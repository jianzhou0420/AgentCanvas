from __future__ import annotations

"""SigLIP 2 image-text embeddings — server-mode foundation-model nodeset.

The modern successor to ``model_clip``: same shared image-text embedding space,
but trained with a *sigmoid* pairwise loss (SigLIP) rather than CLIP's softmax
contrastive loss, which gives stronger zero-shot transfer and — the practical
consequence — **independent per-label probabilities** instead of a distribution
that must sum to 1. Where CLIP's ``classify`` softmaxes across the label set
(the labels compete), SigLIP scores each image·label pair on its own, so "is
this a kitchen? is this a hallway?" are answered independently (both can be
high, both can be low). That is exactly what open-vocabulary mapping and
multi-label scene tagging want.

Three pure single-step primitives (FM-nodeset template — stateless server,
engines keyed by ``model_id`` in a lazy registry, load-failure latch +
single-flight GPU lock, everything procedural lives in the graph)::

    model_siglip2__encode_image  (images: list[{rgb_base64}] | list[b64])
                                 → embeddings: TEXT envelope (N, D), L2-normalized
    model_siglip2__encode_text   (texts: list[str] | JSON)
                                 → embeddings: TEXT envelope (N, D), L2-normalized
    model_siglip2__classify      (images + labels: list[str])
                                 → scores: JSON {labels, probs[N][L], logits, top}

``classify`` reports ``probs = sigmoid(logit_scale·⟨img,txt⟩ + bias)`` using the
model's own learned scale/bias — genuine SigLIP zero-shot. Because it is a
sigmoid, ``probs[n]`` does **not** sum to 1 (multi-label by construction);
``top`` is still the single highest-scoring label per image. Labels are passed
**verbatim** — SigLIP's own eval uses ``"This is a photo of a {}."``; supply the
full prompt if you want a template.

Embedding envelope (CLIP/DINOv2's sibling — the raw C-contiguous float32 buffer
base64-encoded, byte-exact across the HTTP boundary and ~4× smaller than a JSON
float list)::

    {"shape":[N,D], "dtype":"float32", "b64":…, "model_id":…, "normalized":true}

Image and text embeddings from the same ``model_id`` are directly comparable
(cosine = dot product, since both are L2-normalized).

Note vs CLIP internals: SigLIP folds the projection *into* the towers — the
image/text embedding is the tower ``pooler_output`` (no separate
``visual_projection`` head), and ``get_image_features`` returns the raw tower
output object in transformers 5.x (same gotcha as CLIP), so we read
``vision_model``/``text_model`` ``pooler_output`` and L2-normalize ourselves —
bit-exact with the model's own ``image_embeds``/``text_embeds``. SigLIP text is
tokenised with ``padding="max_length", max_length=64`` (its training regime);
we pin that so scores match the model's intended behaviour.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers`` ``AutoModel`` +
``AutoProcessor`` (SigLIP 2 checkpoints, ungated). Override the env with
$SIGLIP2_PYTHON and the device with $SIGLIP2_DEVICE (auto → cuda when
available). This file must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_siglip2/load?mode=server

last updated: 2026-07-08
"""

import asyncio
import base64
import io
import json
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.model_siglip2")

_MODEL_ID_DEFAULT = "google/siglip2-base-patch16-224"

# Curated SigLIP2 variants (fixed-res + NaFlex); shared by all three nodes.
_MODEL_OPTIONS = [
    {"value": "google/siglip2-base-patch16-224", "label": "SigLIP2 Base p16 (224)"},
    {"value": "google/siglip2-base-patch16-256", "label": "SigLIP2 Base p16 (256)"},
    {"value": "google/siglip2-large-patch16-256", "label": "SigLIP2 Large p16 (256)"},
    {"value": "google/siglip2-so400m-patch14-384", "label": "SigLIP2 SoViT-400m p14 (384)"},
    {"value": "google/siglip2-base-patch16-naflex", "label": "SigLIP2 Base NaFlex"},
    {"value": "google/siglip2-so400m-patch16-naflex", "label": "SigLIP2 SoViT-400m NaFlex"},
]
# SigLIP's training/eval tokenisation: pad every caption to a fixed 64 tokens.
_TEXT_MAX_LEN = 64


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("SIGLIP2_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _Siglip2Engine:
    """Lazy singleton registry: one frozen SigLIP 2 per ``model_id``.

    Holds only loaded weights — no cache, no per-call state. Concurrent eval
    workers coalesce onto the one shared engine; the single-flight inference
    lock bounds peak VRAM to a single in-flight forward.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = None
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str) -> "_Siglip2Engine":
        with cls._lock:
            if model_id not in cls._instances:
                cls._instances[model_id] = cls(model_id)
            return cls._instances[model_id]

    def _ensure(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import torch  # noqa: F401
                from transformers import AutoModel, AutoProcessor

                self.device = _resolve_device()
                model = AutoModel.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("SigLIP2 load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("SigLIP2 ready (%s, device=%s)", self.model_id, self.device)
            return True

    def encode_image(self, images: list) -> "np.ndarray | None":
        """Batched image forward → (N, D) L2-normalized float32, input order.

        Reads the vision tower's ``pooler_output`` (SigLIP folds the projection
        into the tower) and L2-normalizes — bit-exact with the model's own
        normalized ``image_embeds``. ``get_image_features`` is avoided: in
        transformers 5.x it returns the raw tower output object, not the vector.
        """
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                feats = self.model.vision_model(pixel_values=pixel_values).pooler_output
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype(np.float32)

    def encode_text(self, texts: list) -> "np.ndarray | None":
        """Batched text forward → (N, D) L2-normalized float32, input order.

        Text tower ``pooler_output`` + L2-norm (see ``encode_image``). Tokenised
        with SigLIP's fixed ``max_length=64`` padding regime.
        """
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(
                text=texts, return_tensors="pt",
                padding="max_length", max_length=_TEXT_MAX_LEN, truncation=True)
            input_ids = pp["input_ids"].to(self.device)
            with torch.no_grad():
                feats = self.model.text_model(input_ids=input_ids).pooler_output
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype(np.float32)

    def classify(self, images: list, labels: list) -> "dict | None":
        """Zero-shot: (N_img, N_label) logits + sigmoid probs via the model's
        learned logit_scale/bias. Probs are per-pair (multi-label), NOT softmax."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(
                text=labels, images=images, return_tensors="pt",
                padding="max_length", max_length=_TEXT_MAX_LEN, truncation=True)
            pp = {k: v.to(self.device) for k, v in pp.items()}
            with torch.no_grad():
                out = self.model(**pp)
                logits = out.logits_per_image  # (N_img, N_label), scaled + biased
                probs = torch.sigmoid(logits)  # SigLIP: per-pair sigmoid, not softmax
            return {
                "logits": logits.detach().cpu().numpy().astype(np.float32),
                "probs": probs.detach().cpu().numpy().astype(np.float32),
            }


# ══════════════════════════════════════════════════════════════════════
# Input / output helpers
# ══════════════════════════════════════════════════════════════════════


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _images_from_input(items: list) -> "list | None":
    """Accept a list of {rgb_base64} dicts or raw base64 strings → RGB arrays.

    Returns None if any entry is missing/malformed (consumer degrades).
    """
    images = []
    for it in items:
        if isinstance(it, dict):
            b64 = it.get("rgb_base64") or it.get("image_base64")
        elif isinstance(it, str):
            b64 = it
        else:
            b64 = None
        if not b64:
            return None
        images.append(_decode_rgb(b64))
    return images


def _as_str_list(x: Any) -> "list | None":
    """Normalize texts/labels: list[str], a JSON list string, or a single str."""
    if x is None:
        return None
    if isinstance(x, list):
        out = [str(t) for t in x]
        return out or None
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed] or None
            except Exception:
                pass
        return [s] if s else None
    return None


def _embedding_envelope(feats: np.ndarray, model_id: str) -> str:
    buf = np.ascontiguousarray(feats, dtype=np.float32)
    return json.dumps({
        "shape": list(buf.shape),
        "dtype": "float32",
        "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        "model_id": model_id,
        "normalized": True,
    })


def _model_id(node: BaseCanvasNode) -> str:
    cfg = getattr(node, "config", None) or {}
    return cfg.get("model_id", _MODEL_ID_DEFAULT)


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class Siglip2EncodeImageTool(BaseCanvasNode):
    """L2-normalized SigLIP 2 image embeddings for a list of images."""

    node_type: ClassVar[str] = "model_siglip2__encode_image"
    display_name: ClassVar[str] = "SigLIP2: Encode Image"
    description: ClassVar[str] = (
        "L2-normalized SigLIP2 image embedding per image; base64-npy (N,D) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Image"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings"),
    ]
    output_ports = [
        PortDef(
            "embeddings", "TEXT",
            'JSON envelope {"shape":[N,D],"dtype":"float32","b64":…,"model_id":…,"normalized":true}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"embeddings": ""}
        model_id = _model_id(self)
        engine = _Siglip2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            feats = engine.encode_image(images)
            if feats is None:
                return ""
            return _embedding_envelope(feats, model_id)

        envelope = await loop.run_in_executor(None, _run)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no image embeddings (load failure or bad input)")
        return {"embeddings": envelope}


class Siglip2EncodeTextTool(BaseCanvasNode):
    """L2-normalized SigLIP 2 text embeddings for a list of strings."""

    node_type: ClassVar[str] = "model_siglip2__encode_text"
    display_name: ClassVar[str] = "SigLIP2: Encode Text"
    description: ClassVar[str] = (
        "L2-normalized SigLIP2 text embedding per string; base64-npy (N,D) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Type"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("texts", "ANY", "List of strings (or a JSON list / single string)"),
    ]
    output_ports = [
        PortDef(
            "embeddings", "TEXT",
            'JSON envelope {"shape":[N,D],"dtype":"float32","b64":…,"model_id":…,"normalized":true}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        texts = _as_str_list(inputs.get("texts"))
        if not texts:
            return {"embeddings": ""}
        model_id = _model_id(self)
        engine = _Siglip2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            feats = engine.encode_text(texts)
            if feats is None:
                return ""
            return _embedding_envelope(feats, model_id)

        envelope = await loop.run_in_executor(None, _run)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no text embeddings (load failure)")
        return {"embeddings": envelope}


class Siglip2ClassifyTool(BaseCanvasNode):
    """Zero-shot classification of images against caller-supplied text labels.

    Uses SigLIP's learned ``logit_scale``/``logit_bias`` with a **sigmoid**, so
    ``probs`` is per image·label pair and does NOT sum to 1 (multi-label). Labels
    are used verbatim — supply full prompts if you want a template.
    """

    node_type: ClassVar[str] = "model_siglip2__classify"
    display_name: ClassVar[str] = "SigLIP2: Zero-Shot Classify"
    description: ClassVar[str] = (
        "Zero-shot image↔label sigmoid scores (per-pair, multi-label); JSON {labels,probs,logits,top}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tags"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings"),
        PortDef("labels", "ANY", "Candidate text labels (list[str] / JSON / single string)"),
    ]
    output_ports = [
        PortDef(
            "scores", "TEXT",
            'JSON {"labels":[…],"probs":[N][L],"logits":[N][L],"top":[{label,prob}],"model_id":…}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        labels = _as_str_list(inputs.get("labels"))
        if not items or not labels:
            return {"scores": ""}
        model_id = _model_id(self)
        engine = _Siglip2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            res = engine.classify(images, labels)
            if res is None:
                return ""
            probs = res["probs"]
            logits = res["logits"]
            top = []
            for row in probs:
                j = int(np.argmax(row))
                top.append({"label": labels[j], "prob": float(row[j])})
            return json.dumps({
                "labels": labels,
                "probs": probs.tolist(),
                "logits": logits.tolist(),
                "top": top,
                "model_id": model_id,
            })

        out = await loop.run_in_executor(None, _run)
        if out:
            self._self_log("top", [t["label"] for t in json.loads(out)["top"]])
        else:
            self._self_log("degraded", "no scores (load failure or bad input)")
        return {"scores": out}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class Siglip2NodeSet(BaseNodeSet):
    """SigLIP 2 image-text embedding primitives — server-mode FM nodeset."""

    name = "model_siglip2"
    description = (
        "SigLIP 2 image-text embeddings (encode_image / encode_text / sigmoid "
        "zero-shot classify) — stronger open-vocab visual features on the shared ac-fm server"
    )
    # Stateless embedding primitives — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers SigLIP2 is native there).
    # Override with $SIGLIP2_PYTHON; device via $SIGLIP2_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "SIGLIP2_PYTHON")

    def get_tools(self) -> list:
        return [Siglip2EncodeImageTool(), Siglip2EncodeTextTool(), Siglip2ClassifyTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_siglip2 ready (server_python=%s); engines load lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
