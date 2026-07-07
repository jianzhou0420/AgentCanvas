from __future__ import annotations

"""CLIP image-text embeddings — server-mode foundation-model nodeset.

The language-aligned counterpart to ``model_dinov2``: where DINOv2 gives a
self-supervised per-image feature (no language), CLIP gives an embedding that
lives in a *shared image-text space*, so a picture and the phrase describing it
land near each other. That shared space is the geometry behind open-vocabulary
maps (VLMaps / ConceptFusion), zero-shot retrieval, and CLIP-reward models —
the biggest hole in the FM palette before this nodeset.

Three pure single-step primitives (see the FM-nodeset design doc / model_sam
for the template — stateless server, engines keyed by ``model_id`` in a lazy
registry, load-failure latch + single-flight GPU lock, everything procedural
lives in the graph)::

    model_clip__encode_image  (images: list[{rgb_base64}] | list[b64])
                              → embeddings: TEXT envelope (N, D), L2-normalized
    model_clip__encode_text   (texts: list[str] | JSON)
                              → embeddings: TEXT envelope (N, D), L2-normalized
    model_clip__classify      (images + labels: list[str])
                              → scores: JSON {labels, probs[N][L], logits, top}

``classify`` is CLIP's *native* zero-shot capability — it uses the model's own
learned ``logit_scale`` temperature, not a hand-rolled cosine, so the softmax
probabilities reproduce true CLIP zero-shot behaviour. Labels are passed
**verbatim** (no hidden "a photo of a {}" template): prompt engineering is the
caller's decision, kept out of the primitive.

Embedding envelope (DINOv2's sibling — the raw C-contiguous float32 buffer
base64-encoded, byte-exact across the HTTP boundary and ~4× smaller than a
JSON float list)::

    {"shape":[N,D], "dtype":"float32", "b64":…, "model_id":…, "normalized":true}

Image and text embeddings from the same ``model_id`` are directly comparable
(cosine = dot product, since both are L2-normalized) — a graph can wire
``encode_image`` + ``encode_text`` into its own similarity/retrieval logic
without ``classify``.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers`` ``CLIPModel`` +
``CLIPProcessor``. Override the env with $CLIP_PYTHON and the device with
$CLIP_DEVICE (auto → cuda when available). This file must stay
Python-3.8-parseable (the override may point at a py3.8 env).

Load: POST /api/components/nodesets/model_clip/load?mode=server

last updated: 2026-07-07
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

log = logging.getLogger("agentcanvas.model_clip")

_MODEL_ID_DEFAULT = "openai/clip-vit-base-patch32"


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("CLIP_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _ClipEngine:
    """Lazy singleton registry: one frozen CLIP per ``model_id``.

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
    def get(cls, model_id: str) -> "_ClipEngine":
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
                from transformers import CLIPModel, CLIPProcessor

                self.device = _resolve_device()
                model = CLIPModel.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("CLIP load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("CLIP ready (%s, device=%s)", self.model_id, self.device)
            return True

    def encode_image(self, images: list) -> "np.ndarray | None":
        """Batched image forward → (N, D) L2-normalized float32, input order.

        Goes through the vision tower + ``visual_projection`` head rather than
        ``get_image_features`` — in transformers 5.x the latter returns the raw
        vision output (pre-projection), not the projected embedding. This path
        is bit-exact with the model's own normalized ``image_embeds`` (verified
        2026-07-07).
        """
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                feats = self.model.visual_projection(vision_out.pooler_output)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype(np.float32)

    def encode_text(self, texts: list) -> "np.ndarray | None":
        """Batched text forward → (N, D) L2-normalized float32, input order.

        Text tower + ``text_projection`` head (see ``encode_image`` for why we
        avoid ``get_text_features``); bit-exact with the model's normalized
        ``text_embeds``.
        """
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True)
            pp = {k: v.to(self.device) for k, v in pp.items()}
            with torch.no_grad():
                text_out = self.model.text_model(
                    input_ids=pp["input_ids"],
                    attention_mask=pp.get("attention_mask"),
                )
                feats = self.model.text_projection(text_out.pooler_output)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype(np.float32)

    def classify(self, images: list, labels: list) -> "dict | None":
        """Zero-shot: (N_img, N_label) logits/probs via the model's logit_scale."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(
                text=labels, images=images, return_tensors="pt",
                padding=True, truncation=True)
            pp = {k: v.to(self.device) for k, v in pp.items()}
            with torch.no_grad():
                out = self.model(**pp)
                logits = out.logits_per_image  # (N_img, N_label), logit_scale-scaled
                probs = logits.softmax(dim=-1)
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


class ClipEncodeImageTool(BaseCanvasNode):
    """L2-normalized CLIP image embeddings for a list of images."""

    node_type: ClassVar[str] = "model_clip__encode_image"
    display_name: ClassVar[str] = "CLIP: Encode Image"
    description: ClassVar[str] = (
        "L2-normalized CLIP image embedding per image; base64-npy (N,D) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Image"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "text", "HF CLIP model repo id", default=_MODEL_ID_DEFAULT),
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
        engine = _ClipEngine.get(model_id)
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


class ClipEncodeTextTool(BaseCanvasNode):
    """L2-normalized CLIP text embeddings for a list of strings."""

    node_type: ClassVar[str] = "model_clip__encode_text"
    display_name: ClassVar[str] = "CLIP: Encode Text"
    description: ClassVar[str] = (
        "L2-normalized CLIP text embedding per string; base64-npy (N,D) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Type"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "text", "HF CLIP model repo id", default=_MODEL_ID_DEFAULT),
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
        engine = _ClipEngine.get(model_id)
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


class ClipClassifyTool(BaseCanvasNode):
    """Zero-shot classification of images against caller-supplied text labels.

    Uses the model's learned ``logit_scale`` (true CLIP zero-shot), so ``probs``
    is the genuine softmax over image·label similarities, not a bare cosine.
    Labels are used verbatim — supply full prompts if you want a template.
    """

    node_type: ClassVar[str] = "model_clip__classify"
    display_name: ClassVar[str] = "CLIP: Zero-Shot Classify"
    description: ClassVar[str] = (
        "Zero-shot image↔label scores using CLIP's own logit_scale; JSON {labels,probs,logits,top}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tags"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("model_id", "text", "HF CLIP model repo id", default=_MODEL_ID_DEFAULT),
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
        engine = _ClipEngine.get(model_id)
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


class ClipNodeSet(BaseNodeSet):
    """CLIP image-text embedding primitives — server-mode FM nodeset."""

    name = "model_clip"
    description = (
        "CLIP image-text embeddings (encode_image / encode_text / zero-shot "
        "classify) — language-aligned visual features on the shared ac-fm server"
    )
    # Stateless embedding primitives — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers CLIPModel is native there).
    # Override with $CLIP_PYTHON; device via $CLIP_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "CLIP_PYTHON")

    def get_tools(self) -> list:
        return [ClipEncodeImageTool(), ClipEncodeTextTool(), ClipClassifyTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_clip ready (server_python=%s); engines load lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
