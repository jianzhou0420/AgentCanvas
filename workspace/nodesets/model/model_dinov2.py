from __future__ import annotations

"""DINOv2 / DINOv3 image features — server-mode foundation-model nodeset.

Extracted from ``smartway_waypoint`` (where DINOv2 lived inside the waypoint
engine as the RGB backbone) per the method / foundation-model boundary
principle (roadmap TODO #56) and an explicit user decision (2026-07-04): the
DDPPO depth encoder stays method-internal, but the DINOv2 backbone is exposed
as a generic per-image feature primitive.

Two backends, one tool (``backend`` picks per node):

  * **hub** (default) — Load + forward are byte-faithful to the smartway engine
    (upstream ``Policy_ViewSelection_VLNBERT.py:111`` + ``base_il_trainer.py:356``):
    ``torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')`` frozen/eval,
    preprocessed by ``AutoImageProcessor('facebook/dinov2-small')``, one batched
    forward → pooled per-image embeddings (ViT-S/14-reg → 384-d).
  * **hf** — transformers-native ``AutoModel`` + ``AutoImageProcessor``, per-image
    descriptor = the CLS ``pooler_output``. This is the path for **DINOv3**
    (``facebook/dinov3-*`` — a gated repo: accept the licence on Hugging Face
    once, with ``HF_TOKEN`` set) and for any HF DINOv2 checkpoint
    (``facebook/dinov2-*``, ungated). Not byte-exact with the hub path (different
    weights/preprocessing) — use it when you want DINOv3 or a HF-hosted variant,
    not for smartway parity.

DINOv2/DINOv3 are per-image ViTs (no cross-sample ops), so features computed
here in input order equal the ones the old engine computed after its clockwise
reorder — the consumer applies its own ordering.

Single tool::

    model_dinov2__extract_features  (views: ordered list of {rgb_base64})
                                    → features: TEXT JSON envelope
                                      {"shape":[N,D],"dtype":"float32","b64":…}

The envelope carries the raw C-contiguous float32 buffer base64-encoded —
byte-exact across the HTTP boundary (a JSON float list would round-trip
through decimal text) and ~4× smaller.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) since 2026-07-05. Numeric parity of the *hub*
path with the old ``ac-smartway`` hosting is BYTE-EXACT under two pinned
conditions, both verified in the 2026-07-05 parity probe: the ViT forward is
bit-identical across torch 2.1.1→2.8.0, and the processor is forced to the PIL
backend (``use_fast=False`` below — transformers 5.x flipped the default to a
torchvision implementation whose resize numerics differ). Override with
$DINOV2_PYTHON. This file must stay Python-3.8-parseable (override may point
at a py3.8 env).

Load: POST /api/components/nodesets/model_dinov2/load?mode=server

FM-template alignment (2026-07-05): single-flight GPU inference lock added
(one in-flight forward per engine). Model identity (`hub_model` /
`processor_id`) was already node config; registry, load-failure latch and
degraded self-log were already in place.

last updated: 2026-07-08 (added the transformers-native ``hf`` backend for DINOv3)
"""

import asyncio
import base64
import io
import json
import logging
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

log = logging.getLogger("agentcanvas.model_dinov2")

_HUB_REPO_DEFAULT = "facebookresearch/dinov2"
_HUB_MODEL_DEFAULT = "dinov2_vits14_reg"
_PROCESSOR_DEFAULT = "facebook/dinov2-small"
_BACKEND_DEFAULT = "hub"
# hf-backend default: the ungated HF DINOv2-reg (parity-adjacent, testable).
# Set model_id to facebook/dinov3-* for DINOv3 (gated — accept the licence once).
_HF_MODEL_DEFAULT = "facebook/dinov2-with-registers-small"

# torch.hub DINOv2 backbone entry names (from facebookresearch/dinov2 hubconf).
_HUB_MODEL_OPTIONS = [
    {"value": "dinov2_vits14_reg", "label": "ViT-S/14 + registers"},
    {"value": "dinov2_vitb14_reg", "label": "ViT-B/14 + registers"},
    {"value": "dinov2_vitl14_reg", "label": "ViT-L/14 + registers"},
    {"value": "dinov2_vitg14_reg", "label": "ViT-g/14 + registers"},
    {"value": "dinov2_vits14", "label": "ViT-S/14"},
    {"value": "dinov2_vitb14", "label": "ViT-B/14"},
    {"value": "dinov2_vitl14", "label": "ViT-L/14"},
    {"value": "dinov2_vitg14", "label": "ViT-g/14"},
]
# HF image-processor repos paired with the hub backbone (preprocessing only).
_PROCESSOR_OPTIONS = [
    {"value": "facebook/dinov2-small", "label": "dinov2-small processor"},
    {"value": "facebook/dinov2-base", "label": "dinov2-base processor"},
    {"value": "facebook/dinov2-large", "label": "dinov2-large processor"},
    {"value": "facebook/dinov2-giant", "label": "dinov2-giant processor"},
]
# HF transformers repos for the hf backend: DINOv2 (ungated) + DINOv3 (gated).
_HF_MODEL_OPTIONS = [
    {"value": "facebook/dinov2-with-registers-small", "label": "DINOv2 ViT-S + reg (HF)"},
    {"value": "facebook/dinov2-with-registers-base", "label": "DINOv2 ViT-B + reg (HF)"},
    {"value": "facebook/dinov2-with-registers-large", "label": "DINOv2 ViT-L + reg (HF)"},
    {"value": "facebook/dinov2-small", "label": "DINOv2 ViT-S (HF)"},
    {"value": "facebook/dinov2-base", "label": "DINOv2 ViT-B (HF)"},
    {"value": "facebook/dinov2-large", "label": "DINOv2 ViT-L (HF)"},
    {"value": "facebook/dinov2-giant", "label": "DINOv2 ViT-g (HF)"},
    {"value": "facebook/dinov3-vits16-pretrain-lvd1689m", "label": "DINOv3 ViT-S/16 (gated)"},
    {"value": "facebook/dinov3-vitb16-pretrain-lvd1689m", "label": "DINOv3 ViT-B/16 (gated)"},
    {"value": "facebook/dinov3-vitl16-pretrain-lvd1689m", "label": "DINOv3 ViT-L/16 (gated)"},
    {"value": "facebook/dinov3-convnext-base-pretrain-lvd1689m", "label": "DINOv3 ConvNeXt-Base (gated)"},
]


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per (backend, model key)
# ══════════════════════════════════════════════════════════════════════


class _DinoV2Engine:
    """Lazy singleton registry: one frozen DINO per (backend, model key).

    ``backend="hub"`` loads the torch.hub DINOv2 (byte-exact smartway parity);
    ``backend="hf"`` loads a transformers ``AutoModel`` (DINOv3 or any HF DINOv2
    checkpoint), reading the CLS ``pooler_output`` as the per-image descriptor.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, backend: str, hub_repo: str, hub_model: str, processor_id: str, hf_model: str) -> None:
        self.backend = backend
        self.hub_repo = hub_repo
        self.hub_model = hub_model
        self.processor_id = processor_id
        self.hf_model = hf_model
        self.device = None
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_failed = False
        # Single-flight GPU section: one in-flight forward per engine bounds
        # peak VRAM under concurrent eval workers (house FM-engine template).
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, backend: str, hub_repo: str, hub_model: str, processor_id: str, hf_model: str) -> "_DinoV2Engine":
        if backend == "hf":
            key = ("hf", hf_model)
        else:
            key = ("hub", hub_repo, hub_model, processor_id)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(backend, hub_repo, hub_model, processor_id, hf_model)
            return cls._instances[key]

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
            if self.backend == "hf":
                return self._ensure_hf()
            return self._ensure_hub()

    def _ensure_hub(self) -> bool:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Verbatim smartway engine load (upstream
        # Policy_ViewSelection_VLNBERT.py:111): frozen, eval.
        try:
            model = torch.hub.load(self.hub_repo, self.hub_model).to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.model = model
        except Exception as exc:
            log.warning("DINOv2 hub load failed: %s", exc)
            self._load_failed = True
            return False
        try:
            from transformers import AutoImageProcessor  # type: ignore

            # use_fast=False pins the PIL preprocessing backend: byte-equal
            # pixels/features vs the pre-2026-07-05 ac-smartway hosting.
            # transformers 5.x defaults to a torchvision backend whose
            # resize numerics differ; the kwarg is valid (no-op) on 4.x.
            self.processor = AutoImageProcessor.from_pretrained(
                self.processor_id, use_fast=False)
        except Exception as exc:
            log.warning("DINO processor load failed: %s", exc)
            self._load_failed = True
            return False
        self._loaded = True
        log.info("DINOv2 ready (hub %s/%s, device=%s)", self.hub_repo, self.hub_model, self.device)
        return True

    def _ensure_hf(self) -> bool:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore

            model = AutoModel.from_pretrained(self.hf_model).to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.model = model
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model)
        except Exception as exc:
            log.warning(
                "DINO hf load failed (%s) — gated DINOv3? accept the licence at "
                "https://huggingface.co/%s with HF_TOKEN set: %s",
                self.hf_model, self.hf_model, exc)
            self._load_failed = True
            return False
        self._loaded = True
        log.info("DINO ready (hf %s, device=%s)", self.hf_model, self.device)
        return True

    def extract(self, images: list) -> "np.ndarray | None":
        """Batched forward over HWC uint8 arrays → (N, D) float32, input order."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                if self.backend == "hf":
                    # transformers ViT → CLS pooler_output as the per-image vector.
                    feats = self.model(pixel_values=pixel_values).pooler_output
                else:
                    feats = self.model(pixel_values)
            return feats.detach().cpu().numpy().astype(np.float32)


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class ExtractFeaturesTool(BaseCanvasNode):
    """Per-image DINOv2 embeddings for an ordered view list.

    Output order matches input order; the consumer owns any reordering
    (e.g. smartway's clockwise remap inside the waypoint predictor).
    On load failure emits an empty ``features`` string — consumers keep
    their degraded-mode fallback (the old in-engine behaviour was zeros).
    """

    node_type: ClassVar[str] = "model_dinov2__extract_features"
    display_name: ClassVar[str] = "DINOv2/v3: Extract Features"
    description: ClassVar[str] = (
        "Pooled DINOv2/DINOv3 embedding per view; base64-npy envelope (hub: ViT-S/14-reg → 384-d)"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid3x3"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "backend", "select", "Backend (hub = byte-exact smartway; hf = DINOv3 / HF DINOv2)",
                options=[
                    {"value": "hub", "label": "hub (torch.hub DINOv2, smartway parity)"},
                    {"value": "hf", "label": "hf (transformers — DINOv3 / HF DINOv2)"},
                ],
                default=_BACKEND_DEFAULT,
            ),
            ConfigField("hub_model", "select", label="Hub model (backend=hub)", options=list(_HUB_MODEL_OPTIONS), default=_HUB_MODEL_DEFAULT),
            ConfigField("processor_id", "select", label="Processor (backend=hub)", options=list(_PROCESSOR_OPTIONS), default=_PROCESSOR_DEFAULT),
            ConfigField(
                "hf_model", "select", label="HF model (backend=hf)",
                options=list(_HF_MODEL_OPTIONS), default=_HF_MODEL_DEFAULT,
            ),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "Ordered list of {rgb_base64} dicts (e.g. 12 directions)"),
    ]
    output_ports = [
        PortDef(
            "features", "TEXT",
            'JSON envelope {"shape":[N,D],"dtype":"float32","b64":…} of the (N,D) buffer',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"features": ""}

        cfg = getattr(self, "config", None) or {}
        backend = (cfg.get("backend", _BACKEND_DEFAULT) or _BACKEND_DEFAULT).strip()
        hub_model = cfg.get("hub_model", _HUB_MODEL_DEFAULT)
        processor_id = cfg.get("processor_id", _PROCESSOR_DEFAULT)
        hf_model = (cfg.get("hf_model", _HF_MODEL_DEFAULT) or _HF_MODEL_DEFAULT).strip()

        loop = asyncio.get_running_loop()
        engine = _DinoV2Engine.get(backend, _HUB_REPO_DEFAULT, hub_model, processor_id, hf_model)

        def _extract() -> str:
            images = []
            for v in views:
                b64 = v.get("rgb_base64") if isinstance(v, dict) else None
                if not b64:
                    return ""  # incomplete panorama → let consumer degrade
                images.append(_decode_rgb(b64))
            feats = engine.extract(images)
            if feats is None:
                return ""
            buf = np.ascontiguousarray(feats)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            })

        envelope = await loop.run_in_executor(None, _extract)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no features (load failure or missing view)")
        return {"features": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class DinoV2NodeSet(BaseNodeSet):
    """DINOv2 per-image feature extraction — server-mode FM nodeset."""

    name = "model_dinov2"
    description = (
        "DINOv2/DINOv3 pooled per-image features (hub = byte-exact smartway parity; "
        "hf = transformers DINOv3 / HF DINOv2) — server-mode FM nodeset"
    )
    # Stateless feature extractor — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env) — byte-equal features vs the old
    # ac-smartway hosting given the use_fast=False PIL pin above (parity gate
    # 2026-07-05). Override with $DINOV2_PYTHON.
    server_python = conda_env_python("ac-fm", "DINOV2_PYTHON")

    def get_tools(self) -> list:
        return [ExtractFeaturesTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("DinoV2NodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        pass
