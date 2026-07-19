from __future__ import annotations

"""AIMv2 image features — server-mode foundation-model nodeset.

A second self-supervised visual backbone alongside ``model_dinov2``, from a
different pre-training recipe: AIMv2 (Apple) is trained *autoregressively* — a
multimodal decoder predicts image patches (and text) — which yields patch
features that transfer strongly to recognition, grounding and dense tasks. Same
role as DINOv2 in a graph (a generic per-image descriptor, no language head),
but a different feature geometry, so it is worth having both when a downstream
head is sensitive to the backbone.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_aimv2__extract_features  (images: list[{rgb_base64}] | list[b64])
                                   → features: TEXT envelope (N, D), mean-pooled

The AIMv2 vision tower emits per-patch tokens ``(N, P, D)`` with **no CLS /
pooler token**, so the per-image descriptor here is the **mean over patch
tokens** — the standard global-feature reduction for a CLS-less ViT. Features
are *not* L2-normalized (raw pooled activations); normalize downstream if a
cosine metric is wanted. Envelope (DINOv2/CLIP's sibling — raw C-contiguous
float32 base64, byte-exact across the HTTP boundary)::

    {"shape":[N,D], "dtype":"float32", "b64":…, "model_id":…, "pooled":"mean"}

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers`` ``AutoModel``
(``Aimv2VisionModel``) + ``AutoImageProcessor`` (AIMv2 checkpoints, ungated).
Override the env with $AIMV2_PYTHON and the device with $AIMV2_DEVICE (auto →
cuda when available). This file must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_aimv2/load?mode=server

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

log = logging.getLogger("agentcanvas.model_aimv2")

_MODEL_ID_DEFAULT = "apple/aimv2-large-patch14-224"

# Curated AIMv2 size ladder at the standard patch14 / 224px config.
_MODEL_OPTIONS = [
    {"value": "apple/aimv2-large-patch14-224", "label": "AIMv2 Large (patch14, 224)"},
    {"value": "apple/aimv2-huge-patch14-224", "label": "AIMv2 Huge (patch14, 224)"},
    {"value": "apple/aimv2-1B-patch14-224", "label": "AIMv2 1B (patch14, 224)"},
    {"value": "apple/aimv2-3B-patch14-224", "label": "AIMv2 3B (patch14, 224)"},
]


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("AIMV2_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _Aimv2Engine:
    """Lazy singleton registry: one frozen AIMv2 vision tower per ``model_id``."""

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
    def get(cls, model_id: str) -> "_Aimv2Engine":
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
                from transformers import AutoImageProcessor, AutoModel

                self.device = _resolve_device()
                model = AutoModel.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("AIMv2 load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("AIMv2 ready (%s, device=%s)", self.model_id, self.device)
            return True

    def extract(self, images: list) -> "np.ndarray | None":
        """Batched forward → (N, D) mean-pooled float32, input order.

        AIMv2 has no CLS/pooler token; the per-image descriptor is the mean over
        the ``(P)`` patch tokens of ``last_hidden_state``.
        """
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                feats = self.model(pixel_values=pixel_values).last_hidden_state  # (N, P, D)
                feats = feats.mean(dim=1)  # (N, D)
            return feats.detach().cpu().numpy().astype(np.float32)


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _images_from_input(items: list) -> "list | None":
    """Accept {rgb_base64} dicts or raw base64 strings → RGB arrays (None on bad)."""
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


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class Aimv2ExtractFeaturesTool(BaseCanvasNode):
    """Per-image AIMv2 mean-pooled embeddings for a list of images."""

    node_type: ClassVar[str] = "model_aimv2__extract_features"
    display_name: ClassVar[str] = "AIMv2: Extract Features"
    description: ClassVar[str] = (
        "Mean-pooled AIMv2 embedding per image (patch-token mean); base64-npy (N,D) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid3x3"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings"),
    ]
    output_ports = [
        PortDef(
            "features", "TEXT",
            'JSON envelope {"shape":[N,D],"dtype":"float32","b64":…,"model_id":…,"pooled":"mean"}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"features": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)
        engine = _Aimv2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            feats = engine.extract(images)
            if feats is None:
                return ""
            buf = np.ascontiguousarray(feats, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "pooled": "mean",
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no features (load failure or bad input)")
        return {"features": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class Aimv2NodeSet(BaseNodeSet):
    """AIMv2 per-image feature extraction — server-mode FM nodeset."""

    name = "model_aimv2"
    description = (
        "AIMv2 (autoregressive ViT) mean-pooled per-image features — a second "
        "self-supervised backbone alongside DINOv2 on the shared ac-fm server"
    )
    # Stateless feature extractor — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Aimv2 is native there).
    # Override with $AIMV2_PYTHON; device via $AIMV2_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "AIMV2_PYTHON")

    def get_tools(self) -> list:
        return [Aimv2ExtractFeaturesTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_aimv2 ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
