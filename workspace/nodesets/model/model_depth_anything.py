from __future__ import annotations

"""Depth Anything V2 monocular depth — server-mode foundation-model nodeset.

Turns a single RGB frame into a dense depth map — the geometric-perception
primitive the FM palette was missing. RGB-only agents, sim2real, obstacle
avoidance, and lifting a 2D detection into 3D all need per-pixel depth without
a depth sensor; this is the foundation-model way to get it.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight
GPU lock, everything procedural lives in the graph)::

    model_depth_anything__estimate_depth  (images: list[{rgb_base64}] | list[b64])
                                          → depth: TEXT envelope (N, H, W) float32

Depth envelope (the raw C-contiguous float32 buffer base64-encoded, byte-exact
across the HTTP boundary, ~4× smaller than a JSON float list)::

    {"shape":[N,H,W], "dtype":"float32", "b64":…, "model_id":…, "depth_type":…}

``depth_type`` labels the units the consumer should assume:
    relative  — unitless inverse-ish depth (default checkpoint); larger = nearer
    metric    — metres (Metric-* checkpoints); use those + depth_type=metric

Depth maps are returned at the **original input resolution** (the model runs at
its own patch grid, then the prediction is bicubic-upsampled per forward). All
images in one call must share resolution — the envelope is a single (N,H,W)
buffer; mixed sizes degrade to empty with a self-log (split into uniform
batches). In an agent loop N is typically 1 (per-frame depth).

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``AutoModelForDepthEstimation`` + ``AutoImageProcessor`` (Depth-Anything-V2-*
checkpoints, ungated). Override the env with $DEPTH_ANYTHING_PYTHON and the
device with $DEPTH_ANYTHING_DEVICE (auto → cuda when available). This file must
stay Python-3.8-parseable (the override may point at a py3.8 env).

Load: POST /api/components/nodesets/model_depth_anything/load?mode=server

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

log = logging.getLogger("agentcanvas.model_depth_anything")

_MODEL_ID_DEFAULT = "depth-anything/Depth-Anything-V2-Small-hf"

# Curated Depth-Anything-V2 transformers (-hf) checkpoints: relative + metric.
_MODEL_OPTIONS = [
    {"value": "depth-anything/Depth-Anything-V2-Small-hf", "label": "V2 Small (relative)"},
    {"value": "depth-anything/Depth-Anything-V2-Base-hf", "label": "V2 Base (relative)"},
    {"value": "depth-anything/Depth-Anything-V2-Large-hf", "label": "V2 Large (relative)"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "label": "V2 Metric Indoor Small"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", "label": "V2 Metric Indoor Base"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "label": "V2 Metric Indoor Large"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "label": "V2 Metric Outdoor Small"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", "label": "V2 Metric Outdoor Base"},
    {"value": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "label": "V2 Metric Outdoor Large"},
]
# depth_type: blank auto-derives (metric if the id contains "metric", else relative).
_DEPTH_TYPE_OPTIONS = [
    {"value": "", "label": "auto (from model id)"},
    {"value": "relative", "label": "relative"},
    {"value": "metric", "label": "metric"},
]


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("DEPTH_ANYTHING_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _DepthAnythingEngine:
    """Lazy singleton registry: one frozen Depth-Anything per ``model_id``.

    Holds only loaded weights — no cache, no per-call state. The single-flight
    inference lock bounds peak VRAM to one in-flight forward under concurrent
    eval workers (house FM-engine template).
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
    def get(cls, model_id: str) -> "_DepthAnythingEngine":
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
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                self.device = _resolve_device()
                model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("Depth-Anything load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("Depth-Anything ready (%s, device=%s)", self.model_id, self.device)
            return True

    def estimate(self, images: list) -> "np.ndarray | None":
        """Batched forward over same-size HWC uint8 images → (N, H, W) float32.

        Depth is bicubic-upsampled back to the original (H, W). Returns None on
        load failure; the caller has already enforced uniform input size.
        """
        if not self._ensure():
            return None
        import torch
        import torch.nn.functional as F

        h, w = images[0].shape[:2]
        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                pred = self.model(pixel_values=pixel_values).predicted_depth  # (N, h', w')
                pred = pred.unsqueeze(1)  # (N, 1, h', w')
                pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)
                pred = pred.squeeze(1)  # (N, H, W)
            return pred.detach().cpu().numpy().astype(np.float32)


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


class DepthAnythingEstimateTool(BaseCanvasNode):
    """Per-image monocular depth for a list of same-resolution images."""

    node_type: ClassVar[str] = "model_depth_anything__estimate_depth"
    display_name: ClassVar[str] = "Depth Anything: Estimate Depth"
    description: ClassVar[str] = (
        "Monocular depth per image at original resolution; base64-npy (N,H,W) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Mountain"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
            ConfigField(
                "depth_type", "select", label="Units label",
                options=list(_DEPTH_TYPE_OPTIONS), default="",
            ),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "depth", "TEXT",
            'JSON envelope {"shape":[N,H,W],"dtype":"float32","b64":…,"model_id":…,"depth_type":…}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"depth": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)
        depth_type = (cfg.get("depth_type") or "").strip()
        if not depth_type:
            depth_type = "metric" if "metric" in model_id.lower() else "relative"

        engine = _DepthAnythingEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            shapes = {img.shape[:2] for img in images}
            if len(shapes) != 1:
                log.warning("Depth-Anything: mixed input resolutions %s — degrading", shapes)
                return "MIXED"
            depth = engine.estimate(images)
            if depth is None:
                return ""
            buf = np.ascontiguousarray(depth, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "depth_type": depth_type,
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MIXED":
            self._self_log("degraded", "mixed input resolutions — split into uniform batches")
            return {"depth": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no depth (load failure or bad input)")
        return {"depth": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class DepthAnythingNodeSet(BaseNodeSet):
    """Depth Anything V2 monocular depth — server-mode FM nodeset."""

    name = "model_depth_anything"
    description = (
        "Depth Anything V2 monocular depth estimation (relative / metric "
        "checkpoints as config) — dense per-pixel depth on the shared ac-fm server"
    )
    # Stateless depth estimator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers AutoModelForDepthEstimation
    # is native there). Override with $DEPTH_ANYTHING_PYTHON; device via
    # $DEPTH_ANYTHING_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "DEPTH_ANYTHING_PYTHON")

    def get_tools(self) -> list:
        return [DepthAnythingEstimateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_depth_anything ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
