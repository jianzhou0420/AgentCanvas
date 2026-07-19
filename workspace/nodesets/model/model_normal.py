from __future__ import annotations

"""Surface-normal estimation (Sapiens) — server-mode foundation-model nodeset.

Turns a single RGB frame into a dense per-pixel surface-normal map — the
geometry-orientation primitive that complements monocular depth. Ground-plane
and wall detection, walkable-surface reasoning, obstacle facing, and lifting a
2D scene into oriented 3D structure all want per-pixel normals; this is the
foundation-model way to get them from RGB alone.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_normal__estimate_normals  (images: list[{rgb_base64}] | list[b64])
                                    → normals: TEXT envelope (N, H, W, 3) float32

Normal envelope (the raw C-contiguous float32 buffer base64-encoded, byte-exact
across the HTTP boundary, ~4× smaller than a JSON float list)::

    {"shape":[N,H,W,3], "dtype":"float32", "b64":…, "model_id":…}

``normals[n,y,x]`` is a unit 3-vector (x, y, z) in the camera frame (the raw
prediction is bicubic-upsampled to the original (H, W) then L2-renormalized per
pixel). All images in one call must share resolution — the envelope is a single
(N,H,W,3) buffer; mixed sizes degrade to empty with a self-log (split into
uniform batches). In an agent loop N is typically 1 (per-frame normals).

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``AutoModelForNormalEstimation`` + ``AutoImageProcessor`` (Sapiens normal
checkpoints, ungated). Override the env with $NORMAL_PYTHON and the device with
$NORMAL_DEVICE (auto → cuda when available). This file must stay
Python-3.8-parseable.

Load: POST /api/components/nodesets/model_normal/load?mode=server

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

log = logging.getLogger("agentcanvas.model_normal")

_MODEL_ID_DEFAULT = "facebook/sapiens2-normal-0.4b"

# Curated Sapiens2 normal-head size ladder.
_MODEL_OPTIONS = [
    {"value": "facebook/sapiens2-normal-0.4b", "label": "Sapiens2 Normal 0.4B"},
    {"value": "facebook/sapiens2-normal-0.8b", "label": "Sapiens2 Normal 0.8B"},
    {"value": "facebook/sapiens2-normal-1b", "label": "Sapiens2 Normal 1B"},
    {"value": "facebook/sapiens2-normal-5b", "label": "Sapiens2 Normal 5B"},
]


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("NORMAL_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _NormalEngine:
    """Lazy singleton registry: one frozen normal estimator per ``model_id``.

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
    def get(cls, model_id: str) -> "_NormalEngine":
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
                from transformers import AutoImageProcessor, AutoModelForNormalEstimation

                self.device = _resolve_device()
                model = AutoModelForNormalEstimation.from_pretrained(self.model_id)
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("Normal-estimator load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("Normal-estimator ready (%s, device=%s)", self.model_id, self.device)
            return True

    def estimate(self, images: list) -> "np.ndarray | None":
        """Batched forward over same-size HWC uint8 images → (N, H, W, 3) float32.

        The normal map is bicubic-upsampled back to the original (H, W), then
        each pixel's vector is L2-renormalized to unit length. Returns None on
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
                normals = self.model(pixel_values=pixel_values).normals  # (N, 3, h', w')
                normals = F.interpolate(normals, size=(h, w), mode="bicubic", align_corners=False)
                normals = torch.nn.functional.normalize(normals, dim=1)  # unit length per pixel
                normals = normals.permute(0, 2, 3, 1)  # (N, H, W, 3)
            return normals.detach().cpu().numpy().astype(np.float32)


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


class NormalEstimateTool(BaseCanvasNode):
    """Per-image surface normals for a list of same-resolution images."""

    node_type: ClassVar[str] = "model_normal__estimate_normals"
    display_name: ClassVar[str] = "Surface Normals: Sapiens"
    description: ClassVar[str] = (
        "Per-pixel unit surface normals at original resolution; base64-npy (N,H,W,3) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Compass"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "normals", "TEXT",
            'JSON envelope {"shape":[N,H,W,3],"dtype":"float32","b64":…,"model_id":…}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"normals": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)

        engine = _NormalEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            shapes = {img.shape[:2] for img in images}
            if len(shapes) != 1:
                log.warning("Normal-estimator: mixed input resolutions %s — degrading", shapes)
                return "MIXED"
            normals = engine.estimate(images)
            if normals is None:
                return ""
            buf = np.ascontiguousarray(normals, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MIXED":
            self._self_log("degraded", "mixed input resolutions — split into uniform batches")
            return {"normals": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no normals (load failure or bad input)")
        return {"normals": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class NormalNodeSet(BaseNodeSet):
    """Surface-normal estimation (Sapiens) — server-mode FM nodeset."""

    name = "model_normal"
    description = (
        "Sapiens surface-normal estimation — dense per-pixel unit normals from a "
        "single RGB frame on the shared ac-fm server"
    )
    # Stateless normal estimator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers AutoModelForNormalEstimation
    # is native there). Override with $NORMAL_PYTHON; device via $NORMAL_DEVICE
    # (auto → cuda).
    server_python = conda_env_python("ac-fm", "NORMAL_PYTHON")

    def get_tools(self) -> list:
        return [NormalEstimateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_normal ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
