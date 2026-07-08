from __future__ import annotations

"""Single-image 3D pointmap (Sapiens) — server-mode foundation-model nodeset.

Turns a single RGB frame into a dense per-pixel 3D pointmap: for every pixel, an
(X, Y, Z) coordinate in the camera frame. This is the monocular counterpart to
VGGT's multi-view pointmap — where VGGT fuses several views into one 3D field,
this lifts a *single* image into 3D geometry directly, no camera intrinsics, no
second view. Obstacle geometry, free-space extent, "how far is that wall" and
back-projecting a 2D detection into a 3D point all want this.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_pointmap__estimate_pointmap  (images: list[{rgb_base64}] | list[b64])
                                       → pointmap: TEXT envelope (N, H, W, 3) float32

Pointmap envelope (the raw C-contiguous float32 buffer base64-encoded, byte-exact
across the HTTP boundary)::

    {"shape":[N,H,W,3], "dtype":"float32", "b64":…, "model_id":…, "scales":[…]}

``pointmap[n,y,x]`` is an (X, Y, Z) coordinate in canonical camera space, resized
back to the original (H, W). Divide by ``scales[n]`` (one scalar per image) to
convert to metric coordinates — the raw prediction is scale-canonical, so the
scale factor is carried alongside rather than silently applied. All images in one
call must share resolution — the envelope is a single (N,H,W,3) buffer; mixed
sizes degrade to empty with a self-log (split into uniform batches). In an agent
loop N is typically 1 (per-frame geometry).

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``Sapiens2ForPointmapEstimation`` + ``AutoImageProcessor`` (Sapiens pointmap
checkpoints, ungated). Override the env with $POINTMAP_PYTHON and the device with
$POINTMAP_DEVICE (auto → cuda when available). This file must stay
Python-3.8-parseable.

Load: POST /api/components/nodesets/model_pointmap/load?mode=server

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

log = logging.getLogger("agentcanvas.model_pointmap")

_MODEL_ID_DEFAULT = "facebook/sapiens2-pointmap-0.4b"


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("POINTMAP_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _PointmapEngine:
    """Lazy singleton registry: one frozen pointmap estimator per ``model_id``.

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
    def get(cls, model_id: str) -> "_PointmapEngine":
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
                from transformers import AutoImageProcessor, Sapiens2ForPointmapEstimation

                self.device = _resolve_device()
                model = Sapiens2ForPointmapEstimation.from_pretrained(self.model_id)
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("Pointmap-estimator load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("Pointmap-estimator ready (%s, device=%s)", self.model_id, self.device)
            return True

    def estimate(self, images: list) -> "tuple | None":
        """Batched forward over same-size HWC uint8 images.

        Returns ``(pointmaps (N,H,W,3) float32, scales (N,) float32)`` in
        canonical camera space at the original (H, W), or None on load failure.
        The caller has already enforced uniform input size.
        """
        if not self._ensure():
            return None
        import torch

        h, w = images[0].shape[:2]
        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                # Dedicated post-process: crops preprocessing padding and resizes
                # each prediction back to its original (H, W); returns per-image
                # {"pointmap": (3, H, W)} in canonical camera space.
                results = self.processor.post_process_pointmap_estimation(
                    outputs, target_sizes=[(h, w)] * len(images)
                )
            pts = np.stack([
                r["pointmap"].permute(1, 2, 0).detach().cpu().numpy() for r in results
            ]).astype(np.float32)  # (N, H, W, 3)
            scales = np.asarray(
                outputs.scales.detach().cpu().numpy(), dtype=np.float32
            ).reshape(-1)
            return pts, scales


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


class PointmapEstimateTool(BaseCanvasNode):
    """Per-image 3D pointmap for a list of same-resolution images."""

    node_type: ClassVar[str] = "model_pointmap__estimate_pointmap"
    display_name: ClassVar[str] = "Pointmap: Sapiens"
    description: ClassVar[str] = (
        "Per-pixel 3D (X,Y,Z) pointmap at original resolution; base64-npy (N,H,W,3) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField("model_id", "text", "HF pointmap-estimation model repo id", default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "pointmap", "TEXT",
            'JSON envelope {"shape":[N,H,W,3],"dtype":"float32","b64":…,"model_id":…,"scales":[…]}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"pointmap": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)

        engine = _PointmapEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            shapes = {img.shape[:2] for img in images}
            if len(shapes) != 1:
                log.warning("Pointmap-estimator: mixed input resolutions %s — degrading", shapes)
                return "MIXED"
            out = engine.estimate(images)
            if out is None:
                return ""
            pts, scales = out
            buf = np.ascontiguousarray(pts, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "scales": [round(float(s), 6) for s in scales.tolist()],
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MIXED":
            self._self_log("degraded", "mixed input resolutions — split into uniform batches")
            return {"pointmap": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no pointmap (load failure or bad input)")
        return {"pointmap": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class PointmapNodeSet(BaseNodeSet):
    """Single-image 3D pointmap estimation (Sapiens) — server-mode FM nodeset."""

    name = "model_pointmap"
    description = (
        "Sapiens single-image pointmap estimation — dense per-pixel 3D (X,Y,Z) "
        "coordinates from one RGB frame on the shared ac-fm server"
    )
    # Stateless pointmap estimator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Sapiens2ForPointmapEstimation
    # is native there). Override with $POINTMAP_PYTHON; device via $POINTMAP_DEVICE
    # (auto → cuda).
    server_python = conda_env_python("ac-fm", "POINTMAP_PYTHON")

    def get_tools(self) -> list:
        return [PointmapEstimateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_pointmap ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
