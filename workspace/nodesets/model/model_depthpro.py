from __future__ import annotations

"""DepthPro metric depth — server-mode foundation-model nodeset.

Turns a single RGB frame into a dense **metric** depth map (metres) plus the
recovered field of view — the absolute-scale geometry primitive that Depth
Anything's relative/inverse depth cannot give. Obstacle distance in metres,
metric 3D lifting, sim2real scale, and camera-intrinsic recovery from a lone RGB
frame all want this; DepthPro is the foundation-model way to get it zero-shot.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_depthpro__estimate_metric_depth  (images: list[{rgb_base64}] | list[b64])
                                           → depth: TEXT envelope (N, H, W) float32

Depth envelope (the raw C-contiguous float32 buffer base64-encoded, byte-exact
across the HTTP boundary, ~4× smaller than a JSON float list)::

    {"shape":[N,H,W], "dtype":"float32", "b64":…, "model_id":…,
     "depth_type":"metric", "field_of_view":[fov_deg, …]}

Depth is in **metres** at the original input resolution; ``field_of_view`` gives
the horizontal FoV in degrees per image (DepthPro estimates focal length as part
of the forward). All images in one call must share resolution — the envelope is
a single (N,H,W) buffer; mixed sizes degrade to empty with a self-log (split into
uniform batches). In an agent loop N is typically 1 (per-frame depth).

Note: DepthPro runs at a fixed high internal resolution (1536²) regardless of
input size — a single forward is heavier than Depth Anything's, so this is the
"metric when you need it" companion, not a per-step replacement.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``AutoModelForDepthEstimation`` + ``AutoImageProcessor`` (``apple/DepthPro-hf``,
ungated). Override the env with $DEPTHPRO_PYTHON and the device with
$DEPTHPRO_DEVICE (auto → cuda when available). This file must stay
Python-3.8-parseable.

Load: POST /api/components/nodesets/model_depthpro/load?mode=server

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

log = logging.getLogger("agentcanvas.model_depthpro")

_MODEL_ID_DEFAULT = "apple/DepthPro-hf"


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("DEPTHPRO_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _DepthProEngine:
    """Lazy singleton registry: one frozen DepthPro per ``model_id``.

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
    def get(cls, model_id: str) -> "_DepthProEngine":
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
                log.warning("DepthPro load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("DepthPro ready (%s, device=%s)", self.model_id, self.device)
            return True

    def estimate(self, images: list) -> "tuple | None":
        """Batched forward over same-size HWC uint8 images.

        Returns ((N, H, W) float32 metric depth in metres, [fov_deg, …]) or None
        on load failure. Post-processing resizes depth back to the original
        (H, W); the caller has already enforced uniform input size.
        """
        if not self._ensure():
            return None
        import torch

        target_sizes = [img.shape[:2] for img in images]
        with self._infer_lock:
            pp = self.processor(images=images, return_tensors="pt")
            pixel_values = pp["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            results = self.processor.post_process_depth_estimation(
                outputs, target_sizes=target_sizes
            )
            depths = np.stack([
                r["predicted_depth"].detach().cpu().numpy().astype(np.float32) for r in results
            ])
            fovs = [round(float(r["field_of_view"]), 3) for r in results]
            return depths, fovs


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


class DepthProEstimateTool(BaseCanvasNode):
    """Per-image metric depth + recovered field of view."""

    node_type: ClassVar[str] = "model_depthpro__estimate_metric_depth"
    display_name: ClassVar[str] = "DepthPro: Metric Depth"
    description: ClassVar[str] = (
        "Metric (metres) depth + field of view per image; base64-npy (N,H,W) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Ruler"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField("model_id", "text", "HF DepthPro model repo id", default=_MODEL_ID_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "depth", "TEXT",
            'JSON envelope {"shape":[N,H,W],"dtype":"float32","b64":…,"model_id":…,'
            '"depth_type":"metric","field_of_view":[…]}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"depth": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)

        engine = _DepthProEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _images_from_input(items)
            if images is None:
                return ""
            shapes = {img.shape[:2] for img in images}
            if len(shapes) != 1:
                log.warning("DepthPro: mixed input resolutions %s — degrading", shapes)
                return "MIXED"
            out = engine.estimate(images)
            if out is None:
                return ""
            depth, fovs = out
            buf = np.ascontiguousarray(depth, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "depth_type": "metric",
                "field_of_view": fovs,
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


class DepthProNodeSet(BaseNodeSet):
    """DepthPro metric depth — server-mode FM nodeset."""

    name = "model_depthpro"
    description = (
        "Apple DepthPro zero-shot metric depth estimation — dense per-pixel depth "
        "in metres + recovered field of view on the shared ac-fm server"
    )
    # Stateless depth estimator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers DepthPro is native there).
    # Override with $DEPTHPRO_PYTHON; device via $DEPTHPRO_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "DEPTHPRO_PYTHON")

    def get_tools(self) -> list:
        return [DepthProEstimateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_depthpro ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
