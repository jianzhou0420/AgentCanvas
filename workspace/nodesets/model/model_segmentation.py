from __future__ import annotations

"""Universal segmentation (Mask2Former) — server-mode foundation-model nodeset.

Turns a single RGB frame into a dense labelled segmentation — the scene-parsing
primitive that complements SAM's class-agnostic instance masks. Where SAM answers
"where are the objects", Mask2Former answers "what is each pixel": a per-pixel
semantic label map, or a panoptic map that also separates instances. Semantic
maps for VLN, walkable-vs-obstacle reasoning, and grounding a class name to
image regions all want this.

Two pure single-step primitives (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_segmentation__semantic  (images: list[{rgb_base64}] | list[b64])
                                  → segmentation: TEXT envelope (N, H, W) int32
    model_segmentation__panoptic  (images: list[{rgb_base64}] | list[b64])
                                  → segmentation: TEXT envelope (N, H, W) int32

Semantic envelope — each pixel is a class id; ``id2label`` names every id::

    {"shape":[N,H,W], "dtype":"int32", "b64":…, "model_id":…,
     "id2label":{"0":"wall", …}}

Panoptic envelope — each pixel is a *segment* id (instance-aware); ``segments``
lists, per image, one dict per segment giving its class + score::

    {"shape":[N,H,W], "dtype":"int32", "b64":…, "model_id":…,
     "segments":[[{"id":1,"label_id":3,"label":"chair","score":0.98}, …], …]}

Maps are returned at the **original input resolution**. All images in one call
must share resolution — the envelope is a single (N,H,W) buffer; mixed sizes
degrade to empty with a self-log (split into uniform batches). In an agent loop
N is typically 1 (per-frame parse).

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``Mask2FormerForUniversalSegmentation`` + ``AutoImageProcessor``. Defaults are
ungated Swin-Tiny checkpoints (ADE20K semantic / COCO panoptic); swap the
model_id for a larger backbone. Override the env with $SEGMENTATION_PYTHON and
the device with $SEGMENTATION_DEVICE (auto → cuda when available). This file
must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_segmentation/load?mode=server

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

log = logging.getLogger("agentcanvas.model_segmentation")

_SEMANTIC_MODEL_DEFAULT = "facebook/mask2former-swin-tiny-ade-semantic"
_PANOPTIC_MODEL_DEFAULT = "facebook/mask2former-swin-tiny-coco-panoptic"


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("SEGMENTATION_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _SegmentationEngine:
    """Lazy singleton registry: one frozen Mask2Former per ``model_id``.

    Holds only loaded weights — no cache, no per-call state. The single-flight
    inference lock bounds peak VRAM to one in-flight forward under concurrent
    eval workers (house FM-engine template). A semantic and a panoptic checkpoint
    are distinct weights, so they map to distinct engine instances.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = None
        self.model = None
        self.processor = None
        self.id2label = {}
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str) -> "_SegmentationEngine":
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
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

                self.device = _resolve_device()
                model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
            except Exception as exc:
                log.warning("Mask2Former load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("Mask2Former ready (%s, device=%s)", self.model_id, self.device)
            return True

    def _forward(self, images: list) -> "Any | None":
        """Shared preprocess + forward; returns (outputs, target_sizes) or None."""
        if not self._ensure():
            return None
        import torch

        target_sizes = [img.shape[:2] for img in images]
        pp = self.processor(images=images, return_tensors="pt")
        pp = {k: v.to(self.device) for k, v in pp.items()}
        with torch.no_grad():
            outputs = self.model(**pp)
        return outputs, target_sizes

    def semantic(self, images: list) -> "np.ndarray | None":
        """Batched semantic segmentation → (N, H, W) int32 class-id maps."""
        with self._infer_lock:
            fwd = self._forward(images)
            if fwd is None:
                return None
            outputs, target_sizes = fwd
            maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            return np.stack([m.detach().cpu().numpy().astype(np.int32) for m in maps])

    def panoptic(self, images: list) -> "tuple | None":
        """Batched panoptic segmentation → ((N, H, W) int32 seg-id maps, segments)."""
        with self._infer_lock:
            fwd = self._forward(images)
            if fwd is None:
                return None
            outputs, target_sizes = fwd
            results = self.processor.post_process_panoptic_segmentation(
                outputs, target_sizes=target_sizes
            )
            maps = []
            segments = []
            for res in results:
                maps.append(res["segmentation"].detach().cpu().numpy().astype(np.int32))
                info = []
                for seg in res["segments_info"]:
                    label_id = int(seg["label_id"])
                    info.append({
                        "id": int(seg["id"]),
                        "label_id": label_id,
                        "label": self.id2label.get(label_id, str(label_id)),
                        "score": round(float(seg.get("score", 0.0)), 4),
                    })
                segments.append(info)
            return np.stack(maps), segments


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


def _uniform_or_flag(items: list) -> "list | str | None":
    """Decode + enforce uniform resolution. Returns images, 'MIXED', or None."""
    images = _images_from_input(items)
    if images is None:
        return None
    if len({img.shape[:2] for img in images}) != 1:
        return "MIXED"
    return images


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class SemanticSegmentTool(BaseCanvasNode):
    """Per-pixel semantic class-id map for a list of same-resolution images."""

    node_type: ClassVar[str] = "model_segmentation__semantic"
    display_name: ClassVar[str] = "Segmentation: Semantic"
    description: ClassVar[str] = (
        "Per-pixel semantic class-id map at original resolution; base64-npy (N,H,W) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "LayoutGrid"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField(
                "model_id", "text", "HF Mask2Former semantic model repo id",
                default=_SEMANTIC_MODEL_DEFAULT,
            ),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "segmentation", "TEXT",
            'JSON envelope {"shape":[N,H,W],"dtype":"int32","b64":…,"model_id":…,"id2label":{…}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"segmentation": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _SEMANTIC_MODEL_DEFAULT)
        engine = _SegmentationEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _uniform_or_flag(items)
            if images is None:
                return ""
            if images == "MIXED":
                return "MIXED"
            maps = engine.semantic(images)
            if maps is None:
                return ""
            buf = np.ascontiguousarray(maps, dtype=np.int32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "int32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "id2label": {str(k): v for k, v in engine.id2label.items()},
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MIXED":
            self._self_log("degraded", "mixed input resolutions — split into uniform batches")
            return {"segmentation": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no segmentation (load failure or bad input)")
        return {"segmentation": envelope}


class PanopticSegmentTool(BaseCanvasNode):
    """Instance-aware panoptic segment-id map + per-segment class list."""

    node_type: ClassVar[str] = "model_segmentation__panoptic"
    display_name: ClassVar[str] = "Segmentation: Panoptic"
    description: ClassVar[str] = (
        "Instance-aware panoptic map + per-segment classes; base64-npy (N,H,W) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Boxes"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField(
                "model_id", "text", "HF Mask2Former panoptic model repo id",
                default=_PANOPTIC_MODEL_DEFAULT,
            ),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (uniform size)"),
    ]
    output_ports = [
        PortDef(
            "segmentation", "TEXT",
            'JSON envelope {"shape":[N,H,W],"dtype":"int32","b64":…,"model_id":…,"segments":[[…]]}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"segmentation": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _PANOPTIC_MODEL_DEFAULT)
        engine = _SegmentationEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            images = _uniform_or_flag(items)
            if images is None:
                return ""
            if images == "MIXED":
                return "MIXED"
            out = engine.panoptic(images)
            if out is None:
                return ""
            maps, segments = out
            buf = np.ascontiguousarray(maps, dtype=np.int32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "int32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "model_id": model_id,
                "segments": segments,
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MIXED":
            self._self_log("degraded", "mixed input resolutions — split into uniform batches")
            return {"segmentation": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no segmentation (load failure or bad input)")
        return {"segmentation": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class SegmentationNodeSet(BaseNodeSet):
    """Universal segmentation (Mask2Former) — server-mode FM nodeset."""

    name = "model_segmentation"
    description = (
        "Mask2Former universal segmentation — semantic class-id maps and "
        "instance-aware panoptic maps from a single RGB frame on the shared ac-fm server"
    )
    # Stateless segmenter — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Mask2Former is native there).
    # Override with $SEGMENTATION_PYTHON; device via $SEGMENTATION_DEVICE
    # (auto → cuda).
    server_python = conda_env_python("ac-fm", "SEGMENTATION_PYTHON")

    def get_tools(self) -> list:
        return [SemanticSegmentTool(), PanopticSegmentTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_segmentation ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
