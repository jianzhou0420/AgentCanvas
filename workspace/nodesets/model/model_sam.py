from __future__ import annotations

"""SAM (Segment Anything Model) nodeset — SAM, SAM2, and SAM3 backends.

Provides visual segmentation and object tracking tools for VLN agents.
The agent can segment objects referenced in navigation instructions,
track landmarks across frames, and perform full-scene segmentation.

Backend selection (via environment variables):
    SAM_VERSION   = sam | sam2 | sam3   (default: sam2)
    SAM_MODEL_CFG = checkpoint path or model config name
    SAM_DEVICE    = cuda:0 | cpu       (default: cuda:0)

Capabilities by version:
    SAM   — point / box / auto-mask segmentation
    SAM2  — + video object tracking with memory bank
    SAM3  — + native text-prompted segmentation, 3D-aware masks
"""


import base64
import io
import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

logger = logging.getLogger(__name__)

_DEFAULT_SAM_CHECKPOINT = "data/habitat/checkpoints/sam/sam_vit_b.pth"


# ━━ Image helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG string to (H, W, 3) uint8 numpy array."""
    buf = io.BytesIO(base64.b64decode(image_b64))
    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean mask (H, W) to a base64 PNG string."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_bbox_xyxy(mask: np.ndarray) -> list[int]:
    """Convert a boolean mask to [x1, y1, x2, y2] bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = (
        int(np.where(rows)[0][[0, -1]].tolist()[0]),
        int(np.where(rows)[0][[0, -1]].tolist()[1]),
    )
    x1, x2 = (
        int(np.where(cols)[0][[0, -1]].tolist()[0]),
        int(np.where(cols)[0][[0, -1]].tolist()[1]),
    )
    return [x1, y1, x2, y2]


def _infer_model_type(model_cfg: str) -> str:
    """Infer SAM model type from checkpoint path or config string."""
    cfg = model_cfg.lower()
    if "vit_l" in cfg:
        return "vit_l"
    if "vit_h" in cfg:
        return "vit_h"
    return "vit_b"


def _patch_torchvision_nms() -> None:
    """Monkey-patch torchvision NMS with a pure-Python fallback.

    Needed when torch and torchvision versions are mismatched (e.g.
    torch 2.4 + torchvision 0.10) and the C++ NMS ops fail to load.
    SamAutomaticMaskGenerator calls ``torchvision.ops.batched_nms``,
    which in old torchvision goes through a JIT-traced wrapper — so we
    must patch ``batched_nms`` itself, not just ``nms``.
    """
    import torch

    try:
        import torchvision.ops

        torchvision.ops.nms(
            torch.zeros((0, 4), device="cpu"),
            torch.zeros(0, device="cpu"),
            0.5,
        )
    except RuntimeError:
        logger.warning("torchvision C++ NMS unavailable — installing pure-Python fallback")

        def _py_nms(
            boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
        ) -> torch.Tensor:
            """Pure-Python NMS (greedy, O(n^2)). All ops on CPU."""
            boxes_cpu = boxes.cpu()
            scores_cpu = scores.cpu()
            if boxes_cpu.numel() == 0:
                return torch.empty(0, dtype=torch.int64)
            x1, y1, x2, y2 = boxes_cpu[:, 0], boxes_cpu[:, 1], boxes_cpu[:, 2], boxes_cpu[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores_cpu.argsort(descending=True).tolist()
            keep: list[int] = []
            suppressed = [False] * len(scores_cpu)
            for i in order:
                if suppressed[i]:
                    continue
                keep.append(i)
                for j in order:
                    if suppressed[j] or j == i:
                        continue
                    xx1 = max(float(x1[i]), float(x1[j]))
                    yy1 = max(float(y1[i]), float(y1[j]))
                    xx2 = min(float(x2[i]), float(x2[j]))
                    yy2 = min(float(y2[i]), float(y2[j]))
                    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                    iou = inter / (float(areas[i]) + float(areas[j]) - inter + 1e-6)
                    if iou > iou_threshold:
                        suppressed[j] = True
            return torch.tensor(keep, dtype=torch.int64)

        def _py_batched_nms(
            boxes: torch.Tensor,
            scores: torch.Tensor,
            idxs: torch.Tensor,
            iou_threshold: float,
        ) -> torch.Tensor:
            """Pure-Python batched NMS — runs per-class NMS on CPU."""
            keep_indices: list[int] = []
            for class_id in torch.unique(idxs).tolist():
                curr_mask = idxs == class_id
                curr_idx = torch.where(curr_mask)[0]
                curr_keep = _py_nms(boxes[curr_idx], scores[curr_idx], iou_threshold)
                keep_indices.extend(curr_idx[curr_keep].tolist())
            # Sort by score descending
            keep_indices.sort(key=lambda i: -float(scores[i]))
            return torch.tensor(keep_indices, dtype=torch.int64, device=boxes.device)

        import torchvision.ops.boxes as _tv_boxes

        _tv_boxes.nms = _py_nms
        _tv_boxes.batched_nms = _py_batched_nms
        torchvision.ops.nms = _py_nms
        torchvision.ops.batched_nms = _py_batched_nms
        logger.info("Pure-Python NMS fallback installed")


# ━━ Backend Abstraction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SamBackend(ABC):
    """Abstract backend for SAM model variants.

    Each concrete backend declares its capabilities via class variables.
    Tools check these flags at execute-time and return descriptive errors
    when a feature is unsupported by the active backend.
    """

    version: ClassVar[str]
    supports_tracking: ClassVar[bool] = False
    supports_text_prompt: ClassVar[bool] = False

    @abstractmethod
    async def load(self, model_cfg: str, device: str) -> None:
        """Load model weights onto *device*."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Release model and free GPU memory."""
        ...

    # ── Core segmentation (all versions) ──

    @abstractmethod
    async def segment_points(
        self,
        image_b64: str,
        points: list[list[float]],
        labels: list[int],
        multimask: bool = True,
    ) -> dict: ...

    @abstractmethod
    async def segment_box(
        self,
        image_b64: str,
        box: list[float],
    ) -> dict: ...

    @abstractmethod
    async def auto_mask(
        self,
        image_b64: str,
        points_per_side: int = 32,
        min_mask_area: int = 100,
    ) -> dict: ...

    # ── Video tracking (SAM2+) ──

    async def track_init(
        self,
        image_b64: str,
        mask_index: int,
        points: list[list[float]] | None = None,
    ) -> dict:
        """Initialize object tracking on a segmented mask."""
        return {"error": f"{self.version} does not support tracking. Use sam2 or sam3."}

    async def track_propagate(
        self,
        image_b64: str,
        track_ids: list[str],
    ) -> dict:
        """Propagate tracked objects to a new frame."""
        return {"error": f"{self.version} does not support tracking. Use sam2 or sam3."}

    # ── Text-prompted segmentation (SAM3 / Grounded-SAM) ──

    async def segment_text(
        self,
        image_b64: str,
        text: str,
        threshold: float = 0.3,
    ) -> dict:
        """Segment objects matching a text description."""
        return {"error": f"{self.version} does not support text prompts natively. Use sam3."}


# ── Mock helpers ──


def _mock_segment_result(n_masks: int, prompt_type: str) -> dict:
    """Generate a realistic-looking segmentation result for testing."""
    masks = []
    for i in range(n_masks):
        x1, y1 = random.randint(10, 200), random.randint(10, 200)
        masks.append(
            {
                "mask_index": i,
                "bbox": [x1, y1, x1 + random.randint(40, 200), y1 + random.randint(40, 160)],
                "score": round(random.uniform(0.5, 1.0), 3),
                "area": random.randint(500, 50000),
                "stability_score": round(random.uniform(0.8, 1.0), 3),
            }
        )
    masks.sort(key=lambda m: m["score"], reverse=True)
    return {"masks": masks, "count": len(masks), "prompt_type": prompt_type}


def _mock_auto_result(points_per_side: int) -> dict:
    """Generate mock auto-mask results."""
    n = random.randint(5, min(20, points_per_side))
    masks = []
    for i in range(n):
        x1, y1 = random.randint(0, 300), random.randint(0, 200)
        masks.append(
            {
                "mask_index": i,
                "bbox": [x1, y1, x1 + random.randint(30, 180), y1 + random.randint(30, 160)],
                "score": round(random.uniform(0.3, 1.0), 3),
                "area": random.randint(200, 80000),
                "stability_score": round(random.uniform(0.7, 1.0), 3),
                "predicted_iou": round(random.uniform(0.6, 1.0), 3),
            }
        )
    masks.sort(key=lambda m: m["area"], reverse=True)
    return {"masks": masks, "count": len(masks)}


# ── SAM v1 Backend ──────────────────────────────────────────────────────────


class Sam1Backend(SamBackend):
    """Original Segment Anything Model (Meta, 2023).

    Point/box/auto segmentation. No tracking or text prompts.

    Uses ``segment_anything.sam_model_registry`` →
    ``SamPredictor`` / ``SamAutomaticMaskGenerator``.
    """

    version = "sam"
    supports_tracking = False
    supports_text_prompt = False

    def __init__(self) -> None:
        self._predictor: Any = None
        self._auto_gen: Any = None
        self._device: str = "cpu"

    async def load(self, model_cfg: str, device: str) -> None:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

        _patch_torchvision_nms()
        checkpoint = model_cfg if model_cfg != "default" else _DEFAULT_SAM_CHECKPOINT
        model_type = _infer_model_type(checkpoint)
        self._device = device

        logger.info("SAM v1: loading %s from %s on %s", model_type, checkpoint, device)
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)

        self._predictor = SamPredictor(sam)
        self._auto_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        logger.info(
            "SAM v1: model loaded (%s, %.0fM params)",
            model_type,
            sum(p.numel() for p in sam.parameters()) / 1e6,
        )

    async def unload(self) -> None:
        import torch

        logger.info("SAM v1: unloading")
        self._predictor = None
        self._auto_gen = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def segment_points(self, image_b64, points, labels, multimask=True):
        image = _decode_image(image_b64)
        self._predictor.set_image(image)

        masks, scores, _ = self._predictor.predict(
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=multimask,
        )

        result_masks = []
        for i in range(len(scores)):
            m = masks[i]  # (H, W) bool
            result_masks.append(
                {
                    "mask_index": i,
                    "bbox": _mask_to_bbox_xyxy(m),
                    "score": round(float(scores[i]), 4),
                    "area": int(m.sum()),
                    "mask_b64": _encode_mask(m),
                }
            )
        result_masks.sort(key=lambda x: x["score"], reverse=True)
        return {"masks": result_masks, "count": len(result_masks), "prompt_type": "point"}

    async def segment_box(self, image_b64, box):
        image = _decode_image(image_b64)
        self._predictor.set_image(image)

        masks, scores, _ = self._predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
        )

        result_masks = []
        for i in range(len(scores)):
            m = masks[i]
            result_masks.append(
                {
                    "mask_index": i,
                    "bbox": _mask_to_bbox_xyxy(m),
                    "score": round(float(scores[i]), 4),
                    "area": int(m.sum()),
                    "mask_b64": _encode_mask(m),
                }
            )
        return {"masks": result_masks, "count": len(result_masks), "prompt_type": "box"}

    async def auto_mask(self, image_b64, points_per_side=32, min_mask_area=100):
        image = _decode_image(image_b64)

        # Update generator params if they differ from defaults
        self._auto_gen.points_per_side = points_per_side
        self._auto_gen.min_mask_region_area = min_mask_area

        raw_masks = self._auto_gen.generate(image)

        result_masks = []
        for i, entry in enumerate(raw_masks):
            seg = entry["segmentation"]  # (H, W) bool
            # SAM auto returns bbox as [x, y, w, h] — convert to [x1, y1, x2, y2]
            bx, by, bw, bh = entry["bbox"]
            result_masks.append(
                {
                    "mask_index": i,
                    "bbox": [int(bx), int(by), int(bx + bw), int(by + bh)],
                    "score": round(float(entry["predicted_iou"]), 4),
                    "area": int(entry["area"]),
                    "stability_score": round(float(entry["stability_score"]), 4),
                    "mask_b64": _encode_mask(seg),
                }
            )
        result_masks.sort(key=lambda x: x["area"], reverse=True)
        return {"masks": result_masks, "count": len(result_masks)}


# ── SAM 2 Backend ───────────────────────────────────────────────────────────


class Sam2Backend(SamBackend):
    """SAM 2 — MOCK backend (not installed, requires Python 3.10+).

    Video-capable segmentation with memory attention (Meta, 2024).
    All methods return random placeholder data.

    Real integration: ``sam2.build_sam.build_sam2`` →
    ``SAM2ImagePredictor`` / ``SAM2VideoPredictor``.
    """

    version = "sam2"
    supports_tracking = True
    supports_text_prompt = False

    def __init__(self) -> None:
        self._predictor: Any = None
        self._video_predictor: Any = None
        self._tracks: dict[str, dict[str, Any]] = {}

    async def load(self, model_cfg: str, device: str) -> None:
        logger.info("SAM 2: loading model cfg=%s device=%s", model_cfg, device)
        # TODO: real integration
        # from sam2.build_sam import build_sam2, build_sam2_video_predictor
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # model = build_sam2(model_cfg, device=device)
        # self._predictor = SAM2ImagePredictor(model)
        # self._video_predictor = build_sam2_video_predictor(model_cfg, device=device)
        self._tracks = {}

    async def unload(self) -> None:
        logger.info("SAM 2: unloading")
        self._predictor = None
        self._video_predictor = None
        self._tracks = {}

    async def segment_points(self, image_b64, points, labels, multimask=True):
        return _mock_segment_result(3 if multimask else 1, "point")

    async def segment_box(self, image_b64, box):
        return _mock_segment_result(1, "box")

    async def auto_mask(self, image_b64, points_per_side=32, min_mask_area=100):
        return _mock_auto_result(points_per_side)

    async def track_init(self, image_b64, mask_index, points=None):
        track_id = f"trk_{uuid.uuid4().hex[:8]}"
        self._tracks[track_id] = {"frame_count": 0, "mask_index": mask_index}
        logger.info("SAM 2: initialized track %s (mask_index=%d)", track_id, mask_index)
        # TODO: self._video_predictor.add_new_points_or_box(...)
        return {
            "track_id": track_id,
            "status": "initialized",
            "initial_bbox": [100, 100, 200, 200],
        }

    async def track_propagate(self, image_b64, track_ids):
        # TODO: self._video_predictor.propagate_in_video(...)
        results = {}
        for tid in track_ids:
            state = self._tracks.get(tid)
            if state is not None:
                state["frame_count"] += 1
                results[tid] = {
                    "bbox": [
                        100 + random.randint(-5, 5),
                        100 + random.randint(-5, 5),
                        200 + random.randint(-5, 5),
                        200 + random.randint(-5, 5),
                    ],
                    "score": round(random.uniform(0.85, 0.99), 3),
                    "frame_count": state["frame_count"],
                    "status": "tracking",
                }
            else:
                results[tid] = {"status": "lost", "score": 0.0}
        active = sum(1 for r in results.values() if r["status"] == "tracking")
        return {"tracks": results, "active_count": active}


# ── SAM 3 Backend ───────────────────────────────────────────────────────────


class Sam3Backend(SamBackend):
    """SAM 3 — MOCK backend (not yet released).

    Planned extensions over SAM 2:
    - Built-in text encoder for text-to-mask (no Grounding DINO needed)
    - Depth-conditioned mask proposals for 3D-aware segmentation
    - Improved multi-object tracking with re-identification after occlusion

    All methods return random placeholder data.
    """

    version = "sam3"
    supports_tracking = True
    supports_text_prompt = True

    def __init__(self) -> None:
        self._model: Any = None
        self._tracks: dict[str, dict[str, Any]] = {}

    async def load(self, model_cfg: str, device: str) -> None:
        logger.info("SAM 3: loading model cfg=%s device=%s", model_cfg, device)
        # TODO: integrate when released
        self._tracks = {}

    async def unload(self) -> None:
        logger.info("SAM 3: unloading")
        self._model = None
        self._tracks = {}

    async def segment_points(self, image_b64, points, labels, multimask=True):
        return _mock_segment_result(3 if multimask else 1, "point")

    async def segment_box(self, image_b64, box):
        return _mock_segment_result(1, "box")

    async def auto_mask(self, image_b64, points_per_side=32, min_mask_area=100):
        return _mock_auto_result(points_per_side)

    async def track_init(self, image_b64, mask_index, points=None):
        track_id = f"trk_{uuid.uuid4().hex[:8]}"
        self._tracks[track_id] = {"frame_count": 0, "mask_index": mask_index}
        return {
            "track_id": track_id,
            "status": "initialized",
            "initial_bbox": [100, 100, 200, 200],
        }

    async def track_propagate(self, image_b64, track_ids):
        results = {}
        for tid in track_ids:
            state = self._tracks.get(tid)
            if state is not None:
                state["frame_count"] += 1
                results[tid] = {
                    "bbox": [
                        100 + random.randint(-5, 5),
                        100 + random.randint(-5, 5),
                        200 + random.randint(-5, 5),
                        200 + random.randint(-5, 5),
                    ],
                    "score": round(random.uniform(0.90, 0.99), 3),
                    "frame_count": state["frame_count"],
                    "status": "tracking",
                }
            else:
                results[tid] = {"status": "lost", "score": 0.0}
        active = sum(1 for r in results.values() if r["status"] == "tracking")
        return {"tracks": results, "active_count": active}

    async def segment_text(self, image_b64, text, threshold=0.3):
        # TODO: SAM 3 native text encoder
        n_matches = random.randint(1, 3)
        masks = []
        for i in range(n_matches):
            x1, y1 = random.randint(10, 200), random.randint(10, 200)
            masks.append(
                {
                    "mask_index": i,
                    "bbox": [x1, y1, x1 + random.randint(50, 200), y1 + random.randint(50, 160)],
                    "score": round(random.uniform(max(threshold, 0.4), 1.0), 3),
                    "area": random.randint(500, 20000),
                    "phrase_match": text,
                }
            )
        masks.sort(key=lambda m: m["score"], reverse=True)
        return {"masks": masks, "count": len(masks), "query": text}


# ── Backend registry ──

SAM_BACKENDS: dict[str, type[SamBackend]] = {
    "sam": Sam1Backend,
    "sam2": Sam2Backend,
    "sam3": Sam3Backend,
}


# ━━ Tool Definitions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Each tool holds a back-reference to the SamNodeSet (not the backend directly)
# because get_tools() is called BEFORE initialize() in the WorkspaceComponentRegistry
# lifecycle.  The backend is resolved lazily at execute-time.


class SamSegmentPointTool(BaseCanvasNode):
    """Point-prompted segmentation — all SAM versions."""

    node_type = "model_sam__segment_point"
    display_name = "SAM: Segment Point"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "Segment objects at specified point(s). Provide foreground points "
        "(label=1) on the target and background points (label=0) to exclude "
        "regions. Returns ranked candidate masks."
    )
    category = "tool"
    icon = "Crosshair"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image"),
        PortDef("points", "TEXT", "JSON list of [x, y] pixel coordinates"),
        PortDef("labels", "TEXT", "JSON list of per-point labels: 1=foreground, 0=background"),
        PortDef(
            "multimask",
            "TEXT",
            "Return 3 candidate masks ranked by score (default true)",
            optional=True,
        ),
    ]
    output_ports = [PortDef("result", "TEXT", "Segmentation result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.segment_points(
            image_b64=inputs["image_b64"],
            points=inputs["points"],
            labels=inputs["labels"],
            multimask=inputs.get("multimask", True),
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


class SamSegmentBoxTool(BaseCanvasNode):
    """Box-prompted segmentation — all SAM versions."""

    node_type = "model_sam__segment_box"
    display_name = "SAM: Segment Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "Segment the primary object within a bounding box. Useful when the "
        "target's approximate location is known from detection or instruction."
    )
    category = "tool"
    icon = "Square"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image"),
        PortDef("box", "TEXT", "JSON [x1, y1, x2, y2] bounding box in pixels"),
    ]
    output_ports = [PortDef("result", "TEXT", "Segmentation result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.segment_box(
            image_b64=inputs["image_b64"],
            box=inputs["box"],
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


class SamAutoMaskTool(BaseCanvasNode):
    """Automatic full-scene segmentation — all SAM versions."""

    node_type = "model_sam__auto_mask"
    display_name = "SAM: Auto Mask"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "points_per_side", "slider", label="Points/side", default=32, min=8, max=64, step=8
            )
        ],
    )
    description = (
        "Automatically segment all objects in the image. Returns masks sorted "
        "by area. Useful for scene understanding and discovering objects the "
        "agent can interact with."
    )
    category = "tool"
    icon = "Layers"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image"),
        PortDef(
            "points_per_side", "TEXT", "Grid density for point sampling (default 32)", optional=True
        ),
        PortDef(
            "min_mask_area",
            "TEXT",
            "Discard masks smaller than this pixel area (default 100)",
            optional=True,
        ),
    ]
    output_ports = [PortDef("result", "TEXT", "Segmentation result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.auto_mask(
            image_b64=inputs["image_b64"],
            points_per_side=inputs.get("points_per_side", 32),
            min_mask_area=inputs.get("min_mask_area", 100),
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


class SamSegmentTextTool(BaseCanvasNode):
    """Text-prompted segmentation — SAM3 native (MOCK: returns random data)."""

    node_type = "model_sam__segment_text"
    display_name = "[Mock] SAM: Segment Text"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "[MOCK] Segment objects matching a natural language description (e.g. "
        "'brown wooden chair'). Returns random placeholder masks. "
        "Real implementation requires SAM3 backend (not yet available)."
    )
    category = "tool"
    icon = "Type"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image"),
        PortDef("text", "TEXT", "Natural language description of the target object(s)"),
        PortDef("threshold", "TEXT", "Confidence threshold (default 0.3)", optional=True),
    ]
    output_ports = [PortDef("result", "TEXT", "Segmentation result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.segment_text(
            image_b64=inputs["image_b64"],
            text=inputs["text"],
            threshold=inputs.get("threshold", 0.3),
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


class SamTrackInitTool(BaseCanvasNode):
    """Initialize object tracking — SAM2+ only (MOCK: returns random data)."""

    node_type = "model_sam__track_init"
    display_name = "[Mock] SAM: Track Init"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "[MOCK] Start tracking a segmented object across subsequent frames. "
        "Returns placeholder track_id. Real implementation requires SAM2 or "
        "SAM3 backend (not yet installed)."
    )
    category = "tool"
    icon = "Focus"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image (initial frame)"),
        PortDef("mask_index", "TEXT", "Mask index from a prior segmentation result"),
        PortDef(
            "points",
            "TEXT",
            "Optional JSON list of [x, y] points to identify the object",
            optional=True,
        ),
    ]
    output_ports = [PortDef("result", "TEXT", "Track initialization result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.track_init(
            image_b64=inputs["image_b64"],
            mask_index=inputs["mask_index"],
            points=inputs.get("points"),
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


class SamTrackPropagateTool(BaseCanvasNode):
    """Propagate tracked objects to a new frame — SAM2+ only (MOCK: returns random data)."""

    node_type = "model_sam__track_propagate"
    display_name = "[Mock] SAM: Track Propagate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="teal")
    description = (
        "[MOCK] Feed a new frame to the tracker and get updated bboxes/scores "
        "for all tracked objects. Returns random placeholder data. Real "
        "implementation requires SAM2 or SAM3 backend (not yet installed)."
    )
    category = "tool"
    icon = "ScanLine"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image (new frame)"),
        PortDef("track_ids", "TEXT", "JSON list of track_id values from SamTrackInit"),
    ]
    output_ports = [PortDef("result", "TEXT", "Track propagation result JSON")]

    def __init__(self, nodeset: SamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        backend = self._nodeset._backend if self._nodeset else None
        if backend is None:
            return {"result": '{"error": "SAM nodeset not initialized"}'}
        result = await backend.track_propagate(
            image_b64=inputs["image_b64"],
            track_ids=inputs["track_ids"],
        )
        return {"result": json.dumps(result, ensure_ascii=False)}


# ━━ NodeSet ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SamNodeSet(BaseNodeSet):
    """Segment Anything Model nodeset — SAM / SAM2 / SAM3.

    All six nodes are always registered.  Nodes that require features
    unavailable in the active backend (e.g. tracking on SAM v1) return
    a descriptive error dict at execute-time.

    Configure via environment variables:
        SAM_VERSION   = sam | sam2 | sam3   (default: sam2)
        SAM_MODEL_CFG = checkpoint path     (default: "default")
        SAM_DEVICE    = cuda:0 | cpu        (default: cuda:0)
    """

    name = "model_sam"
    description = "SAM visual segmentation and object tracking (SAM / SAM2 / SAM3)"

    def __init__(self) -> None:
        self._backend: SamBackend | None = None

    async def initialize(self, **kwargs: Any) -> None:
        import os

        version = os.environ.get("SAM_VERSION", "sam")
        model_cfg = os.environ.get("SAM_MODEL_CFG", "default")
        device = os.environ.get("SAM_DEVICE", "cuda:0")

        backend_cls = SAM_BACKENDS.get(version)
        if backend_cls is None:
            raise ValueError(
                f"Unknown SAM_VERSION={version!r}. Choose from: {sorted(SAM_BACKENDS)}"
            )

        self._backend = backend_cls()
        await self._backend.load(model_cfg, device)
        logger.info(
            "SAM nodeset ready: version=%s device=%s tracking=%s text=%s",
            version,
            device,
            self._backend.supports_tracking,
            self._backend.supports_text_prompt,
        )

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.unload()
            self._backend = None
        logger.info("SAM nodeset shut down")

    def get_tools(self) -> list:
        return [
            SamSegmentPointTool(self),
            SamSegmentBoxTool(self),
            SamAutoMaskTool(self),
            SamSegmentTextTool(self),
            SamTrackInitTool(self),
            SamTrackPropagateTool(self),
        ]
