from __future__ import annotations

"""SAM (Segment Anything) foundation-model nodeset — full series, server mode.

Pure single-step segmentation primitives. Five capability nodes, all of them
stateless one-shot forwards; everything procedural lives in the graph:

    model_sam__segment_points   point-prompted segmentation (+ optional
                                mask_input for outer-loop iterative refinement)
    model_sam__segment_box      box-prompted segmentation (single box or batch)
    model_sam__segment_auto     full-scene segmentation (engineered grid over
                                the point capability; original image only)
    model_sam__segment_text     concept/text-prompted segmentation (SAM 3
                                native — instance masks for a noun phrase)
    model_sam__embed_image      image-encoder features as an embedding envelope

Variant matrix (real backends only; ``variant`` + ``ckpt`` are node config,
engines live in a ``(variant, ckpt)`` registry so checkpoints coexist):

    variant  backing                              nodes
    sam1     segment_anything + local .pth ckpt   points / box / auto / embed
    sam2     transformers Sam2Model (SAM 2.1,     points / box / auto
             HF weights, ungated)
    sam3     transformers Sam3Model (HF weights,  text
             GATED — request access on the
             facebook/sam3 model page first;
             until granted the engine latches
             degraded)

Design rulings (2026-07-05, see the FM-nodeset design doc):
    - Variants are config, not node identity; swapping them never changes the
      graph (uniform masks envelope). Video tracking stays deferred until a
      real consumer exists (session state is not welcome on a shared server).
    - The server is fully stateless: no embedding cache, no sessions. Reuse is
      expressed as dataflow — prompted nodes accept EITHER a raw base64 image
      OR a previously computed embedding envelope on their ``image`` port, and
      emit the envelope they used, so a graph can stash it in a state
      container and skip the heavy encoder on later prompts. Embedding in/out
      is sam1-only for now (SAM 2/3 features are multi-level; envelope v2
      would be needed) — sam2 emits ``""`` and rejects envelope injection.
    - Iterative refinement belongs to the outer loop: every prompted candidate
      carries ``low_res_logits_b64``; feed one back into ``mask_input``.

Embedding envelope (TEXT JSON, byte-exact float32 buffer):
    {"b64", "shape", "dtype", "original_hw", "input_hw", "variant", "ckpt_id"}
    Injecting an envelope whose variant/ckpt_id mismatches the target engine
    raises — better a node error than a silently wrong mask.

Masks envelope (TEXT JSON):
    {"masks": [{"mask_index", "mask_b64" (PNG), "bbox_xyxy", "iou_score",
                "area", "low_res_logits_b64" (prompted only),
                "stability_score" (auto only)}],
     "count", "image_w", "image_h"}

Environment:
    Served from the shared ``ac-fm`` env. sam1 needs segment-anything
    (installed by scripts/install/install_ac_fm.sh); sam2/sam3 ride the
    resident transformers (>=5.13 ships Sam2/Sam3) — zero extra packages.
    SAM_PYTHON   — interpreter override for the server subprocess
    SAM_DEVICE   — cuda:0 | cpu (default: cuda:0)
    Default ckpts: sam1 data/habitat/checkpoints/sam/sam_vit_b.pth (model
    type inferred from the filename: vit_b / vit_l / vit_h); sam2
    facebook/sam2.1-hiera-base-plus; sam3 facebook/sam3 (both resolve as HF
    repo ids through the standard HF cache; local dirs also accepted).

This file must stay Python-3.8-parseable (override may point at an old env).

Load: POST /api/components/nodesets/model_sam/load?mode=server

last updated: 2026-07-05
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
from PIL import Image

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

logger = logging.getLogger(__name__)


def _find_repo_root() -> str:
    """Walk upward from this file until a dir containing ``data/`` is found.

    A fixed ``../../..`` breaks when this file is served from a workspace
    OVERLAY copy (different depth than frozen ``workspace/nodesets/model/…``);
    the auto_host subprocess also does not run with the repo root as CWD, so
    relative checkpoint paths must be anchored here (same helper as model_ram).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 7):
        cand = os.path.normpath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), "..", ".."))


_REPO_ROOT = _find_repo_root()


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


def _f32_b64(arr: np.ndarray) -> str:
    """base64 of a C-contiguous float32 buffer (byte-exact, no decimal trip)."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).decode("ascii")


def _as_json(value: Any) -> Any:
    """Accept either a JSON string or an already-decoded list/dict."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _patch_torchvision_nms() -> None:
    """Monkey-patch torchvision NMS with a pure-Python fallback.

    Needed when torch and torchvision versions are mismatched (e.g.
    torch 2.4 + torchvision 0.10) and the C++ NMS ops fail to load.
    SamAutomaticMaskGenerator calls ``torchvision.ops.batched_nms``,
    which in old torchvision goes through a JIT-traced wrapper — so we
    must patch ``batched_nms`` itself, not just ``nms``. A no-op on a
    healthy env (ac-fm: torch 2.8 + torchvision 0.23).
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


# ━━ Engines ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _SamEngineBase:
    """Shared engine shell: lazy load with a failure latch + single-flight lock.

    One instance per ``(variant, ckpt)`` (see ``_get_engine``). Engines hold
    only loaded weights — no cache, no sessions; the GPU section is
    single-flight per engine to bound VRAM under K concurrent eval workers.
    """

    def __init__(self, variant: str, ckpt: str) -> None:
        self.variant = variant
        self.ckpt = ckpt
        self.ckpt_id = os.path.basename(ckpt.rstrip("/"))
        self._model: Any = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        self._device = os.environ.get("SAM_DEVICE", "cuda:0")

    def ensure(self) -> bool:
        """Load the model once; latch on failure (no retry storm). Blocking."""
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
                self._load()
                self._loaded = True
                return True
            except Exception:
                logger.warning(
                    "model_sam: failed to load %s (%s) — latching degraded",
                    self.ckpt,
                    self.variant,
                    exc_info=True,
                )
                self._load_failed = True
                return False

    def _load(self) -> None:
        raise NotImplementedError

    def _reject_envelope_payload(self, image_payload: str) -> None:
        """Embedding injection is a sam1-only capability for now."""
        env = _sniff_embedding_envelope(image_payload)
        if env is not None:
            raise ValueError(
                "model_sam: embedding injection is supported for the sam1 variant "
                "only (envelope variant=%r, this engine=%r)"
                % (env.get("variant"), self.variant)
            )


class _Sam1Engine(_SamEngineBase):
    """SAM 1 via the official ``segment_anything`` package (local .pth ckpt).

    Every call builds a fresh ``SamPredictor`` (a thin wrapper), so no
    per-image state survives a call.
    """

    def _load(self) -> None:
        from segment_anything import sam_model_registry

        _patch_torchvision_nms()
        model_type = _infer_model_type(self.ckpt)
        logger.info(
            "model_sam: loading %s from %s on %s", model_type, self.ckpt, self._device
        )
        sam = sam_model_registry[model_type](checkpoint=self.ckpt)
        sam.to(device=self._device)
        self._model = sam
        logger.info(
            "model_sam: %s loaded (%.0fM params)",
            model_type,
            sum(p.numel() for p in sam.parameters()) / 1e6,
        )

    # -- predictor state ------------------------------------------------------

    def _new_predictor(self) -> Any:
        from segment_anything import SamPredictor

        return SamPredictor(self._model)

    def _set_state(self, predictor: Any, image_payload: str) -> "tuple[int, int]":
        """Set predictor state from a raw base64 image OR an embedding envelope.

        Returns (H, W) of the original image. Envelope variant/ckpt mismatch
        raises — never silently decode against the wrong engine.
        """
        env = _sniff_embedding_envelope(image_payload)
        if env is not None:
            if env.get("variant") != self.variant or env.get("ckpt_id") != self.ckpt_id:
                raise ValueError(
                    "model_sam: embedding envelope was computed by %r/%r but this node "
                    "is configured for %r/%r — rewire or align configs"
                    % (env.get("variant"), env.get("ckpt_id"), self.variant, self.ckpt_id)
                )
            import torch

            feats = np.frombuffer(
                base64.b64decode(env["b64"]), dtype=np.float32
            ).reshape(env["shape"]).copy()
            predictor.features = torch.from_numpy(feats).to(self._device)
            predictor.original_size = tuple(env["original_hw"])
            predictor.input_size = tuple(env["input_hw"])
            predictor.is_image_set = True
            return (int(env["original_hw"][0]), int(env["original_hw"][1]))
        image = _decode_image(image_payload)
        predictor.set_image(image)
        return (int(image.shape[0]), int(image.shape[1]))

    def _embedding_envelope(self, predictor: Any) -> str:
        feats = predictor.features.detach().to("cpu").numpy()
        return json.dumps(
            {
                "b64": _f32_b64(feats),
                "shape": [int(x) for x in feats.shape],
                "dtype": "float32",
                "original_hw": [int(x) for x in predictor.original_size],
                "input_hw": [int(x) for x in predictor.input_size],
                "variant": self.variant,
                "ckpt_id": self.ckpt_id,
            }
        )

    # -- capabilities (sync, called via run_in_executor) ----------------------

    def predict_points(
        self,
        image_payload: str,
        points: list,
        labels: list,
        mask_input_b64: str,
        multimask: bool,
    ) -> "tuple[dict, str]":
        with self._lock:
            predictor = self._new_predictor()
            hw = self._set_state(predictor, image_payload)
            kwargs = {}
            if mask_input_b64:
                kwargs["mask_input"] = (
                    np.frombuffer(base64.b64decode(mask_input_b64), dtype=np.float32)
                    .reshape(1, 256, 256)
                    .copy()
                )
            masks, scores, low_res = predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=multimask,
                **kwargs,
            )
            emb_env = self._embedding_envelope(predictor)
        cands = []
        for i in range(len(scores)):
            m = masks[i]
            cands.append(
                {
                    "mask_index": i,
                    "mask_b64": _encode_mask(m),
                    "bbox_xyxy": _mask_to_bbox_xyxy(m),
                    "iou_score": round(float(scores[i]), 4),
                    "area": int(m.sum()),
                    "low_res_logits_b64": _f32_b64(low_res[i]),
                }
            )
        cands.sort(key=lambda x: x["iou_score"], reverse=True)
        return _masks_envelope(cands, hw), emb_env

    def predict_boxes(self, image_payload: str, boxes: list) -> "tuple[dict, str]":
        with self._lock:
            predictor = self._new_predictor()
            hw = self._set_state(predictor, image_payload)
            cands = []
            for i, box in enumerate(boxes):
                masks, scores, low_res = predictor.predict(
                    box=np.array(box, dtype=np.float32),
                    multimask_output=False,
                )
                m = masks[0]
                cands.append(
                    {
                        "mask_index": i,
                        "mask_b64": _encode_mask(m),
                        "bbox_xyxy": _mask_to_bbox_xyxy(m),
                        "iou_score": round(float(scores[0]), 4),
                        "area": int(m.sum()),
                        "low_res_logits_b64": _f32_b64(low_res[0]),
                    }
                )
            emb_env = self._embedding_envelope(predictor)
        return _masks_envelope(cands, hw), emb_env

    def auto_masks(
        self,
        image_b64: str,
        points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        min_mask_area: int,
        crop_n_layers: int,
    ) -> dict:
        from segment_anything import SamAutomaticMaskGenerator

        image = _decode_image(image_b64)
        with self._lock:
            # A fresh generator per call: its knobs come fully from node config
            # (the old shared instance mutated params across concurrent calls).
            gen = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_area,
                crop_n_layers=crop_n_layers,
            )
            raw_masks = gen.generate(image)
        cands = []
        for i, entry in enumerate(raw_masks):
            seg = entry["segmentation"]  # (H, W) bool
            # SAM auto returns bbox as [x, y, w, h] — convert to [x1, y1, x2, y2]
            bx, by, bw, bh = entry["bbox"]
            cands.append(
                {
                    "mask_index": i,
                    "mask_b64": _encode_mask(seg),
                    "bbox_xyxy": [int(bx), int(by), int(bx + bw), int(by + bh)],
                    "iou_score": round(float(entry["predicted_iou"]), 4),
                    "area": int(entry["area"]),
                    "stability_score": round(float(entry["stability_score"]), 4),
                }
            )
        cands.sort(key=lambda x: x["area"], reverse=True)
        return _masks_envelope(cands, (int(image.shape[0]), int(image.shape[1])))

    def embed(self, image_b64: str) -> str:
        with self._lock:
            predictor = self._new_predictor()
            self._set_state(predictor, image_b64)
            return self._embedding_envelope(predictor)


class _Sam2Engine(_SamEngineBase):
    """SAM 2.1 image path via transformers (``Sam2Model`` + ``Sam2Processor``).

    The HF implementation, not the facebookresearch/sam2 package — the
    resident transformers (>=5.13) ships it, so the variant costs zero new
    dependencies. Point/box prompts + the mask-generation pipeline for auto;
    no embedding in/out (SAM 2 features are multi-level — envelope v2 first).
    """

    def _load(self) -> None:
        from transformers import Sam2Model, Sam2Processor

        logger.info("model_sam: loading sam2 from %s on %s", self.ckpt, self._device)
        self._processor = Sam2Processor.from_pretrained(self.ckpt)
        self._model = Sam2Model.from_pretrained(self.ckpt).to(self._device).eval()
        self._auto_pipe = None
        logger.info(
            "model_sam: sam2 loaded (%.0fM params)",
            sum(p.numel() for p in self._model.parameters()) / 1e6,
        )

    def _post_process(self, outputs: Any, inputs: Any) -> Any:
        args = [outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()]
        if "reshaped_input_sizes" in inputs:
            try:
                return self._processor.post_process_masks(
                    *args, inputs["reshaped_input_sizes"].cpu()
                )[0]
            except TypeError:
                pass
        return self._processor.post_process_masks(*args)[0]

    def _predict(
        self,
        image_payload: str,
        points: "list | None" = None,
        labels: "list | None" = None,
        boxes: "list | None" = None,
        mask_input_b64: str = "",
        multimask: bool = True,
    ) -> "tuple[list, tuple[int, int]]":
        import torch

        self._reject_envelope_payload(image_payload)
        image = _decode_image(image_payload)
        kwargs = {}
        if points is not None:
            kwargs["input_points"] = [[points]]
            kwargs["input_labels"] = [[labels]]
        if boxes is not None:
            kwargs["input_boxes"] = [boxes]
        inputs = self._processor(images=image, return_tensors="pt", **kwargs).to(self._device)
        fwd = {"multimask_output": multimask}
        if mask_input_b64:
            logits = (
                np.frombuffer(base64.b64decode(mask_input_b64), dtype=np.float32)
                .reshape(1, 1, 256, 256)
                .copy()
            )
            fwd["input_masks"] = torch.from_numpy(logits).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, **fwd)
        masks_full = self._post_process(outputs, inputs)  # (n_obj, n_masks, H, W)
        ious = outputs.iou_scores.cpu().numpy()[0]  # (n_obj, n_masks)
        low_res = outputs.pred_masks.float().cpu().numpy()[0]  # (n_obj, n_masks, 256, 256)
        cands = []
        idx = 0
        for o in range(masks_full.shape[0]):
            for m in range(masks_full.shape[1]):
                mask = np.asarray(masks_full[o, m]).astype(bool)
                if not mask.any():
                    continue
                cands.append(
                    {
                        "mask_index": idx,
                        "mask_b64": _encode_mask(mask),
                        "bbox_xyxy": _mask_to_bbox_xyxy(mask),
                        "iou_score": round(float(ious[o, m]), 4),
                        "area": int(mask.sum()),
                        "low_res_logits_b64": _f32_b64(low_res[o, m]),
                    }
                )
                idx += 1
        return cands, (int(image.shape[0]), int(image.shape[1]))

    def predict_points(
        self,
        image_payload: str,
        points: list,
        labels: list,
        mask_input_b64: str,
        multimask: bool,
    ) -> "tuple[dict, str]":
        with self._lock:
            cands, hw = self._predict(
                image_payload,
                points=points,
                labels=labels,
                mask_input_b64=mask_input_b64,
                multimask=multimask,
            )
        cands.sort(key=lambda x: x["iou_score"], reverse=True)
        return _masks_envelope(cands, hw), ""

    def predict_boxes(self, image_payload: str, boxes: list) -> "tuple[dict, str]":
        with self._lock:
            cands, hw = self._predict(image_payload, boxes=boxes, multimask=False)
        return _masks_envelope(cands, hw), ""

    def auto_masks(
        self,
        image_b64: str,
        points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        min_mask_area: int,
        crop_n_layers: int,
    ) -> dict:
        import inspect

        from PIL import Image as PILImage
        from transformers import pipeline

        if int(crop_n_layers) > 0:
            # transformers 5.13 upstream bug: the sam2 mask-generation
            # pipeline torch.stack()s unequal-sized crops and crashes.
            raise ValueError(
                "sam2 auto does not support crop_n_layers>0 (transformers "
                "pipeline bug: unequal crop sizes crash preprocessing) — "
                "use variant=sam1 for cropped auto-segmentation"
            )
        image = _decode_image(image_b64)
        with self._lock:
            if self._auto_pipe is None:
                self._auto_pipe = pipeline(
                    "mask-generation",
                    model=self._model,
                    image_processor=self._processor.image_processor,
                    device=self._device,
                )
            call_kwargs = {
                "points_per_crop": int(points_per_side),
                "pred_iou_thresh": float(pred_iou_thresh),
                "stability_score_thresh": float(stability_score_thresh),
            }
            accepted: set = set()
            for fn_name in ("_sanitize_parameters", "preprocess", "_forward", "postprocess"):
                try:
                    accepted |= set(
                        inspect.signature(getattr(self._auto_pipe, fn_name)).parameters
                    )
                except (AttributeError, ValueError):
                    pass
            dropped = [k for k in call_kwargs if k not in accepted]
            for k in dropped:
                call_kwargs.pop(k)
            if dropped:
                logger.warning(
                    "model_sam: sam2 auto pipeline does not accept %s — using its defaults",
                    dropped,
                )
            out = self._auto_pipe(PILImage.fromarray(image), **call_kwargs)
        masks = out.get("masks") or []
        scores = out.get("scores")
        if scores is None:
            scores = [0.0] * len(masks)
        cands = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask = np.asarray(mask).astype(bool)
            area = int(mask.sum())
            if area == 0 or area < int(min_mask_area):
                continue
            cands.append(
                {
                    "mask_index": i,
                    "mask_b64": _encode_mask(mask),
                    "bbox_xyxy": _mask_to_bbox_xyxy(mask),
                    "iou_score": round(float(score), 4),
                    "area": area,
                    # no stability_score: the HF pipeline does not expose it
                }
            )
        cands.sort(key=lambda x: x["area"], reverse=True)
        return _masks_envelope(cands, (int(image.shape[0]), int(image.shape[1])))


class _Sam3Engine(_SamEngineBase):
    """SAM 3 concept/text segmentation via transformers (``Sam3Model``).

    Weights are HF-gated (facebook/sam3, manual approval): until access is
    granted on the model page, ``from_pretrained`` 403s and the engine
    latches degraded (empty envelopes, logged).
    """

    def _load(self) -> None:
        from transformers import Sam3Model, Sam3Processor

        logger.info("model_sam: loading sam3 from %s on %s", self.ckpt, self._device)
        self._processor = Sam3Processor.from_pretrained(self.ckpt)
        self._model = Sam3Model.from_pretrained(self.ckpt).to(self._device).eval()
        logger.info(
            "model_sam: sam3 loaded (%.0fM params)",
            sum(p.numel() for p in self._model.parameters()) / 1e6,
        )

    def segment_text(
        self, image_b64: str, text: str, score_thresh: float, mask_threshold: float
    ) -> dict:
        import torch

        image = _decode_image(image_b64)
        with self._lock:
            inputs = self._processor(images=image, text=text, return_tensors="pt").to(
                self._device
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            # target_sizes must be passed explicitly — the processor output
            # carries no original_sizes, and without it masks come back at
            # model resolution instead of the original image size.
            res = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=float(score_thresh),
                mask_threshold=float(mask_threshold),
                target_sizes=[(int(image.shape[0]), int(image.shape[1]))],
            )[0]
        masks = res.get("masks")
        scores = res.get("scores")
        boxes = res.get("boxes")

        def _np(x: Any) -> np.ndarray:
            return np.asarray(x.cpu() if hasattr(x, "cpu") else x)

        cands = []
        for i in range(0 if masks is None else len(masks)):
            score = float(_np(scores[i])) if scores is not None else 0.0
            if score < float(score_thresh):
                continue
            mask = _np(masks[i]).astype(bool)
            if not mask.any():
                continue
            if boxes is not None:
                bb = [int(round(float(v))) for v in _np(boxes[i]).tolist()]
            else:
                bb = _mask_to_bbox_xyxy(mask)
            cands.append(
                {
                    "mask_index": i,
                    "mask_b64": _encode_mask(mask),
                    "bbox_xyxy": bb,
                    "iou_score": round(score, 4),
                    "area": int(mask.sum()),
                }
            )
        cands.sort(key=lambda x: x["iou_score"], reverse=True)
        return _masks_envelope(cands, (int(image.shape[0]), int(image.shape[1])))


# ━━ Engine registry ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ENGINE_CLASSES = {"sam1": _Sam1Engine, "sam2": _Sam2Engine, "sam3": _Sam3Engine}
_DEFAULT_CKPT = {
    "sam1": os.path.join(_REPO_ROOT, "data", "habitat", "checkpoints", "sam", "sam_vit_b.pth"),
    "sam2": "facebook/sam2.1-hiera-base-plus",
    "sam3": "facebook/sam3",
}
_ENGINES: dict = {}
_ENGINES_LOCK = threading.Lock()


def _get_engine(variant: str, ckpt: str = "") -> _SamEngineBase:
    """Lazy singleton per (variant, resolved ckpt) — checkpoints coexist."""
    if variant not in _ENGINE_CLASSES:
        raise ValueError(
            "model_sam: unknown variant %r (known: %s)" % (variant, sorted(_ENGINE_CLASSES))
        )
    resolved = ckpt or _DEFAULT_CKPT[variant]
    if variant == "sam1" and not os.path.isabs(resolved):
        # sam1 ckpts are local files; HF variants take repo ids — never anchor those.
        resolved = os.path.join(_REPO_ROOT, resolved)
    key = (variant, resolved)
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = _ENGINE_CLASSES[variant](variant, resolved)
        return _ENGINES[key]


def _sniff_embedding_envelope(payload: str) -> "dict | None":
    """Return the parsed envelope if payload is one, else None (raw image b64).

    An embedding envelope is a JSON object carrying at least b64/shape/variant;
    a raw base64 PNG never parses as a JSON object, so the sniff is exact.
    """
    s = payload.lstrip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict) and "b64" in obj and "shape" in obj and "variant" in obj:
        return obj
    return None


def _masks_envelope(cands: list, hw: "tuple[int, int]") -> dict:
    return {
        "masks": cands,
        "count": len(cands),
        "image_w": hw[1],
        "image_h": hw[0],
    }


def _engine_from_config(node: BaseCanvasNode, default_variant: str = "sam1") -> _SamEngineBase:
    variant = str(node.config.get("variant", default_variant) or default_variant)
    ckpt = str(node.config.get("ckpt", "") or "").strip()
    return _get_engine(variant, ckpt)


_SAM1_OPT = {"value": "sam1", "label": "SAM 1 (vit_b/l/h by ckpt)"}
_SAM2_OPT = {"value": "sam2", "label": "SAM 2.1 (HF hiera)"}
_SAM3_OPT = {"value": "sam3", "label": "SAM 3 (HF, gated weights)"}


def _variant_fields(options: list, default: str) -> list:
    return [
        ConfigField(
            "variant",
            "select",
            label="SAM variant",
            options=list(options),
            default=default,
        ),
        ConfigField(
            "ckpt",
            "text",
            label="Checkpoint override (sam1: .pth path; sam2/sam3: HF repo id or local dir)",
            default="",
        ),
    ]


# ━━ Canvas nodes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SamSegmentPointsTool(BaseCanvasNode):
    """Point-prompted segmentation — pure single-step forward.

    Iterative refinement is the OUTER loop's job: each candidate carries
    ``low_res_logits_b64``; wire one back into ``mask_input`` to close the
    refinement cycle at graph level.
    """

    node_type = "model_sam__segment_points"
    display_name = "SAM: Segment (points)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=_variant_fields([_SAM1_OPT, _SAM2_OPT], "sam1")
        + [
            ConfigField(
                "multimask",
                "toggle",
                label="Return 3 ranked candidates (off = single best)",
                default=True,
            )
        ],
    )
    description = (
        "Segment objects at specified point(s). Foreground points (label=1) on "
        "the target, background points (label=0) to exclude regions. The image "
        "port accepts a raw base64 PNG or a model_sam embedding envelope "
        "(embedding injection: sam1 variant only)."
    )
    category = "tool"
    icon = "Crosshair"
    input_ports = [
        PortDef("image", "TEXT", "Base64 PNG image OR model_sam embedding envelope"),
        PortDef("points", "TEXT", "JSON list of [x, y] pixel coordinates"),
        PortDef("labels", "TEXT", "JSON list of per-point labels: 1=foreground, 0=background"),
        PortDef(
            "mask_input",
            "TEXT",
            "Optional low_res_logits_b64 from a previous round (iterative refinement)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("masks", "TEXT", "Masks envelope JSON (each candidate carries logits)"),
        PortDef("image_embedding", "TEXT", "Embedding envelope used for this call"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        engine = _engine_from_config(self)
        points = _as_json(inputs["points"])
        labels = _as_json(inputs["labels"])
        mask_input_b64 = str(inputs.get("mask_input") or "")
        multimask = bool(self.config.get("multimask", True))
        image_payload = inputs["image"]
        loop = asyncio.get_running_loop()

        def _run() -> "tuple[dict, str] | None":
            if not engine.ensure():
                return None
            return engine.predict_points(image_payload, points, labels, mask_input_b64, multimask)

        out = await loop.run_in_executor(None, _run)
        if out is None:
            self._self_log("degraded", "SAM engine failed to load")
            return {"masks": "", "image_embedding": ""}
        envelope, emb_env = out
        self._self_log("count", envelope["count"])
        return {"masks": json.dumps(envelope, ensure_ascii=False), "image_embedding": emb_env}


class SamSegmentBoxTool(BaseCanvasNode):
    """Box-prompted segmentation — single box or a batch of boxes."""

    node_type = "model_sam__segment_box"
    display_name = "SAM: Segment (box)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet", config_fields=_variant_fields([_SAM1_OPT, _SAM2_OPT], "sam1")
    )
    description = (
        "Segment the primary object inside each bounding box. Accepts a single "
        "[x1,y1,x2,y2] box or a list of boxes (one candidate per box). The "
        "image port accepts a raw base64 PNG or a model_sam embedding envelope."
    )
    category = "tool"
    icon = "Square"
    input_ports = [
        PortDef("image", "TEXT", "Base64 PNG image OR model_sam embedding envelope"),
        PortDef("boxes", "TEXT", "JSON [x1,y1,x2,y2] or list of such boxes"),
    ]
    output_ports = [
        PortDef("masks", "TEXT", "Masks envelope JSON (one candidate per box)"),
        PortDef("image_embedding", "TEXT", "Embedding envelope used for this call"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        engine = _engine_from_config(self)
        boxes = _as_json(inputs["boxes"])
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]  # single [x1,y1,x2,y2] -> batch of one
        image_payload = inputs["image"]
        loop = asyncio.get_running_loop()

        def _run() -> "tuple[dict, str] | None":
            if not engine.ensure():
                return None
            return engine.predict_boxes(image_payload, boxes)

        out = await loop.run_in_executor(None, _run)
        if out is None:
            self._self_log("degraded", "SAM engine failed to load")
            return {"masks": "", "image_embedding": ""}
        envelope, emb_env = out
        self._self_log("count", envelope["count"])
        return {"masks": json.dumps(envelope, ensure_ascii=False), "image_embedding": emb_env}


class SamSegmentAutoTool(BaseCanvasNode):
    """Automatic full-scene segmentation (grid-batched point prompts)."""

    node_type = "model_sam__segment_auto"
    display_name = "SAM: Segment (auto)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=_variant_fields([_SAM1_OPT, _SAM2_OPT], "sam1")
        + [
            ConfigField(
                "points_per_side", "slider", label="Points/side", default=32, min=8, max=64, step=8
            ),
            ConfigField("pred_iou_thresh", "text", label="Pred-IoU threshold", default=0.86),
            ConfigField(
                "stability_score_thresh", "text", label="Stability threshold", default=0.92
            ),
            ConfigField("min_mask_area", "text", label="Min mask area (px)", default=100),
            ConfigField(
                "crop_n_layers",
                "slider",
                label="Crop layers (sam1 only; 0 = whole image; >0 finds smaller objects, slower)",
                default=0,
                min=0,
                max=3,
                step=1,
            ),
        ],
    )
    description = (
        "Segment everything in the image with a point grid (masks sorted by "
        "area). Original image only — the generator re-encodes internally, so "
        "it cannot consume a precomputed embedding."
    )
    category = "tool"
    icon = "Layers"
    input_ports = [PortDef("image_b64", "TEXT", "Base64-encoded PNG image")]
    output_ports = [PortDef("masks", "TEXT", "Masks envelope JSON (area-sorted)")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        engine = _engine_from_config(self)
        pps = int(self.config.get("points_per_side", 32))
        iou_th = float(self.config.get("pred_iou_thresh", 0.86))
        stab_th = float(self.config.get("stability_score_thresh", 0.92))
        min_area = int(float(self.config.get("min_mask_area", 100)))
        crop_layers = int(float(self.config.get("crop_n_layers", 0)))
        image_b64 = inputs["image_b64"]
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            if not engine.ensure():
                return None
            return engine.auto_masks(image_b64, pps, iou_th, stab_th, min_area, crop_layers)

        envelope = await loop.run_in_executor(None, _run)
        if envelope is None:
            self._self_log("degraded", "SAM engine failed to load")
            return {"masks": ""}
        self._self_log("count", envelope["count"])
        return {"masks": json.dumps(envelope, ensure_ascii=False)}


class SamSegmentTextTool(BaseCanvasNode):
    """Concept/text-prompted segmentation — SAM 3 native.

    Returns every instance matching a short noun-phrase concept. The
    graph-level GDINO→segment_box composition remains available as the
    non-SAM3 route; this node is the native one.
    """

    node_type = "model_sam__segment_text"
    display_name = "SAM: Segment (text)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=_variant_fields([_SAM3_OPT], "sam3")
        + [
            ConfigField(
                "score_thresh", "text", label="Instance score threshold", default=0.5
            ),
            ConfigField(
                "mask_threshold",
                "text",
                label="Mask binarization threshold (per-pixel)",
                default=0.5,
            ),
        ],
    )
    description = (
        "Segment every instance matching a short noun-phrase concept (e.g. "
        "'red circle'). SAM 3 native text prompting; weights are HF-gated "
        "(facebook/sam3) — until access is granted the engine runs degraded "
        "(empty envelope)."
    )
    category = "tool"
    icon = "Type"
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64-encoded PNG image"),
        PortDef("text", "TEXT", "Noun-phrase concept to segment"),
    ]
    output_ports = [
        PortDef("masks", "TEXT", "Masks envelope JSON (score-sorted instances)")
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        engine = _engine_from_config(self, default_variant="sam3")
        image_b64 = inputs["image_b64"]
        text = str(inputs["text"] or "").strip()
        thresh = float(self.config.get("score_thresh", 0.5))
        mask_th = float(self.config.get("mask_threshold", 0.5))
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            if not engine.ensure():
                return None
            return engine.segment_text(image_b64, text, thresh, mask_th)

        envelope = await loop.run_in_executor(None, _run)
        if envelope is None:
            self._self_log("degraded", "SAM engine failed to load")
            return {"masks": ""}
        self._self_log("count", envelope["count"])
        return {"masks": json.dumps(envelope, ensure_ascii=False)}


class SamEmbedImageTool(BaseCanvasNode):
    """Image-encoder features as an embedding envelope (encode-once entry).

    Stash the envelope in a state container and feed it to the prompted
    nodes' ``image`` port to skip the heavy encoder on later prompts.
    """

    node_type = "model_sam__embed_image"
    display_name = "SAM: Embed Image"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet", config_fields=_variant_fields([_SAM1_OPT], "sam1")
    )
    description = (
        "Run only the SAM image encoder and return the embedding envelope "
        "(float32 features + sizes + variant/ckpt provenance). Reuse is a "
        "graph-level decision: store it, wire it back into prompted nodes."
    )
    category = "tool"
    icon = "ScanLine"
    input_ports = [PortDef("image_b64", "TEXT", "Base64-encoded PNG image")]
    output_ports = [PortDef("image_embedding", "TEXT", "Embedding envelope JSON")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        engine = _engine_from_config(self)
        image_b64 = inputs["image_b64"]
        loop = asyncio.get_running_loop()

        def _run() -> "str | None":
            if not engine.ensure():
                return None
            return engine.embed(image_b64)

        emb_env = await loop.run_in_executor(None, _run)
        if emb_env is None:
            self._self_log("degraded", "SAM engine failed to load")
            return {"image_embedding": ""}
        return {"image_embedding": emb_env}


# ━━ NodeSet ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SamNodeSet(BaseNodeSet):
    """SAM-family segmentation primitives, served from the shared ac-fm env.

    Stateless server: engines (keyed by variant+ckpt from node config) hold
    only loaded weights; no cache, no sessions. See module docstring for the
    envelope contracts and design rulings.
    """

    name = "model_sam"
    description = (
        "Segment Anything full series (SAM 1 / 2.1 / 3 as config variants) — "
        "point/box/auto/text segmentation + image embedding, pure single-step "
        "primitives on the shared ac-fm server"
    )
    parallelism = "shared"
    server_python = conda_env_python("ac-fm", "SAM_PYTHON")

    def get_tools(self) -> list[BaseCanvasNode]:
        return [
            SamSegmentPointsTool(),
            SamSegmentBoxTool(),
            SamSegmentAutoTool(),
            SamSegmentTextTool(),
            SamEmbedImageTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        logger.info(
            "model_sam ready (server_python=%s); engines load lazily per node config",
            self.server_python,
        )

    async def shutdown(self) -> None:
        # Loaded engines stay resident until subprocess teardown (house convention).
        logger.info("model_sam shutdown")
