from __future__ import annotations

"""GroundingDINO open-vocabulary text->box detector — server-mode FM nodeset.

A thin, generic perception primitive: given a base64 RGB image and a text
prompt (e.g. ``"ground"``), return the detected bounding boxes. Built for the
AO-Planner port's Visual Affordances Prompting layer — AO-Planner uses
``GroundingDINO('ground', box_threshold=0.4, text_threshold=0.4)`` to detect
the navigable floor, then feeds those boxes to SAM (``model_sam__segment_box``)
for the ground mask (DiscussNav-style: detector → segmenter). Generic enough
that any other open-vocab-detection method can reuse it unchanged, which is why
it lives in ``model/`` (TODO #56 method/foundation-model boundary), like
``model_sam`` / ``model_ram``.

The single tool::

    model_grounding_dino__detect  (image_b64: TEXT, [text_prompt: TEXT])
        -> result: TEXT  (JSON {boxes:[{xyxy,score,phrase}], count, image_w, image_h, text_prompt})

Inference recipe is lifted verbatim from ``model_detany3d/__init__.py:287-331``
(``_convert_dino_image`` + ``_dino_predict``), which itself ports DetAny3D
``app_mp.py:94-105, 171-187``: the standard GroundingDINO transform
(RandomResize 800 / ImageNet norm), ``groundingdino.util.inference.predict``,
then ``box_convert`` from normalized cxcywh to pixel xyxy.

Runs **server mode** (own subprocess + CUDA context) so the parent eval holds
no GroundingDINO VRAM and worker pools coalesce onto one shared server.

ENV CHOICE: reuses the ``detany3d`` conda env, which already has
``groundingdino-py`` 0.4.0 (torch 2.1.2+cu118) + the bundled
``GroundingDINO_SwinT_OGC`` config. This is a conscious, reversible deviation
from the dedicated-env-per-model norm (reusing the env that already *is* the
GroundingDINO host avoids a redundant multi-GB build); override with
``$GROUNDING_DINO_PYTHON`` to point at a dedicated env. Backbone is **Swin-T
OGC — AO-Planner's exact detector** (``data/detany3d/weights/
groundingdino_swint_ogc.pth``); set ``$GROUNDING_DINO_WEIGHTS`` to a Swin-B
checkpoint (+ ``$GROUNDING_DINO_CONFIG``) for the stronger backbone. Thresholds
default to AO-Planner's 0.4 / 0.4, caption ``"ground"`` (C-2 fidelity alignment
2026-06-17: was Swin-B @ 0.25 / "floor . ground .").

BACKEND SELECTION (2026-07-04, TODO #56 sweep): ``$GROUNDING_DINO_BACKEND``
picks the implementation at load time (SAM-style env-var selection):

    native   (default)  groundingdino-py + Swin-T OGC ckpt, ``ac-detany3d`` env
    hf_tiny             HF transformers ``IDEA-Research/grounding-dino-tiny``
                        (the variant the retired navgpt open_vocab_detect node
                        ran). Requires transformers>=4.40 — point
                        ``$GROUNDING_DINO_PYTHON`` at the ``agentcanvas`` env
                        (4.45.2) for this backend; ac-detany3d predates it.

Both backends emit the same ``result`` JSON schema, so graphs are
backend-agnostic; NavGPT-style post-processing lives in the pure
``navgpt_mp3d_tools__format_detections`` node.

Load:  POST /api/components/nodesets/model_grounding_dino/load

last updated: 2026-07-04
"""

import asyncio
import base64
import concurrent.futures
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

log = logging.getLogger("agentcanvas.model_grounding_dino")

# ── Weights + config resolution (reuse the detany3d data dir) ──────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# workspace/nodesets/model/model_grounding_dino.py -> ../../../ == repo root
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
_DEFAULT_WEIGHTS = os.path.join(
    _REPO_ROOT, "data", "detany3d", "weights", "groundingdino_swint_ogc.pth"
)
_GROUNDINGDINO_WEIGHTS = os.environ.get("GROUNDING_DINO_WEIGHTS", _DEFAULT_WEIGHTS)
# SAM ViT-H (AO-Planner's exact SAM variant) — present in the detany3d data dir.
_DEFAULT_SAM_VIT_H = os.path.join(_REPO_ROOT, "data", "detany3d", "weights", "sam_vit_h_4b8939.pth")
_SAM_VIT_H_WEIGHTS = os.environ.get("GROUNDING_DINO_SAM_WEIGHTS", _DEFAULT_SAM_VIT_H)

# AO-Planner defaults (llm/run_grounded_sam.sh: box/text_threshold 0.4).
_DEFAULT_TEXT_PROMPT = "ground"
_DEFAULT_BOX_THRESHOLD = 0.4
_DEFAULT_TEXT_THRESHOLD = 0.4

# Backend selection: "native" (groundingdino-py, Swin-T OGC) | "hf_tiny"
# (HF transformers grounding-dino-tiny; needs transformers>=4.40).
_BACKEND = os.environ.get("GROUNDING_DINO_BACKEND", "native")
_HF_MODEL_ID = os.environ.get("GROUNDING_DINO_HF_MODEL", "IDEA-Research/grounding-dino-tiny")

# Lazy singleton (per server subprocess) + single-thread executor for GPU
# affinity (mirrors DetAny3DEnvManager's executor pattern).
_model = None
_device = None
_hf_model = None
_hf_processor = None
_sam_predictor = None
_load_lock = threading.Lock()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gdino")


def _resolve_groundingdino_config() -> str:
    """Locate the GroundingDINO config in the pip-installed package.

    Defaults to ``GroundingDINO_SwinT_OGC.py`` (AO-Planner's exact backbone);
    ``$GROUNDING_DINO_CONFIG`` overrides the basename (e.g.
    ``GroundingDINO_SwinB_cfg.py`` for the stronger Swin-B). Adapted from
    ``model_detany3d/__init__.py:113-127`` (which pinned Swin-B).
    """
    cfg_name = os.environ.get("GROUNDING_DINO_CONFIG", "GroundingDINO_SwinT_OGC.py")
    try:
        import groundingdino  # type: ignore[import-not-found]

        gd_pkg = os.path.dirname(os.path.abspath(groundingdino.__file__))
        candidate = os.path.join(gd_pkg, "config", cfg_name)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        f"{cfg_name} not found in installed groundingdino package. "
        "Run scripts/install/install_ac_detany3d.sh to install groundingdino-py, or set "
        "$GROUNDING_DINO_PYTHON to an env that has it."
    )


def _ensure_model():
    """Lazy-load GroundingDINO (Swin-T OGC by default) once per server subprocess."""
    global _model, _device
    if _model is not None:
        return _model, _device
    with _load_lock:
        if _model is not None:
            return _model, _device
        import torch
        from groundingdino.util.inference import load_model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Loading GroundingDINO on %s (ckpt=%s) …", device, os.path.basename(_GROUNDINGDINO_WEIGHTS))
        model = load_model(_resolve_groundingdino_config(), _GROUNDINGDINO_WEIGHTS)
        model.to(device)
        model.eval()
        _model, _device = model, device
        log.info("GroundingDINO loaded (%s)", device)
        return _model, _device


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _ensure_hf_model():
    """Lazy-load the HF grounding-dino-tiny backend (verbatim from the retired
    navgpt ``_get_gdino``; requires transformers>=4.40)."""
    global _hf_model, _hf_processor, _device
    if _hf_model is not None:
        return _hf_model, _hf_processor, _device
    with _load_lock:
        if _hf_model is not None:
            return _hf_model, _hf_processor, _device
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Loading GroundingDINO (hf backend) %s on %s …", _HF_MODEL_ID, device)
        _hf_processor = AutoProcessor.from_pretrained(_HF_MODEL_ID)
        _hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(_HF_MODEL_ID).to(device)
        _hf_model.eval()
        _device = device
        log.info("GroundingDINO hf backend loaded (%s)", device)
        return _hf_model, _hf_processor, _device


def _detect_hf(b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
    """HF-transformers detect path — same output schema as the native path.
    Inference recipe verbatim from the retired navgpt ``OpenVocabDetectNode``."""
    import torch
    from PIL import Image

    model, processor, device = _ensure_hf_model()
    img = _decode_rgb(b64)
    pil = Image.fromarray(img, "RGB")
    proc_inputs = processor(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**proc_inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        proc_inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil.size[::-1]],
    )[0]
    h, w = img.shape[:2]
    out_boxes: list[dict] = []
    for box, score, label in zip(
        results["boxes"].cpu().numpy(), results["scores"].cpu().numpy(), results["labels"]
    ):
        out_boxes.append(
            {
                "xyxy": [int(c) for c in box],
                "score": float(score),
                "phrase": str(label).strip(),
            }
        )
    return {
        "boxes": out_boxes,
        "count": len(out_boxes),
        "image_w": int(w),
        "image_h": int(h),
        "text_prompt": text,
    }


def _detect(b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
    """Run GroundingDINO → list of pixel-xyxy boxes. Backend picked by
    ``$GROUNDING_DINO_BACKEND``; native recipe from
    ``model_detany3d/__init__.py:287-331`` (DetAny3D ``app_mp.py:94-105, 171-187``)."""
    if _BACKEND == "hf_tiny":
        return _detect_hf(b64, text, box_threshold, text_threshold)
    import groundingdino.datasets.transforms as T  # type: ignore[import-not-found]
    import torch
    from groundingdino.util.inference import predict as dino_predict
    from PIL import Image
    from torchvision.ops import box_convert

    model, device = _ensure_model()
    img = _decode_rgb(b64)  # HxWx3 uint8
    src = Image.fromarray(img, "RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_t, _ = transform(src, None)
    with torch.no_grad():
        # remove_combined omitted — not in the groundingdino-py 0.4.0 signature.
        boxes, logits, phrases = dino_predict(
            model=model,
            image=image_t,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    h, w = img.shape[:2]
    out_boxes: list[dict] = []
    if len(boxes) > 0:
        xyxy = box_convert(boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy")
        for i in range(len(xyxy)):
            out_boxes.append(
                {
                    "xyxy": xyxy[i].to(torch.int).cpu().numpy().tolist(),
                    "score": float(logits[i]),
                    "phrase": phrases[i],
                }
            )
    return {
        "boxes": out_boxes,
        "count": len(out_boxes),
        "image_w": int(w),
        "image_h": int(h),
        "text_prompt": text,
    }


def _ensure_sam():
    """Lazy-load SAM ViT-H (the AO-Planner SAM variant) once per server."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    with _load_lock:
        if _sam_predictor is not None:
            return _sam_predictor
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("Loading SAM ViT-H on %s (ckpt=%s) …", device, _SAM_VIT_H_WEIGHTS)
        sam = sam_model_registry["vit_h"](checkpoint=_SAM_VIT_H_WEIGHTS)
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
        log.info("SAM ViT-H loaded (%s)", device)
        return _sam_predictor


def _ground_mask(b64: str, text: str, box_threshold: float, text_threshold: float) -> dict:
    """GroundingDINO('ground') boxes -> SAM masks (per box) -> union ground mask.

    Faithful AO-Planner Grounded-SAM: detect ground boxes, segment each with SAM
    ViT-H (multimask_output=False), union the masks. Returns a base64 PNG mask.
    """
    import numpy as np
    from PIL import Image

    det = _detect(b64, text, box_threshold, text_threshold)
    boxes = det.get("boxes") or []
    img = _decode_rgb(b64)
    h, w = img.shape[:2]
    if not boxes:
        return {"mask_b64": "", "n_boxes": 0, "image_w": int(w), "image_h": int(h)}
    predictor = _ensure_sam()
    predictor.set_image(img)
    union = None
    for b in boxes:
        xy = b.get("xyxy")
        if not xy or len(xy) != 4:
            continue
        masks, _scores, _ = predictor.predict(box=np.asarray(xy, dtype=float), multimask_output=False)
        m = np.asarray(masks[0]).astype(bool)
        union = m if union is None else (union | m)
    if union is None:
        return {"mask_b64": "", "n_boxes": 0, "image_w": int(w), "image_h": int(h)}
    buf = io.BytesIO()
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
    return {
        "mask_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "n_boxes": len(boxes),
        "image_w": int(w),
        "image_h": int(h),
    }


class GroundingDinoDetectTool(BaseCanvasNode):
    """Open-vocabulary text→box detection with GroundingDINO Swin-B.

    Returns every box matching ``text_prompt`` (default ``"ground"``) as pixel
    ``xyxy`` + score + matched phrase, aligned to the input image's resolution.
    """

    node_type: ClassVar[str] = "model_grounding_dino__detect"
    display_name: ClassVar[str] = "GroundingDINO: Detect (open-vocab)"
    description: ClassVar[str] = (
        "Open-vocabulary text→box detection; returns boxes matching a text prompt (e.g. 'ground')"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ScanSearch"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "text_prompt", "text", "Object/region to detect (open-vocab)",
                default=_DEFAULT_TEXT_PROMPT,
            ),
            ConfigField(
                "box_threshold", "text", "Box confidence threshold (AO-Planner 0.4)",
                default=str(_DEFAULT_BOX_THRESHOLD),
            ),
            ConfigField(
                "text_threshold", "text", "Text-match threshold (AO-Planner 0.4)",
                default=str(_DEFAULT_TEXT_THRESHOLD),
            ),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
        PortDef(
            "text_prompt", "TEXT", "Optional: override the configured text prompt",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "result", "TEXT",
            "JSON {boxes:[{xyxy:[x1,y1,x2,y2], score, phrase}], count, image_w, image_h, text_prompt}",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("image_b64") or ""
        if not b64:
            return {"result": json.dumps({"boxes": [], "count": 0, "error": "no image_b64"})}

        config = getattr(self, "config", None) or {}
        text = (inputs.get("text_prompt") or config.get("text_prompt") or _DEFAULT_TEXT_PROMPT).strip()
        box_threshold = float(config.get("box_threshold", _DEFAULT_BOX_THRESHOLD) or _DEFAULT_BOX_THRESHOLD)
        text_threshold = float(config.get("text_threshold", _DEFAULT_TEXT_THRESHOLD) or _DEFAULT_TEXT_THRESHOLD)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, _detect, b64, text, box_threshold, text_threshold)
        self._self_log("n_boxes", result.get("count", 0))
        self._self_log("text_prompt", text)
        return {"result": json.dumps(result)}


class GroundingDinoGroundMaskTool(BaseCanvasNode):
    """GroundingDINO('ground') + SAM ViT-H → union navigable-ground mask.

    AO-Planner's Grounded-SAM low-level segmenter, run entirely in the detany3d
    server env (GroundingDINO Swin-B + SAM ViT-H + their weights all live
    there). Output ``mask_b64`` feeds ``aoplanner__sample_waypoints``.
    """

    node_type: ClassVar[str] = "model_grounding_dino__ground_mask"
    display_name: ClassVar[str] = "Grounded-SAM: Ground Mask"
    description: ClassVar[str] = "GroundingDINO('ground') + SAM ViT-H → union navigable-ground mask (base64 PNG)"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Layers"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("text_prompt", "text", "Ground prompt (AO-Planner 'ground')", default="ground"),
            ConfigField("box_threshold", "text", "Box confidence threshold (AO-Planner 0.4)", default="0.4"),
            ConfigField("text_threshold", "text", "Text-match threshold (AO-Planner 0.4)", default="0.4"),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
    ]
    output_ports = [
        PortDef("mask_b64", "TEXT", "Base64 PNG of the union ground mask (empty if no ground)"),
        PortDef("n_boxes", "ANY", "Number of ground boxes segmented"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("image_b64") or ""
        if not b64:
            return {"mask_b64": "", "n_boxes": 0}
        config = getattr(self, "config", None) or {}
        text = (config.get("text_prompt") or _DEFAULT_TEXT_PROMPT).strip()
        box_threshold = float(config.get("box_threshold", _DEFAULT_BOX_THRESHOLD) or _DEFAULT_BOX_THRESHOLD)
        text_threshold = float(config.get("text_threshold", _DEFAULT_TEXT_THRESHOLD) or _DEFAULT_TEXT_THRESHOLD)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, _ground_mask, b64, text, box_threshold, text_threshold)
        self._self_log("n_boxes", result.get("n_boxes", 0))
        return {"mask_b64": result["mask_b64"], "n_boxes": result.get("n_boxes", 0)}


class GroundingDinoNodeSet(BaseNodeSet):
    """GroundingDINO open-vocab detector — server-mode FM nodeset."""

    name = "model_grounding_dino"
    description = "GroundingDINO Swin-B open-vocabulary text→box detector — server-mode FM nodeset"
    # Stateless detector — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    # Reuse the detany3d env (has groundingdino-py 0.4.0 + the SwinB weights).
    # Override with $GROUNDING_DINO_PYTHON for a dedicated env.
    server_python = conda_env_python("ac-detany3d", "GROUNDING_DINO_PYTHON")

    def get_tools(self) -> list:
        return [GroundingDinoDetectTool(), GroundingDinoGroundMaskTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("GroundingDinoNodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        global _model
        _model = None
