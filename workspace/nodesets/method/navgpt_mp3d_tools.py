"""NavGPT MP3D Tools — BLIP-2 captioning and Faster R-CNN object detection.

Online vision nodes that replicate NavGPT's offline preprocessing pipeline
at runtime.  Designed to run in the **agentcanvas** env (Python 3.10+,
torch 2.x, transformers, torchvision) in local mode.

Original NavGPT pre-computes scene descriptions offline using:
  - BLIP-2 ViT-G FlanT5-XL for captioning 24 egocentric views per viewpoint
  - Faster R-CNN for object detection within 3 m

These nodes do the same work online, accepting the env per-view primitive
(``views`` LIST[IMAGE] + ``view_meta``) from env_mp3d and returning text
descriptions / object lists matching the original NavGPT observation format.

last updated: 2026-04-10
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.navgpt_mp3d_tools")

# ══════════════════════════════════════════════════════════════════════
# Lazy model singletons — loaded once on first use, stay in GPU memory
# ══════════════════════════════════════════════════════════════════════

_blip2_model = None
_blip2_processor = None
_blip2_device = None
_blip2_load_lock = threading.Lock()

_rcnn_model = None
_rcnn_device = None
_rcnn_load_lock = threading.Lock()
# COCO class names (91 classes, index 0 = __background__)
_COCO_CLASSES: list[str] = []

_gdino_model = None
_gdino_processor = None
_gdino_device = None
_gdino_load_lock = threading.Lock()

_instructblip_model = None
_instructblip_processor = None
_instructblip_device = None
_instructblip_load_lock = threading.Lock()

# R2R-relevant indoor vocabulary for GroundingDINO open-vocab detection.
# Period-delimited per GroundingDINO convention. Covers terminal landmarks
# the original NavGPT BUTD detector (Visual Genome 1600 classes) saw, that
# COCO 80 classes miss.
_R2R_INDOOR_VOCAB = (
    "chair . table . sofa . bed . bathtub . toilet . sink . door . window . "
    "television . nightstand . wardrobe . mirror . lamp . plant . staircase . "
    "railing . fireplace . urn . archway . credenza . cabinet . dresser . "
    "shelf . desk . bookcase . painting . curtain . rug . pillow . bench . "
    "piano . refrigerator . oven . microwave . stove . counter . island . "
    "kitchen . dishwasher . armchair . ottoman . loveseat . coffee table . "
    "end table . dining table . tv stand . pool table . fountain . statue . "
    "vase . candle . clock . banister . step . landing . hallway . corridor . "
    "doorway . column . pillar . arch . skylight . chandelier . sconce . "
    "pendant light . floor lamp . ceiling fan . exit . entrance ."
)


def _get_blip2(model_name: str = "Salesforce/blip2-flan-t5-xl", device: str = "auto"):
    """Lazy-load BLIP-2 model and processor."""
    global _blip2_model, _blip2_processor, _blip2_device

    if _blip2_model is not None:
        return _blip2_model, _blip2_processor, _blip2_device

    with _blip2_load_lock:
        if _blip2_model is not None:
            return _blip2_model, _blip2_processor, _blip2_device

        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Loading BLIP-2 model %s on %s …", model_name, device)
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        _blip2_processor = processor
        _blip2_model = model
        _blip2_device = device
        log.info("BLIP-2 loaded (%s)", device)
        return _blip2_model, _blip2_processor, _blip2_device


def _get_rcnn(device: str = "auto"):
    """DEPRECATED — kept for back-compat with v0 snapshots; use ``_get_gdino`` instead."""
    global _rcnn_model, _rcnn_device, _COCO_CLASSES

    if _rcnn_model is not None:
        return _rcnn_model, _rcnn_device

    with _rcnn_load_lock:
        if _rcnn_model is not None:
            return _rcnn_model, _rcnn_device

        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Loading Faster R-CNN on %s …", device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
        model.eval()
        _rcnn_model = model
        _rcnn_device = device
        _COCO_CLASSES = weights.meta["categories"]
        log.info("Faster R-CNN loaded (%s, %d classes)", device, len(_COCO_CLASSES))
        return _rcnn_model, _rcnn_device


def _get_instructblip(
    model_name: str = "Salesforce/instructblip-flan-t5-xl",
    device: str = "auto",
):
    """Lazy-load InstructBLIP (FlanT5-XL) for DiscussNav-style scene description.

    Source: DiscussNav.py:133 (LAVIS load_model_and_preprocess
    'blip2_t5_instruct/flant5xl'). We use the HuggingFace transformers
    equivalent because LAVIS is not installed in the agentcanvas env.
    """
    global _instructblip_model, _instructblip_processor, _instructblip_device

    if _instructblip_model is not None:
        return _instructblip_model, _instructblip_processor, _instructblip_device

    with _instructblip_load_lock:
        if _instructblip_model is not None:
            return _instructblip_model, _instructblip_processor, _instructblip_device

        import torch
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Loading InstructBLIP %s on %s …", model_name, device)
        processor = InstructBlipProcessor.from_pretrained(model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        _instructblip_processor = processor
        _instructblip_model = model
        _instructblip_device = device
        log.info("InstructBLIP loaded (%s)", device)
        return _instructblip_model, _instructblip_processor, _instructblip_device


def _get_gdino(device: str = "auto"):
    """Lazy-load GroundingDINO-tiny for open-vocabulary text-conditioned detection.

    Replaces COCO 80-class Faster R-CNN. Original NavGPT used BUTD Faster R-CNN
    on Visual Genome 1600 classes; GroundingDINO's open-vocab text conditioning
    is a paper-near substitute (queries via ``_R2R_INDOOR_VOCAB``).
    """
    global _gdino_model, _gdino_processor, _gdino_device

    if _gdino_model is not None:
        return _gdino_model, _gdino_processor, _gdino_device

    with _gdino_load_lock:
        if _gdino_model is not None:
            return _gdino_model, _gdino_processor, _gdino_device

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "IDEA-Research/grounding-dino-tiny"
        log.info("Loading GroundingDINO-tiny (%s) on %s …", model_id, device)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()
        _gdino_processor = processor
        _gdino_model = model
        _gdino_device = device
        log.info("GroundingDINO-tiny loaded (%s)", device)
        return _gdino_model, _gdino_processor, _gdino_device


# ══════════════════════════════════════════════════════════════════════
# Per-view inputs (env_mp3d emits views: LIST[IMAGE] + view_meta: TEXT)
# ══════════════════════════════════════════════════════════════════════

# env_mp3d renders with a fixed 3-elevation sweep [-30, 0, 30]; views are
# ordered OUTER=elevation, INNER=heading, so view_index = e * n_headings + h.
_N_ELEVATIONS = 3


def _parse_view_meta(raw: Any) -> list[dict]:
    """Parse the env ``view_meta`` JSON into a list of per-view dicts.

    Each entry: ``{view_index, heading_deg, elevation_deg, direction}``,
    aligned 1:1 with the ``views`` LIST[IMAGE].
    """
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    try:
        parsed = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        return []
    return [m for m in parsed if isinstance(m, dict)] if isinstance(parsed, list) else []


def _infer_n_headings(n_views: int) -> int:
    """Headings per elevation ring, given the fixed 3-elevation render.

    Returns ``n_views // 3`` when evenly divisible (24→8, 36→12), else
    ``n_views`` (horizon-only / unknown layout → treat as no elevation rings).
    """
    if n_views >= _N_ELEVATIONS and n_views % _N_ELEVATIONS == 0:
        return n_views // _N_ELEVATIONS
    return n_views


# ══════════════════════════════════════════════════════════════════════
# BLIP-2 Caption Node
# ══════════════════════════════════════════════════════════════════════


class BLIP2CaptionNode(BaseCanvasNode):
    """Caption panorama views using BLIP-2, matching NavGPT's offline pipeline.

    Accepts the env per-view primitive (``views`` LIST[IMAGE] + ``view_meta``)
    from env_mp3d, runs BLIP-2 captioning on each view, and returns:

    - ``descriptions``: 8-direction scene descriptions as formatted text
    - ``summary``: GPT-3.5-style 1-sentence summary (approximated by
      concatenating the front-facing caption)
    - ``descriptions_json``: raw JSON array of per-direction captions

    The original NavGPT uses BLIP-2 ViT-G FlanT5-XL with the prompt
    *"This is a scene of"*.
    """

    node_type = "navgpt_mp3d_tools__blip2_caption"
    display_name = "BLIP-2 NavGPT"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model_name", "text", "HuggingFace model ID", default="Salesforce/blip2-flan-t5-xl"
            ),
            ConfigField("prompt", "text", "Captioning prompt", default="This is a scene of"),
            ConfigField(
                "device",
                "select",
                "Device",
                options=[
                    {"value": "auto", "label": "Auto"},
                    {"value": "cuda", "label": "CUDA"},
                    {"value": "cpu", "label": "CPU"},
                ],
                default="auto",
            ),
            ConfigField(
                "max_new_tokens",
                "slider",
                "Max tokens per caption",
                default=64,
                min=16,
                max=256,
                step=16,
            ),
            ConfigField(
                "merge_elevations",
                "toggle",
                "Merge 3 elevation views per heading (NavGPT 24-view mode)",
                default=True,
            ),
            ConfigField(
                "merge_style",
                "select",
                "Elevation merge style",
                options=[
                    {"value": "primary", "label": "Primary (ahead only)"},
                    {"value": "concat", "label": "Concat (down/ahead/up)"},
                ],
                default="primary",
            ),
        ],
    )
    description = "Caption panorama views using BLIP-2 (NavGPT perception)"
    category = "perception"
    icon = "ScanEye"
    input_ports = [
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env_mp3d"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
    ]
    output_ports = [
        PortDef("descriptions", "TEXT", "8-direction scene descriptions (formatted text)"),
        PortDef("summary", "TEXT", "1-sentence scene summary"),
        PortDef("descriptions_json", "TEXT", "Per-direction captions as JSON array"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import asyncio

        import torch
        from PIL import Image

        raw_views = inputs.get("views")
        views = list(raw_views) if isinstance(raw_views, list) else []
        directions = _parse_view_meta(inputs.get("view_meta"))

        if not views:
            self._self_log("error", "No views received")
            return {"descriptions": "", "summary": "", "descriptions_json": "[]"}

        n_views = len(views)

        config = getattr(self, "config", None) or {}
        model_name = config.get("model_name", "Salesforce/blip2-flan-t5-xl")
        prompt = config.get("prompt", "This is a scene of")
        device = config.get("device", "auto")
        max_new_tokens = int(config.get("max_new_tokens", 64))
        merge_elevations = bool(config.get("merge_elevations", True))
        merge_style = config.get("merge_style", "primary")

        self._self_log("model", model_name)
        self._self_log("n_views", n_views)
        self._self_log("merge_elevations", merge_elevations)
        self._self_log("views_received", len(views))

        # Run BLIP-2 captioning in thread (model is synchronous)
        loop = asyncio.get_running_loop()

        def _caption_all() -> list[str]:
            model, processor, dev = _get_blip2(model_name, device)
            captions: list[str] = []
            for view_arr in views:
                pil_img = Image.fromarray(view_arr).convert("RGB")
                inputs_blip = processor(images=pil_img, text=prompt, return_tensors="pt").to(
                    dev,
                    dtype=torch.float16 if dev == "cuda" else torch.float32,
                )
                with torch.no_grad():
                    out = model.generate(**inputs_blip, max_new_tokens=max_new_tokens)
                caption = processor.decode(out[0], skip_special_tokens=True).strip()
                captions.append(caption)
            return captions

        captions = await loop.run_in_executor(None, _caption_all)

        # Elevation merging: group views by heading across elevation levels.
        # env_mp3d renders OUTER=elevation, INNER=heading (3 elevations),
        # so heading h at elevation e is at index: h + e * n_headings.
        n_headings = _infer_n_headings(n_views)
        n_elevs = n_views // n_headings if n_headings else 1
        _elevation_labels = ("Looking down", "Ahead", "Looking up")  # elev -30, 0, +30
        _did_merge = False
        if merge_elevations and n_elevs > 1 and n_views == n_headings * n_elevs:
            merged: list[str] = []
            for h in range(n_headings):
                group = [captions[h + e * n_headings] for e in range(n_elevs)]
                if merge_style == "primary":
                    # Use only the ahead/0° elevation (index 1 for [-30,0,30])
                    merged.append(group[1] if len(group) > 1 else group[0])
                else:
                    parts = [f"{_elevation_labels[e]}: {group[e]}" for e in range(len(group))]
                    merged.append(", ".join(parts))
            self._self_log("merged_headings", n_headings)
            output_captions = merged
            _did_merge = True
        else:
            output_captions = captions

        # Format output — one line per heading direction
        desc_lines: list[str] = []
        dir_labels = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        n_output = len(output_captions)
        for i, caption in enumerate(output_captions):
            label = dir_labels[i] if i < len(dir_labels) else f"View {i}"
            if not _did_merge and directions and i < len(directions):
                # Non-merge mode: use per-view direction metadata directly
                label = directions[i].get("direction", label)
            desc_lines.append(f"{label}: {caption}")

        descriptions = "\n".join(desc_lines)
        # Summary: use front-facing merged (or raw) caption as a 1-sentence approximation
        summary = output_captions[0] if output_captions else ""

        self._self_log("output_captions_count", n_output)
        for i, c in enumerate(output_captions):
            self._self_log(f"caption_{i}", c[:200])
        self._self_log("summary", summary[:200])

        return {
            "descriptions": descriptions,
            "summary": summary,
            "descriptions_json": json.dumps(output_captions),
        }


# ══════════════════════════════════════════════════════════════════════
# InstructBLIP Caption Node — DiscussNav per-direction VLM
# ══════════════════════════════════════════════════════════════════════


# Source: DiscussNav.py:140 (Vision_Perception_Experts.instructblip_description).
_INSTRUCTBLIP_PROMPT = "Describe this indoor scene in details"


class InstructBlipCaptionNode(BaseCanvasNode):
    """Per-direction scene description with InstructBLIP — DiscussNav perception.

    Consumes the env per-view primitive (``views`` LIST[IMAGE] + ``view_meta``)
    and runs InstructBLIP-FlanT5-XL on each view with the verbatim DiscussNav
    prompt *"Describe this indoor scene in details"*.

    Output ``captions_per_dir`` is a ``LIST[TEXT]`` aligned 1:1 with the
    ``directions`` JSON (index = view_index). Empty entries when a view is
    blocked / out of bounds preserve length.
    """

    node_type = "navgpt_mp3d_tools__instructblip_caption"
    display_name = "InstructBLIP Caption (DiscussNav)"
    description = "Describe each panorama direction with InstructBLIP-FlanT5-XL"
    category = "perception"
    icon = "ScanEye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model_name",
                "text",
                "HuggingFace model ID",
                default="Salesforce/instructblip-flan-t5-xl",
            ),
            ConfigField("prompt", "text", "Description prompt", default=_INSTRUCTBLIP_PROMPT),
            ConfigField(
                "device",
                "select",
                "Device",
                options=[
                    {"value": "auto", "label": "Auto"},
                    {"value": "cuda", "label": "CUDA"},
                    {"value": "cpu", "label": "CPU"},
                ],
                default="auto",
            ),
            ConfigField(
                "max_new_tokens",
                "slider",
                "Max tokens per caption",
                default=128,
                min=32,
                max=384,
                step=16,
            ),
        ],
    )
    input_ports = [
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env_mp3d"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
    ]
    output_ports = [
        PortDef(
            "captions_per_dir",
            "LIST[TEXT]",
            "Per-direction descriptions, aligned 1:1 with `views` / `view_meta`",
        ),
        PortDef("captions_json", "TEXT", "Same list serialised as JSON"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import asyncio

        import torch
        from PIL import Image

        raw_views = inputs.get("views")
        views = list(raw_views) if isinstance(raw_views, list) else []

        if not views:
            self._self_log("error", "No views received")
            return {"captions_per_dir": [], "captions_json": "[]"}

        n_views = len(views)

        config = getattr(self, "config", None) or {}
        model_name = config.get("model_name", "Salesforce/instructblip-flan-t5-xl")
        prompt = config.get("prompt", _INSTRUCTBLIP_PROMPT)
        device = config.get("device", "auto")
        max_new_tokens = int(config.get("max_new_tokens", 128))

        self._self_log("model", model_name)
        self._self_log("n_views", n_views)
        self._self_log("views_received", len(views))

        loop = asyncio.get_running_loop()

        def _caption_all() -> list[str]:
            # Manual unpacking — transformers' InstructBlipProcessor.__call__
            # concatenates image-token list with text Tensor (processing_
            # instructblip.py:134-136) and trips "list + Tensor" TypeError.
            # We bypass by calling the sub-tokenizers/image-processor
            # directly and prepending image tokens to the prompt ourselves.
            model, processor, dev = _get_instructblip(model_name, device)
            num_q = processor.num_query_tokens or 32
            img_token_str = processor.image_token.content * num_q
            cast_dtype = torch.float16 if dev == "cuda" else torch.float32
            out: list[str] = ["" for _ in views]
            for i, view_arr in enumerate(views):
                if view_arr is None or view_arr.size == 0:
                    continue
                pil = Image.fromarray(view_arr).convert("RGB")
                full_text = img_token_str + prompt
                text_enc = processor.tokenizer(full_text, return_tensors="pt")
                qf_enc = processor.qformer_tokenizer(prompt, return_tensors="pt")
                img_enc = processor.image_processor(pil, return_tensors="pt")
                proc_inputs = {
                    "input_ids": text_enc["input_ids"].to(dev),
                    "attention_mask": text_enc["attention_mask"].to(dev),
                    "qformer_input_ids": qf_enc["input_ids"].to(dev),
                    "qformer_attention_mask": qf_enc["attention_mask"].to(dev),
                    "pixel_values": img_enc["pixel_values"].to(dev, dtype=cast_dtype),
                }
                with torch.no_grad():
                    gen = model.generate(
                        **proc_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                decoded = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
                out[i] = (decoded[0] if decoded else "").strip()
            return out

        captions = await loop.run_in_executor(None, _caption_all)
        for i, c in enumerate(captions):
            self._self_log(f"caption_{i}", c[:200])

        return {
            "captions_per_dir": captions,
            "captions_json": json.dumps(captions),
        }


# ══════════════════════════════════════════════════════════════════════
# Faster R-CNN Object Detection Node
# ══════════════════════════════════════════════════════════════════════


class FasterRCNNDetectNode(BaseCanvasNode):
    """Detect objects in panorama views using Faster R-CNN.

    Accepts the env per-view primitive (``views`` + ``view_meta``), runs
    Faster R-CNN (ResNet-50 FPN, COCO-pretrained) on each view, and
    returns per-direction object lists matching NavGPT's format:
    object name + approximate relative heading within the 45-degree
    sector + distance (estimated from bounding box size).

    The original NavGPT retains only objects within 3 metres.
    """

    node_type = "navgpt_mp3d_tools__fasterrcnn_detect"
    display_name = "Faster R-CNN NavGPT"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "device",
                "select",
                "Device",
                options=[
                    {"value": "auto", "label": "Auto"},
                    {"value": "cuda", "label": "CUDA"},
                    {"value": "cpu", "label": "CPU"},
                ],
                default="auto",
            ),
            ConfigField(
                "confidence",
                "slider",
                "Min confidence threshold",
                default=0.5,
                min=0.1,
                max=0.95,
                step=0.05,
            ),
            ConfigField(
                "max_objects_per_view",
                "slider",
                "Max objects per view",
                default=10,
                min=1,
                max=30,
                step=1,
            ),
        ],
    )
    description = "Detect objects in panorama views using Faster R-CNN (NavGPT perception)"
    category = "perception"
    icon = "BoxSelect"
    input_ports = [
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env_mp3d"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
    ]
    output_ports = [
        PortDef("objects_json", "TEXT", "Per-direction objects as JSON (list of 8 dicts)"),
        PortDef("objects_text", "TEXT", "Formatted objects text for LLM prompt"),
        PortDef("total_count", "TEXT", "Total objects detected across all views"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import asyncio

        import torch
        from torchvision.transforms import functional as F

        raw_views = inputs.get("views")
        views = list(raw_views) if isinstance(raw_views, list) else []
        directions = _parse_view_meta(inputs.get("view_meta"))

        if not views:
            self._self_log("error", "No views received")
            return {"objects_json": "[]", "objects_text": "", "total_count": "0"}

        n_views = len(views)

        config = getattr(self, "config", None) or {}
        device = config.get("device", "auto")
        confidence = float(config.get("confidence", 0.5))
        max_per_view = int(config.get("max_objects_per_view", 10))

        self._self_log("n_views", n_views)
        self._self_log("confidence_threshold", confidence)
        self._self_log("views_received", len(views))

        loop = asyncio.get_running_loop()

        def _detect_all() -> list[list[dict]]:
            model, dev = _get_rcnn(device)
            all_objects: list[list[dict]] = []

            for _view_idx, view_arr in enumerate(views):
                # Convert to tensor [C, H, W] float32 in [0, 1]
                tensor = F.to_tensor(view_arr).to(dev)
                with torch.no_grad():
                    preds = model([tensor])[0]

                boxes = preds["boxes"].cpu().numpy()
                labels = preds["labels"].cpu().numpy()
                scores = preds["scores"].cpu().numpy()

                h, w = view_arr.shape[:2]
                fov_deg = 45.0  # NavGPT uses 45-degree FOV per view

                view_objects: list[dict] = []
                for box, label_idx, score in zip(boxes, labels, scores, strict=True):
                    if score < confidence:
                        continue
                    if len(view_objects) >= max_per_view:
                        break

                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    # Estimate relative heading within this view's FOV
                    # cx=0 → left edge (-22.5°), cx=w → right edge (+22.5°)
                    rel_heading_in_view = (cx / w - 0.5) * fov_deg

                    # Rough distance estimate from bounding box height
                    # (larger box = closer object, heuristic only)
                    box_h = y2 - y1
                    box_ratio = box_h / h
                    # Heuristic: full-height box ≈ 0.5m, tiny box ≈ 5m
                    estimated_distance = max(0.3, min(5.0, 1.5 / max(box_ratio, 0.01)))

                    class_name = (
                        _COCO_CLASSES[label_idx]
                        if label_idx < len(_COCO_CLASSES)
                        else f"class_{label_idx}"
                    )

                    view_objects.append(
                        {
                            "name": class_name,
                            "confidence": round(float(score), 3),
                            "rel_heading_deg": round(float(rel_heading_in_view), 1),
                            "estimated_distance_m": round(float(estimated_distance), 2),
                            "bbox": [round(float(c), 1) for c in [x1, y1, x2, y2]],
                        }
                    )

                all_objects.append(view_objects)
            return all_objects

        all_objects = await loop.run_in_executor(None, _detect_all)

        # Format output
        dir_labels = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        text_lines: list[str] = []
        total = 0
        for i, objs in enumerate(all_objects):
            label = dir_labels[i] if i < len(dir_labels) else f"View {i}"
            if directions and i < len(directions):
                label = directions[i].get("direction", label)
            total += len(objs)
            if objs:
                obj_strs = [
                    f"{o['name']} ({o['confidence']:.0%}, ~{o['estimated_distance_m']:.1f}m)"
                    for o in objs
                ]
                text_lines.append(f"{label} Objects: {', '.join(obj_strs)}")
            else:
                text_lines.append(f"{label} Objects: None")

        objects_text = "\n".join(text_lines)

        self._self_log("total_objects", total)
        for i, objs in enumerate(all_objects):
            if objs:
                names = [o["name"] for o in objs]
                self._self_log(f"view_{i}_objects", names)

        return {
            "objects_json": json.dumps(all_objects),
            "objects_text": objects_text,
            "total_count": str(total),
        }


# ══════════════════════════════════════════════════════════════════════
# Open-Vocab Detection Node (GroundingDINO-tiny — text-conditioned)
# ══════════════════════════════════════════════════════════════════════


class OpenVocabDetectNode(BaseCanvasNode):
    """Open-vocabulary object detection via GroundingDINO-tiny.

    Drop-in replacement for ``FasterRCNNDetectNode`` (COCO 80-class) with
    text-conditioned detection over ``_R2R_INDOOR_VOCAB`` (~75 R2R-relevant
    indoor nouns). Approximates the paper's Visual Genome 1600-class BUTD
    coverage of indoor objects (bathtub, urn, archway, credenza, nightstand,
    etc.) that COCO misses entirely.

    Distance estimation uses the same bbox-height heuristic as the legacy
    node — Phase 3 of the fidelity-recovery plan replaces this with real
    MatterSim depth.
    """

    node_type = "navgpt_mp3d_tools__open_vocab_detect"
    display_name = "Open-Vocab Detector NavGPT"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "device",
                "select",
                "Device",
                options=[
                    {"value": "auto", "label": "Auto"},
                    {"value": "cuda", "label": "CUDA"},
                    {"value": "cpu", "label": "CPU"},
                ],
                default="auto",
            ),
            ConfigField(
                "box_threshold",
                "slider",
                "Min box-confidence threshold",
                default=0.3,
                min=0.1,
                max=0.7,
                step=0.05,
            ),
            ConfigField(
                "text_threshold",
                "slider",
                "Min text-confidence threshold",
                default=0.25,
                min=0.1,
                max=0.7,
                step=0.05,
            ),
            ConfigField(
                "max_objects_per_view",
                "slider",
                "Max objects per view",
                default=10,
                min=1,
                max=30,
                step=1,
            ),
            ConfigField(
                "text_queries",
                "text",
                "Period-delimited candidate classes (GroundingDINO format)",
                default=_R2R_INDOOR_VOCAB,
            ),
        ],
    )
    description = "Open-vocabulary detection (GroundingDINO-tiny, text-conditioned)"
    category = "perception"
    icon = "BoxSelect"
    input_ports = [
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env_mp3d"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
        PortDef(
            "depth_views",
            "LIST[DEPTH]",
            "Per-view depth (metres) aligned 1:1 with views (optional)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("objects_json", "TEXT", "Per-direction objects as JSON (list of 8 dicts)"),
        PortDef("objects_text", "TEXT", "Formatted objects text for LLM prompt"),
        PortDef("total_count", "TEXT", "Total objects detected across all views"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import asyncio

        import torch
        from PIL import Image

        raw_views = inputs.get("views")
        views = list(raw_views) if isinstance(raw_views, list) else []
        directions = _parse_view_meta(inputs.get("view_meta"))
        raw_depth = inputs.get("depth_views")

        if not views:
            self._self_log("error", "No views received")
            return {"objects_json": "[]", "objects_text": "", "total_count": "0"}

        n_views = len(views)

        config = getattr(self, "config", None) or {}
        device = config.get("device", "auto")
        box_thr = float(config.get("box_threshold", 0.3))
        text_thr = float(config.get("text_threshold", 0.25))
        max_per_view = int(config.get("max_objects_per_view", 10))
        # NavGPT paper filters detections to ≤ 3m via real MatterSim depth.
        max_distance_m = float(config.get("max_distance_m", 3.0))
        text_queries = str(config.get("text_queries", _R2R_INDOOR_VOCAB)).strip()
        if not text_queries.endswith("."):
            text_queries = text_queries + " ."

        self._self_log("n_views", n_views)
        self._self_log("box_threshold", box_thr)
        self._self_log("text_threshold", text_thr)
        self._self_log("vocab_chars", len(text_queries))
        self._self_log("max_distance_m", max_distance_m)
        self._self_log("views_received", len(views))

        depth_views: list[np.ndarray] | None = None
        if isinstance(raw_depth, list) and raw_depth:
            depth_views = list(raw_depth)
            self._self_log("depth_source", "real_mattersim_skybox")
            self._self_log("depth_views_count", len(depth_views))
        else:
            self._self_log("depth_source", "heuristic_fallback")

        loop = asyncio.get_running_loop()

        def _detect_all() -> list[list[dict]]:
            model, processor, dev = _get_gdino(device)
            all_objects: list[list[dict]] = []
            for view_idx, view_arr in enumerate(views):
                depth_view = depth_views[view_idx] if depth_views is not None else None
                pil = Image.fromarray(view_arr.astype("uint8"))
                proc_inputs = processor(
                    images=pil,
                    text=text_queries,
                    return_tensors="pt",
                ).to(dev)
                with torch.no_grad():
                    outputs = model(**proc_inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    proc_inputs.input_ids,
                    threshold=box_thr,
                    text_threshold=text_thr,
                    target_sizes=[pil.size[::-1]],
                )[0]

                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]  # list of strings — no index lookup needed
                h, w = view_arr.shape[:2]
                fov_deg = 45.0

                view_objects: list[dict] = []
                for box, score, label in zip(boxes, scores, labels, strict=True):
                    if len(view_objects) >= max_per_view:
                        break
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    rel_heading_in_view = (cx / w - 0.5) * fov_deg
                    if depth_view is not None:
                        cy_i = int(np.clip(cy, 0, depth_view.shape[0] - 1))
                        cx_i = int(np.clip(cx, 0, depth_view.shape[1] - 1))
                        depth_m = float(depth_view[cy_i, cx_i])
                        # depth_view is float32 metres (converted at the
                        # MP3DGraphPanoramaTool boundary). 0 = unknown
                        # (rendering hole or out of range).
                        if depth_m > 0:
                            estimated_distance = depth_m
                            distance_source = "real_depth"
                        else:
                            box_ratio = (y2 - y1) / h
                            estimated_distance = max(0.3, min(5.0, 1.5 / max(box_ratio, 0.01)))
                            distance_source = "heuristic_hole"
                    else:
                        box_ratio = (y2 - y1) / h
                        estimated_distance = max(0.3, min(5.0, 1.5 / max(box_ratio, 0.01)))
                        distance_source = "heuristic_no_depth"

                    # Paper-faithful 3m filter — only retain objects within reach.
                    if estimated_distance > max_distance_m:
                        continue

                    view_objects.append(
                        {
                            "name": str(label).strip(),
                            "confidence": round(float(score), 3),
                            "rel_heading_deg": round(float(rel_heading_in_view), 1),
                            "estimated_distance_m": round(float(estimated_distance), 2),
                            "distance_source": distance_source,
                            "bbox": [round(float(c), 1) for c in [x1, y1, x2, y2]],
                        }
                    )
                all_objects.append(view_objects)
            return all_objects

        all_objects = await loop.run_in_executor(None, _detect_all)

        # Format output — identical schema to FasterRCNNDetectNode
        dir_labels = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        text_lines: list[str] = []
        total = 0
        for i, objs in enumerate(all_objects):
            label = dir_labels[i] if i < len(dir_labels) else f"View {i}"
            if directions and i < len(directions):
                label = directions[i].get("direction", label)
            total += len(objs)
            if objs:
                obj_strs = [
                    f"{o['name']} ({o['confidence']:.0%}, ~{o['estimated_distance_m']:.1f}m)"
                    for o in objs
                ]
                text_lines.append(f"{label} Objects: {', '.join(obj_strs)}")
            else:
                text_lines.append(f"{label} Objects: None")

        objects_text = "\n".join(text_lines)

        self._self_log("total_objects", total)
        for i, objs in enumerate(all_objects):
            if objs:
                self._self_log(f"view_{i}_objects", [o["name"] for o in objs])

        return {
            "objects_json": json.dumps(all_objects),
            "objects_text": objects_text,
            "total_count": str(total),
        }


# ══════════════════════════════════════════════════════════════════════
# Paper Objects Cache — load NavGPT's pre-computed objects_list/{scan}.json
# ══════════════════════════════════════════════════════════════════════

_PAPER_OBJECTS_CACHE: dict[str, dict] = {}
_PAPER_OBJECTS_LOCK = threading.Lock()
# nodesets/navgpt_mp3d_tools.py → nodesets → workspace → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAPER_OBJECTS_DIR = str(_REPO_ROOT / "data" / "mp3d" / "v1" / "objects_list")

_PAPER_CAPTIONS_LIST_CACHE: dict[str, dict] = {}
_PAPER_CAPTIONS_SUMMARY_CACHE: dict[str, dict] = {}
_PAPER_CAPTIONS_LOCK = threading.Lock()
_PAPER_CAPTIONS_LIST_DIR = str(_REPO_ROOT / "data" / "mp3d" / "v1" / "observations_list_summarized")
_PAPER_CAPTIONS_SUMMARY_DIR = str(_REPO_ROOT / "data" / "mp3d" / "v1" / "observations_summarized")


def _load_paper_captions_list(scan_id: str) -> dict:
    """Load NavGPT's per-heading 3-elev-summarized captions ({scan}.json -> {vp: [8 sentences]})."""
    if scan_id in _PAPER_CAPTIONS_LIST_CACHE:
        return _PAPER_CAPTIONS_LIST_CACHE[scan_id]
    with _PAPER_CAPTIONS_LOCK:
        if scan_id in _PAPER_CAPTIONS_LIST_CACHE:
            return _PAPER_CAPTIONS_LIST_CACHE[scan_id]
        with open(f"{_PAPER_CAPTIONS_LIST_DIR}/{scan_id}.json") as f:
            data = json.load(f)
        _PAPER_CAPTIONS_LIST_CACHE[scan_id] = data
        log.info("Loaded paper captions_list for scan %s (%d viewpoints)", scan_id, len(data))
        return data


def _load_paper_captions_summary(scan_id: str) -> dict:
    """Load NavGPT's per-viewpoint summary sentence ({scan}_summarized.json -> {vp: str})."""
    if scan_id in _PAPER_CAPTIONS_SUMMARY_CACHE:
        return _PAPER_CAPTIONS_SUMMARY_CACHE[scan_id]
    with _PAPER_CAPTIONS_LOCK:
        if scan_id in _PAPER_CAPTIONS_SUMMARY_CACHE:
            return _PAPER_CAPTIONS_SUMMARY_CACHE[scan_id]
        with open(f"{_PAPER_CAPTIONS_SUMMARY_DIR}/{scan_id}_summarized.json") as f:
            data = json.load(f)
        _PAPER_CAPTIONS_SUMMARY_CACHE[scan_id] = data
        log.info("Loaded paper captions_summary for scan %s (%d viewpoints)", scan_id, len(data))
        return data


def _load_paper_objects(scan_id: str) -> dict:
    """Lazy-load and cache the paper's pre-computed objects_list JSON for one scan."""
    if scan_id in _PAPER_OBJECTS_CACHE:
        return _PAPER_OBJECTS_CACHE[scan_id]
    with _PAPER_OBJECTS_LOCK:
        if scan_id in _PAPER_OBJECTS_CACHE:
            return _PAPER_OBJECTS_CACHE[scan_id]
        path = f"{_PAPER_OBJECTS_DIR}/{scan_id}.json"
        with open(path) as f:
            data = json.load(f)
        _PAPER_OBJECTS_CACHE[scan_id] = data
        log.info("Loaded paper objects_list for scan %s (%d viewpoints)", scan_id, len(data))
        return data


class PaperObjectsCacheNode(BaseCanvasNode):
    """Drop-in replacement for OpenVocabDetectNode that uses paper-released objects.

    Loads the per-viewpoint object dict from NavGPT's pre-computed
    ``objects_list/{scan}.json`` (downloaded from the paper's Dropbox release;
    1916-class vocabulary, ~7.2 objects/viewpoint average). Each viewpoint maps
    to a list of 8 sector dicts ``{obj_name: {heading_deg, distance_m}}`` —
    same shape as ``OpenVocabDetectNode.objects_json`` after sector merging.

    Headings in the paper data are in DEGREES; we convert to RADIANS to match
    the rest of the navgpt_mp3d_tools pipeline (``_format_observation_compass``
    expects radians).
    """

    node_type = "navgpt_mp3d_tools__paper_objects_cache"
    display_name = "Paper Objects Cache (NavGPT)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = "Load paper-released objects_list/{scan}.json for current viewpoint"
    category = "perception"
    icon = "Package"
    input_ports = [
        PortDef("scan_id", "TEXT", "Scan ID from env (reset.scan_id or panorama.scan_id)"),
        PortDef("viewpoint_id", "TEXT", "Viewpoint ID from env"),
        PortDef("trigger", "TEXT", "Optional sequencing trigger", optional=True),
    ]
    output_ports = [
        PortDef(
            "objects_json",
            "TEXT",
            "Per-direction objects as JSON (list of 8 dicts, headings in radians)",
        ),
        PortDef("objects_text", "TEXT", "Formatted objects text for LLM prompt"),
        PortDef("total_count", "TEXT", "Total objects loaded across all sectors"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import math

        scan_id = str(inputs.get("scan_id", "")).strip()
        viewpoint_id = str(inputs.get("viewpoint_id", "")).strip()

        if not scan_id or not viewpoint_id:
            self._self_log("error", f"missing scan/vp: scan={scan_id!r} vp={viewpoint_id!r}")
            return {"objects_json": "[]", "objects_text": "", "total_count": "0"}

        try:
            scan_data = _load_paper_objects(scan_id)
        except FileNotFoundError as exc:
            self._self_log("error", f"objects_list missing for scan {scan_id}: {exc}")
            return {"objects_json": "[]", "objects_text": "", "total_count": "0"}

        sectors = scan_data.get(viewpoint_id)
        if sectors is None:
            self._self_log(
                "error", f"viewpoint {viewpoint_id} not in objects_list for scan {scan_id}"
            )
            return {"objects_json": "[]", "objects_text": "", "total_count": "0"}

        # Convert headings from degrees → radians; keep distance as-is.
        # Paper schema: [{obj_name: {heading: deg, distance: m}}, ...] x8
        sectors_rad: list[dict] = []
        total = 0
        for sector in sectors:
            converted: dict = {}
            for obj_name, info in sector.items():
                if not isinstance(info, dict):
                    continue
                h_deg = float(info.get("heading", 0.0))
                d_m = float(info.get("distance", 0.0))
                converted[obj_name] = {
                    "heading": math.radians(h_deg),
                    "distance": d_m,
                }
                total += 1
            sectors_rad.append(converted)

        # Format objects_text for LLM (per-sector listing — same format as detector nodes)
        dir_labels = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        text_lines: list[str] = []
        for i, sector in enumerate(sectors_rad):
            label = dir_labels[i] if i < len(dir_labels) else f"View {i}"
            if sector:
                items = [f"{name} ({info['distance']:.2f}m)" for name, info in sector.items()]
                text_lines.append(f"{label} Objects: {', '.join(items)}")
            else:
                text_lines.append(f"{label} Objects: None")
        objects_text = "\n".join(text_lines)

        self._self_log("scan_id", scan_id)
        self._self_log("viewpoint_id", viewpoint_id)
        self._self_log("total_objects", total)
        for i, sector in enumerate(sectors_rad):
            if sector:
                self._self_log(f"sector_{i}_objects", list(sector.keys()))

        return {
            "objects_json": json.dumps(sectors_rad),
            "objects_text": objects_text,
            "total_count": str(total),
        }


class PaperCaptionsCacheNode(BaseCanvasNode):
    """Drop-in replacement for BLIP-2 + summarizer chain using paper-released captions.

    Loads two pre-computed JSON artifacts shipped in NavGPT's Dropbox release:
    - ``observations_list_summarized/{scan}.json[vp]`` → list of 8 per-heading
      sentences, each summarising the 3 elevation captures via GPT-3.5.
    - ``observations_summarized/{scan}_summarized.json[vp]`` → single sentence
      summarising the entire viewpoint.

    Output ports match BLIP-2 + summarizer chain so this can replace both
    nodes via graph rewiring without touching downstream consumers.

    Direction labels follow the same 8-compass convention as BLIP-2 and
    ``_format_observation_compass`` — ``Front, Front Right, Right, …``.
    """

    node_type = "navgpt_mp3d_tools__paper_captions_cache"
    display_name = "Paper Captions Cache (NavGPT)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = "Load paper-released observations_list_summarized + observations_summarized"
    category = "perception"
    icon = "FileText"
    input_ports = [
        PortDef("scan_id", "TEXT", "Scan ID from env (reset.scan_id or panorama.scan_id)"),
        PortDef("viewpoint_id", "TEXT", "Viewpoint ID from env"),
        PortDef("trigger", "TEXT", "Optional sequencing trigger", optional=True),
    ]
    output_ports = [
        PortDef("descriptions", "TEXT", "8-direction scene descriptions (formatted text)"),
        PortDef("summary", "TEXT", "1-sentence per-viewpoint summary"),
        PortDef("descriptions_json", "TEXT", "Per-direction captions as JSON array of 8 strings"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        scan_id = str(inputs.get("scan_id", "")).strip()
        viewpoint_id = str(inputs.get("viewpoint_id", "")).strip()

        if not scan_id or not viewpoint_id:
            self._self_log("error", f"missing scan/vp: scan={scan_id!r} vp={viewpoint_id!r}")
            return {"descriptions": "", "summary": "", "descriptions_json": "[]"}

        # Load list (per-heading sentences)
        try:
            scan_list = _load_paper_captions_list(scan_id)
        except FileNotFoundError as exc:
            self._self_log("error", f"observations_list_summarized missing for {scan_id}: {exc}")
            return {"descriptions": "", "summary": "", "descriptions_json": "[]"}

        sentences = scan_list.get(viewpoint_id)
        if sentences is None:
            self._self_log(
                "error", f"vp {viewpoint_id} not in observations_list_summarized/{scan_id}"
            )
            return {"descriptions": "", "summary": "", "descriptions_json": "[]"}

        # Pad/truncate to 8 (defensive — paper always ships 8)
        if not isinstance(sentences, list):
            sentences = []
        if len(sentences) < 8:
            sentences = list(sentences) + [""] * (8 - len(sentences))
        elif len(sentences) > 8:
            sentences = sentences[:8]

        # Load summary
        summary = ""
        try:
            scan_summary = _load_paper_captions_summary(scan_id)
            summary = str(scan_summary.get(viewpoint_id, "")).strip()
        except FileNotFoundError as exc:
            self._self_log("warn", f"observations_summarized missing for {scan_id}: {exc}")

        # Format descriptions text (matches BLIP-2 output convention)
        dir_labels = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]
        desc_lines = [f"{dir_labels[i]}: {sentences[i]}" for i in range(8)]
        descriptions = "\n".join(desc_lines)

        self._self_log("scan_id", scan_id)
        self._self_log("viewpoint_id", viewpoint_id)
        self._self_log("summary", summary[:200])
        for i, s in enumerate(sentences):
            self._self_log(f"caption_{i}", s[:200])

        return {
            "descriptions": descriptions,
            "summary": summary,
            "descriptions_json": json.dumps(sentences),
        }


# ══════════════════════════════════════════════════════════════════════
# NavGPT-MP3D Method Nodes — agent reasoning logic (not env access)
# ══════════════════════════════════════════════════════════════════════

_MAX_SCRATCHPAD_LENGTH = 7000  # Match original NavGPT MAX_SCRATCHPAD_LENGTH


def _lr_label(h_deg: float) -> str:
    """Format heading as ``right X.XX`` / ``left X.XX`` — NavGPT lr() helper."""
    if h_deg > 0:
        return f"right {h_deg:.2f}"
    if h_deg < 0:
        return f"left {-h_deg:.2f}"
    return "right 0.00"


def _normalize_heading_deg(h_deg: float) -> float:
    """Normalize heading to the (-180, 180] range used by NavGPT narration."""
    h = h_deg
    while h > 180:
        h -= 360
    while h <= -180:
        h += 360
    return h


def _format_turned_angle(delta_deg: float, curr_heading_deg: float) -> str:
    """Format the per-step ``Turn heading direction X from Y to Z.`` prefix.

    Mirrors NavGPT ``make_equiv_action`` heading-delta narration.  The LLM
    reads this as part of each step's Observation so it can track rotation.
    """
    curr = _normalize_heading_deg(curr_heading_deg)
    prev = _normalize_heading_deg(curr - delta_deg)
    return (
        f"Turn heading direction {abs(delta_deg):.2f} degrees "
        f"from {_lr_label(prev)} to {_lr_label(curr)}."
    )


def _demote_last_full_obs(scratchpad: str) -> str:
    """Demote the most recent FULL 8-compass obs block back to its compact line.

    Paper-faithful to ``_construct_scratchpad`` (agent.py:117-130) — only the
    last intermediate step keeps the full observation; prior steps are
    compressed to the one-line ``Current viewpoint "X": Scene from ...``
    narrative. On each new step, we strip the 8-compass text sitting between
    that summary line and the next ``\\nThought:``.
    """
    marker_idx = scratchpad.rfind("Scene from the viewpoint is a ")
    if marker_idx == -1:
        return scratchpad
    end_of_summary_line = scratchpad.find("\n", marker_idx)
    if end_of_summary_line == -1:
        return scratchpad
    thought_idx = scratchpad.find("\nThought:", end_of_summary_line)
    if thought_idx == -1:
        return scratchpad
    return scratchpad[:end_of_summary_line] + scratchpad[thought_idx:]


def _parse_navgpt_action(raw: str) -> str:
    """Extract a viewpoint ID or STOP from raw LLM output.

    Mirrors the original NavGPT ``NavGPTOutputParser``:
    1. ``Action Input: "([a-fA-F0-9]{32})"``
    2. ``Final Answer: (STOP|Finished!|32-char hex)``
    3. Fallback: any 32-char hex anywhere in string
    """
    # 1. Action Input format (ReAct tool-calling)
    m = re.search(
        r'Action\s*\d*\s*Input\s*\d*\s*:[\s]*"?([a-fA-F0-9]{32})"?',
        raw,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # 2. Final Answer format
    fa = re.search(r"Final Answer:\s*(.*)", raw, re.DOTALL)
    if fa:
        answer = fa.group(1).strip().strip('"').strip("'")
        if answer.upper() in ("STOP", "FINISHED", "FINISHED!"):
            return "STOP"
        if re.match(r"^[a-fA-F0-9]{32}$", answer):
            return answer

    # 3. Fallback: any 32-char hex in the string
    hex_m = re.search(r"[a-fA-F0-9]{32}", raw)
    if hex_m:
        return hex_m.group(0)

    # Check for stop keywords anywhere
    upper = raw.upper()
    if "FINISHED" in upper or "STOP" in upper:
        return "STOP"

    return raw  # pass through — graph_navigate will handle the error


# ══════════════════════════════════════════════════════════════════════
# Episode-info access (live env lookup via component registry)
# Mirrors the pattern in basic_agent.py::_try_get_habitat_mgr — avoids
# importing matterport3d directly (cross-nodeset coupling).
# ══════════════════════════════════════════════════════════════════════


def _try_get_mp3d_mgr() -> Any:
    """Return the live MP3DEnvManager from the registered EnvMP3DNodeSet, or None.

    Only returns a value in local-mode (manager runs in-process).  In
    server-mode the backend-side ``EnvMP3DNodeSet`` is a shell without
    ``_mgr``; callers must fall back to
    :func:`_fetch_episode_info_via_env_panel`.
    """
    try:
        from app.state import get_services

        for ns in get_services().workspace_component_registry._live_nodesets.values():
            if ns.__class__.__name__ != "EnvMP3DNodeSet":
                continue
            mgr = getattr(ns, "_mgr", None)
            if mgr is not None and getattr(mgr, "initialized", False):
                return mgr
    except Exception:
        pass
    return None


async def _fetch_episode_info_via_env_panel() -> dict | None:
    """Server-mode fallback — read episode metadata via the RemoteEnvPanelProxy.

    When ``env_mp3d`` runs in server mode the manager lives in a subprocess,
    so ``_live_nodesets['env_mp3d']._mgr`` is ``None`` on the backend side.
    The env panel proxy still works: its ``on_load()`` returns the current
    state dict (``{available, dataset, split, episode_index, current_episode: {instruction, instr_id, scan, ...}}``).
    """
    try:
        from app.components.env_panel import get_env_panel

        panel = get_env_panel("env_mp3d")
        if panel is None:
            return None
        state = await panel.on_load()
        if not state or not state.get("available"):
            return None
        ep = state.get("current_episode") or {}
        return {
            "instruction": ep.get("instruction", ""),
            "episode_id": ep.get("instr_id", ""),
            "scan_id": ep.get("scan", ""),
        }
    except Exception:
        return None


class MP3DGetInstructionNode(BaseCanvasNode):
    """Source node — fetch the current R2R episode's navigation instruction.

    Replaces the deleted ``env_mp3d__new_episode`` as the graph's
    instruction source.  Episode selection itself is handled pre-flight
    by ``MP3DEnvPanel`` (ADR-025) — this node only *reads* the
    currently-selected episode's instruction from the live env manager.
    """

    node_type = "navgpt_mp3d_tools__get_instruction"
    display_name = "MP3D: Get Instruction"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")
    description = "Read current R2R episode instruction from live MP3D env"
    category = "tool"
    icon = "FileText"
    input_ports: ClassVar[list] = []  # no inputs — source node
    output_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction for current episode"),
        PortDef("episode_id", "TEXT", "Current episode ID"),
        PortDef("scan_id", "TEXT", "Current scan ID"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import asyncio

        # Local-mode: reach into the live EnvMP3DNodeSet's in-process manager.
        mgr = _try_get_mp3d_mgr()
        if mgr is not None:
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(mgr.executor, mgr.get_episode_info)
        else:
            # Server-mode: the manager lives in a subprocess; read episode
            # metadata via the RemoteEnvPanelProxy's state endpoint.
            info = await _fetch_episode_info_via_env_panel()
            if info is None:
                self._self_log("mgr", "not_loaded")
                return {"instruction": "(MP3D env not loaded)", "episode_id": "", "scan_id": ""}
            self._self_log("mgr", "via_env_panel_proxy")

        instruction = info.get("instruction", "")
        episode_id = str(info.get("episode_id", ""))
        scan_id = str(info.get("scan_id", ""))

        self._self_log("episode_id", episode_id)
        self._self_log("scan_id", scan_id)
        self._self_log("instruction_preview", instruction[:200])

        return {
            "instruction": instruction,
            "episode_id": episode_id,
            "scan_id": scan_id,
        }


class NavGPTParseActionNode(BaseCanvasNode):
    """Parse viewpoint ID from LLM output using NavGPT regex.

    Extracts a 32-char hex viewpoint ID or STOP signal from the
    orchestrator LLM's ReAct-format output.  Pure function — no
    graph_state coupling.

    Mirrors ``navgpt__parse_action`` (CE variant) but for discrete
    viewpoint IDs instead of FORWARD/LEFT/RIGHT/STOP keywords.
    """

    node_type = "navgpt_mp3d_tools__parse_action"
    display_name = "NavGPT: Parse Action (MP3D)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")
    description = "Extract viewpoint ID or STOP from LLM response (NavGPT regex)"
    category = "processing"
    icon = "GitBranch"
    input_ports = [
        PortDef("llm_response", "TEXT", "Raw orchestrator LLM output"),
    ]
    output_ports = [
        PortDef("viewpoint_id", "TEXT", "Extracted viewpoint ID (or STOP)"),
        PortDef("is_stop", "BOOL", "True when agent signals STOP/Finished"),
        PortDef("thought", "TEXT", "Extracted thought text before action"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        llm_response = str(inputs.get("llm_response", "")).strip()
        if not llm_response:
            return {"viewpoint_id": "", "is_stop": False, "thought": ""}

        vp_id = _parse_navgpt_action(llm_response)
        is_stop = vp_id.upper() in ("STOP", "FINISHED", "FINISHED!")

        thought = llm_response
        thought_match = re.search(
            r"(?:Thought)\s*:\s*(.+?)(?:\nAction\s*:|$)",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        self._self_log("parsed_action", vp_id)
        self._self_log("is_stop", is_stop)
        self._self_log("thought", thought[:200] if thought else "")

        return {"viewpoint_id": vp_id, "is_stop": is_stop, "thought": thought}


class NavGPTInitObservationNode(BaseCanvasNode):
    """Format init_observation and initial history for the first step.

    Feeds the ``Initialize`` pivot with values that populate
    ``iter_in.init_ports`` on step 0. Matches original NavGPT ``rollout()``
    and ``init_trajecotry()`` initialization.
    """

    node_type = "navgpt_mp3d_tools__init_observation"
    display_name = "NavGPT: Init Observation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")
    description = "Seed init_observation + initial history for NavGPT orchestrator"
    category = "processing"
    icon = "Play"
    input_ports = [
        PortDef("observation", "TEXT", "8-compass observation text from graph_observe"),
        PortDef("viewpoint_id", "TEXT", "Starting viewpoint ID"),
        PortDef("summary", "TEXT", "1-sentence scene summary"),
    ]
    output_ports = [
        PortDef(
            "init_observation",
            "TEXT",
            "Formatted init observation for orchestrator prompt (step-0 1-sentence GPT summary, paper §3.4)",
        ),
        PortDef(
            "history_0",
            "TEXT",
            "One-line init history: 'Navigation start, no actions taken yet. Current viewpoint {vp}: Scene from the viewpoint is a {summary}' — used as {init_observation} swap on step >= 1 (original NavGPT get_full_inputs L141)",
        ),
        PortDef(
            "scratchpad_init",
            "TEXT",
            "Seed empty scratchpad '' for step-0 — routed through Initialize into iter_in.init_ports.scratchpad",
        ),
        PortDef("observation_out", "TEXT", "Pass-through observation (for downstream sequencing)"),
        PortDef(
            "viewpoint_id_out", "TEXT", "Pass-through viewpoint ID (for downstream sequencing)"
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        observation = str(inputs.get("observation", ""))
        vp_id = str(inputs.get("viewpoint_id", ""))
        summary = str(inputs.get("summary", "")) or "(scene description unavailable)"

        # Step-0 init_observation = full 8-compass observation WITH navigable
        # viewpoint IDs (paper Fig 11 left: `Init Observation:` is the detailed
        # 8-heading layout). This is what gives the LLM the viewpoint IDs it
        # must pick from. `observation` arrives from init_format.
        init_obs = observation

        # From step 1 onward, {init_observation} is swapped with `history_0`:
        # a 1-sentence recap seeded by the GPT-3.5 viewpoint summarizer
        # (paper Fig 11 right, original get_full_inputs L141).
        init_history = (
            f"Navigation start, no actions taken yet.\n"
            f'Current viewpoint "{vp_id}": '
            f"Scene from the viewpoint is a {summary}"
        )

        self._self_log("viewpoint_id", vp_id)
        self._self_log("summary", summary[:200])
        self._self_log("init_obs_length", len(init_obs))

        return {
            "init_observation": init_obs,
            "history_0": init_history,
            "scratchpad_init": "",
            "observation_out": observation,
            "viewpoint_id_out": vp_id,
        }


class NavGPTScratchpadWriterNode(BaseCanvasNode):
    """Build ReAct scratchpad entry from LLM response + observation.

    Uses SHORT history format for intermediate steps (M1), matching the
    original NavGPT ``_construct_scratchpad()`` intermediate step format.
    """

    node_type = "navgpt_mp3d_tools__scratchpad_writer"
    display_name = "NavGPT: Scratchpad Writer"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")
    description = "Build ReAct scratchpad entry from LLM response + observation"
    category = "processing"
    icon = "NotebookPen"
    input_ports = [
        PortDef(
            "scratchpad_in",
            "TEXT",
            "Current scratchpad from iter_in (init '' on step 0, accumulated on step >= 1)",
        ),
        PortDef("llm_response", "TEXT", "Raw orchestrator LLM output"),
        PortDef("observation", "TEXT", "8-compass observation from graph_observe"),
        PortDef("viewpoint_id", "TEXT", "Current viewpoint ID after navigation"),
        PortDef("summary", "TEXT", "1-sentence scene summary"),
        PortDef("success", "TEXT", "Navigation success: 'true' or 'false'"),
        PortDef("error", "TEXT", "Error message from navigate (optional)", optional=True),
        PortDef(
            "turned_angle",
            "TEXT",
            "Signed heading delta in degrees (from navigate_to)",
            optional=True,
        ),
        PortDef("heading", "TEXT", "Current heading in degrees after navigation", optional=True),
    ]
    output_ports = [
        PortDef(
            "scratchpad_out",
            "TEXT",
            "Updated scratchpad after appending Thought/Action/Observation",
        ),
        PortDef("observation_out", "TEXT", "Pass-through observation (for iter_out sequencing)"),
        PortDef("viewpoint_id_out", "TEXT", "Pass-through viewpoint ID (for iter_out sequencing)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        current = str(inputs.get("scratchpad_in", ""))
        llm_response = str(inputs.get("llm_response", ""))
        observation = str(inputs.get("observation", ""))
        vp_id = str(inputs.get("viewpoint_id", ""))
        summary = str(inputs.get("summary", "")) or "(scene description unavailable)"
        success = str(inputs.get("success", "true")).strip().lower()
        error = str(inputs.get("error", ""))
        turned_raw = str(inputs.get("turned_angle", "0")).strip()
        heading_raw = str(inputs.get("heading", "0")).strip()

        # Capture the prior "Visited Viewpoints" line BEFORE stripping it.
        # This is the cumulative source of truth that survives
        # MAX_SCRATCHPAD_LENGTH truncation — re-extracting from
        # `Current viewpoint "<id>"` markers alone would shrink the list
        # whenever early markers fall off the truncated front.
        prior_match = re.search(r"Visited Viewpoints \([^)]*\):([^\n]*)", current)
        prior_visited_ids: list[str] = (
            re.findall(r"[0-9a-f]{32}", prior_match.group(1)) if prior_match else []
        )

        # Strip any prior "Visited Viewpoints" line so we can re-render a
        # fresh one at the bottom of the new scratchpad below.
        current = re.sub(
            r"\n*Visited Viewpoints \([^)]*\):[^\n]*\n*",
            "\n",
            current,
        )

        # Demote the prior step's FULL 8-compass obs back to its compact
        # one-liner (paper: _construct_scratchpad) so only the step we're
        # about to write keeps a full observation.
        current = _demote_last_full_obs(current)

        # Append LLM response (Thought/Action/Action Input)
        current += llm_response

        if success in ("true", "1"):
            # Build the "Turn heading direction X degrees from Y to Z." narration
            # (NavGPT get_history() prefix — lets the LLM track rotation).
            try:
                delta = float(turned_raw)
            except ValueError:
                delta = 0.0
            try:
                curr_heading = float(heading_raw)
            except ValueError:
                curr_heading = 0.0
            turned_str = _format_turned_angle(delta, curr_heading)

            # Paper-faithful: the LAST intermediate step's observation is FULL
            # (8-compass with navigable IDs); prior historic entries keep their
            # summary line. MAX_SCRATCHPAD_LENGTH truncation from the tail
            # approximates the paper's demote-to-short policy — oldest full
            # entries fall off as scratchpad grows.
            obs_block = observation.strip() if observation else ""
            current += (
                f"\nObservation: \n{turned_str}\n"
                f'Current viewpoint "{vp_id}": '
                f"Scene from the viewpoint is a {summary}"
            )
            if obs_block:
                current += f"\n{obs_block}"
            current += "\nThought:"
        else:
            # ── Error: paper-faithful "not valid, agent not moved" message
            #    (mirrors agent.py:402 — prevents the LLM from fabricating IDs) ──
            err_msg = error or f"ViewpointID '{vp_id}' is not valid, agent not moved."
            current += (
                f"\nObservation: {err_msg} "
                f"DO NOT fabricate nonexistent IDs. "
                f"Choose a viewpoint ID from the Navigable Viewpoints listed below."
                f"\nCurrent Viewpoint:\n{observation}"
                f"\nThought:"
            )

        # Truncate to MAX_SCRATCHPAD_LENGTH from the end
        if len(current) > _MAX_SCRATCHPAD_LENGTH:
            current = current[-_MAX_SCRATCHPAD_LENGTH:]

        # Render a "Visited Viewpoints" line just above the trailing
        # "Thought:" stub. The list grows monotonically: take the prior
        # captured list (which already accumulates across iterations) and
        # union with the current vp_id. Robust against truncation —
        # parsing surviving `Current viewpoint "<id>"` markers from
        # `current` after truncation would lose IDs whenever the early
        # part of the scratchpad falls off the front.
        visited_uniq: list[str] = list(prior_visited_ids)
        if vp_id and re.fullmatch(r"[0-9a-f]{32}", vp_id) and vp_id not in visited_uniq:
            visited_uniq.append(vp_id)
        if visited_uniq:
            visited_block = (
                "\nVisited Viewpoints (oldest→newest, MUST NOT revisit "
                f"via action_maker): {', '.join(visited_uniq)}"
            )
            tail = "\nThought:"
            if current.endswith(tail):
                current = current[: -len(tail)] + visited_block + tail
            else:
                current += visited_block

        self._self_log("success", success)
        self._self_log("scratchpad_length", len(current))
        self._self_log("viewpoint_id", vp_id)
        self._self_log("visited_count", len(visited_uniq))

        return {
            "scratchpad_out": current,
            "observation_out": observation,
            "viewpoint_id_out": vp_id,
        }


# ══════════════════════════════════════════════════════════════════════
# Observation formatting helpers (moved from matterport3d.py — Phase 2)
# These are NavGPT-specific prompt-shaping logic, not env plumbing.
# ══════════════════════════════════════════════════════════════════════


def _format_observation_compass(
    navigable: dict,
    current_heading_rad: float,
    scene_descriptions: list | None = None,
    objects_per_sector: list | None = None,
) -> str:
    """Format observation in 8-compass-direction style matching original NavGPT.

    Byte-identical (on matched-unit inputs) to ``NavAgent.modify_heading_angles``
    in NavGPT's ``nav_src/agent.py`` (L251-L327); see
    ``workspace/nodesets/_upstream/navgpt/fetch_upstream.sh``.

    Unit convention (differs from original — consistent across this module):
    ALL headings are in RADIANS — ``current_heading_rad``, each
    ``navigable[vp]["heading"]``, and each
    ``objects_per_sector[i][obj]["heading"]``. ``_merge_rcnn_to_sectors``
    produces radians for downstream consumption.

    Args:
        navigable: dict of ``{vp_id: {heading, elevation, distance}}`` (radians).
        current_heading_rad: agent's current heading in radians.
        scene_descriptions: 8 strings (one per 45-degree sector), or ``None``.
        objects_per_sector: 8 dicts of ``{obj_name: {heading, distance}}``, or ``None``.

    Returns:
        Formatted 8-direction observation string.
    """
    import math

    heading_deg = math.degrees(current_heading_rad)

    def _normalize(angle: float) -> float:
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle

    def _lr(angle: float) -> str:
        return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

    directions = [
        "Front",
        "Front Right",
        "Right",
        "Rear Right",
        "Rear",
        "Rear Left",
        "Left",
        "Front Left",
    ]

    # Which observation index maps to which direction (same formula as original)
    range_idx = int((heading_deg - 22.5) // 45) + 1
    obs_idx = [(i + range_idx) % 8 for i in range(8)]

    # Group navigable viewpoints into direction sectors
    candidate_range: dict[int, dict] = {}
    for vp_id, vp_data in navigable.items():
        vp_heading_deg = math.degrees(vp_data["heading"])
        vp_range_idx = int((vp_heading_deg - 22.5) // 45) + 1
        rel_heading = _normalize(vp_heading_deg - heading_deg)
        vp_desc = f"{_lr(rel_heading)}, {vp_data['distance']:.2f}m"
        candidate_range.setdefault(vp_range_idx, {})[vp_id] = vp_desc

    # Compute angle ranges for each sector
    angle_ranges = [
        (angle - 22.5 - heading_deg, angle + 22.5 - heading_deg) for angle in range(0, 360, 45)
    ]

    formatted: list[str] = []
    for direction, idx in zip(directions, obs_idx, strict=False):
        rel1 = _normalize(angle_ranges[idx][0])
        rel2 = _normalize(angle_ranges[idx][1])

        s = f"{direction}, range ({_lr(rel1)} to {_lr(rel2)}): "

        # Scene description
        if scene_descriptions and idx < len(scene_descriptions) and scene_descriptions[idx]:
            s += f"\n'{scene_descriptions[idx]}'"
        else:
            s += "\n(no scene description available)"

        # Objects
        if objects_per_sector and idx < len(objects_per_sector) and objects_per_sector[idx]:
            obj_dict = {}
            for obj_name, obj_data in objects_per_sector[idx].items():
                obj_heading = obj_data.get("heading", 0)
                if isinstance(obj_heading, (int, float)):
                    rel_obj = _normalize(math.degrees(obj_heading) - heading_deg)
                else:
                    rel_obj = 0.0
                obj_dist = obj_data.get("distance", 0)
                obj_dict[obj_name] = f"{_lr(rel_obj)}, {obj_dist:.2f}m"
            s += f"\n{direction} Objects in 3m: {obj_dict}"
        else:
            s += f"\n{direction} Objects in 3m: None"

        # Navigable viewpoints in this sector
        # Note: NavGPT original uses no space after ":" when present, space when "None"
        if candidate_range.get(idx):
            s += f"\n{direction} Navigable Viewpoints:{candidate_range[idx]}"
        else:
            s += f"\n{direction} Navigable Viewpoints: None"

        formatted.append(s)

    return "\n".join(formatted)


def _merge_rcnn_to_sectors(
    raw_objects: list,
    n_headings: int | None = None,
) -> list:
    """Convert RCNN per-view detections to per-sector dict format.

    ``FasterRCNNDetectNode`` outputs ``list[list[dict]]`` (one list per view,
    ordered outer=elevation inner=heading — matching
    ``render_panorama_with_elevations``).  ``_format_observation_compass``
    expects ``list[dict]`` where each dict maps
    ``{obj_name: {heading: rad, distance: m}}``.

    This function merges all elevation views per heading sector, deduplicates
    by object name (keeping highest-confidence detection), and converts to the
    expected per-sector dict format. ``n_headings`` is inferred from the view
    count (3 elevation rings) when not given.
    """
    import math

    n_views = len(raw_objects)
    if n_headings is None:
        n_headings = _infer_n_headings(n_views)
    if n_views == 0:
        return [{} for _ in range(n_headings)]

    n_elevs = n_views // n_headings if n_views >= n_headings else 1

    sectors: list = []
    for h in range(n_headings):
        sector_objects: dict = {}
        sector_center_deg = h * (360.0 / n_headings)  # 0, 45, 90, ...

        # Gather objects from all elevation views for this heading
        # View ordering: outer=elevation, inner=heading → index = h + e * n_headings
        for e in range(n_elevs):
            view_idx = h + e * n_headings
            if view_idx >= n_views:
                continue
            for obj in raw_objects[view_idx]:
                name = obj.get("name", "unknown")
                # Absolute heading = sector center + relative heading within view
                abs_heading_rad = math.radians(
                    sector_center_deg + obj.get("rel_heading_deg", 0),
                )
                distance = obj.get("estimated_distance_m", 3.0)
                confidence = obj.get("confidence", 0)
                # Keep highest-confidence detection per name within sector
                if name not in sector_objects or confidence > sector_objects[name].get("_conf", 0):
                    sector_objects[name] = {
                        "heading": abs_heading_rad,
                        "distance": distance,
                        "_conf": confidence,
                    }

        # Strip internal _conf key
        for v in sector_objects.values():
            v.pop("_conf", None)
        sectors.append(sector_objects)

    return sectors


class NavGPTObservationFormatNode(BaseCanvasNode):
    """Format MP3D observation into NavGPT 8-compass prompt text.

    Consumes structured env output (navigable dict, scene descriptions,
    scene objects, current heading) and produces the NavGPT observation
    string. This is NavGPT-specific prompt shaping — env returns data,
    this node turns it into prose.

    Ports the ``modify_heading_angles()`` formatting from original NavGPT
    ``agent.py``. RCNN per-view detections are merged into 8 sectors.
    """

    node_type = "navgpt_mp3d_tools__observation_format"
    display_name = "NavGPT: Observation Format"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")
    description = "Format env observation as NavGPT 8-compass prompt text"
    category = "processing"
    icon = "Eye"
    input_ports = [
        PortDef("heading", "TEXT", "Current heading in degrees"),
        PortDef("navigable_json", "TEXT", "Navigable viewpoints as JSON"),
        PortDef(
            "scene_descriptions_json", "TEXT", "Scene descriptions as JSON list", optional=True
        ),
        PortDef(
            "scene_objects_json", "TEXT", "Scene objects as JSON (raw or per-sector)", optional=True
        ),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "NavGPT 8-compass formatted observation"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import math

        heading_deg_raw = str(inputs.get("heading", "0")).strip()
        try:
            heading_deg = float(heading_deg_raw)
        except (ValueError, TypeError):
            heading_deg = 0.0
        current_heading_rad = math.radians(heading_deg)

        try:
            nav = json.loads(inputs.get("navigable_json") or "{}")
        except (ValueError, TypeError) as exc:
            self._self_log("navigable_parse_error", str(exc))
            nav = {}

        scene_descs: list | None = None
        raw_descs = inputs.get("scene_descriptions_json")
        if raw_descs:
            try:
                parsed = json.loads(raw_descs)
                if isinstance(parsed, list) and parsed:
                    scene_descs = parsed
            except (ValueError, TypeError) as exc:
                self._self_log("scene_descs_parse_error", str(exc))

        objects: list | None = None
        raw_objects = inputs.get("scene_objects_json")
        if raw_objects:
            try:
                parsed = json.loads(raw_objects)
                if isinstance(parsed, list) and parsed:
                    objects = parsed
            except (ValueError, TypeError) as exc:
                self._self_log("scene_objects_parse_error", str(exc))

        # Normalize RCNN list-of-lists to per-sector dicts
        if objects is not None and len(objects) > 0 and isinstance(objects[0], list):
            objects = _merge_rcnn_to_sectors(objects)

        observation = _format_observation_compass(
            nav,
            current_heading_rad,
            scene_descs,
            objects,
        )

        self._self_log("observation_length", len(observation))
        self._self_log("observation_preview", observation[:400])
        self._self_log("navigable_count", len(nav))
        self._self_log("has_scene_descs", scene_descs is not None)

        return {"observation": observation}


# ══════════════════════════════════════════════════════════════════════
# NodeSet wrapper
# ══════════════════════════════════════════════════════════════════════


class NavGPTMP3DToolsNodeSet(BaseNodeSet):
    """NavGPT MP3D tools — perception + method-specific reasoning nodes.

    Perception nodes (online vision pipeline):
    - ``BLIP2CaptionNode``: BLIP-2 ViT-G FlanT5-XL image captioning
    - ``FasterRCNNDetectNode``: Faster R-CNN object detection

    Method nodes (NavGPT-specific agent logic, decoupled from env):
    - ``MP3DGetInstructionNode``: Read current episode instruction
    - ``NavGPTObservationFormatNode``: Format env raw data → 8-compass prompt
    - ``NavGPTParseActionNode``: Extract viewpoint ID from LLM output
    - ``NavGPTInitObservationNode``: Format init_observation + history (feeds Initialize)
    - ``NavGPTScratchpadWriterNode``: Build ReAct scratchpad entries

    All run in the agentcanvas environment (Python 3.10+, local mode).
    Vision models are lazy-loaded on first use and stay in GPU memory.
    """

    name = "navgpt_mp3d_tools"
    display_name = "NavGPT MP3D Tools"
    description = (
        "NavGPT MP3D: perception (BLIP-2 + Faster R-CNN) + reasoning (parse, scratchpad, init)"
    )

    def get_tools(self) -> list:
        return [
            BLIP2CaptionNode(),
            InstructBlipCaptionNode(),
            FasterRCNNDetectNode(),
            OpenVocabDetectNode(),
            PaperObjectsCacheNode(),
            PaperCaptionsCacheNode(),
            MP3DGetInstructionNode(),
            NavGPTParseActionNode(),
            NavGPTInitObservationNode(),
            NavGPTScratchpadWriterNode(),
            NavGPTObservationFormatNode(),
        ]

    async def initialize(self) -> None:
        log.info("NavGPT MP3D Tools nodeset initialised (models load on first use)")

    async def shutdown(self) -> None:
        global _blip2_model, _blip2_processor, _blip2_device
        global _rcnn_model, _rcnn_device

        import torch

        if _blip2_model is not None:
            del _blip2_model, _blip2_processor
            _blip2_model = None
            _blip2_processor = None
            _blip2_device = None
            log.info("BLIP-2 model unloaded")

        if _rcnn_model is not None:
            del _rcnn_model
            _rcnn_model = None
            _rcnn_device = None
            log.info("Faster R-CNN model unloaded")

        global _gdino_model, _gdino_processor, _gdino_device
        if _gdino_model is not None:
            del _gdino_model, _gdino_processor
            _gdino_model = None
            _gdino_processor = None
            _gdino_device = None
            log.info("GroundingDINO-tiny model unloaded")

        global _instructblip_model, _instructblip_processor, _instructblip_device
        if _instructblip_model is not None:
            del _instructblip_model, _instructblip_processor
            _instructblip_model = None
            _instructblip_processor = None
            _instructblip_device = None
            log.info("InstructBLIP model unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
