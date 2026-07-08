from __future__ import annotations

"""Florence-2 unified vision — server-mode foundation-model nodeset.

Florence-2 is a single compact seq2seq vision model that does many perception
tasks through one interface: you pick a *task token* and it emits the answer —
a caption, a set of detection boxes, region-grounded phrases, or OCR. One model
covers ground that would otherwise take a captioner + a detector + a grounder +
an OCR engine, which is exactly what an agent's "look at this frame and tell me
X" step wants: switch the task, reuse the weights.

One pure single-step primitive with a task-select config (FM-nodeset template —
stateless server, engines keyed by ``model_id`` in a lazy registry, load-failure
latch + single-flight GPU lock, everything procedural lives in the graph)::

    model_florence2__run  (image: {rgb_base64} | b64)
                          [config: task, text_input] → result: TEXT JSON

The result is Florence-2's ``post_process_generation`` dict for the chosen task,
returned verbatim as JSON. Its shape depends on the task:

    <CAPTION> / <DETAILED_CAPTION> / <MORE_DETAILED_CAPTION>
        {"<TASK>": "a caption string"}
    <OD> / <DENSE_REGION_CAPTION> / <REGION_PROPOSAL>
        {"<TASK>": {"bboxes": [[x1,y1,x2,y2], …], "labels": ["chair", …]}}
    <CAPTION_TO_PHRASE_GROUNDING> / <OPEN_VOCABULARY_DETECTION>  (need text_input)
        {"<TASK>": {"bboxes": […], "labels"/"bboxes_labels": […]}}
    <REFERRING_EXPRESSION_SEGMENTATION> / <REGION_TO_SEGMENTATION>  (need text_input)
        {"<TASK>": {"polygons": [[…]], "labels": […]}}
    <OCR>
        {"<TASK>": "recognized text"}
    <OCR_WITH_REGION>
        {"<TASK>": {"quad_boxes": […], "labels": […]}}

Boxes / polygons are in **original pixel coordinates** (post-process rescales
from Florence-2's internal 1000×1000 grid to the image's (W, H)). One image per
call — Florence-2 generates per image and the result is a per-image dict; loop at
graph level for a batch.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers``
``Florence2ForConditionalGeneration`` + ``AutoProcessor``. The default is
``florence-community/Florence-2-base`` — the transformers-native conversion of
microsoft/Florence-2 (ungated, no trust_remote_code). The original
``microsoft/Florence-2-*`` repos still use the remote-code weight/processor
layout and do **not** load natively (missing lm_head + a tokenizer with no
image_token), so point ``model_id`` at a ``florence-community/*`` repo. Override
the env with $FLORENCE2_PYTHON and the device with $FLORENCE2_DEVICE (auto →
cuda when available). This file must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_florence2/load?mode=server

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

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.model_florence2")

_MODEL_ID_DEFAULT = "florence-community/Florence-2-base"

# Task tokens Florence-2 understands. The three marked (text) consume the
# ``text_input`` config (a phrase / category list); the rest ignore it.
_TASKS = [
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<OCR>",
    "<OCR_WITH_REGION>",
]
_TASK_OPTIONS = [
    {"value": "<CAPTION>", "label": "Caption (short)"},
    {"value": "<DETAILED_CAPTION>", "label": "Caption (detailed)"},
    {"value": "<MORE_DETAILED_CAPTION>", "label": "Caption (most detailed)"},
    {"value": "<OD>", "label": "Object detection"},
    {"value": "<DENSE_REGION_CAPTION>", "label": "Dense region captions"},
    {"value": "<REGION_PROPOSAL>", "label": "Region proposals"},
    {"value": "<CAPTION_TO_PHRASE_GROUNDING>", "label": "Phrase grounding (needs text)"},
    {"value": "<OPEN_VOCABULARY_DETECTION>", "label": "Open-vocab detection (needs text)"},
    {"value": "<REFERRING_EXPRESSION_SEGMENTATION>", "label": "Referring segmentation (needs text)"},
    {"value": "<OCR>", "label": "OCR (plain text)"},
    {"value": "<OCR_WITH_REGION>", "label": "OCR with regions"},
]


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("FLORENCE2_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _Florence2Engine:
    """Lazy singleton registry: one frozen Florence-2 per ``model_id``.

    Holds only loaded weights — no cache, no per-call state. The single-flight
    inference lock bounds peak VRAM to one in-flight generate under concurrent
    eval workers (house FM-engine template).
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = None
        self.dtype = None
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str) -> "_Florence2Engine":
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
                import torch
                from transformers import AutoProcessor, Florence2ForConditionalGeneration

                self.device = _resolve_device()
                self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                model = Florence2ForConditionalGeneration.from_pretrained(
                    self.model_id, torch_dtype=self.dtype
                )
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("Florence-2 load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("Florence-2 ready (%s, device=%s, dtype=%s)", self.model_id, self.device, self.dtype)
            return True

    def run(self, image: Any, task: str, text_input: str, max_new_tokens: int, num_beams: int) -> "dict | None":
        """Single-image task-prompted generation → post-processed result dict.

        The prompt is the task token, optionally followed by ``text_input`` for
        the grounding / open-vocab / referring tasks. Returns None on load
        failure; the parsed dict is keyed by the task token.
        """
        if not self._ensure():
            return None
        import torch

        prompt = task if not text_input else task + text_input
        w, h = image.size  # PIL (W, H)
        with self._infer_lock:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
            input_ids = inputs["input_ids"].to(self.device)
            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    do_sample=False,
                )
            text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            return self.processor.post_process_generation(text, task=task, image_size=(w, h))


def _decode_rgb(b64: str) -> Any:
    from PIL import Image

    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _image_from_input(item: Any) -> "Any | None":
    """Accept a {rgb_base64} dict, a raw base64 string, or a 1-list thereof."""
    if isinstance(item, list):
        item = item[0] if item else None
    if isinstance(item, dict):
        b64 = item.get("rgb_base64") or item.get("image_base64")
    elif isinstance(item, str):
        b64 = item
    else:
        b64 = None
    if not b64:
        return None
    return _decode_rgb(b64)


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class Florence2RunTool(BaseCanvasNode):
    """Run one Florence-2 task on a single image; emit the result as JSON."""

    node_type: ClassVar[str] = "model_florence2__run"
    display_name: ClassVar[str] = "Florence-2: Run Task"
    description: ClassVar[str] = (
        "Unified vision: caption / detect / ground / OCR one image via a task token; JSON result"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Sparkles"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField(
                "model_id", "text",
                "HF Florence-2 repo id (use a florence-community/* native conversion, not microsoft/*)",
                default=_MODEL_ID_DEFAULT,
            ),
            ConfigField(
                "task", "select", label="Task",
                options=list(_TASK_OPTIONS), default="<CAPTION>",
            ),
            ConfigField(
                "text_input", "text",
                "Phrase / category (used only by the grounding, open-vocab and referring tasks)",
                default="",
            ),
            ConfigField(
                "max_new_tokens", "slider", "Max new tokens",
                default=1024, min=64, max=1024, step=64,
            ),
            ConfigField("num_beams", "slider", "Beam count", default=3, min=1, max=5, step=1),
        ],
    )
    input_ports = [
        PortDef("image", "ANY", "A {rgb_base64} dict or raw base64 string (single image)"),
    ]
    output_ports = [
        PortDef("result", "TEXT", "post_process_generation dict as JSON, keyed by the task token"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        item = inputs.get("image")
        if not item:
            return {"result": ""}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)
        task = str(cfg.get("task", "<CAPTION>") or "<CAPTION>")
        if task not in _TASKS:
            self._self_log("degraded", "unknown task %r" % task)
            return {"result": ""}
        text_input = str(cfg.get("text_input", "") or "").strip()
        max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        num_beams = int(cfg.get("num_beams", 3))

        engine = _Florence2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            image = _image_from_input(item)
            if image is None:
                return ""
            result = engine.run(image, task, text_input, max_new_tokens, num_beams)
            if result is None:
                return ""
            return json.dumps(result, ensure_ascii=False)

        envelope = await loop.run_in_executor(None, _run)
        if envelope:
            self._self_log("task", task)
        else:
            self._self_log("degraded", "no result (load failure or bad input)")
        return {"result": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class Florence2NodeSet(BaseNodeSet):
    """Florence-2 unified vision — server-mode FM nodeset."""

    name = "model_florence2"
    description = (
        "Florence-2 unified vision — caption, object detection, phrase grounding, "
        "open-vocab detection, referring segmentation and OCR from one compact "
        "seq2seq model on the shared ac-fm server"
    )
    # Stateless generator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Florence2 is native there).
    # Override with $FLORENCE2_PYTHON; device via $FLORENCE2_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "FLORENCE2_PYTHON")

    def get_tools(self) -> list:
        return [Florence2RunTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_florence2 ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
