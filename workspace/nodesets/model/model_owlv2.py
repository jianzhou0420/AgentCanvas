from __future__ import annotations

"""OWLv2 open-vocabulary detection — server-mode foundation-model nodeset.

The zero-shot detection sibling of ``model_grounding_dino``, from a different
lineage: OWLv2 (OWL-ViT v2) scores **each candidate label independently** with a
CLIP-style open-vocab head, so you hand it a *set* of queries ("chair", "door",
"trash can") and it returns every matching box per query. GroundingDINO parses a
single free-text caption; OWLv2 takes a label list — the natural fit for "find
all of these object classes in the frame" (object goals, affordance targets,
scene inventory), and its scores are directly comparable across the label set.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``model_id`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_owlv2__detect  (image_b64: TEXT, [queries: ANY])
        → result: TEXT  (JSON {boxes:[{xyxy,score,phrase}], count, image_w, image_h, queries})

The ``result`` schema is **identical** to ``model_grounding_dino__detect`` (pixel
``xyxy`` + score + matched phrase), so a graph can swap detector backends without
touching downstream nodes. ``queries`` accepts a list[str], a JSON list, or a
comma/newline-separated string; blank falls back to the configured default. Boxes
are returned at the input image's resolution, filtered by ``threshold``.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers`` ``Owlv2ForObjectDetection``
+ ``AutoProcessor`` (OWLv2 checkpoints, ungated). Override the env with
$OWLV2_PYTHON and the device with $OWLV2_DEVICE (auto → cuda when available). This
file must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_owlv2/load?mode=server

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

log = logging.getLogger("agentcanvas.model_owlv2")

_MODEL_ID_DEFAULT = "google/owlv2-base-patch16-ensemble"

# Curated OWLv2 open-vocab detector variants.
_MODEL_OPTIONS = [
    {"value": "google/owlv2-base-patch16-ensemble", "label": "OWLv2 Base p16 (ensemble)"},
    {"value": "google/owlv2-base-patch16", "label": "OWLv2 Base p16"},
    {"value": "google/owlv2-base-patch16-finetuned", "label": "OWLv2 Base p16 (finetuned)"},
    {"value": "google/owlv2-large-patch14-ensemble", "label": "OWLv2 Large p14 (ensemble)"},
    {"value": "google/owlv2-large-patch14", "label": "OWLv2 Large p14"},
    {"value": "google/owlv2-large-patch14-finetuned", "label": "OWLv2 Large p14 (finetuned)"},
]
_DEFAULT_QUERIES = "chair, door, table"
_DEFAULT_THRESHOLD = 0.1


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("OWLV2_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _Owlv2Engine:
    """Lazy singleton registry: one frozen OWLv2 detector per ``model_id``.

    Holds only loaded weights — no per-call state. The single-flight inference
    lock bounds peak VRAM to one in-flight forward under concurrent eval workers.
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
    def get(cls, model_id: str) -> "_Owlv2Engine":
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
                from transformers import AutoProcessor, Owlv2ForObjectDetection

                self.device = _resolve_device()
                model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as exc:
                log.warning("OWLv2 load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("OWLv2 ready (%s, device=%s)", self.model_id, self.device)
            return True

    def detect(self, image: np.ndarray, queries: list, threshold: float) -> dict:
        """Open-vocab detection: queries → xyxy boxes at input resolution.

        ``labels`` from the post-processor index into the flat ``queries`` list,
        so the matched phrase is ``queries[label]`` (version-robust — no reliance
        on the optional ``text_labels`` key)."""
        import torch
        from PIL import Image

        pil = Image.fromarray(image, "RGB")
        h, w = image.shape[:2]
        with self._infer_lock:
            inp = self.processor(text=[queries], images=pil, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp)
            target = torch.tensor([[h, w]], device=self.device)
            res = self.processor.post_process_grounded_object_detection(
                out, threshold=threshold, target_sizes=target)[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        labels = res["labels"].detach().cpu().numpy()
        out_boxes = []
        for box, score, lab in zip(boxes, scores, labels):
            idx = int(lab)
            out_boxes.append({
                "xyxy": [int(round(float(c))) for c in box],
                "score": float(score),
                "phrase": queries[idx] if 0 <= idx < len(queries) else str(idx),
            })
        return {
            "boxes": out_boxes,
            "count": len(out_boxes),
            "image_w": int(w),
            "image_h": int(h),
            "queries": queries,
        }


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _as_query_list(x: Any) -> "list | None":
    """Normalize queries: list[str], JSON list, or comma/newline-separated string."""
    if x is None:
        return None
    if isinstance(x, list):
        out = [str(t).strip() for t in x if str(t).strip()]
        return out or None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    out = [str(t).strip() for t in parsed if str(t).strip()]
                    return out or None
            except Exception:
                pass
        parts = [p.strip() for chunk in s.split("\n") for p in chunk.split(",")]
        out = [p for p in parts if p]
        return out or None
    return None


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class Owlv2DetectTool(BaseCanvasNode):
    """Open-vocabulary detection over a set of candidate label queries.

    Returns every box matching any query as pixel ``xyxy`` + score + matched
    phrase, aligned to the input image's resolution. Same ``result`` schema as
    ``model_grounding_dino__detect`` — backend-swappable.
    """

    node_type: ClassVar[str] = "model_owlv2__detect"
    display_name: ClassVar[str] = "OWLv2: Detect (open-vocab)"
    description: ClassVar[str] = (
        "Open-vocabulary detection over a label set; boxes matching each query (xyxy+score+phrase)"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ScanSearch"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "select", label="Model", options=list(_MODEL_OPTIONS), default=_MODEL_ID_DEFAULT),
            ConfigField(
                "queries", "text",
                "Candidate labels (comma/newline separated or JSON list)",
                default=_DEFAULT_QUERIES,
            ),
            ConfigField(
                "threshold", "slider", "Detection score threshold (OWLv2 ~0.1)",
                default=_DEFAULT_THRESHOLD, min=0.01, max=1.0, step=0.01,
            ),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
        PortDef(
            "queries", "ANY",
            "Optional: override labels (list[str] / JSON / comma-sep string)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "result", "TEXT",
            "JSON {boxes:[{xyxy:[x1,y1,x2,y2], score, phrase}], count, image_w, image_h, queries}",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("image_b64") or ""
        if not b64:
            return {"result": json.dumps({"boxes": [], "count": 0, "error": "no image_b64"})}

        cfg = getattr(self, "config", None) or {}
        model_id = cfg.get("model_id", _MODEL_ID_DEFAULT)
        queries = _as_query_list(inputs.get("queries")) or _as_query_list(cfg.get("queries")) \
            or _as_query_list(_DEFAULT_QUERIES)
        try:
            threshold = float(cfg.get("threshold", _DEFAULT_THRESHOLD) or _DEFAULT_THRESHOLD)
        except (TypeError, ValueError):
            threshold = _DEFAULT_THRESHOLD

        engine = _Owlv2Engine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            if not engine._ensure():
                return None
            image = _decode_rgb(b64)
            return engine.detect(image, queries, threshold)

        result = await loop.run_in_executor(None, _run)
        if result is None:
            self._self_log("degraded", "OWLv2 engine failed to load")
            return {"result": ""}
        self._self_log("n_boxes", result.get("count", 0))
        self._self_log("queries", queries)
        return {"result": json.dumps(result)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class Owlv2NodeSet(BaseNodeSet):
    """OWLv2 open-vocabulary detector — server-mode FM nodeset."""

    name = "model_owlv2"
    description = (
        "OWLv2 open-vocabulary detection over a label set — zero-shot boxes on "
        "the shared ac-fm server (GroundingDINO-compatible result schema)"
    )
    # Stateless detector — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Owlv2 is native there).
    # Override with $OWLV2_PYTHON; device via $OWLV2_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "OWLV2_PYTHON")

    def get_tools(self) -> list:
        return [Owlv2DetectTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_owlv2 ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
