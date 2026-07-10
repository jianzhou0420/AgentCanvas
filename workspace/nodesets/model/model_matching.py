from __future__ import annotations

"""Sparse keypoint detection + matching — server-mode foundation-model nodeset.

The localization counterpart to ``model_vggt``: where VGGT regresses *dense*
geometry in one feed-forward pass, this nodeset exposes the classic *sparse*
front-end — detect repeatable keypoints in one image, match them across two —
that SLAM / relocalization / loop-closure front-ends consume. It feeds the
pySLAM nodeset's correspondence stage and any two-view geometry step.

Both primitives ride Hugging Face ``transformers`` (the keypoint-detection /
keypoint-matching task heads), so the whole thing lives in the shared **ac-fm**
env alongside CLIP / SAM / Depth-Anything — no dedicated env, no external repo::

    model_matching__detect_keypoints  (image: {rgb_base64} | b64)
                          → keypoints : envelope {keypoints(N,2), scores(N), descriptors(N,D)}
    model_matching__match             (image_a, image_b)
                          → matches   : envelope {keypoints0(M,2), keypoints1(M,2), matching_scores(M)}

``detect_keypoints`` wraps ``SuperPointForKeypointDetection`` (variant as
``model_id``; default ``magic-leap-community/superpoint``). ``match`` wraps the
``AutoModelForKeypointMatching`` family — default ``ETH-CVG/lightglue_superpoint``
(LightGlue on SuperPoint), and by only swapping ``model_id`` you also get the
``superglue_*`` and ``efficientloftr`` matchers for free (all share the same
processor + ``post_process_keypoint_matching`` contract). ``match`` takes an
image **pair** ([[a, b]]) exactly as the transformers processor expects and
returns only the *mutually matched* keypoints, in original-image pixel coords,
above ``threshold``.

Multi-array envelope per port (each array = the raw C-contiguous float32 buffer
base64-encoded, byte-exact across HTTP, ~4× smaller than a JSON float list)::

    {"model_id":…, "image_hw":[H,W], "<name>":{"shape":[…],"dtype":"float32","b64":…}, …}

Runs **server mode** in the shared ``ac-fm`` env (Python 3.11, transformers 5.x).
Override the env with $MATCHING_PYTHON and the device with $MATCHING_DEVICE
(auto → cuda). Weights (SuperPoint / LightGlue checkpoints) download lazily to
the HF cache on first use — none are fetched at load. This file stays
Python-3.8-parseable (the override may point at another env).

Load: POST /api/components/nodesets/model_matching/load?mode=server

last updated: 2026-07-07
"""

import asyncio
import base64
import io
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

log = logging.getLogger("agentcanvas.model_matching")

_DETECT_MODEL_DEFAULT = "magic-leap-community/superpoint"
_MATCH_MODEL_DEFAULT = "ETH-CVG/lightglue_superpoint"

# transformers ships SuperPoint as the only standalone keypoint detector;
# LightGlue matchers exist for SuperPoint and DISK features only.
_DETECT_OPTIONS = [
    {"value": "magic-leap-community/superpoint", "label": "SuperPoint (official)"},
    {"value": "stevenbucaille/superpoint", "label": "SuperPoint (mirror)"},
]
_MATCH_OPTIONS = [
    {"value": "ETH-CVG/lightglue_superpoint", "label": "LightGlue (SuperPoint)"},
    {"value": "ETH-CVG/lightglue_disk", "label": "LightGlue (DISK)"},
]


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("MATCHING_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per (kind, model_id)
# ══════════════════════════════════════════════════════════════════════


class _MatchingEngine:
    """Lazy registry: one frozen processor+model per ``(kind, model_id)``.

    ``kind`` picks the auto-class (``detect`` → keypoint detection, ``match`` →
    keypoint matching). Holds only loaded weights — no per-call state. Workers
    coalesce onto the shared engine; the single-flight lock bounds peak VRAM to
    one in-flight forward. These heads run fp32 (keypoint coords are precision-
    sensitive and the nets are tiny — no autocast).
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, kind: str, model_id: str) -> None:
        self.kind = kind
        self.model_id = model_id
        self.device = None
        self.processor = None
        self.model = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, kind: str, model_id: str) -> "_MatchingEngine":
        key = (kind, model_id)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(kind, model_id)
            return cls._instances[key]

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
                from transformers import AutoImageProcessor

                if self.kind == "detect":
                    from transformers import AutoModelForKeypointDetection as _AutoModel
                else:
                    from transformers import AutoModelForKeypointMatching as _AutoModel

                self.device = _resolve_device()
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                model = _AutoModel.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
            except Exception as exc:
                log.warning("matching load failed (%s/%s): %s", self.kind, self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("matching ready (%s/%s, device=%s)", self.kind, self.model_id, self.device)
            return True

    def detect(self, image: Any) -> "dict | None":
        """SuperPoint keypoints/scores/descriptors for one image (original px)."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            H, W = image.height, image.width
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            res = self.processor.post_process_keypoint_detection(outputs, target_sizes=[(H, W)])[0]
            return {
                "keypoints": res["keypoints"].detach().cpu().numpy().astype(np.float32),     # (N,2)
                "scores": res["scores"].detach().cpu().numpy().astype(np.float32),           # (N,)
                "descriptors": res["descriptors"].detach().cpu().numpy().astype(np.float32),  # (N,D)
                "image_hw": [int(H), int(W)],
            }

    def match(self, image_a: Any, image_b: Any, threshold: float) -> "dict | None":
        """Mutually matched keypoints across a pair (original px, above threshold)."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            Ha, Wa = image_a.height, image_a.width
            Hb, Wb = image_b.height, image_b.width
            inputs = self.processor([[image_a, image_b]], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = [[(Ha, Wa), (Hb, Wb)]]
            res = self.processor.post_process_keypoint_matching(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
            return {
                "keypoints0": res["keypoints0"].detach().cpu().numpy().astype(np.float32),       # (M,2)
                "keypoints1": res["keypoints1"].detach().cpu().numpy().astype(np.float32),       # (M,2)
                "matching_scores": res["matching_scores"].detach().cpu().numpy().astype(np.float32),  # (M,)
                "image_hw": [int(Ha), int(Wa)],
                "image_hw_b": [int(Hb), int(Wb)],
            }


# ══════════════════════════════════════════════════════════════════════
# Input / output helpers
# ══════════════════════════════════════════════════════════════════════


def _pil_from_input(item: Any) -> "Any | None":
    """Decode a {rgb_base64}/{image_base64} dict or raw base64 string → RGB PIL image."""
    from PIL import Image

    if isinstance(item, dict):
        b64 = item.get("rgb_base64") or item.get("image_base64")
    elif isinstance(item, str):
        b64 = item
    else:
        b64 = None
    if not b64:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def _first_image(x: Any) -> Any:
    """Accept a single image entry or a 1-element list (graph wires vary)."""
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _arr_field(a: np.ndarray) -> dict:
    buf = np.ascontiguousarray(a, dtype=np.float32)
    return {
        "shape": list(buf.shape),
        "dtype": "float32",
        "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


def _envelope(model_id: str, image_hw: list, arrays: dict, **scalars: Any) -> str:
    import json

    env: dict = {"model_id": model_id, "image_hw": image_hw}
    env.update(scalars)
    for name, arr in arrays.items():
        env[name] = _arr_field(arr)
    return json.dumps(env)


def _cfg(node: BaseCanvasNode) -> dict:
    return getattr(node, "config", None) or {}


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class DetectKeypointsTool(BaseCanvasNode):
    """SuperPoint keypoint detection — one image → keypoints + scores + descriptors."""

    node_type: ClassVar[str] = "model_matching__detect_keypoints"
    display_name: ClassVar[str] = "Matching: Detect Keypoints"
    description: ClassVar[str] = (
        "SuperPoint keypoints for one image; base64-npy envelope "
        "{keypoints[N,2],scores[N],descriptors[N,D]}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField(
                "model_id", "select", label="Detector",
                options=list(_DETECT_OPTIONS), default=_DETECT_MODEL_DEFAULT,
            ),
        ],
    )
    input_ports = [
        PortDef("image", "ANY", "One {rgb_base64} dict or raw base64 string"),
    ]
    output_ports = [
        PortDef(
            "keypoints", "TEXT",
            'JSON {"model_id","image_hw":[H,W],"keypoints":{[N,2]},"scores":{[N]},"descriptors":{[N,D]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        image_in = _first_image(inputs.get("image"))
        if image_in is None:
            return {"keypoints": ""}
        model_id = _cfg(self).get("model_id", _DETECT_MODEL_DEFAULT)
        engine = _MatchingEngine.get("detect", model_id)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            image = _pil_from_input(image_in)
            if image is None:
                return None
            return engine.detect(image)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no keypoints (load failure or bad input)")
            return {"keypoints": ""}
        env = _envelope(
            model_id, res["image_hw"],
            {"keypoints": res["keypoints"], "scores": res["scores"], "descriptors": res["descriptors"]},
        )
        self._self_log("keypoints", int(res["keypoints"].shape[0]))
        return {"keypoints": env}


class MatchTool(BaseCanvasNode):
    """LightGlue keypoint matching — two images → mutually matched keypoint pairs."""

    node_type: ClassVar[str] = "model_matching__match"
    display_name: ClassVar[str] = "Matching: Match Pair"
    description: ClassVar[str] = (
        "Match keypoints across two images; JSON envelope "
        "{keypoints0[M,2],keypoints1[M,2],matching_scores[M]}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "GitCompare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField(
                "model_id", "select", label="Matcher",
                options=list(_MATCH_OPTIONS), default=_MATCH_MODEL_DEFAULT,
            ),
            ConfigField(
                "threshold", "text", label="Match score threshold",
                default="0.0",
            ),
        ],
    )
    input_ports = [
        PortDef("image_a", "ANY", "First image — {rgb_base64} dict or raw base64 string"),
        PortDef("image_b", "ANY", "Second image — {rgb_base64} dict or raw base64 string"),
    ]
    output_ports = [
        PortDef(
            "matches", "TEXT",
            'JSON {"model_id","image_hw","image_hw_b","keypoints0":{[M,2]},'
            '"keypoints1":{[M,2]},"matching_scores":{[M]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        a_in = _first_image(inputs.get("image_a"))
        b_in = _first_image(inputs.get("image_b"))
        if a_in is None or b_in is None:
            return {"matches": ""}
        cfg = _cfg(self)
        model_id = cfg.get("model_id", _MATCH_MODEL_DEFAULT)
        threshold = float(cfg.get("threshold", 0.0) or 0.0)
        engine = _MatchingEngine.get("match", model_id)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            image_a = _pil_from_input(a_in)
            image_b = _pil_from_input(b_in)
            if image_a is None or image_b is None:
                return None
            return engine.match(image_a, image_b, threshold)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no matches (load failure or bad input)")
            return {"matches": ""}
        env = _envelope(
            model_id, res["image_hw"],
            {
                "keypoints0": res["keypoints0"],
                "keypoints1": res["keypoints1"],
                "matching_scores": res["matching_scores"],
            },
            image_hw_b=res["image_hw_b"],
        )
        self._self_log("matches", int(res["keypoints0"].shape[0]))
        return {"matches": env}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class MatchingNodeSet(BaseNodeSet):
    """Sparse keypoint detection + matching primitives — server-mode FM nodeset."""

    name = "model_matching"
    description = (
        "Sparse keypoint detection + matching (detect_keypoints / match) — "
        "SuperPoint + LightGlue via transformers, the SLAM / relocalization "
        "front-end, on the shared ac-fm server"
    )
    # Stateless perception primitives — one shared server across eval workers.
    parallelism = "shared"
    # Rides transformers' keypoint task heads → shares ac-fm (no dedicated env).
    # Override $MATCHING_PYTHON; device via $MATCHING_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "MATCHING_PYTHON")

    def get_tools(self) -> list:
        return [DetectKeypointsTool(), MatchTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_matching ready (server_python=%s); engines load lazily per (kind, model_id)",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
