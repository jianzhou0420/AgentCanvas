from __future__ import annotations

"""RAFT optical flow — server-mode foundation-model nodeset.

Turns a pair of consecutive RGB frames into a dense per-pixel motion field — the
temporal-geometry primitive the FM palette was missing. Ego-motion cues,
moving-object detection, frame-to-frame warping, and short-horizon dynamics all
need per-pixel flow; RAFT is the foundation-model way to get it without a
hand-tuned pyramid.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``variant`` in a lazy registry, load-failure latch + single-flight GPU
lock, everything procedural lives in the graph)::

    model_opticalflow__estimate_flow  (image_a, image_b: {rgb_base64} | b64)
                                      → flow: TEXT envelope (H, W, 2) float32

Flow envelope (the raw C-contiguous float32 buffer base64-encoded, byte-exact
across the HTTP boundary, ~4× smaller than a JSON float list)::

    {"shape":[H,W,2], "dtype":"float32", "b64":…, "variant":…}

``flow[y,x]`` is the displacement in **pixels** carrying pixel (x, y) of frame A
to its match in frame B: channel 0 = dx (+ right), channel 1 = dy (+ down). The
field is returned at the **original input resolution** (RAFT runs on the frame
padded up to a multiple of 8, then the flow is cropped back). Both frames must
share resolution.

``variant`` selects the torchvision checkpoint:
    raft_large  — full RAFT (Raft_Large_Weights.C_T_SKHT_V2, default; accurate)
    raft_small  — RAFT-S (Raft_Small_Weights, ~10× smaller; faster, coarser)

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126) via ``torchvision.models.optical_flow`` — no transformers, no HF
download (weights ship with torchvision). Override the env with
$OPTICAL_FLOW_PYTHON and the device with $OPTICAL_FLOW_DEVICE (auto → cuda when
available). This file must stay Python-3.8-parseable.

Load: POST /api/components/nodesets/model_opticalflow/load?mode=server

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

log = logging.getLogger("agentcanvas.model_opticalflow")

_VARIANT_DEFAULT = "raft_large"
_VARIANTS = ("raft_large", "raft_small")


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("OPTICAL_FLOW_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per variant
# ══════════════════════════════════════════════════════════════════════


class _OpticalFlowEngine:
    """Lazy singleton registry: one frozen RAFT per ``variant``.

    Holds only loaded weights + the checkpoint's own preprocessing transform —
    no cache, no per-call state. The single-flight inference lock bounds peak
    VRAM to one in-flight forward under concurrent eval workers.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.device = None
        self.model = None
        self.transform = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, variant: str) -> "_OpticalFlowEngine":
        with cls._lock:
            if variant not in cls._instances:
                cls._instances[variant] = cls(variant)
            return cls._instances[variant]

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
                from torchvision.models.optical_flow import (
                    Raft_Large_Weights,
                    Raft_Small_Weights,
                    raft_large,
                    raft_small,
                )

                if self.variant == "raft_small":
                    weights = Raft_Small_Weights.DEFAULT
                    net = raft_small(weights=weights)
                else:
                    weights = Raft_Large_Weights.DEFAULT
                    net = raft_large(weights=weights)

                self.device = _resolve_device()
                net = net.to(self.device)
                net.eval()
                for p in net.parameters():
                    p.requires_grad = False
                self.model = net
                self.transform = weights.transforms()
            except Exception as exc:
                log.warning("RAFT load failed (%s): %s", self.variant, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("RAFT ready (%s, device=%s)", self.variant, self.device)
            return True

    def estimate(self, frame_a: np.ndarray, frame_b: np.ndarray) -> "np.ndarray | None":
        """Flow A→B for one same-size HWC uint8 pair → (H, W, 2) float32 pixels.

        RAFT needs spatial dims divisible by 8, so each frame is zero-padded up
        to the next multiple of 8, run, then the flow is cropped back to the
        original (H, W). Returns None on load failure.
        """
        if not self._ensure():
            return None
        import torch

        h, w = frame_a.shape[:2]
        with self._infer_lock:
            # HWC uint8 → CHW uint8 tensor; the checkpoint transform converts to
            # float and normalizes to [-1, 1] (it does NOT resize). ascontiguousarray
            # gives a writable copy (PIL-decoded arrays are read-only buffers).
            ta = torch.from_numpy(np.ascontiguousarray(frame_a)).permute(2, 0, 1).unsqueeze(0)
            tb = torch.from_numpy(np.ascontiguousarray(frame_b)).permute(2, 0, 1).unsqueeze(0)
            ta, tb = self.transform(ta, tb)
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            if ph or pw:
                ta = torch.nn.functional.pad(ta, (0, pw, 0, ph))
                tb = torch.nn.functional.pad(tb, (0, pw, 0, ph))
            ta = ta.to(self.device)
            tb = tb.to(self.device)
            with torch.no_grad():
                # RAFT returns the iterative-refinement list; the last entry is
                # the final flow, shape (1, 2, Hpad, Wpad).
                flow = self.model(ta, tb)[-1]
            flow = flow[:, :, :h, :w]  # crop padding back to original
            flow = flow.squeeze(0).permute(1, 2, 0)  # (H, W, 2)
            return flow.detach().cpu().numpy().astype(np.float32)


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _frame_from_input(item: Any) -> "np.ndarray | None":
    """Accept a {rgb_base64} dict or a raw base64 string → RGB array (None on bad)."""
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


class OpticalFlowEstimateTool(BaseCanvasNode):
    """Dense RAFT optical flow between two consecutive same-resolution frames."""

    node_type: ClassVar[str] = "model_opticalflow__estimate_flow"
    display_name: ClassVar[str] = "Optical Flow: RAFT"
    description: ClassVar[str] = (
        "Dense per-pixel motion field A→B (pixels); base64-npy (H,W,2) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Wind"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="blue",
        config_fields=[
            ConfigField(
                "variant", "select",
                "RAFT checkpoint: raft_large (accurate) | raft_small (fast)",
                default=_VARIANT_DEFAULT, options=list(_VARIANTS),
            ),
        ],
    )
    input_ports = [
        PortDef("image_a", "ANY", "Frame A: {rgb_base64} dict or raw base64 string"),
        PortDef("image_b", "ANY", "Frame B: {rgb_base64} dict or raw base64 string (same size as A)"),
    ]
    output_ports = [
        PortDef(
            "flow", "TEXT",
            'JSON envelope {"shape":[H,W,2],"dtype":"float32","b64":…,"variant":…}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        item_a = inputs.get("image_a")
        item_b = inputs.get("image_b")
        if item_a is None or item_b is None:
            return {"flow": ""}

        cfg = getattr(self, "config", None) or {}
        variant = cfg.get("variant", _VARIANT_DEFAULT)
        if variant not in _VARIANTS:
            variant = _VARIANT_DEFAULT

        engine = _OpticalFlowEngine.get(variant)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            frame_a = _frame_from_input(item_a)
            frame_b = _frame_from_input(item_b)
            if frame_a is None or frame_b is None:
                return ""
            if frame_a.shape[:2] != frame_b.shape[:2]:
                log.warning(
                    "RAFT: frame size mismatch %s vs %s — degrading",
                    frame_a.shape[:2], frame_b.shape[:2],
                )
                return "MISMATCH"
            flow = engine.estimate(frame_a, frame_b)
            if flow is None:
                return ""
            buf = np.ascontiguousarray(flow, dtype=np.float32)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "variant": variant,
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope == "MISMATCH":
            self._self_log("degraded", "frame A / B resolution mismatch")
            return {"flow": ""}
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no flow (load failure or bad input)")
        return {"flow": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class OpticalFlowNodeSet(BaseNodeSet):
    """RAFT optical flow — server-mode FM nodeset."""

    name = "model_opticalflow"
    description = (
        "RAFT dense optical flow (raft_large / raft_small as config) — per-pixel "
        "frame-to-frame motion field on the shared ac-fm server"
    )
    # Stateless flow estimator — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; torchvision RAFT is native there, no HF
    # download). Override with $OPTICAL_FLOW_PYTHON; device via
    # $OPTICAL_FLOW_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "OPTICAL_FLOW_PYTHON")

    def get_tools(self) -> list:
        return [OpticalFlowEstimateTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_opticalflow ready (server_python=%s); engine loads lazily per variant",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
