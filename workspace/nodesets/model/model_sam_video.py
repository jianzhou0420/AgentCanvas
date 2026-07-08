from __future__ import annotations

"""SAM 2 video object tracking — server-mode foundation-model nodeset.

Tracks a single object through a clip: prompt it once on the first frame (a
click or a box) and SAM 2 propagates a mask for that object across every
subsequent frame. This is the temporal counterpart to ``model_sam``'s per-frame
image segmentation — "keep this object segmented as I walk" rather than "segment
this frame". Following an instructed landmark through an egocentric VLN rollout,
persisting an object identity across steps, and mask-based visual servoing all
want this.

Why a separate nodeset (not a ``model_sam`` variant): ``model_sam`` is a fully
stateless image server with an explicit ruling that video/session state does not
belong on it. This nodeset honours that — the SAM 2 inference session is created,
prompted, propagated and torn down **entirely inside one tool call** over a
whole clip, so nothing survives across calls. The server stays stateless; the
session is a call-local temporary, same as any other forward's scratch state.

One pure single-step primitive (FM-nodeset template — stateless server, engines
keyed by ``(variant, ckpt)`` in a lazy registry, load-failure latch +
single-flight GPU lock, everything procedural lives in the graph)::

    model_sam_video__track  (frames: list[b64] | (T,H,W,3) uint8 envelope,
                             points | box  on frame 0)
                            → masks: TEXT envelope (T, H, W) uint8 {0,1}

Frames envelope in — either a JSON list of base64 PNG/JPEG frames, or a
base64-npy ``{"shape":[T,H,W,3],"dtype":"uint8","b64":…}`` buffer. Prompt the
object on frame 0 with EITHER ``points`` (JSON list of [x,y], foreground unless
``labels`` says otherwise) OR ``box`` (JSON [x1,y1,x2,y2]); box wins if both
given. Masks envelope out — a single (T,H,W) uint8 buffer, ``masks[t]`` the
tracked object's binary mask on frame t at the original resolution::

    {"shape":[T,H,W], "dtype":"uint8", "b64":…, "variant":…, "object_id":1}

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) via ``transformers`` ``Sam2VideoModel`` +
``Sam2VideoProcessor`` (facebook/sam2.1-hiera checkpoints, ungated). Override the
env with $SAM_VIDEO_PYTHON and the device with $SAM_VIDEO_DEVICE (auto → cuda
when available). SAM 3 video (text-concept tracking, gated weights) is a future
variant — deferred until a consumer needs it. This file must stay
Python-3.8-parseable.

Load: POST /api/components/nodesets/model_sam_video/load?mode=server

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

log = logging.getLogger("agentcanvas.model_sam_video")

_CKPT_DEFAULT = "facebook/sam2.1-hiera-tiny"
_OBJ_ID = 1  # single-object tracking


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("SAM_VIDEO_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per checkpoint
# ══════════════════════════════════════════════════════════════════════


class _SamVideoEngine:
    """Lazy singleton registry: one frozen SAM 2 video model per ``ckpt``.

    Holds only loaded weights — no cache, no sessions across calls. Each
    ``track`` builds its own inference session and discards it. The single-flight
    lock bounds peak VRAM to one in-flight clip under concurrent eval workers.
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, ckpt: str) -> None:
        self.ckpt = ckpt
        self.device = None
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, ckpt: str) -> "_SamVideoEngine":
        with cls._lock:
            if ckpt not in cls._instances:
                cls._instances[ckpt] = cls(ckpt)
            return cls._instances[ckpt]

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
                from transformers import Sam2VideoModel, Sam2VideoProcessor

                self.device = _resolve_device()
                model = Sam2VideoModel.from_pretrained(self.ckpt)
                model = model.to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
                self.processor = Sam2VideoProcessor.from_pretrained(self.ckpt)
            except Exception as exc:
                log.warning("SAM2-video load failed (%s): %s", self.ckpt, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("SAM2-video ready (%s, device=%s)", self.ckpt, self.device)
            return True

    def track(
        self, frames: list, points: "list | None", labels: "list | None", box: "list | None"
    ) -> "np.ndarray | None":
        """Prompt one object on frame 0 and propagate its mask across the clip.

        Returns (T, H, W) uint8 {0,1}, or None on load failure. The whole
        session lives and dies inside this call — the server keeps no state.
        """
        if not self._ensure():
            return None
        import torch

        h, w = frames[0].shape[:2]
        t = len(frames)
        out = np.zeros((t, h, w), dtype=np.uint8)
        with self._infer_lock:
            session = self.processor.init_video_session(
                video=frames, inference_device=self.device, dtype=torch.float32
            )
            if box is not None:
                self.processor.process_new_points_or_boxes_for_video_frame(
                    session, frame_idx=0, obj_ids=[_OBJ_ID],
                    input_boxes=[[[float(v) for v in box]]],
                )
            else:
                pts = [[float(x), float(y)] for x, y in points]
                labs = [int(v) for v in labels] if labels else [1] * len(pts)
                self.processor.process_new_points_or_boxes_for_video_frame(
                    session, frame_idx=0, obj_ids=[_OBJ_ID],
                    input_points=[[pts]], input_labels=[[labs]],
                )
            with torch.no_grad():
                for seg in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
                    full = self.processor.post_process_masks(
                        [seg.pred_masks], original_sizes=[[h, w]]
                    )[0]
                    arr = np.asarray(full.detach().cpu() if hasattr(full, "detach") else full)
                    while arr.ndim > 2:  # (batch, channels, H, W) → (H, W): first object/mask
                        arr = arr[0]
                    fi = int(seg.frame_idx)
                    if 0 <= fi < t:
                        out[fi] = (arr > 0).astype(np.uint8)
        return out


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _frames_from_input(item: Any) -> "list | None":
    """Accept a base64-npy (T,H,W,3) envelope or a list of base64 frames.

    Returns a list of (H, W, 3) uint8 arrays (uniform size), or None on bad input.
    """
    frames = None
    if isinstance(item, str):
        s = item.lstrip()
        if s.startswith("{"):
            try:
                env = json.loads(item)
            except (ValueError, TypeError):
                env = None
            if isinstance(env, dict) and "b64" in env and "shape" in env:
                buf = np.frombuffer(base64.b64decode(env["b64"]), dtype=np.dtype(env.get("dtype", "uint8")))
                arr = buf.reshape(env["shape"])
                frames = [np.ascontiguousarray(arr[i], dtype=np.uint8) for i in range(arr.shape[0])]
        elif s.startswith("["):
            try:
                lst = json.loads(item)
            except (ValueError, TypeError):
                lst = None
            if isinstance(lst, list):
                frames = [_decode_rgb(b) for b in lst]
    elif isinstance(item, list):
        frames = [_decode_rgb(b) if isinstance(b, str) else _decode_rgb(b.get("rgb_base64") or b.get("image_base64")) for b in item]
    if not frames:
        return None
    if len({f.shape[:2] for f in frames}) != 1:
        return None
    return frames


def _as_json(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return None
    return value


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class SamVideoTrackTool(BaseCanvasNode):
    """Track one object across a clip from a first-frame point or box prompt."""

    node_type: ClassVar[str] = "model_sam_video__track"
    display_name: ClassVar[str] = "SAM Video: Track"
    description: ClassVar[str] = (
        "Propagate one object's mask across a clip from a frame-0 point/box; base64-npy (T,H,W) envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Video"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("ckpt", "text", "HF SAM 2 video checkpoint (hiera tiny/small/base+/large)", default=_CKPT_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("frames", "ANY", "List of base64 frames OR a base64-npy (T,H,W,3) uint8 envelope"),
        PortDef("points", "TEXT", "JSON list of [x,y] prompt points on frame 0", optional=True),
        PortDef("labels", "TEXT", "JSON list of per-point labels (1=fg, 0=bg); default all fg", optional=True),
        PortDef("box", "TEXT", "JSON [x1,y1,x2,y2] prompt box on frame 0 (wins over points)", optional=True),
    ]
    output_ports = [
        PortDef(
            "masks", "TEXT",
            'JSON envelope {"shape":[T,H,W],"dtype":"uint8","b64":…,"variant":…,"object_id":1}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        frames_in = inputs.get("frames")
        if not frames_in:
            return {"masks": ""}

        cfg = getattr(self, "config", None) or {}
        ckpt = str(cfg.get("ckpt", _CKPT_DEFAULT) or _CKPT_DEFAULT)
        points = _as_json(inputs.get("points"))
        labels = _as_json(inputs.get("labels"))
        box = _as_json(inputs.get("box"))

        if box is None and not points:
            self._self_log("degraded", "no prompt — provide points or a box on frame 0")
            return {"masks": ""}

        engine = _SamVideoEngine.get(ckpt)
        loop = asyncio.get_running_loop()

        def _run() -> str:
            frames = _frames_from_input(frames_in)
            if frames is None:
                return ""
            masks = engine.track(frames, points, labels, box)
            if masks is None:
                return ""
            buf = np.ascontiguousarray(masks, dtype=np.uint8)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "uint8",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "variant": ckpt,
                "object_id": _OBJ_ID,
            })

        envelope = await loop.run_in_executor(None, _run)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no masks (load failure or bad input)")
        return {"masks": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class SamVideoNodeSet(BaseNodeSet):
    """SAM 2 video object tracking — server-mode FM nodeset."""

    name = "model_sam_video"
    description = (
        "SAM 2 video object tracking — prompt an object once on frame 0 (point or "
        "box) and propagate its mask across the whole clip on the shared ac-fm server"
    )
    # Stateless server: the inference session is call-local, no cross-call state.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Sam2VideoModel is native there).
    # Override with $SAM_VIDEO_PYTHON; device via $SAM_VIDEO_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-fm", "SAM_VIDEO_PYTHON")

    def get_tools(self) -> list:
        return [SamVideoTrackTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_sam_video ready (server_python=%s); engine loads lazily per ckpt",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
