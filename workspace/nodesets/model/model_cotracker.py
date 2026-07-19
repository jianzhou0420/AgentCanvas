from __future__ import annotations

"""CoTracker point tracking — server-mode foundation-model nodeset.

The temporal counterpart to ``model_vggt``: where VGGT tracks query pixels
across a *co-visible view set* in a single feed-forward pass, CoTracker
(Meta, CoTracker3) tracks points densely through a **video / frame sequence**,
handling occlusion and long-range motion. It is the specialist that fills the
gap VGGT's ``track_points`` cannot — temporal correspondence over a walk, the
substrate for motion cues, visual odometry priors, and dynamic-object handling.

Two pure primitives (see model_vggt / model_sam for the template — stateless
server, engines keyed by ``variant`` in a lazy registry, load-failure latch +
single-flight GPU lock, everything procedural lives in the graph)::

    model_cotracker__track_grid    (images: list[{rgb_base64}] | list[b64])
                                   → tracks : envelope {tracks(T,N,2), visibility(T,N)}
    model_cotracker__track_points  (images + query_points: list[[t,x,y]] | list[[x,y]])
                                   → tracks : envelope {tracks(T,N,2), visibility(T,N)}

``track_grid`` seeds a dense ``grid_size × grid_size`` point grid on
``grid_query_frame`` and follows every point through the clip — the "track
everything" mode. ``track_points`` follows caller-supplied query pixels
(each ``(t, x, y)`` = frame index + pixel; a bare ``(x, y)`` defaults to
frame 0). Both run CoTracker3's **offline** predictor (whole clip at once) —
the streaming ``online`` variant is reserved.

Frames are decoded to a ``(1, T, 3, H, W)`` uint8→float tensor in **0–255**
range (CoTracker's own convention — no [0,1] normalization) and fed verbatim.
Outputs squeeze the batch dim: ``tracks`` ``(T, N, 2)`` (x, y per frame) and
``visibility`` ``(T, N)`` (float 0/1, cast from the model's bool mask).

Multi-array envelope per port (each array = the raw C-contiguous float32 buffer
base64-encoded, byte-exact across HTTP, ~4× smaller than a JSON float list)::

    {"variant":…, "image_hw":[H,W], "num_frames":T, "<name>":{"shape":[…],"dtype":"float32","b64":…}, …}

Runs **server mode** in a **dedicated** ``ac-cotracker`` env (Python 3.11, torch
2.8.0+cu126). CoTracker installs from git (``cotracker``, pinned to a commit —
not on PyPI, not transformers); it is kept in its own env to keep the shared
ac-fm transformers stack clean and reproducible. Unlike VGGT this is *not* a
numpy<2-forced split (CoTracker is numpy-2-compatible) — a provenance /
cleanliness choice. Weights ``scaled_offline.pth``
(``facebook/cotracker3`` on HF) download lazily to the torch-hub cache; the repo
is NOT cloned (we load via ``CoTrackerPredictor(checkpoint=…)`` on the installed
package). Override the env with $COTRACKER_PYTHON and the device with
$COTRACKER_DEVICE (auto → cuda). This file stays Python-3.8-parseable.

Load: POST /api/components/nodesets/model_cotracker/load?mode=server

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

log = logging.getLogger("agentcanvas.model_cotracker")

_VARIANT_DEFAULT = "cotracker3_offline"
_CKPT_URL = {
    "cotracker3_offline": "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
}
_WINDOW_LEN = 60  # CoTracker3 offline default


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("COTRACKER_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per variant
# ══════════════════════════════════════════════════════════════════════


class _CotrackerEngine:
    """Lazy singleton registry: one frozen CoTrackerPredictor per ``variant``.

    Holds only the loaded predictor — no cache, no per-call state. Workers
    coalesce onto the shared engine; the single-flight inference lock bounds
    peak VRAM to one in-flight forward (CoTracker's cost scales with clip
    length × point count × resolution).
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.device = None
        self.model = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, variant: str) -> "_CotrackerEngine":
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
                import torch
                from cotracker.predictor import CoTrackerPredictor

                url = _CKPT_URL.get(self.variant)
                if url is None:
                    raise ValueError(f"unknown CoTracker variant: {self.variant}")
                # Download only the checkpoint into the torch-hub cache — never
                # clone the repo (the `cotracker` package is already installed).
                ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt = os.path.join(ckpt_dir, os.path.basename(url))
                if not os.path.exists(ckpt):
                    torch.hub.download_url_to_file(url, ckpt)

                self.device = _resolve_device()
                model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=_WINDOW_LEN)
                model.model.eval()
                for p in model.model.parameters():
                    p.requires_grad = False
                self.model = model.to(self.device)
            except Exception as exc:
                log.warning("CoTracker load failed (%s): %s", self.variant, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("CoTracker ready (%s, device=%s)", self.variant, self.device)
            return True

    def track(self, video: Any, queries: Any, grid_size: int, grid_query_frame: int) -> "dict | None":
        """One offline forward → 2D tracks + visibility, batch dim squeezed."""
        if not self._ensure():
            return None
        import torch

        with self._infer_lock:
            video = video.to(self.device)
            H, W = int(video.shape[-2]), int(video.shape[-1])
            T = int(video.shape[1])
            with torch.no_grad():
                if queries is not None:
                    q = torch.tensor(queries, dtype=torch.float32, device=self.device)[None]  # (1,N,3)
                    tracks, vis = self.model(video, queries=q)
                else:
                    tracks, vis = self.model(
                        video, grid_size=grid_size, grid_query_frame=grid_query_frame
                    )
            # tracks (1,T,N,2) float; vis (1,T,N) bool (or (1,T,N,1))
            tr = tracks[0].detach().cpu().numpy().astype(np.float32)
            vs = vis[0].detach().cpu().numpy()
            if vs.ndim == 3 and vs.shape[-1] == 1:
                vs = vs[..., 0]
            vs = vs.astype(np.float32)
            return {"tracks": tr, "visibility": vs, "image_hw": [H, W], "num_frames": T}


# ══════════════════════════════════════════════════════════════════════
# Input / output helpers
# ══════════════════════════════════════════════════════════════════════


def _video_from_input(items: list) -> "Any | None":
    """Decode {rgb_base64}/b64 frames → (1, T, 3, H, W) float tensor in 0–255.

    All frames must share one resolution (CoTracker requires it). Returns None
    on any missing/malformed entry or a size mismatch (consumer degrades).
    """
    import torch
    from PIL import Image

    arrs = []
    for it in items:
        if isinstance(it, dict):
            b64 = it.get("rgb_base64") or it.get("image_base64")
        elif isinstance(it, str):
            b64 = it
        else:
            b64 = None
        if not b64:
            return None
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None
        arrs.append(np.asarray(img, dtype=np.uint8))  # (H,W,3)
    if not arrs:
        return None
    hw = arrs[0].shape[:2]
    if any(a.shape[:2] != hw for a in arrs):
        return None
    stack = np.stack(arrs)  # (T,H,W,3)
    return torch.tensor(stack).permute(0, 3, 1, 2)[None].float()  # (1,T,3,H,W), 0-255


def _query_points_from_input(x: Any) -> "list | None":
    """Normalize queries → list[[t,x,y]]. Accepts [[t,x,y]] or [[x,y]] (t→0)."""
    if x is None:
        return None
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return None
    if not isinstance(x, list) or not x:
        return None
    pts = []
    for p in x:
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            pts.append([0.0, float(p[0]), float(p[1])])
        else:
            return None
    return pts


def _arr_field(a: np.ndarray) -> dict:
    buf = np.ascontiguousarray(a, dtype=np.float32)
    return {
        "shape": list(buf.shape),
        "dtype": "float32",
        "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


def _envelope(variant: str, image_hw: list, num_frames: int, arrays: dict) -> str:
    env: dict = {"variant": variant, "image_hw": image_hw, "num_frames": num_frames}
    for name, arr in arrays.items():
        env[name] = _arr_field(arr)
    return json.dumps(env)


def _variant(node: BaseCanvasNode) -> str:
    cfg = getattr(node, "config", None) or {}
    return cfg.get("variant", _VARIANT_DEFAULT)


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class TrackGridTool(BaseCanvasNode):
    """Track a dense grid of points through a frame sequence (CoTracker offline)."""

    node_type: ClassVar[str] = "model_cotracker__track_grid"
    display_name: ClassVar[str] = "CoTracker: Track Grid"
    description: ClassVar[str] = (
        "Seed a grid_size×grid_size point grid and track it through the clip; "
        "JSON envelope {tracks[T,N,2], visibility[T,N]}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("variant", "text", label="CoTracker variant", default=_VARIANT_DEFAULT),
            ConfigField("grid_size", "slider", label="Grid size (N = g²)", default=10, min=1, max=50, step=1),
            ConfigField("grid_query_frame", "slider", label="Grid seed frame", default=0, min=0, max=64, step=1),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "Ordered frames — list of {rgb_base64} dicts or raw base64 strings (T frames)"),
    ]
    output_ports = [
        PortDef(
            "tracks", "TEXT",
            'JSON {"variant","image_hw","num_frames","tracks":{[T,N,2]},"visibility":{[T,N]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        if not items:
            return {"tracks": ""}
        variant = _variant(self)
        cfg = getattr(self, "config", None) or {}
        grid_size = int(cfg.get("grid_size", 10) or 10)
        grid_query_frame = int(cfg.get("grid_query_frame", 0) or 0)
        engine = _CotrackerEngine.get(variant)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            video = _video_from_input(items)
            if video is None:
                return None
            return engine.track(video, None, grid_size, grid_query_frame)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no tracks (load failure or bad/mismatched frames)")
            return {"tracks": ""}
        env = _envelope(
            variant, res["image_hw"], res["num_frames"],
            {"tracks": res["tracks"], "visibility": res["visibility"]},
        )
        self._self_log("tracked", f"{res['tracks'].shape[1]}pts×{res['num_frames']}f")
        return {"tracks": env}


class TrackPointsTool(BaseCanvasNode):
    """Track caller-supplied query pixels through a frame sequence (CoTracker offline)."""

    node_type: ClassVar[str] = "model_cotracker__track_points"
    display_name: ClassVar[str] = "CoTracker: Track Points"
    description: ClassVar[str] = (
        "Track query pixels (t,x,y) through the clip; JSON envelope "
        "{tracks[T,N,2], visibility[T,N]}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("variant", "text", label="CoTracker variant", default=_VARIANT_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "Ordered frames — list of {rgb_base64} dicts or raw base64 strings (T frames)"),
        PortDef("query_points", "ANY", "Queries list[[t,x,y]] (frame index + pixel; [x,y] defaults t=0), or JSON"),
    ]
    output_ports = [
        PortDef(
            "tracks", "TEXT",
            'JSON {"variant","image_hw","num_frames","tracks":{[T,N,2]},"visibility":{[T,N]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        query_points = _query_points_from_input(inputs.get("query_points"))
        if not items or not query_points:
            return {"tracks": ""}
        variant = _variant(self)
        engine = _CotrackerEngine.get(variant)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            video = _video_from_input(items)
            if video is None:
                return None
            return engine.track(video, query_points, 0, 0)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no tracks (load failure or bad input)")
            return {"tracks": ""}
        env = _envelope(
            variant, res["image_hw"], res["num_frames"],
            {"tracks": res["tracks"], "visibility": res["visibility"]},
        )
        self._self_log("tracked", f"{res['tracks'].shape[1]}pts×{res['num_frames']}f")
        return {"tracks": env}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class CotrackerNodeSet(BaseNodeSet):
    """CoTracker point-tracking primitives — server-mode FM nodeset."""

    name = "model_cotracker"
    description = (
        "CoTracker point tracking (track_grid / track_points) — dense point "
        "tracking through a video / frame sequence with occlusion handling, "
        "on a dedicated ac-cotracker server"
    )
    # Stateless tracking primitives — one shared server across eval workers.
    parallelism = "shared"
    # Dedicated env: CoTracker installs from git (pinned commit, not on PyPI); kept
    # out of the shared ac-fm transformers env for provenance/cleanliness — not a
    # numpy-forced split (CoTracker is numpy-2-compatible). Override
    # $COTRACKER_PYTHON; device via $COTRACKER_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-cotracker", "COTRACKER_PYTHON")

    def get_tools(self) -> list:
        return [TrackGridTool(), TrackPointsTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_cotracker ready (server_python=%s); engines load lazily per variant",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
