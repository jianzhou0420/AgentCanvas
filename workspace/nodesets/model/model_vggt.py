from __future__ import annotations

"""VGGT feed-forward 3D reconstruction — server-mode foundation-model nodeset.

The geometry counterpart to ``model_depth_anything``: where Depth Anything
gives *per-view relative depth*, VGGT (Visual Geometry Grounded Transformer,
Meta, CVPR 2025) ingests **1→N RGB views** and, in a **single forward pass**,
regresses camera poses, dense per-view depth, and a dense world-frame point
map — no SLAM, no pairwise global-alignment loop. That turns a bare RGB stream
into metric-ish geometry, the substrate open-vocab 3D mapping (VGGT points +
CLIP features) and RGB-only navigation build on.

Two pure primitives (see model_sam / model_clip for the template — stateless
server, engines keyed by ``model_id`` in a lazy registry, load-failure latch +
single-flight GPU lock, everything procedural lives in the graph)::

    model_vggt__reconstruct   (images: list[{rgb_base64}] | list[b64])
                              → cameras       : envelope {extrinsics(S,3,4), intrinsics(S,3,3)}
                              → depth         : envelope {depth(S,H,W), depth_conf(S,H,W)}
                              → world_points  : envelope {world_points(S,H,W,3), world_points_conf(S,H,W)}
    model_vggt__track_points  (images + query_points: list[[x,y]] in preprocessed-frame px)
                              → tracks        : envelope {track(S,N,2), vis(S,N), conf(S,N)}

``reconstruct`` runs the camera + depth + point heads in one pass (VGGT's
native shape). ``points_source`` picks the world-point path: ``head`` = the
point head's direct output; ``unproject`` = depth + camera back-projection
(``unproject_depth_map_to_point_map``), which the upstream README notes is
often the more accurate point cloud. ``track_points`` is split out because it
is the only surface needing ``query_points`` (drives the track head).

Preprocessing reuses VGGT's own ``load_and_preprocess_images`` **verbatim** —
PIL ``Image.open`` accepts file-like objects, so decoded base64 is fed as
``BytesIO`` with zero reimplementation and zero preprocessing drift. Cameras
come back as a compact 9-D pose encoding and are decoded to extrinsic[S,3,4] +
intrinsic[S,3,3] via ``pose_encoding_to_extri_intri`` (intrinsics assume a
centred principal point — VGGT's own convention).

Multi-array envelope per port (each array = the raw C-contiguous float32 buffer
base64-encoded, byte-exact across HTTP, ~4× smaller than a JSON float list)::

    {"model_id":…, "image_hw":[H,W], "<name>":{"shape":[…],"dtype":"float32","b64":…}, …}

Runs **server mode** in a **dedicated** ``ac-vggt`` env (Python 3.11, torch
2.8.0+cu126) — VGGT is the standalone ``vggt`` package (``VGGT.from_pretrained``
via huggingface_hub, *not* transformers), and it pins ``numpy<2``, so it cannot
share ac-fm's numpy-2 stack without downgrading it under the other FM nodesets.
Override the env with $VGGT_PYTHON and the device with $VGGT_DEVICE
(auto → cuda). Weights ``facebook/VGGT-1B`` are cc-by-nc-4.0 (research);
``facebook/VGGT-1B-Commercial`` is the commercial variant. This file must stay
Python-3.8-parseable (the override may point at a py3.8 env).

Load: POST /api/components/nodesets/model_vggt/load?mode=server

last updated: 2026-07-07
"""

import asyncio
import base64
import io
import json
import logging
import os
import threading
from contextlib import nullcontext
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

log = logging.getLogger("agentcanvas.model_vggt")

_MODEL_ID_DEFAULT = "facebook/VGGT-1B"


def _resolve_device() -> Any:
    import torch

    want = (os.environ.get("VGGT_DEVICE", "") or "auto").strip()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per model_id
# ══════════════════════════════════════════════════════════════════════


class _VggtEngine:
    """Lazy singleton registry: one frozen VGGT per ``model_id``.

    Holds only loaded weights — no cache, no per-call state. Concurrent eval
    workers coalesce onto the one shared engine; the single-flight inference
    lock bounds peak VRAM to a single in-flight forward (VGGT point/depth maps
    scale with view count × resolution).
    """

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = None
        self.model = None
        self._loaded = False
        self._load_failed = False
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str) -> "_VggtEngine":
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
                from vggt.models.vggt import VGGT

                self.device = _resolve_device()
                model = VGGT.from_pretrained(self.model_id).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
            except Exception as exc:
                log.warning("VGGT load failed (%s): %s", self.model_id, exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info("VGGT ready (%s, device=%s)", self.model_id, self.device)
            return True

    def _autocast(self) -> Any:
        """Match the upstream recipe: bf16/fp16 autocast on CUDA, plain on CPU.

        The heads re-disable autocast internally (fp32); only the aggregator
        runs in reduced precision, exactly as the README's inference snippet.
        """
        import torch

        if self.device is not None and self.device.type == "cuda":
            cap = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if cap >= 8 else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def reconstruct(self, image_files: list, mode: str, points_source: str) -> "dict | None":
        """One forward → cameras + depth + world points, all (S, …) input order."""
        if not self._ensure():
            return None
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        with self._infer_lock:
            images = load_and_preprocess_images(image_files, mode=mode).to(self.device)
            with torch.no_grad():
                with self._autocast():
                    pred = self.model(images)  # 4D input → forward adds B=1
            H, W = int(images.shape[-2]), int(images.shape[-1])
            extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
            # squeeze the B=1 batch dim → S-leading arrays
            E = extr[0].detach().cpu().numpy().astype(np.float32)                 # (S,3,4)
            K = intr[0].detach().cpu().numpy().astype(np.float32)                 # (S,3,3)
            D = pred["depth"][0].squeeze(-1).detach().cpu().numpy().astype(np.float32)   # (S,H,W)
            DC = pred["depth_conf"][0].detach().cpu().numpy().astype(np.float32)         # (S,H,W)
            WPC = pred["world_points_conf"][0].detach().cpu().numpy().astype(np.float32)  # (S,H,W)
            if points_source == "unproject":
                from vggt.utils.geometry import unproject_depth_map_to_point_map

                P = unproject_depth_map_to_point_map(D[..., None], E, K).astype(np.float32)  # (S,H,W,3)
            else:
                P = pred["world_points"][0].detach().cpu().numpy().astype(np.float32)        # (S,H,W,3)
            return {
                "extrinsics": E,
                "intrinsics": K,
                "depth": D,
                "depth_conf": DC,
                "world_points": P,
                "world_points_conf": WPC,
                "image_hw": [H, W],
            }

    def track(self, image_files: list, query_points: list, mode: str) -> "dict | None":
        """Forward with query points → 2D tracks + visibility/confidence, (S,N,·)."""
        if not self._ensure():
            return None
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images

        with self._infer_lock:
            images = load_and_preprocess_images(image_files, mode=mode).to(self.device)
            qp = torch.tensor(query_points, dtype=torch.float32, device=self.device)  # (N,2)
            with torch.no_grad():
                with self._autocast():
                    pred = self.model(images, query_points=qp)
            if "track" not in pred:
                return None
            H, W = int(images.shape[-2]), int(images.shape[-1])
            # The track head runs under the outer autocast (unlike camera/depth/point,
            # which re-disable it to fp32), so its outputs can be bf16 — numpy has no
            # bf16, so cast to float32 in torch before .numpy().
            return {
                "track": pred["track"][0].detach().float().cpu().numpy(),  # (S,N,2)
                "vis": pred["vis"][0].detach().float().cpu().numpy(),      # (S,N)
                "conf": pred["conf"][0].detach().float().cpu().numpy(),    # (S,N)
                "image_hw": [H, W],
            }


# ══════════════════════════════════════════════════════════════════════
# Input / output helpers
# ══════════════════════════════════════════════════════════════════════


def _image_files_from_input(items: list) -> "list | None":
    """Accept {rgb_base64} dicts or raw base64 strings → list of BytesIO.

    BytesIO is a valid ``Image.open`` target, so VGGT's own
    ``load_and_preprocess_images`` runs verbatim on in-memory bytes.
    Returns None on any missing/malformed entry (consumer degrades).
    """
    files = []
    for it in items:
        if isinstance(it, dict):
            b64 = it.get("rgb_base64") or it.get("image_base64")
        elif isinstance(it, str):
            b64 = it
        else:
            b64 = None
        if not b64:
            return None
        files.append(io.BytesIO(base64.b64decode(b64)))
    return files


def _query_points_from_input(x: Any) -> "list | None":
    """Normalize query points: list[[x,y], …] or a JSON string of the same."""
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
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append([float(p[0]), float(p[1])])
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


def _envelope(model_id: str, image_hw: list, arrays: dict, **scalars: Any) -> str:
    env: dict = {"model_id": model_id, "image_hw": image_hw}
    env.update(scalars)
    for name, arr in arrays.items():
        env[name] = _arr_field(arr)
    return json.dumps(env)


def _model_id(node: BaseCanvasNode) -> str:
    cfg = getattr(node, "config", None) or {}
    return cfg.get("model_id", _MODEL_ID_DEFAULT)


_MODE_OPTIONS = [
    {"value": "crop", "label": "crop (width=518, center-crop height)"},
    {"value": "pad", "label": "pad (longest=518, pad to square)"},
]
_POINTS_OPTIONS = [
    {"value": "head", "label": "head (point-head output)"},
    {"value": "unproject", "label": "unproject (depth × camera — often sharper)"},
]


# ══════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════


class VggtReconstructTool(BaseCanvasNode):
    """Feed-forward 3D reconstruction of a set of RGB views in one forward pass."""

    node_type: ClassVar[str] = "model_vggt__reconstruct"
    display_name: ClassVar[str] = "VGGT: Reconstruct"
    description: ClassVar[str] = (
        "N RGB views → camera poses + dense depth + world point map (single "
        "forward); base64-npy envelopes on cameras / depth / world_points"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "text", label="HF VGGT model repo id", default=_MODEL_ID_DEFAULT),
            ConfigField("mode", "select", label="Preprocess", default="crop", options=_MODE_OPTIONS),
            ConfigField(
                "points_source", "select", label="World points",
                default="head", options=_POINTS_OPTIONS,
            ),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (1..N views)"),
    ]
    output_ports = [
        PortDef(
            "cameras", "TEXT",
            'JSON {"model_id","image_hw":[H,W],"extrinsics":{shape[S,3,4]},"intrinsics":{shape[S,3,3]}}',
        ),
        PortDef(
            "depth", "TEXT",
            'JSON {"model_id","image_hw","depth":{[S,H,W]},"depth_conf":{[S,H,W]}}',
        ),
        PortDef(
            "world_points", "TEXT",
            'JSON {"model_id","image_hw","points_source","world_points":{[S,H,W,3]},"world_points_conf":{[S,H,W]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        empty = {"cameras": "", "depth": "", "world_points": ""}
        if not items:
            return empty
        model_id = _model_id(self)
        cfg = getattr(self, "config", None) or {}
        mode = cfg.get("mode", "crop")
        points_source = cfg.get("points_source", "head")
        engine = _VggtEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            files = _image_files_from_input(items)
            if files is None:
                return None
            return engine.reconstruct(files, mode, points_source)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no reconstruction (load failure or bad input)")
            return empty
        hw = res["image_hw"]
        cameras = _envelope(model_id, hw, {"extrinsics": res["extrinsics"], "intrinsics": res["intrinsics"]})
        depth = _envelope(model_id, hw, {"depth": res["depth"], "depth_conf": res["depth_conf"]})
        world_points = _envelope(
            model_id, hw,
            {"world_points": res["world_points"], "world_points_conf": res["world_points_conf"]},
            points_source=points_source,
        )
        self._self_log("views", int(res["extrinsics"].shape[0]))
        return {"cameras": cameras, "depth": depth, "world_points": world_points}


class VggtTrackPointsTool(BaseCanvasNode):
    """Track caller-supplied 2D query points across the view set (VGGT track head)."""

    node_type: ClassVar[str] = "model_vggt__track_points"
    display_name: ClassVar[str] = "VGGT: Track Points"
    description: ClassVar[str] = (
        "Track query pixels across N views; JSON envelope {track[S,N,2],vis[S,N],conf[S,N]}"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "text", label="HF VGGT model repo id", default=_MODEL_ID_DEFAULT),
            ConfigField("mode", "select", label="Preprocess", default="crop", options=_MODE_OPTIONS),
        ],
    )
    input_ports = [
        PortDef("images", "ANY", "List of {rgb_base64} dicts or raw base64 strings (1..N views)"),
        PortDef("query_points", "ANY", "Query pixels list[[x,y]] in preprocessed-frame coords (or JSON)"),
    ]
    output_ports = [
        PortDef(
            "tracks", "TEXT",
            'JSON {"model_id","image_hw","track":{[S,N,2]},"vis":{[S,N]},"conf":{[S,N]}}',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        items = inputs.get("images") or []
        query_points = _query_points_from_input(inputs.get("query_points"))
        if not items or not query_points:
            return {"tracks": ""}
        model_id = _model_id(self)
        cfg = getattr(self, "config", None) or {}
        mode = cfg.get("mode", "crop")
        engine = _VggtEngine.get(model_id)
        loop = asyncio.get_running_loop()

        def _run() -> "dict | None":
            files = _image_files_from_input(items)
            if files is None:
                return None
            return engine.track(files, query_points, mode)

        res = await loop.run_in_executor(None, _run)
        if res is None:
            self._self_log("degraded", "no tracks (load failure or bad input)")
            return {"tracks": ""}
        tracks = _envelope(
            model_id, res["image_hw"],
            {"track": res["track"], "vis": res["vis"], "conf": res["conf"]},
        )
        self._self_log("tracked", int(res["track"].shape[1]))
        return {"tracks": tracks}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VggtNodeSet(BaseNodeSet):
    """VGGT feed-forward 3D reconstruction primitives — server-mode FM nodeset."""

    name = "model_vggt"
    description = (
        "VGGT feed-forward 3D reconstruction (reconstruct / track_points) — "
        "N RGB views → camera poses + dense depth + world point map in one pass, "
        "on a dedicated ac-vggt server"
    )
    # Stateless geometry primitives — one shared server across eval workers.
    parallelism = "shared"
    # Dedicated env: VGGT is the standalone `vggt` package (not transformers) and
    # pins numpy<2, so it cannot share ac-fm's numpy-2 stack. Override $VGGT_PYTHON;
    # device via $VGGT_DEVICE (auto → cuda).
    server_python = conda_env_python("ac-vggt", "VGGT_PYTHON")

    def get_tools(self) -> list:
        return [VggtReconstructTool(), VggtTrackPointsTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "model_vggt ready (server_python=%s); engines load lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
