from __future__ import annotations

"""ModelPySlamNodeSet — pySLAM as a server-mode, session-type ``model/`` nodeset.

Wraps pySLAM (Freda, v2.10.6, GPL-3.0) on the canvas. Two node tiers from the
design note (``docs/pages/developer-guide/tmp/2026-07-06-pyslam-nodeset-design.html``):
a **streaming SLAM session** (Tier-1) whose whole algorithm matrix is reached by
*config passthrough* (not one-node-per-algorithm), plus **stateless perception**
nodes (Tier-2) that use pyslam's standalone classes as pure functions.

    Tier-1 — streaming session (mutable map state):
    model_pyslam__reset            trigger → clear the map for a new episode
    model_pyslam__track            rgb + depth → pose + tracking_state (the workhorse)
    model_pyslam__get_trajectory   trigger → accumulated poses
    model_pyslam__get_map          trigger → sparse-map handle (path, not inline)
    model_pyslam__get_dense_map    trigger → dense volumetric map handle (PYSLAM_VOLUMETRIC=1)

    Tier-2 — stateless perception (no session; share the container):
    model_pyslam__extract_features rgb → keypoints + descriptors
    model_pyslam__match_features   descriptors_a + descriptors_b → matched idx pairs
    model_pyslam__eval_trajectory  poses_est + poses_gt → ATE / RPE (evo)

    Full-surface neural nodes (stateless; GPU — the :cuda image ships a cu128 torch):
    model_pyslam__predict_depth        rgb [+ image_right] → dense metric depth map
    model_pyslam__segment_semantic     rgb → per-pixel semantic (+ instance) map
    model_pyslam__reconstruct_multiview [rgb, …] → fused 3-D scene (DUSt3R/MASt3R/VGGT)

Goal (2026-07-08): expose pySLAM's *whole* capability surface as first-class
nodes — this is AgentCanvas supporting pySLAM-the-model, not a curated subset for
one downstream graph. The eleven nodes above ARE that whole callable surface:
session verbs + stateless perception + neural depth / semantics / multi-view.
A ``get_loop_closures`` node was considered and **deliberately dropped**
(2026-07-08): loop closure is not a callable capability but an internal mechanism
of the running SLAM session, and its *effect* (drift correction) is already
exposed through the corrected ``get_trajectory`` / ``get_map`` output. Reading the
loop events themselves means instrumenting pySLAM's loop-closing thread (transient
state, no clean getter), so it is left unbuilt until a concrete consumer needs it
(e.g. SLAM-accuracy attribution). Multi-view weights mount in from
data/models/pyslam/ (external-weights decision 2026-07-08), keeping the image lean
(code + compiled curope only).

Classification (design §1): pySLAM has no MDP / episode selection / actions —
it is "feed frames in, map accumulates, read products out", so it lives in
``model/`` with ``model_sam`` as the template (SAM 2's cross-call ``_tracks``
state is isomorphic to pySLAM's accumulating map). It is **not** an ``env/``.

Deployment — **container bridge** (design §1, §5; superseding the conda-env plan):
    - This nodeset runs in **local mode** (``server_python = None``). It does *not*
      import pyslam itself; instead :meth:`initialize` ``docker run``s the
      ``agentcanvas/pyslam`` image and holds a :class:`_client.PySlamContainerClient`
      as its session. The client speaks HTTP to a FastAPI shim (``_server.py``)
      running **inside** the container, which drives the real
      :class:`_backend.PySlamSession`. So ``import pyslam`` happens only in the
      container — the GPL-3.0 source is never vendored into this repo and never
      loaded into the framework env (same treatment as ``habitat_sim``). Rootless
      Docker keeps the whole thing sudo-free (see the ``reference_rootless_docker``
      memory).
    - The nodeset dir is bind-mounted read-only into the container, so
      ``_server``/``_backend`` are available inside without baking our code into
      the image (only fastapi/uvicorn + pyslam live in the image).
    - ``parallelism = "replicated"``: the map is mutable per-episode state; each
      batch-eval worker gets its own container (one container per client). For a
      single graph run that is exactly one container.

Configure via environment variables (defaults = pure-CPU, no weights):
    PYSLAM_SENSOR   = mono | stereo | rgbd     (default: rgbd)
    PYSLAM_FEATURE  = ORB2 | SUPERPOINT | ...  (default: ORB2  — feature_tracker_configs)
    PYSLAM_LOOP     = DBOW3 | off | ...        (default: DBOW3 — loop_detector_configs)
    PYSLAM_CAM_W / PYSLAM_CAM_H / PYSLAM_CAM_HFOV  — default camera intrinsics
    PYSLAM_IMAGE    — pin the container image; unset → GPU picks :cuda (full surface,
                      incl. reconstruct_multiview), no-GPU picks :cpu-fixed
    PYSLAM_GPU      = 0 | 1                     GPU on/off (default 1; CPU fallback if absent)
    PYSLAM_WEIGHTS_DIR — host folder of external multiview weights (default data/models/pyslam)
    PYSLAM_ARTIFACT_DIR — host dir where get_map writes handles (default outputs/pyslam_maps)
    PYSLAM_VOLUMETRIC = 0 | 1                  enable Slam's dense volumetric integrator
    PYSLAM_VOLUMETRIC_TYPE = VOXEL_GRID | TSDF | VOXEL_SEMANTIC_GRID   (get_dense_map output)
    PYSLAM_ENV      = INDOOR | OUTDOOR         dense depth-truncation regime (default INDOOR)

Status: the backend pipeline (``_backend.py``) is validated end-to-end on a TUM
RGB-D sequence through the real container (both former seams — camera marshalling,
sparse-map export — confirmed; see ``_backend.py`` header). The container bridge
(``_client.py`` + ``_server.py``) wires that backend into the graph executor.

last updated: 2026-07-07 (container bridge — local-mode nodeset ↔ dockerised pyslam)
"""

import asyncio
import json
import logging
import os
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

log = logging.getLogger("agentcanvas.pyslam")

# The graph executor fires FRESH ``node_cls()`` instances (graph_executor.py
# ~L1123) — it never uses the ``get_tools(self)`` instances that carry a
# ``self._nodeset`` back-reference. So a local-mode nodeset can't reach its
# session through the node's back-ref; nodes read this process-level singleton
# instead, which ``initialize()`` sets. Local mode is always single-process
# (worker_count>1 is auto-routed to server mode), so one module global is the
# correct per-worker isolation — each replicated worker is its own subprocess.
_NODESET: ModelPySlamNodeSet | None = None

# Per-node call counters for get_map's ``stream_every`` throttle (live in-loop
# map streaming). Keyed by graph node id; the executor fires a fresh node
# instance each step, so the counter can't live on the instance. Only the
# throttled (in-loop) get_map node touches this — the default stream_every=0
# (after-loop / one-shot) never counts.
_GETMAP_COUNTERS: dict[str, int] = {}


# ── payload decoding (mirrors model_detany3d: accept ndarray or path) ──────


def _decode_image_input(value: Any) -> np.ndarray:
    """Accept an already-decoded ``np.ndarray`` or a path to a .npy/image file.

    Server-mode wires carry numpy directly (msgpack); a path string is the
    fallback for large frames staged on disk.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, str):
        if value.endswith(".npy"):
            return np.load(value)
        from PIL import Image  # local — Pillow lives in the ac-pyslam env

        return np.asarray(Image.open(value).convert("RGB"), dtype=np.uint8)
    raise TypeError(f"cannot decode image input of type {type(value).__name__}")


def _decode_depth_input(value: Any) -> np.ndarray | None:
    """Depth is optional (mono). ndarray/list → float32 array; path → .npy load."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str) and value.endswith(".npy"):
        return np.load(value).astype(np.float32, copy=False)
    raise TypeError(f"cannot decode depth input of type {type(value).__name__}")


def _pose_to_matrix(pose: Any) -> Any:
    """Normalise an env pose to a 4x4 SE3 (list of lists), for trajectory eval.

    ``env_habitat``'s observe ``pose`` port emits
    ``{"position": [x,y,z], "orientation": [qx,qy,qz,qw]}`` (quaternion,
    scalar-last). pySLAM's estimated pose translation is the **camera centre in
    world coords** (``tracking.cur_t = -Rwc·tcw``), so a matching world pose here
    needs the rotation from the quaternion and the translation from the position —
    ATE(translation) only compares the centres, alignment absorbs the frame
    offset. Passes an already-4x4 pose through unchanged.
    """
    if isinstance(pose, np.ndarray):
        return pose.tolist() if pose.shape == (4, 4) else pose
    if isinstance(pose, list):
        return pose  # assume already 4x4 list-of-lists
    if isinstance(pose, dict):
        pos = pose.get("position") or [0.0, 0.0, 0.0]
        quat = pose.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        qx, qy, qz, qw = (float(v) for v in quat)
        n = qx * qx + qy * qy + qz * qz + qw * qw
        s = 0.0 if n == 0.0 else 2.0 / n
        xs, ys, zs = qx * s, qy * s, qz * s
        wx, wy, wz = qw * xs, qw * ys, qw * zs
        xx, xy, xz = qx * xs, qx * ys, qx * zs
        yy, yz, zz = qy * ys, qy * zs, qz * zs
        return [
            [1.0 - (yy + zz), xy - wz, xz + wy, float(pos[0])],
            [xy + wz, 1.0 - (xx + zz), yz - wx, float(pos[1])],
            [xz - wy, yz + wx, 1.0 - (xx + yy), float(pos[2])],
            [0.0, 0.0, 0.0, 1.0],
        ]
    raise TypeError(f"cannot convert pose of type {type(pose).__name__} to 4x4")


def _resize_depth_to(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resample a HxW metric depth map onto an (H,W) grid.

    Env RGB and depth sensors are co-located with the same FOV + centred
    principal point, so a graph that renders them at different resolutions
    (e.g. CMA's RGB 224 vs depth 256) still has pixel-for-pixel-alignable maps.
    pySLAM's RGB-D front-end requires depth.shape == rgb.shape, so we resample
    depth onto the RGB grid. Nearest-neighbour (not bilinear) keeps metric depth
    values intact — no interpolated ghosts across depth discontinuities.
    """
    h0, w0 = depth.shape[:2]
    if (h0, w0) == (height, width):
        return depth
    ys = np.minimum((np.arange(height) * (h0 / height)).astype(np.int64), h0 - 1)
    xs = np.minimum((np.arange(width) * (w0 / width)).astype(np.int64), w0 - 1)
    return depth[ys][:, xs]


# ── Tier-1 tool nodes ──────────────────────────────────────────────────────
#
# Each node holds a back-reference to the nodeset (not the session directly)
# because get_tools() runs BEFORE initialize() in the registry lifecycle; the
# session is resolved lazily at execute-time (same as model_sam's backend).


class PySlamResetTool(BaseCanvasNode):
    """Start / restart a SLAM session — clear the map for a new episode.

    First call builds the ``Slam`` instance from the configured intrinsics and
    presets; subsequent calls only clear the map (``reset_session``), never
    rebuild — building is seconds of thread spin-up we don't repeat per episode.
    """

    node_type = "model_pyslam__reset"
    display_name = "pySLAM: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "sensor_type", "select", label="Sensor", default="rgbd",
                options=[
                    {"value": "rgbd", "label": "RGB-D"},
                    {"value": "mono", "label": "Monocular"},
                    {"value": "stereo", "label": "Stereo"},
                ],
            ),
            # Phase 3 turns feature/loop into curated selects over the pyslam
            # factory names + a raw escape hatch; kept as text for the skeleton.
            ConfigField("feature_preset", "text", label="Feature preset", default="ORB2"),
            ConfigField("loop_preset", "text", label="Loop preset", default="DBOW3"),
            ConfigField("cam_hfov", "text", label="Camera HFOV (deg)", default="90"),
            ConfigField("depth_scale", "text",
                        label="Depth scale (metres per depth unit)", default="1.0"),
        ],
    )
    description = (
        "Begin or reset a pySLAM session. Clears the accumulated map so a new "
        "episode starts clean. Configure sensor type, feature/loop presets, and "
        "camera field-of-view here; the Slam instance is built lazily on first use."
    )
    category = "tool"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Fire to (re)start the session", optional=True),
        PortDef("intrinsics", "ANY",
                "Camera intrinsics {fx,fy,cx,cy,width,height} (e.g. env observe port) "
                "— preferred over hfov for an exact pinhole", optional=True),
        PortDef("cam_width", "ANY", "Camera width in px (overrides PYSLAM_CAM_W)", optional=True),
        PortDef("cam_height", "ANY", "Camera height in px (overrides PYSLAM_CAM_H)", optional=True),
        PortDef("cam_hfov", "ANY", "Horizontal FOV in degrees (overrides config)", optional=True),
    ]
    output_ports = [
        PortDef("episode_ok", "BOOL", "True once the session is ready"),
        PortDef("info", "TEXT", "Session state summary or error message"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"episode_ok": False, "info": '{"error": "pyslam nodeset not initialized"}'}
        try:
            ns._ensure_built(
                intrinsics=inputs.get("intrinsics"),
                width=inputs.get("cam_width"),
                height=inputs.get("cam_height"),
                hfov=inputs.get("cam_hfov") or ns._config.get("cam_hfov"),
            )
            # Stash metres-per-depth-unit for the track node (Habitat renders
            # depth normalised to [0,1] over max_depth → scale = max_depth).
            ns._config["depth_scale"] = float(self.config.get("depth_scale") or 1.0)
            ns._session.reset()
            ns._gt_traj.clear()  # new episode → drop the previous GT trajectory
            info = {
                "sensor": ns._session.sensor_type,
                "feature": ns._session.feature_preset,
                "loop": ns._session.loop_preset,
                "built": ns._session.is_built,
            }
            return {"episode_ok": True, "info": json.dumps(info, ensure_ascii=False)}
        except Exception as exc:
            log.exception("pyslam reset failed")
            return {"episode_ok": False, "info": json.dumps({"error": str(exc)})}


class PySlamTrackTool(BaseCanvasNode):
    """Feed one frame — the SLAM session's step. Returns the estimated pose.

    This is ``env.step`` for the map: rgb (+ depth in RGB-D mode) in, camera
    pose + tracking state out, map accumulates as a side effect.
    """

    node_type = "model_pyslam__track"
    display_name = "pySLAM: Track"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")
    description = (
        "Track one frame through the SLAM front-end and update the map. Provide "
        "an RGB image (and a depth map in RGB-D mode). Returns the estimated "
        "camera pose (4x4 world-from-camera), the tracking state (OK / LOST / "
        "INIT), and the current map-point count."
    )
    category = "tool"
    icon = "Camera"
    input_ports = [
        PortDef("rgb", "IMAGE", "RGB frame (np.ndarray HxWx3 uint8, or .npy path)"),
        PortDef("depth", "DEPTH", "Depth frame in metres (np.ndarray HxW, RGB-D only)", optional=True),
        PortDef("timestamp", "ANY", "Frame timestamp in seconds (default = frame index)", optional=True),
        PortDef("gt_pose", "ANY",
                "Ground-truth agent pose this step (env observe.pose) — captured "
                "frame-aligned for eval_trajectory when tracking succeeds", optional=True),
    ]
    output_ports = [
        PortDef("pose", "POSE", "4x4 world-from-camera pose (list of lists), or null if LOST"),
        PortDef("tracking_state", "TEXT", "SLAM tracking state (OK / LOST / INIT / ...)"),
        PortDef("num_map_points", "ANY", "Number of 3-D points in the map"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None or not ns._session.is_built:
            return {
                "pose": None,
                "tracking_state": "ERROR",
                "num_map_points": 0,
            }
        try:
            rgb = _decode_image_input(inputs["rgb"])
            depth = _decode_depth_input(inputs.get("depth"))
            if depth is not None and depth.shape[:2] != rgb.shape[:2]:
                depth = _resize_depth_to(depth, rgb.shape[1], rgb.shape[0])
            if depth is not None:
                depth = depth * float(ns._config.get("depth_scale") or 1.0)
            result = ns._session.track(rgb, depth, inputs.get("timestamp"))
            # Capture GT frame-aligned to the estimated trajectory: append iff
            # pySLAM produced a pose this step (same condition the session uses to
            # grow its estimated trajectory), keeping the two lists index-aligned.
            # A None slot (GT unwired that step) is dropped later by the eval node.
            if result["pose"] is not None:
                gt = inputs.get("gt_pose")
                ns._gt_traj.append(_pose_to_matrix(gt) if gt is not None else None)
            log.info("pyslam track: state=%s map_pts=%s pose=%s rgb=%s depth=%s range=[%.2f,%.2f]",
                     result["state"], result["num_map_points"],
                     "Y" if result["pose"] is not None else "-",
                     rgb.shape, None if depth is None else depth.shape,
                     float(depth.min()) if depth is not None else -1.0,
                     float(depth.max()) if depth is not None else -1.0)
            return {
                "pose": result["pose"],
                "tracking_state": result["state"],
                "num_map_points": result["num_map_points"],
            }
        except Exception as exc:
            log.exception("pyslam track failed")
            return {"pose": None, "tracking_state": f"ERROR: {exc}", "num_map_points": 0}


class PySlamGetTrajectoryTool(BaseCanvasNode):
    """Read the accumulated camera trajectory (one pose per tracked frame)."""

    node_type = "model_pyslam__get_trajectory"
    display_name = "pySLAM: Get Trajectory"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")
    description = (
        "Return the accumulated camera trajectory as a list of 4x4 poses. Use "
        "after a run to evaluate against ground truth (ATE / RPE) or to drive "
        "downstream map consumers."
    )
    category = "tool"
    icon = "Route"
    input_ports = [PortDef("trigger", "ANY", "Fire to read the trajectory", optional=True)]
    output_ports = [
        PortDef("poses", "ANY", "List of 4x4 world-from-camera poses (estimated)"),
        PortDef("num_frames", "ANY", "Total frames fed to track()"),
        PortDef("gt_poses", "ANY",
                "Frame-aligned ground-truth poses captured by track (feed to "
                "eval_trajectory alongside poses); empty if no gt_pose was wired"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"poses": [], "num_frames": 0, "gt_poses": []}
        traj = ns._session.get_trajectory()
        return {"poses": traj["poses"], "num_frames": traj["num_frames"],
                "gt_poses": list(ns._gt_traj)}


class PySlamGetMapTool(BaseCanvasNode):
    """Export the sparse map (keyframes + landmarks) to a handle.

    Heavy geometry never rides an inline wire (design §5a): the map is dumped
    to disk in the SLAM process and only the path travels the wire.
    """

    node_type = "model_pyslam__get_map"
    display_name = "pySLAM: Get Map"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "stream_every", "text",
                label="Stream every N frames (0 = one-shot / after-loop; N>0 = live in-loop)",
                default="0",
            ),
        ],
    )
    description = (
        "Export the current sparse map (3-D landmarks + keyframes) to a file and "
        "return its path (a handle). Consumers load the handle on demand rather "
        "than receiving megabytes of points inline on every wire. Set "
        "stream_every=N>0 to fire this node inside the loop and export only every "
        "Nth frame — a live-growing map for pointCloudViewer without dumping the "
        "full map every step (skipped frames return an empty handle, cheaply)."
    )
    category = "tool"
    icon = "Boxes"
    input_ports = [PortDef("trigger", "ANY", "Fire to export the map", optional=True)]
    output_ports = [
        PortDef("map_handle", "TEXT", "Path to the exported map file (.npz)"),
        PortDef("num_points", "ANY", "Number of 3-D points exported"),
        PortDef("num_keyframes", "ANY", "Number of keyframes in the map"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None or not ns._session.is_built:
            return {"map_handle": "", "num_points": 0, "num_keyframes": 0}
        # stream_every throttle: when >0 (live in-loop streaming) only export the
        # map every Nth firing; skipped firings return an empty handle with no
        # container round-trip / disk dump, and the pointCloudViewer sink no-ops
        # on an empty cloud (keeps the last render). 0 = export every call.
        try:
            every = int(self.config.get("stream_every") or 0)
        except (TypeError, ValueError):
            every = 0
        if every > 0:
            key = getattr(self, "node_id", "") or "_"
            n = _GETMAP_COUNTERS.get(key, 0) + 1
            _GETMAP_COUNTERS[key] = n
            if n % every != 0:
                return {"map_handle": "", "num_points": 0, "num_keyframes": 0}
        try:
            result = ns._session.get_map()
            return {
                "map_handle": result["map_handle"],
                "num_points": result["num_points"],
                "num_keyframes": result["num_keyframes"],
            }
        except Exception as exc:
            log.exception("pyslam get_map failed")
            return {"map_handle": f"ERROR: {exc}", "num_points": 0, "num_keyframes": 0}


class PySlamGetDenseMapTool(BaseCanvasNode):
    """Export the dense volumetric map (TSDF mesh / fused voxel cloud) to a handle.

    Unlike Get Map (sparse landmarks), this is Slam's own volumetric integrator
    output — a fused surface. It requires the session to be built with volumetric
    integration ON (set ``PYSLAM_VOLUMETRIC=1`` in the nodeset env); otherwise it
    returns an empty handle. Heavy geometry never rides the wire — only the path.
    """

    node_type = "model_pyslam__get_dense_map"
    display_name = "pySLAM: Get Dense Map"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")
    description = (
        "Export the dense volumetric map — a fused voxel point cloud or TSDF mesh "
        "built by pySLAM's volumetric integrator over the tracked frames — to a "
        "file and return its path. Requires volumetric integration enabled "
        "(PYSLAM_VOLUMETRIC=1); returns an empty handle otherwise."
    )
    category = "tool"
    icon = "Box"
    input_ports = [PortDef("trigger", "ANY", "Fire to export the dense map", optional=True)]
    output_ports = [
        PortDef("dense_handle", "TEXT", "Path to the exported dense map (.npz), or empty"),
        PortDef("num_points", "ANY", "Number of fused voxel points"),
        PortDef("num_vertices", "ANY", "Number of mesh vertices (TSDF)"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None or not ns._session.is_built:
            return {"dense_handle": "", "num_points": 0, "num_vertices": 0}
        try:
            result = ns._session.get_dense_map()
            log.info("pyslam get_dense_map: type=%s points=%s verts=%s tris=%s",
                     result.get("type"), result["num_points"],
                     result["num_vertices"], result["num_triangles"])
            return {
                "dense_handle": result["dense_handle"],
                "num_points": result["num_points"],
                "num_vertices": result["num_vertices"],
            }
        except Exception as exc:
            log.exception("pyslam get_dense_map failed")
            return {"dense_handle": f"ERROR: {exc}", "num_points": 0, "num_vertices": 0}


# ── Tier-2 tool nodes (stateless perception) ────────────────────────────────
#
# These do not touch the SLAM session's map — they are pure functions over
# pyslam's standalone feature front-end and evo trajectory eval (design note
# nodes 7-12). They need the container up (``ns._session``) but not a built
# ``Slam`` — so, unlike track/get_map, they do not gate on ``is_built``.


def _decode_array_input(value: Any) -> np.ndarray:
    """Accept an ndarray / list / .npy path → ndarray (dtype preserved for ndarray).

    In local mode the executor passes objects in-process, so a descriptor block
    keeps its uint8/float32 dtype across the wire — which the matcher relies on
    to pick Hamming vs L2. Only a list literal or a staged .npy needs decoding.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str) and value.endswith(".npy"):
        return np.load(value)
    return np.asarray(value)


class PySlamExtractFeaturesTool(BaseCanvasNode):
    """Detect keypoints + descriptors on one image (stateless — no SLAM session).

    pySLAM's local-feature front-end used as a standalone tool: RGB in, keypoints
    (Nx6) and descriptors (NxD) out. Classical detectors (ORB / SIFT / AKAZE)
    need no weights; learned ones (SuperPoint / XFeat) auto-download on first use.
    """

    node_type = "model_pyslam__extract_features"
    display_name = "pySLAM: Extract Features"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "detector", "select", label="Detector", default="ORB",
                options=[
                    {"value": "ORB", "label": "ORB (OpenCV, no weights)"},
                    {"value": "SIFT", "label": "SIFT"},
                    {"value": "AKAZE", "label": "AKAZE"},
                    {"value": "BRISK", "label": "BRISK"},
                    {"value": "SUPERPOINT", "label": "SuperPoint (learned)"},
                    {"value": "XFEAT", "label": "XFeat (learned)"},
                ],
            ),
            ConfigField("descriptor", "text", label="Descriptor (blank = same as detector)",
                        default=""),
            ConfigField("num_features", "text", label="Max features", default="2000"),
        ],
    )
    description = (
        "Detect keypoints and compute descriptors on a single image using pySLAM's "
        "feature front-end. Returns keypoints (x, y, size, angle, response, octave) "
        "and the descriptor block — feed both to Match Features."
    )
    category = "tool"
    icon = "ScanSearch"
    input_ports = [PortDef("rgb", "IMAGE", "RGB frame (np.ndarray HxWx3 uint8, or .npy path)")]
    output_ports = [
        PortDef("keypoints", "ANY", "Nx6 array [x, y, size, angle, response, octave]"),
        PortDef("descriptors", "ANY", "NxD descriptor block (uint8 for ORB, float for SIFT)"),
        PortDef("num_keypoints", "ANY", "Number of keypoints detected"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"keypoints": None, "descriptors": None, "num_keypoints": 0}
        try:
            rgb = _decode_image_input(inputs["rgb"])
            out = ns._session.extract_features(
                rgb,
                detector=self.config.get("detector") or "ORB",
                descriptor=self.config.get("descriptor") or self.config.get("detector") or "ORB",
                num_features=int(self.config.get("num_features") or 2000),
            )
            return {
                "keypoints": out["keypoints"],
                "descriptors": out["descriptors"],
                "num_keypoints": out["num_keypoints"],
            }
        except Exception as exc:
            log.exception("pyslam extract_features failed")
            return {"keypoints": None, "descriptors": None, "num_keypoints": f"ERROR: {exc}"}


class PySlamMatchFeaturesTool(BaseCanvasNode):
    """Match two descriptor blocks (stateless), returning matched index pairs.

    pySLAM's matcher used as a standalone tool: descriptors A + B in, the indices
    of mutually-matched features out. Classical matchers (BF / FLANN) run on the
    arrays alone — no images, no weights.
    """

    node_type = "model_pyslam__match_features"
    display_name = "pySLAM: Match Features"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "matcher_type", "select", label="Matcher", default="BF",
                options=[
                    {"value": "BF", "label": "Brute-force"},
                    {"value": "FLANN", "label": "FLANN"},
                ],
            ),
            ConfigField("ratio_test", "text", label="Lowe ratio test", default="0.7"),
        ],
    )
    description = (
        "Match two sets of descriptors (from Extract Features) and return the "
        "matched index pairs. Descriptor dtype picks the distance automatically "
        "(Hamming for binary ORB/AKAZE, L2 for SIFT)."
    )
    category = "tool"
    icon = "Spline"
    input_ports = [
        PortDef("descriptors_a", "ANY", "First image's descriptor block (NxD)"),
        PortDef("descriptors_b", "ANY", "Second image's descriptor block (MxD)"),
    ]
    output_ports = [
        PortDef("idxs_a", "ANY", "Matched indices into descriptors_a"),
        PortDef("idxs_b", "ANY", "Matched indices into descriptors_b (aligned with idxs_a)"),
        PortDef("num_matches", "ANY", "Number of matches"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"idxs_a": None, "idxs_b": None, "num_matches": 0}
        try:
            des_a = _decode_array_input(inputs["descriptors_a"])
            des_b = _decode_array_input(inputs["descriptors_b"])
            out = ns._session.match_features(
                des_a, des_b,
                matcher_type=self.config.get("matcher_type") or "BF",
                ratio_test=float(self.config.get("ratio_test") or 0.7),
            )
            return {
                "idxs_a": out["idxs_a"],
                "idxs_b": out["idxs_b"],
                "num_matches": out["num_matches"],
            }
        except Exception as exc:
            log.exception("pyslam match_features failed")
            return {"idxs_a": None, "idxs_b": None, "num_matches": f"ERROR: {exc}"}


class PySlamEvalTrajectoryTool(BaseCanvasNode):
    """Evaluate an estimated trajectory against ground truth — ATE + RPE (stateless).

    pySLAM's evo-based trajectory eval as a standalone tool: Umeyama-aligns the
    estimate to GT (scale-corrected iff monocular) and reports ATE/RPE stats.
    Pairs naturally with Get Trajectory. Pure math — no weights, no GPU.
    """

    node_type = "model_pyslam__eval_trajectory"
    display_name = "pySLAM: Eval Trajectory"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "is_monocular", "select", label="Monocular (scale-align)", default="false",
                options=[
                    {"value": "false", "label": "No (metric, SE3 align)"},
                    {"value": "true", "label": "Yes (Sim3 scale align)"},
                ],
            ),
        ],
    )
    description = (
        "Compare an estimated camera trajectory against ground truth and report "
        "ATE and RPE statistics (RMSE / mean / median / std). Aligns with Umeyama; "
        "enable monocular to also correct scale."
    )
    category = "tool"
    icon = "Ruler"
    input_ports = [
        PortDef("poses_est", "ANY", "Estimated trajectory (list of 4x4 poses)"),
        PortDef("poses_gt", "ANY", "Ground-truth trajectory (list of 4x4 poses)"),
        PortDef("is_monocular", "ANY", "Override the monocular/scale-align flag", optional=True),
    ]
    output_ports = [
        PortDef("ate_rmse", "ANY", "Absolute Trajectory Error RMSE (metres)"),
        PortDef("rpe_rmse", "ANY", "Relative Pose Error RMSE (metres, delta=1 frame)"),
        PortDef("metrics", "TEXT", "Full ATE + RPE statistics as JSON"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"ate_rmse": None, "rpe_rmse": None,
                    "metrics": '{"error": "pyslam nodeset not initialized"}'}
        try:
            mono_in = inputs.get("is_monocular")
            if mono_in is None:
                mono = str(self.config.get("is_monocular") or "false").lower() == "true"
            else:
                mono = bool(mono_in) if not isinstance(mono_in, str) else mono_in.lower() == "true"
            # Keep only frame-aligned est/gt pairs (get_trajectory may hand over
            # None gt slots for steps where no ground truth was wired).
            est_in = inputs.get("poses_est") or []
            gt_in = inputs.get("poses_gt") or []
            pairs = [(e, g) for e, g in zip(est_in, gt_in, strict=False)
                     if e is not None and g is not None]
            if not pairs:
                return {"ate_rmse": None, "rpe_rmse": None,
                        "metrics": json.dumps({"error": "no frame-aligned est/gt pose pairs"})}
            est = [e for e, _ in pairs]
            gt = [g for _, g in pairs]
            out = ns._session.eval_trajectory(est, gt, is_monocular=mono)
            return {
                "ate_rmse": out.get("ate", {}).get("rmse"),
                "rpe_rmse": out.get("rpe", {}).get("rmse"),
                "metrics": json.dumps(out, ensure_ascii=False),
            }
        except Exception as exc:
            log.exception("pyslam eval_trajectory failed")
            return {"ate_rmse": None, "rpe_rmse": None,
                    "metrics": json.dumps({"error": str(exc)})}


# ── Full-surface neural nodes ───────────────────────────────────────────────
#
# Beyond the SLAM session + feature/eval tools, pyslam bundles a bank of dense
# perception backends (depth / semantics / multi-view reconstruction / dense
# mapping). Exposing pyslam's *whole* surface — not a curated subset — means these
# get first-class nodes too. Like Tier-2 they are stateless (no Slam session), but
# they run neural nets on the GPU (the cpu-fixed image ships a cu128 torch), so the
# first call may download a checkpoint. This is the first: dense depth prediction.


class PySlamPredictDepthTool(BaseCanvasNode):
    """Predict a dense depth map from an image (stateless — pyslam's estimator bank).

    Exposes pyslam ``depth_estimation.depth_estimator_factory``: mono estimators
    (Depth-Anything V2 / Depth Pro / MASt3R) need only ``rgb`` and run on the GPU;
    stereo estimators (SGBM / RAFT-Stereo / CREStereo) also consume ``image_right``.
    Learned models auto-download a checkpoint on first use; SGBM is pure OpenCV/CPU.
    """

    node_type = "model_pyslam__predict_depth"
    display_name = "pySLAM: Predict Depth"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "estimator", "select", label="Estimator", default="DEPTH_ANYTHING_V2",
                options=[
                    {"value": "DEPTH_ANYTHING_V2", "label": "Depth-Anything V2 (mono, GPU)"},
                    {"value": "DEPTH_PRO", "label": "Depth Pro (mono, GPU)"},
                    {"value": "DEPTH_MAST3R", "label": "MASt3R (mono, GPU)"},
                    {"value": "DEPTH_SGBM", "label": "SGBM (stereo, CPU, needs right image)"},
                    {"value": "DEPTH_RAFT_STEREO", "label": "RAFT-Stereo (stereo, GPU)"},
                    {"value": "DEPTH_CRESTEREO_PYTORCH", "label": "CREStereo (stereo, GPU)"},
                ],
            ),
            ConfigField(
                "environment", "select", label="Scene type", default="INDOOR",
                options=[
                    {"value": "INDOOR", "label": "Indoor"},
                    {"value": "OUTDOOR", "label": "Outdoor"},
                ],
            ),
            ConfigField("min_depth", "text", label="Min depth (m)", default="0.0"),
            ConfigField("max_depth", "text", label="Max depth (m)", default="10.0"),
        ],
    )
    description = (
        "Predict a dense depth map from a single RGB image using pySLAM's depth "
        "estimator bank. Mono estimators (Depth-Anything V2, Depth Pro, MASt3R) run "
        "on the GPU from RGB alone; stereo estimators (SGBM, RAFT-Stereo, CREStereo) "
        "also take a right image. Returns an HxW metric depth map (metres)."
    )
    category = "tool"
    icon = "Layers"
    input_ports = [
        PortDef("rgb", "IMAGE", "RGB frame (np.ndarray HxWx3 uint8, or .npy path)"),
        PortDef("image_right", "IMAGE",
                "Right RGB frame — stereo estimators only (SGBM / RAFT / CREStereo)",
                optional=True),
    ]
    output_ports = [
        PortDef("depth", "DEPTH", "Predicted dense depth map (HxW float32, metres)"),
        PortDef("depth_range", "ANY", "[min, max] of the predicted depth (metres)"),
        PortDef("estimator", "TEXT", "Estimator that produced the depth"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"depth": None, "depth_range": None, "estimator": "ERROR: not initialized"}
        try:
            rgb = _decode_image_input(inputs["rgb"])
            right = inputs.get("image_right")
            right = _decode_image_input(right) if right is not None else None
            out = ns._session.predict_depth(
                rgb, right,
                estimator=self.config.get("estimator") or "DEPTH_ANYTHING_V2",
                min_depth=float(self.config.get("min_depth") or 0.0),
                max_depth=float(self.config.get("max_depth") or 10.0),
                environment=self.config.get("environment") or "INDOOR",
            )
            log.info("pyslam predict_depth: est=%s shape=%s range=[%.3f,%.3f]",
                     out["estimator"], out["shape"], out["min"], out["max"])
            return {
                "depth": out["depth"],
                "depth_range": [out["min"], out["max"]],
                "estimator": out["estimator"],
            }
        except Exception as exc:
            log.exception("pyslam predict_depth failed")
            return {"depth": None, "depth_range": None, "estimator": f"ERROR: {exc}"}


class PySlamSegmentSemanticTool(BaseCanvasNode):
    """Semantic-segment one image (stateless — pyslam's segmentation bank).

    Exposes pyslam ``semantics.semantic_segmentation_factory``: DeepLabV3 /
    SegFormer / CLIP / Detic / YOLO / … run on the GPU, RGB in, a per-pixel label
    map (or probability / feature volume) out, plus an optional instance map.
    Learned models auto-download a checkpoint on first use.
    """

    node_type = "model_pyslam__segment_semantic"
    display_name = "pySLAM: Segment Semantic"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "model", "select", label="Model", default="DEEPLABV3",
                options=[
                    {"value": "DEEPLABV3", "label": "DeepLabV3 (torchvision)"},
                    {"value": "SEGFORMER", "label": "SegFormer"},
                    {"value": "CLIP", "label": "CLIP (open-vocab)"},
                    {"value": "DETIC", "label": "Detic (open-vocab, instances)"},
                    {"value": "YOLO", "label": "YOLO (instances)"},
                    {"value": "EOV_SEG", "label": "EOV-Seg"},
                ],
            ),
            ConfigField(
                "feature_type", "select", label="Output", default="LABEL",
                options=[
                    {"value": "LABEL", "label": "Label map (HxW ints)"},
                    {"value": "PROBABILITY_VECTOR", "label": "Probabilities (HxWxC)"},
                    {"value": "FEATURE_VECTOR", "label": "Feature volume (HxWxD)"},
                ],
            ),
            ConfigField(
                "dataset", "select", label="Label set", default="CITYSCAPES",
                options=[
                    {"value": "CITYSCAPES", "label": "Cityscapes"},
                    {"value": "ADE20K", "label": "ADE20K (indoor+outdoor)"},
                    {"value": "VOC", "label": "Pascal VOC"},
                    {"value": "NYU40", "label": "NYU40 (indoor)"},
                ],
            ),
        ],
    )
    description = (
        "Semantic-segment a single RGB image using pySLAM's segmentation bank "
        "(DeepLabV3 / SegFormer / CLIP / Detic / YOLO). Returns a per-pixel "
        "semantic map (label ints, or a probability/feature volume) and an "
        "optional instance map. Runs on the GPU."
    )
    category = "tool"
    icon = "Shapes"
    input_ports = [PortDef("rgb", "IMAGE", "RGB frame (np.ndarray HxWx3 uint8, or .npy path)")]
    output_ports = [
        PortDef("semantics", "ANY",
                "Per-pixel semantic map: HxW int labels, or HxWxC/HxWxD volume"),
        PortDef("instances", "ANY", "Per-pixel instance map (HxW), or null if not produced"),
        PortDef("num_classes", "ANY", "Number of classes in the label set"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"semantics": None, "instances": None, "num_classes": 0}
        try:
            rgb = _decode_image_input(inputs["rgb"])
            out = ns._session.segment_semantic(
                rgb,
                model=self.config.get("model") or "DEEPLABV3",
                feature_type=self.config.get("feature_type") or "LABEL",
                dataset=self.config.get("dataset") or "CITYSCAPES",
            )
            log.info("pyslam segment_semantic: model=%s shape=%s classes=%s instances=%s",
                     out["model"], out["shape"], out["num_classes"],
                     "Y" if out["instances"] is not None else "-")
            return {
                "semantics": out["semantics"],
                "instances": out["instances"],
                "num_classes": out["num_classes"],
            }
        except Exception as exc:
            log.exception("pyslam segment_semantic failed")
            return {"semantics": None, "instances": None, "num_classes": f"ERROR: {exc}"}


class PySlamReconstructMultiviewTool(BaseCanvasNode):
    """Feed-forward multi-view 3-D reconstruction from N images (stateless — GPU).

    Exposes pyslam ``scene_from_views.scene_from_views_factory`` — the DUSt3R /
    MASt3R / VGGT / MV-DUSt3R family. Give a list of ≥2 overlapping RGB views;
    returns a fused global point cloud (handle), per-view camera-to-world poses,
    and (optionally) a mesh. MASt3R / MV-DUSt3R load weights from the mounted
    data/models/pyslam/ folder; the HF-runtime backends download on first use.
    """

    node_type = "model_pyslam__reconstruct_multiview"
    display_name = "pySLAM: Reconstruct Multi-view"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="indigo",
        config_fields=[
            ConfigField(
                "backend", "select", label="Backend", default="MAST3R",
                options=[
                    {"value": "MAST3R", "label": "MASt3R (metric, mounted weights)"},
                    {"value": "DUST3R", "label": "DUSt3R (HF weights)"},
                    # MVDUST3R: known gap — pyslam's bundled mvdust3r has a version
                    # drift against its own dust3r copy (import error on
                    # normalize_pointclouds); the other backends cover multi-view.
                    {"value": "MVDUST3R", "label": "MV-DUSt3R (known gap — unsupported)"},
                    {"value": "VGGT", "label": "VGGT (HF weights)"},
                    {"value": "VGGT_ROBUST", "label": "VGGT-Robust (HF weights)"},
                    {"value": "FAST3R", "label": "Fast3r (HF weights)"},
                    {"value": "DEPTH_ANYTHING_V3", "label": "Depth-Anything V3 (HF weights)"},
                ],
            ),
            ConfigField(
                "as_pointcloud", "select", label="Output", default="true",
                options=[
                    {"value": "true", "label": "Point cloud"},
                    {"value": "false", "label": "Mesh (+ point cloud)"},
                ],
            ),
        ],
    )
    description = (
        "Reconstruct a fused 3-D scene from a list of overlapping RGB images using "
        "pySLAM's feed-forward multi-view stack (MASt3R / DUSt3R / VGGT / MV-DUSt3R). "
        "Returns a global point-cloud handle plus per-view camera-to-world poses. "
        "Runs on the GPU; the first call may load or download the model weights."
    )
    category = "tool"
    icon = "Box"
    input_ports = [
        PortDef("images", "ANY",
                "List of RGB frames (np.ndarray HxWx3 uint8, or .npy/image paths) — "
                "at least 2 overlapping views"),
    ]
    output_ports = [
        PortDef("scene_handle", "TEXT", "Path to the fused scene (.npz: points, colors, mesh)"),
        PortDef("camera_poses", "ANY", "Per-view 4x4 camera-to-world poses"),
        PortDef("num_points", "ANY", "Number of points in the fused cloud"),
        PortDef("num_views", "ANY", "Number of input views reconstructed"),
    ]

    def __init__(self, nodeset: ModelPySlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = self._nodeset or _NODESET
        if ns is None or ns._session is None:
            return {"scene_handle": "ERROR: not initialized", "camera_poses": None,
                    "num_points": 0, "num_views": 0}
        try:
            raw = inputs["images"]
            if not isinstance(raw, (list, tuple)):
                raise TypeError("reconstruct_multiview: 'images' must be a list of ≥2 views")
            images = [_decode_image_input(im) for im in raw]
            out = ns._session.reconstruct_multiview(
                images,
                backend=self.config.get("backend") or "MAST3R",
                as_pointcloud=str(self.config.get("as_pointcloud") or "true").lower() == "true",
            )
            log.info("pyslam reconstruct_multiview: backend=%s views=%s points=%s verts=%s",
                     out["backend"], out["num_views"], out["num_points"], out["num_vertices"])
            return {
                "scene_handle": out["scene_handle"],
                "camera_poses": out["camera_poses"],
                "num_points": out["num_points"],
                "num_views": out["num_views"],
            }
        except Exception as exc:
            log.exception("pyslam reconstruct_multiview failed")
            return {"scene_handle": f"ERROR: {exc}", "camera_poses": None,
                    "num_points": 0, "num_views": 0}


# ── NodeSet ────────────────────────────────────────────────────────────────


class ModelPySlamNodeSet(BaseNodeSet):
    """pySLAM streaming-SLAM session nodeset (Tier-1 skeleton).

    All four nodes are always registered. When the session is not yet built (no
    camera configured / Slam not started), nodes return a benign error rather
    than crashing the server — the capability-flag discipline from ``model_sam``.

    Runs in **local mode** (``server_python = None``): the session is a
    :class:`_client.PySlamContainerClient` that ``docker run``s the
    ``agentcanvas/pyslam`` image and drives it over HTTP. The only place
    ``import pyslam`` happens is inside that container (``_server.py`` →
    ``_backend.py``), keeping the GPL dependency behind the container boundary.
    """

    name = "model_pyslam"
    description = "pySLAM streaming SLAM session (track / trajectory / sparse map)"
    # Local mode: the pyslam dependency lives in a Docker container reached over
    # HTTP (see _client.py), NOT a conda env — so no server_python interpreter.
    server_python: ClassVar[str | None] = None
    parallelism: ClassVar[str] = "replicated"  # map is mutable per-episode state

    def __init__(self) -> None:
        self._session: Any = None
        self._config: dict = {}
        # Frame-aligned ground-truth poses for trajectory eval — the track node
        # appends one here iff pySLAM produced an estimated pose that step, so
        # this list stays index-aligned with the session's estimated trajectory
        # (the same association pySLAM's own eval does by timestamp).
        self._gt_traj: list = []

    async def initialize(self, **kwargs: Any) -> None:
        # Runs in the framework process. Create the container client (cheap) then
        # `docker run` the pyslam image + wait for its shim to answer — the heavy
        # part, offloaded off the event loop. The Slam instance itself is built
        # lazily on first reset once camera intrinsics are known.
        from ._client import PySlamContainerClient

        self._config = {
            "sensor_type": os.environ.get("PYSLAM_SENSOR", "rgbd"),
            "feature_preset": os.environ.get("PYSLAM_FEATURE", "ORB2"),
            "loop_preset": os.environ.get("PYSLAM_LOOP", "DBOW3"),
            "cam_width": os.environ.get("PYSLAM_CAM_W"),
            "cam_height": os.environ.get("PYSLAM_CAM_H"),
            "cam_hfov": os.environ.get("PYSLAM_CAM_HFOV", "90"),
            "volumetric": os.environ.get("PYSLAM_VOLUMETRIC", "0").lower() in ("1", "true", "yes"),
            "volumetric_type": os.environ.get("PYSLAM_VOLUMETRIC_TYPE", "VOXEL_GRID"),
            "environment": os.environ.get("PYSLAM_ENV", "INDOOR"),
        }
        self._session = PySlamContainerClient(
            sensor_type=self._config["sensor_type"],
            feature_preset=self._config["feature_preset"],
            loop_preset=self._config["loop_preset"],
            headless=True,
            volumetric=self._config["volumetric"],
            volumetric_type=self._config["volumetric_type"],
            environment=self._config["environment"],
        )
        await asyncio.to_thread(self._session.start_container)
        global _NODESET
        _NODESET = self  # nodes reach the session through this (see _NODESET note)
        log.info(
            "pyslam nodeset ready (container bridge): sensor=%s feature=%s loop=%s",
            self._config["sensor_type"], self._config["feature_preset"],
            self._config["loop_preset"],
        )

    def _ensure_built(self, intrinsics: Any = None, width: Any = None,
                      height: Any = None, hfov: Any = None) -> None:
        """Build the Slam instance on first use. Prefer an explicit ``intrinsics``
        dict (``fx/fy/cx/cy/width/height`` — e.g. env_habitat's observe port) so
        the pinhole matches the real camera exactly; else derive fx from a
        horizontal FOV. Idempotent — a no-op once built. (Depth is scaled to
        metres in the track node, not here — pySLAM's DepthMapFactor only applies
        to file-loaded depth, not an ndarray passed straight to ``track()``.)"""
        if self._session.is_built:
            return
        if isinstance(intrinsics, dict) and intrinsics.get("fx"):
            w = int(intrinsics.get("width") or width or self._config.get("cam_width") or 640)
            h = int(intrinsics.get("height") or height or self._config.get("cam_height") or 480)
            fx = float(intrinsics["fx"])
            self._session.configure_camera(
                width=w, height=h, fx=fx, fy=float(intrinsics.get("fy") or fx),
                cx=float(intrinsics.get("cx") or w / 2.0),
                cy=float(intrinsics.get("cy") or h / 2.0),
            )
        else:
            w = int(width if width is not None else (self._config.get("cam_width") or 640))
            h = int(height if height is not None else (self._config.get("cam_height") or 480))
            fov = float(hfov if hfov is not None else (self._config.get("cam_hfov") or 90))
            self._session.configure_camera(width=w, height=h, hfov_deg=fov)
        self._session.start()

    async def shutdown(self) -> None:
        if self._session is not None:
            # close() quits Slam in-container then `docker rm -f`s it; offload
            # the blocking docker call off the event loop.
            await asyncio.to_thread(self._session.close)
            self._session = None
        global _NODESET
        if _NODESET is self:
            _NODESET = None
        log.info("pyslam nodeset shut down")

    def get_tools(self) -> list:
        return [
            # Tier-1: streaming SLAM session
            PySlamResetTool(self),
            PySlamTrackTool(self),
            PySlamGetTrajectoryTool(self),
            PySlamGetMapTool(self),
            PySlamGetDenseMapTool(self),
            # Tier-2: stateless perception (share the container, no Slam session)
            PySlamExtractFeaturesTool(self),
            PySlamMatchFeaturesTool(self),
            PySlamEvalTrajectoryTool(self),
            # Full-surface neural nodes (GPU; first call may download a checkpoint)
            PySlamPredictDepthTool(self),
            PySlamSegmentSemanticTool(self),
            PySlamReconstructMultiviewTool(self),
        ]
