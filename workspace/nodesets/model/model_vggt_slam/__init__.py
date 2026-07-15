from __future__ import annotations

"""ModelVggtSlamNodeSet — VGGT-SLAM 2.0 as a server-mode, session-type ``model/`` nodeset.

Wraps VGGT-SLAM 2.0 (MIT-SPARK, RSS 2026; BSD-2-Clause) on the canvas: dense
RGB-only SLAM — VGGT feed-forward submap reconstruction + DINOv2-SALAD
loop-closure retrieval + GTSAM SL(4) pose-graph optimization — plus the
upstream's optional open-set 3D object detection (Perception Encoder CLIP +
SAM 3, ``run_os``). Upstream pin: MIT-SPARK/VGGT-SLAM @ 35327ac (reference
clone: ``third_party/zz_just_for_refer/vggt_slam/``).

    Streaming SLAM session (mutable map state):
    model_vggt_slam__reset           trigger → fresh Solver/map for a new episode
    model_vggt_slam__track           rgb (+timestamp, +gt_pose) → pose + flags (the workhorse)
    model_vggt_slam__finalize        trigger → process the trailing partial submap
    model_vggt_slam__get_trajectory  trigger → TUM-format trajectory + 4x4 poses
    model_vggt_slam__get_map         trigger → dense colored .pcd handle (path, not inline)
    model_vggt_slam__eval_trajectory traj (+gt) → ATE via evo_ape tum -as (upstream ruler)
    model_vggt_slam__query_object    text → best frame + SAM3 masks + 3D OBBs (run_os only)

Pose semantics (IMPORTANT for graph authors): VGGT-SLAM processes frames in
SUBMAP BURSTS (default 16 keyframes + 1 overlap). ``track.pose`` is None until
the first submap completes, then updates once per submap boundary, and earlier
poses are retro-corrected by later optimization / loop closures. The
authoritative trajectory is ``get_trajectory`` fired AFTER ``finalize``.
Scale note: poses live on the SL(4) projective manifold (up-to-scale) —
evaluate with Sim(3) alignment (``evo_ape tum -as``), never raw.

Classification: like model_pyslam, no MDP / episode selection / actions —
"feed frames in, map accumulates, read products out" → ``model/``, session
shape. Unlike pyslam it is RGB-only and self-calibrating (VGGT predicts
intrinsics), so ``track`` takes no depth and ``reset`` takes no intrinsics.

Deployment — dedicated conda env, server mode (the model_vggt pattern, NOT
pyslam's docker bridge): upstream is pip-installable, so it lives in the
``ac-vggt-slam`` env (torch 2.3.1 — incompatible with ac-vggt's 2.8) and this
package is imported directly inside the AutoServerApp subprocess. Session
state persists on the nodeset instance across /call's (auto_server_app.py
constructs the nodeset once). ``parallelism = "replicated"``: the map is
mutable per-episode state; each eval worker gets its own server process.

Environment:
    ac-vggt-slam (Python 3.11)  — scripts/install/install_ac_vggt_slam.sh
    $VGGT_SLAM_PYTHON           — override the interpreter
    Weights: VGGT-1B (torch-hub auto), dino_salad.ckpt (install script),
    PE-Core-L14-336 + SAM 3 (HF auto, run_os only).

Deviations from upstream (full list + rationale in ``_backend.py`` docstring):
    D1 headless viewer patch (viser :8080 hardcoded upstream) ·
    D2 SAM3 lazy-load at first query (VRAM) ·
    D3 keyframes re-encoded to timestamp-named PNGs (streaming source has no files).

Load: POST /api/components/nodesets/model_vggt_slam/load?mode=server

last updated: 2026-07-14 (initial port)
"""

import asyncio
import functools
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
    conda_env_python,
)

log = logging.getLogger("agentcanvas.vggt_slam")

# The graph executor fires FRESH node_cls() instances (never the get_tools()
# instances carrying self._nodeset) — nodes reach the live session through
# this module global, set by initialize(). One process per replicated worker,
# so a module global is the correct per-worker isolation (pyslam pattern,
# model_pyslam/__init__.py:104-119).
#
# Resolution order is _NODESET FIRST, node backref second (`_NODESET or
# self._nodeset`): AutoServerApp calls get_functions() BEFORE on_startup to
# build the manifest, instantiating a throwaway nodeset whose tools the /call
# handlers close over; on_startup then constructs and initializes a SECOND
# instance (auto_server_app.py:248,330). The backref therefore points at the
# never-initialized throwaway (truthy, _session=None) — only _NODESET names
# the instance initialize() actually blessed. Caught live 2026-07-14 (first
# graph smoke: every /call returned "not initialized").
_NODESET: ModelVggtSlamNodeSet | None = None


def _decode_image_input(value: Any) -> np.ndarray:
    """Accept an ndarray (msgpack wire) or a path to a .npy / image file."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(value)
    if isinstance(value, str) and os.path.isfile(value):
        if value.endswith(".npy"):
            return np.load(value)
        import cv2

        bgr = cv2.imread(value, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image file: {value}")
        return bgr[..., ::-1].copy()
    if isinstance(value, list):
        return np.asarray(value, dtype=np.uint8)
    raise ValueError(f"unsupported image input type: {type(value)!r}")


async def _run(ns: ModelVggtSlamNodeSet, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a session method on the session's 1-worker executor (thread affinity)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        ns._session.executor, functools.partial(fn, *args, **kwargs)
    )


# ══════════════════════════════════════════════════════════════════════
# Session nodes
# ══════════════════════════════════════════════════════════════════════


class VggtSlamResetTool(BaseCanvasNode):
    """Start / restart the SLAM session — fresh Solver, map, and pose graph."""

    node_type = "model_vggt_slam__reset"
    display_name = "VGGT-SLAM: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            # Defaults mirror upstream argparse (main.py:29-34 @ 35327ac).
            ConfigField("submap_size", "text", label="Submap size (new frames)", default="16"),
            ConfigField("max_loops", "text", label="Max loop closures (0 disables)", default="1"),
            ConfigField("min_disparity", "text", label="Min keyframe disparity (px)", default="50"),
            ConfigField("conf_threshold", "text",
                        label="Conf percentile filtered (%)", default="25.0"),
            ConfigField("lc_thres", "text", label="Retrieval threshold [0-1]", default="0.95"),
            ConfigField(
                "run_os", "select", label="Open-set semantics (PE-CLIP during tracking)",
                default="false",
                options=[
                    {"value": "false", "label": "off"},
                    {"value": "true", "label": "on (needed for query_object)"},
                ],
            ),
        ],
    )
    description = (
        "Begin or reset a VGGT-SLAM session: fresh submap map, SL(4) pose graph, "
        "and SALAD retrieval database. VGGT-1B loads once per worker and is reused "
        "across episodes. Enable run_os here if query_object will be used — CLIP "
        "embeddings are computed during tracking and cannot be backfilled."
    )
    category = "tool"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Fire to (re)start the session", optional=True),
    ]
    output_ports = [
        PortDef("episode_ok", "BOOL", "True once the session is ready"),
        PortDef("info", "TEXT", "Session config summary or error message"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"episode_ok": False, "info": '{"error": "vggt_slam nodeset not initialized"}'}
        try:
            cfg = {
                "submap_size": int(float(self.config.get("submap_size") or 16)),
                "max_loops": int(float(self.config.get("max_loops") or 1)),
                "min_disparity": float(self.config.get("min_disparity") or 50),
                "conf_threshold": float(self.config.get("conf_threshold") or 25.0),
                "lc_thres": float(self.config.get("lc_thres") or 0.95),
                "run_os": str(self.config.get("run_os") or "false").lower() == "true",
            }
            info = await _run(ns, ns._session.reset, cfg)
            return {"episode_ok": True, "info": json.dumps(info, ensure_ascii=False)}
        except Exception as exc:
            log.exception("vggt_slam reset failed")
            return {"episode_ok": False, "info": json.dumps({"error": str(exc)})}


class VggtSlamTrackTool(BaseCanvasNode):
    """Feed one RGB frame — keyframe gate, submap batching, burst-updated pose."""

    node_type = "model_vggt_slam__track"
    display_name = "VGGT-SLAM: Track"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = (
        "Feed one RGB frame to the SLAM session. Frames pass an optical-flow "
        "keyframe gate; keyframes accumulate until a submap (submap_size+1 "
        "frames) triggers a VGGT forward + loop-closure check + SL(4) optimize. "
        "pose is the latest optimized camera pose — None until the first submap "
        "completes, then updated once per submap (burst semantics; see nodeset "
        "docstring). Wire timestamp from the env for eval-grade trajectories."
    )
    category = "tool"
    icon = "Camera"
    input_ports = [
        PortDef("rgb", "IMAGE", "RGB frame (HxWx3 uint8 ndarray, or image path)"),
        PortDef("timestamp", "ANY",
                "Frame timestamp in seconds (TUM convention) — load-bearing for "
                "eval association; default = frame index", optional=True),
        PortDef("gt_pose", "POSE",
                "Ground-truth pose {position, orientation} (env observe.pose) — "
                "captured keyframe-aligned for eval_trajectory", optional=True),
    ]
    output_ports = [
        PortDef("pose", "POSE", "Latest optimized 4x4 cam-to-world pose (list of lists), or null"),
        PortDef("is_keyframe", "BOOL", "Frame passed the disparity gate"),
        PortDef("submap_processed", "BOOL", "This fire ran a full submap (VGGT+optimize)"),
        PortDef("num_keyframes", "ANY", "Keyframes accepted so far"),
        PortDef("num_submaps", "ANY", "Non-loop-closure submaps in the map"),
        PortDef("num_loops", "ANY", "Loop closures accepted so far"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        empty = {"pose": None, "is_keyframe": False, "submap_processed": False,
                 "num_keyframes": 0, "num_submaps": 0, "num_loops": 0}
        if ns is None or ns._session is None:
            return empty
        try:
            rgb = _decode_image_input(inputs["rgb"])
            ts = inputs.get("timestamp")
            ts = float(ts) if ts is not None else None
            result = await _run(ns, ns._session.track, rgb, ts, inputs.get("gt_pose"))
            if result["submap_processed"]:
                log.info("vggt_slam track: submap done — kf=%s submaps=%s loops=%s",
                         result["num_keyframes"], result["num_submaps"], result["num_loops"])
            return result
        except Exception as exc:
            log.exception("vggt_slam track failed")
            return {**empty, "pose": None}


class VggtSlamFinalizeTool(BaseCanvasNode):
    """Process the trailing partial submap after the frame stream ends."""

    node_type = "model_vggt_slam__finalize"
    display_name = "VGGT-SLAM: Finalize"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = (
        "Flush the keyframe buffer as a final (possibly partial) submap — the "
        "streaming equivalent of upstream's last-image trigger. Fire once after "
        "the loop (iterOut.final_stop); its ok output gates the getters/eval. "
        "Idempotent."
    )
    category = "tool"
    icon = "FlagTriangleRight"
    input_ports = [PortDef("trigger", "ANY", "Fire after the frame loop ends", optional=True)]
    output_ports = [
        PortDef("ok", "BOOL", "Finalize completed"),
        PortDef("submaps_processed", "ANY", "Total submaps processed this episode"),
        PortDef("num_loops", "ANY", "Total loop closures accepted"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"ok": False, "submaps_processed": 0, "num_loops": 0}
        try:
            result = await _run(ns, ns._session.finalize)
            return {"ok": True,
                    "submaps_processed": result["submaps_processed"],
                    "num_loops": result["num_loops"]}
        except Exception as exc:
            log.exception("vggt_slam finalize failed")
            return {"ok": False, "submaps_processed": 0, "num_loops": 0}


class VggtSlamGetTrajectoryTool(BaseCanvasNode):
    """Read the optimized trajectory (TUM text + 4x4 pose list)."""

    node_type = "model_vggt_slam__get_trajectory"
    display_name = "VGGT-SLAM: Get Trajectory"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = (
        "Export the optimized keyframe trajectory: TUM-format text (timestamp tx "
        "ty tz qx qy qz qw — upstream's exact pose writer) plus parsed 4x4 "
        "matrices, and the keyframe-aligned ground-truth rows captured by track. "
        "Fire after finalize for the authoritative result."
    )
    category = "tool"
    icon = "Route"
    input_ports = [PortDef("trigger", "ANY", "Fire to read the trajectory", optional=True)]
    output_ports = [
        PortDef("poses", "ANY", "List of 4x4 cam-to-world poses (SE(3)-decomposed)"),
        PortDef("traj_tum", "TEXT", "Estimated trajectory, TUM format"),
        PortDef("gt_tum", "TEXT", "Keyframe-aligned ground truth, TUM format (may be empty)"),
        PortDef("num_poses", "ANY", "Number of keyframe poses"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"poses": [], "traj_tum": "", "gt_tum": "", "num_poses": 0}
        try:
            return await _run(ns, ns._session.get_trajectory)
        except Exception:
            log.exception("vggt_slam get_trajectory failed")
            return {"poses": [], "traj_tum": "", "gt_tum": "", "num_poses": 0}


class VggtSlamGetMapTool(BaseCanvasNode):
    """Export the dense colored point-cloud map to a .pcd handle."""

    node_type = "model_vggt_slam__get_map"
    display_name = "VGGT-SLAM: Get Map"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = (
        "Export the merged confidence-filtered world-frame point cloud: the "
        "upstream .pcd writer runs verbatim (pcd_path artifact), and the same "
        "cloud is re-exported as an .npz handle (points/colors keys — what "
        "pointCloudViewer and downstream consumers np.load). Heavy geometry "
        "rides disk handles, never an inline wire. Fire after finalize."
    )
    category = "tool"
    icon = "Map"
    input_ports = [PortDef("trigger", "ANY", "Fire to export the map", optional=True)]
    output_ports = [
        PortDef("map_handle", "TEXT", "Path to the .npz cloud handle (points/colors; empty on failure)"),
        PortDef("pcd_path", "TEXT", "Path to the upstream-format merged colored .pcd"),
        PortDef("num_points", "ANY", "Confidence-filtered point count"),
        PortDef("num_submaps", "ANY", "Non-loop-closure submaps merged"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"map_handle": "", "pcd_path": "", "num_points": 0, "num_submaps": 0}
        try:
            return await _run(ns, ns._session.get_map)
        except Exception:
            log.exception("vggt_slam get_map failed")
            return {"map_handle": "", "pcd_path": "", "num_points": 0, "num_submaps": 0}


class VggtSlamEvalTrajectoryTool(BaseCanvasNode):
    """ATE against ground truth — the upstream eval ruler (evo_ape tum -as)."""

    node_type = "model_vggt_slam__eval_trajectory"
    display_name = "VGGT-SLAM: Eval Trajectory"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "align_scale", "select", label="Sim(3) scale alignment (-s)",
                default="true",
                options=[
                    {"value": "true", "label": "on (-as, upstream ruler — REQUIRED for SL(4) poses)"},
                    {"value": "false", "label": "off (-a only)"},
                ],
            ),
        ],
    )
    description = (
        "Compute ATE RMSE with the exact upstream ruler (evals/eval_tum.sh): "
        "evo_ape tum <gt> <est> -as. Ground truth resolves from the sequence "
        "name (data/tum/<seq>/groundtruth.txt — full-file association, parity "
        "grade) or falls back to the keyframe-aligned gt_tum text from "
        "get_trajectory."
    )
    category = "tool"
    icon = "Ruler"
    input_ports = [
        PortDef("traj_tum", "ANY", "Estimated trajectory in TUM format (get_trajectory.traj_tum)"),
        PortDef("gt_tum", "ANY", "Ground-truth TUM text (fallback)", optional=True),
        PortDef("sequence", "ANY", "TUM sequence name → data/tum/<seq>/groundtruth.txt", optional=True),
    ]
    output_ports = [
        PortDef("ate_rmse", "ANY", "ATE RMSE in metres after Sim(3) alignment (null on failure)"),
        PortDef("metrics", "METRICS", "Full evo metrics {rmse, mean, median, std, ...}"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"ate_rmse": None, "metrics": {"error": "nodeset not initialized"}}
        try:
            align_scale = str(self.config.get("align_scale") or "true").lower() == "true"
            result = await _run(
                ns, ns._session.eval_trajectory,
                str(inputs.get("traj_tum") or ""),
                inputs.get("gt_tum"),
                inputs.get("sequence"),
                align_scale,
            )
            parsed = json.loads(result["metrics"])
            if result["ate_rmse"] is not None:
                parsed["ate_rmse"] = result["ate_rmse"]  # canonical key for eval harvest
            return {"ate_rmse": result["ate_rmse"], "metrics": parsed}
        except Exception as exc:
            log.exception("vggt_slam eval_trajectory failed")
            return {"ate_rmse": None, "metrics": {"error": str(exc)}}


class VggtSlamQueryObjectTool(BaseCanvasNode):
    """Open-set 3D object query: text → best frame → SAM3 masks → world OBBs."""

    node_type = "model_vggt_slam__query_object"
    display_name = "VGGT-SLAM: Query Object"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("sam3_conf", "text", label="SAM3 confidence threshold", default="0.50"),
            ConfigField("default_text", "text",
                        label="Default query (used when the text port is unwired)", default=""),
        ],
    )
    description = (
        "Open-set query over the built map (requires run_os=true at reset): "
        "PE-CLIP text embedding retrieves the best-matching keyframe, SAM 3 "
        "segments the prompt in it, and each mask is lifted to world-frame "
        "points + a PCA oriented bounding box. Post-hoc — fire any time after "
        "tracking (SAM 3 loads lazily on first query)."
    )
    category = "tool"
    icon = "ScanSearch"
    input_ports = [
        PortDef("text", "TEXT", "Object text prompt (e.g. 'keyboard'); falls back to "
                "the default_text config when unwired", optional=True),
        PortDef("trigger", "ANY", "Optional gate (e.g. finalize.ok)", optional=True),
    ]
    output_ports = [
        PortDef("best_submap_id", "ANY", "Submap id of the best-matching frame"),
        PortDef("best_frame_id", "ANY", "Frame index inside that submap"),
        PortDef("score", "ANY", "CLIP cosine similarity of the retrieval"),
        PortDef("num_instances", "ANY", "SAM3 instances lifted to 3D"),
        PortDef("obb", "ANY", "Per-instance {center, extent, rotation, sam3_score}"),
        PortDef("points_path", "TEXT", "NPZ of per-instance world-frame points"),
        PortDef("overlay_path", "TEXT", "PNG of the query frame with mask overlay"),
        PortDef("best_frame_image_path", "TEXT", "PNG of the raw best-matching frame"),
        PortDef("info", "TEXT", "Status or error message"),
    ]

    def __init__(self, nodeset: ModelVggtSlamNodeSet | None = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        empty = {"best_submap_id": None, "best_frame_id": None, "score": None,
                 "num_instances": 0, "obb": [], "points_path": "", "overlay_path": "",
                 "best_frame_image_path": "", "info": ""}
        if ns is None or ns._session is None:
            return {**empty, "info": "vggt_slam nodeset not initialized"}
        try:
            text = str(inputs.get("text") or self.config.get("default_text") or "").strip()
            if not text:
                return {**empty, "info": "empty text prompt"}
            sam3_conf = float(self.config.get("sam3_conf") or 0.50)
            result = await _run(ns, ns._session.query_object, text, sam3_conf)
            if "error" in result:
                return {**empty, "info": result["error"]}
            return {**result, "info": "ok"}
        except Exception as exc:
            log.exception("vggt_slam query_object failed")
            return {**empty, "info": str(exc)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class ModelVggtSlamNodeSet(BaseNodeSet):
    """VGGT-SLAM 2.0 dense RGB SLAM session — server-mode on ac-vggt-slam."""

    name = "model_vggt_slam"
    description = (
        "VGGT-SLAM 2.0 (MIT-SPARK): dense RGB SLAM — VGGT submap reconstruction "
        "+ SALAD loop closure + GTSAM SL(4) optimization, with optional open-set "
        "3D object queries (PE-CLIP + SAM 3). Session nodes: reset / track / "
        "finalize / get_trajectory / get_map / eval_trajectory / query_object."
    )
    # Stateful session (accumulating map) — one server process per eval worker.
    parallelism = "replicated"
    # Dedicated env: upstream pins torch 2.3.1 (ac-vggt runs 2.8; ac-fm numpy 2).
    server_python = conda_env_python("ac-vggt-slam", "VGGT_SLAM_PYTHON")
    # VGGT-1B bf16 (~5 GB weights + submap activations) + SALAD, + PE/SAM3 under run_os.
    expected_vram_mb = 16000
    # A submap-boundary track fires a 17-frame VGGT forward + LC + optimize (1-3 s
    # on desktop GPU); non-boundary tracks are <50 ms. Budget for the boundary.
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None

    def get_tools(self) -> list:
        return [
            VggtSlamResetTool(self),
            VggtSlamTrackTool(self),
            VggtSlamFinalizeTool(self),
            VggtSlamGetTrajectoryTool(self),
            VggtSlamGetMapTool(self),
            VggtSlamEvalTrajectoryTool(self),
            VggtSlamQueryObjectTool(self),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        global _NODESET
        # Heavy import (torch / vggt_slam / gtsam) — resolvable only inside the
        # ac-vggt-slam server process; the framework env never gets past the
        # module docstring because server_python auto-routes loads to server mode.
        from . import _backend

        self._session = _backend.VggtSlamSession()
        _NODESET = self
        log.info("model_vggt_slam ready (server_python=%s); VGGT-1B loads on first reset",
                 self.server_python)

    async def shutdown(self) -> None:
        global _NODESET
        if self._session is not None:
            session = self._session
            self._session = None
            await asyncio.to_thread(session.close)
        _NODESET = None
