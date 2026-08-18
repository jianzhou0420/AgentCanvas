from __future__ import annotations

"""EnvSlamVlnceNodeSet — R2R-CE over SLAM-instrumented habitat-sim 0.3.3.

The env side of the 2026-08-17 SLAM-instrument probe (bare-vs-instrumented
arms over SlamEnv), promoted from the slam-frontier worktree into a proper
env nodeset. The world is habitat-sim 0.3.3 (ac-habitat033, no habitat-lab)
running R2R_VLNCE_v1-3 episodes, wrapped by the SlamEnv middleware
(_slam_env.py): every coarse VLN-CE action (FORWARD 0.25 m, LEFT/RIGHT 15°)
decomposes into 5 micro-steps (0.05 m / 3°), and every micro-frame is
auto-fed to ORB-SLAM3 (dockerized, agentcanvas/orbslam3 — the same image as
model_orbslam3) plus a 2D occupancy grid as a side effect of motion. The
agent's only relationship with SLAM is read-only queries.

The ``instruments`` env-panel field is the arm switch (set it BEFORE
placement):
  0 (bare)  — pose_source="gt": no SLAM container, no bootstrap; the mapping
              stack runs off GT poses but nothing exposes it to the agent.
  1 (slam)  — pose_source="slam": ORB-SLAM3 container per episode, a
              disclosed 360° bootstrap scan at placement, and the three
              query verbs return SLAM estimates.

Node surface (env template + the three instrument reads):

  env_slam_vlnce__reset               — ensure-live; episode metadata
  env_slam_vlnce__step_discrete       — 0=STOP 1=FWD 2=LEFT 3=RIGHT (coarse)
  env_slam_vlnce__observe_egocentric  — rgb/depth/pose/intrinsics (512²)
  env_slam_vlnce__get_pose            — SLAM pose estimate (x, z, yaw_deg)
  env_slam_vlnce__get_map             — occupancy map image + frontier list
  env_slam_vlnce__get_trajectory      — estimated path so far
  env_slam_vlnce__evaluate            — NE/SR/SPL/OSR/nDTW/TL (+SLAM stats)

IMPORTANT — reporting caveat: this is the "modern (0.3.3) caliber" sibling
of the std R2R line, NOT 0.1.7-comparable (same caveat as env_vlnce033 on
main: navmesh, sliding and rendering differ across habitat versions). Its
board (slamr2r_*) is its own line; never mix into 0.1.7 tables.

Caliber (probe-faithful): square RGB-D 512², hfov 90, camera 1.25 m, METRIC
depth (ORB-SLAM3 RGBD needs metres — deliberately unlike env_vlnce033's
normalized depth); success distance 3.0 m; 500 coarse-step cap. A fresh
SlamEnv (fresh simulator + fresh SLAM container in the instrumented arm) is
built per episode — the occupancy grid and SLAM anchor are episode-scoped.

Data layout (shared with env_vlnce033, same override env vars):
    data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
    data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
    data/scene_datasets/mp3d/{scan}/{scan}.glb + .navmesh

last updated: 2026-08-17
"""

import asyncio
import concurrent.futures
import logging
import math
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

from ._env import SCENE_ROOT, camera_intrinsics, list_splits, load_episodes, load_gt_locations
from ._frontier import frontier_mask
from ._map_render import render_annotated_map, render_annotated_map_v2
from ._slam_env import SlamEnv

# Pose/map frame facts, verified empirically 2026-08-18 (probe: 6× RIGHT →
# yaw +90; forward at yaw 0 → +z; forward at yaw 90 → +x): world anchored at
# the episode start pose, up on the map = +z = starting heading, right = +x,
# yaw_deg increases turning RIGHT, 0 = starting heading.
_POSE_CONVENTION = ("x right / z forward of your START pose (metres); "
                    "yaw_deg: 0 = your starting heading, increases turning "
                    "right; on the map, up = +z (your starting heading)")

log = logging.getLogger("agentcanvas.env_slam_vlnce")

_DEFAULTS: dict = {
    "img_size": 512,          # square aligned RGB-D pair (probe caliber)
    "coarse_step_m": 0.25,    # VLN-CE FORWARD_STEP_SIZE
    "coarse_turn_deg": 15.0,  # VLN-CE TURN_ANGLE
    "cell_size": 0.10,        # occupancy grid resolution
    "map_size_m": 48.0,
    "success_distance": 3.0,
    "max_steps": 500,         # coarse-action cap per episode
}

_ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def _ndtw(path: list, gt: list, threshold: float) -> float:
    """Standard nDTW (euclidean, success-threshold normalization) — same
    implementation as env_vlnce033's."""
    if not path or not gt:
        return 0.0
    p = np.asarray(path, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    n, m = len(p), len(g)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(p[i - 1] - g[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(np.exp(-dtw[n, m] / (m * threshold)))


# ══════════════════════════════════════════════════════════════════════
# SlamVlnceEnvManager — singleton runtime (one SlamEnv per live episode)
# ══════════════════════════════════════════════════════════════════════


class SlamVlnceEnvManager:
    """Singleton SlamEnv runtime.

    Unlike env_vlnce033 (which reuses the simulator across same-scene
    episodes) a FRESH SlamEnv is built per placement: the occupancy grid and
    the SLAM session are anchored at the episode's start pose, so nothing
    outlives an episode. set_episode is idempotent on an identical, still
    fresh placement — the driver's field-push + play + reset sequence costs
    ONE build (the SLAM container + 360° bootstrap are ~40 s in the
    instrumented arm, so double-seating matters here).
    """

    _instance: SlamVlnceEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slam_vlnce"
        )
        self._config: dict = dict(_DEFAULTS)

        self._split: str = "val_unseen"
        self._episodes: list = []
        self._gt_locations: dict = {}
        self._instruments = 0  # 0=bare, 1=SLAM instruments, 2=+map v2

        self._se: SlamEnv | None = None
        self._ep: dict = {}
        self._ep_index: int = -1
        self._seated_key: tuple | None = None  # (split, index, instruments)
        self._boot: dict | None = None
        self._step_index = 0
        self._stop_called = False
        self._done = False
        self._agent_path: list = []  # GT position per coarse action (oracle/nDTW)

    @classmethod
    def get(cls) -> SlamVlnceEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return self._executor

    # ── Lifecycle ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k in self._config and v is not None:
                    self._config[k] = type(_DEFAULTS[k])(v)
            splits = list_splits()
            if self._split not in splits and splits:
                self._split = "val_unseen" if "val_unseen" in splits else splits[0]
            self._episodes = load_episodes(self._split)
            self._gt_locations = load_gt_locations(self._split)
            log.info("slam_vlnce: initialized split=%s (%d episodes, %d gt paths)",
                     self._split, len(self._episodes), len(self._gt_locations))

    def shutdown(self) -> None:
        with self._lock:
            self._close_env_unlocked()

    def _close_env_unlocked(self) -> None:
        if self._se is not None:
            try:
                self._se.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.warning("slam_vlnce: SlamEnv close failed", exc_info=True)
            self._se = None
            self._seated_key = None

    # ── Episode control ────────────────────────────────────────────────

    def list_splits(self) -> list:
        return list_splits()

    def set_split(self, split: str) -> dict:
        with self._lock:
            if split == self._split and self._episodes:
                return {"split": split, "episode_count": len(self._episodes)}
            episodes = load_episodes(split)
            if not episodes:
                return {"error": f"split '{split}' not found or empty"}
            self._split = split
            self._episodes = episodes
            self._gt_locations = load_gt_locations(split)
            self._ep_index = -1
            self._ep = {}
            return {"split": split, "episode_count": len(episodes)}

    def set_instruments(self, level: int) -> dict:
        """Arm switch: 0 = bare, 1 = SLAM instruments (map v1),
        2 = SLAM instruments with map v2 (no frontier, floor-layered,
        snapped grow-only window). Takes effect on the NEXT placement
        (set_episode rebuilds when the level differs from the seated one)."""
        with self._lock:
            self._instruments = int(level)
            return {"instruments": self._instruments}

    def list_episodes(self, split: str | None = None) -> list:
        with self._lock:
            if split and split != self._split:
                episodes = load_episodes(split)
            else:
                episodes = self._episodes
            return [
                {
                    "index": i,
                    "episode_id": str(e.get("episode_id", i)),
                    "scan": str(e.get("scene_id", "")).split("/")[-2]
                    if "/" in str(e.get("scene_id", "")) else "",
                }
                for i, e in enumerate(episodes)
            ]

    def set_episode_by_index(self, index: int) -> dict:
        with self._lock:
            if not self._episodes:
                return {"error": "not initialized — no episodes loaded"}
            if index < 0 or index >= len(self._episodes):
                return {"error": f"index {index} out of range (0..{len(self._episodes) - 1})"}

            key = (self._split, index, self._instruments)
            fresh = (self._se is not None and self._step_index == 0
                     and not self._done)
            if key == self._seated_key and fresh:
                return self._episode_meta_unlocked()  # idempotent re-seat

            ep = self._episodes[index]
            scene_path = os.path.join(SCENE_ROOT, str(ep["scene_id"]))
            if not os.path.isfile(scene_path):
                return {"error": f"scene mesh missing: {scene_path}"}

            self._close_env_unlocked()
            cfg = self._config
            se = SlamEnv(
                scene_path,
                img_size=int(cfg["img_size"]),
                coarse_step_m=float(cfg["coarse_step_m"]),
                coarse_turn_deg=float(cfg["coarse_turn_deg"]),
                cell_size=float(cfg["cell_size"]),
                map_size_m=float(cfg["map_size_m"]),
                pose_source=("slam" if self._instruments else "gt"),
                map_mode=("v2" if self._instruments == 2 else "v1"),
            )
            se.reset(ep.get("start_position"), ep.get("start_rotation"))
            self._boot = se.bootstrap() if self._instruments else None

            self._se = se
            self._ep = ep
            self._ep_index = index
            self._seated_key = key
            self._step_index = 0
            self._stop_called = False
            self._done = False
            self._agent_path = [se.env.agent_position()]

            log.info("slam_vlnce: episode %d (id=%s) scene=%s instruments=%s",
                     index, ep.get("episode_id"), ep.get("scene_id"),
                     self._instruments)
            return self._episode_meta_unlocked()

    def _episode_meta_unlocked(self) -> dict:
        ep = self._ep
        return {
            "episode_id": str(ep.get("episode_id", "")),
            "scene_id": str(ep.get("scene_id", "")),
            "instruction": (ep.get("instruction") or {}).get("instruction_text", ""),
            "geodesic_distance": float((ep.get("info") or {}).get("geodesic_distance", 0.0)),
            "instruments": self._instruments,
            "bootstrap": self._boot,
            "step_count": self._step_index,
            "done": self._done,
        }

    def ensure_live(self) -> dict:
        """Template reset semantics: live episode read untouched; done
        episode re-armed in place. Never chooses an episode."""
        with self._lock:
            live = self._se is not None and self._ep and not self._done
            index = self._ep_index
        if live:
            with self._lock:
                return self._episode_meta_unlocked()
        if index < 0:
            index = 0
        return self.set_episode_by_index(index)

    # ── Transition + perception ────────────────────────────────────────

    def step(self, action: int) -> dict:
        with self._lock:
            if self._se is None or not self._ep:
                return {"error": "no live episode — set an episode first"}
            if self._done:
                return {
                    "reward": 0.0, "terminated": True, "truncated": False,
                    "info": {"note": "episode already done"},
                    "step_count": self._step_index,
                }

            action = int(action)
            act_result: dict = {}
            if action == 0:
                self._stop_called = True
                self._done = True
            elif action in (1, 2, 3):
                act_result = self._se.act(action)
                self._agent_path.append(self._se.env.agent_position())
            else:
                return {"error": f"invalid action {action} (expected 0-3)"}

            self._step_index += 1
            truncated = False
            if not self._done and self._step_index >= int(self._config["max_steps"]):
                truncated = True
                self._done = True

            info = {
                "action": action,
                "action_name": _ACTION_NAMES.get(action, "UNKNOWN"),
                "step_count": self._step_index,
                "collision": bool(act_result.get("collision", False)),
                "collisions": self._se.env.collisions,
                "slam": act_result.get("slam", self._se.status),
                "reanchors": self._se.reanchors,
                "tick": self._se.tick,
                "episode_id": str(self._ep.get("episode_id", "")),
            }
            if self._done:
                info["metrics"] = self._metrics_unlocked()
            return {
                "reward": 0.0,
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "info": info,
                "step_count": self._step_index,
                "metrics": info.get("metrics"),
            }

    def observe(self) -> dict:
        with self._lock:
            if self._se is None or not self._ep:
                return {"error": "no live episode"}
            obs = self._se.env.observe()
            pos = self._se.env.agent_position()
            return {
                "rgb": obs["rgb"],
                "depth": obs["depth"],  # METRIC metres (not normalized)
                "pose": {"position": pos},
                "intrinsics": camera_intrinsics(
                    self._se.env.hfov_deg, self._se.env.img_size),
                "instruction_text": (self._ep.get("instruction") or {}).get(
                    "instruction_text", ""),
            }

    # ── Instrument reads (read-only; SLAM estimates, not ground truth) ──

    def get_pose(self) -> dict:
        with self._lock:
            if self._se is None:
                return {"error": "no live episode"}
            pose = self._se.get_pose()
            pose["convention"] = _POSE_CONVENTION
            return pose

    def get_map(self) -> dict:
        with self._lock:
            if self._se is None:
                return {"error": "no live episode"}
            se = self._se
            if se.map_mode == "v2":
                return self._get_map_v2_unlocked(se)
            m = se.get_map()
            m.pop("png", None)  # file-path variant unused on the node surface
            # circle N on the render pairs with JSON id "FN"
            frontier_cells = [(int(f["id"][1:]), tuple(f.pop("cell")))
                              for f in m["frontiers"]]
            agent_xz = None
            yaw = None
            if se.last_pose is not None:
                agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
                yaw = math.radians(se.get_pose()["yaw_deg"])
            img, window = render_annotated_map(
                se.occ.grid, frontier_mask(se.occ.grid),
                se.occ.cell_size, se.occ.origin,
                agent_xz=agent_xz, agent_yaw=yaw,
                trajectory=se.est_track, frontier_cells=frontier_cells,
            )
            m["map"] = img
            p = se.get_pose()
            if "x" in p:
                m["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
            m["map_window_m"] = round(window, 1)
            m["orientation"] = "up = +z (your starting heading); right = +x"
            return m

    def _get_map_v2_unlocked(self, se: SlamEnv) -> dict:
        """Map v2 (SLAM-02): no frontier layer; current-floor render only;
        crop window snapped to 2 m multiples and grow-only per episode."""
        agent_xz = None
        yaw = None
        if se.last_pose is not None:
            agent_xz = (float(se.last_pose[0, 3]), float(se.last_pose[2, 3]))
            yaw = math.radians(se.get_pose()["yaw_deg"])
        floor_y = se.occ.current_floor_y
        # other storeys' track segments would scribble over this floor's map
        traj = [p for p in se.est_track
                if abs(p[1] - floor_y) <= se.occ.floor_merge_m]
        floor_label = f"floor {se.occ.current_floor + 1}/{len(se.occ.layers)}"
        start_here = bool(traj) and bool(se.est_track) and traj[0] is se.est_track[0]
        img, window, se.map_window = render_annotated_map_v2(
            se.occ.grid, se.occ.cell_size, se.occ.origin,
            agent_xz=agent_xz, agent_yaw=yaw, trajectory=traj,
            window_cells=se.map_window, floor_label=floor_label,
            mark_start=start_here,
        )
        out = {
            "map": img,
            "floor": floor_label,
            "free_m2": se.occ.free_area_m2(),
            "explored_m2": se.occ.explored_area_m2(),
            "slam": se.status, "tick": se.tick,
            "map_window_m": round(window, 1),
            "orientation": "up = +z (your starting heading); right = +x",
        }
        p = se.get_pose()
        if "x" in p:
            out["pose"] = {"x": p["x"], "z": p["z"], "yaw_deg": p["yaw_deg"]}
        return out

    def get_trajectory(self) -> dict:
        with self._lock:
            if self._se is None:
                return {"error": "no live episode"}
            return self._se.get_trajectory()

    # ── Metrics ────────────────────────────────────────────────────────

    def _metrics_unlocked(self) -> dict:
        cfg = self._config
        ep = self._ep
        se = self._se
        goal = (ep.get("goals") or [{}])[0].get("position")
        if goal is None or se is None:
            return {}

        cur = se.env.agent_position()
        d_goal = se.env.geodesic_distance(cur, goal)
        success = float(self._stop_called and d_goal <= cfg["success_distance"])

        # Oracle success: geodesic from every per-coarse-action position.
        oracle = 0.0
        for p in self._agent_path:
            if se.env.geodesic_distance(p, goal) <= cfg["success_distance"]:
                oracle = 1.0
                break

        # Path length over the micro-step GT track (turns contribute 0; the
        # bootstrap spin is in-place so it contributes 0 too).
        path_len = 0.0
        for i in range(1, len(se.env.path)):
            path_len += float(np.linalg.norm(
                np.asarray(se.env.path[i]) - np.asarray(se.env.path[i - 1])))

        gd = float((ep.get("info") or {}).get("geodesic_distance", 0.0))
        spl = success * gd / max(path_len, gd) if gd > 0 else 0.0

        gt = self._gt_locations.get(str(ep.get("episode_id", "")), [])
        ndtw = _ndtw(self._agent_path, gt, cfg["success_distance"]) if gt else 0.0

        metrics = {
            "distance_to_goal": round(d_goal, 4),
            "success": success,
            "spl": round(spl, 4),
            "oracle_success": oracle,
            "ndtw": round(ndtw, 4),
            "path_length": round(path_len, 4),
            "steps_taken": float(self._step_index),
            "stop_called": float(self._stop_called),
            "collisions": float(se.env.collisions),
            "instruments": float(self._instruments),
        }
        if self._instruments:
            slam = se.metrics()
            ate = slam.get("ate_rmse_m")
            metrics.update({
                # SLAM-quality block (instrumented arm only — in the bare arm
                # the mapping stack runs off GT and these would read as fake
                # perfection)
                "ate_rmse_m": (None if ate is None or math.isnan(ate)
                               else round(float(ate), 4)),
                "tracking_ratio": round(float(slam["tracking_ratio"]), 4),
                "lost_ticks": float(slam["lost_ticks"]),
                "reanchors": float(slam["reanchors"]),
                "explored_m2": round(float(slam["explored_m2"]), 2),
                "micro_ticks": float(slam["ticks"]),
            })
        return metrics

    def evaluate(self) -> dict:
        with self._lock:
            if self._se is None or not self._ep:
                return {"error": "no live episode"}
            return self._metrics_unlocked()


def _mgr() -> SlamVlnceEnvManager:
    return SlamVlnceEnvManager.get()


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas nodes — the four template verbs + the three instrument reads
# ══════════════════════════════════════════════════════════════════════


class ResetSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__reset"
    display_name = "SLAM-VLNCE: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Ensure a live R2R-CE episode (re-arm if done) — metadata only"
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Episode NL instruction"),
        PortDef("episode_id", "TEXT", "Episode identifier"),
        PortDef("scene_id", "TEXT", "MP3D scene id"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        meta = await _run(_mgr().ensure_live)
        if "error" in meta:
            self._self_log("error", meta["error"])
            return {"instruction": "", "episode_id": "", "scene_id": ""}
        self._self_log("episode_id", meta.get("episode_id"))
        self._self_log("instruments", meta.get("instruments"))
        return {
            "instruction": meta.get("instruction", ""),
            "episode_id": meta.get("episode_id", ""),
            "scene_id": meta.get("scene_id", ""),
        }


class StepDiscreteSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__step_discrete"
    display_name = "SLAM-VLNCE: Step (discrete)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = ("Advance one coarse action (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT); "
                   "5 micro-steps + SLAM auto-feed inside")
    category = "environment"
    icon = "Play"
    input_ports = [
        PortDef("action", "ACTION", "Discrete action (0-3)"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar; always 0.0)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (500 default)"),
        PortDef("info", "ANY", "Per-step diagnostics + terminal metrics"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        action = int(inputs.get("action", 1))
        result = await _run(_mgr().step, action)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"reward": 0.0, "terminated": True, "truncated": False,
                    "info": {"error": result["error"]}}
        self._self_log("action", action)
        self._self_log("action_name", _ACTION_NAMES.get(action, "UNKNOWN"))
        self._self_log("terminated", result["terminated"])
        self._self_log("step_count", result.get("step_count"))
        if result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": result.get("reward", 0.0),
            "terminated": bool(result.get("terminated", False)),
            "truncated": bool(result.get("truncated", False)),
            "info": result.get("info", {}),
        }


class ObserveEgocentricSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__observe_egocentric"
    display_name = "SLAM-VLNCE: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull the current first-person observation: RGB, metric depth, pose, intrinsics"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB observation (512×512×3)"),
        PortDef("depth", "DEPTH", "Current depth map (512×512, METRIC metres)"),
        PortDef("pose", "POSE", "Agent GT position (graph-side; the bridge never exposes it)"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height}"),
        PortDef("instruction_text", "TEXT", "Raw NL instruction for the current episode"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().observe)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"rgb": None, "depth": None, "pose": None, "intrinsics": None,
                    "instruction_text": ""}
        self._self_log("rgb_shape", list(result["rgb"].shape))
        return result


class GetPoseSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__get_pose"
    display_name = "SLAM-VLNCE: Get Pose"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = "Read the SLAM-estimated pose (x, z, yaw_deg) — free, read-only"
    category = "environment"
    icon = "Crosshair"
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("pose", "ANY", "{x, z, yaw_deg, slam, tick} (SLAM estimate)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().get_pose)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"pose": {"error": result["error"]}}
        self._self_log("slam", result.get("slam"))
        return {"pose": result}


class GetMapSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__get_map"
    display_name = "SLAM-VLNCE: Get Map"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = ("Read the auto-built occupancy map (top-down image) + "
                   "frontier list — free, read-only")
    category = "environment"
    icon = "Map"
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("map", "IMAGE", ("Cropped annotated top-down render: white=free, "
                                 "black=obstacle, gray=unknown; blue arrow=agent "
                                 "heading, blue line=trajectory, S=start, numbered "
                                 "green circles=frontiers (N='FN'), 2 m gridlines "
                                 "with x/z labels, scale bar; up=starting heading")),
        PortDef("frontiers", "ANY", ("[{id, dir_deg, dist_m, size_cells}] sorted by "
                                     "distance; ids stable across calls (position-"
                                     "matched, retired ids never reused)")),
        PortDef("info", "ANY", "{dir_convention, orientation, pose, map_window_m, free_m2, explored_m2, slam, tick}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().get_map)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"map": None, "frontiers": [], "info": {"error": result["error"]}}
        frontiers = result.pop("frontiers", [])
        map_img = result.pop("map", None)
        self._self_log("frontiers", len(frontiers))
        return {"map": map_img, "frontiers": frontiers, "info": result}


class GetTrajectorySlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__get_trajectory"
    display_name = "SLAM-VLNCE: Get Trajectory"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    description = "Read the estimated path so far as (x, z) points — free, read-only"
    category = "environment"
    icon = "Route"
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("trajectory", "ANY", "{points: [[x, z]...], n_total, slam, tick}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().get_trajectory)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"trajectory": {"error": result["error"]}}
        self._self_log("n_total", result.get("n_total"))
        return {"trajectory": result}


class EvaluateSlamVlnceTool(BaseCanvasNode):
    node_type = "env_slam_vlnce__evaluate"
    display_name = "SLAM-VLNCE: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = "Pull R2R-CE metrics (NE/SR/SPL/OSR/nDTW/TL + SLAM stats) without stepping"
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "R2R-CE metric dict (+SLAM block when instrumented)"),
        PortDef("success", "TEXT", "1 if agent stopped within 3 m of goal, 0 otherwise"),
        PortDef("spl", "TEXT", "Success weighted by Path Length"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().evaluate)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"metrics": {}, "success": "0", "spl": "0"}
        for k in ("success", "spl", "ndtw", "distance_to_goal", "path_length"):
            if k in result:
                self._self_log(k, result[k])
        return {
            "metrics": result,
            "success": str(int(result.get("success", 0))),
            "spl": f"{result.get('spl', 0.0):.4f}",
        }


# ══════════════════════════════════════════════════════════════════════
# Env panel
# ══════════════════════════════════════════════════════════════════════


class SlamVlnceEnvPanel(BaseEnvPanel):
    name: ClassVar[str] = "env_slam_vlnce"
    display_name: ClassVar[str] = "SLAM-VLNCE Env"

    fields: ClassVar[list] = [
        EnvPanelField("split", "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
        EnvPanelField("instruments", "select", "SLAM instruments"),
    ]
    actions: ClassVar[list] = [
        EnvPanelAction("play", "Play", side_effect="run_start", enabled_when="idle"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop", "Stop", side_effect="run_stop", enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none", enabled_when="idle"),
    ]

    def __init__(self) -> None:
        self._state: dict = {"split": "val_unseen", "episode_index": 0,
                             "instruments": 0}

    async def on_load(self) -> dict:
        mgr = _mgr()
        splits = await _run(mgr.list_splits)
        split = self._state["split"] if self._state["split"] in splits else (
            splits[0] if splits else ""
        )
        self._state["split"] = split
        if split:
            await _run(mgr.set_split, split)
        episodes = await _run(mgr.list_episodes) if split else []
        return {
            "split": split,
            "episode_index": int(self._state.get("episode_index", 0)),
            "instruments": int(self._state.get("instruments", 0)),
            "episode_count": len(episodes),
            "splits": splits,
            "step_budget": int(_DEFAULTS["max_steps"]),
        }

    async def _seat_episode(self) -> dict:
        mgr = _mgr()
        await _run(mgr.set_split, self._state["split"])
        await _run(mgr.set_instruments, int(self._state["instruments"]))
        return await _run(mgr.set_episode_by_index, int(self._state["episode_index"]))

    async def on_field_change(self, name: str, value: Any) -> dict:
        self._state[name] = value if name == "split" else int(value)
        if name == "split":
            self._state["episode_index"] = 0
        if name == "instruments":
            # arm switch only — placement (and the expensive SLAM bootstrap)
            # happens on the next episode_index push / play / reset
            await _run(_mgr().set_instruments, bool(int(value)))
            state = await self.on_load()
            return state
        await self._seat_episode()
        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = {
            "split": self._state["split"],
            "episode_index": self._state["episode_index"],
        }
        return state

    async def on_action(self, name: str, params: dict) -> dict:
        if name in ("play", "reset"):
            result = await self._seat_episode()
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            if "error" not in result:
                return {
                    "ok": True,
                    "side_effect": "signal",
                    "signal_name": "episode_reset",
                    "signal_payload": {
                        "split": self._state["split"],
                        "episode_index": self._state["episode_index"],
                    },
                }
            return {"ok": False, "side_effect": "none", "error": result.get("error")}
        if name in ("pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list:
        mgr = _mgr()
        if field == "split":
            return [{"value": s, "label": s} for s in await _run(mgr.list_splits)]
        if field == "episode_index":
            episodes = await _run(mgr.list_episodes)
            return [
                {"value": e["index"], "label": f'{e["index"]}: {e["episode_id"]} ({e["scan"]})'}
                for e in episodes
            ]
        if field == "instruments":
            return [
                {"value": 0, "label": "off (bare)"},
                {"value": 1, "label": "on (SLAM pose/map/trajectory)"},
                {"value": 2, "label": "on, map v2 (no frontier, floor-layered)"},
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ══════════════════════════════════════════════════════════════════════


class EnvSlamVlnceNodeSet(BaseNodeSet):
    name = "env_slam_vlnce"
    description = ("R2R-CE over SLAM-instrumented habitat-sim 0.3.3 — coarse "
                   "actions auto-fed to ORB-SLAM3 + occupancy grid; modern "
                   "caliber, not 0.1.7-comparable")
    server_python = conda_env_python("ac-habitat033", "SLAMVLNCE_PYTHON")
    env_panel = SlamVlnceEnvPanel
    # ADR-server-003: stateful sim (per-worker scene + pose + SLAM session)
    statefulness: ClassVar[str] = "stateful"
    # One coarse step = 5 sim micro-steps, each SLAM-tracked in the
    # instrumented arm (~1 s total); generous headroom for scene loads.
    default_per_step_budget_sec: ClassVar[float] = 10.0

    def get_tools(self) -> list:
        return [
            ResetSlamVlnceTool(),
            StepDiscreteSlamVlnceTool(),
            ObserveEgocentricSlamVlnceTool(),
            GetPoseSlamVlnceTool(),
            GetMapSlamVlnceTool(),
            GetTrajectorySlamVlnceTool(),
            EvaluateSlamVlnceTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        await _run(lambda: _mgr().initialize(**kwargs))

    async def shutdown(self) -> None:
        await _run(_mgr().shutdown)

    async def get_eval_metadata(self) -> dict:
        splits = await _run(_mgr().list_splits)
        return {
            "env_name": "slam_vlnce",
            "splits": splits,
            "episode_counts": {},
            "metrics": [
                "distance_to_goal", "success", "spl", "oracle_success",
                "ndtw", "path_length", "steps_taken",
            ],
            "supports_set_episode": True,
            "step_budget": int(_DEFAULTS["max_steps"]),
        }
