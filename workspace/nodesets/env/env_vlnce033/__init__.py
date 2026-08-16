from __future__ import annotations

"""EnvVlnce033NodeSet — "VLN-CE @ hs0.3.3": R2R-CE episodes on modern habitat-sim.

Runs the R2R_VLNCE_v1-3 episode datasets on habitat-sim 0.3.3 (the
``ac-habitat033`` conda env), bypassing habitat-lab entirely: episode loading,
the discrete action space, and the VLN-CE metric suite are implemented
directly against the ``habitat_sim`` API (same approach as env_hmeqa /
env_express). The legacy ``env_habitat`` nodeset (habitat-sim 0.1.7 via
the ``vlnce`` env) stays frozen as the literature-comparable oracle.

IMPORTANT — reporting caveat: numbers from this env are the "modern
(0.3.3) caliber", NOT the 0.1.7 literature caliber. Navmesh generation,
sliding, and rendering differ across habitat versions; always label
results accordingly and never mix them into 0.1.7 comparison tables.

Node surface mirrors ``env_habitat`` so the verified ``straightforward``
graph topology carries over with only node-type renames:

  env_vlnce033__reset               — ensure-live; episode metadata
  env_vlnce033__step_discrete       — 0=STOP 1=FWD 2=LEFT 3=RIGHT
  env_vlnce033__observe_egocentric  — rgb/depth/pose/intrinsics/
                                          raw_obs/instruction_text
  env_vlnce033__evaluate            — NE/SR/SPL/OSR/nDTW/TL sink

Faithful VLN-CE task caliber (vlnce_task.yaml + habitat-lab 0.1.7
defaults):

- actions: STOP / MOVE_FORWARD 0.25 m / TURN_LEFT 15° / TURN_RIGHT 15°,
  sliding allowed
- sensors: RGB 224×224 hfov 90, depth 256×256 hfov 90, both at
  [0, 1.25, 0]; depth normalized to [0, 1] over 0–10 m (habitat-lab
  ``NORMALIZE_DEPTH: True`` default — CMA checkpoints expect this)
- success distance 3.0 m; episode cap 500 steps
- metrics: distance_to_goal (geodesic), success (STOP within 3 m), spl,
  oracle_success, ndtw (gt locations, euclidean, threshold 3 m),
  path_length, steps_taken

Extra action caliber for SLAM work (NOT part of VLN-CE): the
``step_size_m`` / ``turn_angle_deg`` config on initialize lets a graph
shrink the motion quantum (e.g. 0.05 m / 5°) for continuous-trajectory
experiments. Defaults are the faithful VLN-CE values.

Data layout:
    data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
    data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
    data/scene_datasets/mp3d/{scan}/{scan}.glb + .navmesh

last updated: 2026-08-15
"""

import asyncio
import concurrent.futures
import gzip
import json
import logging
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

log = logging.getLogger("agentcanvas.env_vlnce033")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))
_DATASET_ROOT = os.environ.get(
    "VLNCE033_DATA_ROOT",
    os.path.join(_REPO_ROOT, "data", "habitat", "datasets", "R2R_VLNCE_v1-3_preprocessed"),
)
_SCENE_ROOT = os.environ.get(
    "VLNCE033_SCENE_ROOT", os.path.join(_REPO_ROOT, "data", "scene_datasets")
)

_DEFAULTS: dict[str, Any] = {
    "rgb_size": 224,
    "depth_size": 256,
    "hfov": 90.0,
    "camera_height": 1.25,   # habitat-lab 0.1.7 default sensor POSITION
    "depth_max": 10.0,       # habitat-lab MAX_DEPTH default; depth → [0,1]
    "step_size_m": 0.25,     # FORWARD_STEP_SIZE
    "turn_angle_deg": 15.0,  # TURN_ANGLE
    "success_distance": 3.0,
    "max_steps": 500,        # MAX_EPISODE_STEPS
    "seed": 42,
}

_ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def _list_splits() -> list[str]:
    """Splits = subdirs of the dataset root that carry {split}.json.gz."""
    if not os.path.isdir(_DATASET_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(_DATASET_ROOT)):
        if os.path.isfile(os.path.join(_DATASET_ROOT, name, f"{name}.json.gz")):
            out.append(name)
    return out


def _load_episodes(split: str) -> list[dict[str, Any]]:
    path = os.path.join(_DATASET_ROOT, split, f"{split}.json.gz")
    if not os.path.isfile(path):
        log.error("R2R-CE split file missing: %s", path)
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["episodes"]


def _load_gt_locations(split: str) -> dict[str, list[list[float]]]:
    """episode_id → gt locations (for nDTW). Empty dict when unavailable."""
    path = os.path.join(_DATASET_ROOT, split, f"{split}_gt.json.gz")
    if not os.path.isfile(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        gt = json.load(f)
    return {str(k): v.get("locations", []) for k, v in gt.items()}


def _cam_intrinsics(hfov_deg: float, height: int, width: int) -> dict[str, float]:
    hfov = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    vfov = 2 * np.arctan(np.tan(hfov / 2.0) * height / width)
    fy = (height / 2.0) / np.tan(vfov / 2.0)
    return {"fx": float(fx), "fy": float(fy), "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height}


def _ndtw(path: list[list[float]], gt: list[list[float]], threshold: float) -> float:
    """Standard nDTW (euclidean, success-threshold normalization)."""
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
# Vlnce033EnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class Vlnce033EnvManager:
    """Singleton R2R-CE runtime on raw habitat-sim.

    The simulator is rebuilt only on scene change (episodes within one
    MP3D scan reuse the live sim — R2R-CE splits are scene-clustered).
    All simulator calls run on a pinned single-thread executor.
    """

    _instance: Vlnce033EnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlnce033"
        )
        self._config: dict[str, Any] = dict(_DEFAULTS)

        self._split: str = "val_unseen"
        self._episodes: list[dict[str, Any]] = []
        self._gt_locations: dict[str, list[list[float]]] = {}

        self._simulator: Any = None
        self._scene_path: str = ""
        self._ep: dict[str, Any] = {}
        self._ep_index: int = -1
        self._step_index = 0
        self._stop_called = False
        self._done = False
        self._agent_path: list[list[float]] = []
        self._collisions = 0

    @classmethod
    def get(cls) -> Vlnce033EnvManager:
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
            splits = _list_splits()
            if self._split not in splits and splits:
                self._split = "val_unseen" if "val_unseen" in splits else splits[0]
            self._episodes = _load_episodes(self._split)
            self._gt_locations = _load_gt_locations(self._split)
            log.info(
                "VLN-CE@hs033: initialized split=%s (%d episodes, %d gt paths)",
                self._split, len(self._episodes), len(self._gt_locations),
            )

    def shutdown(self) -> None:
        with self._lock:
            self._close_sim_unlocked()

    def _close_sim_unlocked(self) -> None:
        if self._simulator is not None:
            try:
                self._simulator.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.warning("VLN-CE@hs033: simulator close failed", exc_info=True)
            self._simulator = None
            self._scene_path = ""

    # ── Episode control ────────────────────────────────────────────────

    def list_splits(self) -> list[str]:
        return _list_splits()

    def set_split(self, split: str) -> dict[str, Any]:
        with self._lock:
            if split == self._split and self._episodes:
                return {"split": split, "episode_count": len(self._episodes)}
            episodes = _load_episodes(split)
            if not episodes:
                return {"error": f"split '{split}' not found or empty"}
            self._split = split
            self._episodes = episodes
            self._gt_locations = _load_gt_locations(split)
            self._ep_index = -1
            self._ep = {}
            return {"split": split, "episode_count": len(episodes)}

    def list_episodes(self, split: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if split and split != self._split:
                episodes = _load_episodes(split)
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

    def _make_sim_cfg(self, scene_path: str) -> Any:
        import habitat_sim  # lazy — only importable in the ac-habitat033 subprocess

        cfg = self._config
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.allow_sliding = True  # HABITAT_SIM_V0.ALLOW_SLIDING
        sim_cfg.random_seed = int(cfg["seed"])

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [cfg["rgb_size"], cfg["rgb_size"]]
        rgb_spec.position = [0.0, cfg["camera_height"], 0.0]
        rgb_spec.hfov = cfg["hfov"]

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [cfg["depth_size"], cfg["depth_size"]]
        depth_spec.position = [0.0, cfg["camera_height"], 0.0]
        depth_spec.hfov = cfg["hfov"]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.1
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["step_size_m"])),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["turn_angle_deg"])),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["turn_angle_deg"])),
            ),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def set_episode_by_index(self, index: int) -> dict[str, Any]:
        with self._lock:
            if not self._episodes:
                return {"error": "not initialized — no episodes loaded"}
            if index < 0 or index >= len(self._episodes):
                return {"error": f"index {index} out of range (0..{len(self._episodes) - 1})"}

            ep = self._episodes[index]
            scene_rel = str(ep["scene_id"])  # e.g. "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
            scene_path = os.path.join(_SCENE_ROOT, scene_rel)
            if not os.path.isfile(scene_path):
                return {"error": f"scene mesh missing: {scene_path}"}

            import habitat_sim  # lazy

            if self._simulator is None or scene_path != self._scene_path:
                self._close_sim_unlocked()
                self._simulator = habitat_sim.Simulator(self._make_sim_cfg(scene_path))
                self._scene_path = scene_path
                navmesh_path = os.path.splitext(scene_path)[0] + ".navmesh"
                if os.path.isfile(navmesh_path):
                    self._simulator.pathfinder.load_nav_mesh(navmesh_path)
                else:
                    log.warning("VLN-CE@hs033: no navmesh at %s — recomputing", navmesh_path)
                    settings = habitat_sim.NavMeshSettings()
                    settings.set_defaults()
                    settings.agent_height = 1.5
                    settings.agent_radius = 0.1
                    if not self._simulator.recompute_navmesh(
                        self._simulator.pathfinder, settings
                    ):
                        raise RuntimeError(f"recompute_navmesh failed for {scene_path}")
                self._simulator.seed(int(self._config["seed"]))

            import quaternion  # noqa: F401 — enables np.quaternion

            agent = self._simulator.get_agent(0)
            state = habitat_sim.AgentState()
            state.position = np.asarray(ep["start_position"], dtype=np.float32)
            rot = ep["start_rotation"]  # dataset order [x, y, z, w]
            state.rotation = np.quaternion(rot[3], rot[0], rot[1], rot[2])
            agent.set_state(state)

            self._ep = ep
            self._ep_index = index
            self._step_index = 0
            self._stop_called = False
            self._done = False
            self._collisions = 0
            self._agent_path = [list(map(float, agent.get_state().position))]

            log.info(
                "VLN-CE@hs033: episode %d (id=%s) scene=%s",
                index, ep.get("episode_id"), scene_rel,
            )
            return self._episode_meta_unlocked()

    def _episode_meta_unlocked(self) -> dict[str, Any]:
        ep = self._ep
        return {
            "episode_id": str(ep.get("episode_id", "")),
            "scene_id": str(ep.get("scene_id", "")),
            "instruction": (ep.get("instruction") or {}).get("instruction_text", ""),
            "geodesic_distance": float((ep.get("info") or {}).get("geodesic_distance", 0.0)),
            "step_count": self._step_index,
            "done": self._done,
        }

    def ensure_live(self) -> dict[str, Any]:
        """Template reset semantics: live episode read untouched; done
        episode re-armed in place. Never chooses an episode."""
        with self._lock:
            live = self._simulator is not None and self._ep and not self._done
            index = self._ep_index
        if live:
            with self._lock:
                return self._episode_meta_unlocked()
        if index < 0:
            index = 0
        return self.set_episode_by_index(index)

    # ── Transition + perception ────────────────────────────────────────

    def step(self, action: int) -> dict[str, Any]:
        with self._lock:
            if self._simulator is None or not self._ep:
                return {"error": "no live episode — set an episode first"}
            if self._done:
                return {
                    "reward": 0.0, "terminated": True, "truncated": False,
                    "info": {"note": "episode already done"},
                    "step_count": self._step_index,
                }

            action = int(action)
            if action == 0:
                self._stop_called = True
                self._done = True
            else:
                name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(action)
                if name is None:
                    return {"error": f"invalid action {action} (expected 0-3)"}
                prev = np.asarray(self._simulator.get_agent(0).get_state().position)
                self._simulator.step(name)
                pos = np.asarray(self._simulator.get_agent(0).get_state().position)
                if name == "move_forward" and float(np.linalg.norm(pos - prev)) < 1e-4:
                    self._collisions += 1
                self._agent_path.append(list(map(float, pos)))

            self._step_index += 1
            truncated = False
            if not self._done and self._step_index >= int(self._config["max_steps"]):
                truncated = True
                self._done = True

            info = {
                "action": action,
                "action_name": _ACTION_NAMES.get(action, "UNKNOWN"),
                "step_count": self._step_index,
                "collisions": self._collisions,
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

    def observe(self) -> dict[str, Any]:
        with self._lock:
            if self._simulator is None or not self._ep:
                return {"error": "no live episode"}
            obs = self._simulator.get_sensor_observations()
            rgb = np.asarray(obs["color_sensor"], dtype=np.uint8)
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            depth_m = np.asarray(obs["depth_sensor"], dtype=np.float32).squeeze()
            depth = np.clip(depth_m, 0.0, self._config["depth_max"]) / self._config["depth_max"]

            state = self._simulator.get_agent(0).get_state()
            pose = {
                "position": list(map(float, state.position)),
                "orientation": [
                    float(state.rotation.x), float(state.rotation.y),
                    float(state.rotation.z), float(state.rotation.w),
                ],
            }
            return {
                "rgb": rgb,
                "depth": depth,
                "pose": pose,
                "raw_obs": {
                    "rgb": rgb,
                    "depth": depth,
                    "step_index": self._step_index,
                    "episode_id": str(self._ep.get("episode_id", "")),
                },
                "instruction_text": (self._ep.get("instruction") or {}).get(
                    "instruction_text", ""
                ),
            }

    def get_intrinsics(self) -> dict[str, float]:
        cfg = self._config
        return _cam_intrinsics(cfg["hfov"], cfg["rgb_size"], cfg["rgb_size"])

    # ── Metrics ────────────────────────────────────────────────────────

    def _geodesic_unlocked(self, a: list[float], b: list[float]) -> float:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        if self._simulator.pathfinder.find_path(path):
            return float(path.geodesic_distance)
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _metrics_unlocked(self) -> dict[str, float]:
        cfg = self._config
        ep = self._ep
        goal = (ep.get("goals") or [{}])[0].get("position")
        if goal is None or self._simulator is None:
            return {}

        cur = self._agent_path[-1] if self._agent_path else ep["start_position"]
        d_goal = self._geodesic_unlocked(cur, goal)
        success = float(self._stop_called and d_goal <= cfg["success_distance"])

        # Oracle success: geodesic distance from every visited position.
        oracle = 0.0
        for p in self._agent_path:
            if self._geodesic_unlocked(p, goal) <= cfg["success_distance"]:
                oracle = 1.0
                break

        path_len = 0.0
        for i in range(1, len(self._agent_path)):
            path_len += float(
                np.linalg.norm(
                    np.asarray(self._agent_path[i]) - np.asarray(self._agent_path[i - 1])
                )
            )

        gd = float((ep.get("info") or {}).get("geodesic_distance", 0.0))
        spl = success * gd / max(path_len, gd) if gd > 0 else 0.0

        gt = self._gt_locations.get(str(ep.get("episode_id", "")), [])
        ndtw = _ndtw(self._agent_path, gt, cfg["success_distance"]) if gt else 0.0

        return {
            "distance_to_goal": round(d_goal, 4),
            "success": success,
            "spl": round(spl, 4),
            "oracle_success": oracle,
            "ndtw": round(ndtw, 4),
            "path_length": round(path_len, 4),
            "steps_taken": float(self._step_index),
            "stop_called": float(self._stop_called),
            "collisions": float(self._collisions),
        }

    def evaluate(self) -> dict[str, Any]:
        with self._lock:
            if self._simulator is None or not self._ep:
                return {"error": "no live episode"}
            return self._metrics_unlocked()


def _mgr() -> Vlnce033EnvManager:
    return Vlnce033EnvManager.get()


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas nodes — the four verbs (surface mirrors env_habitat)
# ══════════════════════════════════════════════════════════════════════


class ResetVlnce033Tool(BaseCanvasNode):
    node_type = "env_vlnce033__reset"
    display_name = "VLN-CE@hs033: Reset"
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
        return {
            "instruction": meta.get("instruction", ""),
            "episode_id": meta.get("episode_id", ""),
            "scene_id": meta.get("scene_id", ""),
        }


class StepDiscreteVlnce033Tool(BaseCanvasNode):
    node_type = "env_vlnce033__step_discrete"
    display_name = "VLN-CE@hs033: Step (discrete)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Advance one tick with a discrete action (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT)"
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


class ObserveEgocentricVlnce033Tool(BaseCanvasNode):
    node_type = "env_vlnce033__observe_egocentric"
    display_name = "VLN-CE@hs033: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull the current first-person observation: RGB, depth, pose, intrinsics"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB observation (224×224×3)"),
        PortDef("depth", "DEPTH", "Current depth map (256×256, normalized [0,1])"),
        PortDef("pose", "POSE", "Agent position + orientation quaternion [x,y,z,w]"),
        PortDef("intrinsics", "ANY", "RGB camera intrinsics {fx,fy,cx,cy,width,height}"),
        PortDef("raw_obs", "ANY", "Raw observation dict (for policy adapter stage 1)"),
        PortDef(
            "instruction_text",
            "TEXT",
            "Raw natural-language instruction for the current episode",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().observe)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"rgb": None, "depth": None, "pose": None, "intrinsics": None,
                    "raw_obs": None, "instruction_text": ""}
        self._self_log("rgb_shape", list(result["rgb"].shape))
        return {
            "rgb": result["rgb"],
            "depth": result["depth"],
            "pose": result["pose"],
            "intrinsics": _mgr().get_intrinsics(),
            "raw_obs": result["raw_obs"],
            "instruction_text": result["instruction_text"],
        }


class EvaluateVlnce033Tool(BaseCanvasNode):
    node_type = "env_vlnce033__evaluate"
    display_name = "VLN-CE@hs033: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = "Pull R2R-CE metrics (NE/SR/SPL/OSR/nDTW/TL) without stepping"
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "R2R-CE metric dict"),
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


class Vlnce033EnvPanel(BaseEnvPanel):
    name: ClassVar[str] = "env_vlnce033"
    display_name: ClassVar[str] = "VLN-CE @ hs0.3.3 Env"

    fields: ClassVar[list[EnvPanelField]] = [
        EnvPanelField("split", "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions: ClassVar[list[EnvPanelAction]] = [
        EnvPanelAction("play", "Play", side_effect="run_start", enabled_when="idle"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop", "Stop", side_effect="run_stop", enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none", enabled_when="idle"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"split": "val_unseen", "episode_index": 0}

    async def on_load(self) -> dict[str, Any]:
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
            "episode_count": len(episodes),
            "splits": splits,
            "step_budget": int(_DEFAULTS["max_steps"]),
        }

    async def _seat_episode(self) -> dict[str, Any]:
        mgr = _mgr()
        await _run(mgr.set_split, self._state["split"])
        return await _run(mgr.set_episode_by_index, int(self._state["episode_index"]))

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        self._state[name] = value if name == "split" else int(value)
        if name == "split":
            self._state["episode_index"] = 0
        await self._seat_episode()
        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = {
            "split": self._state["split"],
            "episode_index": self._state["episode_index"],
        }
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
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

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        mgr = _mgr()
        if field == "split":
            return [{"value": s, "label": s} for s in await _run(mgr.list_splits)]
        if field == "episode_index":
            episodes = await _run(mgr.list_episodes)
            return [
                {"value": e["index"], "label": f'{e["index"]}: {e["episode_id"]} ({e["scan"]})'}
                for e in episodes
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ══════════════════════════════════════════════════════════════════════


class EnvVlnce033NodeSet(BaseNodeSet):
    name = "env_vlnce033"
    description = "VLN-CE @ hs0.3.3 — R2R-CE episodes on habitat-sim 0.3.3; modern caliber, not 0.1.7-comparable"
    server_python = conda_env_python("ac-habitat033", "VLNCE033_PYTHON")
    env_panel = Vlnce033EnvPanel
    parallelism: ClassVar[str] = "replicated"  # stateful sim: per-worker scene + pose
    default_per_step_budget_sec: ClassVar[float] = 2.0  # habitat step is fast

    def get_tools(self) -> list:
        return [
            ResetVlnce033Tool(),
            StepDiscreteVlnce033Tool(),
            ObserveEgocentricVlnce033Tool(),
            EvaluateVlnce033Tool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        await _run(lambda: _mgr().initialize(**kwargs))

    async def shutdown(self) -> None:
        await _run(_mgr().shutdown)

    async def get_eval_metadata(self) -> dict:
        splits = await _run(_mgr().list_splits)
        return {
            "env_name": "vlnce033",
            "splits": splits,
            "episode_counts": {},
            "metrics": [
                "distance_to_goal", "success", "spl", "oracle_success",
                "ndtw", "path_length", "steps_taken",
            ],
            "supports_set_episode": True,
            "step_budget": int(_DEFAULTS["max_steps"]),
        }
