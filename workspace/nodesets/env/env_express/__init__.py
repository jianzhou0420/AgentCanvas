from __future__ import annotations

"""EnvExpressNodeSet — EXPRESS-Bench environment as a NodeSet.

Wraps Habitat-Sim + HM3D scenes for EXPRESS-Bench (Jiang et al., "Beyond
the Destination: A Novel Benchmark for Exploration-Aware Embodied
Question Answering", ICCV 2025, arXiv 2503.11117). Open-vocabulary EQA:
the agent explores from a fixed start pose and answers a free-form
question; scoring is exploration-aware — a GPT judge grades the answer
against BOTH the ground truth and the agent's final first-person image.

Runs in server mode inside the ``ac-hmeqa`` conda env (same stack as
env_hmeqa: Python 3.9, habitat-sim 0.3.x, pure habitat_sim — the
upstream repo has no habitat-lab dependency either).

Faithful to ``tmp/thirdparty/EXPRESS-Bench`` (fork of explore-eqa):

- camera: 512×512, hfov 90, height 1.5 m, tilt 0 (fine_eqa.yaml)
- start pose: ``start_position`` + identity rotation, angle 0
  (main.py:77-112; ``start_rotation`` is [1,0,0,0] in all 2,044 records)
- step = free-pose teleport; waypoints snapped via ``pathfinder.snap_point``
  with the NaN → ``get_random_navigable_point_near(last, 3)`` fallback
  (main.py:407-409); path length accumulates ``ShortestPath``
  geodesic distance between consecutive positions (main.py:410-415)
- per-episode budget ``num_step = int(sqrt(scene_size) * 3)`` (main.py:117)
- termination is method-side (VLM self-stop) or budget; the env only
  reports ``truncated`` at the budget like env_hmeqa
- metrics (evaluation.py): judge returns "δ, σ" (grounding ∈ {0,0.5,1}
  FIRST, correctness 1-5 SECOND); per episode
  C = 100·clip(δ·σ,0,5)/5, C* = 100·clip(σ,0,5)/5,
  E_path = C · l/max(p,l), d_T = geodesic(stop → goal_position)
- the judge is gpt-4o-mini with ``prompt/evaluation.txt`` + the final
  frame (main.py:425-427). The LLM call itself lives in the graph as a
  vanilla llmCall (env_openeqa_em precedent); this nodeset provides
  ``judge_prompt`` (assembles the exact system/user strings from the
  vendored prompt) and ``evaluate`` (parses "δ, σ" and folds in the
  env-side path/goal terms).

Deliberate deviations from upstream (document, don't "fix"):

- Scene resolution: upstream loads ``data/{scene_id}/{hash}.basis.glb``
  (scene_id = "hm3d/{split}/{NNNNN-HASH}") plus the *annotated* scene
  dataset config. We resolve the basename against the flat
  ``data/hm3d/hm3dsem/`` pool (shared with env_hmeqa) and load the bare
  glb: upstream's ``semantic_sensor: True`` flag is dead code — its
  ``make_simple_cfg`` builds color+depth sensors only, so semantic
  annotations are never read at inference time.
- ``d_T`` can be geodesically unreachable (inf). evaluation.py skips
  such episodes when averaging; METRICS must stay numeric, so evaluate
  emits ``d_t = -1.0`` with ``d_t_valid = 0.0`` — filter on d_t_valid
  when aggregating d_t.

Data layout:
    data/hm3d/express_bench/express-bench.json  — 2,044 QA records
    data/hm3d/hm3dsem/{scene}/{scene[6:]}.basis.glb / .basis.navmesh

last updated: 2026-07-29
"""


import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
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

log = logging.getLogger("agentcanvas.express")


# ══════════════════════════════════════════════════════════════════════
# Paths & defaults
# ══════════════════════════════════════════════════════════════════════

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))
_DATA_ROOT = os.environ.get(
    "EXPRESS_DATA_ROOT", os.path.join(_REPO_ROOT, "data", "hm3d", "express_bench")
)
_SCENE_ROOT = os.environ.get(
    "EXPRESS_SCENE_ROOT", os.path.join(_REPO_ROOT, "data", "hm3d", "hm3dsem")
)
_JUDGE_PROMPT_PATH = os.path.join(_PKG_DIR, "prompts", "evaluation.txt")

# Camera + agent defaults (mirror EXPRESS-Bench fine_eqa.yaml)
_DEFAULTS = {
    "img_height": 512,
    "img_width": 512,
    "hfov": 90,
    "camera_height": 1.5,
    "camera_tilt_deg": 0.0,
    "max_step_room_size_ratio": 3.0,
    "black_pixel_ratio": 0.5,
    "seed": 42,
}

_SPLITS = ["val", "train", "all"]


# ══════════════════════════════════════════════════════════════════════
# Coordinate helpers (vendored from explore-eqa src/habitat.py + geom.py,
# same math as env_hmeqa — duplicated so this package loads standalone
# via auto_host)
# ══════════════════════════════════════════════════════════════════════


def _pos_normal_to_habitat(pts: np.ndarray) -> np.ndarray:
    """Rotate +90° around x-axis: normal (x,y,z)→habitat (x,z,-y)."""
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def _pos_habitat_to_normal(pts: np.ndarray) -> np.ndarray:
    """Inverse of the above."""
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def _pose_habitat_to_normal(pose: np.ndarray) -> np.ndarray:
    """4×4 extrinsic: habitat→normal frame."""
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        pose,
    )


def _pose_normal_to_tsdf(pose: np.ndarray) -> np.ndarray:
    """4×4 extrinsic: normal→TSDF frame (y-flip, z-flip)."""
    return np.dot(
        pose,
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    )


def _get_cam_intr(hfov: float, img_height: int, img_width: int) -> np.ndarray:
    """3×3 camera intrinsics from pinhole params."""
    hfov_rad = hfov * np.pi / 180
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
    fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
    fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
    cx = img_width // 2
    cy = img_height // 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _get_scene_bnds(pathfinder: Any, floor_height: float) -> tuple[np.ndarray, float]:
    """TSDF bounds + scene_size from habitat pathfinder."""
    scene_bnds = pathfinder.get_bounds()
    lo = _pos_habitat_to_normal(scene_bnds[0])
    hi = _pos_habitat_to_normal(scene_bnds[1])
    scene_size = float(np.abs(np.prod(hi[:2] - lo[:2])))
    tsdf_bnds = np.array(
        [
            [min(lo[0], hi[0]), max(lo[0], hi[0])],
            [min(lo[1], hi[1]), max(lo[1], hi[1])],
            [floor_height - 0.2, floor_height + 3.5],
        ],
        dtype=np.float64,
    )
    return tsdf_bnds, scene_size


def _make_sim_cfg(
    scene_path: str, img_height: int, img_width: int, hfov: float, camera_height: float
) -> Any:
    """Build habitat_sim.Configuration with RGB + depth sensors."""
    import habitat_sim  # lazy — only works in the ac-hmeqa env subprocess

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [img_height, img_width]
    rgb_spec.position = [0.0, camera_height, 0.0]
    rgb_spec.hfov = hfov

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [img_height, img_width]
    depth_spec.position = [0.0, camera_height, 0.0]
    depth_spec.hfov = hfov

    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# ══════════════════════════════════════════════════════════════════════
# Dataset loading — express-bench.json
# ══════════════════════════════════════════════════════════════════════


def _load_records(path: str) -> list[dict[str, Any]]:
    """Load the flat EXPRESS-Bench JSON array."""
    if not os.path.isfile(path):
        log.error("EXPRESS-Bench JSON missing: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _record_split(rec: dict[str, Any]) -> str:
    """"hm3d/train/00006-HkseAnWCgqk" → "train"."""
    parts = str(rec.get("scene_id", "")).split("/")
    return parts[1] if len(parts) >= 2 else ""


def _record_scene(rec: dict[str, Any]) -> str:
    """"hm3d/train/00006-HkseAnWCgqk" → "00006-HkseAnWCgqk"."""
    return str(rec.get("scene_id", "")).split("/")[-1]


def _load_judge_prompt() -> tuple[str, str]:
    """Mirror of upstream ``gpt.py:prompt_make`` on the vendored prompt.

    system = line index 1, user = lines 3..end (the caller appends the
    per-episode "Question/Answer/Response/Your mark:" block).
    """
    with open(_JUDGE_PROMPT_PATH, encoding="utf-8") as f:
        txt = f.readlines()
    prompt_system = txt[1]
    prompt = txt[3]
    for i in range(4, len(txt)):
        prompt = prompt + txt[i]
    return prompt_system, prompt


def _parse_judge_marks(judge_text: str) -> tuple[float, int] | None:
    """Parse the judge's "δ, σ" reply — mirror of evaluation.py:8-9.

    Returns (grounding δ, correctness σ) or None when unparseable.
    Primary path is upstream-exact (strip "Your mark:", split on ","),
    with a numeric-extraction fallback for chatty judges.
    """
    raw = str(judge_text or "").replace("Your mark:", "").strip()
    try:
        parts = raw.split(",")
        return float(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(nums) >= 2:
        try:
            return float(nums[0]), int(float(nums[1]))
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# ExpressEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class ExpressEnvManager:
    """Singleton env manager for EXPRESS-Bench.

    Same shape as ``HMEQAEnvManager``: the simulator is torn down and
    rebuilt per episode (one .glb per record); all simulator calls run
    on a pinned single-thread executor for GL/physics affinity.
    """

    _instance: ExpressEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="express",
        )

        # Static data (loaded once on initialize)
        self._records: list[dict[str, Any]] = []
        self._split: str = "val"
        self._split_indices: list[int] = []  # positions into _records
        self._config: dict[str, Any] = dict(_DEFAULTS)

        # Episode-scoped state (rebuilt on set_episode)
        self._simulator: Any = None
        self._agent: Any = None
        self._pathfinder: Any = None
        self._current_episode_idx: int = -1  # index within the split
        self._ep_record: dict[str, Any] = {}
        self._ep_floor_height: float = 0.0
        self._ep_tsdf_bnds: np.ndarray | None = None
        self._ep_num_step: int = 0

        # Runtime pose (mutates per step)
        self._pts: np.ndarray = np.zeros(3)
        self._angle: float = 0.0
        self._step_index: int = 0
        self._path_len: float = 0.0

    # ── Singleton + lifecycle ──

    @classmethod
    def get(cls) -> ExpressEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return bool(self._records)

    def initialize(self, **kwargs: Any) -> None:
        """Load the QA JSON. Does NOT open a scene — that happens on set_episode."""
        with self._lock:
            split = str(kwargs.pop("split", self._split))
            self._config.update({k: v for k, v in kwargs.items() if k in _DEFAULTS})
            self._records = _load_records(os.path.join(_DATA_ROOT, "express-bench.json"))
            self._apply_split_unlocked(split if split in _SPLITS else "val")
            log.info(
                "ExpressEnvManager: loaded %d records (%s: %d episodes)",
                len(self._records),
                self._split,
                len(self._split_indices),
            )

    def shutdown(self) -> None:
        with self._lock:
            self._close_simulator_unlocked()
            self._records = []
            self._split_indices = []

    def _close_simulator_unlocked(self) -> None:
        if self._simulator is not None:
            try:
                self._simulator.close()
            except Exception:
                log.debug("Simulator.close() raised (non-fatal)", exc_info=True)
            self._simulator = None
            self._agent = None
            self._pathfinder = None

    def _apply_split_unlocked(self, split: str) -> None:
        self._split = split
        if split == "all":
            self._split_indices = list(range(len(self._records)))
        else:
            self._split_indices = [
                i for i, rec in enumerate(self._records) if _record_split(rec) == split
            ]

    def switch_split(self, split: str) -> dict[str, Any]:
        """Re-slice the episode list; tears down any live simulator."""
        with self._lock:
            if split not in _SPLITS:
                return {"error": f"unknown split {split!r}"}
            if split == self._split and self._split_indices:
                return {"split": split, "episode_count": len(self._split_indices)}
            self._close_simulator_unlocked()
            self._current_episode_idx = -1
            self._apply_split_unlocked(split)
            return {"split": split, "episode_count": len(self._split_indices)}

    def split_counts(self) -> dict[str, int]:
        counts = {"all": len(self._records)}
        for s in ("val", "train"):
            counts[s] = sum(1 for rec in self._records if _record_split(rec) == s)
        return counts

    # ── Episode control ──

    def get_total_episodes(self) -> int:
        return len(self._split_indices)

    def get_episode_info(self, index: int) -> dict[str, Any]:
        """Info for a specific episode, without switching to it."""
        if not self._records:
            return {"error": "EXPRESS-Bench not initialized"}
        if index < 0 or index >= len(self._split_indices):
            return {"error": f"index {index} out of range (0, {len(self._split_indices)})"}
        rec = self._records[self._split_indices[index]]
        return {
            "index": index,
            "episode_id": str(rec.get("episode_id", "")),
            "trajectory_id": str(rec.get("trajectory_id", "")),
            "scene": _record_scene(rec),
            "split": _record_split(rec),
            "type": rec.get("type", ""),
            "question": rec.get("question", ""),
            "answer": rec.get("answer", ""),
            "geodesic_distance": float(rec.get("geodesic_distance", 0.0)),
            "gt_step_length": int(rec.get("step_length", 0)),
        }

    def set_episode_by_index(self, index: int) -> dict[str, Any]:
        """Tear down any live simulator, open the scene for episode ``index``."""
        with self._lock:
            if not self._records:
                return {"error": "EXPRESS-Bench not initialized — call initialize() first"}
            if index < 0 or index >= len(self._split_indices):
                return {"error": f"index {index} out of range"}

            rec = self._records[self._split_indices[index]]
            scene = _record_scene(rec)
            scene_short = scene[6:] if len(scene) > 6 else scene
            mesh_path = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.glb")
            navmesh_path = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.navmesh")
            if not os.path.isfile(mesh_path):
                return {"error": f"scene mesh missing: {mesh_path}"}

            self._close_simulator_unlocked()

            import habitat_sim  # lazy

            sim_cfg = _make_sim_cfg(
                scene_path=mesh_path,
                img_height=self._config["img_height"],
                img_width=self._config["img_width"],
                hfov=self._config["hfov"],
                camera_height=self._config["camera_height"],
            )
            self._simulator = habitat_sim.Simulator(sim_cfg)
            self._pathfinder = self._simulator.pathfinder
            self._pathfinder.seed(int(self._config["seed"]))
            if os.path.isfile(navmesh_path):
                self._pathfinder.load_nav_mesh(navmesh_path)
            else:
                log.warning(
                    "EXPRESS: navmesh missing at %s — recomputing from scene mesh",
                    navmesh_path,
                )
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                navmesh_settings.agent_height = float(self._config["camera_height"])
                if not self._simulator.recompute_navmesh(self._pathfinder, navmesh_settings):
                    raise RuntimeError(f"EXPRESS: recompute_navmesh failed for scene {scene}")
            self._agent = self._simulator.initialize_agent(0)

            # Episode state — main.py:77-120: start at start_position with
            # identity rotation (angle 0); budget from scene size.
            init_pts = np.array(rec["start_position"], dtype=np.float64)
            pts_normal = _pos_habitat_to_normal(init_pts)
            floor_height = float(pts_normal[-1])
            tsdf_bnds, scene_size = _get_scene_bnds(self._pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * self._config["max_step_room_size_ratio"])

            self._current_episode_idx = index
            self._ep_record = rec
            self._ep_floor_height = floor_height
            self._ep_tsdf_bnds = tsdf_bnds
            self._ep_num_step = num_step

            self._pts = init_pts
            self._angle = 0.0
            self._step_index = 0
            self._path_len = 0.0

            self._set_agent_pose_unlocked(init_pts, self._angle)

            log.info(
                "EXPRESS: episode %d (id=%s) scene=%s scene_size=%.1f num_step=%d",
                index,
                rec.get("episode_id"),
                scene,
                scene_size,
                num_step,
            )
            return self._current_obs_unlocked()

    def _set_agent_pose_unlocked(self, pts: np.ndarray, angle: float) -> None:
        """Teleport the agent to (pts, angle). Requires the lock + live sim."""
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

        camera_tilt = self._config["camera_tilt_deg"] * np.pi / 180
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.asarray(pts, dtype=np.float64)
        agent_state.rotation = rotation
        self._agent.set_state(agent_state)

    def _current_obs_unlocked(self) -> dict[str, Any]:
        """Render RGB + depth at the current agent pose and build obs bundle."""
        import quaternion

        obs = self._simulator.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32).squeeze()

        sensor = self._agent.get_state().sensor_states["depth_sensor"]
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
        cam_pose[:3, 3] = sensor.position
        cam_pose_tsdf = _pose_normal_to_tsdf(_pose_habitat_to_normal(cam_pose))

        pts_normal = _pos_habitat_to_normal(np.asarray(self._pts, dtype=np.float64))

        h, w = rgb.shape[:2]
        num_black = int(np.sum(np.sum(rgb, axis=-1) == 0))
        is_black = num_black > self._config["black_pixel_ratio"] * h * w

        return {
            "rgb": rgb,
            "depth": depth,
            "pose": {
                "position": list(self._pts) if hasattr(self._pts, "__iter__") else [0, 0, 0],
                "orientation": [0.0, 0.0, 0.0, 1.0],  # canvas UI uses position only
            },
            "cam_pose_matrix": cam_pose_tsdf,
            "pose_normal": pts_normal,
            "angle": float(self._angle),
            "floor_height": float(self._ep_floor_height),
            "question": self._ep_record.get("question", ""),
            "episode_id": str(self._ep_record.get("episode_id", "")),
            "is_black": is_black,
            "step_index": int(self._step_index),
            "num_step": int(self._ep_num_step),
            "path_len": float(self._path_len),
        }

    def step_freepose(self, position_normal: list[float], angle: float) -> dict[str, Any]:
        """Teleport to a new pose (normal-frame 2D + floor height).

        Mirrors main.py's waypoint hop: snap the target to the navmesh
        (NaN → random navigable point near the last position, radius 3),
        accumulate the geodesic distance from the previous position, then
        set the agent state.
        """
        with self._lock:
            if self._simulator is None:
                return {"error": "no active simulator — call set_episode_by_index first"}

            import habitat_sim

            pts_normal_3d = np.append(
                np.asarray(position_normal, dtype=np.float64), self._ep_floor_height
            )
            pts_habitat = _pos_normal_to_habitat(pts_normal_3d)
            last_pts = np.asarray(self._pts, dtype=np.float64)

            pts_habitat = np.asarray(self._pathfinder.snap_point(pts_habitat), dtype=np.float64)
            if np.isnan(pts_habitat).any():
                pts_habitat = np.asarray(
                    self._pathfinder.get_random_navigable_point_near(last_pts, 3),
                    dtype=np.float64,
                )

            # Path-length accumulation (main.py:410-415) — only when a
            # geodesic path exists, exactly like upstream.
            path = habitat_sim.ShortestPath()
            path.requested_start = last_pts
            path.requested_end = pts_habitat
            if self._pathfinder.find_path(path):
                self._path_len += float(path.geodesic_distance)

            self._pts = pts_habitat
            self._angle = float(angle)
            self._set_agent_pose_unlocked(pts_habitat, self._angle)
            self._step_index += 1
            return self._current_obs_unlocked()

    def sample_waypoint(self, radius: float = 3.0) -> dict[str, Any]:
        """Random-exploration action: navigable point near the agent + random yaw.

        Mirrors the paper's RE baseline hop (``get_random_navigable_point_near``
        with radius 3 — same call step_freepose uses as its NaN fallback) and
        returns the free-pose dict ``step_freepose`` consumes. Yaw is uniform
        in [0, 2π) from a seed-derived RNG so runs replay deterministically.
        """
        with self._lock:
            if self._simulator is None:
                return {"error": "no active simulator — call set_episode_by_index first"}
            rng = getattr(self, "_np_rng", None)
            if rng is None:
                rng = np.random.RandomState(int(self._config["seed"]))
                self._np_rng = rng
            last_pts = np.asarray(self._pts, dtype=np.float64)
            pts = np.asarray(
                self._pathfinder.get_random_navigable_point_near(last_pts, float(radius)),
                dtype=np.float64,
            )
            if np.isnan(pts).any():
                pts = last_pts
            pts_normal = _pos_habitat_to_normal(pts)
            return {
                "position_normal": [float(pts_normal[0]), float(pts_normal[1])],
                "angle": float(rng.uniform(0.0, 2.0 * np.pi)),
            }

    def current_obs(self) -> dict[str, Any]:
        with self._lock:
            if self._simulator is None:
                return {"error": "no active simulator"}
            return self._current_obs_unlocked()

    def current_episode(self) -> dict[str, Any]:
        """Metadata for the currently loaded episode (no observation)."""
        with self._lock:
            if self._current_episode_idx < 0:
                return {"error": "no active episode"}
            rec = self._ep_record
            return {
                "index": self._current_episode_idx,
                "episode_id": str(rec.get("episode_id", "")),
                "trajectory_id": str(rec.get("trajectory_id", "")),
                "scene": _record_scene(rec),
                "split": _record_split(rec),
                "type": rec.get("type", ""),
                "question": rec.get("question", ""),
                "answer": rec.get("answer", ""),
                "geodesic_distance": float(rec.get("geodesic_distance", 0.0)),
                "floor_height": self._ep_floor_height,
                "num_step": self._ep_num_step,
                "step_index": int(self._step_index),
                "path_len": float(self._path_len),
                "tsdf_bnds": (
                    self._ep_tsdf_bnds.tolist() if self._ep_tsdf_bnds is not None else None
                ),
            }

    def goal_distance(self) -> float:
        """Geodesic distance from the current position to goal_position.

        Mirror of main.py:441-445 (d_T). Returns inf when unreachable.
        """
        with self._lock:
            if self._simulator is None or not self._ep_record:
                return float("inf")
            import habitat_sim

            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(self._pts, dtype=np.float64)
            path.requested_end = np.asarray(
                self._ep_record.get("goal_position", [0, 0, 0]), dtype=np.float64
            )
            self._pathfinder.find_path(path)
            return float(path.geodesic_distance)

    def get_cam_intrinsics(self) -> np.ndarray:
        return _get_cam_intr(
            self._config["hfov"], self._config["img_height"], self._config["img_width"]
        )

    def list_episodes(self, start: int = 0, count: int = 10000) -> list[dict[str, Any]]:
        """Shallow metadata for episodes in [start, start+count)."""
        out: list[dict[str, Any]] = []
        for i in range(start, min(start + count, len(self._split_indices))):
            rec = self._records[self._split_indices[i]]
            out.append(
                {
                    "index": i,
                    "episode_id": str(rec.get("episode_id", "")),
                    "scene": _record_scene(rec),
                    "type": rec.get("type", ""),
                    "question": str(rec.get("question", ""))[:80],
                }
            )
        return out


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> ExpressEnvManager:
    return ExpressEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    mgr = _get_mgr()
    return await asyncio.get_running_loop().run_in_executor(mgr.executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


class ResetExpressTool(BaseCanvasNode):
    node_type = "env_express__reset"
    display_name = "EXPRESS: Reset"
    description = (
        "Begin episode — emit question + metadata (no observation; pull via observe_egocentric)"
    )
    category = "environment"
    icon = "RotateCcw"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef(
            "trigger", "ANY", "Optional trigger — fires reset when data arrives", optional=True
        ),
    ]
    output_ports = [
        PortDef("question", "TEXT", "Open-vocabulary question text"),
        PortDef("answer", "TEXT", "Ground-truth free-form answer (judge input; never show the agent)"),
        PortDef("scene", "TEXT", "HM3D scene id (e.g. 00006-HkseAnWCgqk)"),
        PortDef("episode_id", "TEXT", "Global EXPRESS episode id"),
        PortDef("trajectory_id", "TEXT", "GT trajectory id (777 unique)"),
        PortDef("question_type", "TEXT", "One of state/existence/attribute/object/location/counting/knowledge"),
        PortDef("num_step", "ANY", "Per-episode step budget (scene-size-dependent)"),
        PortDef("floor_height", "ANY", "Floor z (episode constant)"),
        PortDef("tsdf_bnds", "ANY", "3×2 TSDF voxel-volume bounds (normal frame)"),
        PortDef("geodesic_distance", "ANY", "GT start→goal geodesic length l_i (metres)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        idx = mgr._current_episode_idx if mgr._current_episode_idx >= 0 else 0
        res = await _run_sync(mgr.set_episode_by_index, idx)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", res["error"])
        info = await _run_sync(mgr.current_episode)
        if "error" in info:
            self._self_log("error", info["error"])
            return {
                "question": "",
                "answer": "",
                "scene": "",
                "episode_id": "",
                "trajectory_id": "",
                "question_type": "",
                "num_step": 0,
                "floor_height": 0.0,
                "tsdf_bnds": None,
                "geodesic_distance": 0.0,
            }
        self._self_log("episode_id", info.get("episode_id"))
        self._self_log("scene", info.get("scene"))
        self._self_log("question", str(info.get("question", ""))[:200])
        return {
            "question": info.get("question", ""),
            "answer": info.get("answer", ""),
            "scene": info.get("scene", ""),
            "episode_id": info.get("episode_id", ""),
            "trajectory_id": info.get("trajectory_id", ""),
            "question_type": info.get("type", ""),
            "num_step": info.get("num_step", 0),
            "floor_height": info.get("floor_height", 0.0),
            "tsdf_bnds": info.get("tsdf_bnds"),
            "geodesic_distance": info.get("geodesic_distance", 0.0),
        }


class StepPoseExpressTool(BaseCanvasNode):
    node_type = "env_express__step_pose"
    display_name = "EXPRESS: Step (pose teleport)"
    description = (
        "Teleport to a waypoint (navmesh-snapped, path length accumulated); "
        "returns control signals only (pull obs via observe_egocentric)"
    )
    category = "environment"
    icon = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef(
            "action",
            "TEXT",
            'Free-pose JSON: {"position_normal": [x, y], "angle": float}',
        ),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar; 0)"),
        PortDef("terminated", "BOOL", "MDP terminal: env-side error / bad action"),
        PortDef("truncated", "BOOL", "True once step_index reaches the num_step budget"),
        PortDef(
            "info",
            "ANY",
            "Diagnostics: {step_index, pose, pose_normal, angle, cam_pose_matrix, path_len}",
        ),
        PortDef("step_index", "ANY", "Step counter (1-based after first step)"),
        PortDef("episode_id", "TEXT", "Episode id"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        raw = inputs.get("action", "")
        try:
            action = json.loads(raw) if isinstance(raw, str) else raw
            position_normal = list(action["position_normal"])
            angle = float(action["angle"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as e:
            self._self_log("error", f"bad action: {e!r} raw={raw!r}")
            return {
                "reward": 0.0,
                "terminated": True,
                "truncated": True,  # unified loop-stop signal (wired to iterOut.stop)
                "info": {"error": str(e)},
                "step_index": 0,
                "episode_id": "",
            }

        result = await _run_sync(_get_mgr().step_freepose, position_normal, angle)
        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "reward": 0.0,
                "terminated": True,
                "truncated": True,
                "info": {"error": result["error"]},
                "step_index": 0,
                "episode_id": "",
            }
        self._self_log("step_index", result.get("step_index"))
        self._self_log("path_len", round(result.get("path_len", 0.0), 2))
        info = {
            k: result.get(k)
            for k in (
                "step_index",
                "pose",
                "pose_normal",
                "angle",
                "cam_pose_matrix",
                "floor_height",
                "path_len",
            )
        }
        _si = int(result.get("step_index", 0) or 0)
        _ns = int(result.get("num_step", 0) or 0)
        return {
            "reward": 0.0,
            "terminated": False,
            "truncated": bool(_ns) and _si >= _ns,
            "info": info,
            "step_index": result.get("step_index", 0),
            "episode_id": result.get("episode_id", ""),
        }


class SampleWaypointExpressTool(BaseCanvasNode):
    node_type = "env_express__sample_waypoint"
    display_name = "EXPRESS: Sample Waypoint (random)"
    description = (
        "Random-exploration action source (paper RE baseline): navigable point "
        "within 3 m of the agent + uniform yaw, as step_pose's free-pose JSON"
    )
    category = "environment"
    icon = "Shuffle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef("trigger", "ANY", "Fires a sample when data arrives"),
    ]
    output_ports = [
        PortDef("action", "TEXT", 'Free-pose JSON: {"position_normal": [x, y], "angle": float}'),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run_sync(_get_mgr().sample_waypoint, 3.0)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"action": ""}
        self._self_log("waypoint", result)
        return {"action": json.dumps(result)}


class ObserveEgocentricExpressTool(BaseCanvasNode):
    node_type = "env_express__observe_egocentric"
    display_name = "EXPRESS: Observe (egocentric)"
    description = (
        "Pull current first-person observation: RGB, depth, pose, intrinsics, TSDF-frame extrinsic"
    )
    category = "environment"
    icon = "Eye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB observation"),
        PortDef(
            "depth",
            "ANY",
            "Current depth (ANY = lossless metric depth over HTTP; DEPTH wire normalizes to [0,1])",
        ),
        PortDef("pose", "POSE", "Habitat-frame agent pose"),
        PortDef("intrinsics", "ANY", "3×3 camera intrinsics matrix (episode-constant)"),
        PortDef("cam_pose_matrix", "ANY", "4×4 TSDF-frame camera extrinsic"),
        PortDef("pose_normal", "ANY", "3-vector normal-frame position"),
        PortDef("angle", "ANY", "Agent yaw angle (radians)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        obs = await _run_sync(mgr.current_obs)
        if "error" in obs:
            self._self_log("error", obs["error"])
            return {
                "rgb": None,
                "depth": None,
                "pose": None,
                "intrinsics": None,
                "cam_pose_matrix": None,
                "pose_normal": None,
                "angle": 0.0,
            }
        K = await _run_sync(mgr.get_cam_intrinsics)
        self._self_log("has_rgb", obs.get("rgb") is not None)
        return {
            "rgb": obs.get("rgb"),
            "depth": obs.get("depth"),
            "pose": obs.get("pose"),
            "intrinsics": K,
            "cam_pose_matrix": obs.get("cam_pose_matrix"),
            "pose_normal": obs.get("pose_normal"),
            "angle": obs.get("angle"),
        }


class JudgePromptExpressTool(BaseCanvasNode):
    node_type = "env_express__judge_prompt"
    display_name = "EXPRESS: Judge prompt"
    description = (
        "Assemble the benchmark's gpt-4o-mini judge prompt (system + user) for the "
        "current episode — feed to a vanilla llmCall together with the final frame"
    )
    category = "environment"
    icon = "Scale"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef("pred_answer", "TEXT", "Agent's free-form answer"),
    ]
    output_ports = [
        PortDef("system", "TEXT", "Judge system prompt (evaluation.txt line 2)"),
        PortDef(
            "user",
            "TEXT",
            'Judge user prompt incl. "Question/Answer/Response/Your mark:" block',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        pred = str(inputs.get("pred_answer", "")).strip()
        info = await _run_sync(_get_mgr().current_episode)
        if "error" in info:
            self._self_log("error", info["error"])
            return {"system": "", "user": ""}
        system, user = _load_judge_prompt()
        # main.py:425 — the ex_prompt appended to the vendored user prompt.
        user = user + (
            f"Question: {info.get('question', '')}\n"
            f"Answer: {info.get('answer', '')}\n"
            f"Response: {pred}\n"
            "Your mark: "
        )
        self._self_log("pred", pred[:120])
        return {"system": system, "user": user}


class EvaluateExpressTool(BaseCanvasNode):
    node_type = "env_express__evaluate"
    display_name = "EXPRESS: Evaluate"
    description = (
        "Parse the judge's 'δ, σ' reply and fold in path efficiency + goal distance "
        "→ per-episode C / C* / E_path / d_T"
    )
    category = "environment"
    icon = "CheckCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef("pred_answer", "TEXT", "Agent's free-form answer (logged into metrics)"),
        PortDef("judge_text", "TEXT", "Raw judge reply — 'δ, σ' (grounding first)"),
    ]
    output_ports = [
        PortDef("c", "ANY", "Exploration-aware correctness C = 100·clip(δ·σ,0,5)/5"),
        PortDef("metrics", "METRICS", "{c, c_star, e_path, d_t, d_t_valid, delta, sigma, …}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        pred = str(inputs.get("pred_answer", "")).strip()
        judge_text = str(inputs.get("judge_text", ""))
        marks = _parse_judge_marks(judge_text)
        judge_ok = marks is not None
        delta, sigma = marks if judge_ok else (0.0, 0)

        info = await _run_sync(_get_mgr().current_episode)
        if "error" in info:
            self._self_log("error", info["error"])
            info = {}
        d_t = await _run_sync(_get_mgr().goal_distance)
        d_t_valid = math.isfinite(d_t)

        # evaluation.py:17-21 — per-episode terms of the paper metrics.
        c = 100.0 * min(max(delta * sigma, 0.0), 5.0) / 5.0
        c_star = 100.0 * min(max(float(sigma), 0.0), 5.0) / 5.0
        p_len = float(info.get("path_len", 0.0))
        l_gt = float(info.get("geodesic_distance", 0.0))
        weight = (l_gt / max(p_len, l_gt)) if max(p_len, l_gt) > 0 else 1.0
        e_path = c * weight

        self._self_log("judge_ok", judge_ok)
        self._self_log("delta_sigma", f"{delta}, {sigma}")
        self._self_log("c", round(c, 1))
        return {
            "c": c,
            "metrics": {
                "c": c,
                "c_star": c_star,
                "e_path": e_path,
                "d_t": float(d_t) if d_t_valid else -1.0,
                "d_t_valid": 1.0 if d_t_valid else 0.0,
                "delta": float(delta),
                "sigma": float(sigma),
                "judge_ok": 1.0 if judge_ok else 0.0,
                "path_len": p_len,
                "gt_geodesic": l_gt,
                "steps_taken": int(info.get("step_index", 0) or 0),
                "num_steps": int(info.get("num_step", 0) or 0),
                "scene": info.get("scene", ""),
                "question_type": info.get("type", ""),
                "pred_answer": pred,
            },
        }


# ══════════════════════════════════════════════════════════════════════
# ExpressEnvPanel — canvas panel env panel
# ══════════════════════════════════════════════════════════════════════


class ExpressEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for EXPRESS-Bench.

    Two-field cascade: ``split → episode_index``. Splits slice the flat
    2,044-record JSON by the scene_id prefix (val 409 / train 1,635 /
    all 2,044); the benchmark itself is evaluated on all records.
    """

    name = "env_express"
    display_name = "EXPRESS-Bench"
    fields = [
        EnvPanelField("split", "select", "Split"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions = [
        EnvPanelAction("play", "Play", side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop", "Stop", side_effect="run_stop", enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "split": "val",
            "episode_index": 0,
        }

    def _mgr(self) -> ExpressEnvManager:
        return ExpressEnvManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split": self._state.get("split", "val"),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        if not mgr.initialized:
            return {
                "available": False,
                "split": self._state.get("split", "val"),
                "episode_index": 0,
                "episode_count": 0,
                "splits": _SPLITS,
                "message": (
                    "EXPRESS-Bench not initialized. Load env_express from "
                    "the NodeSet Manager to enable episode control."
                ),
            }
        total = mgr.get_total_episodes()
        current_idx = mgr._current_episode_idx if mgr._current_episode_idx >= 0 else 0
        self._state["split"] = mgr._split
        self._state["episode_index"] = current_idx
        ep_info = mgr.get_episode_info(current_idx)
        if mgr._current_episode_idx == current_idx and getattr(mgr, "_ep_num_step", 0) > 0:
            step_budget = int(mgr._ep_num_step)
        else:
            step_budget = None
        return {
            "available": True,
            "split": mgr._split,
            "episode_index": current_idx,
            "episode_count": total,
            "splits": _SPLITS,
            # Per-episode dynamic budget — int(sqrt(scene_size) * 3), read
            # by the eval-batch resolver after each env panel on_load.
            "step_budget": step_budget,
            "current_episode": ep_info,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = self._mgr()
        if name == "split":
            split = str(value)
            self._state["split"] = split
            self._state["episode_index"] = 0
            if mgr.initialized:
                res = await self._run(mgr.switch_split, split)
                if isinstance(res, dict) and "error" in res:
                    state = await self.on_load()
                    state["error"] = res["error"]
                    return state
        elif name == "episode_index":
            try:
                idx = int(value)
            except (TypeError, ValueError):
                idx = 0
            self._state["episode_index"] = idx
            if mgr.initialized:
                await self._run(mgr.set_episode_by_index, idx)
        else:
            self._state[name] = value

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        mgr = self._mgr()
        if name in ("play", "reset"):
            if not mgr.initialized:
                return {"ok": False, "side_effect": "none", "error": "EXPRESS not initialized"}
            await self._run(mgr.set_episode_by_index, int(self._state["episode_index"]))
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            return {
                "ok": True,
                "side_effect": "signal",
                "signal_name": "episode_reset",
                "signal_payload": self._episode_reset_payload(),
            }
        if name in ("pause", "stop"):
            return {"ok": True, "side_effect": f"run_{name}"}
        return {"ok": False, "side_effect": "none", "error": f"Unknown action '{name}'"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        if field == "split":
            mgr = self._mgr()
            counts = mgr.split_counts() if mgr.initialized else {}
            return [
                {"value": s, "label": f"{s} ({counts.get(s, 0)} questions)"} for s in _SPLITS
            ]
        if field == "episode_index":
            mgr = self._mgr()
            if not mgr.initialized:
                return []
            episodes = await self._run(mgr.list_episodes, 0, 10000)
            return [
                {
                    "value": ep["index"],
                    "label": "{}: {} [{}] — {}".format(
                        ep["index"],
                        ep["scene"],
                        ep.get("type", ""),
                        ep.get("question", "")[:50],
                    ),
                }
                for ep in episodes
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvExpressNodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


class EnvExpressNodeSet(BaseNodeSet):
    """EXPRESS-Bench (exploration-aware open-vocab EQA) as a NodeSet.

    Shares the ``ac-hmeqa`` conda env with env_hmeqa (identical stack:
    pure habitat_sim, no habitat-lab). ``$EXPRESS_PYTHON`` overrides.
    """

    name = "env_express"
    description = "EXPRESS-Bench — exploration-aware open-vocabulary EQA on HM3D"
    server_python = conda_env_python("ac-hmeqa", "EXPRESS_PYTHON")
    # Same NVIDIA driver-570 EGL workaround as env_hmeqa (habitat-sim
    # 0.3.x SIGSEGV at Simulator() construction) — see env_hmeqa for the
    # full story.
    _SHIM_PATH = os.path.join(
        _REPO_ROOT, "scripts", "install", "hmeqa_libs", "nvidia_egl_workaround.so"
    )
    server_env = {"LD_PRELOAD": _SHIM_PATH} if os.path.exists(_SHIM_PATH) else {}
    env_panel = ExpressEnvPanel
    parallelism = "replicated"  # Stateful simulator: per-worker scene + agent pose.
    # Teleport + render is fast, but EXPRESS graphs put an LLM (and the
    # per-step stop-check) in the loop — same sizing rationale as
    # env_hmeqa/env_objnav.
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = ExpressEnvManager.get()

    def get_tools(self) -> list:
        return [
            # gym-like env interface (see docs: nodesets/env/template.html)
            ResetExpressTool(),  # env_express__reset (metadata only)
            StepPoseExpressTool(),  # env_express__step_pose
            SampleWaypointExpressTool(),  # env_express__sample_waypoint (RE baseline)
            ObserveEgocentricExpressTool(),  # env_express__observe_egocentric
            JudgePromptExpressTool(),  # env_express__judge_prompt
            EvaluateExpressTool(),  # env_express__evaluate
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Load the EXPRESS JSON. Simulator opens lazily on first set_episode.

        Accepted kwargs (all optional):
            split ("val" | "train" | "all"), img_height, img_width, hfov,
            camera_height, camera_tilt_deg, max_step_room_size_ratio, seed
        """
        if self._mgr.initialized:
            log.info("EXPRESS already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("EnvExpressNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        counts = self._mgr.split_counts() if self._mgr.initialized else {}
        return {
            "env_name": "express",
            "datasets": ["EXPRESS-Bench"],
            "splits": _SPLITS,
            "episode_counts": counts,
            "metrics": [
                "c",
                "c_star",
                "e_path",
                "d_t",
                "d_t_valid",
                "delta",
                "sigma",
                "judge_ok",
                "path_len",
                "steps_taken",
            ],
            "supports_set_episode": self._mgr.initialized,
            # Episode length is scene-size-dependent (sqrt(scene_size)·3);
            # this is an upper bound for batch-eval timeout budgeting.
            "step_budget": 50,
        }

    async def shutdown(self) -> None:
        self._mgr.shutdown()
