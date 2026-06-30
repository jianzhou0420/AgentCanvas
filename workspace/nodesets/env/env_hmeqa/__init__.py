from __future__ import annotations

"""EnvHMEQANodeSet — HM-EQA environment as a NodeSet.

Wraps Habitat-Sim + HM3D semantic meshes for House-Mesh Embodied Question
Answering (Ren et al. 2024, explore-eqa). Runs in server mode inside the
dedicated ``hmeqa`` conda env (Python 3.9, habitat-sim latest, torch 2.2)
— the VLN-CE ``vlnce`` env cannot host this because of pinned-package
conflicts (habitat-sim 0.1.7, Py3.8, torch 1.9).

Architecture — mirrors the three-layer pattern from ``habitat.py``:

1. ``HMEQAEnvManager`` (singleton engine)
     Holds a single habitat-sim.Simulator instance bound to the
     currently-active scene. The simulator is torn down and rebuilt on
     every episode change — one .glb file per episode. Threading: one
     pinned ThreadPoolExecutor for GL/physics thread affinity.

2. Canvas tool nodes (``BaseCanvasNode`` adapters)
     env_hmeqa__reset           — start an episode; emits first obs + Q
     env_hmeqa__step            — free-pose teleport; emits new obs
     env_hmeqa__episode_info    — Q, choices, GT answer, scene info
     env_hmeqa__cam_intrinsics  — 3×3 camera matrix (episode-constant)
     env_hmeqa__evaluate        — post-hoc success check (GT comparison)

3. ``EnvHMEQANodeSet`` (collection + lifecycle)
     ``get_tools()`` + ``initialize()`` + ``shutdown()`` +
     ``get_eval_metadata()`` + ``env_panel = HMEQAEnvPanel``. The
     ``server_python`` ClassVar defaults to ``$HMEQA_PYTHON`` so the
     framework auto-hosts this file in the hmeqa env subprocess.

Action contract — free-pose teleport (JSON TEXT):
    {"position_normal": [x, y], "angle": float}
The env appends floor_height, converts normal→habitat coords, and
teleports the agent. This mirrors ``run_vlm_exp.py:313-321`` in
explore-eqa. There is NO mid-episode "answer" action — answer emission
is post-hoc on the method side (the planned cross-env action-manifest
contract is deliberately not exercised here).

Observation bundle (emitted by both reset and step):
    rgb              (IMAGE)  — H×W×3 uint8
    depth            (DEPTH)  — H×W float32
    pose             (POSE)   — habitat-frame agent pose (canvas UI)
    cam_pose_matrix  (ANY)    — 4×4 TSDF-frame camera extrinsic
    pose_normal      (ANY)    — 3-vector normal-frame position
    angle            (ANY)    — scalar yaw
    floor_height     (ANY)    — scalar floor z (episode constant)
    question         (TEXT)   — formatted multi-choice question
    episode_id       (TEXT)   — index as string
    done             (BOOL)   — step only; True only on env-side error
                                  (bad action / sim failure). Per-episode
                                  budget exhaustion is enforced by the
                                  framework via the ``step_budget`` field
                                  the env panel publishes from on_load.
    step_index       (ANY)    — current step counter (step only)

Data layout (ADR-platform-005):
    data/hm3d/hmeqa/questions.csv           — HM-EQA Q&A
    data/hm3d/hmeqa/scene_init_poses.csv    — per-(scene, floor) init
    data/hm3d/hmeqa/Open_Sans/              — annotation font (method-side)
    data/hm3d/hm3dsem/{scene}/{scene[6:]}.basis.glb      — meshes
    data/hm3d/hm3dsem/{scene}/{scene[6:]}.basis.navmesh  — navmesh

last updated: 2026-04-24
"""


import asyncio
import concurrent.futures
import csv
import json
import logging
import math
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
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)

log = logging.getLogger("agentcanvas.hmeqa")


# ══════════════════════════════════════════════════════════════════════
# Paths & defaults
# ══════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
_DATA_ROOT = os.environ.get("HMEQA_DATA_ROOT", os.path.join(_REPO_ROOT, "data", "hm3d", "hmeqa"))
_SCENE_ROOT = os.environ.get(
    "HMEQA_SCENE_ROOT", os.path.join(_REPO_ROOT, "data", "hm3d", "hm3dsem")
)

# Camera + agent defaults (mirror explore-eqa cfg/vlm_exp.yaml)
_DEFAULTS = {
    "img_height": 480,
    "img_width": 640,
    "hfov": 120,
    "camera_height": 1.5,  # paper cfg/vlm_exp.yaml line 18
    "camera_tilt_deg": -30.0,
    "tsdf_grid_size": 0.1,
    "init_clearance": 0.5,
    "max_step_room_size_ratio": 3.0,  # int(sqrt(scene_size) * ratio) = num_step
    "black_pixel_ratio": 0.5,  # paper cfg/vlm_exp.yaml line 30 — obs skipped if #black pixels exceeds this
    "seed": 42,
}


# ══════════════════════════════════════════════════════════════════════
# Coordinate helpers (vendored from explore-eqa/src/habitat.py + geom.py)
#
# Kept in this module rather than as a separate import so the server-
# mode subprocess can load this single file via auto_host. Pure-numpy,
# no external deps beyond what's already in the env.
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
    import habitat_sim  # lazy — only works in the hmeqa env subprocess

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
# Dataset loading — questions.csv + scene_init_poses.csv
# ══════════════════════════════════════════════════════════════════════


def _load_questions(path: str) -> list[dict[str, Any]]:
    """Load HM-EQA questions CSV. Returns list of dict rows."""
    if not os.path.isfile(path):
        log.error("HM-EQA questions CSV missing: %s", path)
        return []
    with open(path, newline="") as f:
        rows = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    return rows


def _load_init_poses(path: str) -> dict[str, dict[str, Any]]:
    """Load scene_init_poses CSV. Keyed by ``"{scene}_{floor}"``."""
    out: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(path):
        log.error("HM-EQA init poses CSV missing: %s", path)
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            out[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    return out


def _parse_choices(raw: str) -> list[str]:
    """Questions CSV stores choices as a quoted comma-list string.

    From ``run_vlm_exp.py:78``:
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
    """
    return [c.split("'")[1] for c in raw.split("',")]


def _format_multichoice_question(question: str, choices: list[str]) -> str:
    """LLaMA-style A/B/C/D formatting used by explore-eqa.

    Mirror of ``run_vlm_exp.py:84-88``.
    """
    out = question
    letters = ["A", "B", "C", "D"]
    for token, choice in zip(letters, choices):
        out += "\n" + token + "." + " " + choice
    return out


# ══════════════════════════════════════════════════════════════════════
# HMEQAEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class HMEQAEnvManager:
    """Singleton env manager for HM-EQA.

    Unlike ``HabitatEnvManager`` (VLN-CE), which initializes one dataset
    and iterates over its pre-defined episode list, HM-EQA re-creates
    the simulator per episode — each question references a different
    scene .glb, and scenes don't share a habitat dataset object.

    All simulator calls run on a pinned single-thread executor for
    GL/physics affinity.
    """

    _instance: HMEQAEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hmeqa",
        )

        # Static data (loaded once on initialize)
        self._questions: list[dict[str, Any]] = []
        self._init_poses: dict[str, dict[str, Any]] = {}
        self._config: dict[str, Any] = dict(_DEFAULTS)

        # Episode-scoped state (rebuilt on set_episode)
        self._simulator: Any = None
        self._agent: Any = None
        self._pathfinder: Any = None
        self._current_episode_idx: int = -1
        self._ep_scene: str = ""
        self._ep_floor: str = ""
        self._ep_question: str = ""
        self._ep_choices: list[str] = []
        self._ep_answer: str = ""
        self._ep_init_pts: list[float] = [0, 0, 0]
        self._ep_init_angle: float = 0.0
        self._ep_floor_height: float = 0.0
        self._ep_tsdf_bnds: np.ndarray | None = None
        self._ep_num_step: int = 0

        # Runtime pose (mutates per step)
        self._pts: np.ndarray = np.zeros(3)
        self._angle: float = 0.0
        self._step_index: int = 0

    # ── Singleton + lifecycle ──

    @classmethod
    def get(cls) -> HMEQAEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return bool(self._questions) and bool(self._init_poses)

    def initialize(self, **kwargs: Any) -> None:
        """Load static CSVs. Does NOT open a scene — that happens on set_episode."""
        with self._lock:
            self._config.update({k: v for k, v in kwargs.items() if k in _DEFAULTS})
            q_path = os.path.join(_DATA_ROOT, "questions.csv")
            p_path = os.path.join(_DATA_ROOT, "scene_init_poses.csv")
            self._questions = _load_questions(q_path)
            self._init_poses = _load_init_poses(p_path)
            log.info(
                "HMEQAEnvManager: loaded %d questions, %d init poses",
                len(self._questions),
                len(self._init_poses),
            )

    def shutdown(self) -> None:
        with self._lock:
            self._close_simulator_unlocked()
            self._questions = []
            self._init_poses = {}

    def _close_simulator_unlocked(self) -> None:
        if self._simulator is not None:
            try:
                self._simulator.close()
            except Exception:
                log.debug("Simulator.close() raised (non-fatal)", exc_info=True)
            self._simulator = None
            self._agent = None
            self._pathfinder = None

    # ── Episode control ──

    def get_total_episodes(self) -> int:
        return len(self._questions)

    def get_episode_info(self, index: int) -> dict[str, Any]:
        """Info for a specific episode, without switching to it.

        Safe on uninitialized env (returns ``{"error": ...}``).
        """
        if not self._questions:
            return {"error": "HM-EQA not initialized"}
        if index < 0 or index >= len(self._questions):
            return {"error": f"index {index} out of range (0, {len(self._questions)})"}
        row = self._questions[index]
        scene = row.get("scene", "")
        floor = row.get("floor", "")
        scene_floor = f"{scene}_{floor}"
        choices = _parse_choices(row.get("choices", ""))
        vlm_q = _format_multichoice_question(row.get("question", ""), choices)
        init = self._init_poses.get(scene_floor, {})
        return {
            "index": index,
            "episode_id": str(index),
            "scene": scene,
            "floor": floor,
            "scene_floor": scene_floor,
            "question": vlm_q,
            "raw_question": row.get("question", ""),
            "choices": choices,
            "answer": row.get("answer", ""),
            "label": row.get("label", ""),
            "init_pts": init.get("init_pts", [0, 0, 0]),
            "init_angle": init.get("init_angle", 0.0),
            "has_init": bool(init),
        }

    def set_episode_by_index(self, index: int) -> dict[str, Any]:
        """Tear down any live simulator, open the scene for episode ``index``."""
        with self._lock:
            if not self._questions:
                return {"error": "HM-EQA not initialized — call initialize() first"}
            if index < 0 or index >= len(self._questions):
                return {"error": f"index {index} out of range"}

            row = self._questions[index]
            scene = row.get("scene", "")
            floor = row.get("floor", "")
            scene_floor = f"{scene}_{floor}"
            init = self._init_poses.get(scene_floor)
            if init is None:
                return {"error": f"init pose missing for {scene_floor}"}

            # Close previous sim, open new one
            self._close_simulator_unlocked()

            # Scene mesh path convention (run_vlm_exp.py:101-105):
            #   {scene_root}/{scene}/{scene[6:]}.basis.glb
            #   {scene_root}/{scene}/{scene[6:]}.basis.navmesh
            # Scene IDs look like "00004-VqCaAuuoeWk" — the first 6 chars
            # are an HM3D prefix we strip from the filename.
            scene_short = scene[6:] if len(scene) > 6 else scene
            mesh_path = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.glb")
            navmesh_path = os.path.join(_SCENE_ROOT, scene, scene_short + ".basis.navmesh")
            if not os.path.isfile(mesh_path):
                return {"error": f"scene mesh missing: {mesh_path}"}

            import habitat_sim  # lazy
            from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

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
                # Without a navmesh, pathfinder.get_bounds() returns a
                # degenerate region → TSDF vol_bnds AssertionError +
                # silent step_count=0 episode failure. Recompute from the
                # scene mesh so the episode runs on a real walkable area.
                log.warning(
                    "HM-EQA: navmesh missing at %s — recomputing from scene mesh",
                    navmesh_path,
                )
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                navmesh_settings.agent_height = float(self._config["camera_height"])
                if not self._simulator.recompute_navmesh(self._pathfinder, navmesh_settings):
                    raise RuntimeError(f"HM-EQA: recompute_navmesh failed for scene {scene}")
            self._agent = self._simulator.initialize_agent(0)

            # Episode state
            init_pts = np.array(init["init_pts"], dtype=np.float64)
            init_angle = float(init["init_angle"])
            pts_normal = _pos_habitat_to_normal(init_pts)
            floor_height = float(pts_normal[-1])
            tsdf_bnds, scene_size = _get_scene_bnds(self._pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * self._config["max_step_room_size_ratio"])

            self._current_episode_idx = index
            self._ep_scene = scene
            self._ep_floor = floor
            self._ep_question = row.get("question", "")
            self._ep_choices = _parse_choices(row.get("choices", ""))
            self._ep_answer = row.get("answer", "")
            self._ep_init_pts = init["init_pts"]
            self._ep_init_angle = init_angle
            self._ep_floor_height = floor_height
            self._ep_tsdf_bnds = tsdf_bnds
            self._ep_num_step = num_step

            self._pts = init_pts
            self._angle = init_angle
            self._step_index = 0

            # Apply initial pose
            self._set_agent_pose_unlocked(init_pts, init_angle)

            log.info(
                "HM-EQA: episode %d scene=%s floor=%s scene_size=%.1f num_step=%d",
                index,
                scene,
                floor,
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
        # Habitat returns RGBA — drop alpha for consumer ports.
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32).squeeze()

        # Camera extrinsic in TSDF frame (what TSDFPlanner.integrate wants)
        sensor = self._agent.get_state().sensor_states["depth_sensor"]
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
        cam_pose[:3, 3] = sensor.position
        cam_pose_tsdf = _pose_normal_to_tsdf(_pose_habitat_to_normal(cam_pose))

        pts_normal = _pos_habitat_to_normal(np.asarray(self._pts, dtype=np.float64))

        # Black-image check (skip obs if agent fell through floor)
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
            "question": self._ep_question,
            "episode_id": str(self._current_episode_idx),
            "is_black": is_black,
            "step_index": int(self._step_index),
            "num_step": int(self._ep_num_step),
        }

    def step_freepose(self, position_normal: list[float], angle: float) -> dict[str, Any]:
        """Teleport the agent to a new pose specified in the normal frame.

        Args:
            position_normal: 2D position in normal frame (x, y). Height
                is taken from the episode's floor_height.
            angle: Yaw angle (radians).
        """
        with self._lock:
            if self._simulator is None:
                return {"error": "no active simulator — call set_episode_by_index first"}

            # Normal-frame 2D + floor_height → 3D → habitat frame
            pts_normal_3d = np.append(
                np.asarray(position_normal, dtype=np.float64), self._ep_floor_height
            )
            pts_habitat = _pos_normal_to_habitat(pts_normal_3d)
            self._pts = pts_habitat
            self._angle = float(angle)
            self._set_agent_pose_unlocked(pts_habitat, self._angle)
            self._step_index += 1
            return self._current_obs_unlocked()

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
            return {
                "index": self._current_episode_idx,
                "episode_id": str(self._current_episode_idx),
                "scene": self._ep_scene,
                "floor": self._ep_floor,
                "scene_floor": f"{self._ep_scene}_{self._ep_floor}",
                "question": self._ep_question,
                "choices": list(self._ep_choices),
                "answer": self._ep_answer,
                "init_pts": list(self._ep_init_pts),
                "init_angle": self._ep_init_angle,
                "floor_height": self._ep_floor_height,
                "num_step": self._ep_num_step,
                "tsdf_bnds": (
                    self._ep_tsdf_bnds.tolist() if self._ep_tsdf_bnds is not None else None
                ),
            }

    def get_cam_intrinsics(self) -> np.ndarray:
        return _get_cam_intr(
            self._config["hfov"], self._config["img_height"], self._config["img_width"]
        )

    def list_episodes(self, start: int = 0, count: int = 10000) -> list[dict[str, Any]]:
        """Shallow metadata for episodes in [start, start+count)."""
        out: list[dict[str, Any]] = []
        for i in range(start, min(start + count, len(self._questions))):
            row = self._questions[i]
            out.append(
                {
                    "index": i,
                    "episode_id": str(i),
                    "scene": row.get("scene", ""),
                    "floor": row.get("floor", ""),
                    "question": row.get("question", "")[:80],
                    "label": row.get("label", ""),
                }
            )
        return out


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> HMEQAEnvManager:
    return HMEQAEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    mgr = _get_mgr()
    return await asyncio.get_running_loop().run_in_executor(mgr.executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


class ResetHMEQATool(BaseCanvasNode):
    node_type = "env_hmeqa__reset"
    display_name = "HM-EQA: Reset"
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
        PortDef("question", "TEXT", "Question text as the VLM sees it (with A/B/C/D tail)"),
        PortDef("raw_question", "TEXT", "Bare question text (no A/B/C/D tail)"),
        PortDef("choices", "ANY", "List of 4 choice strings"),
        PortDef("answer", "TEXT", "Ground-truth letter (A/B/C/D)"),
        PortDef("scene", "TEXT", "HM3D scene id"),
        PortDef("floor", "TEXT", "Floor index"),
        PortDef("episode_id", "TEXT", "Episode index as string"),
        PortDef("num_step", "ANY", "Per-episode step budget (scene-size-dependent)"),
        PortDef("floor_height", "ANY", "Floor z (episode constant)"),
        PortDef("tsdf_bnds", "ANY", "3×2 TSDF voxel-volume bounds (normal frame)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        # Ensure the env panel-selected episode is loaded (placement is
        # env panel-owned); reset only emits metadata — no observation.
        idx = mgr._current_episode_idx if mgr._current_episode_idx >= 0 else 0
        res = await _run_sync(mgr.set_episode_by_index, idx)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", res["error"])
        info = await _run_sync(mgr.current_episode)
        if "error" in info:
            self._self_log("error", info["error"])
            return {
                "question": "",
                "raw_question": "",
                "choices": [],
                "answer": "",
                "scene": "",
                "floor": "",
                "episode_id": "",
                "num_step": 0,
                "floor_height": 0.0,
                "tsdf_bnds": None,
            }
        vlm_q = info.get("question", "")
        raw = vlm_q.split("\nA.")[0] if "\nA." in vlm_q else vlm_q
        self._self_log("episode_id", info.get("episode_id"))
        self._self_log("scene", info.get("scene"))
        self._self_log("question", vlm_q[:200])
        return {
            "question": vlm_q,
            "raw_question": raw,
            "choices": info.get("choices", []),
            "answer": info.get("answer", ""),
            "scene": info.get("scene", ""),
            "floor": info.get("floor", ""),
            "episode_id": info.get("episode_id", ""),
            "num_step": info.get("num_step", 0),
            "floor_height": info.get("floor_height", 0.0),
            "tsdf_bnds": info.get("tsdf_bnds"),
        }


class StepPoseHMEQATool(BaseCanvasNode):
    node_type = "env_hmeqa__step_pose"
    display_name = "HM-EQA: Step (pose teleport)"
    description = "Teleport the agent to a free pose; returns control signals only (pull obs via observe_egocentric)"
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
        # gym-like contract
        PortDef("reward", "ANY", "Per-step reward (scalar; 0)"),
        PortDef("terminated", "BOOL", "MDP terminal: env-side error / bad action"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (env panel step_budget enforces)"),
        PortDef(
            "info", "ANY", "Diagnostics: {step_index, pose, pose_normal, angle, cam_pose_matrix}"
        ),
        # hmeqa-specific extras (also inside info)
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
                "truncated": True,  # unified loop-stop signal (wired to iterOut.stop)
                "info": {"error": result["error"]},
                "step_index": 0,
                "episode_id": "",
            }
        self._self_log("step_index", result.get("step_index"))
        self._self_log("is_black", result.get("is_black"))
        info = {
            k: result.get(k)
            for k in (
                "step_index",
                "pose",
                "pose_normal",
                "angle",
                "cam_pose_matrix",
                "floor_height",
            )
        }
        # Episode horizon: HM-EQA explores for a scene-size-dependent
        # ``num_step`` budget, then answers. truncated=True at that horizon
        # is the gym-idiomatic "stop exploring" signal (the pre-gym env
        # exposed this as ``done``; the migration dropped it). Wired to
        # iterOut.stop so the loop halts instead of oscillating among
        # already-explored frontiers to the static step_budget.
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


class ObserveEgocentricHMEQATool(BaseCanvasNode):
    node_type = "env_hmeqa__observe_egocentric"
    display_name = "HM-EQA: Observe (egocentric)"
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


class EvaluateHMEQATool(BaseCanvasNode):
    node_type = "env_hmeqa__evaluate"
    display_name = "HM-EQA: Evaluate"
    description = "Post-hoc success check — compare method-predicted letter to GT"
    category = "environment"
    icon = "CheckCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    # Post-loop-ness is NOT decided here — it is a per-node graph role set
    # via `config.post_loop: true` in the graph JSON. A graph whose answer
    # is computed post-loop (e.g. explore_eqa_hmeqa's adjudicate chain)
    # marks this evaluate node — and its upstream answer nodes — post_loop;
    # GraphExecutor._post_loop_pass then fires the whole chain once,
    # in dependency order, so `pred_letter` resolves correctly. (The old
    # final_fire ClassVar re-fired a node in isolation and could not do
    # this — retired 2026-05-21.)
    input_ports = [
        PortDef("pred_letter", "TEXT", "Agent's predicted letter (A/B/C/D)"),
    ]
    output_ports = [
        PortDef("success", "BOOL", "pred_letter == ground_truth"),
        PortDef("gt", "TEXT", "Ground-truth letter"),
        PortDef("metrics", "METRICS", "{success, num_steps, scene, floor}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        pred = str(inputs.get("pred_letter", "")).strip().upper()
        info = await _run_sync(_get_mgr().current_episode)
        gt = info.get("answer", "") if "error" not in info else ""
        success = bool(pred and pred == gt)
        self._self_log("pred", pred)
        self._self_log("gt", gt)
        self._self_log("success", success)
        return {
            "success": success,
            "gt": gt,
            "metrics": {
                "success": 1.0 if success else 0.0,
                "num_steps": info.get("num_step", 0) if "error" not in info else 0,
                "scene": info.get("scene", "") if "error" not in info else "",
                "floor": info.get("floor", "") if "error" not in info else "",
            },
        }


# ══════════════════════════════════════════════════════════════════════
# HMEQAEnvPanel — canvas panel env panel
# ══════════════════════════════════════════════════════════════════════


class HMEQAEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for HM-EQA.

    Two-field cascade: ``split → episode_index``. HM-EQA ships only one
    split (``val``) in the questions CSV, so the split selector has a
    single option today — the field is retained for future train/test
    splits without an env panel-schema migration.
    """

    name = "env_hmeqa"
    display_name = "HM-EQA"
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

    def _mgr(self) -> HMEQAEnvManager:
        return HMEQAEnvManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split": self._state.get("split", "val"),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        ctx = getattr(self, "_context", {}) or {}
        mgr = self._mgr()

        # Server mode loads the nodeset in a subprocess — the framework
        # env panel proxy forwards HTTP calls to that subprocess, so
        # the same on_load runs inside the subprocess and sees the real
        # mgr state.

        if not mgr.initialized:
            return {
                "available": False,
                "split": "val",
                "episode_index": 0,
                "episode_count": 0,
                "splits": ["val"],
                "message": (
                    "HM-EQA not initialized. Load env_hmeqa from the "
                    "NodeSet Manager to enable episode control."
                ),
            }
        total = mgr.get_total_episodes()
        current_idx = mgr._current_episode_idx if mgr._current_episode_idx >= 0 else 0
        self._state["episode_index"] = current_idx
        ep_info = mgr.get_episode_info(current_idx)
        # ``num_step`` is computed from ``scene_size``, which requires the
        # scene to be loaded; it is set by ``set_episode_by_index`` (the
        # ``play`` env panel action), not by ``get_episode_info``. Read it
        # straight from the manager state, but only when this idx is the
        # currently-loaded episode — otherwise the resolver should fall
        # through to the graph default.
        if mgr._current_episode_idx == current_idx and getattr(mgr, "_ep_num_step", 0) > 0:
            step_budget = int(mgr._ep_num_step)
        else:
            step_budget = None
        return {
            "available": True,
            "split": self._state.get("split", "val"),
            "episode_index": current_idx,
            "episode_count": total,
            "splits": ["val"],
            # Per-episode dynamic budget — int(sqrt(scene_size) * 3) per
            # Ren et al. 2024 §VI. Read by the framework's eval-batch
            # resolver chain (eval_batch.py:_run_one_episode) after each
            # env panel.on_load(), so the executor's step_budget is set
            # per episode without the env having to wire a "done" boolean.
            "step_budget": step_budget,
            "current_episode": ep_info,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = self._mgr()
        if name == "split":
            self._state["split"] = str(value)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                idx = int(value)
            except (TypeError, ValueError):
                idx = 0
            self._state["episode_index"] = idx
            if mgr.initialized:
                await self._run(mgr.set_episode_by_index, idx)
            state = await self.on_load()
            state["side_effect"] = "signal"
            state["signal_name"] = "episode_reset"
            state["signal_payload"] = self._episode_reset_payload()
            return state
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
                return {"ok": False, "side_effect": "none", "error": "HM-EQA not initialized"}
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
            return [{"value": "val", "label": "val (500 questions)"}]
        if field == "episode_index":
            mgr = self._mgr()
            if not mgr.initialized:
                return []
            episodes = await self._run(mgr.list_episodes, 0, 10000)
            return [
                {
                    "value": ep["index"],
                    "label": "{}: {} floor {} — {}".format(
                        ep["index"],
                        ep["scene"],
                        ep["floor"],
                        ep.get("question", "")[:50],
                    ),
                }
                for ep in episodes
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvHMEQANodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


class EnvHMEQANodeSet(BaseNodeSet):
    """HM-EQA (HM3D question-answering) environment as a NodeSet.

    Loads in server mode against the `hmeqa` conda env by default
    (Python 3.9, habitat-sim latest, Prismatic VLM). The default
    `server_python` reads from `$HMEQA_PYTHON`; for eval/CI this must
    point at the env created by `scripts/install/install_ac_hmeqa.sh`.
    """

    name = "env_hmeqa"
    description = "HM-EQA — HM3D semantic scenes + explore-eqa question-answering"
    server_python = conda_env_python("ac-hmeqa", "HMEQA_PYTHON")
    # NVIDIA driver-570 workaround. habitat-sim 0.3.x SIGSEGVs at Simulator()
    # construction because driver 570 returns a bogus pointer from
    # glGetString(GL_VENDOR), which Magnum's WindowlessEglApplication.cpp:492
    # strlen's into invalid memory. The shim built by install_ac_hmeqa.sh
    # forges clearly-invalid returns to NULL so Magnum's NULL fast-path
    # kicks in. The conda env's activate.d hook only fires for interactive
    # `conda activate ac-hmeqa`, but auto_host spawns the worker via Popen
    # with the Python binary directly — so we have to inject LD_PRELOAD
    # into the subprocess env explicitly here.
    _SHIM_PATH = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        ),
        "scripts",
        "install",
        "hmeqa_libs",
        "nvidia_egl_workaround.so",
    )
    server_env = {"LD_PRELOAD": _SHIM_PATH} if os.path.exists(_SHIM_PATH) else {}
    env_panel = HMEQAEnvPanel
    parallelism = "replicated"  # Stateful simulator: per-worker scene + agent pose.
    # Per-step budget — teleport + 2× sensor render is fast (<0.5s), but
    # ExploreEQA's per-step graph also issues 6× Prismatic VLM token-score
    # calls against a shared-singleton VLM. Under high worker_count those
    # serialize on the GPU; 30s/step absorbs the contention so the
    # episode wall-clock doesn't burn before num_step is reached.
    default_per_step_budget_sec = 30.0
    # v1 replay smooth-mode: hmeqa_replay.py declares supports_smooth_mode()
    # and lazy-spawns hmeqa_renderer.py in the same hmeqa env on first
    # smooth frame request.
    replay_parser = "hmeqa_replay.py"

    def __init__(self) -> None:
        super().__init__()
        self._mgr = HMEQAEnvManager.get()

    def get_tools(self) -> list:
        return [
            # gym-like env interface (see docs: nodesets/env/template.html)
            ResetHMEQATool(),  # env_hmeqa__reset (metadata only)
            StepPoseHMEQATool(),  # env_hmeqa__step_pose
            ObserveEgocentricHMEQATool(),  # env_hmeqa__observe_egocentric
            EvaluateHMEQATool(),  # env_hmeqa__evaluate
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Load HM-EQA CSVs. Simulator opens lazily on first set_episode.

        Accepted kwargs (all optional):
            img_height, img_width, hfov, camera_height, camera_tilt_deg,
            max_step_room_size_ratio, seed
        """
        if self._mgr.initialized:
            log.info("HM-EQA already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("EnvHMEQANodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        count = self._mgr.get_total_episodes() if self._mgr.initialized else 0
        return {
            "env_name": "hmeqa",
            "datasets": ["HM-EQA"],
            "splits": ["val"],
            "episode_counts": {"val": count},
            "metrics": ["success", "num_steps"],
            "supports_set_episode": self._mgr.initialized,
            # HM-EQA episode length is scene-size-dependent — this is an
            # upper bound for batch-eval timeout budgeting. Per-episode
            # value is published per call to the env panel's on_load.
            "step_budget": 30,
        }

    async def shutdown(self) -> None:
        self._mgr.shutdown()
