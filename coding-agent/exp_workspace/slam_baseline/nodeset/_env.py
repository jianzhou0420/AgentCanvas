"""Minimal standalone habitat-sim 0.3.3 wrapper for the SLAM-instrumented line.

Runs in the ``ac-habitat033`` conda env. Ported 2026-08-17 from the
slam-frontier probe worktree (slam_frontier/env.py, trimmed to the R2R-CE
loader this nodeset uses); the FBE baseline scaffolding stayed behind.

Sensors: square RGB + depth (same resolution, so ORB-SLAM3 RGBD gets an
aligned pair), hfov 90, camera at 1.25 m. Depth is METRIC (metres), not
normalized. Discrete actions: 1=FORWARD, 2=LEFT, 3=RIGHT (magnitudes
configurable — SlamEnv drives micro-quanta). 0=STOP is handled by the caller.

Data-root overrides deliberately SHARE env_vlnce033's variable names
(VLNCE033_DATA_ROOT / VLNCE033_SCENE_ROOT): both nodesets read the same
R2R_VLNCE_v1-3 corpus, so one repoint serves both.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

import numpy as np

# package lives at <repo>/coding-agent/exp_workspace/<exp>/nodeset
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", "..", ".."))
DATASET_ROOT = os.environ.get(
    "VLNCE033_DATA_ROOT",
    os.path.join(REPO_ROOT, "data", "habitat", "datasets", "R2R_VLNCE_v1-3_preprocessed"),
)
SCENE_ROOT = os.environ.get(
    "VLNCE033_SCENE_ROOT", os.path.join(REPO_ROOT, "data", "scene_datasets")
)

ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def camera_intrinsics(hfov_deg: float, size: int) -> dict:
    hfov = np.deg2rad(hfov_deg)
    f = (size / 2.0) / np.tan(hfov / 2.0)
    return {"fx": float(f), "fy": float(f), "cx": size / 2.0, "cy": size / 2.0,
            "width": size, "height": size}


def list_splits() -> list:
    """Splits = subdirs of the dataset root that carry {split}.json.gz."""
    if not os.path.isdir(DATASET_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(DATASET_ROOT)):
        if os.path.isfile(os.path.join(DATASET_ROOT, name, f"{name}.json.gz")):
            out.append(name)
    return out


def load_episodes(split: str) -> list:
    path = os.path.join(DATASET_ROOT, split, f"{split}.json.gz")
    if not os.path.isfile(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["episodes"]


def load_gt_locations(split: str) -> dict:
    """episode_id -> gt locations (for nDTW). Empty dict when unavailable."""
    path = os.path.join(DATASET_ROOT, split, f"{split}_gt.json.gz")
    if not os.path.isfile(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        gt = json.load(f)
    return {str(k): v.get("locations", []) for k, v in gt.items()}


class HabitatEnv:
    """One scene, discrete actions, metric RGB-D + ground-truth pose."""

    def __init__(
        self,
        scene_path: str,
        img_size: int = 256,
        hfov_deg: float = 90.0,
        camera_height: float = 1.25,
        step_size_m: float = 0.25,
        turn_angle_deg: float = 15.0,
        seed: int = 42,
    ) -> None:
        import habitat_sim

        self.img_size = img_size
        self.hfov_deg = hfov_deg
        self.camera_height = camera_height
        self.step_size_m = step_size_m
        self.turn_angle_deg = turn_angle_deg

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.allow_sliding = True
        sim_cfg.random_seed = seed

        specs = []
        for uuid, stype in (
            ("color_sensor", habitat_sim.SensorType.COLOR),
            ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = stype
            spec.resolution = [img_size, img_size]
            spec.position = [0.0, camera_height, 0.0]
            spec.hfov = hfov_deg
            specs.append(spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.1
        agent_cfg.sensor_specifications = specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=step_size_m)),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle_deg)),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle_deg)),
        }

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        navmesh = os.path.splitext(scene_path)[0] + ".navmesh"
        if os.path.isfile(navmesh):
            self.sim.pathfinder.load_nav_mesh(navmesh)
        else:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = 1.5
            settings.agent_radius = 0.1
            if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                raise RuntimeError(f"recompute_navmesh failed for {scene_path}")
        self.sim.seed(seed)
        self._habitat_sim = habitat_sim
        self.collisions = 0
        self.path: list = []

    # ── episode control ──

    def reset(self, position: list | None = None,
              rotation_xyzw: list | None = None) -> None:
        import quaternion  # noqa: F401 — enables np.quaternion

        agent = self.sim.get_agent(0)
        state = self._habitat_sim.AgentState()
        if position is None:
            position = self.sim.pathfinder.get_random_navigable_point()
        state.position = np.asarray(position, dtype=np.float32)
        if rotation_xyzw is not None:
            r = rotation_xyzw
            state.rotation = np.quaternion(r[3], r[0], r[1], r[2])
        agent.set_state(state)
        self.collisions = 0
        self.path = [list(map(float, agent.get_state().position))]

    def step(self, action: int) -> bool:
        """Apply a discrete action; returns True if a forward move collided."""
        name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}[int(action)]
        prev = np.asarray(self.sim.get_agent(0).get_state().position)
        self.sim.step(name)
        pos = np.asarray(self.sim.get_agent(0).get_state().position)
        collided = name == "move_forward" and float(np.linalg.norm(pos - prev)) < 1e-4
        if collided:
            self.collisions += 1
        self.path.append(list(map(float, pos)))
        return collided

    # ── perception ──

    def observe(self) -> dict:
        obs = self.sim.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth_m = np.asarray(obs["depth_sensor"], dtype=np.float32).squeeze()
        return {"rgb": rgb, "depth": depth_m}

    def agent_position(self) -> list:
        return list(map(float, self.sim.get_agent(0).get_state().position))

    def gt_camera_pose_cv(self) -> np.ndarray:
        """Ground-truth 4x4 world-from-camera pose in OpenCV convention.

        habitat sensors use the OpenGL convention (camera looks -z, y up);
        right-multiplying by diag(1,-1,-1,1) converts to OpenCV
        (looks +z, y down) so GT poses drop into the same mapping code as
        ORB-SLAM3 output.
        """
        state = self.sim.get_agent(0).get_state()
        node = state.sensor_states["color_sensor"]
        q = node.rotation  # np.quaternion (w, x, y, z)
        import quaternion

        rot = quaternion.as_rotation_matrix(q)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = np.asarray(node.position, dtype=np.float64)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return T @ flip

    def intrinsics(self) -> dict:
        return camera_intrinsics(self.hfov_deg, self.img_size)

    def geodesic_distance(self, a: Any, b: Any) -> float:
        """Geodesic between two points; euclidean fallback when no path is
        found (the probe's own scoring rule, same as env_vlnce033)."""
        sp = self._habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, dtype=np.float32)
        sp.requested_end = np.asarray(b, dtype=np.float32)
        if self.sim.pathfinder.find_path(sp):
            return float(sp.geodesic_distance)
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64)
                                    - np.asarray(b, dtype=np.float64)))

    def navigable_area(self) -> float:
        return float(self.sim.pathfinder.navigable_area)

    def close(self) -> None:
        self.sim.close()
