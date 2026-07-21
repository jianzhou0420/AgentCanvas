from __future__ import annotations

"""EnvObjnavNodeSet — HM3D ObjectNav (habitat-lab 0.2.4) as a NodeSet.

Gym-like interface (docs: nodesets/env/template.html): ``reset`` (metadata
only) / ``step_discrete`` (control signals) / ``observe_egocentric`` (pull) /
``evaluate`` (metric sink).

Works in-process or as an auto-hosted server:
  Local:  POST /api/components/nodesets/env_objnav/load
  Server: POST /api/components/nodesets/env_objnav/load?mode=server

Task: Habitat ObjectNav on HM3D-Semantics — given a goal category (one of
chair / bed / plant / toilet / tv_monitor / sofa), navigate to within 1 m of
any instance and STOP. Benchmark configuration follows the de-facto paper
standard (VLFM / SG-Nav / MVP-Nav lineage, NOT Challenge-2023 Stretch):
habitat 0.2.4, classic agent (0.25 m forward / 30° turn / 640×480 RGB-D /
500-step budget / 1.0 m success), episodes ``objectnav_hm3d_v1`` (2000 val
episodes over 20 scenes; ``v2`` selectable for the OpenFrontier lineage).

Data layout (repo-root-relative, staged 2026-07-20):
  scenes:   data/scene_datasets/hm3d/val/            (HM3D v0.2 val, official)
  episodes: data/datasets/objectnav/hm3d/{v1,v2}/    (train / val / val_mini)

  v1 and v2 disagree on the scene path baked into every episode's ``scene_id``:
  v1 says ``hm3d/val/...`` while v2 says ``hm3d_v0.2/val/...`` — and v2's
  val_mini points at ``hm3d_v0.2/minival/...`` (its 2 scenes, 00800 + 00802,
  are a subset of our val download). Both v2 prefixes are therefore served by
  compat symlinks into the one real scene tree:
      data/scene_datasets/hm3d_v0.2/val     -> ../hm3d/val
      data/scene_datasets/hm3d_v0.2/minival -> ../hm3d/val
  Without them habitat-sim aborts with "Missing (at least) one of scene
  dataset attributes ... Likely an invalid scene name".

Discrete action space (habitat 0.2.4 ObjectNav defaults):
  0 = STOP · 1 = MOVE_FORWARD (0.25 m) · 2 = TURN_LEFT (30°) ·
  3 = TURN_RIGHT (30°) · 4 = LOOK_UP (30°) · 5 = LOOK_DOWN (30°)

Success judgment uses episode-embedded goal viewpoints (verified 2026-07-20:
oracle ShortestPathFollower SR 1.00 without semantic annotations; the
``.semantic.glb`` files staged alongside are only needed by methods/measures
that read the semantic scene, e.g. habitat's top_down_map).
"""

import asyncio
import base64
import concurrent.futures
import io
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
from app.components.env_panel import BaseEnvPanel, EnvPanelAction, EnvPanelField

log = logging.getLogger("agentcanvas.env_objnav")


# ══════════════════════════════════════════════════════════════════════
# Dataset wiring — objectnav_hm3d v1 (paper standard) / v2 (challenge 2023)
# ══════════════════════════════════════════════════════════════════════

_DATASETS: list[str] = ["v1", "v2"]
_SPLITS: list[str] = ["val", "val_mini"]  # train scenes not staged locally

_BENCH_CONFIG = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
_DATA_PATH_TMPL = "data/datasets/objectnav/hm3d/{version}/{{split}}/{{split}}.json.gz"


def _repo_root() -> str:
    # __file__ lives at workspace/nodesets/env/env_objnav.py — three parents.
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
    )


# ══════════════════════════════════════════════════════════════════════
# ObjnavEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class ObjnavEnvManager:
    """Single (non-vectorized) habitat-lab 0.2.4 ObjectNav environment.

    All public methods are blocking — call via
    ``asyncio.get_running_loop().run_in_executor(mgr.executor, fn)``.
    Single-thread executor enforces GL thread affinity.
    """

    _instance: ObjnavEnvManager | None = None

    def __init__(self) -> None:
        self._env: Any = None
        self._config: Any = None
        self._current_obs: dict | None = None
        self._episode_done: bool = False
        self._step_count: int = 0
        self._last_action: int | None = None
        self._lock = threading.Lock()
        self._dataset: str = "v1"
        self._split: str = "val"
        self._episode_index: int = 0
        self._gpu_id: int = 0
        self._max_steps: int = 500
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="objnav",
        )

    @classmethod
    def get(cls) -> ObjnavEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._env is not None

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ── Lifecycle ──

    def initialize(
        self,
        dataset: str = "v1",
        split: str = "val",
        gpu_id: int = 0,
        max_steps: int = 500,
    ) -> dict:
        """Build habitat config, create env, arm the first episode."""
        with self._lock:
            if self._env is not None:
                log.warning("ObjnavEnvManager already initialized — skipping")
                return self._get_episode_info_unlocked()
            return self._build_env_unlocked(dataset, split, gpu_id, max_steps)

    def _build_env_unlocked(
        self,
        dataset: str,
        split: str,
        gpu_id: int,
        max_steps: int,
    ) -> dict:
        # Relative data paths in the habitat YAML resolve from CWD.
        repo_root = _repo_root()
        if os.path.isdir(os.path.join(repo_root, "data")):
            os.chdir(repo_root)

        import habitat
        from habitat.config import read_write
        from habitat.config.default import get_config

        if dataset not in _DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r} (expected {_DATASETS})")
        if split not in _SPLITS:
            raise ValueError(f"Unknown split {split!r} (expected {_SPLITS})")

        config = get_config(_BENCH_CONFIG)
        with read_write(config):
            config.habitat.dataset.split = split
            config.habitat.dataset.data_path = _DATA_PATH_TMPL.format(version=dataset)
            config.habitat.environment.max_episode_steps = int(max_steps)
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)

        log.info(
            "Creating ObjectNav env — dataset=%s split=%s gpu=%d max_steps=%d",
            dataset, split, gpu_id, max_steps,
        )
        self._env = habitat.Env(config=config)
        self._config = config
        self._dataset, self._split = dataset, split
        self._gpu_id, self._max_steps = int(gpu_id), int(max_steps)

        self._current_obs = self._env.reset()
        self._episode_done = False
        self._step_count = 0
        self._last_action = None
        self._episode_index = self._current_episode_index_unlocked()

        info = self._get_episode_info_unlocked()
        log.info(
            "ObjectNav env ready — episode=%s scene=%s goal=%s",
            info.get("episode_id"), info.get("scene_id"), info.get("object_category"),
        )
        return info

    def shutdown(self) -> None:
        with self._lock:
            if self._env is not None:
                log.info("Shutting down ObjectNav env")
                self._env.close()
                self._env = None

    def switch_dataset_split(self, dataset: str, split: str) -> dict:
        """Rebuild the env for a new (dataset, split) selection."""
        with self._lock:
            if self._env is not None:
                self._env.close()
                self._env = None
            return self._build_env_unlocked(dataset, split, self._gpu_id, self._max_steps)

    # ── Episode control (env panel + reset) ──

    def list_splits(self) -> list[str]:
        return list(_SPLITS)

    def get_total_episodes(self) -> int:
        with self._lock:
            if self._env is None:
                return 0
            return len(self._env.episodes)

    def get_episodes_list(self, offset: int = 0, limit: int = 50) -> dict:
        with self._lock:
            if self._env is None:
                return {"episodes": [], "total": 0}
            eps = self._env.episodes
            out = []
            for i, ep in enumerate(eps[offset : offset + limit]):
                out.append(
                    {
                        "index": offset + i,
                        "episode_id": str(ep.episode_id),
                        "scene_id": ep.scene_id,
                        "object_category": getattr(ep, "object_category", ""),
                    },
                )
            return {"episodes": out, "total": len(eps)}

    def _current_episode_index_unlocked(self) -> int:
        try:
            cur_id = str(self._env.current_episode.episode_id)
            cur_scene = self._env.current_episode.scene_id
            for i, ep in enumerate(self._env.episodes):
                if str(ep.episode_id) == cur_id and ep.scene_id == cur_scene:
                    return i
        except Exception:
            pass
        return -1

    def set_episode_by_index(self, index: int) -> dict:
        """Place + arm episode ``index``; resets counters."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            eps = self._env.episodes
            if index < 0 or index >= len(eps):
                return {"error": f"Index {index} out of range (0-{len(eps) - 1})"}
            # habitat 0.2.4 Env.reset() pulls the next episode from
            # episode_iterator, reconfigures the sim if the scene changed and
            # places the agent at the episode start (the 0.1.7 same-scene
            # stale-pose bug is fixed in this generation).
            self._env.episode_iterator = iter([eps[index]])
            self._current_obs = self._env.reset()
            self._episode_done = False
            self._step_count = 0
            self._last_action = None
            self._episode_index = index
            return self._get_episode_info_unlocked()

    def ensure_live(self) -> dict:
        """Template §5.1 reset semantics: a live episode is read untouched;
        a done one is re-armed at the SAME placement. Never chooses."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if not self._episode_done:
                return self._get_episode_info_unlocked()
        # Done → re-arm in place (re-acquires the lock inside).
        return self.set_episode_by_index(max(self._episode_index, 0))

    # ── Transition ──

    def step(self, action: int) -> dict:
        """Advance one tick. Control signals only — no observation."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if self._episode_done:
                return {"error": "Episode already done", "terminated": True}

            action = int(action)
            self._current_obs = self._env.step(action)
            self._step_count += 1
            self._last_action = action
            self._episode_done = bool(self._env.episode_over)

            terminated, truncated = self._split_done_unlocked()
            result: dict[str, Any] = {
                "reward": 0.0,  # habitat.Env is reward-free; eval task
                "terminated": terminated,
                "truncated": truncated,
            }
            info: dict[str, Any] = {
                "step_count": self._step_count,
                "action": action,
            }
            state = self._get_agent_state_unlocked()
            if "error" not in state:
                info.update(state)
            if self._episode_done:
                info["metrics"] = self._metrics_unlocked()
            result["info"] = info
            return result

    def _split_done_unlocked(self) -> tuple[bool, bool]:
        """Split habitat's single episode_over into gym (terminated, truncated).

        STOP (action 0) → terminated; budget exhaustion without STOP →
        truncated. habitat 0.2.4 flips episode_over for both.
        """
        if not self._episode_done:
            return False, False
        budget_hit = bool(self._max_steps and self._step_count >= self._max_steps)
        stopped = self._last_action == 0
        truncated = budget_hit and not stopped
        return (not truncated), truncated

    def navigate_to(self, target: list, goal_radius: float = 0.36,
                    max_nav_steps: int = 50) -> dict:
        """Walk toward a world-frame target via ShortestPathFollower.

        Mirrors env_habitat step_pose: real discrete primitives (each one
        advances the task and counts toward SPL / step budget), stopping at
        ``goal_radius``, ``max_nav_steps``, episode end, or path exhaustion.
        Never dispatches STOP — committing is the caller's decision.
        """
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if self._episode_done:
                return {"error": "Episode already done", "terminated": True}

            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            sim = self._env.sim
            follower = ShortestPathFollower(
                sim, goal_radius=float(goal_radius), return_one_hot=False,
            )
            goal = np.array(target, dtype=np.float32)
            steps = 0
            while steps < int(max_nav_steps) and not self._episode_done:
                try:
                    action = follower.get_next_action(goal)
                except Exception as e:  # noqa: BLE001
                    log.warning("navigate_to: follower failed: %s", e)
                    break
                if action is None or int(action) == 0:
                    break  # follower says goal reached — do NOT dispatch STOP
                self._current_obs = self._env.step(int(action))
                self._step_count += 1
                self._last_action = int(action)
                self._episode_done = bool(self._env.episode_over)
                steps += 1
                pos = np.asarray(sim.get_agent_state().position, dtype=np.float32)
                if float(np.linalg.norm(pos - goal)) < float(goal_radius):
                    break

            terminated, truncated = self._split_done_unlocked()
            state = self._get_agent_state_unlocked()
            dist = None
            if "error" not in state:
                dist = float(np.linalg.norm(
                    np.asarray(state["position"], dtype=np.float32) - goal))
            info: dict[str, Any] = {
                "step_count": self._step_count,
                "nav_steps": steps,
                "distance_to_target": dist,
            }
            if "error" not in state:
                info.update(state)
            if self._episode_done:
                info["metrics"] = self._metrics_unlocked()
            return {
                "reward": 0.0,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }

    # ── Perception (pull) ──

    def render_panorama_rgbd(self, n_views: int = 12) -> dict:
        """Aligned RGB+Depth views at n_views headings from the current pose.

        Read-only: renders via ``sim.get_observations_at`` with
        ``keep_agent_at_new_pose=False`` — the agent does not move and the
        task takes no step.
        """
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            sim = self._env.sim
            state = sim.get_agent_state()
            pos, rot = state.position, state.rotation
            angle_step = 2.0 * np.pi / n_views
            views = []
            for i in range(n_views):
                yaw = i * angle_step
                yaw_quat = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
                new_rot = rot * yaw_quat
                obs = sim.get_observations_at(
                    pos,
                    [float(new_rot.x), float(new_rot.y), float(new_rot.z), float(new_rot.w)],
                    keep_agent_at_new_pose=False,
                )
                if obs is None:
                    continue
                view: dict[str, Any] = {
                    "dir_id": i,
                    "heading_deg": round(float(np.degrees(yaw)) % 360, 1),
                }
                if "rgb" in obs:
                    view["rgb_base64"] = self._encode_rgb_base64(
                        np.asarray(obs["rgb"], dtype=np.uint8))
                if "depth" in obs:
                    depth = np.asarray(obs["depth"], dtype=np.float32).squeeze()
                    view["depth_base64"] = self._encode_depth_base64(depth)
                    view["depth_raw_base64"] = self._encode_depth_raw_base64(depth)
                views.append(view)
            return {"views": views, "n_views": len(views)}

    def render_panorama_composite(self, n_views: int = 12) -> dict:
        """Stitched labelled grid of n_views RGB headings (single image)."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            sim = self._env.sim
            state = sim.get_agent_state()
            pos, rot = state.position, state.rotation
            angle_step = 2.0 * np.pi / n_views
            images, meta = [], []
            for i in range(n_views):
                yaw = i * angle_step
                yaw_quat = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
                new_rot = rot * yaw_quat
                obs = sim.get_observations_at(
                    pos,
                    [float(new_rot.x), float(new_rot.y), float(new_rot.z), float(new_rot.w)],
                    keep_agent_at_new_pose=False,
                )
                if obs is None or "rgb" not in obs:
                    continue
                heading = round(float(np.degrees(yaw)) % 360, 1)
                images.append(np.asarray(obs["rgb"], dtype=np.uint8)[:, :, :3])
                meta.append({"dir_id": i, "heading_deg": heading,
                             "direction": f"{int(heading)}°"})
            composite = self._build_composite(images, [m["direction"] for m in meta])
            return {"views": meta, "n_views": len(meta), "composite": composite}

    @staticmethod
    def _encode_rgb_base64(rgb: np.ndarray) -> str:
        from PIL import Image

        img = Image.fromarray(rgb[:, :, :3].astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_depth_base64(depth: np.ndarray) -> str:
        from PIL import Image

        d = np.squeeze(depth)
        d_min, d_max = float(d.min()), float(d.max())
        if d_max - d_min > 1e-6:
            d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(d_norm, mode="L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_depth_raw_base64(depth: np.ndarray) -> str:
        # 16-bit PNG depth in millimetres — preserves absolute metric depth
        # (same convention as env_habitat.encode_depth_raw_base64).
        from PIL import Image

        d = np.squeeze(depth).astype(np.float32)
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(d_mm, mode="I;16").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _build_composite(images: list, labels: list) -> np.ndarray | None:
        from PIL import Image, ImageDraw

        n = len(images)
        if n == 0:
            return None
        h, w = images[0].shape[:2]
        label_h, border = 20, 2
        if n <= 4:
            cols, rows = 2, 2
        elif n <= 8:
            cols, rows = 4, 2
        elif n <= 12:
            cols, rows = 4, 3
        else:
            cols, rows = 6, 4
        cell_w, cell_h = w + border * 2, h + label_h + border * 2
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, (img_arr, label) in enumerate(zip(images, labels)):
            row, col = divmod(i, cols)
            x, y = col * cell_w + border, row * cell_h + border
            draw.text((x + w // 2 - len(label) * 3, y + 2), label, fill=(255, 255, 100))
            canvas.paste(Image.fromarray(img_arr), (x, y + label_h))
        return np.asarray(canvas, dtype=np.uint8)

    def observe(self) -> dict:
        """Idempotent read of the current frame — never advances the env."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            obs = self._current_obs or {}
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            result: dict[str, Any] = {
                "rgb": np.asarray(rgb, dtype=np.uint8) if rgb is not None else None,
                "depth": (
                    np.asarray(depth, dtype=np.float32).squeeze()
                    if depth is not None
                    else None
                ),
                "pose": self._get_agent_state_unlocked(),
                "intrinsics": self._cam_intrinsics_unlocked(),
                # ObjectNav task sensors (VLFM-style methods consume these).
                "gps": np.asarray(obs["gps"]).tolist() if "gps" in obs else None,
                "compass": float(np.asarray(obs["compass"]).reshape(-1)[0])
                if "compass" in obs
                else None,
            }
            return result

    def _get_agent_state_unlocked(self) -> dict:
        if self._env is None:
            return {"error": "Environment not initialized"}
        try:
            state = self._env.sim.get_agent_state()
            rot = state.rotation
            return {
                "position": np.asarray(state.position, dtype=float).tolist(),
                "orientation": [float(rot.x), float(rot.y), float(rot.z), float(rot.w)],
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"get_agent_state failed: {e}"}

    def _cam_intrinsics_unlocked(self) -> dict | None:
        if self._config is None:
            return None
        try:
            sensor = (
                self._config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )
            w, h = int(sensor.width), int(sensor.height)
            f = (w / 2.0) / float(np.tan(np.radians(float(sensor.hfov)) / 2.0))
            return {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}
        except Exception:
            return None

    # ── Metric sink ──

    def evaluate(self) -> dict:
        """Pull current task metrics on demand — independent of done."""
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            return self._metrics_unlocked()

    def _metrics_unlocked(self) -> dict:
        try:
            info = self._env.get_metrics()
        except Exception as e:  # noqa: BLE001
            return {"error": f"get_metrics failed: {e}"}
        return {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in info.items()
            if isinstance(v, (int, float, np.floating, str, bool))
        }

    # ── Queries ──

    def get_episode_info(self) -> dict:
        with self._lock:
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        if self._env is None:
            return {"error": "Environment not initialized"}
        ep = self._env.current_episode
        return {
            "episode_id": str(ep.episode_id),
            "scene_id": ep.scene_id,
            "object_category": getattr(ep, "object_category", ""),
            "episode_index": self._episode_index,
            "dataset": self._dataset,
            "split": self._split,
            "step_count": self._step_count,
            "done": self._episode_done,
            "max_steps": self._max_steps,
        }


def _mgr() -> ObjnavEnvManager:
    return ObjnavEnvManager.get()


async def _run(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Nodes — gym verbs (template.html)
# ══════════════════════════════════════════════════════════════════════


class ResetObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__reset"
    display_name = "ObjNav: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Ensure a live episode (re-arm if done) — goal category + ids, no observation"
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger reset (any value)", optional=True),
    ]
    output_ports = [
        PortDef("object_category", "TEXT", "Goal category (chair/bed/plant/toilet/tv_monitor/sofa)"),
        PortDef("episode_id", "TEXT", "Episode ID"),
        PortDef("scene_id", "TEXT", "Scene ID"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Episode placement is env-panel-owned; reset only ensures liveness.
        meta = await _run(_mgr().ensure_live)
        self._self_log("episode_id", meta.get("episode_id"))
        self._self_log("object_category", meta.get("object_category"))
        return {
            "object_category": str(meta.get("object_category", "")),
            "episode_id": str(meta.get("episode_id", "")),
            "scene_id": str(meta.get("scene_id", "")),
        }


class StepDiscreteObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__step_discrete"
    display_name = "ObjNav: Step (discrete)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Advance one tick (0=STOP 1=FWD 2=LEFT 3=RIGHT 4=LOOK_UP 5=LOOK_DOWN)"
    category = "environment"
    icon = "Play"
    input_ports = [
        PortDef("action", "ACTION", "Discrete action (0-5)"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (0.0 — eval task is reward-free)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called"),
        PortDef("truncated", "BOOL", "Step-budget cutoff without STOP"),
        PortDef("info", "ANY", "Per-step diagnostics + terminal metrics"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        action = int(inputs.get("action", 1))
        result = await _run(_mgr().step, action)
        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "reward": 0.0,
                "terminated": bool(result.get("terminated", False)),
                "truncated": False,
                "info": {"error": result["error"]},
            }
        self._self_log("terminated", result["terminated"])
        self._self_log("step_count", result["info"].get("step_count"))
        return result


class StepPoseObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__step_pose"
    display_name = "ObjNav: Step (pose)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Navigate toward a target pose via shortest-path following (real steps, no STOP)"
    category = "environment"
    icon = "Navigation"
    input_ports = [
        PortDef("target", "POSE", "Target position [x, y, z] or pose dict"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (0.0 — eval task is reward-free)"),
        PortDef("terminated", "BOOL", "MDP terminal: episode ended during the walk"),
        PortDef("truncated", "BOOL", "Step-budget cutoff during the walk"),
        PortDef("info", "ANY", "Diagnostics: nav_steps, distance_to_target, final pose (+ terminal metrics)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Frontier-style workhorse: the method picks a point, the env walks
        # there with discrete primitives that all count toward SPL/budget.
        # STOP is never dispatched — committing is the reasoning side's call.
        target_pose = inputs.get("target", {})
        target = target_pose.get("position") if isinstance(target_pose, dict) else target_pose
        if target is None:
            return {"reward": 0.0, "terminated": False, "truncated": False, "info": {}}
        result = await _run(_mgr().navigate_to, [float(x) for x in target])
        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "reward": 0.0,
                "terminated": bool(result.get("terminated", False)),
                "truncated": False,
                "info": {"error": result["error"]},
            }
        self._self_log("nav_steps", result["info"].get("nav_steps"))
        self._self_log("distance_to_target", result["info"].get("distance_to_target"))
        return result


class ObserveEgocentricObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__observe_egocentric"
    display_name = "ObjNav: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull the current first-person frame: RGB, depth, pose, intrinsics, gps, compass"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB (640×480)"),
        PortDef("depth", "DEPTH", "Current depth map (metres)"),
        PortDef("pose", "POSE", "Agent position [x,y,z] + orientation quaternion [x,y,z,w]"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height} or None"),
        PortDef("gps", "ANY", "ObjectNav GPS sensor [x,y] relative to start (None if absent)"),
        PortDef("compass", "ANY", "ObjectNav compass (rad, relative to start; None if absent)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Pure read — a finished episode observed again returns the terminal
        # frame; re-arming is reset's job (never auto-reset here).
        result = await _run(_mgr().observe)
        if "error" in result:
            self._self_log("error", result["error"])
            return {
                "rgb": None, "depth": None, "pose": None,
                "intrinsics": None, "gps": None, "compass": None,
            }
        self._self_log("has_rgb", result.get("rgb") is not None)
        return result


class ObserveCameraPoseObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__observe_camera_pose"
    display_name = "ObjNav: Camera Pose"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "World-frame agent position [x,y,z] + orientation quaternion [x,y,z,w]"
    category = "environment"
    icon = "Crosshair"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger (optional)", optional=True),
    ]
    output_ports = [
        PortDef("position", "ANY", "World-frame position [x, y, z]"),
        PortDef("rotation", "ANY", "World-frame orientation quaternion [x, y, z, w]"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        result = await _run(_mgr().observe)
        pose = result.get("pose") or {}
        pos = pose.get("position", [0.0, 0.0, 0.0])
        ori = pose.get("orientation", [0.0, 0.0, 0.0, 1.0])
        self._self_log("position", pos)
        return {"position": pos, "rotation": ori}


class ObservePanoramaObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__observe_panorama"
    display_name = "ObjNav: Observe (panorama)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField(
                "representation",
                "select",
                label="Representation",
                default="views_rgbd",
                options=[
                    {"value": "views_rgbd", "label": "Aligned RGB-D views"},
                    {"value": "composite", "label": "Stitched composite grid (single image)"},
                ],
            ),
            ConfigField(
                "n_views",
                "slider",
                label="Number of views",
                default=12,
                min=4,
                max=24,
                step=4,
            ),
        ],
    )
    description = "Multi-view panorama at the agent's position — aligned RGB-D views or one stitched image"
    category = "environment"
    icon = "Panorama"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-render (optional)", optional=True),
    ]
    output_ports = [
        PortDef("views", "ANY", "List of {dir_id, heading_deg, rgb_base64, depth_base64} (views_rgbd mode)"),
        PortDef("directions", "TEXT", "Per-view heading labels (JSON)"),
        PortDef("n_views", "ANY", "Number of views returned"),
        PortDef("composite", "IMAGE", "Stitched panorama grid (composite mode)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import json

        cfg = self.config or {}
        representation = cfg.get("representation", "views_rgbd")
        n_views = int(cfg.get("n_views", 12))
        # Pure read — renders off-pose via get_observations_at, never steps.
        if representation == "composite":
            result = await _run(_mgr().render_panorama_composite, n_views)
            if "error" in result:
                self._self_log("error", result["error"])
                return {"views": [], "directions": "[]", "n_views": 0, "composite": None}
            directions = json.dumps(result["views"])
            self._self_log("n_views", result["n_views"])
            return {
                "views": result["views"],
                "directions": directions,
                "n_views": result["n_views"],
                "composite": result["composite"],
            }
        result = await _run(_mgr().render_panorama_rgbd, n_views)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"views": [], "directions": "[]", "n_views": 0, "composite": None}
        views = result.get("views", [])
        directions = json.dumps(
            [{"dir_id": v["dir_id"], "heading_deg": v["heading_deg"]} for v in views],
        )
        self._self_log("n_views", result.get("n_views", 0))
        return {
            "views": views,
            "directions": directions,
            "n_views": result.get("n_views", 0),
            "composite": None,
        }


class EvaluateObjnavTool(BaseCanvasNode):
    node_type = "env_objnav__evaluate"
    display_name = "ObjNav: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull episode metrics (success / spl / soft_spl / distance_to_goal)"
    category = "environment"
    icon = "CheckCircle"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger evaluate (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "Episode metrics dict"),
        PortDef("success", "ANY", "Success (1.0 / 0.0)"),
        PortDef("spl", "ANY", "Success weighted by path length"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        metrics = await _run(_mgr().evaluate)
        self._self_log("metrics", metrics)
        return {
            "metrics": metrics,
            "success": metrics.get("success"),
            "spl": metrics.get("spl"),
        }


# ══════════════════════════════════════════════════════════════════════
# Env panel
# ══════════════════════════════════════════════════════════════════════


class ObjnavEnvPanel(BaseEnvPanel):
    name: ClassVar[str] = "env_objnav"
    display_name: ClassVar[str] = "HM3D ObjectNav"

    fields: ClassVar[list[EnvPanelField]] = [
        EnvPanelField("dataset", "select", "Episodes"),
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
        self._state: dict[str, Any] = {
            "dataset": "v1",
            "split": "val",
            "episode_index": 0,
        }

    @staticmethod
    async def _run(fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            _mgr().executor, fn, *args,
        )

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "dataset": self._state["dataset"],
            "split": self._state["split"],
            "episode_index": self._state["episode_index"],
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = _mgr()
        total = await self._run(mgr.get_total_episodes)
        current = await self._run(mgr.get_episode_info)
        if isinstance(current, dict) and "error" not in current:
            self._state["dataset"] = current.get("dataset", self._state["dataset"])
            self._state["split"] = current.get("split", self._state["split"])
            idx = int(current.get("episode_index", -1))
            if idx >= 0:
                self._state["episode_index"] = idx
        return {
            "available": True,
            "dataset": self._state["dataset"],
            "split": self._state["split"],
            "episode_index": int(self._state["episode_index"]),
            "episode_count": int(total),
            "datasets": list(_DATASETS),
            "splits": list(_SPLITS),
            "step_budget": mgr.max_steps or 500,
            "current_episode": current if "error" not in current else None,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = _mgr()
        if name in ("dataset", "split"):
            if name == "dataset" and str(value) in _DATASETS:
                self._state["dataset"] = str(value)
            elif name == "split" and str(value) in _SPLITS:
                self._state["split"] = str(value)
            self._state["episode_index"] = 0
            await self._run(
                mgr.switch_dataset_split,
                self._state["dataset"],
                self._state["split"],
            )
        elif name == "episode_index":
            try:
                new_index = int(value)
            except (TypeError, ValueError):
                new_index = 0
            self._state["episode_index"] = new_index
            state = await self.on_load()
            # on_load rebinds episode_index from the env's current episode,
            # clobbering the just-armed value (env not reset yet). Restore it
            # so a following on_action("play") seats the new index.
            self._state["episode_index"] = new_index
            state["episode_index"] = new_index
            state["side_effect"] = "signal"
            state["signal_name"] = "episode_reset"
            state["signal_payload"] = self._episode_reset_payload()
            return state
        else:
            self._state[name] = value
            return await self.on_load()

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        mgr = _mgr()
        if name in ("play", "reset"):
            if not mgr.initialized:
                return {
                    "ok": False,
                    "side_effect": "none",
                    "error": "ObjectNav environment not initialized",
                }
            result = await self._run(
                mgr.set_episode_by_index, int(self._state["episode_index"]),
            )
            if name == "play":
                return {"ok": True, "side_effect": "run_start"}
            if isinstance(result, dict) and "error" in result:
                return {"ok": False, "side_effect": "none", "error": result["error"]}
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
        mgr = _mgr()
        if field == "dataset":
            return [{"value": d, "label": f"objectnav_hm3d_{d}"} for d in _DATASETS]
        if field == "split":
            return [{"value": s, "label": s} for s in _SPLITS]
        if field == "episode_index":
            if not mgr.initialized:
                return []
            data = await self._run(mgr.get_episodes_list, 0, 10000)
            episodes = data.get("episodes", []) if isinstance(data, dict) else []
            return [
                {
                    "value": ep.get("index", i),
                    "label": "{}: {} → {}".format(
                        ep.get("index", i),
                        (ep.get("scene_id") or "").split("/")[-1].replace(".basis.glb", ""),
                        ep.get("object_category", "?"),
                    ),
                }
                for i, ep in enumerate(episodes)
            ]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvObjnavNodeSet — registration
# ══════════════════════════════════════════════════════════════════════


class EnvObjnavNodeSet(BaseNodeSet):
    """HM3D ObjectNav (habitat-lab 0.2.4) as a NodeSet."""

    name = "env_objnav"
    description = "HM3D ObjectNav environment (habitat 0.2.4, paper-standard config)"
    server_python = conda_env_python("ac-objnav", "OBJNAV_PYTHON")
    env_panel = ObjnavEnvPanel
    # ADR-server-003: stateful simulator — per-worker scene + agent pose.
    parallelism: ClassVar[str] = "replicated"
    # ADR-eval-002: the habitat step itself is sub-second, but every ObjectNav
    # graph puts a VLM/LLM in the loop (the probe graph makes two vision calls
    # per step). 5.0 killed a 30-step probe at step 10 on wall-clock; mirror
    # env_habitat's 30.0, which is sized for exactly this contention.
    default_per_step_budget_sec: ClassVar[float] = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = ObjnavEnvManager.get()

    def get_tools(self) -> list:
        return [
            ResetObjnavTool(),
            StepDiscreteObjnavTool(),
            StepPoseObjnavTool(),
            ObserveEgocentricObjnavTool(),
            ObserveCameraPoseObjnavTool(),
            ObservePanoramaObjnavTool(),
            EvaluateObjnavTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize the ObjectNav simulator.

        Kwargs:
            dataset: "v1" (default, paper standard) or "v2" (challenge 2023).
            split: "val" (default) or "val_mini".
            gpu_id: CUDA device index (default 0).
            max_steps: episode step budget (default 500).
        """
        if self._mgr.initialized:
            log.info("ObjectNav already initialized — skipping")
            return
        await asyncio.get_running_loop().run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(
                dataset=kwargs.get("dataset", "v1"),
                split=kwargs.get("split", "val"),
                gpu_id=int(kwargs.get("gpu_id", 0)),
                max_steps=int(kwargs.get("max_steps", 500)),
            ),
        )
        log.info("EnvObjnavNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        metadata = {
            "env_name": "objnav_hm3d",
            "datasets": list(_DATASETS),
            "splits": list(_SPLITS),
            "episode_counts": {},
            "metrics": ["success", "spl", "soft_spl", "distance_to_goal"],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": 500,
        }
        if self._mgr.initialized:
            metadata["episode_counts"] = {
                self._mgr._split: self._mgr.get_total_episodes(),
            }
        return metadata

    async def shutdown(self) -> None:
        self._mgr.shutdown()
