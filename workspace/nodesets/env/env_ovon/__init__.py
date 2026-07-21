from __future__ import annotations

"""EnvOvonNodeSet — HM3D-OVON open-vocabulary ObjectNav (habitat-lab 0.2.4).

Gym-like interface (docs: nodesets/env/template.html): ``reset`` (metadata
only) / ``step_discrete`` + ``step_pose`` (control signals) / ``observe_*``
(pull) / ``evaluate`` (metric sink). Node surface and port shapes mirror
``env_objnav`` verb-for-verb — a graph written against one binds to the other
by swapping the node_type prefix.

Task: HM3D-OVON (Yokoyama et al., IROS 2024) — given a goal named by
**free-form text** ("bathtub platform", "bathrobe", "stove"), navigate to
within 1 m of an instance and STOP. The open-vocabulary formulation is the
whole point: unlike ObjectNav's fixed 6 categories, the goal vocabulary is
open at test time, and the three val splits grade generalisation:

  val_seen            categories seen in training
  val_seen_synonyms   the same objects named by unseen synonyms
  val_unseen          categories never seen in training

Each val split is 36 scenes × 3000 episodes (the 36 semantically-annotated
HM3D v0.2 val scenes). ``train`` is deliberately NOT offered: its 145 scenes
come from the HM3D *train* download, which is not staged locally — offering a
split whose scenes are absent is the failure mode env_objnav hit with v2.

Agent / benchmark configuration is inherited from the ObjectNav paper standard
(``benchmark/nav/objectnav/objectnav_hm3d.yaml``): classic cylindrical agent,
0.25 m forward / 30° turn / 640×480 RGB-D / 500-step budget / 1.0 m success.
Upstream OVON pins habitat-lab 0.2.3 + python 3.7; we run 0.2.4 + python 3.9
(the ``ac-objnav`` env). Verified compatible 2026-07-21 by loading all three
val splits and stepping — but it IS a deviation, and the first thing to
suspect if numbers ever disagree with the paper.

Three OVON-specific pieces of wiring, all load-bearing:

1. **Vendored dataset class.** habitat-lab 0.2.4 registers no OVON dataset
   (only PointNav / ObjectNav / InstanceImageNav / MP3DEQA / R2RVLN /
   Rearrange). ``_ovon_dataset`` is a verbatim copy of upstream's
   ``OVON-v1``; importing it registers the class. Episodes carry
   ``children_object_categories`` and string ``object_id``s that plain
   ``ObjectNav-v1`` cannot deserialize.

2. **``objectgoal_sensor`` is removed from the task.** habitat's
   ObjectGoalSensor demands ``dataset.category_to_task_category_id`` — a
   fixed category→int table, which is precisely what an open vocabulary does
   not have; leaving it in aborts task construction with AttributeError. The
   goal reaches the agent as text on ``reset.object_category`` instead, which
   is the open-vocab contract. ``compass`` and ``gps`` sensors are kept.

3. **Every measure keeps its stock habitat value**, ``success_distance``
   included (0.1, judged to goal view points). Upstream OVON overrides it to
   0.25, but it does so as part of a whole different package (Stretch agent,
   ``OVONSim-v0``, ``OVONDistanceToGoal``); taking one knob from that package
   and leaving the rest yields a benchmark comparable to neither. See
   ``_SUCCESS_DISTANCE`` for the escape hatch and the history behind it.

Data layout (repo-root-relative, staged 2026-07-21 from HuggingFace
``nyokoyama/hm3d_ovon``):

  scenes:   data/scene_datasets/hm3d/val/     (shared with env_objnav)
  episodes: data/datasets/ovon/hm3d/{split}/

  **Split-directory / filename mismatch** — the tarball's directory names and
  the json.gz inside them disagree, so habitat's usual
  ``{split}/{split}.json.gz`` template resolves to a missing file for two of
  the three splits. ``_SPLIT_FILES`` maps it out explicitly:

      val_seen/           val_seen.json.gz
      val_seen_synonyms/  val_unseen_easy.json.gz    ← not val_seen_synonyms
      val_unseen/         val_unseen_hard.json.gz    ← not val_unseen

  The tarball also ships macOS AppleDouble turds (``._*``); habitat's
  ``_get_scenes_from_folder`` reads them as scene files and dies with
  BadGzipFile. They were deleted at staging time — re-extracting the tarball
  means deleting them again.

Discrete action space (habitat 0.2.4 ObjectNav defaults):
  0 = STOP · 1 = MOVE_FORWARD (0.25 m) · 2 = TURN_LEFT (30°) ·
  3 = TURN_RIGHT (30°) · 4 = LOOK_UP (30°) · 5 = LOOK_DOWN (30°)

Success judgment uses episode-embedded goal viewpoints, so no semantic mesh
read is required at runtime (same as env_objnav).
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

log = logging.getLogger("agentcanvas.env_ovon")


# ══════════════════════════════════════════════════════════════════════
# Dataset wiring — HM3D-OVON val splits
# ══════════════════════════════════════════════════════════════════════

# train omitted on purpose: HM3D train scenes are not staged (see docstring).
_SPLITS: list[str] = ["val_seen", "val_seen_synonyms", "val_unseen"]

# Directory name → the json.gz actually inside it. Upstream's tarball uses the
# generation-time names (easy/hard) for two of the three published splits.
_SPLIT_FILES: dict[str, str] = {
    "val_seen": "val_seen",
    "val_seen_synonyms": "val_unseen_easy",
    "val_unseen": "val_unseen_hard",
}

_SPLIT_LABELS: dict[str, str] = {
    "val_seen": "val_seen — seen categories",
    "val_seen_synonyms": "val_seen_synonyms — unseen synonyms",
    "val_unseen": "val_unseen — unseen categories",
}

# The agent/measure config is ObjectNav's paper standard; the dataset type,
# data path, objectgoal sensor and success threshold differ (see docstring).
_BENCH_CONFIG = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
_DATA_ROOT = "data/datasets/ovon/hm3d"

# Success radius around a goal viewpoint. LEAVE AT None = habitat's stock
# value (0.1 in objectnav_hm3d.yaml) — that is the standard, and the standard
# is what this nodeset exists to run.
#
# History, so nobody re-"fixes" this: on 2026-07-21 it was briefly pinned to
# 0.25 (upstream OVON's override) after an oracle scored only SR 0.50 at 0.1.
# That was a bad inference — the oracle used at the time chased ONE hand-picked
# viewpoint and used a follower radius below the step size, so it was the test
# that was broken, not the threshold. Overriding a single knob out of upstream's
# config while keeping ObjectNav's agent also produces a benchmark comparable
# to neither. If a future measurement genuinely shows the stock value is
# unreachable, change it with the full upstream OVON config, not à la carte.
_SUCCESS_DISTANCE: float | None = None


def _data_path(split: str) -> str:
    """Fully-resolved episode path for ``split``.

    Returned with no ``{split}`` placeholder left in it: habitat calls
    ``.format(split=...)`` on this string, which is a no-op once resolved.
    Going through the mapping instead of the template is what handles the
    directory/filename mismatch.
    """
    return f"{_DATA_ROOT}/{split}/{_SPLIT_FILES[split]}.json.gz"


def _repo_root() -> str:
    # __file__ lives at workspace/nodesets/env/env_ovon/__init__.py — four parents.
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
    )


# ══════════════════════════════════════════════════════════════════════
# OvonEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class OvonEnvManager:
    """Single (non-vectorized) habitat-lab 0.2.4 HM3D-OVON environment.

    All public methods are blocking — call via
    ``asyncio.get_running_loop().run_in_executor(mgr.executor, fn)``.
    Single-thread executor enforces GL thread affinity.
    """

    _instance: OvonEnvManager | None = None

    def __init__(self) -> None:
        self._env: Any = None
        self._config: Any = None
        self._current_obs: dict | None = None
        self._episode_done: bool = False
        self._step_count: int = 0
        self._last_action: int | None = None
        self._lock = threading.Lock()
        self._split: str = "val_seen"
        self._episode_index: int = 0
        self._gpu_id: int = 0
        self._max_steps: int = 500
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ovon",
        )

    @classmethod
    def get(cls) -> OvonEnvManager:
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
        split: str = "val_seen",
        gpu_id: int = 0,
        max_steps: int = 500,
    ) -> dict:
        """Build habitat config, create env, arm the first episode."""
        with self._lock:
            if self._env is not None:
                log.warning("OvonEnvManager already initialized — skipping")
                return self._get_episode_info_unlocked()
            return self._build_env_unlocked(split, gpu_id, max_steps)

    def _build_env_unlocked(self, split: str, gpu_id: int, max_steps: int) -> dict:
        # Relative data paths in the habitat YAML resolve from CWD.
        repo_root = _repo_root()
        if os.path.isdir(os.path.join(repo_root, "data")):
            os.chdir(repo_root)

        # Registers OVON-v1 in habitat's dataset registry — must precede
        # habitat.Env construction (see module docstring §1).
        from . import _ovon_dataset  # noqa: F401

        import habitat
        from habitat.config import read_write
        from habitat.config.default import get_config

        if split not in _SPLITS:
            raise ValueError(f"Unknown split {split!r} (expected {_SPLITS})")

        config = get_config(_BENCH_CONFIG)
        with read_write(config):
            config.habitat.dataset.type = "OVON-v1"
            config.habitat.dataset.split = split
            config.habitat.dataset.data_path = _data_path(split)
            config.habitat.environment.max_episode_steps = int(max_steps)
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)
            # Open vocabulary has no category→id table (docstring §2).
            config.habitat.task.lab_sensors.pop("objectgoal_sensor", None)
            # Stock habitat threshold unless explicitly overridden (docstring §3).
            if _SUCCESS_DISTANCE is not None:
                config.habitat.task.measurements.success.success_distance = (
                    _SUCCESS_DISTANCE
                )

        log.info(
            "Creating OVON env — split=%s gpu=%d max_steps=%d",
            split, gpu_id, max_steps,
        )
        self._env = habitat.Env(config=config)
        self._config = config
        self._split = split
        self._gpu_id, self._max_steps = int(gpu_id), int(max_steps)

        self._current_obs = self._env.reset()
        self._episode_done = False
        self._step_count = 0
        self._last_action = None
        self._episode_index = self._current_episode_index_unlocked()

        info = self._get_episode_info_unlocked()
        log.info(
            "OVON env ready — episodes=%d episode=%s scene=%s goal=%s",
            len(self._env.episodes), info.get("episode_id"),
            info.get("scene_id"), info.get("object_category"),
        )
        return info

    def shutdown(self) -> None:
        with self._lock:
            if self._env is not None:
                log.info("Shutting down OVON env")
                self._env.close()
                self._env = None

    def switch_split(self, split: str) -> dict:
        """Rebuild the env for a new split selection."""
        with self._lock:
            if self._env is not None:
                self._env.close()
                self._env = None
            return self._build_env_unlocked(split, self._gpu_id, self._max_steps)

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

        Mirrors env_objnav / env_habitat step_pose: real discrete primitives
        (each one advances the task and counts toward SPL / step budget),
        stopping at ``goal_radius``, ``max_nav_steps``, episode end, or path
        exhaustion. Never dispatches STOP — committing is the caller's
        decision. This is the workhorse for frontier-style OVON methods.
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
                # Task sensors kept from the ObjectNav config (VLFM-style
                # methods consume these); objectgoal is absent by design.
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
            "split": self._split,
            "step_count": self._step_count,
            "done": self._episode_done,
            "max_steps": self._max_steps,
        }


def _mgr() -> OvonEnvManager:
    return OvonEnvManager.get()


async def _run(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Nodes — gym verbs (template.html)
# ══════════════════════════════════════════════════════════════════════


class ResetOvonTool(BaseCanvasNode):
    node_type = "env_ovon__reset"
    display_name = "OVON: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Ensure a live episode (re-arm if done) — open-vocab goal text + ids, no observation"
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger reset (any value)", optional=True),
    ]
    output_ports = [
        PortDef("object_category", "TEXT", "Open-vocabulary goal, free-form text (e.g. 'bathtub platform')"),
        PortDef("episode_id", "TEXT", "Episode ID (unique per scene only — key on (scene_id, episode_id))"),
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


class StepDiscreteOvonTool(BaseCanvasNode):
    node_type = "env_ovon__step_discrete"
    display_name = "OVON: Step (discrete)"
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


class StepPoseOvonTool(BaseCanvasNode):
    node_type = "env_ovon__step_pose"
    display_name = "OVON: Step (pose)"
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


class ObserveEgocentricOvonTool(BaseCanvasNode):
    node_type = "env_ovon__observe_egocentric"
    display_name = "OVON: Observe (egocentric)"
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
        PortDef("gps", "ANY", "GPS sensor [x,y] relative to episode start (None if absent)"),
        PortDef("compass", "ANY", "Compass (rad, relative to start; None if absent)"),
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


class ObserveCameraPoseOvonTool(BaseCanvasNode):
    node_type = "env_ovon__observe_camera_pose"
    display_name = "OVON: Camera Pose"
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


class ObservePanoramaOvonTool(BaseCanvasNode):
    node_type = "env_ovon__observe_panorama"
    display_name = "OVON: Observe (panorama)"
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


class EvaluateOvonTool(BaseCanvasNode):
    node_type = "env_ovon__evaluate"
    display_name = "OVON: Evaluate"
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


class OvonEnvPanel(BaseEnvPanel):
    name: ClassVar[str] = "env_ovon"
    display_name: ClassVar[str] = "HM3D-OVON"
    # No dataset selector: HM3D-OVON ships one dataset, and the split IS the
    # experimental axis (seen / synonyms / unseen).
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
        self._state: dict[str, Any] = {
            "split": "val_seen",
            "episode_index": 0,
        }

    @staticmethod
    async def _run(fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            _mgr().executor, fn, *args,
        )

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split": self._state["split"],
            "episode_index": self._state["episode_index"],
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = _mgr()
        total = await self._run(mgr.get_total_episodes)
        current = await self._run(mgr.get_episode_info)
        if isinstance(current, dict) and "error" not in current:
            self._state["split"] = current.get("split", self._state["split"])
            idx = int(current.get("episode_index", -1))
            if idx >= 0:
                self._state["episode_index"] = idx
        return {
            "available": True,
            "split": self._state["split"],
            "episode_index": int(self._state["episode_index"]),
            "episode_count": int(total),
            "splits": list(_SPLITS),
            "step_budget": mgr.max_steps or 500,
            "current_episode": current if "error" not in current else None,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = _mgr()
        if name == "split":
            if str(value) in _SPLITS:
                self._state["split"] = str(value)
            self._state["episode_index"] = 0
            await self._run(mgr.switch_split, self._state["split"])
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
                    "error": "OVON environment not initialized",
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
        if field == "split":
            return [{"value": s, "label": _SPLIT_LABELS.get(s, s)} for s in _SPLITS]
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
# EnvOvonNodeSet — registration
# ══════════════════════════════════════════════════════════════════════


class EnvOvonNodeSet(BaseNodeSet):
    """HM3D-OVON open-vocabulary ObjectNav (habitat-lab 0.2.4) as a NodeSet."""

    name = "env_ovon"
    description = "HM3D-OVON open-vocabulary ObjectNav environment (habitat 0.2.4)"
    # Shares ac-objnav: same habitat-sim 0.2.4 / habitat-lab stack, and the
    # OVON dataset class is vendored rather than pip-installed.
    server_python = conda_env_python("ac-objnav", "OVON_PYTHON")
    env_panel = OvonEnvPanel
    # ADR-server-003: stateful simulator — per-worker scene + agent pose.
    parallelism: ClassVar[str] = "replicated"
    # ADR-eval-002: same sizing as env_objnav — the habitat step is
    # sub-second, but every OVON graph puts a VLM in the loop.
    default_per_step_budget_sec: ClassVar[float] = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = OvonEnvManager.get()

    def get_tools(self) -> list:
        return [
            ResetOvonTool(),
            StepDiscreteOvonTool(),
            StepPoseOvonTool(),
            ObserveEgocentricOvonTool(),
            ObserveCameraPoseOvonTool(),
            ObservePanoramaOvonTool(),
            EvaluateOvonTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize the HM3D-OVON simulator.

        Kwargs:
            split: "val_seen" (default) / "val_seen_synonyms" / "val_unseen".
            gpu_id: CUDA device index (default 0).
            max_steps: episode step budget (default 500).
        """
        if self._mgr.initialized:
            log.info("OVON already initialized — skipping")
            return
        await asyncio.get_running_loop().run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(
                split=kwargs.get("split", "val_seen"),
                gpu_id=int(kwargs.get("gpu_id", 0)),
                max_steps=int(kwargs.get("max_steps", 500)),
            ),
        )
        log.info("EnvOvonNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        metadata = {
            "env_name": "ovon_hm3d",
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
