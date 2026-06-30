from __future__ import annotations

"""EnvHabitatNodeSet — Habitat VLN-CE environment as a unified NodeSet.

Works both as a local in-process nodeset and as an auto-hosted server:
  Local:  POST /api/components/nodesets/env_habitat/load
  Server: POST /api/components/nodesets/env_habitat/load?mode=server

Replaces the former BaseEnv-based ``HabitatEnv`` and ``HabitatServerNodeSet``.

Architecture — three layers
---------------------------
1. ``HabitatEnvManager`` (singleton engine)
     Wraps the raw Habitat-Sim / VLN-CE API. Owns the env instance, a
     threading.Lock, and a single-thread ThreadPoolExecutor that pins all
     simulator work to one OS thread (GL/physics affinity). Exposes clean
     blocking methods: ``initialize``, ``step``, ``reset_episode``,
     ``get_agent_state``, ``get_observations``, ``render_panorama``,
     ``set_episode_by_index``, ``get_episodes_list``, etc. Pure runtime
     plumbing — no canvas awareness.

2. Canvas tool nodes (thin ``BaseCanvasNode`` adapters)
     Each tool class declares metadata (``node_type``, ``display_name``,
     ``category``, ``icon``, ``input_ports``, ``output_ports``,
     ``ui_config``) plus an async ``forward(inputs, ctx)`` that dispatches
     to the manager via ``_run_sync(...)`` — hopping onto the manager's
     executor thread — then maps the manager's return dict into the
     declared output ports and emits ``_self_log`` entries for two-layer
     observability (ADR-022).

     User-facing tools:    observe, step, localize, episode_info,
                           navigate, panorama (+ frontiers/query_map mocks).
     Episode management lives on ``HabitatEnvPanel`` (ADR-025) — no
     canvas nodes for split/episode control; the env panel calls
     ``mgr.set_episode`` / ``mgr.list_episodes`` directly and works
     identically in local and server mode via ``RemoteEnvPanelProxy``.

3. ``EnvHabitatNodeSet`` (collection + lifecycle)
     The ``BaseNodeSet`` binding. ``get_tools()`` returns the tool list,
     ``initialize()`` / ``shutdown()`` drive manager setup/teardown and
     register/unregister the ``HabitatEnvPanel``,
     ``get_eval_metadata()`` feeds the eval page (splits, metrics, counts),
     and ``server_python`` steers ``ComponentRegistry`` to auto-host this
     module in the vlnce conda env when loaded in server mode.

Separation of concerns: manager = engine · tools = canvas adapters ·
nodeset = lifecycle + registry entry. This is the canonical shape for
env nodesets in AgentCanvas.

last updated: 2026-04-13
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

log = logging.getLogger("agentcanvas.habitat")


# ══════════════════════════════════════════════════════════════════════
# Dataset wiring — multi-dataset (R2R-CE / RxR-CE) config routing
#
# The three-field HabitatEnvPanel cascade (dataset → split → episode)
# drives which VLN-CE YAML we feed to ``get_config`` and which base
# split name we hand to ``HabitatEnvManager.initialize``. RxR splits
# carry a BCP-47 language suffix (``val_unseen_en`` / ``_hi`` / ``_te``)
# that selects the per-language RxR baseline YAML; the base split
# (``val_unseen``) is what VLN-CE's ``RxRVLNCEDatasetV1`` actually
# reads from ``data/habitat/datasets/RxR_VLNCE_v0/{base}/``.
# ══════════════════════════════════════════════════════════════════════

_DATASETS: list[str] = ["R2R-CE", "RxR-CE"]

_R2R_CONFIG = "vlnce_baselines/config/r2r_baselines/cma_pm_da.yaml"

# Per-RxR-language baseline configs already ship with VLN-CE upstream.
_RXR_CONFIGS: dict[str, str] = {
    "en": "vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml",
    "hi": "vlnce_baselines/config/rxr_baselines/rxr_cma_hi.yaml",
    "te": "vlnce_baselines/config/rxr_baselines/rxr_cma_te.yaml",
}

# BCP-47 prefix table — mirrors matterport3d.py for cross-nodeset parity.
# Key is the split-name suffix (``_en``); value is the BCP-47 prefix used
# inside ``ep.instruction.language`` strings (e.g. ``en-IN``, ``en-US``).
_RXR_LANG_MAP: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "te": "te",
}

_R2R_BASE_SPLITS: list[str] = ["val_unseen", "val_seen", "train", "rand100"]
_RXR_BASE_SPLITS: list[str] = ["val_unseen", "val_seen", "train", "test_challenge"]


def _dataset_splits(dataset: str) -> list[str]:
    """Return the display split list for a given dataset selection."""
    if dataset == "RxR-CE":
        return [f"{base}_{lang}" for base in _RXR_BASE_SPLITS for lang in _RXR_LANG_MAP]
    return list(_R2R_BASE_SPLITS)


def _resolve_dataset_config(dataset: str, split: str) -> tuple[str, str]:
    """Pick (exp_config, base_split) for a (dataset, split) pair.

    RxR splits must carry a ``_en`` / ``_hi`` / ``_te`` suffix. The suffix
    selects the per-language YAML; the base (``val_unseen``) is what
    VLN-CE's RxR dataset loader consumes. R2R-CE passes through
    unchanged.

    Raises ValueError for unknown datasets or malformed RxR splits.
    """
    if dataset in ("", "R2R-CE"):
        return _R2R_CONFIG, split
    if dataset == "RxR-CE":
        for suffix, _ in _RXR_LANG_MAP.items():
            if split.endswith("_" + suffix):
                base = split[: -(len(suffix) + 1)]
                return _RXR_CONFIGS[suffix], base
        raise ValueError(f"RxR-CE split {split!r} is missing a language suffix (_en / _hi / _te)")
    raise ValueError(f"Unknown dataset {dataset!r}")


def _extract_episode_extras(ep: Any) -> dict[str, Any]:
    """Build the RxR-compatible ``extras`` dict for an episode.

    Mirrors the shape emitted by the MP3D nodeset (``matterport3d.py::
    _load_rxr``) so downstream agent graphs can be dataset-agnostic.
    Every key is populated with ``None`` when the underlying VLN-CE
    episode does not carry that field — keeping the schema stable
    across R2R-CE, RxR-CE, and any future dataset.
    """
    inst = getattr(ep, "instruction", None)
    extras: dict[str, Any] = {
        "language": getattr(inst, "language", None) if inst else None,
        "instruction_id": getattr(inst, "instruction_id", None) if inst else None,
        "annotator_id": getattr(inst, "annotator_id", None) if inst else None,
        "timed_instruction": getattr(ep, "timed_instruction", None),
        "pose_trace_path": None,  # CE-side pose traces are out-of-scope (E8)
    }
    return extras


# ══════════════════════════════════════════════════════════════════════
# HabitatEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


# ── temporary demo-recording tap (guarded by outputs/_record/ENABLE) ──────
# Dumps per-low-level-step RGB to outputs/_record/<subdir>/ep_<id>/ ONLY when
# the sentinel file exists; no-op otherwise. Safe to delete after recording.
_REC_DIR = "outputs/_record"


def _rec_save(subdir: str, ep, idx, img) -> None:
    import os
    if img is None or not os.path.exists(_REC_DIR + "/ENABLE"):
        return
    try:
        import numpy as _np
        from PIL import Image as _Image
        d = os.path.join(_REC_DIR, subdir, "ep_" + str(ep).replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        _Image.fromarray(_np.asarray(img)).save(os.path.join(d, "f%06d.png" % int(idx)))
    except Exception:
        pass


class HabitatEnvManager:
    """Manages a single (non-vectorized) Habitat VLN-CE environment.

    All public methods are blocking and should be called via
    ``asyncio.run_in_executor(mgr.executor, fn)`` from async code.
    Single-thread executor enforces GL/physics thread affinity.
    """

    _instance: HabitatEnvManager | None = None

    def __init__(self) -> None:
        self._env = None
        self._current_obs: dict | None = None
        self._episode_done: bool = False
        self._step_count: int = 0
        self._lock = threading.Lock()
        self._config = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="habitat",
        )

    # ── Singleton access ──

    @classmethod
    def get(cls) -> HabitatEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lifecycle ──

    def initialize(
        self,
        exp_config: str,
        split: str = "val_unseen",
        gpu_id: int = 0,
        max_steps: int = 500,
    ) -> dict:
        """Build Habitat config, create env, reset to first episode."""
        with self._lock:
            if self._env is not None:
                log.warning("HabitatEnvManager already initialized — skipping")
                return self.get_episode_info()

            # __file__ lives at workspace/nodesets/env/env_habitat.py — three
            # parents to reach the repo root.
            repo_root = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
            )
            candidates = [
                os.environ.get("VLNCE_ROOT", ""),
                os.path.join(repo_root, "..", "VLN-CE"),
                os.path.join(repo_root, "third_party", "VLN-CE"),
            ]
            vlnce_root = None
            for c in candidates:
                c = os.path.normpath(c) if c else ""
                if c and os.path.isdir(os.path.join(c, "data")):
                    vlnce_root = c
                    break
            if vlnce_root is None:
                vlnce_root = os.path.normpath(
                    os.path.join(repo_root, "third_party", "VLN-CE"),
                )
            if os.path.isdir(vlnce_root):
                log.info("Changing cwd to VLN-CE root: %s", vlnce_root)
                os.chdir(vlnce_root)

            import habitat_extensions  # noqa: F401
            import vlnce_baselines  # noqa: F401
            from habitat_baselines.common.environments import get_env_class
            from vlnce_baselines.config.default import get_config

            config = get_config(exp_config)
            config.defrost()

            config.TASK_CONFIG.DATASET.SPLIT = split
            config.TASK_CONFIG.DATASET.ROLES = ["guide"]
            config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
            config.TASK_CONFIG.TASK.NDTW.SPLIT = split
            config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
            config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
            config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = max_steps
            config.EVAL.EPISODE_COUNT = -1
            config.SIMULATOR_GPU_IDS = [gpu_id]
            config.TORCH_GPU_ID = gpu_id
            config.NUM_ENVIRONMENTS = 1

            config.freeze()
            self._config = config

            log.info("Creating Habitat env (ENV_NAME=%s) ...", config.ENV_NAME)
            env_cls = get_env_class(config.ENV_NAME)
            self._env = env_cls(config=config)

            log.info("Resetting to first episode ...")
            self._current_obs = self._env.reset()
            self._episode_done = False
            self._step_count = 0

            info = self._get_episode_info_unlocked()
            log.info(
                "Habitat env ready — episode=%s scene=%s",
                info.get("episode_id"),
                info.get("scene_id"),
            )
            return info

    def shutdown(self) -> None:
        with self._lock:
            if self._env is not None:
                log.info("Shutting down Habitat env")
                self._env.close()
                self._env = None

    @property
    def initialized(self) -> bool:
        return self._env is not None

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    def get_spaces(self) -> tuple:
        with self._lock:
            if self._env is None:
                return None, None
            return self._env.observation_space, self._env.action_space

    @property
    def max_steps(self) -> int:
        """Episode step budget (TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS)."""
        if self._config is None:
            return 0
        try:
            return int(self._config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS)
        except Exception:
            return 0

    def _split_done(self, done: bool) -> tuple[bool, bool]:
        """Split habitat's single ``done`` into gym ``(terminated, truncated)``.

        ``truncated`` = the episode ended because the step budget was hit;
        ``terminated`` = it ended for a task reason (STOP / goal). Habitat
        flips ``done`` for both, so we infer truncation from the step count.
        """
        if not done:
            return False, False
        ms = self.max_steps
        truncated = bool(ms and self._step_count >= ms)
        return (not truncated), truncated

    def get_cam_intrinsics(self) -> dict | None:
        """Best-effort pinhole intrinsics from the RGB sensor config.

        Returns ``{fx, fy, cx, cy, width, height}`` or ``None`` when the
        sensor config is unavailable. Folded into ``observe_egocentric``.
        """
        if self._config is None:
            return None
        try:
            sensor = self._config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
            w, h = int(sensor.WIDTH), int(sensor.HEIGHT)
            hfov = float(sensor.HFOV)
            f = (w / 2.0) / float(np.tan(np.radians(hfov) / 2.0))
            return {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}
        except Exception:
            return None

    # ── Actions ──

    def step(self, action: int) -> dict:
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if self._episode_done:
                return {"error": "Episode already done", "done": True}

            obs, reward, done, info = self._env.step(action)
            self._current_obs = obs
            self._episode_done = done
            self._step_count += 1

            agent_state = self._get_agent_state_unlocked()
            result: dict[str, Any] = {
                "position": agent_state["position"],
                "orientation": agent_state["orientation"],
                "done": done,
                "step_count": self._step_count,
                "action": action,
                "reward": float(reward),
            }
            terminated, truncated = self._split_done(done)
            result["terminated"] = terminated
            result["truncated"] = truncated

            rgb = self._extract_rgb(obs)
            depth = self._extract_depth(obs)
            if rgb is not None:
                result["rgb_base64"] = self.encode_rgb_base64(rgb)
            if depth is not None:
                result["depth_base64"] = self.encode_depth_base64(depth)

            if done and info:
                result["metrics"] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in info.items()
                    if isinstance(v, (int, float, np.floating, str, bool))
                }

            return result

    def step_hightolow(self, angle: float, distance: float) -> dict:
        """Open-Nav HIGHTOLOW action: rotate by ``angle`` (rad) then walk
        ``distance`` (m) forward via repeated MOVE_FORWARD steps.

        Mirrors ``Open-Nav/habitat_extensions/nav.py:28-67``: builds a yaw
        quaternion, composes it onto the current rotation, sets the agent
        state via ``sim.set_agent_state``, then dispatches ``ksteps =
        int(distance // 0.25)`` MOVE_FORWARD primitive actions. Returns the
        same dict shape as ``step()``.

        Boundary: ``angle == 0 and distance == 0`` is the canonical "no
        movement" signal an LLM-driven graph uses to mean STOP. Habitat
        records success only when the agent explicitly calls action 0;
        without this dispatch the episode times out with success=0 even
        when the agent is at the goal. We forward such no-op tuples to
        ``self.step(0)`` so the habitat task's measurements register the
        STOP and the next ``evaluate`` pull reflects it.
        """
        if angle == 0.0 and distance == 0.0:
            return self.step(0)

        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if self._episode_done:
                return {"error": "Episode already done", "done": True}

            sim = self._env._env.sim
            agent_state = sim.get_agent_state()
            pos = agent_state.position
            rot = agent_state.rotation

            yaw_quat = np.quaternion(
                np.cos(float(angle) / 2.0),
                0.0,
                np.sin(float(angle) / 2.0),
                0.0,
            )
            new_rot = rot * yaw_quat
            sim.set_agent_state(pos, new_rot)

            forward_step = 0.25
            ksteps = max(1, int(float(distance) // forward_step))
            forward_action = 1  # MOVE_FORWARD

            obs = self._current_obs
            done = False
            info: dict = {}
            for _ in range(ksteps):
                if self._episode_done:
                    break
                obs, _reward, done, info = self._env.step(forward_action)
                self._step_count += 1
                try:
                    _rec_ep = str(self._env._env.current_episode.episode_id)
                except Exception:
                    _rec_ep = "x"
                _rec_save("habitat", _rec_ep, self._step_count, self._extract_rgb(obs))
                if done:
                    self._episode_done = True
                    break

            self._current_obs = obs
            agent_state = self._get_agent_state_unlocked()
            result: dict[str, Any] = {
                "position": agent_state["position"],
                "orientation": agent_state["orientation"],
                "done": self._episode_done,
                "step_count": self._step_count,
                "angle": float(angle),
                "distance": float(distance),
                "ksteps": ksteps,
            }
            terminated, truncated = self._split_done(self._episode_done)
            result["terminated"] = terminated
            result["truncated"] = truncated
            rgb = self._extract_rgb(obs)
            depth = self._extract_depth(obs)
            if rgb is not None:
                result["rgb_base64"] = self.encode_rgb_base64(rgb)
            if depth is not None:
                result["depth_base64"] = self.encode_depth_base64(depth)
            if self._episode_done and info:
                result["metrics"] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in info.items()
                    if isinstance(v, (int, float, np.floating, str, bool))
                }
            return result

    # ── AO-Planner multi-point Path execution (env_habitat__step_path) ──
    # Faithful port of AO-Planner ``environments_llm.py``: ``single_step_control``
    # (:387), ``multi_step_control`` (:447), ``turn`` (:364), with the try-out
    # obstacle-sliding block (:398-445). The path arrives as a POLAR sequence
    # (angle_rad, distance_m) in the same convention as ``step_hightolow`` (yaw
    # relative to the current/observation heading), so it reuses the verified
    # pixel->polar projection. ``step_path`` converts the sequence to WORLD targets
    # once (from the captured init pose), then walks each via discrete TURN/MOVE
    # primitives, re-deriving the bearing from the ACTUAL pose at every hop (exactly
    # like ``multi_step_control``) so a collision/slide on one hop does not throw off
    # the aim of the next. Try-out uses ``random.choice`` for the slide direction —
    # upstream is itself non-deterministic here (environments_llm.py:409).

    @staticmethod
    def _heading_from_quaternion(quat: Any) -> float:
        """Agent yaw in AO-Planner's convention (0 == facing -Z), in [0, 2pi).

        Verbatim from ``habitat_extensions/utils.py:843-849``.
        """
        from habitat.tasks.utils import cartesian_to_polar
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        hv = quaternion_rotate_vector(quat.inverse(), np.array([0.0, 0.0, -1.0]))
        phi = cartesian_to_polar(-hv[2], hv[0])[1]
        return float(phi % (2 * np.pi))

    @staticmethod
    def _calculate_vp_rel_pos(p1: Any, p2: Any, base_heading: float = 0.0) -> tuple:
        """Relative (heading, ground-distance) from p1 to p2 minus base_heading.

        Verbatim from ``environments_llm.py:27-43`` (``calculate_vp_rel_pos``).
        """
        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        xz_dist = max(float(np.sqrt(dx ** 2 + dz ** 2)), 1e-8)
        heading = float(np.arcsin(-dx / xz_dist))
        if p2[2] > p1[2]:
            heading = np.pi - heading
        heading -= base_heading
        while heading < 0:
            heading += 2 * np.pi
        heading = heading % (2 * np.pi)
        return float(heading), float(xz_dist)

    def step_path(self, path_angles: Any, path_distances: Any, tryout: bool = True) -> dict:
        """Walk a multi-point Path (AO-Planner VAP) given as a polar sequence.

        ``path_angles``/``path_distances`` are parallel lists in the
        ``step_hightolow`` polar convention (yaw rad relative to current heading,
        ground-distance m). An empty path (or all near-zero hops) means STOP ->
        ``step(0)``.
        """
        import math
        import random

        angs = [float(a) for a in (path_angles or [])]
        dists = [float(d) for d in (path_distances or [])]
        # Drop near-zero hops up front (e.g. the foot / path-start anchor at the
        # image bottom edge, whose depth is ~0 — D-5: usable as a path start, but
        # contributes no movement).
        hops = [(a, d) for a, d in zip(angs, dists) if d > 1e-3]
        if not hops:
            return self.step(0)  # STOP / no-op

        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            if self._episode_done:
                return {"error": "Episode already done", "done": True}

            from habitat.sims.habitat_simulator.actions import HabitatSimActions
            from habitat.utils.geometry_utils import quaternion_rotate_vector

            sim = self._env._env.sim
            act_f = HabitatSimActions.MOVE_FORWARD
            act_l = HabitatSimActions.TURN_LEFT
            act_r = HabitatSimActions.TURN_RIGHT
            try:
                sim_cfg = self._config.TASK_CONFIG.SIMULATOR
                uni_deg = float(getattr(sim_cfg, "TURN_ANGLE", 15) or 15)
                uni_f = float(getattr(sim_cfg, "FORWARD_STEP_SIZE", 0.25) or 0.25)
            except Exception:  # noqa: BLE001
                uni_deg, uni_f = 15.0, 0.25

            # 1. Convert the polar sequence to WORLD targets from the captured pose
            #    (each hop lands where step_hightolow(ang, dist) would: rotate the
            #    base heading by ang, then move dist along -Z).
            init_state = sim.get_agent_state()
            pos0 = np.array(init_state.position, dtype=float)
            rot0 = init_state.rotation
            targets = []
            for ang, dist in hops:
                yaw = np.quaternion(np.cos(ang / 2.0), 0.0, np.sin(ang / 2.0), 0.0)
                head = rot0 * yaw
                wd = quaternion_rotate_vector(head, np.array([0.0, 0.0, -1.0]))
                targets.append(pos0 + wd * dist)

            walk = {"obs": self._current_obs, "info": {}}

            def _prim(act: int) -> bool:
                obs, _reward, done, info = self._env.step(act)
                walk["obs"], walk["info"] = obs, info
                self._step_count += 1
                if done:
                    self._episode_done = True
                return done

            def _turn(ang_rad: float) -> None:
                ang_deg = math.degrees(ang_rad)
                ang_deg = round(ang_deg / uni_deg) * uni_deg
                if 180 < ang_deg <= 360:
                    ang_deg -= 360
                turns = [act_l] * int(ang_deg // uni_deg) if ang_deg >= 0 else [act_r] * int(-ang_deg // uni_deg)
                for t in turns:
                    if self._episode_done or _prim(t):
                        return

            def _single_step(target: Any) -> None:
                st = sim.get_agent_state()
                base_h = self._heading_from_quaternion(st.rotation)
                ang, dis = self._calculate_vp_rel_pos(st.position, target, base_h)
                _turn(ang)
                if self._episode_done:
                    return
                ksteps = int(dis // uni_f)
                if not tryout:
                    for _ in range(ksteps):
                        if _prim(act_f):
                            return
                    return
                # try-out obstacle sliding (environments_llm.py:398-445)
                cnt = 0
                for _ in range(ksteps):
                    if _prim(act_f):
                        return
                    if sim.previous_step_collided:
                        break
                    cnt += 1
                ksteps -= cnt
                if ksteps <= 0:
                    return
                try_ang = random.choice([math.radians(90), math.radians(270)])
                _turn(try_ang)
                if self._episode_done:
                    return
                if try_ang == math.radians(90):
                    turn_seqs = [(0, 270), (330, 300), (330, 330), (300, 30), (330, 60), (330, 90)]
                else:
                    turn_seqs = [(0, 90), (30, 60), (30, 30), (60, 330), (30, 300), (30, 270)]
                for head_t, tail_t in turn_seqs:
                    _turn(math.radians(head_t))
                    if self._episode_done:
                        return
                    prev = sim.get_agent_state().position
                    if _prim(act_f):
                        return
                    post = sim.get_agent_state().position
                    if list(prev) != list(post):
                        _turn(math.radians(tail_t))
                        if self._episode_done:
                            return
                        for _ in range(ksteps):
                            if _prim(act_f):
                                return
                            if sim.previous_step_collided:
                                break
                        return

            # 2. Walk every target, re-deriving the bearing from the actual pose.
            for tgt in targets:
                if self._episode_done:
                    break
                _single_step(tgt)

            obs, info = walk["obs"], walk["info"]
            self._current_obs = obs
            agent_state = self._get_agent_state_unlocked()
            result: dict[str, Any] = {
                "position": agent_state["position"],
                "orientation": agent_state["orientation"],
                "done": self._episode_done,
                "step_count": self._step_count,
                "n_hops": len(targets),
            }
            terminated, truncated = self._split_done(self._episode_done)
            result["terminated"] = terminated
            result["truncated"] = truncated
            rgb = self._extract_rgb(obs)
            depth = self._extract_depth(obs)
            if rgb is not None:
                result["rgb_base64"] = self.encode_rgb_base64(rgb)
            if depth is not None:
                result["depth_base64"] = self.encode_depth_base64(depth)
            if self._episode_done and info:
                result["metrics"] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in info.items()
                    if isinstance(v, (int, float, np.floating, str, bool))
                }
            return result

    def evaluate(self) -> dict:
        """Pull current habitat task metrics on demand — independent of done.

        Mirrors ``env_mp3d.HabitatEnvManager.graph_evaluate``: returns the
        same metric dict that ``step``/``step_hightolow`` produce when
        ``done=True``, but callable mid-episode (e.g. after step_budget cap
        is hit by a graph that never triggers habitat's terminal flag).

        Uses ``habitat.Env.get_metrics()`` which reads the task's measurement
        cache — no env-stepping side effect. Returns ``{}`` if no episode
        is live.
        """
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            try:
                info = self._env._env.get_metrics()
            except Exception as e:
                return {"error": f"get_metrics failed: {e}"}
            if not isinstance(info, dict):
                return {"error": f"get_metrics returned {type(info).__name__}"}
            return {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in info.items()
                if isinstance(v, (int, float, np.floating, str, bool))
            }

    def render_panorama_rgbd(self, n_views: int = 12) -> dict:
        """Aligned RGB+Depth panorama for waypoint prediction.

        Returns ``{views: [{dir_id, heading_deg, rgb_base64, depth_base64}],
        n_views}``. Used by the Open-Nav waypoint predictor which needs
        12 RGB and 12 Depth tensors aligned to 12 30° heading bins.
        """
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}

            sim = self._env._env.sim
            agent_state = sim.get_agent_state()
            pos = agent_state.position
            rot = agent_state.rotation

            angle_step = 2.0 * np.pi / n_views
            views = []
            for i in range(n_views):
                yaw = i * angle_step
                yaw_quat = np.quaternion(
                    np.cos(yaw / 2),
                    0.0,
                    np.sin(yaw / 2),
                    0.0,
                )
                new_rot = rot * yaw_quat
                rot_list = [
                    float(new_rot.x),
                    float(new_rot.y),
                    float(new_rot.z),
                    float(new_rot.w),
                ]
                obs = sim.get_observations_at(pos.tolist(), rot_list)
                if obs is None:
                    continue
                rgb = self._extract_rgb(obs)
                depth = self._extract_depth(obs)
                view: dict[str, Any] = {
                    "dir_id": i,
                    "heading_deg": round(np.degrees(yaw) % 360, 1),
                }
                if rgb is not None:
                    view["rgb_base64"] = self.encode_rgb_base64(np.asarray(rgb, dtype=np.uint8))
                if depth is not None:
                    view["depth_base64"] = self.encode_depth_base64(depth)
                    view["depth_raw_base64"] = self.encode_depth_raw_base64(depth)
                views.append(view)
            return {"views": views, "n_views": len(views)}

    # ── Queries ──

    def get_agent_state(self) -> dict:
        with self._lock:
            return self._get_agent_state_unlocked()

    def get_state(self) -> dict:
        return self.get_agent_state()

    def _get_agent_state_unlocked(self) -> dict:
        if self._env is None:
            return {"error": "Environment not initialized"}
        sim = self._env._env.sim
        state = sim.get_agent_state()
        pos = state.position.tolist()
        rot = state.rotation
        orientation = [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
        return {"position": pos, "orientation": orientation}

    def get_observations(self) -> dict:
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            obs = self._current_obs
            rgb = self._extract_rgb(obs)
            depth = self._extract_depth(obs)
            result: dict[str, Any] = {}
            if rgb is not None:
                result["rgb_base64"] = self.encode_rgb_base64(rgb)
                result["width"] = rgb.shape[1]
                result["height"] = rgb.shape[0]
            if depth is not None:
                result["depth_base64"] = self.encode_depth_base64(depth)
            return result

    def get_observation(self) -> dict:
        return self.get_observations()

    def get_episode_info(self) -> dict:
        with self._lock:
            return self._get_episode_info_unlocked()

    def _get_episode_info_unlocked(self) -> dict:
        if self._env is None:
            return {"error": "Environment not initialized"}
        ep = self._env.current_episode
        info: dict[str, Any] = {
            "episode_id": str(ep.episode_id),
            "scene_id": ep.scene_id,
            "step_count": self._step_count,
            "done": self._episode_done,
        }
        if hasattr(ep, "instruction") and ep.instruction:
            inst = ep.instruction
            if hasattr(inst, "instruction_text"):
                info["instruction"] = inst.instruction_text
            elif isinstance(inst, str):
                info["instruction"] = inst
        extras = _extract_episode_extras(ep)
        info["language"] = extras["language"]
        info["extras"] = extras
        if hasattr(ep, "goals"):
            info["goals"] = [{"position": g.position} for g in ep.goals if hasattr(g, "position")]
        return info

    def get_instruction_text(self) -> str | None:
        """Current episode's raw natural-language instruction (per-episode
        metadata, NOT a per-step sensor). Emitted by observe so a downstream
        tokenizer node can derive token ids — the env stays raw-text only."""
        with self._lock:
            if self._env is None:
                return None
            inst = getattr(self._env.current_episode, "instruction", None)
            if inst is None:
                return None
            if hasattr(inst, "instruction_text"):
                return inst.instruction_text
            if isinstance(inst, str):
                return inst
            return None

    def reset_episode(self) -> dict:
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}
            self._current_obs = self._env.reset()
            self._episode_done = False
            self._step_count = 0

            info = self._get_episode_info_unlocked()
            agent_state = self._get_agent_state_unlocked()
            info.update(agent_state)

            rgb = self._extract_rgb(self._current_obs)
            depth = self._extract_depth(self._current_obs)
            if rgb is not None:
                info["rgb_base64"] = self.encode_rgb_base64(rgb)
            if depth is not None:
                info["depth_base64"] = self.encode_depth_base64(depth)
            return info

    def reset(self) -> dict:
        return self.reset_episode()

    def get_total_episodes(self) -> int:
        with self._lock:
            if self._env is None:
                return 0
            try:
                inner = self._env._env
                if hasattr(inner, "_dataset") and hasattr(inner._dataset, "episodes"):
                    return len(inner._dataset.episodes)
            except Exception:
                pass
            return 0

    def get_status(self) -> dict:
        with self._lock:
            if self._env is None:
                return {"initialized": False}
            ep_info = self._get_episode_info_unlocked()
            result = {
                "initialized": True,
                "step_count": self._step_count,
                "done": self._episode_done,
                "episode_id": ep_info.get("episode_id"),
                "scene_id": ep_info.get("scene_id"),
            }
            try:
                inner = self._env._env
                if hasattr(inner, "_dataset") and hasattr(inner._dataset, "episodes"):
                    result["total_episodes"] = len(inner._dataset.episodes)
            except Exception:
                pass
            return result

    def get_raw_obs(self) -> dict | None:
        with self._lock:
            return self._current_obs

    def get_config(self):
        return self._config

    def get_current_episode(self):
        with self._lock:
            if self._env is None:
                return None
            return self._env.current_episode

    def set_episode_by_index(self, index: int) -> dict:
        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}

            inner = self._env._env
            dataset = inner._dataset
            if index < 0 or index >= len(dataset.episodes):
                return {"error": f"Index {index} out of range (0-{len(dataset.episodes) - 1})"}

            target_ep = dataset.episodes[index]
            inner._episode_iterator = iter([target_ep])
            self._current_obs = self._env.reset()
            # habitat 0.1.7 Env.reset() advances the iterator and reloads
            # the scene if it changed — but when the new episode is in
            # the same scene as the previous one, sim.reset() does NOT
            # re-place the agent at the episode's start. Consecutive
            # same-scene episodes inherit the previous episode's terminal
            # pose, silently breaking every metric beyond the first
            # per-scene episode (observed 2026-05-15 on smartway2_ce eval:
            # ep10/ep11 same scene → identical step-0 panorama hash +
            # pose). Force-place the agent at the canonical start, then
            # refresh observations + task measurements.
            sim = inner.sim
            start_pos = np.array(target_ep.start_position, dtype=np.float32)
            sr = target_ep.start_rotation
            # Episode stores rotation as [x, y, z, w]; np.quaternion takes w first.
            start_rot = np.quaternion(float(sr[3]), float(sr[0]), float(sr[1]), float(sr[2]))
            # task.reset first (it may reposition the agent; we override after).
            try:
                inner._task.reset(episode=target_ep)
            except Exception:
                pass
            sim.set_agent_state(start_pos, start_rot)
            self._current_obs = sim.get_sensor_observations()
            self._episode_done = False
            self._step_count = 0

            info = self._get_episode_info_unlocked()
            info.update(self._get_agent_state_unlocked())
            rgb = self._extract_rgb(self._current_obs)
            depth = self._extract_depth(self._current_obs)
            if rgb is not None:
                info["rgb_base64"] = self.encode_rgb_base64(rgb)
            if depth is not None:
                info["depth_base64"] = self.encode_depth_base64(depth)
            return info

    def set_episode(self, index: int) -> dict:
        return self.set_episode_by_index(index)

    def get_episodes_list(self, offset: int = 0, limit: int = 50) -> dict:
        with self._lock:
            if self._env is None:
                return {"episodes": [], "total": 0}
            dataset = self._env._env._dataset
            eps = dataset.episodes[offset : offset + limit]
            result = []
            for i, ep in enumerate(eps):
                entry: dict[str, Any] = {
                    "index": offset + i,
                    "episode_id": str(ep.episode_id),
                    "scene_id": ep.scene_id,
                }
                if hasattr(ep, "instruction") and ep.instruction:
                    entry["instruction"] = getattr(
                        ep.instruction,
                        "instruction_text",
                        str(ep.instruction),
                    )
                extras = _extract_episode_extras(ep)
                entry["language"] = extras["language"]
                entry["extras"] = extras
                result.append(entry)
            return {"episodes": result, "total": len(dataset.episodes)}

    def get_episodes(self, offset: int = 0, limit: int = 50) -> dict:
        return self.get_episodes_list(offset, limit)

    # ── Observation helpers ──

    @staticmethod
    def _extract_rgb(obs: dict | None) -> np.ndarray | None:
        if obs is None:
            return None
        for key in ("rgb", "RGB"):
            if key in obs:
                return np.asarray(obs[key])
        return None

    @staticmethod
    def _extract_depth(obs: dict | None) -> np.ndarray | None:
        if obs is None:
            return None
        for key in ("depth", "DEPTH"):
            if key in obs:
                return np.asarray(obs[key])
        return None

    @staticmethod
    def encode_rgb_base64(rgb: np.ndarray) -> str:
        from PIL import Image

        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def encode_depth_base64(depth: np.ndarray) -> str:
        from PIL import Image

        d = np.squeeze(depth)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        img = Image.fromarray(d_norm, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def encode_depth_raw_base64(depth: np.ndarray) -> str:
        # 16-bit PNG depth in millimetres — preserves absolute metric depth
        # so downstream consumers (e.g. SpatialBot's 3-channel packer in
        # Open-Nav) can reconstruct the full habitat depth range. Separate
        # from encode_depth_base64 which normalises to 8-bit for viewers.
        from PIL import Image

        d = np.squeeze(depth).astype(np.float32)
        # Habitat depth is in metres; cap at 65.535 m to fit uint16 mm.
        d_mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        img = Image.fromarray(d_mm, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── Panoramic Rendering ──

    def render_panorama(self, n_views: int = 4) -> dict:
        import quaternion as _quat_mod  # noqa: F401

        with self._lock:
            if self._env is None:
                return {"error": "Environment not initialized"}

            sim = self._env._env.sim
            agent_state = sim.get_agent_state()
            pos = agent_state.position
            rot = agent_state.rotation

            direction_names = {
                4: ["Front", "Right", "Back", "Left"],
                8: [
                    "Front",
                    "Front-Right",
                    "Right",
                    "Back-Right",
                    "Back",
                    "Back-Left",
                    "Left",
                    "Front-Left",
                ],
                12: [
                    "Front(0°)",
                    "Front-Right(30°)",
                    "Front-Right(60°)",
                    "Right(90°)",
                    "Back-Right(120°)",
                    "Back-Right(150°)",
                    "Back(180°)",
                    "Back-Left(210°)",
                    "Back-Left(240°)",
                    "Left(270°)",
                    "Front-Left(300°)",
                    "Front-Left(330°)",
                ],
                24: [f"{i * 15}°" for i in range(24)],
            }.get(n_views, [f"View_{i}" for i in range(n_views)])

            angle_step = 2.0 * np.pi / n_views
            views = []
            rgb_arrays = []

            for i in range(n_views):
                yaw = i * angle_step
                yaw_quat = np.quaternion(
                    np.cos(yaw / 2),
                    0.0,
                    np.sin(yaw / 2),
                    0.0,
                )
                new_rot = rot * yaw_quat
                rot_list = [
                    float(new_rot.x),
                    float(new_rot.y),
                    float(new_rot.z),
                    float(new_rot.w),
                ]
                pos_list = pos.tolist()

                obs = sim.get_observations_at(pos_list, rot_list)
                if obs is None:
                    continue

                rgb = self._extract_rgb(obs)
                if rgb is not None:
                    rgb = np.asarray(rgb, dtype=np.uint8)
                    rgb_arrays.append(rgb)
                    views.append(
                        {
                            "direction": direction_names[i],
                            "heading_deg": round(np.degrees(yaw) % 360, 1),
                            "rgb_base64": self.encode_rgb_base64(rgb),
                        }
                    )

            composite_b64 = ""
            if rgb_arrays:
                composite_b64 = self._build_composite(
                    rgb_arrays,
                    [v["direction"] for v in views],
                )

            return {
                "views": views,
                "composite_base64": composite_b64,
                "n_views": len(views),
            }

    @staticmethod
    def _build_composite(images: list[np.ndarray], labels: list[str]) -> str:
        from PIL import Image, ImageDraw

        n = len(images)
        if n == 0:
            return ""

        h, w = images[0].shape[:2]
        label_h = 20
        border = 2  # thin black border around each sub-image

        if n <= 4:
            cols, rows = 2, 2
        elif n <= 8:
            cols, rows = 4, 2
        elif n <= 12:
            cols, rows = 4, 3
        else:
            cols, rows = 6, 4

        cell_w = w + border * 2
        cell_h = h + label_h + border * 2
        canvas_w = cols * cell_w
        canvas_h = rows * cell_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for i, (img_arr, label) in enumerate(zip(images, labels)):
            row, col = divmod(i, cols)
            x = col * cell_w + border
            y = row * cell_h + border
            # Label text (yellow on black background above image)
            draw.text((x + w // 2 - len(label) * 3, y + 2), label, fill=(255, 255, 100))
            # Sub-image with border gap (black canvas shows through as border)
            pil_img = Image.fromarray(img_arr)
            canvas.paste(pil_img, (x, y + label_h))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ══════════════════════════════════════════════════════════════════════
# Habitat Tools
# ══════════════════════════════════════════════════════════════════════


def _get_env() -> HabitatEnvManager:
    return HabitatEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    env_mgr = _get_env()
    return await asyncio.get_running_loop().run_in_executor(env_mgr.executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Nodes — gym-like env interface (see docs: nodesets/env/template.html)
#
#   reset                episode metadata only (no observation)
#   step_discrete        action:ACTION  → reward / terminated / truncated / info
#   step_pose            target:POSE    → reward / terminated / truncated / info
#   step_hightolow       angle,distance → reward / terminated / truncated / info
#   observe_egocentric   → rgb, depth, pose, intrinsics, raw_obs
#   observe_panorama     → views, directions, n_views
#   evaluate             → metrics, success, spl   (thin metric sink)
#
# Episode lifecycle (env.reset / episode selection) is env panel-owned
# (HabitatEnvPanel.set_episode); `reset` here only reads metadata. step
# nodes return NO observation — perception is pulled via the observe_* family.
# ══════════════════════════════════════════════════════════════════════


def _habitat_step_info(result: dict) -> dict:
    """Assemble the gym ``info`` dict from a manager step result."""
    info: dict[str, Any] = {
        "step_count": result.get("step_count"),
        "position": result.get("position"),
        "orientation": result.get("orientation"),
    }
    if result.get("metrics"):
        info["metrics"] = result["metrics"]
    if "ksteps" in result:
        info["ksteps"] = result["ksteps"]
    return info


class ResetHabitatTool(BaseCanvasNode):
    node_type = "env_habitat__reset"
    display_name = "Habitat: Reset"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Ensure a live episode (re-arm if done) — emit instruction + ids, no observation"
    category = "environment"
    icon = "RotateCcw"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger reset (any value)", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("episode_id", "TEXT", "Episode ID"),
        PortDef("scene_id", "TEXT", "Scene ID"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Episode placement is env panel-owned (set_episode); reset ensures the
        # placed episode is live. A done episode (canvas re-run) is re-armed in
        # place; a live one — the batch-eval path, where the runner just placed
        # a fresh episode — is read without disturbance. Rollover lives HERE,
        # never in observe_* (pure reads): a post-loop re-fire of an observe
        # must not silently roll the env into a new episode under evaluate.
        mgr = _get_env()
        if mgr._episode_done:
            await _run_sync(mgr.reset_episode)
        info = await _run_sync(mgr.get_episode_info)
        self._self_log("episode_id", info.get("episode_id"))
        self._self_log("instruction", info.get("instruction", "")[:200])
        self._self_log("scene_id", info.get("scene_id"))
        return {
            "instruction": info.get("instruction", ""),
            "episode_id": str(info.get("episode_id", "")),
            "scene_id": str(info.get("scene_id", "")),
        }


class StepDiscreteHabitatTool(BaseCanvasNode):
    node_type = "env_habitat__step_discrete"
    display_name = "Habitat: Step (discrete)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Advance one tick with a discrete action (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT)"
    category = "environment"
    icon = "Play"
    input_ports = [
        PortDef("action", "ACTION", "Discrete action (0-3)"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called / goal reached"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Per-step diagnostics + terminal metrics"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        action = int(inputs.get("action", 1))
        result = await _run_sync(_get_env().step, action)
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        reward = result.get("reward", 0.0)
        info = _habitat_step_info(result)

        from app.standard.actions import ACTION_NAMES

        self._self_log("action", action)
        self._self_log("action_name", ACTION_NAMES.get(action, "UNKNOWN"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        self._self_log("reward", reward)
        self._self_log("step_count", result.get("step_count"))
        if (terminated or truncated) and result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class StepPoseHabitatTool(BaseCanvasNode):
    node_type = "env_habitat__step_pose"
    display_name = "Habitat: Step (pose)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Navigate toward a target SE(3) pose via shortest-path following"
    category = "environment"
    icon = "Navigation"
    input_ports = [
        PortDef("target", "POSE", "Target position [x, y, z] or pose dict"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called / goal reached"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Diagnostics + terminal metrics + final pose"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        target_pose = inputs.get("target", {})
        target = target_pose.get("position") if isinstance(target_pose, dict) else target_pose
        env_mgr = _get_env()
        if target is None:
            return {"reward": 0.0, "terminated": False, "truncated": False, "info": {}}

        def _navigate() -> dict:
            with env_mgr._lock:
                if env_mgr._env is None:
                    return {"error": "Environment not initialized"}

                sim = env_mgr._env._env.sim
                try:
                    from habitat.tasks.nav.shortest_path_follower import (
                        ShortestPathFollower,
                    )

                    follower = ShortestPathFollower(sim, goal_radius=0.36, return_one_hot=False)
                except Exception:
                    log.warning("ShortestPathFollower unavailable — returning target only")
                    return {"done": False, "pose": None}

                goal_pos = np.array(target, dtype=np.float32)
                path: list = []
                steps = 0
                max_nav_steps = 50

                while steps < max_nav_steps and not env_mgr._episode_done:
                    try:
                        action = follower.get_next_action(goal_pos)
                    except Exception:
                        break
                    if action is None or action == 0:
                        break

                    obs, _reward, done, _info = env_mgr._env.step(action)
                    env_mgr._current_obs = obs
                    env_mgr._episode_done = done
                    env_mgr._step_count += 1
                    steps += 1

                    state = sim.get_agent_state()
                    pos = state.position.tolist()
                    path.append(pos)

                    if done:
                        break

                    dist = np.linalg.norm(np.array(pos) - goal_pos)
                    if dist < 0.5:
                        break

            final_state = sim.get_agent_state()
            return {
                "done": env_mgr._episode_done,
                "pose": {
                    "position": final_state.position.tolist(),
                    "orientation": [
                        float(final_state.rotation.x),
                        float(final_state.rotation.y),
                        float(final_state.rotation.z),
                        float(final_state.rotation.w),
                    ],
                },
            }

        nav = await asyncio.get_running_loop().run_in_executor(env_mgr.executor, _navigate)
        if "error" in nav:
            return {"reward": 0.0, "terminated": False, "truncated": False, "info": {}}

        done = bool(nav.get("done", False))
        terminated, truncated = env_mgr._split_done(done)
        info: dict[str, Any] = {"step_count": env_mgr._step_count, "pose": nav.get("pose")}
        if done:
            metrics = await _run_sync(env_mgr.evaluate)
            if isinstance(metrics, dict) and "error" not in metrics:
                info["metrics"] = metrics
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        return {
            "reward": 0.0,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class StepHighToLowHabitatTool(BaseCanvasNode):
    """Open-Nav HIGHTOLOW action — rotate by angle then walk distance.

    Mirrors the custom Habitat action registered in
    ``Open-Nav/habitat_extensions/nav.py``: the agent yaw is rotated by
    ``angle`` radians, then a series of MOVE_FORWARD primitive actions walks
    it ``distance`` metres (``ksteps = int(distance / 0.25)``). This is the
    action interface the Open-Nav waypoint predictor was trained against.
    """

    node_type = "env_habitat__step_hightolow"
    display_name = "Habitat: Step (HIGHTOLOW)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Rotate by angle (rad) then walk distance (m) — Open-Nav HIGHTOLOW"
    category = "environment"
    icon = "Compass"
    input_ports = [
        PortDef("angle", "TEXT", "Rotation in radians (yaw) — scalar float"),
        PortDef("distance", "TEXT", "Forward distance in metres — scalar float"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called / goal reached"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Diagnostics + terminal metrics (incl. ksteps)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        angle = float(inputs.get("angle", 0.0))
        distance = float(inputs.get("distance", 0.0))
        result = await _run_sync(_get_env().step_hightolow, angle, distance)
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        info = _habitat_step_info(result)
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)
        self._self_log("ksteps", result.get("ksteps"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        if (terminated or truncated) and result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class StepPathHabitatTool(BaseCanvasNode):
    """AO-Planner multi-point Path execution — walk a polar route with try-out sliding.

    Faithful port of ``environments_llm.multi_step_control``/``single_step_control``:
    the path is a polar sequence (``path_angles[i]``, ``path_distances[i]``) in the
    ``step_hightolow`` convention; each hop is walked via discrete TURN/MOVE
    primitives, re-deriving the bearing from the actual pose, with try-out obstacle
    sliding (when ``ALLOW_SLIDING`` is off). An empty path means STOP.
    """

    node_type = "env_habitat__step_path"
    display_name = "Habitat: Step (Path)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="cyan",
        config_fields=[
            ConfigField("tryout", "boolean", "Obstacle try-out sliding on collision (AO-Planner)", default=True),
        ],
    )
    description = "Walk a multi-point polar Path with try-out sliding — AO-Planner VAP execution"
    category = "environment"
    icon = "Route"
    input_ports = [
        PortDef("path_angles", "TEXT", "JSON list of per-hop yaw radians (relative to current heading)"),
        PortDef("path_distances", "TEXT", "JSON list of per-hop ground distances (m)"),
    ]
    output_ports = [
        PortDef("reward", "ANY", "Per-step reward (scalar)"),
        PortDef("terminated", "BOOL", "MDP terminal: STOP called / goal reached"),
        PortDef("truncated", "BOOL", "Step-budget cutoff"),
        PortDef("info", "ANY", "Diagnostics + terminal metrics (incl. n_hops)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import json

        def _parse(v: Any) -> list:
            if isinstance(v, list):
                return v
            try:
                x = json.loads(v or "[]")
                return x if isinstance(x, list) else []
            except Exception:  # noqa: BLE001
                return []

        angles = [float(a) for a in _parse(inputs.get("path_angles"))]
        distances = [float(d) for d in _parse(inputs.get("path_distances"))]
        tryout = bool((getattr(self, "config", None) or {}).get("tryout", True))
        result = await _run_sync(_get_env().step_path, angles, distances, tryout)
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        info = _habitat_step_info(result)
        self._self_log("n_hops", result.get("n_hops"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        if (terminated or truncated) and result.get("metrics"):
            self._self_log("metrics", result["metrics"])
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }


class ObserveEgocentricHabitatTool(BaseCanvasNode):
    node_type = "env_habitat__observe_egocentric"
    display_name = "Habitat: Observe (egocentric)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Pull the current first-person observation: RGB, depth, pose, intrinsics"
    category = "environment"
    icon = "Eye"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Current RGB observation"),
        PortDef("depth", "DEPTH", "Current depth map"),
        PortDef("pose", "POSE", "Agent position + orientation"),
        PortDef("intrinsics", "ANY", "Camera intrinsics {fx,fy,cx,cy,width,height} or None"),
        PortDef("raw_obs", "ANY", "Raw Habitat observation dict (for policy nodes)"),
        PortDef(
            "instruction_text",
            "TEXT",
            "Raw natural-language instruction for the current episode (per-episode "
            "metadata; feed a tokenizer node, e.g. policy_vlnce__tokenize_instruction)",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Pure read — no lifecycle action. A finished episode observed again
        # returns the terminal frame; re-arming is reset's job.
        mgr = _get_env()
        raw = await _run_sync(mgr.get_raw_obs)
        rgb = depth = None
        if raw:
            for k in ("rgb", "RGB"):
                if k in raw:
                    rgb = np.asarray(raw[k], dtype=np.uint8)
                    break
            for k in ("depth", "DEPTH"):
                if k in raw:
                    depth = np.asarray(raw[k], dtype=np.float32).squeeze()
                    break
        pose = await _run_sync(mgr.get_agent_state)
        if not isinstance(pose, dict) or "error" in pose:
            pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        intrinsics = mgr.get_cam_intrinsics()
        instruction_text = await _run_sync(mgr.get_instruction_text)
        self._self_log("has_rgb", rgb is not None)
        self._self_log("has_depth", depth is not None)
        if rgb is not None:
            self._self_log("rgb_shape", list(rgb.shape))
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "intrinsics": intrinsics,
            "raw_obs": raw,
            "instruction_text": instruction_text,
        }


class ObserveCameraPoseHabitatTool(BaseCanvasNode):
    """Return the agent's current world-frame camera pose.

    Exposes the position + quaternion rotation that ``sim.get_agent_state()``
    already tracks internally.  Consumers (e.g. ``project_waypoints``) can use
    these to convert pixel→camera-frame coordinates into absolute world
    coordinates, matching upstream ``pixel_to_world`` (``utils.py:87-130``).

    Orientation convention: ``[x, y, z, w]`` quaternion (habitat default).
    """

    node_type = "env_habitat__observe_camera_pose"
    display_name = "Habitat: Camera Pose"
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
        mgr = _get_env()
        state = await _run_sync(mgr.get_agent_state)
        if isinstance(state, dict) and "error" not in state:
            pos = state.get("position", [0.0, 0.0, 0.0])
            ori = state.get("orientation", [0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        else:
            pos, ori = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        self._self_log("position", pos)
        self._self_log("rotation", ori)
        return {"position": pos, "rotation": ori}


class ObservePanoramaHabitatTool(BaseCanvasNode):
    """Multi-view aligned RGB-D panorama at the agent's position.

    Returns per-direction RGB and depth base64 strings keyed by ``dir_id``
    (0 = current heading, increasing clockwise in 30° steps for
    ``n_views=12``). Consumed by the Open-Nav waypoint predictor and scene
    perception nodes. Expensive (one render per view) — pull on demand only.
    """

    node_type = "env_habitat__observe_panorama"
    display_name = "Habitat: Observe (panorama)"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    description = "Multi-view aligned RGB+Depth panorama at the agent's position"
    category = "environment"
    icon = "Camera"
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-render (optional)", optional=True),
    ]
    output_ports = [
        PortDef(
            "views",
            "ANY",
            "List of {dir_id, heading_deg, rgb_base64, depth_base64} (views_rgbd mode)",
        ),
        PortDef("directions", "TEXT", "Per-view heading labels (JSON)"),
        PortDef("n_views", "ANY", "Number of views returned"),
        PortDef("composite", "IMAGE", "Stitched panorama grid (composite mode)"),
    ]
    config_fields = [
        ConfigField(
            name="representation",
            field_type="select",
            label="Representation",
            default="views_rgbd",
            options=[
                {"value": "views_rgbd", "label": "Aligned RGB-D views (waypoint predictor)"},
                {"value": "composite", "label": "Stitched composite grid (single image)"},
            ],
        ),
        ConfigField(
            name="n_views",
            field_type="slider",
            label="Number of views",
            default=12,
            min=4,
            max=24,
            step=4,
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import json

        cfg = self.config or {}
        representation = cfg.get("representation", "views_rgbd")
        n_views = int(cfg.get("n_views", 12))
        # Pure read — initialize() already resets to the first episode, so a
        # live manager never has _current_obs unset; no lifecycle action here.
        mgr = _get_env()

        if representation == "composite":
            result = await _run_sync(mgr.render_panorama, n_views)
            if "error" in result:
                self._self_log("error", result["error"])
                return {"views": [], "directions": "[]", "n_views": 0, "composite": None}
            composite_arr = None
            composite_b64 = result.get("composite_base64", "")
            if composite_b64:
                import io as _io

                from PIL import Image

                raw = base64.b64decode(composite_b64)
                composite_arr = np.asarray(
                    Image.open(_io.BytesIO(raw)).convert("RGB"), dtype=np.uint8
                )
            directions = json.dumps(
                [
                    {"direction": v.get("direction"), "heading_deg": v.get("heading_deg")}
                    for v in result.get("views", [])
                ]
            )
            self._self_log("n_views", result.get("n_views", 0))
            self._self_log(
                "composite_shape", list(composite_arr.shape) if composite_arr is not None else None
            )
            return {
                "views": result.get("views", []),
                "directions": directions,
                "n_views": result.get("n_views", 0),
                "composite": composite_arr,
            }

        result = await _run_sync(mgr.render_panorama_rgbd, n_views)
        views = result.get("views", [])
        directions = json.dumps(
            [{"dir_id": v.get("dir_id"), "heading_deg": v.get("heading_deg")} for v in views]
        )
        self._self_log("n_views", result.get("n_views", 0))
        return {
            "views": views,
            "directions": directions,
            "n_views": result.get("n_views", 0),
            "composite": None,
        }


class EvaluateHabitatTool(BaseCanvasNode):
    """Thin metric sink over the env's terminal state (after-loop band).

    Emits the current habitat task's measurement dict (success, spl, ndtw,
    sdtw, distance_to_goal, path_length). Wire ``trigger`` from the loop's
    ``iter_out.final_stop`` so it fires exactly once when the episode ends
    (env-template decision E); per-step telemetry rides ``step_*.info`` into
    a viewer instead. The pull is read-only and works whether the episode
    ended by ``terminated`` or by step-budget truncation.
    """

    node_type = "env_habitat__evaluate"
    display_name = "Habitat: Evaluate"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    description = "Pull current habitat task metrics (SR/SPL/nDTW/SDTW) without stepping"
    category = "evaluation"
    icon = "BarChart"
    input_ports = [
        PortDef("trigger", "TEXT", "Trigger evaluation (any value)", optional=True),
    ]
    output_ports = [
        PortDef("metrics", "METRICS", "Habitat task metrics dict"),
        PortDef("success", "TEXT", "1 if agent reached goal, 0 otherwise"),
        PortDef("spl", "TEXT", "Success weighted by Path Length"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_env()
        result = await _run_sync(mgr.evaluate)
        if "error" in result:
            self._self_log("error", result["error"])
            return {"metrics": {}, "success": "0", "spl": "0"}

        success = result.get("success", 0)
        spl = result.get("spl", 0)
        self._self_log("success", success)
        self._self_log("spl", f"{spl:.3f}" if isinstance(spl, (int, float)) else str(spl))
        for k in ("ndtw", "sdtw", "distance_to_goal", "path_length"):
            if k in result:
                v = result[k]
                self._self_log(k, f"{v:.3f}" if isinstance(v, (int, float)) else str(v))
        return {
            "metrics": result,
            "success": str(int(success)) if isinstance(success, (int, float)) else "0",
            "spl": f"{spl:.4f}" if isinstance(spl, (int, float)) else "0",
        }


# ══════════════════════════════════════════════════════════════════════
# HabitatEnvPanel — episode/split control plane for the canvas panel
# Implements BaseEnvPanel; registered via EnvHabitatNodeSet.env panel.
# All Habitat-specific control logic lives here, not in any framework file.
# ══════════════════════════════════════════════════════════════════════

import contextlib

# Import is intentional at module load (not lazy). Failure to import means
# the agentcanvas backend itself is broken; we want a hard error.
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)


def _peek_episode_sync(mgr: Any, index: int) -> dict[str, Any]:
    """Read episode info by index without switching (blocking, thread-safe)."""
    try:
        with mgr._lock:
            dataset = mgr._env._env._dataset
            if index < 0 or index >= len(dataset.episodes):
                return {"error": "Index out of range"}
            ep = dataset.episodes[index]
            info: dict[str, Any] = {
                "index": index,
                "episode_id": str(ep.episode_id),
                "scene_id": ep.scene_id,
            }
            if hasattr(ep, "instruction") and ep.instruction:
                info["instruction"] = getattr(
                    ep.instruction, "instruction_text", str(ep.instruction)
                )
            extras = _extract_episode_extras(ep)
            info["language"] = extras["language"]
            info["extras"] = extras
            return info
    except Exception as e:
        return {"error": str(e)}


def _get_current_ep_index(mgr: Any) -> int:
    """Best-effort current episode index."""
    try:
        if mgr._env is None:
            return -1
        current_ep = mgr._env.current_episode
        dataset = mgr._env._env._dataset
        for i, ep in enumerate(dataset.episodes):
            if ep.episode_id == current_ep.episode_id:
                return i
    except Exception:
        pass
    return 0


class HabitatEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for the Habitat VLN-CE environment.

    Three-field cascade (``dataset → split → episode_index``) — mirrors
    ``MP3DEnvPanel``. Changing dataset resets split + episode; changing
    split re-initializes the underlying ``HabitatEnvManager`` against
    the matching VLN-CE YAML (R2R-CE or per-language RxR-CE). Each
    field change emits an ``episode_reset`` signal so any
    ``lifetime="episode"`` state container clears downstream.
    """

    name = "env_habitat"
    display_name = "Habitat VLN-CE"
    fields = [
        EnvPanelField("dataset", "select", "Dataset"),
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
            "dataset": "R2R-CE",
            "split": "val_unseen",
            "episode_index": 0,
        }

    # ── Helpers ──

    def _mgr(self) -> HabitatEnvManager:
        return HabitatEnvManager.get()

    async def _run(self, fn, *args):
        mgr = self._mgr()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(mgr.executor, fn, *args)

    async def _episode_info(self, index: int) -> dict[str, Any]:
        return await self._run(_peek_episode_sync, self._mgr(), index)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "dataset": self._state.get("dataset", ""),
            "split": self._state.get("split", ""),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    # ── Lifecycle hooks ──

    async def on_load(self) -> dict[str, Any]:
        # Server mode: the env runs in a spawned subprocess, so the local
        # HabitatEnvManager singleton is empty by design. Episode control
        # cannot reach the subprocess yet (no control-plane bridge).
        ctx = getattr(self, "_context", {}) or {}
        if ctx.get("mode") == "server":
            return {
                "available": False,
                "dataset": self._state.get("dataset", "R2R-CE"),
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": list(_DATASETS),
                "splits": _dataset_splits(self._state.get("dataset", "R2R-CE")),
                "message": (
                    "Habitat is running in server mode (subprocess). Episode "
                    "control from this panel is not yet supported."
                ),
            }
        mgr = self._mgr()
        dataset = self._state.get("dataset", "R2R-CE")
        splits = _dataset_splits(dataset)

        if mgr._env is None:
            return {
                "available": False,
                "dataset": dataset,
                "split": "",
                "episode_index": 0,
                "episode_count": 0,
                "datasets": list(_DATASETS),
                "splits": splits,
                "message": (
                    "Habitat environment not initialized. Load env_habitat from "
                    "the NodeSet Manager to enable episode control."
                ),
            }

        current_split = ""
        max_steps = 500
        if mgr._config is not None:
            with contextlib.suppress(Exception):
                current_split = mgr._config.TASK_CONFIG.DATASET.SPLIT
            with contextlib.suppress(Exception):
                max_steps = mgr._config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

        # Reconcile cached split: for RxR-CE, the VLN-CE config carries
        # only the base split name (e.g. ``val_unseen``); the display
        # form carries the language suffix. Prefer the cached suffixed
        # value if it starts with the same base.
        cached_split = self._state.get("split", "")
        display_split = cached_split
        if current_split:
            if dataset == "RxR-CE" and cached_split.startswith(current_split + "_"):
                display_split = cached_split
            else:
                display_split = current_split

        total = await self._run(mgr.get_total_episodes)
        current_index = await self._run(_get_current_ep_index, mgr)
        if current_index < 0:
            current_index = self._state.get("episode_index", 0)

        # Refresh cached state so subsequent actions use the latest values.
        self._state["split"] = display_split
        self._state["episode_index"] = current_index

        ep_info = await self._episode_info(current_index)
        return {
            "available": True,
            "dataset": dataset,
            "split": display_split,
            "episode_index": current_index,
            "episode_count": total,
            "datasets": list(_DATASETS),
            "splits": splits,
            "step_budget": max_steps,
            "current_episode": ep_info,
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        if name == "dataset":
            new_dataset = str(value)
            if new_dataset not in _DATASETS:
                return await self.on_load()
            prev_dataset = self._state.get("dataset")
            self._state["dataset"] = new_dataset
            splits = _dataset_splits(new_dataset)
            self._state["split"] = splits[0] if splits else ""
            self._state["episode_index"] = 0
            # If the dataset actually changed, the env (if any) is bound to
            # the OLD YAML and must be re-initialized so subsequent split /
            # episode pushes hit the right scene + sensor config. Without
            # this, an immediately-following split push that happens to
            # equal the new dataset's default split is short-circuited by
            # the equality check below, leaving the env on the prior
            # dataset (e.g. R2R obs shape persists during an RxR eval).
            if prev_dataset and prev_dataset != new_dataset and self._state["split"]:
                await self._switch_split(self._state["split"])
        elif name == "split":
            new_split = str(value)
            if new_split and new_split != self._state.get("split"):
                await self._switch_split(new_split)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                new_index = int(value)
            except (TypeError, ValueError):
                new_index = 0
            self._state["episode_index"] = new_index
            ep_info = await self._episode_info(new_index)
            state = await self.on_load()
            # ``on_load()`` rebinds ``episode_index`` from the env's current
            # episode (last value passed to ``set_episode_by_index``), which
            # clobbers the value we just armed when the env hasn't been
            # reset yet. Restore the just-set value so the subsequent
            # ``on_action("play")`` reads the new index — without this,
            # batch eval reuses episode 0 for every dispatched index.
            self._state["episode_index"] = new_index
            state["episode_index"] = new_index
            state["current_episode"] = ep_info
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
        mgr = self._mgr()
        if name in ("play", "reset"):
            if mgr._env is None:
                return {
                    "ok": False,
                    "side_effect": "none",
                    "error": "Habitat environment not initialized",
                }
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
        mgr = self._mgr()
        if field == "dataset":
            return [{"value": d, "label": d} for d in _DATASETS]
        if field == "split":
            splits = _dataset_splits(self._state.get("dataset", "R2R-CE"))
            return [{"value": s, "label": s} for s in splits]
        if field == "episode_index":
            if mgr._env is None:
                return []
            data = await self._run(mgr.get_episodes_list, 0, 10000)
            episodes = data.get("episodes", []) if isinstance(data, dict) else []
            return [
                {
                    "value": ep.get("index", i),
                    "label": "{}: {}".format(
                        ep.get("index", i),
                        (ep.get("scene_id") or "").split("/")[-1].replace(".glb", ""),
                    ),
                }
                for i, ep in enumerate(episodes)
            ]
        return []

    # ── Split switching ──

    async def _switch_split(self, split: str) -> None:
        """Re-initialize the env for a new display split.

        For R2R-CE the display split equals the VLN-CE base split. For
        RxR-CE the display carries a language suffix (``val_unseen_en``)
        and we resolve (YAML, base split) via ``_resolve_dataset_config``.
        """
        mgr = self._mgr()
        dataset = self._state.get("dataset", "R2R-CE")
        if split not in _dataset_splits(dataset):
            log.warning(
                "HabitatEnvPanel: split %r not valid for dataset %r",
                split,
                dataset,
            )
            return
        try:
            exp_config, base_split = _resolve_dataset_config(dataset, split)
        except ValueError as e:
            log.error("HabitatEnvPanel: %s", e)
            return

        gpu_id = 0
        max_steps = 500
        if mgr._config is not None:
            with contextlib.suppress(Exception):
                gpu_id = mgr._config.SIMULATOR_GPU_IDS[0]
            with contextlib.suppress(Exception):
                max_steps = mgr._config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

        log.info(
            "HabitatEnvPanel: switching to dataset=%s split=%s (config=%s base=%s)",
            dataset,
            split,
            exp_config,
            base_split,
        )
        await self._run(mgr.shutdown)
        mgr._env = None
        await self._run(mgr.initialize, exp_config, base_split, gpu_id, max_steps)
        self._state["split"] = split


# ══════════════════════════════════════════════════════════════════════
# EnvHabitatNodeSet — the unified nodeset
# ══════════════════════════════════════════════════════════════════════


class EnvHabitatNodeSet(BaseNodeSet):
    """Habitat VLN-CE environment as a NodeSet.

    Exposes ``env_mgr`` for graph executor nodes that need direct
    simulator access (envStep, envObserve, iterIn, policyForward, etc.).

    Works both locally and as an auto-hosted server (``?mode=server``).
    """

    name = "env_habitat"
    description = "Habitat-Sim VLN-CE environment"
    server_python = conda_env_python("ac-vlnce", "VLNCE_PYTHON")
    env_panel = HabitatEnvPanel
    parallelism = "replicated"  # Stateful simulator: per-worker scene + agent pose.
    # ADR-028: Habitat steps are physics-bound and typically complete in
    # well under a second; 30s/step is loose headroom that absorbs
    # per-step overhead from policy/VLM contention under high worker_count
    # without prematurely burning the episode wall-clock.
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = HabitatEnvManager.get()

    def get_tools(self) -> list:
        return [
            ResetHabitatTool(),
            StepDiscreteHabitatTool(),
            StepPoseHabitatTool(),
            StepHighToLowHabitatTool(),
            StepPathHabitatTool(),
            ObserveEgocentricHabitatTool(),
            ObserveCameraPoseHabitatTool(),
            ObservePanoramaHabitatTool(),
            EvaluateHabitatTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Initialize the Habitat simulator.

        Kwargs:
            dataset: "R2R-CE" (default) or "RxR-CE". Ignored when
                ``exp_config`` is supplied explicitly.
            split: Display split. For RxR-CE the split must carry a
                language suffix (``val_unseen_en`` / ``_hi`` / ``_te``).
                Default: dataset-appropriate first split.
            exp_config: Explicit VLN-CE config path. When supplied,
                ``dataset`` / ``split`` are not used for routing and the
                YAML's declared split is honoured.
            gpu_id: CUDA device index (default: 0).
            max_steps: Max episode steps (default: 500).
        """
        if self._mgr.initialized:
            log.info("Habitat already initialized — skipping")
            return
        dataset = kwargs.get("dataset", "R2R-CE")
        explicit_config = kwargs.get("exp_config")
        if explicit_config:
            exp_config = explicit_config
            split = kwargs.get("split", "val_unseen")
        else:
            split = kwargs.get(
                "split",
                _dataset_splits(dataset)[0] if _dataset_splits(dataset) else "val_unseen",
            )
            try:
                exp_config, split = _resolve_dataset_config(dataset, split)
            except ValueError as e:
                log.error("EnvHabitatNodeSet.initialize: %s — falling back to R2R-CE", e)
                exp_config, split = _R2R_CONFIG, "val_unseen"
        gpu_id = kwargs.get("gpu_id", 0)
        max_steps = kwargs.get("max_steps", 500)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            self._mgr.initialize,
            exp_config,
            split,
            gpu_id,
            max_steps,
        )
        log.info("EnvHabitatNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        """Return eval metadata for the Habitat VLN-CE environment.

        Splits are union of both datasets so the Eval page can list
        every selectable option. Dynamic info (episode_counts) is
        populated for the currently-loaded dataset only.
        """
        splits = list(dict.fromkeys(_dataset_splits("R2R-CE") + _dataset_splits("RxR-CE")))
        metadata = {
            "env_name": "habitat_vlnce",
            "datasets": list(_DATASETS),
            "splits": splits,
            "episode_counts": {},
            "metrics": ["spl", "success", "ndtw", "sdtw", "path_length", "distance_to_goal"],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": 500,
        }
        if self._mgr.initialized and self._mgr._config is not None:
            try:
                split = self._mgr._config.TASK_CONFIG.DATASET.SPLIT
                env = self._mgr._env
                if env is not None and hasattr(env, "number_of_episodes"):
                    metadata["episode_counts"] = {split: env.number_of_episodes()}
                elif env is not None and hasattr(env, "_env") and hasattr(env._env, "episodes"):
                    metadata["episode_counts"] = {split: len(env._env.episodes)}
            except Exception:
                pass
        return metadata

    async def shutdown(self) -> None:
        self._mgr.shutdown()
