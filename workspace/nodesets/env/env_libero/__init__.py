from __future__ import annotations

"""EnvLiberoNodeSet — LIBERO manipulation benchmark as a NodeSet.

Wraps the LIBERO suite (ICRA 2024, Liu et al., arXiv:2306.03310) — 130
manipulation tasks split across 5 benchmarks (libero_spatial / object /
goal / 10 / 90), each with 50 init-state variations, executed on a Panda
arm in robosuite + MuJoCo.

Architecture — three-layer pattern mirroring ``hmeqa.py``:

1. ``LiberoEnvManager`` (singleton)
     One ``OffScreenRenderEnv`` + ``LiberoWrapper`` instance, rebuilt on
     every ``set_episode`` (each task uses a different BDDL file).
     Single-thread executor enforces robosuite/MuJoCo GL affinity.

2. Gym-like canvas tool nodes (see env/template.html for the contract)
     env_libero__reset              — episode start (metadata only, no obs);
                                      applies init_state + runs
                                      num_steps_wait physics-settle steps
     env_libero__step_continuous    — runtime-variable-length 7-D action
                                      chunk (JSON list-of-lists), NaN-clips,
                                      executes K steps with early-break on
                                      success; control signals only
     env_libero__step_pose          — absolute EE waypoint via closed-loop
                                      OSC convergence; control signals only
     env_libero__observe_egocentric — pull agentview/wrist RGB + proprio
     env_libero__observe_objects    — pull privileged GT scene snapshot
                                      (BDDL names, AABB PCs, EE pose,
                                      bounds) — feeds GT planners (VoxPoser)
     env_libero__evaluate           — post-hoc success boolean + metrics

3. ``EnvLiberoNodeSet`` (collection + lifecycle)
     server_python defaults to ``$LIBERO_PYTHON`` (the env created by
     ``scripts/install/install_ac_libero.sh``). ``parallelism="replicated"``
     because the robosuite scene is per-worker stateful.

Action contract (TEXT JSON):
    Either a single 7-vec or a list of 7-vecs (chunk). K is runtime-variable.
        "[ax, ay, az, arx, ary, arz, grip]"
        "[[ax, ay, az, arx, ary, arz, grip], ...]"
    Indices: 0-2 delta-pos, 3-5 delta-axis-angle, 6 gripper (-1=close, +1=open).
    NaN/Inf clipped to [-1, 1] before stepping.

Observation bundle (pulled via observe_egocentric):
    agentview_image  (IMAGE)        — H×W×3 uint8 (180° flipped to match training)
    wrist_image      (IMAGE)        — H×W×3 uint8 (180° flipped)
    observation      (LIST[IMAGE])  — [agentview_image, wrist_image] (Tier-1)
    state            (ANY)          — 8-D float32: eef_pos + axis-angle + gripper_qpos
    pose             (POSE)         — None (LIBERO is manipulation, no nav pose)
    instruction      (TEXT)         — task language (Tier-1)
    episode_id       (TEXT)         — "{suite}/{task_id}/{episode_id}"
    suite, task_id, max_steps, step_index, reward, success, done, truncated

Data layout:
    data/libero/                          — asset namespace anchor
      └─ datasets/                        — symlink to LIBERO HDF5 datasets
      └─ bddl/                            — auto-resolved via libero.libero.get_libero_path

last updated: 2026-06-10 (gym-like interface + VoxPoser decoupling)
"""


import asyncio
import concurrent.futures
import json
import logging
import os
import pathlib
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


log = logging.getLogger("agentcanvas.libero")


# ══════════════════════════════════════════════════════════════════════
# Paths & defaults
# ══════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_DATA_ROOT = os.environ.get(
    "LIBERO_DATA_ROOT", os.path.join(_REPO_ROOT, "data", "libero")
)

_SUITE_NAMES: list[str] = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]

# Per-suite max episode steps (longest training demo × ~1.5–2.5).
# Source: libero_runner.py:33-39 in vlaworkspace.
# Bumped 1500→2500 (2026-05-16) because VoxPoser's bounded-advance OSC
# (5mm/tick) takes ~2x more env ticks than VLA's per-step deltas (~10mm).
# Smoke 20260516_134041 truncated mid-transport at 1500. VLA evaluations
# typically converge well under 1500, so the extra budget is benign for
# them but unblocks LMP-driven methods.
_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 2500,
    "libero_object": 2500,
    "libero_goal": 2500,
    "libero_10": 2500,
    "libero_90": 2500,
}

_DEFAULTS: dict[str, Any] = {
    "resolution": 256,        # camera_heights / camera_widths for OffScreenRenderEnv
    "num_steps_wait": 10,     # dummy gripper-close steps after reset (physics settle)
    "seed": 42,
}

# Fixed dummy action used during num_steps_wait. Indices 0-5 are zero
# delta; index 6 is -1.0 (gripper closed).
_DUMMY_ACTION: list[float] = [0.0] * 6 + [-1.0]


# ══════════════════════════════════════════════════════════════════════
# LiberoEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


# ── temporary demo-recording tap (guarded by outputs/_record/ENABLE) ──────
# Dumps per-step camera frames to outputs/_record/<subdir>/ep_<id>/ ONLY when
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


class LiberoEnvManager:
    """Singleton env manager for LIBERO.

    Each task uses a different BDDL file, so the underlying env is
    rebuilt on every ``set_episode``. All robosuite calls run on a
    pinned single-thread executor for GL/physics thread affinity.
    """

    _instance: LiberoEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="libero",
        )

        # Static state (loaded once on initialize).
        self._initialized: bool = False
        self._config: dict[str, Any] = dict(_DEFAULTS)
        self._benchmark_dict: dict[str, Any] | None = None
        self._task_count_by_suite: dict[str, int] = {}

        # Episode-scoped state (rebuilt on set_episode).
        self._suite: str = ""
        self._task_id: int = -1
        self._episode_id: int = -1
        self._task: Any = None              # libero task descriptor
        self._instruction: str = ""
        self._initial_states: np.ndarray | None = None  # for current task
        self._max_steps: int = 0
        self._wrapper: Any = None           # LiberoWrapper instance
        self._step_index: int = 0
        self._cumulative_reward: float = 0.0
        self._success: bool = False
        self._done: bool = False
        self._last_obs: dict[str, Any] | None = None

    def _log_env_step(self, source: str, reward: float = 0.0, done: bool = False, success: bool = False) -> None:
        """Per-env-step log shared by all wrapper.step() call sites.

        Step index reflects post-increment value (current step just executed).
        """
        log.info(
            "LIBERO env step=%d/%d source=%s reward=%.4f cum=%.3f success=%s done=%s",
            self._step_index,
            self._max_steps,
            source,
            float(reward),
            self._cumulative_reward,
            success,
            done,
        )

    # ── Singleton + lifecycle ──────────────────────────────────────────

    @classmethod
    def get(cls) -> LiberoEnvManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, **kwargs: Any) -> None:
        """Lazy-import LIBERO benchmark dict; cache per-suite task counts."""
        with self._lock:
            self._config.update({k: v for k, v in kwargs.items() if k in _DEFAULTS})
            from libero.libero import benchmark  # heavy import — env scope

            bdict = benchmark.get_benchmark_dict()
            counts: dict[str, int] = {}
            for s in _SUITE_NAMES:
                if s in bdict:
                    counts[s] = bdict[s]().n_tasks
                else:
                    log.warning("LIBERO benchmark missing suite %s", s)
                    counts[s] = 0
            self._benchmark_dict = bdict
            self._task_count_by_suite = counts
            self._initialized = True
            log.info(
                "LiberoEnvManager: initialized — task counts %s",
                {k: v for k, v in counts.items()},
            )

    def shutdown(self) -> None:
        with self._lock:
            self._close_wrapper_unlocked()
            self._benchmark_dict = None
            self._task_count_by_suite = {}
            self._initialized = False

    def _close_wrapper_unlocked(self) -> None:
        if self._wrapper is not None:
            try:
                self._wrapper.close()
            except Exception:  # noqa: BLE001
                log.debug("LiberoWrapper.close() raised (non-fatal)", exc_info=True)
            self._wrapper = None

    # ── Inspection ─────────────────────────────────────────────────────

    def list_suites(self) -> list[str]:
        return list(_SUITE_NAMES)

    def list_tasks(self, suite: str) -> list[dict[str, Any]]:
        """Per-task metadata for a suite. Safe on uninitialized env."""
        if not self._initialized or self._benchmark_dict is None:
            return []
        if suite not in self._benchmark_dict:
            return []
        ts = self._benchmark_dict[suite]()
        out: list[dict[str, Any]] = []
        for tid in range(ts.n_tasks):
            task = ts.get_task(tid)
            try:
                init_states = ts.get_task_init_states(tid)
                n_init = int(init_states.shape[0]) if hasattr(init_states, "shape") else len(init_states)
            except Exception:  # noqa: BLE001
                n_init = 0
            out.append(
                {
                    "task_id": tid,
                    "language": str(getattr(task, "language", "")),
                    "n_init_states": n_init,
                    "max_steps": _SUITE_MAX_STEPS.get(suite, 280),
                }
            )
        return out

    def get_total_episodes(self, suite: str) -> int:
        return sum(t["n_init_states"] for t in self.list_tasks(suite))

    # ── Episode control ────────────────────────────────────────────────

    def set_episode(self, suite: str, task_id: int, episode_id: int) -> dict[str, Any]:
        """Tear down any live env, build a fresh one for (suite, task, ep)."""
        with self._lock:
            if not self._initialized or self._benchmark_dict is None:
                return {"error": "LIBERO not initialized — call initialize() first"}
            if suite not in self._benchmark_dict:
                return {"error": f"unknown suite '{suite}'"}

            ts = self._benchmark_dict[suite]()
            if task_id < 0 or task_id >= ts.n_tasks:
                return {"error": f"task_id {task_id} out of range (0, {ts.n_tasks})"}

            task = ts.get_task(task_id)
            try:
                init_states = ts.get_task_init_states(task_id)
            except Exception as e:  # noqa: BLE001
                return {"error": f"get_task_init_states failed: {e!r}"}
            if init_states is None or len(init_states) == 0:
                return {"error": f"no init_states for {suite}/{task_id}"}

            n_states = int(init_states.shape[0]) if hasattr(init_states, "shape") else len(init_states)
            ep_idx_resolved = int(episode_id) % n_states  # tolerate over-index

            # Resolve BDDL file path.
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            bddl_root = pathlib.Path(get_libero_path("bddl_files"))
            bddl_path = str(bddl_root / task.problem_folder / task.bddl_file)
            if not os.path.isfile(bddl_path):
                return {"error": f"BDDL file missing: {bddl_path}"}

            # Tear down prev env (different BDDL → fresh sim).
            self._close_wrapper_unlocked()

            resolution = int(self._config["resolution"])
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_heights=resolution,
                    camera_widths=resolution,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("env init failed for %s", bddl_path)
                return {"error": f"env init failed: {e!r}"}

            # robosuite's underlying env defaults to horizon=1000 and returns
            # done=True at horizon (truncation). LiberoWrapper.step then mislabels
            # this as info["success"]=True (line 195), so our manager thinks the
            # task succeeded and refuses further actions ("executing action in
            # terminated episode"). For LMP-driven methods (VoxPoser) that take
            # >1000 ticks to complete a pick-and-place, we'd never finish the
            # plan. Bump horizon to match _SUITE_MAX_STEPS so our manager's
            # own _step_index >= _max_steps check fires first.
            # Diagnosed in run 20260516_134457: trajectory truncated at waypoint 3
            # of 17 in "move to plate" phase, num_steps=990 ≈ default horizon.
            try:
                env.env.horizon = _SUITE_MAX_STEPS.get(suite, 280) + 100
            except Exception:  # noqa: BLE001
                pass

            # Lazy import — avoids import-time dep on robosuite at module load.
            from ._wrapper import LiberoWrapper

            wrapper = LiberoWrapper(
                env,
                render_hw=(resolution, resolution),
            )
            wrapper.init_state = init_states[ep_idx_resolved]
            try:
                wrapper.seed(int(self._config["seed"]))
            except Exception:  # noqa: BLE001
                pass

            # Reset → applies init_state, returns processed obs.
            obs = wrapper.reset()

            # Physics settle: gripper-close dummy actions.
            num_wait = int(self._config["num_steps_wait"])
            for _ in range(num_wait):
                obs, _, _, _ = wrapper.step(_DUMMY_ACTION)

            # Episode bookkeeping.
            self._suite = suite
            self._task_id = task_id
            self._episode_id = ep_idx_resolved
            self._task = task
            self._instruction = str(getattr(task, "language", ""))
            self._initial_states = init_states
            self._max_steps = _SUITE_MAX_STEPS.get(suite, 280)
            self._wrapper = wrapper
            self._step_index = 0
            self._cumulative_reward = 0.0
            self._success = False
            self._done = False
            self._last_obs = obs

            log.info(
                "LIBERO: episode set %s/task=%d/ep=%d (n_init=%d, max_steps=%d) — %s",
                suite, task_id, ep_idx_resolved, n_states, self._max_steps,
                self._instruction[:60],
            )
            return self._bundle_unlocked()

    # ── Stepping ───────────────────────────────────────────────────────

    def step_chunk(self, action_chunk: np.ndarray) -> dict[str, Any]:
        """Execute a (K, 7) action chunk; early-break on success.

        ``action_chunk`` may also be (7,) — broadcast to (1, 7) by caller.
        Returns the observation bundle for the *last* step taken.
        """
        with self._lock:
            if self._wrapper is None:
                return {"error": "no active env — call set_episode first"}

            arr = np.asarray(action_chunk, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[1] != 7:
                return {
                    "error": f"action shape {tuple(arr.shape)} invalid; "
                    "expected (K, 7) or (7,)"
                }

            # NaN/Inf clip — robosuite rejects non-finite actions.
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = np.clip(arr, -1.0, 1.0)

            for k in range(arr.shape[0]):
                obs, reward, done, info = self._wrapper.step(arr[k])
                self._step_index += 1
                self._cumulative_reward += float(reward)
                self._last_obs = obs
                _rec_save(
                    "libero",
                    "%s_%s_%s" % (self._suite, self._task_id, self._episode_id),
                    self._step_index,
                    obs.get("agentview_image") if isinstance(obs, dict) else None,
                )
                _succ = bool(info.get("success", False))
                self._log_env_step("step_chunk", reward=reward, done=bool(done), success=_succ)
                if _succ or bool(done):
                    self._success = True
                    self._done = True
                    break
                if self._step_index >= self._max_steps:
                    self._done = True
                    break

            return self._bundle_unlocked()

    # ── Observations / metadata ────────────────────────────────────────

    def current_obs(self) -> dict[str, Any]:
        with self._lock:
            if self._wrapper is None or self._last_obs is None:
                return {"error": "no active env"}
            return self._bundle_unlocked()

    def current_episode(self) -> dict[str, Any]:
        with self._lock:
            if self._task_id < 0:
                return {"error": "no active episode"}
            return {
                "suite": self._suite,
                "task_id": int(self._task_id),
                "episode_id": str(self._episode_id),
                "instruction": self._instruction,
                "max_steps": int(self._max_steps),
                "step_index": int(self._step_index),
                "cumulative_reward": float(self._cumulative_reward),
                "success": bool(self._success),
                "done": bool(self._done),
            }

    def _bundle_unlocked(self) -> dict[str, Any]:
        """Build the observation bundle returned by reset / step / observe.

        Caller must hold the lock.
        """
        obs = self._last_obs or {}
        agent_img = obs.get("agentview_image")
        wrist_img = obs.get("wrist_image")
        state = obs.get("state")
        truncated = (
            self._done
            and not self._success
            and self._step_index >= self._max_steps
        )
        return {
            "agentview_image": agent_img,
            "wrist_image": wrist_img,
            # Tier-1 contract: ``observation`` is a list of images so the
            # canonical reset/step bundle is uniform across env types.
            "observation": (
                [agent_img, wrist_img]
                if (agent_img is not None and wrist_img is not None)
                else None
            ),
            "state": state,
            # Tier-1 ``pose`` is None — LIBERO is manipulation, not navigation.
            "pose": None,
            "instruction": self._instruction,
            "episode_id": "{}/{}/{}".format(
                self._suite, self._task_id, self._episode_id
            ),
            "suite": self._suite,
            "task_id": int(self._task_id),
            "max_steps": int(self._max_steps),
            "step_index": int(self._step_index),
            "reward": float(self._cumulative_reward),
            "success": bool(self._success),
            "done": bool(self._done),
            "truncated": bool(truncated),
        }


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> LiberoEnvManager:
    return LiberoEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(
        _get_mgr().executor, fn, *args
    )


def _gym_control_fields(mgr: LiberoEnvManager, *, converged: bool = False) -> dict[str, Any]:
    """Gym-like control signals (reward/terminated/truncated/info + extras)
    read from the manager's current episode state. Shared by step_pose's
    return paths so every exit carries the full step contract."""
    truncated = bool(
        mgr._done and not mgr._success and mgr._step_index >= mgr._max_steps
    )
    return {
        "reward": float(mgr._cumulative_reward),
        "terminated": bool(mgr._done and not truncated),
        "truncated": truncated,
        "success": bool(mgr._success),
        "step_index": int(mgr._step_index),
        "info": {
            "step_index": int(mgr._step_index),
            "success": bool(mgr._success),
            "cumulative_reward": float(mgr._cumulative_reward),
            "converged": converged,
        },
    }


def _parse_action(raw: Any) -> np.ndarray:
    """Parse a step input into a (K, 7) float32 action chunk.

    Accepts:
        - A JSON string of a 7-list or list-of-7-lists
        - A Python list / numpy array (passed straight through)

    Raises ``ValueError`` on bad shape; the step node catches and aborts.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"action TEXT is not valid JSON: {e!r}") from e
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"action shape {tuple(arr.shape)} invalid; "
            "expected 7-vec or list of 7-vecs"
        )
    return arr


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


class ResetLiberoTool(BaseCanvasNode):
    node_type = "env_libero__reset"
    display_name = "LIBERO: Reset"
    description = (
        "Begin episode — re-run set_episode for the env panel-selected "
        "episode and emit metadata only (pull obs via observe_egocentric)."
    )
    category = "environment"
    icon = "RotateCcw"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Task language description"),
        PortDef("episode_id", "TEXT", "{suite}/{task}/{episode}"),
        PortDef("suite", "TEXT", "Active task suite"),
        PortDef("task_id", "ANY", "Task index within suite"),
        PortDef("max_steps", "ANY", "Per-suite max episode steps"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        # Episode placement is env panel-owned; re-run set_episode for the
        # current selection so reset is idempotent. Metadata only — no obs.
        result = await _run_sync(
            mgr.set_episode,
            mgr._suite or _SUITE_NAMES[0],
            mgr._task_id if mgr._task_id >= 0 else 0,
            mgr._episode_id if mgr._episode_id >= 0 else 0,
        )
        if isinstance(result, dict) and "error" in result:
            self._self_log("error", result["error"])
            return {
                "instruction": "", "episode_id": "",
                "suite": "", "task_id": 0, "max_steps": 0,
            }
        self._self_log("episode_id", result.get("episode_id"))
        self._self_log("instruction", (result.get("instruction") or "")[:80])
        self._self_log("max_steps", result.get("max_steps"))
        return {
            "instruction": result.get("instruction", ""),
            "episode_id": result.get("episode_id", ""),
            "suite": result.get("suite", ""),
            "task_id": result.get("task_id", 0),
            "max_steps": result.get("max_steps", 0),
        }


class StepContinuousLiberoTool(BaseCanvasNode):
    node_type = "env_libero__step_continuous"
    display_name = "LIBERO: Step (continuous)"
    description = (
        "Execute a 7-DoF action chunk; returns control signals only (pull obs "
        "via observe_egocentric). Input JSON: a 7-vec or list of 7-vecs "
        "[delta_pos(3), delta_axis_angle(3), gripper(±1)]. K is runtime-"
        "variable; loop aborts early on success."
    )
    category = "environment"
    icon = "Play"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "max_chunk_steps",
                "number",
                label="Max chunk steps",
                default=0,
                placeholder="0 = run full chunk; 8 matches Pi0 replan rate",
            ),
        ],
    )
    input_ports = [
        PortDef(
            "action",
            "TEXT",
            'JSON: "[ax,ay,az,arx,ary,arz,grip]" or "[[…],…]" — chunk length variable',
        ),
    ]
    output_ports = [
        # gym-like contract
        PortDef("reward", "ANY", "Cumulative reward over the episode"),
        PortDef("terminated", "BOOL", "MDP terminal: task success / env-terminal"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (max_steps without success)"),
        PortDef("info", "ANY", "Diagnostics: {step_index, success, cumulative_reward, …}"),
        # libero-specific extras (also inside info)
        PortDef("success", "BOOL", "Task success flag"),
        PortDef("step_index", "ANY", "Step counter"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        raw = inputs.get("action", "")
        try:
            arr = _parse_action(raw)
        except ValueError as e:
            self._self_log("error", f"bad action: {e!s} raw={raw!r}")
            return {
                "reward": 0.0, "terminated": True, "truncated": False,
                "info": {"error": str(e)}, "success": False, "step_index": 0,
            }

        # Optional truncation — Pi0/SmolVLA emit 50-action chunks, but running
        # 50 env steps between re-plans drifts off-policy. Replanning every 8
        # steps (Pi0 default) is what the smoke runner uses and yields 5/5 on
        # libero_spatial. 0 = run full chunk (backward-compat for evals that
        # genuinely want long chunks).
        try:
            max_chunk = int(self.config.get("max_chunk_steps", 0) or 0)
        except (TypeError, ValueError):
            max_chunk = 0
        if 0 < max_chunk < arr.shape[0]:
            arr = arr[:max_chunk]

        result = await _run_sync(_get_mgr().step_chunk, arr)
        if isinstance(result, dict) and "error" in result:
            self._self_log("error", result["error"])
            return {
                "reward": 0.0, "terminated": True, "truncated": False,
                "info": {"error": result["error"]}, "success": False, "step_index": 0,
            }
        done = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        terminated = bool(done and not truncated)
        self._self_log("step_index", result.get("step_index"))
        self._self_log("terminated", terminated)
        self._self_log("truncated", truncated)
        self._self_log("success", result.get("success"))
        info = {
            k: result.get(k)
            for k in ("step_index", "success", "instruction", "episode_id", "suite", "task_id")
        }
        info["cumulative_reward"] = float(result.get("reward", 0.0) or 0.0)
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "success": bool(result.get("success", False)),
            "step_index": result.get("step_index", 0),
        }


class ObserveEgocentricLiberoTool(BaseCanvasNode):
    node_type = "env_libero__observe_egocentric"
    display_name = "LIBERO: Observe (egocentric)"
    description = (
        "Pull the current observation: agentview + wrist RGB + proprio state "
        "(read-only, no env step)."
    )
    category = "environment"
    icon = "Eye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Agentview RGB (alias of agentview_image)"),
        PortDef("agentview_image", "IMAGE", "Third-person view (180°-flipped)"),
        PortDef("wrist_image", "IMAGE", "Wrist camera (180°-flipped)"),
        PortDef("state", "ANY", "8-D float32: eef_pos + axis-angle + gripper_qpos"),
        PortDef("observation", "LIST[IMAGE]", "[agentview_image, wrist_image]"),
        PortDef("pose", "POSE", "None (manipulation, no nav pose)"),
        PortDef("intrinsics", "ANY", "Reserved — always None (no camera intrinsics surface)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        cur = await _run_sync(_get_mgr().current_obs)
        if isinstance(cur, dict) and "error" in cur:
            self._self_log("error", cur["error"])
            return {
                "rgb": None, "agentview_image": None, "wrist_image": None,
                "state": None, "observation": [], "pose": None, "intrinsics": None,
            }
        av = cur.get("agentview_image")
        self._self_log("has_rgb", av is not None)
        return {
            "rgb": av,
            "agentview_image": av,
            "wrist_image": cur.get("wrist_image"),
            "state": cur.get("state"),
            "observation": cur.get("observation", []),
            "pose": cur.get("pose"),
            "intrinsics": None,
        }


class EvaluateLiberoTool(BaseCanvasNode):
    node_type = "env_libero__evaluate"
    display_name = "LIBERO: Evaluate"
    description = "Post-hoc success boolean + rollout metrics."
    category = "environment"
    icon = "CheckCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    # Run-end metric harvest: this node must read live env state AFTER the
    # loop so aggregate metrics (esp. num_steps) reflect the actual final
    # step count, not the one-cycle-delayed value the cross-scope dataflow
    # trigger would capture. Declared per-node in the graph JSON via
    # `config.post_loop: true` (see GraphExecutor._post_loop_pass) —
    # voxposer_libero_decomposed.json sets it on its evaluate node.
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("success", "BOOL"),
        PortDef("metrics", "METRICS", "{success, num_steps, suite, task_id, cumulative_reward}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        info = await _run_sync(_get_mgr().current_episode)
        if isinstance(info, dict) and "error" in info:
            self._self_log("error", info["error"])
            return {
                "success": False,
                "metrics": {
                    "success": 0.0, "num_steps": 0,
                    "suite": "", "task_id": 0, "cumulative_reward": 0.0,
                },
            }
        success = bool(info.get("success", False))
        self._self_log("success", success)
        self._self_log("num_steps", info.get("step_index", 0))
        return {
            "success": success,
            "metrics": {
                "success": 1.0 if success else 0.0,
                "num_steps": int(info.get("step_index", 0)),
                "suite": info.get("suite", ""),
                "task_id": int(info.get("task_id", 0)),
                "cumulative_reward": float(info.get("cumulative_reward", 0.0)),
            },
        }


# ══════════════════════════════════════════════════════════════════════
# Shared GT helpers — BDDL parsing + MuJoCo body lookup
# ══════════════════════════════════════════════════════════════════════
#
# Consumed by env_libero__observe_objects (privileged GT snapshot) and
# the EE-control extras (step_pose / reset_to_home / close_gripper).
# Frozen v1 logic mirrored from the retired LiberoVoxPoserAdapter.

import re as _re


def _need_active(mgr: LiberoEnvManager) -> str | None:
    """Return error string if no active episode."""
    if mgr._wrapper is None:
        return "no active episode — call env_libero__reset first"
    return None


def _resolve_body_id(sim: Any, name: str) -> int | None:
    """Mirror LiberoVoxPoserAdapter._resolve_body_id (frozen v1 logic)."""
    model = sim.model
    underscored = name.replace(" ", "_")
    candidates = [name, f"{name}_main", underscored, f"{underscored}_main"]
    for c in candidates:
        try:
            return int(model.body_name2id(c))
        except (ValueError, KeyError):
            continue
    names = list(model.body_names)
    target = underscored.lower()
    for i, n in enumerate(names):
        if n and target in n.lower():
            return int(i)
    if _re.match(r".+_\d+$", underscored):
        stem = _re.sub(r"_\d+$", "", underscored).lower()
        for i, n in enumerate(names):
            if n and stem in n.lower():
                return int(i)
    return None


def _parse_bddl_objects(bddl_path: str) -> list[str]:
    """Mirror LiberoVoxPoserAdapter._parse_bddl_objects (frozen v1 logic)."""
    with open(bddl_path, "r") as f:
        text = f.read()
    out: list[str] = []
    seen: set[str] = set()
    for section in ("objects", "fixtures"):
        m = _re.search(rf"\(:{section}\s+(.+?)\)", text, _re.DOTALL)
        if not m:
            continue
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split(" - ")
            if not parts:
                continue
            for nm in parts[0].split():
                nm = nm.strip()
                if nm and nm not in seen:
                    seen.add(nm)
                    out.append(nm)
    return out


# ── env_libero__observe_objects (gym-like privileged obs space) ───────

# GT snapshot constants — mirror the retired LiberoVoxPoserAdapter (frozen
# v1 logic). AABB-sampled pseudo point clouds are sufficient for voxel-
# occupancy consumers (VoxPoser); no depth/segmentation rendering needed.
#
# x/y bounds are frame-invariant across LIBERO suites (robot base sits at
# the same x/y, workspace extends in front) and were calibrated 2026-05-16
# on libero_spatial task 0. The z bounds are NOT frame-invariant: suites
# anchor the world frame differently — libero_spatial is TABLE-mounted
# (robot base z≈0.91, table top z≈0.90, objects z≈0.87–0.98) while
# libero_object is FLOOR-mounted (robot base z≈0.0, floor z≈0.0, objects
# z≈-0.07–0.15). A single hardcoded z box (the old [0.85, 1.25]) put the
# whole voxel workspace ~0.9 m ABOVE the objects in floor-mounted suites,
# so every grasp target clamped to the unreachable top and SR was 0
# (diagnosed 2026-06-28, run 20260628_115719). Fix: derive the z box per
# scene by anchoring to the robot base body z (≈ the support surface in
# both frames, within ~1 cm), keeping the proven table-frame offsets
# below/above the base.
_GT_BOUNDS_X: tuple[float, float] = (-0.7, 0.3)
_GT_BOUNDS_Y: tuple[float, float] = (-0.4, 0.5)
# z box relative to robot base z. libero_spatial base z=0.912 →
# [0.812, 1.262] (≈ the proven [0.85, 1.25], EE home 1.174 inside);
# libero_object base z=0.0 → [-0.10, 0.35] (EE home 0.248 inside,
# basket bottom -0.068 inside).
_Z_BELOW_BASE = 0.10
_Z_ABOVE_BASE = 0.35
# Fall-back base z if robot0_base is unreadable — the libero_spatial frame
# (table-mounted) keeps the historical [0.85, 1.25]-ish box.
_FALLBACK_BASE_Z = 0.912
_GT_PC_DENSITY = 200


def _robot_base_z(sim: Any) -> float:
    """World-frame z of the robot mount — the scene's support-surface anchor.

    Robust across robosuite/LIBERO frame anchorings (table- vs floor-mounted
    suites). Tries the canonical body name, then any robot base body, then a
    table-frame fall-back."""
    for name in ("robot0_base", "robot0_mount", "base"):
        try:
            return float(sim.data.body_xpos[sim.model.body_name2id(name)][2])
        except (ValueError, KeyError):
            continue
    try:
        for bid in range(sim.model.nbody):
            nm = (sim.model.body_id2name(bid) or "").lower()
            if "robot" in nm and "base" in nm:
                return float(sim.data.body_xpos[bid][2])
    except Exception:  # noqa: BLE001 — fall through to constant
        pass
    return _FALLBACK_BASE_Z


def _workspace_bounds(sim: Any) -> tuple[list[float], list[float]]:
    """Per-scene workspace box: fixed x/y, z anchored to the robot base."""
    base_z = _robot_base_z(sim)
    bounds_min = [_GT_BOUNDS_X[0], _GT_BOUNDS_Y[0], base_z - _Z_BELOW_BASE]
    bounds_max = [_GT_BOUNDS_X[1], _GT_BOUNDS_Y[1], base_z + _Z_ABOVE_BASE]
    return bounds_min, bounds_max


class ObserveObjectsLiberoTool(BaseCanvasNode):
    node_type = "env_libero__observe_objects"
    display_name = "LIBERO: Observe (objects, GT)"
    description = (
        "Pull a privileged ground-truth scene snapshot (read-only, no env "
        "step): BDDL object names, per-object AABB-sampled point clouds + "
        "normals, scene collision PC, EE pose, gripper state, workspace "
        "bounds. Single dict output so GT planners (VoxPoser) read one "
        "atomic state."
    )
    category = "environment"
    icon = "Boxes"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef(
            "snapshot", "ANY",
            "{object_names, object_pcs: {name: {pc, normals}}, scene_pc, "
            "ee_pos, ee_quat, gripper_open, bounds_min, bounds_max, error}",
        ),
        PortDef("error", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        if (err := _need_active(mgr)):
            self._self_log("error", err)
            return {"snapshot": {"error": err}, "error": err}

        def _build():
            from transforms3d.quaternions import mat2quat
            wrapper = mgr._wrapper
            sim = wrapper.env.sim

            def _aabb_pc(bid: int) -> np.ndarray:
                """Uniform sample inside the body's geom AABB (frozen v1 logic:
                LiberoVoxPoserAdapter.get_3d_obs_by_name)."""
                center = np.array(sim.data.body_xpos[bid], dtype=np.float32)
                geom_ids = [
                    g for g in range(sim.model.ngeom)
                    if sim.model.geom_bodyid[g] == bid
                ]
                if geom_ids:
                    geom_centers = np.array(
                        [sim.data.geom_xpos[g] for g in geom_ids], dtype=np.float32
                    )
                    geom_sizes = np.array(
                        [sim.model.geom_size[g] for g in geom_ids], dtype=np.float32
                    )
                    mins = np.min(geom_centers - geom_sizes, axis=0)
                    maxs = np.max(geom_centers + geom_sizes, axis=0)
                else:
                    mins = center - 0.025
                    maxs = center + 0.025
                rng = np.random.default_rng(int(bid))
                return rng.uniform(mins, maxs, size=(_GT_PC_DENSITY, 3)).astype(np.float32)

            # Object names from the active BDDL.
            names: list[str] = []
            task = mgr._task
            if task is not None:
                from libero.libero import get_libero_path
                bddl_root = get_libero_path("bddl_files")
                bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
                if os.path.isfile(bddl_path):
                    names = _parse_bddl_objects(bddl_path)

            # Per-object PCs (+z normals — VoxPoser consumes occupancy only).
            object_pcs: dict[str, Any] = {}
            for name in names:
                bid = _resolve_body_id(sim, name)
                if bid is None:
                    continue
                pc = _aabb_pc(bid)
                normals = np.zeros_like(pc)
                normals[:, 2] = 1.0
                object_pcs[name] = {"pc": pc.tolist(), "normals": normals.tolist()}

            # Scene collision PC over all named non-robot bodies (frozen v1
            # logic: LiberoVoxPoserAdapter.get_scene_3d_obs, ignore_robot).
            scene_chunks: list[np.ndarray] = []
            for bid in range(sim.model.nbody):
                try:
                    name = sim.model.body_id2name(bid) or ""
                except Exception:  # noqa: BLE001 — unnamed body
                    continue
                lname = name.lower()
                if not name or "robot" in lname or "gripper" in lname or "hand" in lname:
                    continue
                scene_chunks.append(_aabb_pc(bid))
            scene_pc = (
                np.concatenate(scene_chunks, axis=0).tolist() if scene_chunks else []
            )

            # EE pose from MuJoCo sites.
            ee_pos = None
            ee_quat = None
            for sname in ("robot0_eef_site", "gripper0_grip_site", "robot0_grip_site"):
                try:
                    sid = sim.model.site_name2id(sname)
                    ee_pos = np.array(sim.data.site_xpos[sid], dtype=np.float32).tolist()
                    mat = sim.data.site_xmat[sid].reshape(3, 3)
                    ee_quat = mat2quat(mat).astype(np.float32).tolist()
                    break
                except (ValueError, KeyError):
                    continue
            if ee_pos is None or ee_quat is None:
                return {"error": "could not locate EE site in MuJoCo model"}

            # Gripper open ≈ positive finger-qpos sum.
            raw = wrapper.env.env._get_observations()
            gq = raw.get("robot0_gripper_qpos")
            gripper_open = bool(gq is not None and float(np.sum(gq)) > 0.04)

            # Workspace bounds anchored to the live scene frame (see
            # _workspace_bounds) — NOT a fixed constant, because suites
            # differ in world-frame z anchoring by ~0.9 m.
            bounds_min, bounds_max = _workspace_bounds(sim)

            return {
                "object_names": names,
                "object_pcs": object_pcs,
                "scene_pc": scene_pc,
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "gripper_open": gripper_open,
                "bounds_min": bounds_min,
                "bounds_max": bounds_max,
                "error": "",
            }

        snapshot = await _run_sync(_build)
        err = snapshot.get("error") or ""
        if err:
            self._self_log("error", err)
        else:
            self._self_log("n_objects", len(snapshot.get("object_names") or []))
        return {"snapshot": snapshot, "error": err}


# ── env_libero__step_pose ─────────────────────────────────────────


# Bounded-step OSC parameters (mirror mono adapter _voxposer/libero_adapter.py:
# _STEP_POS_M / _STEP_ROT_RAD / _OUTPUT_MAX_POS / _OUTPUT_MAX_ROT). Per-substep
# we command the OSC goal to advance at most _STEP_POS_M toward target, then
# scale into delta-space (delta=1 → _OUTPUT_MAX_POS commanded goal shift in
# robosuite OSC_POSE). Without bounding, gain-saturated delta integrates the
# OSC goal_pose far past the physical EE on large errors and causes overshoot.
_STEP_POS_M = 0.005          # max 5mm commanded OSC advance per substep
_STEP_ROT_RAD = 0.05         # max ~3° commanded OSC rotation per substep
_OUTPUT_MAX_POS = 0.05       # robosuite default: delta=1 → 5cm goal shift
_OUTPUT_MAX_ROT = 0.5        # robosuite default: delta=1 → 0.5 rad goal shift


def _quat_log(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion log → axis-angle (3-vec) magnitude is rotation angle in rad."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    w = float(np.clip(q[0], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arccos(w))
    return q[1:] * (angle / den)


def _quat_mul(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a_wxyz
    bw, bx, by, bz = b_wxyz
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_inv(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


class StepPoseLiberoTool(BaseCanvasNode):
    node_type = "env_libero__step_pose"
    display_name = "LIBERO: Step (pose)"
    description = (
        "Drive EE to an absolute world-frame pose via OSC delta convergence "
        "(position AND rotation closed-loop); returns gym control signals "
        "(pull obs via observe_egocentric / observe_objects). Action input is "
        "either a JSON 8-vec '[x, y, z, qw, qx, qy, qz, gripper]' (legacy) "
        "or a JSON object {'pose': [8-vec], 'hold_steps': int}. When "
        "hold_steps > 0, OSC convergence is skipped and the node forces "
        "N zero-delta env ticks with the requested gripper command — "
        "used post-flip to let fingers actually close."
    )
    category = "environment"
    icon = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("max_steps", "number", default=100),
            ConfigField("pos_tol_m", "number", default=0.01),
            ConfigField("rot_tol_rad", "number", default=0.10),
        ],
    )
    input_ports = [
        PortDef("action", "TEXT",
                "JSON 8-vec OR {'pose': [8-vec], 'hold_steps': int}"),
    ]
    output_ports = [
        # gym-like contract
        PortDef("reward", "ANY", "Cumulative reward over the episode"),
        PortDef("terminated", "BOOL", "MDP terminal: task success / env-terminal"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (max_steps without success)"),
        PortDef("info", "ANY", "Diagnostics: {step_index, success, cumulative_reward, converged}"),
        # libero-specific extras (also inside info)
        PortDef("success", "BOOL", "Task success flag"),
        PortDef("step_index", "ANY", "Env step counter"),
        # converged=True iff within tol OR env terminated mid-move; False when
        # the per-call step budget exhausted without reaching tol — error
        # carries the diagnostic (final pos/rot residuals).
        PortDef("converged", "BOOL", "True iff converged or env terminated"),
        PortDef("final_pose", "ANY", "[x,y,z,qw,qx,qy,qz] reached"),
        PortDef("step_count", "ANY", "Env ticks consumed by this call"),
        PortDef("error", "TEXT", "Empty on success; non-empty when budget exhausted without convergence"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        if (err := _need_active(mgr)):
            self._self_log("error", err)
            return {"converged": False, "final_pose": None,
                    "step_count": 0, "error": err,
                    **_gym_control_fields(mgr)}
        raw = inputs.get("action", "")
        hold_steps = 0
        try:
            parsed = (
                json.loads(raw) if isinstance(raw, str) else raw
            )
            if isinstance(parsed, dict):
                pose_vec = parsed.get("pose")
                hold_steps = int(parsed.get("hold_steps", 0) or 0)
                target = np.asarray(pose_vec, dtype=np.float32).reshape(-1)
            else:
                target = np.asarray(parsed, dtype=np.float32).reshape(-1)
            # Empty input = no-op tick (cursor-exhausted overshoot from
            # multi-scope inner loop: dispense_waypoint emits waypoint=None,
            # vp_plan_executor forwards "[]"). Treat as clean done so the
            # composite's check_waypoint_done path stays untainted.
            if target.shape[0] == 0:
                return {"converged": True, "final_pose": None,
                        "step_count": 0, "error": "",
                        **_gym_control_fields(mgr, converged=True)}
            if target.shape[0] != 8:
                raise ValueError(f"expected 8-vec, got {target.shape[0]}")
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            err = f"bad pose input: {e!s}"
            self._self_log("error", err)
            return {"converged": False, "final_pose": None,
                    "step_count": 0, "error": err,
                    **_gym_control_fields(mgr)}

        max_steps = int(self.config.get("max_steps", 100) or 100)
        pos_tol = float(self.config.get("pos_tol_m", 0.01) or 0.01)
        rot_tol = float(self.config.get("rot_tol_rad", 0.10) or 0.10)

        def _move():
            from transforms3d.quaternions import mat2quat
            wrapper = mgr._wrapper
            sim = wrapper.env.sim
            target_xyz = target[:3]
            target_q = target[3:7]  # wxyz
            grip_in = float(target[7])
            # LIBERO/robosuite gripper action convention is +1=CLOSE, -1=OPEN
            # (verified empirically 2026-06-28 on the installed robosuite: 10×
            # action -1 → finger qpos 0.039 = apart/open; 60× action +1 → qpos
            # 0.0005 = together/closed). The VLA step_continuous path sends the
            # policy's native action so it was unaffected, but this pose path
            # was remapping closed→-1 (which actually OPENS), so every VoxPoser
            # grasp descended with the gripper closing and "released" at the
            # object — the can was never captured. grip_in is downstream conv
            # (1.0=closed, 0.0=open). NB: the "−1=close" comments elsewhere in
            # this file are stale doc errors, not the live convention.
            libero_grip = 1.0 if grip_in > 0.5 else -1.0

            def _read_current_pose():
                for sname in ("robot0_eef_site", "gripper0_grip_site", "robot0_grip_site"):
                    try:
                        sid = sim.model.site_name2id(sname)
                        p = np.array(sim.data.site_xpos[sid], dtype=np.float64)
                        m = sim.data.site_xmat[sid].reshape(3, 3)
                        q = mat2quat(m).astype(np.float64)
                        return p, q
                    except (ValueError, KeyError):
                        continue
                return None, None

            # Hold-pose branch: skip OSC convergence; force `hold_steps` env
            # ticks with zero pos/rot delta + the requested gripper command.
            # Mirrors v1 LiberoVoxPoserAdapter's _GRIPPER_TRANSITION_SETTLE
            # (default 60) — open-loop dispatch alone never gives MuJoCo
            # enough ticks for fingers to close on the object.
            if hold_steps > 0:
                steps = 0
                for _ in range(hold_steps):
                    if mgr._done:
                        break
                    obs, reward, done_step, info = wrapper.step(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, libero_grip]
                    )
                    mgr._step_index += 1
                    mgr._cumulative_reward += float(reward)
                    mgr._last_obs = obs
                    _rec_save(
                        "libero",
                        "%s_%s_%s" % (mgr._suite, mgr._task_id, mgr._episode_id),
                        mgr._step_index,
                        obs.get("agentview_image") if isinstance(obs, dict) else None,
                    )
                    steps += 1
                    mgr._log_env_step(
                        "move_to_pose__hold",
                        reward=reward,
                        done=bool(done_step),
                        success=bool(info.get("success", False)),
                    )
                    if done_step:
                        mgr._done = True
                        mgr._success = bool(info.get("success", reward >= 1.0))
                        break
                    if mgr._step_index >= mgr._max_steps:
                        mgr._done = True
                        break
                cp, cq = _read_current_pose()
                final_pose = (
                    cp.tolist() + cq.tolist() if cp is not None and cq is not None else None
                )
                return {
                    "converged": True, "final_pose": final_pose,
                    "step_count": int(steps), "error": "",
                }

            def _read_pose():
                for sname in ("robot0_eef_site", "gripper0_grip_site", "robot0_grip_site"):
                    try:
                        sid = sim.model.site_name2id(sname)
                        p = np.array(sim.data.site_xpos[sid], dtype=np.float64)
                        m = sim.data.site_xmat[sid].reshape(3, 3)
                        q = mat2quat(m).astype(np.float64)
                        return p, q
                    except (ValueError, KeyError):
                        continue
                return None, None

            steps = 0
            cur_pos = None
            cur_quat = None
            pos_dist = float("inf")
            rot_dist = float("inf")
            converged = False
            env_terminated = False
            for _ in range(max_steps):
                if mgr._done:
                    env_terminated = True
                    break
                cur_pos, cur_quat = _read_pose()
                if cur_pos is None or cur_quat is None:
                    return {
                        "converged": False, "final_pose": None,
                        "step_count": int(steps),
                        "error": "could not locate EE site",
                    }

                pos_err = target_xyz.astype(np.float64) - cur_pos
                pos_dist = float(np.linalg.norm(pos_err))
                q_err = _quat_mul(target_q.astype(np.float64), _quat_inv(cur_quat))
                rot_axisangle = _quat_log(q_err)
                rot_dist = float(np.linalg.norm(rot_axisangle))

                if pos_dist < pos_tol and rot_dist < rot_tol:
                    converged = True
                    break

                # Bounded-step OSC delta: command goal to advance at most
                # _STEP_POS_M per substep, then scale into delta-space.
                if pos_dist > _STEP_POS_M:
                    commanded_pos = pos_err * (_STEP_POS_M / pos_dist)
                else:
                    commanded_pos = pos_err
                d_pos = np.clip(commanded_pos / _OUTPUT_MAX_POS, -1.0, 1.0)
                if rot_dist > _STEP_ROT_RAD:
                    commanded_rot = rot_axisangle * (_STEP_ROT_RAD / rot_dist)
                else:
                    commanded_rot = rot_axisangle
                d_rot = np.clip(commanded_rot / _OUTPUT_MAX_ROT, -1.0, 1.0)
                action = [
                    float(d_pos[0]), float(d_pos[1]), float(d_pos[2]),
                    float(d_rot[0]), float(d_rot[1]), float(d_rot[2]),
                    libero_grip,
                ]
                obs, reward, done_step, info = wrapper.step(action)
                mgr._step_index += 1
                mgr._cumulative_reward += float(reward)
                mgr._last_obs = obs
                _rec_save(
                    "libero",
                    "%s_%s_%s" % (mgr._suite, mgr._task_id, mgr._episode_id),
                    mgr._step_index,
                    obs.get("agentview_image") if isinstance(obs, dict) else None,
                )
                steps += 1
                mgr._log_env_step(
                    "move_to_pose",
                    reward=reward,
                    done=bool(done_step),
                    success=bool(info.get("success", False)),
                )
                if done_step:
                    mgr._done = True
                    mgr._success = bool(info.get("success", reward >= 1.0))
                    env_terminated = True
                    break
                if mgr._step_index >= mgr._max_steps:
                    mgr._done = True
                    env_terminated = True
                    break

            # Re-read pose AFTER the final action so reported final_pose
            # reflects the actual landing spot (not the pre-action read).
            # Also lets us check whether the very last action achieved tol.
            if not converged and not env_terminated:
                p, q = _read_pose()
                if p is not None:
                    cur_pos, cur_quat = p, q
                    pos_err = target_xyz.astype(np.float64) - cur_pos
                    pos_dist = float(np.linalg.norm(pos_err))
                    q_err = _quat_mul(target_q.astype(np.float64), _quat_inv(cur_quat))
                    rot_dist = float(np.linalg.norm(_quat_log(q_err)))
                    if pos_dist < pos_tol and rot_dist < rot_tol:
                        converged = True

            final_pos = cur_pos.tolist() if cur_pos is not None else None
            final_quat = cur_quat.tolist() if cur_quat is not None else None
            final_pose = (final_pos + final_quat) if (final_pos and final_quat) else None

            if not converged and not env_terminated:
                err = (
                    f"move_to_pose did not converge in {steps} env steps: "
                    f"pos_err={pos_dist*1000:.1f}mm (tol={pos_tol*1000:.1f}mm), "
                    f"rot_err={rot_dist:.3f}rad (tol={rot_tol:.3f}rad)"
                )
                log.error("LIBERO: %s", err)
                return {
                    "converged": False, "final_pose": final_pose,
                    "step_count": int(steps), "error": err,
                }

            return {
                "converged": True, "final_pose": final_pose,
                "step_count": int(steps), "error": "",
            }

        result = await _run_sync(_move)
        return {
            **result,
            **_gym_control_fields(mgr, converged=bool(result.get("converged", False))),
        }


# ── env_libero__reset_to_home ─────────────────────────────────────


class ResetToHomeLiberoTool(BaseCanvasNode):
    node_type = "env_libero__reset_to_home"
    display_name = "LIBERO: Reset to Home"
    description = (
        "Best-effort home pose: re-apply the initial joint state, settle 5 ticks. "
        "v1 fall-back if init_states is unavailable: 3 zero-delta steps."
    )
    category = "environment"
    icon = "Home"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("trigger", "ANY", "Optional fire trigger", optional=True)]
    output_ports = [
        PortDef("done", "BOOL"),
        PortDef("error", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        if (err := _need_active(mgr)):
            self._self_log("error", err)
            return {"done": False, "error": err}

        def _reset():
            wrapper = mgr._wrapper
            init_states = mgr._initial_states
            ep = mgr._episode_id
            try:
                if init_states is not None and ep is not None and 0 <= ep < len(init_states):
                    wrapper.env.set_init_state(init_states[ep])
                # Settle: a few zero-delta ticks with gripper open.
                for _ in range(5):
                    if mgr._done:
                        break
                    wrapper.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    mgr._step_index += 1
                    mgr._log_env_step("reset_to_home")
                return True
            except Exception as e:  # noqa: BLE001
                return f"reset_to_home failed: {e!r}"

        result = await _run_sync(_reset)
        if isinstance(result, str):
            self._self_log("error", result)
            return {"done": False, "error": result}
        return {"done": True, "error": ""}


# ── env_libero__close_gripper ─────────────────────────────────────


class CloseGripperLiberoTool(BaseCanvasNode):
    node_type = "env_libero__close_gripper"
    display_name = "LIBERO: Close Gripper"
    description = "Apply zero-delta + close-gripper for N settle steps."
    category = "environment"
    icon = "Hand"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("settle_steps", "number", default=5),
        ],
    )
    input_ports = [PortDef("trigger", "ANY", "Optional fire trigger", optional=True)]
    output_ports = [
        PortDef("done", "BOOL"),
        PortDef("error", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        if (err := _need_active(mgr)):
            self._self_log("error", err)
            return {"done": False, "error": err}
        steps = int(self.config.get("settle_steps", 5) or 5)

        def _close():
            wrapper = mgr._wrapper
            for _ in range(steps):
                if mgr._done:
                    break
                wrapper.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                mgr._step_index += 1
                mgr._log_env_step("close_gripper")
            return True

        await _run_sync(_close)
        return {"done": True, "error": ""}


# ══════════════════════════════════════════════════════════════════════
# LiberoEnvPanel — canvas panel env panel
# ══════════════════════════════════════════════════════════════════════


class LiberoEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for LIBERO.

    Three-field cascade: ``suite → task_id → episode_index``. Field
    changes emit the ``episode_reset`` signal so ``lifetime="episode"``
    state containers clear automatically.
    """

    name = "env_libero"
    display_name = "LIBERO"
    fields = [
        EnvPanelField("suite",          "select", "Task Suite"),
        EnvPanelField("task_id",        "select", "Task"),
        EnvPanelField("episode_index",  "select", "Episode"),
    ]
    actions = [
        EnvPanelAction("play",  "Play",  side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop",  "Stop",  side_effect="run_stop",  enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "suite":         _SUITE_NAMES[0],
            "task_id":       0,
            "episode_index": 0,
        }

    def _mgr(self) -> LiberoEnvManager:
        return LiberoEnvManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "suite":         self._state.get("suite", _SUITE_NAMES[0]),
            "task_id":       int(self._state.get("task_id", 0)),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        if not mgr.initialized:
            return {
                "available": False,
                "suite":         self._state.get("suite", _SUITE_NAMES[0]),
                "task_id":       0,
                "episode_index": 0,
                "episode_count": 0,
                "splits":        list(_SUITE_NAMES),
                "message": (
                    "LIBERO not initialized. Load env_libero from the "
                    "NodeSet Manager to enable episode control."
                ),
            }
        suite = self._state.get("suite", _SUITE_NAMES[0])
        task_id = int(self._state.get("task_id", 0))
        ep_idx = int(self._state.get("episode_index", 0))
        tasks = await self._run(mgr.list_tasks, suite)
        episode_count = tasks[task_id]["n_init_states"] if 0 <= task_id < len(tasks) else 0
        return {
            "available":         True,
            "suite":             suite,
            "task_id":           task_id,
            "episode_index":     ep_idx,
            "episode_count":     episode_count,
            "splits":            list(_SUITE_NAMES),
            "step_budget": _SUITE_MAX_STEPS.get(suite, 280),
            "current_episode": {
                "suite":      suite,
                "task_id":    task_id,
                "episode":    ep_idx,
                "instruction":
                    tasks[task_id]["language"] if 0 <= task_id < len(tasks) else "",
            },
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = self._mgr()
        if name == "suite":
            self._state["suite"] = str(value)
            self._state["task_id"] = 0
            self._state["episode_index"] = 0
        elif name == "task_id":
            try:
                tid = int(value)
            except (TypeError, ValueError):
                tid = 0
            self._state["task_id"] = tid
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                idx = int(value)
            except (TypeError, ValueError):
                idx = 0
            self._state["episode_index"] = idx
            if mgr.initialized:
                await self._run(
                    mgr.set_episode,
                    self._state["suite"],
                    int(self._state["task_id"]),
                    idx,
                )
        elif name == "split" and str(value) in _SUITE_NAMES:
            # Eval harnesses push the run's `split` selector through the panel
            # cascade. For LIBERO a split that names a suite selects that suite
            # — this lets a graph eval target e.g. libero_object via
            # `split=libero_object` without a stateful panel-default edit. A
            # non-suite split (e.g. the generic "val_unseen") falls through to
            # the no-op store below and leaves the default suite untouched.
            self._state["suite"] = str(value)
            self._state["task_id"] = 0
            self._state["episode_index"] = 0
        else:
            self._state[name] = value

        if name == "split" and str(value) in _SUITE_NAMES and mgr.initialized:
            await self._run(
                mgr.set_episode,
                self._state["suite"],
                int(self._state["task_id"]),
                int(self._state["episode_index"]),
            )

        # For suite / task changes, also push set_episode so the manager's
        # current scene matches the dropdown (so reset on the canvas is
        # idempotent and reflects the chosen task).
        if name in ("suite", "task_id") and mgr.initialized:
            await self._run(
                mgr.set_episode,
                self._state["suite"],
                int(self._state["task_id"]),
                int(self._state["episode_index"]),
            )

        state = await self.on_load()
        state["side_effect"] = "signal"
        state["signal_name"] = "episode_reset"
        state["signal_payload"] = self._episode_reset_payload()
        return state

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        mgr = self._mgr()
        if name in ("play", "reset"):
            if not mgr.initialized:
                return {"ok": False, "side_effect": "none",
                        "error": "LIBERO not initialized"}
            await self._run(
                mgr.set_episode,
                self._state["suite"],
                int(self._state["task_id"]),
                int(self._state["episode_index"]),
            )
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
        if field == "suite":
            options: list[dict[str, Any]] = []
            for s in _SUITE_NAMES:
                count = mgr._task_count_by_suite.get(s, 0) if mgr.initialized else 0
                label = f"{s} ({count} tasks)" if count else s
                options.append({"value": s, "label": label})
            return options
        if field == "task_id":
            if not mgr.initialized:
                return []
            tasks = await self._run(mgr.list_tasks, self._state.get("suite", _SUITE_NAMES[0]))
            return [
                {
                    "value": t["task_id"],
                    "label": "{}: {}".format(
                        t["task_id"],
                        (t.get("language", "") or "")[:60],
                    ),
                }
                for t in tasks
            ]
        if field == "episode_index":
            if not mgr.initialized:
                return []
            tasks = await self._run(mgr.list_tasks, self._state.get("suite", _SUITE_NAMES[0]))
            tid = int(self._state.get("task_id", 0))
            n = tasks[tid]["n_init_states"] if 0 <= tid < len(tasks) else 0
            return [{"value": i, "label": f"ep {i}"} for i in range(n)]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvLiberoNodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


class EnvLiberoNodeSet(BaseNodeSet):
    """LIBERO manipulation benchmark as a NodeSet.

    Loads in server mode against the ``ac-libero`` conda env by
    default. ``server_python`` reads from ``$LIBERO_PYTHON``; for eval/CI
    point this at the env created by ``scripts/install/install_ac_libero.sh``.
    """

    name = "env_libero"
    description = "LIBERO — manipulation benchmark (5 task suites, 130 tasks)"
    server_python = conda_env_python("ac-libero", "LIBERO_PYTHON")
    env_panel = LiberoEnvPanel
    parallelism = "replicated"  # Per-worker robosuite scene state.
    # Robosuite step ≈ 50 ms; with chunk=K and num_steps_wait=10 the first
    # reset takes ~1 s. 30 s / step is loose headroom that absorbs policy
    # inference + VLM contention under high worker_count.
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = LiberoEnvManager.get()

    def get_tools(self) -> list:
        return [
            # Gym-like core (reset / step_<actionspace> / observe_<obsspace> /
            # evaluate) — see docs/pages/developer-guide/nodesets/env/template.html
            ResetLiberoTool(),
            StepContinuousLiberoTool(),
            StepPoseLiberoTool(),
            ObserveEgocentricLiberoTool(),
            ObserveObjectsLiberoTool(),
            EvaluateLiberoTool(),
            # EE-control extras (sim-mutating helpers outside the gym verbs)
            ResetToHomeLiberoTool(),
            CloseGripperLiberoTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Lazy-load LIBERO benchmark dict. No env opens here — happens on
        first set_episode (i.e. first canvas reset or env panel play).

        Accepted kwargs: resolution, num_steps_wait, seed.
        """
        if self._mgr.initialized:
            log.info("LIBERO already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("EnvLiberoNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        if not self._mgr.initialized:
            counts: dict[str, int] = {s: 0 for s in _SUITE_NAMES}
        else:
            counts = {
                s: self._mgr.get_total_episodes(s) for s in _SUITE_NAMES
            }
        return {
            "env_name": "libero",
            "datasets": ["LIBERO"],
            "splits": list(_SUITE_NAMES),
            "episode_counts": counts,
            "metrics": ["success", "num_steps", "cumulative_reward"],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": max(_SUITE_MAX_STEPS.values()),
        }

    async def shutdown(self) -> None:
        self._mgr.shutdown()
