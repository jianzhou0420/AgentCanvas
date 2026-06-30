from __future__ import annotations

"""EnvSimplerNodeSet — SIMPLER VLA evaluation benchmark as a NodeSet.

Wraps the SIMPLER suite (https://github.com/simpler-env/SimplerEnv) — 25
manipulation tasks split across two embodiments: WidowX/Bridge (4 tasks)
and Google Robot (21 tasks), executed in SAPIEN/ManiSkill2_real2sim.

Architecture — three-layer pattern mirroring ``libero.py``:

1. ``SimplerEnvManager`` (singleton)
     One ``simpler_env.make(task_id)`` instance at a time, rebuilt on
     every ``set_episode`` (each task is a different SAPIEN scene).
     Single-thread executor enforces SAPIEN GL/physics affinity.

2. Five canvas tool nodes
     env_simpler__reset            — episode start; emits initial obs
                                     after a deterministic gym reset(seed=...)
     env_simpler__step             — accepts a runtime-variable-length
                                     7-D action chunk (JSON list-of-lists),
                                     loops one-at-a-time over the env,
                                     early-break on terminated/truncated
     env_simpler__get_observation  — read-only snapshot of last obs
     env_simpler__episode_info     — current split/task/episode metadata
     env_simpler__evaluate         — post-hoc success boolean + metrics

3. ``EnvSimplerNodeSet`` (collection + lifecycle)
     server_python defaults to ``$SIMPLER_PYTHON`` (the env created by
     ``scripts/install/install_ac_simpler.sh``). ``parallelism="replicated"``
     because the SAPIEN scene is per-worker stateful.

Action contract (TEXT JSON):
    Either a single 7-vec or a list of 7-vecs (chunk). K is runtime-variable.
        "[ax, ay, az, arx, ary, arz, grip]"
        "[[ax, ay, az, arx, ary, arz, grip], ...]"
    Indices: 0-2 delta-pos (range ±1), 3-5 delta-axis-angle (range ±π/2 radians),
    6 gripper (range ±1, -1=close, +1=open — same convention as LIBERO).
    NaN/Inf clipped to [-1, 1] for 0-2 and 6; clipped to [-π/2, π/2] for 3-5.

Observation bundle (emitted by reset / step / get_observation):
    agentview_image  (IMAGE)        — third-person camera, H×W×3 uint8
    wrist_image      (IMAGE)        — None (SIMPLER has no wrist cam)
    observation      (LIST[IMAGE])  — [agentview_image] (single-view; LIBERO emits two)
    state            (ANY)          — flat float32 proprio from obs['agent']
    pose             (POSE)         — None (manipulation, not nav)
    instruction      (TEXT)         — env.get_language_instruction()
    episode_id       (TEXT)         — "{split}/{task_id}/{episode_index}"
    split, task_id, max_steps, step_index, reward, success, done, truncated

last updated: 2026-05-01
"""


import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef, conda_env_python
from app.components.env_panel import (
    BaseEnvPanel,
    EnvPanelAction,
    EnvPanelField,
)


log = logging.getLogger("agentcanvas.simpler")


# ══════════════════════════════════════════════════════════════════════
# Splits, cameras, defaults
# ══════════════════════════════════════════════════════════════════════

_SPLIT_NAMES: list[str] = ["bridge", "google_robot"]

# Map split → task_id prefix used to filter simpler_env.ENVIRONMENTS.
_SPLIT_PREFIX: dict[str, str] = {
    "bridge": "widowx_",
    "google_robot": "google_robot_",
}

# Map split → third-person camera name SAPIEN exposes for that embodiment.
# WidowX: 3rd_view_camera (640×480), Google Robot: overhead_camera (640×512).
_SPLIT_CAMERA: dict[str, str] = {
    "bridge": "3rd_view_camera",
    "google_robot": "overhead_camera",
}

# Number of seeds we expose per task — SIMPLER doesn't ship a curated init-state
# list, episodes are seed-determined. Match LIBERO's 50.
_SEEDS_PER_TASK: int = 50

# Fallback max-steps if env.spec.max_episode_steps is missing.
# Verified per-task: widowx_spoon_on_towel = 60. Per-task value is read from
# env.spec at set_episode time; this constant is only the back-stop.
_MAX_STEPS_DEFAULT: int = 60

_DEFAULTS: dict[str, Any] = {
    "seed_base": 2022,  # SIMPLER's own canonical default seed.
}


# ══════════════════════════════════════════════════════════════════════
# SimplerEnvManager — singleton simulator runtime
# ══════════════════════════════════════════════════════════════════════


class SimplerEnvManager:
    """Singleton env manager for SIMPLER.

    Each task is a different SAPIEN scene, so the underlying env is
    rebuilt on every ``set_episode``. All SAPIEN calls run on a pinned
    single-thread executor for GL/physics affinity.
    """

    _instance: SimplerEnvManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="simpler",
        )

        # Static state (loaded once on initialize).
        self._initialized: bool = False
        self._config: dict[str, Any] = dict(_DEFAULTS)
        self._tasks_by_split: dict[str, list[str]] = {s: [] for s in _SPLIT_NAMES}

        # Episode-scoped state (rebuilt on set_episode).
        self._split: str = ""
        self._task_id: str = ""
        self._episode_index: int = -1
        self._instruction: str = ""
        self._max_steps: int = 0
        self._wrapper: Any = None              # SimplerWrapper instance
        self._step_index: int = 0
        self._cumulative_reward: float = 0.0
        self._success: bool = False
        self._done: bool = False
        self._last_obs: dict[str, Any] | None = None

    # ── Singleton + lifecycle ──────────────────────────────────────────

    @classmethod
    def get(cls) -> SimplerEnvManager:
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
        """Lazy-import simpler_env, partition task IDs by embodiment."""
        with self._lock:
            self._config.update({k: v for k, v in kwargs.items() if k in _DEFAULTS})
            import simpler_env  # heavy import — env scope

            tasks_by_split: dict[str, list[str]] = {s: [] for s in _SPLIT_NAMES}
            for tid in simpler_env.ENVIRONMENTS:
                for split, prefix in _SPLIT_PREFIX.items():
                    if tid.startswith(prefix):
                        tasks_by_split[split].append(tid)
                        break
            self._tasks_by_split = tasks_by_split
            self._initialized = True
            counts = {s: len(v) for s, v in tasks_by_split.items()}
            log.info("SimplerEnvManager: initialized — task counts %s", counts)

    def shutdown(self) -> None:
        with self._lock:
            self._close_wrapper_unlocked()
            self._tasks_by_split = {s: [] for s in _SPLIT_NAMES}
            self._initialized = False

    def _close_wrapper_unlocked(self) -> None:
        # Match upstream SimplerEnv pattern (maniskill2_evaluator never calls
        # env.close()): just drop the ref and let GC reclaim. ManiSkill2's
        # BaseEnv.close() only nulls Python refs anyway — it doesn't destroy
        # SAPIEN renderer/scene C++ objects, so calling it can interleave
        # with the new env's construction and segfault the dynamic linker
        # (observed: ld-2.31.so error 4 on the 5th close+make cycle).
        if self._wrapper is not None:
            self._wrapper = None
            import gc
            gc.collect()

    # ── Inspection ─────────────────────────────────────────────────────

    def list_splits(self) -> list[str]:
        return list(_SPLIT_NAMES)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """Per-task metadata for a split. Safe on uninitialized env."""
        if not self._initialized:
            return []
        if split not in self._tasks_by_split:
            return []
        return [
            {
                "task_id": tid,
                "n_episodes": _SEEDS_PER_TASK,
                "max_steps": _MAX_STEPS_DEFAULT,
            }
            for tid in self._tasks_by_split[split]
        ]

    def get_total_episodes(self, split: str) -> int:
        return len(self._tasks_by_split.get(split, [])) * _SEEDS_PER_TASK

    # ── Episode control ────────────────────────────────────────────────

    def set_episode(self, split: str, task_id: str, episode_index: int) -> dict[str, Any]:
        """Tear down any live env, build a fresh one for (split, task, episode)."""
        with self._lock:
            if not self._initialized:
                return {"error": "SIMPLER not initialized — call initialize() first"}
            if split not in _SPLIT_PREFIX:
                return {"error": f"unknown split '{split}'"}
            prefix = _SPLIT_PREFIX[split]
            if not isinstance(task_id, str) or not task_id.startswith(prefix):
                return {"error": f"task_id '{task_id}' doesn't match split '{split}' (prefix '{prefix}')"}
            if task_id not in self._tasks_by_split.get(split, []):
                return {"error": f"task_id '{task_id}' not registered for split '{split}'"}

            ep_idx = max(0, int(episode_index))
            seed = int(self._config["seed_base"]) + ep_idx

            # Reuse the existing SAPIEN env when the task_id is unchanged —
            # only the seed differs across episodes within a task. Repeated
            # close()+make() leaks Vulkan/GL contexts in SAPIEN and segfaults
            # the dynamic linker after a handful of cycles (observed in
            # multi-episode evals).
            reuse_env = (
                self._wrapper is not None
                and self._task_id == task_id
                and self._split == split
            )
            if not reuse_env:
                self._close_wrapper_unlocked()

                # Imports are env-scoped — only the ac-simpler env has them.
                import simpler_env

                try:
                    env = simpler_env.make(task_id)
                except Exception as e:  # noqa: BLE001
                    log.exception("simpler_env.make failed for %s", task_id)
                    return {"error": f"simpler_env.make failed: {e!r}"}

                # Lazy import — avoids import-time dep on sapien at module load.
                from ._wrapper import SimplerWrapper

                camera_name = _SPLIT_CAMERA[split]
                wrapper = SimplerWrapper(env, camera_name=camera_name)
            else:
                wrapper = self._wrapper
                env = wrapper.env

            try:
                obs = wrapper.reset(seed=seed)
            except Exception as e:  # noqa: BLE001
                log.exception("SimplerWrapper.reset failed for %s seed=%d", task_id, seed)
                if not reuse_env:
                    wrapper.close()
                return {"error": f"reset failed: {e!r}"}

            # Pull instruction + max_steps from the live env. Use .unwrapped
            # to bypass gymnasium's wrapper-attribute deprecation warning.
            try:
                instruction = str(env.unwrapped.get_language_instruction())
            except Exception:  # noqa: BLE001
                try:
                    instruction = str(env.get_language_instruction())
                except Exception:  # noqa: BLE001
                    instruction = ""

            max_steps = _MAX_STEPS_DEFAULT
            spec = getattr(env, "spec", None)
            if spec is not None and getattr(spec, "max_episode_steps", None):
                max_steps = int(spec.max_episode_steps)

            # Episode bookkeeping.
            self._split = split
            self._task_id = task_id
            self._episode_index = ep_idx
            self._instruction = instruction
            self._max_steps = max_steps
            self._wrapper = wrapper
            self._step_index = 0
            self._cumulative_reward = 0.0
            self._success = False
            self._done = False
            self._last_obs = obs

            log.info(
                "SIMPLER: episode set %s/%s/ep=%d (seed=%d, max_steps=%d) — %s",
                split, task_id, ep_idx, seed, max_steps, instruction[:60],
            )
            return self._bundle_unlocked()

    # ── Stepping ───────────────────────────────────────────────────────

    def step_chunk(self, action_chunk: np.ndarray) -> dict[str, Any]:
        """Execute a (K, 7) action chunk; early-break on terminated/truncated.

        ``action_chunk`` may also be (7,) — broadcast to (1, 7) by caller.
        Indices 0-2 (delta-pos) and 6 (gripper) clipped to [-1, 1]; indices
        3-5 (axis-angle, radians) clipped to env's [-π/2, π/2] range.
        Returns the bundle for the *last* step taken.
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

            # NaN/Inf scrub before clipping.
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            # Per-channel clip — pos & gripper to ±1, axis-angle to ±π/2.
            arr[:, 0:3] = np.clip(arr[:, 0:3], -1.0, 1.0)
            arr[:, 3:6] = np.clip(arr[:, 3:6], -np.pi / 2, np.pi / 2)
            arr[:, 6:7] = np.clip(arr[:, 6:7], -1.0, 1.0)

            for k in range(arr.shape[0]):
                obs, reward, done, info = self._wrapper.step(arr[k])
                self._step_index += 1
                self._cumulative_reward += float(reward)
                self._last_obs = obs
                if bool(info.get("success", False)):
                    self._success = True
                if done:
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
            if not self._task_id:
                return {"error": "no active episode"}
            return {
                "split": self._split,
                "task_id": self._task_id,
                "episode_id": str(self._episode_index),
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
        wrist_img = obs.get("wrist_image")  # always None for SIMPLER
        state = obs.get("state")
        truncated = (
            self._done
            and not self._success
            and self._step_index >= self._max_steps
        )
        return {
            "agentview_image": agent_img,
            "wrist_image": wrist_img,
            # Tier-1 contract: ``observation`` is a list of images. SIMPLER
            # has only one camera so the list is single-entry.
            "observation": [agent_img] if agent_img is not None else None,
            "state": state,
            # Tier-1 ``pose`` is None — SIMPLER is manipulation, not navigation.
            "pose": None,
            "instruction": self._instruction,
            "episode_id": "{}/{}/{}".format(
                self._split, self._task_id, self._episode_index
            ),
            "split": self._split,
            "task_id": self._task_id,
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


def _get_mgr() -> SimplerEnvManager:
    return SimplerEnvManager.get()


async def _run_sync(fn: Any, *args: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(
        _get_mgr().executor, fn, *args
    )


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


_NULL_BUNDLE = {
    "agentview_image": None,
    "wrist_image": None,
    "observation": None,
    "state": None,
    "pose": None,
    "instruction": "",
    "episode_id": "",
    "split": "",
    "task_id": "",
    "max_steps": 0,
    "step_index": 0,
    "reward": 0.0,
    "success": False,
    "done": True,        # mark done so a caller's loop halts cleanly on error
    "truncated": False,
}


def _select(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: d.get(k, _NULL_BUNDLE.get(k)) for k in keys}


_RESET_PORT_KEYS = [
    "instruction", "episode_id", "observation", "pose",
    "agentview_image", "wrist_image", "state",
    "split", "task_id", "max_steps",
]

_STEP_PORT_KEYS = [
    "instruction", "episode_id", "observation", "pose", "done",
    "agentview_image", "wrist_image", "state",
    "reward", "success", "step_index", "truncated",
]


class ResetSimplerTool(BaseCanvasNode):
    node_type = "env_simpler__reset"
    display_name = "SIMPLER: Reset"
    description = "Begin episode — emit instruction + ids (metadata only, no observation)"
    category = "environment"
    icon = "RotateCcw"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("instruction", "TEXT", "Task language description"),
        PortDef("episode_id", "TEXT", "{split}/{task_id}/{episode_index}"),
        PortDef("split", "TEXT", "Active embodiment ('bridge' or 'google_robot')"),
        PortDef("task_id", "TEXT", "SIMPLER task identifier"),
        PortDef("max_steps", "ANY", "Per-episode step budget (env.spec.max_episode_steps)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        mgr = _get_mgr()
        # Episode placement is env panel-owned; reset only emits metadata.
        split = mgr._split or _SPLIT_NAMES[0]
        task_id = mgr._task_id or (
            mgr._tasks_by_split.get(split, [""])[0] if mgr.initialized else ""
        )
        ep_idx = mgr._episode_index if mgr._episode_index >= 0 else 0
        result = await _run_sync(mgr.set_episode, split, task_id, ep_idx)
        if isinstance(result, dict) and "error" in result:
            self._self_log("error", result["error"])
            return {"instruction": "", "episode_id": "", "split": "", "task_id": "", "max_steps": 0}
        self._self_log("episode_id", result.get("episode_id"))
        self._self_log("instruction", (result.get("instruction") or "")[:80])
        self._self_log("max_steps", result.get("max_steps"))
        return {
            "instruction": result.get("instruction", ""),
            "episode_id": result.get("episode_id", ""),
            "split": result.get("split", ""),
            "task_id": result.get("task_id", ""),
            "max_steps": result.get("max_steps", 0),
        }


class StepContinuousSimplerTool(BaseCanvasNode):
    node_type = "env_simpler__step_continuous"
    display_name = "SIMPLER: Step (continuous)"
    description = (
        "Execute a 7-DoF action chunk; returns control signals only (pull obs via "
        "observe_egocentric). Input JSON: a 7-vec or list of 7-vecs "
        "[delta_pos(3), delta_axis_angle(3), gripper(0=open/1=close)]."
    )
    category = "environment"
    icon = "Play"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef(
            "action",
            "TEXT",
            'JSON: "[ax,ay,az,arx,ary,arz,grip]" or "[[…],…]" — pos/grip ±1, rot ±π/2',
        ),
    ]
    output_ports = [
        # gym-like contract
        PortDef("reward", "ANY", "Cumulative reward over the episode"),
        PortDef("terminated", "BOOL", "MDP terminal: task success / env-terminal"),
        PortDef("truncated", "BOOL", "Step-budget cutoff (max_steps without success)"),
        PortDef("info", "ANY", "Diagnostics: {step_index, success, cumulative_reward}"),
        # simpler-specific extras (also inside info)
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
            for k in ("step_index", "success", "cumulative_reward", "instruction", "episode_id")
        }
        return {
            "reward": result.get("reward", 0.0),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "success": bool(result.get("success", False)),
            "step_index": result.get("step_index", 0),
        }


class ObserveEgocentricSimplerTool(BaseCanvasNode):
    node_type = "env_simpler__observe_egocentric"
    display_name = "SIMPLER: Observe (egocentric)"
    description = "Pull the current observation: agentview RGB + proprio state (read-only, no env step)."
    category = "environment"
    icon = "Eye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Trigger re-observe (optional)", optional=True),
    ]
    output_ports = [
        PortDef("rgb", "IMAGE", "Agentview RGB (third-person)"),
        PortDef("agentview_image", "IMAGE", "Third-person view (alias of rgb, per-embodiment camera)"),
        PortDef("wrist_image", "IMAGE", "None (SIMPLER has no wrist cam)"),
        PortDef("state", "ANY", "Flat float32 proprio from obs['agent']"),
        PortDef("observation", "LIST[IMAGE]", "[agentview_image]"),
        PortDef("pose", "POSE", "None (manipulation, no nav pose)"),
        PortDef("intrinsics", "ANY", "None (not exposed by SIMPLER)"),
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


class EvaluateSimplerTool(BaseCanvasNode):
    node_type = "env_simpler__evaluate"
    display_name = "SIMPLER: Evaluate"
    description = "Post-hoc success boolean + rollout metrics."
    category = "environment"
    icon = "CheckCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("trigger", "ANY", "Optional fire trigger", optional=True),
    ]
    output_ports = [
        PortDef("success", "BOOL"),
        PortDef("metrics", "METRICS", "{success, num_steps, split, task_id, cumulative_reward}"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        info = await _run_sync(_get_mgr().current_episode)
        if isinstance(info, dict) and "error" in info:
            self._self_log("error", info["error"])
            return {
                "success": False,
                "metrics": {
                    "success": 0.0, "num_steps": 0,
                    "split": "", "task_id": "", "cumulative_reward": 0.0,
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
                "split": info.get("split", ""),
                "task_id": info.get("task_id", ""),
                "cumulative_reward": float(info.get("cumulative_reward", 0.0)),
            },
        }


# ══════════════════════════════════════════════════════════════════════
# SimplerEnvPanel — canvas panel env panel
# ══════════════════════════════════════════════════════════════════════


class SimplerEnvPanel(BaseEnvPanel):
    """Canvas panel env panel for SIMPLER.

    Three-field cascade: ``split → task_id → episode_index``. Field
    changes emit the ``episode_reset`` signal so ``lifetime="episode"``
    state containers clear automatically.
    """

    name = "env_simpler"
    display_name = "SIMPLER"
    fields = [
        EnvPanelField("split",         "select", "Embodiment"),
        EnvPanelField("task_id",       "select", "Task"),
        EnvPanelField("episode_index", "select", "Episode"),
    ]
    actions = [
        EnvPanelAction("play",  "Play",  side_effect="run_start"),
        EnvPanelAction("pause", "Pause", side_effect="run_pause", enabled_when="running"),
        EnvPanelAction("stop",  "Stop",  side_effect="run_stop",  enabled_when="running"),
        EnvPanelAction("reset", "Reset", side_effect="none"),
    ]

    def __init__(self) -> None:
        # Default to the smallest split (Bridge / 4 tasks).
        self._state: dict[str, Any] = {
            "split":         _SPLIT_NAMES[0],
            "task_id":       "",            # filled lazily on first on_load
            "episode_index": 0,
        }

    def _mgr(self) -> SimplerEnvManager:
        return SimplerEnvManager.get()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mgr().executor, fn, *args)

    def _resolve_default_task(self, split: str) -> str:
        mgr = self._mgr()
        if not mgr.initialized:
            return ""
        tasks = mgr._tasks_by_split.get(split, [])
        return tasks[0] if tasks else ""

    def _episode_reset_payload(self) -> dict[str, Any]:
        return {
            "split":         self._state.get("split", _SPLIT_NAMES[0]),
            "task_id":       self._state.get("task_id", ""),
            "episode_index": int(self._state.get("episode_index", 0)),
        }

    async def on_load(self) -> dict[str, Any]:
        mgr = self._mgr()
        if not mgr.initialized:
            return {
                "available": False,
                "split":         self._state.get("split", _SPLIT_NAMES[0]),
                "task_id":       self._state.get("task_id", ""),
                "episode_index": 0,
                "episode_count": 0,
                "splits":        list(_SPLIT_NAMES),
                "message": (
                    "SIMPLER not initialized. Load env_simpler from the "
                    "NodeSet Manager to enable episode control."
                ),
            }
        split = self._state.get("split", _SPLIT_NAMES[0])
        task_id = self._state.get("task_id") or self._resolve_default_task(split)
        if task_id and not task_id.startswith(_SPLIT_PREFIX[split]):
            task_id = self._resolve_default_task(split)
        self._state["task_id"] = task_id
        ep_idx = int(self._state.get("episode_index", 0))
        return {
            "available":         True,
            "split":             split,
            "task_id":           task_id,
            "episode_index":     ep_idx,
            "episode_count":     _SEEDS_PER_TASK if task_id else 0,
            "splits":            list(_SPLIT_NAMES),
            "step_budget": _MAX_STEPS_DEFAULT,
            "current_episode": {
                "split":   split,
                "task_id": task_id,
                "episode": ep_idx,
                "instruction": "",  # filled by manager once env is built
            },
        }

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        mgr = self._mgr()
        if name == "split":
            split = str(value)
            if split not in _SPLIT_NAMES:
                split = _SPLIT_NAMES[0]
            self._state["split"] = split
            self._state["task_id"] = self._resolve_default_task(split)
            self._state["episode_index"] = 0
        elif name == "task_id":
            self._state["task_id"] = str(value)
            self._state["episode_index"] = 0
        elif name == "episode_index":
            try:
                idx = int(value)
            except (TypeError, ValueError):
                idx = 0
            self._state["episode_index"] = idx
        else:
            self._state[name] = value

        # For any cascade change, push set_episode so the manager's scene
        # matches the dropdown — makes canvas reset idempotent.
        if mgr.initialized and self._state.get("task_id"):
            await self._run(
                mgr.set_episode,
                self._state["split"],
                self._state["task_id"],
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
                        "error": "SIMPLER not initialized"}
            if not self._state.get("task_id"):
                self._state["task_id"] = self._resolve_default_task(self._state["split"])
            if not self._state.get("task_id"):
                return {"ok": False, "side_effect": "none",
                        "error": f"no tasks registered for split '{self._state['split']}'"}
            await self._run(
                mgr.set_episode,
                self._state["split"],
                self._state["task_id"],
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
        if field == "split":
            options: list[dict[str, Any]] = []
            for s in _SPLIT_NAMES:
                count = len(mgr._tasks_by_split.get(s, [])) if mgr.initialized else 0
                label = f"{s} ({count} tasks)" if count else s
                options.append({"value": s, "label": label})
            return options
        if field == "task_id":
            if not mgr.initialized:
                return []
            split = self._state.get("split", _SPLIT_NAMES[0])
            return [
                {"value": tid, "label": tid}
                for tid in mgr._tasks_by_split.get(split, [])
            ]
        if field == "episode_index":
            return [{"value": i, "label": f"ep {i} (seed={2022 + i})"}
                    for i in range(_SEEDS_PER_TASK)]
        return []


# ══════════════════════════════════════════════════════════════════════
# EnvSimplerNodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


class EnvSimplerNodeSet(BaseNodeSet):
    """SIMPLER VLA evaluation benchmark as a NodeSet.

    Loads in server mode against the ``ac-simpler`` conda env by
    default. ``server_python`` reads from ``$SIMPLER_PYTHON``; for eval/CI
    point this at the env created by ``scripts/install/install_ac_simpler.sh``.
    """

    name = "env_simpler"
    description = "SIMPLER — VLA evaluation benchmark (SAPIEN/ManiSkill2, 25 tasks)"
    server_python = conda_env_python("ac-simpler", "SIMPLER_PYTHON")
    env_panel = SimplerEnvPanel
    parallelism = "replicated"  # Per-worker SAPIEN scene state.
    # SAPIEN step latency varies by task (3 Hz Google Robot to 5 Hz WidowX
    # control freq); ~50-200 ms per env.step. 30 s / step is loose headroom
    # for VLA inference + worker contention.
    default_per_step_budget_sec = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = SimplerEnvManager.get()

    def get_tools(self) -> list:
        return [
            # gym-like env interface (see docs: nodesets/env/template.html)
            ResetSimplerTool(),                # env_simpler__reset (metadata only)
            StepContinuousSimplerTool(),       # env_simpler__step_continuous
            ObserveEgocentricSimplerTool(),    # env_simpler__observe_egocentric
            EvaluateSimplerTool(),             # env_simpler__evaluate
        ]

    async def initialize(self, **kwargs: Any) -> None:
        """Lazy-load simpler_env, partition task IDs by embodiment.

        Accepted kwargs: seed_base.
        """
        if self._mgr.initialized:
            log.info("SIMPLER already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("EnvSimplerNodeSet initialized")

    async def get_eval_metadata(self) -> dict:
        if not self._mgr.initialized:
            counts: dict[str, int] = {s: 0 for s in _SPLIT_NAMES}
        else:
            counts = {s: self._mgr.get_total_episodes(s) for s in _SPLIT_NAMES}
        return {
            "env_name": "simpler",
            "datasets": ["SIMPLER"],
            "splits": list(_SPLIT_NAMES),
            "episode_counts": counts,
            "metrics": ["success", "num_steps", "cumulative_reward"],
            "supports_set_episode": self._mgr.initialized,
            "step_budget": _MAX_STEPS_DEFAULT,
        }

    async def shutdown(self) -> None:
        self._mgr.shutdown()
