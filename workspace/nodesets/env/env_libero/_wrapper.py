from __future__ import annotations

"""Vendored LIBERO wrapper — internal helper for env_libero.

Copied (with minor trims) from
``vlaworkspace/src/vlaworkspace/env_runner/env/libero/libero_wrapper.py``.
Per AgentCanvas policy, no upstream import — all runtime code lives in
this repo. Original wrapper authored by the vlaworkspace project (2026).

What it does:
    Adapts LIBERO's ``OffScreenRenderEnv`` (which is *not* a gym.Env) into a
    minimal reset/step interface compatible with our env nodeset:
      * 180° image flip on agentview / wrist (matches train preprocessing)
      * Quaternion → axis-angle conversion for the 8-D state vector
      * robosuite ``MjRenderContext.__del__`` monkey-patch to silence
        AttributeError spam during multiprocess GC

Trimmed vs upstream:
    * Removed dill + ``run_dill_function`` + ``get_attr`` (we don't use
      AsyncVectorEnv's dill-based env init in the canvas path).
"""

import math

import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# robosuite MjRenderContext destructor patch
#
# Upstream's ``__del__`` assumes ``self.con`` exists; during multiprocess
# garbage collection the attribute may be missing and we get noisy
# AttributeError tracebacks. Patch silently no-ops in that case.
# ---------------------------------------------------------------------------

def _patch_robosuite_render_context() -> None:
    try:
        from robosuite.utils import binding_utils

        def safe_del(self):
            if hasattr(self, "con") and self.con is not None:
                try:
                    self.con.free()
                except Exception:  # noqa: BLE001
                    pass

        binding_utils.MjRenderContext.__del__ = safe_del
    except Exception:  # noqa: BLE001
        # robosuite may not be importable yet (lazy env), or version may
        # not expose binding_utils. Skip silently — patch is best-effort.
        pass


_patch_robosuite_render_context()


# ---------------------------------------------------------------------------
# LiberoWrapper
# ---------------------------------------------------------------------------


class LiberoWrapper:
    """Adapter around LIBERO's ``OffScreenRenderEnv``.

    Standalone (not a ``gymnasium.Wrapper``) because the wrapped env is
    not a gym.Env. Exposes ``reset`` / ``step`` / ``close`` / ``seed`` and
    a ``success`` property aggregating per-step done flags.

    Observation dict keys returned by ``reset`` and ``step``:
        agentview_image: (H, W, 3) uint8 — third-person view, 180° flipped
        wrist_image:     (H, W, 3) uint8 — robot0_eye_in_hand, 180° flipped
        state:           (8,) float32   — eef_pos(3) + axis_angle(3) + grip_qpos(2)

    Action format (passed to ``step``):
        7-vector float32: [delta_pos(3), delta_axis_angle(3), gripper(1)]
        gripper: -1 = close, +1 = open.
    """

    def __init__(
        self,
        env,
        render_hw: tuple[int, int] = (256, 256),
    ) -> None:
        self.env = env
        self.render_hw = render_hw
        self.init_state: np.ndarray | None = None  # caller sets before reset
        self._rewards: list[float] = []
        self._successes: list[float] = []

        self.observation_space = spaces.Dict(
            {
                "agentview_image": spaces.Box(0, 255, (*render_hw, 3), np.uint8),
                "wrist_image": spaces.Box(0, 255, (*render_hw, 3), np.uint8),
                "state": spaces.Box(-np.inf, np.inf, (8,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, (7,), np.float32)
        self.metadata = getattr(env, "metadata", {})

    # ── reset ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.env.reset()
        if self.init_state is not None:
            obs = self.env.set_init_state(self.init_state)
        else:
            obs = self.env._get_observations()
        self._rewards = []
        self._successes = []
        return self._process_obs(obs)

    # ── step ────────────────────────────────────────────────────────────

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        obs, reward, done, info = self.env.step(action)
        self._rewards.append(float(reward))
        # LIBERO returns done=True only on task success.
        self._successes.append(float(done))
        info = dict(info) if info else {}
        info["success"] = bool(done)
        return self._process_obs(obs), float(reward), bool(done), info

    # ── obs assembly ────────────────────────────────────────────────────

    def _process_obs(self, obs) -> dict:
        out = {
            "agentview_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    self._quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ).astype(np.float32),
        }
        return out

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """quat (xyzw) → axis-angle (3-vec). Mirrors robosuite's helper."""
        quat = quat.copy()
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    # ── lifecycle ───────────────────────────────────────────────────────

    @property
    def success(self) -> float:
        return max(self._successes) if self._successes else 0.0

    @property
    def cumulative_reward(self) -> float:
        return float(sum(self._rewards))

    def close(self) -> None:
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass

    def seed(self, seed: int | None = None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None
