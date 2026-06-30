from __future__ import annotations

"""SIMPLER (SAPIEN/ManiSkill2) wrapper — internal helper for env_simpler.

Adapts ``simpler_env.make(task_id)`` (a gymnasium env) into the minimal
interface our env nodeset expects:
    * Resolve the per-embodiment third-person camera RGB image
    * Pack a flat float32 proprio vector from ``obs['agent']``
    * Surface ``info['success']`` / ``terminated`` / ``truncated`` consistently

What we deliberately don't do here:
    * No gripper convention conversion. SIMPLER's native (0=open, 1=close)
      is exposed verbatim on the wire — see ``simpler.py`` action contract.
    * No language instruction caching — the manager calls
      ``env.get_language_instruction()`` directly at episode set time.

last updated: 2026-05-01
"""

import logging
from typing import Any

import numpy as np


log = logging.getLogger("agentcanvas.simpler.wrapper")


# ── Image extraction (Color vs rgb fallback) ──────────────────────────

def _extract_rgb(image_dict: dict[str, Any]) -> np.ndarray | None:
    """Pull the RGB image out of an obs['image'][cam_name] dict.

    Verified against SimplerEnv 2026-05-01: per-camera entries expose
    keys ``rgb`` (uint8 H×W×3), ``depth``, ``Segmentation``. We use ``rgb``
    directly. The ``Color`` fallback (SAPIEN's raw float32 RGBA) is kept
    for forward-compat in case a future SimplerEnv version exposes raw
    camera output.
    """
    if "rgb" in image_dict:
        rgb = image_dict["rgb"]
        if isinstance(rgb, np.ndarray):
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(rgb)
    if "Color" in image_dict:
        col = image_dict["Color"]
        if isinstance(col, np.ndarray):
            arr = col[..., :3] if col.ndim == 3 and col.shape[-1] == 4 else col
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(arr)
    return None


def _flat_proprio(agent_obs: Any) -> np.ndarray | None:
    """Concatenate ``obs['agent']`` array values into a flat float32 vector.

    SIMPLER's ``agent`` dict has 4 entries: ``qpos`` (8,), ``qvel`` (8,),
    ``base_pose`` (7,) — all numpy arrays — plus ``controller``, a nested
    dict that we skip. Result is a (23,) flat float32 for WidowX.
    """
    if agent_obs is None:
        return None
    try:
        if isinstance(agent_obs, dict):
            parts = []
            for v in agent_obs.values():
                if isinstance(v, np.ndarray):
                    parts.append(v.reshape(-1).astype(np.float32))
                elif isinstance(v, (int, float)):
                    parts.append(np.asarray([float(v)], dtype=np.float32))
                # Skip nested dicts (e.g. agent['controller']) — they're
                # not flat proprio and would break concatenate.
            return np.concatenate(parts) if parts else None
        if isinstance(agent_obs, np.ndarray):
            return agent_obs.reshape(-1).astype(np.float32)
    except Exception:  # noqa: BLE001
        log.debug("proprio extraction failed", exc_info=True)
    return None


# ── SimplerWrapper ────────────────────────────────────────────────────


class SimplerWrapper:
    """Thin adapter around a ``simpler_env`` gymnasium env.

    The underlying env IS already a gym.Env, so we don't subclass
    ``gymnasium.Wrapper`` — we just expose the bundle dict shape that
    ``SimplerEnvManager`` consumes.
    """

    def __init__(self, env: Any, camera_name: str) -> None:
        self.env = env
        self.camera_name = camera_name
        self._cumulative_reward: float = 0.0
        self._success: bool = False

    # ── reset ───────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = int(seed)
        obs, _info = self.env.reset(**kwargs)
        self._cumulative_reward = 0.0
        self._success = False
        return self._process_obs(obs)

    # ── step ────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict]:
        # gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._cumulative_reward += float(reward)
        # In SIMPLER, terminated == info['success']; truncated comes from TimeLimit.
        success = bool(info.get("success", terminated))
        self._success = self._success or success
        info = dict(info) if info else {}
        info["success"] = success
        info["truncated"] = bool(truncated)
        done = bool(terminated or truncated)
        return self._process_obs(obs), float(reward), done, info

    # ── obs assembly ───────────────────────────────────────────────

    def _process_obs(self, obs: Any) -> dict[str, Any]:
        agent_img = None
        image_root = obs.get("image") if isinstance(obs, dict) else None
        if isinstance(image_root, dict):
            cam_dict = image_root.get(self.camera_name)
            if cam_dict is None:
                # Fall back to the first camera SAPIEN gave us — better
                # than emitting None when the camera name dispatch is wrong.
                cam_dict = next(iter(image_root.values()), None)
            if isinstance(cam_dict, dict):
                agent_img = _extract_rgb(cam_dict)

        state = _flat_proprio(obs.get("agent") if isinstance(obs, dict) else None)

        return {
            "agentview_image": agent_img,
            # SIMPLER has no wrist camera; emit None to match the LIBERO
            # bundle shape (Tier-1 portability rule).
            "wrist_image": None,
            "state": state,
        }

    # ── metrics ────────────────────────────────────────────────────

    @property
    def success(self) -> bool:
        return bool(self._success)

    @property
    def cumulative_reward(self) -> float:
        return float(self._cumulative_reward)

    # ── lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
