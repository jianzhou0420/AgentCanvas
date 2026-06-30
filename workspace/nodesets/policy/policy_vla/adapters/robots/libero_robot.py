"""LIBERO robot adaptor.

Vendored from vlaworkspace/adaptors/robots/libero_robot.py — no upstream import.

LIBERO state: pos[3] + quat[4] + gripper[1] = 8D
LIBERO actions: delta_pos[3] + delta_axis_angle[3] + abs_gripper[1] = 7D
Cameras: front=observation.image (256x256), wrist=observation.wrist_image (256x256)
"""

from __future__ import annotations

import logging

import numpy as np

from ..canonical import CanonicalDict, CanonicalInfo, make_canonical_obs
from .base_robot import RobotAdaptor

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _parse_image_to_chw_float(image) -> np.ndarray:
    """Parse image to CHW float32 [0,1] format.

    Handles:
    - CHW float [0,1] (from LeRobot dataset) -> pass through
    - HWC uint8 [0,255] (from env) -> CHW float [0,1]
    - CHW uint8 [0,255] -> float [0,1]
    """
    image = np.asarray(_to_numpy(image), dtype=None)

    if image.dtype == np.uint8:
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32) / 255.0

    image = image.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
        image = np.transpose(image, (2, 0, 1))

    if image.max() > 1.0:
        image = image / 255.0

    return image


class LiberoRobot(RobotAdaptor):
    """Robot adaptor for LIBERO environment.

    State layout: [pos(3), quat(4), gripper(1)] = 8D
    Action layout: [delta_pos(3), delta_axis_angle(3), abs_gripper(1)] = 7D
    """

    ENV_KEY_REMAP = {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
    }

    def __init__(
        self,
        *,
        tasks: dict[int, str] | None = None,
        default_prompt: str | None = None,
        use_delta_actions: bool = False,
        delta_action_mask: tuple[bool, ...] | None = None,
    ) -> None:
        self.tasks = tasks
        self.default_prompt = default_prompt
        self.use_delta_actions = use_delta_actions
        if use_delta_actions and delta_action_mask is None:
            self.delta_action_mask = (True, True, True, True, True, True, False)
        else:
            self.delta_action_mask = delta_action_mask

    def get_canonical_info(self) -> CanonicalInfo:
        action_type = {
            "pos": "delta" if self.use_delta_actions else "absolute",
            "rot": "delta" if self.use_delta_actions else "absolute",
            "gripper": "absolute",
        }
        return CanonicalInfo(
            state_type={"pos": "absolute", "rot": "absolute", "gripper": "absolute"},
            state_rot_repr="quat",
            action_type=action_type,
            action_rot_repr="axis_angle",
            state_dims={"pos": 3, "rot": 4, "gripper": 1},
            action_dims={"pos": 3, "rot": 3, "gripper": 1},
        )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        prompt = ""
        if self.tasks and "task_index" in data:
            task_idx = int(data["task_index"])
            prompt = self.tasks.get(task_idx, self.default_prompt or "")
        elif self.default_prompt:
            prompt = self.default_prompt

        front_img = None
        if "observation.image" in data:
            img = _to_numpy(data["observation.image"])
            if img.ndim >= 2 and img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
            front_img = _parse_image_to_chw_float(img)

        wrist_img = None
        if "observation.wrist_image" in data:
            img = _to_numpy(data["observation.wrist_image"])
            if img.ndim >= 2 and img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
            wrist_img = _parse_image_to_chw_float(img)

        state_raw = np.asarray(_to_numpy(data["observation.state"]), dtype=np.float32)
        if state_raw.ndim >= 2 and state_raw.shape[0] == 1:
            state_raw = state_raw.squeeze(0)

        state_pos = state_raw[..., :3]
        state_rot = state_raw[..., 3:7]
        state_gripper = state_raw[..., 7:8]

        actions_raw = np.asarray(_to_numpy(data["actions"]), dtype=np.float32)
        if actions_raw.ndim >= 2 and actions_raw.shape[0] == 1:
            actions_raw = actions_raw.squeeze(0)

        if self.use_delta_actions and self.delta_action_mask is not None:
            mask = np.asarray(self.delta_action_mask)
            dims = mask.shape[-1]
            current_state = state_raw[-1] if state_raw.ndim >= 2 else state_raw
            actions_raw = actions_raw.copy()
            actions_raw[..., :dims] -= np.expand_dims(
                np.where(mask, current_state[..., :dims], 0), axis=-2
            )

        action_pos = actions_raw[..., :3]
        action_rot = actions_raw[..., 3:6]
        action_gripper = actions_raw[..., 6:7]

        return make_canonical_obs(
            images={"front": front_img, "wrist": wrist_img},
            state={"pos": state_pos, "rot": state_rot, "gripper": state_gripper},
            actions={"pos": action_pos, "rot": action_rot, "gripper": action_gripper},
            prompt=prompt,
            info=self.get_canonical_info(),
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        remapped = {}
        for key, value in data.items():
            new_key = self.ENV_KEY_REMAP.get(key, key)
            remapped[new_key] = value

        front_img = None
        if "image" in remapped:
            front_img = _parse_image_to_chw_float(remapped["image"])

        wrist_img = None
        if "wrist_image" in remapped:
            wrist_img = _parse_image_to_chw_float(remapped["wrist_image"])

        state_raw = np.asarray(_to_numpy(remapped["state"]), dtype=np.float32)
        state_pos = state_raw[..., :3]
        state_rot = state_raw[..., 3:7]
        state_gripper = state_raw[..., 7:8]

        prompt = remapped.get("prompt", self.default_prompt or "")

        return make_canonical_obs(
            images={"front": front_img, "wrist": wrist_img},
            state={"pos": state_pos, "rot": state_rot, "gripper": state_gripper},
            actions={},
            prompt=prompt,
            info=self.get_canonical_info(),
        )

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        actions_data = canonical_action["data"]["actions"]

        action_pos = actions_data["pos"]
        action_rot = actions_data["rot"]
        action_gripper = actions_data["gripper"]

        actions = np.concatenate([action_pos, action_rot, action_gripper], axis=-1).astype(np.float32)

        if self.use_delta_actions and self.delta_action_mask is not None and state is not None:
            current_state = np.asarray(_to_numpy(state.get("state", state)), dtype=np.float32)
            if hasattr(current_state, "shape") and current_state.ndim >= 1:
                mask = np.asarray(self.delta_action_mask)
                dims = mask.shape[-1]
                actions = actions.copy()
                actions[..., :dims] += np.expand_dims(
                    np.where(mask, current_state[..., :dims], 0), axis=-2
                )

        return {"actions": actions}

    def get_state_dim(self) -> int:
        return 8

    def get_action_dim(self) -> int:
        return 7

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state/pos", "state/rot", "state/gripper", "actions/pos", "actions/rot", "actions/gripper")

    def env_obs(self) -> dict:
        return {
            "image": "[256, 256, 3] uint8 [0, 255]",
            "wrist_image": "[256, 256, 3] uint8 [0, 255]",
            "state": "[8] float32",
            "prompt": "str",
        }

    def env_action(self) -> dict:
        return {"actions": "[horizon, 7] float32"}

    def datasets(self) -> dict:
        return {
            "observation.image": "[1, 3, 256, 256] float32 [0, 1]",
            "observation.wrist_image": "[1, 3, 256, 256] float32 [0, 1]",
            "observation.state": "[1, 8] float32",
            "actions": "[horizon, 7] float32",
            "task_index": "int",
        }


# ───── DEFAULTS (transcribed from CORL_pi0_libero_lerobot.yaml + diffusion_policy_libero.yaml) ─────
# Pi0 / DP / DROID-DP-finetune all use delta+mask on LIBERO; SmolVLA + DROID-DP-pretrained use
# absolute — for that variant see libero_robot_absolute.py (same class, different default kwargs).
DEFAULT_KWARGS: dict = {
    "use_delta_actions": True,
    "delta_action_mask": [True, True, True, True, True, True, False],
}
