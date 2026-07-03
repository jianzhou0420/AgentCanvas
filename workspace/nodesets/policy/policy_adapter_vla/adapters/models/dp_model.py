"""Diffusion Policy model adaptor.

Vendored from vlaworkspace/adaptors/models/dp_model.py — no upstream import.

DP_C model format:
    Input:
        - obs.agentview_image: [T, C, H, W] float32 [-1,1]
        - obs.robot0_eye_in_hand_image: [T, C, H, W] float32 [-1,1]
        - obs.robot0_eef_pos: [T, 3] float32
        - obs.robot0_eef_quat: [T, 4] float32
        - obs.robot0_gripper_qpos: [T, 2] float32
        - action: [horizon, 10] float32  (pos[3] + rotation_6d[6] + gripper[1])
    Output:
        - action: [horizon, 10] float32

Normalization: limits mode, on pos + gripper only (NOT rotation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..canonical import CanonicalDict, CanonicalInfo, make_canonical_action
from .base_model import ModelAdaptor

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _get_rotation_transformer(from_rep: str, to_rep: str):
    # Lazy import — vendored model tree under workspace/nodesets/policy/policy_vla/models/dp/
    from ...models.dp.common.rotation_transformer import RotationTransformer

    return RotationTransformer(from_rep=from_rep, to_rep=to_rep)


def _convert_rotation(data: np.ndarray, from_repr: str, to_repr: str) -> np.ndarray:
    if from_repr == to_repr:
        return data
    transformer = _get_rotation_transformer(from_repr, to_repr)
    original_shape = data.shape
    flat = data.reshape(-1, original_shape[-1])
    converted = transformer.forward(flat)
    return converted.reshape(*original_shape[:-1], converted.shape[-1])


class DPModel(ModelAdaptor):
    """Model adaptor for Diffusion Policy (DP_C)."""

    CAMERA_MAP = {
        "front": "agentview_image",
        "wrist": "robot0_eye_in_hand_image",
    }

    def __init__(
        self,
        *,
        norm_stats_path: str | None = None,
        norm_stats: dict | None = None,
        n_obs_steps: int = 2,
        crop_shape: tuple[int, int] | None = (76, 76),
        model_action_dim: int = 10,
        action_norm_stats_path: str | None = None,
    ) -> None:
        self._auto_norm_stats = norm_stats_path == "auto"
        if self._auto_norm_stats:
            super().__init__(norm_stats=norm_stats)
        else:
            super().__init__(norm_stats_path=norm_stats_path, norm_stats=norm_stats)
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.model_action_dim = model_action_dim
        self._action_norm_stats_path = action_norm_stats_path

    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]
        info = canonical["info"]

        obs = {}

        for canonical_name, dp_name in self.CAMERA_MAP.items():
            img = data.get("images", {}).get(canonical_name)
            if img is not None:
                img = np.asarray(img, dtype=np.float32)
                if img.ndim == 3 and img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
                    img = np.transpose(img, (2, 0, 1))
                elif img.ndim == 4 and img.shape[-1] in (1, 3, 4) and img.shape[1] not in (1, 3, 4):
                    img = np.transpose(img, (0, 3, 1, 2))
                if img.max() > 1.0:
                    img = img / 255.0
                obs[dp_name] = img

        if self.crop_shape is not None:
            for dp_name in self.CAMERA_MAP.values():
                if dp_name in obs:
                    obs[dp_name] = self._crop_image(obs[dp_name])

        for dp_name in self.CAMERA_MAP.values():
            if dp_name in obs:
                obs[dp_name] = (obs[dp_name] * 2.0 - 1.0).astype(np.float32)

        state_data = data.get("state", {})
        if "pos" in state_data and state_data["pos"] is not None:
            obs["robot0_eef_pos"] = np.asarray(state_data["pos"], dtype=np.float32)
        if "rot" in state_data and state_data["rot"] is not None:
            obs["robot0_eef_quat"] = np.asarray(state_data["rot"], dtype=np.float32)
        if "gripper" in state_data and state_data["gripper"] is not None:
            obs["robot0_gripper_qpos"] = np.asarray(state_data["gripper"], dtype=np.float32)

        result = {"obs": obs}

        action_data = data.get("actions", {})
        if action_data:
            action_components = []

            if "pos" in action_data and action_data["pos"] is not None:
                action_components.append(np.asarray(action_data["pos"], dtype=np.float32))

            if "rot" in action_data and action_data["rot"] is not None:
                rot = np.asarray(action_data["rot"], dtype=np.float32)
                rot6d = _convert_rotation(rot, info.action_rot_repr, "rotation_6d")
                action_components.append(rot6d)

            if "gripper" in action_data and action_data["gripper"] is not None:
                gripper = np.asarray(action_data["gripper"], dtype=np.float32)
                if gripper.shape[-1] > 1:
                    gripper = gripper[..., :1]
                action_components.append(gripper)

            if action_components:
                result["action"] = np.concatenate(action_components, axis=-1).astype(np.float32)

        prompt = data.get("prompt")
        if prompt is not None and prompt != "":
            result["prompt"] = prompt

        if self._norm_stats is not None:
            self._normalize_dp(result)

        return result

    def model_to_canonical(self, model_output: dict, info: CanonicalInfo) -> CanonicalDict:
        action_key = "action" if "action" in model_output else "actions"
        actions = np.asarray(_to_numpy(model_output[action_key]), dtype=np.float32)

        if self._norm_stats is not None:
            actions = self._unnormalize_actions(actions)

        pos_dim = info.action_dims.get("pos", 3)
        action_pos = actions[..., :pos_dim]
        action_rot6d = actions[..., pos_dim : pos_dim + 6]
        action_gripper = actions[..., pos_dim + 6 :]

        action_rot = _convert_rotation(action_rot6d, "rotation_6d", info.action_rot_repr)

        expected_gripper_dim = info.action_dims.get("gripper", 1)
        if action_gripper.shape[-1] != expected_gripper_dim:
            if action_gripper.shape[-1] < expected_gripper_dim:
                pad = np.zeros(
                    (*action_gripper.shape[:-1], expected_gripper_dim - action_gripper.shape[-1]),
                    dtype=np.float32,
                )
                action_gripper = np.concatenate([action_gripper, pad], axis=-1)
            else:
                action_gripper = action_gripper[..., :expected_gripper_dim]

        return make_canonical_action(
            actions={"pos": action_pos, "rot": action_rot, "gripper": action_gripper},
            info=info,
        )

    def get_norm_stats_mode(self) -> str:
        return "limits"

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("obs/robot0_eef_pos", "obs/robot0_gripper_qpos", "action")

    def model_input(self) -> dict:
        h = self.crop_shape[0] if self.crop_shape else "H"
        w = self.crop_shape[1] if self.crop_shape else "W"
        return {
            "obs": {
                "agentview_image": f"[T, 3, {h}, {w}] float32 [-1, 1]",
                "robot0_eye_in_hand_image": f"[T, 3, {h}, {w}] float32 [-1, 1]",
                "robot0_eef_pos": "[T, 3] float32",
                "robot0_eef_quat": "[T, 4] float32",
                "robot0_gripper_qpos": "[T, 2] float32",
            },
            "action": f"[horizon, {self.model_action_dim}] float32",
        }

    def model_output(self) -> dict:
        return {"action": f"[horizon, {self.model_action_dim}] float32"}

    def canonical_to_norm_stats_format(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]
        info = canonical["info"]
        obs = {}

        state_data = data.get("state", {})
        if "pos" in state_data and state_data["pos"] is not None:
            obs["robot0_eef_pos"] = np.asarray(state_data["pos"], dtype=np.float32)
        if "gripper" in state_data and state_data["gripper"] is not None:
            obs["robot0_gripper_qpos"] = np.asarray(state_data["gripper"], dtype=np.float32)

        result = {}
        if obs:
            result["obs"] = obs

        action_data = data.get("actions", {})
        if action_data:
            action_components = []
            if "pos" in action_data and action_data["pos"] is not None:
                action_components.append(np.asarray(action_data["pos"], dtype=np.float32))
            if "rot" in action_data and action_data["rot"] is not None:
                rot = np.asarray(action_data["rot"], dtype=np.float32)
                rot6d = _convert_rotation(rot, info.action_rot_repr, "rotation_6d")
                action_components.append(rot6d)
            if "gripper" in action_data and action_data["gripper"] is not None:
                gripper = np.asarray(action_data["gripper"], dtype=np.float32)
                if gripper.shape[-1] > 1:
                    gripper = gripper[..., :1]
                action_components.append(gripper)
            if action_components:
                result["action"] = np.concatenate(action_components, axis=-1).astype(np.float32)

        return result

    def auto_compute_norm_stats(self, hf_dataset) -> None:
        """Compute norm stats from dataset (LeRobot HF format)."""
        columns = hf_dataset.column_names

        if "observation.eePose" in columns:
            from .dp_defaults import ABCDEFGH_ACTION_STATS, ABCDEFGH_POS_STATS

            eePose = hf_dataset["observation.eePose"]
            if hasattr(eePose, "numpy"):
                eePose = eePose.numpy()
            else:
                eePose = np.array(eePose)
            gripper = eePose[:, 7:9]
            gripper_min = gripper.min(axis=0).astype(np.float32)
            gripper_max = gripper.max(axis=0).astype(np.float32)
            gripper_mean = gripper.mean(axis=0).astype(np.float32)
            gripper_std = gripper.std(axis=0).astype(np.float32)

            gripper_stats = {
                "min": gripper_min,
                "max": gripper_max,
                "mean": gripper_mean,
                "std": gripper_std,
            }

            self._norm_stats = {
                "obs/robot0_eef_pos": ABCDEFGH_POS_STATS,
                "obs/robot0_gripper_qpos": gripper_stats,
                "action": ABCDEFGH_ACTION_STATS,
            }
            logger.info(
                f"Auto-computed norm stats (MimicGen): gripper min={gripper_min}, max={gripper_max}"
            )

        elif "observation.state" in columns:
            from .dp_defaults import LIBERO_ACTION_STATS, LIBERO_POS_STATS

            state = hf_dataset["observation.state"]
            if hasattr(state, "numpy"):
                state = state.numpy()
            else:
                state = np.array(state)

            gripper = state[:, 7:8].astype(np.float32)
            gripper_stats = {
                "min": gripper.min(axis=0).astype(np.float32),
                "max": gripper.max(axis=0).astype(np.float32),
                "mean": gripper.mean(axis=0).astype(np.float32),
                "std": gripper.std(axis=0).astype(np.float32),
            }

            self._norm_stats = {
                "obs/robot0_eef_pos": LIBERO_POS_STATS,
                "obs/robot0_gripper_qpos": gripper_stats,
                "action": LIBERO_ACTION_STATS,
            }
            logger.info(
                f"LIBERO norm stats: pos hardcoded, gripper computed from data, action hardcoded. "
                f"gripper min={gripper_stats['min']}, max={gripper_stats['max']}"
            )

        else:
            raise ValueError(
                f"Cannot auto-compute norm stats: dataset has neither 'observation.eePose' "
                f"nor 'observation.state' column. Available columns: {columns}"
            )

        self._auto_norm_stats = False

        if self._action_norm_stats_path:
            import json as _json

            with open(self._action_norm_stats_path) as f:
                loaded = _json.load(f)
            if "norm_stats" in loaded:
                loaded = loaded["norm_stats"]
            if "action" in loaded:
                self._norm_stats["action"] = {
                    k: np.array(v, dtype=np.float32) for k, v in loaded["action"].items()
                }
                logger.info(f"Overrode action norm stats from {self._action_norm_stats_path}")
            else:
                logger.warning(
                    f"action_norm_stats_path set to {self._action_norm_stats_path} "
                    f"but no 'action' key found in file. Available keys: {list(loaded.keys())}"
                )

    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        crop_h, crop_w = self.crop_shape

        if image.ndim == 3:
            _, h, w = image.shape
            if self.training:
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
            else:
                top = (h - crop_h) // 2
                left = (w - crop_w) // 2
            return image[:, top : top + crop_h, left : left + crop_w]
        elif image.ndim == 4:
            t, _, h, w = image.shape
            if self.training:
                crops = []
                for i in range(t):
                    top = np.random.randint(0, h - crop_h + 1)
                    left = np.random.randint(0, w - crop_w + 1)
                    crops.append(image[i, :, top : top + crop_h, left : left + crop_w])
                return np.stack(crops, axis=0)
            else:
                top = (h - crop_h) // 2
                left = (w - crop_w) // 2
                return image[:, :, top : top + crop_h, left : left + crop_w]
        return image

    def _normalize_dp(self, data: dict) -> None:
        if "obs" in data:
            for key in ("robot0_eef_pos", "robot0_gripper_qpos"):
                stats_key = f"obs/{key}"
                if key in data["obs"] and stats_key in self._norm_stats:
                    stats = self._norm_stats[stats_key]
                    x = np.asarray(data["obs"][key], dtype=np.float32)
                    data["obs"][key] = self._normalize_limits(x, stats)

        if "action" in data and "action" in self._norm_stats:
            stats = self._norm_stats["action"]
            x = np.asarray(data["action"], dtype=np.float32)
            action_dim = 3
            sliced_stats = {
                k: np.asarray(v, dtype=np.float32)[..., :action_dim] for k, v in stats.items()
            }
            normalized_pos = self._normalize_limits(x[..., :action_dim], sliced_stats)
            data["action"] = np.concatenate([normalized_pos, x[..., action_dim:]], axis=-1).astype(
                np.float32
            )

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if "action" not in self._norm_stats:
            return actions

        stats = self._norm_stats["action"]
        action_dim = 3
        sliced_stats = {
            k: np.asarray(v, dtype=np.float32)[..., :action_dim] for k, v in stats.items()
        }
        unnormalized_pos = self._unnormalize_limits(actions[..., :action_dim], sliced_stats)
        return np.concatenate([unnormalized_pos, actions[..., action_dim:]], axis=-1).astype(
            np.float32
        )

    @staticmethod
    def _normalize_limits(x: np.ndarray, stats: dict) -> np.ndarray:
        min_val = np.asarray(stats["min"], dtype=np.float32)
        max_val = np.asarray(stats["max"], dtype=np.float32)
        input_range = max_val - min_val
        input_range = np.where(input_range < 1e-7, 2.0, input_range)
        scale = 2.0 / input_range
        offset = -1.0 - scale * min_val
        return (x * scale + offset).astype(np.float32)

    @staticmethod
    def _unnormalize_limits(x: np.ndarray, stats: dict) -> np.ndarray:
        min_val = np.asarray(stats["min"], dtype=np.float32)
        max_val = np.asarray(stats["max"], dtype=np.float32)
        input_range = max_val - min_val
        input_range = np.where(input_range < 1e-7, 2.0, input_range)
        scale = 2.0 / input_range
        offset = -1.0 - scale * min_val
        return ((x - offset) / scale).astype(np.float32)


# norm_stats vendored into the nodeset (policy_vla/_assets/norm_stats/); resolve
# relative to this file so it works regardless of CWD or install location.
_NORM_STATS_DIR = Path(__file__).resolve().parents[2] / "_assets" / "norm_stats"

# ───── DEFAULTS (transcribed from diffusion_policy_libero.yaml) ─────
DEFAULT_KWARGS: dict = {
    "norm_stats_path": str(_NORM_STATS_DIR / "norm_stats_dp.json"),
    "n_obs_steps": 2,
    "crop_shape": [76, 76],
}
