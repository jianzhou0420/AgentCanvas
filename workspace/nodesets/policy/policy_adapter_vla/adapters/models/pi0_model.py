"""Pi0 model adaptor.

Vendored from vlaworkspace/adaptors/models/pi0_model.py — no upstream import.

Pi0 model format:
    Input:
        - images: {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb} [224, 224, 3] uint8
        - image_masks: {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb} bool
        - state: [32] float32 normalized
        - actions: [horizon, 32] float32 normalized
        - tokenized_prompt: [max_token_len] int64
        - tokenized_prompt_mask: [max_token_len] bool
    Output:
        - actions: [horizon, 32] float32 normalized, padded

Normalization: z-score on ALL state/action components (concatenated).
Images: canonical CHW float [0,1] -> HWC uint8 [0,255], resize to 224x224.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..canonical import CanonicalDict, CanonicalInfo, make_canonical_action
from .base_model import ModelAdaptor

logger = logging.getLogger(__name__)


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _chw_float_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = image * 255.0
    return image.astype(np.uint8)


class Pi0Model(ModelAdaptor):
    """Model adaptor for Pi0 (and Pi0 LoRA)."""

    CAMERA_MAP = {
        "front": "base_0_rgb",
        "wrist": "left_wrist_0_rgb",
    }
    ZERO_CAMERA = "right_wrist_0_rgb"

    def __init__(
        self,
        *,
        norm_stats_path: str | None = None,
        norm_stats: dict | None = None,
        max_token_len: int | None = None,
        model_action_dim: int = 32,
        use_quantile_norm: bool = False,
    ) -> None:
        super().__init__(norm_stats_path=norm_stats_path, norm_stats=norm_stats)
        self.max_token_len = max_token_len
        self.model_action_dim = model_action_dim
        self.use_quantile_norm = use_quantile_norm

        self._tokenizer = None
        if max_token_len is not None:
            self._tokenizer = self._create_tokenizer(max_token_len)

    @staticmethod
    def _create_tokenizer(max_len: int):
        # Lazy import — vendored model tree under workspace/nodesets/policy/policy_vla/models/openpi/
        from ...models.openpi.models.tokenizer import PaligemmaTokenizer

        tokenizer = PaligemmaTokenizer(max_len=max_len)
        logger.info(f"Created PaligemmaTokenizer with max_len={max_len}")
        return tokenizer

    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]
        info = canonical["info"]

        result = {}

        images = {}
        image_masks = {}
        ref_shape = None

        for canonical_name, pi0_name in self.CAMERA_MAP.items():
            img = data.get("images", {}).get(canonical_name)
            if img is not None:
                hwc_img = _chw_float_to_hwc_uint8(img)
                images[pi0_name] = hwc_img
                image_masks[pi0_name] = np.bool_(True)
                ref_shape = hwc_img.shape
            else:
                images[pi0_name] = None
                image_masks[pi0_name] = np.bool_(False)

        if ref_shape is not None:
            images[self.ZERO_CAMERA] = np.zeros(ref_shape, dtype=np.uint8)
        else:
            images[self.ZERO_CAMERA] = np.zeros((224, 224, 3), dtype=np.uint8)
        image_masks[self.ZERO_CAMERA] = np.bool_(False)

        for name in images:
            if images[name] is None:
                images[name] = np.zeros(ref_shape or (224, 224, 3), dtype=np.uint8)

        result["images"] = images
        result["image_masks"] = image_masks

        state_components = []
        state_data = data.get("state", {})
        for component_name in ("pos", "rot", "gripper", "joint_position"):
            if component_name in state_data and state_data[component_name] is not None:
                state_components.append(np.asarray(state_data[component_name], dtype=np.float32))

        if state_components:
            result["state"] = np.concatenate(state_components, axis=-1)
        else:
            result["state"] = np.zeros(0, dtype=np.float32)

        action_data = data.get("actions", {})
        if action_data:
            action_components = []
            for component_name in ("pos", "rot", "gripper", "joint_position"):
                if component_name in action_data and action_data[component_name] is not None:
                    action_components.append(
                        np.asarray(action_data[component_name], dtype=np.float32)
                    )
            if action_components:
                result["actions"] = np.concatenate(action_components, axis=-1)

        if self._norm_stats is not None:
            self._normalize(result)

        result["images"] = {k: resize_with_pad(v, 224, 224) for k, v in result["images"].items()}

        prompt = data.get("prompt", "")
        if self._tokenizer and prompt:
            if not isinstance(prompt, str):
                prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
            tokens, token_masks = self._tokenizer.tokenize(prompt, state=None)
            result["tokenized_prompt"] = tokens
            result["tokenized_prompt_mask"] = token_masks
        elif self._tokenizer:
            tokens, token_masks = self._tokenizer.tokenize("", state=None)
            result["tokenized_prompt"] = tokens
            result["tokenized_prompt_mask"] = token_masks

        if "state" in result:
            result["state"] = self._pad_to_dim(result["state"], self.model_action_dim)
        if "actions" in result:
            result["actions"] = self._pad_to_dim(result["actions"], self.model_action_dim)

        return result

    def model_to_canonical(self, model_output: dict, info: CanonicalInfo) -> CanonicalDict:
        actions = np.asarray(_to_numpy(model_output["action"]), dtype=np.float32)

        if self._norm_stats is not None:
            actions = self._unnormalize_actions(actions)

        actual_dim = sum(info.action_dims.values())
        actions = actions[..., :actual_dim]

        action_components = {}
        offset = 0
        for component_name in ("pos", "rot", "gripper", "joint_position"):
            if component_name in info.action_dims:
                dim = info.action_dims[component_name]
                action_components[component_name] = actions[..., offset : offset + dim]
                offset += dim

        return make_canonical_action(actions=action_components, info=info)

    def get_norm_stats_mode(self) -> str:
        return "gaussian"

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state", "actions")

    def model_input(self) -> dict:
        return {
            "images": {
                "base_0_rgb": "[224, 224, 3] uint8 [0, 255]",
                "left_wrist_0_rgb": "[224, 224, 3] uint8 [0, 255]",
                "right_wrist_0_rgb": "[224, 224, 3] uint8 [0, 255]",
            },
            "image_masks": {
                "base_0_rgb": "bool",
                "left_wrist_0_rgb": "bool",
                "right_wrist_0_rgb": "bool",
            },
            "state": f"[{self.model_action_dim}] float32",
            "actions": f"[horizon, {self.model_action_dim}] float32",
            "tokenized_prompt": f"[{self.max_token_len}] int64",
            "tokenized_prompt_mask": f"[{self.max_token_len}] bool",
        }

    def model_output(self) -> dict:
        return {"actions": f"[horizon, {self.model_action_dim}] float32"}

    def canonical_to_norm_stats_format(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]
        result = {}

        state_components = []
        state_data = data.get("state", {})
        for component_name in ("pos", "rot", "gripper", "joint_position"):
            if component_name in state_data and state_data[component_name] is not None:
                state_components.append(np.asarray(state_data[component_name], dtype=np.float32))
        if state_components:
            result["state"] = np.concatenate(state_components, axis=-1)

        action_components = []
        action_data = data.get("actions", {})
        for component_name in ("pos", "rot", "gripper", "joint_position"):
            if component_name in action_data and action_data[component_name] is not None:
                action_components.append(np.asarray(action_data[component_name], dtype=np.float32))
        if action_components:
            result["actions"] = np.concatenate(action_components, axis=-1)

        return result

    def _normalize(self, data: dict) -> None:
        for key in ("state", "actions"):
            if key not in data or key not in self._norm_stats:
                continue
            stats = self._norm_stats[key]
            x = np.asarray(data[key], dtype=np.float32)

            if self.use_quantile_norm:
                q01 = np.asarray(stats["q01"], dtype=np.float32)[..., : x.shape[-1]]
                q99 = np.asarray(stats["q99"], dtype=np.float32)[..., : x.shape[-1]]
                data[key] = ((x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)
            else:
                mean = np.asarray(stats["mean"], dtype=np.float32)[..., : x.shape[-1]]
                std = np.asarray(stats["std"], dtype=np.float32)[..., : x.shape[-1]]
                data[key] = ((x - mean) / (std + 1e-6)).astype(np.float32)

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if "actions" not in self._norm_stats:
            return actions

        stats = self._norm_stats["actions"]

        if self.use_quantile_norm:
            q01 = np.asarray(stats["q01"], dtype=np.float32)
            q99 = np.asarray(stats["q99"], dtype=np.float32)
            dim = q01.shape[-1]
            if dim < actions.shape[-1]:
                normalized_part = (actions[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                return np.concatenate([normalized_part, actions[..., dim:]], axis=-1).astype(
                    np.float32
                )
            return ((actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01).astype(np.float32)
        else:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            if mean.shape[-1] < actions.shape[-1]:
                pad_len = actions.shape[-1] - mean.shape[-1]
                mean = np.concatenate([mean, np.zeros(pad_len, dtype=np.float32)])
                std = np.concatenate([std, np.ones(pad_len, dtype=np.float32)])
            return (actions * (std + 1e-6) + mean).astype(np.float32)

    @staticmethod
    def _pad_to_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
        x = np.asarray(x)
        current_dim = x.shape[-1]
        if current_dim < target_dim:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target_dim - current_dim)]
            return np.pad(x, pad_width, constant_values=0.0)
        return x


# norm_stats vendored into the nodeset (policy_vla/_assets/norm_stats/); resolve
# relative to this file so it works regardless of CWD or install location.
_NORM_STATS_DIR = Path(__file__).resolve().parents[2] / "_assets" / "norm_stats"

# ───── DEFAULTS (transcribed from CORL_pi0_libero_lerobot.yaml) ─────
DEFAULT_KWARGS: dict = {
    "norm_stats_path": str(_NORM_STATS_DIR / "norm_stats.json"),
    "max_token_len": 48,
    "model_action_dim": 32,
    "use_quantile_norm": False,
}
