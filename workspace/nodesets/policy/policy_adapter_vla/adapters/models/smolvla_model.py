"""SmolVLA model adaptor.

Vendored from vlaworkspace/adaptors/models/smolvla_model.py — no upstream import.

SmolVLA model format:
    Input:
        - observation.images.front: [C, H, W] float32 [0,1]
        - observation.images.wrist: [C, H, W] float32 [0,1] (optional, zero if missing)
        - observation.state: [state_dim] float32, z-score normalized
        - action: [horizon, action_dim] float32, z-score normalized
        - observation.language.tokens: [max_token_len] int64
        - observation.language.attention_mask: [max_token_len] bool
    Output:
        - action: [horizon, action_dim] float32 normalized
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base_model import ModelAdaptor
from ..canonical import CanonicalDict, CanonicalInfo, make_canonical_action

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


class SmolVLAModel(ModelAdaptor):
    """Model adaptor for SmolVLA."""

    def __init__(
        self,
        *,
        norm_stats_path: str | None = None,
        norm_stats: dict | None = None,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        max_token_len: int = 48,
    ) -> None:
        super().__init__(norm_stats_path=norm_stats_path, norm_stats=norm_stats)
        self.tokenizer_name = tokenizer_name
        self.max_token_len = max_token_len
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.tokenizer_name)
            self._tokenizer = processor.tokenizer
            logger.info(f"Loaded SmolVLM tokenizer from {self.tokenizer_name}")
        return self._tokenizer

    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]

        result = {}

        images_data = data.get("images", {})
        front_img = images_data.get("front")
        wrist_img = images_data.get("wrist")

        if front_img is not None:
            result["observation.images.front"] = np.asarray(front_img, dtype=np.float32)
        else:
            result["observation.images.front"] = np.zeros((3, 256, 256), dtype=np.float32)

        if wrist_img is not None:
            result["observation.images.wrist"] = np.asarray(wrist_img, dtype=np.float32)
        else:
            result["observation.images.wrist"] = np.zeros(
                result["observation.images.front"].shape, dtype=np.float32
            )

        state_components = []
        state_data = data.get("state", {})
        for component_name in ("pos", "rot", "gripper", "joint_position"):
            if component_name in state_data and state_data[component_name] is not None:
                state_components.append(np.asarray(state_data[component_name], dtype=np.float32))

        if state_components:
            flat_state = np.concatenate(state_components, axis=-1)
        else:
            flat_state = np.zeros(0, dtype=np.float32)

        action_data = data.get("actions", {})
        flat_actions = None
        if action_data:
            action_components = []
            for component_name in ("pos", "rot", "gripper", "joint_position"):
                if component_name in action_data and action_data[component_name] is not None:
                    action_components.append(np.asarray(action_data[component_name], dtype=np.float32))
            if action_components:
                flat_actions = np.concatenate(action_components, axis=-1)

        norm_data = {"state": flat_state}
        if flat_actions is not None:
            norm_data["actions"] = flat_actions

        if self._norm_stats is not None:
            self._normalize(norm_data)

        result["observation.state"] = norm_data["state"]
        if "actions" in norm_data:
            result["action"] = norm_data["actions"]

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

        if prompt and not prompt.endswith("\n"):
            prompt = prompt + "\n"
        elif not prompt:
            prompt = "\n"

        tokenizer = self._get_tokenizer()
        encoded = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="np",
        )
        result["observation.language.tokens"] = encoded["input_ids"][0].astype(np.int64)
        result["observation.language.attention_mask"] = encoded["attention_mask"][0].astype(np.bool_)

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
                action_components[component_name] = actions[..., offset:offset + dim]
                offset += dim

        return make_canonical_action(actions=action_components, info=info)

    def get_norm_stats_mode(self) -> str:
        return "gaussian"

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ("state", "actions")

    def model_input(self) -> dict:
        return {
            "observation.images.front": "[C, H, W] float32 [0, 1]",
            "observation.images.wrist": "[C, H, W] float32 [0, 1]",
            "observation.state": "[state_dim] float32 normalized",
            "action": "[horizon, action_dim] float32 normalized",
            "observation.language.tokens": f"[{self.max_token_len}] int64",
            "observation.language.attention_mask": f"[{self.max_token_len}] bool",
        }

    def model_output(self) -> dict:
        return {"action": "[horizon, action_dim] float32 normalized"}

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
            mean = np.asarray(stats["mean"], dtype=np.float32)[..., :x.shape[-1]]
            std = np.asarray(stats["std"], dtype=np.float32)[..., :x.shape[-1]]
            data[key] = ((x - mean) / (std + 1e-8)).astype(np.float32)

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if "actions" not in self._norm_stats:
            return actions

        stats = self._norm_stats["actions"]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

        dim = mean.shape[-1]
        if dim < actions.shape[-1]:
            real_part = actions[..., :dim] * std + mean
            return np.concatenate([real_part, actions[..., dim:]], axis=-1).astype(np.float32)
        return (actions * std + mean).astype(np.float32)


# norm_stats vendored into the nodeset (policy_vla/_assets/norm_stats/); resolve
# relative to this file so it works regardless of CWD or install location.
_NORM_STATS_DIR = Path(__file__).resolve().parents[2] / "_assets" / "norm_stats"

# ───── DEFAULTS (transcribed from smolvla_libero.yaml) ─────
DEFAULT_KWARGS: dict = {
    "norm_stats_path": str(_NORM_STATS_DIR / "norm_stats_smolvla.json"),
    "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "max_token_len": 48,
}
