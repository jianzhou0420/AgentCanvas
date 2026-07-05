"""R2R-CE env adapter — stage 1 + stage 5 for R2R-CE family.

R2R-CE config (vlnce_task.yaml):
  - 4-action discrete: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
  - RGB 224×224, depth 256×256, both 90° HFOV
  - sliding=True, agent height 1.5m
  - Instruction: token id sequence under raw_obs["instruction"]["tokens"]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import (
    CanonicalDict,
    CanonicalNavInfo,
    make_canonical_obs,
)
from .base_env import VlnEnvAdaptor

DEFAULT_KWARGS: dict[str, Any] = {}


_R2RCE_INFO = CanonicalNavInfo(
    env_family="r2rce",
    action_dim=4,
    image_size=(224, 224),
    depth_max=10.0,
    instruction_kind="raw_text",
)


class R2rceEnv(VlnEnvAdaptor):
    """R2R-CE adapter: vocab instruction, 4-action discrete."""

    env_family = "r2rce"

    def get_canonical_info(self) -> CanonicalNavInfo:
        return _R2RCE_INFO

    def env_to_canonical(
        self, raw_obs: dict[str, Any], instruction: Any = None
    ) -> CanonicalDict:
        rgb = np.asarray(raw_obs.get("rgb"))
        # Habitat-Sim's RGBA sensor (and the gym-interface env_habitat) emit a
        # 4-channel (H, W, 4) frame; CMA's frozen ImageNet RGB encoder and the
        # released checkpoint expect 3-channel (H, W, 3). Drop the alpha channel
        # to match native VLN-CE obs.
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]

        depth = np.asarray(raw_obs.get("depth"), dtype=np.float32)
        # CMA's depth ResNet encoder reads ``observation_space["depth"].shape[2]``
        # (resnet_policy.py), so depth MUST carry an explicit channel dim
        # (H, W, 1). The gym-interface env_habitat emits a squeezed (H, W) depth
        # — re-add the trailing channel to match native VLN-CE obs.
        if depth.ndim == 2:
            depth = depth[..., None]

        # STANDARDIZE ONLY: the instruction rides in as the raw natural-language
        # string (per-episode, from env_habitat's instruction_text port). Pass it
        # through verbatim — NO tokenization here. CMA-vocab tokenization is a
        # policy-specific step and lives in the model-side adapter
        # (cma.canonical_to_model), per the standardize/process split.
        text = instruction if isinstance(instruction, str) else None
        if not text:
            text = raw_obs.get("instruction_text") or ""
            if not text and isinstance(raw_obs.get("instruction"), str):
                text = raw_obs["instruction"]

        return make_canonical_obs(
            rgb=rgb,
            depth=depth,
            instruction={"kind": "raw_text", "text": text},
            step_meta={
                "step_index": int(raw_obs.get("step_index", 0)),
                "episode_id": str(raw_obs.get("episode_id", "")),
            },
            info=_R2RCE_INFO,
        )

    def canonical_to_env(self, canonical_action: CanonicalDict) -> int:
        action_index = int(canonical_action["data"]["action_index"])
        # Clip to R2R action range (defensive — argmax should already be valid).
        return max(0, min(action_index, _R2RCE_INFO.action_dim - 1))
