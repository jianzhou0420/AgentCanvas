"""RxR-CE env adapter — stage 1 + stage 5 for RxR-CE family.

RxR-CE config (rxr_vlnce_english_task.yaml et al.):
  - 6-action discrete: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN
  - RGB 640×480, depth 640×480, both 79° HFOV
  - sliding=False, agent height 0.88m
  - Instruction: precomputed BERT features under raw_obs["rxr_instruction"]
    (shape (512, 768) float32 zero-padded; NOT a dict with "tokens").

The model adapter (cma.py) routes on info.instruction_kind:
  - "vocab" (R2R): instruction goes back into raw_obs["instruction"]
  - "feat"  (RxR): instruction goes back into raw_obs["rxr_instruction"]
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


_RXRCE_INFO = CanonicalNavInfo(
    env_family="rxrce",
    action_dim=6,
    image_size=(640, 480),
    depth_max=5.0,
    instruction_kind="feat",
)


class RxrceEnv(VlnEnvAdaptor):
    """RxR-CE adapter: BERT-feat instruction, 6-action discrete."""

    env_family = "rxrce"

    def get_canonical_info(self) -> CanonicalNavInfo:
        return _RXRCE_INFO

    def env_to_canonical(
        self, raw_obs: dict[str, Any], instruction: Any = None
    ) -> CanonicalDict:
        rgb = np.asarray(raw_obs.get("rgb"))
        depth = np.asarray(raw_obs.get("depth"), dtype=np.float32)
        # RxR uses precomputed BERT features; prefer the dedicated instruction
        # port if wired, else the BERT-feat sensor bundled in raw_obs. This is
        # already the standardized form — no per-policy processing here.
        embedding = instruction if instruction is not None else raw_obs.get("rxr_instruction")

        return make_canonical_obs(
            rgb=rgb,
            depth=depth,
            instruction={
                "kind": "feat",
                "embedding": embedding,
            },
            step_meta={
                "step_index": int(raw_obs.get("step_index", 0)),
                "episode_id": str(raw_obs.get("episode_id", "")),
            },
            info=_RXRCE_INFO,
        )

    def canonical_to_env(self, canonical_action: CanonicalDict) -> int:
        action_index = int(canonical_action["data"]["action_index"])
        return max(0, min(action_index, _RXRCE_INFO.action_dim - 1))
