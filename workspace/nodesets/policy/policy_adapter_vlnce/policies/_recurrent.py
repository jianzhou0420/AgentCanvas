"""Private helpers shared by CMA + Seq2Seq policy wrappers.

Lifted verbatim from the legacy ``policy_cma.py:262-380`` inline
class hierarchy. Filename starts with ``_`` so the canvas-dropdown
discovery scan skips it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def _torch_imports() -> tuple:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def make_recurrent_policy_classes() -> tuple[type, type]:
    """Construct (BaseVLNPolicy, RecurrentPolicy) on demand.

    Defined as a factory so importing this module doesn't pull torch.
    Caller invokes once per process (the resulting classes are cached
    inside the per-process VlnceManager).
    """
    import torch

    from habitat_baselines.utils.common import CategoricalNet

    class BaseVLNPolicy(torch.nn.Module, ABC):
        def __init__(self) -> None:
            super().__init__()

        @abstractmethod
        def act(self, observations: dict[str, Any], **kwargs: Any) -> Any: ...

        def load_checkpoint(self, path: str, map_location: str = "cpu") -> None:
            ckpt = torch.load(path, map_location=map_location)
            if "state_dict" in ckpt:
                self.load_state_dict(ckpt["state_dict"])
            else:
                self.load_state_dict(ckpt)

        def reset(self) -> None:
            pass

        @property
        def device(self) -> Any:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    class RecurrentPolicy(BaseVLNPolicy):
        def __init__(self, net: Any, dim_actions: int) -> None:
            super().__init__()
            self.net = net
            self.dim_actions = dim_actions
            self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)

        def act(
            self,
            observations: dict[str, Any],
            rnn_states: Any,
            prev_actions: Any,
            masks: Any,
            deterministic: bool = False,
        ) -> tuple[Any, Any]:
            features, rnn_states_out = self.net(observations, rnn_states, prev_actions, masks)
            distribution = self.action_distribution(features)
            action = distribution.mode() if deterministic else distribution.sample()
            return action, rnn_states_out

        @property
        def num_recurrent_layers(self) -> int:
            return self.net.num_recurrent_layers

        @property
        def hidden_size(self) -> int:
            return getattr(self.net, "_hidden_size", -1)

    return BaseVLNPolicy, RecurrentPolicy
