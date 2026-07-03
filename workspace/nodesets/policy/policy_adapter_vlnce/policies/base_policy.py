"""Abstract VlnPolicy — stage 3 (predict) of the 5-stage adapter pipeline.

Subclasses wrap the underlying vlnce_baselines model classes (CMANet,
Seq2SeqNet) and expose a uniform forward() that consumes the model_batch
dict from stage 2 and returns a model_output dict for stage 4.

Each policy module under ``policies/`` must export:
  - one VlnPolicy subclass
  - module-level ``DEFAULT_KWARGS: dict``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VlnPolicy(ABC):
    """Abstract VLN-CE policy wrapper."""

    @abstractmethod
    def build(
        self,
        *,
        observation_space: Any,
        action_space: Any,
        model_config: Any,
        obs_transforms: list,
    ) -> None:
        """Construct the underlying network. Called by the manager once
        the model_adaptor + first canonical observation are available.

        ``obs_transforms`` is the result of
        ``habitat_baselines.common.obs_transformers.get_active_obs_transforms(cfg)``
        — passed at build time so policy.forward() can apply the per-frame
        transforms in-process (avoiding a torch→numpy→torch round-trip
        across the wire from stage 2).
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def to(self, device: Any) -> "VlnPolicy": ...

    @abstractmethod
    def eval(self) -> "VlnPolicy": ...

    @abstractmethod
    def forward(self, model_batch: dict[str, Any]) -> dict[str, Any]:
        """Run policy.act on the batch; return model_output dict.

        model_output must contain at least:
          - "action_index":     int  (deterministic argmax for now)
          - "rnn_states_out":   numpy ndarray (1, num_layers, hidden_size)
        """

    @property
    @abstractmethod
    def num_recurrent_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> Any: ...

    @property
    @abstractmethod
    def num_parameters(self) -> int: ...
