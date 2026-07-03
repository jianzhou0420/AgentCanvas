"""Abstract VlnModelAdaptor — stage 2 + stage 4 of the 5-stage adapter pipeline.

Stage 2 (canonical → model_batch): builds the model-specific input dict
(applies vlnce-baselines obs_transforms, tokenizes instruction, stacks
rnn state). Stage 4 (model_output → canonical_action): extracts the
sampled action index from the policy's logits/distribution output.

Subclasses live in this folder as one ``<arch>.py`` file each (filename
is the dropdown option). Each module must export:
  - one VlnModelAdaptor subclass
  - module-level ``DEFAULT_KWARGS: dict``  (will receive runtime
    ``exp_config_path`` from the canvas node config; merged into
    ctor kwargs by the manager).

Stage 2 owns the ``exp_config`` runtime field because the YAML drives:
  - obs_transforms list (ResizeShortestEdge / CenterCropperPerSensor / …)
  - INSTRUCTION_SENSOR_UUID
  - MODEL config (passed downstream to the policy ctor in stage 3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from workspace.nodesets.policy.policy_adapter_vlnce.adapters.canonical import (
    CanonicalDict,
)


class VlnModelAdaptor(ABC):
    """Abstract base for model-arch adapters."""

    arch: str = ""  # "cma" | "seq2seq" — concrete subclass overrides

    @abstractmethod
    def __init__(self, *, exp_config_path: str) -> None:
        """Load the experiment YAML, build obs_transforms and policy config."""

    @property
    @abstractmethod
    def exp_config_path(self) -> str: ...

    @property
    @abstractmethod
    def policy_config(self) -> Any:
        """The full vlnce_baselines config object (for CMANet ctor)."""

    @abstractmethod
    def canonical_to_model(
        self,
        canonical: CanonicalDict,
        *,
        hidden_in: dict[str, Any] | None,
        device: Any,
    ) -> dict[str, Any]:
        """Pack canonical into a model-ready batch dict for one sample.

        Returns dict with at least:
          - "obs_batch":     dict[str, Tensor]   (post obs_transforms, batched B=1)
          - "rnn_states":    Tensor (1, num_layers, hidden_size)
          - "prev_actions":  Tensor (1, 1) long
          - "not_done_masks": Tensor (1, 1) uint8
        """

    @abstractmethod
    def model_to_canonical(
        self,
        model_output: dict[str, Any],
        info: Any,
    ) -> CanonicalDict:
        """Pick action index from model_output and wrap in canonical_action."""

    @abstractmethod
    def derive_obs_space(self, sample_canonical: CanonicalDict) -> Any:
        """Build the gym Dict obs_space the policy ctor expects.

        Called once by the manager when ensure_policy first runs (it
        needs the *transformed* obs_space to size CMANet correctly).
        """

    @abstractmethod
    def derive_action_space(self, info: Any) -> Any:
        """Build the gym Discrete action_space (4 for R2R, 6 for RxR)."""
