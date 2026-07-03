"""
Base Policy Interface for TaskFusion.

All VLA policies must implement this interface to be compatible with the
TaskFusion training and evaluation system.

Required methods:
    - compute_loss(batch) -> Tensor: Training forward pass
    - predict_action(batch) -> Tensor: Inference forward pass

Optional methods:
    - reset(): Reset stateful policies
    - get_optimizer(): Return custom optimizer (for LoRA, etc.)

Usage:
    class MyVLAPolicy(BasePolicy):
        def __init__(self, config):
            super().__init__()
            self.model = MyModel(config)

        def compute_loss(self, batch: dict) -> torch.Tensor:
            outputs = self.model(batch)
            return outputs.loss

        def predict_action(self, batch: dict) -> torch.Tensor:
            with torch.no_grad():
                return self.model.sample(batch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

import torch
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for all VLA policies.

    A policy transforms observations into actions. It supports both:
    - Training: compute_loss(batch) returns a scalar loss
    - Inference: predict_action(batch) returns action predictions

    Subclasses must implement:
        - compute_loss(): Training forward pass
        - predict_action(): Inference forward pass

    Example:
        class Pi0Policy(BasePolicy):
            def __init__(self, config):
                super().__init__()
                self.model = Pi0Model(config)

            def compute_loss(self, batch: dict) -> torch.Tensor:
                return self.model.forward_loss(batch)

            def predict_action(self, batch: dict) -> torch.Tensor:
                return self.model.sample_actions(batch)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            batch: Dictionary containing:
                - Observations (images, state, etc.)
                - Actions (ground truth)
                - Any additional data needed for training

        Returns:
            Scalar loss tensor for backpropagation.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_action(self, batch: dict) -> torch.Tensor:
        """
        Predict actions from observations.

        Args:
            batch: Dictionary containing observations:
                - Images (possibly multiple cameras)
                - State (robot proprioception)
                - Language prompt (tokenized)

        Returns:
            Action tensor of shape [B, T, action_dim] or [B, action_dim]
        """
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint weights. Subclasses must override."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load_checkpoint()")

    def reset(self) -> None:
        """
        Reset policy state.

        Called at the beginning of each episode for stateful policies
        (e.g., those with recurrent components or action history).
        """
        pass

    def get_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> torch.optim.Optimizer:
        """
        Get optimizer for training.

        Override this method to provide custom parameter groups or
        optimizer settings (e.g., different learning rates for LoRA).

        Args:
            lr: Learning rate
            weight_decay: Weight decay
            **kwargs: Additional optimizer arguments

        Returns:
            Optimizer instance
        """
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )

    def get_scheduler(self, optimizer, num_training_steps, last_epoch=-1, **kwargs):
        """Override to provide policy-specific LR schedule. Return None for trainer default."""
        return None

    def get_param_groups(self) -> list[dict[str, Any]] | None:
        """
        Get parameter groups for optimizer.

        Override this for custom parameter grouping (e.g., LoRA vs base params).

        Returns:
            List of param group dicts, or None to use all parameters
        """
        return None

    @property
    def device(self) -> torch.device:
        """Get the device of the policy's parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the policy's parameters."""
        return next(self.parameters()).dtype


# Type alias for policy output
PolicyOutput = Union[torch.Tensor, dict[str, torch.Tensor]]
