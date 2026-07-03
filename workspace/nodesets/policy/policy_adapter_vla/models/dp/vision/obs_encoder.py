"""Observation Encoder for processing multi-modal observations.

This module provides an encoder that handles multiple observation keys (e.g.,
multiple camera views) and produces a concatenated feature representation.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.visual_core import VisualCore

# Observation modality types
OBS_MODALITY_RGB = "rgb"
OBS_MODALITY_LOW_DIM = "low_dim"


class ObservationEncoder(nn.Module):
    """Encoder that processes multi-modal observations.

    Handles multiple observation keys (e.g., agentview_image, robot0_eye_in_hand_image)
    and concatenates their features into a single output vector.

    For RGB observations, uses VisualCore (ResNet18 + SpatialSoftmax + Linear).
    For low_dim observations, passes through with optional linear projection.

    NOTE: Unlike robomimic's ObservationEncoder, this version does NOT include
    built-in randomizers. Cropping/augmentation should be done in the adaptor
    transform pipeline.

    Args:
        obs_shapes: Dictionary mapping observation key names to their shapes.
        obs_config: Dictionary specifying modality type for each key category.
            Expected format: {'rgb': [key1, key2], 'low_dim': [key3, key4]}
        feature_dimension: Output feature dimension per RGB observation.
        feature_activation: Activation function class to apply after each encoder.
        backbone_kwargs: Kwargs passed to ResNet18Conv backbone.
        pool_kwargs: Kwargs passed to SpatialSoftmax pooling.
    """

    def __init__(
        self,
        obs_shapes: dict[str, list[int]],
        obs_config: dict[str, list[str]] | None = None,
        feature_dimension: int = 64,
        feature_activation: type[nn.Module] | None = nn.ReLU,
        backbone_kwargs: dict | None = None,
        pool_kwargs: dict | None = None,
    ):
        super().__init__()

        self.obs_shapes = OrderedDict(obs_shapes)
        self.feature_dimension = feature_dimension
        self.backbone_kwargs = backbone_kwargs or {}
        self.pool_kwargs = pool_kwargs or {}

        # Build modality mapping from obs_config
        self._obs_modalities = {}
        if obs_config is not None:
            for modality, keys in obs_config.items():
                for key in keys:
                    self._obs_modalities[key] = modality

        # Infer modality from shape if not specified
        for key, shape in self.obs_shapes.items():
            if key not in self._obs_modalities:
                # Assume RGB if 3D shape with channels, otherwise low_dim
                if len(shape) == 3 and shape[0] in (1, 3, 4):
                    self._obs_modalities[key] = OBS_MODALITY_RGB
                else:
                    self._obs_modalities[key] = OBS_MODALITY_LOW_DIM

        # Create encoder networks
        self.obs_nets = nn.ModuleDict()
        for key, shape in self.obs_shapes.items():
            modality = self._obs_modalities[key]
            if modality == OBS_MODALITY_RGB:
                self.obs_nets[key] = VisualCore(
                    input_shape=shape,
                    backbone_kwargs=deepcopy(self.backbone_kwargs),
                    pool_kwargs=deepcopy(self.pool_kwargs),
                    feature_dimension=feature_dimension,
                )
            else:
                # Low-dim: identity (flattening done in forward)
                self.obs_nets[key] = None

        # Optional activation after each encoder
        self.activation = None
        if feature_activation is not None:
            self.activation = feature_activation()

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observations and concatenate features.

        Args:
            obs_dict: Dictionary mapping observation key names to tensors.
                All keys in self.obs_shapes must be present.

        Returns:
            Concatenated features of shape [B, D] where D is the sum of all
            observation feature dimensions.
        """
        # Ensure all required keys are present
        assert set(self.obs_shapes.keys()).issubset(obs_dict.keys()), (
            f"obs_dict keys {list(obs_dict.keys())} missing required keys "
            f"{list(self.obs_shapes.keys())}"
        )

        # Process each modality in order
        feats = []
        for key in self.obs_shapes:
            x = obs_dict[key]

            # Maybe process with encoder network
            if self.obs_nets[key] is not None:
                x = self.obs_nets[key](x)
                if self.activation is not None:
                    x = self.activation(x)

            # Flatten to [B, D]
            x = x.reshape(x.shape[0], -1)
            feats.append(x)

        # Concatenate all features
        return torch.cat(feats, dim=-1)

    def output_shape(self) -> list[int]:
        """Compute the total output feature dimension.

        Returns:
            Output shape [D] where D is the sum of all observation features.
        """
        feat_dim = 0
        for key, shape in self.obs_shapes.items():
            if self.obs_nets[key] is not None:
                key_shape = self.obs_nets[key].output_shape()
            else:
                key_shape = shape
            feat_dim += int(np.prod(key_shape))
        return [feat_dim]

    def __repr__(self) -> str:
        """Pretty print encoder."""
        lines = [f"{self.__class__.__name__}("]
        for key in self.obs_shapes:
            lines.append("  Key(")
            lines.append(f"    name={key}")
            lines.append(f"    shape={self.obs_shapes[key]}")
            lines.append(f"    modality={self._obs_modalities[key]}")
            lines.append(f"    net={self.obs_nets[key]}")
            lines.append("  )")
        lines.append(f"  output_shape={self.output_shape()}")
        lines.append(")")
        return "\n".join(lines)


def create_obs_encoder(
    obs_key_shapes: dict[str, list[int]],
    obs_config: dict[str, list[str]],
    feature_dimension: int = 64,
    obs_encoder_group_norm: bool = False,
    backbone_kwargs: dict | None = None,
    pool_kwargs: dict | None = None,
) -> ObservationEncoder:
    """Factory function to create an ObservationEncoder.

    This is a convenience function that matches the interface expected by
    the diffusion policy classes.

    Args:
        obs_key_shapes: Dictionary mapping observation key names to their shapes.
        obs_config: Dictionary specifying modality type for each key category.
            Expected format: {'rgb': [key1, key2], 'low_dim': [key3, key4]}
        feature_dimension: Output feature dimension per RGB observation.
        obs_encoder_group_norm: If True, replace BatchNorm with GroupNorm.
        backbone_kwargs: Kwargs passed to ResNet18Conv backbone.
        pool_kwargs: Kwargs passed to SpatialSoftmax pooling.

    Returns:
        Configured ObservationEncoder instance.
    """
    encoder = ObservationEncoder(
        obs_shapes=obs_key_shapes,
        obs_config=obs_config,
        feature_dimension=feature_dimension,
        backbone_kwargs=backbone_kwargs,
        pool_kwargs=pool_kwargs,
    )

    # Optionally replace BatchNorm with GroupNorm
    if obs_encoder_group_norm:
        _replace_batchnorm_with_groupnorm(encoder)

    return encoder


def _replace_batchnorm_with_groupnorm(module: nn.Module) -> None:
    """Replace all BatchNorm2d layers with GroupNorm.

    Args:
        module: Module to modify in-place.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = num_channels // 16
            if num_groups < 1:
                num_groups = 1
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            _replace_batchnorm_with_groupnorm(child)
