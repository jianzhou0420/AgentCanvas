"""Visual Core network combining backbone + pooling + projection.

This module provides a complete visual encoder pipeline that combines:
1. Convolutional backbone (ResNet18Conv)
2. Spatial pooling (SpatialSoftmax)
3. Optional flatten and linear projection
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from workspace.nodesets.policy.policy_vla.models.dp.vision.resnet import ResNet18Conv
from workspace.nodesets.policy.policy_vla.models.dp.vision.spatial_softmax import SpatialSoftmax


class VisualCore(nn.Module):
    """Visual encoder combining backbone, pooling, and projection.

    A network block that combines a visual backbone network with optional
    pooling and linear layers.

    Architecture:
        Input [C, H, W]
        -> ResNet18Conv [512, H/32, W/32]
        -> SpatialSoftmax [num_kp, 2]
        -> Flatten [num_kp * 2]
        -> Linear [feature_dimension]
        Output [feature_dimension]

    Args:
        input_shape: Shape of input [C, H, W]. Does not include batch.
        backbone_kwargs: Kwargs for ResNet18Conv backbone.
        pool_kwargs: Kwargs for SpatialSoftmax pooling.
        flatten: Whether to flatten visual features.
        feature_dimension: If not None, add Linear to project to this dimension.
    """

    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
        backbone_kwargs: dict | None = None,
        pool_kwargs: dict | None = None,
        flatten: bool = True,
        feature_dimension: int | None = 64,
    ):
        super().__init__()

        assert len(input_shape) == 3, f"Expected [C, H, W], got {input_shape}"
        self.input_shape = list(input_shape)
        self.flatten = flatten
        self.feature_dimension = feature_dimension

        # Setup backbone kwargs
        if backbone_kwargs is None:
            backbone_kwargs = {}
        backbone_kwargs = backbone_kwargs.copy()
        backbone_kwargs["input_channel"] = input_shape[0]

        # Create backbone
        self.backbone = ResNet18Conv(**backbone_kwargs)
        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # Setup pool kwargs
        if pool_kwargs is None:
            pool_kwargs = {}
        pool_kwargs = pool_kwargs.copy()
        pool_kwargs["input_shape"] = feat_shape

        # Create pooling layer
        self.pool = SpatialSoftmax(**pool_kwargs)
        feat_shape = self.pool.output_shape(feat_shape)
        net_list.append(self.pool)

        # Flatten layer
        if self.flatten:
            net_list.append(nn.Flatten(start_dim=1, end_dim=-1))

        # Optional linear projection
        if feature_dimension is not None:
            assert self.flatten, "feature_dimension requires flatten=True"
            linear = nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through visual core.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Feature tensor of shape [B, feature_dimension] if flatten=True and
            feature_dimension is set, otherwise depends on configuration.
        """
        # Verify input shape matches expected (excluding batch)
        ndim = len(self.input_shape)
        assert tuple(x.shape)[-ndim:] == tuple(self.input_shape), (
            f"Expected input shape ending with {self.input_shape}, got {x.shape}"
        )
        return self.nets(x)

    def output_shape(self, input_shape: list[int] | None = None) -> list[int]:
        """Compute output shape from input shape.

        Args:
            input_shape: Shape of input [C, H, W]. If None, uses self.input_shape.

        Returns:
            Output shape. [feature_dimension] if set, otherwise computed from
            backbone and pool.
        """
        if input_shape is None:
            input_shape = self.input_shape

        if self.feature_dimension is not None:
            return [self.feature_dimension]

        feat_shape = self.backbone.output_shape(input_shape)
        feat_shape = self.pool.output_shape(feat_shape)

        if self.flatten:
            return [int(np.prod(feat_shape))]
        else:
            return feat_shape

    def __repr__(self) -> str:
        """Pretty print network."""
        lines = [
            f"{self.__class__.__name__}(",
            f"  input_shape={self.input_shape}",
            f"  output_shape={self.output_shape()}",
            f"  backbone={self.backbone}",
            f"  pool={self.pool}",
            ")",
        ]
        return "\n".join(lines)
