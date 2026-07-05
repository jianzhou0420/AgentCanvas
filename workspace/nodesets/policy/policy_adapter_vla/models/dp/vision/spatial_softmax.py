"""Spatial Softmax pooling layer for visual features.

Based on "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
https://rll.berkeley.edu/dsae/dsae.pdf
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial Softmax Layer.

    Converts spatial feature maps into keypoint coordinates by treating each
    channel as a probability distribution over spatial locations.

    For each keypoint, a 2D spatial probability distribution is created using
    a softmax, where the support is the pixel locations. This distribution is
    used to compute the expected value of the pixel location, which becomes a
    keypoint of dimension 2. K such keypoints are created.

    Output shape: [B, num_kp, 2] where 2 represents (x, y) coordinates in [-1, 1].

    Args:
        input_shape: Shape of input feature maps [C, H, W].
        num_kp: Number of keypoints to extract. If None, uses input channels.
        temperature: Temperature for softmax (lower = sharper attention).
        learnable_temperature: If True, temperature is learned during training.
        output_variance: If True, also outputs variance of keypoint locations.
        noise_std: Standard deviation of noise added during training.
    """

    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()

        assert len(input_shape) == 3, f"Expected [C, H, W], got {input_shape}"
        self._in_c, self._in_h, self._in_w = input_shape

        # Optional 1x1 conv to reduce channels to num_kp
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c

        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        # Temperature parameter
        if self.learnable_temperature:
            temperature_param = nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True
            )
            self.register_parameter("temperature", temperature_param)
        else:
            temperature_param = nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False
            )
            self.register_buffer("temperature", temperature_param)

        # Create position grids for computing expected coordinates
        # Positions normalized to [-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        # Store last keypoints for visualization
        self.kps = None

    def forward(self, feature: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through spatial softmax layer.

        Args:
            feature: Input feature maps [B, C, H, W].

        Returns:
            Mean keypoints of shape [B, K, 2], and optionally keypoint variance
            of shape [B, K, 2, 2] if output_variance=True.
        """
        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w

        # Optional channel reduction
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W]
        feature = feature.reshape(-1, self._in_h * self._in_w)

        # 2D softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)

        # Compute expected x and y coordinates
        # [1, H * W] x [B * K, H * W] -> [B * K, 1]
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)

        # Stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], dim=1)

        # Reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints = feature_keypoints + noise

        # Optionally compute variance
        if self.output_variance:
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy_cross = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy_cross - expected_x * expected_y
            # Stack to [B * K, 4] and reshape to [B, K, 2, 2]
            feature_covar = (
                torch.cat([var_x, var_xy, var_xy, var_y], dim=1)
                .reshape(-1, self._num_kp, 2, 2)
            )
            feature_keypoints = (feature_keypoints, feature_covar)

        # Store for visualization
        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()

        return feature_keypoints

    def output_shape(self, input_shape: list[int]) -> list[int]:
        """Compute output shape from input shape.

        Args:
            input_shape: Shape of input [C, H, W]. Does not include batch.

        Returns:
            Output shape [num_kp, 2].
        """
        assert len(input_shape) == 3
        assert input_shape[0] == self._in_c
        return [self._num_kp, 2]

    def __repr__(self) -> str:
        """Pretty print network."""
        return (
            f"{self.__class__.__name__}("
            f"num_kp={self._num_kp}, "
            f"temperature={self.temperature.item():.2f}, "
            f"noise={self.noise_std})"
        )
