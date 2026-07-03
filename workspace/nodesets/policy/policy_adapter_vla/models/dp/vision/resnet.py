"""ResNet18 convolutional backbone for visual encoding.

This module provides a ResNet18-based convolutional backbone that can be used
to process input images for visual feature extraction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as vision_models


class CoordConv2d(nn.Conv2d):
    """Convolution with coordinate channels.

    Adds 2 extra channels encoding the (x, y) spatial position of each pixel,
    normalized to [-1, 1]. This provides the network with explicit spatial
    information about pixel locations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        # Add 2 channels for x and y coordinates
        super().__init__(
            in_channels + 2,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # Create coordinate grids normalized to [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Expand to batch dimension and add channel dimension
        # Shape: [1, 1, H, W] -> [B, 1, H, W]
        xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, h, w)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, h, w)

        # Concatenate coordinate channels with input
        x = torch.cat([x, xx, yy], dim=1)
        return super().forward(x)


class ResNet18Conv(nn.Module):
    """ResNet18 convolutional backbone for visual feature extraction.

    A ResNet18 block that can be used to process input images. Removes the
    final avgpool and fc layers to output spatial feature maps.

    Output shape: [512, H/32, W/32] where H, W are input dimensions.

    Args:
        input_channel: Number of input channels. Default 3 for RGB.
        pretrained: If True, load ImageNet pretrained weights.
        input_coord_conv: If True, use CoordConv for the first layer.
    """

    def __init__(
        self,
        input_channel: int = 3,
        pretrained: bool = False,
        input_coord_conv: bool = False,
    ):
        super().__init__()

        # Load ResNet18 with optional pretrained weights
        weights = "IMAGENET1K_V1" if pretrained else None
        net = vision_models.resnet18(weights=weights)

        # Modify first conv layer if needed
        if input_coord_conv:
            net.conv1 = CoordConv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove last 2 layers (avgpool and fc)
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet18 backbone.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Feature tensor of shape [B, 512, H/32, W/32].
        """
        return self.nets(x)

    def output_shape(self, input_shape: list[int]) -> list[int]:
        """Compute output shape from input shape.

        Args:
            input_shape: Shape of input [C, H, W]. Does not include batch.

        Returns:
            Output shape [512, H/32, W/32].
        """
        assert len(input_shape) == 3
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [512, out_h, out_w]

    def __repr__(self) -> str:
        """Pretty print network."""
        return (
            f"{self.__class__.__name__}("
            f"input_channel={self._input_channel}, "
            f"input_coord_conv={self._input_coord_conv})"
        )
