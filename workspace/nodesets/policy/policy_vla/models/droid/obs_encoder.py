"""Observation encoder for DROID Diffusion Policy.

Architecture:
  RGB: Resize(128) -> ColorJitter -> RandomCrop(116) -> ResNet50(ImageNet)
       -> SpatialSoftmax(32 kp) -> Flatten -> Linear(64, 512)
  Low-dim: identity (flatten)
  All -> Concat -> MLP(total -> 1024 -> 512 -> 512)

Only depends on torch, torchvision, numpy.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class ResNet50Conv(nn.Module):
    """ResNet-50 backbone (ImageNet pretrained) without FC layers.

    Produces spatial feature maps of shape [B, 2048, ceil(H/32), ceil(W/32)].
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        # Remove avgpool and fc
        self.nets = nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """Compute output shape given input [C, H, W]."""
        c, h, w = input_shape
        return [2048, math.ceil(h / 32), math.ceil(w / 32)]

    def forward(self, x):
        return self.nets(x)


class SpatialSoftmax(nn.Module):
    """Spatial Softmax layer that outputs expected (x, y) keypoint coordinates.

    Input: [B, C, H, W] feature maps
    Output: [B, num_kp, 2] expected keypoint coordinates
    """

    def __init__(self, input_shape, num_kp=32, temperature=1.0, learnable_temperature=False):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.num_kp = num_kp

        # 1x1 conv to reduce channels to num_kp
        self.nets = nn.Conv2d(self._in_c, self.num_kp, kernel_size=1)

        # Temperature
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
        else:
            self.temperature = temperature

        # Create position grids
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def output_shape(self, input_shape=None):
        return [self.num_kp, 2]

    def forward(self, feature):
        feature = self.nets(feature)  # [B, num_kp, H, W]
        feature = feature.reshape(-1, self.num_kp, self._in_h * self._in_w)
        attention = torch.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)
        return torch.cat([expected_x, expected_y], dim=-1)  # [B, num_kp, 2]


class VisualCore(nn.Module):
    """Visual encoder: Resize -> Augment -> ResNet50 -> SpatialSoftmax -> Flatten -> Linear(512).

    Augmentation (ColorJitter + RandomCrop) only active during training.
    Center crop used during eval.
    """

    def __init__(
        self,
        input_shape,
        feature_dimension=512,
        num_kp=32,
        image_size=128,
        crop_size=116,
        color_jitter=True,
        pretrained=True,
    ):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.image_size = image_size
        self.crop_size = crop_size

        # Augmentation transforms
        self.resize = T.Resize((image_size, image_size), antialias=True)

        jitter_params = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.color_jitter = T.ColorJitter(**jitter_params) if color_jitter else None
        self.random_crop = T.RandomCrop(crop_size)
        self.center_crop = T.CenterCrop(crop_size)

        # Backbone
        self.backbone = ResNet50Conv(pretrained=pretrained)
        backbone_out = self.backbone.output_shape([input_shape[0], crop_size, crop_size])

        # SpatialSoftmax pooling
        self.pool = SpatialSoftmax(input_shape=backbone_out, num_kp=num_kp)
        pool_out = self.pool.output_shape()  # [num_kp, 2]
        pool_out_dim = pool_out[0] * pool_out[1]  # num_kp * 2

        self.fc = nn.Linear(pool_out_dim, feature_dimension)

    def output_shape(self):
        return [self.feature_dimension]

    def forward(self, x):
        # Rescale from [-1,1] to [0,1] (ColorJitter needs non-negative for HSV)
        x = (x + 1.0) / 2.0

        # Resize
        x = self.resize(x)

        # Augmentation (training only)
        if self.training:
            if self.color_jitter is not None:
                x = self.color_jitter(x)
            x = self.random_crop(x)
        else:
            x = self.center_crop(x)

        # Backbone + pool + flatten + linear
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)  # [B, num_kp * 2]
        x = self.fc(x)  # [B, feature_dimension]
        return x


class ObservationEncoder(nn.Module):
    """Multi-modal observation encoder with MLP head.

    For each obs key: if RGB -> VisualCore(512), if low_dim -> identity.
    Concat all -> MLP(total -> 1024 -> 512 -> 512).
    Output: [B, 512].
    """

    def __init__(
        self,
        obs_key_shapes,
        obs_config,
        feature_dimension=512,
        image_size=128,
        crop_size=116,
        color_jitter=True,
    ):
        super().__init__()

        self.obs_key_shapes = obs_key_shapes
        self.obs_config = obs_config
        self.feature_dimension = feature_dimension

        nets = OrderedDict()
        output_dims = OrderedDict()

        for key in obs_config.get("rgb", []):
            if key in obs_key_shapes:
                shape = obs_key_shapes[key]
                nets[key] = VisualCore(
                    input_shape=shape,
                    feature_dimension=feature_dimension,
                    image_size=image_size,
                    crop_size=crop_size,
                    color_jitter=color_jitter,
                )
                output_dims[key] = feature_dimension

        for key in obs_config.get("low_dim", []):
            if key in obs_key_shapes:
                shape = obs_key_shapes[key]
                dim = 1
                for s in shape:
                    dim *= s
                output_dims[key] = dim

        self.nets = nn.ModuleDict(nets)
        self.output_dims = output_dims

        # MLP head: concat_dim -> 1024 -> 512 -> 512
        # Skip MLP when there are no RGB keys (low-dim only) — pass raw features
        # through directly, matching MimicGen DAH-DP behavior and keeping
        # global_cond_dim small for stage 1 training.
        concat_dim = sum(output_dims.values())
        has_rgb = len(obs_config.get("rgb", [])) > 0
        if has_rgb:
            self.mlp = nn.Sequential(
                nn.Linear(concat_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
            self._output_dim = 512
        else:
            self.mlp = None
            self._output_dim = concat_dim

    def output_shape(self):
        """Return output dimension — 512 with MLP, raw concat_dim without."""
        return [self._output_dim]

    def forward(self, obs_dict):
        features = []

        for key in self.obs_config.get("rgb", []):
            if key in obs_dict and key in self.nets:
                features.append(self.nets[key](obs_dict[key]))

        for key in self.obs_config.get("low_dim", []):
            if key in obs_dict:
                x = obs_dict[key]
                features.append(x.reshape(x.shape[0], -1))

        x = torch.cat(features, dim=-1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x


def create_obs_encoder(
    obs_key_shapes,
    obs_config,
    feature_dimension=512,
    obs_encoder_group_norm=True,
    image_size=128,
    crop_size=116,
    color_jitter=True,
):
    """Factory function to create an ObservationEncoder with optional GroupNorm replacement.

    Args:
        obs_key_shapes: Dict mapping obs key names to shapes
        obs_config: Dict with 'rgb' and 'low_dim' key lists
        feature_dimension: Output dim per visual encoder (before MLP)
        obs_encoder_group_norm: If True, replace BatchNorm2d with GroupNorm
        image_size: Resize target for images
        crop_size: Crop size (random during train, center during eval)
        color_jitter: Whether to apply ColorJitter augmentation during training

    Returns:
        ObservationEncoder instance
    """
    from workspace.nodesets.policy.policy_vla.models.droid.conditional_unet1d import (
        replace_bn_with_gn,
    )

    encoder = ObservationEncoder(
        obs_key_shapes=obs_key_shapes,
        obs_config=obs_config,
        feature_dimension=feature_dimension,
        image_size=image_size,
        crop_size=crop_size,
        color_jitter=color_jitter,
    )

    if obs_encoder_group_norm:
        encoder = replace_bn_with_gn(encoder)

    return encoder
