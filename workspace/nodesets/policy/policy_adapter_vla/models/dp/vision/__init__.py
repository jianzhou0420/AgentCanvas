"""Vision encoder modules for observation processing.

This package provides modular components for visual encoding:

- ResNet18Conv: Convolutional backbone for feature extraction
- SpatialSoftmax: Spatial pooling to extract keypoint coordinates
- VisualCore: Complete visual encoder (backbone + pooling + projection)
- ObservationEncoder: Multi-modal encoder for multiple observation keys
- CropRandomizer: Data augmentation via random cropping (for encoder use)
"""

from __future__ import annotations

from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.crop_randomizer import (
    CropRandomizer,
    crop_image_from_indices,
    sample_random_image_crops,
)
from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.obs_encoder import (
    ObservationEncoder,
    create_obs_encoder,
)
from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.resnet import (
    CoordConv2d,
    ResNet18Conv,
)
from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.spatial_softmax import SpatialSoftmax
from workspace.nodesets.policy.policy_adapter_vla.models.dp.vision.visual_core import VisualCore

__all__ = [
    # Backbone
    "ResNet18Conv",
    "CoordConv2d",
    # Pooling
    "SpatialSoftmax",
    # Complete encoders
    "VisualCore",
    "ObservationEncoder",
    "create_obs_encoder",
    # Augmentation (for encoder-level use)
    "CropRandomizer",
    "crop_image_from_indices",
    "sample_random_image_crops",
]
