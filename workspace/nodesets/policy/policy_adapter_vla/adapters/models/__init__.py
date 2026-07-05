"""Model adapter sub-package — model-coupled side of the Adapter system.

Each .py file that defines a `ModelAdaptor` subclass and a module-level
`DEFAULT_KWARGS` dict is auto-discovered by the canvas
`adapt_canonical_to_model` node and surfaces as a dropdown option.

`dp_defaults.py` is excluded from the dropdown — it's a constants module,
not an adapter (no `ModelAdaptor` subclass).
"""

from __future__ import annotations

from .base_model import ModelAdaptor
from .dp_model import DPModel
from .pi0_model import Pi0Model
from .rt1_model import Rt1Model
from .smolvla_model import SmolVLAModel

__all__ = ["DPModel", "ModelAdaptor", "Pi0Model", "Rt1Model", "SmolVLAModel"]
