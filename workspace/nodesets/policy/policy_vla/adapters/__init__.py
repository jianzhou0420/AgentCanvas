"""Vendored adapter system for VLA policies — internal helper for policy_vla.

Sources (verbatim copies with rewritten imports, per AgentCanvas
no-upstream-import policy):
    vlaworkspace/src/vlaworkspace/adaptors/

Folder structure mirrors upstream's two-axis split:

    adapters/
    ├── adaptor.py        Adaptor composer (Robot ⊕ Model via canonical)
    ├── canonical.py      CanonicalDict / CanonicalInfo data contract
    ├── robots/           env-coupled side  (auto-discovered by canvas)
    │   ├── base_robot.py
    │   └── libero_robot.py [+ libero_robot_absolute.py variant]
    └── models/           model-coupled side  (auto-discovered by canvas)
        ├── base_model.py
        ├── pi0_model.py · smolvla_model.py · dp_model.py
        └── dp_defaults.py    (constants — not a ModelAdaptor)

Adding a variant = drop a .py file in the right sub-folder with
`DEFAULT_KWARGS = {...}`. Removing one = delete the file. The canvas node
dropdowns scan their respective sub-folder and rebuild on hot-reload.
"""

from __future__ import annotations

from .adaptor import Adaptor
from .canonical import (
    CanonicalDict,
    CanonicalInfo,
    make_canonical_action,
    make_canonical_obs,
)
from .models import DPModel, ModelAdaptor, Pi0Model, Rt1Model, SmolVLAModel
from .robots import LiberoRobot, RobotAdaptor, SimplerRobot

__all__ = [
    "Adaptor",
    "CanonicalDict",
    "CanonicalInfo",
    "DPModel",
    "LiberoRobot",
    "ModelAdaptor",
    "Pi0Model",
    "RobotAdaptor",
    "Rt1Model",
    "SimplerRobot",
    "SmolVLAModel",
    "make_canonical_action",
    "make_canonical_obs",
]
