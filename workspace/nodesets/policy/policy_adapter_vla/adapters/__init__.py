"""Vendored adapter system for VLA policies — internal helper for policy_adapter_vla.

Sources (verbatim copies with rewritten imports, per AgentCanvas
no-upstream-import policy):
    vlaworkspace/src/vlaworkspace/adaptors/

Model side of upstream's two-axis split. The env-coupled side — robots/ —
now lives in the general env_adapter nodeset
(``workspace/nodesets/env/env_adapter/robots/``); upstream's composed
``Adaptor`` dissolved with that split (the manager calls the model adapter
and policy directly):

    adapters/
    ├── canonical.py      CanonicalDict / CanonicalInfo data contract
    │                     (duplicated in env_adapter/robots/ — plain dicts
    │                     on the wire keep the two copies runtime-independent)
    └── models/           model-coupled side  (auto-discovered by canvas)
        ├── base_model.py
        ├── pi0_model.py · smolvla_model.py · dp_model.py · rt1_model.py
        └── dp_defaults.py    (constants — not a ModelAdaptor)

Adding a variant = drop a .py file in the right sub-folder with
`DEFAULT_KWARGS = {...}`. Removing one = delete the file. The canvas node
dropdowns scan their respective sub-folder and rebuild on hot-reload.
"""

from __future__ import annotations

from .canonical import (
    CanonicalDict,
    CanonicalInfo,
    make_canonical_action,
    make_canonical_obs,
)
from .models import DPModel, ModelAdaptor, Pi0Model, Rt1Model, SmolVLAModel

__all__ = [
    "CanonicalDict",
    "CanonicalInfo",
    "DPModel",
    "ModelAdaptor",
    "Pi0Model",
    "Rt1Model",
    "SmolVLAModel",
    "make_canonical_action",
    "make_canonical_obs",
]
