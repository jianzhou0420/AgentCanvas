"""RT-1-X policy — Google Robot embodiment variant.

Same ``Rt1Policy`` class as ``rt1_policy.py``; only differs in DEFAULT_KWARGS,
which selects ``policy_setup="google_robot"``. Per-embodiment branching inside
RT1Inference applies ``_small_action_filter_google_robot`` (gripper-closedness
small-action zeroing) and uses the Google Robot's axis-angle rotation
convention — the wrong branch on a Google Robot SIMPLER task will produce
ill-scaled actions.

Pick this in the Predict node's policy dropdown when running any
``google_robot_*`` SIMPLER task (drawer / move_near / pick_*). For
``widowx_*`` tasks pick ``rt1_policy`` instead (defaults to widowx_bridge).
"""

from __future__ import annotations

# Re-export the canonical class so the subclass-introspection layer finds it.
from .rt1_policy import Rt1Policy  # noqa: F401


# ───── DEFAULTS — Google Robot branch ─────
DEFAULT_KWARGS: dict = {
    "policy_setup": "google_robot",
    "action_scale": 1.0,
}
