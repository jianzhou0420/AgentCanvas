"""LIBERO robot adaptor — absolute-action variant.

Same `LiberoRobot` class as `libero_robot.py`, different default kwargs.
Empty `DEFAULT_KWARGS` falls back to `LiberoRobot()` ctor defaults
(`use_delta_actions=False`), matching `smolvla_libero.yaml` and
`droid_dp_libero.yaml` (which both omit robot kwargs upstream).

Demonstrates the **drop-a-file = add-an-adapter** pattern: this file owns
its own dropdown entry separately from `libero_robot.py`. The class itself
is shared — the discovery layer keys on file-stem, not class identity.
"""

from __future__ import annotations

# Re-export the canonical class so subclass-introspection finds it.
from .libero_robot import LiberoRobot  # noqa: F401

# ───── DEFAULTS (transcribed from smolvla_libero.yaml + droid_dp_libero.yaml) ─────
DEFAULT_KWARGS: dict = {}
