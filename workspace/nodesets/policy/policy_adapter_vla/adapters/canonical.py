"""Canonical intermediate format for the adaptor system.

Vendored verbatim from vlaworkspace/adaptors/canonical.py — per AgentCanvas
policy, no upstream import.

Defines the standard data format that bridges RobotAdaptors and ModelAdaptors.
Every robot environment maps to this format, and every model maps from it.

Canonical Observation (full, for input):
    {
        "data": {
            "images": {"front": [n_obs, C, H, W] float32 | None, "wrist": ... | None},
            "state": {"pos": [n_obs, 3] | None, "rot": [n_obs, D] | None,
                      "gripper": [n_obs, 1-2] | None, "joint_position": [n_obs, N] | None},
            "actions": {"pos": [horizon, 3] | None, "rot": [horizon, D] | None,
                        "gripper": [horizon, 1-2] | None, "joint_position": [horizon, N] | None},
            "prompt": str,
        },
        "info": CanonicalInfo(...),
    }

Canonical Action (output only):
    {
        "data": {
            "actions": {"pos": ..., "rot": ..., "gripper": ..., ...},
        },
        "info": CanonicalInfo(...),
    }
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CanonicalDict = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class CanonicalInfo:
    """Metadata about canonical data representations."""

    state_type: dict[str, str]
    state_rot_repr: str
    action_type: dict[str, str]
    action_rot_repr: str
    state_dims: dict[str, int] = dataclasses.field(default_factory=dict)
    action_dims: dict[str, int] = dataclasses.field(default_factory=dict)


def make_canonical_obs(
    *,
    images: dict[str, np.ndarray | None] | None = None,
    state: dict[str, np.ndarray | None] | None = None,
    actions: dict[str, np.ndarray | None] | None = None,
    prompt: str = "",
    info: CanonicalInfo,
) -> CanonicalDict:
    return {
        "data": {
            "images": images or {},
            "state": state or {},
            "actions": actions or {},
            "prompt": prompt,
        },
        "info": info,
    }


def make_canonical_action(
    *,
    actions: dict[str, np.ndarray | None],
    info: CanonicalInfo,
) -> CanonicalDict:
    return {
        "data": {
            "actions": actions,
        },
        "info": info,
    }


def validate_canonical(canonical: CanonicalDict, *, require_actions: bool = False) -> None:
    if "data" not in canonical:
        raise ValueError("Canonical dict missing 'data' key")
    if "info" not in canonical:
        raise ValueError("Canonical dict missing 'info' key")

    data = canonical["data"]
    info = canonical["info"]

    if not isinstance(info, CanonicalInfo):
        raise ValueError(f"Expected CanonicalInfo, got {type(info)}")

    if "images" in data:
        for name, img in data["images"].items():
            if img is not None:
                if not isinstance(img, np.ndarray):
                    raise ValueError(f"Image '{name}' should be ndarray, got {type(img)}")
                if img.dtype != np.float32:
                    raise ValueError(f"Image '{name}' should be float32, got {img.dtype}")

    if "state" in data:
        for name, s in data["state"].items():
            if s is not None and not isinstance(s, np.ndarray):
                raise ValueError(f"State '{name}' should be ndarray, got {type(s)}")

    if require_actions and "actions" in data:
        actions = data["actions"]
        if not actions:
            raise ValueError("Actions required but empty")
        for name, a in actions.items():
            if a is not None and not isinstance(a, np.ndarray):
                raise ValueError(f"Action '{name}' should be ndarray, got {type(a)}")
