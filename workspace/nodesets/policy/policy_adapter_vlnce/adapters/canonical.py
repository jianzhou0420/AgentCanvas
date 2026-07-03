"""Canonical intermediate format for the VLN-CE adapter system.

VLN-flavored sibling of ``policy_adapter_vla/adapters/canonical.py``. The canvas
nodes between stage 1 (env→canonical) and stage 5 (canonical→env) move
opaque CanonicalDicts on ANY-typed wires; this module defines the schema
and the dataclass info carrier.

The canonical carries STANDARDIZED RAW data only — env-native content in a
uniform shape, no policy-specific processing. The instruction is the raw
natural-language string (R2R-CE) or the precomputed BERT feature (RxR-CE);
tokenization / obs_transforms / normalization are policy-specific and run in
the model-side adapter (``canonical_to_model``), never here.

Canonical Observation (stage 1 → stage 2):
    {
        "data": {
            "rgb":         np.ndarray uint8 HxWx3,        # env-native size, standardized
            "depth":       np.ndarray float HxWx1,        # in metres (raw)
            "instruction": {"kind": "raw_text", "text": str}                      # R2R-CE
                         | {"kind": "feat",     "embedding": np.ndarray (T, 768)},  # RxR-CE
            "step_meta":   {"step_index": int, "episode_id": str},
        },
        "info": CanonicalNavInfo(...),
    }

Canonical Action (stage 4 → stage 5):
    {
        "data": {"action_index": int, "action_dim": int},
        "info": CanonicalNavInfo(...),
    }
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import numpy as np

CanonicalDict = Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class CanonicalNavInfo:
    """Static metadata describing the env family the canonical originated from.

    Stage-1 env adapters fill this once per call; downstream stages use it
    to dispatch family-specific paths (e.g. vocab-vs-feat instruction
    handling in cma model adapter).
    """

    env_family: str           # "r2rce" | "rxrce"
    action_dim: int           # 4 (R2R: STOP, FORWARD, LEFT, RIGHT) | 6 (RxR: + LOOK_UP/DOWN)
    image_size: tuple[int, int]
    depth_max: float          # metres; 10.0 (R2R default) | 5.0 (RxR cap)
    instruction_kind: str     # "raw_text" | "feat"


def make_canonical_obs(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    instruction: dict[str, Any],
    step_meta: dict[str, Any] | None = None,
    info: CanonicalNavInfo,
) -> CanonicalDict:
    return {
        "data": {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
            "step_meta": step_meta or {},
        },
        # asdict() so the canonical stays msgpack-serializable across the
        # auto_host ANY-wire — a raw CanonicalNavInfo dataclass is not packable
        # and 500s the server. Downstream stages re-derive info from
        # env_adapter.get_canonical_info(), so this carrier is metadata only:
        # no consumer reads it off the wire.
        "info": dataclasses.asdict(info),
    }


def make_canonical_action(
    *,
    action_index: int,
    info: CanonicalNavInfo,
) -> CanonicalDict:
    return {
        "data": {
            "action_index": int(action_index),
            "action_dim": info.action_dim,
        },
        # asdict() for msgpack-wire safety (see make_canonical_obs).
        "info": dataclasses.asdict(info),
    }


def validate_canonical_obs(canonical: CanonicalDict) -> None:
    if "data" not in canonical or "info" not in canonical:
        raise ValueError("CanonicalDict missing 'data' or 'info' key")
    info = canonical["info"]
    # info travels as a plain dict (dataclasses.asdict of CanonicalNavInfo) so
    # it survives the msgpack wire; accept either form for robustness.
    if not isinstance(info, (CanonicalNavInfo, dict)) or (
        isinstance(info, dict) and "env_family" not in info
    ):
        raise ValueError(f"Expected CanonicalNavInfo or its asdict, got {type(info)}")
    data = canonical["data"]
    if "rgb" not in data or "depth" not in data or "instruction" not in data:
        raise ValueError("Canonical obs missing rgb/depth/instruction key")
    instr = data["instruction"]
    if not isinstance(instr, dict) or "kind" not in instr:
        raise ValueError("instruction must be {'kind': str, ...}")
    if instr["kind"] not in ("raw_text", "feat"):
        raise ValueError(f"unknown instruction kind {instr['kind']!r}")
