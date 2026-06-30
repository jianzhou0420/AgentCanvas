"""RT-1-X model adapter — pass-through (no normalization, no tokenization).

RT-1-X's TF SavedModel does its own image resize / language embedding
internally; the adapter's job is just to translate canonical ↔ what
``Rt1Policy.predict_action`` expects, and back.

Data flow:
    canonical (CHW float [0,1] image, str prompt)
        ──► canonical_to_model    ──► {image: HWC uint8, instruction: str}
                                        ↓ predict_action
                                      RT1Inference.step internally:
                                        resize → USE-embed → tf-policy.action
                                        ↓
                                      → {action: (1, 7) float32}
    {action: (1, 7) float32}
        ──► model_to_canonical    ──► canonical_action
                                      {pos: (1,3), rot: (1,3), gripper: (1,1)}

Why no norm_stats: RT-1's per-embodiment rescaling is baked into the
TF graph + the wrapper's ``unnormalize_action_widowx_bridge`` /
``_small_action_filter_google_robot`` paths. Norm stats here would
re-normalize an already-rescaled action. ``norm_stats_mode="none"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..canonical import CanonicalDict, CanonicalInfo, make_canonical_action
from .base_model import ModelAdaptor

logger = logging.getLogger(__name__)


def _canonical_image_to_uint8_hwc(image: Any) -> np.ndarray:
    """Convert canonical's CHW float32 [0,1] image back to HWC uint8.

    SimplerRobot.env_to_canonical produces CHW float32 [0,1]; RT1Inference's
    step() expects HWC uint8 [0,255]. This is the single conversion point.
    """
    if image is None:
        raise ValueError("Rt1Model.canonical_to_model: front image is None")
    arr = np.asarray(image)
    if arr.dtype.kind == "f":
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        max_val = float(arr.max()) if arr.size > 0 else 0.0
        if max_val <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        return arr.round().astype(np.uint8)
    arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


class Rt1Model(ModelAdaptor):
    """Pass-through adapter for RT-1-X.

    Has no learnable parameters and no norm-stats. Constructor takes no kwargs
    beyond the BaseAdaptor's signature so ``DEFAULT_KWARGS = {}`` is sufficient.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Accept (and ignore) norm_stats_path / norm_stats — the manager passes
        # norm_stats_path through ensure_model_adaptor; we don't use it.
        norm_stats_path = kwargs.pop("norm_stats_path", None)
        if norm_stats_path:
            logger.info("Rt1Model: ignoring norm_stats_path=%r (RT-1 has no norm-stats)",
                        norm_stats_path)
        super().__init__()
        if kwargs:
            logger.debug("Rt1Model: unused kwargs %s", list(kwargs))

    # ── canonical → model ──

    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        data = canonical["data"]
        front = data.get("images", {}).get("front")
        image = _canonical_image_to_uint8_hwc(front)
        prompt = data.get("prompt", "") or ""
        return {"image": image, "instruction": str(prompt)}

    # ── model → canonical ──

    def model_to_canonical(self, model_output: dict, info: CanonicalInfo) -> CanonicalDict:
        action = model_output.get("action")
        if action is None:
            raise ValueError("Rt1Model.model_to_canonical: model_output missing 'action' key")
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 7:
            chunk = arr[None, :]
        elif arr.ndim == 2 and arr.shape[-1] == 7:
            chunk = arr
        else:
            raise ValueError(
                f"Rt1Model.model_to_canonical: action shape {arr.shape} not (7,) or (K,7)"
            )
        return make_canonical_action(
            actions={
                "pos":     chunk[:, 0:3],
                "rot":     chunk[:, 3:6],
                "gripper": chunk[:, 6:7],
            },
            info=info,
        )

    # ── norm-stats interface (no-op for RT-1) ──

    def get_norm_stats_mode(self) -> str:
        return "none"

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return ()

    def canonical_to_norm_stats_format(self, canonical: CanonicalDict) -> dict:
        return {}

    # ── descriptors (used by canvas tooltips / inspectors) ──

    def model_input(self) -> dict:
        return {
            "image":       "(H, W, 3) uint8 — passed straight to Rt1Policy.predict_action",
            "instruction": "str — task description (USE-embedded inside the model)",
        }

    def model_output(self) -> dict:
        return {
            "action": "(K=1, 7) float32 — [dpos(3), daxis_angle(3), gripper(1)]",
        }


# ───── DEFAULTS — RT-1-X is configuration-free at the adapter layer ─────
# All per-embodiment behaviour is selected on the policy side via
# Rt1Policy(policy_setup=...). The model adapter has nothing to tune.
DEFAULT_KWARGS: dict = {}
