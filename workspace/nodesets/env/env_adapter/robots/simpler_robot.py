"""SIMPLER (SimplerEnv) robot adaptor.

Covers both embodiments exposed by the env_simpler nodeset:
    bridge        — WidowX, third-person 3rd_view_camera (640x480)
    google_robot  — Google Robot, overhead_camera (640x512)

Wire contract — matches what env_simpler emits and what env_simpler__step accepts
(see workspace/nodesets/env/env_simpler/__init__.py docstring):

    obs.image          H x W x 3 uint8 — third-person view (per-embodiment camera)
    obs.wrist_image    None             — SIMPLER has no wrist cam
    obs.state          (D,) float32     — flat proprio from obs['agent']
                                          (qpos + qvel + base_pose for WidowX)
    action             (K, 7) float32   — [delta_pos(3), delta_axis_angle(3),
                                          gripper(1)] per step
                                          gripper: -1=close, +1=open

Design choices vs LiberoRobot:

1.  No wrist camera. canonical.images.wrist = None.

2.  State is **joint_position**, not (pos, rot, gripper). SIMPLER's `obs['agent']`
    flat proprio is qpos+qvel+base_pose, *not* EE pose. We don't fabricate an
    EE pose from joint angles here — the honest representation is to pass the
    raw flat vector through canonical's `state.joint_position` slot.

3.  use_delta_actions=False by default. SIMPLER's env_simpler__step takes raw
    deltas in env space already, so canonical_to_env is a straight stack.
    LiberoRobot's "delta = action - current_state" trick presumes an EE-pose
    state that we don't have.

CAVEAT — Pi0 LIBERO finetune ≠ SIMPLER inference. A Pi0 checkpoint trained
against LIBERO's 8-D EE state will not produce meaningful actions when fed
SIMPLER's joint-space proprio. This adapter wires the pipeline end-to-end so
graphs run; getting good numbers needs a SIMPLER-trained (or co-trained)
checkpoint. Pure-vision policies (some DP variants) that ignore state are
the closest things to "drop-in" today.
"""

from __future__ import annotations

import logging

import numpy as np

from .canonical import CanonicalDict, CanonicalInfo, make_canonical_obs
from .base_robot import RobotAdaptor

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _parse_image_to_chw_float(image) -> np.ndarray:
    """HWC uint8 [0,255] → CHW float32 [0,1]. Pass-through for already-CHW/float inputs."""
    image = np.asarray(_to_numpy(image), dtype=None)

    if image.dtype == np.uint8:
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32) / 255.0

    image = image.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
        image = np.transpose(image, (2, 0, 1))

    if image.max() > 1.0:
        image = image / 255.0

    return image


# WidowX flat proprio = qpos(8) + qvel(8) + base_pose(7).
# Google Robot is a different arm + base; concrete dim is verified at runtime
# from the first env_to_canonical call (state.shape[-1]).
_DEFAULT_STATE_DIM = 23


class SimplerRobot(RobotAdaptor):
    """Robot adaptor for SimplerEnv (WidowX / Google Robot).

    State layout: flat float32 of length D (joint-space, embodiment-dependent).
    Action layout: [delta_pos(3), delta_axis_angle(3), gripper(1)] = 7D.
    """

    ENV_KEY_REMAP = {
        "observation/image": "image",
        "observation/state": "state",
    }

    def __init__(
        self,
        *,
        tasks: dict[int, str] | None = None,
        default_prompt: str | None = None,
        state_dim: int = _DEFAULT_STATE_DIM,
        use_delta_actions: bool = False,
    ) -> None:
        self.tasks = tasks
        self.default_prompt = default_prompt
        self.state_dim = int(state_dim)
        self.use_delta_actions = use_delta_actions  # kept for API symmetry; not used.

    def get_canonical_info(self) -> CanonicalInfo:
        return CanonicalInfo(
            state_type={"joint_position": "absolute"},
            state_rot_repr="none",
            action_type={"pos": "delta", "rot": "delta", "gripper": "absolute"},
            action_rot_repr="axis_angle",
            state_dims={"joint_position": self.state_dim},
            action_dims={"pos": 3, "rot": 3, "gripper": 1},
        )

    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        # No SIMPLER training corpus is plumbed through AgentCanvas today —
        # this method exists for RobotAdaptor contract symmetry. If/when a
        # SIMPLER-format LeRobot dataset shows up, fill this in from the
        # observed keys (likely `observation.image`, `observation.state`,
        # `actions`, `task_index`).
        raise NotImplementedError(
            "SimplerRobot.dataset_to_canonical: no SIMPLER training data "
            "ingestion path implemented yet — adapter is inference-only."
        )

    def env_to_canonical(self, data: dict) -> CanonicalDict:
        remapped = {}
        for key, value in data.items():
            new_key = self.ENV_KEY_REMAP.get(key, key)
            remapped[new_key] = value

        front_img = None
        if remapped.get("image") is not None:
            front_img = _parse_image_to_chw_float(remapped["image"])

        # SIMPLER has no wrist camera — env_simpler emits None on this port.
        # Drop it on the floor; canonical.images.wrist stays None.

        state_raw = remapped.get("state")
        if state_raw is None:
            joint_pos = None
        else:
            arr = np.asarray(_to_numpy(state_raw), dtype=np.float32).reshape(-1)
            # Adopt observed dim if it differs from configured default — first
            # call wins, so subsequent shape changes (would be embodiment swap
            # mid-episode) are caught by the model adapter's norm-stats step.
            if arr.shape[-1] != self.state_dim:
                logger.debug(
                    "SimplerRobot: state_dim %d → %d (from env)",
                    self.state_dim,
                    arr.shape[-1],
                )
                self.state_dim = int(arr.shape[-1])
            joint_pos = arr

        prompt = remapped.get("prompt", self.default_prompt or "")

        return make_canonical_obs(
            images={"front": front_img, "wrist": None},
            state={"joint_position": joint_pos},
            actions={},
            prompt=prompt,
            info=self.get_canonical_info(),
        )

    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        actions_data = canonical_action["data"]["actions"]

        action_pos = actions_data.get("pos")
        action_rot = actions_data.get("rot")
        action_gripper = actions_data.get("gripper")

        if action_pos is None or action_rot is None or action_gripper is None:
            raise ValueError(
                "SimplerRobot.canonical_to_env: model output missing one of "
                f"pos/rot/gripper (got keys {list(actions_data.keys())}). "
                "Models that emit joint-space actions need a different env "
                "step contract — out of scope for this adapter."
            )

        actions = np.concatenate(
            [action_pos, action_rot, action_gripper],
            axis=-1,
        ).astype(np.float32)

        # No add-current-state step: SIMPLER takes deltas natively, and we
        # don't have an EE-pose state to add anyway (state is joint-space).
        return {"actions": actions}

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return 7

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return (
            "state/joint_position",
            "actions/pos",
            "actions/rot",
            "actions/gripper",
        )

    def env_obs(self) -> dict:
        return {
            "image": "[H, W, 3] uint8 [0, 255] — third-person view",
            "wrist_image": "None — SIMPLER has no wrist cam",
            "state": f"[{self.state_dim}] float32 — flat proprio (qpos+qvel+base_pose)",
            "prompt": "str",
        }

    def env_action(self) -> dict:
        return {"actions": "[horizon, 7] float32 — [dpos(3), daxis_angle(3), gripper(1)]"}

    def datasets(self) -> dict:
        # No upstream LeRobot SIMPLER corpus is wired in — kept for contract
        # symmetry with LiberoRobot. See dataset_to_canonical().
        return {}


# ───── DEFAULTS — SIMPLER inference-only baseline ─────
# state_dim defaults to WidowX (23-D); env_to_canonical auto-corrects from the
# observed shape on first call so Google Robot works without changing this.
DEFAULT_KWARGS: dict = {}
