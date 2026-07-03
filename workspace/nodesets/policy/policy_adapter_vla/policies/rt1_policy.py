"""RT-1-X policy — stateless wrapper around the vendored TF SavedModel.

Architectural note:
    BasePolicy is ``nn.Module`` because the framework calls
    ``policy.to(device)``, ``policy.eval()``, ``policy.parameters()``,
    ``policy.device`` etc. RT-1's actual model is a TensorFlow SavedModel,
    not a torch module — there are no torch parameters to put on GPU.
    We satisfy the framework by registering a single dummy ``nn.Parameter``.

Stateless contract (mirrors Pi0Policy):
    ``predict_action`` is a pure function of its ``batch`` input — no
    instance attribute is mutated across calls. RT-1's recurrent
    ``policy_state`` (a 6-step token history) is threaded explicitly
    through the ``batch["policy_state_in"]`` input and the
    ``"policy_state"`` key of the returned dict, so the canvas can wire
    it through an IterIn/IterOut feedback edge (mirroring policy_cma).

    The USE language embedding is computed inside ``predict_action`` and
    memoised at module scope keyed by instruction string — this is a
    pure cache, not state, and is bounded to the last 64 entries.

Inference path:
    Rt1Model.canonical_to_model produces ``{image: HWC uint8, instruction: str}``.
    VlaPolicyManager.predict pre-processes this via ``_batch_to_torch`` —
    the image becomes ``torch.uint8 (1, H, W, 3)`` on the policy's device,
    the instruction string passes through. ``predict_action`` undoes the
    torch wrap (cpu().numpy(), drop B dim), runs one TF eager step, and
    packs the 7-D action into a ``(1, 1, 7)`` torch tensor;
    ``VlaPolicyManager.predict`` then squeezes the leading 1.

Usage from canvas:
    Pick ``rt1_model`` in the Canonical→Model node's model dropdown,
    ``rt1_policy`` in the Predict node's policy dropdown, and put the path
    to the TF SavedModel directory (containing ``saved_model.pb``) in the
    Predict node's checkpoint_path. Wire ``policy_state_in/out`` through
    IterIn/IterOut for feedback (see workspace/graphs/vla_policy_simpler.json).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from workspace.nodesets.policy.policy_adapter_vla.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


# ── Module-level USE embedding memo (pure cache, not state) ────────────────
# Keyed by raw instruction string. Bounded to the last 64 entries via
# OrderedDict eviction; SimplerEnv only uses a handful of unique strings,
# so this is just defence against pathological multi-task evaluations.
_USE_EMBEDDING_CACHE_MAX = 64
_USE_EMBEDDING_CACHE: OrderedDict[str, Any] = OrderedDict()


def _coerce_image_for_rt1(image_t: Any) -> np.ndarray:
    """Strip torch wrapping and return (H, W, 3) uint8 numpy."""
    if hasattr(image_t, "detach"):
        arr = image_t.detach().to("cpu").numpy()
    else:
        arr = np.asarray(image_t)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            max_val = float(arr.max()) if arr.size > 0 else 0.0
            arr = (np.clip(arr, 0.0, 1.0) * 255.0) if max_val <= 1.0 else arr
            arr = arr.round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Rt1Policy.predict_action: image shape {arr.shape} not (H,W,3)")
    return arr


_EMBODIMENT_PARAMS: dict[str, dict[str, Any]] = {
    "google_robot": {
        "unnormalize_action": False,
        "unnormalize_action_fxn": None,
        "invert_gripper_action": False,
        "action_rotation_mode": "axis_angle",
    },
    "widowx_bridge": {
        # `unnormalize_action_fxn` filled in below — defined later in module.
        "unnormalize_action": True,
        "unnormalize_action_fxn": None,
        "invert_gripper_action": True,
        "action_rotation_mode": "rpy",
    },
}


def _detect_embodiment(image_hw: tuple[int, int], fallback: str) -> str:
    """Pick the SIMPLER embodiment from the front-camera (H, W).

    SimplerEnv exposes two embodiments with non-overlapping image shapes:
    WidowX bridge → (480, 640), Google Robot → (512, 640). We branch on the
    height alone since width matches. Anything else falls through to the
    constructor's ``policy_setup`` so non-SIMPLER use of this policy keeps
    the explicit setting authoritative.
    """
    h = image_hw[0]
    if h == 480:
        return "widowx_bridge"
    if h == 512:
        return "google_robot"
    return fallback


def _coerce_instruction(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _resize_image_tf(image: np.ndarray, target_w: int, target_h: int) -> Any:
    """Resize via tf.image.resize_with_pad → uint8 tf.Tensor."""
    import tensorflow as tf

    out = tf.image.resize_with_pad(image, target_width=target_w, target_height=target_h)
    out = tf.cast(out, tf.uint8)
    return out


def _rescale_action_with_bound(
    actions: Any,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    resc = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
    return np.clip(resc, post_scaling_min + safety_margin, post_scaling_max - safety_margin)


def _unnormalize_action_widowx_bridge(action: dict[str, Any]) -> dict[str, Any]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-1.75,
        high=1.75,
        post_scaling_max=0.05,
        post_scaling_min=-0.05,
    )
    action["rotation_delta"] = _rescale_action_with_bound(
        action["rotation_delta"],
        low=-1.4,
        high=1.4,
        post_scaling_max=0.25,
        post_scaling_min=-0.25,
    )
    return action


# Late-bind the widowx unnormalizer reference now that the function exists.
_EMBODIMENT_PARAMS["widowx_bridge"]["unnormalize_action_fxn"] = _unnormalize_action_widowx_bridge


def _small_action_filter_google_robot(
    raw_action: dict[str, Any],
    arm_movement: bool = False,
    gripper: bool = True,
) -> dict[str, Any]:
    import tensorflow as tf

    if arm_movement:
        raw_action["world_vector"] = tf.where(
            tf.abs(raw_action["world_vector"]) < 5e-3,
            tf.zeros_like(raw_action["world_vector"]),
            raw_action["world_vector"],
        )
        raw_action["rotation_delta"] = tf.where(
            tf.abs(raw_action["rotation_delta"]) < 5e-3,
            tf.zeros_like(raw_action["rotation_delta"]),
            raw_action["rotation_delta"],
        )
        raw_action["base_displacement_vector"] = tf.where(
            raw_action["base_displacement_vector"] < 5e-3,
            tf.zeros_like(raw_action["base_displacement_vector"]),
            raw_action["base_displacement_vector"],
        )
        raw_action["base_displacement_vertical_rotation"] = tf.where(
            raw_action["base_displacement_vertical_rotation"] < 1e-2,
            tf.zeros_like(raw_action["base_displacement_vertical_rotation"]),
            raw_action["base_displacement_vertical_rotation"],
        )
    if gripper:
        raw_action["gripper_closedness_action"] = tf.where(
            tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
            tf.zeros_like(raw_action["gripper_closedness_action"]),
            raw_action["gripper_closedness_action"],
        )
    return raw_action


def _policy_state_to_numpy(state: Any) -> Any:
    """Recursively convert a TF nested policy_state to pure-numpy structures.

    The HTTP boundary between auto_host workers and the shared policy
    subprocess serialises this dict; TF EagerTensors are not JSON-friendly,
    so we strip them down to numpy here. The inverse is
    :func:`_policy_state_to_tf` on the way back in.
    """
    import tensorflow as tf

    if isinstance(state, dict):
        return {k: _policy_state_to_numpy(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        seq = [_policy_state_to_numpy(v) for v in state]
        return type(state)(seq) if isinstance(state, tuple) else seq
    if isinstance(state, tf.Tensor):
        return state.numpy()
    return state


def _policy_state_to_tf(state: Any) -> Any:
    """Inverse of :func:`_policy_state_to_numpy` — numpy → tf.constant tree."""
    import tensorflow as tf

    if isinstance(state, dict):
        return {k: _policy_state_to_tf(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        seq = [_policy_state_to_tf(v) for v in state]
        return type(state)(seq) if isinstance(state, tuple) else seq
    if isinstance(state, np.ndarray):
        return tf.constant(state)
    return state


class Rt1Policy(BasePolicy):
    """Stateless inference wrapper around the RT-1-X TF SavedModel.

    Constructor kwargs (all in ``DEFAULT_KWARGS``):
        policy_setup:  "widowx_bridge" | "google_robot" — per-embodiment branch
        action_scale:  float — multiplier on world_vector + rot_axangle
        image_width / image_height: TF resize target before policy.action
        lang_embed_model_path: TF Hub path for Universal Sentence Encoder
    """

    def __init__(
        self,
        *,
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        image_width: int = 320,
        image_height: int = 256,
        lang_embed_model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        **_unused: Any,
    ) -> None:
        super().__init__()
        # Dummy parameter — RT-1 has no torch params, but BasePolicy.device /
        # policy.to() / sum(p.numel() for p in policy.parameters()) all expect
        # at least one. Frozen (requires_grad=False) so no gradient tracking.
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._policy_setup = str(policy_setup)
        self._action_scale = float(action_scale)
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._lang_embed_model_path = str(lang_embed_model_path)

        # Filled by load_checkpoint — set-once, never mutated.
        self._tfa_policy: Any = None
        self._lang_embed_model: Any = None
        self._loaded_path: str = ""

        if self._policy_setup == "google_robot":
            self._unnormalize_action = False
            self._unnormalize_action_fxn = None
            self._invert_gripper_action = False
            self._action_rotation_mode = "axis_angle"
        elif self._policy_setup == "widowx_bridge":
            self._unnormalize_action = True
            self._unnormalize_action_fxn = _unnormalize_action_widowx_bridge
            self._invert_gripper_action = True
            self._action_rotation_mode = "rpy"
        else:
            raise NotImplementedError(
                f"Rt1Policy.policy_setup={self._policy_setup!r} not in "
                "('widowx_bridge', 'google_robot')"
            )

    def load_checkpoint(self, path: str) -> None:
        import tensorflow_hub as hub
        from tf_agents.policies import py_tf_eager_policy

        # Allow either pointing at the SavedModel dir directly or one level up.
        if os.path.isfile(os.path.join(path, "saved_model.pb")):
            saved = path
        else:
            candidates = (
                [
                    os.path.join(path, n)
                    for n in os.listdir(path)
                    if os.path.isfile(os.path.join(path, n, "saved_model.pb"))
                ]
                if os.path.isdir(path)
                else []
            )
            if not candidates:
                raise FileNotFoundError(
                    f"Rt1Policy.load_checkpoint: no saved_model.pb under {path!r} "
                    "or its immediate children."
                )
            saved = candidates[0]
        logger.info(
            "Loading RT-1-X SavedModel from %s (policy_setup=%s)", saved, self._policy_setup
        )
        self._tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved,
            load_specs_from_pbtxt=True,
            use_tf_function=True,
        )
        self._lang_embed_model = hub.load(self._lang_embed_model_path)
        self._loaded_path = saved

    def get_initial_policy_state(self, batch_size: int = 1) -> Any:
        """Produce the per-episode initial recurrent state (numpy-nested)."""
        if self._tfa_policy is None:
            raise RuntimeError("Rt1Policy.get_initial_policy_state: checkpoint not loaded.")
        tf_state = self._tfa_policy.get_initial_state(batch_size=batch_size)
        return _policy_state_to_numpy(tf_state)

    def _embed_instruction(self, instruction: str) -> Any:
        """Memoised USE embedding for a single instruction string."""
        if instruction in _USE_EMBEDDING_CACHE:
            _USE_EMBEDDING_CACHE.move_to_end(instruction)
            return _USE_EMBEDDING_CACHE[instruction]
        if not instruction:
            import tensorflow as tf

            emb = tf.zeros((512,), dtype=tf.float32)
        else:
            emb = self._lang_embed_model([instruction])[0]
        _USE_EMBEDDING_CACHE[instruction] = emb
        if len(_USE_EMBEDDING_CACHE) > _USE_EMBEDDING_CACHE_MAX:
            _USE_EMBEDDING_CACHE.popitem(last=False)
        return emb

    def predict_action(self, batch: dict) -> dict[str, Any]:
        """Pure function: ``(image, instruction, policy_state_in) → {action, policy_state}``.

        No ``self`` mutation. ``policy_state_in`` is the previous-step
        recurrent state (numpy-nested dict) or ``None`` on the first
        step (we lazily call ``get_initial_state(1)``).
        """
        if self._tfa_policy is None:
            raise RuntimeError(
                "Rt1Policy.predict_action: checkpoint not loaded — set "
                "checkpoint_path on the Predict node."
            )

        import tf_agents
        from tf_agents.trajectories import time_step as ts

        image = _coerce_image_for_rt1(batch.get("image"))
        instruction = _coerce_instruction(batch.get("instruction"))
        policy_state_in = batch.get("policy_state_in")

        # Auto-detect embodiment from image shape so policy-side post-processing
        # (action unnormalization, gripper sign, rotation encoding) matches the
        # actual SIMPLER embodiment regardless of how the policy was constructed.
        # Fixes the silent bug where the constructor default ("widowx_bridge")
        # was applied to google_robot tasks too — gripper got binarised, action
        # rescaled wrong, rotation decoded as RPY instead of axis-angle, and
        # every pick task scored 0%.
        embodiment = _detect_embodiment(
            (int(image.shape[0]), int(image.shape[1])), self._policy_setup
        )
        params = _EMBODIMENT_PARAMS[embodiment]

        # Lazy initial state when iter 0 (no in-wire value yet).
        if policy_state_in is None:
            policy_state_tf = self._tfa_policy.get_initial_state(batch_size=1)
        else:
            policy_state_tf = _policy_state_to_tf(policy_state_in)

        embedding = self._embed_instruction(instruction)

        # Build a fresh observation dict + time_step locally — no scratch buffer.
        observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self._tfa_policy.time_step_spec.observation)
        )
        observation["image"] = _resize_image_tf(image, self._image_width, self._image_height)
        observation["natural_language_embedding"] = embedding
        tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

        policy_step = self._tfa_policy.action(tfa_time_step, policy_state_tf)
        raw_action = dict(policy_step.action)
        if embodiment == "google_robot":
            raw_action = _small_action_filter_google_robot(
                raw_action,
                arm_movement=False,
                gripper=True,
            )
        if params["unnormalize_action"]:
            raw_action = params["unnormalize_action_fxn"](
                {k: np.asarray(v) for k, v in raw_action.items()}
            )
        for k in list(raw_action.keys()):
            raw_action[k] = np.asarray(raw_action[k])

        # Pack into the SimplerEnv-style 7-D chunk (xyz, axangle, gripper).
        rotation_mode = params["action_rotation_mode"]
        wv = np.asarray(raw_action["world_vector"], dtype=np.float64) * self._action_scale
        if rotation_mode == "axis_angle":
            rd = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            angle = float(np.linalg.norm(rd))
            axis = rd / angle if angle > 1e-6 else np.array([0.0, 1.0, 0.0])
            ra = axis * angle * self._action_scale
        elif rotation_mode in ("rpy", "ypr", "pry"):
            from transforms3d.euler import euler2axangle

            if rotation_mode == "rpy":
                roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            elif rotation_mode == "ypr":
                yaw, pitch, roll = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            else:  # "pry"
                pitch, roll, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            axis, angle = euler2axangle(roll, pitch, yaw)
            ra = axis * angle * self._action_scale
        else:
            raise NotImplementedError(rotation_mode)

        gp_raw = raw_action["gripper_closedness_action"]
        if params["invert_gripper_action"]:
            gp_raw = -gp_raw
        if embodiment == "google_robot":
            gp = np.asarray(gp_raw, dtype=np.float64)
        else:  # widowx_bridge: binarise to ±1
            gp = np.asarray(gp_raw, dtype=np.float64)
            gp = 2.0 * (gp > 0.0) - 1.0
        gp = np.atleast_1d(gp)
        if gp.size == 0:
            gp = np.zeros(1, dtype=np.float64)

        chunk = np.concatenate(
            [wv.reshape(-1)[:3], np.asarray(ra, dtype=np.float64).reshape(-1)[:3], gp[:1]],
            axis=0,
        ).astype(np.float32)  # (7,)
        action_tensor = torch.from_numpy(chunk[None, None, :]).to(self.device)

        return {
            "action": action_tensor,
            "policy_state": _policy_state_to_numpy(policy_step.state),
        }

    def compute_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError(
            "Rt1Policy is inference-only — RT-1-X is a frozen TF SavedModel, "
            "training is not supported through this wrapper."
        )


# ───── DEFAULTS — pick policy_setup to match the active SIMPLER split ─────
# widowx_bridge: WidowX, third-person 3rd_view_camera (640×480), gripper -1=close/+1=open
# google_robot:  Google Robot, overhead_camera (640×512), sticky-gripper inversion
DEFAULT_KWARGS: dict = {
    "policy_setup": "widowx_bridge",
    "action_scale": 1.0,
}
