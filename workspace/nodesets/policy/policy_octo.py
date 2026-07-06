"""Octo policy NodeSet — SimplerEnv-native VLA baseline.

Single-node nodeset wrapping ``simpler_env.policies.octo.octo_model.OctoInference``
(vendored under ``third_party/SimplerEnv``). Octo is a JAX/Flax model so this
runs in its own conda env (``ac-octo``, JAX + Flax + ``octo``) — kept
separate from the TF-based RT-1 env.

Why bypass the canonical adapter pipeline:
    The vendored ``OctoInference`` already handles
      - image resize (lanczos3 → 256×256) + dtype,
      - language tokenization + T5 conditioning task creation,
      - per-embodiment action ensembling (Bridge vs Google Robot),
      - sticky-gripper logic for Google Robot,
      - terminate-episode signal,
    and emits an action dict ``{world_vector, rot_axangle, gripper, terminate_episode}``
    in the env_simpler step contract directly.

Wire shape:
    inputs:
        image:        IMAGE  (H, W, 3) uint8 — third-person view
        instruction:  TEXT   — natural-language task description
    config:
        checkpoint_path: str  default "data/vla_policy/checkpoints/octo-small-1.5"
                              empty = ``hf://rail-berkeley/octo-small`` (downloads on first use)
        model_type:      "octo-small" | "octo-base"
        policy_setup:    "widowx_bridge" | "google_robot"
        horizon:         int   default 2  (Octo's image history length)
        pred_action_horizon: int default 4
        exec_horizon:    int   default 1
        image_size:      int   default 256
        action_scale:    float default 1.0
        init_rng:        int   default 0
    outputs:
        action_chunk: TEXT  JSON ``[[dx, dy, dz, drx, dry, drz, grip]]`` (K=1)
        terminate:    BOOL  Octo's terminate_episode > 0.5

Checkpoint layout (HF snapshot):
    data/vla_policy/checkpoints/octo-small-1.5/
        ├── config.json
        ├── dataset_statistics.json
        ├── *.msgpack
        └── ...

Set ``$VLA_CHECKPOINTS_ROOT`` to override the prefix.

last updated: 2026-05-04
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef, conda_env_python


log = logging.getLogger("agentcanvas.policy-octo")


# ── Class-level model singleton (one per subprocess) ──
_MODEL_BUNDLE: dict | None = None
_MODEL_LOAD_LOCK = threading.Lock()


_MODEL_TYPE_TO_DIR = {
    "octo-small": "octo-small-1.5",
    "octo-base":  "octo-base-1.5",
}


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _resolve_default_checkpoint(model_type: str) -> str:
    root = os.environ.get(
        "VLA_CHECKPOINTS_ROOT",
        os.path.join(_repo_root(), "data", "vla_policy", "checkpoints"),
    )
    return os.path.join(root, _MODEL_TYPE_TO_DIR.get(model_type, model_type))


def _ensure_loaded(
    checkpoint_path: str,
    model_type: str,
    policy_setup: str,
    horizon: int,
    pred_action_horizon: int,
    exec_horizon: int,
    image_size: int,
    action_scale: float,
    init_rng: int,
) -> dict:
    """Lock-protected lazy load. Returns the model bundle dict."""
    global _MODEL_BUNDLE
    cache_key = (checkpoint_path, model_type, policy_setup, horizon, pred_action_horizon,
                 exec_horizon, image_size, action_scale, init_rng)
    if _MODEL_BUNDLE is not None and _MODEL_BUNDLE.get("cache_key") == cache_key:
        return _MODEL_BUNDLE
    with _MODEL_LOAD_LOCK:
        if _MODEL_BUNDLE is not None and _MODEL_BUNDLE.get("cache_key") == cache_key:
            return _MODEL_BUNDLE

        from octo.model.octo_model import OctoModel
        from simpler_env.policies.octo.octo_model import OctoInference

        if checkpoint_path and os.path.exists(checkpoint_path):
            log.info("Loading Octo from local %s", checkpoint_path)
            base_model = OctoModel.load_pretrained(checkpoint_path)
            inference = OctoInference(
                model=base_model,
                model_type=model_type,
                policy_setup=policy_setup,
                horizon=horizon,
                pred_action_horizon=pred_action_horizon,
                exec_horizon=exec_horizon,
                image_size=image_size,
                action_scale=action_scale,
                init_rng=init_rng,
            )
        else:
            if checkpoint_path:
                log.warning("checkpoint_path %r missing — falling back to hf://rail-berkeley/%s",
                            checkpoint_path, model_type)
            log.info("Loading Octo via HF: hf://rail-berkeley/%s", model_type)
            inference = OctoInference(
                model_type=model_type,
                policy_setup=policy_setup,
                horizon=horizon,
                pred_action_horizon=pred_action_horizon,
                exec_horizon=exec_horizon,
                image_size=image_size,
                action_scale=action_scale,
                init_rng=init_rng,
            )

        _MODEL_BUNDLE = {
            "model": inference,
            "cache_key": cache_key,
            "current_instruction": None,
        }
        log.info("Octo (%s, %s) loaded.", model_type, policy_setup)
        return _MODEL_BUNDLE


def _coerce_image(image: Any) -> np.ndarray:
    """Pull an HWC uint8 array out of whatever the wire delivered."""
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image expected (H,W,3) uint8; got shape={arr.shape} dtype={arr.dtype}")
    return arr


class OctoPredictTool(BaseCanvasNode):
    """Octo forward pass — image + instruction → 7-D action chunk."""

    node_type = "policy_octo__predict"
    display_name = "Octo: Predict"
    description = (
        "Octo (RAIL Berkeley) forward pass. Takes a third-person RGB image "
        "and a language instruction; emits a 1-step action chunk matching "
        "env_simpler__step's [dpos(3), daxis_angle(3), gripper(1)] contract. "
        "Per-embodiment behaviour selected via policy_setup config."
    )
    category = "policy"
    icon = "Cpu"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    input_ports = [
        PortDef("image", "IMAGE", "Third-person view, (H,W,3) uint8"),
        PortDef("instruction", "TEXT", "Natural-language task description"),
    ]
    output_ports = [
        PortDef("action_chunk", "TEXT", "JSON [[dx,dy,dz,drx,dry,drz,grip]]"),
        PortDef("terminate", "BOOL", "Octo's terminate_episode > 0.5"),
    ]

    config_schema = {
        "checkpoint_path": {
            "type": "string",
            "default": "",
            "description": (
                "Absolute path to the Octo HF snapshot directory. "
                "Empty = VLA_CHECKPOINTS_ROOT/<model_type>-1.5; if that's missing too, "
                "fall back to hf://rail-berkeley/<model_type>."
            ),
        },
        "model_type": {
            "type": "string",
            "default": "octo-small",
            "enum": ["octo-small", "octo-base"],
        },
        "policy_setup": {
            "type": "string",
            "default": "widowx_bridge",
            "enum": ["widowx_bridge", "google_robot"],
            "description": "Per-embodiment branch — pick to match the active SIMPLER split.",
        },
        "horizon": {"type": "integer", "default": 2},
        "pred_action_horizon": {"type": "integer", "default": 4},
        "exec_horizon": {"type": "integer", "default": 1},
        "image_size": {"type": "integer", "default": 256},
        "action_scale": {"type": "number", "default": 1.0},
        "init_rng": {"type": "integer", "default": 0},
    }

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        model_type = self.config.get("model_type", "octo-small")
        ckpt = (self.config.get("checkpoint_path") or "").strip()
        if not ckpt:
            ckpt = _resolve_default_checkpoint(model_type)
        policy_setup = self.config.get("policy_setup", "widowx_bridge")
        horizon = int(self.config.get("horizon", 2))
        pred_action_horizon = int(self.config.get("pred_action_horizon", 4))
        exec_horizon = int(self.config.get("exec_horizon", 1))
        image_size = int(self.config.get("image_size", 256))
        action_scale = float(self.config.get("action_scale", 1.0))
        init_rng = int(self.config.get("init_rng", 0))

        try:
            image = _coerce_image(inputs.get("image"))
        except ValueError as e:
            self._self_log("error", f"bad image: {e!s}")
            return {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]]), "terminate": False}

        instruction = (inputs.get("instruction") or "").strip()
        if not instruction:
            self._self_log("warn", "empty instruction — Octo will encode the empty string")

        loop = asyncio.get_running_loop()
        try:
            bundle = await loop.run_in_executor(
                None, _ensure_loaded,
                ckpt, model_type, policy_setup, horizon, pred_action_horizon,
                exec_horizon, image_size, action_scale, init_rng,
            )
        except Exception as e:  # noqa: BLE001
            self._self_log("error", f"load failed: {e!r}")
            return {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]]), "terminate": False}

        model = bundle["model"]
        # Octo's reset() rebuilds the task token + clears the image-history deque
        # and action ensembler. Call only when the instruction changes.
        if bundle.get("current_instruction") != instruction:
            await loop.run_in_executor(None, model.reset, instruction)
            bundle["current_instruction"] = instruction
            self._self_log("reset", instruction[:60])

        raw_action, action = await loop.run_in_executor(
            None, lambda: model.step(image, instruction)
        )
        wv = np.asarray(action["world_vector"], dtype=np.float32).reshape(-1)
        ra = np.asarray(action["rot_axangle"], dtype=np.float32).reshape(-1)
        gp_raw = action.get("gripper", np.zeros(1, dtype=np.float32))
        gp = np.asarray(gp_raw, dtype=np.float32).reshape(-1)
        if gp.size == 0:
            gp = np.zeros(1, dtype=np.float32)
        terminate = bool(np.any(np.asarray(action.get("terminate_episode", 0.0)) > 0.5))

        chunk = np.concatenate([wv[:3], ra[:3], gp[:1]], axis=0).astype(np.float32).tolist()
        self._self_log("action_norm", f"|wv|={float(np.linalg.norm(wv)):.3f} |ra|={float(np.linalg.norm(ra)):.3f} grip={float(gp[0]):.2f}")
        return {"action_chunk": json.dumps([chunk]), "terminate": terminate}


class PolicyOctoNodeSet(BaseNodeSet):
    """Octo SimplerEnv-native policy as a NodeSet.

    Loads in server mode against the ``ac-octo`` conda env by default.
    ``server_python`` reads from ``$OCTO_PYTHON`` then falls back to the conda
    env path created by ``scripts/install/install_ac_octo.sh``.
    """

    name = "policy_octo"
    description = (
        "Octo (RAIL Berkeley) — JAX/Flax VLA baseline for SimplerEnv. "
        "Single ``policy_octo__predict`` node."
    )
    server_python = conda_env_python("ac-octo", "OCTO_PYTHON")
    parallelism = "replicated"  # Per-worker JAX state.
    default_per_step_budget_sec = 30.0

    def get_tools(self) -> list:
        return [OctoPredictTool()]

    async def initialize(self, **kwargs: Any) -> None:
        pass

    async def shutdown(self) -> None:
        global _MODEL_BUNDLE
        _MODEL_BUNDLE = None
