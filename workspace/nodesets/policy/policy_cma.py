"""CMA Policy NodeSet — neural VLN-CE policy as a canvas node.

Provides a single tool: ``PolicyForwardTool``, which lazily loads the CMA
checkpoint on first execution and runs recurrent forward passes across
iterations. ADR-028 PC-3: opted into the batched-inference tier
(``batched=True``, ``batch_dim="raw_obs"``). RNN hidden state lives on
the wire — explicit ``hidden_in``/``hidden_out`` ports — so the
policy subprocess can serve K parallel workers from one shared
checkpoint with one stacked forward pass per step.

Standalone: builds observation/action spaces from the raw_obs input port —
no dependency on other nodesets or framework internals. Policy classes
(BaseVLNPolicy, RecurrentPolicy, CMAPolicy) are defined inline inside the
lazy load function so no vlnce-env imports occur at module load time.

Requires:
  - raw_obs input wired from an env node (Habitat observe/step)
  - hidden_in input from the previous iteration (None on iter 0 → fresh state)
  - CMA checkpoint at data/habitat/checkpoints/CMA_PM_DA_Aug.pth

last updated: 2026-04-18
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef
from app.server.batched_inference import OUTPUTS_KEY, SAMPLES_KEY

log = logging.getLogger("agentcanvas.policy-cma")


# ── Class-level policy singleton (ADR-028 PC-3) ──
# One subprocess hosts one policy_cma server hosting one PolicyForwardTool
# instance. The model checkpoint is loaded once for the lifetime of the
# subprocess and shared across all batched calls. A threading.Lock protects
# the load — only one loader thread may run, others wait and reuse the
# bundle. This replaces the prior ``ctx.policy``/``ctx.rnn_states`` per-
# instance attributes which were the per-worker-state side channel that
# the batched server is meant to eliminate.
_POLICY_BUNDLE: dict | None = None
_POLICY_LOAD_LOCK = threading.Lock()


def _build_spaces_from_obs(raw_obs: dict) -> tuple:
    """Build gym observation/action spaces from a raw Habitat observation dict.

    VLN-CE action space is always Discrete(4): STOP, FORWARD, LEFT, RIGHT.
    Observation space is inferred from the numpy array shapes in raw_obs.
    Handles both numpy arrays (local mode) and Python lists (server mode,
    after HTTP JSON round-trip).
    """
    import gym.spaces as spaces
    import numpy as np

    obs_dict = {}
    for key, val in raw_obs.items():
        if not isinstance(val, (np.ndarray, list)):
            continue
        arr = np.asarray(val)
        if arr.size == 0:
            continue
        if np.issubdtype(arr.dtype, np.integer):
            obs_dict[key] = spaces.Box(
                low=np.iinfo(arr.dtype).min,
                high=np.iinfo(arr.dtype).max,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        else:
            obs_dict[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=arr.dtype,
            )
    obs_space = spaces.Dict(obs_dict)
    action_space = spaces.Discrete(4)
    return obs_space, action_space


class PolicyForwardTool(BaseCanvasNode):
    """Run a neural VLN-CE policy (CMA) forward pass.

    Batched (ADR-028 PC-3): K parallel callers rendezvous in the
    :class:`BatchedInferenceServer` hosted by this tool's
    :class:`AutoServerApp`. The server collects K samples, calls
    ``execute`` once with all K under :data:`SAMPLES_KEY`, and the tool
    returns K results under :data:`OUTPUTS_KEY`. Single-sample callers
    (canvas Play, local-mode load) bypass the server and ``execute``
    promotes the call into a degenerate batch of 1.

    RNN hidden state on the wire: ``hidden_in`` is None on iteration 0
    (policy initialises zeros internally) and a dict of numpy arrays on
    subsequent iterations. The graph wires ``hidden_out`` back to
    ``hidden_in`` via the IterIn/IterOut feedback edge, so each worker
    carries its own hidden state across steps without any per-worker
    state on the policy server.
    """

    node_type = "policy_cma__forward"
    display_name = "Policy: CMA Forward"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")
    description = "CMA neural policy forward pass — outputs discrete action"
    category = "policy"
    icon = "Brain"

    # ADR-028 PC-3 opt-in. ``batch_dim`` names the input port carrying the
    # per-sample slot (validated at scan time in ``register_node``).
    batched: ClassVar[bool] = True
    batch_dim: ClassVar[str] = "raw_obs"

    input_ports = [
        PortDef(
            "raw_obs",
            "TEXT",
            "Raw Habitat observation dict (from env observe/step)",
        ),
        PortDef(
            "hidden_in",
            "ANY",
            "RNN hidden state from prior iteration (None on first step)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("action", "ACTION", "Predicted navigation action"),
        PortDef(
            "hidden_out",
            "ANY",
            "RNN hidden state for next iteration (wire back via IterOut → IterIn)",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # Detect batched-server invocation: AutoServerApp passes K samples
        # under SAMPLES_KEY when batched=True. Local/canvas callers pass a
        # single inputs dict — promote to a batch of 1 so one code path
        # handles both.
        raw_samples = inputs.get(SAMPLES_KEY)
        is_batched_call = isinstance(raw_samples, list)
        samples = raw_samples if is_batched_call else [inputs]

        # Validate: at least the first sample must carry a raw_obs dict to
        # build the observation space and load the policy. Mixed-validity
        # batches are not expected in practice (all K workers are
        # mid-episode); fall back to STOP for everyone if validation fails.
        first_obs = samples[0].get("raw_obs") if samples else None
        if not isinstance(first_obs, dict):
            outputs = [{"action": 0, "hidden_out": s.get("hidden_in")} for s in samples]
            return {OUTPUTS_KEY: outputs} if is_batched_call else outputs[0]

        loop = asyncio.get_running_loop()

        # Lazy class-level singleton load — survives across calls and
        # across batches. Threading lock inside _ensure_policy_loaded
        # serialises concurrent first-touch loaders.
        bundle = await loop.run_in_executor(None, _ensure_policy_loaded, first_obs)
        if bundle is None:
            outputs = [{"action": 0, "hidden_out": s.get("hidden_in")} for s in samples]
            return {OUTPUTS_KEY: outputs} if is_batched_call else outputs[0]

        # One stacked forward pass for all K samples.
        actions, hidden_outs = await loop.run_in_executor(
            None,
            _batched_forward,
            bundle,
            samples,
        )

        from app.standard.actions import ACTION_NAMES

        outputs: list[dict] = []
        for action, hidden_out in zip(actions, hidden_outs):
            outputs.append({"action": int(action), "hidden_out": hidden_out})

        # Self-log the head sample so single-sample callers see normal
        # log entries; for K>1 batches, also log batch_size for visibility.
        if outputs:
            head = outputs[0]["action"]
            self._self_log("predicted_action", head)
            self._self_log("action_name", ACTION_NAMES.get(head, "UNKNOWN"))
            if len(outputs) > 1:
                self._self_log("batch_size", len(outputs))

        if is_batched_call:
            return {OUTPUTS_KEY: outputs}
        return outputs[0]


# ── Singleton policy loader ──


def _ensure_policy_loaded(raw_obs: dict) -> dict | None:
    """Load the CMA policy on first call; return the cached bundle thereafter.

    Blocking — call from a thread (run_in_executor). The threading lock
    serialises concurrent first-touch loaders so the checkpoint is read
    exactly once even if K batched callers race.
    """
    global _POLICY_BUNDLE
    if _POLICY_BUNDLE is not None:
        return _POLICY_BUNDLE
    with _POLICY_LOAD_LOCK:
        if _POLICY_BUNDLE is not None:
            return _POLICY_BUNDLE
        try:
            policy, config, obs_transforms, device = _load_cma_policy(raw_obs)
        except Exception:
            log.exception("Failed to load CMA policy")
            return None
        _POLICY_BUNDLE = {
            "policy": policy,
            "policy_config": config,
            "obs_transforms": obs_transforms,
            "device": device,
        }
        log.info(
            "CMA policy bundle ready (params=%d)",
            sum(p.numel() for p in policy.parameters()),
        )
        return _POLICY_BUNDLE


# ── Blocking helpers (run in a worker thread via run_in_executor) ──


def _load_cma_policy(raw_obs: dict) -> tuple:
    """Load the CMA policy checkpoint. Blocking.

    Builds observation/action spaces from raw_obs shapes — no live
    environment manager needed. Policy classes are defined inline here so
    that no vlnce-env imports occur at module load time.
    """
    from abc import ABC, abstractmethod

    import habitat_extensions  # noqa: F401
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import vlnce_baselines  # noqa: F401
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_obs_space,
        get_active_obs_transforms,
    )
    from habitat_baselines.rl.ppo.policy import Net
    from habitat_baselines.utils.common import CategoricalNet
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.cma_policy import CMANet
    from vlnce_baselines.models.utils import CustomFixedCategorical

    # ── BaseVLNPolicy ─────────────────────────────────────────────────────
    class BaseVLNPolicy(nn.Module, ABC):
        """Abstract base class for all VLN-CE policies."""

        def __init__(self):
            super().__init__()

        @abstractmethod
        def act(self, observations: dict[str, Any], **kwargs) -> Any:
            raise NotImplementedError

        @abstractmethod
        def compute_loss(self, batch: dict) -> torch.Tensor:
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def from_config(cls, config, observation_space, action_space) -> BaseVLNPolicy:
            raise NotImplementedError

        def load_checkpoint(self, path: str, map_location: str = "cpu") -> None:
            ckpt = torch.load(path, map_location=map_location)
            if "state_dict" in ckpt:
                self.load_state_dict(ckpt["state_dict"])
            else:
                self.load_state_dict(ckpt)

        def reset(self) -> None:
            pass

        @property
        def device(self) -> torch.device:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        @property
        def is_recurrent(self) -> bool:
            return False

    # ── RecurrentPolicy ───────────────────────────────────────────────────
    class RecurrentPolicy(BaseVLNPolicy):
        """Base class for RNN-based VLN-CE policies (CMA, Seq2Seq).

        Architecture:
            observations → net.forward() → features → action_distribution → action
                                │                                              │
                           rnn_states_in                               rnn_states_out
        """

        def __init__(self, net: Net, dim_actions: int):
            super().__init__()
            self.net = net
            self.dim_actions = dim_actions
            self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)

        def act(
            self,
            observations: dict[str, torch.Tensor],
            rnn_states: torch.Tensor,
            prev_actions: torch.Tensor,
            masks: torch.Tensor,
            deterministic: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            features, rnn_states = self.net(observations, rnn_states, prev_actions, masks)
            distribution = self.action_distribution(features)
            action = distribution.mode() if deterministic else distribution.sample()
            return action, rnn_states

        def compute_loss(
            self,
            observations: dict[str, torch.Tensor],
            prev_actions: torch.Tensor,
            not_done_masks: torch.Tensor,
            corrected_actions: torch.Tensor,
            weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            T, N = corrected_actions.size()
            rnn_states = torch.zeros(
                N,
                self.net.num_recurrent_layers,
                self.net._hidden_size,
                device=self.device,
            )
            distribution = self.build_distribution(
                observations, rnn_states, prev_actions, not_done_masks
            )
            logits = distribution.logits.view(T, N, -1)
            action_loss = F.cross_entropy(
                logits.permute(0, 2, 1), corrected_actions, reduction="none"
            )
            action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()
            return action_loss, action_loss

        def build_distribution(
            self,
            observations: dict[str, torch.Tensor],
            rnn_states: torch.Tensor,
            prev_actions: torch.Tensor,
            masks: torch.Tensor,
        ) -> CustomFixedCategorical:
            features, _ = self.net(observations, rnn_states, prev_actions, masks)
            return self.action_distribution(features)

        @property
        def is_recurrent(self) -> bool:
            return True

        @property
        def num_recurrent_layers(self) -> int:
            return self.net.num_recurrent_layers

        @property
        def output_size(self) -> int:
            return self.net.output_size

        @property
        def num_actions(self) -> int:
            return self.dim_actions

    # ── CMAPolicy ─────────────────────────────────────────────────────────
    class CMAPolicy(RecurrentPolicy):
        """Cross-Modal Attention policy for VLN-CE.

        Two-layer GRU with cross-modal attention between instruction text,
        RGB features, and depth features. See https://arxiv.org/abs/2004.02857
        """

        def __init__(self, observation_space, action_space, model_config):
            net = CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            )
            super().__init__(net, action_space.n)

        @classmethod
        def from_config(cls, config, observation_space, action_space):
            return cls(
                observation_space=observation_space,
                action_space=action_space,
                model_config=config.MODEL,
            )

    # ── load checkpoint ───────────────────────────────────────────────────
    obs_space, action_space = _build_spaces_from_obs(raw_obs)

    # __file__ lives at workspace/nodesets/policy/policy_cma.py — three
    # parents to reach the repo root.
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
    )

    checkpoint = os.path.join(repo_root, "data", "habitat", "checkpoints", "CMA_PM_DA_Aug.pth")
    exp_config = "vlnce_baselines/config/r2r_baselines/cma_pm_da.yaml"

    # VLN-CE root resolution
    candidates = [
        os.environ.get("VLNCE_ROOT", ""),
        os.path.join(repo_root, "..", "VLN-CE"),
        os.path.join(repo_root, "third_party", "VLN-CE"),
    ]
    vlnce_root = None
    for c in candidates:
        c = os.path.normpath(c) if c else ""
        if c and os.path.isdir(os.path.join(c, "data")):
            vlnce_root = c
            break
    if vlnce_root is None:
        vlnce_root = os.path.normpath(
            os.path.join(repo_root, "third_party", "VLN-CE"),
        )
    if os.path.isdir(vlnce_root):
        os.chdir(vlnce_root)

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    config = get_config(exp_config)
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "val_unseen"
    config.TASK_CONFIG.DATASET.ROLES = ["guide"]
    config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    config.TASK_CONFIG.TASK.NDTW.SPLIT = "val_unseen"
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.SIMULATOR_GPU_IDS = [gpu_id]
    config.TORCH_GPU_ID = gpu_id
    config.NUM_ENVIRONMENTS = 1
    config.freeze()

    obs_transforms = get_active_obs_transforms(config)
    transformed_obs_space = apply_obs_transforms_obs_space(obs_space, obs_transforms)

    log.info("Loading CMA policy from %s on %s ...", checkpoint, device)
    policy = CMAPolicy.from_config(
        config=config,
        observation_space=transformed_obs_space,
        action_space=action_space,
    )
    policy.load_checkpoint(checkpoint)
    policy.to(device)
    policy.eval()
    log.info("CMA policy loaded (params=%d)", sum(p.numel() for p in policy.parameters()))
    return policy, config, obs_transforms, device


def _batched_forward(bundle: dict, samples: list[dict]) -> tuple[list[int], list[dict]]:
    """Run a stacked policy forward pass across N samples. Blocking.

    Each sample carries:
      raw_obs:    dict (raw Habitat obs)
      hidden_in:  dict | None — {rnn_states, prev_actions, not_done_masks}
                  None on iteration 0 → fresh state with mask=0

    Returns:
      actions:     list[int] of length N (per-sample discrete action)
      hidden_outs: list[dict] of length N (per-sample state for next call)
    """
    import numpy as np
    import torch
    from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
    from habitat_baselines.utils.common import batch_obs
    from vlnce_baselines.common.utils import extract_instruction_tokens

    policy = bundle["policy"]
    policy_config = bundle["policy_config"]
    obs_transforms = bundle["obs_transforms"]
    device = bundle["device"]

    n = len(samples)
    num_layers = policy.net.num_recurrent_layers
    hidden_size = policy_config.MODEL.STATE_ENCODER.hidden_size

    # Stack observations into a single batched obs dict (N, ...)
    observations = [s["raw_obs"] for s in samples]
    observations = extract_instruction_tokens(
        observations,
        policy_config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
    )
    obs_batch = batch_obs(observations, device)
    obs_batch = apply_obs_transforms_batch(obs_batch, obs_transforms)

    # Stack hidden state per sample. None → zeros + mask=0 (fresh state).
    rnn_states_list = []
    prev_actions_list = []
    masks_list = []
    for s in samples:
        h = s.get("hidden_in")
        if isinstance(h, dict) and "rnn_states" in h:
            rnn_states_list.append(
                torch.as_tensor(np.asarray(h["rnn_states"]), dtype=torch.float32).to(device)
            )
            prev_actions_list.append(
                torch.as_tensor(np.asarray(h["prev_actions"]), dtype=torch.long).to(device)
            )
            masks_list.append(
                torch.as_tensor(np.asarray(h["not_done_masks"]), dtype=torch.uint8).to(device)
            )
        else:
            rnn_states_list.append(torch.zeros(1, num_layers, hidden_size, device=device))
            prev_actions_list.append(torch.zeros(1, 1, dtype=torch.long, device=device))
            masks_list.append(torch.zeros(1, 1, dtype=torch.uint8, device=device))

    rnn_states = torch.cat(rnn_states_list, dim=0)  # (N, num_layers, hidden_size)
    prev_actions = torch.cat(prev_actions_list, dim=0)  # (N, 1)
    not_done_masks = torch.cat(masks_list, dim=0)  # (N, 1)

    with torch.no_grad():
        actions, new_rnn_states = policy.act(
            obs_batch,
            rnn_states,
            prev_actions,
            not_done_masks,
            deterministic=True,
        )

    # Split per-sample. After this step every sample is "not done" from the
    # RNN's point of view, so future calls should use mask=1 to carry the
    # state forward. (Episode termination is handled by the env, not here.)
    out_actions: list[int] = []
    hidden_outs: list[dict] = []
    for i in range(n):
        out_actions.append(int(actions[i, 0].item()))
        hidden_outs.append(
            {
                "rnn_states": new_rnn_states[i : i + 1].detach().cpu().numpy(),
                "prev_actions": actions[i : i + 1].detach().cpu().long().numpy(),
                "not_done_masks": np.ones((1, 1), dtype=np.uint8),
            }
        )

    return out_actions, hidden_outs


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class PolicyCMANodeSet(BaseNodeSet):
    """CMA neural policy as a loadable NodeSet.

    Load this alongside EnvHabitatNodeSet to use neural policy
    navigation on the canvas. The policy is loaded lazily on first
    execution (not at nodeset load time) and shared as a class-level
    singleton across batched callers (ADR-028 PC-3).

    Requires the vlnce conda env (Python 3.8 + habitat-sim). When the
    backend runs under the agentcanvas env, load with ?mode=server to
    auto-route to the vlnce interpreter (ADR-020).
    """

    name = "policy_cma"
    description = "CMA cross-modal attention policy (VLN-CE)"
    server_python = os.environ.get("VLNCE_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-vlnce/bin/python"))

    def get_tools(self) -> list:
        return [PolicyForwardTool()]

    async def initialize(self, **kwargs: Any) -> None:
        pass

    async def shutdown(self) -> None:
        pass
