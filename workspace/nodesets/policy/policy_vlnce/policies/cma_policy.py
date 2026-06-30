"""CMA Policy wrapper for stage 3 (predict).

Wraps ``vlnce_baselines.models.cma_policy.CMANet`` with a uniform
``VlnPolicy`` interface so the canvas predict node doesn't need to
know about RecurrentPolicy / CategoricalNet plumbing.

Compatible with all R2R-CE CMA checkpoints
(``CMA.pth, CMA_DA.pth, CMA_PM.pth, CMA_Aug.pth, CMA_PM_DA_Aug.pth, ...``)
and with the RxR-CE monolingual CMA checkpoint (``rxr_cma_en.pth``)
when paired with an exp_config from ``rxr_baselines/``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from workspace.nodesets.policy.policy_vlnce.policies.base_policy import VlnPolicy

DEFAULT_KWARGS: dict[str, Any] = {}


class CmaPolicy(VlnPolicy):
    arch: ClassVar[str] = "cma"

    def __init__(self) -> None:
        self._net: Any = None  # underlying CMANet
        self._wrapper: Any = (
            None  # RecurrentPolicy(net) — owns CategoricalNet head + load_checkpoint
        )
        self._device: Any = None
        self._obs_transforms: list[Any] | None = None  # set in build()

    # ---- VlnPolicy interface ----

    def build(
        self,
        *,
        observation_space: Any,
        action_space: Any,
        model_config: Any,
        obs_transforms: list,
    ) -> None:
        # noqa: side-effect imports register vlnce-extensions / habitat-extensions
        import habitat_extensions  # noqa: F401
        import vlnce_baselines  # noqa: F401
        from vlnce_baselines.models.cma_policy import CMANet

        from workspace.nodesets.policy.policy_vlnce.policies._recurrent import (
            make_recurrent_policy_classes,
        )

        _, RecurrentPolicy = make_recurrent_policy_classes()

        net = CMANet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
        )
        self._net = net
        self._wrapper = RecurrentPolicy(net, action_space.n)
        self._obs_transforms = obs_transforms

    def load_checkpoint(self, path: str) -> None:
        self._wrapper.load_checkpoint(path)

    def to(self, device: Any) -> CmaPolicy:
        self._wrapper = self._wrapper.to(device)
        self._device = device
        return self

    def eval(self) -> CmaPolicy:
        self._wrapper = self._wrapper.eval()
        return self

    def forward(self, model_batch: dict[str, Any]) -> dict[str, Any]:
        import torch
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_batch,
        )
        from habitat_baselines.utils.common import batch_obs

        device = self._device
        # Stage 2 emitted the post-extract_instruction_tokens raw obs list
        # (numpy / python types) on the wire. Build obs_batch + apply
        # transforms HERE in torch on device — avoids the round-trip
        # torch→numpy→torch that breaks downstream behavior.
        observations = model_batch["raw_observations"]
        try:
            obs_batch = batch_obs(observations, device)
        except RuntimeError as e:
            if "Could not infer dtype" in str(e):
                import logging

                _log = logging.getLogger("agentcanvas.policy_vlnce")
                _log.error("batch_obs failed; raw_obs keys + types:")
                for k, v in observations[0].items():
                    _log.error(
                        "  %-30s %s shape=%s",
                        k,
                        type(v).__name__,
                        getattr(v, "shape", None)
                        if hasattr(v, "shape")
                        else len(v)
                        if hasattr(v, "__len__")
                        else "?",
                    )
            raise
        obs_batch = apply_obs_transforms_batch(obs_batch, self._obs_transforms)
        hidden_in = model_batch.get("hidden_in")

        # Build per-sample hidden state tensors (zeros + mask=0 on iter 0).
        # CMA's num_recurrent_layers is a Net property — not in config —
        # so this build happens here (stage 3) where the policy is loaded.
        num_layers = self.num_recurrent_layers
        hidden_size = self.hidden_size
        if isinstance(hidden_in, dict) and "rnn_states" in hidden_in:
            rnn_states = torch.as_tensor(
                np.asarray(hidden_in["rnn_states"]),
                dtype=torch.float32,
                device=device,
            )
            prev_actions = torch.as_tensor(
                np.asarray(hidden_in["prev_actions"]),
                dtype=torch.long,
                device=device,
            )
            not_done_masks = torch.as_tensor(
                np.asarray(hidden_in["not_done_masks"]),
                dtype=torch.uint8,
                device=device,
            )
        else:
            rnn_states = torch.zeros(1, num_layers, hidden_size, device=device)
            prev_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
            not_done_masks = torch.zeros(1, 1, dtype=torch.uint8, device=device)

        with torch.no_grad():
            actions, new_rnn_states = self._wrapper.act(
                obs_batch,
                rnn_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
            )

        action_scalar = int(actions[0, 0].item())
        # Build hidden_out dict (numpy, B=1) for the wire-state contract.
        rnn_states_np = new_rnn_states.detach().cpu().numpy()
        prev_actions_np = actions.detach().cpu().long().numpy()
        return {
            "action_index": action_scalar,
            "rnn_states_out": rnn_states_np,
            "prev_actions_out": prev_actions_np,
            "not_done_masks_out": np.ones((1, 1), dtype=np.uint8),
        }

    @property
    def num_recurrent_layers(self) -> int:
        return self._wrapper.num_recurrent_layers

    @property
    def hidden_size(self) -> int:
        return self._wrapper.hidden_size

    @property
    def device(self) -> Any:
        return self._device

    @property
    def num_parameters(self) -> int:
        try:
            return sum(p.numel() for p in self._wrapper.parameters())
        except Exception:
            return -1

    # ---- internal: torch nn passthrough for to/eval/parameters ----

    def parameters(self) -> Any:
        return self._wrapper.parameters()
