"""Seq2Seq Policy wrapper for stage 3 (predict).

Wraps ``vlnce_baselines.models.seq2seq_policy.Seq2SeqNet`` with the same
``VlnPolicy`` interface used by CmaPolicy. Compatible with the published
Seq2Seq_DA checkpoint (and the un-released Seq2Seq{_PM,_DA,_Aug,_PM_DA_Aug}
variants if a user trains them).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from workspace.nodesets.policy.policy_adapter_vlnce.policies.base_policy import VlnPolicy

DEFAULT_KWARGS: dict[str, Any] = {}


class Seq2SeqPolicy(VlnPolicy):
    arch: ClassVar[str] = "seq2seq"

    def __init__(self) -> None:
        self._net: Any = None
        self._wrapper: Any = None
        self._device: Any = None
        self._obs_transforms: list[Any] | None = None

    def build(
        self,
        *,
        observation_space: Any,
        action_space: Any,
        model_config: Any,
        obs_transforms: list,
    ) -> None:
        import habitat_extensions  # noqa: F401
        import vlnce_baselines  # noqa: F401
        from vlnce_baselines.models.seq2seq_policy import Seq2SeqNet

        from workspace.nodesets.policy.policy_adapter_vlnce.policies._recurrent import (
            make_recurrent_policy_classes,
        )

        _, RecurrentPolicy = make_recurrent_policy_classes()

        net = Seq2SeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
        )
        self._net = net
        self._wrapper = RecurrentPolicy(net, action_space.n)
        self._obs_transforms = obs_transforms

    def load_checkpoint(self, path: str) -> None:
        self._wrapper.load_checkpoint(path)

    def to(self, device: Any) -> Seq2SeqPolicy:
        self._wrapper = self._wrapper.to(device)
        self._device = device
        return self

    def eval(self) -> Seq2SeqPolicy:
        self._wrapper = self._wrapper.eval()
        return self

    def forward(self, model_batch: dict[str, Any]) -> dict[str, Any]:
        import torch
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_batch,
        )
        from habitat_baselines.utils.common import batch_obs

        device = self._device
        observations = model_batch["raw_observations"]
        obs_batch = batch_obs(observations, device)
        obs_batch = apply_obs_transforms_batch(obs_batch, self._obs_transforms)
        hidden_in = model_batch.get("hidden_in")

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
        # Seq2SeqNet doesn't store _hidden_size like CMANet does; pull from
        # the GRU inside its state_encoder so RecurrentPolicy.hidden_size
        # (which reads _hidden_size and defaults to -1) doesn't end up
        # building zero-state tensors with negative dimensions.
        return self._wrapper.net.state_encoder.rnn.hidden_size

    @property
    def device(self) -> Any:
        return self._device

    @property
    def num_parameters(self) -> int:
        try:
            return sum(p.numel() for p in self._wrapper.parameters())
        except Exception:
            return -1

    def parameters(self) -> Any:
        return self._wrapper.parameters()
