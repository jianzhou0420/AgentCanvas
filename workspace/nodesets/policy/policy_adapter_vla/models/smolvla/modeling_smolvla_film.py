#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolVLA modeling — FiLM conditioning variant.

Uses Feature-wise Linear Modulation (FiLM) instead of cross-attention
to condition the action expert on VLM outputs.

For the original cross-attention variant, see modeling_smolvla.py.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn

from workspace.nodesets.policy.policy_adapter_vla.models.action_expert import (
    ConditionalUnet1D,
    MLPForDiffusion,
    TransformerForDiffusionFiLM,
)
from workspace.nodesets.policy.policy_adapter_vla.models.smolvla.configuration_smolvla import SmolVLAConfig
from workspace.nodesets.policy.policy_adapter_vla.models.smolvla.film import create_pooler
from workspace.nodesets.policy.policy_adapter_vla.models.smolvla.modeling_smolvla import (
    aloha_gripper_from_angular,
    aloha_gripper_from_angular_inv,
    aloha_gripper_to_angular,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
    pad_vector,
    resize_with_pad,
)
from workspace.nodesets.policy.policy_adapter_vla.models.smolvla.smolvlm_with_expert_film import (
    SmolVLMWithExpertFilmModel,
)


class VLAFlowMatchingFilm(nn.Module):
    """SmolVLA with FiLM conditioning.

    Instead of cross-attention between VLM and Expert, this variant:
    1. Runs VLM forward on the prefix (images + language + state)
    2. Pools VLM output into a conditioning vector via VLMPooler
    3. Runs Expert with self-attention + FiLM layers modulated by the conditioning vector
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertFilmModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device,
            film_cond_dim=self.config.film_cond_dim,
            film_after_self_attn=self.config.film_after_self_attn,
            film_after_ffn=self.config.film_after_ffn,
            film_pool_mode=self.config.film_pool_mode,
            film_layer_agg=self.config.film_layer_agg,
            film_layer_agg_n=self.config.film_layer_agg_n,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )

        if self.config.expert_type == "gemma":
            # Current: Gemma expert with action projections
            self.action_in_proj = nn.Linear(
                self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size
            )
            self.action_out_proj = nn.Linear(
                self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim
            )
            self.action_time_mlp_in = nn.Linear(
                self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
            )
            self.action_time_mlp_out = nn.Linear(
                self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
            )
            self.action_expert = None
        else:
            # Alternative: self-contained action expert (handles its own projections)
            cond_dim = self._resolve_cond_dim()
            self._override_vlm_pooler(cond_dim)
            self.action_expert = self._build_action_expert(cond_dim)
            self.action_in_proj = None
            self.action_out_proj = None
            self.action_time_mlp_in = None
            self.action_time_mlp_out = None

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def _resolve_cond_dim(self) -> int:
        """Determine the conditioning dimension for alternative action experts."""
        cfg = self.config
        if cfg.film_cond_dim > 0:
            return cfg.film_cond_dim
        ec = cfg.expert_config
        if cfg.expert_type in ("transformer_film", "mlp"):
            return ec.get("n_emb", 256)
        elif cfg.expert_type == "cnn":
            return ec.get("diffusion_step_embed_dim", 256) * 4
        return 256

    def _override_vlm_pooler(self, cond_dim: int):
        """Replace VLM pooler if its output dimension doesn't match cond_dim."""
        current_pooler = self.vlm_with_expert.vlm_pooler
        if hasattr(current_pooler, "proj") and isinstance(current_pooler.proj, nn.Linear):
            current_out = current_pooler.proj.out_features
        elif hasattr(current_pooler, "out_proj") and isinstance(current_pooler.out_proj, nn.Linear):
            current_out = current_pooler.out_proj.out_features
        else:
            current_out = self.vlm_with_expert.config.text_config.hidden_size
        if current_out != cond_dim:
            vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
            self.vlm_with_expert.vlm_pooler = create_pooler(
                self.config.film_pool_mode,
                vlm_hidden,
                cond_dim,
            )

    def _build_action_expert(self, cond_dim: int) -> nn.Module:
        """Factory method to create the appropriate action expert."""
        cfg = self.config
        ec = cfg.expert_config
        if cfg.expert_type == "cnn":
            return ConditionalUnet1D(
                input_dim=cfg.max_action_dim,
                global_cond_dim=cond_dim,
                diffusion_step_embed_dim=ec.get("diffusion_step_embed_dim", 256),
                down_dims=ec.get("down_dims", [256, 512]),
                kernel_size=ec.get("kernel_size", 3),
                n_groups=ec.get("n_groups", 8),
                cond_predict_scale=ec.get("cond_predict_scale", False),
            )
        elif cfg.expert_type == "transformer_film":
            return TransformerForDiffusionFiLM(
                input_dim=cfg.max_action_dim,
                output_dim=cfg.max_action_dim,
                horizon=cfg.chunk_size,
                n_obs_steps=1,
                cond_dim=cond_dim,
                n_layer=ec.get("n_layer", 8),
                n_head=ec.get("n_head", 8),
                n_emb=ec.get("n_emb", 256),
                p_drop_emb=ec.get("p_drop_emb", 0.1),
                p_drop_attn=ec.get("p_drop_attn", 0.1),
                causal_attn=ec.get("causal_attn", False),
                time_as_cond=ec.get("time_as_cond", True),
                enable_film_self_attn=ec.get("enable_film_self_attn", True),
                enable_film_ffn=ec.get("enable_film_ffn", True),
                n_cond_layers=ec.get("n_cond_layers", 0),
            )
        elif cfg.expert_type == "mlp":
            return MLPForDiffusion(
                input_dim=cfg.max_action_dim,
                output_dim=cfg.max_action_dim,
                horizon=cfg.chunk_size,
                n_obs_steps=1,
                cond_dim=cond_dim,
                n_layer=ec.get("n_layer", 8),
                n_emb=ec.get("n_emb", 256),
                p_drop_emb=ec.get("p_drop_emb", 0.1),
                p_drop_attn=ec.get("p_drop_attn", 0.1),
                time_as_cond=ec.get("time_as_cond", True),
                parallel_input_emb=ec.get("parallel_input_emb", True),
            )
        else:
            raise ValueError(f"Unknown expert_type: {cfg.expert_type}")

    def _forward_action_expert(self, x_t: Tensor, timestep: Tensor, film_cond: Tensor) -> Tensor:
        """Forward through the alternative action expert with the correct kwarg name."""
        if self.config.expert_type == "cnn":
            return self.action_expert(sample=x_t, timestep=timestep, global_cond=film_cond)
        else:
            return self.action_expert(sample=x_t, timestep=timestep, cond=film_cond)

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        if self.config.expert_type == "gemma":
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_pos_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1

            (_, suffix_out), _, _ = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_pos_ids,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                prefix_pad_mask=prefix_pad_masks,
                expert_attention_mask=suffix_att_2d,
                expert_position_ids=suffix_pos_ids,
            )

            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
        else:
            # Alternative expert: VLM+pool only, then self-contained expert
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            (_, _), _, film_cond = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_pos_ids,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                fill_kv_cache=False,
                prefix_pad_mask=prefix_pad_masks,
            )
            v_t = self._forward_action_expert(x_t, time, film_cond)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Phase 1: VLM forward + pool to get film_cond
        (vlm_out, _), _, film_cond = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
            prefix_pad_mask=prefix_pad_masks,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                x_t,
                expanded_time,
                film_cond,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        x_t,
        timestep,
        film_cond,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if self.config.expert_type == "gemma":
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_pos_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _, _ = self.vlm_with_expert.forward(
                attention_mask=suffix_att_2d,
                position_ids=suffix_pos_ids,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                film_cond=film_cond,
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
        else:
            v_t = self._forward_action_expert(x_t, timestep, film_cond)
        return v_t


class SmolVLAFilmPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatchingFilm model to train and run inference within LeRobot.

    FiLM conditioning variant — uses Feature-wise Linear Modulation instead of cross-attention.
    For the original cross-attention variant, see SmolVLAPolicy in modeling_smolvla.py.
    """

    config_class = SmolVLAConfig
    name = "smolvla_film"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatchingFilm(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
