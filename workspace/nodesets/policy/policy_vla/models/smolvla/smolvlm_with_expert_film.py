# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""SmolVLM with Expert model — FiLM conditioning variant.

Uses Feature-wise Linear Modulation (FiLM) instead of cross-attention
to condition the action expert on VLM outputs.

For the original cross-attention variant, see smolvlm_with_expert.py.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

from workspace.nodesets.policy.policy_vla.models.smolvla.film import (
    FiLMLayer,
    create_aggregator,
    create_pooler,
)
from workspace.nodesets.policy.policy_vla.models.smolvla.smolvlm_with_expert import (
    apply_rope,
    get_intermediate_size,
)


class SmolVLMWithExpertFilmModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
        film_cond_dim: int = -1,
        film_after_self_attn: bool = True,
        film_after_ffn: bool = True,
        film_pool_mode: str = "mean",
        film_layer_agg: str = "last",
        film_layer_agg_n: int = 4,
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[
                :num_vlm_layers
            ]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(
            int(hidden_size * expert_width_multiplier)
        )
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)

        # FiLM modules (always enabled in this variant)
        cond_dim = film_cond_dim if film_cond_dim > 0 else lm_expert_config.hidden_size

        self.vlm_pooler = create_pooler(
            film_pool_mode,
            config.text_config.hidden_size,
            cond_dim,
        )
        self.vlm_aggregator = create_aggregator(
            film_layer_agg,
            self.num_vlm_layers,
            config.text_config.hidden_size,
            num_layers_agg=film_layer_agg_n,
        )

        num_expert = lm_expert_config.num_hidden_layers
        self.film_layers_attn = nn.ModuleList(
            [
                FiLMLayer(cond_dim, lm_expert_config.hidden_size) if film_after_self_attn else None
                for _ in range(num_expert)
            ]
        )
        self.film_layers_ffn = nn.ModuleList(
            [
                FiLMLayer(cond_dim, lm_expert_config.hidden_size) if film_after_ffn else None
                for _ in range(num_expert)
            ]
        )

        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = "self_attn"
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        film_cond: torch.Tensor | None = None,
        prefix_pad_mask: torch.Tensor | None = None,
        expert_attention_mask: torch.Tensor | None = None,
        expert_position_ids: torch.Tensor | None = None,
    ):
        prefix_embs = inputs_embeds[0]
        suffix_embs = inputs_embeds[1] if len(inputs_embeds) > 1 else None

        vlm_out = None
        # Phase 1: VLM forward (if prefix provided)
        if prefix_embs is not None:
            all_hidden = self._forward_vlm_only(
                prefix_embs,
                attention_mask,
                position_ids,
                use_cache,
                fill_kv_cache,
                past_key_values,
            )  # [B, N, L, H]
            agg_out = self.vlm_aggregator(all_hidden)  # [B, N', L, H] or [B, L, H]
            vlm_norm = self.get_vlm_model().text_model.norm
            agg_out = agg_out.to(dtype=vlm_norm.weight.dtype)
            agg_out = vlm_norm(agg_out)
            film_cond = self.vlm_pooler(agg_out, prefix_pad_mask)  # [B, cond_dim]
            vlm_out = all_hidden[:, -1]  # keep last layer for return value

        # Phase 2: Expert with FiLM (if suffix provided)
        expert_out = None
        if suffix_embs is not None and film_cond is not None:
            attn_mask = (
                expert_attention_mask if expert_attention_mask is not None else attention_mask
            )
            pos_ids = expert_position_ids if expert_position_ids is not None else position_ids
            expert_out = self._forward_expert_film(suffix_embs, attn_mask, pos_ids, film_cond)

        return [vlm_out, expert_out], past_key_values, film_cond

    def _forward_vlm_only(
        self,
        prefix_embs,
        attention_mask,
        position_ids,
        use_cache,
        fill_kv_cache,
        past_key_values,
    ):
        """Run VLM through all layers with self-attention only.

        Returns:
            [B, num_layers, L, H] stacked hidden states (no norm applied).
        """
        vlm_model = self.get_vlm_model().text_model
        vlm_layers = vlm_model.layers
        batch_size = prefix_embs.shape[0]
        head_dim = self.vlm.config.text_config.head_dim
        num_vlm = len(vlm_layers)

        all_hidden = []
        hidden = prefix_embs
        for layer_idx in range(num_vlm):
            layer = vlm_layers[layer_idx]
            # Self-attention
            normed = layer.input_layernorm(hidden)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            normed = normed.to(dtype=layer.self_attn.q_proj.weight.dtype)

            q = layer.self_attn.q_proj(normed).view(hidden_shape)
            k = layer.self_attn.k_proj(normed).view(hidden_shape)
            v = layer.self_attn.v_proj(normed).view(hidden_shape)

            q = apply_rope(q, position_ids)
            k = apply_rope(k, position_ids)

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(attention_mask, batch_size, head_dim, q, k, v)
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            out_emb = out_emb + hidden
            after_first_residual = out_emb.clone()

            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = out_emb.to(dtype=layer.mlp.gate_proj.weight.dtype)
            out_emb = layer.mlp(out_emb)
            out_emb = out_emb + after_first_residual

            hidden = out_emb
            all_hidden.append(hidden)

        return torch.stack(all_hidden, dim=1)  # [B, num_layers, L, H]

    def _forward_expert_film(self, suffix_embs, attention_mask, position_ids, film_cond):
        """Run Expert through all layers with self-attention + FiLM conditioning."""
        expert_model = self.lm_expert
        batch_size = suffix_embs.shape[0]
        head_dim = expert_model.config.head_dim

        # Expert attention head counts (may differ from VLM)
        expert_num_attention_heads = expert_model.config.num_attention_heads
        expert_num_key_value_heads = expert_model.config.num_key_value_heads

        hidden = suffix_embs
        for layer_idx in range(len(expert_model.layers)):
            layer = expert_model.layers[layer_idx]

            # 1. input_layernorm
            normed = layer.input_layernorm(hidden)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            normed = normed.to(dtype=layer.self_attn.q_proj.weight.dtype)

            # 2. Self-attention: q, k, v projections
            q = layer.self_attn.q_proj(normed).view(hidden_shape)
            k = layer.self_attn.k_proj(normed).view(hidden_shape)
            v = layer.self_attn.v_proj(normed).view(hidden_shape)

            # RoPE
            q = apply_rope(q, position_ids)
            k = apply_rope(k, position_ids)

            # Attention (using expert's head counts)
            att_output = self._expert_attention_forward(
                attention_mask,
                batch_size,
                head_dim,
                q,
                k,
                v,
                expert_num_attention_heads,
                expert_num_key_value_heads,
            )

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            attn_out = layer.self_attn.o_proj(att_output)

            # 3. FiLM after self-attn (before residual)
            if self.film_layers_attn[layer_idx] is not None:
                attn_out = self.film_layers_attn[layer_idx](film_cond, attn_out)

            # 4. Residual 1
            hidden = attn_out + hidden
            after_first_residual = hidden.clone()

            # 5. FFN
            ffn_out = layer.post_attention_layernorm(hidden)
            ffn_out = ffn_out.to(dtype=layer.mlp.gate_proj.weight.dtype)
            ffn_out = layer.mlp(ffn_out)

            # 6. FiLM after FFN (before residual)
            if self.film_layers_ffn[layer_idx] is not None:
                ffn_out = self.film_layers_ffn[layer_idx](film_cond, ffn_out)

            # 7. Residual 2
            hidden = ffn_out + after_first_residual

        hidden = hidden.to(dtype=expert_model.norm.weight.dtype)
        hidden = expert_model.norm(hidden)
        return hidden

    def _expert_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
        num_att_heads,
        num_key_value_heads,
    ):
        """Eager attention forward using expert's head counts."""
        num_key_value_groups = num_att_heads // num_key_value_heads
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output
