"""
Unified Transformer backbone for diffusion with configurable conditioning method.

Supports 8 conditioning methods via `cond_method` parameter:
  - cross_attn:    Encoder-decoder cross-attention (Q=actions, KV=obs)
  - prefix:        Obs tokens prepended to action sequence, self-attend
  - film:          FiLM affine modulation (gamma*x + beta)
  - adaln_zero:    DiT-style adaptive LayerNorm + zero-initialized gating
  - adaln:         Adaptive LayerNorm without zero-init gating
  - ada_rms_norm:  Adaptive RMSNorm without zero-init gating
  - lora_cond:     Low-rank conditioned Q/V bias injection
  - additive:      Simple projected bias addition after sublayers
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from workspace.nodesets.policy.policy_adapter_vla.models.dp.common.module_attr_mixin import ModuleAttrMixin
from workspace.nodesets.policy.policy_adapter_vla.models.dp.diffusion.positional_embedding import (
    SinusoidalPosEmb,
)
from workspace.nodesets.policy.policy_adapter_vla.models.dp.diffusion.unified_decoder_layers import (
    AdaLNDecoderLayer,
    AdaLNZeroDecoderLayer,
    AdaRMSNormDecoderLayer,
    AdditiveCondDecoderLayer,
    CondTransformerDecoder,
    FiLMDecoderLayer,
    LoRACondDecoderLayer,
)

logger = logging.getLogger(__name__)

LAYER_MAP = {
    "film": FiLMDecoderLayer,
    "adaln_zero": AdaLNZeroDecoderLayer,
    "adaln": AdaLNDecoderLayer,
    "ada_rms_norm": AdaRMSNormDecoderLayer,
    "lora_cond": LoRACondDecoderLayer,
    "additive": AdditiveCondDecoderLayer,
}

VALID_COND_METHODS = {
    "cross_attn",
    "prefix",
    "film",
    "adaln_zero",
    "adaln",
    "ada_rms_norm",
    "lora_cond",
    "additive",
}


class TransformerForDiffusionUnified(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        cond_method: str = "cross_attn",
        lora_rank: int = 16,
    ) -> None:
        super().__init__()

        assert cond_method in VALID_COND_METHODS, f"Unknown cond_method: {cond_method}"
        self.cond_method = cond_method

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1  # time token
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # Input embedding (shared)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Method-specific modules
        self.time_emb = None
        self.diffusion_step_encoder = None
        self.cond_obs_emb = None
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None

        if cond_method in ("cross_attn", "prefix"):
            # Token-based: time as raw sinusoidal token
            self.time_emb = SinusoidalPosEmb(n_emb)

            if obs_as_cond:
                self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            # Condition encoder
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
                )

            if cond_method == "cross_attn":
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerDecoder(
                    decoder_layer=decoder_layer, num_layers=n_layer
                )
            else:  # prefix
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_layer
                )

        else:  # Global-vector methods: film, adaln_zero, lora_cond, additive
            self.diffusion_step_encoder = nn.Sequential(
                SinusoidalPosEmb(n_emb),
                nn.Linear(n_emb, n_emb * 4),
                nn.Mish(),
                nn.Linear(n_emb * 4, n_emb),
            )
            global_cond_dim = cond_dim * n_obs_steps + n_emb

            layer_cls = LAYER_MAP[cond_method]
            layer_kwargs = dict(
                d_model=n_emb,
                nhead=n_head,
                cond_dim=global_cond_dim,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            if cond_method == "lora_cond":
                layer_kwargs["lora_rank"] = lora_rank

            decoder_layer = layer_cls(**layer_kwargs)
            self.decoder = CondTransformerDecoder(decoder_layer, n_layer)

        # Attention masks
        if causal_attn:
            if cond_method == "prefix":
                # Prefix mask: prefix sees all prefix, not actions;
                # actions see all prefix + causal among actions
                total = T_cond + T
                mask = torch.zeros(total, total)
                mask[:T_cond, T_cond:] = float("-inf")
                causal_block = torch.triu(torch.ones(T, T), diagonal=1).bool()
                mask[T_cond:, T_cond:].masked_fill_(causal_block, float("-inf"))
                self.register_buffer("mask", mask)
                self.memory_mask = None
            elif cond_method == "cross_attn":
                # Causal mask for action sequence
                sz = T
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                mask = (
                    mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
                )
                self.register_buffer("mask", mask)

                # Memory mask (cross-attn)
                if time_as_cond and obs_as_cond:
                    S = T_cond
                    t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
                    mem_mask = t >= (s - 1)
                    mem_mask = (
                        mem_mask.float()
                        .masked_fill(mem_mask == 0, float("-inf"))
                        .masked_fill(mem_mask == 1, 0.0)
                    )
                    self.register_buffer("memory_mask", mem_mask)
                else:
                    self.memory_mask = None
            else:
                # Global vector methods: causal self-attn mask on action tokens
                sz = T
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                mask = (
                    mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
                )
                self.register_buffer("mask", mask)
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # Head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond

        # Initialize weights
        self.apply(self._init_weights)

        # Post-init zero-initialization for identity-at-init behavior
        self._zero_init_cond_params()

        logger.info(
            "number of parameters: %e [cond_method=%s]",
            sum(p.numel() for p in self.parameters()),
            cond_method,
        )

    def _zero_init_cond_params(self):
        """Zero-initialize output projections so conditioning has no effect at init."""
        if self.cond_method == "adaln_zero":
            for layer in self.decoder.layers:
                nn.init.zeros_(layer.adaLN_modulation[-1].weight)
                nn.init.zeros_(layer.adaLN_modulation[-1].bias)
        elif self.cond_method == "lora_cond":
            for layer in self.decoder.layers:
                nn.init.zeros_(layer.lora_up_q.weight)
                nn.init.zeros_(layer.lora_up_v.weight)
        elif self.cond_method == "additive":
            for layer in self.decoder.layers:
                nn.init.zeros_(layer.cond_proj_sa[-1].weight)
                nn.init.zeros_(layer.cond_proj_sa[-1].bias)
                nn.init.zeros_(layer.cond_proj_ff[-1].weight)
                nn.init.zeros_(layer.cond_proj_ff[-1].bias)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            FiLMDecoderLayer,
            AdaLNZeroDecoderLayer,
            AdaLNDecoderLayer,
            AdaRMSNormDecoderLayer,
            LoRACondDecoderLayer,
            AdditiveCondDecoderLayer,
            CondTransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            # Handle elementwise_affine=False (AdaLN-Zero case)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            # Handle elementwise_affine=False (AdaRMSNorm case)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusionUnified):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Separate params into decay/no_decay groups.

        Conditioning-specific params (FiLM, AdaLN, LoRA, Additive projections)
        are placed in the no_decay group.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Conditioning-specific param name patterns -> no weight decay
        cond_patterns = (
            ".adaLN_modulation.",
            ".adaRMSNorm_modulation.",
            ".lora_",
            ".cond_proj_",
            ".film_",
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if any(pat in fpn for pat in cond_patterns) or pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Special cases
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # Validate
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        cond: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T', cond_dim)
        output: (B, T, input_dim)
        """
        # Process timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Input embedding
        input_emb = self.input_emb(sample)

        if self.cond_method == "cross_attn":
            return self._forward_cross_attn(input_emb, timesteps, cond)
        elif self.cond_method == "prefix":
            return self._forward_prefix(input_emb, timesteps, cond)
        else:
            return self._forward_global_vector(input_emb, timesteps, cond)

    def _forward_cross_attn(self, input_emb, timesteps, cond):
        # Time token
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # Build condition tokens
        cond_embeddings = time_emb
        if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(cond)  # (B, To, n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_embeddings + position_embeddings)
        memory = self.encoder(x)  # (B, T_cond, n_emb)

        # Decoder
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + position_embeddings)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)
        return x

    def _forward_prefix(self, input_emb, timesteps, cond):
        # Time token
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # Build prefix tokens
        cond_embeddings = time_emb
        if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(cond)  # (B, To, n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        prefix = self.drop(cond_embeddings + position_embeddings)
        prefix = self.encoder(prefix)  # (B, T_cond, n_emb)

        # Action tokens
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        action_tokens = self.drop(input_emb + position_embeddings)

        # Concatenate [prefix, action_tokens] and self-attend
        combined = torch.cat([prefix, action_tokens], dim=1)  # (B, T_cond+T, n_emb)
        combined = self.decoder(src=combined, mask=self.mask)

        # Strip prefix, keep only action tokens
        x = combined[:, tc:, :]  # (B, T, n_emb)

        x = self.ln_f(x)
        x = self.head(x)
        return x

    def _forward_global_vector(self, input_emb, timesteps, cond):
        # Flatten obs to global vector
        cond_flat = cond.reshape(cond.shape[0], -1)  # (B, cond_dim * n_obs_steps)

        # Time embedding via MLP
        time_feature = self.diffusion_step_encoder(timesteps)  # (B, n_emb)

        # Global conditioning vector
        global_feature = torch.cat([time_feature, cond_flat], dim=-1)

        # Decoder with conditioning
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + position_embeddings)
        x = self.decoder(tgt=x, tgt_mask=self.mask, cond=global_feature)

        x = self.ln_f(x)
        x = self.head(x)
        return x
