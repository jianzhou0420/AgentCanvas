from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.nn.modules import Module

from workspace.nodesets.policy.policy_adapter_vla.models.action_expert.film_layers import FiLMLayer

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# ---
# A simplified FiLM-enabled MLP block to replace a single transformer layer
# ---


class VectorizedParallelLinear(nn.Module):
    def __init__(self, num_layers, in_features, out_features):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features

        # 3D weight tensor with shape (num_layers, in_features, out_features)
        # Using nn.Parameter to make it trainable
        self.weight = nn.Parameter(torch.randn(num_layers, in_features, out_features))
        # 2D bias tensor with shape (num_layers, out_features)
        self.bias = nn.Parameter(torch.randn(num_layers, out_features))

    def forward(self, x):
        # Input x has shape (batch_size, num_layers, in_features)

        # Batch matrix multiplication using einsum
        # 'bni' = (batch_size, num_layers, in_features)
        # 'nio' = (num_layers, in_features, out_features)
        # 'bno' = output (batch_size, num_layers, out_features)
        output = torch.einsum("bni, nio -> bno", x, self.weight)

        # Add bias with broadcasting
        output = output + self.bias

        return output


class FiLMMLPBlock(Module):
    def __init__(
        self, horizon: int, d_model: int, dim_feedforward: int, cond_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        # First linear layer with GELU activation and dropout
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # FiLM layer to modulate the features before the second linear layer
        self.film = FiLMLayer(cond_dim, dim_feedforward, enable=True)

        # Second linear layer
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Residual connection
        residual = x

        # Pass through the first linear layer
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Apply FiLM modulation using the condition
        x = self.film(cond, x)

        # Pass through the second linear layer and add the residual
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.norm(x + residual)

        return x


# ---
# The main class, with the Transformer part replaced by a pure MLP stack
# ---


class MLPForDiffusion(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,  # This is kept for compatibility but not used
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        **kwargs: dict | None,  # Additional parameters for future compatibility
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # Input embedding stem, identical to the original
        if kwargs.get("parallel_input_emb", False):
            self.input_emb = VectorizedParallelLinear(
                num_layers=horizon, in_features=input_dim, out_features=n_emb
            )
        else:
            raise ValueError("parallel_input_emb must be provided in kwargs for MLPForDiffusion")
            self.input_emb = nn.Linear(input_dim, n_emb)

        self.drop = nn.Dropout(p_drop_emb)

        # Cond encoder, identical to the original
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_emb * 4),
            nn.Mish(),
            nn.Linear(n_emb * 4, n_emb),
        )

        # --- MLP STACK REPLACEMENT ---
        # The core of the change: replace the transformer decoder with a stack of MLPs.
        # The total dimension of the conditioning vector
        total_cond_dim = n_emb + (cond_dim * n_obs_steps)
        self.mlp_decoder = nn.Sequential(
            *[
                FiLMMLPBlock(
                    horizon=horizon,
                    d_model=n_emb,
                    dim_feedforward=4 * n_emb,
                    cond_dim=total_cond_dim,
                    dropout=p_drop_attn,
                )
                for _ in range(n_layer)
            ]
        )
        # --- END OF REPLACEMENT ---

        # The rest of the head is the same
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Store constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond

        # init
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        cond: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """

        # This part of the code remains the same as it handles conditioning preprocessing.
        cond_reshaped = cond.reshape(sample.shape[0], -1)

        # 0. process input
        # sample = rearrange(sample, 'b h d -> b (h d)')
        input_emb = self.input_emb(sample)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_feature = self.diffusion_step_encoder(timesteps)

        # 2. cond
        global_feature = torch.cat([time_feature, cond_reshaped], axis=-1)

        # 3. MLP Decoder
        x = self.drop(input_emb)
        # Pass the input through the MLP stack, providing the global_feature as the condition for each layer.
        for layer in self.mlp_decoder:
            x = layer(x, global_feature)

        # 4. head
        x = self.ln_f(x)
        x = self.head(x)

        return x

    # def _init_weights(self, module):
    #     # Initialisation logic is simplified to only handle the new MLP layers
    #     ignore_types = (nn.Dropout,
    #                     SinusoidalPosEmb,
    #                     FiLMMLPBlock,
    #                     FiLMLayer,
    #                     nn.ModuleList,
    #                     nn.Mish,
    #                     nn.GELU,
    #                     nn.Sequential)
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         torch.nn.init.zeros_(module.bias)
    #         torch.nn.init.ones_(module.weight)
    #     elif isinstance(module, ignore_types):
    #         pass
    #     else:
    #         raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        Separates all parameters into two buckets: those that will experience
        weight decay and those that won't (biases, layernorm/embedding weights,
        and FiLM modulation parameters).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, VectorizedParallelLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # Check for module types
                is_whitelist_module = isinstance(m, whitelist_weight_modules)
                is_blacklist_module = isinstance(m, blacklist_weight_modules)

                # --- NEW LOGIC TO HANDLE FILM PARAMETERS ---
                # Parameters within the FiLM layer (cond_encoder, gamma/beta heads)
                # are considered part of the modulation and typically don't get weight decay.
                # We check if the full parameter name contains '.film.' to catch them.
                if ".film." in fpn:
                    no_decay.add(fpn)
                    continue
                # --- END NEW LOGIC ---

                if pn.endswith("bias"):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias" (if it were present)
                    no_decay.add(fpn)
                elif pn.endswith("weight") and is_whitelist_module:
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and is_blacklist_module:
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter (if it existed)
        if hasattr(self, "pos_emb"):
            no_decay.add("pos_emb")
        if hasattr(self, "cond_pos_emb") and self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")
        no_decay.add("_dummy_variable")

        # validate that we considered every parameter
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

        # create the pytorch optimizer object
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
