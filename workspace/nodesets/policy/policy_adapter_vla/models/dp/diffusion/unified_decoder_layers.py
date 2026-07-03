"""
Unified decoder layer implementations for conditioned diffusion transformers.

All layers implement: forward(tgt, tgt_mask=None, cond=None) -> Tensor
where cond is a global conditioning vector (B, cond_dim).

Classes:
    FiLMDecoderLayer       - Affine modulation: gamma * x + beta
    AdaLNZeroDecoderLayer  - DiT-style adaptive LayerNorm + zero-initialized gating
    AdaLNDecoderLayer      - Adaptive LayerNorm without zero-init gating
    AdaRMSNormDecoderLayer - Adaptive RMSNorm without zero-init gating
    LoRACondDecoderLayer   - Low-rank conditioned Q/V bias injection
    AdditiveCondDecoderLayer - Simple projected bias addition
    CondTransformerDecoder - Generic stack of N conditioned layers
"""

import copy
from typing import Optional, Callable, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class FiLMDecoderLayer(nn.Module):
    """Transformer decoder layer with FiLM conditioning (gamma * x + beta)."""

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # FiLM: cond -> (gamma, beta) for self-attn and ffn
        self.film_sa = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d_model * 2))
        self.film_ffn = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d_model * 2))

    def _film_modulate(self, film_layer, cond, x):
        embed = film_layer(cond)  # (B, d_model*2)
        embed = embed.unsqueeze(1)  # (B, 1, d_model*2)
        gamma, beta = embed.chunk(2, dim=-1)
        return x * gamma + beta

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            sa_out = self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._film_modulate(self.film_sa, cond, sa_out)
            ff_out = self._ff_block(self.norm2(x))
            x = x + self._film_modulate(self.film_ffn, cond, ff_out)
        else:
            sa_out = self._sa_block(x, tgt_mask)
            x = self.norm1(x + self._film_modulate(self.film_sa, cond, sa_out))
            ff_out = self._ff_block(x)
            x = self.norm2(x + self._film_modulate(self.film_ffn, cond, ff_out))
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdaLNZeroDecoderLayer(nn.Module):
    """DiT-style adaptive LayerNorm with zero-initialized gating.

    Produces 6 modulation vectors from cond: (gamma1, beta1, alpha1, gamma2, beta2, alpha2).
    LayerNorm uses elementwise_affine=False; adaptive params replace learnable ones.
    adaLN_modulation output Linear is zero-initialized -> alpha=0 at init -> identity blocks.
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        # No learnable affine params - adaptive params replace them
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # AdaLN: cond -> (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 6 * d_model)
        )

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        mod = self.adaLN_modulation(cond)  # (B, 6*d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.unsqueeze(1).chunk(6, dim=-1)

        x = tgt
        # Self-attention with adaptive LN + gating
        x_norm = (1 + gamma1) * self.norm1(x) + beta1
        sa_out = self._sa_block(x_norm, tgt_mask)
        x = x + alpha1 * sa_out

        # FFN with adaptive LN + gating
        x_norm = (1 + gamma2) * self.norm2(x) + beta2
        ff_out = self._ff_block(x_norm)
        x = x + alpha2 * ff_out

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdaLNDecoderLayer(nn.Module):
    """Adaptive LayerNorm WITHOUT zero-initialized gating.

    Like AdaLNZeroDecoderLayer but:
    - Produces 4 modulation vectors (gamma1, beta1, gamma2, beta2) instead of 6 (no alpha gates).
    - No zero-init on adaLN_modulation — uses standard N(0, 0.02) initialization.
    - Sublayer outputs are added directly via standard residual (no gating).

    This isolates the contribution of the zero-init gating strategy by comparing:
    - adaln_zero: 6 vectors, zero-init → identity at init, gradual activation
    - adaln: 4 vectors, standard init → active conditioning from step 1
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        # No learnable affine params - adaptive params replace them
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # AdaLN: cond -> (gamma1, beta1, gamma2, beta2) — no alpha gates
        self.adaLN_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 4 * d_model)
        )

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        mod = self.adaLN_modulation(cond)  # (B, 4*d_model)
        gamma1, beta1, gamma2, beta2 = mod.unsqueeze(1).chunk(4, dim=-1)

        x = tgt
        # Self-attention with adaptive LN (no gating)
        x_norm = (1 + gamma1) * self.norm1(x) + beta1
        sa_out = self._sa_block(x_norm, tgt_mask)
        x = x + sa_out

        # FFN with adaptive LN (no gating)
        x_norm = (1 + gamma2) * self.norm2(x) + beta2
        ff_out = self._ff_block(x_norm)
        x = x + ff_out

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdaRMSNormDecoderLayer(nn.Module):
    """Adaptive RMSNorm WITHOUT zero-initialized gating.

    Like AdaLNDecoderLayer but replaces LayerNorm with RMSNorm:
    - Produces 4 modulation vectors (gamma1, beta1, gamma2, beta2) instead of 6 (no alpha gates).
    - No zero-init on adaRMSNorm_modulation — uses standard N(0, 0.02) initialization.
    - Sublayer outputs are added directly via standard residual (no gating).
    - Formula: (1 + gamma) * RMSNorm(x) + beta

    This isolates the contribution of RMSNorm vs LayerNorm by comparing:
    - adaln: 4 vectors, LayerNorm, standard init
    - ada_rms_norm: 4 vectors, RMSNorm, standard init
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        # No learnable affine params - adaptive params replace them
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # AdaRMSNorm: cond -> (gamma1, beta1, gamma2, beta2) — no alpha gates
        self.adaRMSNorm_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 4 * d_model)
        )

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        mod = self.adaRMSNorm_modulation(cond)  # (B, 4*d_model)
        gamma1, beta1, gamma2, beta2 = mod.unsqueeze(1).chunk(4, dim=-1)

        x = tgt
        # Self-attention with adaptive RMSNorm (no gating)
        x_norm = (1 + gamma1) * self.norm1(x) + beta1
        sa_out = self._sa_block(x_norm, tgt_mask)
        x = x + sa_out

        # FFN with adaptive RMSNorm (no gating)
        x_norm = (1 + gamma2) * self.norm2(x) + beta2
        ff_out = self._ff_block(x_norm)
        x = x + ff_out

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class LoRACondDecoderLayer(nn.Module):
    """Low-rank conditioned Q/V bias injection.

    cond -> rank*2 -> split -> up_q/up_v -> d_model biases added to Q and V.
    lora_up_q, lora_up_v are zero-initialized -> no bias at init.
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True, lora_rank: int = 16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # LoRA: cond -> low-rank -> Q bias, V bias
        self.lora_down = nn.Linear(cond_dim, lora_rank * 2)
        self.lora_up_q = nn.Linear(lora_rank, d_model)
        self.lora_up_v = nn.Linear(lora_rank, d_model)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        x = tgt

        # Compute Q/V biases from condition
        lr = self.lora_down(cond)  # (B, rank*2)
        lr_q, lr_v = lr.chunk(2, dim=-1)  # (B, rank) each
        q_bias = self.lora_up_q(lr_q).unsqueeze(1)  # (B, 1, d_model)
        v_bias = self.lora_up_v(lr_v).unsqueeze(1)  # (B, 1, d_model)

        if self.norm_first:
            x_norm = self.norm1(x)
            sa_out = self._sa_block_biased(x_norm, q_bias, v_bias, tgt_mask)
            x = x + sa_out
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_out = self._sa_block_biased(x, q_bias, v_bias, tgt_mask)
            x = self.norm1(x + sa_out)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block_biased(self, x, q_bias, v_bias, attn_mask):
        q = x + q_bias
        k = x
        v = x + v_bias
        x = self.self_attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdditiveCondDecoderLayer(nn.Module):
    """Simple projected bias addition after sublayers.

    proj output Linears are zero-initialized -> no bias at init.
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable] = 'gelu',
                 batch_first: bool = True, norm_first: bool = True,
                 bias: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # Additive: cond -> bias after self-attn and FFN
        self.cond_proj_sa = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d_model))
        self.cond_proj_ff = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d_model))

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        sa_bias = self.cond_proj_sa(cond).unsqueeze(1)  # (B, 1, d_model)
        ff_bias = self.cond_proj_ff(cond).unsqueeze(1)  # (B, 1, d_model)

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask) + sa_bias
            x = x + self._ff_block(self.norm2(x)) + ff_bias
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask) + sa_bias)
            x = self.norm2(x + self._ff_block(x) + ff_bias)
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CondTransformerDecoder(nn.Module):
    """Generic stack of N conditioned decoder layers.

    Passes cond through to each layer's forward method.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask=tgt_mask, cond=cond)
        if self.norm is not None:
            output = self.norm(output)
        return output
