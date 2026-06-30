"""FiLM (Feature-wise Linear Modulation) conditioning modules for SmolVLA.

Provides FiLMLayer for affine modulation and VLMPooler for pooling VLM outputs
into a conditioning vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_pooler_input(vlm_output, pad_mask):
    """Reshape 4D aggregator output for sequence-level poolers.

    If vlm_output is [B, N, L, H] (multiple layers kept), flatten to [B, N*L, H]
    and expand pad_mask accordingly. If already [B, L, H], pass through unchanged.
    """
    if vlm_output.ndim == 4:
        B, N, L, H = vlm_output.shape
        vlm_output = vlm_output.reshape(B, N * L, H)
        pad_mask = pad_mask.unsqueeze(1).expand(B, N, L).reshape(B, N * L)
    return vlm_output, pad_mask


class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation: x * (1 + gamma) + beta.

    Zero-initialized so that the initial output is the identity: x * (1+0) + 0 = x.

    Args:
        cond_dim: Dimension of the conditioning vector.
        out_dim: Dimension of the features to modulate.
    """

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_dim * 2),
        )
        # Zero-init so FiLM starts as identity
        nn.init.zeros_(self.cond_encoder[1].weight)
        nn.init.zeros_(self.cond_encoder[1].bias)

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: [B, cond_dim] conditioning vector.
            x: [B, T, out_dim] input features.

        Returns:
            [B, T, out_dim] modulated features.
        """
        cond = cond.to(dtype=self.cond_encoder[1].weight.dtype)
        embed = self.cond_encoder(cond)  # [B, out_dim * 2]
        gamma, beta = embed.reshape(embed.shape[0], 2, 1, self.out_dim).unbind(dim=1)
        # gamma: [B, 1, out_dim], beta: [B, 1, out_dim]
        return x * (1 + gamma) + beta


class AttentionPooler(nn.Module):
    """Pools VLM hidden states via learnable query cross-attention.

    K learnable queries attend to the full VLM output sequence, then the
    resulting K vectors are concatenated and projected to cond_dim.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
        num_queries: Number of learnable queries.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        vlm_hidden_size: int,
        cond_dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = vlm_hidden_size // num_heads
        assert vlm_hidden_size % num_heads == 0, (
            f"vlm_hidden_size ({vlm_hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.queries = nn.Parameter(torch.randn(num_queries, vlm_hidden_size) * 0.02)
        self.q_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.k_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.v_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.out_proj = nn.Linear(num_queries * vlm_hidden_size, cond_dim)

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        B, L, H = vlm_output.shape
        dtype = self.q_proj.weight.dtype
        vlm_output = vlm_output.to(dtype=dtype)

        # Expand learnable queries: [num_queries, H] -> [B, num_queries, H]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Project Q, K, V
        q = self.q_proj(q)  # [B, num_queries, H]
        k = self.k_proj(vlm_output)  # [B, L, H]
        v = self.v_proj(vlm_output)  # [B, L, H]

        # Reshape to multi-head: [B, num_heads, seq_len, head_dim]
        q = q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with mask: pad_mask [B, L] -> [B, 1, 1, L]
        attn_mask = ~pad_mask[:, None, None, :]  # True = masked out
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Merge heads: [B, num_heads, num_queries, head_dim] -> [B, num_queries, H]
        attn = attn.transpose(1, 2).reshape(B, self.num_queries, -1)

        # Concat queries and project: [B, num_queries * H] -> [B, cond_dim]
        pooled = attn.reshape(B, -1)
        return self.out_proj(pooled)


class MaxPooler(nn.Module):
    """Pools VLM hidden states via element-wise max over valid tokens.

    Captures the strongest activation per feature dimension.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(vlm_hidden_size, cond_dim) if cond_dim != vlm_hidden_size else nn.Identity()

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        # Fill masked positions with -inf so they don't affect max
        mask = pad_mask.unsqueeze(-1)  # [B, L, 1]
        filled = vlm_output.masked_fill(~mask, float("-inf"))
        pooled = filled.max(dim=1).values  # [B, H]

        if isinstance(self.proj, nn.Linear):
            pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


class MeanMaxPooler(nn.Module):
    """Pools VLM hidden states by concatenating mean and max pooling, then projecting.

    Captures both average signal and salient activations.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(vlm_hidden_size * 2, cond_dim)

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        mask = pad_mask.unsqueeze(-1)  # [B, L, 1]
        mask_float = mask.to(vlm_output.dtype)

        # Mean pooling
        summed = (vlm_output * mask_float).sum(dim=1)  # [B, H]
        count = mask_float.sum(dim=1).clamp(min=1)  # [B, 1]
        mean_pooled = summed / count

        # Max pooling
        filled = vlm_output.masked_fill(~mask, float("-inf"))
        max_pooled = filled.max(dim=1).values  # [B, H]

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # [B, 2H]
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


class WeightedMeanPooler(nn.Module):
    """Pools VLM hidden states via learned per-token scalar weights.

    A lightweight alternative to full attention pooling: learns a scalar
    importance weight per token, then computes a weighted sum.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int):
        super().__init__()
        self.score_proj = nn.Linear(vlm_hidden_size, 1)
        self.proj = nn.Linear(vlm_hidden_size, cond_dim) if cond_dim != vlm_hidden_size else nn.Identity()

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        dtype = self.score_proj.weight.dtype
        vlm_output = vlm_output.to(dtype=dtype)

        # Compute per-token scores: [B, L, 1]
        scores = self.score_proj(vlm_output)  # [B, L, 1]

        # Mask padding with -inf before softmax
        scores = scores.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)  # [B, L, 1]

        # Weighted sum
        pooled = (vlm_output * weights).sum(dim=1)  # [B, H]

        if isinstance(self.proj, nn.Linear):
            pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


class PerModalityPooler(nn.Module):
    """Pools image, language, and state tokens separately, then concatenates and projects.

    Prevents image patches from drowning out language/state signals by giving
    each modality its own pooling.

    Requires a modality_mask tensor indicating token types:
        0 = image, 1 = language, 2 = state, -1 = padding

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
        num_modalities: Number of modality types (default: 3 for image/language/state).
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        self.proj = nn.Linear(vlm_hidden_size * num_modalities, cond_dim)

    def forward(
        self,
        vlm_output: torch.Tensor,
        pad_mask: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).
            modality_mask: [B, L] int tensor (0=image, 1=language, 2=state).
                           Required — raises ValueError if not provided.

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        if modality_mask is None:
            raise ValueError("PerModalityPooler requires modality_mask (0=image, 1=language, 2=state)")

        pools = []
        for mod_id in range(self.num_modalities):
            # Mask for this modality AND valid tokens
            mod_mask = (modality_mask == mod_id) & pad_mask  # [B, L]
            mod_mask_float = mod_mask.unsqueeze(-1).to(vlm_output.dtype)  # [B, L, 1]
            summed = (vlm_output * mod_mask_float).sum(dim=1)  # [B, H]
            count = mod_mask_float.sum(dim=1).clamp(min=1)  # [B, 1]
            pools.append(summed / count)

        pooled = torch.cat(pools, dim=-1)  # [B, num_modalities * H]
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


class GeMPooler(nn.Module):
    """Generalized Mean (GeM) pooling with a learnable power parameter.

    Computes (mean(x^p))^(1/p) over valid tokens. Interpolates between
    mean pooling (p=1) and max pooling (p->inf). Popular in image retrieval.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
        init_p: Initial value of the learnable power parameter (default: 3.0).
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int, init_p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.proj = nn.Linear(vlm_hidden_size, cond_dim) if cond_dim != vlm_hidden_size else nn.Identity()

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        # Clamp p to avoid numerical issues
        p = self.p.clamp(min=1.0)

        mask = pad_mask.unsqueeze(-1).to(vlm_output.dtype)  # [B, L, 1]

        # Clamp to small positive value before raising to power p
        x = vlm_output.clamp(min=1e-6)
        x_p = x.pow(p) * mask  # [B, L, H]
        summed = x_p.sum(dim=1)  # [B, H]
        count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = (summed / count).pow(1.0 / p)  # [B, H]

        if isinstance(self.proj, nn.Linear):
            pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


class VLMPooler(nn.Module):
    """Pools VLM hidden states into a fixed-size conditioning vector.

    Args:
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.
        pool_mode: Pooling strategy, "mean" or "last".
    """

    def __init__(self, vlm_hidden_size: int, cond_dim: int, pool_mode: str = "mean"):
        super().__init__()
        self.pool_mode = pool_mode
        self.proj = nn.Linear(vlm_hidden_size, cond_dim) if cond_dim != vlm_hidden_size else nn.Identity()

    def forward(self, vlm_output: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_output: [B, L, vlm_hidden_size] or [B, N, L, H] VLM hidden states.
            pad_mask: [B, L] boolean mask (True = valid token).

        Returns:
            [B, cond_dim] pooled conditioning vector.
        """
        vlm_output, pad_mask = _prepare_pooler_input(vlm_output, pad_mask)
        if self.pool_mode == "mean":
            # Masked mean pooling
            mask = pad_mask.unsqueeze(-1).to(vlm_output.dtype)  # [B, L, 1]
            summed = (vlm_output * mask).sum(dim=1)  # [B, H]
            count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            pooled = summed / count
        elif self.pool_mode == "last":
            # Take the last valid token per sequence
            # Find index of last True in pad_mask
            lengths = pad_mask.sum(dim=1).long() - 1  # [B]
            lengths = lengths.clamp(min=0)
            pooled = vlm_output[torch.arange(vlm_output.shape[0], device=vlm_output.device), lengths]
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        if isinstance(self.proj, nn.Linear):
            pooled = pooled.to(dtype=self.proj.weight.dtype)
        return self.proj(pooled)


def create_pooler(
    pool_mode: str,
    vlm_hidden_size: int,
    cond_dim: int,
) -> nn.Module:
    """Factory to create the appropriate VLM pooler from config strings.

    Args:
        pool_mode: Sequence pooling strategy — "mean", "last", "attention",
                   "max", "mean_max", "weighted_mean", "gem".
        vlm_hidden_size: Hidden dimension of the VLM.
        cond_dim: Output conditioning dimension.

    Returns:
        A pooler module.
    """
    if pool_mode in ("mean", "last"):
        return VLMPooler(vlm_hidden_size, cond_dim, pool_mode=pool_mode)
    elif pool_mode == "attention":
        return AttentionPooler(vlm_hidden_size, cond_dim)
    elif pool_mode == "max":
        return MaxPooler(vlm_hidden_size, cond_dim)
    elif pool_mode == "mean_max":
        return MeanMaxPooler(vlm_hidden_size, cond_dim)
    elif pool_mode == "weighted_mean":
        return WeightedMeanPooler(vlm_hidden_size, cond_dim)
    elif pool_mode == "gem":
        return GeMPooler(vlm_hidden_size, cond_dim)
    else:
        raise ValueError(
            f"Unknown pool_mode: '{pool_mode}'. "
            f"Valid options: mean, last, attention, max, mean_max, weighted_mean, gem"
        )


# ---------------------------------------------------------------------------
# Layer aggregators: reduce [B, N, L, H] to [B, L, H] or keep [B, N', L, H]
# ---------------------------------------------------------------------------


class LastLayerAggregator(nn.Module):
    """Takes only the last VLM layer output."""

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, N, L, H]
        Returns:
            [B, L, H]
        """
        return all_hidden[:, -1]


class IdentityAggregator(nn.Module):
    """Pass-through — keeps all layers."""

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, N, L, H]
        Returns:
            [B, N, L, H]
        """
        return all_hidden


class LastNAggregator(nn.Module):
    """Keeps the last N layers.

    Args:
        n: Number of layers to keep.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, N, L, H]
        Returns:
            [B, n, L, H]
        """
        return all_hidden[:, -self.n:]


class WeightedSumAggregator(nn.Module):
    """Learnable scalar weight per VLM layer, softmax-normalized weighted sum.

    Args:
        num_layers: Number of VLM layers to aggregate.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, N, L, H]
        Returns:
            [B, L, H]
        """
        weights = F.softmax(self.layer_logits, dim=0)  # [N]
        # weights[None, :, None, None] -> [1, N, 1, 1]
        return (weights[None, :, None, None] * all_hidden).sum(dim=1)


class ConcatAggregator(nn.Module):
    """Concatenate the last N layers along hidden dim, then project back to H.

    Args:
        n: Number of layers to concatenate.
        vlm_hidden_size: Hidden dimension of each layer.
    """

    def __init__(self, n: int, vlm_hidden_size: int):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(n * vlm_hidden_size, vlm_hidden_size)

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, N, L, H] — last n layers are used.
        Returns:
            [B, L, H]
        """
        last_n = all_hidden[:, -self.n:]  # [B, n, L, H]
        B, n, L, H = last_n.shape
        x = last_n.permute(0, 2, 1, 3).reshape(B, L, n * H)
        return self.proj(x)


class LayerAttentionAggregator(nn.Module):
    """Learnable query cross-attends over the layer dimension at each sequence position.

    A single learnable query vector attends to all layers independently per
    (batch, sequence-position), producing a single [B, L, H] output.

    Args:
        num_layers: Number of VLM layers.
        vlm_hidden_size: Hidden dimension of each layer.
        num_heads: Number of attention heads.
    """

    def __init__(self, num_layers: int, vlm_hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vlm_hidden_size // num_heads
        assert vlm_hidden_size % num_heads == 0

        self.query = nn.Parameter(torch.randn(vlm_hidden_size) * 0.02)
        self.q_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.k_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.v_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.out_proj = nn.Linear(vlm_hidden_size, vlm_hidden_size)

    def forward(self, all_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden: [B, num_layers, L, H]
        Returns:
            [B, L, H]
        """
        B, num_layers, L, H = all_hidden.shape
        nH, dH = self.num_heads, self.head_dim

        # Reshape to (B*L, num_layers, H) so attention operates over the layer dim
        kv_input = all_hidden.permute(0, 2, 1, 3).reshape(B * L, num_layers, H)

        # Query: [H] -> [B*L, 1, H]
        q = self.query.unsqueeze(0).unsqueeze(0).expand(B * L, 1, H)
        q = self.q_proj(q)  # [B*L, 1, H]
        k = self.k_proj(kv_input)  # [B*L, num_layers, H]
        v = self.v_proj(kv_input)  # [B*L, num_layers, H]

        # Multi-head reshape: [B*L, seq, nH, dH] -> [B*L, nH, seq, dH]
        q = q.view(B * L, 1, nH, dH).transpose(1, 2)
        k = k.view(B * L, num_layers, nH, dH).transpose(1, 2)
        v = v.view(B * L, num_layers, nH, dH).transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # [B*L, nH, 1, dH]

        # Merge heads: [B*L, 1, H]
        attn = attn.transpose(1, 2).reshape(B * L, 1, H)
        out = self.out_proj(attn).squeeze(1)  # [B*L, H]
        return out.view(B, L, H)


def create_aggregator(
    agg_mode: str,
    num_layers: int,
    vlm_hidden_size: int,
    num_layers_agg: int = 4,
) -> nn.Module:
    """Factory to create a layer aggregator from a mode string.

    Args:
        agg_mode: One of "last", "all", "last_n", "weighted_sum", "concat",
                  "all_concat", "layer_attention".
        num_layers: Total number of VLM layers.
        vlm_hidden_size: Hidden dimension of VLM layers.
        num_layers_agg: Number of layers for "last_n" and "concat" (ignored otherwise).

    Returns:
        An aggregator module.
    """
    if agg_mode == "last":
        return LastLayerAggregator()
    elif agg_mode == "all":
        return IdentityAggregator()
    elif agg_mode == "last_n":
        return LastNAggregator(num_layers_agg)
    elif agg_mode == "weighted_sum":
        return WeightedSumAggregator(num_layers)
    elif agg_mode == "concat":
        return ConcatAggregator(num_layers_agg, vlm_hidden_size)
    elif agg_mode == "all_concat":
        return ConcatAggregator(num_layers, vlm_hidden_size)
    elif agg_mode == "layer_attention":
        return LayerAttentionAggregator(num_layers, vlm_hidden_size)
    else:
        raise ValueError(
            f"Unknown agg_mode: '{agg_mode}'. "
            f"Valid options: last, all, last_n, weighted_sum, concat, all_concat, layer_attention"
        )
