from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import workspace.nodesets.policy.policy_adapter_vla.models.openpi.models.gemma as _gemma
import workspace.nodesets.policy.policy_adapter_vla.models.openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from workspace.nodesets.policy.policy_adapter_vla.models.openpi.models_pytorch.gemma_pytorch import (
    PaliGemmaWithExpertModel,
)

"""
================================================================================
                         π0 / π0.5 Architecture
================================================================================

This file implements the π0 and π0.5 vision-language-action (VLA) models.

See /docs/pi0/pi0_looklike.md for complete architecture diagrams.

================================================================================
                    Overview: Dual-Stream Architecture
================================================================================

Both π0 and π0.5 use a **dual-stream** architecture with **synchronized
layer-by-layer** computation:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PaliGemmaWithExpertModel                             │
│                  (SYNCHRONIZED LAYER-BY-LAYER COMPUTATION)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PREFIX (VLM Stream)                    SUFFIX (Action Expert Stream)      │
│   paligemma.language_model               gemma_expert.model                 │
│                                                                             │
│   ┌─────────────────────┐                ┌─────────────────────┐            │
│   │  VLM Layer 0        │◄══ shared ════►│  Expert Layer 0     │            │
│   └─────────────────────┘    attention   └─────────────────────┘            │
│            │                                       │                        │
│            ▼                                       ▼                        │
│          ...                                     ...                        │
│            │                                       │                        │
│            ▼                                       ▼                        │
│       prefix_out                            suffix_out                      │
│       (discarded)                           (used for action)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

NOT sequential:  ❌ VLM Layer 0→1→...→N, THEN Expert Layer 0→1→...→N
BUT synchronized: ✓ (VLM Layer 0 + Expert Layer 0) → (VLM Layer 1 + Expert Layer 1) → ...

================================================================================
                    π0 vs π0.5 Comparison
================================================================================

| Feature             | π0                      | π0.5                    |
|---------------------|-------------------------|-------------------------|
| State Input         | Yes (state_proj)        | No                      |
| Suffix Tokens       | [state, action⊕time]    | [action] only           |
| Time in Input       | concat([action, time])  | action only             |
|                     | → MLP fusion            |                         |
| Time for AdaRMS     | Not used (None)         | time_mlp(time_emb)      |
| Action Expert AdaRMS| use_adarms=False        | use_adarms=True         |
| Residual Connection | x + y (standard)        | x + y * gate (gated)    |
| Suffix Shape        | [B, 1+H, 1024]          | [B, H, 1024]            |

================================================================================
                    PaliGemma Architecture Diagram
================================================================================

Q: Is PaliGemma a Decoder-Only Transformer?
A: YES, but with a twist!

PaliGemma = SigLIP Vision Encoder + Gemma Decoder-Only LLM

The language model (Gemma) is decoder-only, but the ATTENTION PATTERN is
"Prefix-LM" (not pure causal):
  - PREFIX tokens (images + language): BIDIRECTIONAL attention
  - SUFFIX tokens (actions): CAUSAL attention

This is different from:
  - Pure Decoder-Only (GPT): All tokens use causal attention
  - Encoder-Decoder (T5): Separate encoder (bidirectional) and decoder (causal)

================================================================================
                    PaliGemma Model Structure
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PaliGemmaForConditionalGeneration                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           PaliGemmaModel                                │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                      VISION TOWER (SigLIP)                       │   │ │
│  │  │                                                                  │   │ │
│  │  │   Input: [B, 3, 224, 224] (RGB images)                          │   │ │
│  │  │                                                                  │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  Patch Embedding (Conv2D 16x16)                          │  │   │ │
│  │  │   │    [B, 3, 224, 224] → [B, 196, 1152]                     │  │   │ │
│  │  │   │    (224/16 = 14, 14×14 = 196 patches)                    │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                           │                                      │   │ │
│  │  │                           ▼                                      │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  + Positional Embedding (Sinusoidal 2D)                  │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                           │                                      │   │ │
│  │  │                           ▼                                      │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  Transformer Encoder (27 layers, So400m variant)         │  │   │ │
│  │  │   │    - Multi-Head Self-Attention (16 heads)                │  │   │ │
│  │  │   │    - Feed-Forward Network (MLP)                          │  │   │ │
│  │  │   │    - LayerNorm                                           │  │   │ │
│  │  │   │    [B, 196, 1152] → [B, 196, 1152]                       │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                                                                  │   │ │
│  │  │   Output: [B, 196, 1152] (image patch features)                 │   │ │
│  │  └──────────────────────────────────────────────────────────────────┘   │ │
│  │                                │                                         │ │
│  │                                ▼                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                 MULTIMODAL PROJECTOR                             │   │ │
│  │  │                                                                  │   │ │
│  │  │   Linear: [B, 196, 1152] → [B, 196, 2048]                       │   │ │
│  │  │   (Projects vision features to language model dimension)         │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                │                                         │ │
│  │                                ▼                                         │ │
│  │                         Image Tokens                                     │ │
│  │                        [B, 196, 2048]                                    │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │               LANGUAGE MODEL (Gemma 2B - Decoder-Only)           │   │ │
│  │  │                                                                  │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  Token Embedding (vocab_size=257,152)                    │  │   │ │
│  │  │   │    Text tokens [B, seq_len] → [B, seq_len, 2048]         │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                           │                                      │   │ │
│  │  │                           ▼                                      │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  Concatenate: [Image Tokens, Language Tokens]            │  │   │ │
│  │  │   │    [B, 196, 2048] + [B, 48, 2048] → [B, 244, 2048]       │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                           │                                      │   │ │
│  │  │                           ▼                                      │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  × 18 Transformer Decoder Blocks                         │  │   │ │
│  │  │   │                                                          │  │   │ │
│  │  │   │  ┌────────────────────────────────────────────────────┐  │  │   │ │
│  │  │   │  │  RMSNorm (pre-attention)                           │  │  │   │ │
│  │  │   │  └────────────────────────────────────────────────────┘  │  │   │ │
│  │  │   │                        │                                  │  │   │ │
│  │  │   │                        ▼                                  │  │   │ │
│  │  │   │  ┌────────────────────────────────────────────────────┐  │  │   │ │
│  │  │   │  │  Multi-Head Self-Attention                         │  │  │   │ │
│  │  │   │  │    - 8 Query Heads                                 │  │  │   │ │
│  │  │   │  │    - 1 KV Head (Grouped Query Attention)           │  │  │   │ │
│  │  │   │  │    - Head Dim: 256                                 │  │  │   │ │
│  │  │   │  │    - RoPE (Rotary Position Embedding)              │  │  │   │ │
│  │  │   │  │    - Prefix-LM Attention Mask                      │  │  │   │ │
│  │  │   │  └────────────────────────────────────────────────────┘  │  │   │ │
│  │  │   │                        │                                  │  │   │ │
│  │  │   │                        ▼                                  │  │   │ │
│  │  │   │  ┌────────────────────────────────────────────────────┐  │  │   │ │
│  │  │   │  │  RMSNorm (pre-FFN)                                 │  │  │   │ │
│  │  │   │  └────────────────────────────────────────────────────┘  │  │   │ │
│  │  │   │                        │                                  │  │   │ │
│  │  │   │                        ▼                                  │  │   │ │
│  │  │   │  ┌────────────────────────────────────────────────────┐  │  │   │ │
│  │  │   │  │  Feed-Forward Network (Gated MLP)                  │  │  │   │ │
│  │  │   │  │    Linear: 2048 → 16384 (gate + up projection)     │  │  │   │ │
│  │  │   │  │    GELU activation                                 │  │  │   │ │
│  │  │   │  │    Linear: 16384 → 2048 (down projection)          │  │  │   │ │
│  │  │   │  └────────────────────────────────────────────────────┘  │  │   │ │
│  │  │   │                        │                                  │  │   │ │
│  │  │   │                  + Residual                               │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                           │                                      │   │ │
│  │  │                           ▼                                      │   │ │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │   │ │
│  │  │   │  Final RMSNorm                                           │  │   │ │
│  │  │   └──────────────────────────────────────────────────────────┘  │   │ │
│  │  │                                                                  │   │ │
│  │  │   Output: [B, seq_len, 2048] (hidden states)                    │   │ │
│  │  └──────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘


================================================================================
                    π0's Dual-Expert Architecture
================================================================================

In π0, PaliGemma is extended with an ACTION EXPERT (smaller Gemma 300M):

┌─────────────────────────────────────────────────────────────────────────────┐
│                       PaliGemmaWithExpertModel                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   PaliGemma (for PREFIX)                                │ │
│  │                                                                         │ │
│  │   ┌─────────────────┐    ┌───────────────┐    ┌─────────────────────┐  │ │
│  │   │   Vision Tower  │    │   Projector   │    │   Language Model    │  │ │
│  │   │    (SigLIP)     │ →  │   (Linear)    │ →  │    (Gemma 2B)       │  │ │
│  │   │                 │    │               │    │                     │  │ │
│  │   │  27 layers      │    │ 1152 → 2048   │    │  18 layers          │  │ │
│  │   │  1152 dim       │    │               │    │  2048 dim           │  │ │
│  │   └─────────────────┘    └───────────────┘    │  8 heads, 1 KV head │  │ │
│  │                                               └─────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                  Gemma Expert (for SUFFIX)                              │ │
│  │                                                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                     Gemma 300M                                   │  │ │
│  │   │                                                                  │  │ │
│  │   │   - 18 layers (same depth as Gemma 2B)                          │  │ │
│  │   │   - 1024 dim (smaller width)                                    │  │ │
│  │   │   - 8 heads, 1 KV head                                          │  │ │
│  │   │   - Head dim: 256                                               │  │ │
│  │   │   - MLP dim: 4096                                               │  │ │
│  │   │   - NO embedding layer (uses projected inputs)                  │  │ │
│  │   └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  KEY INSIGHT: Both experts share the SAME ATTENTION COMPUTATION!            │
│  - Queries, Keys, Values are computed separately for each expert            │
│  - But attention scores are computed JOINTLY on concatenated Q, K, V        │
│  - This allows prefix (Gemma 2B) and suffix (Gemma 300M) to attend to each  │
│    other while having separate FFN weights                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
                    Attention Computation in π0
================================================================================

For each of the 18 transformer layers:

    PREFIX (Gemma 2B)                    SUFFIX (Gemma 300M)
    ┌─────────────────┐                  ┌─────────────────┐
    │ prefix_hidden   │                  │ suffix_hidden   │
    │ [B, 636, 2048]  │                  │ [B, 51, 1024]   │
    └────────┬────────┘                  └────────┬────────┘
             │                                    │
             ▼                                    ▼
    ┌─────────────────┐                  ┌─────────────────┐
    │  RMSNorm        │                  │  RMSNorm        │
    │  (Gemma 2B)     │                  │  (Gemma 300M)   │
    └────────┬────────┘                  └────────┬────────┘
             │                                    │
             ▼                                    ▼
    ┌─────────────────┐                  ┌─────────────────┐
    │  Q, K, V proj   │                  │  Q, K, V proj   │
    │  (2048 → 2048)  │                  │  (1024 → 2048)  │
    │  W_q, W_k, W_v  │                  │  W_q, W_k, W_v  │
    │  (separate)     │                  │  (separate)     │
    └────────┬────────┘                  └────────┬────────┘
             │                                    │
             │  Q_prefix [B, 636, 8, 256]        │  Q_suffix [B, 51, 8, 256]
             │  K_prefix [B, 636, 1, 256]        │  K_suffix [B, 51, 1, 256]
             │  V_prefix [B, 636, 1, 256]        │  V_suffix [B, 51, 1, 256]
             │                                    │
             └──────────────┬─────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │        CONCATENATE            │
              │                               │
              │  Q = [Q_prefix, Q_suffix]     │
              │      [B, 687, 8, 256]         │
              │                               │
              │  K = [K_prefix, K_suffix]     │
              │      [B, 687, 1, 256]         │
              │                               │
              │  V = [V_prefix, V_suffix]     │
              │      [B, 687, 1, 256]         │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Apply RoPE to Q, K        │
              │     (Rotary Position Embed)   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   JOINT ATTENTION FORWARD     │
              │                               │
              │   attn_weights = Q @ K.T      │
              │   attn_weights *= scale       │
              │   attn_weights += attn_mask   │  ← Prefix-LM mask applied here
              │   attn_weights = softmax(...)  │
              │   attn_output = attn @ V      │
              │                               │
              │   [B, 687, 8*256] = [B, 687, 2048]
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │           SPLIT               │
              │                               │
              │  prefix_attn = [:, :636, :]   │
              │  suffix_attn = [:, 636:, :]   │
              └───────────────┬───────────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  O_proj (2B)    │               │  O_proj (300M)  │
    │  [2048 → 2048]  │               │  [2048 → 1024]  │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  + Residual     │               │  + Residual     │
    │  RMSNorm        │               │  RMSNorm        │
    │  FFN (2B)       │               │  FFN (300M)     │
    │  + Residual     │               │  + Residual     │
    └────────┬────────┘               └────────┬────────┘
             │                                 │
             ▼                                 ▼
    prefix_output                     suffix_output
    [B, 636, 2048]                    [B, 51, 1024]


================================================================================
                    Prefix-LM Attention Mask
================================================================================

The attention mask enables the "Prefix-LM" pattern:

                            KEY (j)
                    ┌───────────────────────────┐
                    │   PREFIX (636)  │ SUFFIX  │
                    │    (images +    │  (51)   │
                    │    language)    │(actions)│
          ┌─────────┼─────────────────┼─────────┤
          │         │                 │         │
   QUERY  │ PREFIX  │  ✓ ✓ ✓ ✓ ✓ ✓   │  ✗ ✗ ✗  │  Prefix sees prefix (bidirectional)
   (i)    │  (636)  │  (bidirectional)│(blocked)│  Prefix can't see suffix
          │         │                 │         │
          ├─────────┼─────────────────┼─────────┤
          │         │                 │         │
          │ SUFFIX  │  ✓ ✓ ✓ ✓ ✓ ✓   │  ✓ ✗ ✗  │  Suffix sees all prefix
          │  (51)   │ (sees prefix)   │  ✓ ✓ ✗  │  Suffix is causal within itself
          │         │                 │  ✓ ✓ ✓  │
          └─────────┴─────────────────┴─────────┘

This is achieved through the `ar_mask` (autoregressive mask) mechanism:
  - ar_mask = 0: Token shares attention block with previous (bidirectional)
  - ar_mask = 1: Token starts new causal block (previous can't see it)

    PREFIX ar_mask:  [0, 0, 0, 0, ..., 0, 0]  ← All bidirectional
    SUFFIX ar_mask:  [1, 1, 0, 0, ..., 0]     ← First two start new blocks
                      ↑  ↑
                   state first_action

    cumsum:          [0, 0, 0, ..., 0, 1, 2, 2, 2, ..., 2]
                      └── prefix ──┘  └──── suffix ────┘

    Token i attends to j if: cumsum[j] <= cumsum[i]


================================================================================
                    Model Configurations
================================================================================

┌────────────────────────────────────────────────────────────────────────────┐
│                        Gemma 2B (PaliGemma LLM)                            │
├────────────────────────────────────────────────────────────────────────────┤
│  Parameter          │ Value                                                │
│  ───────────────────┼─────────────────────────────────────────────────────│
│  Hidden Size        │ 2048                                                 │
│  Num Layers         │ 18                                                   │
│  Num Attention Heads│ 8                                                    │
│  Num KV Heads       │ 1 (Grouped Query Attention)                          │
│  Head Dimension     │ 256                                                  │
│  MLP Dimension      │ 16,384                                               │
│  Vocab Size         │ 257,152                                              │
│  Activation         │ GELU                                                 │
│  Normalization      │ RMSNorm                                              │
│  Position Encoding  │ RoPE (Rotary Position Embedding)                     │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                    Gemma 300M (Action Expert)                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Parameter          │ Value                                                │
│  ───────────────────┼─────────────────────────────────────────────────────│
│  Hidden Size        │ 1024                                                 │
│  Num Layers         │ 18 (same as Gemma 2B for layer-wise shared attn)     │
│  Num Attention Heads│ 8                                                    │
│  Num KV Heads       │ 1 (Grouped Query Attention)                          │
│  Head Dimension     │ 256                                                  │
│  MLP Dimension      │ 4,096                                                │
│  Vocab Size         │ 257,152 (shared, but embedding not used)             │
│  Activation         │ GELU                                                 │
│  Normalization      │ RMSNorm (or AdaRMSNorm for π0.5)                     │
│  Position Encoding  │ RoPE (Rotary Position Embedding)                     │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                        SigLIP Vision Encoder                               │
├────────────────────────────────────────────────────────────────────────────┤
│  Parameter          │ Value                                                │
│  ───────────────────┼─────────────────────────────────────────────────────│
│  Variant            │ So400m/14 (SigLIP 400M, patch size 14)              │
│  Input Size         │ 224 × 224                                            │
│  Patch Size         │ 16 × 16                                              │
│  Num Patches        │ 14 × 14 = 196                                        │
│  Hidden Size        │ 1152                                                 │
│  Num Layers         │ 27                                                   │
│  Num Attention Heads│ 16                                                   │
│  Intermediate Size  │ 4304                                                 │
│  Projection Dim     │ 2048 (to match Gemma 2B)                             │
└────────────────────────────────────────────────────────────────────────────┘


================================================================================
                    Summary: Why This Architecture?
================================================================================

1. DECODER-ONLY with PREFIX-LM:
   - Simpler than encoder-decoder (single model)
   - Prefix-LM allows bidirectional understanding of images/language
   - Causal suffix enables autoregressive-style action generation

2. DUAL EXPERTS:
   - Gemma 2B: Pre-trained on text, understands language well
   - Gemma 300M: Smaller, specialized for action generation
   - Shared attention allows cross-modal reasoning
   - Separate FFN allows task-specific processing

3. KV CACHE EFFICIENCY:
   - Prefix (images + language) computed once and cached
   - Only suffix recomputed during denoising loop
   - 10x speedup for inference

4. FLOW MATCHING:
   - Enables high-quality action generation
   - Velocity field prediction is simpler than noise prediction
   - Straight-line trajectories are easier to learn

================================================================================
"""


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """
    π0 / π0.5 Vision-Language-Action Model

    This class implements the full π0/π0.5 architecture with:
    - Dual-stream synchronized layer-by-layer computation
    - Flow matching for action prediction
    - Shared attention across VLM and Action Expert streams

    Key Architecture (see pi0_looklike.md for full diagrams):

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        PaliGemmaWithExpertModel                             │
    │                  (SYNCHRONIZED LAYER-BY-LAYER COMPUTATION)                  │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   PREFIX (VLM Stream)                    SUFFIX (Action Expert Stream)      │
    │   paligemma.language_model               gemma_expert.model                 │
    │                                                                             │
    │   ┌─────────────────────┐                ┌─────────────────────┐            │
    │   │  VLM Layer 0        │◄══ shared ════►│  Expert Layer 0     │            │
    │   └─────────────────────┘    attention   └─────────────────────┘            │
    │            │                                       │                        │
    │            ▼                                       ▼                        │
    │          ...                                     ...                        │
    │            │                                       │                        │
    │            ▼                                       ▼                        │
    │       prefix_out                            suffix_out                      │
    │       (discarded)                           (used for action)               │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

    NOT sequential:  ❌ VLM Layer 0→1→...→N, THEN Expert Layer 0→1→...→N
    BUT synchronized: ✓ (VLM Layer 0 + Expert Layer 0) → (VLM Layer 1 + Expert Layer 1) → ...

    π0 vs π0.5 Configuration:
        π0:   use_adarms=[False, False], state_proj, action_time_mlp
        π0.5: use_adarms=[False, True], no state, time_mlp → adarms_cond
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ========================================================================
        # Dual-stream model: VLM (Gemma 2B) + Action Expert (Gemma 300M)
        # ========================================================================
        # use_adarms controls AdaRMSNorm usage:
        #   - π0:   [False, False] - standard residual (x + y)
        #   - π0.5: [False, True]  - VLM standard, Expert gated (x + y*gate)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # ========================================================================
        # Action Projection Layers (shared between π0 and π0.5)
        # ========================================================================
        self.action_in_proj = nn.Linear(32, action_expert_config.width)  # [B, H, 32] → [B, H, 1024]
        self.action_out_proj = nn.Linear(
            action_expert_config.width, 32
        )  # [B, H, 1024] → [B, H, 32]

        # ========================================================================
        # π0 vs π0.5 Specific Layers
        # ========================================================================
        if self.pi05:
            # π0.5: Time goes to AdaRMSNorm conditioning (gated residual)
            # No state embedding, no action-time fusion
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            # adarms_cond = silu(time_mlp_out(silu(time_mlp_in(time_emb))))
        else:
            # π0: State as separate token, action-time concatenation + MLP fusion
            self.state_proj = nn.Linear(32, action_expert_config.width)  # [B, 32] → [B, 1024]
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_config.width, action_expert_config.width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )
            # action_time = mlp_out(silu(mlp_in(concat([action_emb, time_emb]))))

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        try:
            from workspace.nodesets.policy.policy_adapter_vla.models.openpi.models_pytorch.transformers_replace.models.siglip import (
                check,
            )

            if not check.check_whether_transformers_replace_is_installed_correctly():
                msg = "transformers_replace is not installed correctly."
                raise ValueError(msg)
        except ImportError:
            msg = "transformers_replace is not installed correctly."
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed PREFIX: images + language tokens for perception.

        The prefix uses BIDIRECTIONAL attention - all tokens can attend to each other.
        This allows the model to fully understand the visual and language context
        before generating actions.

        Args:
            images: List of camera images, each [B, 224, 224, 3]
            img_masks: List of image validity masks, each [B]
            lang_tokens: Tokenized language prompt [B, max_token_len]
            lang_masks: Language token validity mask [B, max_token_len]

        Returns:
            embs: Concatenated embeddings [B, num_prefix_tokens, 2048]
            pad_masks: Validity mask [B, num_prefix_tokens]
            att_masks: Attention mask [B, num_prefix_tokens] (all 0s = bidirectional)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # ============================================================================
        # Process images through SigLIP vision encoder
        # ============================================================================
        # Each image: [B, 224, 224, 3] -> [B, 196, 2048]
        # 196 = 14×14 patches (224/16 = 14 patches per side)
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            # img_emb: [B, 196, 2048]

            bsize, num_img_embs = img_emb.shape[:2]  # B, 196

            embs.append(img_emb)

            # Expand image mask to cover all patches
            # img_mask [B] -> [B, 196]
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # att_mask = 0 means bidirectional attention (tokens can see each other)
            att_masks += [0] * num_img_embs

        # ============================================================================
        # Process language tokens through Gemma embedding layer
        # ============================================================================
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            # lang_emb: [B, max_token_len, 2048]

            # Scale by sqrt(dim) following Gemma/PaLM convention
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        # lang_emb: [B, 48, 2048]

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Full bidirectional attention between image and language
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # ============================================================================
        # Concatenate all prefix tokens
        # ============================================================================
        embs = torch.cat(embs, dim=1)
        # embs: [B, 196*3 + 48, 2048] = [B, 636, 2048]

        pad_masks = torch.cat(pad_masks, dim=1)
        # pad_masks: [B, 636]

        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        # att_masks: [636] all zeros

        # Expand to batch dimension
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        # att_masks: [B, 636] all zeros = bidirectional attention

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """
        Embed SUFFIX: state + noisy_actions + timestep for action generation.

        The suffix uses a CAUSAL attention pattern within itself, but can attend
        to all prefix tokens. The first suffix token starts a new causal block,
        preventing prefix tokens from attending to suffix tokens.

        π0 SUFFIX EMBEDDING:
        ┌─────────────────────────────────────────┐
        │  SUFFIX EMBEDDING (π0)                  │
        │                                         │
        │  state_proj: [B,32] → [B,1,1024]       │
        │                                         │
        │  noisy_actions x_t = t*noise + (1-t)*a │
        │  action_in_proj: [B,H,32] → [B,H,1024] │
        │                                         │
        │  time_emb: sinusoidal [B,1024]         │
        │  expand: [B,1024] → [B,H,1024]         │
        │                                         │
        │  concat: [action, time] → [B,H,2048]   │
        │  action_time_mlp → [B,H,1024]          │
        │                                         │
        │  suffix_embs = [state, action_time]    │
        │              = [B, 1+H, 1024]          │
        │                                         │
        │  adarms_cond = None  ← NOT USED        │
        └─────────────────────────────────────────┘

        π0.5 SUFFIX EMBEDDING:
        ┌─────────────────────────────────────────┐
        │  SUFFIX EMBEDDING (π0.5)                │
        │                                         │
        │  (NO state embedding)                   │
        │                                         │
        │  noisy_actions x_t = t*noise + (1-t)*a │
        │  action_in_proj: [B,H,32] → [B,H,1024] │
        │                                         │
        │  time_emb: sinusoidal [B,1024]         │
        │  time_mlp: [B,1024] → [B,1024]         │
        │                                         │
        │  suffix_embs = action_emb  (no concat) │
        │              = [B, H, 1024]            │
        │                                         │
        │  adarms_cond = time_emb  ← USED!       │
        └─────────────────────────────────────────┘

        Args:
            state: Robot proprioception [B, 32]
            noisy_actions: Noisy actions at timestep t [B, action_horizon, action_dim] = [B, 50, 32]
            timestep: Flow matching timestep [B], value in [0.001, 0.999]

        Returns:
            embs: Concatenated embeddings [B, num_suffix_tokens, 1024]
                  - π0:   [B, 1+H, 1024] (state + action_time)
                  - π0.5: [B, H, 1024]   (action only)
            pad_masks: Validity mask [B, num_suffix_tokens]
            att_masks: Attention mask [B, num_suffix_tokens] (marks causal boundaries)
            adarms_cond: Adaptive RMSNorm conditioning
                  - π0:   None (not used)
                  - π0.5: time_emb [B, 1024] (used for gated residual)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # ============================================================================
        # Process state token (π0 only, not π0.5)
        # ============================================================================
        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Project state to action expert dimension
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            # state_emb: [B, 1024]

            # Add as single token
            embs.append(state_emb[:, None, :])
            # state_emb: [B, 1, 1024]

            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # att_mask = 1 starts a NEW CAUSAL BLOCK
            # This prevents prefix tokens from attending to suffix tokens
            att_masks += [1]

        # ============================================================================
        # Create sinusoidal timestep embedding
        # ============================================================================
        # Encodes the flow matching timestep t ∈ [0, 1] as a high-dimensional vector
        # Uses frequencies in range [4e-3, 4.0] for sensitivity to small timestep changes
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        # time_emb: [B, 1024]
        time_emb = time_emb.type(dtype=timestep.dtype)

        # ============================================================================
        # Project noisy actions to embedding dimension
        # ============================================================================
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        # action_emb: [B, 50, 1024]

        # ============================================================================
        # Fuse action + time information
        # ============================================================================
        if not self.pi05:
            # π0: Concatenate and MLP fusion
            # Expand time to match action sequence length
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            # time_emb: [B, 50, 1024]

            # Concatenate action and time embeddings
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            # action_time_emb: [B, 50, 2048]

            # Apply 2-layer MLP with SiLU activation
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)  # [B, 50, 2048] -> [B, 50, 1024]
                x = F.silu(x)  # swish == silu activation
                return self.action_time_mlp_out(x)  # [B, 50, 1024] -> [B, 50, 1024]

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            # action_time_emb: [B, 50, 1024]

            adarms_cond = None  # No adaptive normalization for π0
        else:
            # π0.5: Use adaptive RMSNorm instead of concatenation
            # Process time through MLP for adaptive normalization conditioning
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            # time_emb: [B, 1024]

            action_time_emb = action_emb  # No fusion, just action embeddings
            adarms_cond = time_emb  # Time passed to adaptive RMSNorm

        # ============================================================================
        # Add action tokens to sequence
        # ============================================================================
        embs.append(action_time_emb)
        # action_time_emb: [B, 50, 1024]

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        # Attention mask for action tokens:
        #   [1, 0, 0, 0, ..., 0]
        #    ↑
        #    First action starts new block (prefix can't see it)
        #    Subsequent actions share block (can see previous actions)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # ============================================================================
        # Concatenate all suffix tokens
        # ============================================================================
        embs = torch.cat(embs, dim=1)
        # embs: [B, 1 + 50, 1024] = [B, 51, 1024] (state + actions)

        pad_masks = torch.cat(pad_masks, dim=1)
        # pad_masks: [B, 51]

        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        # att_masks: [51] = [1, 1, 0, 0, ..., 0]
        #                    ↑  ↑
        #                 state first_action
        #
        # The cumsum of att_masks creates causal blocks:
        # cumsum = [1, 2, 2, 2, ..., 2]
        # Token i can attend to j if cumsum[j] <= cumsum[i]
        # So state (cumsum=1) and actions (cumsum=2) can see prefix (cumsum=0)
        # But prefix (cumsum=0) cannot see state/actions (cumsum>=1)

        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        # att_masks: [B, 51]

        return embs, pad_masks, att_masks, adarms_cond

    # region forward
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """
        Training forward pass using Flow Matching objective.

        Flow Matching learns to predict a velocity field v_θ that transforms
        noise into target actions. The model learns: v_θ(x_t, t) ≈ u_t = noise - actions

        Args:
            observation: Contains images, language tokens, and robot state
            actions: Target actions [B, action_horizon, action_dim] = [B, 50, 32]
            noise: Optional pre-sampled noise, same shape as actions
            time: Optional pre-sampled timesteps [B]

        Returns:
            MSE loss [B, action_horizon, action_dim] between predicted and target velocity
        """

        # ============================================================================
        # STEP 1: Preprocess observation
        # ============================================================================
        # Extract and normalize inputs from observation dataclass:
        #   - images: list of 3 camera images, each [B, 224, 224, 3]
        #   - img_masks: list of 3 masks [B], True if image is valid
        #   - lang_tokens: tokenized language prompt [B, max_token_len]
        #   - lang_masks: mask for valid tokens [B, max_token_len]
        #   - state: robot proprioception [B, 32]
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        # ============================================================================
        # STEP 2: Sample noise and timestep for flow matching
        # ============================================================================
        # Sample random Gaussian noise ε ~ N(0, I)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
            # noise: [B, 50, 32]

        # Sample timestep t ~ Beta(1.5, 1) * 0.999 + 0.001, so t ∈ [0.001, 0.999]
        # Beta(1.5, 1) is skewed towards higher values, meaning more training
        # emphasis on the noisier end of the trajectory
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
            # time: [B]

        # ============================================================================
        # STEP 3: Create noisy actions (flow matching interpolation)
        # ============================================================================
        # Flow matching defines a straight-line path from actions (t=0) to noise (t=1):
        #   x_t = t * noise + (1 - t) * actions
        #
        # At t=0: x_0 = actions (clean target)
        # At t=1: x_1 = noise (pure noise)
        # At t=0.5: x_0.5 = 0.5*noise + 0.5*actions (halfway interpolation)
        time_expanded = time[:, None, None]  # [B, 1, 1] for broadcasting
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # x_t: [B, 50, 32] - noisy actions at timestep t

        # The target velocity field is the direction from actions to noise:
        #   u_t = d(x_t)/dt = noise - actions
        # This is constant along the trajectory (straight line)
        u_t = noise - actions
        # u_t: [B, 50, 32] - target velocity the model should predict

        # ============================================================================
        # STEP 4: Embed PREFIX (images + language)
        # ============================================================================
        # PREFIX contains perception information with BIDIRECTIONAL attention
        #
        # embed_prefix() does:
        #   1. Pass each image through SigLIP: [B, 224, 224, 3] -> [B, 196, 2048]
        #      (196 = 14x14 patches from 224/16 patch size)
        #   2. Embed language tokens: [B, 48] -> [B, 48, 2048]
        #   3. Concatenate all: [B, 196*3 + 48, 2048] = [B, 636, 2048]
        #   4. Create attention masks: all zeros (bidirectional attention)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # prefix_embs: [B, 636, 2048] - embedded image + language tokens
        # prefix_pad_masks: [B, 636] - True where tokens are valid (not padding)
        # prefix_att_masks: [B, 636] - all 0s (bidirectional, tokens can see each other)

        # ============================================================================
        # STEP 5: Embed SUFFIX (state + noisy_actions + time)
        # ============================================================================
        # SUFFIX contains action generation tokens with CAUSAL attention
        #
        # embed_suffix() does:
        #   1. Project state: [B, 32] -> [B, 1, 1024] (state token)
        #   2. Project noisy actions: [B, 50, 32] -> [B, 50, 1024]
        #   3. Create sinusoidal time embedding: [B] -> [B, 1024]
        #   4. Fuse action + time with MLP: [B, 50, 2048] -> [B, 50, 1024]
        #   5. Concatenate state + action: [B, 1+50, 1024] = [B, 51, 1024]
        #   6. Create attention masks: [1, 1, 0, 0, ..., 0] (causal block)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )
        # suffix_embs: [B, 51, 1024] - embedded state + action tokens
        # suffix_pad_masks: [B, 51] - all True (no padding in suffix)
        # suffix_att_masks: [B, 51] - [1, 1, 0, 0, ...] marks causal boundary
        # adarms_cond: None for π0, [B, 1024] time embedding for π0.5

        # Cast to bfloat16 if model weights are in bfloat16
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # ============================================================================
        # STEP 6: Concatenate prefix and suffix, create attention mask
        # ============================================================================
        # Combine prefix and suffix into one sequence
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        # pad_masks: [B, 687] - which tokens are valid

        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        # att_masks: [B, 687]
        # [0,0,0,...,0, 0,0,...,0, 1, 1,0,0,...,0]
        #  └─ images ─┘ └─ lang ─┘ │  └─ actions ─┘
        #                        state
        # 0 = bidirectional (share attention with previous)
        # 1 = starts new causal block (previous tokens can't see this)

        # Create 2D attention mask using cumsum trick:
        # Token i can attend to token j if cumsum[j] <= cumsum[i]
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # att_2d_masks: [B, 687, 687] - True where attention is allowed
        #
        # Result pattern:
        #                PREFIX (636)              SUFFIX (51)
        #         ┌─────────────────────────┬─────────────────┐
        # PREFIX  │    ✓ (bidirectional)    │   ✗ (blocked)   │
        #         ├─────────────────────────┼─────────────────┤
        # SUFFIX  │  ✓ (can see prefix)     │  ✓ (causal)     │
        #         └─────────────────────────┴─────────────────┘

        # Compute position IDs for RoPE (Rotary Position Embedding)
        # cumsum gives: [1, 2, 3, ..., 687], subtract 1 to get [0, 1, 2, ..., 686]
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # position_ids: [B, 687]

        # ============================================================================
        # STEP 7: Prepare 4D attention mask for transformer
        # ============================================================================
        # Transformers expect attention mask as [B, 1, T, S] with float values
        # True (can attend) -> 0.0
        # False (blocked) -> -inf (or very large negative number)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        # att_2d_masks_4d: [B, 1, 687, 687]

        # ============================================================================
        # STEP 8: Forward pass through PaliGemma + Action Expert
        # ============================================================================
        # The model uses DUAL-STREAM architecture with SYNCHRONIZED LAYER-BY-LAYER:
        #   - Stream 0 (Gemma 2B): processes prefix (images + language)
        #   - Stream 1 (Gemma 300M): processes suffix (state + actions)
        #
        # Both streams process each layer TOGETHER, sharing attention computation.
        # Q, K, V are concatenated across streams, attention computed jointly,
        # then split back for separate o_proj, MLP, and residuals.
        #
        # inputs_embeds = [prefix_embs, suffix_embs]
        #                  [B, 636, 2048]  [B, 51, 1024] (π0)
        #                                  [B, 50, 1024] (π0.5, no state)
        #
        # adarms_cond = [None, adarms_cond]
        #   - None for prefix (VLM never uses AdaRMS)
        #   - π0:   None (no adaptive normalization)
        #   - π0.5: time_emb [B, 1024] (for gated residual: x + y * gate)
        #
        # IMPORTANT: prefix_out is DISCARDED! Only suffix_out is used.
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = (
                self.paligemma_with_expert.forward(  # Note: _ = prefix_out (discarded)
                    attention_mask=att_2d_masks_4d,  # [B, 1, 687, 687]
                    position_ids=position_ids,  # [B, 687]
                    past_key_values=None,  # No KV cache during training
                    inputs_embeds=[prefix_embs, suffix_embs],  # [prefix, suffix]
                    use_cache=False,  # Don't store KV cache
                    adarms_cond=[None, adarms_cond],  # Adaptive RMSNorm conditioning
                )
            )
            return suffix_out

        # Apply gradient checkpointing if enabled (saves memory during training)
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        # suffix_out: [B, 51, 1024] - transformer output for suffix tokens

        # ============================================================================
        # STEP 9: Extract action outputs and project to velocity prediction
        # ============================================================================
        # Take only the last `action_horizon` tokens (the action tokens, not state)
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out: [B, 50, 1024]

        suffix_out = suffix_out.to(dtype=torch.float32)

        # Project from expert width (1024) to action dimension (32)
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        # v_t: [B, 50, 32] - predicted velocity field

        # ============================================================================
        # STEP 10: Compute flow matching loss
        # ============================================================================
        # Loss = MSE(predicted_velocity, target_velocity)
        #      = MSE(v_θ(x_t, t), u_t)
        #      = MSE(v_θ(x_t, t), noise - actions)
        #
        # The model learns to predict the direction from clean actions to noise,
        # which can be reversed during inference to denoise.
        return F.mse_loss(u_t, v_t, reduction="none")
        # Returns: [B, 50, 32] - per-element MSE loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """
        Inference: Generate actions by iteratively denoising from pure noise.

        Uses Euler integration to follow the learned velocity field from t=1 (noise)
        to t=0 (clean actions). The prefix (images + language) is computed once
        and cached, while only the suffix is recomputed at each denoising step.

        Args:
            device: Device to run inference on
            observation: Contains images, language tokens, and robot state
            noise: Optional starting noise [B, 50, 32], sampled if not provided
            num_steps: Number of denoising steps (default: 10)

        Returns:
            Denoised actions [B, 50, 32]
        """
        bsize = observation.state.shape[0]

        # ============================================================================
        # STEP 1: Initialize with pure noise
        # ============================================================================
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
            # noise: [B, 50, 32] ~ N(0, I)

        # ============================================================================
        # STEP 2: Preprocess observation
        # ============================================================================
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        # ============================================================================
        # STEP 3: Embed prefix and create KV cache
        # ============================================================================
        # The prefix (images + language) doesn't change during denoising,
        # so we compute it once and cache the key-value pairs for efficiency
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # prefix_embs: [B, 636, 2048]

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache (PREFILL step)
        # This stores K, V for all prefix tokens so we don't recompute them
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # Only prefix, no suffix
            use_cache=True,  # Store KV cache
        )
        # past_key_values: cached K, V tensors for all transformer layers

        # ============================================================================
        # STEP 4: Denoising loop (Euler integration)
        # ============================================================================
        # We integrate from t=1 (noise) to t=0 (clean actions)
        # Using Euler method: x_{t+dt} = x_t + dt * v_θ(x_t, t)
        #
        # Note: dt is negative since we're going from t=1 to t=0
        dt = -1.0 / num_steps  # e.g., -0.1 for 10 steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise  # Start with pure noise at t=1
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Iterate: t = 1.0 -> 0.9 -> 0.8 -> ... -> 0.1 -> 0.0
        while time >= -dt / 2:  # Robust to floating point errors
            expanded_time = time.expand(bsize)  # [B]

            # Get predicted velocity at current (x_t, t)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,  # Reuse cached prefix KV
                x_t,
                expanded_time,
            )
            # v_t: [B, 50, 32] - predicted velocity field

            # Euler step: x_{t+dt} = x_t + dt * v_t
            # Since dt < 0, this moves x_t towards cleaner actions
            x_t = x_t + dt * v_t
            time += dt  # t decreases: 1.0 -> 0.9 -> ... -> 0.0

        # ============================================================================
        # STEP 5: Return denoised actions
        # ============================================================================
        return x_t
        # x_t: [B, 50, 32] - clean actions at t≈0

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """
        Apply one denoising step: predict velocity v_θ(x_t, t) at current state.

        This function uses the cached prefix KV and only computes the suffix
        (state + noisy actions + time) for efficiency.

        Args:
            state: Robot proprioception [B, 32]
            prefix_pad_masks: Mask for valid prefix tokens [B, 636]
            past_key_values: Cached K, V from prefix forward pass
            x_t: Current noisy actions [B, 50, 32]
            timestep: Current timestep [B], value in [0, 1]

        Returns:
            Predicted velocity v_t [B, 50, 32]
        """

        # ============================================================================
        # STEP 1: Embed suffix (state + noisy actions + time)
        # ============================================================================
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep
        )
        # suffix_embs: [B, 51, 1024] - state token + action tokens with time
        # suffix_att_masks: [1, 1, 0, 0, ...] - causal attention pattern

        suffix_len = suffix_pad_masks.shape[1]  # 51
        batch_size = prefix_pad_masks.shape[0]  # B
        prefix_len = prefix_pad_masks.shape[1]  # 636

        # ============================================================================
        # STEP 2: Build attention mask for suffix attending to prefix + suffix
        # ============================================================================
        # The suffix tokens need to attend to:
        #   1. All prefix tokens (via KV cache)
        #   2. Previous suffix tokens (causal within suffix)

        # Suffix can attend to all valid prefix tokens
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        # prefix_pad_2d_masks: [B, 51, 636] - all True where prefix is valid

        # Causal attention within suffix
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # suffix_att_2d_masks: [B, 51, 51] - lower triangular (causal)

        # Combine: suffix attends to [prefix, suffix]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        # full_att_2d_masks: [B, 51, 687]
        #
        # For each suffix token (row), it can attend to:
        #   - All 636 prefix tokens (from cache)
        #   - Previous suffix tokens (causal)

        # ============================================================================
        # STEP 3: Compute position IDs for suffix tokens
        # ============================================================================
        # Suffix positions continue from where prefix ended
        # If prefix has 636 tokens, suffix positions are [636, 637, ..., 686]
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # [B, 1]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        # position_ids: [B, 51] with values [636, 637, ..., 686]

        # ============================================================================
        # STEP 4: Forward pass with KV cache
        # ============================================================================
        # Prepare 4D attention mask
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # full_att_2d_masks_4d: [B, 1, 51, 687]

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # Forward pass:
        #   - inputs_embeds[0] = None: don't compute prefix, use cache
        #   - inputs_embeds[1] = suffix_embs: compute suffix
        #   - past_key_values: cached K, V from prefix
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,  # [B, 1, 51, 687]
            position_ids=position_ids,  # [B, 51]
            past_key_values=past_key_values,  # Cached prefix KV
            inputs_embeds=[None, suffix_embs],  # Only compute suffix
            use_cache=False,  # Don't update cache
            adarms_cond=[None, adarms_cond],  # Time conditioning for π0.5
        )
        # outputs_embeds: [None, suffix_out]

        # ============================================================================
        # STEP 5: Extract action output and project to velocity
        # ============================================================================
        suffix_out = outputs_embeds[1]
        # suffix_out: [B, 51, 1024]

        # Take only action tokens (last 50), not state token
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out: [B, 50, 1024]

        suffix_out = suffix_out.to(dtype=torch.float32)

        # Project to action dimension
        return self.action_out_proj(suffix_out)
        # Returns: [B, 50, 32] - predicted velocity v_θ(x_t, t)
