"""
PaliGemma with Action Expert Model - Comprehensive Explanation
==============================================================

This file implements the dual-stream architecture used in π0 (Pi-Zero) and π0.5
for vision-language-action models.

The architecture combines PaliGemma (vision-language model) with a lightweight
Gemma action expert using **SYNCHRONIZED LAYER-BY-LAYER** computation.

==============================================================
ARCHITECTURE OVERVIEW: DUAL-STREAM GEMMA
==============================================================

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
│   ┌─────────────────────┐                ┌─────────────────────┐            │
│   │  VLM Layer 1        │◄══ shared ════►│  Expert Layer 1     │            │
│   └─────────────────────┘    attention   └─────────────────────┘            │
│            │                                       │                        │
│            ▼                                       ▼                        │
│          ...                                     ...                        │
│            │                                       │                        │
│            ▼                                       ▼                        │
│   ┌─────────────────────┐                ┌─────────────────────┐            │
│   │  VLM Layer N        │◄══ shared ════►│  Expert Layer N     │            │
│   └─────────────────────┘    attention   └─────────────────────┘            │
│            │                                       │                        │
│            ▼                                       ▼                        │
│       prefix_out                            suffix_out                      │
│       (discarded)                           (used for action)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

NOT sequential:  ❌ VLM Layer 0→1→...→N, THEN Expert Layer 0→1→...→N
BUT synchronized: ✓ (VLM Layer 0 + Expert Layer 0) → (VLM Layer 1 + Expert Layer 1) → ...

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL SPECIFICATIONS                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐           │
│   │     PaliGemma (Gemma 2B)    │    │   Action Expert (Gemma 300M)│           │
│   │         [PREFIX]            │    │         [SUFFIX]            │           │
│   ├─────────────────────────────┤    ├─────────────────────────────┤           │
│   │ • width = 2048              │    │ • width = 1024              │           │
│   │ • mlp_dim = 16384           │    │ • mlp_dim = 4096            │           │
│   │ • num_heads = 8             │    │ • num_heads = 8             │           │
│   │ • head_dim = 256            │    │ • head_dim = 128            │           │
│   │ • kv_heads = 1              │    │ • kv_heads = 1              │           │
│   │ • depth = 18 layers         │    │ • depth = 18 layers         │           │
│   │ • ~2B parameters            │    │ • ~300M parameters          │           │
│   │ • use_adarms = False        │    │ • use_adarms = False (π0)   │           │
│   │                             │    │ • use_adarms = True (π0.5)  │           │
│   │ • residual: x + y           │    │ • residual: x + y (π0)      │           │
│   │                             │    │ • residual: x + y*gate(π0.5)│           │
│   └─────────────────────────────┘    └─────────────────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


==============================================================
ATTENTION PATTERN: PREFIX-LM (Bidirectional Prefix, Causal Suffix)
==============================================================

The attention mask allows:
- PREFIX tokens can attend to ALL prefix tokens (bidirectional)
- SUFFIX tokens can attend to ALL prefix tokens AND preceding suffix tokens (causal)

┌────────────────────────────────────────────────────────────────────────────────┐
│                         ATTENTION MASK PATTERN                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│              │ img1 │ img2 │ txt1 │ txt2 │ act1 │ act2 │ act3 │                │
│         ─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤                │
│         img1 │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │  ✗   │  ✗   │                │
│         img2 │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │  ✗   │  ✗   │   PREFIX      │
│         txt1 │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │  ✗   │  ✗   │ (bidirectional)│
│         txt2 │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │  ✗   │  ✗   │                │
│         ─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤                │
│         act1 │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │  ✗   │                │
│         act2 │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✗   │   SUFFIX      │
│         act3 │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │   (causal)    │
│                                                                                 │
│         ✓ = can attend    ✗ = cannot attend                                    │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘


==============================================================
SHARED ATTENTION MECHANISM (Key to Dual-Stream Architecture)
==============================================================

The key to the dual-stream architecture is **shared attention** - both streams
compute Q, K, V separately, then CONCATENATE for joint attention:

┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHARED ATTENTION (per layer)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VLM hidden [B, L_prefix, 2048]      Expert hidden [B, L_suffix, 1024]     │
│       │                                     │                               │
│       ▼                                     ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ input_norm  │                      │ input_norm  │ (AdaRMSNorm for π0.5) │
│  └─────────────┘                      └─────────────┘                       │
│       │                                     │                               │
│       ▼                                     ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ Q_vlm, K_vlm│                      │ Q_exp, K_exp│                       │
│  │ V_vlm       │                      │ V_exp       │                       │
│  └─────────────┘                      └─────────────┘                       │
│       │                                     │                               │
│       └──────────────┬──────────────────────┘                               │
│                      │                                                      │
│                      ▼                                                      │
│         ┌───────────────────────────────────────┐                           │
│         │  CONCATENATE along sequence dim        │                          │
│         │  Q = [Q_vlm, Q_exp]  [B, H, L_total, D]│                          │
│         │  K = [K_vlm, K_exp]  [B, H, L_total, D]│                          │
│         │  V = [V_vlm, V_exp]  [B, H, L_total, D]│                          │
│         └───────────────────────────────────────┘                           │
│                      │                                                      │
│                      ▼                                                      │
│         ┌───────────────────────────────────────┐                           │
│         │  SINGLE ATTENTION COMPUTATION          │                          │
│         │  attn = softmax(Q @ K.T / √d) @ V     │                          │
│         │  with attention_mask applied          │                           │
│         └───────────────────────────────────────┘                           │
│                      │                                                      │
│                      ▼                                                      │
│         ┌───────────────────────────────────────┐                           │
│         │  SPLIT back to each stream            │                           │
│         │  attn_vlm = attn[:, :L_prefix]        │                           │
│         │  attn_exp = attn[:, L_prefix:]        │                           │
│         └───────────────────────────────────────┘                           │
│                      │                                                      │
│       ┌──────────────┴──────────────────────┐                               │
│       │                                     │                               │
│       ▼                                     ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ o_proj_vlm  │                      │ o_proj_exp  │                       │
│  │ + residual  │                      │ + residual  │ (gated for π0.5)      │
│  │ + MLP_vlm   │                      │ + MLP_exp   │                       │
│  └─────────────┘                      └─────────────┘                       │
│       │                                     │                               │
│       ▼                                     ▼                               │
│  VLM hidden (updated)              Expert hidden (updated)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

**Why this works:**
- Action tokens can **attend to image+text tokens** (cross-modal attention)
- Image+text tokens can attend to each other (but NOT to action tokens due to attention mask)
- Each stream maintains its own **o_proj, MLP, residuals** (different hidden sizes)


==============================================================
FORWARD FUNCTION: THREE EXECUTION MODES
==============================================================

The forward function supports THREE distinct modes:

MODE 1: PREFIX-ONLY (inputs_embeds[1] is None)
─────────────────────────────────────────────
Used for: KV caching during inference prefill
- Only processes prefix tokens (image + language)
- Returns prefix hidden states and KV cache
- Action expert is NOT used

    inputs_embeds = [prefix_embeds, None]
                           │
                           ▼
              ┌─────────────────────────┐
              │   PaliGemma Language    │
              │   Model Forward         │
              └─────────────────────────┘
                           │
                           ▼
              (prefix_output, past_key_values)


MODE 2: SUFFIX-ONLY (inputs_embeds[0] is None)
─────────────────────────────────────────────
Used for: Autoregressive decoding with KV cache
- Only processes suffix/action tokens
- Uses cached KV from prefix
- PaliGemma language model is NOT used

    inputs_embeds = [None, suffix_embeds]
                              │
                              ▼
              ┌─────────────────────────┐
              │   Gemma Expert Model    │
              │   Forward               │
              └─────────────────────────┘
                              │
                              ▼
                    suffix_output


MODE 3: JOINT TRAINING (both inputs_embeds provided)
───────────────────────────────────────────────────
Used for: Training with flow matching
- Processes BOTH prefix and suffix together
- SHARED ATTENTION across both models
- SEPARATE FFN for each model
- This is the core innovation of π0

    inputs_embeds = [prefix_embeds, suffix_embeds]
                           │              │
                           ▼              ▼
              ┌─────────────────────────────────────┐
              │     JOINT LAYER COMPUTATION         │
              │  (Shared Attention + Separate FFN)  │
              │         × 18 layers                 │
              └─────────────────────────────────────┘
                           │              │
                           ▼              ▼
              (prefix_output, suffix_output)


==============================================================
JOINT LAYER COMPUTATION (compute_layer_complete)
==============================================================

This is the core of the dual-expert architecture. Each layer:
1. Applies LayerNorm to BOTH prefix and suffix separately
2. Computes Q, K, V for BOTH prefix and suffix separately
3. CONCATENATES Q, K, V along sequence dimension
4. Applies rotary position embeddings
5. Computes SHARED attention
6. Splits attention output back to prefix and suffix
7. Applies SEPARATE FFN for each

STEP-BY-STEP DIAGRAM:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    JOINT LAYER COMPUTATION (per layer)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: prefix_embeds [B, T_prefix, 2048]    suffix_embeds [B, T_suffix, 1024]  │
│                │                                      │                         │
│                ▼                                      ▼                         │
│  ┌──────────────────────────┐          ┌──────────────────────────┐            │
│  │  input_layernorm (2B)    │          │  input_layernorm (300M)  │            │
│  │  + AdaRMSNorm gate       │          │  + AdaRMSNorm gate       │            │
│  └──────────────────────────┘          └──────────────────────────┘            │
│                │                                      │                         │
│                ▼                                      ▼                         │
│  ┌──────────────────────────┐          ┌──────────────────────────┐            │
│  │  Q_proj: 2048 → 8×256    │          │  Q_proj: 1024 → 8×128    │            │
│  │  K_proj: 2048 → 1×256    │          │  K_proj: 1024 → 1×128    │            │
│  │  V_proj: 2048 → 1×256    │          │  V_proj: 1024 → 1×128    │            │
│  └──────────────────────────┘          └──────────────────────────┘            │
│                │                                      │                         │
│                │    Q: [B, 8, T_prefix, 256]          │    Q: [B, 8, T_suffix, 128]
│                │    K: [B, 1, T_prefix, 256]          │    K: [B, 1, T_suffix, 128]
│                │    V: [B, 1, T_prefix, 256]          │    V: [B, 1, T_suffix, 128]
│                │                                      │                         │
│                └──────────────┬───────────────────────┘                         │
│                               │                                                 │
│                               ▼                                                 │
│                  ┌─────────────────────────────┐                                │
│                  │   CONCATENATE along seq dim │                                │
│                  │   Q: [B, 8, T_total, 256]   │   Note: head_dim must match!  │
│                  │   K: [B, 1, T_total, 256]   │   (This impl uses 256 for both)│
│                  │   V: [B, 1, T_total, 256]   │                                │
│                  └─────────────────────────────┘                                │
│                               │                                                 │
│                               ▼                                                 │
│                  ┌─────────────────────────────┐                                │
│                  │   Apply Rotary Pos Emb      │                                │
│                  │   (RoPE on Q and K)         │                                │
│                  └─────────────────────────────┘                                │
│                               │                                                 │
│                               ▼                                                 │
│                  ┌─────────────────────────────┐                                │
│                  │   SHARED ATTENTION          │                                │
│                  │   eager_attention_forward   │                                │
│                  │   with PREFIX-LM mask       │                                │
│                  │                             │                                │
│                  │   Output: [B, T_total, D]   │                                │
│                  └─────────────────────────────┘                                │
│                               │                                                 │
│                ┌──────────────┴──────────────┐                                  │
│                │         SPLIT               │                                  │
│                ▼                             ▼                                  │
│  att_out[:, :T_prefix]            att_out[:, T_prefix:]                         │
│                │                             │                                  │
│                ▼                             ▼                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                    │
│  │  O_proj (Gemma 2B)       │  │  O_proj (Gemma 300M)     │                    │
│  │  8×256 → 2048            │  │  8×128 → 1024            │                    │
│  └──────────────────────────┘  └──────────────────────────┘                    │
│                │                             │                                  │
│                ▼                             ▼                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                    │
│  │  Residual + LayerNorm   │  │  Residual + LayerNorm    │                    │
│  │  (post_attention)        │  │  (post_attention)        │                    │
│  └──────────────────────────┘  └──────────────────────────┘                    │
│                │                             │                                  │
│                ▼                             ▼                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                    │
│  │  MLP (Gemma 2B)          │  │  MLP (Gemma 300M)        │                    │
│  │  2048→16384→2048         │  │  1024→4096→1024          │                    │
│  │  (GeGLU activation)      │  │  (GeGLU activation)      │                    │
│  └──────────────────────────┘  └──────────────────────────┘                    │
│                │                             │                                  │
│                ▼                             ▼                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                    │
│  │  Second Residual         │  │  Second Residual         │                    │
│  │  (_gated_residual)       │  │  (_gated_residual)       │                    │
│  └──────────────────────────┘  └──────────────────────────┘                    │
│                │                             │                                  │
│                ▼                             ▼                                  │
│  OUTPUT: prefix_embeds          OUTPUT: suffix_embeds                           │
│          [B, T_prefix, 2048]            [B, T_suffix, 1024]                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


==============================================================
AdaRMSNorm (π0.5 Time Conditioning) - ONLY USED IN π0.5
==============================================================

For π0.5, an additional time conditioning mechanism is used via AdaRMSNorm.
This modulates the layer normalization based on the flow matching timestep.

┌─────────────────────────────────────────────────────────────────────────────┐
│                              AdaRMSNorm                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  x [B, T, 1024]                    cond [B, 1024] (time embedding)          │
│       │                                  │                                  │
│       ▼                                  ▼                                  │
│  ┌─────────────┐                  ┌──────────────────┐                      │
│  │  RMSNorm(x) │                  │ dense(cond)      │                      │
│  │  normalize  │                  │ Linear(1024,3072)│                      │
│  └─────────────┘                  └──────────────────┘                      │
│       │                                  │                                  │
│       │                                  ▼                                  │
│       │                           chunk into 3:                             │
│       │                           scale [B, 1024]                           │
│       │                           shift [B, 1024]                           │
│       │                           gate  [B, 1024]                           │
│       │                                  │                                  │
│       └────────────┬─────────────────────┘                                  │
│                    │                                                        │
│                    ▼                                                        │
│         output = normed_x * (1 + scale) + shift                             │
│                                                                             │
│         return (output, gate)                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

The gate from AdaRMSNorm is used for GATED RESIDUAL connections:
    out = residual + gate * layer_output   (π0.5)
    out = residual + layer_output          (π0, standard)

Configuration flag: use_adarms = [bool_for_vlm, bool_for_expert]
    π0:   use_adarms = [False, False]  - No adaptive normalization
    π0.5: use_adarms = [False, True]   - Only expert uses AdaRMS

==============================================================
π0 vs π0.5: GemmaDecoderLayer Differences
==============================================================

π0 (Standard):                              π0.5 (AdaRMS + Gated):

hidden_states                               hidden_states    adarms_cond
     │                                           │               │
     ├────────────┐ (residual)                   ├───────────────┤ (residual)
     │            │                              │               │
     ▼            │                              ▼               │
┌─────────────┐   │                        ┌─────────────┐       │
│ RMSNorm     │   │                        │ AdaRMSNorm  │←──────┤
│ → output    │   │                        │ → (out,gate)│       │
└─────────────┘   │                        └─────────────┘       │
     │            │                              │     │         │
     ▼            │                              │     └─ gate   │
┌─────────────┐   │                              ▼         │     │
│ Attention   │   │                        ┌─────────────┐ │     │
└─────────────┘   │                        │ Attention   │ │     │
     │            │                        └─────────────┘ │     │
     ▼            │                              │         │     │
   x + y ◄────────┘                              ▼         ▼     │
     │                                      x + y * gate ◄───────┘
     │                                           │
     ├────────────┐ (residual)                   ├───────────────┐
     │            │                              │               │
     ▼            │                              ▼               │
┌─────────────┐   │                        ┌─────────────┐       │
│ RMSNorm     │   │                        │ AdaRMSNorm  │←── cond
└─────────────┘   │                        │ → (out,gate)│       │
     │            │                        └─────────────┘       │
     ▼            │                              │     │         │
┌─────────────┐   │                              ▼     └─ gate   │
│ MLP         │   │                        ┌─────────────┐ │     │
└─────────────┘   │                        │ MLP         │ │     │
     │            │                        └─────────────┘ │     │
     ▼            │                              │         │     │
   x + y ◄────────┘                              ▼         ▼     │
                                            x + y * gate ◄───────┘


==============================================================
GRADIENT CHECKPOINTING
==============================================================

The forward function supports gradient checkpointing to reduce memory usage
during training. When enabled:
- Each layer computation is wrapped in torch.utils.checkpoint.checkpoint
- Activations are recomputed during backward pass instead of stored
- Significantly reduces GPU memory at cost of ~30% slower training

    if use_gradient_checkpointing:
        inputs_embeds = torch.utils.checkpoint.checkpoint(
            compute_layer_complete,
            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
            use_reentrant=False,
            preserve_rng_state=False,
        )


==============================================================
PRECISION HANDLING
==============================================================

The model supports mixed precision training:
- Most parameters: bfloat16 for memory efficiency
- Critical parameters kept in float32 for numerical stability:
  - Vision tower patch embedding
  - Vision tower position embedding
  - All LayerNorm weights (input_layernorm, post_attention_layernorm, model.norm)

This prevents numerical instability in normalization layers while maintaining
memory efficiency for the bulk of the computation.

"""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING

from workspace.nodesets.policy.policy_vla.models.openpi.models_pytorch.transformers_replace.models.gemma import (
    modeling_gemma,
)
from workspace.nodesets.policy.policy_vla.models.openpi.models_pytorch.transformers_replace.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
)
from workspace.nodesets.policy.policy_vla.models.openpi.models_pytorch.transformers_replace.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)


class PaliGemmaWithExpertModel(nn.Module):
    """
    Dual-Stream Model: PaliGemma (Gemma 2B) + Action Expert (Gemma 300M)

    This class implements the core π0/π0.5 architecture where:
    - PaliGemma handles vision + language understanding (PREFIX stream)
    - Action Expert handles action prediction (SUFFIX stream)
    - Both streams are processed LAYER-BY-LAYER with shared attention
    - prefix_out is DISCARDED; only suffix_out is used for action prediction

    Key Architecture Features:
    - SYNCHRONIZED layer-by-layer computation (not sequential)
    - Shared attention: Q, K, V concatenated, joint attention, then split
    - Separate o_proj, MLP, residuals per stream (different hidden sizes)
    - π0: standard residual (x + y), no AdaRMS
    - π0.5: gated residual (x + y * gate), AdaRMS on expert only

    Architecture Diagram:
    ┌────────────────────────────────────────────────────────────────┐
    │                    PaliGemmaWithExpertModel                    │
    │              (SYNCHRONIZED LAYER-BY-LAYER)                     │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  self.paligemma (PaliGemmaForConditionalGeneration)           │
    │  ├── vision_tower (SigLIP ViT)                                │
    │  ├── multi_modal_projector                                    │
    │  └── language_model (GemmaForCausalLM - 2B)                   │
    │       ├── embed_tokens                                        │
    │       ├── layers[0:17] (18 GemmaDecoderLayers)               │
    │       └── norm (RMSNorm)                                      │
    │                                                                │
    │  self.gemma_expert (GemmaForCausalLM - 300M)                  │
    │  ├── embed_tokens = None  (receives pre-embedded actions)    │
    │  ├── layers[0:17] (18 GemmaDecoderLayers)                    │
    │  │   └── use_adarms = True for π0.5                          │
    │  └── norm (RMSNorm or AdaRMSNorm)                            │
    │                                                                │
    │  OUTPUT: [prefix_out, suffix_out]                             │
    │           ↓          ↓                                        │
    │        DISCARDED   → action_out_proj → velocity prediction    │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """
        Initialize the dual-expert model.

        Args:
            vlm_config: Configuration for PaliGemma (Gemma 2B)
                - width: 2048 (hidden dimension)
                - mlp_dim: 16384 (FFN intermediate dimension)
                - num_heads: 8 (attention heads)
                - head_dim: 256 (dimension per head)
                - num_kv_heads: 1 (GQA with 1 KV head)
                - depth: 18 (number of layers)

            action_expert_config: Configuration for Action Expert (Gemma 300M)
                - width: 1024 (hidden dimension)
                - mlp_dim: 4096 (FFN intermediate dimension)
                - num_heads: 8 (attention heads)
                - head_dim: 128 (dimension per head)
                - num_kv_heads: 1 (GQA with 1 KV head)
                - depth: 18 (number of layers)

            use_adarms: [bool, bool] - Enable AdaRMSNorm for [VLM, Expert]
                - Used in π0.5 for time conditioning
                - Default: [False, False] (standard π0)

            precision: "bfloat16" or "float32"
                - bfloat16: Memory efficient, good for training
                - float32: Full precision, good for debugging
        """
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # ================================================================
        # STEP 1: Configure PaliGemma (Vision-Language Model)
        # ================================================================
        # PaliGemma = SigLIP Vision Encoder + Gemma 2B Language Model
        #
        # Key config values:
        # - vocab_size: 257152 (extended vocabulary)
        # - image_token_index: 257152 (special token for image features)
        # - hidden_size: 2048 (Gemma 2B dimension)
        # - num_hidden_layers: 18 (depth)

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152

        # Text (Gemma 2B) configuration
        vlm_config_hf.text_config.hidden_size = vlm_config.width  # 2048
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim  # 16384
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads  # 8
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim  # 256
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth  # 18
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads  # 1 (GQA)
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None

        # Vision (SigLIP) configuration
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # ================================================================
        # STEP 2: Configure Action Expert (Gemma 300M)
        # ================================================================
        # Lightweight decoder for action prediction
        # Shares attention with PaliGemma but has separate FFN weights
        #
        # Key differences from Gemma 2B:
        # - hidden_size: 1024 (vs 2048)
        # - intermediate_size: 4096 (vs 16384)
        # - head_dim: 128 (vs 256)

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,  # 128
            hidden_size=action_expert_config.width,  # 1024
            intermediate_size=action_expert_config.mlp_dim,  # 4096
            num_attention_heads=action_expert_config.num_heads,  # 8
            num_hidden_layers=action_expert_config.depth,  # 18
            num_key_value_heads=action_expert_config.num_kv_heads,  # 1 (GQA)
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # ================================================================
        # STEP 3: Initialize Models
        # ================================================================

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)

        # IMPORTANT: Remove embed_tokens from expert
        # The action expert doesn't embed tokens directly - it receives
        # pre-embedded action tokens from the π0 model
        self.gemma_expert.model.embed_tokens = None

        # ================================================================
        # STEP 4: Set Precision
        # ================================================================
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        """
        Convert model to bfloat16 while keeping critical params in float32.

        Why mixed precision?
        - bfloat16: 2x memory savings, faster matmuls on modern GPUs
        - float32 needed for: LayerNorm, Embeddings (numerical stability)

        Params kept in float32:
        ┌─────────────────────────────────────────────────────────────────┐
        │  - vision_tower.vision_model.embeddings.patch_embedding.weight │
        │  - vision_tower.vision_model.embeddings.patch_embedding.bias   │
        │  - vision_tower.vision_model.embeddings.position_embedding     │
        │  - input_layernorm (all layers)                                │
        │  - post_attention_layernorm (all layers)                       │
        │  - model.norm (final layer norm)                               │
        └─────────────────────────────────────────────────────────────────┘
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep critical parameters in float32 for numerical stability
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """
        Extract image features using SigLIP vision encoder.

        Input:  image [B, 3, 224, 224] - RGB image
        Output: image_embeds [B, 256, 2048] - 256 patch tokens with 2048 dim

        Processing:
        ┌─────────────────────────────────────────────────────────────┐
        │  image [B, 3, 224, 224]                                    │
        │           │                                                 │
        │           ▼                                                 │
        │  SigLIP ViT Encoder                                        │
        │  - Patch size: 14x14                                       │
        │  - (224/14)² = 256 patches                                 │
        │           │                                                 │
        │           ▼                                                 │
        │  Multi-modal Projector (Linear)                            │
        │  - Projects to Gemma dimension (2048)                      │
        │           │                                                 │
        │           ▼                                                 │
        │  image_embeds [B, 256, 2048]                               │
        └─────────────────────────────────────────────────────────────┘
        """
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        """
        Embed language tokens using PaliGemma's embedding layer.

        Input:  tokens [B, T] - Token IDs
        Output: embeds [B, T, 2048] - Token embeddings

        Note: This uses the Gemma 2B embedding table (vocab_size=257152)
        """
        return self.paligemma.language_model.embed_tokens(tokens)

    # region forward
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """
        Forward pass with three execution modes.

        Args:
            attention_mask: [B, 1, T, T] - PREFIX-LM attention pattern
            position_ids: [B, T] - Position indices for RoPE
            past_key_values: KV cache for inference
            inputs_embeds: [prefix_embeds, suffix_embeds]
                - prefix_embeds: [B, T_prefix, 2048] or None
                - suffix_embeds: [B, T_suffix, 1024] or None
            use_cache: Whether to return KV cache
            adarms_cond: [cond_vlm, cond_expert] - AdaRMSNorm conditioning

        Returns:
            outputs: [prefix_output, suffix_output]
            past_key_values: KV cache (only for MODE 1)

        ╔═══════════════════════════════════════════════════════════════╗
        ║                    EXECUTION MODE SELECTION                    ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║                                                               ║
        ║  inputs_embeds = [prefix, suffix]                             ║
        ║                                                               ║
        ║  MODE 1: [prefix, None]   → PREFIX-ONLY (KV cache prefill)   ║
        ║  MODE 2: [None, suffix]   → SUFFIX-ONLY (AR decoding)        ║
        ║  MODE 3: [prefix, suffix] → JOINT (training)                 ║
        ║                                                               ║
        ╚═══════════════════════════════════════════════════════════════╝
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # ================================================================
        # MODE 1: PREFIX-ONLY
        # ================================================================
        # Used during inference to process image + language and cache KV
        #
        # Flow:
        #   prefix_embeds → PaliGemma Language Model → (output, KV cache)
        #
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        # ================================================================
        # MODE 2: SUFFIX-ONLY
        # ================================================================
        # Used during inference for autoregressive action decoding
        # Uses KV cache from prefix to attend to image+language
        #
        # Flow:
        #   suffix_embeds + KV_cache → Gemma Expert → output
        #
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None

        # ================================================================
        # MODE 3: JOINT TRAINING
        # ================================================================
        # This is the core π0 training mode with shared attention
        #
        # Key Innovation:
        # - Both models process their inputs through the same attention
        # - Action tokens can attend to vision+language tokens
        # - Enables end-to-end gradient flow
        #
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # ============================================================
            # Gradient Checkpointing Setup
            # ============================================================
            # Reduces memory by recomputing activations during backward
            # Trade-off: ~30% slower but ~40% less memory

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # ============================================================
            # Joint Layer Computation Function
            # ============================================================
            # This function implements shared attention + separate FFN
            #
            # DETAILED FLOW per layer:
            #
            # ┌─────────────────────────────────────────────────────────┐
            # │  1. LayerNorm both inputs (separate norms)              │
            # │  2. Compute Q, K, V for both (separate projections)     │
            # │  3. Concatenate Q, K, V along sequence dimension        │
            # │  4. Apply RoPE to concatenated Q, K                     │
            # │  5. Compute attention (SHARED)                          │
            # │  6. Split attention output                              │
            # │  7. Apply O_proj + residual (separate)                  │
            # │  8. LayerNorm (post_attention, separate)                │
            # │  9. MLP (separate)                                      │
            # │  10. Residual connection (separate)                     │
            # └─────────────────────────────────────────────────────────┘

            def compute_layer_complete(
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
            ):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                # --------------------------------------------------------
                # STEP 1 & 2: LayerNorm + Q/K/V Projection (SEPARATE)
                # --------------------------------------------------------
                # Each model uses its own layernorm and projection weights
                #
                # Gemma 2B:  hidden_states [B, T_prefix, 2048]
                #            Q [B, 8, T_prefix, 256], K/V [B, 1, T_prefix, 256]
                #
                # Gemma 300M: hidden_states [B, T_suffix, 1024]
                #            Q [B, 8, T_suffix, 128], K/V [B, 1, T_suffix, 128]

                query_states = []
                key_states = []
                value_states = []
                gates = []

                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]

                    # Apply input LayerNorm (returns gate for gated residual)
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                    gates.append(gate)

                    # Compute Q, K, V projections
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                    # Q: [B, T, hidden] → [B, T, num_heads * head_dim] → [B, num_heads, T, head_dim]
                    query_state = (
                        layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    )
                    # K: [B, T, hidden] → [B, T, num_kv_heads * head_dim] → [B, num_kv_heads, T, head_dim]
                    key_state = (
                        layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    )
                    # V: same as K
                    value_state = (
                        layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    )

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # --------------------------------------------------------
                # STEP 3: Concatenate Q, K, V (SHARED ATTENTION)
                # --------------------------------------------------------
                # Concatenate along sequence dimension (dim=2)
                #
                # Before: Q_prefix [B, 8, T_prefix, 256], Q_suffix [B, 8, T_suffix, 128]
                # After:  Q_concat [B, 8, T_total, head_dim]
                #
                # NOTE: This implementation assumes head_dim matches between models
                # The original π0 paper handles different head_dims differently

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # --------------------------------------------------------
                # STEP 4: Apply Rotary Position Embeddings (RoPE)
                # --------------------------------------------------------
                # RoPE encodes position information into Q and K
                # Using PaliGemma's rotary_emb for both models

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(
                    dummy_tensor, position_ids
                )
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # --------------------------------------------------------
                # STEP 5: Shared Attention Computation
                # --------------------------------------------------------
                # Uses PREFIX-LM attention pattern:
                # - Prefix tokens: bidirectional attention to all prefix
                # - Suffix tokens: causal attention (can see prefix + prior suffix)
                #
                # attention_mask enforces this pattern

                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )

                # Reshape attention output
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # --------------------------------------------------------
                # STEP 6-10: Split and Process Separately
                # --------------------------------------------------------
                # After shared attention, split output back to prefix/suffix
                # Then apply separate O_proj, LayerNorm, MLP, and residuals

                outputs_embeds = []
                start_pos = 0

                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    # STEP 6: Split attention output
                    # Extract this model's portion of the attention output

                    # Handle dtype conversion if needed
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                    # STEP 7: O_proj + First Residual
                    # O_proj: [B, T, num_heads * head_dim] → [B, T, hidden_size]
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # Gated residual connection (if AdaRMSNorm is used)
                    # out = residual + gate * layer_output
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                    after_first_residual = out_emb.clone()

                    # STEP 8: Post-attention LayerNorm
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])

                    # Convert to bfloat16 if the MLP uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    # STEP 9: MLP (Feed-Forward Network)
                    # Gemma uses GeGLU: FFN(x) = GELU(x·W_gate) ⊙ (x·W_up) · W_down
                    out_emb = layer.mlp(out_emb)

                    # STEP 10: Second Residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # ============================================================
            # Process All Layers
            # ============================================================
            # Iterate through 18 layers, applying gradient checkpointing if enabled

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # Gradient checkpointing: recompute activations during backward
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            # ============================================================
            # Final LayerNorm
            # ============================================================
            # Apply final RMSNorm to both outputs

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        # ================================================================
        # Return Results
        # ================================================================
        # Returns: [prefix_output, suffix_output], past_key_values
        #
        # prefix_output: [B, T_prefix, 2048] - Hidden states for vision+language
        # suffix_output: [B, T_suffix, 1024] - Hidden states for actions
        # past_key_values: KV cache (only populated in MODE 1)

        return [prefix_output, suffix_output], prefix_past_key_values
