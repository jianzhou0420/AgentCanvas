#!/bin/bash
# =============================================================================
# ac-fm — shared Foundation-Model environment
# =============================================================================
# Creates the `ac-fm` conda env (Python 3.11): the common landing zone for
# foundation-model nodesets that only declare *lower-bound* requirements
# (BLIP-2, InstructBLIP, DINOv2/DINOv3, Grounding-DINO hf backend, OWLv2,
# SigLIP2, AIMv2, Qwen2.5-VL / Qwen3-VL, InternVL3, Gemma3, SmolVLM2).
# Models whose code pins *exact* old versions (Prismatic, SpatialBot/Bunny
# remote code, DetAny3D vendored stack) stay in their dedicated envs — they
# can never share. RAM/RAM++ was evaluated and reclassified frozen
# (2026-07-05): its vendored BERT + utils pin five distinct old APIs
# (transformers modeling_utils symbols, tokenizer attrs, post_init/tied
# weights, get_head_mask, scipy interp2d) — it stays in ac-ram.
#
# Version rationale (2026-07-05):
#   - torch 2.8.0+cu126 — newest torch with prebuilt flash-attn wheels
#     (flash-attn 2.8.3 ships cu12torch2.4..2.8 only); cu126 is the
#     mainstream CUDA-12 line, chosen for Docker distribution reach.
#   - transformers 5.13.0 — current 5.x line; all residents pass an
#     import + processor spike.
#   - Python 3.11 — 3.10 EOLs 2026-10; 3.11 has full wheel coverage.
#
# flash-attn note: the prebuilt wheel needs glibc >= 2.32. On older hosts
# (this box: 2.31) the package installs but its import fails — nodesets
# catch that and fall back to sdpa attention. Inside a Docker image with a
# newer-glibc base the same wheel activates flash-attention automatically.
#
# No weights are downloaded here — model checkpoints stay under data/ (or
# the HF cache) and are volume-mounted in the Docker distribution picture.
# SAM 3 weights (facebook/sam3, used by model_sam variant=sam3) are HF-GATED:
# request access on the model page with the deploying account first, or the
# sam3 engine runs degraded (empty outputs). sam2.1 weights are ungated.
#
# Usage:
#   bash scripts/install/install_ac_fm.sh
#
# Since 2026-07-05 (GPU parity gate passed) the five resident nodesets
# DEFAULT to this env — no exports needed. Per-nodeset overrides remain:
# $BLIP2_PYTHON, $INSTRUCTBLIP_PYTHON, $DINOV2_PYTHON,
# $GROUNDING_DINO_PYTHON (hf_tiny backend; native stays on ac-detany3d),
# $QWENVL_PYTHON.
#
# Reproducible install: scripts/install/envs/ac_fm.lock is the frozen
# package set this env was verified with (pip install -r it inside a
# Dockerfile for byte-stable images).
# =============================================================================
set -euo pipefail

ENV_NAME="ac-fm"
PY_VER="3.11"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[skip] conda env '${ENV_NAME}' already exists."
else
    echo "[1/5] Creating conda env '${ENV_NAME}' (Python ${PY_VER})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[2/5] Installing torch 2.8.0 (cu126)"
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

echo "[3/5] Installing transformers 5.x + FM resident deps"
pip install \
    "transformers==5.13.0" \
    "accelerate>=1.1" \
    timm sentencepiece einops \
    "qwen-vl-utils==0.0.14" \
    num2words \
    scipy \
    pillow numpy
# num2words: required by the SmolVLM2 processor (spells out numbers in the
# prompt template); pure-Python, no build step.
# SAM 1 (model_sam nodeset) — pure-Python, lower-bound-only deps;
# opencv-headless is required by SamAutomaticMaskGenerator's postprocessing.
pip install \
    "segment-anything==1.0" \
    opencv-python-headless
# Server stack — auto_host serves these nodesets via FastAPI/uvicorn in
# server mode (ADR-server-001); httpx for the loopback proxy; msgpack is
# the default /call transport (app/server/serialization.py).
pip install \
    "uvicorn==0.39.0" \
    "fastapi==0.128.8" \
    "httpx==0.28.1" \
    msgpack

echo "[4/5] flash-attn (best-effort; sdpa fallback if import fails)"
pip install flash-attn==2.8.3 --no-build-isolation || \
    echo "[warn] flash-attn install failed — nodesets will use sdpa attention."

echo "[5/5] Sanity spike (imports + no weights)"
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
from transformers import (  # noqa: F401
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    AutoModelForZeroShotObjectDetection,
    Qwen2_5_VLForConditionalGeneration,
    # wave-3 residents (2026-07-08):
    Siglip2Model,
    Owlv2ForObjectDetection,
    InternVLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    SmolVLMForConditionalGeneration,
)
import qwen_vl_utils  # noqa: F401
import num2words  # noqa: F401  (SmolVLM2 processor dep)
try:
    import flash_attn  # noqa: F401
    print("flash-attn: active")
except Exception:
    print("flash-attn: import failed (old glibc?) — sdpa fallback will engage")
print("ac-fm sanity spike passed")
EOF

echo "Done. Env: ${ENV_NAME} — see header for per-nodeset *_PYTHON overrides."
