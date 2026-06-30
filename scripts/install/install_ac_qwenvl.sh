#!/bin/bash
# =============================================================================
# Qwen2.5-VL Environment Installation Script
# =============================================================================
# Creates the `ac-qwenvl` conda env (Python 3.10) for serving
# Qwen2.5-VL as the ReAct reasoning + VQA VLM behind the ToolEQA method
# nodeset.
#
# Used in server mode by the `vlm_qwen2_5_vl` nodeset at
# `workspace/nodesets/server/vlm_qwen2_5_vl.py`. ToolEQA's reasoning loop
# calls its `generate` node over the standard server-mode HTTP route.
#
# Why 3B (not the paper's 7B `worker()` default):
#   This box has a single RTX 3090 (24 GB). ToolEQA must co-host Qwen-VL +
#   DetAny3D (~10 GB) + Habitat-sim. 7B (~16 GB) won't co-fit; 3B (~7 GB)
#   does, and `config/react-eqa.yaml` (the upstream HM-EQA config) itself
#   specifies the 3B variant. Override QWENVL_MODEL_DIR for 7B on a bigger GPU.
#
# Workspace-standalone: no reference to `third_party/`. Deps from PyPI;
# weights from HuggingFace under `data/qwen2_5_vl/`.
#
# Usage:
#   bash scripts/install/install_ac_qwenvl.sh
#
# After install:
#   export QWENVL_PYTHON=/home/$(whoami)/miniforge3/envs/ac-qwenvl/bin/python
#
# Prerequisites:
#   - mamba or conda
#   - NVIDIA GPU + recent CUDA driver (torch cu121)
#   - ~8 GB disk for the 3B weights
# =============================================================================
set -euo pipefail

ENV_NAME="ac-qwenvl"
PY_VER="3.10"
MODEL_REPO="Qwen/Qwen2.5-VL-3B-Instruct"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/data/qwen2_5_vl/Qwen2.5-VL-3B-Instruct"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[skip] conda env '${ENV_NAME}' already exists."
else
    echo "[1/4] Creating conda env '${ENV_NAME}' (Python ${PY_VER})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[2/4] Installing torch (cu121) + Qwen2.5-VL deps"
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# Qwen2.5-VL needs transformers>=4.49; pin to the upstream-tested line.
pip install \
    "transformers==4.50.2" \
    "qwen-vl-utils==0.0.8" \
    "accelerate>=0.30" \
    "huggingface_hub>=0.24" \
    pillow numpy
# Server stack — auto_host serves this nodeset via FastAPI/uvicorn in server
# mode (ADR-server-001); httpx for the loopback proxy. Without these the
# auto_host subprocess dies on import ("No module named 'uvicorn'") and the
# parent waits out the full health-check timeout.
pip install \
    "uvicorn==0.39.0" \
    "fastapi==0.128.8" \
    "httpx==0.28.1"

echo "[3/4] flash-attn (best-effort; falls back to sdpa attention if it fails)"
pip install flash-attn==2.7.2.post1 --no-build-isolation || \
    echo "[warn] flash-attn install failed — nodeset will use sdpa attention."

echo "[4/4] Downloading weights ${MODEL_REPO} -> ${WEIGHTS_DIR}"
mkdir -p "${WEIGHTS_DIR}"
huggingface-cli download "${MODEL_REPO}" --local-dir "${WEIGHTS_DIR}"

echo
echo "Done. Add to your shell:"
echo "  export QWENVL_PYTHON=${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
echo "  export QWENVL_MODEL_DIR=${WEIGHTS_DIR}"
