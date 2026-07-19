#!/bin/bash
# =============================================================================
# ac-vggt — VGGT feed-forward 3D-reconstruction environment
# =============================================================================
# Creates the `ac-vggt` conda env (Python 3.11) for the model_vggt nodeset:
# VGGT (Visual Geometry Grounded Transformer) reconstructs camera poses, dense
# depth, and a world point map from 1..N RGB views in a single forward pass,
# plus optional point tracking.
#
# Why a DEDICATED env (not the shared ac-fm):
#   VGGT pins numpy<2. ac-fm runs numpy 2.x for CLIP / Depth-Anything / SAM /
#   DINOv2 / the VLMs. Installing VGGT into ac-fm would downgrade numpy under
#   those residents and break them — the split is mandatory, not cosmetic.
#
# Version rationale (2026-07-07):
#   - torch 2.8.0+cu126 / torchvision 0.23.0+cu126 — same cu126 line as ac-fm,
#     chosen for Docker distribution reach. VGGT runs on torch SDPA attention;
#     no flash-attn is needed (unlike ac-fm).
#   - vggt — git-only (no PyPI). Pinned to the exact upstream commit this env
#     was verified with, for reproducible builds. Its own metadata drags in
#     numpy<2, einops, huggingface_hub, opencv-python, Pillow, safetensors;
#     the numpy pin is what forces the separate env (see above).
#   - Python 3.11 — matches ac-fm; full wheel coverage.
#
# No weights are downloaded here. `facebook/VGGT-1B` (~5 GB) auto-downloads to
# the HF cache on first `VGGT.from_pretrained`; it is cc-by-nc-4.0 (RESEARCH
# ONLY). `facebook/VGGT-1B-Commercial` is the commercial-licensed variant —
# set the node's `model_id` config to it for commercial use.
#
# Usage:
#   bash scripts/install/install_ac_vggt.sh
#
# The model_vggt nodeset resolves this env via conda_env_python("ac-vggt",
# "VGGT_PYTHON"); override the interpreter with $VGGT_PYTHON and the device
# with $VGGT_DEVICE (auto -> cuda).
#
# Reproducible install: scripts/install/envs/ac_vggt.lock is the frozen
# package set this env was verified with (pip install -r it inside a
# Dockerfile for byte-stable images).
# =============================================================================
set -euo pipefail

ENV_NAME="ac-vggt"
PY_VER="3.11"
# Upstream commit this env was verified against (facebookresearch/vggt).
VGGT_COMMIT="a288dd0f14786c93483e45524328726ab7b1b4ce"

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

echo "[3/5] Installing VGGT (git @ ${VGGT_COMMIT:0:7}) — drags in numpy<2, einops, hf_hub, opencv, safetensors"
pip install "git+https://github.com/facebookresearch/vggt.git@${VGGT_COMMIT}"

echo "[4/5] Installing server stack (auto_host serves the nodeset via FastAPI/uvicorn, ADR-server-001)"
# msgpack is the default /call transport (app/server/serialization.py); httpx
# backs the loopback proxy paths in app/server/*.
pip install \
    "uvicorn==0.39.0" \
    "fastapi==0.128.8" \
    "httpx==0.28.1" \
    "msgpack==1.2.1"

echo "[5/5] Sanity spike (imports + numpy<2 guard; no weights)"
python - <<'EOF'
import numpy, torch
assert numpy.__version__.startswith("1."), f"numpy must be <2 for VGGT, got {numpy.__version__}"
print("torch", torch.__version__, "| numpy", numpy.__version__)
# The exact symbols model_vggt.py imports — catch any upstream drift here.
from vggt.models.vggt import VGGT  # noqa: F401
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: F401
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: F401
import fastapi, uvicorn, httpx, msgpack  # noqa: F401  (server stack)
print("ac-vggt sanity spike passed (weights fetched lazily on first from_pretrained)")
EOF

echo "Done. Env: ${ENV_NAME} — override the interpreter with \$VGGT_PYTHON, device with \$VGGT_DEVICE."
