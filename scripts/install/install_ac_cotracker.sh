#!/bin/bash
# =============================================================================
# ac-cotracker — CoTracker point-tracking environment
# =============================================================================
# Creates the `ac-cotracker` conda env (Python 3.11) for the model_cotracker
# nodeset: CoTracker3 (Meta) tracks points densely through a video / frame
# sequence, handling occlusion and long-range motion — the temporal
# correspondence primitive that complements VGGT's single-pass track head.
#
# Why a DEDICATED env (not the shared ac-fm):
#   CoTracker installs from git (the `cotracker` package is pinned to a commit,
#   not published on PyPI, and is not a transformers model). It is kept in its
#   own env to keep the shared ac-fm transformers stack clean and reproducible.
#   Unlike ac-vggt this is NOT a numpy<2-forced split — CoTracker is
#   numpy-2-compatible — it is a provenance / cleanliness choice.
#
# Version rationale (2026-07-07):
#   - torch 2.8.0+cu126 / torchvision 0.23.0+cu126 — same cu126 line as ac-fm /
#     ac-vggt, for Docker distribution reach. No flash-attn (not needed).
#   - cotracker — git-only. Pinned to the exact upstream commit this env was
#     verified with, for reproducible builds. Its install is lightweight
#     (torch, numpy, Pillow); the imageio/matplotlib visualization extras are
#     not pulled by the predictor path.
#   - Python 3.11 — matches ac-fm / ac-vggt.
#
# No weights are downloaded here. `scaled_offline.pth` (`facebook/cotracker3`
# on HF) is fetched lazily to the torch-hub cache on first track; the nodeset
# loads it via `CoTrackerPredictor(checkpoint=…)` on the installed package —
# the co-tracker repo is NEVER cloned (unlike a bare `torch.hub.load`).
#
# Usage:
#   bash scripts/install/install_ac_cotracker.sh
#
# The model_cotracker nodeset resolves this env via
# conda_env_python("ac-cotracker", "COTRACKER_PYTHON"); override the interpreter
# with $COTRACKER_PYTHON and the device with $COTRACKER_DEVICE (auto -> cuda).
#
# Reproducible install: scripts/install/envs/ac_cotracker.lock is the frozen
# package set this env was verified with (pip install -r it inside a
# Dockerfile for byte-stable images).
# =============================================================================
set -euo pipefail

ENV_NAME="ac-cotracker"
PY_VER="3.11"
# Upstream commit this env was verified against (facebookresearch/co-tracker).
COTRACKER_COMMIT="82e02e8029753ad4ef13cf06be7f4fc5facdda4d"

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

echo "[3/5] Installing CoTracker (git @ ${COTRACKER_COMMIT:0:7})"
pip install "git+https://github.com/facebookresearch/co-tracker.git@${COTRACKER_COMMIT}"

echo "[4/5] Installing server stack (auto_host serves the nodeset via FastAPI/uvicorn, ADR-server-001)"
# msgpack is the default /call transport (app/server/serialization.py); httpx
# backs the loopback proxy paths in app/server/*.
pip install \
    "uvicorn==0.39.0" \
    "fastapi==0.128.8" \
    "httpx==0.28.1" \
    "msgpack==1.2.1"

echo "[5/5] Sanity spike (imports; no weights)"
python - <<'EOF'
import numpy, torch
print("torch", torch.__version__, "| numpy", numpy.__version__)
# The symbol model_cotracker.py loads — catch upstream drift here.
from cotracker.predictor import CoTrackerPredictor  # noqa: F401
import fastapi, uvicorn, httpx, msgpack  # noqa: F401  (server stack)
print("ac-cotracker sanity spike passed (weights fetched lazily on first track)")
EOF

echo "Done. Env: ${ENV_NAME} — override the interpreter with \$COTRACKER_PYTHON, device with \$COTRACKER_DEVICE."
