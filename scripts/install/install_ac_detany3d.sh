#!/bin/bash
# =============================================================================
# DetAny3D Environment Installation Script
# =============================================================================
# Creates the `detany3d` conda env (Python 3.9 + CUDA 11.8) for DetAny3D
# (3D detection model used by ToolEQA).
#
# Used in server mode by `model_detany3d` nodeset at
# `workspace/nodesets/model/model_detany3d/` (folder form; DetAny3D source
# is vendored locally inside `_vendor/`).
#
# Workspace-standalone: this script does NOT reference `third_party/`. All
# Python dependencies are pip-installed from PyPI / GitHub. Model weights
# live under `data/detany3d/weights/`.
#
# Why separate from `hmeqa`:
#   DetAny3D pins specific torch + flash-attn + GroundingDINO + UniDepth +
#   SAM versions. Mixing into hmeqa risks habitat-sim ABI breakage.
#
# Usage:
#   bash scripts/install/install_ac_detany3d.sh
#
# After install:
#   export DETANY3D_PYTHON=/home/$(whoami)/miniforge3/envs/ac-detany3d/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - NVIDIA GPU + CUDA 11.8 driver (DetAny3D pins torch+cu118)
#   - ~10-15 GB disk for model weights
#
# Models downloaded to data/detany3d/weights/:
#   - GroundingDINO Swin-B  (~700 MB)  groundingdino_swinb_cogcoor.pth
#   - SAM ViT-H              (~2.4 GB)  sam_vit_h_4b8939.pth
#   - DetAny3D checkpoint    (~3-5 GB)  see _vendor/UPSTREAM_README.md
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NODESET_ROOT="$PROJECT_ROOT/workspace/nodesets/model/model_detany3d"
VENDOR_ROOT="$NODESET_ROOT/_vendor"
DATA_ROOT="$PROJECT_ROOT/data/detany3d"
WEIGHTS_DIR="$DATA_ROOT/weights"
ENV_NAME="ac-detany3d"

echo "=== DetAny3D Environment Installation ==="
echo "Project root:   $PROJECT_ROOT"
echo "Vendor root:    $VENDOR_ROOT"
echo "Weights dir:    $WEIGHTS_DIR"
echo ""

# ── Step 0: Check prerequisites ──

if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "[ERROR] Neither mamba nor conda found. Install miniforge/mamba first."
    exit 1
fi
echo "Using: $CONDA_CMD"

if [ ! -d "$VENDOR_ROOT/detect_anything" ]; then
    echo "[ERROR] DetAny3D source not vendored at $VENDOR_ROOT"
    echo "  The folder-form nodeset workspace/nodesets/model/model_detany3d/"
    echo "  must contain _vendor/{detect_anything,wrap_model.py,train_utils.py,utils.py}."
    exit 1
fi

# ── Step 1: Create conda environment ──

echo ""
echo "=== Step 1: Creating conda env '$ENV_NAME' ==="
$CONDA_CMD env remove -n "$ENV_NAME" -y 2>/dev/null || true
$CONDA_CMD create -n "$ENV_NAME" python=3.9 -y

DETANY3D_PYTHON="/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
if [ ! -f "$DETANY3D_PYTHON" ]; then
    DETANY3D_PYTHON="$(conda run -n $ENV_NAME which python)"
fi
DETANY3D_PIP="${DETANY3D_PYTHON%/python}/pip"
echo "detany3d Python: $DETANY3D_PYTHON"

# ── Step 2: Install PyTorch + CUDA 11.8 (DetAny3D pinning) ──

echo ""
echo "=== Step 2: Installing PyTorch 2.1 + CUDA 11.8 ==="
$DETANY3D_PIP install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# ── Step 3: Install vendored DetAny3D requirements ──

echo ""
echo "=== Step 3: Installing DetAny3D requirements (from vendored copy) ==="
if [ -f "$VENDOR_ROOT/requirements.txt" ]; then
    $DETANY3D_PIP install -r "$VENDOR_ROOT/requirements.txt" || \
        echo "[WARN] some entries in vendored requirements.txt failed — continuing"
fi

# Common deps shared by model_detany3d/__init__.py + vendored model code
$DETANY3D_PIP install \
    flask==3.0.0 \
    pyyaml \
    python-box \
    pillow \
    opencv-python \
    omegaconf \
    open3d \
    scipy

# ── Step 4: Install GroundingDINO from PyPI / upstream GitHub ──

echo ""
echo "=== Step 4: Installing GroundingDINO ==="
# `groundingdino-py` is the community PyPI fork; falls back to direct GitHub install.
$DETANY3D_PIP install groundingdino-py || \
    $DETANY3D_PIP install "git+https://github.com/IDEA-Research/GroundingDINO.git"

# ── Step 5: Install UniDepth from upstream GitHub ──

echo ""
echo "=== Step 5: Installing UniDepth ==="
$DETANY3D_PIP install \
    "git+https://github.com/lpiccinelli-eth/UniDepth.git" \
    --extra-index-url https://download.pytorch.org/whl/cu118 || \
    echo "[WARN] UniDepth install failed; manual install may be needed"

# ── Step 6: Install Segment Anything ──

echo ""
echo "=== Step 6: Installing Segment Anything ==="
$DETANY3D_PIP install "git+https://github.com/facebookresearch/segment-anything.git"

# ── Step 7: Download model weights ──

echo ""
echo "=== Step 7: Downloading model weights ==="
mkdir -p "$WEIGHTS_DIR"

# GroundingDINO Swin-B
DINO_WEIGHTS="$WEIGHTS_DIR/groundingdino_swinb_cogcoor.pth"
if [ ! -f "$DINO_WEIGHTS" ]; then
    echo "Downloading GroundingDINO Swin-B (~700 MB)..."
    wget -O "$DINO_WEIGHTS" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
fi

# SAM ViT-H
SAM_WEIGHTS="$WEIGHTS_DIR/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_WEIGHTS" ]; then
    echo "Downloading SAM ViT-H (~2.4 GB)..."
    wget -O "$SAM_WEIGHTS" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# DetAny3D checkpoint — see vendored UPSTREAM_README.md for the download URL.
echo ""
echo "[NOTE] DetAny3D model checkpoint must be downloaded manually."
echo "       See $VENDOR_ROOT/UPSTREAM_README.md (or the upstream paper's"
echo "       project page at https://tooleqa.github.io)."
echo "       Place the checkpoint under $WEIGHTS_DIR/ and update the"
echo "       cfg.resume + cfg.model.checkpoint paths in"
echo "       $VENDOR_ROOT/detect_anything/configs/demo.yaml"
echo "       to point at $WEIGHTS_DIR/<filename>."

# ── Step 8: Smoke test ──

echo ""
echo "=== Step 8: Smoke test (no model load) ==="
$DETANY3D_PYTHON -c "
import sys
sys.path.insert(0, '$VENDOR_ROOT')
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
import groundingdino
print('groundingdino:', groundingdino.__file__)
print('SMOKE OK')
"

echo ""
echo "=== DetAny3D Environment Installation Complete ==="
echo ""
echo "Add to your shell rc:"
echo "  export DETANY3D_PYTHON=$DETANY3D_PYTHON"
echo ""
echo "Then load model_detany3d from the AgentCanvas NodeSet Manager."
