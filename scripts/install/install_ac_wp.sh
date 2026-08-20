#!/bin/bash
# =============================================================================
# ac-wp — habitat-free waypoint-predictor server env (coding-agent wp/hybrid)
# =============================================================================
# Creates the `ac-wp` conda env that hosts the SmartWay waypoint predictor
# (`SmartWayWaypointNodeSet`, auto_host :9210) for the coding-agent wp /
# hybrid / imagine arms, over the habitat-free shim tree
#   coding-agent/exp_workspace/wp/ac_wp_predictor_shim/   (SMARTWAY_REPO_PATH)
#
# Why a separate env: the predictor's stock home `ac-smartway` is pinned to
# py3.8 + torch 2.1.1+cu121 (habitat-sim 0.1.7 import chain) and cannot drive
# sm_120 cards (RTX 5090 -> cu128 -> torch >= 2.7 -> py >= 3.9). The shim cuts
# the habitat imports, so this env is torch + gym + pytorch-transformers +
# the backend wire stack — no simulator. On older cards (3090, cu121) the
# shim also runs under ac-smartway; this env is only REQUIRED on sm_120.
#
# Spec: scripts/install/envs/ac_wp.yaml — pins mirror the verified reference
# env on an RTX 5090 (sm_120) dev box (`conda create python=3.10` + pip,
# built 2026-07-15, captured 2026-08-19).
#
# Usage:
#   bash scripts/install/install_ac_wp.sh
#
# Prerequisites:
#   - mamba or conda
#   - NVIDIA driver with CUDA 12.8 runtime support (>= 570.x)
#   - checkpoints: data/smartway/waypoint_ckpt/best.pth +
#     data/smartway/ddppo/gibson-2plus-resnet50.pth
#     (bash scripts/data/fetch_ckpt_smartway.sh — shared with ac-smartway)
#   - the smartway vendored tree (the shim symlinks into it):
#     workspace/nodesets/method/smartway_waypoint/_vendored/waypoint_predictor
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_YAML="$SCRIPT_DIR/envs/ac_wp.yaml"
SHIM_DIR="$PROJECT_ROOT/coding-agent/exp_workspace/wp/ac_wp_predictor_shim"

echo "=== ac-wp (waypoint predictor server) Environment Installation ==="
echo "Project root: $PROJECT_ROOT"
echo "Env spec:     $ENV_YAML"
echo "Shim tree:    $SHIM_DIR"
echo ""

# ── Step 0: Prerequisites ──
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "[ERROR] Neither mamba nor conda found. Install miniforge/mamba first."
    exit 1
fi
echo "Using: $CONDA_CMD"

if [ ! -d "$SHIM_DIR" ]; then
    echo "[ERROR] shim tree missing: $SHIM_DIR"
    exit 1
fi
SMARTWAY_VENDORED="$PROJECT_ROOT/workspace/nodesets/method/smartway_waypoint/_vendored/waypoint_predictor"
if [ ! -d "$SMARTWAY_VENDORED" ]; then
    echo "[WARN] $SMARTWAY_VENDORED missing — the shim's waypoint_predictor/ symlink"
    echo "       dangles until it exists. Re-vendor via"
    echo "       workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh."
fi

# ── Step 1: Create conda env ──
echo ""
echo "=== Step 1: Creating conda env from $ENV_YAML ==="
if $CONDA_CMD env list | grep -qE '^\s*ac-wp\s'; then
    echo "  [skip] ac-wp env already exists — use 'conda env remove -n ac-wp -y' to recreate"
else
    cd "$PROJECT_ROOT"
    $CONDA_CMD env create -f "$ENV_YAML"
fi

WP_PYTHON="$(conda run -n ac-wp which python 2>/dev/null || true)"
if [ -z "$WP_PYTHON" ]; then
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        [ -x "$base/envs/ac-wp/bin/python" ] && WP_PYTHON="$base/envs/ac-wp/bin/python" && break
    done
fi
if [ -z "$WP_PYTHON" ]; then
    echo "[ERROR] cannot locate the ac-wp python"
    exit 1
fi
echo "ac-wp Python: $WP_PYTHON"

# ── Step 2: Checkpoints (shared with ac-smartway; guarded) ──
echo ""
echo "=== Step 2: SmartWay checkpoints (data/smartway/) ==="
if [ -f "$PROJECT_ROOT/data/smartway/waypoint_ckpt/best.pth" ] && \
   [ -e "$PROJECT_ROOT/data/smartway/ddppo/gibson-2plus-resnet50.pth" ]; then
    echo "  present: waypoint_ckpt/best.pth + ddppo/gibson-2plus-resnet50.pth"
else
    PYTHON="$WP_PYTHON" bash "$SCRIPT_DIR/../data/fetch_ckpt_smartway.sh" \
        || echo "[WARN] ckpt fetch failed — env installed OK; fetch later: bash scripts/data/fetch_ckpt_smartway.sh"
fi

# ── Step 3: Verify (bare env python, no conda hooks — how auto_host spawns) ──
echo ""
echo "=== Step 3: Verifying installation ==="
echo -n "  Python: "
"$WP_PYTHON" -c "import sys; print(sys.version.split()[0])"
echo -n "  PyTorch: "
"$WP_PYTHON" -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| available:', torch.cuda.is_available())" 2>&1
echo -n "  habitat absent (shim is the point): "
"$WP_PYTHON" -c "import importlib.util as u; print('OK' if u.find_spec('habitat') is None else 'WARN habitat importable')" 2>&1
echo -n "  shim import chain (SMARTWAY_REPO_PATH=$SHIM_DIR): "
SMARTWAY_REPO_PATH="$SHIM_DIR" "$WP_PYTHON" -s - << 'PYEOF' 2>&1 || echo "WARN"
import os, sys
p = os.environ["SMARTWAY_REPO_PATH"]
sys.path.insert(0, p)
sys.path.insert(0, os.path.join(p, "waypoint_predictor"))
import vlnce_baselines.models.encoders.resnet_encoders as r   # shim copy (torch/gym only)
from waypoint_predictor.TRM_net import BinaryDistPredictor_TRM  # TRM + pytorch_transformers
from waypoint_predictor.utils import nms
print("OK", r.ResNetEncoder.__module__)
PYEOF
echo -n "  mcp (/mcp projection, optional): "
"$WP_PYTHON" -c "import mcp; print('OK')" 2>&1 || echo "absent — auto_host runs manifest-only"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Set:    export WP_PYTHON=$WP_PYTHON"
echo "Or:     conda activate ac-wp"
echo ""
echo "Launch the predictor server (see coding-agent/README.md):"
echo "  cd $PROJECT_ROOT/agentcanvas/backend && PYTHONPATH=\$PWD:\$PWD/../.. \\"
echo "    SMARTWAY_REPO_PATH=$SHIM_DIR \\"
echo "    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\"
echo "    $WP_PYTHON -m app.server.auto_host \\"
echo "    --file ../../workspace/nodesets/method/smartway_waypoint/__init__.py \\"
echo "    --class SmartWayWaypointNodeSet --port 9210"
echo ""

# ── Install-time server probes (2026-07-31 campaign; see lib/server_probe.sh) ──
source "$SCRIPT_DIR/lib/server_probe.sh"
probe_server_stack "$WP_PYTHON"
SMARTWAY_REPO_PATH="$SHIM_DIR" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    probe_auto_host "$WP_PYTHON" "$PROJECT_ROOT/workspace/nodesets/method/smartway_waypoint/__init__.py" "SmartWayWaypointNodeSet"
