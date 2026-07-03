#!/bin/bash
# =============================================================================
# VLN-CE Environment Installation Script
# =============================================================================
# Creates the vlnce conda env for Habitat-Sim 0.1.7 (Python 3.8).
# This env is used in server mode by env_habitat and policy_adapter_vlnce nodesets.
#
# Usage:
#   bash scripts/install/install_ac_vlnce.sh
#
# Prerequisites:
#   - mamba or conda installed
#   - NVIDIA GPU driver with EGL support
#   - (third_party/VLN-CE + third_party/habitat-lab are auto-cloned + pinned)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_YAML="$SCRIPT_DIR/envs/ac_vlnce.yaml"

echo "=== VLN-CE Environment Installation ==="
echo "Project root: $PROJECT_ROOT"
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

# Fetch third_party sources (formerly git submodules; now pinned clones —
# commit IDs live in scripts/install/lib/thirdparty.sh).
source "$SCRIPT_DIR/lib/thirdparty.sh"
ensure_thirdparty VLN-CE
ensure_thirdparty habitat-lab

# Apply our patches to the VLN-CE submodule (idempotent; required by Step 2's
# `pip install -e third_party/VLN-CE`, which reads our added setup.py).
echo ""
bash "$SCRIPT_DIR/patches/apply_thirdparty_patches.sh"

# ── Step 1: Create conda environment ──

echo ""
echo "=== Step 1: Creating conda environment from $ENV_YAML ==="
$CONDA_CMD env remove -n ac-vlnce -y 2>/dev/null || true
$CONDA_CMD env create -f "$ENV_YAML"

VLNCE_PYTHON="/home/$(whoami)/miniforge3/envs/ac-vlnce/bin/python"
if [ ! -f "$VLNCE_PYTHON" ]; then
    # Fallback: find the env wherever conda put it
    VLNCE_PYTHON="$(conda run -n ac-vlnce which python)"
fi
CONDA_PREFIX="$(dirname "$(dirname "$VLNCE_PYTHON")")"
echo "vlnce Python: $VLNCE_PYTHON"

# ── Step 2: Install VLN-CE (editable) ──

echo ""
echo "=== Step 2: Installing VLN-CE package ==="
"$VLNCE_PYTHON" -m pip install -e "$PROJECT_ROOT/third_party/VLN-CE" 2>&1 | tail -3

# ── Step 3: Remove conda OpenGL libs (use system NVIDIA drivers) ──

echo ""
echo "=== Step 3: Removing conda OpenGL libs (use NVIDIA drivers instead) ==="
echo "Without this, habitat-sim crashes with: GL::Context: cannot retrieve OpenGL version"
$CONDA_CMD remove -n ac-vlnce libgl libglvnd libglx libegl --force -y 2>/dev/null || true

# ── Step 4: Setup data symlinks ──

echo ""
echo "=== Step 4: Setting up data symlinks ==="
VLNCE_DATA="$PROJECT_ROOT/third_party/VLN-CE/data"
mkdir -p "$VLNCE_DATA"   # keep parent real to preserve submodule-tracked connectivity_graphs.pkl + res/
for subdir in datasets scene_datasets checkpoints ddppo-models; do
    src="$PROJECT_ROOT/data/habitat/$subdir"
    dst="$VLNCE_DATA/$subdir"
    target_rel="../../../data/habitat/$subdir"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "  [WARN] $dst is a real directory (not a symlink). Skipping — manual intervention required."
        continue
    fi
    if [ -L "$dst" ]; then
        current="$(readlink "$dst")"
        if [ "$current" = "$target_rel" ]; then
            echo "  Already linked: $subdir"
            continue
        fi
        rm "$dst"
    fi
    if [ -d "$src" ]; then
        ln -s "$target_rel" "$dst"
        echo "  Linked: third_party/VLN-CE/data/$subdir -> data/habitat/$subdir"
    else
        echo "  [SKIP] $subdir (not present — run: bash scripts/data/fetch_data_vlnce.sh --$subdir)"
    fi
done

# ── Step 5: Setup LD_LIBRARY_PATH activation hook ──

echo ""
echo "=== Step 5: Setting up environment activation hooks ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF

# ── Step 6: Verify installation ──

echo ""
echo "=== Step 6: Verifying installation ==="

echo -n "  PyTorch: "
"$VLNCE_PYTHON" -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1

echo -n "  habitat-sim: "
"$VLNCE_PYTHON" -c "import habitat_sim; print(habitat_sim.__version__)" 2>&1

echo -n "  habitat-lab: "
"$VLNCE_PYTHON" -c "import habitat; print('OK')" 2>&1

echo -n "  vlnce_baselines: "
"$VLNCE_PYTHON" -s -c "import vlnce_baselines; print('OK')" 2>&1 || echo "WARN: import failed (may need PYTHONPATH)"

echo -n "  habitat_extensions: "
"$VLNCE_PYTHON" -s -c "import habitat_extensions; print('OK')" 2>&1 || echo "WARN: import failed (may need PYTHONPATH)"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The vlnce env is used automatically by server-mode nodesets."
echo "To set it explicitly:  export VLNCE_PYTHON=$VLNCE_PYTHON"
echo "To activate manually:  conda activate ac-vlnce"
echo ""
echo "Download data if needed:  bash scripts/data/fetch_data_vlnce.sh --status"
