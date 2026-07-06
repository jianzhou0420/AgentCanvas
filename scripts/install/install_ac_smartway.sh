#!/bin/bash
# =============================================================================
# SmartWay (IROS 2025) Environment Installation Script
# =============================================================================
# Creates the `ac-smartway` conda env for the SmartWay nodeset stack:
#   - workspace/nodesets/method/smartway_waypoint  (DINOv2 + DDPPO + TRM)
#   - workspace/nodesets/method/smartway_perception  (RAM+ tagging)
#
# This is a SIDE EXPERIMENT port — not PortBench v1 (author-relationship
# constraint, see docs/research/embodied-ai-lit-review/vln-methods.html
# § 3.2). The method-side nodeset (workspace/nodesets/method/smartway/) runs in the
# main `agentcanvas` env; only the two server-mode model nodesets need this env.
#
# Usage:
#   bash scripts/install/install_ac_smartway.sh
#
# Prerequisites:
#   - mamba or conda
#   - NVIDIA GPU + driver (CUDA 12.1+ runtime)
#   - (third_party/VLN-CE + third_party/habitat-lab are auto-cloned + pinned)
#   - data/habitat/ddppo-models/gibson-2plus-resnet50.pth (shared with vlnce env)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_YAML="$SCRIPT_DIR/envs/ac_smartway.yaml"

echo "=== SmartWay Environment Installation ==="
echo "Project root: $PROJECT_ROOT"
echo "Env spec:     $ENV_YAML"
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

# Fetch third_party sources — same chain as install_ac_vlnce.sh (formerly git
# submodules; now pinned clones, commit IDs in scripts/install/lib/thirdparty.sh).
source "$SCRIPT_DIR/lib/thirdparty.sh"
ensure_thirdparty VLN-CE
ensure_thirdparty habitat-lab

# SmartWay-Code vendored sub-tree (loaded via sys.path-insert at engine time; the
# server nodeset DOES NOT pip install it — it just needs to be present).
SMARTWAY_VENDORED="$PROJECT_ROOT/workspace/nodesets/method/smartway_waypoint/_vendored/waypoint_predictor"
if [ ! -d "$SMARTWAY_VENDORED" ]; then
    echo "[WARN] $SMARTWAY_VENDORED missing."
    echo "       The waypoint server nodeset reads TRM_net.py / ID_CrossAttention from this path."
    echo "       Re-vendor via workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh."
fi

# ── Step 1: Create conda env ──
echo ""
echo "=== Step 1: Creating conda env from $ENV_YAML ==="
if $CONDA_CMD env list | grep -qE '^\s*ac-smartway\s'; then
    echo "  [skip] ac-smartway env already exists — use 'conda env remove -n ac-smartway -y' to recreate"
else
    cd "$PROJECT_ROOT"
    $CONDA_CMD env create -f "$ENV_YAML"
fi

SMARTWAY_PYTHON="/home/$(whoami)/miniforge3/envs/ac-smartway/bin/python"
if [ ! -f "$SMARTWAY_PYTHON" ]; then
    SMARTWAY_PYTHON="$(conda run -n ac-smartway which python)"
fi
CONDA_PREFIX="$(dirname "$(dirname "$SMARTWAY_PYTHON")")"
echo "smartway Python: $SMARTWAY_PYTHON"

# ── Step 2: Remove conda OpenGL libs (same NVIDIA-driver hand-off as vlnce) ──
echo ""
echo "=== Step 2: Removing conda OpenGL libs (use NVIDIA drivers) ==="
$CONDA_CMD remove -n ac-smartway libgl libglvnd libglx libegl --force -y 2>/dev/null || true

# ── Step 3: LD_LIBRARY_PATH activation hook ──
echo ""
echo "=== Step 3: Setting up environment activation hooks ==="
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

# ── Step 4: Symlink VLN-CE data dirs (so vlnce_baselines imports work) ──
echo ""
echo "=== Step 4: Setting up data symlinks (mirrors install_ac_vlnce.sh) ==="
VLNCE_DATA="$PROJECT_ROOT/third_party/VLN-CE/data"
mkdir -p "$VLNCE_DATA"
for subdir in datasets scene_datasets checkpoints ddppo-models; do
    src="$PROJECT_ROOT/data/habitat/$subdir"
    dst="$VLNCE_DATA/$subdir"
    target_rel="../../../data/habitat/$subdir"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "  [WARN] $dst is a real dir (not a symlink). Skipping."
        continue
    fi
    if [ -L "$dst" ]; then
        if [ "$(readlink "$dst")" = "$target_rel" ]; then
            echo "  Already linked: $subdir"
            continue
        fi
        rm "$dst"
    fi
    if [ -d "$src" ]; then
        ln -s "$target_rel" "$dst"
        echo "  Linked: third_party/VLN-CE/data/$subdir -> data/habitat/$subdir"
    else
        echo "  [SKIP] $subdir (run install_ac_vlnce.sh or scripts/data/fetch_data_vlnce.sh first)"
    fi
done

# ── Step 5: Checkpoint downloads ──
echo ""
echo "=== Step 5: Downloading SmartWay checkpoints ==="
bash "$SCRIPT_DIR/../data/fetch_ckpt_smartway.sh"

# ── Step 6: Verify ──
echo ""
echo "=== Step 6: Verifying installation ==="
echo -n "  PyTorch: "
"$SMARTWAY_PYTHON" -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1
echo -n "  habitat-sim: "
"$SMARTWAY_PYTHON" -c "import habitat_sim; print(habitat_sim.__version__)" 2>&1 || echo "WARN"
echo -n "  vlnce_baselines (depth encoder import chain): "
"$SMARTWAY_PYTHON" -s -c "import vlnce_baselines; print('OK')" 2>&1 || echo "WARN"
echo -n "  ram (RAM+ provider): "
"$SMARTWAY_PYTHON" -s -c "from ram.models import ram_plus; print('OK')" 2>&1 || echo "WARN"
echo -n "  transformers (AutoImageProcessor): "
"$SMARTWAY_PYTHON" -s -c "from transformers import AutoImageProcessor; print('OK')" 2>&1 || echo "WARN"

# DINOv2 is pulled lazily at first predict() — verify torch.hub cache is reachable.
echo -n "  torch.hub reachable: "
"$SMARTWAY_PYTHON" -c "import torch.hub; print('OK')" 2>&1

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Set:    export SMARTWAY_PYTHON=$SMARTWAY_PYTHON"
echo "Or:     conda activate ac-smartway"
echo ""
echo "Next:"
echo "  1. Verify ckpts exist under data/smartway/ (see fetch_ckpt_smartway.sh output)."
echo "  2. When backend slot opens: add 'smartway' profile to"
echo "     .claude/commands/experiment/profiles.yaml (vram_mb: 5000, exclusive_gpu: false)."
echo "  3. Smoke-test individual servers via standalone auto_host before any"
echo "     full graph run — see plan file under .claude/plans/."
echo ""
