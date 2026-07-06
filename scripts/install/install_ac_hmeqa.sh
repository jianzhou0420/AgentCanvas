#!/bin/bash
# =============================================================================
# HM-EQA Environment Installation Script
# =============================================================================
# Creates the `ac-hmeqa` conda env (Python 3.9) for explore-eqa / HM-EQA.
# Used in server mode by the `env_hmeqa` nodeset
# (`workspace/nodesets/env/env_hmeqa/`) and the `vlm_prismatic` FM nodeset
# (`workspace/nodesets/model/vlm_prismatic.py`) — server_python points here.
#
# Why separate from `vlnce`:
#   vlnce pins habitat-sim 0.1.7 + Python 3.8 + torch 1.9 (VLN-CE stack).
#   HM-EQA uses latest habitat-sim + Python 3.9 + torch 2.2.1 (Prismatic
#   VLM). The two stacks cannot coexist in one interpreter.
#
# Usage:
#   bash scripts/install/install_ac_hmeqa.sh
#
# After install:
#   export HMEQA_PYTHON=/home/$(whoami)/miniforge3/envs/ac-hmeqa/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - NVIDIA GPU + CUDA 12.x driver (for torch 2.2.1 + Prismatic inference)
#   - HuggingFace access token in HF_TOKEN (Prismatic weights are gated)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_YAML="$SCRIPT_DIR/envs/ac_hmeqa.yaml"

echo "=== HM-EQA Environment Installation ==="
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

# ── Step 1: Create conda environment ──

echo ""
echo "=== Step 1: Creating conda env from $ENV_YAML ==="
$CONDA_CMD env remove -n ac-hmeqa -y 2>/dev/null || true
$CONDA_CMD env create -f "$ENV_YAML"

HMEQA_PYTHON="/home/$(whoami)/miniforge3/envs/ac-hmeqa/bin/python"
if [ ! -f "$HMEQA_PYTHON" ]; then
    HMEQA_PYTHON="$(conda run -n ac-hmeqa which python)"
fi
CONDA_PREFIX="$(dirname "$(dirname "$HMEQA_PYTHON")")"
echo "hmeqa Python: $HMEQA_PYTHON"

# ── Step 2: Install Prismatic VLM (from pinned upstream commit) ──
#
# Prismatic is installed directly from upstream at a pinned commit.
# For source-level inspection / local edits, see
#   workspace/nodesets/_upstream/prismatic-vlms/fetch_upstream.sh
# (clones to ./upstream/ for `pip install -e ./upstream/`).

echo ""
echo "=== Step 2: Installing Prismatic VLM (pinned upstream) ==="
PRISMATIC_COMMIT="7573aeb4f8cb49b4107b6ef0dc7845377c57b4a7"
PRISMATIC_URL="git+https://github.com/allenzren/prismatic-vlms.git@${PRISMATIC_COMMIT}"
"$HMEQA_PYTHON" -m pip install "$PRISMATIC_URL" 2>&1 | tail -5

# ── Step 3: Remove conda OpenGL libs (use system NVIDIA drivers) ──
#
# Same workaround as install_ac_vlnce.sh — habitat-sim crashes with
# "GL::Context: cannot retrieve OpenGL version" if conda's bundled
# libGL / libEGL shadow the system NVIDIA drivers.

echo ""
echo "=== Step 3: Removing conda OpenGL libs (use NVIDIA drivers instead) ==="
$CONDA_CMD remove -n ac-hmeqa libgl libglvnd libglx libegl --force -y 2>/dev/null || true

# ── Step 4: Build NVIDIA driver-570 EGL workaround shim ──
#
# Magnum's WindowlessEglApplication.cpp:492 calls glGetString(GL_VENDOR)
# right after EGL context creation, with no defense against bogus
# (non-NULL, non-string) return pointers. NVIDIA driver 570.x has a
# regression where the headless EGL path returns such a pointer instead
# of either NULL or a real vendor string, leading to a hard SIGSEGV in
# strlen. The shim intercepts glGetString via LD_PRELOAD and forges
# clearly-invalid pointers to NULL so Magnum's NULL fast-path kicks in.
# See nvidia_egl_workaround.c for the full root-cause analysis.

echo ""
echo "=== Step 4: Building NVIDIA driver-570 EGL workaround shim ==="
SHIM_SRC="$SCRIPT_DIR/hmeqa_libs/nvidia_egl_workaround.c"
SHIM_SO="$SCRIPT_DIR/hmeqa_libs/nvidia_egl_workaround.so"
if [ -f "$SHIM_SRC" ]; then
    gcc -shared -fPIC -O2 -o "$SHIM_SO" "$SHIM_SRC" -ldl
    echo "  built: $SHIM_SO"
else
    echo "[WARN] $SHIM_SRC not found — skipping shim build."
    echo "       habitat-sim 0.3.x will SIGSEGV on driver 570+."
fi

# ── Step 5: Setup activation hooks (LD_LIBRARY_PATH + LD_PRELOAD shim) ──

echo ""
echo "=== Step 5: Setting up env activation hooks ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << EOF
#!/bin/bash
export OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

# NVIDIA driver-570 workaround for habitat-sim 0.3.x (see install_ac_hmeqa.sh
# step 4). The shim forges bogus glGetString returns to NULL so Magnum's
# NULL fast-path kicks in instead of strlen'ing invalid memory. Safe
# wrt other GL apps — valid strings pass through untouched.
if [ -f "$SHIM_SO" ]; then
    export OLD_LD_PRELOAD="\$LD_PRELOAD"
    export LD_PRELOAD="$SHIM_SO\${LD_PRELOAD:+:\$LD_PRELOAD}"
fi
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH

if [ -n "${OLD_LD_PRELOAD+x}" ]; then
    if [ -z "$OLD_LD_PRELOAD" ]; then
        unset LD_PRELOAD
    else
        export LD_PRELOAD="$OLD_LD_PRELOAD"
    fi
    unset OLD_LD_PRELOAD
fi
EOF

# ── Step 6: Verify installation ──

echo ""
echo "=== Step 6: Verifying installation ==="

echo -n "  PyTorch: "
"$HMEQA_PYTHON" -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1

echo -n "  habitat-sim: "
"$HMEQA_PYTHON" -c "import habitat_sim; print(habitat_sim.__version__)" 2>&1

echo -n "  transformers: "
"$HMEQA_PYTHON" -c "import transformers; print(transformers.__version__)" 2>&1

echo -n "  numba: "
"$HMEQA_PYTHON" -c "import numba; print(numba.__version__)" 2>&1

echo -n "  scikit-image: "
"$HMEQA_PYTHON" -c "import skimage; print(skimage.__version__)" 2>&1

echo -n "  prismatic: "
"$HMEQA_PYTHON" -c "import prismatic; print('OK')" 2>&1 || echo "WARN: Prismatic import failed — install manually if needed"

echo -n "  agentcanvas app: "
"$HMEQA_PYTHON" -c "from app.components import BaseCanvasNode; print('OK')" 2>&1 || echo "WARN: agentcanvas app not importable"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The ac-hmeqa env is used by env_hmeqa + vlm_prismatic in server mode."
echo "To set it explicitly:  export HMEQA_PYTHON=$HMEQA_PYTHON"
echo "To activate manually:  conda activate ac-hmeqa"
echo ""
echo "HuggingFace auth (required for Prismatic weights):"
echo "  export HF_TOKEN=<your_token>"
echo "  $HMEQA_PYTHON -c \"from huggingface_hub import login; login(token='\$HF_TOKEN')\""
echo ""
echo "Next:"
echo "  bash scripts/data/fetch_episodes_vln.sh --hmeqa    # fetch HM-EQA CSVs + HM3D instructions"
