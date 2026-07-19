#!/bin/bash
# =============================================================================
# LIBERO Environment Installation Script
# =============================================================================
# Creates the `ac-libero` conda env (Python 3.10) for the LIBERO
# manipulation benchmark. Used in server mode by `env_libero` nodeset at
# `workspace/nodesets/env/env_libero/` (server_python points here).
#
# Why a separate env:
#   LIBERO depends on robosuite 1.4.x + MuJoCo + numpy<2 + LIBERO's own
#   pinned tensorflow / dataclasses-json. These conflict with the other
#   AgentCanvas envs (vlnce: habitat-sim 0.1.7 + Py3.8, hmeqa: habitat-sim
#   0.3 + Py3.9 + torch 2.2). Keeping LIBERO in its own env avoids hours
#   of dependency conflict bisection.
#
# Usage:
#   bash scripts/install/install_ac_libero.sh
#
# After install:
#   export LIBERO_PYTHON=/home/$(whoami)/miniforge3/envs/ac-libero/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - LIBERO source available at one of:
#       third_party/libero           (preferred — public upstream, cloned +
#                                     pinned below via lib/thirdparty.sh)
#       $LIBERO_SOURCE_DIR           (override)
#       https://github.com/Lifelong-Robot-Learning/LIBERO  (fallback git clone)
#   - LIBERO datasets at data/libero/datasets (repo-relative; a symlink or real
#     dir you populate — or set LIBERO_DATASETS_PATH manually after install)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${LIBERO_ENV_NAME:-ac-libero}"
PY_VERSION="3.10"
LIBERO_SRC_DEFAULT="$PROJECT_ROOT/third_party/libero"
LIBERO_DATASETS_DEFAULT="$PROJECT_ROOT/data/libero/datasets"
DATA_ROOT="$PROJECT_ROOT/data/libero"

echo "=== LIBERO Environment Installation ==="
echo "Project root: $PROJECT_ROOT"
echo "Env name:     $ENV_NAME"
echo ""

# ── Step 0: Conda CLI ──

if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "[ERROR] Neither mamba nor conda found. Install miniforge/mamba first."
    exit 1
fi
echo "Using: $CONDA_CMD"

# ── Step 1: Create / reuse conda env ──

echo ""
echo "=== Step 1: Creating conda env '$ENV_NAME' (Python $PY_VERSION) ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  exists: '$ENV_NAME' — reusing (pip-install steps will idempotently update)"
else
    $CONDA_CMD create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

LIBERO_PYTHON="/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
if [ ! -f "$LIBERO_PYTHON" ]; then
    LIBERO_PYTHON="$(conda run -n "$ENV_NAME" which python)"
fi
echo "LIBERO Python: $LIBERO_PYTHON"

# ── Step 2: Resolve LIBERO source ──

echo ""
echo "=== Step 2: Resolving LIBERO source ==="
LIBERO_SRC="${LIBERO_SOURCE_DIR:-$LIBERO_SRC_DEFAULT}"

# If using the default in-repo source and it isn't present yet, clone + pin
# libero from its public upstream. Commit ID lives in lib/thirdparty.sh.
if [ -z "$LIBERO_SOURCE_DIR" ] && [ ! -f "$LIBERO_SRC/setup.py" ]; then
    echo "  fetching libero source (public upstream clone)..."
    source "$SCRIPT_DIR/lib/thirdparty.sh"
    ensure_thirdparty libero
fi

if [ -d "$LIBERO_SRC" ] && [ -f "$LIBERO_SRC/setup.py" ]; then
    echo "  found local source: $LIBERO_SRC"
    LIBERO_INSTALL_TARGET="$LIBERO_SRC"

    # Apply our patches before pip install -e (idempotent). Only meaningful
    # when the source is the in-repo submodule; if user overrode with
    # $LIBERO_SOURCE_DIR, apply_thirdparty_patches.sh will still target the
    # in-repo path — harmless, just doesn't patch their override.
    echo ""
    bash "$SCRIPT_DIR/patches/apply_thirdparty_patches.sh"
else
    echo "  no local LIBERO source at $LIBERO_SRC — will pip install from GitHub"
    echo "  [WARN] git-URL install bypasses our libero patches (PyTorch 2.6 weights_only,"
    echo "         gym->gymnasium). Reset and use the in-repo submodule if you hit those issues."
    LIBERO_INSTALL_TARGET="git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
fi

# ── Step 3: Install LIBERO + deps ──

echo ""
echo "=== Step 3: Installing LIBERO + runtime deps ==="
"$LIBERO_PYTHON" -m pip install --upgrade pip wheel
# Order matters: numpy<2 first, then robosuite, then LIBERO, then app.
"$LIBERO_PYTHON" -m pip install \
    'numpy<2' \
    'robosuite==1.4.1' \
    'dill' \
    'gymnasium' \
    'gym' \
    'imageio[ffmpeg]' \
    'opencv-python-headless' \
    'pyyaml' \
    'easydict' \
    'hydra-core' \
    'bddl' \
    'matplotlib' \
    'cloudpickle' \
    'einops' \
    'future' \
    'thop' \
    'uvicorn' \
    'fastapi' \
    'httpx' \
    'pydantic' \
    'websockets'

if [ "$LIBERO_INSTALL_TARGET" = "$LIBERO_SRC" ]; then
    "$LIBERO_PYTHON" -m pip install -e "$LIBERO_INSTALL_TARGET"
else
    "$LIBERO_PYTHON" -m pip install "$LIBERO_INSTALL_TARGET"
fi

# LIBERO prompts interactively ("Do you want to specify a custom path...") on the
# FIRST `import libero`, writing ~/.libero/config.yaml. Under a non-interactive
# install the input() hits EOF and import fails. Answer "N" (use default paths)
# once to create the config so later imports — the verify below and server-mode
# spawns — are non-interactive.
echo "N" | "$LIBERO_PYTHON" -c "import libero" >/dev/null 2>&1 \
    && echo "  [ok] libero config initialized (~/.libero/config.yaml)" \
    || echo "  [WARN] libero config init failed — first import may still prompt."

# AgentCanvas backend has no setup.py — the framework injects
# PYTHONPATH=<backend>:<workspace> at server-mode spawn time
# (registry.py:289-308). No pip install needed here. The verification
# step below sets PYTHONPATH temporarily to confirm the surface imports.

# ── Step 4: Symlink datasets ──

echo ""
echo "=== Step 4: Linking LIBERO datasets ==="
mkdir -p "$DATA_ROOT"
if [ -d "$LIBERO_DATASETS_DEFAULT" ] && [ ! -e "$DATA_ROOT/datasets" ]; then
    ln -s "$LIBERO_DATASETS_DEFAULT" "$DATA_ROOT/datasets"
    echo "  linked: $DATA_ROOT/datasets -> $LIBERO_DATASETS_DEFAULT"
elif [ -e "$DATA_ROOT/datasets" ]; then
    echo "  exists: $DATA_ROOT/datasets — leaving in place"
else
    echo "  MISSING: $LIBERO_DATASETS_DEFAULT not found"
    echo ""
    echo "  Action required:"
    echo "    Either symlink or copy your LIBERO HDF5 datasets to:"
    echo "        $DATA_ROOT/datasets/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}"
    echo "    Or set LIBERO_DATASETS_PATH to point at them."
fi

# ── Step 5: Verify ──

echo ""
echo "=== Step 5: Verifying installation ==="
echo -n "  numpy:      "
"$LIBERO_PYTHON" -c "import numpy; print(numpy.__version__)" 2>&1
echo -n "  robosuite:  "
"$LIBERO_PYTHON" -c "import robosuite; print(robosuite.__version__)" 2>&1 || echo "FAIL"
echo -n "  libero:     "
"$LIBERO_PYTHON" -c "from libero.libero import benchmark; ks=list(benchmark.get_benchmark_dict().keys()); print(', '.join(ks))" 2>&1 || echo "FAIL"
echo -n "  bddl path:  "
"$LIBERO_PYTHON" -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))" 2>&1 || echo "FAIL"
echo -n "  app:        "
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$LIBERO_PYTHON" -c "from app.components import BaseCanvasNode; print('OK (PYTHONPATH-injected at runtime by registry.py)')" 2>&1 \
    || echo "WARN: agentcanvas app not importable even with PYTHONPATH set"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The $ENV_NAME env is used by workspace/nodesets/env/env_libero/ in server mode."
echo "To set it explicitly:  export LIBERO_PYTHON=$LIBERO_PYTHON"
echo "To activate manually:  conda activate $ENV_NAME"
echo ""
echo "Next:"
echo "  1. cd agentcanvas && bash run_dev.sh"
echo "  2. POST /api/components/nodesets/env_libero/load?mode=server"
echo "  3. Open the canvas — drop env_libero__reset / env_libero__step nodes,"
echo "     pick a suite/task/episode in the LIBERO controller panel."
