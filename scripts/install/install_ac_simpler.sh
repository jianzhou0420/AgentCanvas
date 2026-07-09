#!/bin/bash
# =============================================================================
# SIMPLER Environment Installation Script
# =============================================================================
# Creates the `ac-simpler` conda env (Python 3.10) for the SIMPLER
# VLA evaluation benchmark (https://github.com/simpler-env/SimplerEnv).
# Used in server mode by `env_simpler` nodeset at
# `workspace/nodesets/env/env_simpler/` (server_python points here).
#
# Why a separate env:
#   SIMPLER depends on SAPIEN 2.x (Vulkan-rendered) + ManiSkill2_real2sim +
#   gymnasium >= 0.29. SAPIEN's Vulkan stack and robosuite/MuJoCo (LIBERO)
#   conflict at the system-OpenGL level. Keeping SIMPLER in its own env
#   avoids dependency churn.
#
# Usage:
#   bash scripts/install/install_ac_simpler.sh
#
# After install:
#   export SIMPLER_PYTHON=/home/$(whoami)/miniforge3/envs/ac-simpler/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - NVIDIA GPU + Vulkan ICD (SAPIEN cannot run CPU-only)
#   - git (clones two upstream repos into third_party/)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${SIMPLER_ENV_NAME:-ac-simpler}"
PY_VERSION="3.10"
SIMPLER_SRC="${SIMPLER_SOURCE_DIR:-$PROJECT_ROOT/third_party}"
MANISKILL_REPO="$SIMPLER_SRC/ManiSkill2_real2sim"
SIMPLERENV_REPO="$SIMPLER_SRC/SimplerEnv"

echo "=== SIMPLER Environment Installation ==="
echo "Project root: $PROJECT_ROOT"
echo "Env name:     $ENV_NAME"
echo "Sources:      $SIMPLER_SRC/{ManiSkill2_real2sim,SimplerEnv}"
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

# ── Step 0b: Vulkan precheck (warn, do not auto-install) ──

echo ""
echo "=== Vulkan precheck ==="
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | head -1 || echo "  vulkaninfo present but summary failed"
else
    echo "  WARN: 'vulkaninfo' not found. SAPIEN requires a working Vulkan ICD."
    echo "        On Ubuntu: sudo apt install -y libvulkan-dev vulkan-tools"
    echo "        Continuing — install will succeed but rendering will fail at runtime."
fi

# ── Step 1: Create / reuse conda env ──

echo ""
echo "=== Step 1: Creating conda env '$ENV_NAME' (Python $PY_VERSION) ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  exists: '$ENV_NAME' — reusing (pip-install steps will idempotently update)"
else
    $CONDA_CMD create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

SIMPLER_PYTHON="/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
if [ ! -f "$SIMPLER_PYTHON" ]; then
    SIMPLER_PYTHON="$(conda run -n "$ENV_NAME" which python)"
fi
echo "SIMPLER Python: $SIMPLER_PYTHON"

# ── Step 2: Resolve / clone upstream sources ──

echo ""
echo "=== Step 2: Resolving SIMPLER + ManiSkill2_real2sim sources ==="
mkdir -p "$SIMPLER_SRC"

if [ -d "$MANISKILL_REPO/.git" ]; then
    echo "  found:    $MANISKILL_REPO (existing clone — leaving in place)"
else
    echo "  cloning:  ManiSkill2_real2sim into $MANISKILL_REPO"
    git clone --depth 1 https://github.com/simpler-env/ManiSkill2_real2sim.git "$MANISKILL_REPO"
fi

if [ -d "$SIMPLERENV_REPO/.git" ]; then
    echo "  found:    $SIMPLERENV_REPO (existing clone — leaving in place)"
else
    echo "  cloning:  SimplerEnv into $SIMPLERENV_REPO"
    git clone --depth 1 https://github.com/simpler-env/SimplerEnv.git "$SIMPLERENV_REPO"
fi

# ── Step 3: Install SIMPLER + deps ──

echo ""
echo "=== Step 3: Installing SIMPLER + runtime deps ==="
"$SIMPLER_PYTHON" -m pip install --upgrade pip wheel
# SAPIEN's renderer_config.py imports pkg_resources, which setuptools 81+ removed.
# Pin setuptools<81 to keep the import path working.
"$SIMPLER_PYTHON" -m pip install 'setuptools<81'

# ruckig (transitive dep of mani_skill2_real2sim) ships only sdists past 0.12.x,
# and the latest (0.17.x) uses a `cmake.targets` config that scikit-build-core
# >= 0.10 rejects ("Use build.targets instead of cmake.targets") — so building
# it from source fails during the ManiSkill editable install below. Pre-install
# the last version with a cp310 wheel so pip never builds ruckig from source.
"$SIMPLER_PYTHON" -m pip install --only-binary :all: 'ruckig==0.12.2'

# ManiSkill2_real2sim brings sapien, mani_skill2_real2sim, gymnasium, etc.
"$SIMPLER_PYTHON" -m pip install -e "$MANISKILL_REPO"
"$SIMPLER_PYTHON" -m pip install -e "$SIMPLERENV_REPO"

# numpy<2 is required: numpy 2.x triggers segfaults in SAPIEN's step path.
# opencv-python-headless<4.10 keeps the numpy<2 pin satisfied.
"$SIMPLER_PYTHON" -m pip install 'numpy<2' 'opencv-python-headless<4.10'

# AgentCanvas server-mode deps (match libero install).
"$SIMPLER_PYTHON" -m pip install \
    'fastapi' \
    'uvicorn' \
    'httpx' \
    'pydantic' \
    'websockets'

# AgentCanvas backend has no setup.py — registry.py injects PYTHONPATH at
# server-mode spawn (see install_ac_libero.sh:124-127 for the same pattern).

# ── Step 4: Verify ──

echo ""
echo "=== Step 4: Verifying installation ==="

echo -n "  numpy:           "
"$SIMPLER_PYTHON" -c "import numpy; print(numpy.__version__)" 2>&1 || echo "FAIL"
echo -n "  sapien:          "
"$SIMPLER_PYTHON" -c "import sapien; print(getattr(sapien, '__version__', '?'))" 2>&1 || echo "FAIL"
echo -n "  gymnasium:       "
"$SIMPLER_PYTHON" -c "import gymnasium; print(gymnasium.__version__)" 2>&1 || echo "FAIL"
echo -n "  simpler_env:     "
"$SIMPLER_PYTHON" -c "import simpler_env; print('tasks=', len(simpler_env.ENVIRONMENTS))" 2>&1 || echo "FAIL"
echo -n "  app:             "
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$SIMPLER_PYTHON" -c "from app.components import BaseCanvasNode; print('OK (PYTHONPATH-injected at runtime by registry.py)')" 2>&1 \
    || echo "WARN: agentcanvas app not importable even with PYTHONPATH set"

echo ""
echo "=== Step 5: Probing obs structure (pins Color-vs-rgb key + proprio shape) ==="
"$SIMPLER_PYTHON" - <<'PY' || echo "  WARN: probe failed (Vulkan/GPU issue?). Wrapper will try both keys at runtime."
import os
os.environ.setdefault("DISPLAY", "")
import simpler_env, numpy as np
print("  total task IDs:    ", len(simpler_env.ENVIRONMENTS))
bridge = [t for t in simpler_env.ENVIRONMENTS if t.startswith("widowx_")]
google = [t for t in simpler_env.ENVIRONMENTS if t.startswith("google_robot_")]
print(f"  Bridge tasks:       {len(bridge)} {bridge}")
print(f"  Google Robot tasks: {len(google)} {google[:3]}... ({len(google)} total)")
try:
    env = simpler_env.make("widowx_spoon_on_towel")
    obs, info = env.reset(seed=0)
    print("  obs top-level keys:", list(obs.keys()))
    print("  image entries:     ", list(obs["image"].keys()))
    cam = next(iter(obs["image"]))
    inner = obs["image"][cam]
    print(f"  obs['image']['{cam}'] keys: {list(inner.keys())}")
    for k, v in inner.items():
        if hasattr(v, "shape"):
            print(f"    .{k:14s} shape={v.shape} dtype={v.dtype}")
    # proprio probe
    if "agent" in obs:
        agent = obs["agent"]
        if isinstance(agent, dict):
            print("  agent keys:        ", list(agent.keys()))
            for k, v in agent.items():
                shape = getattr(v, "shape", None)
                print(f"    agent.{k:14s} shape={shape}")
        elif hasattr(agent, "shape"):
            print(f"  agent shape:        {agent.shape}")
    inst = env.get_language_instruction()
    print(f"  instruction:        {inst!r}")
    print(f"  spec.max_episode_steps: {env.spec.max_episode_steps}")
    env.close()
    print("  PROBE OK")
except Exception as e:
    print(f"  PROBE FAIL: {type(e).__name__}: {e}")
    print("  (Likely Vulkan / GPU runtime issue — install completed but render won't work.)")
PY

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The $ENV_NAME env is used by workspace/nodesets/env/env_simpler/ in server mode."
echo "To set it explicitly:  export SIMPLER_PYTHON=$SIMPLER_PYTHON"
echo "To activate manually:  conda activate $ENV_NAME"
echo ""
echo "Next:"
echo "  1. cd agentcanvas && bash run_dev.sh"
echo "  2. POST /api/components/nodesets/env_simpler/load?mode=server"
echo "  3. Open the canvas — drop env_simpler__reset / env_simpler__step nodes,"
echo "     pick a split/task/episode in the SIMPLER controller panel."
