#!/bin/bash
# =============================================================================
# VLNVerse Environment Installation Script
# =============================================================================
# Creates the ac-vlnverse conda env (Python 3.11) for the env_vlnverse nodeset.
# This is deliberately the lightest env install in the repo: the nodeset's
# manager layer is pure python (numpy/msgpack/PIL + fastapi for auto_host);
# Isaac Sim 5.1 runs in its own bundled python behind the msgpack RPC seam
# (workspace/nodesets/env/env_vlnverse/_isaac_worker.py) and is a MANUAL
# prerequisite — this script only verifies it.
#
# Usage:
#   bash scripts/install/install_ac_vlnverse.sh
#
# Env overrides:
#   VLNVERSE_ISAAC_PYTHON  Isaac Sim 5.1 launcher (default: ~/isaacsim5.1/python.sh)
#   VLNVERSE_DATA_ROOT     Directory containing raw_data/ + scene/ for the
#                          data/vlnverse symlinks (default: ../NavHarness/data)
#
# Prerequisites:
#   - mamba or conda installed
#   - Isaac Sim 5.1 installed manually (tens of GB), e.g. under ~/isaacsim5.1
#     https://docs.isaacsim.omniverse.nvidia.com/ — workstation install
#   - VLNVerse data: episodes (raw_data/) + scenes; scenes can be fetched with
#     scripts/data/fetch_scenes_vlnverse.py if you have no local copy
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_YAML="$SCRIPT_DIR/envs/ac_vlnverse.yaml"

echo "=== VLNVerse Environment Installation ==="
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
echo "=== Step 1: Creating conda environment from $ENV_YAML ==="
$CONDA_CMD env remove -n ac-vlnverse -y 2>/dev/null || true
$CONDA_CMD env create -f "$ENV_YAML"

VLNVERSE_PYTHON="/home/$(whoami)/miniforge3/envs/ac-vlnverse/bin/python"
if [ ! -f "$VLNVERSE_PYTHON" ]; then
    # Fallback: find the env wherever conda put it
    VLNVERSE_PYTHON="$(conda run -n ac-vlnverse which python)"
fi
echo "vlnverse Python: $VLNVERSE_PYTHON"

# ── Step 2: Verify Isaac Sim 5.1 (manual prerequisite) ──

echo ""
echo "=== Step 2: Verifying Isaac Sim 5.1 ==="
ISAAC_LAUNCHER="${VLNVERSE_ISAAC_PYTHON:-$HOME/isaacsim5.1/python.sh}"
if [ -f "$ISAAC_LAUNCHER" ]; then
    echo "  Found Isaac launcher: $ISAAC_LAUNCHER"
else
    echo "  [WARN] Isaac Sim 5.1 launcher not found: $ISAAC_LAUNCHER"
    echo "         Isaac Sim 5.1 is a manual prerequisite (tens of GB) — install it"
    echo "         from https://docs.isaacsim.omniverse.nvidia.com/ and either place"
    echo "         it at ~/isaacsim5.1 or set VLNVERSE_ISAAC_PYTHON to its python.sh."
    echo "         Continuing: the ac-vlnverse env is still usable for the package's"
    echo "         pure-logic tests; only rendering needs Isaac."
fi

# ── Step 3: Setup data symlinks ──
# data/ is gitignored, so every fresh clone/worktree/machine re-runs this.

echo ""
echo "=== Step 3: Setting up data symlinks (data/vlnverse) ==="
VLNVERSE_DIR="$PROJECT_ROOT/data/vlnverse"
mkdir -p "$VLNVERSE_DIR"
DATA_ROOT="${VLNVERSE_DATA_ROOT:-$PROJECT_ROOT/../NavHarness/data}"
for subdir in raw_data scene; do
    src="$DATA_ROOT/$subdir"
    dst="$VLNVERSE_DIR/$subdir"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "  [WARN] $dst is a real directory (not a symlink). Skipping — manual intervention required."
        continue
    fi
    if [ ! -e "$src" ]; then
        if [ "$subdir" = "scene" ]; then
            echo "  [SKIP] scene (no source at $src — download with:"
            echo "         python3 scripts/data/fetch_scenes_vlnverse.py)"
        else
            echo "  [SKIP] raw_data (no source at $src — set VLNVERSE_DATA_ROOT to a"
            echo "         directory containing the VLNVerse raw_data/ episodes)"
        fi
        continue
    fi
    # Absolute target: the source lives OUTSIDE this repo (unlike the
    # habitat symlinks, which stay in-repo and use relative targets).
    target="$(readlink -f "$src")"
    if [ -L "$dst" ]; then
        current="$(readlink -f "$dst")"
        if [ "$current" = "$target" ]; then
            echo "  Already linked: $subdir"
            continue
        fi
        rm "$dst"
    fi
    ln -s "$target" "$dst"
    echo "  Linked: data/vlnverse/$subdir -> $target"
done

# ── Step 4: Isaac worker smoke check ──

echo ""
echo "=== Step 4: Isaac worker smoke check ==="
WORKER="$PROJECT_ROOT/workspace/nodesets/env/env_vlnverse/_isaac_worker.py"
if [ ! -f "$ISAAC_LAUNCHER" ]; then
    echo "  [SKIP] Isaac launcher missing (see Step 2)."
else
    echo "  Running: $ISAAC_LAUNCHER $WORKER --check  (first run may take a minute)"
    if "$ISAAC_LAUNCHER" "$WORKER" --check; then
        echo "  Isaac worker check: OK"
    else
        echo "  [WARN] Isaac worker check failed — see output above. Common causes:"
        echo "         GPU driver/EGL setup, or a partial Isaac Sim install."
    fi
fi

# ── Step 5: Verify installation ──

echo ""
echo "=== Step 5: Verifying installation ==="
# Verify steps are diagnostics only — never abort the install (set -e) on a
# failed import.
echo -n "  numpy: "
"$VLNVERSE_PYTHON" -c "import numpy; print(numpy.__version__)" 2>&1 || echo "WARN: numpy import failed"

echo -n "  msgpack(+numpy): "
"$VLNVERSE_PYTHON" -c "import msgpack, msgpack_numpy; print(msgpack.version)" 2>&1 || echo "WARN: msgpack import failed"

echo -n "  PIL: "
"$VLNVERSE_PYTHON" -c "import PIL; print(PIL.__version__)" 2>&1 || echo "WARN: PIL import failed"

echo -n "  fastapi: "
"$VLNVERSE_PYTHON" -c "import fastapi; print(fastapi.__version__)" 2>&1 || echo "WARN: fastapi import failed"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The ac-vlnverse env is used automatically by server-mode nodesets."
echo "To set it explicitly:  export VLNVERSE_PYTHON=$VLNVERSE_PYTHON"
echo "To activate manually:  conda activate ac-vlnverse"
echo ""
echo "Scenes not downloaded yet?  python3 scripts/data/fetch_scenes_vlnverse.py"
echo "(rate-limit friendly loop:  bash scripts/data/fetch_scenes_vlnverse_loop.sh)"
