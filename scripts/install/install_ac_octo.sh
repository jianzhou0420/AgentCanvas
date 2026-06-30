#!/bin/bash
# =============================================================================
# Octo Policy Environment Installation Script
# =============================================================================
# Creates the `ac-octo` conda env (Python 3.10), clones + installs
# Octo at the SimplerEnv-pinned commit (octo-1.0), and (optionally) snapshots
# the Octo HF checkpoint to data/vla_policy/checkpoints/. Used in server mode
# by `policy_octo` nodeset at workspace/nodesets/server/policy_octo.py.
#
# Why a separate env:
#   Octo runs on JAX/Flax. Coexisting with TF (RT-1) or Torch+CUDA (Pi0) in
#   one env is brittle — different CUDA stacks fight for cuDNN/CUDA. Each
#   policy gets its own env.
#
# Usage:
#   bash scripts/install/install_ac_octo.sh                    # full install (env + ckpt snapshot)
#   bash scripts/install/install_ac_octo.sh --skip-ckpt        # env only
#   bash scripts/install/install_ac_octo.sh --ckpt-only        # snapshot ckpt only
#   bash scripts/install/install_ac_octo.sh --model octo-base  # also fetch octo-base-1.5
#
# After install:
#   export OCTO_PYTHON=/home/$(whoami)/miniforge3/envs/ac-octo/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - NVIDIA GPU with CUDA 12.2+ (for JAX GPU). Octo can run CPU-only too;
#     if so, the [cuda12_pip] extra is harmless — JAX falls back to CPU.
#   - third_party/SimplerEnv + third_party/ManiSkill2_real2sim cloned (clone
#     them via scripts/install/install_ac_simpler.sh first if missing).
#   - huggingface-cli (installed by this script if absent) for the local
#     ckpt snapshot. Without it, the policy still works but downloads on
#     first use into HF cache (~/.cache/huggingface/).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${OCTO_ENV_NAME:-ac-octo}"
PY_VERSION="3.10"
SIMPLER_SRC="${SIMPLER_SOURCE_DIR:-$PROJECT_ROOT/third_party}"
MANISKILL_REPO="$SIMPLER_SRC/ManiSkill2_real2sim"
SIMPLERENV_REPO="$SIMPLER_SRC/SimplerEnv"
OCTO_REPO="$SIMPLER_SRC/octo"
OCTO_PIN_SHA="653c54acde686fde619855f2eac0dd6edad7116b"  # octo-1.0
CKPT_ROOT="$PROJECT_ROOT/data/vla_policy/checkpoints"

DO_ENV=1
DO_CKPT=1
MODELS=("octo-small")
while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-ckpt) DO_CKPT=0; shift ;;
        --ckpt-only) DO_ENV=0;  shift ;;
        --model) MODELS+=("$2"); shift 2 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Octo Policy Environment Installation ==="
echo "Project root:    $PROJECT_ROOT"
echo "Env name:        $ENV_NAME"
echo "Sources:         $SIMPLER_SRC/{ManiSkill2_real2sim,SimplerEnv,octo}"
echo "Octo pinned at:  $OCTO_PIN_SHA"
echo "Models:          ${MODELS[*]}"
echo "Steps:           env=$DO_ENV ckpt=$DO_CKPT"
echo ""

if [ "$DO_ENV" = "1" ]; then
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
    else
        echo "[ERROR] Neither mamba nor conda found. Install miniforge/mamba first."
        exit 1
    fi
    echo "Using: $CONDA_CMD"

    # ── Step 1: Source repos ──

    for repo in "$MANISKILL_REPO" "$SIMPLERENV_REPO"; do
        if [ ! -d "$repo" ]; then
            echo "[ERROR] $repo not found. Run scripts/install/install_ac_simpler.sh first."
            exit 1
        fi
    done

    if [ ! -d "$OCTO_REPO/.git" ]; then
        echo ""
        echo "=== Cloning octo into $OCTO_REPO (pinned to octo-1.0) ==="
        git clone https://github.com/octo-models/octo.git "$OCTO_REPO"
        git -C "$OCTO_REPO" checkout "$OCTO_PIN_SHA"
    else
        cur_sha="$(git -C "$OCTO_REPO" rev-parse HEAD)"
        if [ "$cur_sha" != "$OCTO_PIN_SHA" ]; then
            echo "  octo at $cur_sha — pinning to $OCTO_PIN_SHA"
            git -C "$OCTO_REPO" fetch
            git -C "$OCTO_REPO" checkout "$OCTO_PIN_SHA"
        else
            echo "  octo already at pinned commit"
        fi
    fi

    # ── Step 2: Create / reuse conda env ──

    echo ""
    echo "=== Step 2: Creating conda env '$ENV_NAME' (Python $PY_VERSION) ==="
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "  exists — reusing"
    else
        $CONDA_CMD create -n "$ENV_NAME" "python=$PY_VERSION" -y
    fi

    OCTO_PYTHON="/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
    if [ ! -f "$OCTO_PYTHON" ]; then
        OCTO_PYTHON="$(conda run -n "$ENV_NAME" which python)"
    fi
    echo "Octo Python: $OCTO_PYTHON"

    # ── Step 3: JAX (CUDA 12) + base deps ──

    echo ""
    echo "=== Step 3: Installing JAX + transforms3d ==="
    "$OCTO_PYTHON" -m pip install --upgrade pip wheel
    "$OCTO_PYTHON" -m pip install 'setuptools<81'
    # JAX 0.4.20 is the version SimplerEnv pins.
    # Use cuda12_pip extra; on CUDA 11 boxes swap to cuda11_pip.
    "$OCTO_PYTHON" -m pip install --upgrade \
        'jax[cuda12_pip]==0.4.20' \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        || "$OCTO_PYTHON" -m pip install 'jax==0.4.20' 'jaxlib==0.4.20'  # CPU fallback
    "$OCTO_PYTHON" -m pip install \
        'transforms3d' \
        'mediapy' \
        'numpy==1.24.4'

    # ── Step 4: Editable installs ──

    echo ""
    echo "=== Step 4: Editable install of ManiSkill2_real2sim + SimplerEnv + octo ==="
    "$OCTO_PYTHON" -m pip install -e "$MANISKILL_REPO"
    "$OCTO_PYTHON" -m pip install -e "$SIMPLERENV_REPO"
    "$OCTO_PYTHON" -m pip install -e "$OCTO_REPO"

    # ── Step 5: AgentCanvas server-mode deps ──

    echo ""
    echo "=== Step 5: AgentCanvas server-mode deps ==="
    "$OCTO_PYTHON" -m pip install \
        'fastapi' 'uvicorn' 'httpx' 'pydantic' 'websockets' \
        'huggingface_hub'

    # ── Step 6: Verify ──

    echo ""
    echo "=== Step 6: Verifying installation ==="
    echo -n "  numpy:           "
    "$OCTO_PYTHON" -c "import numpy; print(numpy.__version__)" 2>&1 || echo "FAIL"
    echo -n "  jax / flax:      "
    "$OCTO_PYTHON" -c "import jax, flax; print(f'jax={jax.__version__} flax={flax.__version__} devices={jax.devices()}')" 2>&1 || echo "FAIL"
    echo -n "  octo:            "
    "$OCTO_PYTHON" -c "from octo.model.octo_model import OctoModel; print('OK')" 2>&1 || echo "FAIL"
    echo -n "  simpler_env:     "
    "$OCTO_PYTHON" -c "from simpler_env.policies.octo.octo_model import OctoInference; print('OK')" 2>&1 || echo "FAIL"
    echo -n "  policy_octo:     "
    PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
        "$OCTO_PYTHON" -c "
from workspace.nodesets.server.policy_octo import PolicyOctoNodeSet
ns = PolicyOctoNodeSet()
print(f'OK — name={ns.name}, tools={[t.node_type for t in ns.get_tools()]}')
" 2>&1 || echo "FAIL"
fi

# ── Step 7: Download Octo HF snapshots ──

if [ "$DO_CKPT" = "1" ]; then
    echo ""
    echo "=== Step 7: Snapshotting Octo HF checkpoints ==="
    HF_PYTHON="${OCTO_PYTHON:-$(conda run -n "$ENV_NAME" which python 2>/dev/null || echo python)}"
    "$HF_PYTHON" -m pip install --quiet 'huggingface_hub' 2>/dev/null || true

    for model in "${MODELS[@]}"; do
        local_dir="$CKPT_ROOT/${model}-1.5"
        if [ -d "$local_dir" ] && [ -f "$local_dir/config.json" ]; then
            echo "  exists: $local_dir — skipping"
            continue
        fi
        mkdir -p "$local_dir"
        echo "  fetching: rail-berkeley/$model  →  $local_dir"
        "$HF_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='rail-berkeley/$model',
                  local_dir='$local_dir',
                  local_dir_use_symlinks=False)
print('  OK')
" || { echo "  [WARN] snapshot_download failed for $model — policy will fall back to hf:// URL on first use"; }
    done
fi

echo ""
echo "=== Installation Complete ==="
echo ""
if [ "$DO_ENV" = "1" ]; then
    echo "The $ENV_NAME env is used by workspace/nodesets/server/policy_octo.py in server mode."
    echo "To set it explicitly:  export OCTO_PYTHON=/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
fi
if [ "$DO_CKPT" = "1" ]; then
    echo "Checkpoints under:    $CKPT_ROOT/octo-{small,base}-1.5/"
fi
echo ""
echo "Next:"
echo "  1. cd agentcanvas && bash run_dev.sh"
echo "  2. POST /api/components/nodesets/policy_octo/load?mode=server"
echo "  3. Open the canvas — drop env_simpler__reset/step + policy_octo__predict,"
echo "     pick model_type={octo-small,octo-base} and policy_setup={widowx_bridge,google_robot},"
echo "     wire image+instruction in, action_chunk back to env_simpler__step."
