#!/bin/bash
# =============================================================================
# VLA Policy Environment Installation Script (single-shot)
# =============================================================================
# Creates the `ac-vla-policy` conda env from the project-owned yaml
# at scripts/install/envs/ac_vla_policy.yaml. ONE shot — conda packages, pip
# packages (incl. RT-1-X TF stack), and the three editable third_party trees
# all install in a single `mamba env create`.
#
# Why single-shot:
#   - Avoids the previous two-step pitfall where libstdcxx-ng landed AFTER
#     pytorch/TF were already loaded, so CXXABI_1.3.15 was missing on
#     Ubuntu 20.04 (same fix as vlnce.yaml / hmeqa.yaml / mp3d.yaml).
#   - Avoids the typing_extensions==4.5.0 hard pin from tf-agents 0.19.0
#     (which torch 2.6 doesn't satisfy — `TypeIs` needs >=4.10).
#   - Yaml is the single source of truth, mirrors the rest of envs/.
#
# Patch ordering (Step 1 in this script): the two libero source patches + the
# LIBERO __init__.py touch are applied to third_party/libero BEFORE the editable
# installs in Step 2e read that source tree.
#
# Usage:
#   bash scripts/install/install_ac_vla_policy.sh
#
# After install:
#   export VLA_POLICY_PYTHON=/home/$(whoami)/miniforge3/envs/ac-vla-policy/bin/python
#
# Prerequisites:
#   - mamba or conda installed
#   - (lerobot + libero at third_party/{lerobot,libero} are auto-cloned + pinned
#     from public upstream; openpi-client is vendored in the policy_vla nodeset
#     — see lib/thirdparty.sh)
#   - Vendored adapter / policy / models trees in workspace/nodesets/policy/policy_vla/.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="${VLA_POLICY_ENV_NAME:-ac-vla-policy}"
ENV_YAML="$PROJECT_ROOT/scripts/install/envs/ac_vla_policy.yaml"
# Runtime deps (formerly editable-installed out of the private vla_workspace repo):
#   lerobot + libero — public upstreams, cloned + pinned via lib/thirdparty.sh
#   openpi-client    — vendored into the policy_vla nodeset (small; was private)
LEROBOT_DIR="$PROJECT_ROOT/third_party/lerobot"
LIBERO_DIR="$PROJECT_ROOT/third_party/libero"
OPENPI_CLIENT_DIR="$PROJECT_ROOT/workspace/nodesets/policy/policy_vla/_vendored/openpi-client"

echo "=== VLA Policy Environment Installation ==="
echo "Project root:  $PROJECT_ROOT"
echo "Env name:      $ENV_NAME"
echo "Env yaml:      $ENV_YAML"
echo "lerobot:       $LEROBOT_DIR"
echo "libero:        $LIBERO_DIR"
echo "openpi-client: $OPENPI_CLIENT_DIR"
echo ""

# ── Step 0: Conda CLI + prerequisites ──

if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "[ERROR] Neither mamba nor conda found. Install miniforge/mamba first."
    exit 1
fi
echo "Using: $CONDA_CMD"

[ -f "$ENV_YAML" ] || { echo "[ERROR] $ENV_YAML missing"; exit 1; }
echo "  found $ENV_YAML"

# Clone + pin the public upstreams (lerobot, libero); commit IDs live in
# scripts/install/lib/thirdparty.sh. openpi-client is vendored in-repo under the
# policy_vla nodeset, so there is nothing to fetch for it.
source "$SCRIPT_DIR/lib/thirdparty.sh"
ensure_thirdparty lerobot
ensure_thirdparty libero

for d in "$LEROBOT_DIR" "$LIBERO_DIR" "$OPENPI_CLIENT_DIR"; do
    if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "[ERROR] $d missing or empty — check git access (lerobot/libero) and retry."
        exit 1
    fi
done
echo "  lerobot + libero (cloned) + openpi-client (vendored) present"

for tree in adapters policies models; do
    [ -d "$PROJECT_ROOT/workspace/nodesets/policy/policy_vla/$tree" ] || \
        { echo "[ERROR] policy_vla/$tree/ missing"; exit 1; }
done
echo "  policy_vla vendored trees present"

# ── Step 1: Apply our libero source patches (idempotent) ──
#
# libero needs two source fixes (torch.load weights_only fallback + gym ->
# gymnasium); the patch files live in scripts/install/patches/ and are applied
# by the shared applier, which targets third_party/libero. (The old lerobot
# debug-print patch is intentionally dropped — it only touched
# lerobot/envs/libero.py, a module AgentCanvas never imports.)

echo ""
echo "=== Step 1: Applying libero source patches (idempotent) ==="
bash "$SCRIPT_DIR/patches/apply_thirdparty_patches.sh"

# LIBERO upstream forgets to ship libero/__init__.py — without it,
# find_packages() in setup.py finds nothing and the editable install leaves
# an empty MAPPING (so `import libero` fails). Replicate the touch here so the
# install is reproducible.
LIBERO_INIT="$LIBERO_DIR/libero/__init__.py"
if [ ! -f "$LIBERO_INIT" ]; then
    touch "$LIBERO_INIT"
    echo "  [touch] $LIBERO_INIT (LIBERO upstream package-root quirk)"
fi

# ── Step 2: Single-shot env create from yaml ──
# yaml's `pip: -e ./third_party/...` paths resolve relative to CWD, so we cd
# into PROJECT_ROOT first.

echo ""
echo "=== Step 2: Creating conda env '$ENV_NAME' ==="
cd "$PROJECT_ROOT"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  exists — updating in place via 'env update'"
    $CONDA_CMD env update -f "$ENV_YAML" -n "$ENV_NAME"
else
    $CONDA_CMD env create -f "$ENV_YAML"
fi

VLA_PYTHON="/home/$(whoami)/miniforge3/envs/$ENV_NAME/bin/python"
if [ ! -f "$VLA_PYTHON" ]; then
    VLA_PYTHON="$(conda run -n "$ENV_NAME" which python)"
fi
echo "VLA Python: $VLA_PYTHON"

# ── Step 2b: AgentCanvas server-mode deps (uvicorn / fastapi worker) ──

echo ""
echo "=== Step 2b: AgentCanvas backend deps (server-mode) ==="
"$VLA_PYTHON" -m pip install \
    'uvicorn' 'fastapi' 'httpx' 'pydantic' 'websockets' \
    'pyyaml' 'easydict' 'imageio[ffmpeg]'

# ── Step 2c: RT-1-X TF stack ──
#
# Installed in a SECOND pip pass, separate from the yaml's pip block.
# Reason: pip's strict resolver fails when asked to globally satisfy
# tf-agents 0.19.0's transitive pins together with this env's pin set
# (jaxtyping / typeguard / orbax / gymnasium / tyro / ...). Running this
# install on top of the already-resolved env reduces the search space and
# pip succeeds.
#
# RT-1's deps don't conflict with torch 2.6 / jax 0.5.3 / flax 0.10.2 in
# practice; SimplerEnv runs them concurrently. The vendored RT1Inference
# (workspace/nodesets/policy/policy_vla/rt1_inference.py) avoids
# simpler_env/__init__.py's `import mani_skill2_real2sim.envs` so SAPIEN
# stays out of this env.
#
# Modernized 2026-05-05: TF 2.15.0 (used to be TF 2.15.0 + manual nvidia
# wheels because tensorflow[and-cuda]==2.15.0 hit unpublished tensorrt-libs).
# TF 2.19.1's [and-cuda] extra dropped tensorrt entirely — clean install path
# (cuBLAS 12.5.3.2 / cuDNN 9.3.0.75 / etc., 12 wheels, all on PyPI). cuDNN 9
# also matches what jax 0.5.3 needs in this env (no version conflict).
#
# tf-agents 0.19.0 hard-pins tensorflow==2.15.x, but pip --no-deps lets us
# install it on top of TF 2.19. The 4 tf-agents APIs we use through
# rt1_inference.py (SavedModelPyTFEagerPolicy / specs.zero_spec_nest /
# specs.from_spec / trajectories.time_step.transition) are thin wrappers over
# tf.saved_model.load + NamedTuples — stable across TF 2.x. Validated 2026-05-05.
#
# Manual deps that tf-agents 0.19.0 would have dragged via its requires_dist
# (we now pin them ourselves since --no-deps skipped them):
#   - gin-config>=0.4.0
#   - gym<=0.23.0,>=0.17.0  (legacy gym, NOT gymnasium 0.29.1 in env yaml —
#     they coexist; tf-agents imports `gym`, the rest of the env imports
#     `gymnasium`)
#   - pygame==2.1.3  (gym 0.23 transitive)
#   - dm-tree
# Skipped: tensorflow==2.15 (replaced by 2.19 above), typing_extensions==4.5.0
# (incompatible with torch 2.6 — Step 2d already pins >=4.12).

echo ""
echo "=== Step 2c: RT-1-X TF stack ==="
"$VLA_PYTHON" -m pip install \
    'tensorflow[and-cuda]==2.19.1' \
    'tensorflow-hub==0.16.0' \
    'tensorflow-datasets==4.9.4' \
    'tensorflow-probability==0.23.0' \
    'transforms3d' \
    'mediapy'

# tf-agents installed --no-deps to bypass its TF 2.15 hard-pin, then its
# runtime deps installed manually.
"$VLA_PYTHON" -m pip install --no-deps 'tf-agents==0.19.0'
"$VLA_PYTHON" -m pip install \
    'gin-config>=0.4.0' \
    'gym>=0.17.0,<=0.23.0' \
    'pygame==2.1.3' \
    'dm-tree'

# ── Step 2d: Force-upgrade typing_extensions ──
#
# tf-agents 0.19.0 hard-pins `typing-extensions==4.5.0`. torch 2.6's
# `from typing_extensions import TypeIs` needs >=4.10 — without this
# upgrade, torch import fails. pip CLI prints a "dependency resolver does
# not currently take into account..." warning but completes successfully;
# tf-agents tolerates 4.15 in practice (uses are TypeVar/Protocol/Union).

echo ""
echo "=== Step 2d: Force typing_extensions floor for torch 2.6 ==="
"$VLA_PYTHON" -m pip install --upgrade 'typing_extensions>=4.12'

# ── Step 2e: Editable third_party (yaml can't host -e ./relative/path) ──
#
# `--no-deps` because lerobot 0.4.1 setup pins `gymnasium>=1.1.1` +
# `diffusers<0.36`, but this env pins `gymnasium==0.29.1` + `diffusers 0.37`.
# lerobot's runtime code is compatible with the env's pins; the setup
# constraints are overly tight for installation. Without --no-deps, pip's
# resolver does exhaustive backtracking and aborts with `resolution-too-deep`.

echo ""
echo "=== Step 2e: Editable install of lerobot / libero / openpi-client (--no-deps) ==="
"$VLA_PYTHON" -m pip install --no-deps \
    -e "$LEROBOT_DIR" \
    -e "$LIBERO_DIR" \
    -e "$OPENPI_CLIENT_DIR"

# ── Step 2f: lerobot transitive deps that --no-deps skipped ──
#
# These are lerobot 0.4.1's pyproject.toml deps NOT covered by the
# VLA pip block. Listed explicitly (rather than relying on
# pip's resolver) because a global re-resolve including lerobot's
# `gymnasium>=1.1.1` pin against env's `gymnasium==0.29.1` triggers
# `resolution-too-deep`. We deliberately leave gymnasium at 0.29.1 —
# lerobot's runtime imports succeed with it (validated 2026-05-04).
#
# `numpy<2.0` is hard-pinned in this command because rerun-sdk and other
# packages in this batch will silently upgrade numpy to 2.x otherwise,
# breaking jax/jaxlib/tensorflow (all compiled against numpy 1.x). We
# drop rerun-sdk entirely — it's only used by lerobot's teleop UI, not
# for policy inference.

echo ""
echo "=== Step 2f: lerobot transitive deps ==="
"$VLA_PYTHON" -m pip install \
    'numpy<2.0' \
    'draccus==0.10.0' \
    'jsonlines>=4.0.0,<5.0.0' \
    'pynput>=1.7.7,<1.9.0' \
    'pyserial>=3.5,<4.0' \
    'deepdiff>=7.0.1,<9.0.0' \
    'torchcodec>=0.2.1,<0.6.0' \
    'opencv-python-headless>=4.9.0,<4.13.0'

# ── Step 2g: Force-downgrade numpy if anything snuck it up to 2.x ──

echo ""
echo "=== Step 2g: Pin numpy<2.0 (jax/jaxlib/TF compiled against 1.x) ==="
"$VLA_PYTHON" -m pip install --upgrade 'numpy<2.0'

# ── Step 3: Verify ──

echo ""
echo "=== Step 3: Verifying installation ==="
echo -n "  numpy:        "; "$VLA_PYTHON" -c "import numpy; print(numpy.__version__)" 2>&1
echo -n "  torch:        "; "$VLA_PYTHON" -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 || echo "FAIL"
echo -n "  transformers: "; "$VLA_PYTHON" -c "import transformers; print(transformers.__version__)" 2>&1 || echo "FAIL"
echo -n "  diffusers:    "; "$VLA_PYTHON" -c "import diffusers; print(diffusers.__version__)" 2>&1 || echo "FAIL"
echo -n "  hf_hub:       "; "$VLA_PYTHON" -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>&1 || echo "FAIL"
echo -n "  jax / flax:   "; "$VLA_PYTHON" -c "import jax, flax; print(f'jax={jax.__version__} flax={flax.__version__}')" 2>&1 || echo "FAIL"
echo -n "  jaxtyping:    "; "$VLA_PYTHON" -c "import jaxtyping; print(jaxtyping.__version__)" 2>&1 || echo "FAIL"
echo -n "  tensorflow:   "; "$VLA_PYTHON" -c "import tensorflow as tf; print(tf.__version__, 'gpu', len(tf.config.list_physical_devices('GPU')))" 2>&1 | tail -1 || echo "FAIL"
echo -n "  tf_hub:       "; "$VLA_PYTHON" -c "import tensorflow_hub; print(tensorflow_hub.__version__)" 2>&1 | tail -1 || echo "FAIL"
echo -n "  tf_agents:    "; "$VLA_PYTHON" -c "import tf_agents; print(tf_agents.__version__)" 2>&1 | tail -1 || echo "FAIL"
echo -n "  transforms3d: "; "$VLA_PYTHON" -c "import transforms3d; print('OK')" 2>&1 || echo "FAIL"
echo -n "  typing_ext:   "; "$VLA_PYTHON" -c "from typing_extensions import TypeIs; print('TypeIs OK')" 2>&1 || echo "FAIL"
echo -n "  lerobot:      "; "$VLA_PYTHON" -c "from lerobot.policies.pretrained import PreTrainedPolicy; print('OK')" 2>&1 || echo "FAIL"
echo -n "  libero:       "; "$VLA_PYTHON" -c "from libero.libero import benchmark; print(len(benchmark.get_benchmark_dict()), 'suites')" 2>&1 || echo "FAIL"
echo -n "  openpi-client:"; "$VLA_PYTHON" -c "import openpi_client; print('OK')" 2>&1 || echo "FAIL"

echo -n "  TF + torch coexist: "
"$VLA_PYTHON" -c "import torch; import tensorflow as tf; print('OK')" 2>&1 | tail -1 || echo "FAIL"

# Adapter system imports (no torch needed — pure numpy)
echo -n "  adapters:     "
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$VLA_PYTHON" -c "
from workspace.nodesets.policy.policy_vla.adapters import (
    Adaptor, LiberoRobot, SimplerRobot, Pi0Model, SmolVLAModel, DPModel, Rt1Model,
)
print('OK')
" 2>&1 || echo "FAIL"

echo -n "  policies:     "
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$VLA_PYTHON" -c "
from workspace.nodesets.policy.policy_vla.policies import (
    BasePolicy, Pi0Policy, SmolVLAPolicy,
    DiffusionUnetHybridImagePolicy, DroidDiffusionPolicy, Rt1Policy,
)
print('OK')
" 2>&1 || echo "FAIL"

echo -n "  rt1 dropdowns:"
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$VLA_PYTHON" -c "
from workspace.nodesets.policy.policy_vla import MODEL_OPTIONS, POLICY_OPTIONS, ROBOT_OPTIONS
assert 'rt1_model' in MODEL_OPTIONS, f'rt1_model not in {MODEL_OPTIONS}'
assert 'rt1_policy' in POLICY_OPTIONS, f'rt1_policy not in {POLICY_OPTIONS}'
print(f' OK — models={MODEL_OPTIONS}, policies={POLICY_OPTIONS}, robots={ROBOT_OPTIONS}')
" 2>&1 || echo " FAIL"

# 4 policy classes (Pi0/SmolVLA/DP/DroidDP) construct without checkpoint
echo "  policy construction smoke (no checkpoint):"
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$VLA_PYTHON" - <<'PYEOF' 2>&1 | sed 's/^/    /' || true
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from workspace.nodesets.policy.policy_vla.policies import (
    Pi0Policy, SmolVLAPolicy,
    DiffusionUnetHybridImagePolicy, DroidDiffusionPolicy,
)
from workspace.nodesets.policy.policy_vla.policies.pi0_policy import Pi0Config

print("  Pi0Policy:        ", end="")
try:
    p = Pi0Policy(config=Pi0Config(), num_inference_steps=5,
                  use_pretrained_weight=False, gradient_checkpointing=False, _enable_compile=False)
    print(f"OK ({sum(x.numel() for x in p.parameters())/1e6:.0f}M params)")
    del p
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e!s}"[:200])

print("  SmolVLAPolicy:    ", end="")
try:
    p = SmolVLAPolicy(
        image_features={"observation.images.front": [3, 256, 256], "observation.images.wrist": [3, 256, 256]},
        state_dim=8, action_dim=7, chunk_size=10, n_action_steps=5, num_steps=5,
        load_vlm_weights=False)
    print(f"OK ({sum(x.numel() for x in p.parameters())/1e6:.0f}M params)")
    del p
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e!s}"[:200])

shape_meta = {
    "action": {"shape": [10]},
    "obs": {
        "agentview_image": {"shape": [3, 76, 76], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 76, 76], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
        "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
        "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
    },
}
print("  DiffusionPolicy:  ", end="")
try:
    p = DiffusionUnetHybridImagePolicy(shape_meta=shape_meta, noise_scheduler=DDPMScheduler(num_train_timesteps=100),
                                       horizon=16, n_action_steps=8, n_obs_steps=2, crop_shape=(76, 76),
                                       num_inference_steps=10)
    print(f"OK ({sum(x.numel() for x in p.parameters())/1e6:.0f}M params)")
    del p
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e!s}"[:200])

print("  DroidDP:          ", end="")
try:
    p = DroidDiffusionPolicy(shape_meta=shape_meta, noise_scheduler=DDIMScheduler(num_train_timesteps=100),
                             horizon=16, n_action_steps=8, n_obs_steps=2, num_inference_steps=10, use_language=False)
    print(f"OK ({sum(x.numel() for x in p.parameters())/1e6:.0f}M params)")
    del p
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e!s}"[:200])
PYEOF

echo -n "  policy_vla:   "
PYTHONPATH="$PROJECT_ROOT/agentcanvas/backend:$PROJECT_ROOT" \
    "$VLA_PYTHON" -c "
from workspace.nodesets.policy.policy_vla import PolicyVlaNodeSet
ns = PolicyVlaNodeSet()
print(f'OK — name={ns.name}, tools={len(ns.get_tools())}')
" 2>&1 || echo "FAIL"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The $ENV_NAME env is used by workspace/nodesets/policy/policy_vla/ in server mode."
echo "To set it explicitly:  export VLA_POLICY_PYTHON=$VLA_PYTHON"
echo "To activate manually:  conda activate $ENV_NAME"
echo ""
echo "RT-1-X next step:"
echo "  bash scripts/data/fetch_ckpt_rt1.sh    # downloads SavedModel ckpt via gsutil"
echo ""
echo "Then:"
echo "  1. cd agentcanvas && bash run_dev.sh"
echo "  2. POST /api/components/nodesets/policy_vla/load?mode=server"
echo "  3. Open the canvas — VLA Policy controller will appear."
echo "     Pick model/policy/robot dropdowns + checkpoint path, click Load Model."
