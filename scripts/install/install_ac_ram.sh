#!/bin/bash
# =============================================================================
# ac-ram — RAM / RAM++ + SpatialBot server environment
# =============================================================================
# Creates the `ac-ram` conda env (Python 3.10): the shared home for two
# FROZEN perception models that cannot ride ac-fm's bleeding-edge stack:
#   - model_ram      (workspace/nodesets/model/model_ram.py)       RAM / RAM++
#   - vlm_spatialbot (workspace/nodesets/model/vlm_spatialbot.py)  SpatialBot-3B
#
# Why a dedicated env (not ac-fm):
#   RAM/RAM++ (recognize-anything) pins old transformers/scipy APIs — its
#   vendored BERT tag-decoder and Swin position-embedding interpolation break
#   on transformers >= 4.40 and scipy >= 1.14 (interp2d removed). SpatialBot
#   is Bunny-Phi remote code (trust_remote_code) written against transformers
#   ~4.36-4.44. The intersection is transformers 4.39.3 — incompatible with
#   ac-fm's 5.13. So the two frozen models share this window.
#
# No habitat / no simulator: unlike ac-smartway (which also hosts RAM+ but
# needs habitat-sim 0.1.7 for SmartWay's depth-encoder import chain), ac-ram
# is a pure model-server env — RAM tagging + SpatialBot VLM + the auto_host
# HTTP harness, nothing more.
#
# Usage:
#   bash scripts/install/install_ac_ram.sh
#
# Prerequisites:
#   - mamba or conda
#   - NVIDIA GPU + CUDA 12.1 runtime (torch 2.4.1+cu121)
#
# No weights are downloaded here. RAM/RAM++ checkpoints (the standard
# ram_swin_large_14m.pth / ram_plus_swin_large_14m.pth) and the SpatialBot-3B
# HF snapshot live under data/ (or the HF cache) — see each nodeset's default
# ckpt / model_path.
# =============================================================================

set -euo pipefail

ENV_NAME="ac-ram"

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

# ── Step 1: Create conda env (Python 3.10) ──
echo ""
echo "=== Step 1: Creating $ENV_NAME conda env (Python 3.10) ==="
$CONDA_CMD env remove -n "$ENV_NAME" -y 2>/dev/null || true
$CONDA_CMD create -n "$ENV_NAME" python=3.10 -y

RAM_PYTHON="$(conda run -n "$ENV_NAME" which python)"
echo "$ENV_NAME Python: $RAM_PYTHON"
RAM_PIP="$RAM_PYTHON -m pip"

# ── Step 2: PyTorch 2.4.1 + CUDA 12.1 ──
echo ""
echo "=== Step 2: Installing PyTorch 2.4.1 + CUDA 12.1 ==="
$RAM_PIP install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1

# ── Step 3: Frozen transformers window (RAM < 4.40  ∩  Bunny 4.36-4.44) ──
echo ""
echo "=== Step 3: Installing transformers 4.39.3 (the shared frozen window) ==="
$RAM_PIP install "transformers==4.39.3" "tokenizers==0.15.2"

# ── Step 4: RAM / RAM++ (recognize-anything) ──
# Installs the `ram` package (ram / ram_plus model factories). Same git source
# as opennav + ac-smartway.
echo ""
echo "=== Step 4: Installing Recognize Anything (RAM / RAM++) ==="
$RAM_PIP install "git+https://github.com/xinyu1205/recognize-anything.git#egg=ram"

# ── Step 5: SpatialBot (Bunny-Phi) + RAM shared model deps ──
# SpatialBot loads via AutoModelForCausalLM(trust_remote_code=True); its Bunny
# remote code needs accelerate + einops + timm + sentencepiece. fairscale/ftfy
# are RAM deps; scipy is pinned < 1.14 so RAM's Swin pos-embed interp2d call
# still resolves.
echo ""
echo "=== Step 5: Installing model deps (SpatialBot / RAM shared) ==="
$RAM_PIP install \
    "accelerate==1.14.0" "einops==0.8.2" "timm==1.0.26" "sentencepiece==0.2.1" \
    "fairscale==0.4.13" "ftfy" "safetensors" "scipy==1.13.1" "pillow"

# ── Step 6: auto_host HTTP harness (server-mode /call transport) ──
# msgpack is the default /call codec — without it every server-mode call 500s
# (ModuleNotFoundError). fastapi/uvicorn/pydantic host the nodeset subprocess.
echo ""
echo "=== Step 6: Installing server-mode harness ==="
$RAM_PIP install \
    "msgpack" "fastapi[standard]" "uvicorn" "httpx" \
    "pydantic>=2.10" "pydantic-settings" "python-dotenv" "requests"

# ── Step 7: Verify ──
echo ""
echo "=== Step 7: Verifying installation ==="
echo -n "  PyTorch:      "
$RAM_PYTHON -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1
echo -n "  transformers: "
$RAM_PYTHON -c "import transformers; print(transformers.__version__)" 2>&1
echo -n "  RAM / RAM++:  "
$RAM_PYTHON -s -c "from ram.models import ram, ram_plus; print('OK')" 2>&1 || echo "WARN: ram import failed"
echo -n "  msgpack:      "
$RAM_PYTHON -c "import msgpack; print('OK')" 2>&1 || echo "WARN: msgpack missing — server-mode calls will 500"
echo -n "  fastapi:      "
$RAM_PYTHON -c "import fastapi; print('OK')" 2>&1 || echo "WARN"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The $ENV_NAME env hosts model_ram + vlm_spatialbot in server mode."
echo "To set explicitly:  export RAM_PERCEPTION_PYTHON=$RAM_PYTHON   # model_ram"
echo "                    export SPATIALBOT_PYTHON=$RAM_PYTHON       # vlm_spatialbot"
echo "Or activate:        conda activate $ENV_NAME"
echo ""
echo "Weights (not downloaded here):"
echo "  - RAM / RAM++:  under data/  (ram_swin_large_14m.pth / ram_plus_swin_large_14m.pth)"
echo "  - SpatialBot:   HF snapshot at the nodeset's default model_path"
echo ""
