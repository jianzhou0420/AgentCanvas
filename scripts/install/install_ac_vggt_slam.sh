#!/bin/bash
# =============================================================================
# ac-vggt-slam — VGGT-SLAM 2.0 dense RGB SLAM environment
# =============================================================================
# Creates the `ac-vggt-slam` conda env (Python 3.11) for the model_vggt_slam
# nodeset: VGGT-SLAM 2.0 (MIT-SPARK, RSS 2026) — VGGT feed-forward submap
# reconstruction + DINOv2-SALAD loop-closure retrieval + GTSAM SL(4) pose-graph
# optimization, with an optional open-set 3D object detection path
# (Perception Encoder CLIP + SAM 3).
#
# Why a DEDICATED env (not ac-vggt, not ac-fm):
#   Upstream pins torch==2.3.1 / torchvision==0.18.1 (requirements.txt) to
#   satisfy its third-party stack (VGGT_SPARK fork, salad, sam3, gtsam-develop
#   wheel ABI). ac-vggt runs torch 2.8.0+cu126 and ac-fm the same line —
#   incompatible. The split is mandatory, not cosmetic.
#
# Version rationale (2026-07-14):
#   - torch 2.3.1+cu121 / torchvision 0.18.1+cu121 — upstream's exact pins;
#     cu121 is the newest CUDA line published for torch 2.3.1.
#   - gtsam-develop (pinned dev build) — the pip wheel line that ships the
#     SL(4) types (SL4 / PriorFactorSL4 / BetweenFactorSL4) merged for
#     VGGT-SLAM 2.0. cp311 manylinux x86_64 wheels exist.
#   - Upstream repos pinned to the exact commits this env was verified with.
#   - opencv-python-headless instead of upstream's opencv-python: server env,
#     no display; also sidesteps the SAM3<->OpenCV libxcb clash upstream
#     documents in main_realtime.py. Same cv2 API surface we use (LK flow,
#     goodFeaturesToTrack, imread/imwrite/cvtColor).
#   - Omitted from upstream requirements.txt: gradio (never imported by any
#     first-party module — grep-verified) and virtualenv (env tooling, N/A).
#
# Weights:
#   - facebook/VGGT-1B model.pt (~5 GB) auto-downloads to the torch-hub cache
#     on first use (torch.hub.load_state_dict_from_url), cc-by-nc-4.0.
#   - dino_salad.ckpt is NOT auto-downloaded by upstream setup.sh; upstream
#     loop_closure.py:55 loads it from <torch_hub>/checkpoints/dino_salad.ckpt
#     unconditionally in every Solver construction. This script fetches it via
#     gdown (Google Drive id from the salad README). If the fetch 403s, download
#     manually from the salad README link and place it at that exact path.
#   - PE-Core-L14-336 and SAM 3 weights auto-download from HF on first use
#     (open-set path only, run_os=true).
#
# Open-set (PE + SAM3) install policy: installed --no-deps with explicit leaf
# pins so torch stays 2.3.1. A PE/SAM3 install or import failure must NOT
# break the SLAM mainline — those steps and their spike imports are guarded.
#
# Usage:
#   bash scripts/install/install_ac_vggt_slam.sh
#
# The model_vggt_slam nodeset resolves this env via
# conda_env_python("ac-vggt-slam", "VGGT_SLAM_PYTHON"); override the
# interpreter with $VGGT_SLAM_PYTHON.
#
# Reproducible install: scripts/install/envs/ac_vggt_slam.lock is the frozen
# package set this env was verified with.
# =============================================================================
set -euo pipefail

ENV_NAME="ac-vggt-slam"
PY_VER="3.11"

# Upstream commits this env was verified against (2026-07-14 HEADs).
VGGT_SLAM_COMMIT="35327ac28b7d193df9ccc39ba6346052bb6f1207"   # MIT-SPARK/VGGT-SLAM (main = 2.0)
VGGT_SPARK_COMMIT="6e6e16107b88e8e76c751826af10d4295d87ecd2"  # MIT-SPARK/VGGT_SPARK (vggt fork, +compute_similarity)
SALAD_COMMIT="33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d"       # Dominic101/salad
PE_COMMIT="3e352cca660658d4b5c90f42a7808b11469e4c66"          # facebookresearch/perception_models
SAM3_COMMIT="5dd401d1c5c1d5c3eedff06d41b77af824517619"        # facebookresearch/sam3

GTSAM_DEVELOP_PIN="4.3a1.dev202607020747"                     # cp311 manylinux wheel verified on PyPI
SALAD_CKPT_GDRIVE_ID="1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1"      # from salad README "pretrained DINOv2 SALAD model"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[skip] conda env '${ENV_NAME}' already exists."
else
    echo "[1/8] Creating conda env '${ENV_NAME}' (Python ${PY_VER})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[2/8] Installing torch 2.3.1 (cu121) — upstream's exact pin"
pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "[3/8] Installing gtsam-develop ${GTSAM_DEVELOP_PIN} (SL(4) solver)"
pip install "gtsam-develop==${GTSAM_DEVELOP_PIN}"

echo "[4/8] Installing base deps (upstream requirements.txt minus torch/gtsam/gradio/virtualenv)"
# numpy<2: torch 2.3.1 wheels are built against numpy 1.x and the vggt fork's
# metadata pins it anyway — make it explicit so later steps can't drift it.
pip install \
    "numpy<2" \
    Pillow \
    open3d \
    huggingface_hub \
    einops \
    safetensors \
    pytorch_metric_learning \
    pytorch-lightning \
    termcolor \
    "viser==0.2.23" \
    tqdm \
    omegaconf \
    opencv-python-headless \
    scipy \
    requests \
    trimesh \
    matplotlib \
    lz4 \
    ftfy \
    regex

echo "[5/8] Installing pinned upstream repos"
# vggt (SPARK fork — adds compute_similarity/image_match_ratio used by solver.py)
pip install "git+https://github.com/MIT-SPARK/VGGT_SPARK.git@${VGGT_SPARK_COMMIT}"
# salad (loop-closure retrieval; provides salad.eval.load_model)
pip install "git+https://github.com/Dominic101/salad.git@${SALAD_COMMIT}"
# vggt_slam itself (setup.py has no install_requires — pure package install)
pip install "git+https://github.com/MIT-SPARK/VGGT-SLAM.git@${VGGT_SLAM_COMMIT}"

echo "[6/8] Installing open-set path (PE + SAM3, --no-deps, guarded — failure does not break SLAM mainline)"
set +e
(
    set -e
    pip install --no-deps "git+https://github.com/facebookresearch/perception_models.git@${PE_COMMIT}"
    pip install --no-deps "git+https://github.com/facebookresearch/sam3.git@${SAM3_COMMIT}"
    # Leaf deps PE/SAM3 need at IMPORT/inference time that the mainline didn't
    # already install (their full metadata pins a training stack — wandb,
    # lm-eval, datatrove, exact-pinned numpy 2.x — which we deliberately do NOT
    # honor; pip's resolver warnings about it are expected and harmless).
    # Kept explicit so torch stays 2.3.1:
    #   setuptools<81 — sam3 imports pkg_resources (removed in setuptools 81+)
    #   iopath / pycocotools — sam3 import chain; timm — PE vision encoder
    pip install "setuptools<81" timm iopath pycocotools
)
OS_INSTALL_RC=$?
set -e
if [ "${OS_INSTALL_RC}" -ne 0 ]; then
    echo "[warn] open-set (PE/SAM3) install FAILED (rc=${OS_INSTALL_RC}); SLAM mainline unaffected — run_os stays unusable until fixed."
fi

echo "[7/8] dino_salad.ckpt + evo + server stack"
pip install gdown evo \
    "uvicorn==0.39.0" \
    "fastapi==0.128.8" \
    "httpx==0.28.1" \
    "msgpack==1.2.1"

HUB_CKPT_DIR="$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints"
mkdir -p "${HUB_CKPT_DIR}"
if [ -f "${HUB_CKPT_DIR}/dino_salad.ckpt" ]; then
    echo "[skip] dino_salad.ckpt already present."
else
    echo "Fetching dino_salad.ckpt -> ${HUB_CKPT_DIR}"
    gdown "${SALAD_CKPT_GDRIVE_ID}" -O "${HUB_CKPT_DIR}/dino_salad.ckpt" || {
        echo "[warn] gdown failed (Drive quota/403?). Download manually from the"
        echo "       salad README pretrained-model link and place it at:"
        echo "       ${HUB_CKPT_DIR}/dino_salad.ckpt"
    }
fi

echo "[8/8] Sanity spike (imports + numpy<2 guard; VGGT weights fetched lazily)"
python - <<'EOF'
import os
import numpy, torch
assert numpy.__version__.startswith("1."), f"numpy must be <2, got {numpy.__version__}"
assert torch.__version__.startswith("2.3.1"), f"torch must be 2.3.1, got {torch.__version__}"
print("torch", torch.__version__, "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())

# GTSAM SL(4) symbols — the load-bearing gtsam surface (vggt_slam/graph.py).
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4  # noqa: F401
import gtsam.symbol_shorthand  # noqa: F401
print("gtsam SL(4) symbols OK")

# VGGT fork symbols the driver uses (main.py / solver.py).
from vggt.models.vggt import VGGT  # noqa: F401
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: F401
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: F401
print("vggt (SPARK fork) OK")

# vggt_slam package — solver pulls in viewer(viser)+loop_closure(salad)+graph(gtsam).
from salad.eval import load_model  # noqa: F401
import vggt_slam.solver, vggt_slam.map, vggt_slam.graph, vggt_slam.loop_closure  # noqa: F401,E401
import vggt_slam.slam_utils, vggt_slam.submap, vggt_slam.frame_overlap  # noqa: F401,E401
print("vggt_slam OK")

ckpt = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
assert os.path.isfile(ckpt), f"missing {ckpt} — Solver() cannot construct without it (loop_closure.py:55)"
print("dino_salad.ckpt present")

# Open-set path — guarded: absence is a warning, not a failure.
try:
    from sam3.model_builder import build_sam3_image_model  # noqa: F401
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    import core.vision_encoder.pe as pe  # noqa: F401
    import core.vision_encoder.transforms as pe_transforms  # noqa: F401
    print("open-set path (PE + SAM3) OK")
except Exception as e:  # pragma: no cover
    print(f"[warn] open-set path unavailable: {type(e).__name__}: {e}")

import fastapi, uvicorn, httpx, msgpack  # noqa: F401,E401  (server stack)
import evo  # noqa: F401  (ATE ruler, evo_ape tum -as)
print("ac-vggt-slam sanity spike passed")
EOF

echo "Done. Env: ${ENV_NAME} — override the interpreter with \$VGGT_SLAM_PYTHON."
