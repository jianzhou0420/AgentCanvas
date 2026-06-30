#!/usr/bin/env bash
# Open-Nav data + checkpoint setup.
#
# Downloads the four ML artifacts that the AgentCanvas Open-Nav port needs
# at runtime:
#
#   1. RAM Swin-L 14M tag model       (recognise-anything)
#   2. SpatialBot-3B VLM               (Hugging Face)
#   3. Open-Nav waypoint predictor     (BinaryDistPredictor_TRM)
#   4. DDPPO ResNet50 depth encoder    (gibson-2plus-resnet50.pth)
#
# Companion of:
#   scripts/install/envs/opennav.yaml             (perception env)
#   workspace/nodesets/server/opennav_waypoint.py (waypoint predictor)
#   workspace/nodesets/server/opennav_perception.py (RAM + SpatialBot)
#
# Usage:
#   bash scripts/data/fetch_ckpt_opennav.sh [--data-dir <dir>]
#
# Default DATA_DIR is data/opennav/ (intentionally top-level — these are
# tool-agnostic perception models shared across env nodesets, not tied to
# any particular simulator framework). Override with --data-dir or by setting
# the OPENNAV_DATA_DIR environment variable. The corresponding env vars
# (OPENNAV_RAM_CKPT, OPENNAV_SPATIALBOT_PATH, OPENNAV_WAYPOINT_CKPT,
# OPENNAV_DDPPO_CKPT) are echoed at the end so you can copy them into your
# shell or .env file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${OPENNAV_DATA_DIR:-${REPO_ROOT}/data/opennav}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${DATA_DIR}"

echo ">>> Open-Nav data setup"
echo "    DATA_DIR = ${DATA_DIR}"
echo

# ── 1. RAM Swin-L 14M ────────────────────────────────────────────────
RAM_CKPT="${DATA_DIR}/ram_swin_large_14m.pth"
if [[ -f "${RAM_CKPT}" ]]; then
  echo "[skip] RAM Swin-L already present: ${RAM_CKPT}"
else
  echo "[1/4] Downloading RAM Swin-L 14M (~5.6 GB) ..."
  RAM_URL="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth"
  curl -L --fail -o "${RAM_CKPT}" "${RAM_URL}" \
    || wget -O "${RAM_CKPT}" "${RAM_URL}"
fi

# ── 2. SpatialBot-3B ─────────────────────────────────────────────────
SPATIALBOT_DIR="${DATA_DIR}/SpatialBot-3B"
if [[ -d "${SPATIALBOT_DIR}" && -n "$(ls -A "${SPATIALBOT_DIR}" 2>/dev/null)" ]]; then
  echo "[skip] SpatialBot-3B already present: ${SPATIALBOT_DIR}"
else
  echo "[2/4] Cloning SpatialBot-3B from Hugging Face (large; uses git-lfs) ..."
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "ERROR: git-lfs is required to fetch SpatialBot-3B. Install it first." >&2
    exit 1
  fi
  git lfs install --skip-smudge >/dev/null
  git clone "https://huggingface.co/RussRobin/SpatialBot-3B" "${SPATIALBOT_DIR}"
  (cd "${SPATIALBOT_DIR}" && git lfs pull)
fi

# ── 3. Waypoint predictor checkpoint ─────────────────────────────────
# Hosted on Google Drive by the Open-Nav authors — RGB-D FoV 90 weights used
# in the paper. File ID from the README: 16Vk3ummmyLvpQr16TzBL-iwZNlrELOdk.
WAYPOINT_CKPT="${DATA_DIR}/check_val_best_avg_wayscore"
if [[ -f "${WAYPOINT_CKPT}" ]]; then
  echo "[skip] Waypoint predictor already present: ${WAYPOINT_CKPT}"
else
  echo "[3/4] Downloading Open-Nav waypoint predictor (Google Drive via gdown) ..."
  if ! command -v gdown >/dev/null 2>&1; then
    pip install --quiet gdown
  fi
  gdown "https://drive.google.com/uc?id=16Vk3ummmyLvpQr16TzBL-iwZNlrELOdk" -O "${WAYPOINT_CKPT}" \
    || echo "      WARNING: gdown failed — fetch manually from"
    echo "      https://drive.google.com/file/d/16Vk3ummmyLvpQr16TzBL-iwZNlrELOdk/view"
fi

# ── 4. DDPPO ResNet50 depth encoder ──────────────────────────────────
# Zenodo mirror — the original facebookresearch URL no longer serves the file.
DDPPO_DIR="${DATA_DIR}/ddppo-models"
DDPPO_CKPT="${DDPPO_DIR}/gibson-2plus-resnet50.pth"
if [[ -f "${DDPPO_CKPT}" ]]; then
  echo "[skip] DDPPO depth encoder already present: ${DDPPO_CKPT}"
else
  echo "[4/4] Downloading DDPPO gibson-2plus-resnet50 depth encoder (Zenodo) ..."
  mkdir -p "${DDPPO_DIR}"
  DDPPO_URL="https://zenodo.org/record/6634113/files/gibson-2plus-resnet50.pth"
  curl -L --fail -o "${DDPPO_CKPT}" "${DDPPO_URL}" \
    || wget -O "${DDPPO_CKPT}" "${DDPPO_URL}"
fi

cat <<EOF

>>> Open-Nav data setup complete.

Add these to your shell or .env (the AgentCanvas Open-Nav nodesets read
them at server startup):

  export OPENNAV_RAM_CKPT="${RAM_CKPT}"
  export OPENNAV_SPATIALBOT_PATH="${SPATIALBOT_DIR}"
  export OPENNAV_WAYPOINT_CKPT="${WAYPOINT_CKPT}"
  export OPENNAV_DDPPO_CKPT="${DDPPO_CKPT}"

Conda env: bash scripts/install/install_opennav.sh   (or:
  conda env create -f scripts/install/envs/opennav.yaml)
EOF
