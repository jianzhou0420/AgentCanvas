#!/bin/bash
# =============================================================================
# SmartWay Data Setup
# =============================================================================
# Idempotent ckpt downloads + symlinks for the SmartWay nodeset stack.
#
# Targets under PROJECT_ROOT/data/smartway/:
#
#   waypoint_ckpt/best.pth                Enhanced waypoint predictor
#                                         (DINOv2 + masked cross-attn + TRM head).
#                                         Source: Google Drive (upstream README).
#   ram_plus/ram_plus_swin_large_14m.pth  RAM+ Plus tagging model (swin_l, ~1.5GB).
#                                         Source: huggingface.co (recognize-anything).
#   ddppo/gibson-2plus-resnet50.pth       DDPPO depth encoder — symlink shared
#                                         with the existing data/habitat/ddppo-models/
#                                         (vlnce + opennav already use it).
#
# Each target is gated by [[ -f ... ]] → skip-if-present, so the script can be
# re-run safely.
#
# Usage:
#   bash scripts/data/fetch_ckpt_smartway.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/smartway"

# Interpreter for gdown. Honor $PYTHON (install_ac_smartway.sh passes
# SMARTWAY_PYTHON so this resolves under non-interactive spawn where no env is
# activated and a bare `python` is off PATH); fall back to system python3.
PY="${PYTHON:-python3}"

# Upstream README points at this Google Drive file id for the waypoint ckpt.
WAYPOINT_GDRIVE_ID="1TsKqtdR1oir4UFIGhq15ffETz6r8-P2Q"
# RAM+ HF release; switch to a mirror if HF rate-limits.
RAM_PLUS_URL="https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_plus_swin_large_14m.pth"

echo "=== SmartWay Data Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Data dir:     $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR/waypoint_ckpt" "$DATA_DIR/ram_plus"

# ── Step 1: Waypoint predictor checkpoint (Google Drive) ────────────
WAYP_CKPT="$DATA_DIR/waypoint_ckpt/best.pth"
if [[ -f "$WAYP_CKPT" ]]; then
    echo "  [skip] waypoint_ckpt/best.pth ($(du -h "$WAYP_CKPT" | cut -f1))"
else
    echo "  [get]  waypoint_ckpt/best.pth via gdown ($WAYPOINT_GDRIVE_ID)"
    if ! "$PY" -c "import gdown" >/dev/null 2>&1; then
        echo "    Installing gdown into $PY ..."
        "$PY" -m pip install --quiet gdown
    fi
    "$PY" -m gdown --id "$WAYPOINT_GDRIVE_ID" -O "$WAYP_CKPT" || {
        echo "    [WARN] gdown failed — manually download from"
        echo "    https://drive.google.com/file/d/$WAYPOINT_GDRIVE_ID/view"
        echo "    and place at $WAYP_CKPT"
    }
fi

# ── Step 2: RAM+ weights ────────────────────────────────────────────
RAM_PLUS_CKPT="$DATA_DIR/ram_plus/ram_plus_swin_large_14m.pth"
if [[ -f "$RAM_PLUS_CKPT" ]]; then
    echo "  [skip] ram_plus/ram_plus_swin_large_14m.pth ($(du -h "$RAM_PLUS_CKPT" | cut -f1))"
else
    echo "  [get]  ram_plus/ram_plus_swin_large_14m.pth (~1.5GB)"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$RAM_PLUS_CKPT" "$RAM_PLUS_URL" || {
            echo "    [WARN] curl failed — try wget or manual download from"
            echo "    https://github.com/xinyu1205/recognize-anything releases"
        }
    else
        wget -O "$RAM_PLUS_CKPT" "$RAM_PLUS_URL"
    fi
fi

# ── Step 3: DDPPO depth encoder — symlink to shared dir ─────────────
DDPPO_SRC="$PROJECT_ROOT/data/habitat/ddppo-models"
DDPPO_LINK="$DATA_DIR/ddppo"
if [[ -L "$DDPPO_LINK" ]]; then
    current="$(readlink "$DDPPO_LINK")"
    if [[ "$current" == "../habitat/ddppo-models" ]]; then
        echo "  [skip] ddppo (already symlinked)"
    else
        rm "$DDPPO_LINK"
        ln -s "../habitat/ddppo-models" "$DDPPO_LINK"
        echo "  [link] ddppo -> ../habitat/ddppo-models (refreshed)"
    fi
elif [[ -e "$DDPPO_LINK" ]]; then
    echo "  [WARN] $DDPPO_LINK exists and is not a symlink — leaving alone"
elif [[ -d "$DDPPO_SRC" ]]; then
    ln -s "../habitat/ddppo-models" "$DDPPO_LINK"
    echo "  [link] ddppo -> ../habitat/ddppo-models"
else
    echo "  [SKIP] $DDPPO_SRC missing — run install_ac_vlnce.sh or download"
    echo "         gibson-2plus-resnet50.pth from zenodo.org/record/6634113"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Done. Set these env vars (or accept defaults under $DATA_DIR/): ==="
echo "  export SMARTWAY_PYTHON=/home/\$(whoami)/miniforge3/envs/ac-smartway/bin/python"
echo "  export SMARTWAY_WAYPOINT_CKPT=$WAYP_CKPT"
echo "  export SMARTWAY_RAM_PLUS_CKPT=$RAM_PLUS_CKPT"
echo "  export SMARTWAY_DDPPO_CKPT=$DATA_DIR/ddppo/gibson-2plus-resnet50.pth"
echo ""
