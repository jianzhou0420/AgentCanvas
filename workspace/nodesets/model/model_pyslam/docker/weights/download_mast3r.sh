#!/usr/bin/env bash
# Download the MASt3R metric checkpoint into the external weights folder.
#
# The pySLAM multi-view backends' weights are NOT baked into the image (the image
# carries only code + compiled curope); they live in a host folder that mounts
# read-only into the container at runtime. This script populates that folder.
#
# Target folder: $PYSLAM_WEIGHTS_DIR, else <repo>/data/models/pyslam (the runtime
# mount source; see _client.py::_weight_mounts). Runs on the HOST — no container.
# URL copied verbatim from pyslam scripts/install_thirdparty.sh (naver labs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PYSLAM_WEIGHTS_DIR:-}" ]; then
    WEIGHTS_DIR="$PYSLAM_WEIGHTS_DIR"
else
    ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)"
    [ -z "$ROOT" ] && ROOT="$(cd "$HERE/../../../../../.." && pwd)"
    WEIGHTS_DIR="$ROOT/data/models/pyslam"
fi
DEST="$WEIGHTS_DIR/mast3r/checkpoints"
mkdir -p "$DEST"
URL="https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
FILE="$DEST/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ -f "$FILE" ]; then
    echo "[mast3r] already present: $FILE"
else
    echo "[mast3r] downloading $(basename "$FILE") (~2.5GB) -> $DEST ..."
    wget -c -O "$FILE" "$URL"
fi
echo "[mast3r] done -> $FILE"
