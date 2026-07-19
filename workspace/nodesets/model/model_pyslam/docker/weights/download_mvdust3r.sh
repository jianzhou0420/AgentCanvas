#!/usr/bin/env bash
# Download the MV-DUSt3R checkpoints into the external weights folder.
#
# Target folder: $PYSLAM_WEIGHTS_DIR, else <repo>/data/models/pyslam (the runtime
# mount source; see _client.py::_weight_mounts). Runs on the HOST — no container.
# URLs copied verbatim from pyslam thirdparty/mvdust3r_scripts/download_models.py
# (HuggingFace Zhenggang/MV-DUSt3R).
#
# NOTE: MV-DUSt3R is currently an annotated KNOWN GAP in the nodeset — pySLAM's
# bundled mvdust3r has a version drift against its own dust3r copy (import error
# on normalize_pointclouds). These weights are provided for completeness; the
# other multi-view backends (MASt3R / DUSt3R / VGGT) cover reconstruction today.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PYSLAM_WEIGHTS_DIR:-}" ]; then
    WEIGHTS_DIR="$PYSLAM_WEIGHTS_DIR"
else
    ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)"
    [ -z "$ROOT" ] && ROOT="$(cd "$HERE/../../../../../.." && pwd)"
    WEIGHTS_DIR="$ROOT/data/models/pyslam"
fi
DEST="$WEIGHTS_DIR/mvdust3r/checkpoints"
mkdir -p "$DEST"
BASE="https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints"
FILES=(
    "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    "MVD.pth"
    "MVDp_s1.pth"
    "MVDp_s2.pth"
)
for f in "${FILES[@]}"; do
    if [ -f "$DEST/$f" ]; then
        echo "[mvdust3r] already present: $f"
    else
        echo "[mvdust3r] downloading $f -> $DEST ..."
        wget -c -O "$DEST/$f" "$BASE/$f"
    fi
done
echo "[mvdust3r] done -> $DEST"
