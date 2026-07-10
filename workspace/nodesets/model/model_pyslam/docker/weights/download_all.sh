#!/usr/bin/env bash
# Populate the pySLAM external multi-view weights folder (host-side, no container).
#
# Only backends with an EXPLICIT checkpoint URL are fetched here (mast3r, mvdust3r).
# The others (dust3r, vggt, vggt_robust, depth_anything_v3, fast3r) pull their
# weights at RUNTIME from HuggingFace into a mounted cache, so they need no script —
# first use downloads + caches them.
#
# Weights land in $PYSLAM_WEIGHTS_DIR, else <repo>/data/models/pyslam (the runtime
# mount source; see _client.py::_weight_mounts).
#
# Usage:  bash download_all.sh [mast3r|mvdust3r|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
which="${1:-all}"
run() { echo "== $1 =="; bash "$HERE/$1"; }
case "$which" in
    mast3r)   run download_mast3r.sh ;;
    mvdust3r) run download_mvdust3r.sh ;;
    all)      run download_mast3r.sh; run download_mvdust3r.sh ;;
    *) echo "usage: $0 [mast3r|mvdust3r|all]"; exit 2 ;;
esac
echo "[download_all] $which complete."
