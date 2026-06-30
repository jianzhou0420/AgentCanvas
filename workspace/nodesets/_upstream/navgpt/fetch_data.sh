#!/usr/bin/env bash
# Populate data/navgpt/r2r/ — pre-computed observation data consumed by
# workspace/nodesets/server/matterport3d.py via _NAVGPT_OBS_ROOT.
#
# This script depends on fetch_upstream.sh having already cloned NavGPT
# to ./upstream/ (or pass a custom upstream path as $1).
#
# Note: as of pinned commit b3fc8a2, the upstream NavGPT repo does NOT
# include datasets/R2R/ — these are typically distributed separately
# (see upstream README / paper appendix for download instructions, or
# contact authors). This script is a placeholder for the copy step once
# the data is on hand.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
UPSTREAM="${1:-$SCRIPT_DIR/upstream}"
DST="$REPO_ROOT/data/navgpt/r2r"

if [ ! -d "$UPSTREAM" ]; then
    echo "[error] upstream clone not found at $UPSTREAM" >&2
    echo "        run fetch_upstream.sh first" >&2
    exit 1
fi

SRC="$UPSTREAM/datasets/R2R"
if [ ! -d "$SRC" ]; then
    echo "[warn] datasets/R2R/ not present in upstream clone — NavGPT ships" >&2
    echo "       these as separate downloads. See upstream README or paper" >&2
    echo "       for the data release; place files under:" >&2
    echo "         $DST/observations_list_summarized/{scan}.json" >&2
    echo "         $DST/observations_summarized/{scan}_summarized.json" >&2
    echo "         $DST/objects_list/{scan}.json" >&2
    exit 1
fi

mkdir -p "$DST"
cp -r "$SRC"/* "$DST/"
echo "Copied $SRC → $DST"
echo "matterport3d nodeset will pick this up automatically (default path)."
