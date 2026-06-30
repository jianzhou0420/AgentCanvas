#!/usr/bin/env bash
# Fetch the full upstream NavGPT repo for reference / data extraction.
#
# Vendored attribution in workspace/nodesets/navgpt_mp3d_tools.py and the
# data path default in workspace/nodesets/server/matterport3d.py
# (_NAVGPT_OBS_ROOT) point at this repo's `datasets/R2R/` sub-tree
# (pre-computed per-viewpoint scene descriptions).
#
# Upstream: https://github.com/GengzeZhou/NavGPT
# Pinned commit: b3fc8a21b8a1f66b09dde833bcfc25767ef1962d
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored via
# workspace/nodesets/_upstream/*/upstream/ rule). Override with $1.
#
# To populate the runtime data layout (data/navgpt/r2r/), run fetch_data.sh
# after this completes — it copies the relevant sub-trees from ./upstream/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="b3fc8a21b8a1f66b09dde833bcfc25767ef1962d"
URL="https://github.com/GengzeZhou/NavGPT.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched NavGPT @ $COMMIT to $DEST"
echo "Next step: bash $SCRIPT_DIR/fetch_data.sh   # copies datasets/R2R into data/navgpt/r2r/"
