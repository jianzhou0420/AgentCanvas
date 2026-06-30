#!/usr/bin/env bash
# Fetch the full upstream explore-eqa repo for reference / data extraction.
#
# Vendored content:
#   - workspace/nodesets/server/_explore_eqa_tsdf.py (verbatim from
#     src/tsdf.py + src/geom.py, see file header for the 3 tweaks)
#   - Per-step / per-frontier numeric defaults in the hmeqa nodeset
#     mirror cfg/vlm_exp.yaml (see docs/pages/developer-guide/
#     nodesets/explore-eqa.html for the full mapping)
#
# Data files at data/hm3d/hmeqa/ (questions.csv, scene_init_poses.csv,
# Open_Sans/) come from upstream's data/ subtree — see fetch_data.sh.
#
# Upstream: https://github.com/Stanford-ILIAD/explore-eqa
# Pinned commit: 18381da3370c5f7729594f10ebc49b5644bbb88c
# License: UNSPECIFIED — upstream has no LICENSE file as of this commit.
# Note: also fetches the nested prismatic-vlms submodule (MIT, see
#       ../prismatic-vlms/LICENSE).
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="18381da3370c5f7729594f10ebc49b5644bbb88c"
URL="https://github.com/Stanford-ILIAD/explore-eqa.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
git -C "$DEST" submodule update --init --recursive
echo
echo "Fetched explore-eqa @ $COMMIT to $DEST (with prismatic-vlms nested submodule)"
echo "Next step (data files): bash $SCRIPT_DIR/fetch_data.sh"
