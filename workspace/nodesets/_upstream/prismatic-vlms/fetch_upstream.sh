#!/usr/bin/env bash
# Fetch upstream prismatic-vlms — the VLM package the hmeqa nodeset
# depends on (loaded inside ExploreEQANodeSet.initialize()).
#
# Runtime install path: scripts/install/install_ac_hmeqa.sh uses
#   pip install git+https://github.com/allenzren/prismatic-vlms.git@<commit>
# so the package lands in the hmeqa conda env normally — no clone
# under third_party/ needed.
#
# This script is for source-level inspection / local modification only.
# If you edit code under ./upstream/, reinstall editable:
#   pip install -e $SCRIPT_DIR/upstream
#
# Upstream: https://github.com/allenzren/prismatic-vlms
# Pinned commit: 7573aeb4f8cb49b4107b6ef0dc7845377c57b4a7
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="7573aeb4f8cb49b4107b6ef0dc7845377c57b4a7"
URL="https://github.com/allenzren/prismatic-vlms.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched prismatic-vlms @ $COMMIT to $DEST"
echo "To install editable into hmeqa env: pip install -e $DEST"
