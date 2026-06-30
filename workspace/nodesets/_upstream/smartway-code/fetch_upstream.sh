#!/usr/bin/env bash
# Fetch upstream SmartWay-Code repo for reference / re-vendoring.
#
# Vendored at:
#   workspace/nodesets/server/smartway_waypoint/_vendored/waypoint_predictor/
# (verbatim copy of the upstream waypoint_predictor/ sub-tree). The
# server-mode engine sys.path-injects two paths at load time:
#   - the _vendored/ root (for `from waypoint_predictor.TRM_net import ...`)
#   - _vendored/waypoint_predictor/ itself (for bare `import utils` inside TRM_net)
#
# To re-vendor from a fresh upstream snapshot:
#   bash $0                  # clones to ./upstream/
#   rm -rf ../../server/smartway_waypoint/_vendored/waypoint_predictor
#   cp -r ./upstream/waypoint_predictor ../../server/smartway_waypoint/_vendored/
#
# Upstream: https://github.com/sxyxs/SmartWay-Code
# Pinned commit: daa2dd856872727832b4b07cd3a09db34cf211d4
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="daa2dd856872727832b4b07cd3a09db34cf211d4"
URL="https://github.com/sxyxs/SmartWay-Code.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched SmartWay-Code @ $COMMIT to $DEST"
