#!/usr/bin/env bash
# Fetch upstream Open-Nav repo for reference / re-vendoring.
#
# Vendored at:
#   workspace/nodesets/server/opennav_waypoint/_vendored/waypoint_prediction/
# (verbatim copy of the upstream waypoint_prediction/ sub-tree). The
# server-mode engine sys.path-injects this directory at load time so
# `from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM` works.
#
# To re-vendor from a fresh upstream snapshot:
#   bash $0                  # clones to ./upstream/
#   rm -rf ../../server/opennav_waypoint/_vendored/waypoint_prediction
#   cp -r ./upstream/waypoint_prediction ../../server/opennav_waypoint/_vendored/
#
# Upstream: https://github.com/YanyuanQiao/Open-Nav
# Pinned commit: 3a8dcefe5bfdab5192c3c3bf80b14fb096cb08c7
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="3a8dcefe5bfdab5192c3c3bf80b14fb096cb08c7"
URL="https://github.com/YanyuanQiao/Open-Nav.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched Open-Nav @ $COMMIT to $DEST"
