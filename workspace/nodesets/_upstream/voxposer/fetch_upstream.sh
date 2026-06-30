#!/usr/bin/env bash
# Fetch the full upstream VoxPoser repo for reference / comparison.
#
# Vendored snippets in workspace/nodesets/server/libero/_perception.py mirror
# the GT-perception surface from src/envs/rlbench_env.py (lines 128-218).
# This script clones the pinned commit so you can diff or pull additional
# code as needed.
#
# Upstream: https://github.com/huangwl18/VoxPoser
# Pinned commit: e3a4c9e57b6ecb45f91e19c80510091c8cbbcbce
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored via
# workspace/nodesets/_upstream/*/upstream/ rule). Override with $1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="e3a4c9e57b6ecb45f91e19c80510091c8cbbcbce"
URL="https://github.com/huangwl18/VoxPoser.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched VoxPoser @ $COMMIT to $DEST"
echo "Diff our vendored impl against upstream:"
echo "  diff -u workspace/nodesets/server/libero/_perception.py $DEST/src/envs/rlbench_env.py"
