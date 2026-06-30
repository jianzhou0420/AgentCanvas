#!/usr/bin/env bash
# Fetch the full upstream OpenEQA repo for reference / data extraction.
#
# At runtime we only use data/open-eqa-v0.json (~466K, question set).
# The runtime copy lives at data/openeqa/open-eqa-v0.json — see
# scripts/data/fetch_dataset_openeqa.sh for the curl-based fetch path.
# This clone is for full-tree comparison (eval harness, code samples).
#
# Upstream: https://github.com/facebookresearch/open-eqa
# Pinned commit: cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b
# License: MIT (see ./LICENSE)
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b"
URL="https://github.com/facebookresearch/open-eqa.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched OpenEQA @ $COMMIT to $DEST"
