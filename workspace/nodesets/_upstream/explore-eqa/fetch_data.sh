#!/usr/bin/env bash
# Populate data/hm3d/hmeqa/ — runtime data consumed by the hmeqa nodeset.
#
# Files needed:
#   questions.csv          — HM-EQA question set
#   scene_init_poses.csv   — agent start poses per scene
#   Open_Sans/             — font dir for frontier-scoring node label rendering
#
# Depends on fetch_upstream.sh having cloned the repo to ./upstream/.
# Override the upstream path with $1 if you cloned elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
UPSTREAM="${1:-$SCRIPT_DIR/upstream}"
SRC="$UPSTREAM/data"
DST="$REPO_ROOT/data/hm3d/hmeqa"

if [ ! -d "$SRC" ]; then
    echo "[error] $SRC not found." >&2
    echo "        Run fetch_upstream.sh first, or pass the upstream path as \$1." >&2
    exit 1
fi

mkdir -p "$DST"

for f in questions.csv scene_init_poses.csv; do
    if [ -f "$DST/$f" ]; then
        echo "  [skip] $DST/$f already exists"
    elif [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$DST/$f"
        echo "  [ok]   copied $f"
    else
        echo "  [warn] $SRC/$f missing" >&2
    fi
done

if [ -d "$SRC/Open_Sans" ]; then
    if [ -d "$DST/Open_Sans" ]; then
        echo "  [skip] $DST/Open_Sans/ already exists"
    else
        cp -r "$SRC/Open_Sans" "$DST/Open_Sans"
        echo "  [ok]   copied Open_Sans/"
    fi
fi

echo
echo "HM-EQA data populated at $DST"
