#!/usr/bin/env bash
# Fetch a TUM RGB-D benchmark sequence into data/tum/ (gitignored).
#
# Default: fr3/long_office_household (~1.4 GB) — a long handheld trajectory with
# a big loop closure, pySLAM's showcase sequence. Pass a different sequence name
# as $1 (e.g. rgbd_dataset_freiburg1_xyz) to fetch another.
#
# Source: TUM Computer Vision Group RGB-D SLAM dataset
#   https://vision.in.tum.de/data/datasets/rgbd-dataset/download
#
# Usage:
#   bash workspace/nodesets/env/env_tum/download.sh
#   bash workspace/nodesets/env/env_tum/download.sh rgbd_dataset_freiburg1_xyz
set -euo pipefail

SEQ="${1:-rgbd_dataset_freiburg3_long_office_household}"

# Derive the freiburgN folder on the TUM server from the sequence name.
case "$SEQ" in
  *freiburg1*) GRP="freiburg1" ;;
  *freiburg2*) GRP="freiburg2" ;;
  *freiburg3*) GRP="freiburg3" ;;
  *) echo "ERROR: cannot infer freiburg group from '$SEQ'"; exit 1 ;;
esac

# Repo root = four levels up from this script (env_tum/ → env/ → nodesets/ → workspace/ → root)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DEST="${TUM_DATA_ROOT:-$ROOT/data/tum}"
URL="https://vision.in.tum.de/rgbd/dataset/${GRP}/${SEQ}.tgz"
TGZ="$DEST/${SEQ}.tgz"

mkdir -p "$DEST"

if [ -d "$DEST/$SEQ" ] && [ -f "$DEST/$SEQ/rgb.txt" ]; then
  echo "Already present: $DEST/$SEQ (rgb.txt found) — skipping download."
  exit 0
fi

echo "Downloading $SEQ"
echo "  from $URL"
echo "  to   $TGZ"
# -C - resumes a partial download; --fail surfaces a 404 instead of saving HTML.
curl -L --fail -C - -o "$TGZ" "$URL"

echo "Extracting into $DEST ..."
tar -xzf "$TGZ" -C "$DEST"

if [ -f "$DEST/$SEQ/rgb.txt" ]; then
  echo "Done: $DEST/$SEQ"
  echo "  rgb frames:   $(grep -vc '^#' "$DEST/$SEQ/rgb.txt")"
  echo "  depth frames: $(grep -vc '^#' "$DEST/$SEQ/depth.txt")"
  rm -f "$TGZ"
else
  echo "ERROR: extraction did not yield $DEST/$SEQ/rgb.txt"; exit 1
fi
