#!/usr/bin/env bash
# Preprocess MP3D skybox zips into the merged-and-downsized layout that
# MatterSim actually reads at runtime.
#
# For each scan (by default all 90 from third_party/Matterport3DSimulator/connectivity/scans.txt):
#   1. Unzip {scans_dir}/{scan}/matterport_skybox_images.zip
#   2. Run downsizeWithMerge(scan) from third_party/Matterport3DSimulator/scripts/downsize_skybox.py
#      -> produces {vp}_skybox_small.jpg (3072x512, six 512x512 faces concatenated)
#   3. Delete the extracted raw {vp}_skybox{0-5}_sami.jpg faces
#      (the merged files are what MatterSim reads; raw faces are intermediate only)
#
# Keeps peak extra disk usage per scan to ~263 MB (raw faces) before cleanup;
# final footprint ~1.8 GB for all 90 scans on top of the ~18 GB of zips.
#
# Usage:
#   bash scripts/data/gen_skybox_rgb_mp3d.sh            # all 90 scans
#   bash scripts/data/gen_skybox_rgb_mp3d.sh ac26ZMwG7aT 17DRP5sb8fy  # subset
#   bash scripts/data/gen_skybox_rgb_mp3d.sh --keep-raw ac26ZMwG7aT  # keep raw faces
#   SKIP_CLEANUP=1 bash scripts/data/gen_skybox_rgb_mp3d.sh          # same, env form
#
# Prereqs:
#   - data/mp3d/v1/scans/<scan>/matterport_skybox_images.zip exists per scan
#   - mp3d conda env with cv2 + numpy available
#   - third_party/Matterport3DSimulator/connectivity/scans.txt (submodule bundled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MP3D_SIM="$REPO_ROOT/third_party/Matterport3DSimulator"
SCANS_DIR="$REPO_ROOT/data/mp3d/v1/scans"
SCANS_LIST="$MP3D_SIM/connectivity/scans.txt"
DOWNSIZE_SCRIPT="$MP3D_SIM/scripts/downsize_skybox.py"
DEPTH_HELPER="$MP3D_SIM/scripts/depth_to_skybox.py"

# Resolve cv2-capable python. Prefer explicit override, else mp3d, else vlnce.
PY="${MP3D_PY:-${HOME}/miniforge3/envs/ac-mp3d/bin/python}"
[ -x "$PY" ] || PY="${HOME}/miniforge3/envs/ac-vlnce/bin/python"
if ! "$PY" -c "import cv2, numpy" >/dev/null 2>&1; then
  echo "[gen-rgb][err] python $PY lacks cv2/numpy. Set MP3D_PY=/path/to/python" >&2
  exit 2
fi

KEEP_RAW=${SKIP_CLEANUP:-0}
SCAN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --keep-raw) KEEP_RAW=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) SCAN_ARGS+=("$arg") ;;
  esac
done

# Resolve target scans list.
if [ "${#SCAN_ARGS[@]}" -gt 0 ]; then
  SCANS=("${SCAN_ARGS[@]}")
else
  mapfile -t SCANS < "$SCANS_LIST"
fi

# Sanity.
[ -d "$MP3D_SIM" ]         || { echo "[gen-rgb][err] missing submodule: $MP3D_SIM" >&2; exit 2; }
[ -f "$DOWNSIZE_SCRIPT" ]  || { echo "[gen-rgb][err] missing script: $DOWNSIZE_SCRIPT" >&2; exit 2; }
[ -f "$DEPTH_HELPER" ]     || { echo "[gen-rgb][err] missing helper: $DEPTH_HELPER" >&2; exit 2; }
[ -d "$SCANS_DIR" ]        || { echo "[gen-rgb][err] missing scans dir: $SCANS_DIR (run fetch_scans_mp3d.py first)" >&2; exit 2; }

# downsize_skybox.py hardcodes base_dir='data/v1/scans' and imports
# depth_to_skybox. Work around both by chdir-ing to MP3D_SIM, symlinking
# data/v1/scans -> our real scans dir, and injecting scripts/ onto sys.path.
WORKDIR="$MP3D_SIM"

# Ensure data/v1/scans inside the submodule resolves to our real scans dir.
# Create a single-level symlink so the hardcoded 'data/v1/scans' path works.
cd "$WORKDIR"
mkdir -p data/v1
if [ -L data/v1/scans ]; then
  current="$(readlink data/v1/scans)"
  if [ "$current" != "$SCANS_DIR" ]; then
    rm data/v1/scans
    ln -s "$SCANS_DIR" data/v1/scans
  fi
elif [ ! -e data/v1/scans ]; then
  ln -s "$SCANS_DIR" data/v1/scans
elif [ -d data/v1/scans ] && [ ! -L data/v1/scans ]; then
  echo "[gen-rgb][err] $WORKDIR/data/v1/scans is a real dir (not a symlink). Manual cleanup required." >&2
  exit 2
fi

processed=0
skipped=0
failed=0
total=${#SCANS[@]}

for scan in "${SCANS[@]}"; do
  [ -n "$scan" ] || continue
  scan_dir="$SCANS_DIR/$scan"
  skybox_dir="$scan_dir/matterport_skybox_images"
  zip_path="$skybox_dir.zip"

  # Detect fully-processed scan: any *_skybox_small.jpg present AND no raw faces left.
  # find returns non-zero on missing dirs; mask with `|| true` so pipefail doesn't trip.
  if [ -d "$skybox_dir" ]; then
    small_count=$(find "$skybox_dir" -maxdepth 1 -name '*_skybox_small.jpg' 2>/dev/null | wc -l)
    raw_count=$(find "$skybox_dir" -maxdepth 1 -name '*_skybox[0-5]_sami.jpg' 2>/dev/null | wc -l)
  else
    small_count=0
    raw_count=0
  fi
  if [ "$small_count" -gt 0 ] && [ "$raw_count" -eq 0 ]; then
    echo "[gen-rgb] [skip-done] $scan ($small_count panoramas already merged)"
    skipped=$((skipped + 1))
    continue
  fi

  if [ ! -f "$zip_path" ] && [ "$raw_count" -eq 0 ]; then
    echo "[gen-rgb] [skip-missing] $scan (no zip, no raw faces)"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[gen-rgb] [$((processed + skipped + failed + 1))/$total] $scan"

  # Step 1: unzip (if raw faces aren't already extracted).
  if [ "$raw_count" -eq 0 ]; then
    if [ ! -f "$zip_path" ]; then
      echo "[gen-rgb][err]   $scan: zip missing at $zip_path" >&2
      failed=$((failed + 1))
      continue
    fi
    # Zip contains `<scan>/matterport_skybox_images/*.jpg` at top level,
    # so extract under $SCANS_DIR (not $scan_dir) to land at the right depth.
    echo "[gen-rgb]   unzip -> $skybox_dir"
    if ! unzip -q -o "$zip_path" -d "$SCANS_DIR"; then
      echo "[gen-rgb][err]   $scan: unzip failed" >&2
      failed=$((failed + 1))
      continue
    fi
  else
    echo "[gen-rgb]   raw faces already extracted ($raw_count files) — reusing"
  fi

  # Step 2: merge 6 skybox faces per panorama into a 3072x512 concat.
  # Inline impl (doesn't need undistorted_camera_parameters/*.conf that
  # downsize_skybox.camera_parameters requires — that bundle is NOT
  # downloaded for skybox-only rigs).
  if ! SKYBOX_DIR="$skybox_dir" "$PY" - <<'PY'
import os, re, sys, glob
import cv2
import numpy as np

skybox_dir = os.environ["SKYBOX_DIR"]
raw_glob = os.path.join(skybox_dir, "*_skybox0_sami.jpg")
pano_ids = sorted({
    os.path.basename(p).split("_skybox0_sami.jpg")[0]
    for p in glob.glob(raw_glob)
})
if not pano_ids:
    print(f"[err] no *_skybox0_sami.jpg under {skybox_dir}", file=sys.stderr)
    sys.exit(2)

print(f"[inline-merge] {len(pano_ids)} panoramas")
W = H = 512
for pano in pano_ids:
    faces = []
    for i in range(6):
        fp = os.path.join(skybox_dir, f"{pano}_skybox{i}_sami.jpg")
        im = cv2.imread(fp)
        if im is None:
            print(f"[err] unreadable: {fp}", file=sys.stderr)
            sys.exit(3)
        faces.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
    merged = np.concatenate(faces, axis=1)
    out = os.path.join(skybox_dir, f"{pano}_skybox_small.jpg")
    ok = cv2.imwrite(out, merged)
    if not ok:
        print(f"[err] imwrite failed: {out}", file=sys.stderr)
        sys.exit(4)
PY
  then
    echo "[gen-rgb][err]   $scan: merge step failed" >&2
    failed=$((failed + 1))
    continue
  fi

  # Step 3: cleanup raw faces (unless --keep-raw).
  if [ "$KEEP_RAW" = "0" ]; then
    removed=$(find "$skybox_dir" -maxdepth 1 -name '*_skybox[0-5]_sami.jpg' -delete -print 2>/dev/null | wc -l)
    echo "[gen-rgb]   cleanup: removed $removed raw face files"
  else
    echo "[gen-rgb]   keep-raw: leaving raw faces in place"
  fi

  merged=$(find "$skybox_dir" -maxdepth 1 -name '*_skybox_small.jpg' 2>/dev/null | wc -l)
  echo "[gen-rgb]   ok: $merged panoramas merged"
  processed=$((processed + 1))
done

echo ""
echo "[gen-rgb] done: processed=$processed skipped=$skipped failed=$failed total=$total"
if [ "$failed" -gt 0 ]; then
  exit 1
fi
