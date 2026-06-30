#!/bin/bash
# =============================================================================
# OpenEQA (EM-EQA) Dataset Setup — full HM3D + ScanNet via AIGeeksGroup mirror
# =============================================================================
# Stages the on-disk layout consumed by `workspace/nodesets/server/openeqa.py`:
#
#   data/openeqa/
#       open-eqa-v0.json                                # symlink from third_party
#       episodes/hm3d-v0/<episode>/                     # extracted from .tar
#           00000-rgb.png, 00001-rgb.png, ...           # 1920×1080 PNG (lossless)
#       episodes/scannet-v0/<episode>/
#           000000-rgb.png, 000000-depth.png, 000000.txt, ...
#           intrinsic_color.txt, intrinsic_depth.txt, extrinsic_depth.txt
#
# Why AIGeeksGroup mirror?
#
#   - Original Meta Dropbox tarball is permanently expired (HTTP 200 returns
#     "Link Expired" HTML, verified 2026-05-04).
#   - Previous mirror `Embodied1/open-eqa` (parquet) is hard-capped at
#     32 frames/episode — multiframe graphs with K>32 silently truncate.
#   - `AIGeeksGroup/OpenEQA` provides full episodes as per-episode tarballs:
#       * HM3D:    1920×1080 PNG, lossless, paper-canonical resolution
#                  (matches `data/hm3d/extract-frames.py` script defaults).
#       * ScanNet: 1296×968 RGB PNG + 640×480 uint16 depth + per-frame
#                  4×4 camera-to-world pose + color/depth intrinsics +
#                  extrinsics. All lossless, paper-canonical resolutions.
#                  Bonus depth/pose useful for SpatialNav-class methods.
#
# Total download: ~87 GB (12 GB HM3D + 75 GB ScanNet) over 152 .tar files.
# Final on-disk: ~87 GB (the .tar content unpacks ~1:1; PNGs already compressed).
#
# What this script does:
#   1. Curls open-eqa-v0.json from upstream (facebookresearch/open-eqa @ pinned commit)
#   2. Downloads `AIGeeksGroup/OpenEQA` via huggingface_hub.snapshot_download
#      to `_downloads/aig/` (resumable; HF Hub handles retries + chunked).
#   3. Extracts each `.tar` to `episodes/<subset>/<episode-id>/`.
#   4. Idempotent: skips episodes that already have ≥ 1 `*-rgb.png` file.
#
# Usage:
#   bash scripts/data/fetch_dataset_openeqa.sh                  # JSON + HM3D + ScanNet
#   bash scripts/data/fetch_dataset_openeqa.sh --json-only      # JSON only
#   bash scripts/data/fetch_dataset_openeqa.sh --filter hm3d    # JSON + HM3D
#   bash scripts/data/fetch_dataset_openeqa.sh --filter scannet # JSON + ScanNet
#   bash scripts/data/fetch_dataset_openeqa.sh --keep-tars      # don't delete _downloads after extract
#   bash scripts/data/fetch_dataset_openeqa.sh -y               # skip prompt
#
# Env overrides:
#   OPENEQA_DATA_DIR        Target root (default: $PROJECT_ROOT/data/openeqa)
#   OPENEQA_PYTHON          Python with huggingface_hub installed
#                           (default: $CONDA_PREFIX/envs/agentcanvas/bin/python
#                            falling back to PATH `python3`)
#   HF_HOME                 HuggingFace cache root (default: ~/.cache/huggingface)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${OPENEQA_DATA_DIR:-$PROJECT_ROOT/data/openeqa}"
QUESTIONS_JSON="$DATA_DIR/open-eqa-v0.json"
EPISODES_DIR="$DATA_DIR/episodes"
DOWNLOADS_DIR="$DATA_DIR/_downloads/aig"
OPENEQA_COMMIT="cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b"
OPENEQA_JSON_URL="https://raw.githubusercontent.com/facebookresearch/open-eqa/${OPENEQA_COMMIT}/data/open-eqa-v0.json"

HF_REPO="AIGeeksGroup/OpenEQA"
MIN_FREE_GB=100      # ~87 GB final + headroom

SKIP_PROMPT=0
JSON_ONLY=0
FILTER="all"
KEEP_TARS=0

usage() {
    sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes)        SKIP_PROMPT=1 ;;
        --json-only)     JSON_ONLY=1 ;;
        --keep-tars)     KEEP_TARS=1 ;;
        --filter)
            shift
            case "${1:-}" in
                hm3d|scannet|all) FILTER="$1" ;;
                *) echo "--filter must be hm3d|scannet|all" >&2; exit 2 ;;
            esac
            ;;
        -h|--help)       usage ;;
        *) echo "Unknown arg: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# ── Pick Python with huggingface_hub ──────────────────────────────────────
PYTHON="${OPENEQA_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in \
        "${CONDA_PREFIX:-/opt/conda}/envs/agentcanvas/bin/python" \
        "${HOME}/miniforge3/envs/agentcanvas/bin/python" \
        "$(command -v python3 || true)"; do
        if [ -x "$cand" ] && "$cand" -c 'import huggingface_hub' 2>/dev/null; then
            PYTHON="$cand"
            break
        fi
    done
fi
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import huggingface_hub' 2>/dev/null; then
    echo "[ERROR] No Python with huggingface_hub found." >&2
    echo "        Set OPENEQA_PYTHON to a venv with huggingface_hub installed." >&2
    exit 1
fi

mkdir -p "$DATA_DIR" "$EPISODES_DIR" "$DOWNLOADS_DIR"

echo "=== OpenEQA setup (full, via AIGeeksGroup mirror) ==="
echo "  Project root:  $PROJECT_ROOT"
echo "  Data dir:      $DATA_DIR"
echo "  Filter:        $FILTER"
echo "  Python:        $PYTHON"
echo "  HF repo:       $HF_REPO"
echo ""

# ── Step 1: question JSON ────────────────────────────────────────────────
echo "=== Step 1: question JSON ==="
if [ -e "$QUESTIONS_JSON" ]; then
    n_json="$("$PYTHON" -c "import json; print(len(json.load(open('$QUESTIONS_JSON'))))" 2>/dev/null || echo "?")"
    echo "  [skip] $QUESTIONS_JSON exists ($n_json records)"
else
    echo "  [fetch] $OPENEQA_JSON_URL"
    if ! curl -fsSL "$OPENEQA_JSON_URL" -o "$QUESTIONS_JSON"; then
        echo "  [ERROR] failed to download open-eqa-v0.json" >&2
        echo "    URL: $OPENEQA_JSON_URL" >&2
        exit 1
    fi
    n_json="$("$PYTHON" -c "import json; print(len(json.load(open('$QUESTIONS_JSON'))))")"
    echo "  [ok]   downloaded ($n_json records)"
fi

if [ "$JSON_ONLY" -eq 1 ]; then
    echo ""
    echo "=== --json-only complete ==="
    echo "Reload the nodeset to pick up the JSON:"
    echo "  curl -X POST http://localhost:8000/api/components/reload"
    exit 0
fi

# ── Step 2: disk-space check ─────────────────────────────────────────────
echo ""
echo "=== Step 2: disk-space check ==="
avail_kb="$(df -kP "$DATA_DIR" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
fs="$(df -P "$DATA_DIR" | awk 'NR==2 {print $6}')"
if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
    echo "  [ERROR] Free space on $fs is ${avail_gb} GB; need at least ${MIN_FREE_GB} GB" >&2
    exit 1
fi
echo "  [ok]   free space on $fs: ${avail_gb} GB"

if [ "$SKIP_PROMPT" -eq 0 ]; then
    echo ""
    echo "About to download $HF_REPO (~87 GB) into $DOWNLOADS_DIR,"
    echo "then extract per-episode tars under $EPISODES_DIR (~87 GB)."
    if [ "$KEEP_TARS" -eq 0 ]; then
        echo "Tars under _downloads/ will be deleted after extraction."
    fi
    read -r -p "Proceed? [y/N] " yn
    case "$yn" in
        [Yy]|[Yy][Ee][Ss]) ;;
        *) echo "aborted." >&2; exit 1 ;;
    esac
fi

# ── Step 3: download tars ────────────────────────────────────────────────
echo ""
echo "=== Step 3: download $HF_REPO ==="

ALLOW_PATTERNS=""
case "$FILTER" in
    hm3d)    ALLOW_PATTERNS="hm3d-v0/*.tar" ;;
    scannet) ALLOW_PATTERNS="scannet-v0/*.tar" ;;
    all)     ALLOW_PATTERNS="hm3d-v0/*.tar,scannet-v0/*.tar" ;;
esac

"$PYTHON" - <<PYEOF
import os
from huggingface_hub import snapshot_download

allow_patterns = "$ALLOW_PATTERNS".split(",") if "$ALLOW_PATTERNS" else None
path = snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$DOWNLOADS_DIR",
    allow_patterns=allow_patterns,
    max_workers=8,
)
print(f"  [ok]   downloaded to {path}")
PYEOF

# ── Step 4: extract per-episode tars ─────────────────────────────────────
echo ""
echo "=== Step 4: extract per-episode tars ==="

extract_subset() {
    local subset="$1"
    local src_dir="$DOWNLOADS_DIR/$subset"
    local dst_dir="$EPISODES_DIR/$subset"
    [ -d "$src_dir" ] || { echo "  [skip] $subset: no tars downloaded"; return; }
    mkdir -p "$dst_dir"
    local tar_count
    tar_count="$(find "$src_dir" -maxdepth 1 -name '*.tar' | wc -l)"
    echo "  $subset: $tar_count tar files to process"
    local extracted=0 skipped=0
    for tar_path in "$src_dir"/*.tar; do
        [ -f "$tar_path" ] || continue
        local episode_id
        episode_id="$(basename "$tar_path" .tar)"
        local ep_dir="$dst_dir/$episode_id"
        if [ -d "$ep_dir" ] && find "$ep_dir" -maxdepth 1 -name '*-rgb.png' | head -n1 | grep -q .; then
            skipped=$((skipped+1))
            continue
        fi
        mkdir -p "$ep_dir"
        # Tars are packaged as `<episode_id>/<files>` — strip the leading
        # episode_id dir so files land directly in $ep_dir.
        if ! tar -xf "$tar_path" -C "$ep_dir" --strip-components=1 2>/dev/null; then
            tar -xf "$tar_path" -C "$ep_dir"
        fi
        extracted=$((extracted+1))
    done
    echo "  $subset: extracted=$extracted, skipped=$skipped"
}

if [ "$FILTER" = "hm3d" ] || [ "$FILTER" = "all" ]; then
    extract_subset "hm3d-v0"
fi
if [ "$FILTER" = "scannet" ] || [ "$FILTER" = "all" ]; then
    extract_subset "scannet-v0"
fi

# ── Step 5: cleanup tars ─────────────────────────────────────────────────
if [ "$KEEP_TARS" -eq 0 ]; then
    echo ""
    echo "=== Step 5: cleanup tars ==="
    rm -rf "$DOWNLOADS_DIR"
    echo "  [ok]   removed $DOWNLOADS_DIR (use --keep-tars to skip this)"
fi

# ── Summary ──────────────────────────────────────────────────────────────
n_json="$("$PYTHON" -c "import json; print(len(json.load(open('$QUESTIONS_JSON'))))" 2>/dev/null || echo "?")"
n_hm3d=0
n_scannet=0
[ -d "$EPISODES_DIR/hm3d-v0" ] && \
    n_hm3d="$(find "$EPISODES_DIR/hm3d-v0" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[ -d "$EPISODES_DIR/scannet-v0" ] && \
    n_scannet="$(find "$EPISODES_DIR/scannet-v0" -mindepth 1 -maxdepth 1 -type d | wc -l)"
n_hm3d_q="$("$PYTHON" -c "
import json
print(sum(1 for r in json.load(open('$QUESTIONS_JSON'))
         if r.get('episode_history','').startswith('hm3d-v0/')))" 2>/dev/null || echo "?")"
n_scannet_q="$("$PYTHON" -c "
import json
print(sum(1 for r in json.load(open('$QUESTIONS_JSON'))
         if r.get('episode_history','').startswith('scannet-v0/')))" 2>/dev/null || echo "?")"

echo ""
echo "=== Summary ==="
echo "  Questions in JSON:          $n_json"
echo "  HM3D episodes staged:       $n_hm3d  (1920×1080 PNG, variable frame count)"
echo "  ScanNet episodes staged:    $n_scannet  (1296×968 RGB + 640×480 depth + pose + intrinsics)"
echo "  HM3D-backed questions:      $n_hm3d_q"
echo "  ScanNet-backed questions:   $n_scannet_q"

echo ""
echo "=== Provenance ==="
echo "  - Mirror: $HF_REPO (HuggingFace, third-party packaging)."
echo "  - Original Meta Dropbox tarball is permanently expired."
echo "  - HM3D: paper-canonical 1920×1080 PNG, lossless."
echo "  - ScanNet: paper-canonical 1296×968 RGB + 640×480 depth + pose +"
echo "    color/depth intrinsics + extrinsics. All lossless."
echo "  - HM3D scenes are released under Matterport academic-use EULA."

echo ""
echo "=== Next steps ==="
echo "  1. (optional) Add the openeqa-judge LLM profile:"
echo "       cd agentcanvas && python -m app.llm.cli add openeqa-judge \\"
echo "           --provider <openai|anthropic|google> \\"
echo "           --model <model-id> --api-key \$YOUR_API_KEY"
echo "  2. Reload the nodeset:"
echo "       curl -X POST http://localhost:8000/api/components/reload"
echo "  3. Open one of:"
echo "       workspace/graphs/openeqa_em_blind_llm.json   (frames not consumed)"
echo "       workspace/graphs/openeqa_em_multiframe.json  (uses HM3D + ScanNet frames)"
echo ""
