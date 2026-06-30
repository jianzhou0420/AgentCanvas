#!/usr/bin/env bash
# =============================================================================
# fetch_dataset_libero.sh — LIBERO demo datasets (+ optional openpi norm_stats)
# =============================================================================
# Re-obtains the data that used to be symlinked into the private
# vlaworkspace tree, so the LIBERO / VLA stack is reproducible
# from public sources.
#
# Two independent sections (run all, or pass a selector — see Usage):
#
#   [1] LIBERO HDF5 demo datasets  ->  data/libero/datasets/
#       Official LIBERO benchmark demos (libero_object / _goal / _spatial /
#       _100), downloaded by LIBERO's own downloader from the UT-Austin box.com
#       mirror. NOTE: per data/libero/README.md these are TRAINING-TIME ONLY —
#       the runtime env_libero nodeset does NOT read them (init-states come from
#       the LIBERO pip package). Fetch only if you do data conversion / training.
#
#   [2] openpi pi0 LIBERO norm_stats  ->  data/vla_policy/norm_stats/
#       The OFFICIAL Physical-Intelligence pi0 normalization stats, bundled
#       inside the openpi release checkpoint at
#       gs://openpi-assets/checkpoints/pi0_libero . These back the (currently
#       vestigial) data/vla_policy/norm_stats/libero_pi0.json reference. The
#       runtime policies default to the VENDORED JianZhou0420 stats
#       (policy_vla/_assets/norm_stats/), so this section is OPTIONAL — only
#       needed if you want to run against the official pi0 baseline stats.
#
#   NOT fetched: the dp / smolvla "physical-intelligence" norm_stats variants.
#   openpi does not release dp / smolvla LIBERO checkpoints, and the provenance
#   of those two JSON on this machine could not be verified (they appear to be
#   locally computed). They are not auto-downloaded here — do not assume a
#   source. Recompute or copy them in manually if you need them.
#
# Usage:
#   bash scripts/data/fetch_dataset_libero.sh                 # all sections
#   bash scripts/data/fetch_dataset_libero.sh --datasets      # [1] only
#   bash scripts/data/fetch_dataset_libero.sh --norm-stats    # [2] only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASETS_DIR="$PROJECT_ROOT/data/libero/datasets"
NORM_STATS_DIR="$PROJECT_ROOT/data/vla_policy/norm_stats"

LIBERO_PYTHON="${LIBERO_PYTHON:-/home/$(whoami)/miniforge3/envs/ac-libero/bin/python}"
VLA_PYTHON="${VLA_POLICY_PYTHON:-/home/$(whoami)/miniforge3/envs/ac-vla-policy/bin/python}"
LIBERO_SRC="$PROJECT_ROOT/third_party/libero"

DO_DATASETS=true
DO_NORM_STATS=true
case "${1:-}" in
    --datasets)   DO_NORM_STATS=false ;;
    --norm-stats) DO_DATASETS=false ;;
    "" )          ;;
    * ) echo "[fetch-libero] unknown arg: $1 (use --datasets | --norm-stats)"; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# [1] LIBERO HDF5 demo datasets
# -----------------------------------------------------------------------------
if [ "$DO_DATASETS" = true ]; then
    echo "=== [1] LIBERO HDF5 demo datasets -> $DATASETS_DIR ==="
    if [ ! -x "$LIBERO_PYTHON" ]; then
        echo "[fetch-libero] ERROR: ac-libero python not found at $LIBERO_PYTHON"
        echo "                run scripts/install/install_ac_libero.sh first (or set LIBERO_PYTHON)."
        exit 1
    fi
    if [ ! -f "$LIBERO_SRC/benchmark_scripts/download_libero_datasets.py" ]; then
        echo "[fetch-libero] ERROR: $LIBERO_SRC not present — run install_ac_libero.sh first."
        exit 1
    fi
    mkdir -p "$DATASETS_DIR"
    # LIBERO's downloader uses libero.libero.utils.download_utils, which pulls
    # the official zips from https://utexas.box.com/... (libero_object / _goal /
    # _spatial / _100) and unpacks them into --download-dir.
    ( cd "$LIBERO_SRC/benchmark_scripts" && \
      "$LIBERO_PYTHON" download_libero_datasets.py --download-dir "$DATASETS_DIR" --datasets all )
    echo "[fetch-libero] datasets ready under $DATASETS_DIR"
fi

# -----------------------------------------------------------------------------
# [2] openpi pi0 LIBERO norm_stats (optional, official baseline)
# -----------------------------------------------------------------------------
if [ "$DO_NORM_STATS" = true ]; then
    echo ""
    echo "=== [2] openpi pi0 LIBERO norm_stats -> $NORM_STATS_DIR ==="
    if [ ! -x "$VLA_PYTHON" ]; then
        echo "[fetch-libero] WARN: ac-vla-policy python not found at $VLA_PYTHON — skipping [2]."
        echo "               run scripts/install/install_ac_vla_policy.sh first (or set VLA_POLICY_PYTHON)."
    else
        mkdir -p "$NORM_STATS_DIR"
        # The vendored openpi downloader (fsspec/gcsfs) pulls the release
        # checkpoint dir from gs://openpi-assets; norm_stats live in its assets.
        PYTHONPATH="$PROJECT_ROOT" "$VLA_PYTHON" - "$NORM_STATS_DIR" <<'PYEOF'
import sys, shutil, pathlib
dst = pathlib.Path(sys.argv[1])
from workspace.nodesets.policy.policy_vla.models.openpi.shared import download
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
# norm_stats.json is shipped under the checkpoint's assets/ tree.
hits = list(pathlib.Path(ckpt).rglob("norm_stats.json"))
if not hits:
    print(f"[fetch-libero] no norm_stats.json found under {ckpt} — inspect manually")
    sys.exit(0)
out = dst / "libero_pi0.json"
shutil.copyfile(hits[0], out)
print(f"[fetch-libero] copied {hits[0]} -> {out}")
PYEOF
    fi
fi

echo ""
echo "=== fetch_dataset_libero.sh done ==="
