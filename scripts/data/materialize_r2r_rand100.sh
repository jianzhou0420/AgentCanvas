#!/bin/bash
# =============================================================================
# Materialize the r2r_rand100 evaluation split into the data tree.
# =============================================================================
# The paper's R2R-CE split (SmartWay/OpenNav protocol) is versioned whole under
# coding-agent/bridges/splits/r2r_rand100/ (data/ is gitignored, so the repo
# copy is the one tracked record — see the README there for provenance).
# This script copies it to where the env_habitat nodeset expects splits:
#   data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/rand100/
#
# Usage:
#   bash scripts/data/materialize_r2r_rand100.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$PROJECT_ROOT/coding-agent/bridges/splits/r2r_rand100"
DST="$PROJECT_ROOT/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/rand100"

for f in rand100.json.gz rand100_gt.json.gz; do
    [ -f "$SRC/$f" ] || { echo "[ERROR] missing $SRC/$f — repo checkout incomplete?"; exit 1; }
done

mkdir -p "$DST"
cp "$SRC/rand100.json.gz" "$SRC/rand100_gt.json.gz" "$DST/"
echo "rand100 split materialized -> $DST"
ls -la "$DST"
