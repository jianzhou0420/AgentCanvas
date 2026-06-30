#!/usr/bin/env bash
# Episode-data setup for MP3D-based discrete VLN tasks and Habitat-CE
# (continuous-environment) benchmarks.
#
# Discrete (MatterSim) — lands under data/mp3d/tasks/ :
#   --r2r       Room-to-Room           (Anderson et al. CVPR 2018)    ~6 MB
#   --r4r       Room-for-Room          (Jain et al. ACL 2019)         ~55 MB (generated)
#   --rxr       Room-across-Room       (Ku et al. EMNLP 2020)         ~19 GB  (annot+poses)
#   --rxr-full  RxR + BERT features                                    +142 GB  (rarely needed)
#   --reverie   REVERIE                (Qi et al. CVPR 2020)          ~30 MB  (annot+BBox)
#   --reverie-features                                                  +2.9 GB  (ResNet-152)
#   --cvdn      CVDN + NDH             (Thomason et al. CoRL 2019)    ~20 MB
#
# Continuous (Habitat-Sim / VLN-CE) — lands under data/habitat/datasets/ :
#   --rxr-ce    RxR-VLN-CE v0          (Krantz et al. RxR-Habitat)    ~150 MB (episodes only)
#
# EQA (HM3D-based) — lands under data/hm3d/hmeqa/ :
#   --hmeqa     HM-EQA + Open_Sans font (Ren et al. 2024)             ~200 KB (CSVs)
#               Copies questions.csv + scene_init_poses.csv from the
#               vendored explore-eqa repo. HM3D scene meshes (~15 GB)
#               require a manual HuggingFace download — printed as an
#               instruction, not auto-fetched.
#
# All discrete targets land under data/mp3d/tasks/ — the asset-agnostic
# namespace established by ADR-030. Every discrete task reuses the
# Matterport3D scans at data/mp3d/v1/scans/ and the MatterSim
# connectivity graphs shipped with the third_party/Matterport3DSimulator
# submodule. Continuous targets land under data/habitat/datasets/ —
# the VLN-CE framework mirror (ADR-platform-005) whose child symlinks
# already resolve the scene-dataset path for the Habitat nodeset.
#
# Target layout:
#   data/mp3d/tasks/
#     R2R/     R2R_{train,val_seen,val_unseen,test}.json
#     R4R/     R4R_{train,val_seen,val_unseen}.json            (locally generated)
#     RxR/     rxr_{split}_{en,hi,te}_guide.jsonl.gz
#              rxr_{split}_{en,hi,te}_follower.jsonl.gz
#              pose_traces/{instruction_id}_{guide,follower}_pose_trace.npz
#              [text_features/ only with --rxr-full]
#     REVERIE/ REVERIE_{train,val_seen,val_unseen,test}.json
#              BBox/{scan}_{vp}.json
#     CVDN/    {train,val_seen,val_unseen}.json
#     NDH/     {train,val_seen,val_unseen,test}.json
#   data/habitat/datasets/
#     RxR_VLNCE_v0/{train,val_seen,val_unseen,test_challenge}/
#       {split}_guide.json.gz
#       {split}_guide_gt.json.gz
#       {split}_follower.json.gz      (optional)
#       {split}_follower_gt.json.gz   (optional)
#
# Usage:
#   bash scripts/data/fetch_episodes_vln.sh                  # all default datasets
#   bash scripts/data/fetch_episodes_vln.sh --r2r --reverie  # subset
#   bash scripts/data/fetch_episodes_vln.sh --rxr-full       # include 142 GB BERT features
#   bash scripts/data/fetch_episodes_vln.sh --dry-run        # print actions only
#   bash scripts/data/fetch_episodes_vln.sh --force          # re-download existing files
#
# Prereqs:
#   - wget          (R2R, REVERIE features, CVDN/NDH)
#   - git           (REVERIE sparse-checkout of annotations)
#   - gsutil        (RxR only — https://cloud.google.com/storage/docs/gsutil_install)
#   - python3 + networkx>=2.3 (R4R generator; any env with `pip install networkx` works)
#
# Idempotent: files already on disk are skipped unless --force.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && cd .. && pwd)"

TASKS_DIR="$REPO_ROOT/data/mp3d/tasks"
MP3D_SIM="$REPO_ROOT/third_party/Matterport3DSimulator"
CONNECTIVITY_DIR="$MP3D_SIM/connectivity"
VENDORED_DIR="$SCRIPT_DIR/../vendor"
R4R_SCRIPT="$VENDORED_DIR/r4r_generate_data.py"
R4R_GRAPH_UTILS="$VENDORED_DIR/graph_utils.py"
R4R_URL_BASE="https://raw.githubusercontent.com/google-research/google-research/master/r4r"

# Python interpreter for R4R generation. Override via DATA_SETUP_PY env var.
# Default tries a conda env that ships networkx (vlnce), then falls back to system python3.
if [ -n "${DATA_SETUP_PY:-}" ]; then
  PY3="$DATA_SETUP_PY"
elif [ -x "${HOME}/miniforge3/envs/ac-vlnce/bin/python" ]; then
  PY3="${HOME}/miniforge3/envs/ac-vlnce/bin/python"
else
  PY3="python3"
fi

# ── flag parsing ───────────────────────────────────────────────────────
WANT_R2R=0 WANT_R4R=0 WANT_RXR=0 WANT_RXR_FULL=0
WANT_REVERIE=0 WANT_REVERIE_FEATS=0 WANT_CVDN=0
WANT_RXR_CE=0
WANT_HMEQA=0
DRY_RUN=0 FORCE=0
ANY_FLAG=0

usage() { sed -n '2,50p' "$0"; exit "${1:-0}"; }

while (( $# )); do
  case "$1" in
    --r2r)                WANT_R2R=1; ANY_FLAG=1 ;;
    --r4r)                WANT_R4R=1; ANY_FLAG=1 ;;
    --rxr)                WANT_RXR=1; ANY_FLAG=1 ;;
    --rxr-full)           WANT_RXR=1; WANT_RXR_FULL=1; ANY_FLAG=1 ;;
    --rxr-ce)             WANT_RXR_CE=1; ANY_FLAG=1 ;;
    --reverie)            WANT_REVERIE=1; ANY_FLAG=1 ;;
    --reverie-features)   WANT_REVERIE=1; WANT_REVERIE_FEATS=1; ANY_FLAG=1 ;;
    --cvdn)               WANT_CVDN=1; ANY_FLAG=1 ;;
    --hmeqa)              WANT_HMEQA=1; ANY_FLAG=1 ;;
    --all)                ANY_FLAG=1
                          WANT_R2R=1; WANT_R4R=1; WANT_RXR=1
                          WANT_REVERIE=1; WANT_CVDN=1
                          WANT_RXR_CE=1; WANT_HMEQA=1 ;;
    --dry-run)            DRY_RUN=1 ;;
    --force)              FORCE=1 ;;
    -h|--help)            usage 0 ;;
    *) echo "Unknown flag: $1" >&2; usage 1 ;;
  esac
  shift
done

# Default (no dataset flags given) → discrete MP3D datasets only. The
# continuous-env RxR-CE (--rxr-ce) is opt-in because its only reliable
# source is a Google Drive zip that requires ``gdown`` or a manual
# download step.
if (( ANY_FLAG == 0 )); then
  WANT_R2R=1; WANT_R4R=1; WANT_RXR=1; WANT_REVERIE=1; WANT_CVDN=1
fi

log()  { printf '\033[1;34m[fetch-episodes]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fetch-episodes]\033[0m %s\n' "$*" >&2; }
run()  { if (( DRY_RUN )); then echo "DRY-RUN: $*"; else eval "$@"; fi; }

need_bin() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing prereq: $1 — install it and retry." >&2; exit 2; }
}

# ── R2R: wget 4 Dropbox URLs; also absorb any legacy submodule copy ────
do_r2r() {
  need_bin wget
  local dst="$TASKS_DIR/R2R"
  run "mkdir -p '$dst'"

  # One-time absorb: if R2R files still sit inside the submodule's tasks/R2R/data,
  # move them into the canonical location so subsequent runs skip them.
  local legacy="$MP3D_SIM/tasks/R2R/data"
  if [ -d "$legacy" ]; then
    for f in R2R_train.json R2R_val_seen.json R2R_val_unseen.json R2R_test.json; do
      if [ -f "$legacy/$f" ] && [ ! -f "$dst/$f" ]; then
        log "Relocating $f from submodule → data/mp3d/tasks/R2R/"
        run "mv '$legacy/$f' '$dst/$f'"
      fi
    done
  fi

  # Dropbox URLs mirror third_party/Matterport3DSimulator/tasks/R2R/data/download.sh.
  declare -A urls=(
    [R2R_train.json]="https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json"
    [R2R_val_seen.json]="https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json"
    [R2R_val_unseen.json]="https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json"
    [R2R_test.json]="https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json"
  )
  for name in "${!urls[@]}"; do
    local out="$dst/$name"
    if [ -f "$out" ] && (( FORCE == 0 )); then
      log "R2R: $name already present — skip"
    else
      log "R2R: fetching $name"
      run "wget -q --show-progress -O '$out.part' '${urls[$name]}?dl=1' && mv '$out.part' '$out'"
    fi
  done
}

# ── R4R: generate locally from R2R via Google Research's script ────────
do_r4r() {
  [ -d "$TASKS_DIR/R2R" ] || { warn "R4R needs R2R first — run with --r2r (or --all)"; return; }
  [ -d "$CONNECTIVITY_DIR" ] || { warn "R4R needs MatterSim connectivity at $CONNECTIVITY_DIR — init submodules"; return; }
  "$PY3" -c 'import sys' 2>/dev/null || { warn "R4R needs a python interpreter (DATA_SETUP_PY='$PY3' not runnable)"; return; }
  "$PY3" -c 'import networkx' 2>/dev/null || { warn "R4R needs networkx — pip install 'networkx>=2.3' into $PY3"; return; }
  need_bin wget

  local src="$TASKS_DIR/R2R"
  local dst="$TASKS_DIR/R4R"
  run "mkdir -p '$dst' '$VENDORED_DIR'"

  if [ ! -f "$R4R_SCRIPT" ] || [ ! -f "$R4R_GRAPH_UTILS" ] || (( FORCE )); then
    log "R4R: fetching generator + graph_utils from google-research/r4r"
    run "wget -q -O '$R4R_SCRIPT.part' '$R4R_URL_BASE/r4r_generate_data.py' && mv '$R4R_SCRIPT.part' '$R4R_SCRIPT'"
    run "wget -q -O '$R4R_GRAPH_UTILS.part' '$R4R_URL_BASE/graph_utils.py' && mv '$R4R_GRAPH_UTILS.part' '$R4R_GRAPH_UTILS'"
  fi

  # R4R has no test split — test path concatenation is ill-defined without GT.
  for split in train val_seen val_unseen; do
    local in="$src/R2R_$split.json"
    local out="$dst/R4R_$split.json"
    [ -f "$in" ] || { warn "R4R: $in missing — skipping $split"; continue; }
    if [ -f "$out" ] && (( FORCE == 0 )); then
      log "R4R: $split already present — skip"
      continue
    fi
    log "R4R: generating $split (this may take a minute)"
    run "'$PY3' '$R4R_SCRIPT' --input_file_path='$in' --output_file_path='$out' --connections_dir='$CONNECTIVITY_DIR' --distance_threshold=3.0"
  done
}

# ── RxR: gsutil from public GCS bucket ─────────────────────────────────
do_rxr() {
  need_bin gsutil
  local dst="$TASKS_DIR/RxR"
  run "mkdir -p '$dst'"

  # Annotations (~100 MB) — always pulled.
  if [ -z "$(ls -A "$dst"/rxr_*.jsonl.gz 2>/dev/null)" ] || (( FORCE )); then
    log "RxR: fetching annotation JSONL.gz files"
    run "gsutil -m cp 'gs://rxr-data/rxr_*.jsonl.gz' '$dst/'"
  else
    log "RxR: annotations already present — skip"
  fi

  # Pose traces (~18 GB)
  if [ ! -d "$dst/pose_traces" ] || [ -z "$(ls -A "$dst/pose_traces" 2>/dev/null)" ] || (( FORCE )); then
    log "RxR: fetching pose_traces/ (~18 GB, long)"
    run "gsutil -m cp -r 'gs://rxr-data/pose_traces' '$dst/'"
  else
    log "RxR: pose_traces/ already present — skip"
  fi

  if (( WANT_RXR_FULL )); then
    if [ ! -d "$dst/text_features" ] || [ -z "$(ls -A "$dst/text_features" 2>/dev/null)" ] || (( FORCE )); then
      log "RxR: fetching text_features/ (~142 GB, very long)"
      run "gsutil -m cp -r 'gs://rxr-data/text_features' '$dst/'"
    else
      log "RxR: text_features/ already present — skip"
    fi
  fi
}

# ── RxR-CE: continuous-env episodes for the Habitat nodeset (ADR E2) ──
#
# Target: data/habitat/datasets/RxR_VLNCE_v0/{split}/{split}_guide.json.gz
# Source: RxR_VLNCE_v0.zip on Google Drive (ID 145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t).
# Google Drive does not expose a stable wget-friendly URL for files of
# this size, so we rely on ``gdown`` when available and fall back to a
# clear manual instruction otherwise. Guide and guide_gt files alone
# (~100 MB) are sufficient for eval; follower files are optional.
do_rxr_ce() {
  local dst="$REPO_ROOT/data/habitat/datasets/RxR_VLNCE_v0"
  local gdrive_id="145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t"
  run "mkdir -p '$dst'"

  # Idempotency gate — the dataset is already present in most dev
  # checkouts (placed manually before this flag existed). If every
  # expected guide file is on disk, no-op. ``test_challenge`` has no
  # public ground truth, so ``*_guide_gt.json.gz`` is train/val only.
  local have_all=1
  for split in train val_seen val_unseen test_challenge; do
    if [ ! -f "$dst/$split/${split}_guide.json.gz" ]; then
      have_all=0
      break
    fi
    if [ "$split" != "test_challenge" ] && \
       [ ! -f "$dst/$split/${split}_guide_gt.json.gz" ]; then
      have_all=0
      break
    fi
  done
  if (( have_all == 1 )) && (( FORCE == 0 )); then
    log "RxR-CE: all guide episodes already present under $dst — skip"
    return
  fi

  if ! command -v gdown >/dev/null 2>&1; then
    warn "RxR-CE: 'gdown' not installed — cannot auto-download RxR_VLNCE_v0.zip."
    warn "  (a) pip install gdown  then re-run with --rxr-ce, or"
    warn "  (b) download manually from"
    warn "      https://drive.google.com/file/d/$gdrive_id/view"
    warn "      and unzip so that files land at"
    warn "      $dst/{train,val_seen,val_unseen,test_challenge}/*.json.gz"
    return
  fi

  local zip_path="$dst/RxR_VLNCE_v0.zip"
  log "RxR-CE: fetching RxR_VLNCE_v0.zip via gdown (~150 MB)"
  run "gdown --id '$gdrive_id' -O '$zip_path'"
  log "RxR-CE: extracting into $dst"
  # The zip expands to a top-level RxR_VLNCE_v0/ directory; strip it so
  # the files land directly under $dst/{split}/.
  run "unzip -qo '$zip_path' -d '$dst/.unpack'"
  if [ -d "$dst/.unpack/RxR_VLNCE_v0" ]; then
    run "cp -rn '$dst/.unpack/RxR_VLNCE_v0/.' '$dst/'"
  else
    run "cp -rn '$dst/.unpack/.' '$dst/'"
  fi
  run "rm -rf '$dst/.unpack' '$zip_path'"
  log "RxR-CE: done. Expected files under $dst/{split}/"
}

# ── REVERIE: sparse-checkout annotations + BBox; optional features ─────
do_reverie() {
  need_bin git
  local dst="$TASKS_DIR/REVERIE"
  run "mkdir -p '$dst'"

  # Annotations + BBox via sparse checkout. The REVERIE repo stores these
  # at tasks/REVERIE/data/ and tasks/REVERIE/data/BBox/; we flatten both
  # one level up inside data/mp3d/tasks/REVERIE/.
  local have_annot=1
  for f in REVERIE_train.json REVERIE_val_seen.json REVERIE_val_unseen.json REVERIE_test.json; do
    [ -f "$dst/$f" ] || have_annot=0
  done
  if (( have_annot == 0 )) || (( FORCE )); then
    log "REVERIE: shallow-clone annotations + BBox from YuankaiQi/REVERIE"
    local tmp
    tmp="$(mktemp -d)"
    run "rmdir '$tmp'"   # git clone wants non-existent target
    run "git clone --depth 1 https://github.com/YuankaiQi/REVERIE '$tmp'"
    if [ -d "$tmp/tasks/REVERIE/data" ]; then
      run "cp -rn '$tmp/tasks/REVERIE/data/.' '$dst/'"
    else
      warn "REVERIE: expected tasks/REVERIE/data in clone, not found — repo layout may have changed"
    fi
    run "rm -rf '$tmp'"
  else
    log "REVERIE: annotations already present — skip"
  fi

  if (( WANT_REVERIE_FEATS )); then
    # The ResNet-152 image-feature .tsv ships on Dropbox per the REVERIE README.
    # Kept optional because almost no 2026-era method consumes these directly.
    local feats="$dst/img_features"
    run "mkdir -p '$feats'"
    if [ -z "$(ls -A "$feats" 2>/dev/null)" ] || (( FORCE )); then
      warn "REVERIE image features download requires the live Dropbox link from"
      warn "  https://github.com/YuankaiQi/REVERIE#data-preparation"
      warn "Dropbox URLs rotate — copy the current one by hand into $feats/"
    else
      log "REVERIE: image features already present — skip"
    fi
  fi
}

# ── HM-EQA: copy CSVs + Open_Sans font from vendored explore-eqa repo ──
#
# Target: data/hm3d/hmeqa/
#   questions.csv          — 500 multiple-choice EQA questions
#   scene_init_poses.csv   — initial agent pose per (scene, floor)
#   Open_Sans/             — TrueType font used for frontier annotation
#
# HM3D scene meshes themselves (~15 GB) are gated behind a Matterport
# HuggingFace access request — we print the instruction and do not
# attempt to fetch. Consumer (env_hmeqa nodeset) expects meshes under
# data/hm3d/hm3dsem/{scene}/{scene[6:]}.basis.glb.
do_hmeqa() {
  local upstream="$REPO_ROOT/workspace/nodesets/_upstream/explore-eqa/upstream"
  local src="$upstream/data"
  local dst="$REPO_ROOT/data/hm3d/hmeqa"
  if [ ! -d "$src" ]; then
    warn "HM-EQA: source missing at $src"
    warn "  Run: bash workspace/nodesets/_upstream/explore-eqa/fetch_upstream.sh"
    warn "  Then: bash workspace/nodesets/_upstream/explore-eqa/fetch_data.sh"
    return
  fi
  run "mkdir -p '$dst'"

  for f in questions.csv scene_init_poses.csv; do
    local out="$dst/$f"
    if [ -f "$out" ] && (( FORCE == 0 )); then
      log "HM-EQA: $f already present — skip"
    elif [ -f "$src/$f" ]; then
      log "HM-EQA: copying $f"
      run "cp '$src/$f' '$out'"
    else
      warn "HM-EQA: $src/$f not found — skipping"
    fi
  done

  # Open_Sans font directory — used by the frontier-scoring node to
  # render letter labels on annotated frontier images.
  if [ -d "$src/Open_Sans" ]; then
    if [ -d "$dst/Open_Sans" ] && (( FORCE == 0 )); then
      log "HM-EQA: Open_Sans/ already present — skip"
    else
      log "HM-EQA: copying Open_Sans/ font directory"
      run "cp -rn '$src/Open_Sans' '$dst/'"
    fi
  fi

  # HM3D scene meshes — require HuggingFace access token + manual request.
  local scene_root="$REPO_ROOT/data/hm3d/hm3dsem"
  if [ ! -d "$scene_root" ] || [ -z "$(ls -A "$scene_root" 2>/dev/null)" ]; then
    warn ""
    warn "HM-EQA: HM3D scene meshes not found at $scene_root."
    warn "  Scene meshes (~15 GB) must be downloaded manually:"
    warn "    1. Request access at https://aihabitat.org/datasets/hm3d-semantics/"
    warn "    2. Generate a HuggingFace access token (https://huggingface.co/settings/tokens)"
    warn "    3. Follow instructions at https://github.com/matterport/habitat-matterport-3dresearch"
    warn "       to download hm3d-train-habitat-v0.2.tar (semantic train split)"
    warn "    4. Extract so that meshes land at:"
    warn "         $scene_root/{scene}/{scene[6:]}.basis.glb"
    warn "         $scene_root/{scene}/{scene[6:]}.basis.navmesh"
    warn ""
  else
    log "HM-EQA: HM3D scene root exists at $scene_root"
  fi
}

# ── CVDN + NDH: wget from cvdn.dev ─────────────────────────────────────
do_cvdn() {
  need_bin wget
  local cvdn_dst="$TASKS_DIR/CVDN"
  local ndh_dst="$TASKS_DIR/NDH"
  run "mkdir -p '$cvdn_dst' '$ndh_dst'"

  local -A cvdn_urls=(
    [train.json]="https://cvdn.dev/dataset/CVDN/train_val/train.json"
    [val_seen.json]="https://cvdn.dev/dataset/CVDN/train_val/val_seen.json"
    [val_unseen.json]="https://cvdn.dev/dataset/CVDN/train_val/val_unseen.json"
  )
  for name in "${!cvdn_urls[@]}"; do
    local out="$cvdn_dst/$name"
    if [ -f "$out" ] && (( FORCE == 0 )); then
      log "CVDN: $name already present — skip"
    else
      log "CVDN: fetching $name"
      run "wget -q --show-progress -O '$out.part' '${cvdn_urls[$name]}' && mv '$out.part' '$out'"
    fi
  done

  local -A ndh_urls=(
    [train.json]="https://cvdn.dev/dataset/NDH/train_val/train.json"
    [val_seen.json]="https://cvdn.dev/dataset/NDH/train_val/val_seen.json"
    [val_unseen.json]="https://cvdn.dev/dataset/NDH/train_val/val_unseen.json"
    [test.json]="https://cvdn.dev/dataset/NDH/test_cleaned/test_cleaned.json"
  )
  for name in "${!ndh_urls[@]}"; do
    local out="$ndh_dst/$name"
    if [ -f "$out" ] && (( FORCE == 0 )); then
      log "NDH: $name already present — skip"
    else
      log "NDH: fetching $name"
      run "wget -q --show-progress -O '$out.part' '${ndh_urls[$name]}' && mv '$out.part' '$out'"
    fi
  done
}

# ── main ───────────────────────────────────────────────────────────────
log "Target root: $TASKS_DIR"
run "mkdir -p '$TASKS_DIR'"

(( WANT_R2R ))     && { log "=== R2R ===";     do_r2r; }
(( WANT_R4R ))     && { log "=== R4R ===";     do_r4r; }
(( WANT_RXR ))     && { log "=== RxR ===";     do_rxr; }
(( WANT_REVERIE )) && { log "=== REVERIE ==="; do_reverie; }
(( WANT_CVDN ))    && { log "=== CVDN/NDH ==="; do_cvdn; }
(( WANT_RXR_CE ))  && { log "=== RxR-CE ===";  do_rxr_ce; }
(( WANT_HMEQA ))   && { log "=== HM-EQA ===";  do_hmeqa; }

log "Done. Discrete under $TASKS_DIR; CE under $REPO_ROOT/data/habitat/datasets; EQA under $REPO_ROOT/data/hm3d/hmeqa"
