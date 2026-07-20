#!/usr/bin/env bash
# Loop the VLNVerse scene downloader, sleeping a cooldown between bursts so
# HF's 5-minute resolver rate-limit window can slide. Each burst downloads
# until it hits a 429, then exits; the loop sleeps and starts a fresh process.
#
# Adapted from /home/xunyi/Desktop/Data/script/download_vlnverse_loop.sh —
# same exit-code contract, retargeted at scripts/data/fetch_scenes_vlnverse.py
# (which resolves the repo-root data/vlnverse/scene output dir itself, so no
# cd is needed here).
#
# Usage:
#   bash scripts/data/fetch_scenes_vlnverse_loop.sh                  # all missing scenes
#   bash scripts/data/fetch_scenes_vlnverse_loop.sh kujiale_0011     # specific scene(s)
#   COOLDOWN=600 bash scripts/data/fetch_scenes_vlnverse_loop.sh     # custom cooldown (seconds)
#   MAX_ATTEMPTS=20 bash scripts/data/fetch_scenes_vlnverse_loop.sh  # cap retries
#
# Exit codes:
#   0   — all scenes downloaded
#   42  — gave up after MAX_ATTEMPTS rate-limited bursts (unset = unlimited)
#   *   — python script exited with a non-rate-limit error; loop stopped
set -u

COOLDOWN=${COOLDOWN:-480}            # 8 min by default
MAX_ATTEMPTS=${MAX_ATTEMPTS:-0}      # 0 = unlimited
PY=${PY:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

attempt=1
while true; do
    echo
    echo "===== attempt $attempt @ $(date '+%F %T') ====="
    "$PY" "$SCRIPT_DIR/fetch_scenes_vlnverse.py" --exit-on-rate-limit "$@"
    code=$?

    case "$code" in
        0)
            echo "All scenes downloaded."
            exit 0
            ;;
        42)
            if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
                echo "Hit MAX_ATTEMPTS=$MAX_ATTEMPTS while rate-limited; giving up."
                exit 42
            fi
            echo "Rate-limited. Sleeping ${COOLDOWN}s before next attempt..."
            sleep "$COOLDOWN"
            ;;
        130)
            echo "Interrupted by Ctrl-C; exiting."
            exit 130
            ;;
        *)
            echo "Script exited with code $code (not a rate-limit). Stopping loop."
            exit "$code"
            ;;
    esac
    attempt=$((attempt + 1))
done
