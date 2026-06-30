#!/usr/bin/env bash
# Fetch the upstream ToolEQA paper-release repo for reference.
#
# Vendored content: workspace/nodesets/tooleqa/prompts/system_prompt.txt
# is a verbatim copy of the upstream prompt/system_prompt.txt.
# Action constants in workspace/nodesets/tooleqa/_actions.py mirror
# src/tools/go_next_point.py (move_forward, turn_left/right/around).
#
# Paper:    ToolEQA (Zhai et al., ICLR 2026 in-review), arXiv:2510.20310
# Upstream: TODO — confirm upstream repo URL. The local clone at
#           third_party/tooleqa/ (since removed) carried an origin of
#           github.com/jianzhou0420/AgentCanvas.git, which is the user's
#           own working fork, not the canonical paper release.
#           Likely candidates: the paper's GitHub link (check arxiv.org/abs/2510.20310)
#           or the authors' personal pages.
# License: UNKNOWN — no LICENSE file in the working copy. Treat vendored
#          content as fair-use attribution until upstream is identified.
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"

# TODO: replace with the canonical ToolEQA upstream URL once confirmed.
URL="${TOOLEQA_UPSTREAM_URL:-}"

if [ -z "$URL" ]; then
    echo "[error] upstream URL not yet configured." >&2
    echo "        Set TOOLEQA_UPSTREAM_URL=<url> and re-run, e.g.:" >&2
    echo "        TOOLEQA_UPSTREAM_URL=https://github.com/<user>/<repo>.git $0" >&2
    exit 1
fi

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
echo
echo "Fetched ToolEQA from $URL to $DEST"
