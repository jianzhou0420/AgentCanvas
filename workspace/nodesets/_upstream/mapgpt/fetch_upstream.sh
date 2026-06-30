#!/usr/bin/env bash
# Fetch the full upstream MapGPT repo for reference / comparison.
#
# Vendored content in workspace/nodesets/mapgpt.py: verbatim system prompt
# + planner constants from GPT/one_stage_prompt_manager.py (make_r2r_json_prompts,
# lines ~218-236) and rollout constants. This script clones the pinned commit
# so you can diff or pull additional code as needed.
#
# Upstream: https://github.com/chen-judge/MapGPT
# Pinned commit: 7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322
# License: UNSPECIFIED — upstream repo has no LICENSE file as of this commit.
#          Vendored content is treated as fair-use attribution; if you plan
#          to redistribute, contact the authors (Jiaqi Chen et al., ACL 2024).
#
# Default destination: ./upstream/ (sibling of this script, gitignored via
# workspace/nodesets/_upstream/*/upstream/ rule). Override with $1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322"
URL="https://github.com/chen-judge/MapGPT"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched MapGPT @ $COMMIT to $DEST"
echo "Diff our vendored constants against upstream:"
echo "  diff <(sed -n '/Verbatim/,/^def \\|^class /p' workspace/nodesets/mapgpt.py) $DEST/GPT/one_stage_prompt_manager.py"
