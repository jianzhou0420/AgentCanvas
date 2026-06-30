#!/usr/bin/env bash
# Fetch upstream AO-Planner repo for reference (fidelity audit + faithful re-port).
#
# Used to read the released code's controller + prompt manager when bringing the
# port closer to upstream (Path execution, ghost-map merge, image history):
#   environments_llm.py        multi-point pixel-Path execution + try-out sliding
#   llm/prompting/prompt_manager.py   make_graph_* prompts + interleaved image history
#   zero_shot_agent.py         rollout loop, ghost-graph, parse_num/parse_results
#
# Upstream: https://github.com/chen-judge/AO-Planner
# Pinned commit: 719f42a1c9bbb3ebee4815287713ec6c73c0e468  (== main HEAD, AAAI 2025)
# License: see ./LICENSE in the cloned tree.
#
# Default destination: ./upstream/ (sibling of this script, gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/upstream}"
COMMIT="719f42a1c9bbb3ebee4815287713ec6c73c0e468"
URL="https://github.com/chen-judge/AO-Planner.git"

if [ -d "$DEST" ]; then
    echo "[skip] $DEST already exists. Remove it first or pass a different path." >&2
    exit 1
fi

git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo
echo "Fetched AO-Planner @ $COMMIT to $DEST"
