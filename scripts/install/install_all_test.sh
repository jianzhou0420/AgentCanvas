#!/usr/bin/env bash
# ==============================================================================
# AgentCanvas — Test Install & Launch
#
# Same as install_all.sh but uses an isolated test env: agentcanvas-test.
# The doc-site is hand-authored HTML with no install step.
#
# Usage:  bash scripts/install/install_all_test.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install-test]${NC} $*"; }
fail() { echo -e "${RED}[install-test]${NC} $*"; exit 1; }

# --------------- install with test env name -----------------------------------
log "Installing agentcanvas (env: agentcanvas-test)..."
bash "$SCRIPT_DIR/install_agentcanvas.sh" -n agentcanvas-test

# --------------- launch -------------------------------------------------------
log "Launching canvas..."
bash "$REPO_ROOT/agentcanvas/launch.sh"
