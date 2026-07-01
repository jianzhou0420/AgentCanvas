#!/usr/bin/env bash
# ==============================================================================
# AgentCanvas — Core Install & Launch
#
# Installs the agentcanvas conda env + frontend deps, then launches the canvas
# (backend :8000 + frontend :5173 in foreground).
#
# This is the CORE hub env ONLY — it does NOT install any server-mode env.
# Each simulator / model env has its own scripts/install/install_ac_*.sh.
#
# The doc-site (under docs/) is hand-authored HTML with no install step — run
# `bash docs/run_dev.sh` separately to serve it on :8092.
#
# Usage:  bash scripts/install/install_core.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install]${NC} $*"; }
fail() { echo -e "${RED}[install]${NC} $*"; exit 1; }

# --------------- install ------------------------------------------------------
log "Installing agentcanvas..."
bash "$SCRIPT_DIR/install_agentcanvas.sh"

# --------------- launch -------------------------------------------------------
log "Launching canvas..."
bash "$REPO_ROOT/agentcanvas/launch.sh"
