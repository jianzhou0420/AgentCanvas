#!/usr/bin/env bash
# Launch the docs/ live-reload dev server (pure-stdlib HTTP + SSE).
#
# Usage:
#   ./docs/run_dev.sh                 # port 8092
#   ./docs/run_dev.sh 9000            # custom port
#   INTERVAL=1.0 ./docs/run_dev.sh    # slower poll (default 0.5s)
#   HOST=127.0.0.1 ./docs/run_dev.sh  # bind localhost only (default 0.0.0.0)

set -e

cd "$(dirname "$0")"

PORT=${1:-8092}
INTERVAL=${INTERVAL:-0.5}
HOST=${HOST:-0.0.0.0}

GREEN='\033[0;32m'
BLUE='\033[0;34m'
DIM='\033[2m'
NC='\033[0m'

echo -e "${GREEN}=== AgentCanvas Docs — Live Reload ===${NC}"
echo -e "${BLUE}URL     :${NC} http://${HOST}:${PORT}/"
echo -e "${BLUE}Root    :${NC} $(pwd)"
echo -e "${BLUE}Python  :${NC} $(command -v python3) ($(python3 --version 2>&1))"
echo -e "${DIM}Ctrl+C to stop.${NC}"
echo ""

# Forward SIGINT/SIGTERM into the python child so Ctrl+C is responsive.
cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup SIGINT SIGTERM

python3 _lib/_serve.py --host "$HOST" --port "$PORT" --interval "$INTERVAL" &
SERVER_PID=$!
wait "$SERVER_PID"
