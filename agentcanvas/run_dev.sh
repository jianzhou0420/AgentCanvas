#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"

# Ports are overridable for hosts where the defaults are taken
# (e.g. BACKEND_PORT=8080 bash run_dev.sh). BACKEND_URL is what the
# vite proxy reads (vite.config.ts), so the frontend follows automatically.
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:$BACKEND_PORT}"

# Activate agentcanvas conda env for the backend (ADR-020)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate agentcanvas 2>/dev/null || true

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== AgentCanvas Dev Server ===${NC}"

# Frontend: npm install if needed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo -e "${BLUE}Installing frontend dependencies...${NC}"
  (cd "$FRONTEND_DIR" && npm install)
fi

# Trap Ctrl+C to kill both processes
cleanup() {
  echo -e "\n${GREEN}Shutting down...${NC}"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend (uses active conda env)
echo -e "${BLUE}Starting backend on :$BACKEND_PORT...${NC}"
(cd "$BACKEND_DIR" && uvicorn app.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT") &
BACKEND_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend on :$FRONTEND_PORT...${NC}"
(cd "$FRONTEND_DIR" && npx vite --host --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

echo -e "${GREEN}Backend: http://localhost:$BACKEND_PORT${NC}"
echo -e "${GREEN}Frontend: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${GREEN}Swagger:  http://localhost:$BACKEND_PORT/docs${NC}"
echo ""

wait $BACKEND_PID $FRONTEND_PID
