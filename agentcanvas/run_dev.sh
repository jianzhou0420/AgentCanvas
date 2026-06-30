#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"

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
echo -e "${BLUE}Starting backend on :8000...${NC}"
(cd "$BACKEND_DIR" && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend on :5173...${NC}"
(cd "$FRONTEND_DIR" && npx vite --host) &
FRONTEND_PID=$!

echo -e "${GREEN}Backend: http://localhost:8000${NC}"
echo -e "${GREEN}Frontend: http://localhost:5173${NC}"
echo -e "${GREEN}Swagger:  http://localhost:8000/docs${NC}"
echo ""

wait $BACKEND_PID $FRONTEND_PID
