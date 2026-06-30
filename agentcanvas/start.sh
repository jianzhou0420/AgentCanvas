#!/usr/bin/env bash
# Launch both backend and frontend dev servers.
# Usage: ./agentcanvas/start.sh
#   --backend-only   Start only the backend
#   --frontend-only  Start only the frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_PORT=8000
FRONTEND_PORT=5173

# Activate agentcanvas conda env for the backend (ADR-020)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate agentcanvas 2>/dev/null || true

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null
  wait 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

start_backend() {
  echo "Starting backend on http://localhost:$BACKEND_PORT ..."
  cd "$BACKEND_DIR"
  python -m uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
  BACKEND_PID=$!
  echo "  Backend PID: $BACKEND_PID"
}

start_frontend() {
  echo "Starting frontend on http://localhost:$FRONTEND_PORT ..."
  cd "$FRONTEND_DIR"
  npx vite --port "$FRONTEND_PORT" &
  FRONTEND_PID=$!
  echo "  Frontend PID: $FRONTEND_PID"
}

case "${1:-}" in
  --backend-only)
    start_backend
    ;;
  --frontend-only)
    start_frontend
    ;;
  *)
    start_backend
    sleep 1
    start_frontend
    echo ""
    echo "========================================="
    echo "  Frontend: http://localhost:$FRONTEND_PORT"
    echo "  Backend:  http://localhost:$BACKEND_PORT"
    echo "  Press Ctrl+C to stop both"
    echo "========================================="
    ;;
esac

wait
