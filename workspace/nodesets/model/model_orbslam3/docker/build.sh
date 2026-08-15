#!/usr/bin/env bash
# Build agentcanvas/orbslam3 against the ROOTLESS docker daemon.
set -euo pipefail
cd "$(dirname "$0")"

uid=$(id -u)
export XDG_RUNTIME_DIR="/run/user/${uid}"
export DOCKER_HOST="unix:///run/user/${uid}/docker.sock"

# requirements-serve.txt is owned by the backend; copy fresh into context.
repo_root=$(git rev-parse --show-toplevel)
cp "${repo_root}/agentcanvas/backend/requirements-serve.txt" .

docker build -t agentcanvas/orbslam3:latest "$@" .
