#!/usr/bin/env bash
# Build the headless pySLAM image via ROOTLESS docker (no sudo).
# See the reference_rootless_docker memory for why these env vars are needed.
set -euo pipefail

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
export PATH="/usr/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-agentcanvas/pyslam:cpu}"

echo "Building ${TAG} (rootless docker)..."
exec docker build --progress=plain -t "${TAG}" -f "${HERE}/Dockerfile" "${HERE}"
