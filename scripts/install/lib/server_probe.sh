#!/bin/bash
# =============================================================================
# server_probe.sh — install-time probes for server-mode conda envs
# =============================================================================
# Motivation (2026-07-31 deployment campaign): install verifies used to test
# only "activated-context top-level import", while production spawns workers
# with the BARE env python (no conda hooks) and talks msgpack over /call.
# Several envs shipped without the server wire stack and only failed at the
# first real run. These probes close that gap at install time.
#
#   probe_server_stack <env_python>
#       Bare-binary import of the AgentCanvas server wire stack. Every
#       auto_host worker needs these at boot / first proxied call — a miss
#       means every graph call 500s. FAILS (returns 1) when incomplete.
#
#   probe_auto_host <env_python> <nodeset_file> <class_name> [timeout_sec=45]
#       Boots app.server.auto_host with the bare env python — the exact
#       production loading context (PYTHONPATH injected the same way
#       registry.py does at spawn time) — and polls /health. WARN-ONLY:
#       env-type nodesets may legitimately need data or a GPU at
#       initialize() that an install host does not have yet.
# =============================================================================

_PROBE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

probe_server_stack() {
    local py="$1"
    echo ""
    echo "=== Probe: server wire stack (bare env python) ==="
    if "$py" -c "import fastapi, msgpack, httpx, pydantic, websockets" </dev/null 2>/dev/null; then
        echo "  [ok] fastapi + msgpack + httpx + pydantic + websockets import cleanly"
        return 0
    fi
    echo "  [ERROR] server wire stack incomplete in: $py"
    "$py" -c "import fastapi, msgpack, httpx, pydantic, websockets" </dev/null 2>&1 | tail -1 | sed 's/^/    | /'
    echo "          auto_host workers need these at boot; a run would 500 on the"
    echo "          first proxied call. Add the missing package to this env's install."
    return 1
}

probe_auto_host() {
    local py="$1" file="$2" cls="$3" timeout="${4:-45}"
    local port=$(( (RANDOM % 2000) + 47000 ))
    local log
    log=$(mktemp)
    echo ""
    echo "=== Probe: auto_host boot — $cls (bare env python, ${timeout}s budget) ==="
    (
        cd "$_PROBE_REPO_ROOT" &&
        exec env PYTHONPATH="$_PROBE_REPO_ROOT/agentcanvas/backend:$_PROBE_REPO_ROOT" \
            "$py" -m app.server.auto_host --file "$file" --class "$cls" --port "$port"
    ) </dev/null >"$log" 2>&1 &
    local pid=$!
    local ok=false
    local i
    for i in $(seq 1 "$timeout"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        if curl -s --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"status"'; then
            ok=true
            break
        fi
        sleep 1
    done
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true  # killed -> 143; must not trip the caller's set -e
    if [ "$ok" = true ]; then
        echo "  [ok] $cls booted and answered /health in the bare server context"
    else
        echo "  [WARN] $cls did not become healthy within ${timeout}s — worker tail:"
        tail -5 "$log" | sed 's/^/    | /'
        echo "  (warn-only: env-type nodesets may need data/GPU not present at install time)"
    fi
    rm -f "$log"
    return 0
}
