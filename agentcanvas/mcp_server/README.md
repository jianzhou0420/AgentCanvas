# AgentCanvas Backend MCP Server

A Model Context Protocol server that exposes the AgentCanvas batch-eval surface
as typed tools for coding agents (Claude Code, Cursor, custom MCP harnesses).

Acts as a **process-pool manager**: each conversation gets its own dedicated
backend (1 of 5 slots), runs experiments in parallel without colliding, and
the backend self-terminates when the eval finishes. Historical results are
served by the MCP itself (file read), so backend death never loses data.

## Tools (6)

| Tool | Source | Description |
|---|---|---|
| `eval_start` | `POST /api/eval/v2/start` (own backend) | Start a batch eval run; returns `run_id` |
| `eval_status` | `GET /api/eval/v2/status` (own backend) | Poll progress of the active run |
| `eval_stop` | `POST /api/eval/v2/stop` (own backend) | Cancel the active run |
| `eval_export` | filesystem (`outputs/eval_runs/{run_id}/summary.json`) | Full results — works for any historical run, no backend needed |
| `eval_runs_list` | filesystem (`outputs/eval_runs/`) | List recent runs (newest first), no backend needed |
| `graph_list` | filesystem (`workspace/graphs/` + `workspace/architect/exp_profiles/`) | List graphs + advisory exp profiles |

## Pool semantics

The MCP holds a 5-slot pool on ports 8765–8769, mediated by `flock`
lockfiles in `~/.cache/agentcanvas-mcp/locks/port-{N}.lock`. Each
conversation:

1. Walks the ports, claims the first lockfile it can `flock(LOCK_EX|LOCK_NB)`.
2. Spawns a fresh backend on that port (`conda run -n agentcanvas uvicorn …`).
3. Runs experiments. Backend lives only as long as it's needed.
4. **Backend self-exits** in two cases:
   - **Eval finished** — after `save_run` writes `summary.json`, the
     executor's `finally` block hard-exits (`os._exit(0)`). Override
     with `AGENTCANVAS_BACKEND_EXIT_ON_EVAL_END=0` for legacy
     multi-eval workflows (e.g. `run_dev.sh`).
   - **Idle 60s** — backend's idle watchdog hard-exits if no non-`/health`
     request arrives for 60s AND `ExecutionGuard` reports no active mode.
     Slow-path safety net for "MCP died without firing PDEATHSIG".
5. On clean MCP exit (atexit / SIGTERM / SIGINT) the backend group is
   `killpg`'d and the lock fd closed (kernel auto-releases the flock).
6. On `kill -9 <mcp>`: PDEATHSIG fires the SIGTERM cascade; if that's
   somehow missed, the idle watchdog reaps within ~90s; the kernel
   releases the flock the moment the fd closes.

Pool full → MCP raises `pool full (5/5 slots taken)` at startup. Wait
for another conversation to finish or stop one explicitly.

Cold-start spawn is ~5-10s (conda + torch import). Lock claim is sub-ms.

## Registration with Claude Code

```bash
claude mcp add agentcanvas-backend \
  -- conda run -n agentcanvas --no-capture-output python -m mcp_server
```

The `--no-capture-output` flag is important — without it, stdout buffering
breaks the MCP JSON-RPC framing.

After registration, restart Claude Code. The 5 tools will appear in the
LLM's tool catalog. Try:

```
List the available agent graphs.
Run navgpt_mp3d for 3 episodes with 1 worker.
```

## Manual smoke test (no Claude Code)

```bash
conda run -n agentcanvas python -c "
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    p = StdioServerParameters(
        command='conda',
        args=['run', '-n', 'agentcanvas', '--no-capture-output', 'python', '-m', 'mcp_server'],
        cwd='/path/to/vlnworkspace/agentcanvas',
    )
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print([t.name for t in tools.tools])

asyncio.run(run())
"
```

Expected: `['eval_start', 'eval_status', 'eval_stop', 'eval_export', 'graph_list', 'eval_runs_list']`.

## Files

```
mcp_server/
  __init__.py
  __main__.py            entry: claims pool slot, spawns backend, runs FastMCP on stdio
  server.py              FastMCP wiring + 6 tool definitions
  backend_manager.py     flock-based pool spawn + PGID-targeted cleanup
  tools/
    __init__.py
    eval.py              eval_start/status/stop httpx + eval_export direct file read
    graph.py             graph_list (filesystem-only)
    runs.py              eval_runs_list (filesystem-only)
```

## Logs

Backend logs land at `~/.cache/agentcanvas-mcp/backend-{ISO}-port{N}.log`.
MCP server's own logs go to stderr (stdout is reserved for JSON-RPC).

## Troubleshooting

- **`pool full (5/5 slots taken)`**: 5 conversations are already holding
  pool slots. Inspect with `ls ~/.cache/agentcanvas-mcp/locks/` (each
  `.lock` file is held by some MCP); kill stale ones via `lslocks` →
  `kill <pid>` (or `kill -9` and let PDEATHSIG / the idle watchdog
  reap the backend within ~90s).
- **`/health` 90s timeout on cold start**: usually means a torch/CUDA import
  failure. Check the latest log under `~/.cache/agentcanvas-mcp/`.
- **Tool returns `{"error": "another run is already active"}`**: someone
  reused the same backend (rare in pool mode; only happens if you set
  `AGENTCANVAS_BACKEND_EXIT_ON_EVAL_END=0`). Call `eval_stop` first or
  wait for completion.
- **`eval_export` returns 404 immediately**: file isn't there yet.
  Either the run is still in flight (use `eval_status`), or the run_id
  is wrong (cross-check with `eval_runs_list`).
- **MCP server doesn't appear in Claude Code**: confirm registration with
  `claude mcp list`; common cause is conda env not on PATH (use the full
  `conda run` invocation above rather than bare `python`).

## Out of scope (deferred to v2)

- `logs_read`, `nodeset_reload`, `episode_get`, `introspect`
- WebSocket streaming (`/api/eval/v2/ws`) → `notifications/progress`
- HTTP/SSE transport (currently stdio only)
- Auto-generation via `fastmcp.from_fastapi` (curated tools chosen instead)
- Lease-based heartbeat protocol (current 60s idle watchdog is sufficient)
- Migrating `/architect:*` skills off the curl path onto MCP
