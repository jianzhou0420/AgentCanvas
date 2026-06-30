# /experiment:run — submit a graph eval to the backend's JobScheduler

Submit a batch eval to the **already-running** agentcanvas backend's
JobScheduler (M1 design, 2026-05-07). The eval runs as a subprocess of
the backend, queueable + admission-controlled by `marginal_vram_mb` +
`exclusive_gpu` from the profile. Multiple jobs run concurrently when
their VRAM declarations fit. Ctrl-C posts a cancel to the scheduler.

## Usage

```
/experiment:run <profile> <graph_name> [--workspace=<path>] [--eval-overrides=<file>] [key=value ...]
```

- `profile` — name in `.claude/commands/experiment/profiles.yaml` (`defaults` block
  used as fallback for unknown names; warns on stderr)
- `graph_name` — name in `workspace/graphs/` (no `.json` extension) —
  resolved against `--workspace` overlay first, falls through to frozen
- `--workspace=<absolute-path>` — optional active-workspace overlay
  (e.g. an architect iter's `active_workspace/`). Backend loads this
  dir on top of frozen workspace; same-named nodesets/graphs/policies
  override frozen. Default = no overlay (run against frozen only)
- `key=value` — eval block overrides; common keys:
  `episode_count`, `worker_count`, `step_budget`, `per_step_budget_sec`,
  `dataset`, `split`, `start_episode_index`. `key=value` parsing handles
  **scalars only**.
- `--eval-overrides=<file>` — JSON object merged into the eval block
  before the `key=value` pairs. The only way to pass **list/dict** eval
  params (`episode_indices`, `episode_selectors`); an explicit `key=value`
  still overrides a key also present in the file

Example (baseline against frozen workspace):
```
/experiment:run mapgpt-mp3d mapgpt_mp3d episode_count=10 worker_count=2
```

Example (replay an architect iter's modified state without dirtying frozen):
```
/experiment:run mapgpt-mp3d mapgpt_mp3d \
    --workspace=/path/to/vlnworkspace/outputs/design_runs/mapgpt_mp3d/v0/iter_3/active_workspace \
    episode_count=10
```

The backend URL defaults to `http://127.0.0.1:5173` (Vite frontend proxy
that forwards `/api` + `/ws` to the actual backend at `:8000`) and can
be overridden via `AGENTCANVAS_BACKEND_URL`.

## Lifecycle

```
read profiles.yaml ──► build payload (eval block + scheduling block)
                  ──► POST /api/eval/v2/start (via_subprocess=true)
                  ──► poll /api/eval/v2/runs/{run_id} until _DONE
                  ──► tail stderr.log error markers if non-success
```

## Steps

```bash
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
exec python \
  "$ROOT/.claude/commands/experiment/bin/submit.py" "$@"
```

That single line is the entire harness now — `submit.py` does everything:
profile lookup, payload build, submit, poll, cancel-on-interrupt,
error-tail-on-failure. Exit codes:

| code | meaning |
|---|---|
| 0   | run completed |
| 1   | run terminated with status != completed (error / aborted / cancelled) |
| 2   | submit-time failure (backend unreachable, malformed spec, /start 4xx) |
| 130 | Ctrl-C / SIGTERM (cancel POST sent best-effort first) |

## Differences from the legacy version (pre-2026-05-07)

| Old | New |
|---|---|
| Spawned its own uvicorn on `:8765-8769` | Talks to existing backend (default via Vite proxy `:5173` → backend `:8000`) |
| Owned the backend's PGID + cleanup trap | Backend is long-lived, owns its own subprocesses |
| `admit.py` / `release.py` | JobScheduler.admit (server-side) |
| `alloc_port.py` / `register_backend.py` | Not needed |
| `<profile> -- <command...>` (arbitrary command) | `<profile> <graph_name> [key=value ...]` (graph-only) |
| `BACKEND_URL` exported into a sub-shell | Direct HTTP from `submit.py` |
| Backend log tail on non-zero exit | Subprocess stderr tail on non-success |

The legacy `<profile> -- <command>` form is **gone** — arbitrary commands
no longer go through admission control because admission is now keyed to
graph-eval submissions, not scripts. If you need to run an arbitrary
command, run it directly in your shell.

## What survives teardown

Per-run artefacts at `outputs/eval_runs/{run_id}/` — `run_id` is a
second-precision timestamp (e.g. `20260515_143052`):

```
spec.json           full submitted spec
shared_urls.json    backend's auto_host URL table at submit time
summary.json        EvalRun snapshot (live during run, terminal at exit)
stdout.log          subprocess stdout
stderr.log          subprocess stderr (logger output + tracebacks)
graph.json          run-level graph snapshot
episodes/           one self-contained subdir per episode:
  ep{idx:04d}/
    log.jsonl         this episode's per-node-firing ExecutionLogger stream
    assets/           this episode's image artefacts
    episode.json      this episode's row of summary.json (self-describing)
_DONE               clean-exit marker (presence ⇔ status was finalized)
```

When `_DONE` is missing but the subprocess PID is gone (e.g. backend
restart, OOM-killer SIGKILL, segfault), the scheduler's reap loop
flips `summary.json` to `status='aborted'` and writes `_DONE` itself
on next tick.

## Cancelling

```
# graceful, sent automatically on Ctrl-C
curl -X POST $BACKEND/api/eval/v2/runs/$RUN_ID/cancel
# → queued: cancelled immediately
# → running: cancelling (SIGTERM to subprocess pgid; subprocess
#   wraps in next tick → final status='cancelled' + _DONE)
```

## Invariants

- **Backend lifetime is not owned by this skill.** No spawn, no kill.
  The backend on `:8000` belongs to the user; we're a guest. If it
  isn't reachable, `submit.py` prints a reminder with the launch
  command (`cd agentcanvas && bash run_dev.sh`) and exits code 2.
- **Admission is server-side.** Admission decisions (VRAM accounting,
  exclusive_gpu, queue order) live in `JobScheduler` inside the
  backend, visible cross-session via `GET /api/eval/v2/queue`.
- **Cross-session visibility is automatic.** Other terminals running
  this same skill against the same backend share one queue + one
  resource ledger. No more "session A's admit script doesn't know
  about session B's running run".

## See also

- `.claude/commands/experiment/profiles.yaml` — profile catalog (vram_mb + exclusive_gpu per name)
- `.claude/commands/experiment/bin/submit.py` — the actual implementation
- `outputs/eval_runs/` — per-run artefacts (cross-session)
- `GET /api/eval/v2/queue` — scheduler view across all sessions
- `agentcanvas/backend/app/services/job_scheduler.py` — server-side admission + queue
