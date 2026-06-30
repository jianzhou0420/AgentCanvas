# `.claude/commands/experiment/` — thin client over backend's JobScheduler

Single user-managed agentcanvas backend per host (typically `:8000`).
The backend owns admission control, queueing, and per-run subprocess
lifecycle via `agentcanvas/backend/app/services/job_scheduler.py`.
This directory is just the **client side**: profiles + a submission
script that the `/experiment:run` skill execs.

## Layout

```
.claude/commands/experiment/
├── run.md status.md teardown.md   # slash-command defs (/experiment:run etc.)
├── profiles.yaml                  # resource catalog (vram_mb + exclusive_gpu per name)
└── bin/
    └── submit.py                  # HTTP client → POST /api/eval/v2/start (via_subprocess=true)
```

The command defs and the client code now live together in this one directory.

## How it works now

1. User starts one backend (e.g. `cd agentcanvas/backend && uvicorn app.main:app --reload --port 8000`).
2. Each `/experiment:run` invocation calls `bin/submit.py`, which:
   - reads the named profile → builds a `scheduling: {marginal_vram_mb, exclusive_gpu, priority}` block
   - POSTs `/api/eval/v2/start` with the eval block + `via_subprocess=true`
   - polls `/api/eval/v2/runs/{run_id}` until `_DONE` exists
   - on Ctrl-C, POSTs `/api/eval/v2/runs/{run_id}/cancel`
3. The backend's scheduler tick admits jobs that fit the VRAM budget,
   spawns each as a separate Python subprocess (`app.eval_subprocess_main`),
   reaps on exit, writes terminal `summary.json` + `_DONE`.

Cross-session admission is automatic: every Claude session that hits the
same backend shares one queue + one ledger. `GET /api/eval/v2/queue`
shows the global state.

## Adding a new experiment profile

1. Pick a kebab-case name in `profiles.yaml` under `experiments:`.
2. Set `vram_mb` (round up) and `exclusive_gpu`.
3. Run: `/experiment:run <name> <graph_name> [k=v ...]`.

## Inspecting state

```bash
# scheduler view
curl -s http://127.0.0.1:8000/api/eval/v2/queue | jq

# per-run artefacts (cross-session, persistent; run_id = timestamp)
ls outputs/eval_runs/{run_id}/
#   spec.json shared_urls.json summary.json stderr.log stdout.log _DONE graph.json
#   episodes/ep{NNNN}/{log.jsonl,assets/,episode.json}
```

## What changed (2026-05-07)

The pre-2026-05-07 skill was a local file-based admission controller
(`admit.py` / `release.py` / `running.json` / `backends.json`) that
spawned a per-run uvicorn on `:8765-8769`. That was replaced by:

- **Server-side admission** inside the backend (`JobScheduler`).
- **One long-lived backend**, user-managed; not per-experiment.
- **Run subprocesses** spawned by the scheduler (`app.eval_subprocess_main`),
  PR_SET_PDEATHSIG-tied to backend, fault-isolated from each other.

The old skill's local state (`runtime/`) was removed once nothing read it.

## See also

- `agentcanvas/backend/app/services/job_scheduler.py` — server-side scheduler
- `agentcanvas/backend/app/eval_subprocess_main.py` — subprocess entry point
- `.claude/plan/2026-05-07-experiment-subprocess-scheduler.md` — design doc
- `bin/submit.py` — the actual client
