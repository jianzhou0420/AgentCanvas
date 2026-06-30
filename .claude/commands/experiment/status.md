# /experiment:status — query JobScheduler runs without killing them

Companion to `/experiment:run`. `/experiment:run` is a foreground
submitter — it polls inline and Ctrl-C cancels the run. `/experiment:status`
is the **read/observe** side: detach from a previously-submitted `run_id`
and come back later, across conversations, without killing the
subprocess.

## Usage

```
/experiment:status                      # list queue + running
/experiment:status <run_id>             # one-shot snapshot
/experiment:status <run_id> --watch     # follow until terminal; Ctrl-C exits without cancel
/experiment:status <run_id> --cancel    # POST cancel and exit
```

`run_id` is what `/experiment:run` printed on submission
(`[submit] run_id=170e3eb2-… initial_state=pending`). Same id
identifies the run dir at `outputs/eval_runs/{run_id}/`.

The backend URL defaults to `http://127.0.0.1:5173` (Vite frontend proxy
forwarding `/api` + `/ws` to backend `:8000`); override with
`AGENTCANVAS_BACKEND_URL`.

## Lifecycle

```
no args        ──► GET /api/eval/v2/queue (+ per-run augment)
<run_id>       ──► GET /api/eval/v2/runs/{run_id}  → print snapshot
--watch        ──► poll every --interval (default 2s) until status ∈ {done,error,aborted,cancelled}
                   Ctrl-C exits the watcher; run keeps running on the backend
--cancel       ──► POST /api/eval/v2/runs/{run_id}/cancel
```

## Steps

```bash
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
exec python \
  "$ROOT/.claude/commands/experiment/bin/status.py" "$@"
```

## Exit codes

| code | meaning |
|---|---|
| 0   | query succeeded (or `--watch` reached `done` / `completed`) |
| 1   | `--watch` reached `error` / `aborted` / `cancelled` |
| 2   | backend unreachable / `run_id` not found / bad flag combination |

## Difference from /experiment:run

| | `/experiment:run` | `/experiment:status` |
|---|---|---|
| Submits a new job | yes (POST `/start`) | no — query-only by default |
| Polls inline | yes, foreground until `_DONE` | only with `--watch` |
| Ctrl-C semantics | **cancel the run** | exit watcher; run keeps running |
| Cross-conversation | no — bound to the submitter | yes — any conversation can attach by `run_id` |
| Print depth | submit + state changes + final tail | snapshot or one-line per state change |

## Backend lifetime: not ours

This skill **never spawns a backend**. If the URL isn't reachable it
prints a reminder showing how to start one (`cd agentcanvas && bash
run_dev.sh`) and exits with code 2 — same contract as
`/experiment:run`. ADR-eval-003 mandates one long-lived backend per
host so sibling sessions share the same `JobScheduler` admission
ledger; auto-spawning here would create a second ledger and trample
on the first.

## Why this is needed

`submit.py`'s SIGINT handler fires `POST /runs/{run_id}/cancel` (see
`submit.py:226`), so killing the foreground submitter kills the job.
That's the right default for the launching session, but if you want to:

- **detach + resume later** (e.g. close a terminal, hit Ctrl-C in error,
  run a different skill in parallel),
- **inspect a job started in another Claude conversation**,
- **monitor without holding a poll loop open**,

you need a query-only entry point. `status.py` is exactly that — never
spawns, never sends `cancel` unless you ask with `--cancel`.

## See also

- `/experiment:run` — submit + foreground watch (cancels on interrupt)
- `/experiment:teardown` — drain the entire queue (admin / nuclear)
- `GET /api/eval/v2/queue` — same data, raw JSON
- `GET /api/eval/v2/runs/{run_id}` — same data, raw JSON
- Monitor page (frontend) — same data with sparklines for CPU/MEM/GPU
- `outputs/eval_runs/{run_id}/` — per-run artefacts (`summary.json`,
  `stderr.log`, `stdout.log`, `_DONE`)
