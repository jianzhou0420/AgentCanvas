# /experiment:teardown — cancel all in-flight jobs in the scheduler

Drain the JobScheduler: cancel every queued + running job in the
backend at `$AGENTCANVAS_BACKEND_URL` (default `http://127.0.0.1:8000`).
Does NOT touch the backend process itself — that's user-owned now.

The legacy semantics ("kill registered backends") is obsolete because
backends are no longer per-run. Scheduler auto-reaps subprocesses on
exit; this skill only exists for the rare "drain everything before I
restart the backend" case.

## Usage

```
/experiment:teardown
```

No args. Prints what it cancels.

## Steps

```bash
set -uo pipefail
BACKEND="${AGENTCANVAS_BACKEND_URL:-http://127.0.0.1:8000}"
curl -sf "$BACKEND/health" >/dev/null \
  || { echo "[teardown] backend unreachable at $BACKEND" >&2; exit 1; }

curl -s "$BACKEND/api/eval/v2/queue" | python3 -c "
import json, sys, urllib.request
q = json.load(sys.stdin)
ids = [j['run_id'] for j in q['queued']] + [j['run_id'] for j in q['running']]
if not ids:
    print('[teardown] queue empty')
    sys.exit(0)
for rid in ids:
    req = urllib.request.Request(
        f'$BACKEND/api/eval/v2/runs/{rid}/cancel',
        method='POST',
        headers={'Content-Type': 'application/json'},
        data=b'{}',
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            new = json.loads(r.read())['status']
        print(f'[teardown] {rid} -> {new}')
    except Exception as e:
        print(f'[teardown] {rid} cancel failed: {e}', file=sys.stderr)
"
```

## Invariants

- **Does not kill the backend.** Backend is user-managed; teardown only
  drains the scheduler's queue.
- **Idempotent.** Re-running on an empty queue is a no-op.
- **Cross-session.** Cancels every session's jobs at this backend, not
  just yours. Use deliberately.

## See also

- `GET /api/eval/v2/queue` — list state without cancelling
- `POST /api/eval/v2/runs/{id}/cancel` — single-job cancel
