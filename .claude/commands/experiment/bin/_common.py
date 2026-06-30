"""Shared bits for experiment skill CLIs (submit.py, status.py).

Single source of truth for the default backend URL and the
"backend not running" reminder. Per ADR-eval-003 the experiment skill
NEVER hosts its own backend — that's the user's job. We're a guest.
"""

from __future__ import annotations

import os
import sys

DEFAULT_BACKEND = os.environ.get("AGENTCANVAS_BACKEND_URL", "http://127.0.0.1:5173")


def die_unreachable(backend: str, exc: BaseException, prog: str = "experiment") -> int:
    """Print a friendly, actionable reminder when the backend is unreachable.

    The skill does NOT spawn a backend on the user's behalf (that would
    fight any backend already running, fight the user's conda env, and
    create a second admission ledger). Instead we ask the user to start
    one and try again. Returns 2 so callers can ``sys.exit`` directly.
    """
    msg = f"""[{prog}] backend unreachable at {backend}: {exc}

The experiment skill does not host a backend itself — by design, see
ADR-eval-003 (one long-lived backend per host, multiple sessions submit
to the same JobScheduler).

Please start the agentcanvas backend yourself, then retry:

    cd agentcanvas && bash run_dev.sh
        # backend on :8000 + Vite proxy on :5173 (recommended)

    # OR backend only:
    cd agentcanvas/backend && \\
      uvicorn \\
        app.main:app --reload --port 8000

If your backend is on a non-default URL, point this skill at it via:

    export AGENTCANVAS_BACKEND_URL=http://127.0.0.1:8000   # or your URL

Quick health probe:

    curl -fs {backend}/api/eval/v2/queue && echo OK
"""
    print(msg, file=sys.stderr)
    return 2
