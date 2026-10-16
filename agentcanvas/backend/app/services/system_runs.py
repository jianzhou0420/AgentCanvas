"""System Log — per-run aggregation for the Monitor page's Run view.

Reads a run's execution logs (``ExecutionLogger`` ``log.jsonl``) plus the
tee'd resource samples (``system.jsonl``) to produce a per-run performance
summary, for both run families — eval runs (one log per episode) and canvas
runs (one log) — through the ``LogStore``, which owns the layout and the
aggregation. The node-timing breakdown is whatever ``GraphExecutor``
recorded (``queue_wait_ms`` / ``compute_ms``), written once and shared by
both families.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from typing import Any

from ..logging.store import LOG_FILE, LogStore
from .run_state_io import list_run_summaries, read_summary

log = logging.getLogger("agentcanvas.system-runs")


def _store() -> LogStore:
    return LogStore.default()


def _canvas_active_run_id() -> str | None:
    """run_id of the live canvas execution, if one is running/paused."""
    try:
        from ..agent_loop.loop_runner import get_loop_runner

        st = get_loop_runner().get_status()
        if st.get("status") in ("running", "paused"):
            return st.get("execution_id")
    except Exception:
        return None
    return None


# ── unified run list ──


def list_all_runs() -> list[dict[str, Any]]:
    """Unified eval + canvas run list at run granularity, newest first."""
    store = _store()
    out: list[dict[str, Any]] = []

    for r in list_run_summaries(store.eval_runs):
        out.append(
            {
                "run_id": r.get("run_id"),
                "source": "eval",
                "graph_name": (r.get("config") or {}).get("graph_name"),
                "status": r.get("status"),
                "started": r.get("created_at"),
                "finished": r.get("finished_at"),
                "episode_count": r.get("total_episodes") or r.get("episode_count_saved"),
            }
        )

    active = _canvas_active_run_id()
    for run_id in store.canvas_run_ids():
        run_dir = store.canvas_run_dir(run_id)
        graph_name = None
        with contextlib.suppress(Exception):
            graph_name = (store.read_graph(run_id) or {}).get("name")
        try:
            mtime = (run_dir / LOG_FILE).stat().st_mtime
            started = datetime.utcfromtimestamp(mtime).isoformat()
        except OSError:
            started = None
        out.append(
            {
                "run_id": run_id,
                "source": "canvas",
                "graph_name": graph_name,
                "status": "running" if run_id == active else "done",
                "started": started,
                "finished": None,
                "episode_count": None,
            }
        )

    out.sort(key=lambda x: x.get("started") or "", reverse=True)
    return out


# ── per-run detail ──


def run_detail(run_id: str) -> dict[str, Any] | None:
    """Per-run System Log detail: node timing + totals + run-local resource
    series. Recomputed on demand (Run view doesn't poll) and write-through
    cached to ``system_summary.json`` as an artifact."""
    store = _store()
    resolved = store.resolve_run(run_id)
    if resolved is None:
        return None
    run_dir, source = resolved

    agg = store.timing(store.run_log_files(run_dir, source))
    detail: dict[str, Any] = {
        "run_id": run_id,
        "source": source,
        "node_timing": agg["node_timing"],
        "totals": agg["totals"],
        "resources": store.resource_samples(run_dir),
    }
    if source == "eval":
        detail["eval"] = read_summary(run_dir)  # full summary incl. episodes/metrics

    with contextlib.suppress(OSError):
        (run_dir / "system_summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "source": source,
                    "node_timing": agg["node_timing"],
                    "totals": agg["totals"],
                },
                indent=2,
            )
        )

    return detail
