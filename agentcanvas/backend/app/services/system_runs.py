"""System Log — per-run aggregation for the Monitor page's Run view.

Reads a run's existing execution logs (ExecutionLogger ``log.jsonl``) plus the
tee'd resource samples (``system.jsonl``) to produce a per-run performance
summary. Works for both run families:

* eval   → ``outputs/eval_runs/{id}/episodes/ep*/log.jsonl`` + run-local ``system.jsonl``
* canvas → ``outputs/runs/{id}/log.jsonl``                  + run-local ``system.jsonl``

The node-timing breakdown is whatever ``GraphExecutor`` recorded
(``queue_wait_ms`` / ``compute_ms``) — written once, shared by both families
(canvas Play and every eval episode run through the same executor).
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..api.execution.eval_storage import EVAL_RUNS_DIR
from ..api.execution.eval_storage import list_runs as _list_eval_runs
from ..api.execution.eval_storage import load_run as _load_eval_run

log = logging.getLogger("agentcanvas.system-runs")

OUTPUTS_DIR = EVAL_RUNS_DIR.parent
CANVAS_RUNS_DIR = OUTPUTS_DIR / "runs"


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
    out: list[dict[str, Any]] = []

    for r in _list_eval_runs():
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
    if CANVAS_RUNS_DIR.is_dir():
        for sub in CANVAS_RUNS_DIR.iterdir():
            if not sub.is_dir() or not (sub / "log.jsonl").exists():
                continue
            graph_name = None
            gp = sub / "graph.json"
            if gp.exists():
                try:
                    graph_name = (json.loads(gp.read_text()) or {}).get("name")
                except Exception:
                    graph_name = None
            try:
                mtime = (sub / "log.jsonl").stat().st_mtime
                started = datetime.utcfromtimestamp(mtime).isoformat()
            except OSError:
                started = None
            out.append(
                {
                    "run_id": sub.name,
                    "source": "canvas",
                    "graph_name": graph_name,
                    "status": "running" if sub.name == active else "done",
                    "started": started,
                    "finished": None,
                    "episode_count": None,
                }
            )

    out.sort(key=lambda x: x.get("started") or "", reverse=True)
    return out


# ── per-run detail ──


def _resolve_run_dir(run_id: str) -> tuple[Path | None, str]:
    ed = EVAL_RUNS_DIR / run_id
    if ed.is_dir():
        return ed, "eval"
    cd = CANVAS_RUNS_DIR / run_id
    if cd.is_dir():
        return cd, "canvas"
    return None, ""


def _log_files(run_dir: Path, source: str) -> list[Path]:
    if source == "eval":
        return sorted(run_dir.glob("episodes/ep*/log.jsonl"))
    p = run_dir / "log.jsonl"
    return [p] if p.exists() else []


def _pctile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, round((q / 100.0) * (len(s) - 1))))
    return round(s[i], 2)


def _aggregate(run_dir: Path, source: str) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    firings = 0
    grand_compute = 0.0
    tokens = 0
    cost = 0.0
    llm_calls = 0
    total_transport = 0.0
    total_bytes = 0

    for lf in _log_files(run_dir, source):
        try:
            with open(lf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    nt = e.get("node_type", "?")
                    comp = e.get("compute_ms")
                    if comp is None:  # old logs / terminal-band → fall back to duration
                        comp = e.get("duration_ms") or 0.0
                    qw = e.get("queue_wait_ms")
                    b = by_type.setdefault(
                        nt, {"count": 0, "compute": [], "queue": [], "transport": [], "bytes": 0}
                    )
                    b["count"] += 1
                    b["compute"].append(float(comp))
                    if qw is not None:
                        b["queue"].append(float(qw))
                    tm = e.get("transport_ms")
                    if tm is not None:
                        b["transport"].append(float(tm))
                        total_transport += float(tm)
                    tb = e.get("transfer_bytes")
                    if tb:
                        b["bytes"] += int(tb)
                        total_bytes += int(tb)
                    firings += 1
                    grand_compute += float(comp)
                    for il in e.get("inner_log") or []:
                        if isinstance(il, dict) and il.get("key") == "usage":
                            u = il.get("value") or {}
                            tokens += int(u.get("total_tokens") or 0)
                            cost += float(u.get("usd_cost") or 0.0)
                            llm_calls += int(u.get("calls") or 0)
        except OSError:
            continue

    node_timing: list[dict[str, Any]] = []
    for nt, b in by_type.items():
        comp = b["compute"]
        total = sum(comp)
        node_timing.append(
            {
                "node_type": nt,
                "count": b["count"],
                "compute_ms": {
                    "mean": round(total / len(comp), 2) if comp else 0.0,
                    "p5": _pctile(comp, 5),
                    "p50": _pctile(comp, 50),
                    "p95": _pctile(comp, 95),
                    "total": round(total, 2),
                },
                "queue_wait_ms": {
                    "mean": round(sum(b["queue"]) / len(b["queue"]), 2) if b["queue"] else None,
                    "p95": _pctile(b["queue"], 95) if b["queue"] else None,
                },
                "transport_ms": (
                    {
                        "total": round(sum(b["transport"]), 2),
                        "mean": round(sum(b["transport"]) / len(b["transport"]), 2),
                        "p5": _pctile(b["transport"], 5),
                        "p95": _pctile(b["transport"], 95),
                    }
                    if b["transport"]
                    else None
                ),
                "transfer_bytes": b["bytes"],
                "share_pct": round(total / grand_compute * 100.0, 1) if grand_compute else 0.0,
            }
        )
    node_timing.sort(key=lambda x: x["compute_ms"]["total"], reverse=True)

    return {
        "node_timing": node_timing,
        "totals": {
            "firings": firings,
            "compute_ms": round(grand_compute, 2),
            "tokens": tokens,
            "usd_cost": round(cost, 6),
            "llm_calls": llm_calls,
            "transport_ms": round(total_transport, 2),
            "transfer_bytes": total_bytes,
        },
    }


def _read_resources(run_dir: Path, cap: int = 1200) -> list[dict[str, Any]]:
    p = run_dir / "system.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):
                    out.append(json.loads(line))
    except OSError:
        return []
    return out[-cap:]


def run_detail(run_id: str) -> dict[str, Any] | None:
    """Per-run System Log detail: node timing + totals + run-local resource
    series. Recomputed on demand (Run view doesn't poll) and write-through
    cached to ``system_summary.json`` as an artifact."""
    run_dir, source = _resolve_run_dir(run_id)
    if run_dir is None:
        return None

    agg = _aggregate(run_dir, source)
    detail: dict[str, Any] = {
        "run_id": run_id,
        "source": source,
        "node_timing": agg["node_timing"],
        "totals": agg["totals"],
        "resources": _read_resources(run_dir),
    }
    if source == "eval":
        detail["eval"] = _load_eval_run(run_id)  # full summary incl. episodes/metrics

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
