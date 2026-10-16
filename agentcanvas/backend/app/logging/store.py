"""The execution-log store: where every run's ``log.jsonl`` lives, how an
execution is addressed, and how its entries are read and aggregated.

One ``ExecutionLogger`` writes a run's log (ADR-observability-003); this
module is the one place that knows the layout the writer uses and the
readers depend on — the logs API, the replay API, the Monitor's run view::

    {canvas_runs}/{execution_id}/            — one canvas Play execution
        log.jsonl · assets/ · graph.json
    {eval_runs}/{run_id}/                    — one batch-eval run
        graph.json · summary.json · system.jsonl
        episodes/ep{idx:04d}/                — one episode = one execution
            log.jsonl · assets/ · episode.json

An eval episode is addressed by the composite execution id
``{run_id}_ep{idx:04d}``; :meth:`LogStore.locate` resolves either kind.
Every reader goes through :meth:`LogStore.read_entries` (tolerant JSONL:
blank and malformed lines are skipped) and the aggregations here, so a
layout change is one edit and a ``tmp_path`` is enough to test them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_FILE = "log.jsonl"
ASSETS_DIR = "assets"
GRAPH_FILE = "graph.json"
EPISODES_DIR = "episodes"
RESOURCES_FILE = "system.jsonl"


@dataclass(frozen=True)
class LogDir:
    """One execution's directory: ``source`` is ``"canvas"`` or ``"eval"``."""

    path: Path
    source: str

    @property
    def log_path(self) -> Path:
        return self.path / LOG_FILE

    def asset_path(self, filename: str) -> Path:
        return self.path / ASSETS_DIR / filename


class LogStore:
    """The layout, the reader and the aggregations over execution logs."""

    def __init__(self, canvas_runs: Path, eval_runs: Path) -> None:
        self.canvas_runs = Path(canvas_runs)
        self.eval_runs = Path(eval_runs)

    @classmethod
    def default(cls) -> LogStore:
        """The process's pools (``config.canvas_runs_dir`` / ``config.eval_runs_dir``)."""
        from ..config import canvas_runs_dir, eval_runs_dir

        return cls(canvas_runs_dir(), eval_runs_dir())

    # ── addressing ──────────────────────────────────────────────────

    @staticmethod
    def episode_execution_id(run_id: str, episode_index: int) -> str:
        """The composite execution id of one eval episode."""
        return f"{run_id}_ep{episode_index:04d}"

    @staticmethod
    def split_execution_id(execution_id: str) -> tuple[str, int] | None:
        """``(run_id, episode_index)`` of a composite id, ``None`` for a canvas id."""
        run_id, sep, ep = execution_id.rpartition("_ep")
        if not sep or not ep.isdigit():
            return None
        return run_id, int(ep)

    @staticmethod
    def episode_subdir(run_dir: Path | str, episode_index: int) -> Path:
        """An episode's directory under its run dir (the writer's rule)."""
        return Path(run_dir) / EPISODES_DIR / f"ep{episode_index:04d}"

    def canvas_run_dir(self, execution_id: str) -> Path:
        return self.canvas_runs / execution_id

    def eval_run_dir(self, run_id: str) -> Path:
        return self.eval_runs / run_id

    def episode_dir(self, run_id: str, episode_index: int) -> Path:
        return self.episode_subdir(self.eval_run_dir(run_id), episode_index)

    def locate(self, execution_id: str) -> LogDir | None:
        """The directory holding this execution's ``log.jsonl`` — a canvas run
        first, else the eval episode the composite id names; ``None`` when
        neither has a log."""
        canvas = self.canvas_run_dir(execution_id)
        if (canvas / LOG_FILE).is_file():
            return LogDir(canvas, "canvas")
        parts = self.split_execution_id(execution_id)
        if parts is not None:
            episode = self.episode_dir(*parts)
            if (episode / LOG_FILE).is_file():
                return LogDir(episode, "eval")
        return None

    def log_path(self, execution_id: str) -> Path | None:
        located = self.locate(execution_id)
        return located.log_path if located is not None else None

    def asset_path(self, execution_id: str, filename: str) -> Path | None:
        """An asset of this execution (``assets/{filename}``), or ``None``."""
        for candidate in self._candidate_dirs(execution_id):
            path = candidate.asset_path(filename)
            if path.is_file():
                return path
        return None

    def graph_path(self, execution_id: str) -> Path | None:
        """The graph that ran: a canvas run's own ``graph.json``, or the
        run-level ``graph.json`` shared by every episode of an eval run."""
        candidates = [self.canvas_run_dir(execution_id) / GRAPH_FILE]
        parts = self.split_execution_id(execution_id)
        if parts is not None:
            candidates.append(self.eval_run_dir(parts[0]) / GRAPH_FILE)
        return next((p for p in candidates if p.is_file()), None)

    def resolve_run(self, run_id: str) -> tuple[Path, str] | None:
        """A run-level dir by id: ``(dir, "eval" | "canvas")``."""
        eval_dir = self.eval_run_dir(run_id)
        if eval_dir.is_dir():
            return eval_dir, "eval"
        canvas_dir = self.canvas_run_dir(run_id)
        if canvas_dir.is_dir():
            return canvas_dir, "canvas"
        return None

    def run_log_files(self, run_dir: Path, source: str) -> list[Path]:
        """Every ``log.jsonl`` of a run: the one file of a canvas run, the
        per-episode files of an eval run (in episode order)."""
        if source == "eval":
            return sorted(run_dir.glob(f"{EPISODES_DIR}/ep*/{LOG_FILE}"))
        path = run_dir / LOG_FILE
        return [path] if path.is_file() else []

    def _candidate_dirs(self, execution_id: str) -> list[LogDir]:
        dirs = [LogDir(self.canvas_run_dir(execution_id), "canvas")]
        parts = self.split_execution_id(execution_id)
        if parts is not None:
            dirs.append(LogDir(self.episode_dir(*parts), "eval"))
        return dirs

    # ── reading ─────────────────────────────────────────────────────

    @staticmethod
    def read_entries(path: Path | str) -> list[dict[str, Any]]:
        """Every entry of one ``log.jsonl``; a missing file reads as empty,
        blank and malformed lines are skipped."""
        entries: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        entries.append(entry)
        except OSError:
            return []
        return entries

    def entries(self, execution_id: str) -> list[dict[str, Any]]:
        path = self.log_path(execution_id)
        return self.read_entries(path) if path is not None else []

    def read_graph(self, execution_id: str) -> dict[str, Any] | None:
        """The graph that ran, or ``None`` when none was saved. Raises
        ``OSError`` / ``json.JSONDecodeError`` on an unreadable file."""
        path = self.graph_path(execution_id)
        if path is None:
            return None
        return json.loads(path.read_text())

    def resource_samples(self, run_dir: Path, cap: int = 1200) -> list[dict[str, Any]]:
        """The last ``cap`` rows of a run's ``system.jsonl`` resource tee."""
        rows = self.read_entries(run_dir / RESOURCES_FILE)
        return rows[-cap:]

    # ── listing ─────────────────────────────────────────────────────

    def canvas_run_ids(self) -> list[str]:
        """Canvas executions that have a log, newest id first."""
        if not self.canvas_runs.is_dir():
            return []
        return sorted(
            (p.name for p in self.canvas_runs.iterdir() if (p / LOG_FILE).is_file()),
            reverse=True,
        )

    def list_executions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Recent executions with a log, newest modification first:
        ``{execution_id, source, size_bytes, modified}`` — canvas runs, then
        one row per eval episode under the composite id."""
        rows: list[dict[str, Any]] = []
        for name in self.canvas_run_ids()[:50]:
            rows.append(self._row(name, "canvas", self.canvas_run_dir(name) / LOG_FILE))
        if self.eval_runs.is_dir():
            for run_dir in sorted(self.eval_runs.iterdir(), reverse=True)[:50]:
                episodes_dir = run_dir / EPISODES_DIR
                if not episodes_dir.is_dir():
                    continue
                for ep_dir in sorted(episodes_dir.iterdir()):
                    log_path = ep_dir / LOG_FILE
                    if not log_path.is_file():
                        continue
                    ep = ep_dir.name[2:] if ep_dir.name.startswith("ep") else ep_dir.name
                    rows.append(self._row(f"{run_dir.name}_ep{ep}", "eval", log_path))
                if len(rows) > 200:
                    break
        rows.sort(key=lambda r: r["modified"], reverse=True)
        return rows[:limit]

    @staticmethod
    def _row(execution_id: str, source: str, log_path: Path) -> dict[str, Any]:
        stat = os.stat(log_path)
        return {
            "execution_id": execution_id,
            "source": source,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }

    # ── aggregation ─────────────────────────────────────────────────

    @staticmethod
    def summarize(execution_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """The logs API's execution summary (``ExecutionSummary`` shape)."""
        if not entries:
            return {"execution_id": execution_id, "total_firings": 0}
        node_types: set[str] = set()
        error_count = 0
        max_step = 0
        for e in entries:
            node_types.add(str(e.get("node_type", "")))
            if e.get("error"):
                error_count += 1
            max_step = max(max_step, int(e.get("step", 0) or 0))
        return {
            "execution_id": execution_id,
            "source": entries[0].get("source", "unknown"),
            "started_at": entries[0].get("timestamp"),
            "ended_at": entries[-1].get("timestamp"),
            "total_steps": max_step,
            "total_firings": len(entries),
            "error_count": error_count,
            "node_types_fired": sorted(node_types),
        }

    @classmethod
    def timing(cls, log_files: list[Path]) -> dict[str, Any]:
        """The Monitor's per-run performance breakdown over a run's logs:
        per node type — firings, ``compute_ms`` / ``queue_wait_ms`` /
        ``transport_ms`` distributions, bytes, share of compute — plus the
        run totals (firings, compute, LLM tokens / cost / calls from the
        nodes' ``usage`` inner-log records, transport). Old entries without
        ``compute_ms`` count their ``duration_ms``."""
        by_type: dict[str, dict[str, Any]] = {}
        firings = 0
        grand_compute = 0.0
        tokens = 0
        cost = 0.0
        llm_calls = 0
        total_transport = 0.0
        total_bytes = 0
        for log_file in log_files:
            for e in cls.read_entries(log_file):
                nt = str(e.get("node_type", "?"))
                comp = e.get("compute_ms")
                if comp is None:  # old logs / terminal-band → fall back to duration
                    comp = e.get("duration_ms") or 0.0
                b = by_type.setdefault(
                    nt, {"count": 0, "compute": [], "queue": [], "transport": [], "bytes": 0}
                )
                b["count"] += 1
                b["compute"].append(float(comp))
                qw = e.get("queue_wait_ms")
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


def _pctile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, round((q / 100.0) * (len(s) - 1))))
    return round(s[i], 2)
