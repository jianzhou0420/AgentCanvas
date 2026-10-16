"""The run files: what crosses the process boundary between the parent
backend (JobScheduler, the eval API) and the eval subprocess, typed.

Layout under ``outputs/eval_runs/{run_id}/`` (``run_id`` = a second-precision
timestamp, e.g. ``20260515_143052``)::

    spec.json          — :class:`JobSpec`, backend writes pre-spawn
    shared_urls.json   — shared-nodeset URL table, backend writes pre-spawn
    summary.json       — :class:`RunSummary`, written at submit (pending), at
                         subprocess boot (running), after every episode and at
                         the end; the parent's cancel / spawn-failure / abort
                         paths rewrite it through :meth:`RunSummary.finish`
    _DONE              — subprocess touches last on clean exit. Absence + dead
                         PID ⇒ ``aborted`` (Q1).
    graph.json         — run-level graph snapshot (written by BatchEvalRunner).
    episodes/          — one self-contained subdir per episode:
        ep{idx:04d}/
            log.jsonl      — this episode's per-node-firing log (no interleave)
            assets/        — this episode's image artefacts
            episode.json   — this episode's row of summary.json (self-describing)

Every write is atomic (tempfile + rename, ADR-eval-003 rule 3), every read is
tolerant (a missing file is ``None``, a missing field takes its default —
``EvalConfig`` states the eval defaults, the dataclasses below state the
rest). ``summary.json`` is also read by the eval page, the run history, the
replay API, ``system_runs`` and external tooling, so its keys only ever grow.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..agent_loop.eval_config import EvalConfig

_TERMINAL = {"completed", "error", "cancelled", "aborted"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── files ───────────────────────────────────────────────────────────────


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via tempfile + rename so concurrent readers never see torn writes.

    Tempfile name includes pid + thread id so concurrent writers from
    different threads/processes do not race on the same temp inode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_shared_urls(run_dir: Path, urls: dict[str, str]) -> None:
    atomic_write_json(run_dir / "shared_urls.json", {"urls": urls})


def read_shared_urls(run_dir: Path) -> dict[str, str]:
    return (_read_json(run_dir / "shared_urls.json") or {}).get("urls", {})


def is_done(run_dir: Path) -> bool:
    return (run_dir / "_DONE").exists()


def touch_done(run_dir: Path) -> None:
    (run_dir / "_DONE").write_text(_now() + "\n")


# ── JobSpec ─────────────────────────────────────────────────────────────


@dataclass
class Scheduling:
    """The admission request that rides with a job (JobScheduler reads it)."""

    marginal_vram_mb: int = 0
    exclusive_gpu: bool = False
    priority: str = "normal"  # high / normal / low

    def to_dict(self) -> dict[str, Any]:
        return {
            "marginal_vram_mb": self.marginal_vram_mb,
            "exclusive_gpu": self.exclusive_gpu,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Scheduling:
        d = d or {}
        return cls(
            marginal_vram_mb=int(d.get("marginal_vram_mb") or 0),
            exclusive_gpu=bool(d.get("exclusive_gpu", False)),
            priority=str(d.get("priority") or "normal"),
        )


@dataclass
class JobSpec:
    """One eval job as the parent hands it to the subprocess: the eval config,
    the graph, the admission request and the workspace overlay. Nothing the
    subprocess could only recompute — ``eval.env_nodeset`` travels too.

    ``run_id`` / ``created_at`` are stamped by ``JobScheduler.submit`` when
    empty. The shared-nodeset URL table is deliberately not here: it is the
    parent's runtime knowledge, written next to the spec as
    ``shared_urls.json`` and passed to ``submit`` as an argument.
    """

    eval: EvalConfig
    graph: dict[str, Any] = field(default_factory=dict)  # GraphDefinition.to_dict()
    scheduling: Scheduling = field(default_factory=Scheduling)
    active_workspace_dir: str | None = None
    run_id: str = ""
    created_at: str = ""

    @property
    def node_types(self) -> list[str]:
        """The graph's node types (admission charges and calibration key on them)."""
        return [str(n.get("type", "")) for n in (self.graph.get("nodes") or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "eval": self.eval.to_dict(),
            "scheduling": self.scheduling.to_dict(),
            "graph": self.graph,
            "active_workspace_dir": self.active_workspace_dir,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobSpec:
        return cls(
            eval=EvalConfig.from_dict(d.get("eval")),
            graph=dict(d.get("graph") or {}),
            scheduling=Scheduling.from_dict(d.get("scheduling")),
            active_workspace_dir=d.get("active_workspace_dir") or None,
            run_id=str(d.get("run_id") or ""),
            created_at=str(d.get("created_at") or ""),
        )

    def write(self, run_dir: Path) -> None:
        atomic_write_json(run_dir / "spec.json", self.to_dict())

    @classmethod
    def read(cls, run_dir: Path) -> JobSpec | None:
        data = _read_json(run_dir / "spec.json")
        return cls.from_dict(data) if data is not None else None


# ── RunSummary ──────────────────────────────────────────────────────────


def episode_to_dict(ep: Any) -> dict:
    """Serialize one EpisodeResult to a JSON-safe dict.

    Single source of truth for episode shape — used both for the
    per-episode ``episodes/ep{idx}/episode.json`` (written by
    BatchEvalRunner) and inline inside :class:`RunSummary`'s
    ``episodes`` list.
    """
    return {
        "episode_index": ep.episode_index,
        "episode_id": ep.episode_id,
        "scene_id": ep.scene_id,
        "instruction": ep.instruction,
        "metrics": ep.metrics,
        "step_count": ep.step_count,
        "elapsed_sec": ep.elapsed_sec,
        "status": ep.status,
        "error": ep.error,
        # Effective selectors actually pushed for this episode (run-level
        # merged with the per-episode override). Lets offline tooling
        # re-group results by task without re-merging config.episode_selectors.
        "selectors": dict(ep.selectors),
        "worker_id": ep.worker_id,
    }


@dataclass
class RunSummary:
    """``summary.json`` — one run's config, progress and result.

    Built from a live ``EvalRun`` (:meth:`from_run`) or, before any episode
    ran, from the job's spec (:meth:`initial`); finished by the parent's
    cancel / abort / spawn-failure paths (:meth:`finish`). Readers get the
    on-disk dict (:meth:`to_dict`) or the eval page's progress view
    (:meth:`page_summary`, the ``EvalRun.to_summary()`` shape).
    """

    run_id: str
    config: EvalConfig
    status: str = "pending"
    episodes: list[dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    aggregate_by_task: dict[str, Any] = field(default_factory=dict)
    total_episodes: int = 0
    created_at: str | None = None
    finished_at: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None

    # ── builders ────────────────────────────────────────────────────

    @classmethod
    def from_run(cls, run: Any) -> RunSummary:
        """Snapshot a live ``EvalRun`` (status enum → its value)."""
        status = getattr(run.status, "value", run.status)
        return cls(
            run_id=run.run_id,
            config=run.config,
            status=str(status),
            episodes=[episode_to_dict(ep) for ep in run.episodes],
            aggregate_metrics=dict(run.aggregate_metrics),
            aggregate_by_task=dict(run.aggregate_by_task),
            total_episodes=run.total_episodes,
            created_at=run.created_at,
            finished_at=run.finished_at,
            elapsed_sec=run.elapsed_sec,
            error=run.error,
        )

    @classmethod
    def initial(cls, spec: JobSpec, status: str = "running") -> RunSummary:
        """The snapshot written before any episode has run — at submit
        (``pending``) and at subprocess boot (``running``) — so ``/runs/{id}``
        has something to serve from the first second."""
        return cls(run_id=spec.run_id, config=spec.eval, status=status, created_at=spec.created_at)

    def finish(self, status: str, error: str | None = None) -> RunSummary:
        """Mark the run terminal from the parent's side: ``status``, an
        appended ``error`` line when given, ``finished_at`` now."""
        self.status = status
        if error:
            self.error = f"{self.error}\n{error}" if self.error else error
        self.finished_at = _now()
        return self

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    # ── wire shapes ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status,
            "episodes": list(self.episodes),
            "aggregate_metrics": self.aggregate_metrics,
            "aggregate_by_task": dict(self.aggregate_by_task),
            "total_episodes": self.total_episodes,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunSummary:
        known = {f.name for f in fields(cls)} - {"config", "episodes"}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.setdefault("run_id", "")
        kwargs["status"] = str(kwargs.get("status") or "pending")
        kwargs["elapsed_sec"] = float(kwargs.get("elapsed_sec") or 0.0)
        kwargs["total_episodes"] = int(kwargs.get("total_episodes") or 0)
        for key in ("aggregate_metrics", "aggregate_by_task"):
            kwargs[key] = dict(kwargs.get(key) or {})
        return cls(
            config=EvalConfig.from_dict(d.get("config")),
            episodes=[dict(e) for e in (d.get("episodes") or [])],
            **kwargs,
        )

    def page_summary(self) -> dict[str, Any]:
        """The ``EvalRun.to_summary()`` shape the eval page consumes —
        ``completed_count`` counts every terminal episode (clean or errored),
        ``error_count`` the errored subset; neither is a task success rate."""
        completed = sum(1 for ep in self.episodes if ep.get("status") in ("completed", "error"))
        errored = sum(1 for ep in self.episodes if ep.get("status") == "error")
        return {
            "run_id": self.run_id,
            "graph_name": self.config.graph_name,
            "env_nodeset": self.config.env_nodeset,
            "status": self.status,
            "selectors": dict(self.config.selectors),
            "dataset": self.config.dataset,
            "split": self.config.split,
            "episode_count": self.config.episode_count,
            "total_episodes": self.total_episodes,
            "completed_count": completed,
            "error_count": errored,
            "elapsed_sec": round(self.elapsed_sec or 0.0, 1),
            "aggregate_metrics": self.aggregate_metrics,
            "aggregate_by_task": dict(self.aggregate_by_task),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    # ── disk ────────────────────────────────────────────────────────

    def write(self, run_dir: Path) -> None:
        atomic_write_json(run_dir / "summary.json", self.to_dict())

    @classmethod
    def read(cls, run_dir: Path) -> RunSummary | None:
        data = _read_json(run_dir / "summary.json")
        return cls.from_dict(data) if data is not None else None


def read_summary(run_dir: Path) -> dict | None:
    """``summary.json`` as the on-disk dict (``None`` when absent or unreadable)
    — for API responders that hand the dict straight back."""
    summary = RunSummary.read(run_dir)
    return summary.to_dict() if summary is not None else None


def mark_aborted(run_dir: Path, reason: str = "in-flight loss on backend restart") -> None:
    """Promote a stale ``running`` / ``pending`` summary to ``aborted`` when the
    PID is gone but ``_DONE`` was never written. Idempotent.
    """
    summary = RunSummary.read(run_dir)
    if summary is None or summary.status not in {"running", "pending"}:
        return
    summary.finish("aborted", f"[aborted] {reason}").write(run_dir)
