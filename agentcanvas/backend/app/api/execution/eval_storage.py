"""Eval run persistence — JSON files in outputs/eval_runs/{run_id}/summary.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ...config import eval_runs_dir
from ...services.run_state_io import RunSummary, episode_to_dict, list_run_summaries

__all__ = ["delete_run", "episode_to_dict", "list_runs", "load_run", "run_to_dict", "save_run"]

log = logging.getLogger("agentcanvas.eval-storage")

# The eval run pool. One resolver (config.eval_runs_dir) for the scheduler,
# the runner, and these readers, so a backend configured with its own pool
# (AGENTCANVAS_EVAL_RUNS_DIR) writes runs where /runs/{run_id} looks.
EVAL_RUNS_DIR = eval_runs_dir()


def get_runs_dir() -> Path:
    """Resolve outputs/eval_runs/ path, creating it if needed."""
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_RUNS_DIR


def _summary_path(run_id: str) -> Path:
    """Run-level summary JSON. Siblings: graph.json + episodes/ep{idx}/."""
    return EVAL_RUNS_DIR / run_id / "summary.json"


def run_to_dict(run: Any) -> dict:
    """Serialize an EvalRun to its ``summary.json`` dict (``RunSummary``)."""
    return RunSummary.from_run(run).to_dict()


# Backward-compat alias for the previous private name.
_run_to_dict = run_to_dict


def save_run(run: Any) -> None:
    """Write an EvalRun's summary.json — atomically (ADR-eval-003 rule 3)."""
    get_runs_dir()
    RunSummary.from_run(run).write(EVAL_RUNS_DIR / run.run_id)
    log.info("Saved eval run %s to %s", run.run_id, _summary_path(run.run_id))


def load_run(run_id: str) -> dict | None:
    """Load a run by ID. Returns the dict or None if not found."""
    get_runs_dir()
    path = _summary_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("Failed to load eval run %s: %s", run_id, exc)
        return None


def list_runs() -> list[dict]:
    """List all runs as lightweight summaries (excludes episodes list)."""
    return list_run_summaries(get_runs_dir())


def delete_run(run_id: str) -> bool:
    """Delete a run's summary file (preserves graph.json + episodes/ siblings).

    Returns True if deleted, False if not found.
    """
    get_runs_dir()
    path = _summary_path(run_id)
    if not path.exists():
        return False
    path.unlink()
    log.info("Deleted eval run summary %s", run_id)
    return True
