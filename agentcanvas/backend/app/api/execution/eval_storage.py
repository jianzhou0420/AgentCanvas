"""Eval run persistence — JSON files in outputs/eval_runs/{run_id}/summary.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("agentcanvas.eval-storage")

# Resolve outputs/eval_runs/ relative to this file's location:
# execution/eval_storage.py -> execution/ -> api/ -> app/ -> backend/ -> agentcanvas/ -> vlnworkspace/
_REPO_ROOT = Path(__file__).resolve().parents[5]
EVAL_RUNS_DIR = _REPO_ROOT / "outputs" / "eval_runs"


def get_runs_dir() -> Path:
    """Resolve outputs/eval_runs/ path, creating it if needed."""
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_RUNS_DIR


def _summary_path(run_id: str) -> Path:
    """Run-level summary JSON. Siblings: graph.json + episodes/ep{idx}/."""
    return EVAL_RUNS_DIR / run_id / "summary.json"


def episode_to_dict(ep: Any) -> dict:
    """Serialize one EpisodeResult to a JSON-safe dict.

    Single source of truth for episode shape — used both for the
    per-episode ``episodes/ep{idx}/episode.json`` (written by
    BatchEvalRunner) and inline inside :func:`run_to_dict`'s
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


def run_to_dict(run: Any) -> dict:
    """Serialize an EvalRun to a JSON-safe dict (excludes non-serializable fields)."""
    config_dict = {
        "graph_name": run.config.graph_name,
        "env_nodeset": run.config.env_nodeset,
        # Generic cascade — the canonical "what was actually pushed
        # through the env panel" record. Legacy ``split`` is also
        # serialized below so old viewers / replay tooling keep working.
        "selectors": dict(run.config.selectors),
        # Per-episode selector overrides, parallel to episode order.
        # ``None`` for runs that didn't use the cross-task path.
        "episode_selectors": (
            [dict(s) for s in run.config.episode_selectors]
            if run.config.episode_selectors is not None
            else None
        ),
        "split": run.config.split,
        "episode_count": run.config.episode_count,
        "step_budget": run.config.step_budget,
    }

    episodes_list = [episode_to_dict(ep) for ep in run.episodes]

    return {
        "run_id": run.run_id,
        "config": config_dict,
        "status": run.status.value,
        "episodes": episodes_list,
        "aggregate_metrics": run.aggregate_metrics,
        "aggregate_by_task": dict(run.aggregate_by_task),
        "total_episodes": run.total_episodes,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "elapsed_sec": run.elapsed_sec,
        "error": run.error,
    }


# Backward-compat alias for the previous private name.
_run_to_dict = run_to_dict


def save_run(run: Any) -> None:
    """Serialize an EvalRun to outputs/eval_runs/{run_id}/summary.json."""
    get_runs_dir()
    path = _summary_path(run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = run_to_dict(run)
    path.write_text(json.dumps(data, indent=2))
    log.info("Saved eval run %s to %s", run.run_id, path)


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
    runs_dir = get_runs_dir()
    results: list[dict] = []
    for sub in sorted(runs_dir.iterdir(), reverse=True) if runs_dir.exists() else []:
        if not sub.is_dir():
            continue
        path = sub / "summary.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            # Return summary without full episodes list
            summary = {k: v for k, v in data.items() if k != "episodes"}
            summary["episode_count_saved"] = len(data.get("episodes", []))
            # Subprocess-path summaries carry graph_name only inside config;
            # promote it so list consumers (RunHistory rows) see a name.
            if not summary.get("graph_name"):
                summary["graph_name"] = (data.get("config") or {}).get("graph_name", "")
            results.append(summary)
        except Exception as exc:
            log.warning("Failed to read eval run %s: %s", path, exc)
    return results


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
