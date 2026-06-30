"""File protocol shared between JobScheduler (parent) and eval_subprocess_main (subprocess).

Layout under ``outputs/eval_runs/{run_id}/`` (``run_id`` = a second-precision
timestamp, e.g. ``20260515_143052``)::

    spec.json          — backend writes pre-spawn (full JobSpec + graph)
    shared_urls.json   — backend writes pre-spawn (shared nodeset URL table)
    summary.json       — subprocess writes (initial running snapshot, per-episode,
                          terminal). Schema = eval_storage.run_to_dict.
    _DONE              — subprocess touches last on clean exit. Absence + dead PID
                          ⇒ ``aborted`` (Q1).
    graph.json         — run-level graph snapshot (written by BatchEvalRunner).
    episodes/          — one self-contained subdir per episode:
        ep{idx:04d}/
            log.jsonl      — this episode's per-node-firing log (no interleave)
            assets/        — this episode's image artefacts
            episode.json   — this episode's row of summary.json (self-describing)

``spec.json`` schema (assembled in ``api/execution/eval.py``)::

    {
        "eval": {graph_name, selectors, dataset, split, episode_count,
                 step_budget, start_episode_index, worker_count,
                 per_step_budget_sec, episode_indices, episode_selectors},
        "scheduling": {marginal_vram_mb, exclusive_gpu, priority},
        "graph": GraphDefinition.to_dict(),
        "_shared_urls": {nodeset_name: auto_host_url},  # popped before disk write
        "active_workspace_dir": str | None,
            # When non-null, JobScheduler._spawn() sets ACTIVE_WORKSPACE_DIR
            # in the subprocess env so its WorkspaceComponentRegistry overlays this
            # dir on top of frozen workspace. Used by architect skills.
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via tempfile + rename so concurrent readers never see torn writes.

    Tempfile name includes pid + thread id so concurrent writers from
    different threads/processes do not race on the same temp inode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def write_spec(run_dir: Path, spec: dict) -> None:
    atomic_write_json(run_dir / "spec.json", spec)


def write_shared_urls(run_dir: Path, urls: dict[str, str]) -> None:
    atomic_write_json(run_dir / "shared_urls.json", {"urls": urls})


def read_spec(run_dir: Path) -> dict | None:
    p = run_dir / "spec.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def read_shared_urls(run_dir: Path) -> dict[str, str]:
    p = run_dir / "shared_urls.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("urls", {})


def read_summary(run_dir: Path) -> dict | None:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def is_done(run_dir: Path) -> bool:
    return (run_dir / "_DONE").exists()


def touch_done(run_dir: Path) -> None:
    (run_dir / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))


def mark_aborted(run_dir: Path, reason: str = "in-flight loss on backend restart") -> None:
    """Promote a stale ``status='running'`` summary to ``aborted`` when the
    PID is gone but ``_DONE`` was never written. Idempotent.
    """
    summary = read_summary(run_dir)
    if summary is None:
        return
    if summary.get("status") not in {"running", "pending"}:
        return
    summary["status"] = "aborted"
    summary["error"] = (summary.get("error") or "") + f"\n[aborted] {reason}"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(run_dir / "summary.json", summary)


def initial_running_summary(run_id: str, eval_block: dict, created_at: str) -> dict:
    """Return the summary snapshot the subprocess writes the moment it boots,
    before the first episode finishes. Schema-compatible with run_to_dict
    so the existing /runs and /export endpoints can serve it as-is.
    """
    return {
        "run_id": run_id,
        "config": {
            "graph_name": eval_block.get("graph_name", ""),
            "env_nodeset": "",
            "selectors": dict(eval_block.get("selectors") or {}),
            "episode_selectors": eval_block.get("episode_selectors"),
            "split": eval_block.get("split", ""),
            "episode_count": eval_block.get("episode_count", -1),
            "step_budget": eval_block.get("step_budget"),
        },
        "status": "running",
        "episodes": [],
        "aggregate_metrics": {},
        "aggregate_by_task": {},
        "total_episodes": 0,
        "created_at": created_at,
        "finished_at": None,
        "elapsed_sec": 0.0,
        "error": None,
    }
