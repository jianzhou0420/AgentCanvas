"""Eval tools — httpx wrappers for live ops, direct file read for export.

The backend is single-slot (one active run at a time, enforced by
ExecutionGuard) and self-terminates when the run finishes. So:

- ``eval_start`` / ``eval_status`` / ``eval_stop`` go over HTTP to the
  per-conversation backend (live state lives in memory there).
- ``eval_export`` reads ``outputs/eval_runs/{run_id}/summary.json``
  directly — no backend needed. This survives backend death and works
  for any historical run, regardless of which conversation produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT_SEC = 30.0

# tools/eval.py → tools → mcp_server → agentcanvas → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_RUNS_DIR = _REPO_ROOT / "outputs" / "eval_runs"


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT_SEC)


def _structured_error(r: httpx.Response) -> dict[str, Any]:
    """Convert a 4xx response into a structured ``{error, status_code, detail}``
    dict so the LLM can react without seeing a tool-execution exception."""
    try:
        detail = r.json()
    except ValueError:
        detail = r.text
    return {
        "error": f"backend returned HTTP {r.status_code}",
        "status_code": r.status_code,
        "detail": detail,
    }


def eval_start(
    base_url: str,
    *,
    graph_name: str,
    episode_count: int = 10,
    worker_count: int = 1,
    step_budget: int | None = None,
    split: str = "val_unseen",
    selectors: dict[str, Any] | None = None,
    start_episode_index: int = 0,
    episode_indices: list[int] | None = None,
    episode_selectors: list[dict[str, Any]] | None = None,
    per_step_budget_sec: float | None = None,
) -> dict[str, Any]:
    """Start a batch eval run. Returns ``{run_id, status}`` or ``{error}``.

    ``step_budget`` is the per-episode iteration cap. ``None`` (default)
    lets the framework's resolver chain pick from the env's per-episode
    value (e.g. HM-EQA's ``int(sqrt(scene_size) * 3)``), then the graph's
    authored value, then the system default.

    ``selectors`` is the generic env-controller cascade dict — insertion
    order = the order in which the runner pushes fields through the
    controller before each episode. Use it for envs whose controller has
    fields beyond ``split``, e.g. SIMPLER (``{"task_id":
    "google_robot_pick_horizontal_coke_can"}`` alongside
    ``split="google_robot"``) or LIBERO. Do NOT include ``episode_index``
    — the runner pushes that itself per episode.

    ``episode_selectors`` enables cross-task sweeps in a single run. When
    set, each entry is merged on top of run-level ``selectors`` for that
    one episode. Length must match the resolved index list
    (``episode_indices`` length, or ``episode_count``). Order entries
    **task-contiguous** — workers consume contiguous chunks, so
    interleaving tasks would force every worker subprocess to switch
    tasks on every episode (SAPIEN/SIMPLER will crash after ~15 task
    switches per subprocess). Do NOT include ``episode_index``.

    Example — SIMPLER 25 tasks x 4 episodes each in one run::

        tasks = [
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            ...  # 25 task ids
        ]
        # Task-contiguous: 4 episodes of task[0], then 4 of task[1], ...
        episode_selectors = [
            {"task_id": t} for t in tasks for _ in range(4)
        ]
        episode_indices = [i for _ in tasks for i in range(4)]
        eval_start(
            base_url,
            graph_name="simpler_pi0",
            split="google_robot",
            selectors={},  # task_id flows through episode_selectors
            episode_selectors=episode_selectors,
            episode_indices=episode_indices,
            worker_count=4,
        )
    """
    payload: dict[str, Any] = {
        "graph_name": graph_name,
        "episode_count": episode_count,
        "worker_count": worker_count,
        "start_episode_index": start_episode_index,
    }
    if step_budget is not None:
        payload["step_budget"] = step_budget
    # Only send `split` when non-empty — env nodesets without splits
    # (e.g. EQA-shaped graphs) reject the field.
    if split:
        payload["split"] = split
    if selectors:
        payload["selectors"] = dict(selectors)
    if episode_indices is not None:
        payload["episode_indices"] = episode_indices
    if episode_selectors is not None:
        payload["episode_selectors"] = [dict(s) for s in episode_selectors]
    if per_step_budget_sec is not None:
        payload["per_step_budget_sec"] = per_step_budget_sec

    with _client(base_url) as c:
        r = c.post("/api/eval/v2/start", json=payload)
        if 400 <= r.status_code < 500:
            return _structured_error(r)
        r.raise_for_status()
        return r.json()


def eval_status(base_url: str) -> dict[str, Any]:
    """Get status of the currently active run.

    Returns ``{"status": "none", "run": null}`` when no run is active,
    or the full run summary dict when one is.
    """
    with _client(base_url) as c:
        r = c.get("/api/eval/v2/status")
        r.raise_for_status()
        return r.json()


def eval_export(run_id: str) -> dict[str, Any]:
    """Export a run's full results by reading ``summary.json`` directly.

    The backend writes the authoritative summary to
    ``outputs/eval_runs/{run_id}/summary.json`` in the eval executor's
    ``finally`` block, then exits (in MCP-pool mode). Reading the file
    directly means this works without a live backend — historical runs
    from any past conversation are queryable as long as the file exists.

    Trade-off: while a run is still in flight the file is stale (only
    written at completion). For live progress, use ``eval_status``.
    """
    summary = EVAL_RUNS_DIR / run_id / "summary.json"
    if not summary.exists():
        return {
            "error": f"run '{run_id}' not found",
            "status_code": 404,
            "detail": f"expected file at {summary}",
        }
    try:
        return json.loads(summary.read_text())
    except (OSError, ValueError) as exc:
        return {
            "error": f"failed to read run '{run_id}'",
            "status_code": 500,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def eval_stop(base_url: str) -> dict[str, Any]:
    """Stop the active run. Returns ``{run_id, status}`` or ``{error}`` if idle."""
    with _client(base_url) as c:
        r = c.post("/api/eval/v2/stop")
        if 400 <= r.status_code < 500:
            return _structured_error(r)
        r.raise_for_status()
        return r.json()
