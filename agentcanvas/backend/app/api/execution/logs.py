"""REST API for execution logs — query per-node I/O from canvas and eval runs.

Endpoints:
    GET /api/logs                              — list recent executions
    GET /api/logs/{execution_id}               — log entries (paginated, filterable)
    GET /api/logs/{execution_id}/node/{node_id} — one node's history across steps
    GET /api/logs/{execution_id}/summary       — execution summary
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter()

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _outputs_dir() -> str:
    """Return outputs base path (repo_root/outputs)."""
    # execution/logs.py -> execution/ -> api/ -> app/ -> backend/ -> agentcanvas/ -> vlnworkspace/
    repo_root = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
        )
    )
    return os.path.join(repo_root, "outputs")


def _sanitize_id(value: str) -> str:
    """Sanitize execution_id or filename to prevent path traversal."""
    base = os.path.basename(value)
    if not base or base != value or ".." in value:
        return ""
    return base


def _eval_episode_dir(execution_id: str) -> str | None:
    """Resolve an eval per-episode execution_id to its episode dir.

    Eval episode execution_ids have the shape ``{run_id}_ep{NNNN}`` where
    ``run_id`` is a second-precision timestamp. Returns the absolute
    episode dir (may not exist), or None if the id isn't an eval-episode id.
    """
    run_id, sep, ep = execution_id.rpartition("_ep")
    if not sep or not ep.isdigit():
        return None
    return os.path.join(_outputs_dir(), "eval_runs", run_id, "episodes", f"ep{ep}")


def _find_jsonl(execution_id: str) -> str | None:
    """Locate the log.jsonl file for a given execution_id.

    Canvas runs: ``runs/{execution_id}/log.jsonl``.
    Eval episodes: ``eval_runs/{run_id}/episodes/ep{NNNN}/log.jsonl``,
    addressed by the composite ``{run_id}_ep{NNNN}`` execution_id.
    """
    base = _outputs_dir()
    # Canvas runs
    canvas_path = os.path.join(base, "runs", execution_id, "log.jsonl")
    if os.path.isfile(canvas_path):
        return canvas_path
    # Eval runs — per-episode log under episodes/ep{NNNN}/
    ep_dir = _eval_episode_dir(execution_id)
    if ep_dir:
        eval_path = os.path.join(ep_dir, "log.jsonl")
        if os.path.isfile(eval_path):
            return eval_path
    return None


def _read_jsonl(path: str) -> list[dict]:
    """Read all entries from a JSONL file."""
    entries: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return entries


@router.get("")
async def list_executions() -> dict[str, Any]:
    """List recent executions that have log files."""
    base = _outputs_dir()
    executions: list[dict] = []

    # Scan canvas runs
    runs_dir = os.path.join(base, "runs")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir), reverse=True)[:50]:
            jsonl = os.path.join(runs_dir, name, "log.jsonl")
            if os.path.isfile(jsonl):
                stat = os.stat(jsonl)
                executions.append(
                    {
                        "execution_id": name,
                        "source": "canvas",
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )

    # Scan eval runs — one execution entry per episode dir, addressed by
    # the composite ``{run_id}_ep{NNNN}`` id.
    eval_dir = os.path.join(base, "eval_runs")
    if os.path.isdir(eval_dir):
        for run_name in sorted(os.listdir(eval_dir), reverse=True)[:50]:
            episodes_dir = os.path.join(eval_dir, run_name, "episodes")
            if not os.path.isdir(episodes_dir):
                continue
            for ep_name in sorted(os.listdir(episodes_dir)):
                jsonl = os.path.join(episodes_dir, ep_name, "log.jsonl")
                if not os.path.isfile(jsonl):
                    continue
                stat = os.stat(jsonl)
                ep_idx = ep_name[2:] if ep_name.startswith("ep") else ep_name
                executions.append(
                    {
                        "execution_id": f"{run_name}_ep{ep_idx}",
                        "source": "eval",
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
            if len(executions) > 200:
                break

    executions.sort(key=lambda x: x["modified"], reverse=True)
    return {"executions": executions[:100]}


@router.get("/{execution_id}")
async def get_log_entries(
    execution_id: str,
    node_id: str | None = Query(None),
    node_type: str | None = Query(None),
    step: int | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Get log entries for an execution, with optional filters."""
    # Try in-memory logger first (for active executions)
    from ...agent_loop.loop_runner import get_loop_runner

    runner = get_loop_runner()
    if (
        runner._executor
        and runner._executor._logger
        and runner._executor._logger.execution_id == execution_id
    ):
        entries = runner._executor._logger.get_entries(
            node_id=node_id,
            node_type=node_type,
            step=step,
            limit=limit,
            offset=offset,
        )
        return {
            "execution_id": execution_id,
            "entries": [e.dict() for e in entries],
            "total": len(entries),
            "source": "memory",
        }

    # Fall back to JSONL file
    path = _find_jsonl(execution_id)
    if not path:
        return {"execution_id": execution_id, "entries": [], "total": 0, "source": "not_found"}

    all_entries = _read_jsonl(path)

    # Apply filters
    if node_id:
        all_entries = [e for e in all_entries if e.get("node_id") == node_id]
    if node_type:
        all_entries = [e for e in all_entries if e.get("node_type") == node_type]
    if step is not None:
        all_entries = [e for e in all_entries if e.get("step") == step]

    paginated = all_entries[offset : offset + limit]
    return {
        "execution_id": execution_id,
        "entries": paginated,
        "total": len(all_entries),
        "source": "file",
    }


@router.get("/{execution_id}/node/{node_id}")
async def get_node_history(
    execution_id: str,
    node_id: str,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Get output history for a specific node across all steps."""
    # Delegate to the main endpoint with node_id filter
    result = await get_log_entries(
        execution_id=execution_id,
        node_id=node_id,
        node_type=None,
        step=None,
        limit=limit,
        offset=0,
    )
    return result


@router.get("/{execution_id}/summary")
async def get_execution_summary(execution_id: str) -> dict[str, Any]:
    """Get aggregate stats for an execution."""
    # Try in-memory logger first
    from ...agent_loop.loop_runner import get_loop_runner

    runner = get_loop_runner()
    if (
        runner._executor
        and runner._executor._logger
        and runner._executor._logger.execution_id == execution_id
    ):
        summary = runner._executor._logger.get_summary()
        return summary.dict()

    # Fall back to JSONL
    path = _find_jsonl(execution_id)
    if not path:
        return {"execution_id": execution_id, "error": "not_found"}

    entries = _read_jsonl(path)
    if not entries:
        return {"execution_id": execution_id, "total_firings": 0}

    node_types = set()
    error_count = 0
    max_step = 0
    for e in entries:
        node_types.add(e.get("node_type", ""))
        if e.get("error"):
            error_count += 1
        s = e.get("step", 0)
        if s > max_step:
            max_step = s

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


@router.get("/{execution_id}/graph")
async def get_execution_graph(execution_id: str) -> dict[str, Any]:
    """Return the graph definition that was executed (saved at run start)."""
    safe_id = _sanitize_id(execution_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid execution_id")

    base = _outputs_dir()
    # Canvas: graph.json sits in the run dir. Eval: graph.json is
    # run-level (shared by every episode), so an eval-episode id
    # ``{run_id}_ep{NNNN}`` resolves to ``eval_runs/{run_id}/graph.json``.
    candidates = [os.path.join(base, "runs", safe_id, "graph.json")]
    run_id, sep, ep = safe_id.rpartition("_ep")
    if sep and ep.isdigit():
        candidates.append(os.path.join(base, "eval_runs", run_id, "graph.json"))

    for graph_path in candidates:
        if os.path.isfile(graph_path):
            try:
                with open(graph_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise HTTPException(status_code=500, detail=f"Failed to read graph: {e}") from e

    raise HTTPException(status_code=404, detail="Graph not saved for this execution")


@router.get("/{execution_id}/assets/{filename}")
async def get_asset(execution_id: str, filename: str) -> FileResponse:
    """Serve an asset file (image) from a log run's assets/ folder."""
    safe_id = _sanitize_id(execution_id)
    safe_name = _sanitize_id(filename)
    if not safe_id or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid path")

    path = _find_asset(safe_id, safe_name)
    if not path:
        raise HTTPException(status_code=404, detail="Asset not found")

    ext = os.path.splitext(safe_name)[1].lower()
    media_type = MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


def _find_asset(execution_id: str, filename: str) -> str | None:
    """Locate an asset file for a given execution_id.

    Canvas: ``runs/{execution_id}/assets/{filename}``.
    Eval episodes: ``eval_runs/{run_id}/episodes/ep{NNNN}/assets/{filename}``.
    """
    base = _outputs_dir()
    canvas_path = os.path.join(base, "runs", execution_id, "assets", filename)
    if os.path.isfile(canvas_path):
        return canvas_path
    ep_dir = _eval_episode_dir(execution_id)
    if ep_dir:
        eval_asset = os.path.join(ep_dir, "assets", filename)
        if os.path.isfile(eval_asset):
            return eval_asset
    return None
