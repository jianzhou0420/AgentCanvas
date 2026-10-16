"""REST API for execution logs — query per-node I/O from canvas and eval runs.

Endpoints:
    GET /api/logs                              — list recent executions
    GET /api/logs/{execution_id}               — log entries (paginated, filterable)
    GET /api/logs/{execution_id}/node/{node_id} — one node's history across steps
    GET /api/logs/{execution_id}/summary       — execution summary
    GET /api/logs/{execution_id}/graph         — the graph that ran
    GET /api/logs/{execution_id}/assets/{file} — an image asset

A live execution is answered from its in-memory ``ExecutionLogger``; every
other answer comes from the ``LogStore`` (``app/logging/store.py``), which
owns the on-disk layout and the JSONL reader.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ...logging.store import LogStore

router = APIRouter()

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _store() -> LogStore:
    return LogStore.default()


def _sanitize_id(value: str) -> str:
    """Sanitize execution_id or filename to prevent path traversal."""
    base = os.path.basename(value)
    if not base or base != value or ".." in value:
        return ""
    return base


def _live_logger(execution_id: str) -> Any | None:
    """The in-memory logger of the canvas execution now running, if it is this one."""
    from ...agent_loop.loop_runner import get_loop_runner

    runner = get_loop_runner()
    logger = runner._executor._logger if runner._executor else None
    if logger is not None and logger.execution_id == execution_id:
        return logger
    return None


@router.get("")
async def list_executions() -> dict[str, Any]:
    """List recent executions that have log files."""
    return {"executions": _store().list_executions(limit=100)}


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
    logger = _live_logger(execution_id)
    if logger is not None:
        entries = logger.get_entries(
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

    store = _store()
    path = store.log_path(execution_id)
    if path is None:
        return {"execution_id": execution_id, "entries": [], "total": 0, "source": "not_found"}

    all_entries = store.read_entries(path)
    if node_id:
        all_entries = [e for e in all_entries if e.get("node_id") == node_id]
    if node_type:
        all_entries = [e for e in all_entries if e.get("node_type") == node_type]
    if step is not None:
        all_entries = [e for e in all_entries if e.get("step") == step]

    return {
        "execution_id": execution_id,
        "entries": all_entries[offset : offset + limit],
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
    return await get_log_entries(
        execution_id=execution_id,
        node_id=node_id,
        node_type=None,
        step=None,
        limit=limit,
        offset=0,
    )


@router.get("/{execution_id}/summary")
async def get_execution_summary(execution_id: str) -> dict[str, Any]:
    """Get aggregate stats for an execution."""
    logger = _live_logger(execution_id)
    if logger is not None:
        return logger.get_summary().dict()

    store = _store()
    path = store.log_path(execution_id)
    if path is None:
        return {"execution_id": execution_id, "error": "not_found"}
    return store.summarize(execution_id, store.read_entries(path))


@router.get("/{execution_id}/graph")
async def get_execution_graph(execution_id: str) -> dict[str, Any]:
    """Return the graph definition that was executed (saved at run start)."""
    safe_id = _sanitize_id(execution_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid execution_id")
    try:
        graph = _store().read_graph(safe_id)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read graph: {e}") from e
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not saved for this execution")
    return graph


@router.get("/{execution_id}/assets/{filename}")
async def get_asset(execution_id: str, filename: str) -> FileResponse:
    """Serve an asset file (image) from a log run's assets/ folder."""
    safe_id = _sanitize_id(execution_id)
    safe_name = _sanitize_id(filename)
    if not safe_id or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid path")

    path = _store().asset_path(safe_id, safe_name)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset not found")

    ext = os.path.splitext(safe_name)[1].lower()
    media_type = MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(str(path), media_type=media_type)
