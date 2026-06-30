"""Navigate REST endpoints — canvas graph execution."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...agent_loop.loop_runner import get_loop_runner
from ...graph_def import GraphDefinition, validate_graph_connectivity
from ...state import ExecutionGuard, ExecutionMode, get_services

log = logging.getLogger("agentcanvas.navigate-api")
router = APIRouter()


# ── Request models ──


class RunRequest(BaseModel):
    """Single endpoint: receives graph definition, runs it via GraphExecutor."""

    loop_definition: dict
    execution_id: str | None = None
    step_delay_ms: int = 200


# ── Node schema discovery ──


@router.get("/policies")
async def list_policies():
    """List available neural policies for policyForward nodes."""
    return get_services().workspace_component_registry.list_policies()


# ── Loop execution ──


@router.post("/run")
async def run_loop(req: RunRequest):
    """Execute a graph definition from the canvas.

    Single endpoint: the frontend sends the complete GraphDefinition JSON
    and the backend executes it via GraphExecutor. All environment
    interaction happens through nodeset nodes wired on the canvas.
    """
    # Check exclusive execution guard
    if not ExecutionGuard.acquire(ExecutionMode.canvas, "canvas"):
        guard = ExecutionGuard.current()
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start canvas execution: {guard['mode']} is active (holder: {guard['holder']})",
        )

    runner = get_loop_runner()

    # Stop any running execution
    if runner._status in ("running", "paused"):
        await runner.stop()

    # Parse and validate the graph definition at the API boundary
    try:
        graph = GraphDefinition.from_dict(req.loop_definition)
        validate_graph_connectivity(graph)
    except ValueError as e:
        ExecutionGuard.release("canvas")
        raise HTTPException(status_code=400, detail=f"Invalid graph: {e}") from e
    except Exception as e:
        ExecutionGuard.release("canvas")
        raise HTTPException(status_code=400, detail=f"Invalid graph definition: {e}") from e

    # Auto-load any nodesets the graph needs but that aren't loaded (e.g. after
    # a cold start or uvicorn --reload wiped the registry). Idempotent.
    registry = get_services().workspace_component_registry
    try:
        await registry.ensure_nodesets_for_graph(graph)
    except Exception as e:
        ExecutionGuard.release("canvas")
        raise HTTPException(status_code=500, detail=f"Nodeset load failed: {e}") from e

    # Whole-graph wire-type check, now that nodeset proxy classes are in
    # NODE_HANDLERS (the pre-load validate_graph_connectivity only saw
    # builtins). Warn-only for now: it surfaces nodeset env↔method shape
    # mismatches without aborting the run, pending a clean sweep before this
    # is promoted to a hard 400 (ADR-027 staged enforcement).
    from ...graph_def import validate_edge_wire_types

    wire_errs = validate_edge_wire_types(graph)
    if wire_errs:
        log.warning(
            "wire-type mismatches (post-nodeset-load, warn-only):\n  - %s",
            "\n  - ".join(wire_errs),
        )

    # Fetch global hooks from component registry
    global_hooks = registry.get_global_hooks()

    async def _run_and_release():
        try:
            await runner.run(graph, req.step_delay_ms, global_hooks=global_hooks)
        finally:
            ExecutionGuard.release("canvas")

    # Run with validated GraphDefinition
    runner._execution_id = req.execution_id
    runner._stop_event.clear()
    runner._task = asyncio.create_task(_run_and_release())

    return {"ok": True, "execution_id": req.execution_id}


@router.post("/run/pause")
async def run_pause():
    """Pause the running loop."""
    runner = get_loop_runner()
    if runner._status == "running":
        await runner.pause()
    return {"ok": True}


@router.post("/run/stop")
async def run_stop():
    """Stop the running loop."""
    runner = get_loop_runner()
    await runner.stop()
    ExecutionGuard.release("canvas")
    return {"ok": True}


@router.get("/execution-mode")
async def get_execution_mode():
    """Get current execution mode (idle/canvas/eval)."""
    return ExecutionGuard.current()


@router.get("/run/status")
async def run_status():
    """Get loop runner status."""
    runner = get_loop_runner()
    return runner.get_status()


@router.get("/run/checkpoints")
async def run_checkpoints():
    """List available checkpoint steps."""
    runner = get_loop_runner()
    return {"steps": runner.get_checkpoints()}


@router.post("/run/restore/{step}")
async def run_restore(step: int):
    """Restore execution state to a previous checkpoint step."""
    runner = get_loop_runner()
    if runner._status not in ("paused", "done"):
        raise HTTPException(400, "Must pause or complete before restoring")
    ok = runner.restore_step(step)
    if not ok:
        raise HTTPException(404, f"No checkpoint at step {step}")
    return {"ok": True, "restored_to": step}
