"""Eval API v2 — graph-native batch eval endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from ...agent_loop.eval_batch import BatchEvalRunner, EvalConfig, EvalRun, EvalStatus
from ...graph_def import GraphDefinition, validate_graph_connectivity
from ...state import ExecutionGuard, ExecutionMode, get_services
from .eval_storage import delete_run, list_runs, load_run, save_run

log = logging.getLogger("agentcanvas.eval-v2")
router = APIRouter()

# Graph resolution is delegated to WorkspaceComponentRegistry.resolve_graph_path()
# so the active-workspace overlay (Settings.active_workspace_dir) is
# honored — see _load_graph_by_name() below. There used to be a
# module-level GRAPHS_DIR hardwired to <repo>/workspace/graphs/; that
# bypassed Settings entirely and is now removed.

# In-memory active run (one at a time)
_current_run: EvalRun | None = None


# ── Request/Response models ──


class StartEvalV2Request(BaseModel):
    graph_name: str
    # Legacy convenience fields — flat shorthand for the two most common
    # cascade keys. They get merged into ``selectors`` (in this order)
    # at request handling time. New callers needing more cascade fields
    # (e.g. SIMPLER's ``task_id``, LIBERO's ``task_suite``) should
    # populate ``selectors`` directly, optionally alongside these.
    dataset: str = ""
    split: str = "val_unseen"
    # Generic cascade — dict insertion order = order in which the eval
    # batch pushes fields through the env panel before each episode.
    # Each env nodeset's env panel declares its own cascade
    # (``HabitatEnvPanel``: dataset → split → episode_index;
    # ``SimplerEnvPanel``: split → task_id → episode_index;
    # ``HMEQAEnvPanel``: split → episode_index). Push only the fields
    # you need; ``episode_index`` is always pushed last by the runner
    # and must NOT appear here.
    selectors: dict[str, Any] = {}
    episode_count: int = -1
    # Per-episode iteration cap. When None, the resolver chain in
    # BatchEvalRunner picks env-supplied → graph → DEFAULT_STEP_BUDGET.
    # Setting it explicitly forces this value for every episode.
    step_budget: int | None = None
    start_episode_index: int = 0  # ADR-023: start from specific episode
    worker_count: int = 1  # ADR-028: parallel env subprocesses (1 = sequential)
    per_step_budget_sec: float | None = None  # ADR-028: per-step timeout; None = nodeset default
    # Optional explicit episode list. When set, overrides start/count and
    # dispatches workers across exactly these indices (in order). Used for
    # random sampling across the dataset.
    episode_indices: list[int] | None = None
    # Optional per-episode selector overrides, parallel to the resolved
    # index list. Each entry is merged on top of run-level ``selectors``
    # before the cascade push for that episode. Use for cross-task sweeps
    # in a single run (e.g. SIMPLER 25 tasks x N episodes). Length must
    # match the resolved index list (``episode_indices`` length, or
    # ``episode_count``). Do NOT include ``episode_index`` — the runner
    # pushes that itself per episode.
    episode_selectors: list[dict[str, Any]] | None = None

    # Subprocess scheduler path. Default True (flipped 2026-05-07): runs
    # are spawned as separate Python processes by JobScheduler with
    # admission + queue. Set explicitly to False to fall back to the
    # legacy in-process asyncio.task path (or set
    # AGENTCANVAS_EVAL_SUBPROCESS=0 globally).
    via_subprocess: bool = True
    # Resource declaration used only when via_subprocess=True. Caller's
    # estimate of how much VRAM this run will marginally use on top of
    # already-loaded shared singletons (Prismatic etc. are NOT counted —
    # they're in the backend's baseline). 0 = CPU-only.
    marginal_vram_mb: int = 0
    exclusive_gpu: bool = False
    priority: str = "normal"  # high | normal | low

    # Optional active-workspace overlay. When set, the run is executed
    # with WorkspaceComponentRegistry scanning this dir on top of the frozen
    # workspace — same-named nodesets/nodes/policies/graphs override
    # frozen by name. Used by architect skills to run a per-iter
    # mutation set without writing to <repo>/workspace/. Pass an
    # absolute path. Default None = run against frozen workspace only.
    active_workspace_dir: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_max_steps(cls, data: Any) -> Any:
        # Backward compat for clients still posting ``max_steps``.
        # Translate it onto ``step_budget`` so callers do not need to
        # update immediately.
        if isinstance(data, dict) and "max_steps" in data:
            data = dict(data)
            legacy = data.pop("max_steps")
            data.setdefault("step_budget", legacy)
        return data


class IntrospectRequest(BaseModel):
    graph_name: str
    # Optional active-workspace overlay path. When set, the graph is
    # loaded from <active>/graphs/{name}.json before falling through to
    # frozen. Used by architect skills introspecting an iter's modified
    # graph without writing to frozen workspace.
    active_workspace_dir: str | None = None


# ── Helpers ──


def _load_graph_by_name(
    graph_name: str,
    active_workspace_dir: str | None = None,
) -> GraphDefinition:
    """Load a graph by name — active-workspace overlay first, then frozen.

    Resolution checks (in order):
        1. ``active_workspace_dir/graphs/{name}.json`` if the kwarg is set
           (per-request overlay from /api/eval/v2/start payload).
        2. The parent registry's ``resolve_graph_path()`` — honors the
           backend's startup-configured ``Settings.active_workspace_dir``
           (env var ``ACTIVE_WORKSPACE_DIR``).
        3. Frozen ``<repo>/workspace/graphs/{name}.json`` (via the same
           registry resolver, which falls through to frozen).
    """
    path: Path | None = None
    if active_workspace_dir:
        candidate = Path(active_workspace_dir) / "graphs" / f"{graph_name}.json"
        if candidate.exists():
            path = candidate
    if path is None:
        registry = get_services().workspace_component_registry
        path = registry.resolve_graph_path(graph_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{graph_name}' not found")
    try:
        import json

        raw = json.loads(path.read_text())
        graph = GraphDefinition.from_dict(raw)
        validate_graph_connectivity(graph)
        return graph
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid graph: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load graph: {exc}") from exc


# ── Endpoints ──


@router.post("/start")
async def start_eval_v2(req: StartEvalV2Request):
    """Start a batch eval run for a named graph."""
    # Validate active workspace dir up front, before claiming any resources.
    if req.active_workspace_dir:
        active_path = Path(req.active_workspace_dir)
        if not active_path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"active_workspace_dir must be an absolute path; got "
                    f"{req.active_workspace_dir!r}"
                ),
            )
        if not active_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"active_workspace_dir does not exist: {req.active_workspace_dir}",
            )
    graph = _load_graph_by_name(req.graph_name, req.active_workspace_dir)

    # Split nodeset-load by path. Subprocess eval (the default) owns
    # replicated / env nodesets in the eval subprocess, so the parent
    # backend only loads shared singletons (Prismatic VLM, etc.) for
    # the URL handoff — no more spawning env_habitat in the parent just
    # to kill it 2 minutes later. Canvas-Play / legacy in-process eval
    # still need everything loaded in-process, so they fall through to
    # ensure_nodesets_for_graph below.
    registry = get_services().workspace_component_registry
    use_subprocess = (
        req.via_subprocess and os.environ.get("AGENTCANVAS_EVAL_SUBPROCESS", "1") != "0"
    )
    if use_subprocess:
        load_result = await registry.ensure_shared_nodesets_for_graph(graph)
    else:
        load_result = await registry.ensure_nodesets_for_graph(graph, worker_count=req.worker_count)
    if load_result["loaded"]:
        log.info("Auto-loaded nodesets for eval: %s", load_result["loaded"])
    if load_result["failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load required nodesets: {load_result['failed']}",
        )

    # Whole-graph wire-type check now that nodeset proxy classes are in
    # NODE_HANDLERS (the pre-load validate_graph_connectivity saw builtins
    # only). Warn-only for now — surfaces nodeset env↔method shape mismatches
    # without aborting the eval, pending a clean sweep before promotion to a
    # hard 400 (ADR-027 staged enforcement).
    from ...graph_def import validate_edge_wire_types

    wire_errs = validate_edge_wire_types(graph)
    if wire_errs:
        log.warning(
            "wire-type mismatches (post-nodeset-load, warn-only):\n  - %s",
            "\n  - ".join(wire_errs),
        )

    # ADR-028 PB-2: resolve the env nodeset name from the graph so the
    # worker pool can populate tagged env panel + URL overrides per worker
    # (the StartEvalV2Request doesn't carry env_nodeset; the user only
    # picks a graph). First env-nodeset (one with non-empty eval metadata)
    # wins — same rule as /introspect.
    env_nodeset = ""
    for ns_name in registry.detect_env_nodesets_for_graph(graph):
        metadata = await registry.get_eval_metadata_for_nodeset(ns_name)
        if metadata:
            env_nodeset = ns_name
            break

    # Validate per-episode selectors length against the resolved index
    # list before claiming the execution slot — a bad request must not
    # tie up the single eval slot.
    if req.episode_selectors is not None:
        if req.episode_indices is not None:
            expected_len = len(req.episode_indices)
        elif req.episode_count > 0:
            expected_len = req.episode_count
        else:
            expected_len = 10  # matches BatchEvalRunner default fallback
        if len(req.episode_selectors) != expected_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"episode_selectors length {len(req.episode_selectors)} "
                    f"does not match resolved episode count {expected_len}"
                ),
            )
        for i, entry in enumerate(req.episode_selectors):
            if "episode_index" in entry:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"episode_selectors[{i}] must not contain 'episode_index'; "
                        f"the runner pushes it per episode"
                    ),
                )

    # Normalize episode_count when the caller supplied an explicit list.
    # Without this, requests with `episode_indices`/`episode_selectors` leave
    # the request-level sentinel `-1` in the saved config — confusing for any
    # consumer that reads `config.episode_count` (the authoritative total is
    # `EvalRun.total_episodes`). Compute the effective count once and pass it
    # down to both the subprocess spec and the in-process EvalConfig.
    if req.episode_indices is not None:
        effective_episode_count = len(req.episode_indices)
    elif req.episode_selectors is not None:
        effective_episode_count = len(req.episode_selectors)
    else:
        effective_episode_count = req.episode_count

    # Merge legacy (dataset, split) into the canonical cascade dict.
    # Insertion order = cascade order: legacy fields go first (dataset →
    # split, the historical default), then the explicit ``selectors``
    # dict (which can append env panel-specific fields like SIMPLER's
    # ``task_id`` or LIBERO's ``task_suite``, or override the legacy
    # values). Empty legacy strings are skipped so they don't pollute
    # the cascade for envs that don't declare them.
    selectors: dict[str, Any] = {}
    if req.dataset:
        selectors["dataset"] = req.dataset
    if req.split:
        selectors["split"] = req.split
    for k, v in (req.selectors or {}).items():
        selectors[k] = v

    # ── Subprocess scheduler path ──
    if use_subprocess:
        scheduler = get_services().job_scheduler
        if scheduler is None:
            raise HTTPException(status_code=503, detail="JobScheduler not initialized")

        # Compute shared_urls: nodesets the graph needs that are currently
        # loaded as shared singletons in this backend's registry.
        # Also include ALL currently-loaded shared singletons, since some
        # nodesets (e.g. model_detany3d for ToolEQA) are accessed via
        # ``ctx._executor.get_server_url(...)`` from inside method-side
        # nodes rather than as canvas-visible nodes.
        shared_urls: dict[str, str] = {}
        needed: set[str] = set()
        for node in graph.nodes:
            if "__" in node.type:
                needed.add(node.type.split("__")[0])
        for ns_name, _ns in registry._live_nodesets.items():
            if registry._get_parallelism(ns_name) == "shared":
                needed.add(ns_name)
        for ns_name in needed:
            if registry._get_parallelism(ns_name) != "shared":
                continue
            url = registry.get_server_url(ns_name)
            if url:
                shared_urls[ns_name] = url

        # Everything ensure_shared_nodesets_for_graph freshly loaded above
        # is — by construction — a shared singleton this request brought
        # up. JobScheduler refcounts these and auto-unloads on the last
        # job's reap. Pre-existing (canvas-Play-loaded) ones are in
        # already_loaded, not loaded, so they're never auto-unloaded.
        shared_loaded_by_us: list[str] = list(load_result.get("loaded", []))

        spec = {
            "eval": {
                "graph_name": req.graph_name,
                "selectors": selectors,
                "dataset": req.dataset,
                "split": req.split,
                "episode_count": effective_episode_count,
                "step_budget": req.step_budget,
                "start_episode_index": req.start_episode_index,
                "worker_count": req.worker_count,
                "per_step_budget_sec": req.per_step_budget_sec,
                "episode_indices": req.episode_indices,
                "episode_selectors": req.episode_selectors,
            },
            "scheduling": {
                "marginal_vram_mb": req.marginal_vram_mb,
                "exclusive_gpu": req.exclusive_gpu,
                "priority": req.priority,
            },
            "graph": graph.to_dict(),
            "_shared_urls": shared_urls,
            # Shared singletons THIS request freshly loaded (subset of
            # _shared_urls keys). JobScheduler marks these eligible for
            # auto-unload once no remaining queued/running job references
            # them. Pre-existing (canvas-Play-loaded) singletons are NOT
            # in this list, so they're never auto-unloaded.
            "_shared_loaded_by_us": shared_loaded_by_us,
            # Active-workspace overlay path, threaded through to the
            # eval subprocess via ACTIVE_WORKSPACE_DIR env var in
            # JobScheduler._spawn(). None = run against frozen workspace.
            "active_workspace_dir": req.active_workspace_dir,
        }
        try:
            run_id = scheduler.submit(spec)
        except ValueError as exc:
            # P3 feasibility rejection (e.g. declaration exceeds the
            # machine's physical ceiling) — a client error, not a 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "status": "queued", "via_subprocess": True}

    # ── Legacy in-process path (default) ──
    global _current_run
    run_id = str(uuid.uuid4())[:8]
    if not ExecutionGuard.acquire(ExecutionMode.eval, run_id):
        current = ExecutionGuard.current()
        raise HTTPException(
            status_code=409,
            detail=f"Canvas is busy (mode={current['mode']}, holder={current['holder']})",
        )

    config = EvalConfig(
        graph_name=req.graph_name,
        env_nodeset=env_nodeset,
        selectors=selectors,
        dataset=req.dataset,
        split=req.split,
        episode_count=effective_episode_count,
        step_budget=req.step_budget,
        start_episode_index=req.start_episode_index,
        worker_count=req.worker_count,
        per_step_budget_sec=req.per_step_budget_sec,
        episode_indices=req.episode_indices,
        episode_selectors=req.episode_selectors,
    )
    run = EvalRun(
        run_id=run_id,
        config=config,
        status=EvalStatus.pending,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _current_run = run

    runner = BatchEvalRunner(run, graph)

    async def _run_and_persist() -> None:
        try:
            await runner.execute()
        finally:
            save_run(run)

    run.task = asyncio.create_task(_run_and_persist())

    return {"run_id": run_id, "status": "pending"}


@router.post("/stop")
async def stop_eval_v2():
    """Stop the current active eval run."""
    global _current_run
    run = _current_run
    if run is None or run.status not in (EvalStatus.pending, EvalStatus.running):
        raise HTTPException(status_code=404, detail="No active eval run")
    run.stop_event.set()
    return {"run_id": run.run_id, "status": "stopping"}


@router.get("/status")
async def eval_v2_status():
    """Current run status and progress."""
    run = _current_run
    if run is None:
        return {"status": "none", "run": None}
    return {"status": run.status.value, "run": run.to_summary()}


@router.get("/episodes")
async def eval_v2_episodes():
    """Episode results for the current run."""
    run = _current_run
    if run is None:
        return {"episodes": []}
    return {"episodes": [run.to_episode_summary(ep) for ep in run.episodes]}


@router.get("/runs")
async def list_past_runs():
    """List all past runs stored on disk."""
    return {"runs": list_runs()}


@router.get("/runs/{run_id}")
async def get_past_run(run_id: str):
    """Load a specific past run by ID."""
    data = load_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return data


@router.delete("/runs/{run_id}")
async def delete_past_run(run_id: str):
    """Delete a past run file."""
    deleted = delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"deleted": run_id}


@router.get("/queue")
async def eval_v2_queue():
    """Scheduler view: queued + running jobs across all sessions."""
    scheduler = get_services().job_scheduler
    if scheduler is None:
        return {"queued": [], "running": [], "usable_vram_mb": 0, "reserved_vram_mb": 0}
    return scheduler.list_active()


@router.post("/runs/{run_id}/cancel")
async def cancel_subprocess_run(run_id: str):
    """Cancel a subprocess-path run (queued → cancelled, running → cancelling).

    Note: legacy in-process runs are cancelled via /stop, not this endpoint.
    """
    scheduler = get_services().job_scheduler
    if scheduler is None:
        raise HTTPException(status_code=503, detail="JobScheduler not initialized")
    new_status = scheduler.cancel(run_id)
    if new_status == "unknown":
        raise HTTPException(status_code=404, detail=f"run {run_id} not in scheduler")
    return {"run_id": run_id, "status": new_status}


@router.get("/estimate")
async def estimate_eval_v2(graph_name: str, worker_count: int = 1):
    """Advisory resource estimate for a run (VRAM + RAM, one entry per
    resource under ``resources``).

    Read-only: resolves the graph and consults the calibration store —
    loads no nodesets, reserves nothing. Per resource, ``estimate_mb`` is
    null until every component the graph needs has calibration data (run
    it once to calibrate); ``uncalibrated`` lists the gaps. Top-level
    ``max_workers`` is the largest worker_count whose estimate fits the
    measured free of every measurable resource.
    """
    scheduler = get_services().job_scheduler
    if scheduler is None:
        raise HTTPException(status_code=503, detail="JobScheduler not initialized")
    graph = _load_graph_by_name(graph_name, None)
    return scheduler.estimate_run(graph_name, [n.type for n in graph.nodes], worker_count)


@router.post("/introspect")
async def introspect_graph(req: IntrospectRequest):
    """Introspect a graph: detect env nodesets, check loaded state, get metadata."""
    graph = _load_graph_by_name(req.graph_name, req.active_workspace_dir)
    state = get_services()
    registry = state.workspace_component_registry

    env_nodesets = registry.detect_env_nodesets_for_graph(graph)

    # Return the first env nodeset with eval metadata, matching frontend GraphIntrospection type
    if not env_nodesets:
        return {
            "graph_name": req.graph_name,
            "env_nodeset": None,
            "loaded": False,
            "metadata": None,
        }

    # Find the first nodeset that has eval metadata (is an env)
    for ns_name in env_nodesets:
        loaded = registry.is_nodeset_loaded(ns_name)
        metadata = await registry.get_eval_metadata_for_nodeset(ns_name)
        if metadata:  # has eval metadata — it's an env nodeset
            return {
                "graph_name": req.graph_name,
                "env_nodeset": ns_name,
                "loaded": loaded,
                "metadata": metadata,
            }

    # Found nodesets but none with eval metadata — return first with loaded status
    ns_name = env_nodesets[0]
    loaded = registry.is_nodeset_loaded(ns_name)
    return {
        "graph_name": req.graph_name,
        "env_nodeset": ns_name,
        "loaded": loaded,
        "metadata": None,
    }


@router.get("/export/{run_id}")
async def export_run(run_id: str):
    """Export a run as full JSON (includes episodes)."""
    # Check active run first
    run = _current_run
    if run is not None and run.run_id == run_id:
        return {
            "run_id": run.run_id,
            "config": {
                "graph_name": run.config.graph_name,
                "env_nodeset": run.config.env_nodeset,
                "selectors": dict(run.config.selectors),
                "episode_selectors": (
                    [dict(s) for s in run.config.episode_selectors]
                    if run.config.episode_selectors is not None
                    else None
                ),
                "split": run.config.split,
                "episode_count": run.config.episode_count,
                "step_budget": run.config.step_budget,
            },
            "status": run.status.value,
            "episodes": [run.to_episode_summary(ep) for ep in run.episodes],
            "aggregate_metrics": run.aggregate_metrics,
            "aggregate_by_task": dict(run.aggregate_by_task),
            "total_episodes": run.total_episodes,
            "created_at": run.created_at,
            "finished_at": run.finished_at,
            "elapsed_sec": run.elapsed_sec,
            "error": run.error,
        }

    # Fall back to persisted run
    data = load_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return data
