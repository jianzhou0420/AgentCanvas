"""AgentCanvas Backend — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.canvas import env_panel, graphs
from .api.execution import eval as eval_api_v2
from .api.execution import internal_containers, internal_events, logs, run, websocket
from .api.platform import components, profiles, providers
from .api.platform import config as config_api
from .api.platform import errors as errors_api
from .api.platform import system as system_api
from .api.registry import snapshot as registry_snapshot
from .api.replay import router as replay_router
from .config import get_settings
from .errors import ErrorEnvelope, get_bus, install_log_bridge
from .models import WSMessage
from .services.job_scheduler import JobScheduler, reconcile_aborted_runs
from .state import (
    CANVAS_ORPHAN_GRACE_SEC,
    ExecutionGuard,
    ExecutionMode,
    broadcast,
    canvas_guard_orphan_decision,
    get_services,
    ws_client_count,
)


def _arm_pdeathsig() -> None:
    """Linux ``PR_SET_PDEATHSIG``: SIGTERM when our parent dies.

    Belt-and-suspenders pair with any spawn-side ``preexec_fn`` that
    arms PDEATHSIG (e.g. ``base_server._preexec_setsid_pdeathsig`` for
    auto_host children, or external spawners that wrap uvicorn). The
    spawn-side flag is cleared by ``conda run --no-capture-output``'s
    internal fork before exec'ing the python interpreter, so we re-arm
    here once uvicorn's process is the actual one running. Either
    layer dying triggers our SIGTERM. No-op on non-Linux.
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1,  # PR_SET_PDEATHSIG
            signal.SIGTERM,
            0,
            0,
            0,
        )
    except OSError:
        pass


_arm_pdeathsig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("agentcanvas")


async def _broadcast_envelope(env: ErrorEnvelope) -> None:
    """Bus subscriber — fan out every published envelope as an `error_event` WS frame."""
    await broadcast(WSMessage(type="error_event", data=env.to_dict()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # ── Error bus wiring (must happen before any subsystem logs) ──
    bus = get_bus()
    bus.attach_loop(asyncio.get_running_loop())
    bus.subscribe(_broadcast_envelope)
    install_log_bridge(level=logging.INFO)

    # Sync the loopback-proxy bypass flag from Settings into the framework
    # module (httpx call sites read the module-level flag, not Settings).
    from .server._loopback_proxy import set_ignore_loopback_proxy

    set_ignore_loopback_proxy(settings.ignore_loopback_proxy)

    log.info(f"AgentCanvas Backend starting on {settings.host}:{settings.port}")

    # Initialize in-memory state
    state = get_services()

    # ── Scan workspace for class-based components ──
    # All tools, envs, skills, agents, and policies come from workspace
    counts = state.workspace_component_registry.scan_all()
    log.info("workspace components: %s", counts)

    # Initialize nodesets (separate from scan — potentially slow)
    await state.workspace_component_registry.initialize_all()

    # ── JobScheduler (subprocess-path admission + queue) ──
    # Budget derivation (usable VRAM etc.) lives in JobScheduler.create so
    # all admission inputs are centralized in job_scheduler.py.
    import os
    from pathlib import Path

    # Slot backends (/host) point this at their own pool (e.g.
    # outputs/eval_runs_c) so the startup reconcile sweep and the queue
    # never touch the primary tree's runs — outputs/ is a shared symlink
    # across worktree slots. Default: the primary shared pool.
    eval_runs_dir = Path(
        os.environ.get("AGENTCANVAS_EVAL_RUNS_DIR")
        or Path(__file__).resolve().parents[3] / "outputs" / "eval_runs"
    )
    backend_url = f"http://{settings.host}:{settings.port}"
    state.job_scheduler = JobScheduler.create(eval_runs_dir, backend_url)
    state.job_scheduler.set_canvas_lock_callback(
        lambda: ExecutionGuard.current()["mode"] == ExecutionMode.canvas.value
    )
    # TODO #60: scheduler reaches into the registry at admit time to
    # spawn ephemeral auto_host children for overlay-redefined shared
    # nodesets, and at reap time to tear them down.
    state.job_scheduler.set_workspace_component_registry(state.workspace_component_registry)
    fixed = reconcile_aborted_runs(eval_runs_dir)
    log.info(
        "JobScheduler ready: usable_vram=%d MB, aborted_reconciled=%d",
        state.job_scheduler.usable_vram_mb,
        fixed,
    )

    # Background tick loop: eval admission/reap + orphaned-canvas-guard reap, every 1s.
    canvas_orphan_since: float | None = None

    async def _scheduler_loop() -> None:
        nonlocal canvas_orphan_since
        while True:
            try:
                await state.job_scheduler.tick()
            except Exception:
                log.exception("scheduler tick raised")
            # ── Reap an orphaned canvas ExecutionGuard ──
            # Leaks when the frontend closes a paused run without /run/stop;
            # the held lock then starves all eval admission. WS-liveness reaper.
            try:
                guard = ExecutionGuard.current()
                should_reap, canvas_orphan_since = canvas_guard_orphan_decision(
                    guard["mode"], ws_client_count(), canvas_orphan_since, time.monotonic()
                )
                if should_reap:
                    from .agent_loop.loop_runner import get_loop_runner

                    runner = get_loop_runner()
                    log.warning(
                        "Reaping orphaned canvas ExecutionGuard (holder=%s, runner=%s): "
                        "no WebSocket clients for >%.0fs — frontend closed without /run/stop.",
                        guard["holder"],
                        runner.get_status().get("status"),
                        CANVAS_ORPHAN_GRACE_SEC,
                    )
                    await runner.stop()
                    ExecutionGuard.release("canvas")
            except Exception:
                log.exception("canvas-guard reaper raised")
            await asyncio.sleep(1.0)

    sched_task = asyncio.create_task(_scheduler_loop())

    # ── Resource sampler (System Log) — 1 Hz machine snapshot to disk ──
    from .services.resource_sampler import ResourceSampler, set_sampler

    system_dir = Path(__file__).resolve().parents[3] / "outputs" / "system"
    resource_sampler = ResourceSampler(out_dir=system_dir, ws_clients_fn=ws_client_count)
    set_sampler(resource_sampler)
    resource_sampler.start()

    # ── Graph-directory watcher (push graphs_changed on external disk edits) ──
    from .api.canvas.graphs import GRAPH_NODES_DIR, GRAPHS_DIR
    from .services.graph_watcher import run_graph_watch_loop

    watch_task = asyncio.create_task(run_graph_watch_loop((GRAPHS_DIR, GRAPH_NODES_DIR)))

    # ── Nodeset-source watcher (hot-reload nodeset .py edits, no restart) ──
    from .services.nodeset_watcher import run_nodeset_watch_loop

    nodeset_watch_task = asyncio.create_task(
        run_nodeset_watch_loop(state.workspace_component_registry)
    )

    yield

    nodeset_watch_task.cancel()
    watch_task.cancel()
    sched_task.cancel()
    await resource_sampler.stop()
    if state.job_scheduler is not None:
        await state.job_scheduler.shutdown()

    # Shutdown all nodesets and envs
    await state.workspace_component_registry.shutdown_all()
    log.info("AgentCanvas Backend shutdown complete")


app = FastAPI(
    title="AgentCanvas",
    description="Web dashboard backend for AgentCanvas navigation visualization",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
# Canvas — graph editing + nodeset env panels
app.include_router(graphs.router, prefix="/api/graphs", tags=["graphs"])
app.include_router(env_panel.router, prefix="/api/env-panels", tags=["env-panels"])
# Execution — run, eval, logs, websocket
app.include_router(run.router, prefix="/api/navigate", tags=["navigate"])
# Internal reverse-channel: server-mode nodes read/write executor-home state
# containers (cross-nodeset container-access prototype, face B). Not public.
app.include_router(internal_containers.router, prefix="/api/internal", tags=["internal"])
# Internal reverse-channel: server-mode subprocesses push log/error events here;
# republished on the ErrorBus → canvas WebSocket (Move 3, #54). Not public.
app.include_router(internal_events.router, prefix="/api/internal", tags=["internal"])
app.include_router(eval_api_v2.router, prefix="/api/eval/v2", tags=["eval-v2"])
app.include_router(logs.router, prefix="/api/logs", tags=["logs"])
app.include_router(replay_router, prefix="/api/replay", tags=["replay"])
app.include_router(websocket.router, tags=["websocket"])
# Platform — config, components, profiles
app.include_router(config_api.router, prefix="/api/config", tags=["config"])
app.include_router(components.router, prefix="/api/components", tags=["components"])
app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
app.include_router(providers.router, prefix="/api/providers", tags=["providers"])
app.include_router(errors_api.router, prefix="/api/errors", tags=["errors"])
app.include_router(registry_snapshot.router, prefix="/api/registry", tags=["registry"])
app.include_router(system_api.router, prefix="/api/system", tags=["system"])


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so unhandled API exceptions surface in the Report tab.

    HTTPException is re-raised by FastAPI before reaching here; this only
    fires for bugs (e.g. uncaught KeyError in a route handler).
    """
    bus = get_bus()
    env = bus.from_exception(
        exc,
        source="api",
        code="API_UNHANDLED",
        scope={"endpoint": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=500,
        content={"error": env.to_dict()},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
