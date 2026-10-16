"""Subprocess entry point for one batch eval run.

Invoked by JobScheduler as::

    python -m app.eval_subprocess_main \
        --run-dir   /abs/path/outputs/eval_runs/{run_id} \
        --backend-url http://127.0.0.1:8765

Contract:

- ``{run_dir}/spec.json`` is written by the backend before spawn. It
  contains the full JobSpec (eval block + scheduling block) and the
  graph definition, so this subprocess does not need to talk to the
  backend filesystem.
- ``{run_dir}/shared_urls.json`` is also written by the backend before
  spawn. It maps shared nodeset name → auto_host URL. The subprocess
  uses these URLs as remote-mode overrides instead of loading those
  nodesets locally (the whole point of having one backend: VLMs etc.
  load once and serve every run).
- The subprocess ``ensure``s only env-tagged nodesets locally (those
  spawn their own auto_host children, owned by this subprocess via
  PR_SET_PDEATHSIG so they die with us).
- Progress is written to ``{run_dir}/summary.json`` (atomic via
  tempfile + rename) on every episode boundary and on terminal exit.
- On clean exit (success, error, cancelled) the subprocess touches
  ``{run_dir}/_DONE``. Absence of ``_DONE`` after the PID is reaped =
  ``aborted`` per Q1 (in-flight loss accepted).

The subprocess never imports backend FastAPI / scheduler code — it is
strictly a runner. All scheduling decisions stay in the parent.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import httpx

from .services.run_state_io import (
    atomic_write_json,
    initial_running_summary,
    read_shared_urls,
    read_spec,
    read_summary,
    touch_done,
)

log = logging.getLogger("agentcanvas.eval-runner")


def _set_pdeathsig() -> None:
    """Linux-only: parent dies → kernel sends SIGTERM to this process.

    Called via ``preexec_fn``-equivalent at the very start of main so
    any child subprocess we spawn (env workers via ensure_nodesets_for_graph)
    inherits a process group that dies with us, and we ourselves die
    with the parent backend if it crashes.
    """
    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        log.warning("PR_SET_PDEATHSIG unavailable; orphan-on-parent-crash possible")


def _apply_vram_cap() -> None:
    """P3 quota teeth: cap this process's torch allocator at the admission
    charge (``AGENTCANVAS_VRAM_CAP_MB``, set by JobScheduler._spawn), so a
    run that blows past its estimate OOMs itself instead of its neighbors.

    Best-effort by design: only torch allocations in THIS process are
    capped — EGL render contexts and child auto_host servers (separate
    processes with their own allocators) are not. The measured admission
    gate remains the primary defense; this is the backstop.
    """
    cap_mb = int(os.environ.get("AGENTCANVAS_VRAM_CAP_MB", "0") or 0)
    if cap_mb <= 0:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = min(1.0, cap_mb * 1024 * 1024 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        log.info("vram cap: %d MB (fraction %.3f) via AGENTCANVAS_VRAM_CAP_MB", cap_mb, fraction)
    except Exception:
        log.warning("vram cap requested (%d MB) but could not be applied", cap_mb)


def _write_initial_summary(run_dir: Path, spec: dict) -> None:
    """Mark status='running' the moment the subprocess is up."""
    atomic_write_json(
        run_dir / "summary.json",
        initial_running_summary(
            run_id=spec["run_id"],
            eval_block=spec.get("eval", {}),
            created_at=spec.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ),
    )


async def _fetch_shared_url_overrides(backend_url: str, names: list[str]) -> dict[str, str]:
    """Ask backend for current auto_host URLs of shared nodesets.

    Returns ``{nodeset_name: url}``. Names not currently loaded by the
    backend raise — the backend is supposed to pre-load shared singletons
    before scheduler dispatches any job that needs them.
    """
    if not names:
        return {}
    async with httpx.AsyncClient(base_url=backend_url, timeout=10) as client:
        r = await client.post("/api/registry/snapshot", json={"names": names})
        r.raise_for_status()
        snap = r.json()
    missing = [n for n in names if n not in snap.get("urls", {})]
    if missing:
        raise RuntimeError(f"backend does not have these shared nodesets loaded: {missing}")
    return snap["urls"]


async def _build_local_registry_for_run(spec: dict, shared_urls: dict[str, str]) -> tuple:
    """Build a WorkspaceComponentRegistry inside this subprocess.

    Returns ``(graph, env_nodeset_name)`` — graph is GraphDefinition,
    env_nodeset is the one resolved env nodeset string that worker
    pool will tag-spawn.

    Heavy imports happen here (not at module top) so spec parsing /
    spawn validation is cheap before paying the import tax.
    """
    from .agent_loop.graph_run import prepare_graph
    from .components.registry import WorkspaceComponentRegistry  # heavy
    from .graph_def import GraphDefinition
    from .state import get_services

    graph = GraphDefinition.from_dict(spec["graph"])

    registry: WorkspaceComponentRegistry = get_services().workspace_component_registry

    # Subprocess starts with a fresh ProcessServices (in-process singleton not
    # shared across PIDs). The parent backend ran scan_all in lifespan;
    # we have to redo it here so workspace nodesets are discovered before
    # ensure_nodesets_for_graph can load them. Otherwise nodesets fall
    # into the silent "unknown" bucket and the run "completes" with 0
    # steps because no handlers are registered for the graph's node types.
    registry.scan_all()

    for name, url in shared_urls.items():
        await registry.register_remote_nodeset(name, url)

    # Strict: this process exists to run exactly this graph, so a nodeset
    # that fails to load or is unknown is fatal, not a no-op.
    worker_count = int(spec["eval"].get("worker_count", 1))
    prepared = await prepare_graph(
        registry, graph, worker_count=worker_count, on_failed="raise", on_unknown="raise"
    )
    return prepared.graph, prepared.env_nodeset


async def _shutdown_local_registry() -> None:
    """Kill every auto_host env-server subprocess this run spawned.

    Without this, a job that completes (or fails) leaks its env servers:
    they survive because the ``/bin/sh`` wrapper breaks the
    ``PR_SET_PDEATHSIG`` chain, and ``JobScheduler._reap`` only reaps the
    eval_runner subprocess itself, not the env servers it spawned. The
    registry holds the only references to those ``BaseServer`` handles, so
    the kill has to happen here, in-process, before this subprocess exits.
    Idempotent: ``BaseServer.stop()`` is a no-op once already stopped.
    """
    from .state import get_services

    try:
        await get_services().workspace_component_registry.shutdown_all()
    except Exception:
        log.exception("eval_runner: registry shutdown failed; env servers may leak")


async def _run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    spec_path = run_dir / "spec.json"
    if not spec_path.exists():
        sys.stderr.write(f"missing spec: {spec_path}\n")
        return 2

    spec = read_spec(run_dir)
    if spec is None:
        sys.stderr.write(f"missing spec: {spec_path}\n")
        return 2
    _write_initial_summary(run_dir, spec)
    shared_urls = read_shared_urls(run_dir)

    try:
        graph, env_nodeset = await _build_local_registry_for_run(spec, shared_urls)
    except Exception:
        err = traceback.format_exc()
        summary = read_summary(run_dir) or {}
        summary["status"] = "error"
        summary["error"] = err
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(run_dir / "summary.json", summary)
        await _shutdown_local_registry()
        touch_done(run_dir)
        return 1

    # Build EvalRun + EvalConfig from spec, run BatchEvalRunner.
    from .agent_loop.eval_batch import (
        BatchEvalRunner,
        EvalConfig,
        EvalRun,
        EvalStatus,
    )
    from .api.execution.eval_storage import save_run

    eval_block = spec["eval"]
    config = EvalConfig(
        graph_name=eval_block["graph_name"],
        env_nodeset=env_nodeset,
        selectors=eval_block.get("selectors", {}),
        dataset=eval_block.get("dataset", ""),
        split=eval_block.get("split", "val_unseen"),
        episode_count=eval_block.get("episode_count", -1),
        start_episode_index=eval_block.get("start_episode_index", 0),
        worker_count=eval_block.get("worker_count", 1),
        per_step_budget_sec=eval_block.get("per_step_budget_sec"),
        step_budget=eval_block.get("step_budget"),
        episode_indices=eval_block.get("episode_indices"),
        episode_selectors=eval_block.get("episode_selectors"),
    )
    run = EvalRun(
        run_id=spec["run_id"],
        config=config,
        status=EvalStatus.pending,
        created_at=spec.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
    )

    # Episode-boundary summary.json writer: BatchEvalRunner already calls
    # save_run on terminal exit; the per-episode cadence gives the backend's
    # status poll live data. ``run_dir`` is the scheduler's pool, so the
    # runner's graph.json / episodes/ land next to this summary.json.
    from .api.execution.eval_storage import run_to_dict

    def _write_summary(_ep) -> None:
        atomic_write_json(run_dir / "summary.json", run_to_dict(run))

    runner = BatchEvalRunner(run, graph, run_dir=run_dir, on_episode=_write_summary)
    rc = 0
    try:
        await runner.execute()
    except Exception:
        run.status = EvalStatus.error
        run.error = traceback.format_exc()
        rc = 1
    finally:
        await _shutdown_local_registry()
        save_run(run)
        touch_done(run_dir)
    return rc


def main() -> int:
    _set_pdeathsig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _apply_vram_cap()

    parser = argparse.ArgumentParser(prog="app.eval_subprocess_main")
    parser.add_argument(
        "--run-dir", required=True, help="Absolute path to outputs/eval_runs/{run_id}/"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("AGENTCANVAS_BACKEND_URL", "http://127.0.0.1:8765"),
        help="Parent backend URL (for /api/registry/snapshot fetch). Currently unused if shared_urls.json is pre-written.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
