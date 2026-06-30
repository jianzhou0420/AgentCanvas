"""Worker-pool abstraction for batch eval — ADR-028 (Phase A foundation).

Sits between :class:`BatchEvalRunner` and :class:`LoopRunner`. Each worker
holds the ``env_panel_overrides`` dict that worker's :class:`LoopRunner`
will use to resolve env-panel lookups (ADR-028 rule 2).

PA-3 ships the abstraction at ``worker_count=1`` with **zero behaviour
change**: a single worker handle, empty overrides, no subprocess spawning,
no concurrency. PB-1 will add the N tagged subprocess spawn inside
``__aenter__``; PB-2 will swap the sequential acquire-loop for
``asyncio.gather`` over a shared episode queue. This module is the seam
both phases hook into.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agentcanvas.env-worker-pool")


@dataclass
class WorkerHandle:
    """One slot in the pool. Carries the per-runner routing for both the
    nodeset's ``BaseEnvPanel`` (set-episode/play side) and the in-graph
    proxy node calls (step/observe side).

    At ``worker_count=1`` (PA-3): both override dicts are empty, the
    runner's executor falls through to the global env panel registry, and
    proxy nodes use the URL baked into their class closure — the canvas
    singleton path, bit-identical to today.

    At ``worker_count>1`` (PB-1 + PB-1.5): ``env_panel_overrides`` maps
    the env nodeset name (e.g. ``"env_habitat"``) to this worker's tagged
    :class:`RemoteEnvPanelProxy`, and ``server_url_overrides`` maps the
    same name to this worker's tagged subprocess URL so in-graph proxy
    calls route to the same env as the env panel.
    """

    worker_id: int
    env_panel_overrides: dict[str, Any] = field(default_factory=dict)
    server_url_overrides: dict[str, str] = field(default_factory=dict)


def resolve_per_step_budget(
    env_nodeset_name: str | None,
    override: float | None,
) -> float:
    """Resolve the per-step wall-clock budget for the eval timeout.

    Precedence (ADR-028 rule 3):
      1. Explicit ``EvalConfig.per_step_budget_sec`` (user override).
      2. The env nodeset's ``default_per_step_budget_sec`` ClassVar.
      3. ``BaseNodeSet.default_per_step_budget_sec`` framework default.
    """
    if override is not None and override > 0:
        return float(override)
    # Lazy imports keep this module free of circular deps with state/registry.
    from ..components.bases import BaseNodeSet

    if env_nodeset_name:
        try:
            from ..state import get_services

            registry = get_services().workspace_component_registry
            ns = registry._live_nodesets.get(env_nodeset_name)
            if ns is not None:
                return float(type(ns).default_per_step_budget_sec)
        except Exception as e:  # pragma: no cover — defensive
            log.debug("Falling back to BaseNodeSet default; lookup failed: %s", e)
    return float(BaseNodeSet.default_per_step_budget_sec)


class EnvWorkerPool:
    """N-worker pool driving env subprocesses for batch eval.

    Lifecycle: use as an async context manager. ``__aenter__`` populates the
    pool; ``__aexit__`` tears it down. Workers are leased through
    :meth:`acquire`, an async context manager that yields a
    :class:`WorkerHandle`.

    PA-3 (this commit): ``worker_count=1`` only. The pool exposes one
    worker with empty overrides — semantically identical to today's
    BatchEvalRunner, but the orchestration goes through this seam so PB-1
    and PB-2 can swap the implementation without touching the runner.
    """

    def __init__(
        self,
        worker_count: int = 1,
        env_nodeset: str | None = None,
    ) -> None:
        if worker_count < 1:
            raise ValueError(f"worker_count must be >= 1, got {worker_count}")
        self._worker_count = worker_count
        self._env_nodeset = env_nodeset or None
        self._available: asyncio.Queue[WorkerHandle] | None = None
        self._workers: list[WorkerHandle] = []
        # PB-2: cleanup gating. True only when this pool's enter populated
        # tagged overrides (multi-worker mode against an env nodeset). At
        # ``worker_count=1`` or no env nodeset, the pool is a no-op shell
        # over the global registry and __aexit__ must not unload anything.
        self._owns_tagged_spawn: bool = False

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def env_nodeset(self) -> str | None:
        return self._env_nodeset

    async def __aenter__(self) -> EnvWorkerPool:
        self._available = asyncio.Queue()

        if self._worker_count > 1 and self._env_nodeset:
            # PB-2: bind each worker to its tagged subprocess. Spawning
            # already happened in start_eval_v2 → ensure_nodesets_for_graph
            # (PB-1); the pool just looks up the registered tagged
            # env panel proxies and URLs.
            from ..components.env_panel import get_env_panel
            from ..state import get_services

            registry = get_services().workspace_component_registry
            for k in range(self._worker_count):
                tagged_name = f"{self._env_nodeset}#{k}"
                panel = get_env_panel(tagged_name)
                url = registry.get_server_url(self._env_nodeset, tag=k)
                if panel is None or url is None:
                    raise RuntimeError(
                        f"EnvWorkerPool: tagged spawn for {tagged_name} is incomplete "
                        f"(env panel={'ok' if panel else 'missing'}, "
                        f"url={'ok' if url else 'missing'}). "
                        f"Did ensure_nodesets_for_graph(worker_count={self._worker_count}) succeed?"
                    )
                handle = WorkerHandle(
                    worker_id=k,
                    env_panel_overrides={self._env_nodeset: panel},
                    server_url_overrides={self._env_nodeset: url},
                )
                self._workers.append(handle)
                await self._available.put(handle)
            self._owns_tagged_spawn = True
        else:
            # Single-worker / no env nodeset: empty overrides, fall through
            # to the global registry + baked proxy URLs (Phase A invariant
            # preserved bit-identically).
            for k in range(self._worker_count):
                handle = WorkerHandle(worker_id=k)
                self._workers.append(handle)
                await self._available.put(handle)

        log.info(
            "EnvWorkerPool ready: %d worker(s), env=%s, tagged_spawn=%s",
            self._worker_count,
            self._env_nodeset,
            self._owns_tagged_spawn,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Drain the queue + drop handles before subprocess teardown so a
        # buggy caller can't reuse a stale handle while we're stopping the
        # subprocess it points at.
        if self._available is not None:
            while not self._available.empty():
                try:
                    self._available.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self._workers.clear()
        self._available = None

        if self._owns_tagged_spawn and self._env_nodeset:
            # PB-2: tear down the N tagged subprocesses spawned for this
            # pool's lifetime. ``unload_nodeset`` walks every tagged key
            # (PB-1) and stops each subprocess + deregisters its tagged
            # env panel. Failures are logged but not re-raised so the
            # pool exit always returns control to the caller cleanly.
            try:
                from ..state import get_services

                registry = get_services().workspace_component_registry
                await registry.unload_nodeset(self._env_nodeset)
                log.info(
                    "EnvWorkerPool teardown: unloaded tagged copies of %s",
                    self._env_nodeset,
                )
            except Exception:
                log.exception("EnvWorkerPool teardown: failed to unload %s", self._env_nodeset)

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[WorkerHandle]:
        """Lease a worker for one episode. Returns it to the pool on exit."""
        if self._available is None:
            raise RuntimeError("EnvWorkerPool.acquire() called outside async with")
        handle = await self._available.get()
        try:
            yield handle
        finally:
            await self._available.put(handle)

    @contextlib.asynccontextmanager
    async def acquire_specific(self, worker_id: int) -> AsyncIterator[WorkerHandle]:
        """Lease a specific worker by id, used for task-contiguous dispatch.

        BatchEvalRunner partitions cross-task sweeps into contiguous chunks
        (one per worker_id) so each worker's underlying subprocess sees a
        small bounded number of task switches. SAPIEN-backed envs (SIMPLER)
        crash the subprocess after ~15 ``set_episode()`` rebuilds across
        tasks, so an idle worker MUST NOT pick an episode from a different
        task chunk — every chunk is pinned to its own subprocess for the
        whole run. ``acquire`` (any-worker) is unsafe for that case.

        At ``worker_count=1`` this collapses to ``acquire`` semantics.
        """
        if self._available is None:
            raise RuntimeError("EnvWorkerPool.acquire_specific() called outside async with")
        if not 0 <= worker_id < len(self._workers):
            raise ValueError(f"worker_id={worker_id} out of range for pool of {len(self._workers)}")
        # Spin until the requested worker is free. The pool is a closed
        # set of N handles, all leased through this pool, so this loop
        # only re-queues other workers; it cannot deadlock as long as
        # callers always release. Buffer rotated handles back in order.
        target = self._workers[worker_id]
        held: list[WorkerHandle] = []
        try:
            while True:
                handle = await self._available.get()
                if handle.worker_id == worker_id:
                    break
                held.append(handle)
        finally:
            for h in held:
                await self._available.put(h)
        try:
            yield target
        finally:
            await self._available.put(target)
