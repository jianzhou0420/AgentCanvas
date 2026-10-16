"""Fleet — the N tagged copies of every replicated nodeset a graph needs,
routed per worker.

Spawning is the registry's job: ``prepare_graph(worker_count=N)`` (through
``ensure_nodesets_for_graph``) brings up ``N`` tagged subprocesses
(``name#0 … name#N-1``) for each ``parallelism="replicated"`` nodeset in
the graph. A :class:`Fleet` is the routing table over them: worker ``k``
gets ``{nodeset: url#k}`` for every replicated nodeset — env or not — so
its in-graph proxy calls hit its own subprocess, plus ``{nodeset:
panel#k}`` for the ones that expose an env panel, so its episode cascade
hits the same one. On exit the fleet unloads those copies.

At ``worker_count == 1`` the fleet is an empty routing table over the
global registry — the canvas-Play path, bit-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..graph_def import GraphDefinition, nodeset_names
from .launcher import server_key

log = logging.getLogger("agentcanvas.fleet")


@dataclass
class WorkerHandle:
    """One worker's routing: the per-run overrides its executor carries.

    ``env_panel_overrides`` maps nodeset name → this worker's tagged
    :class:`RemoteEnvPanelProxy` (set-episode/play side);
    ``server_url_overrides`` maps nodeset name → this worker's tagged
    subprocess URL (the in-graph proxy nodes' step/observe side). Both are
    empty at ``worker_count == 1``: the executor falls through to the
    global env panel registry and the URL baked into each proxy class.
    """

    worker_id: int
    env_panel_overrides: dict[str, Any] = field(default_factory=dict)
    server_url_overrides: dict[str, str] = field(default_factory=dict)


class Fleet:
    """Routing table + teardown for a graph's replicated nodesets at ``N``.

    Use as an async context manager; :meth:`worker` hands out the handle
    for worker ``k``. Each eval worker coroutine owns exactly one ``k`` for
    the whole run (task-contiguous dispatch: SAPIEN-backed envs crash after
    a dozen cross-task ``set_episode()`` rebuilds, so an episode chunk is
    pinned to its subprocess), so there is no lease queue — only lookup.
    """

    def __init__(
        self,
        graph: GraphDefinition,
        worker_count: int = 1,
        *,
        registry: Any = None,
    ) -> None:
        if worker_count < 1:
            raise ValueError(f"worker_count must be >= 1, got {worker_count}")
        self._graph = graph
        self._worker_count = worker_count
        self._registry = registry
        self._workers: list[WorkerHandle] = []
        # Replicated nodesets this fleet routed — and therefore owns the
        # teardown of. Empty at worker_count == 1.
        self._routed: list[str] = []

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def routed(self) -> list[str]:
        """Replicated nodesets with a per-worker route (empty at N == 1)."""
        return list(self._routed)

    def _get_registry(self) -> Any:
        if self._registry is None:
            from ..state import get_services

            self._registry = get_services().workspace_component_registry
        return self._registry

    async def __aenter__(self) -> Fleet:
        n = self._worker_count
        if n == 1:
            self._workers = [WorkerHandle(worker_id=0)]
            log.info("Fleet ready: 1 worker, no per-worker routing")
            return self

        from .env_panel import get_env_panel

        registry = self._get_registry()
        names = sorted(
            name
            for name in nodeset_names(self._graph)
            if registry.parallelism(name) == "replicated"
        )
        workers: list[WorkerHandle] = []
        for k in range(n):
            urls: dict[str, str] = {}
            panels: dict[str, Any] = {}
            for name in names:
                url = registry.get_server_url(name, tag=k)
                if url is None:
                    raise RuntimeError(
                        f"Fleet: tagged spawn {server_key(name, k)} is missing. "
                        f"Did ensure_nodesets_for_graph(worker_count={n}) succeed?"
                    )
                urls[name] = url
                panel = get_env_panel(server_key(name, k))
                if panel is not None:
                    panels[name] = panel
            workers.append(
                WorkerHandle(worker_id=k, env_panel_overrides=panels, server_url_overrides=urls)
            )
        self._workers = workers
        self._routed = names
        log.info("Fleet ready: %d workers over replicated nodesets %s", n, names)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._workers = []
        if not self._routed:
            return
        # Tear down the N tagged subprocesses this fleet routed.
        # ``unload_nodeset`` walks every tagged key and stops each
        # subprocess + deregisters its tagged env panel. Failures are
        # logged but not re-raised so exit always returns control cleanly.
        registry = self._get_registry()
        for name in self._routed:
            try:
                await registry.unload_nodeset(name)
                log.info("Fleet teardown: unloaded tagged copies of %s", name)
            except Exception:
                log.exception("Fleet teardown: failed to unload %s", name)
        self._routed = []

    def worker(self, worker_id: int) -> WorkerHandle:
        """The handle for worker ``worker_id``."""
        if not self._workers:
            raise RuntimeError("Fleet.worker() called outside async with")
        if not 0 <= worker_id < len(self._workers):
            raise ValueError(
                f"worker_id={worker_id} out of range for fleet of {len(self._workers)}"
            )
        return self._workers[worker_id]
