"""AutoServerApp — auto-generates a server-mode process from any BaseNodeSet.

Introspects the nodeset's nodes' ``input_ports`` / ``output_ports`` to
build ``ServerFunction`` entries.  No manual port re-declaration needed.

Usage::

    from app.server.auto_server_app import AutoServerApp
    from workspace.nodesets.model.model_sam import SamNodeSet

    app = AutoServerApp(SamNodeSet)
    app.serve()

Or via the CLI entry point::

    python -m app.server.auto_host --file workspace/nodesets/model/model_sam.py \\
        --class SamNodeSet --port 9200
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from ..components.bases import BaseCanvasNode, BaseNodeSet, ui_config_to_dict
from .manifest import PortSchema
from .server_app import ServerApp, ServerFunction

log = logging.getLogger(__name__)


class _ServerContext:
    """Minimal execution context for server-mode nodes.

    Mirrors the attribute-bag pattern of ``_NodeStateProxy`` (see
    ``agent_loop/graph_executor.py``) so nodes like
    policy_adapter_vlnce__predict can store state across calls (RNN hidden states,
    loaded checkpoints, etc.).
    """

    def __getattr__(self, name: str) -> Any:
        # Return None for any unset attribute — matches the proxy's dict-backed default.
        return None


def _ctx_for_call(base_ctx: Any, containers: dict | None, exec_ctx: dict | None) -> Any:
    """Per-call context for cross-nodeset container access (face B prototype).

    With no ``_exec`` (the common case) return the shared ``base_ctx`` unchanged
    — zero behaviour change. When the proxy passed granted executor-home
    container ids + an execution_id, return a fresh ``_ServerContext`` whose
    ``containers`` is the nodeset's own dict overlaid with a
    ``RemoteContainerProxy`` per granted id. Per-call (not mutating the shared
    ctx) so concurrent calls with different execution_ids never alias.
    """
    if not exec_ctx:
        return base_ctx
    grants = exec_ctx.get("grants") or []
    execution_id = exec_ctx.get("execution_id")
    base_url = os.environ.get("AGENTCANVAS_EXECUTOR_URL")
    if not (grants and execution_id):
        return base_ctx
    if not base_url:
        raise RuntimeError(
            "cross-process container grant present but AGENTCANVAS_EXECUTOR_URL "
            "is unset — the subprocess cannot call back to the executor home"
        )
    from .remote_container import RemoteContainerProxy

    merged = dict(containers or {})
    for cid in grants:
        merged[cid] = RemoteContainerProxy(base_url, execution_id, cid)
    call_ctx = _ServerContext()
    call_ctx.containers = merged
    return call_ctx


def _make_handler(
    tool: BaseCanvasNode,
    inference_lock: Callable[[], asyncio.Lock] | None = None,
    containers: dict | None = None,
) -> Callable:
    """Create a server handler closure for a single node instance.

    A ``serialized = True`` node (``BaseCanvasNode.serialized``) fires under
    the app's one inference lock — ``inference_lock()`` returns it, created
    inside the running loop on first use — so its calls from K workers run
    one at a time on the device this subprocess owns. Every other node's
    calls are concurrent.
    """
    ctx = _ServerContext()
    # Inject the nodeset's owned containers by-reference. ``containers`` is the
    # app's stable dict (possibly empty at closure-creation time, populated
    # later in place), so assign the reference itself — never a fresh copy.
    ctx.containers = containers if containers is not None else {}
    serialized = bool(getattr(type(tool), "serialized", False))
    if serialized and inference_lock is None:
        raise RuntimeError(
            f"{type(tool).__name__}: serialized=True but no inference lock was "
            f"provided to _make_handler"
        )

    async def fire(inputs: dict, config: dict, exec_ctx: dict | None) -> dict:
        tool.config = config
        # Cross-nodeset container-access prototype (face B): when the proxy
        # passed ``_exec`` (granted executor-home container ids + the
        # execution_id), build a per-call ctx whose ``containers`` overlays
        # RemoteContainerProxy handles onto the nodeset's own containers.
        call_ctx = _ctx_for_call(ctx, containers, exec_ctx)
        return await tool.forward(inputs, call_ctx)

    if not serialized:

        async def handler(inputs: dict, config: dict, exec_ctx: dict | None = None) -> dict:
            return await fire(inputs, config, exec_ctx)

        return handler

    async def serialized_handler(inputs: dict, config: dict, exec_ctx: dict | None = None) -> dict:
        async with inference_lock():
            return await fire(inputs, config, exec_ctx)

    return serialized_handler


def _portdefs_to_schemas(portdefs: list) -> list[PortSchema]:
    """Convert a list of PortDef to PortSchema."""
    return [
        PortSchema(
            name=p.name,
            wire_type=p.wire_type,
            description=p.description,
            optional=p.optional,
        )
        for p in portdefs
    ]


class AutoServerApp(ServerApp):
    """A ServerApp that auto-generates functions from a BaseNodeSet.

    Instead of manually declaring ``ServerFunction`` entries, this class
    introspects the nodeset's nodes and builds them from ``input_ports``
    and ``output_ports``.
    """

    def __init__(self, nodeset_cls: type[BaseNodeSet]) -> None:
        super().__init__()
        self._nodeset_cls = nodeset_cls
        self._nodeset: BaseNodeSet | None = None
        self._env_panel: Any | None = None
        self.name = nodeset_cls.name
        self.description = getattr(nodeset_cls, "description", "")
        # MCP session policy: explicit ``mcp_exclusive`` declaration wins;
        # None derives from the parallelism contract — "replicated" marks the
        # nodeset stateful (env scene state, simulator handles), which is
        # exactly what makes interleaved MCP clients unsafe.
        declared = getattr(nodeset_cls, "mcp_exclusive", None)
        if declared is None:
            self.mcp_exclusive = getattr(nodeset_cls, "parallelism", "shared") == "replicated"
        else:
            self.mcp_exclusive = bool(declared)
        # The one inference lock for ``serialized = True`` nodes hosted in
        # this subprocess (one device, one forward in flight). Created lazily
        # inside the server's running loop — see ``_get_inference_lock``;
        # the Python 3.8 envs that host some nodesets bind a Lock to the
        # loop current at construction time.
        self._inference_lock: asyncio.Lock | None = None
        # Nodeset-level state containers (owned by this nodeset, living in this
        # subprocess; shared by-reference by the nodeset's own nodes). Built
        # once into this stable dict — handler closures in ``get_functions``
        # capture the reference and see it populated even though
        # ``get_functions`` may run (for the manifest) before ``on_startup``.
        self._containers: dict[str, Any] = {}
        self._containers_built = False
        # Instantiate the nodeset's BaseEnvPanel (if declared) here so
        # ``_build_app`` can see it when mounting /env-panel/* routes —
        # FastAPI reads ``get_env_panel_instance()`` at app-construction
        # time, which runs before ``on_startup``.
        panel_cls = getattr(self._nodeset_cls, "env_panel", None)
        if panel_cls is not None:
            try:
                panel = panel_cls()
                panel._context = {
                    "mode": "local",
                    "server_url": None,
                    "nodeset_name": self.name,
                }
                self._env_panel = panel
                log.info(
                    "AutoServerApp: hosting env panel %s for nodeset %s",
                    getattr(panel, "name", "?"),
                    self.name,
                )
            except Exception:
                log.exception(
                    "AutoServerApp: failed to instantiate env panel for %s",
                    self.name,
                )

    async def on_startup(self) -> None:
        # Reuse the instance get_functions() may have created for the manifest:
        # the /call handlers close over ITS tools (self._nodeset back-refs), so
        # constructing a second instance here would initialize an object the
        # handlers never see — session-type nodesets (state on the nodeset
        # instance, e.g. model_vggt_slam2) then serve every call from the
        # never-initialized throwaway. Caught 2026-07-14.
        if self._nodeset is None:
            self._nodeset = self._nodeset_cls()
        await self._nodeset.initialize()
        self._ensure_containers()
        log.info("AutoServerApp: initialized nodeset %s", self.name)

    def _get_inference_lock(self) -> asyncio.Lock:
        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()
        return self._inference_lock

    async def on_shutdown(self) -> None:
        if self._nodeset is not None:
            await self._nodeset.shutdown()
        self._containers.clear()
        log.info("AutoServerApp: shut down nodeset %s", self.name)

    def get_env_panel_instance(self) -> Any | None:
        return self._env_panel

    def _ensure_containers(self) -> None:
        """Build this nodeset's owned (nodeset-level) containers once, in place.

        Mutates ``self._containers`` (the stable dict from __init__) so handler
        closures that captured the reference in ``get_functions`` observe the
        built containers. ``allow_opaque=True``: nodeset-level containers are
        never serialized, so they may hold opaque values.
        """
        if self._containers_built:
            return
        self._containers_built = True
        if self._nodeset is None:
            self._nodeset = self._nodeset_cls()
        from ..agent_loop.state_containers import build_containers

        built = build_containers(self._nodeset.get_containers(), allow_opaque=True)
        self._containers.update(built)
        if built:
            log.info(
                "AutoServerApp: built %d owned container(s) for nodeset %s: %s",
                len(built),
                self.name,
                list(built.keys()),
            )

    def get_owned_container(self, container_id: str) -> Any:
        """A container this nodeset homes, for the cross-process read/write
        endpoints (faces A/C). The executor broker forwards here."""
        return self._containers.get(container_id)

    def get_container_previews(self) -> dict:
        """Read-only preview of this nodeset's owned containers (for display).

        The only state that crosses the process boundary — piggybacked on the
        ``/call`` response (see ``server_app.call_function``) and surfaced in
        the canvas State panel. NOT a cross-process access path.
        """
        if not self._containers:
            return {}
        return {
            cid: {"label": c.label, "states": c.get_preview()}
            for cid, c in self._containers.items()
        }

    def evict_container_key(self, key: str) -> int:
        """Drop one key's partition from every owned container — worker-safe
        per-episode cleanup driven from the eval worker loop (POST
        /containers/evict). Returns the number of containers touched."""
        for container in self._containers.values():
            container.evict(key)
        return len(self._containers)

    def get_manifest(self):  # type: ignore[override]
        manifest = super().get_manifest()
        ns = self._nodeset or self._nodeset_cls()
        try:
            manifest.containers = [c.to_dict() for c in ns.get_containers()]
        except Exception:
            log.exception(
                "AutoServerApp: failed to collect owned container schema for %s",
                self.name,
            )
        return manifest

    def get_functions(self) -> list[ServerFunction]:
        if self._nodeset is None:
            # Before on_startup — create a temporary instance for manifest
            self._nodeset = self._nodeset_cls()

        # Build owned containers before binding handlers so each handler's ctx
        # captures the populated (stable) dict by reference.
        self._ensure_containers()

        tools = self._nodeset.get_tools()
        functions: list[ServerFunction] = []

        for tool in tools:
            node_type = getattr(tool, "node_type", "") or getattr(tool, "name", "unknown")
            description = getattr(tool, "description", "")
            input_ports = getattr(tool, "input_ports", [])
            output_ports = getattr(tool, "output_ports", [])
            config_schema = getattr(tool, "config_schema", {})
            ui_config = ui_config_to_dict(getattr(tool, "ui_config", None))

            functions.append(
                ServerFunction(
                    name=node_type,
                    description=description,
                    input_ports=_portdefs_to_schemas(input_ports),
                    output_ports=_portdefs_to_schemas(output_ports),
                    config_schema=config_schema,
                    ui_config=ui_config,
                    handler=_make_handler(tool, self._get_inference_lock, self._containers),
                )
            )

        log.info(
            "AutoServerApp: generated %d functions from nodeset %s",
            len(functions),
            self.name,
        )
        return functions
