"""ServerNodeProxy — auto-generates BaseCanvasNode subclasses from server manifests.

Each server function becomes a canvas node whose ``forward()`` forwards
inputs over HTTP to the remote server and returns the deserialized outputs.
"""

from __future__ import annotations

import logging
from typing import Any

from ..components.bases import BaseCanvasNode, PortDef, ui_config_from_dict
from ..standard.telemetry import _current_node_usage, merge_usage
from .client import AutoHostClient, CallError, ServerUnreachable
from .manifest import FunctionSchema, ServerManifest

log = logging.getLogger("agentcanvas.server-proxy")

# Per-call wall-clock ceiling for one hosted firing.
CALL_TIMEOUT_S = 120.0


def _make_forward(
    server_url: str, function_name: str, input_port_defs: list, output_port_defs: list
):
    """Create a forward() method bound to a specific server URL and function.

    The baked URL is the default route. At call time, ``forward`` consults
    ``ctx._executor.get_server_url(nodeset_name)`` (the Fleet's per-worker
    routing table) and routes to the per-worker URL when an override
    exists. With no override (canvas Play / single-worker eval), the baked
    URL is used.

    The HTTP contract lives in :class:`AutoHostClient`; this closure only
    adds what the executor needs: the URL override, the container grant,
    and folding the hosted node's log + usage into THIS firing — a server
    error raises, convicting the firing like a local node exception.
    """

    input_wire_map: dict[str, str] = {p.name: p.wire_type for p in input_port_defs}
    output_wire_map: dict[str, str] = {p.name: p.wire_type for p in output_port_defs}
    base_url = server_url.rstrip("/")
    # Convention: server-mode node types are "{nodeset}__{tool}".
    nodeset_name = function_name.split("__", 1)[0] if "__" in function_name else ""

    async def forward(self: Any, inputs: dict, ctx: Any) -> dict:
        active_url = base_url
        executor = getattr(ctx, "_executor", None)
        if executor is not None and nodeset_name:
            override = executor.get_server_url(nodeset_name)
            if override:
                active_url = override.rstrip("/")

        # Cross-nodeset container-access prototype (face B): if this proxy node
        # holds access-grants to executor-home containers, the executor already
        # injected them into ``ctx.containers``. Pass their ids + the current
        # execution_id so the subprocess can build a RemoteContainerProxy that
        # calls back to the executor's /api/internal/containers endpoint.
        exec_ctx: dict | None = None
        try:
            exec_id = getattr(getattr(ctx, "session", None), "_execution_id", None)
            granted = list(getattr(ctx, "containers", {}) or {})
            if exec_id and granted:
                exec_ctx = {"execution_id": exec_id, "grants": granted}
        except Exception:
            log.debug("exec-context attach failed", exc_info=True)

        client = AutoHostClient(active_url, timeout=CALL_TIMEOUT_S)
        try:
            res = await client.call(
                function_name,
                inputs,
                self.config,
                exec_ctx=exec_ctx,
                input_wire_types=input_wire_map,
                output_wire_types=output_wire_map,
            )
        except ServerUnreachable:
            log.warning("Server unreachable: %s/call/%s", active_url, function_name)
            raise
        except CallError:
            log.error("Server call failed: %s/call/%s", active_url, function_name)
            raise

        # The hosted node's own log entries and LLM usage land on THIS
        # firing, exactly as if the node had run in-process.
        for entry in res.log:
            self._self_log(entry.get("key", ""), entry.get("value"))
        bucket = _current_node_usage.get()
        if bucket is not None:
            merge_usage(bucket, res.usage)

        # Stash read-only owned-container previews onto the executor for the
        # canvas State panel. Best-effort — display only, never fatal.
        if res.containers and executor is not None:
            try:
                executor.record_subprocess_containers(nodeset_name, res.containers)
            except Exception:
                log.debug("record_subprocess_containers failed", exc_info=True)

        # C.5 Dynamic Fire-List passthrough: the executor's ``_fire_node``
        # recognises the FireList and dispatches each child.
        if res.firelist is not None:
            return res.firelist
        return res.outputs

    return forward


def create_proxy_node(
    server_url: str,
    server_name: str,
    func: FunctionSchema,
) -> type[BaseCanvasNode]:
    """Dynamically create a BaseCanvasNode subclass for a server function.

    Args:
        server_url: Base URL of the server (e.g. ``http://localhost:9001``).
        server_name: Server name from the manifest.
        func: Function schema describing ports and metadata.

    Returns:
        A new class that extends BaseCanvasNode with the correct ports
        and a forward() that forwards to the remote server.
    """
    # Use the original node_type from the manifest so graphs work identically
    # in both local and server mode (ADR-020, TODO #33).
    node_type = func.name
    display_name = func.name
    category = f"server:{server_name}"

    # Convert PortSchema → PortDef
    input_ports = [
        PortDef(
            name=p.name,
            wire_type=p.wire_type,
            description=p.description,
            optional=p.optional,
        )
        for p in func.input_ports
    ]
    output_ports = [
        PortDef(
            name=p.name,
            wire_type=p.wire_type,
            description=p.description,
        )
        for p in func.output_ports
    ]

    # Create the forward method bound to this server/function
    forward_method = _make_forward(server_url, func.name, input_ports, output_ports)

    # Dynamically create the class
    cls = type(
        f"ServerProxy_{server_name}_{func.name}",
        (BaseCanvasNode,),
        {
            "node_type": node_type,
            "display_name": display_name,
            "description": func.description,
            "category": category,
            "icon": "Globe",
            "input_ports": input_ports,
            "output_ports": output_ports,
            "config_schema": func.config_schema,
            "ui_config": ui_config_from_dict(func.ui_config),
            "forward": forward_method,
        },
    )

    return cls  # type: ignore[return-value]


def generate_proxy_nodes(
    server_url: str,
    manifest: ServerManifest,
) -> list[type[BaseCanvasNode]]:
    """Generate one proxy node class per function in the manifest.

    Args:
        server_url: Base URL of the server.
        manifest: Parsed server manifest.

    Returns:
        List of dynamically created BaseCanvasNode subclasses.
    """
    nodes: list[type[BaseCanvasNode]] = []
    for func in manifest.functions:
        cls = create_proxy_node(server_url, manifest.name, func)
        nodes.append(cls)
        log.info(
            "Generated proxy node: %s (%d in, %d out)",
            cls.node_type,
            len(func.input_ports),
            len(func.output_ports),
        )
    return nodes
