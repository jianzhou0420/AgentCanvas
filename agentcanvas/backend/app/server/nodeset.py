"""ServerNodeSet — a NodeSet backed by an external server process.

Combines :class:`BaseNodeSet` (node interface) with :class:`BaseServer`
(process management) via multiple inheritance.  Each function from the
server's manifest becomes a :class:`BaseCanvasNode` tool on the canvas.

Usage::

    class HabitatNodeSet(ServerNodeSet):
        name = "habitat"
        description = "Habitat-Sim VLN-CE environment"
        command = "/path/to/python habitat_server.py --port 9100"
        port = 9100
        startup_timeout = 60

That's it — 5 lines.  ``load()`` starts the server, fetches the manifest,
and generates proxy canvas nodes.  ``unload()`` stops the server.
"""

from __future__ import annotations

import logging
from typing import Any

from ..components.bases import BaseCanvasNode, BaseNodeSet, PortDef
from . import serialization
from .base_server import BaseServer
from .manifest import FunctionSchema
from .serialization import deserialize_value, serialize_value

log = logging.getLogger("agentcanvas.server-nodeset")


# ── Proxy Node Factory ──


def _make_proxy_forward(
    server_url: str,
    function_name: str,
    input_port_defs: list[PortDef],
    output_port_defs: list[PortDef],
):
    """Create a forward() method that forwards to the server over HTTP."""

    input_wire_map: dict[str, str] = {p.name: p.wire_type for p in input_port_defs}
    output_wire_map: dict[str, str] = {p.name: p.wire_type for p in output_port_defs}
    call_url = "{}/call/{}".format(server_url.rstrip("/"), function_name)

    async def forward(self: Any, inputs: dict, ctx: Any) -> dict:
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        use_msgpack = serialization.MSGPACK_OK
        payload: dict[str, Any] = {}
        for port_name, wire_type in input_wire_map.items():
            val = inputs.get(port_name)
            if val is None:
                continue
            payload[port_name] = val if use_msgpack else serialize_value(val, wire_type)

        request_body = {"inputs": payload, "config": self.config}

        resp_is_msgpack = False
        try:
            async with httpx.AsyncClient(timeout=60.0, **loopback_httpx_kwargs()) as client:
                if use_msgpack:
                    resp = await client.post(
                        call_url,
                        content=serialization.pack_body(request_body),
                        headers={
                            "Content-Type": serialization.MSGPACK_CONTENT_TYPE,
                            "Accept": serialization.MSGPACK_CONTENT_TYPE,
                        },
                    )
                else:
                    resp = await client.post(call_url, json=request_body)
                resp.raise_for_status()
                resp_is_msgpack = resp.headers.get("content-type", "").startswith(
                    serialization.MSGPACK_CONTENT_TYPE
                )
                data = serialization.unpack_body(resp.content) if resp_is_msgpack else resp.json()
        except Exception as e:
            log.warning("Server call failed: %s — %s", call_url, e)
            return {"error": str(e)}

        raw_outputs = data.get("outputs", data)
        result: dict[str, Any] = {}
        for port_name, wire_type in output_wire_map.items():
            val = raw_outputs.get(port_name)
            if val is None:
                continue
            result[port_name] = val if resp_is_msgpack else deserialize_value(val, wire_type)

        return result

    return forward


def _create_proxy_node(
    server_url: str,
    server_name: str,
    func: FunctionSchema,
) -> type[BaseCanvasNode]:
    """Create a BaseCanvasNode subclass for one server function."""

    # Use the original node_type from the manifest so graphs work identically
    # in both local and server mode (ADR-020, TODO #33).
    node_type = func.name
    display_name = func.name
    category = f"server:{server_name}"

    input_ports = [
        PortDef(name=p.name, wire_type=p.wire_type, description=p.description, optional=p.optional)
        for p in func.input_ports
    ]
    output_ports = [
        PortDef(name=p.name, wire_type=p.wire_type, description=p.description)
        for p in func.output_ports
    ]

    forward_method = _make_proxy_forward(server_url, func.name, input_ports, output_ports)

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
            "forward": forward_method,
        },
    )
    return cls  # type: ignore[return-value]


# ── ServerNodeSet ──


class ServerNodeSet(BaseNodeSet, BaseServer):
    """A NodeSet whose nodes come from an external server process.

    Inherits two interfaces:
    - :class:`BaseNodeSet` — ``load()`` / ``unload()`` / ``get_tools()``
    - :class:`BaseServer` — ``start()`` / ``stop()`` / ``health_check()``

    ``load()`` starts the server process, waits for it to become healthy,
    fetches the manifest, and generates proxy canvas nodes for each function.

    ``unload()`` stops the server process and clears the proxy nodes.

    Subclass and set class attributes::

        class HabitatNodeSet(ServerNodeSet):
            name = "habitat"
            command = "... python habitat_server.py --port 9100"
            port = 9100
    """

    # Proxy nodes generated from manifest
    _proxy_nodes: list[type[BaseCanvasNode]]

    def __init__(self, **overrides: Any) -> None:
        BaseNodeSet.__init__(self)
        BaseServer.__init__(self, **overrides)
        self._proxy_nodes = []

    # ── BaseNodeSet interface ──

    async def initialize(self, **kwargs: Any) -> None:
        """Start the server and generate proxy nodes from its manifest."""
        self.start()  # BaseServer.start() — launches subprocess, waits for /health

        manifest = self.fetch_manifest()
        if manifest is None:
            log.error("ServerNodeSet %s: connected but failed to fetch manifest", self.name)
            return

        self._proxy_nodes = [
            _create_proxy_node(self.url, self.name, func) for func in manifest.functions
        ]
        log.info(
            "ServerNodeSet %s loaded: %d nodes from %s",
            self.name,
            len(self._proxy_nodes),
            self.url,
        )

    async def shutdown(self) -> None:
        """Stop the server and clear proxy nodes."""
        self.stop()  # BaseServer.stop() — kills subprocess
        self._proxy_nodes = []
        log.info("ServerNodeSet %s unloaded", self.name)

    def get_tools(self) -> list:
        """Return proxy canvas nodes as nodes.

        Each proxy node is a dynamically-created BaseCanvasNode subclass
        whose forward() forwards to the server over HTTP.
        """
        # Return instances (WorkspaceComponentRegistry expects instances)
        return [cls() for cls in self._proxy_nodes]

    # ── Convenience ──

    def get_node_types(self) -> list[str]:
        """Return node_type strings for all proxy nodes."""
        return [cls.node_type for cls in self._proxy_nodes]
