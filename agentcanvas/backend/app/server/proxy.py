"""ServerNodeProxy — auto-generates BaseCanvasNode subclasses from server manifests.

Each server function becomes a canvas node whose ``forward()`` forwards
inputs over HTTP to the remote server and returns the deserialized outputs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..components.bases import (
    BaseCanvasNode,
    ConfigField,
    DisplayField,
    NodeUIConfig,
    PortDef,
)
from . import serialization
from .manifest import FunctionSchema, ServerManifest
from .serialization import deserialize_value, serialize_value

log = logging.getLogger("agentcanvas.server-proxy")


def _make_forward(
    server_url: str, function_name: str, input_port_defs: list, output_port_defs: list
):
    """Create a forward() method bound to a specific server URL and function.

    The baked ``call_url`` is the default route. ADR-028 PB-1.5: at call
    time, ``forward`` consults ``ctx._executor.get_server_url(nodeset_name)``
    (set by the worker-pool eval path) and routes to the per-worker URL
    when an override exists. With no override (canvas Play / single-worker
    eval), the baked URL is used — bit-identical to pre-PB-1.5 behaviour.
    """

    input_wire_map: dict[str, str] = {p.name: p.wire_type for p in input_port_defs}
    output_wire_map: dict[str, str] = {p.name: p.wire_type for p in output_port_defs}
    baked_call_url = "{}/call/{}".format(server_url.rstrip("/"), function_name)
    # Convention: server-mode node types are "{nodeset}__{tool}".
    nodeset_name = function_name.split("__", 1)[0] if "__" in function_name else ""

    async def forward(self: Any, inputs: dict, ctx: Any) -> dict:
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        # Resolve per-runner URL override (ADR-028 PB-1.5). Empty/missing →
        # use the URL baked into this proxy class at registry-load time.
        call_url = baked_call_url
        active_server_url = server_url
        executor = getattr(ctx, "_executor", None)
        if executor is not None and nodeset_name:
            override = executor.get_server_url(nodeset_name)
            if override:
                active_server_url = override.rstrip("/")
                call_url = f"{active_server_url}/call/{function_name}"

        # Build the request body with NATIVE values. The msgpack codec is
        # type-driven and encodes binary types (ndarray / torch / PIL) itself;
        # only the legacy JSON fallback needs per-port wire-type serialization.
        use_msgpack = serialization.MSGPACK_OK
        payload: dict[str, Any] = {}
        for port_name, wire_type in input_wire_map.items():
            val = inputs.get(port_name)
            if val is None:
                continue
            payload[port_name] = val if use_msgpack else serialize_value(val, wire_type)

        # Add config from node instance
        request_body = {"inputs": payload, "config": self.config}

        # Cross-nodeset container-access prototype (face B): if this proxy node
        # holds access-grants to executor-home containers, the executor already
        # injected them into ``ctx.containers``. Pass their ids + the current
        # execution_id so the subprocess can build a RemoteContainerProxy that
        # calls back to the executor's /api/internal/containers endpoint.
        try:
            exec_id = getattr(getattr(ctx, "session", None), "_execution_id", None)
            granted = list(getattr(ctx, "containers", {}) or {})
            if exec_id and granted:
                request_body["_exec"] = {"execution_id": exec_id, "grants": granted}
        except Exception:
            log.debug("exec-context attach failed", exc_info=True)

        # POST to server. msgpack by default (Content-Type negotiated); the
        # server accepts JSON too (migration window) and echoes the format.
        resp_is_msgpack = False
        try:
            # System Log P2: time the serialize / round-trip / deserialize legs
            # + payload sizes, and fold them into the firing's transport bucket.
            _ser_ms = 0.0
            _req_bytes = 0
            _packed = None
            if use_msgpack:
                _t = time.perf_counter()
                _packed = serialization.pack_body(request_body)
                _ser_ms = (time.perf_counter() - _t) * 1000
                _req_bytes = len(_packed)
            async with httpx.AsyncClient(timeout=120.0, **loopback_httpx_kwargs()) as client:
                _t_rtt = time.perf_counter()
                if use_msgpack:
                    resp = await client.post(
                        call_url,
                        content=_packed,
                        headers={
                            "Content-Type": serialization.MSGPACK_CONTENT_TYPE,
                            "Accept": serialization.MSGPACK_CONTENT_TYPE,
                        },
                    )
                else:
                    resp = await client.post(call_url, json=request_body)
                _rtt_ms = (time.perf_counter() - _t_rtt) * 1000
                resp.raise_for_status()
                resp_is_msgpack = resp.headers.get("content-type", "").startswith(
                    serialization.MSGPACK_CONTENT_TYPE
                )
                _resp_bytes = len(resp.content)
                _t_de = time.perf_counter()
                data = serialization.unpack_body(resp.content) if resp_is_msgpack else resp.json()
                _de_ms = (time.perf_counter() - _t_de) * 1000
            serialization.accumulate_transport(
                rtt_ms=_rtt_ms,
                req_bytes=_req_bytes,
                resp_bytes=_resp_bytes,
                serialize_ms=_ser_ms,
                deserialize_ms=_de_ms,
            )
        except httpx.ConnectError:
            log.warning("Server unreachable: %s", call_url)
            return {"error": f"Server unreachable: {active_server_url}"}
        except Exception as e:
            log.exception("Server call failed: %s", call_url)
            return {"error": str(e)}

        # Stash read-only owned-container previews onto the executor for the
        # canvas State panel (see ``server_app.call_function``). Best-effort —
        # display only, never fatal to the call.
        if isinstance(data, dict) and data.get("containers") and executor is not None:
            try:
                executor.record_subprocess_containers(nodeset_name, data["containers"])
            except Exception:
                log.debug("record_subprocess_containers failed", exc_info=True)

        # C.5 Dynamic Fire-List passthrough. A server-side
        # ``DynamicFireListNode`` returns a FireList instead of a port-output
        # dict; the server serializes it as ``{"__firelist__": {...}}``
        # (see ``server_app.py`` /call route). Reconstruct the FireList here
        # so the executor's ``_fire_node`` recognises the sentinel via
        # ``isinstance(result, FireList)`` and dispatches each child.
        if isinstance(data, dict) and "__firelist__" in data:
            from ..components.bases import FireList, FireSpec

            fl = data["__firelist__"]
            return FireList(
                specs=[
                    FireSpec(
                        node_type=s.get("node_type", ""),
                        inputs=s.get("inputs", {}) or {},
                        config=s.get("config", {}) or {},
                        label=s.get("label", "") or "",
                        capture_outputs=s.get("capture_outputs"),
                    )
                    for s in fl.get("specs", [])
                ],
                spawner_outputs=fl.get("spawner_outputs", {}) or {},
                aggregator=fl.get("aggregator", {}) or {},
            )

        # Decode outputs. msgpack responses are already native (the codec
        # restored binary types on unpack); only JSON needs per-port decode.
        raw_outputs = data.get("outputs", data)
        result: dict[str, Any] = {}
        for port_name, wire_type in output_wire_map.items():
            val = raw_outputs.get(port_name)
            if val is None:
                continue
            result[port_name] = val if resp_is_msgpack else deserialize_value(val, wire_type)

        return result

    return forward


def _dict_to_nodeuiconfig(d: dict) -> NodeUIConfig:
    """Reconstruct a NodeUIConfig from the dict shape carried in the
    server manifest. Empty / missing dict → default NodeUIConfig().

    Reconstructing real dataclass instances (rather than storing the dict
    directly on the proxy class) keeps the local-vs-proxy serialization
    path uniform: ``_serialize_ui_config`` in
    ``app/api/platform/components.py`` does ``asdict(f)`` on each
    config_field / display_field, which would crash on a raw dict.
    """
    if not d:
        return NodeUIConfig()
    cf_dicts = d.get("config_fields") or []
    df_dicts = d.get("display_fields") or []
    return NodeUIConfig(
        color=d.get("color", ""),
        layout=d.get("layout", "block"),
        width=d.get("width", ""),
        min_width=d.get("min_width", ""),
        max_width=d.get("max_width", ""),
        min_height=d.get("min_height", ""),
        rounding=d.get("rounding", ""),
        config_fields=[ConfigField(**cf) for cf in cf_dicts],
        display_fields=[DisplayField(**df) for df in df_dicts],
    )


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
            "ui_config": _dict_to_nodeuiconfig(func.ui_config),
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
