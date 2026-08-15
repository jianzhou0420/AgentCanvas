"""MCP projection — the native ``/mcp`` face of a server-mode nodeset.

One source of truth, two projections: the node-class declarations that
``get_manifest()`` introspects are re-expressed here as MCP tools, served
over Streamable HTTP alongside (never instead of) the manifest protocol.
``GET /manifest`` + msgpack ``POST /call/{fn}`` stay the framework-facing
face; ``/mcp`` is the model-facing face any MCP client (Claude Code, codex)
can consume with zero per-tool hand-coding.

The schema synthesis (wire_type → JSON type, ``optional`` → ``required``,
config_fields → properties) is lifted from the verified 2026-07-20 generic
bridge spike (``coding-agent/bridges/nodeset_mcp.py``, deleted at 68a2f0b).
Unlike the spike this is in-process: ``call_tool`` invokes the registered
handler directly — same arity contract and same JSON-regime serialization
as the ``/call`` route, no HTTP self-call.

This module imports the ``mcp`` SDK (py3.10+) at top level. On the pinned
py3.8 server hosts (ac-vlnce) the import fails; ``server_app`` guards the
import and runs manifest-only — those nodesets keep the out-of-process
projector path (the resurrected spike form).

SDK compatibility: written against mcp 1.x (verified on 1.27.0 — install
``mcp==1.27.0`` in nodeset envs). mcp 2.0 replaced the lowlevel decorator
API (no ``Server.list_tools``); under it the build fails and ``server_app``'s
guard degrades the process to manifest-only, so the mismatch is safe but
the /mcp face is absent.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Awaitable, Callable

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from . import event_push
from .serialization import deserialize_value, serialize_value

log = logging.getLogger(__name__)

# ── wire_type → JSON Schema type ──
# The manifest's per-port type is a coarse 7-value enum, not JSON Schema.
# DEPTH / POSE / ANY are left unconstrained ({}) — POSE may arrive as
# [x,y,z], a dict, or an __ndarray__ marker; ANY is by definition open.
# IMAGE-as-input is a base64 PNG string (deserialized before the handler).
_WIRE_JSON_TYPE = {
    "TEXT": "string",
    "BOOL": "boolean",
    "ACTION": "integer",
    "IMAGE": "string",
    "METRICS": "object",
}


# ── pure: schema synthesis (manifest dict in → dict out, no server) ──


def _infer_type(default: Any) -> str:
    """JSON type of a ``text``/``textarea`` config field, inferred from its
    default — because many numeric knobs are declared ``field_type="text"``
    with a numeric default and cast in ``forward`` (bool must precede int:
    bool ⊂ int)."""
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    return "string"


def _config_field_property(cf: dict) -> Any:
    """One ``ui_config.config_fields[]`` entry → a JSON Schema property
    fragment. Returns None for display-only ``label`` fields (skip). All
    config fields are optional (they carry defaults), so none joins
    ``required``."""
    ftype = cf.get("field_type")
    if ftype == "label":
        return None
    label = cf.get("label") or ""
    default = cf.get("default")
    if ftype == "select":
        opts = cf.get("options") or []
        prop: dict = {"enum": [o["value"] for o in opts]}
        if default is not None:
            prop["default"] = default
    elif ftype == "slider":
        prop = {"type": "number"}
        for key, src in (("minimum", cf.get("min")), ("maximum", cf.get("max")),
                         ("multipleOf", cf.get("step")), ("default", default)):
            if src is not None:
                prop[key] = src
    elif ftype == "toggle":
        prop = {"type": "boolean"}
        if default is not None:
            prop["default"] = default
    else:  # text / textarea (and any unknown field_type) — infer from default
        prop = {"type": _infer_type(default)}
        if default is not None:
            prop["default"] = default
    if label:
        prop["description"] = label
    return prop


def _input_port_property(port: dict) -> dict:
    """One input PortSchema dict → a JSON Schema property fragment.
    DEPTH/POSE/ANY stay unconstrained so the model can pass whatever the
    node accepts."""
    prop: dict = {}
    jtype = _WIRE_JSON_TYPE.get(port.get("wire_type", ""))
    if jtype:
        prop["type"] = jtype
    desc = port.get("description")
    if desc:
        prop["description"] = desc
    return prop


def build_input_schema(fn: dict) -> dict:
    """A manifest function dict → its MCP ``inputSchema`` (a JSON Schema
    object). Properties = the union of input ports and config fields; required = ports with
    ``optional=False``. Input ports win a name collision with a config field
    (ports are the required data path)."""
    properties: dict = {}
    required: list = []

    for cf in (fn.get("ui_config") or {}).get("config_fields") or []:
        name = cf.get("name")
        if not name:
            continue
        prop = _config_field_property(cf)
        if prop is not None:
            properties[name] = prop

    for port in fn.get("input_ports") or []:
        name = port.get("name")
        if not name:
            continue
        properties[name] = _input_port_property(port)  # port wins on collision
        if not port.get("optional"):
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ── pure: output mapping ──


def _compact(value: Any) -> Any:
    """Recursively strip the base64 payload out of ``__ndarray__`` markers so
    a DEPTH map or a raw_obs dict full of arrays doesn't dump megabytes into
    the model's context. Shape/dtype are kept so the model still knows what
    was there. Base64-in-TEXT ports are plain strings, not markers →
    untouched."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return {"__ndarray__": "<omitted>",
                    "dtype": value.get("dtype"), "shape": value.get("shape")}
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def outputs_to_content(fn: dict, outputs: dict) -> list:
    """A JSON-regime outputs dict → MCP content list, mapped by output-port
    wire_type. IMAGE string ports → one ImageContent each (already base64
    PNG, no decode); everything else is compacted into one trailing
    TextContent(JSON). Ports absent from ``outputs`` are skipped."""
    images: list = []
    text_dict: dict = {}
    for port in fn.get("output_ports") or []:
        name = port.get("name")
        if name not in outputs:
            continue
        value = outputs[name]
        if port.get("wire_type") == "IMAGE" and isinstance(value, str):
            images.append(types.ImageContent(type="image", data=value,
                                             mimeType="image/png"))
        else:
            text_dict[name] = _compact(value)
    content: list = list(images)
    if text_dict or not images:
        content.append(types.TextContent(type="text",
                                         text=json.dumps(text_dict, default=str)))
    return content


# ── pure: argument routing ──


def split_args(fn: dict, arguments: dict) -> "tuple[dict, dict]":
    """Split the model's flat kwargs into handler inputs vs config. Keys
    naming an input port go to inputs, keys naming a config field go to
    config; anything else is dropped (the inputSchema already gate-kept, so
    this rarely fires)."""
    config_names = {cf.get("name")
                    for cf in (fn.get("ui_config") or {}).get("config_fields") or []}
    port_names = {p.get("name") for p in fn.get("input_ports") or []}
    inputs: dict = {}
    config: dict = {}
    for key, val in (arguments or {}).items():
        if key in port_names:
            inputs[key] = val
        elif key in config_names:
            config[key] = val
    return inputs, config


def select_functions(manifest: dict, allowlist: Any) -> list:
    """Manifest functions filtered by the allowlist. A function is kept if
    its full node_type or its ``__``-suffix is listed; falsy ⇒ keep all."""
    fns = manifest.get("functions") or []
    if not allowlist:
        return list(fns)
    wanted = set(allowlist)
    return [f for f in fns
            if f.get("name") in wanted or f.get("name", "").split("__")[-1] in wanted]


def _structured_error(exc: Exception) -> dict:
    """A handler failure → a structured dict the model can read, instead of
    an opaque isError traceback."""
    return {"error": str(exc), "type": type(exc).__name__}


# ── wiring: ServerApp → MCP Server → ASGI ──


def build_mcp_server(server_app: Any, allowlist: Any = None) -> Server:
    """Wire a lowlevel MCP ``Server`` whose tools are the given ServerApp's
    manifest functions. ``list_tools`` and ``call_tool`` close over the same
    selected set so they always agree; ``call_tool`` invokes the registered
    handler in-process with the /call route's exact semantics (JSON-regime
    port serialization, 2-vs-3-arg arity contract, no exec_ctx grants)."""
    manifest = server_app.get_manifest().to_dict()  # registers functions too
    fns = select_functions(manifest, allowlist)
    by_name = {f["name"]: f for f in fns}
    server = Server(manifest.get("name") or "nodeset")

    @server.list_tools()
    async def _list_tools() -> "list[types.Tool]":
        return [types.Tool(name=f["name"],
                           description=f.get("description", ""),
                           inputSchema=build_input_schema(f))
                for f in fns]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list:
        fn = by_name.get(name)
        if fn is None:  # unreachable — MCP only dispatches registered tools
            return [types.TextContent(type="text",
                                      text=json.dumps({"error": f"unknown tool {name}"}))]
        sf = server_app._functions.get(name)
        if sf is None or sf.handler is None:
            return [types.TextContent(type="text",
                                      text=json.dumps({"error": f"no handler for {name}"}))]

        raw_inputs, config = split_args(fn, arguments or {})
        port_map = server_app._port_maps[name]
        inputs: dict = {}
        for port_name, wire_type in port_map["inputs"].items():
            if port_name in raw_inputs:
                inputs[port_name] = deserialize_value(raw_inputs[port_name], wire_type)

        # Same arity contract as /call: 3-arg (auto-generated) handlers take
        # an exec_ctx — always None here, MCP calls carry no container grants.
        try:
            nparams = len(inspect.signature(sf.handler).parameters)
        except (ValueError, TypeError):
            nparams = 2

        try:
            if nparams >= 3:
                result = await sf.handler(inputs, config, None)
            else:
                result = await sf.handler(inputs, config)
        except Exception as e:
            log.exception("MCP call_tool %s failed", name)
            return [types.TextContent(type="text",
                                      text=json.dumps(_structured_error(e)))]
        finally:
            # Same call-granularity event drain as /call, so server-node logs
            # buffered during the call still reach the executor bridge.
            event_push.flush()

        # Dynamic Fire-List handover is a graph-engine protocol — it has no
        # meaning for an MCP client, which cannot dispatch child fires.
        from ..components.bases import FireList as _FireList

        if isinstance(result, _FireList):
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"{name} returned a FireList — graph-engine only, "
                          f"not callable over MCP"}))]

        outputs: dict = {}
        for port_name, wire_type in port_map["outputs"].items():
            if port_name in result:
                outputs[port_name] = serialize_value(result[port_name], wire_type)
        return outputs_to_content(fn, outputs)

    return server


# Exclusive servers reap a session that has gone quiet for this long, so a
# crashed client (no DELETE) can't hold the simulator lock forever. Generous
# enough for agent think-gaps between tool calls.
EXCLUSIVE_IDLE_TIMEOUT_S = 600.0


def _session_header(scope: dict) -> Any:
    for key, value in scope.get("headers") or []:
        if key == b"mcp-session-id":
            return value
    return None


async def _reject_second_session(name: str, send: Any) -> None:
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": "server-error",
        "error": {
            "code": -32000,
            "message": (
                f"nodeset {name!r} is mcp_exclusive: another MCP session is "
                f"active. Retry after it disconnects (DELETE its session, or "
                f"idle timeout {EXCLUSIVE_IDLE_TIMEOUT_S:.0f}s)."
            ),
        },
    }).encode()
    await send({"type": "http.response.start", "status": 409,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


class McpPathShim:
    """ASGI middleware dispatching ``/mcp`` (and ``/mcp/``) to the MCP
    transport, everything else to the wrapped app.

    Used instead of ``app.mount("/mcp", ...)`` because Starlette's Mount
    307-redirects the bare mount path to ``/mcp/`` — an extra round trip
    for redirect-following MCP clients and a hard failure for ones that
    don't re-POST. The transport routes by HTTP method, not path, so
    dispatching both spellings straight to it is exact."""

    def __init__(self, app: Any, mcp_asgi: Any) -> None:
        self._app = app
        self._mcp_asgi = mcp_asgi

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            await self._mcp_asgi(scope, receive, send)
            return
        await self._app(scope, receive, send)


def build_streamable_asgi(
    server_app: Any, allowlist: Any = None, exclusive: bool = False,
) -> "tuple[StreamableHTTPSessionManager, Callable[..., Awaitable[None]]]":
    """The ``/mcp`` mount: (session manager, ASGI handler). The manager's
    ``run()`` context must wrap the serving period; ``server_app`` enters it
    on startup and exits it on shutdown. ``json_response=True``: plain JSON
    replies, no SSE stream per call.

    Non-exclusive (stateless tools, model nodesets): every request gets a
    fresh transport, concurrent MCP clients are fine. Exclusive (stateful
    envs): sessions are tracked, and at most one may be live — a second
    initialize (a POST without a session id while one is tracked) is
    rejected with 409 + a JSON-RPC error the client can read. The lock is
    released by the client's DELETE or by the idle-timeout reaper.

    Security matches the existing /call surface (no auth, CORS ``*``): DNS
    rebinding protection stays off. Bind to 127.0.0.1 when only local MCP
    clients should reach the process."""
    mcp_server = build_mcp_server(server_app, allowlist)
    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=not exclusive,
        session_idle_timeout=EXCLUSIVE_IDLE_TIMEOUT_S if exclusive else None,
    )
    name = getattr(server_app, "name", "nodeset")

    async def asgi(scope: dict, receive: Any, send: Any) -> None:
        if (
            exclusive
            and scope.get("type") == "http"
            and scope.get("method") == "POST"
            and _session_header(scope) is None
            # ``_server_instances`` is the manager's session-tracking dict
            # (stateful mode only). DELETE-terminated transports stay in it
            # by design (only crashed sessions are evicted), so the gate
            # counts sessions whose transport is still live.
            and any(not t.is_terminated for t in manager._server_instances.values())
        ):
            await _reject_second_session(name, send)
            return
        await manager.handle_request(scope, receive, send)

    return manager, asgi
