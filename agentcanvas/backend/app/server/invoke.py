"""invoke — one firing of a server-hosted node, in-process.

The server side of the ``/call`` contract (ADR-server-004), owned once:
resolve the function, decode the inputs for the wire regime, fire the
handler with a fresh log buffer and usage bucket, encode the outputs. Both
``POST /call/{fn}`` and the MCP ``tools/call`` projection are adapters over
:func:`invoke`; neither re-implements decode → fire → encode.

A hosted node is a long-lived instance serving many calls, unlike the
executor's fresh instance per firing. The per-call log buffer
(``_current_node_log``) and usage bucket (``_current_node_usage``) give
each call its own, so a node's ``_self_log`` entries and its LLM usage
come back on the wire and the proxy folds them into the executor's firing
— a hosted node logs exactly like a local one.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from ..components.bases import FireList, firelist_to_dict
from ..errors import get_bus
from ..standard.telemetry import _current_node_log, _current_node_usage, new_usage_bucket
from .serialization import deserialize_value, serialize_value

log = logging.getLogger("agentcanvas.server-invoke")


class InvokeError(RuntimeError):
    """The handler raised. ``error_type`` names the original exception."""

    def __init__(self, message: str, error_type: str = "Exception") -> None:
        super().__init__(message)
        self.error_type = error_type


class UnknownFunction(LookupError):
    """No function of that name on this server."""


class NoHandler(LookupError):
    """The function is declared but has no handler."""


@dataclass
class Invocation:
    """What one call produced, encoded for its wire regime."""

    outputs: dict[str, Any] = field(default_factory=dict)
    # The envelope of a FireList the handler returned (outputs is then {}).
    firelist: dict | None = None
    # The node's voluntary ``_self_log`` entries for this call.
    log: list[dict] = field(default_factory=list)
    # LLM usage accumulated during this call, or None when no call was made.
    usage: dict | None = None


def handler_arity(handler: Any) -> int:
    """How many positional parameters a handler takes. Auto-generated
    handlers take ``(inputs, config, exec_ctx)``; manual ``ServerFunction``
    handlers take ``(inputs, config)``. Inspected, not try/except, so an
    internal TypeError is never mistaken for a signature mismatch."""
    try:
        return len(inspect.signature(handler).parameters)
    except (ValueError, TypeError):
        return 2


async def invoke(
    server_app: Any,
    function_name: str,
    raw_inputs: dict,
    config: dict | None,
    exec_ctx: dict | None = None,
    *,
    json_regime: bool = False,
) -> Invocation:
    """Fire ``function_name`` on ``server_app`` once.

    ``json_regime`` is the wire's: JSON callers (the migration window, MCP)
    need per-port ``deserialize_value`` / ``serialize_value``; msgpack
    callers pass native values both ways. ``exec_ctx`` is the cross-nodeset
    container grant (face B), handed only to 3-arg handlers.

    Raises :class:`UnknownFunction`, :class:`NoHandler`, or
    :class:`InvokeError` (the handler raised; the error was pushed to the
    executor's event bus first). Buffered events are flushed on every exit.
    """
    server_app._register_functions()
    fn = server_app._functions.get(function_name)
    if fn is None:
        raise UnknownFunction(function_name)
    if fn.handler is None:
        raise NoHandler(function_name)

    port_map = server_app._port_maps[function_name]
    inputs: dict[str, Any] = {}
    for port_name, wire_type in port_map["inputs"].items():
        if port_name in raw_inputs:
            val = raw_inputs[port_name]
            inputs[port_name] = deserialize_value(val, wire_type) if json_regime else val
    config = config or {}

    call_log: list[dict] = []
    usage = new_usage_bucket()
    log_token = _current_node_log.set(call_log)
    usage_token = _current_node_usage.set(usage)
    try:
        if handler_arity(fn.handler) >= 3:
            result = await fn.handler(inputs, config, exec_ctx)
        else:
            result = await fn.handler(inputs, config)
    except Exception as e:
        log.exception("Handler %s failed", function_name)
        # First-class the error (Move 3, #54): the process's sink carries it
        # to the parent with node / execution scope, so it surfaces on the
        # canvas like a local-node failure.
        execution_id = exec_ctx.get("execution_id") if isinstance(exec_ctx, dict) else None
        get_bus().from_exception(
            e,
            source="node",
            code="SUBPROC_NODE_FAIL",
            scope={
                "node_id": function_name,
                "nodeset": getattr(server_app, "name", None),
                "execution_id": execution_id,
            },
            title=f"Node {function_name} failed in {getattr(server_app, 'name', 'server')}",
        )
        raise InvokeError(str(e), type(e).__name__) from e
    finally:
        _current_node_usage.reset(usage_token)
        _current_node_log.reset(log_token)
        # Drain events buffered during this call so server-node logs reach
        # the canvas at call granularity.
        get_bus().flush()

    usage_out = usage if usage["calls"] > 0 else None
    if usage_out is not None:
        usage_out["usd_cost"] = round(usage_out["usd_cost"], 6)

    if isinstance(result, FireList):
        return Invocation(firelist=firelist_to_dict(result), log=call_log, usage=usage_out)

    outputs: dict[str, Any] = {}
    for port_name, wire_type in port_map["outputs"].items():
        if port_name in result:
            val = result[port_name]
            outputs[port_name] = serialize_value(val, wire_type) if json_regime else val
    return Invocation(outputs=outputs, log=call_log, usage=usage_out)
