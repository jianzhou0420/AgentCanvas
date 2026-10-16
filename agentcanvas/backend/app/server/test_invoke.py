"""Tests for the server-side ``invoke`` — one firing of a hosted node, in
process, with no HTTP: decode for the wire regime, fire with a fresh log
buffer and usage bucket, encode, and the error policy.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    DynamicFireListNode,
    FireList,
    FireSpec,
    PortDef,
)
from app.server.auto_server_app import AutoServerApp
from app.server.invoke import InvokeError, NoHandler, UnknownFunction, handler_arity, invoke
from app.server.server_app import ServerApp, ServerFunction
from app.standard.telemetry import _current_node_usage


class _Echo(BaseCanvasNode):
    node_type = "inv__echo"
    input_ports = [PortDef("text", "TEXT", ""), PortDef("frame", "IMAGE", "", optional=True)]
    output_ports = [PortDef("echo", "TEXT", ""), PortDef("rgb", "IMAGE", "")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        self._self_log("prompt", f"echo:{inputs['text']}")
        bucket = _current_node_usage.get()
        if bucket is not None and self.config.get("bill"):
            bucket["calls"] += 1
            bucket["total_tokens"] += 7
            bucket["model"] = "fake-model"
        frame = inputs.get("frame")
        shape = tuple(frame.shape) if frame is not None else (1, 1, 3)
        return {
            "echo": inputs["text"] * int(self.config.get("n", 1)),
            "rgb": np.zeros(shape, np.uint8),
        }


class _Boom(BaseCanvasNode):
    node_type = "inv__boom"
    input_ports: list = []
    output_ports = [PortDef("x", "TEXT", "")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        raise ValueError("kaboom")


class _Spawner(DynamicFireListNode):
    node_type = "inv__spawn"
    input_ports: list = []
    output_ports = [PortDef("plan", "TEXT", "")]

    async def forward(self, inputs: dict, ctx: Any) -> FireList:
        return FireList(
            specs=[FireSpec(node_type="inv__echo", inputs={"text": "child"}, label="c0")],
            spawner_outputs={"plan": "p"},
            aggregator={"kind": "passthrough_last"},
        )


class _NS(BaseNodeSet):
    name = "inv"

    def get_tools(self) -> list:
        return [_Echo(), _Boom(), _Spawner()]


def _app() -> AutoServerApp:
    app = AutoServerApp(_NS)
    app._register_functions()
    return app


def test_invoke_msgpack_regime_passes_native_values_and_returns_log() -> None:
    app = _app()
    frame = np.zeros((2, 3, 3), np.uint8)
    inv = asyncio.run(invoke(app, "inv__echo", {"text": "a", "frame": frame}, {"n": 2}))
    assert inv.outputs["echo"] == "aa"
    assert isinstance(inv.outputs["rgb"], np.ndarray) and inv.outputs["rgb"].shape == (2, 3, 3)
    assert inv.log == [{"key": "prompt", "value": "echo:a"}]
    assert inv.usage is None  # no LLM call happened
    assert inv.firelist is None


def test_invoke_json_regime_decodes_and_encodes_per_port() -> None:
    app = _app()
    from app.server.serialization import serialize_value

    b64 = serialize_value(np.zeros((4, 5, 3), np.uint8), "IMAGE")
    inv = asyncio.run(invoke(app, "inv__echo", {"text": "a", "frame": b64}, {}, json_regime=True))
    assert inv.outputs["echo"] == "a"
    assert isinstance(inv.outputs["rgb"], str) and inv.outputs["rgb"]  # base64 PNG, not ndarray


def test_invoke_fresh_log_buffer_per_call() -> None:
    app = _app()
    first = asyncio.run(invoke(app, "inv__echo", {"text": "1"}, {}))
    second = asyncio.run(invoke(app, "inv__echo", {"text": "2"}, {}))
    assert [e["value"] for e in first.log] == ["echo:1"]
    assert [e["value"] for e in second.log] == ["echo:2"]  # not accumulated across calls
    tool = next(t for t in app._nodeset.get_tools() if t.node_type == "inv__echo")
    assert tool._log_buffer == []  # the long-lived instance never kept them


def test_invoke_usage_bucket_comes_back() -> None:
    app = _app()
    inv = asyncio.run(invoke(app, "inv__echo", {"text": "a"}, {"bill": True}))
    assert inv.usage is not None
    assert (inv.usage["calls"], inv.usage["total_tokens"], inv.usage["model"]) == (
        1,
        7,
        "fake-model",
    )


def test_invoke_firelist_crosses_as_envelope() -> None:
    inv = asyncio.run(invoke(_app(), "inv__spawn", {}, {}))
    assert inv.outputs == {}
    assert inv.firelist is not None
    body = inv.firelist["__firelist__"]
    assert body["specs"][0]["node_type"] == "inv__echo" and body["spawner_outputs"] == {"plan": "p"}


def test_invoke_errors_are_typed() -> None:
    app = _app()
    with pytest.raises(UnknownFunction):
        asyncio.run(invoke(app, "inv__nope", {}, {}))
    with pytest.raises(InvokeError) as ei:
        asyncio.run(invoke(app, "inv__boom", {}, {}))
    assert str(ei.value) == "kaboom" and ei.value.error_type == "ValueError"


def test_invoke_manual_two_arg_handler_and_no_handler() -> None:
    from app.server.manifest import PortSchema

    class _Manual(ServerApp):
        name = "manual"

        def get_functions(self) -> list[ServerFunction]:
            async def add(inputs: dict, config: dict) -> dict:
                return {"sum": inputs["a"] + inputs["b"]}

            return [
                ServerFunction(
                    name="add",
                    input_ports=[PortSchema("a", "ANY", ""), PortSchema("b", "ANY", "")],
                    output_ports=[PortSchema("sum", "ANY", "")],
                    handler=add,
                ),
                ServerFunction(name="declared_only"),
            ]

    app = _Manual()
    inv = asyncio.run(invoke(app, "add", {"a": 1, "b": 2}, {}))
    assert inv.outputs == {"sum": 3}  # 2-arg manual handler: no exec_ctx passed
    with pytest.raises(NoHandler):
        asyncio.run(invoke(app, "declared_only", {}, {}))
    assert handler_arity(lambda a, b: None) == 2
    assert handler_arity(lambda a, b, c: None) == 3
