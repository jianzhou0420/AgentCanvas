"""Tests for ``AutoHostClient`` — the one client of a hosted nodeset —
driven against a real ``AutoServerApp`` through an in-process ASGI
transport (no subprocess, no sockets).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import numpy as np
import pytest

from app.components.bases import BaseCanvasNode, BaseNodeSet, FireList, PortDef
from app.server.auto_server_app import AutoServerApp
from app.server.client import AutoHostClient, CallError, ServerUnreachable
from app.server.test_invoke import _NS


def _asgi_client(app_cls: type[BaseNodeSet] = _NS, **kw: Any) -> AutoHostClient:
    app = AutoServerApp(app_cls)
    asgi = app._build_app()
    return AutoHostClient("http://hosted", transport=httpx.ASGITransport(app=asgi), **kw)


def test_call_msgpack_roundtrip_with_log_and_usage() -> None:
    client = _asgi_client()
    frame = np.zeros((2, 3, 3), np.uint8)
    wire_in = {"text": "TEXT", "frame": "IMAGE"}
    wire_out = {"echo": "TEXT", "rgb": "IMAGE"}
    res = asyncio.run(
        client.call(
            "inv__echo",
            {"text": "hi", "frame": frame},
            {"n": 2, "bill": True},
            input_wire_types=wire_in,
            output_wire_types=wire_out,
        )
    )
    assert res.outputs["echo"] == "hihi"
    assert isinstance(res.outputs["rgb"], np.ndarray) and res.outputs["rgb"].shape == (2, 3, 3)
    assert res.log == [{"key": "prompt", "value": "echo:hi"}]
    assert res.usage["total_tokens"] == 7
    assert res.firelist is None
    assert res.transport["req_bytes"] > 0 and res.transport["resp_bytes"] > 0


def test_call_json_only_regime_keeps_base64_images() -> None:
    client = _asgi_client(json_only=True)
    res = asyncio.run(
        client.call(
            "inv__echo",
            {"text": "a"},
            input_wire_types={"text": "TEXT"},
            output_wire_types={"echo": "TEXT", "rgb": "IMAGE"},
        )
    )
    assert res.outputs["echo"] == "a"
    # the JSON regime decodes IMAGE ports back to ndarray on this side ...
    assert isinstance(res.outputs["rgb"], np.ndarray)
    # ... and without wire types the raw base64 string passes through
    raw = asyncio.run(client.call("inv__echo", {"text": "a"}))
    assert isinstance(raw.outputs["rgb"], str)


def test_call_firelist_is_rebuilt() -> None:
    res = asyncio.run(_asgi_client().call("inv__spawn", {}))
    assert isinstance(res.firelist, FireList)
    assert res.firelist.specs[0].node_type == "inv__echo"
    assert res.outputs == {}


def test_call_server_error_raises_call_error_with_status() -> None:
    with pytest.raises(CallError) as ei:
        asyncio.run(_asgi_client().call("inv__boom", {}))
    assert ei.value.status == 500 and "kaboom" in str(ei.value)
    with pytest.raises(CallError) as ei2:
        asyncio.run(_asgi_client().call("inv__nope", {}))
    assert ei2.value.status == 404


def test_unreachable_server_raises_server_unreachable() -> None:
    def _down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = AutoHostClient("http://down", transport=httpx.MockTransport(_down))
    with pytest.raises(ServerUnreachable):
        asyncio.run(client.call("inv__echo", {"text": "a"}))
    with pytest.raises(ServerUnreachable):
        client.call_sync("inv__echo", {"text": "a"})
    assert client.health_sync() is False


def test_sync_call_and_health_and_manifest() -> None:
    def _serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/manifest":
            return httpx.Response(
                200, json={"name": "m", "version": "1", "description": "", "functions": []}
            )
        if request.url.path == "/call/f":
            return httpx.Response(
                200, json={"outputs": {"y": 1}, "log": [{"key": "k", "value": 2}]}
            )
        return httpx.Response(404, json={"detail": "nope"})

    client = AutoHostClient("http://mock", json_only=True, transport=httpx.MockTransport(_serve))
    assert client.health_sync() is True
    assert client.manifest_sync().name == "m"
    res = client.call_sync("f", {"a": 1})
    assert res.outputs == {"y": 1} and res.log == [{"key": "k", "value": 2}]
    with pytest.raises(CallError, match="nope"):
        client.panel_info_sync()


class _PanelNode(BaseCanvasNode):
    node_type = "pan__noop"
    input_ports: list = []
    output_ports = [PortDef("ok", "BOOL", "")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"ok": True}


class _Panel:
    name = "pan"
    display_name = "Pan"
    fields: list = []
    actions: list = []

    def __init__(self) -> None:
        self.seen: list = []

    async def on_load(self) -> dict:
        return {"state": "idle"}

    async def on_field_change(self, name: str, value: Any) -> dict:
        self.seen.append((name, value))
        return {"ok": True}

    async def on_action(self, name: str, params: dict) -> dict:
        return {"ok": True, "action": name, "params": params}

    async def get_options(self, field: str) -> list:
        return [{"value": field}]


class _PanelNS(BaseNodeSet):
    name = "pan"
    env_panel = _Panel

    def get_tools(self) -> list:
        return [_PanelNode()]


def test_env_panel_routes() -> None:
    client = _asgi_client(_PanelNS)

    async def run() -> None:
        assert await client.panel_state() == {"state": "idle"}
        assert await client.panel_options("split") == [{"value": "split"}]
        assert (await client.panel_field("split", "val_unseen"))["ok"] is True
        act = await client.panel_action("play", {"x": 1})
        assert act == {"ok": True, "action": "play", "params": {"x": 1}}

    asyncio.run(run())
