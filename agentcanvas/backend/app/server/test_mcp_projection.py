"""Tests for the native MCP projection (``mcp_projection.py``).

Tier 0: pure schema-synthesis / output-mapping functions on canned manifest
dicts. Tier 1: a dummy nodeset served through ``AutoServerApp`` — the /mcp
``tools/list`` must reconcile with ``GET /manifest``. Tier 2: real
``tools/call`` roundtrips in-process (TextContent JSON + ImageContent PNG),
plus the ``--mcp-tools`` allowlist. Run from the backend dir:

    python -m pytest app/server/test_mcp_projection.py -v
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp", reason="mcp SDK not installed (py3.8 host)")

from fastapi.testclient import TestClient

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)
from app.server import mcp_projection as P
from app.server.auto_server_app import AutoServerApp

# ── canned manifest function dicts (Tier 0) ──

_FN = {
    "name": "env_dummy__step",
    "description": "One step.",
    "input_ports": [
        {"name": "action", "wire_type": "ACTION", "description": "Discrete action", "optional": False},
        {"name": "trigger", "wire_type": "BOOL", "description": "", "optional": True},
        {"name": "depth", "wire_type": "DEPTH", "description": "", "optional": True},
    ],
    "output_ports": [
        {"name": "rgb", "wire_type": "IMAGE", "description": "", "optional": False},
        {"name": "info", "wire_type": "METRICS", "description": "", "optional": True},
    ],
    "config_schema": {},
    "ui_config": {
        "config_fields": [
            {"name": "mode", "field_type": "select", "label": "Mode",
             "default": "fast", "options": [{"value": "fast", "label": "Fast"},
                                            {"value": "slow", "label": "Slow"}]},
            {"name": "gain", "field_type": "slider", "label": "Gain",
             "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "verbose", "field_type": "toggle", "label": "", "default": False},
            {"name": "retries", "field_type": "text", "label": "Retries", "default": 3},
            {"name": "note", "field_type": "label", "label": "Display only", "default": "x"},
            # Name-collides with the input port — the port must win.
            {"name": "action", "field_type": "text", "label": "Shadowed", "default": "no"},
        ]
    },
}


def test_infer_type_bool_before_int() -> None:
    assert P._infer_type(True) == "boolean"
    assert P._infer_type(3) == "integer"
    assert P._infer_type(0.5) == "number"
    assert P._infer_type("x") == "string"
    assert P._infer_type(None) == "string"


def test_build_input_schema() -> None:
    schema = P.build_input_schema(_FN)
    props = schema["properties"]
    assert schema["type"] == "object"
    # Ports: ACTION → integer + required; optional BOOL → boolean, not required;
    # DEPTH → unconstrained.
    assert props["action"] == {"type": "integer", "description": "Discrete action"}
    assert props["trigger"] == {"type": "boolean"}
    assert props["depth"] == {}
    assert schema["required"] == ["action"]
    # Config fields: select → enum, slider → bounded number, toggle → boolean,
    # text with int default → integer, label → skipped entirely.
    assert props["mode"] == {"enum": ["fast", "slow"], "default": "fast",
                             "description": "Mode"}
    assert props["gain"] == {"type": "number", "minimum": 0.0, "maximum": 2.0,
                             "multipleOf": 0.5, "default": 1.0, "description": "Gain"}
    assert props["verbose"] == {"type": "boolean", "default": False}
    assert props["retries"] == {"type": "integer", "default": 3,
                                "description": "Retries"}
    assert "note" not in props


def test_select_functions_and_split_args() -> None:
    manifest = {"functions": [_FN, {"name": "env_dummy__reset"}]}
    assert len(P.select_functions(manifest, None)) == 2
    assert [f["name"] for f in P.select_functions(manifest, ["step"])] == ["env_dummy__step"]
    assert [f["name"] for f in P.select_functions(manifest, ["env_dummy__reset"])] == ["env_dummy__reset"]
    inputs, config = P.split_args(_FN, {"action": 2, "mode": "slow", "junk": 1})
    assert inputs == {"action": 2}
    assert config == {"mode": "slow"}


def test_compact_strips_ndarray_payload() -> None:
    marker = {"__ndarray__": "QUFB", "dtype": "float32", "shape": [2, 2]}
    out = P._compact({"a": marker, "b": [marker, 1], "c": "plain"})
    assert out["a"] == {"__ndarray__": "<omitted>", "dtype": "float32", "shape": [2, 2]}
    assert out["b"][0]["__ndarray__"] == "<omitted>"
    assert out["c"] == "plain"


def test_outputs_to_content_shapes() -> None:
    b64 = base64.b64encode(b"png-bytes").decode()
    content = P.outputs_to_content(_FN, {"rgb": b64, "info": {"sr": 1.0}})
    assert content[0].type == "image" and content[0].data == b64
    assert content[-1].type == "text"
    assert json.loads(content[-1].text) == {"info": {"sr": 1.0}}
    # Image-only outputs: no trailing empty TextContent.
    only_img = P.outputs_to_content(_FN, {"rgb": b64})
    assert [c.type for c in only_img] == ["image"]
    # No outputs at all: one empty-JSON TextContent, never [].
    empty = P.outputs_to_content(_FN, {})
    assert [c.type for c in empty] == ["text"] and json.loads(empty[0].text) == {}


# ── dummy nodeset (Tier 1/2) ──


class _EchoNode(BaseCanvasNode):
    node_type = "dummy__echo"
    display_name = "Echo"
    description = "Echo text back, optionally repeated."
    input_ports = [
        PortDef("text", "TEXT", "Text to echo"),
        PortDef("count", "ACTION", "Repeat count", optional=True),
    ]
    output_ports = [PortDef("echo", "TEXT", "The echoed text")]
    ui_config = NodeUIConfig(config_fields=[
        ConfigField(name="sep", field_type="text", label="Separator", default="-"),
    ])

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        sep = self.config.get("sep", "-")
        count = int(inputs.get("count") or 1)
        return {"echo": sep.join([inputs["text"]] * count)}


class _FrameNode(BaseCanvasNode):
    node_type = "dummy__frame"
    display_name = "Frame"
    description = "A tiny RGB frame plus a note."
    input_ports = [PortDef("trigger", "BOOL", "", optional=True)]
    output_ports = [
        PortDef("rgb", "IMAGE", "2x2 test frame"),
        PortDef("note", "TEXT", "Side channel"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "note": "ok"}


class _DummyNodeSet(BaseNodeSet):
    name = "dummy"
    description = "MCP projection test nodeset"

    def get_tools(self) -> list:
        return [_EchoNode(), _FrameNode()]


def _rpc(client: TestClient, method: str, params: Any = None, id_: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        body["params"] = params
    resp = client.post("/mcp", json=body, headers={
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18",
    })
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "error" not in payload, payload
    return payload["result"]


@pytest.fixture()
def client() -> Any:
    app = AutoServerApp(_DummyNodeSet)
    with TestClient(app._build_app()) as c:
        yield c


def test_tools_list_reconciles_with_manifest(client: TestClient) -> None:
    manifest = client.get("/manifest").json()
    tools = _rpc(client, "tools/list")["tools"]
    by_name = {t["name"]: t for t in tools}
    # Same function set, same descriptions, schema rebuilt from the same dicts.
    assert set(by_name) == {f["name"] for f in manifest["functions"]}
    for f in manifest["functions"]:
        tool = by_name[f["name"]]
        assert tool["description"] == f["description"]
        assert tool["inputSchema"] == P.build_input_schema(f)
    assert by_name["dummy__echo"]["inputSchema"]["required"] == ["text"]


def test_tools_call_text_roundtrip(client: TestClient) -> None:
    result = _rpc(client, "tools/call", {
        "name": "dummy__echo",
        "arguments": {"text": "hi", "count": 2, "sep": "+"},
    }, id_=2)
    assert result.get("isError") in (None, False)
    text_blocks = [c for c in result["content"] if c["type"] == "text"]
    assert json.loads(text_blocks[-1]["text"]) == {"echo": "hi+hi"}


def test_tools_call_image_roundtrip(client: TestClient) -> None:
    result = _rpc(client, "tools/call", {
        "name": "dummy__frame", "arguments": {},
    }, id_=3)
    kinds = [c["type"] for c in result["content"]]
    assert kinds == ["image", "text"]
    img = result["content"][0]
    assert img["mimeType"] == "image/png"
    assert base64.b64decode(img["data"])[:4] == b"\x89PNG"
    assert json.loads(result["content"][1]["text"]) == {"note": "ok"}


def test_mcp_tools_allowlist() -> None:
    app = AutoServerApp(_DummyNodeSet)
    app.mcp_tools = ["echo"]
    with TestClient(app._build_app()) as c:
        tools = _rpc(c, "tools/list")["tools"]
        assert [t["name"] for t in tools] == ["dummy__echo"]
        # The manifest protocol is never narrowed by the MCP allowlist.
        assert len(c.get("/manifest").json()["functions"]) == 2


def test_manifest_and_call_untouched(client: TestClient) -> None:
    """The projection is additive: the manifest protocol still answers."""
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/call/dummy__echo", json={"inputs": {"text": "a"}})
    assert resp.json()["outputs"]["echo"] == "a"


# ── mcp_exclusive session policy (P2) ──


class _StatefulNodeSet(_DummyNodeSet):
    name = "dummy_env"
    statefulness = "stateful"  # stateful → exclusive by derivation


def test_mcp_exclusive_resolution() -> None:
    assert AutoServerApp(_DummyNodeSet).mcp_exclusive is False
    assert AutoServerApp(_StatefulNodeSet).mcp_exclusive is True
    # The auto_host --mcp-exclusive flag overrides the derivation both ways.
    assert AutoServerApp(_StatefulNodeSet, mcp_exclusive="off").mcp_exclusive is False
    assert AutoServerApp(_DummyNodeSet, mcp_exclusive="on").mcp_exclusive is True


_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}


def _initialize(client: TestClient) -> Any:
    return client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}},
    }, headers=_MCP_HEADERS)


def test_exclusive_session_gate() -> None:
    app = AutoServerApp(_StatefulNodeSet)
    with TestClient(app._build_app()) as c:
        # First client: full stateful handshake, then a real call.
        r1 = _initialize(c)
        assert r1.status_code == 200, r1.text
        sid = r1.headers["mcp-session-id"]
        with_sid = dict(_MCP_HEADERS, **{"mcp-session-id": sid})
        n = c.post("/mcp", json={"jsonrpc": "2.0",
                                 "method": "notifications/initialized"},
                   headers=with_sid)
        assert n.status_code in (200, 202), n.text
        r = c.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "dummy__echo", "arguments": {"text": "x"}},
        }, headers=with_sid)
        assert r.status_code == 200, r.text
        content = r.json()["result"]["content"]
        assert json.loads(content[-1]["text"]) == {"echo": "x"}

        # Second client: rejected while the first session is live.
        r2 = _initialize(c)
        assert r2.status_code == 409
        assert "mcp_exclusive" in r2.json()["error"]["message"]

        # DELETE releases the lock; a new session may then initialize.
        d = c.delete("/mcp", headers=with_sid)
        assert d.status_code in (200, 204), d.text
        r3 = _initialize(c)
        assert r3.status_code == 200, r3.text
