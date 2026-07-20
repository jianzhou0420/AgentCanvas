"""Concept-level, manifest-driven generic MCP bridge (概念级尝试).

Reads a running auto_host's ``GET /manifest`` and auto-exposes each nodeset
function as an MCP tool — no hand-coding per tool. This is the generic form of
what ``bridges/mcp_bridge.py`` does by hand for env_habitat: instead of writing
``observe``/``step``/``look_around`` by name, it derives the whole tool surface
from the manifest the auto_host already publishes.

Pure HTTP client — does NOT spawn auto_host (the coding-agent driver owns server
lifecycle). Runs in the ``agentcanvas`` env (has ``mcp`` + ``requests``); the
auto_host runs separately in its own env (e.g. ``ac-vlnce`` for env_habitat).

NOT a replacement for the frozen ``bridges/mcp_bridge.py`` and NOT on the
experiment path. It proves the concept "a nodeset IS an MCP server in all but
format"; it deliberately drops the hand bridge's *policy* layer, which the
manifest cannot carry:

  * no action batching — a manifest ``step_discrete`` is one action per call,
    where the frozen ``step`` takes a list with early-halt (turn budget explodes;
    unusable for real nav, fine for a concept demo);
  * no STOP-gate (``_stop_armed`` placement confirmation) — raw ``action=0``
    ends the episode immediately;
  * no budget broadcast, no clearance readout, no ``look_around`` composite,
    no live spectating, no BARE toggle;
  * port-level over-exposure — the ``--tools`` allowlist narrows *tools*, not
    *ports*, so a generic ``observe`` still returns depth/pose/intrinsics/raw_obs
    (compacted) alongside rgb.

Experiment-safety that IS preserved: with an allowlist, ``reset``/``evaluate``
are simply not registered, so the agent cannot re-roll episodes or read SR/SPL.

Usage::

    python coding-agent/bridges/nodeset_mcp.py \
        --server-url http://127.0.0.1:9200 \
        --tools env_habitat__observe_egocentric,env_habitat__step_discrete
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import anyio
import mcp.types as types
import requests
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# ── wire_type → JSON Schema type ──
# The manifest's per-port type is a coarse 7-value enum, not JSON Schema. DEPTH /
# POSE / ANY are left unconstrained ({}) — POSE may arrive as [x,y,z], a dict, or
# an __ndarray__ marker; ANY is by definition open. IMAGE-as-input is rare (base64
# string) but modeled so FM/SAM-style TEXT-image ports stay string-typed.
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
    default — because many numeric knobs are declared ``field_type="text"`` with
    a numeric default and cast in ``forward`` (bool must precede int: bool ⊂ int)."""
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    return "string"


def _config_field_property(cf: dict) -> dict | None:
    """One ``ui_config.config_fields[]`` entry → a JSON Schema property fragment.
    Returns None for display-only ``label`` fields (skip). All config fields are
    optional (they carry defaults), so none joins ``required``."""
    ftype = cf.get("field_type")
    if ftype == "label":
        return None
    label = cf.get("label") or ""
    default = cf.get("default")
    if ftype == "select":
        opts = cf.get("options") or []
        prop: dict[str, Any] = {"enum": [o["value"] for o in opts]}
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
    """One input PortDef → a JSON Schema property fragment. DEPTH/POSE/ANY stay
    unconstrained so the model can pass whatever the node accepts."""
    prop: dict[str, Any] = {}
    jtype = _WIRE_JSON_TYPE.get(port.get("wire_type", ""))
    if jtype:
        prop["type"] = jtype
    desc = port.get("description")
    if desc:
        prop["description"] = desc
    return prop


def build_input_schema(fn: dict) -> dict:
    """A manifest function → its MCP ``inputSchema`` (a JSON Schema object).
    Properties = input ports ∪ config fields; required = ports with
    ``optional=False``. Input ports win a name collision with a config field
    (ports are the required data path); doesn't occur in env_habitat."""
    properties: dict[str, dict] = {}
    required: list[str] = []

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

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ── pure: output mapping ──


def _compact(value: Any) -> Any:
    """Recursively strip the base64 payload out of ``__ndarray__`` markers so a
    DEPTH map or a raw_obs dict full of arrays doesn't dump megabytes into the
    model's context. Shape/dtype are kept so the model still knows what was there.
    Base64-in-TEXT ports (SAM-style) are plain strings, not markers → untouched."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return {"__ndarray__": "<omitted>",
                    "dtype": value.get("dtype"), "shape": value.get("shape")}
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def outputs_to_content(fn: dict, outputs: dict) -> list:
    """A ``/call`` outputs dict → MCP content list, mapped by output-port
    wire_type. IMAGE string ports → one ImageContent each (already base64 PNG in
    the JSON regime, no decode); everything else is compacted into one trailing
    TextContent(JSON). Ports absent from ``outputs`` are skipped."""
    images: list = []
    text_dict: dict[str, Any] = {}
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


def _split_args(fn: dict, arguments: dict) -> tuple[dict, dict]:
    """Split the model's flat kwargs into ``/call`` inputs vs config. Keys naming a
    config field go to config, keys naming an input port go to inputs; anything
    else is dropped (the inputSchema already gate-kept, so this rarely fires)."""
    config_names = {cf.get("name")
                    for cf in (fn.get("ui_config") or {}).get("config_fields") or []}
    port_names = {p.get("name") for p in fn.get("input_ports") or []}
    inputs: dict[str, Any] = {}
    config: dict[str, Any] = {}
    for key, val in (arguments or {}).items():
        if key in port_names:
            inputs[key] = val
        elif key in config_names:
            config[key] = val
    return inputs, config


# ── I/O: manifest + call ──


def fetch_manifest(server_url: str) -> dict:
    """GET /manifest (always plain JSON) → the nodeset's tool inventory."""
    resp = requests.get(f"{server_url.rstrip('/')}/manifest", timeout=30)
    resp.raise_for_status()
    return resp.json()


def select_functions(manifest: dict, allowlist: list[str] | None) -> list[dict]:
    """Manifest functions filtered by the allowlist. A function is kept if its
    full node_type or its ``__``-suffix is listed; None ⇒ keep all."""
    fns = manifest.get("functions") or []
    if not allowlist:
        return list(fns)
    wanted = set(allowlist)
    return [f for f in fns
            if f.get("name") in wanted or f.get("name", "").split("__")[-1] in wanted]


def _structured_error(exc: Exception) -> dict:
    """A requests failure → a structured dict the model can read, instead of an
    MCP isError traceback (pattern lifted from the stale mcp_server)."""
    resp = getattr(exc, "response", None)
    return {
        "error": str(exc),
        "status_code": getattr(resp, "status_code", None),
        "detail": (resp.text[:500] if resp is not None else None),
    }


def call_function(server_url: str, name: str, inputs: dict, config: dict) -> dict:
    """POST /call/{name} with JSON content-type — the JSON regime returns IMAGE
    ports as base64 PNG and other ndarrays as ``__ndarray__`` markers."""
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(f"{server_url.rstrip('/')}/call/{name}", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["outputs"]


# ── server wiring ──


def build_server(server_url: str, allowlist: list[str] | None, label: str | None) -> Server:
    """Fetch the manifest, select functions, and wire an MCP Server whose tools
    are generated from them. list_tools and call_tool close over the same
    selected set so they always agree (call_tool caches list_tools' defs)."""
    manifest = fetch_manifest(server_url)
    fns = select_functions(manifest, allowlist)
    by_name = {f["name"]: f for f in fns}
    server = Server(label or manifest.get("name") or "nodeset")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
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
        inputs, config = _split_args(fn, arguments)
        try:
            outputs = call_function(server_url, name, inputs, config)
        except requests.RequestException as exc:
            return [types.TextContent(type="text",
                                      text=json.dumps(_structured_error(exc)))]
        return outputs_to_content(fn, outputs)

    return server


async def _serve(server: Server) -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-url", required=True,
                    help="base URL of a running auto_host, e.g. http://127.0.0.1:9200")
    ap.add_argument("--tools", default=os.environ.get("NODESET_MCP_TOOLS", ""),
                    help="comma allowlist of node_types or __-suffixes; empty = all")
    ap.add_argument("--nodeset", default="",
                    help="MCP server label (default: manifest name)")
    args = ap.parse_args()
    allow = [t.strip() for t in args.tools.split(",") if t.strip()] or None
    server = build_server(args.server_url, allow, args.nodeset or None)
    anyio.run(_serve, server)


if __name__ == "__main__":
    main()
