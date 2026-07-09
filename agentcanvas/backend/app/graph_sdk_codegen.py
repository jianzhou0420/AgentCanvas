"""Compile a :class:`app.graph_def.GraphDefinition` back into a standalone
``graph_sdk`` builder script — the inverse of authoring a graph in Python.

This is the codegen side of the Graph SDK surface (roadmap F4): take any
graph — authored on the canvas, loaded from JSON, or built in code — and emit
runnable Python that rebuilds the *same* topology through
:class:`app.graph_sdk.Graph`.  The emitted module is self-contained: it
imports only ``app.graph_sdk`` and reconstructs every node, wire, container,
access grant and hook.

    from app.graph_def import GraphDefinition
    from app.graph_sdk_codegen import graph_to_code
    gd = GraphDefinition.from_dict(json.load(open("graph.json")))
    print(graph_to_code(gd))

Round-trip guarantee: the graph the emitted script builds is *semantically*
identical to the input (node types + configs, wire multiset, containers,
grants, hooks, budget) — modulo UI-only noise (positions, edge ids, and the
synthesised iterIn ``ports``, which re-synthesise from ``initPorts``).
"""

from __future__ import annotations

import keyword
import re
from typing import Any

from .graph_def import GraphDefinition, NodeDef


def _ident(node_id: str, used: set[str]) -> str:
    """A unique, valid Python identifier for a node id (used as its local var)."""
    base = re.sub(r"\W", "_", node_id)
    if not base or base[0].isdigit():
        base = f"n_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_"
    cand, i = base, 1
    while cand in used:
        cand, i = f"{base}_{i}", i + 1
    used.add(cand)
    return cand


def _clean_node_config(node_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Drop UI-only / synthesised keys so codegen emits authored config only."""
    cfg = dict(config or {})
    cfg.pop("_expanded", None)  # canvas fold state
    if node_type == "iterIn":
        cfg.pop("ports", None)  # synthesised from initPorts + paired iterOut
    return cfg


def _config_src(config: dict[str, Any]) -> str:
    """Render a config dict as trailing call arguments.

    All-identifier keys become ``k=<repr>`` kwargs (the readable common case);
    otherwise the whole dict is spread as ``**{...}``.
    """
    if not config:
        return ""
    if all(isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k) for k in config):
        return "".join(f", {k}={v!r}" for k, v in config.items())
    return f", **{config!r}"


def _render_graph_body(
    gd: GraphDefinition,
    var: str,
    indent: str,
    subfuncs: list[str],
    used_funcs: set[str],
) -> list[str]:
    """Emit the lines that build one graph level into local ``var``.

    Composite (subgraph) nodes recurse: each gets a top-level ``_sub_*``
    builder appended to ``subfuncs``.
    """
    lines: list[str] = []
    ctor = (
        f"{indent}{var} = Graph(name={gd.name!r}"
        + (f", description={gd.description!r}" if gd.description else "")
        + (f", eval_graph={gd.eval_graph!r}" if gd.eval_graph else "")
        + f", step_budget={gd.step_budget!r})"
    )
    lines.append(ctor)

    # node id → local var name
    used_vars: set[str] = set()
    idvar: dict[str, str] = {nd.id: _ident(nd.id, used_vars) for nd in gd.nodes}

    # -- nodes --
    lines.append("")
    lines.append(f"{indent}# nodes")
    for nd in gd.nodes:
        v = idvar[nd.id]
        cfg = _clean_node_config(nd.type, nd.config)
        if nd.subgraph is not None:
            sub_fn = _sub_builder(nd, subfuncs, used_funcs)
            ctype = "" if nd.type == "compositeNode" else f", type={nd.type!r}"
            lines.append(
                f"{indent}{v} = {var}.composite({nd.id!r}, {sub_fn}(){ctype}{_config_src(cfg)})"
            )
        else:
            lines.append(f"{indent}{v} = {var}.add({nd.type!r}, id={nd.id!r}{_config_src(cfg)})")

    # -- containers --
    if gd.containers:
        lines.append("")
        lines.append(f"{indent}# state containers")
        for c in gd.containers:
            states = {name: sd.to_dict() for name, sd in c.states.items()}
            label = f", label={c.label!r}" if c.label else ""
            lines.append(f"{indent}{var}.container({c.id!r}{label}, states={states!r})")

    # -- access grants --
    if gd.access_grants:
        lines.append("")
        lines.append(f"{indent}# access grants")
        for ag in gd.access_grants:
            lines.append(
                f"{indent}{var}.grant({ag.node_id!r}, {ag.container_id!r}, id={ag.id!r})"
            )

    # -- wires --
    if gd.edges:
        lines.append("")
        lines.append(f"{indent}# wires")
        for e in gd.edges:
            sv = idvar.get(e.source, e.source)
            dv = idvar.get(e.target, e.target)
            lines.append(
                f"{indent}{var}.connect({sv}.out({e.sourceHandle!r}), "
                f"{dv}.in_({e.targetHandle!r}))"
            )

    # -- hooks --
    if gd.hooks:
        lines.append("")
        lines.append(f"{indent}# lifecycle hooks")
        for h in gd.hooks:
            extra = ""
            if h.match_node_type != "*":
                extra += f", match_node_type={h.match_node_type!r}"
            if h.match_node_id is not None:
                extra += f", match_node_id={h.match_node_id!r}"
            if h.timeout_ms != 1000:
                extra += f", timeout_ms={h.timeout_ms!r}"
            if not h.enabled:
                extra += ", enabled=False"
            lines.append(f"{indent}{var}.hook({h.event!r}, {h.command!r}{extra})")

    lines.append("")
    lines.append(f"{indent}return {var}")
    return lines


def _sub_builder(node: NodeDef, subfuncs: list[str], used_funcs: set[str]) -> str:
    """Emit a top-level ``_sub_*`` builder for a composite node's subgraph."""
    fn = _ident(f"_sub_{node.id}", used_funcs)
    body = _render_graph_body(node.subgraph, "sg", "    ", subfuncs, used_funcs)
    subfuncs.append(f"def {fn}() -> Graph:\n" + "\n".join(body) + "\n")
    return fn


def graph_to_code(gd: GraphDefinition, *, func_name: str = "build", var: str = "g") -> str:
    """Return standalone ``graph_sdk`` builder source for ``gd``."""
    subfuncs: list[str] = []
    used_funcs: set[str] = {func_name}
    body = _render_graph_body(gd, var, "    ", subfuncs, used_funcs)

    header = (
        '"""Auto-generated by app.graph_sdk_codegen — rebuilds '
        f"{gd.name!r} via the Graph SDK.\n\n"
        "    python -m <thismodule>   # build the graph and print a summary\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "from app.graph_sdk import Graph\n\n"
    )
    parts = [header]
    parts.extend(subfuncs)
    parts.append(f"def {func_name}() -> Graph:\n" + "\n".join(body) + "\n")
    parts.append(
        '\nif __name__ == "__main__":\n'
        f"    {var} = {func_name}()\n"
        f"    print({var})\n"
    )
    return "\n".join(parts)
