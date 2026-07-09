"""Runnable proof that AgentCanvas can be driven Graph SDK.

    python -m app.graph_sdk_demo

Defines three tiny pure-Python nodes, builds a graph with the Graph SDK
:class:`app.graph_sdk.Graph` builder, runs it in-process (no FastAPI, no
canvas GUI), and shows that the same graph round-trips through canvas JSON.
No env, no GPU, no LLM — this is the equivalent of the executor unit tests,
exercised through the public Graph SDK surface.
"""

from __future__ import annotations

from app.agent_loop.builtin_nodes import register_node
from app.graph_sdk import Graph
from app.components.bases import BaseCanvasNode, PortDef
from app.graph_def import GraphDefinition

# ── three tiny pure-Python demo nodes ────────────────────────────────────


class ConstNode(BaseCanvasNode):
    """Emits ``config.value`` on its single output port."""

    node_type = "demo_const"
    input_ports: list[PortDef] = []
    output_ports = [PortDef("value", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"value": self.config.get("value", 0)}


class AddNode(BaseCanvasNode):
    """Fires when BOTH inputs arrive; emits their sum."""

    node_type = "demo_add"
    input_ports = [PortDef("a", "ANY"), PortDef("b", "ANY")]
    output_ports = [PortDef("result", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"result": inputs["a"] + inputs["b"]}


class ScaleNode(BaseCanvasNode):
    """Multiplies its input by ``config.factor``."""

    node_type = "demo_scale"
    input_ports = [PortDef("x", "ANY")]
    output_ports = [PortDef("y", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"y": inputs["x"] * self.config.get("factor", 1)}


for _cls in (ConstNode, AddNode, ScaleNode):
    register_node(_cls)


# ── build the graph in Python ────────────────────────────────────────────


def build() -> Graph:
    """(7 + 5) * 3 == 36, expressed as a Graph SDK graph."""
    g = Graph(name="graph_sdk_demo", eval_graph=False)

    c7 = g.add("demo_const", value=7)
    c5 = g.add("demo_const", value=5)
    add = g.add("demo_add")
    scale = g.add("demo_scale", factor=3)
    out = g.graph_out("total")

    # fluent wiring — `>>` is sugar for g.connect(...)
    c7.out("value") >> add.in_("a")
    c5.out("value") >> add.in_("b")
    add.out("result") >> scale.in_("x")
    scale.out("y") >> out.in_("value")

    return g


def main() -> None:
    g = build()
    print(f"built: {g!r}")

    result = g.run(validate=True)
    print(f"outputs: {result.outputs}")
    print(f"  add node state:   {result.node_state('demo_add_0')}")
    print(f"  scale node state: {result.node_state('demo_scale_0')}")
    assert result["total"] == 36, result["total"]
    print("in-process run OK  →  (7 + 5) * 3 == 36")

    # round-trip: code → JSON → dataclass → run again, identical result
    graph_json = g.to_json()
    reloaded_def: GraphDefinition = GraphDefinition.from_dict(g.to_dict())
    reloaded = Graph.from_definition(reloaded_def)
    result2 = reloaded.run()
    assert result2["total"] == 36, result2["total"]
    print(f"round-trip OK      →  JSON ({len(graph_json)} chars) reloaded, still 36")
    print("\n--- canvas-loadable JSON (first 400 chars) ---")
    print(graph_json[:400])


if __name__ == "__main__":
    main()
