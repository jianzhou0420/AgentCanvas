"""Example 01 — build a pure-Python graph and run it in-process.

    PYTHONPATH=. python examples/01_pure_python.py     # from agentcanvas/backend

No backend, no GPU, no nodesets: three tiny BaseCanvasNode subclasses wired
(7 + 5) * 3 = 36, run entirely in-process. This is the "hello world" of the
Graph SDK API — everything downstream is the same three moves (add / connect /
run) on bigger node palettes.
"""

from __future__ import annotations

from agentcanvas import Graph
from app.agent_loop.builtin_nodes import register_node
from app.components.bases import BaseCanvasNode, PortDef


class ExConst(BaseCanvasNode):
    node_type = "ex_const"
    input_ports: list[PortDef] = []
    output_ports = [PortDef("value", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"value": self.config.get("value", 0)}


class ExAdd(BaseCanvasNode):
    node_type = "ex_add"
    input_ports = [PortDef("a", "ANY"), PortDef("b", "ANY")]
    output_ports = [PortDef("sum", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"sum": inputs["a"] + inputs["b"]}


class ExScale(BaseCanvasNode):
    node_type = "ex_scale"
    input_ports = [PortDef("x", "ANY")]
    output_ports = [PortDef("y", "ANY")]

    async def forward(self, inputs: dict, ctx) -> dict:
        return {"y": inputs["x"] * self.config.get("factor", 1)}


for _cls in (ExConst, ExAdd, ExScale):
    register_node(_cls)


def build() -> Graph:
    g = Graph(name="pure-python-demo")
    seven = g.add("ex_const", id="seven", value=7)
    five = g.add("ex_const", id="five", value=5)
    add = g.add("ex_add", id="add")
    scale = g.add("ex_scale", id="scale", factor=3)
    total = g.graph_out("total")

    g.connect(seven.out("value"), add.in_("a"))
    g.connect(five.out("value"), add.in_("b"))
    g.connect(add.out("sum"), scale.in_("x"))
    g.connect(scale.out("y"), total.in_("value"))
    return g


if __name__ == "__main__":
    result = build().run(validate=True)
    print("result:", result["total"])
    assert result["total"] == 36, result["total"]
    print("OK — (7 + 5) * 3 = 36, run in-process with no backend.")
