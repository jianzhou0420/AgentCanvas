"""Example 02 — authoring sugar: loop / hook / composite.

    PYTHONPATH=. python examples/02_authoring_sugar.py

Builds (does not run) a graph exercising the three ergonomic helpers, then
serialises it to canvas JSON. No env / GPU — this is a structure demo.

* g.loop(init=…, carry=…)  → an iterIn/iterOut episode loop with the correct
  persist flags; wire it through loop.seed / feed / carry / stop.
* g.hook(event, command)   → a lifecycle shell hook.
* g.composite(id, subgraph) → a nested-subgraph composite node.
"""

from __future__ import annotations

from agentcanvas import Graph


def build() -> Graph:
    g = Graph(name="sugar-demo", eval_graph=True, step_budget=20)

    # A tiny episode loop: seed an instruction (init-only, persists), carry a
    # running "state" string across steps.
    loop = g.loop(init=[("instruction", "TEXT")], carry=[("state", "TEXT")])

    src = g.add("graphIn", id="instr_in", portName="instruction", wireType="TEXT")
    step = g.add("noop", id="step")           # stand-in for a real per-step node
    stopper = g.add("noop", id="stopper")

    # seed the run-start values, feed them to the step node, carry the result
    loop.seed("instruction", src.out("instruction"))
    loop.feed("instruction", step.in_("instruction"))
    loop.feed("state", step.in_("prev_state"))
    loop.carry("state", step.out("next_state"))
    loop.stop(stopper.out("done"))

    # a composite node wrapping a nested subgraph
    sub = Graph(name="scorer")
    si = sub.graph_in("x")
    so = sub.graph_out("score")
    sub.connect(si.out("x"), so.in_("value"))
    g.composite("scorer_box", sub, label="Scorer")

    # a lifecycle hook
    g.hook("GraphComplete", "echo done", match_node_type="*")

    return g


if __name__ == "__main__":
    g = build()
    d = g.to_dict()
    print(g)
    print(f"nodes={len(d['nodes'])} edges={len(d['edges'])} hooks={len(d.get('hooks', []))}")
    composite = next(n for n in d["nodes"] if n.get("subgraph"))
    print(f"composite {composite['id']!r} wraps {len(composite['subgraph']['nodes'])} sub-nodes")
    print("OK — loop + hook + composite built; serialises to canvas JSON.")
