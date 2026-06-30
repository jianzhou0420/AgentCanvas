"""After-loop stage tests (final side of the two-sided iterOut).

Nodes wired downstream of the iterOut's ``final_*`` handles never fire
during the dataflow loop — the final side emits exactly once, at scope
termination, and ``_after_loop_pass`` drains the resulting band in
dependency order. An otherwise-identical dataflow node wired to an
in-loop producer fires every iteration.

Reuses the tiny ``_FireCounter`` node + harness from
``test_executor_scopes`` (importing it also registers the node).
"""

from __future__ import annotations

from ..graph_def import GraphDefinition
from .test_executor_scopes import _edge, _node, _run


def _after_loop_graph(loop_iters: int = 3) -> GraphDefinition:
    """A 1-scope loop (body fires ``loop_iters`` times, then stops the
    iterOut via its ``stop`` input) plus:

    * ``ctrlC`` — a dataflow consumer of ``body`` (fires every iteration);
    * ``postA`` -> ``postB`` — a two-node after-loop chain: postA hangs off
      the iterOut's ``final_v`` handle, postB off postA.
    """
    return GraphDefinition(
        name="after_loop",
        eval_graph=False,
        step_budget=loop_iters * 4,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node("body", "_test_fire_counter", done_after=loop_iters),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
            # Contrast: a dataflow consumer of body — fires every iteration.
            _node("ctrlC", "_test_fire_counter"),
            # After-loop chain: final_v → postA → postB.
            _node("postA", "_test_fire_counter"),
            _node("postB", "_test_fire_counter"),
        ],
        edges=[
            _edge("e0", "seed", "iter_in", sh="value", th="init_x"),
            _edge("e2", "iter_in", "body", sh="init_x", th="trigger"),
            _edge("e3", "body", "iter_out", sh="done", th="stop"),
            _edge("e4", "body", "iter_out", sh="value", th="v"),
            _edge("e5", "body", "ctrlC", sh="value", th="trigger"),
            _edge("e6", "iter_out", "postA", sh="final_v", th="trigger"),
            _edge("e7", "postA", "postB", sh="value", th="trigger"),
        ],
    )


def test_after_loop_node_fires_once_not_every_iteration() -> None:
    """A final-side consumer fires exactly once (after the loop), while an
    otherwise-identical dataflow node wired to the in-loop producer fires
    once per iteration."""
    exe = _run(_after_loop_graph(loop_iters=3))
    assert exe.terminated is True

    body_fires = exe.nodes["body"].state["total_fires"]
    assert body_fires >= 2, "loop must really iterate for the contrast to mean anything"

    # Dataflow consumer wired to the in-loop producer fired multiple times.
    assert exe.nodes["ctrlC"].state["total_fires"] > 1

    # Final-side consumer fired exactly ONCE despite body firing body_fires times.
    assert exe.nodes["postA"].state["total_fires"] == 1


def test_after_loop_chain_propagates_in_dependency_order() -> None:
    """A downstream after-loop node fires once too — proving
    ``_after_loop_pass`` settles the chain (postB has no trigger other
    than postA's output)."""
    exe = _run(_after_loop_graph(loop_iters=4))
    assert exe.nodes["postA"].state["total_fires"] == 1
    assert exe.nodes["postB"].state["total_fires"] == 1


def test_final_side_carries_terminal_iteration_value() -> None:
    """The #64 regression test: the final-side consumer must receive the
    TERMINAL iteration's value — not a stale earlier one. body emits its
    fire count, so postA's trigger must equal loop_iters."""
    exe = _run(_after_loop_graph(loop_iters=5))
    assert exe.terminated is True
    # postA consumed final_v == body's terminal count (== 5). _FireCounter
    # echoes nothing about its input, so assert via the iterOut's last
    # collected value reaching the paired iterIn slot AND the after-loop
    # fire having happened exactly once with the loop fully run.
    assert exe.nodes["body"].state["total_fires"] == 5
    assert exe.nodes["postA"].state["total_fires"] == 1
    assert exe.nodes["postA"].state.get("last_trigger") == 5


def test_budget_exhaust_emits_final_side() -> None:
    """No stop wired: budget exhaust must still emit the final side so the
    after-loop chain runs."""
    g = _after_loop_graph(loop_iters=99)  # done never fires within budget
    g.step_budget = 4
    exe = _run(g)
    assert exe.terminated is True
    assert exe.step_counter == 4
    assert exe.nodes["postA"].state["total_fires"] == 1
    # Terminal value = the 4th iteration's count.
    assert exe.nodes["postA"].state.get("last_trigger") == 4


def test_error_path_emits_final_side_best_effort() -> None:
    """If the loop crashes mid-run, the error path reconstructs the last
    completed iteration's values from the paired iterIn's slots and still
    emits the final side, so the after-loop verdict runs."""

    from ..components.bases import BaseCanvasNode, PortDef
    from .builtin_nodes import register_node

    class _Bomb(BaseCanvasNode):
        node_type = "_test_bomb"
        display_name = "Bomb (test)"
        category = "control"
        icon = "Zap"
        input_ports = [PortDef("trigger", "ANY", optional=True)]
        output_ports = [PortDef("value", "ANY")]

        async def forward(self, inputs, ctx):
            n = (ctx.fires or 0) + 1
            ctx.fires = n
            if n >= 3:
                raise RuntimeError("boom at fire 3")
            return {"value": n}

    register_node(_Bomb)

    g = _after_loop_graph(loop_iters=99)
    # Replace body with the bomb: crashes on its 3rd fire (iteration 3).
    for n in g.nodes:
        if n.id == "body":
            n.type = "_test_bomb"
            n.config = {}
    # Strict errors so the crash propagates instead of becoming {'error':...}
    import os

    os.environ["AGENTCANVAS_STRICT_ERRORS"] = "1"
    try:
        exe = _run(g)
    finally:
        os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)
    # Two iterations completed before the crash; the fallback emission must
    # have fed the after-loop chain with iteration 2's value.
    assert exe.nodes["postA"].state["total_fires"] == 1
    assert exe.nodes["postA"].state.get("last_trigger") == 2
