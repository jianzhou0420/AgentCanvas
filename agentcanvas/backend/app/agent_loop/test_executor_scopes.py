"""Executor scope tests — V1 backward-compat + multi-scope behavior.

Two test surfaces:

1. **Single-scope golden**: a minimal 1-scope graph runs and produces
   exactly the pre-refactor step_counter / stop behavior. Asserts
   the per-scope code path is bit-equivalent to the legacy single-scope
   path in 0/1-scope graphs.

2. **Multi-scope nested**: a 2-scope graph (outer + inner) runs and
   produces the expected per-scope counter advance, inner-vs-outer
   stop semantics, and graphOut latch behavior.

These tests use a tiny custom node (``_FireCounter``) registered into
the global handler registry, plus a minimal session stub. No env, no
LLM, no IO.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..components.bases import BaseCanvasNode, PortDef
from ..graph_def import EdgeDef, GraphDefinition, NodeDef
from .builtin_nodes import register_node
from .graph_executor import GraphExecutor

# ── tiny test node + session ────────────────────────────────────────────


class _FireCounter(BaseCanvasNode):
    """Test-only node — fires when triggered, increments state['total_fires'],
    optionally toggles state['done'] after N fires (configurable via
    ``config.done_after``).
    """

    node_type = "_test_fire_counter"
    display_name = "Fire Counter (test)"
    category = "control"
    icon = "Hash"
    input_ports = [PortDef("trigger", "ANY", optional=True)]
    output_ports = [
        PortDef("count", "ANY"),
        PortDef("done", "BOOL"),
        PortDef("value", "ANY"),
    ]

    async def forward(self, inputs, ctx):
        # _NodeStateProxy uses attribute access (__getattr__/__setattr__)
        # for persistent state, NOT subscript. ctx.fires returns None for
        # uninitialised keys.
        # ``fires`` is the per-scope counter: increments each fire, resets
        # to 0 after emitting done=true so scope re-entry starts fresh.
        # ``total_fires`` accumulates across all fires (used by tests to
        # assert total invocation count).
        n = (ctx.fires or 0) + 1
        total = (ctx.total_fires or 0) + 1
        ctx.total_fires = total
        # Record the trigger value received this fire (None when unwired) —
        # lets tests assert WHICH iteration's value reached a consumer.
        ctx.last_trigger = inputs.get("trigger")
        done_after = int(self.config.get("done_after", -1) or -1)
        done = bool(done_after > 0 and n >= done_after)
        if done:
            ctx.fires = 0  # reset for next scope re-entry
        else:
            ctx.fires = n
        return {"count": n, "done": done, "value": n}


# Register once at import time
register_node(_FireCounter)


class _StubSession:
    """Minimal session shape for GraphExecutor.run()."""

    _status = "idle"
    _current_step = 0
    _metrics: dict | None = None
    _execution_id = "test"
    principles = None

    def _ws(self, msg_type: str, data: Any = None):
        from ..models import WSMessage

        return WSMessage(type=msg_type, data=data, execution_id=self._execution_id)


def _node(id: str, type: str, label: str = "", **config) -> NodeDef:
    return NodeDef(id=id, type=type, label=label or id, config=dict(config))


def _edge(id: str, source: str, target: str, sh: str = "value", th: str = "trigger") -> EdgeDef:
    return EdgeDef(id=id, source=source, target=target, sourceHandle=sh, targetHandle=th)


def _run(
    graph: GraphDefinition, step_budget: int = 50, step_budget_override: int | None = None
) -> GraphExecutor:
    """Drive a graph through a fresh executor and return it for inspection.

    Calls ``_synthesize_iterin_ports`` first because direct GraphDefinition
    construction (used by these tests) bypasses the from_dict load path
    that normally invokes it.
    """
    from ..graph_def import _synthesize_iterin_ports

    _synthesize_iterin_ports(graph)
    if graph.step_budget is None:
        graph.step_budget = step_budget
    exe = GraphExecutor()
    sess = _StubSession()
    asyncio.run(
        exe.run(
            graph,
            sess,
            step_delay_ms=0,
            step_budget_override=step_budget_override,
        )
    )
    return exe


# ── 1. Single-scope golden tests ────────────────────────────────────────


def _single_scope_graph(loop_iters: int = 3) -> GraphDefinition:
    """Build a minimal 1-scope graph: counter inside iter loop, stops
    (via the iterOut ``stop`` input) when count == loop_iters.

    Note: iter_in emits prefixed slot names (init_x from initPorts.x).
    See `_synthesize_iterin_ports` for the convention.

    Need a seed wired into iter_in's init side so it receives a non-empty
    port slot — without it the iterIn never fires (mirrors real graphs
    where env_reset seeds the loop).
    """
    return GraphDefinition(
        name="ss",
        eval_graph=False,
        step_budget=loop_iters * 4,  # generous; stop will fire first
        nodes=[
            _node("seed", "_test_fire_counter"),  # fires once at run-start
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
        ],
        edges=[
            _edge("e0", "seed", "iter_in", sh="value", th="init_x"),
            _edge("e2", "iter_in", "body", sh="init_x", th="trigger"),
            _edge("e3", "body", "iter_out", sh="done", th="stop"),
            _edge("e4", "body", "iter_out", sh="value", th="v"),
        ],
    )


def test_single_scope_step_counter_advances_per_iter() -> None:
    """1-scope graph with body terminating after 3 fires: step_counter
    should reach 3, scope_state[outermost].step_counter == 3."""
    g = _single_scope_graph(loop_iters=3)
    exe = _run(g)
    assert exe.terminated is True
    assert exe.step_counter == 3
    outer_id = exe._outermost_scope_id
    assert outer_id == "iter_in"
    assert exe.scope_state[outer_id].step_counter == 3
    assert exe.scope_state[outer_id].terminated is True
    body = exe.nodes["body"]
    assert body.state["total_fires"] == 3


def test_single_scope_step_budget_exhausts_when_no_termination() -> None:
    """No stop wired; budget should cap iteration."""
    g = GraphDefinition(
        name="ss_budget",
        eval_graph=False,
        step_budget=5,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node("body", "_test_fire_counter"),  # never sets done
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
        ],
        edges=[
            _edge("e0", "seed", "iter_in", sh="value", th="init_x"),
            _edge("e2", "iter_in", "body", sh="init_x", th="trigger"),
            _edge("e3", "body", "iter_out", sh="value", th="v"),
        ],
    )
    exe = _run(g)
    assert exe.step_counter == 5
    assert exe.terminated is True


# ── 2. Multi-scope nested tests ─────────────────────────────────────────


def _two_scope_nested_graph(
    outer_iters: int = 3,
    inner_iters: int = 5,
) -> GraphDefinition:
    """Nested 2-scope graph:

      iter_in_o ──→ outer_body ──→ graph_in_a ──→ iter_in_in (init side)
                                                                  ↓
                    (done → stop) ←── inner_body ──→ iter_out_in ──final_r──→ graph_out_x
                                                                                     ↓
      iter_out_o.stop ←── outer_body.done          graph_out_x ──→ iter_out_o.s

    For test simplicity: outer_body fires once per outer iter, triggers
    inner scope (which runs ``inner_iters`` then stops via its iterOut's
    stop input), then outer_body's done flag stops the outer iterOut at
    ``outer_iters``.
    """
    return GraphDefinition(
        name="nested2",
        eval_graph=False,
        step_budget=outer_iters * 10,  # outer cap; inner has its own budget
        nodes=[
            # seed
            _node("seed", "_test_fire_counter"),
            # outer scope
            _node(
                "iter_in_o",
                "iterIn",
                pairedWith="iter_out_o",
                step_budget=outer_iters * 2,
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node("outer_body", "_test_fire_counter", done_after=outer_iters),
            _node(
                "iter_out_o",
                "iterOut",
                pairedWith="iter_in_o",
                ports=[{"name": "s", "wire_type": "ANY"}],
            ),
            # inner scope
            _node("graph_in_a", "graphIn", portName="a", wireType="ANY"),
            _node(
                "iter_in_in",
                "iterIn",
                pairedWith="iter_out_in",
                step_budget=inner_iters * 2,
                initPorts=[{"name": "a", "wire_type": "ANY", "persist": True}],
            ),
            _node("inner_body", "_test_fire_counter", done_after=inner_iters),
            _node(
                "iter_out_in",
                "iterOut",
                pairedWith="iter_in_in",
                ports=[{"name": "r", "wire_type": "ANY"}],
            ),
            _node("graph_out_x", "graphOut", portName="x", wireType="ANY"),
        ],
        edges=[
            # seed → iter_in_o init side
            _edge("e0", "seed", "iter_in_o", sh="value", th="init_x"),
            # outer entry: iter_in_o emits init_x → outer_body
            _edge("e2", "iter_in_o", "outer_body", sh="init_x", th="trigger"),
            # outer body fires inner scope (graph_in_a as IO bridge)
            _edge("e3", "outer_body", "graph_in_a", sh="value", th="value"),
            _edge("e4", "graph_in_a", "iter_in_in", sh="value", th="init_a"),
            # iter_in_in emits init_a → inner_body
            _edge("e6", "iter_in_in", "inner_body", sh="init_a", th="trigger"),
            _edge("e7", "inner_body", "iter_out_in", sh="done", th="stop"),
            _edge("e8", "inner_body", "iter_out_in", sh="value", th="r"),
            # final side: inner terminal value bridges out via graphOut latch
            _edge("e9", "iter_out_in", "graph_out_x", sh="final_r", th="value"),
            # outer continuation
            _edge("e10", "graph_out_x", "iter_out_o", sh="value", th="s"),
            _edge("e11", "outer_body", "iter_out_o", sh="done", th="stop"),
        ],
    )


def test_nested_scope_inner_runs_inner_iters_per_outer_iter() -> None:
    """Outer 0→3 outer-iters; per outer iter inner runs inner_iters fires.
    Total inner_body fires = outer_iters x inner_iters = 15."""
    g = _two_scope_nested_graph(outer_iters=3, inner_iters=5)
    exe = _run(g)
    # Outer body fires once per outer iter
    assert exe.nodes["outer_body"].state["total_fires"] == 3
    # Inner body fires inner_iters times per outer iter
    assert exe.nodes["inner_body"].state["total_fires"] == 3 * 5, (
        f"expected 15 inner fires, got {exe.nodes['inner_body'].state['total_fires']}"
    )
    # Per-scope counters
    assert exe.scope_state["iter_in_o"].step_counter == 3
    # Inner scope counter: re-initialized at each outer iter, last value
    # after final outer iter is the final inner count for that iter.
    # (We don't accumulate across outer iters because each outer iter
    # rebuilds the inner scope's state via Initialize.)
    # End state: inner.terminated=True (inner_body sets done at fire 5)
    assert exe.scope_state["iter_in_in"].terminated is True
    # Outer terminated by outer_body's done flag at outer_iters
    assert exe.scope_state["iter_in_o"].terminated is True


def test_nested_inner_termination_does_not_kill_outer() -> None:
    """Inner termination should NOT halt the run; outer must complete its
    own iterations."""
    g = _two_scope_nested_graph(outer_iters=2, inner_iters=3)
    exe = _run(g)
    # Outer must reach 2 iters
    assert exe.nodes["outer_body"].state["total_fires"] == 2
    # Inner must run 2 outer x 3 inner = 6 times
    assert exe.nodes["inner_body"].state["total_fires"] == 6


def test_nested_outer_termination_kills_run() -> None:
    """Outer termination should set self.terminated=True (root)."""
    g = _two_scope_nested_graph(outer_iters=1, inner_iters=2)
    exe = _run(g)
    assert exe.terminated is True


def test_nested_per_scope_step_budget_outer_independent_of_inner() -> None:
    """Inner step_budget=2 caps inner regardless of outer budget."""
    g = _two_scope_nested_graph(outer_iters=3, inner_iters=99)  # inner won't reach 99
    # Override inner step_budget via direct edit
    inner_node = next(n for n in g.nodes if n.id == "iter_in_in")
    inner_node.config["step_budget"] = 2
    inner_body_node = next(n for n in g.nodes if n.id == "inner_body")
    inner_body_node.config["done_after"] = 99  # never fires done
    exe = _run(g)
    # Inner capped to 2 fires per outer iter (budget exhaust); outer still
    # completes its 3 iters (outer_body reaches done_after=3)
    assert exe.nodes["inner_body"].state["total_fires"] == 3 * 2, (
        f"inner should fire 6 (outer_iters=3 x inner_budget=2), "
        f"got {exe.nodes['inner_body'].state['total_fires']}"
    )


def test_nested_graphout_latches_inner_value_to_outer() -> None:
    """graphOut should buffer inner value to state['latched_value'] and
    propagate on inner termination — NOT every inner iter."""
    g = _two_scope_nested_graph(outer_iters=2, inner_iters=4)
    exe = _run(g)
    # graphOut should have latched the LAST inner value (= 4) on the last
    # outer iter. Since the buffer is overwritten each fire, end state
    # holds inner_iters' final value.
    graphout = exe.nodes["graph_out_x"]
    assert graphout.state.get("latched_value") == 4, (
        f"expected graphOut to latch inner final count=4, got {graphout.state.get('latched_value')}"
    )
