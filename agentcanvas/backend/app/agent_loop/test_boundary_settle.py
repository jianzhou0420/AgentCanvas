"""Terminal-step sibling-sink regression — boundary phase 2/4 (settle).

When the loop body raises the stop signal, any sibling sink fed by the
same iteration's output is queued in that wave but has not necessarily
fired when the iterOut boundary runs. The settle phase drains those
sinks BEFORE the decide (stop) check, so a sink still observes the
terminal step's value.

History: before the settle loop landed (2026-05-07), the executor
exited straight from the stop pull-check and sibling ``evaluate`` sinks
kept stale step-N-1 metrics — manifesting as 0/60 SR for RT-1-X on
SIMPLER while the env itself flagged success. The multi-scope refactor
later re-opened the hole for dead-end sinks (graph-scope, skipped by
the settle drain's same-scope filter) until the root-boundary exception
landed 2026-07-04. This file pins the fix so an executor refactor
cannot silently reintroduce either regression.

The sink edge is declared AFTER the iterOut edges on purpose: the
propagation loop then enqueues iterOut ahead of the sink, so on the
terminal wave the sink is still queued when the boundary fires — the
exact ordering the settle phase exists for.
"""

from __future__ import annotations

from ..graph_def import GraphDefinition
from .test_executor_scopes import _edge, _node, _run

_LOOP_ITERS = 3


def _sibling_sink_graph() -> GraphDefinition:
    return GraphDefinition(
        name="sibling_sink",
        eval_graph=False,
        step_budget=_LOOP_ITERS * 4,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "driver", "wire_type": "ANY", "persist": True}],
            ),
            _node("body", "_test_fire_counter", done_after=_LOOP_ITERS),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
            _node("sink", "_test_fire_counter"),
        ],
        edges=[
            _edge("e_seed", "seed", "iter_in", sh="value", th="init_driver"),
            _edge("e_drive", "iter_in", "body", sh="init_driver", th="trigger"),
            _edge("e_stop", "body", "iter_out", sh="done", th="stop"),
            _edge("e_v", "body", "iter_out", sh="value", th="v"),
            # Sibling sink fed the SAME per-iteration output — declared
            # after the iterOut edges so the terminal wave leaves it
            # queued behind the boundary (see module docstring).
            _edge("e_sink", "body", "sink", sh="value", th="trigger"),
        ],
    )


def test_sibling_sink_fires_on_terminal_step() -> None:
    """A dead-end sink is never on a path to the iterOut, so scope
    analysis leaves it in the graph scope (``''``). The settle drain's
    same-scope filter therefore has a root-boundary exception: graph-
    scope nodes are drained too, so the sink still observes the terminal
    step instead of being left in the queue when the run exits (fixed
    2026-07-04; before that it fired N-1 times with the step-N-1 value).
    """
    exe = _run(_sibling_sink_graph())
    sink = exe.nodes["sink"].state
    # Every iteration including the terminal one — not loop_iters - 1.
    assert sink["total_fires"] == _LOOP_ITERS
    # And it saw the terminal step's value, not a stale step-N-1 one.
    assert sink.get("last_trigger") == _LOOP_ITERS
