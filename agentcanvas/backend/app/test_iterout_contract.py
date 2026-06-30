"""Validator tests for the two-sided iterOut contract.

Covers the load-time rules added with the termination-node removal
(2026-06-11): removed-type rejection, iterOut handle validation (stop in,
final_* out), after-loop band purity, and the eval graphOut source rule.
"""

from __future__ import annotations

import pytest

from .graph_def import (
    EdgeDef,
    GraphDefinition,
    NodeDef,
    _synthesize_iterin_ports,
    validate_graph_connectivity,
)


def _validate(graph: GraphDefinition) -> None:
    _synthesize_iterin_ports(graph)
    validate_graph_connectivity(graph)


def _node(id: str, type: str, **config) -> NodeDef:
    return NodeDef(id=id, type=type, label=id, config=dict(config))


def _edge(id: str, source: str, target: str, sh: str = "value", th: str = "value") -> EdgeDef:
    return EdgeDef(id=id, source=source, target=target, sourceHandle=sh, targetHandle=th)


def _loop_graph(extra_nodes=(), extra_edges=(), eval_graph=False) -> GraphDefinition:
    """Minimal valid single-scope loop: note→iterIn(init)→note→iterOut."""
    return GraphDefinition(
        name="vg",
        eval_graph=eval_graph,
        nodes=[
            _node("seed", "note"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node("body", "note"),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
            *extra_nodes,
        ],
        edges=[
            _edge("e0", "seed", "iter_in", sh="out", th="init_x"),
            _edge("e1", "iter_in", "body", sh="init_x", th="in"),
            _edge("e2", "body", "iter_out", sh="done", th="stop"),
            _edge("e3", "body", "iter_out", sh="out", th="v"),
            *extra_edges,
        ],
    )


def _errors_of(graph: GraphDefinition) -> str:
    with pytest.raises(ValueError) as exc:
        _validate(graph)
    return str(exc.value)


def test_valid_loop_graph_passes() -> None:
    _validate(_loop_graph())


def test_termination_node_rejected_with_migration_hint() -> None:
    g = _loop_graph(extra_nodes=[_node("term", "termination")])
    msg = _errors_of(g)
    assert "termination" in msg and "stop" in msg


def test_edge_from_iterout_must_use_final_handles() -> None:
    g = _loop_graph(
        extra_nodes=[_node("sink", "note")],
        extra_edges=[_edge("e4", "iter_out", "sink", sh="v", th="in")],
    )
    msg = _errors_of(g)
    assert "final-side only" in msg


def test_final_handles_accepted() -> None:
    g = _loop_graph(
        extra_nodes=[_node("sink", "note")],
        extra_edges=[
            _edge("e4", "iter_out", "sink", sh="final_v", th="in"),
            _edge("e5", "iter_out", "sink", sh="final_stop", th="trigger"),
        ],
    )
    _validate(g)


def test_unknown_handle_into_iterout_rejected() -> None:
    g = _loop_graph(extra_edges=[_edge("e4", "body", "iter_out", sh="out", th="bogus")])
    msg = _errors_of(g)
    assert "into iterOut" in msg and "bogus" in msg


def test_after_loop_band_purity_rejects_in_loop_feed() -> None:
    """An after-loop node fed by a loop-body node (not via the final side)
    is rejected — verdict inputs must ride the pivot."""
    g = _loop_graph(
        extra_nodes=[_node("verdict", "note")],
        extra_edges=[
            _edge("e4", "iter_out", "verdict", sh="final_stop", th="trigger"),
            _edge("e5", "body", "verdict", sh="out", th="extra"),
        ],
    )
    msg = _errors_of(g)
    assert "ride the pivot" in msg


def test_eval_graphout_must_be_fed_from_band() -> None:
    """eval graph: a graph-scope graphOut fed from the loop body is the
    last-write-wins bug shape — rejected."""
    g = _loop_graph(
        eval_graph=True,
        extra_nodes=[_node("g_out", "graphOut", portName="metrics")],
        extra_edges=[_edge("e4", "body", "g_out", sh="out", th="value")],
    )
    msg = _errors_of(g)
    assert "after-loop band" in msg


def test_eval_graphout_from_final_side_passes() -> None:
    g = _loop_graph(
        eval_graph=True,
        extra_nodes=[_node("g_out", "graphOut", portName="metrics")],
        extra_edges=[_edge("e4", "iter_out", "g_out", sh="final_v", th="value")],
    )
    _validate(g)
