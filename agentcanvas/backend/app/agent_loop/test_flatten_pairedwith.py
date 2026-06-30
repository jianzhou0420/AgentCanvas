"""Tests for flatten.py's pairedWith id-rewrite pass.

Composites that contain an iteration loop inside (iterIn / iterOut paired
by `config.pairedWith`) need their `pairedWith` strings rewritten to the
post-flatten prefixed ids — otherwise scope_analysis +
_synthesize_iterin_ports cannot resolve the inner pair.

Three cases:
  1. Composite with one inner pivot pair — pairedWith prefixed,
     analyze_scopes finds exactly 1 scope at the prefixed id.
  2. Composite WITHOUT a pivot pair (DAG-only) — no false rewrites
     of unrelated config string values.
  3. Recursive: composite-inside-composite, both with their own
     pivot pair — pairedWith prefixed correctly at each level.
"""

from __future__ import annotations

from ..graph_def import EdgeDef, GraphDefinition, NodeDef
from .flatten import flatten_graph
from .scope_analysis import analyze_scopes


def _node(id: str, type: str, **config) -> NodeDef:
    return NodeDef(id=id, type=type, config=dict(config))


def _edge(id: str, source: str, target: str, sh: str = "out", th: str = "in") -> EdgeDef:
    return EdgeDef(id=id, source=source, target=target, sourceHandle=sh, targetHandle=th)


def _two_pivot_subgraph() -> GraphDefinition:
    """A minimal composite body: graphIn → iter_in(init side) → body → iter_out → graphOut."""
    return GraphDefinition(
        nodes=[
            _node("graph_in__x", "graphIn", portName="x"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                step_budget=10,
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node("body", "note"),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "y", "wire_type": "ANY", "persist": True}],
            ),
            _node("graph_out__y", "graphOut", portName="y"),
        ],
        edges=[
            _edge("e1", "graph_in__x", "iter_in", "value", "init_x"),
            _edge("e2", "iter_in", "body", "init_x", "in"),
            _edge("e3", "body", "iter_out", "done", "stop"),
            _edge("e4", "body", "iter_out", "out", "y"),
            _edge("e5", "iter_out", "graph_out__y", "final_y", "value"),
        ],
    )


# ── Case 1 ──────────────────────────────────────────────────────────────


def test_composite_with_pivot_pair_pairedwith_rewritten() -> None:
    """A composite holding one iteration loop — after flatten the
    pivots must reference each other by their prefixed ids and
    analyze_scopes must find exactly one scope.
    """
    parent = GraphDefinition(
        nodes=[
            _node("seed", "note"),
            NodeDef(
                id="ce",  # composite id, becomes the prefix
                type="composite",
                config={},
                subgraph=_two_pivot_subgraph(),
            ),
            _node("sink", "note"),
        ],
        edges=[
            _edge("pe1", "seed", "ce", "out", "x"),  # parent → composite graphIn
            _edge("pe2", "ce", "sink", "y", "in"),  # composite graphOut → parent
        ],
    )
    flat, _ = flatten_graph(parent)

    nodes_by_id = {n.id: n for n in flat.nodes}
    # Pivot ids must be prefixed
    assert "ce__iter_in" in nodes_by_id
    assert "ce__iter_out" in nodes_by_id
    # Bare ids must be gone
    assert "iter_in" not in nodes_by_id
    assert "iter_out" not in nodes_by_id

    # pairedWith must point at the prefixed ids (THE FIX)
    assert nodes_by_id["ce__iter_out"].config["pairedWith"] == "ce__iter_in"
    assert nodes_by_id["ce__iter_in"].config["pairedWith"] == "ce__iter_out"

    # analyze_scopes should now resolve exactly one scope at ce__iter_in
    forest, errs = analyze_scopes(flat)
    assert errs == [], f"unexpected errors: {errs}"
    assert list(forest.scopes.keys()) == ["ce__iter_in"]


# ── Case 2 ──────────────────────────────────────────────────────────────


def test_composite_without_pivot_pair_no_false_rewrites() -> None:
    """A DAG-only composite — no pivots, no pairedWith. Other config
    values that might coincidentally match an inner id must NOT be
    rewritten (the fix is intentionally narrow to `pairedWith`).
    """
    sub = GraphDefinition(
        nodes=[
            _node("graph_in", "graphIn", portName="x"),
            # Note: a config field whose VALUE coincidentally matches "filter"
            # (an inner id) — must be left alone by the rewrite.
            _node("filter", "note", related="filter"),
            _node("graph_out", "graphOut", portName="y"),
        ],
        edges=[
            _edge("e1", "graph_in", "filter", "value", "in"),
            _edge("e2", "filter", "graph_out", "out", "value"),
        ],
    )
    parent = GraphDefinition(
        nodes=[
            _node("seed", "note"),
            NodeDef(id="cmp", type="composite", config={}, subgraph=sub),
            _node("sink", "note"),
        ],
        edges=[
            _edge("pe1", "seed", "cmp", "out", "x"),
            _edge("pe2", "cmp", "sink", "y", "in"),
        ],
    )
    flat, _ = flatten_graph(parent)
    nodes_by_id = {n.id: n for n in flat.nodes}
    # The body node's `related` config value must NOT have been prefixed
    # (the rewrite is scoped to the `pairedWith` key only).
    assert nodes_by_id["cmp__filter"].config.get("related") == "filter"
    # And no pairedWith should exist on this DAG-only composite
    for n in flat.nodes:
        assert "pairedWith" not in (n.config or {}), f"unexpected pairedWith on {n.id}: {n.config}"


# ── Case 3 ──────────────────────────────────────────────────────────────


def test_recursive_composite_pairedwith_rewritten_at_each_level() -> None:
    """Composite-inside-composite, both with their own pivot pair.
    After (recursive) flatten, both levels' pairedWith strings must be
    prefixed with their respective parent ids.
    """
    inner_composite = NodeDef(
        id="inner",
        type="composite",
        config={},
        subgraph=_two_pivot_subgraph(),
    )

    # Outer composite contains: graphIn → outer pivot pair wrapping inner composite → graphOut
    outer_sub = GraphDefinition(
        nodes=[
            _node("graph_in__a", "graphIn", portName="a"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "a", "wire_type": "ANY", "persist": True}],
            ),
            inner_composite,  # nested composite
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "b", "wire_type": "ANY", "persist": True}],
            ),
            _node("graph_out__b", "graphOut", portName="b"),
        ],
        edges=[
            _edge("e1", "graph_in__a", "iter_in", "value", "init_a"),
            _edge("e2", "iter_in", "inner", "init_a", "x"),  # outer → inner composite
            _edge("e3", "inner", "iter_out", "y", "stop"),  # inner graphOut → outer stop
            _edge("e4", "inner", "iter_out", "y", "b"),  # inner composite graphOut → outer iter_out
        ],
    )

    parent = GraphDefinition(
        nodes=[
            _node("seed", "note"),
            NodeDef(id="outer", type="composite", config={}, subgraph=outer_sub),
            _node("sink", "note"),
        ],
        edges=[
            _edge("pe1", "seed", "outer", "out", "a"),
            _edge("pe2", "outer", "sink", "b", "in"),
        ],
    )
    flat, _ = flatten_graph(parent)
    nodes_by_id = {n.id: n for n in flat.nodes}

    # Outer-level pivots prefixed by "outer__"
    assert "outer__iter_in" in nodes_by_id
    assert "outer__iter_out" in nodes_by_id
    assert nodes_by_id["outer__iter_out"].config["pairedWith"] == "outer__iter_in"

    # Inner-level pivots prefixed by "outer__inner__" (recursive)
    assert "outer__inner__iter_in" in nodes_by_id
    assert "outer__inner__iter_out" in nodes_by_id
    assert nodes_by_id["outer__inner__iter_out"].config["pairedWith"] == "outer__inner__iter_in"

    # analyze_scopes finds two scopes: outer + nested inner
    forest, errs = analyze_scopes(flat)
    assert errs == [], f"unexpected errors: {errs}"
    assert set(forest.scopes.keys()) == {"outer__iter_in", "outer__inner__iter_in"}
    assert forest.scopes["outer__inner__iter_in"].parent_scope_id == "outer__iter_in"
