"""Unit tests for scope topology analysis (multi-scope iteration).

Hand-built graphs exercise the four shapes that matter:
  1. Empty graph (no scopes)
  2. Single scope (the legacy shape — must produce 1 root, 0 errors)
  3. Two nested scopes (outer + inner with graphIn/graphOut at boundary)
  4. Two sequential peer scopes (no nesting)

Plus validation negatives:
  5. Cross-author-scope wire bypassing graphIn/graphOut (must error)
  6. Two iterOuts claiming the same iterIn (must error)
  7. iterIn with no paired iterOut (must error)
"""

from __future__ import annotations

from ..graph_def import EdgeDef, GraphDefinition, NodeDef
from .scope_analysis import analyze_scopes


def _node(id: str, type: str, **config) -> NodeDef:
    return NodeDef(id=id, type=type, config=dict(config))


def _edge(id: str, source: str, target: str, sh: str = "out", th: str = "in") -> EdgeDef:
    return EdgeDef(id=id, source=source, target=target, sourceHandle=sh, targetHandle=th)


# ── Shape 1: empty / no-scope graph ─────────────────────────────────────


def test_empty_graph_no_scopes() -> None:
    g = GraphDefinition(nodes=[], edges=[])
    forest, errs = analyze_scopes(g)
    assert forest.scopes == {}
    assert forest.root_scope_ids == []
    assert forest.graph_scope_node_ids == []
    assert errs == []


def test_no_pivots_only_seeds() -> None:
    g = GraphDefinition(
        nodes=[_node("a", "note"), _node("b", "note")],
        edges=[_edge("e", "a", "b")],
    )
    forest, errs = analyze_scopes(g)
    assert forest.scopes == {}
    assert sorted(forest.graph_scope_node_ids) == ["a", "b"]
    assert errs == []


# ── Shape 2: single scope (legacy single-iter graph shape) ──────────────


def test_single_scope_legacy_shape() -> None:
    g = GraphDefinition(
        nodes=[
            _node("seed", "note"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "x", "wire_type": "ANY"}],
            ),
            _node("body", "note"),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "y", "wire_type": "ANY"}],
            ),
            _node("sink", "note"),
        ],
        edges=[
            _edge("e1", "seed", "iter_in", "out", "init_x"),
            _edge("e3", "iter_in", "body"),
            _edge("e4", "body", "iter_out", "done", "stop"),
            _edge("e5", "body", "iter_out"),
            _edge("e6", "iter_out", "sink", "final_y", "value"),
        ],
    )
    forest, errs = analyze_scopes(g)
    assert errs == []
    assert list(forest.scopes.keys()) == ["iter_in"]
    assert forest.root_scope_ids == ["iter_in"]
    s = forest.scopes["iter_in"]
    assert s.parent_scope_id is None
    assert s.child_scope_ids == []
    assert s.iter_out_id == "iter_out"
    assert sorted(s.member_node_ids) == ["body", "iter_in", "iter_out"]
    assert sorted(forest.graph_scope_node_ids) == ["seed", "sink"]
    assert forest.is_single_scope is True


# ── Shape 3: nested scope (outer + inner) ───────────────────────────────


def _two_scope_nested_graph() -> GraphDefinition:
    """Reusable 2-scope nested fixture used by several tests."""
    return GraphDefinition(
        nodes=[
            _node("seed", "note"),
            _node(
                "iter_in_o",
                "iterIn",
                pairedWith="iter_out_o",
                step_budget=5,
                initPorts=[{"name": "instr", "wire_type": "TEXT"}],
            ),
            _node("outer_body", "note"),
            _node("graph_in_a", "graphIn", portName="a"),
            _node(
                "iter_in_i",
                "iterIn",
                pairedWith="iter_out_i",
                step_budget=50,
                initPorts=[{"name": "a", "wire_type": "ANY"}],
            ),
            _node("inner_body", "note"),
            _node(
                "iter_out_i",
                "iterOut",
                pairedWith="iter_in_i",
                ports=[{"name": "r", "wire_type": "ANY"}],
            ),
            _node("graph_out_x", "graphOut", portName="x"),
            _node(
                "iter_out_o",
                "iterOut",
                pairedWith="iter_in_o",
                ports=[{"name": "s", "wire_type": "ANY"}],
            ),
            _node("sink", "note"),
        ],
        edges=[
            _edge("e1", "seed", "iter_in_o", "out", "init_instr"),
            _edge("e3", "iter_in_o", "outer_body"),
            _edge("e4", "outer_body", "graph_in_a", "out", "value"),
            _edge("e5", "graph_in_a", "iter_in_i", "out", "init_a"),  # intra-inner
            _edge("e7", "iter_in_i", "inner_body"),
            _edge("e8", "inner_body", "iter_out_i", "done", "stop"),
            _edge("e9", "inner_body", "iter_out_i"),
            _edge("e10", "iter_out_i", "graph_out_x", "final_r", "value"),
            _edge("e11", "graph_out_x", "iter_out_o"),  # intra-outer
            _edge("e12", "outer_body", "iter_out_o", "done", "stop"),
            _edge("e13", "iter_out_o", "sink", "final_s", "value"),
        ],
    )


def test_nested_two_scopes_topology() -> None:
    g = _two_scope_nested_graph()
    forest, errs = analyze_scopes(g)
    assert errs == [], f"unexpected errors: {errs}"
    assert set(forest.scopes.keys()) == {"iter_in_o", "iter_in_i"}
    assert forest.root_scope_ids == ["iter_in_o"]
    outer = forest.scopes["iter_in_o"]
    inner = forest.scopes["iter_in_i"]
    assert outer.parent_scope_id is None
    assert outer.child_scope_ids == ["iter_in_i"]
    assert inner.parent_scope_id == "iter_in_o"
    assert inner.child_scope_ids == []
    assert forest.is_single_scope is False


def test_nested_stop_wires_stay_in_scope() -> None:
    """The stop wires live on the pivots themselves — membership must keep
    each body + pivots inside its own scope, nothing leaks to graph scope
    except the seed and the final sink."""
    g = _two_scope_nested_graph()
    forest, _ = analyze_scopes(g)
    outer = forest.scopes["iter_in_o"]
    inner = forest.scopes["iter_in_i"]
    assert "iter_out_o" in outer.member_node_ids
    assert "iter_out_i" in inner.member_node_ids
    assert "sink" in forest.graph_scope_node_ids


def test_nested_graphin_graphout_binding() -> None:
    g = _two_scope_nested_graph()
    forest, _ = analyze_scopes(g)
    inner = forest.scopes["iter_in_i"]
    # graphIn / graphOut belong to the inner scope (their adjacent edges
    # land/originate in inner-scope nodes).
    assert inner.graphin_node_ids == ["graph_in_a"]
    assert inner.graphout_node_ids == ["graph_out_x"]


def test_nested_resolved_step_budgets() -> None:
    g = _two_scope_nested_graph()
    forest, _ = analyze_scopes(g)
    assert forest.scopes["iter_in_o"].step_budget == 5
    assert forest.scopes["iter_in_i"].step_budget == 50


def test_nested_writes_child_scope_ids_hint() -> None:
    g = _two_scope_nested_graph()
    analyze_scopes(g)  # mutates outer iter_in's config
    outer_node = next(n for n in g.nodes if n.id == "iter_in_o")
    inner_node = next(n for n in g.nodes if n.id == "iter_in_i")
    assert outer_node.config.get("nested_scope_ids") == ["iter_in_i"]
    # Inner has no children — should not have the key
    assert "nested_scope_ids" not in inner_node.config


# ── Shape 4: peer (sequential) scopes — non-nested ──────────────────────


def test_two_peer_scopes_no_nesting() -> None:
    """Two scopes that don't contain each other — both are roots."""
    g = GraphDefinition(
        nodes=[
            _node(
                "iter_in_a",
                "iterIn",
                pairedWith="iter_out_a",
                initPorts=[{"name": "x"}],
            ),
            _node("body_a", "note"),
            _node("iter_out_a", "iterOut", pairedWith="iter_in_a", ports=[{"name": "y"}]),
            _node(
                "iter_in_b",
                "iterIn",
                pairedWith="iter_out_b",
                initPorts=[{"name": "x"}],
            ),
            _node("body_b", "note"),
            _node("iter_out_b", "iterOut", pairedWith="iter_in_b", ports=[{"name": "y"}]),
        ],
        edges=[
            _edge("e2", "iter_in_a", "body_a"),
            _edge("e3", "body_a", "iter_out_a"),
            # sequential handoff: iterOut_a → iter_in_b's init side
            _edge("e4", "iter_out_a", "iter_in_b", "y", "init_x"),
            _edge("e6", "iter_in_b", "body_b"),
            _edge("e7", "body_b", "iter_out_b"),
        ],
    )
    forest, errs = analyze_scopes(g)
    assert errs == []
    assert sorted(forest.root_scope_ids) == ["iter_in_a", "iter_in_b"]
    assert forest.scopes["iter_in_a"].parent_scope_id is None
    assert forest.scopes["iter_in_b"].parent_scope_id is None


# ── Validation negatives ────────────────────────────────────────────────


def test_cross_author_scope_wire_without_port_errors() -> None:
    """Direct wire from outer body to inner body bypassing graphIn errors."""
    g = GraphDefinition(
        nodes=[
            _node(
                "iter_in_o",
                "iterIn",
                pairedWith="iter_out_o",
                initPorts=[{"name": "x"}],
            ),
            _node("outer_body", "note"),
            _node(
                "iter_in_i",
                "iterIn",
                pairedWith="iter_out_i",
                initPorts=[{"name": "a"}],
            ),
            _node("inner_body", "note"),
            _node("iter_out_i", "iterOut", pairedWith="iter_in_i", ports=[{"name": "r"}]),
            _node("iter_out_o", "iterOut", pairedWith="iter_in_o", ports=[{"name": "s"}]),
        ],
        edges=[
            _edge("e2", "iter_in_o", "outer_body"),
            _edge("e3", "outer_body", "inner_body"),  # ! direct cross-scope, should error
            _edge("e5", "iter_in_i", "inner_body"),
            _edge("e6", "inner_body", "iter_out_i"),
            _edge("e7", "iter_out_i", "iter_out_o"),
            _edge("e8", "outer_body", "iter_out_o"),
        ],
    )
    _forest, errs = analyze_scopes(g)
    assert any("crosses an author scope boundary" in e for e in errs), errs


def test_duplicate_pairedWith_errors() -> None:
    """Two iterOuts both claim pairedWith=iter_in_x → error."""
    g = GraphDefinition(
        nodes=[
            _node("iter_in_x", "iterIn", pairedWith="iter_out_a"),
            _node("iter_out_a", "iterOut", pairedWith="iter_in_x", ports=[{"name": "y"}]),
            _node("iter_out_b", "iterOut", pairedWith="iter_in_x", ports=[{"name": "y"}]),
        ],
        edges=[],
    )
    _, errs = analyze_scopes(g)
    assert any("Two iterOut nodes claim pairedWith" in e for e in errs), errs


def test_iter_in_without_paired_iter_out_errors() -> None:
    g = GraphDefinition(
        nodes=[_node("ii", "iterIn", pairedWith="not_a_real_io")],
        edges=[],
    )
    _, errs = analyze_scopes(g)
    assert any("no paired iterOut" in e for e in errs), errs
