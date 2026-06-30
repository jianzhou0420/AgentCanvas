"""Auto-layout regression tests (``app/layout.py``).

These pin the two-pivot redesign (2026-06-17): the init/loop *band* split was
removed, so a graph lays out as a single left→right flow::

    seed / init ──► iterIn ──► loop body … ──► iterOut

The run-start seeds that feed ``iterIn`` are ordinary DAG predecessors, so they
land to the LEFT of ``iterIn`` on the same spine — no vertical band, no gap.

Run: ``cd agentcanvas/backend && python -m pytest app/test_layout.py -v``
"""

from __future__ import annotations

from app.layout import H_SPACING, layout_graph


def _node(nid: str, ntype: str, **config):
    n: dict = {"id": nid, "type": ntype}
    if config:
        n["config"] = config
    return n


def _edge(src: str, tgt: str):
    return {"source": src, "target": tgt}


def _loop_graph(seeds: list[str]):
    """Build a minimal loop: seeds → iterIn → body → iterOut → (back) iterIn."""
    nodes = [_node(s, "promptTemplate") for s in seeds]
    nodes += [
        _node("iter_in", "iterIn"),
        _node("body", "llmCall"),
        _node("iter_out", "iterOut", pairedWith="iter_in"),
    ]
    edges = [_edge(s, "iter_in") for s in seeds]
    edges += [
        _edge("iter_in", "body"),
        _edge("body", "iter_out"),
        _edge("iter_out", "iter_in"),  # back-edge (pairedWith iter_in)
    ]
    return {"nodes": nodes, "edges": edges}


def _pos(result: dict) -> dict[str, dict]:
    return {n["id"]: n["position"] for n in result["nodes"]}


def test_single_seed_loop_is_one_flow():
    """Seed, iterIn, body, iterOut lie on one spine (same y), x strictly L→R.

    This is the core of the band removal: the seed is on the *same horizontal
    line* as the loop body, not stacked in a separate band above it.
    """
    result = layout_graph(_loop_graph(["seed"]))
    p = _pos(result)

    # One spine → identical y for the whole chain (no up/down band split).
    ys = {p["seed"]["y"], p["iter_in"]["y"], p["body"]["y"], p["iter_out"]["y"]}
    assert len(ys) == 1, f"expected one spine lane, got ys={ys}"

    # Strictly left→right in flow order.
    assert p["seed"]["x"] < p["iter_in"]["x"] < p["body"]["x"] < p["iter_out"]["x"]


def test_seed_is_one_layer_left_of_iterin():
    """The seed feeds iterIn, so iterIn sits exactly one column to its right.

    Also proves the iterOut→iterIn back-edge is excluded from layering: were it
    kept, seed/iterIn/body/iterOut would form a cycle and collapse to layer 0.
    """
    result = layout_graph(_loop_graph(["seed"]))
    p = _pos(result)
    assert p["iter_in"]["x"] == p["seed"]["x"] + H_SPACING
    assert p["iter_out"]["x"] == p["seed"]["x"] + 3 * H_SPACING


def test_iterout_is_rightmost():
    """iterOut is the last node in the flow → maximum x."""
    result = layout_graph(_loop_graph(["seed"]))
    p = _pos(result)
    max_x = max(v["x"] for v in p.values())
    assert p["iter_out"]["x"] == max_x


def test_multi_seed_all_left_of_iterin():
    """Every run-start seed lands strictly left of iterIn (none stacked above)."""
    seeds = ["reset", "seed_nav", "seed_pano"]
    result = layout_graph(_loop_graph(seeds))
    p = _pos(result)
    for s in seeds:
        assert p[s]["x"] < p["iter_in"]["x"], f"{s} not left of iter_in"


def test_no_position_overlap():
    """No two laid-out nodes share an identical (x, y)."""
    result = layout_graph(_loop_graph(["reset", "seed_nav", "seed_pano"]))
    coords = [(n["position"]["x"], n["position"]["y"]) for n in result["nodes"]]
    assert len(coords) == len(set(coords)), f"overlapping positions: {coords}"


def test_acyclic_chain_left_to_right():
    """A plain chain lays out with strictly increasing x and a single y."""
    graph = {
        "nodes": [
            _node("a", "promptTemplate"),
            _node("b", "llmCall"),
            _node("c", "promptTemplate"),
        ],
        "edges": [_edge("a", "b"), _edge("b", "c")],
    }
    result = layout_graph(graph)
    p = _pos(result)
    assert p["a"]["x"] < p["b"]["x"] < p["c"]["x"]
    assert p["a"]["y"] == p["b"]["y"] == p["c"]["y"]


def test_empty_graph_is_noop():
    """An empty graph returns without raising."""
    assert layout_graph({"nodes": [], "edges": []}) == {"nodes": [], "edges": []}
