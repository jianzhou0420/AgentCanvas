"""Auto-layout for AgentCanvas graph JSON files.

Layered (Sugiyama-style) algorithm:

1. break any cycles so layering can't stall (``_break_cycles``);
2. longest-path layering fixes each node's column / X;
3. median-heuristic ordering within each layer cuts edge crossings;
4. a barycentre + isotonic (PAVA) pass assigns Y — straight chains stay
   straight, branches sit next to where they connect, nothing overlaps;
5. semantic post-processing places the viewer row and state containers.

Usage (module)::

    from app.layout import layout_graph
    result = layout_graph(graph_dict)

Usage (CLI)::

    cd agentcanvas/backend
    python -m app.layout ../../workspace/graphs/navgpt_ce.json
    python -m app.layout --preview ../../workspace/graphs/*.json
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

log = logging.getLogger("agentcanvas.layout")

# ── Defaults ──────────────────────────────────────────────────────────

H_SPACING = 280  # horizontal gap between layers (px)
V_SPACING = 160  # vertical gap between nodes in a layer (px)
BAND_GAP = 80  # vertical gap between the main graph and the bottom viewer row (px)

# ── Type-based classification (explicit sets) ─────────────────────────

VIEWER_TYPES: frozenset[str] = frozenset(
    {
        "imageViewer",
        "textViewer",
        "textScroll",
        "actionLog",
        "metrics",
    }
)
OUTPUT_TYPES: frozenset[str] = frozenset({"graphOut"})
CONTROL_TYPES: frozenset[str] = frozenset({"iterIn", "iterOut"})
ANNOTATION_TYPES: frozenset[str] = frozenset({"note"})
KNOWN_PROCESSING: frozenset[str] = frozenset(
    {
        "promptTemplate",
        "llmCall",
        "vlmCall",
        "graphIn",
        "compositeNode",
    }
)


CONTAINER_TYPE = "stateContainer"

# ── Node height estimation ───────────────────────────────────────────
# Heights are derived from BaseCanvasNode ui_config + ports when the
# component registry is available (API path).  The static _TYPE_HEIGHT
# table serves as a fallback for CLI usage where no registry is loaded.

_TYPE_HEIGHT: dict[str, int] = {
    "iterIn": 50,
    "iterOut": 50,
    "graphIn": 50,
    "graphOut": 50,
    "llmCall": 120,
    "vlmCall": 120,
    "promptTemplate": 100,
    "compositeNode": 120,
    "imageViewer": 220,
    "textViewer": 160,
    "textScroll": 200,
    "actionLog": 180,
    "metrics": 160,
    "stateContainer": 150,
}
_DEFAULT_HEIGHT = 80
_LANE_GAP = 40  # minimum vertical gap between lanes

# ── Node width estimation ────────────────────────────────────────────
# The canvas measures every node's rendered width and passes it in
# (``node_dims``); columns are then spaced by real width so wide nodes
# never bleed into the next column.  These fallbacks cover the no-dims
# path (CLI / tests), where a fixed h_spacing pitch is used instead.

_COL_GAP = 72  # horizontal gap between a column's right edge and the next
_DEFAULT_WIDTH = 200
_DUMMY_H = 22  # channel height reserved per long edge crossing a column
_TYPE_WIDTH: dict[str, int] = {
    "iterIn": 150,
    "iterOut": 150,
    "graphIn": 130,
    "graphOut": 130,
    "imageViewer": 260,
    "textViewer": 240,
    "textScroll": 240,
    "actionLog": 240,
    "metrics": 220,
    "stateContainer": 200,
}

# Per-widget height contributions (px)
_HEADER_HEIGHT = 40
_PORT_ROW_HEIGHT = 28
_DISPLAY_HEIGHTS: dict[str, int] = {
    "image_viewer": 180,
    "log_list": 120,
    "metric_table": 100,
}
_DISPLAY_DEFAULT_HEIGHT = 80


def compute_height_from_node_cls(node_cls: type) -> int:
    """Derive estimated rendered height from a BaseCanvasNode subclass.

    Uses ui_config.min_height, input/output ports, config_fields, and
    display_fields to compute height.  Returns pixels.

    On the canvas, ports and config fields share rows (port handles sit
    alongside content), so the body height is driven by the *max* of
    port count and visible config count, not their sum.  Display fields
    (image viewers, log lists) stack below and add their full height.
    """
    ui = getattr(node_cls, "ui_config", None)
    if ui is None:
        return _DEFAULT_HEIGHT

    layout = getattr(ui, "layout", "block")

    # Strip layout nodes have a fixed narrow shape
    if layout == "strip":
        min_h = getattr(ui, "min_height", "")
        if min_h and min_h.endswith("px"):
            return int(min_h[:-2])
        return 50

    # Start from explicit min_height or header baseline
    min_h_str = getattr(ui, "min_height", "")
    base = int(min_h_str[:-2]) if min_h_str and min_h_str.endswith("px") else _HEADER_HEIGHT

    # Folded state: config fields collapsed, only port handles visible.
    # Cap visible rows — extra ports stack as compact handles on the edge.
    _FOLDED_MAX_ROWS = 3
    n_in = len(getattr(node_cls, "input_ports", []))
    n_out = len(getattr(node_cls, "output_ports", []))
    body_rows = min(max(n_in, n_out), _FOLDED_MAX_ROWS)
    body_h = body_rows * _PORT_ROW_HEIGHT

    # Display fields (image viewers, logs) stack below the body
    display_h = 0
    for df in getattr(ui, "display_fields", []):
        dt = getattr(df, "display_type", "")
        display_h += _DISPLAY_HEIGHTS.get(dt, _DISPLAY_DEFAULT_HEIGHT)

    # Viewer layout: display fields dominate the height
    if layout == "viewer":
        return max(base, base + display_h)

    # Image grid: dynamic grid of image cells; estimate from default rows.
    if layout == "imageGrid":
        default_cfg = getattr(node_cls, "default_config", {}) or {}
        rows = int(default_cfg.get("rows", 1))
        return max(base, _HEADER_HEIGHT + rows * 180 + 20)

    return max(base, base + body_h + display_h)


def build_height_map() -> dict[str, int]:
    """Build a node_type → height map from the live NODE_HANDLERS registry.

    Returns an empty dict if the registry is not importable (e.g. CLI mode).
    """
    try:
        from app.agent_loop.builtin_nodes import NODE_HANDLERS
    except Exception:
        return {}
    return {ntype: compute_height_from_node_cls(cls) for ntype, cls in NODE_HANDLERS.items()}


def _node_height(node_type: str, overrides: dict[str, int] | None = None) -> int:
    """Estimated rendered height for a node type.

    Checks ``overrides`` (live registry) first, then static table, then default.
    """
    if overrides and node_type in overrides:
        return overrides[node_type]
    if node_type in _TYPE_HEIGHT:
        return _TYPE_HEIGHT[node_type]
    return _DEFAULT_HEIGHT


def _classify(node_type: str) -> str:
    """Classify a node by its type string (topology-independent)."""
    if node_type in VIEWER_TYPES:
        return "viewer"
    if node_type in OUTPUT_TYPES:
        return "output"
    if node_type in CONTROL_TYPES:
        return "control"
    if node_type == CONTAINER_TYPE:
        return "container"
    return "processing"


def _is_known_type(t: str) -> bool:
    """Check whether a type string matches any known pattern."""
    return (
        t in VIEWER_TYPES
        or t in OUTPUT_TYPES
        or t in CONTROL_TYPES
        or t in KNOWN_PROCESSING
        or "__" in t  # nodeset__node convention (e.g. env_habitat__observe)
    )


# ── Cycle breaking (make an arbitrary digraph acyclic for layering) ────


def _break_cycles(
    node_ids: list[str], fwd: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Drop back-edges so longest-path layering never stalls on a cycle.

    Kahn's layering only makes progress on a DAG: any cycle leaves its nodes
    (and everything downstream of them) stuck at layer 0, piling them into one
    overlapping column.  The loop's ``iterOut → iterIn`` edge is dropped
    upstream, but graphs routinely contain *other* incidental cycles — e.g. a
    cache node that both feeds and is fed by an observer.  A depth-first sweep
    keeps every tree / forward / cross edge and drops each edge that points
    back at a node still on the recursion stack (a cycle-closing back-edge).
    Nodes and neighbours are visited in sorted order, so the resulting DAG is
    deterministic across interpreter runs.

    Returns ``(dag_fwd, dag_rev)`` adjacency with parallel edges collapsed.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in node_ids}
    dag_fwd: dict[str, list[str]] = defaultdict(list)

    for root in node_ids:
        if color[root] != WHITE:
            continue
        color[root] = GREY
        stack: list[tuple[str, Any]] = [(root, iter(sorted(set(fwd.get(root, [])))))]
        while stack:
            u, succ_it = stack[-1]
            descended = False
            for v in succ_it:
                if color.get(v, BLACK) == GREY:
                    continue  # back-edge → drop (would close a cycle)
                dag_fwd[u].append(v)  # tree / forward / cross edge → keep
                if color[v] == WHITE:
                    color[v] = GREY
                    stack.append((v, iter(sorted(set(fwd.get(v, []))))))
                    descended = True
                    break
            if not descended:
                color[u] = BLACK
                stack.pop()

    dag_rev: dict[str, list[str]] = defaultdict(list)
    for u, vs in dag_fwd.items():
        for v in vs:
            dag_rev[v].append(u)
    return dag_fwd, dag_rev


# ── Within-layer ordering (crossing minimisation) ─────────────────────


def _weighted_median(vals: list[float]) -> float | None:
    """Weighted-median ordering key (Sugiyama).  ``None`` when no neighbours."""
    if not vals:
        return None
    vals = sorted(vals)
    m = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[m])
    if len(vals) == 2:
        return (vals[0] + vals[1]) / 2.0
    left = vals[m - 1] - vals[0]
    right = vals[-1] - vals[m]
    if left + right == 0:
        return (vals[m - 1] + vals[m]) / 2.0
    return (vals[m - 1] * right + vals[m] * left) / (left + right)


def _order_layers(
    layers: dict[int, list[str]],
    adj_prev: dict[str, list[str]],
    adj_next: dict[str, list[str]],
    sweeps: int = 6,
) -> dict[str, int]:
    """Order nodes within each layer to reduce edge crossings.

    Alternates downward / upward median sweeps, re-sorting every layer by the
    weighted median of its neighbours' positions in the adjacent layer.
    Mutates ``layers`` in place and returns the final ``node → index`` map.
    A node with no neighbours on the reference side keeps its current index as
    the sort key, so the pass is stable and deterministic.
    """
    pos: dict[str, int] = {}
    for li in layers:
        for i, nid in enumerate(layers[li]):
            pos[nid] = i

    layer_ids = sorted(layers)
    for s in range(sweeps):
        going_down = s % 2 == 0
        seq = layer_ids[1:] if going_down else layer_ids[-2::-1]
        adj = adj_prev if going_down else adj_next
        for li in seq:
            keyed = [
                (nid, _weighted_median([pos[u] for u in adj.get(nid, []) if u in pos]))
                for nid in layers[li]
            ]
            keyed = [(nid, k if k is not None else float(pos[nid])) for nid, k in keyed]
            keyed.sort(key=lambda t: t[1])
            layers[li] = [nid for nid, _ in keyed]
            for i, nid in enumerate(layers[li]):
                pos[nid] = i
    return pos


# ── Coordinate assignment (Y) ─────────────────────────────────────────


def _pava(targets: list[float], min_gaps: list[float]) -> list[float]:
    """Isotonic layer placement: top-y values closest (least-squares) to
    ``targets`` that preserve order and keep ``min_gaps[i]`` between
    consecutive nodes.

    Substituting ``z_i = top_i - Sum_{k<i} gap_k`` turns the separation
    constraints into plain monotonicity (``z`` non-decreasing), solved exactly
    by pool-adjacent-violators — minimal-movement overlap removal.
    """
    n = len(targets)
    if n == 0:
        return []
    prefix = [0.0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + min_gaps[i - 1]
    t = [targets[i] - prefix[i] for i in range(n)]

    vals: list[float] = []
    counts: list[int] = []
    weights: list[int] = []
    for x in t:
        vals.append(x)
        counts.append(1)
        weights.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, c2, w2 = vals.pop(), counts.pop(), weights.pop()
            v1, c1, w1 = vals.pop(), counts.pop(), weights.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            counts.append(c1 + c2)
            weights.append(w1 + w2)
    z: list[float] = []
    for v, c in zip(vals, counts, strict=True):
        z.extend([v] * c)
    return [z[i] + prefix[i] for i in range(n)]


def _coord_pass(
    layers: dict[int, list[str]],
    adj: dict[str, list[str]],
    layer: dict[str, int],
    h_of: Any,
    direction: str,
) -> dict[str, float]:
    """One coordinate sweep: give each node a top-y near the barycentre of its
    already-placed neighbours on one side, resolving overlaps per layer.

    ``direction='down'`` anchors leftward (layers processed left→right, pull to
    lower-layer neighbours); ``'up'`` anchors rightward.  A node with no
    reference-side neighbour keeps its stacked initial position.
    """
    top: dict[str, float] = {}
    for li in layers:
        cy = 0.0
        for nid in layers[li]:
            top[nid] = cy
            cy += h_of(nid) + _LANE_GAP

    seq = sorted(layers) if direction == "down" else sorted(layers, reverse=True)
    for li in seq:
        col = layers[li]
        if not col:
            continue
        desired: list[float] = []
        for nid in col:
            refs = [
                top[u]
                for u in adj.get(nid, [])
                if (layer[u] < li if direction == "down" else layer[u] > li)
            ]
            desired.append(sum(refs) / len(refs) if refs else top[nid])
        gaps = [h_of(col[i]) + _LANE_GAP for i in range(len(col) - 1)]
        for nid, c in zip(col, _pava(desired, gaps), strict=True):
            top[nid] = c
    return top


# ── Core algorithm ────────────────────────────────────────────────────


def layout_graph(
    graph: dict[str, Any],
    *,
    h_spacing: int = H_SPACING,
    v_spacing: int = V_SPACING,
    node_heights: dict[str, int] | None = None,
    node_dims: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Compute auto-layout positions for a graph definition.

    Args:
        node_heights: Optional ``{node_type: height_px}`` map from live
            registry (see :func:`build_height_map`).  Falls back to
            static estimates when ``None``.
        node_dims: Optional ``{node_id: {"width": px, "height": px}}`` map of
            *real rendered* sizes measured by the canvas.  When present, rows
            are spaced by each node's true height and columns by each column's
            widest node (cumulative X) — so wide nodes never overlap the next
            column.  When ``None``, columns fall back to the fixed
            ``h_spacing`` pitch and heights to ``node_heights`` / static
            estimates (the CLI / test path).

    Returns a **new** dict with only ``nodes[].position`` and
    ``containers[].position`` updated.  All other fields are preserved.
    """
    graph = copy.deepcopy(graph)
    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    edges: list[dict[str, Any]] = graph.get("edges", [])
    containers: list[dict[str, Any]] = graph.get("containers", [])
    access_grants: list[dict[str, Any]] = graph.get("access_grants", [])

    if not nodes:
        return graph

    node_map: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}

    # Long-edge routing dummies (Phase 5): id -> (width, height).  Consulted by
    # the size accessors so dummies stay thin (0 width, a slim channel height).
    _dummy_sizes: dict[str, tuple[int, int]] = {}

    # Real-size accessors: prefer the canvas-measured dimension, else estimate.
    def _dim(nid: str, key: str) -> float | None:
        if node_dims and nid in node_dims:
            v = node_dims[nid].get(key)
            if v:
                return float(v)
        return None

    def _h_of(nid: str) -> int:
        if nid in _dummy_sizes:
            return _dummy_sizes[nid][1]
        real = _dim(nid, "height")
        if real is not None:
            return round(real)
        return _node_height(node_map.get(nid, {}).get("type", ""), node_heights)

    def _w_of(nid: str) -> int:
        if nid in _dummy_sizes:
            return _dummy_sizes[nid][0]
        real = _dim(nid, "width")
        if real is not None:
            return round(real)
        return _TYPE_WIDTH.get(node_map.get(nid, {}).get("type", ""), _DEFAULT_WIDTH)

    # Annotation nodes (notes) are pulled out of the main layout: their
    # positions are computed by Phase 10 from the bounding box of the placed
    # graph, so the rest of the algorithm produces bit-identical output to
    # the no-note case.
    annotation_nids: set[str] = {
        nid for nid, n in node_map.items() if n.get("type") in ANNOTATION_TYPES
    }

    # ── Phase 1: Build DAG (exclude IterOut→IterIn back-edges) ────────

    fwd: dict[str, list[str]] = defaultdict(list)
    rev: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in node_map or tgt not in node_map:
            continue
        src_n, tgt_n = node_map[src], node_map[tgt]
        # Back-edge detection: iterOut → paired iterIn
        if (
            src_n.get("type") == "iterOut"
            and tgt_n.get("type") == "iterIn"
            and src_n.get("config", {}).get("pairedWith") == tgt
        ):
            continue
        fwd[src].append(tgt)
        rev[tgt].append(src)

    # ── Phase 1b: Single unified DAG (no init/loop bands) ─────────────
    #
    # Two-pivot model (ADR-dataflow-008): the run-start init inputs enter
    # via the *left* side of iterIn as ordinary canvas wires, so the seed
    # closure feeding iterIn is already part of the same DAG.  Layered
    # left-to-right, those seeds land naturally to the LEFT of iterIn — no
    # separate band, no vertical stacking.  The whole graph reads as one
    # flow::
    #
    #     seed / init ──► iterIn ──► loop body … ──► iterOut
    #
    # (The retired three-pivot model split an "init band" above a "loop
    # band"; that partitioning died with the Initialize node, 2026-06-10.)

    layout_nids: set[str] = set(node_map) - annotation_nids

    # Restrict adjacency to layout nodes (drops any stray annotation wires).
    lf0: dict[str, list[str]] = defaultdict(list)
    for nid in layout_nids:
        for succ in fwd.get(nid, []):
            if succ in layout_nids:
                lf0[nid].append(succ)

    # Break any residual cycles (beyond the loop back-edge already dropped in
    # Phase 1) so longest-path layering never stalls.  A single incidental
    # cycle — e.g. a cache node that both feeds and is fed by an observer —
    # would otherwise leave every downstream node at layer 0, collapsing the
    # whole graph into one overlapping column at x=0.
    lf, lr = _break_cycles(sorted(layout_nids), lf0)

    in_deg: dict[str, int] = {nid: len(lr.get(nid, [])) for nid in layout_nids}

    # ── Phase 2: Longest-path layering (Kahn's algorithm) ────────────

    layer: dict[str, int] = {nid: 0 for nid in layout_nids}
    remaining = {nid: len(lr.get(nid, [])) for nid in layout_nids}
    q: deque[str] = deque(sorted(nid for nid, d in remaining.items() if d == 0))
    while q:
        nid = q.popleft()
        for succ in lf.get(nid, []):
            layer[succ] = max(layer[succ], layer[nid] + 1)
            remaining[succ] -= 1
            if remaining[succ] == 0:
                q.append(succ)
    for nid, d in remaining.items():
        if d > 0:  # pragma: no cover — _break_cycles guarantees a DAG
            log.warning("Unreachable node (possible cycle): %s", nid)

    # ── Phase 3: Classify nodes ───────────────────────────────────────

    cat: dict[str, str] = {}
    for n in nodes:
        nid = n["id"]
        if nid in annotation_nids:
            continue
        ntype = n.get("type", "")
        c = _classify(ntype)
        # Seeds: in-band in-degree 0 processing nodes
        if c == "processing" and in_deg.get(nid, 0) == 0:
            c = "seed"
        cat[nid] = c
        # Warn for unknown types
        if c == "processing" and ntype and not _is_known_type(ntype):
            log.warning("Unknown node type '%s' (id=%s), treating as processing", ntype, nid)

    # ── Phase 4: Split flow nodes from viewers / containers ──────────
    #
    # Viewers / outputs are placed as a single horizontal row at the bottom of
    # the canvas (Phase 9); state containers go below the flow (Phase 8).  The
    # remaining "regulars" are what the layered flow below arranges.

    viewer_nids = {nid for nid in cat if cat[nid] in ("viewer", "output")}
    container_nids = {nid for nid in cat if cat[nid] == "container"}
    regulars = layout_nids - viewer_nids - container_nids

    # ── Phase 5: Long-edge routing dummies + within-layer ordering ───
    #
    # (a) Reserve channels.  A DAG edge spanning >1 column would be drawn
    # straight through the columns between its endpoints and vanish behind
    # their nodes (wires render beneath node bodies).  Replace each such edge
    # with a chain of thin dummy nodes — one per crossed column — so the
    # ordering + coordinate passes push real nodes off the edge's lane, leaving
    # a clear horizontal channel.  Dummies never get an output position; their
    # coordinates become the edge's routing waypoints (Phase 10).
    #
    # (b) Order each layer (real + dummy) by the median heuristic to cut
    # crossings — replacing the old fixed "spine centre, others alternating"
    # lane assignment that ignored connectivity.

    layers: dict[int, list[str]] = defaultdict(list)
    for nid in sorted(regulars, key=lambda n: (layer[n], n)):
        layers[layer[nid]].append(nid)

    edge_chain: dict[tuple[str, str], list[str]] = {}
    _dcount = 0
    for u in sorted(regulars):
        for v in sorted(set(lf.get(u, []))):
            if v not in regulars or layer[v] - layer[u] <= 1:
                continue
            chain: list[str] = []
            for li in range(layer[u] + 1, layer[v]):
                did = f"__route_{_dcount}"
                _dcount += 1
                _dummy_sizes[did] = (0, _DUMMY_H)
                layer[did] = li
                layers[li].append(did)
                chain.append(did)
            edge_chain[(u, v)] = chain

    dummy_ids = set(_dummy_sizes)
    route_nids = regulars | dummy_ids

    # Routing edges: each long forward edge threaded through its dummy chain,
    # so every routing edge spans exactly one column.
    redges: list[tuple[str, str]] = []
    for u in sorted(regulars):
        for v in sorted(set(lf.get(u, []))):
            if v not in regulars:
                continue
            chain = edge_chain.get((u, v))
            if chain:
                prev = u
                for d in chain:
                    redges.append((prev, d))
                    prev = d
                redges.append((prev, v))
            elif layer[v] - layer[u] == 1:
                redges.append((u, v))

    adj_prev: dict[str, list[str]] = defaultdict(list)
    adj_next: dict[str, list[str]] = defaultdict(list)
    for a, b in redges:
        adj_next[a].append(b)
        adj_prev[b].append(a)
    _order_layers(layers, adj_prev, adj_next)

    # ── Phase 6: Y positions (barycentre pull + non-overlap) ─────────
    #
    # Undirected neighbour set over the routing graph (long edges threaded
    # through dummies, so the edge straightens and real nodes are pushed off
    # its lane) plus leftward / loop edges linked directly (so the loop's two
    # ends align vertically).  Two passes — anchored left, then right — pull
    # each node toward the mean top-y of its neighbours; per-layer isotonic
    # resolution (_pava) enforces non-overlap while moving nodes as little as
    # possible.  Averaging keeps straight chains straight and centres branches
    # on their connection points.

    adj: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    def _link(a: str, b: str) -> None:
        if a != b and (a, b) not in seen_pairs:
            seen_pairs.add((a, b))
            adj[a].append(b)
            adj[b].append(a)

    for a, b in redges:
        _link(a, b)
    # Leftward / loop edges (dropped from the DAG) stay direct links — no
    # dummies — so the loop's ends align but we don't route them.
    for e in edges:
        s, t = e.get("source", ""), e.get("target", "")
        if s in regulars and t in regulars and layer.get(t, 0) <= layer.get(s, 0):
            _link(s, t)

    down = _coord_pass(layers, adj, layer, _h_of, "down")
    up = _coord_pass(layers, adj, layer, _h_of, "up")
    top = {nid: (down[nid] + up[nid]) / 2.0 for nid in route_nids}
    for li in sorted(layers):
        col = layers[li]
        gaps = [_h_of(col[i]) + _LANE_GAP for i in range(len(col) - 1)]
        for nid, c in zip(col, _pava([top[nid] for nid in col], gaps), strict=True):
            top[nid] = c

    node_y: dict[str, float] = {}
    if top:
        min_top = min(top.values())
        for nid in top:
            node_y[nid] = top[nid] - min_top

    # ── Phase 7: Viewer-row placement (layer + y only; X in Phase 8) ─
    #
    # Viewers/outputs live in one dedicated row below every flow lane.  Each is
    # anchored to its upstream source column so it sits under the node that
    # feeds it; duplicate columns bump one column rightward to avoid overlap.
    # Only layer[] and node_y[] are set here — pixel X is assigned in Phase 8
    # once every column's width is known.

    if viewer_nids:
        non_viewer_bottom = max(
            (
                node_y.get(nid, 0) + _h_of(nid)
                for nid in node_map
                if nid not in viewer_nids and nid not in annotation_nids
            ),
            default=0.0,
        )
        viewer_row_y = non_viewer_bottom + BAND_GAP

        def _primary_source_col(nid: str) -> int:
            srcs = [
                layer[e["source"]]
                for e in edges
                if e.get("target") == nid
                and e.get("source") in layer
                and e.get("source") not in viewer_nids
            ]
            return max(srcs) if srcs else 0

        sorted_viewers = sorted(viewer_nids, key=lambda nid: (_primary_source_col(nid), nid))
        used_cols: set[int] = set()
        for nid in sorted_viewers:
            col = _primary_source_col(nid)
            while col in used_cols:
                col += 1
            used_cols.add(col)
            layer[nid] = col
            node_y[nid] = viewer_row_y

    # ── Phase 8: Column X (width-aware) + apply positions ────────────
    #
    # Every flow node and viewer now has a column index (layer[]) and a top-y
    # (node_y[]).  Columns are laid out left→right: each column's pixel X is the
    # running sum of the preceding columns' widths + _COL_GAP, where a column's
    # width is its widest member's real (canvas-measured) width.  So a node
    # wider than the old fixed pitch pushes the whole rest of the graph right
    # instead of overlapping its neighbour.  Without node_dims this collapses
    # to the fixed h_spacing pitch (the CLI / test path).

    placed = regulars | viewer_nids
    col_w: dict[int, int] = {}
    for nid in placed:
        li = layer[nid]
        col_w[li] = max(col_w.get(li, 0), _w_of(nid))

    if node_dims and col_w:
        col_x: dict[int, int] = {}
        cx = 0
        for li in range(min(col_w), max(col_w) + 1):
            col_x[li] = cx
            cx += col_w.get(li, _DEFAULT_WIDTH) + _COL_GAP

        def _col_x(li: int) -> int:
            return col_x.get(li, li * h_spacing)
    else:

        def _col_x(li: int) -> int:
            return li * h_spacing

    for n in nodes:
        nid = n["id"]
        if nid not in placed:
            continue
        n["position"] = {"x": _col_x(layer[nid]), "y": round(node_y.get(nid, 0.0))}

    # ── Phase 9: Container positioning ────────────────────────────────

    if containers:
        ag_map: dict[str, list[str]] = defaultdict(list)
        for ag in access_grants:
            ag_map[ag.get("container_id", "")].append(ag.get("node_id", ""))

        all_bottom = max(
            (
                node_y.get(nid, 0) + _h_of(nid)
                for nid in node_map
                if nid not in annotation_nids
            ),
            default=0,
        )
        container_y_offset = all_bottom + _LANE_GAP

        for c in containers:
            cid = c.get("id", "")
            connected = [nid for nid in ag_map.get(cid, []) if nid in layer]
            xs = (
                [_col_x(layer[nid]) for nid in connected]
                if connected
                else [_col_x(layer[nid]) for nid in node_map if nid in layer]
            )
            c["position"] = {
                "x": round(sum(xs) / len(xs)) if xs else 0,
                "y": round(container_y_offset),
            }
            c_real = _dim(cid, "height")
            c_h = round(c_real) if c_real is not None else _node_height("stateContainer", node_heights)
            container_y_offset += c_h + _LANE_GAP

    # ── Phase 10: Long-edge routing waypoints ────────────────────────
    #
    # Each long forward edge gets the centre points of its dummy chain as
    # ``waypoints`` — the clear channel the frontend can route the wire through
    # so it never disappears behind the nodes in the columns it crosses.  Short
    # edges get none (adjacent columns route fine as a plain curve).

    if edge_chain:

        def _col_center(li: int) -> float:
            return _col_x(li) + col_w.get(li, _DEFAULT_WIDTH) / 2.0

        wp_map: dict[tuple[str, str], list[dict[str, int]]] = {
            (u, v): [
                {
                    "x": round(_col_center(layer[d])),
                    "y": round(node_y.get(d, 0.0) + _DUMMY_H / 2),
                }
                for d in chain
            ]
            for (u, v), chain in edge_chain.items()
            if chain
        }
        for e in edges:
            key = (e.get("source", ""), e.get("target", ""))
            if key in wp_map:
                e["waypoints"] = wp_map[key]

    # Annotation nodes (notes) are pass-through: their `position` was set
    # by the user on the canvas and copied into `graph` by the deepcopy at
    # the top of this function. Phases 1-10 exclude them from every layout
    # calculation, so we leave their positions untouched here.

    return graph


# ── CLI entry point ───────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-layout AgentCanvas graph JSON files",
    )
    parser.add_argument("files", nargs="+", help="Graph JSON file(s)")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print position table to stdout without modifying files",
    )
    parser.add_argument(
        "--h-spacing",
        type=int,
        default=H_SPACING,
        help=f"Horizontal spacing between layers (default: {H_SPACING})",
    )
    parser.add_argument(
        "--v-spacing",
        type=int,
        default=V_SPACING,
        help=f"Vertical spacing between nodes (default: {V_SPACING})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            log.error("File not found: %s", path)
            continue

        raw = json.loads(path.read_text())
        result = layout_graph(raw, h_spacing=args.h_spacing, v_spacing=args.v_spacing)

        if args.preview:
            name = result.get("name", path.stem)
            nodes = result.get("nodes", [])
            containers = result.get("containers", [])
            print(f"\n{'=' * 72}")
            print(f"  {name}  ({path.name})")
            print(f"{'=' * 72}")
            print(f"{'node_id':<30} {'type':<28} {'layer':>5}  {'(x, y)'}")
            print(f"{'-' * 30} {'-' * 28} {'-' * 5}  {'-' * 14}")
            for n in nodes:
                pos = n["position"]
                print(
                    f"{n['id']:<30} {n.get('type', '?'):<28} "
                    f"{pos['x'] // args.h_spacing:>5}  ({pos['x']}, {pos['y']})"
                )
            for c in containers:
                pos = c["position"]
                print(f"{c['id']:<30} {'[container]':<28} {'':>5}  ({pos['x']}, {pos['y']})")
        else:
            path.write_text(json.dumps(result, indent=2) + "\n")
            n_nodes = len(result.get("nodes", []))
            n_containers = len(result.get("containers", []))
            print(
                f"  {path.name}: laid out {n_nodes} nodes"
                + (f" + {n_containers} containers" if n_containers else "")
            )


if __name__ == "__main__":
    _cli()
