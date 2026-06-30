"""Auto-layout for AgentCanvas graph JSON files.

Two-pass algorithm: topological layering (left-to-right X) followed by
semantic post-processing (viewer stacking, container placement).

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

# Semantic priority for Y-sorting within a layer (lower = higher on canvas)
_PRIORITY: dict[str, int] = {
    "seed": 0,
    "control": 1,
    "processing": 2,
    "viewer": 3,
    "output": 3,
}


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


# ── Chain decomposition ───────────────────────────────────────────────


def _find_chains(
    node_ids: list[str],
    fwd: dict[str, list[str]],
    rev: dict[str, list[str]],
    layer: dict[str, int],
) -> list[list[str]]:
    """Greedy longest-path chain decomposition.

    Returns chains ordered by length (longest first = spine).
    Each node appears in exactly one chain.
    """
    remaining = set(node_ids)
    chains: list[list[str]] = []

    while remaining:
        # Tie-break on node id to make chain decomposition deterministic
        # across interpreter runs (Python hash randomization otherwise
        # shuffles set iteration, and therefore lane assignment).
        ordered = sorted(remaining, key=lambda nid: (layer.get(nid, 0), nid))
        dp: dict[str, int] = {}
        parent: dict[str, str | None] = {}

        for nid in ordered:
            preds = sorted(p for p in rev.get(nid, []) if p in remaining and p in dp)
            if preds:
                best = max(preds, key=lambda p: (dp[p], p))
                dp[nid] = dp[best] + 1
                parent[nid] = best
            else:
                dp[nid] = 1
                parent[nid] = None

        end = max(sorted(remaining), key=lambda nid: dp.get(nid, 0))
        chain: list[str] = []
        cur: str | None = end
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        chain.reverse()
        chains.append(chain)
        remaining -= set(chain)

    return chains


# ── Core algorithm ────────────────────────────────────────────────────


def layout_graph(
    graph: dict[str, Any],
    *,
    h_spacing: int = H_SPACING,
    v_spacing: int = V_SPACING,
    node_heights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute auto-layout positions for a graph definition.

    Args:
        node_heights: Optional ``{node_type: height_px}`` map from live
            registry (see :func:`build_height_map`).  Falls back to
            static estimates when ``None``.

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
    lf: dict[str, list[str]] = defaultdict(list)
    lr: dict[str, list[str]] = defaultdict(list)
    for nid in layout_nids:
        for succ in fwd.get(nid, []):
            if succ in layout_nids:
                lf[nid].append(succ)
                lr[succ].append(nid)

    in_deg: dict[str, int] = {nid: len(lr.get(nid, [])) for nid in layout_nids}

    # ── Phase 2: Longest-path layering (Kahn's algorithm) ────────────

    layer: dict[str, int] = {nid: 0 for nid in layout_nids}
    remaining = {nid: len(lr.get(nid, [])) for nid in layout_nids}
    q: deque[str] = deque(nid for nid, d in remaining.items() if d == 0)
    while q:
        nid = q.popleft()
        for succ in lf.get(nid, []):
            layer[succ] = max(layer[succ], layer[nid] + 1)
            remaining[succ] -= 1
            if remaining[succ] == 0:
                q.append(succ)
    for nid, d in remaining.items():
        if d > 0:
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

    # ── Phase 4: (no dedicated viewer column) ────────────────────────
    #
    # Viewers / outputs are NOT promoted to a dedicated column here — they
    # are placed as a single horizontal row at the very bottom of the
    # canvas in Phase 9, so their layer/lane carry through from the natural
    # topo order and will be overridden when they land in the viewer row.

    # ── Phase 5: Swim-lane layout ────────────────────────────────────
    #
    # One global chain decomposition over the whole graph: the longest
    # chain is the spine (center lane), the rest alternate above / below.

    viewer_nids = {nid for nid in cat if cat[nid] in ("viewer", "output")}
    container_nids = {nid for nid in cat if cat[nid] == "container"}

    local_lane: dict[str, int] = {}
    regulars = sorted(
        layout_nids - viewer_nids - container_nids,
        key=lambda nid: layer[nid],
    )
    if regulars:
        chains = _find_chains(regulars, lf, lr, layer)
        n_chains = len(chains)
        mid = n_chains // 2
        remap: dict[int, int] = {0: mid}
        above, below = mid - 1, mid + 1
        for i in range(1, n_chains):
            if i % 2 == 1 and above >= 0:
                remap[i] = above
                above -= 1
            else:
                remap[i] = below
                below += 1
        for chain_idx, chain in enumerate(chains):
            for nid in chain:
                local_lane[nid] = remap[chain_idx]

    # Viewers: excluded from lane assignment — they live in the dedicated
    # bottom row computed in Phase 9.

    # Container nodes (type=stateContainer in nodes array): placed in fresh
    # lanes below every regular lane, centered on their connections.
    for nid in container_nids:
        connected = list(
            {e["source"] for e in edges if e.get("target") == nid and e["source"] in local_lane}
            | {e["target"] for e in edges if e.get("source") == nid and e["target"] in local_lane}
        )
        max_lane = max(local_lane.values(), default=-1)
        local_lane[nid] = max_lane + 1
        if connected:
            layer[nid] = round(sum(layer[c] for c in connected) / len(connected))

    # ── Phase 6: Y positions (single lane stack, no band gap) ────────
    #
    # Every laned node (regulars + containers) contributes its height to its
    # lane's span; lanes stack top-to-bottom by index with _LANE_GAP.

    spans: dict[int, int] = {}
    for nid, li in local_lane.items():
        h = _node_height(node_map[nid].get("type", ""), node_heights)
        spans[li] = max(spans.get(li, 0), h)

    y_base: dict[int, float] = {}
    y_cursor = 0.0
    for li in sorted(spans.keys()):
        y_base[li] = y_cursor
        y_cursor += spans[li] + _LANE_GAP

    node_y: dict[str, float] = {}
    for nid in node_map:
        if nid in viewer_nids or nid in annotation_nids:
            continue
        node_y[nid] = y_base.get(local_lane.get(nid, 0), 0.0)

    # ── Phase 7: Apply positions to non-viewer nodes ─────────────────

    for n in nodes:
        nid = n["id"]
        if nid in viewer_nids or nid in annotation_nids:
            continue
        n["position"] = {
            "x": layer[nid] * h_spacing,
            "y": round(node_y.get(nid, 0)),
        }

    # ── Phase 9: Viewer row (single horizontal row at the canvas bottom)
    #
    # All viewers/outputs live in one dedicated row below every lane of
    # the graph.  X is anchored to each viewer's upstream source column
    # so a viewer visually sits under the node that feeds it; duplicate
    # columns bump rightward by one h_spacing to avoid collisions.

    if viewer_nids:
        non_viewer_bottom = max(
            (
                node_y.get(nid, 0) + _node_height(node_map[nid].get("type", ""), node_heights)
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
            for n in nodes:
                if n["id"] == nid:
                    n["position"] = {
                        "x": col * h_spacing,
                        "y": round(viewer_row_y),
                    }
                    break

    # ── Phase 8: Container positioning ────────────────────────────────

    if containers:
        ag_map: dict[str, list[str]] = defaultdict(list)
        for ag in access_grants:
            ag_map[ag.get("container_id", "")].append(ag.get("node_id", ""))

        all_bottom = max(
            (
                node_y.get(nid, 0) + _node_height(node_map[nid].get("type", ""), node_heights)
                for nid in node_map
                if nid not in annotation_nids
            ),
            default=0,
        )
        container_y_offset = all_bottom + _LANE_GAP

        for c in containers:
            cid = c.get("id", "")
            connected = [nid for nid in ag_map.get(cid, []) if nid in node_map]
            if connected:
                xs = [layer[nid] * h_spacing for nid in connected]
                c["position"] = {
                    "x": round(sum(xs) / len(xs)),
                    "y": round(container_y_offset),
                }
            else:
                all_x = [layer[nid] * h_spacing for nid in node_map]
                c["position"] = {
                    "x": round(sum(all_x) / len(all_x)) if all_x else 0,
                    "y": round(container_y_offset),
                }
            container_y_offset += _node_height("stateContainer", node_heights) + _LANE_GAP

    # Annotation nodes (notes) are pass-through: their `position` was set
    # by the user on the canvas and copied into `graph` by the deepcopy at
    # the top of this function. Phases 1-9 exclude them from every layout
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
