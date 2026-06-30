"""Scope topology analysis for multi-scope iteration (ADR-multi-scope).

Pure function on a ``GraphDefinition``. Computes the scope tree implied by
each (iterIn, iterOut) pair plus their graphIn/graphOut IO boundary nodes,
and assigns every node + graphIn/graphOut to its innermost containing
scope.

A **scope** is a connected sub-region of the graph defined by one
``pairedWith`` pair. Multiple scopes coexist in a single flat graph;
nesting is determined by topology (an inner scope's ``iter_in`` and
``iter_out`` both lie within an outer scope's body).

This module does NOT execute anything — it only analyses topology and
emits ``ScopeForest`` + validation errors. The executor (Phase B) reads
the ``ScopeForest`` to drive per-scope step counters / stop checks /
settle-loop behaviour.

See `.claude/plans/zany-frolicking-acorn.md` for the full design.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..graph_def import GraphDefinition

log = logging.getLogger("agentcanvas.scope-analysis")


# Synthetic id for the implicit "graph scope" — nodes that are outside
# any author-declared (iterIn, iterOut) pair. Single-scope
# graphs whose pivots are at graph level still produce ONE author scope;
# the graph scope holds run-start entry nodes (env_reset, vp_init, ...) and
# post-loop sinks (evaluate, viewers).
GRAPH_SCOPE_ID = ""


@dataclass
class ScopeInfo:
    """One iteration scope — an (iterIn, iterOut) pair plus its body
    nodes, IO boundary nodes, and resolved config. The scope's halt signal
    is the iterOut's own ``stop`` input (no separate termination node).
    """

    scope_id: str  # iter_in node id (the canonical key)
    iter_out_id: str
    parent_scope_id: str | None  # None = root scope
    child_scope_ids: list[str] = field(default_factory=list)
    member_node_ids: set[str] = field(default_factory=set)  # innermost membership
    graphin_node_ids: list[str] = field(default_factory=list)
    graphout_node_ids: list[str] = field(default_factory=list)
    step_budget: int | None = None  # resolved from iter_in.config.step_budget


@dataclass
class ScopeForest:
    """Result of ``analyze_scopes``: the full scope topology."""

    scopes: dict[str, ScopeInfo] = field(default_factory=dict)
    root_scope_ids: list[str] = field(default_factory=list)
    # innermost scope id per node; GRAPH_SCOPE_ID for nodes outside any scope
    node_to_scope: dict[str, str] = field(default_factory=dict)
    # nodes outside any author scope (graph-scope entry nodes + post-loop sinks)
    graph_scope_node_ids: list[str] = field(default_factory=list)

    @property
    def is_single_scope(self) -> bool:
        """True if the graph has 0 or 1 author scopes — the legacy shape.
        Used by the executor to take a fast-path that mirrors pre-refactor
        behaviour bit-for-bit.
        """
        return len(self.scopes) <= 1


# ── public API ─────────────────────────────────────────────────────────


def analyze_scopes(graph: GraphDefinition) -> tuple[ScopeForest, list[str]]:
    """Analyze a graph's scope topology.

    Returns ``(forest, errors)`` where ``errors`` is a list of validation
    error strings (empty on a clean graph). The forest is always
    well-formed even when errors are present — callers that must reject
    invalid graphs should check ``errors`` before using the forest.

    Pure function: does not mutate ``graph``.
    """
    errors: list[str] = []
    forest = ScopeForest()

    # ── Step 1: collect iter_in/iter_out/graphIn/graphOut nodes
    nodes_by_id: dict[str, Any] = {n.id: n for n in graph.nodes}
    iter_in_ids: list[str] = []
    iter_out_ids: list[str] = []
    graphin_ids: list[str] = []
    graphout_ids: list[str] = []

    for n in graph.nodes:
        if n.type == "iterIn":
            iter_in_ids.append(n.id)
        elif n.type == "iterOut":
            iter_out_ids.append(n.id)
        elif n.type == "graphIn":
            graphin_ids.append(n.id)
        elif n.type == "graphOut":
            graphout_ids.append(n.id)

    # ── Step 2: build scope info for each iter_in
    # Pair iter_in → iter_out (via iter_out.config.pairedWith == iter_in.id).
    iter_out_paired: dict[str, str] = {}  # iter_out_id → iter_in_id (paired back)
    for io_id in iter_out_ids:
        paired = (nodes_by_id[io_id].config or {}).get("pairedWith", "")
        if paired:
            iter_out_paired[io_id] = paired

    # Detect duplicate iter_in pairings (two iter_outs claim the same iter_in)
    iter_in_to_iter_out: dict[str, str] = {}
    for io_id, ii_id in iter_out_paired.items():
        if ii_id in iter_in_to_iter_out:
            errors.append(
                f"Two iterOut nodes claim pairedWith='{ii_id}': "
                f"'{iter_in_to_iter_out[ii_id]}' and '{io_id}'. "
                f"Each iterIn must be paired with at most one iterOut."
            )
        else:
            iter_in_to_iter_out[ii_id] = io_id

    # Build provisional ScopeInfo for each iter_in (regardless of pair completeness)
    for ii_id in iter_in_ids:
        io_id = iter_in_to_iter_out.get(ii_id, "")
        if not io_id:
            errors.append(
                f"iterIn '{ii_id}' has no paired iterOut "
                f"(no iterOut declares pairedWith='{ii_id}')."
            )
            continue
        ii_cfg = nodes_by_id[ii_id].config or {}
        forest.scopes[ii_id] = ScopeInfo(
            scope_id=ii_id,
            iter_out_id=io_id,
            parent_scope_id=None,  # filled later
            step_budget=ii_cfg.get("step_budget"),
        )

    # ── Step 3: per-scope candidate membership via BFS
    # Forward BFS from iter_in (treating iter_out as a terminal node we
    # include but don't traverse past), backward BFS from iter_out, intersect.
    #
    # Augmentation: scope topology must include the implicit pairedWith
    # handoff edge that the executor uses internally — iterOut → iterIn
    # (per-iter loopback). This is NOT a canvas edge (graph.edges) but IS
    # a control-flow path the analyzer must see, otherwise inner-scope
    # nodes downstream of the handoff appear "unreachable" from outer
    # scope's BFS.
    edges = graph.edges or []
    out_adj: dict[str, list[str]] = {}  # source_id → [target_id, ...]
    in_adj: dict[str, list[str]] = {}  # target_id → [source_id, ...]
    for e in edges:
        out_adj.setdefault(e.source, []).append(e.target)
        in_adj.setdefault(e.target, []).append(e.source)
    # Implicit pairedWith handoff.
    # iterOut.pairedWith == X (iterIn id) → implicit edge iterOut → X
    # (the per-iter loopback). Adding this edge means BFS forward from
    # iter_in will eventually traverse the cycle — so we MUST be careful
    # to stop at iter_out (boundary) before this edge would create a loop.
    for io_id in iter_out_ids:
        paired = (nodes_by_id[io_id].config or {}).get("pairedWith", "")
        if paired and paired in nodes_by_id:
            out_adj.setdefault(io_id, []).append(paired)
            in_adj.setdefault(paired, []).append(io_id)

    candidate_members: dict[str, set[str]] = {}  # scope_id → candidate set
    for scope_id, scope in forest.scopes.items():
        ii_id = scope.scope_id
        io_id = scope.iter_out_id
        # Forward reachability from iter_in (include iter_in itself; do not
        # traverse OUT of iter_out — it's the boundary).
        fwd: set[str] = set()
        q: deque[str] = deque([ii_id])
        while q:
            cur = q.popleft()
            if cur in fwd:
                continue
            fwd.add(cur)
            if cur == io_id:
                continue  # don't traverse through the boundary
            for nxt in out_adj.get(cur, []):
                if nxt not in fwd:
                    q.append(nxt)
        # Backward reachability from iter_out (include iter_out; do not
        # traverse OUT of iter_in — symmetric boundary).
        bwd: set[str] = set()
        q = deque([io_id])
        while q:
            cur = q.popleft()
            if cur in bwd:
                continue
            bwd.add(cur)
            if cur == ii_id:
                continue
            for prev in in_adj.get(cur, []):
                if prev not in bwd:
                    q.append(prev)
        cand = fwd & bwd
        # Always include the scope's own pivots.
        cand.add(ii_id)
        cand.add(io_id)
        candidate_members[scope_id] = cand

    # ── Step 4: nesting tree
    # Scope B is nested in scope A iff B.iter_in ∈ A.candidate_members
    # AND B.iter_out ∈ A.candidate_members AND A != B.
    # Innermost parent = the smallest scope containing B's pivots.
    for b_id, b_scope in forest.scopes.items():
        candidate_parents: list[str] = []
        for a_id, a_cand in candidate_members.items():
            if a_id == b_id:
                continue
            if b_scope.scope_id in a_cand and b_scope.iter_out_id in a_cand:
                candidate_parents.append(a_id)
        if not candidate_parents:
            b_scope.parent_scope_id = None
            forest.root_scope_ids.append(b_id)
            continue
        # Pick the innermost parent: the parent whose candidate_members
        # is a (proper) subset of all other candidate parents'.
        innermost: str | None = None
        for p_id in candidate_parents:
            is_innermost = True
            p_cand = candidate_members[p_id]
            for q_id in candidate_parents:
                if q_id == p_id:
                    continue
                # If q's pivots are inside p, q is closer to b (or equal);
                # so p is not innermost.
                q_scope = forest.scopes[q_id]
                if q_scope.scope_id in p_cand and q_scope.iter_out_id in p_cand:
                    is_innermost = False
                    break
            if is_innermost:
                innermost = p_id
                break
        b_scope.parent_scope_id = innermost
        if innermost:
            forest.scopes[innermost].child_scope_ids.append(b_id)
        else:
            # All parents are mutually-incomparable peers — partial overlap.
            errors.append(
                f"Scope '{b_id}' has multiple non-nested parent candidates "
                f"({candidate_parents}). Scopes must be either fully nested "
                f"or fully disjoint, not partially overlapping."
            )
            forest.root_scope_ids.append(b_id)

    # ── Step 5: innermost membership = candidate - union(child candidates)
    for scope_id, scope in forest.scopes.items():
        cand = set(candidate_members[scope_id])
        for child_id in scope.child_scope_ids:
            cand -= candidate_members[child_id]
        # Don't strip the child scope's pivots — they belong to the child,
        # which is innermost for those nodes. The subtraction above already
        # removes them since each pivot is in its own scope's candidate set.
        scope.member_node_ids = cand
        for nid in cand:
            forest.node_to_scope[nid] = scope_id

    # ── Step 5.5: re-assign graphIn / graphOut to the scope
    # they SERVE, not the scope they happen to be reachable from. These
    # node types are sinks-or-sources whose semantic membership is
    # determined by their adjacent edges, not by forward/backward
    # reachability from iter pivots.

    def _innermost_scope_among(node_ids: list[str]) -> str | None:
        """Pick the innermost scope id among the given node ids' scopes.
        Innermost = the deepest in the scope tree (longest parent chain).
        Returns None if none of the node ids are in any author scope.
        """
        candidates: set[str] = set()
        for nid in node_ids:
            sid = forest.node_to_scope.get(nid, GRAPH_SCOPE_ID)
            if sid and sid in forest.scopes:
                candidates.add(sid)
        if not candidates:
            return None

        # Pick the deepest scope by following parent_scope_id.
        def depth(sid: str) -> int:
            d = 0
            cur = sid
            while cur and cur in forest.scopes:
                p = forest.scopes[cur].parent_scope_id
                if p is None:
                    break
                d += 1
                cur = p
            return d

        return max(candidates, key=depth)

    # graphIn: belongs to the innermost scope that any of its OUTGOING
    # targets are in (it's the "parameter slot" of that inner scope).
    for pin_id in graphin_ids:
        targets = out_adj.get(pin_id, [])
        owner = _innermost_scope_among(targets)
        if owner is not None:
            # Re-assign: remove from prior scope's member set, add to owner's
            old = forest.node_to_scope.get(pin_id, GRAPH_SCOPE_ID)
            if old and old in forest.scopes and old != owner:
                forest.scopes[old].member_node_ids.discard(pin_id)
            forest.scopes[owner].member_node_ids.add(pin_id)
            forest.node_to_scope[pin_id] = owner

    # graphOut: belongs to the innermost scope that any of its INCOMING
    # sources are in (it's the "return slot" of that inner scope).
    # Exception: final_* edges from a ROOT scope's iterOut are excluded —
    # a graphOut fed from the root final side is an after-loop eval sink
    # (graph scope), not a scope-internal latch. final_* edges from
    # NESTED iterOuts still bind (composite return-latch pattern).
    def _is_root_scope_node(nid: str) -> bool:
        sid = forest.node_to_scope.get(nid, GRAPH_SCOPE_ID)
        return sid in forest.scopes and forest.scopes[sid].parent_scope_id is None

    _final_fed_sources: dict[str, set[str]] = {}
    for e in graph.edges:
        if (e.sourceHandle or "").startswith("final_") and _is_root_scope_node(e.source):
            _final_fed_sources.setdefault(e.target, set()).add(e.source)
    for pout_id in graphout_ids:
        _skip = _final_fed_sources.get(pout_id, set())
        sources = [src for src in in_adj.get(pout_id, []) if src not in _skip]
        owner = _innermost_scope_among(sources)
        if owner is not None:
            old = forest.node_to_scope.get(pout_id, GRAPH_SCOPE_ID)
            if old and old in forest.scopes and old != owner:
                forest.scopes[old].member_node_ids.discard(pout_id)
            forest.scopes[owner].member_node_ids.add(pout_id)
            forest.node_to_scope[pout_id] = owner

    # ── Step 6: graph-scope nodes (outside any author scope)
    all_member_ids: set[str] = set()
    for s in forest.scopes.values():
        all_member_ids |= s.member_node_ids
    for n in graph.nodes:
        if n.id not in all_member_ids:
            forest.graph_scope_node_ids.append(n.id)
            forest.node_to_scope[n.id] = GRAPH_SCOPE_ID

    # ── Step 7: bind graphIn + graphOut to scopes (now using
    # the post-fixup scope assignments from Step 5.5)
    for pin_id in graphin_ids:
        scope_id = forest.node_to_scope.get(pin_id, GRAPH_SCOPE_ID)
        if scope_id and scope_id in forest.scopes:
            forest.scopes[scope_id].graphin_node_ids.append(pin_id)
    for pout_id in graphout_ids:
        scope_id = forest.node_to_scope.get(pout_id, GRAPH_SCOPE_ID)
        if scope_id and scope_id in forest.scopes:
            forest.scopes[scope_id].graphout_node_ids.append(pout_id)

    # ── Step 8: validate cross-scope wires between AUTHOR scopes go through
    # graphIn/graphOut. The implicit graph scope (id="") is exempt — its
    # connections to/from the outermost author scope (env_reset → iterIn
    # init side, iter_out → evaluate, etc.) are universal in single-scope
    # graphs and should remain free. The structural-IO requirement applies
    # only when crossing between two distinct AUTHOR scopes (e.g. outer
    # iter ↔ inner iter). Pivot nodes (iterIn, iterOut) are themselves the
    # natural boundary into/out of their scope — wires landing on them from
    # graph-scope are part of the single-scope idiom.
    graphin_set = set(graphin_ids)
    graphout_set = set(graphout_ids)
    iter_out_set = set(iter_out_ids)
    iter_in_set = set(iter_in_ids)
    for e in edges:
        sx = forest.node_to_scope.get(e.source, GRAPH_SCOPE_ID)
        sy = forest.node_to_scope.get(e.target, GRAPH_SCOPE_ID)
        if sx == sy:
            continue  # same-scope wire, fine
        # Skip graph-scope ↔ author-scope: these are the entry/sink wires
        # (env_reset → iterIn init side, iter_out → evaluate, ...). The
        # author scope's pivots already name this boundary.
        if sx == GRAPH_SCOPE_ID or sy == GRAPH_SCOPE_ID:
            continue
        # Cross-author-scope wires are legal when crossing through a
        # NAMED IO boundary on either side:
        #   * source is graphOut (inner→outer return)
        #   * target is graphIn (outer→inner parameter)
        #   * source is iterOut (scope's natural exit point) — peer/sequential handoff
        #   * target is iterIn (scope's natural entry point — its init side) — peer/sequential handoff
        source_is_named_exit = e.source in graphout_set or e.source in iter_out_set
        target_is_named_entry = e.target in graphin_set or e.target in iter_in_set
        if not (source_is_named_exit or target_is_named_entry):
            errors.append(
                f"Edge '{e.id}' ({e.source} → {e.target}) crosses an author "
                f"scope boundary (scope '{sx}' → scope '{sy}') without going "
                f"through a named boundary node (graphOut/iterOut on the "
                f"source side, or graphIn/iterIn on the target side)."
            )

    # ── Step 9: write derivative `nested_scope_ids` onto outer iter_in configs
    # (UI hint; canonical config still on each inner iter_in). Mutates
    # graph in place — callers that need a pristine graph should pass a
    # deepcopy. This is the sole mutation in this otherwise-pure function.
    for scope_id, scope in forest.scopes.items():
        node = nodes_by_id.get(scope_id)
        if node is None:
            continue
        if node.config is None:
            node.config = {}
        if scope.child_scope_ids:
            node.config["nested_scope_ids"] = list(scope.child_scope_ids)
        else:
            # Clean stale entry if previous analysis wrote one
            node.config.pop("nested_scope_ids", None)

    return forest, errors
