"""Dataflow execution engine — nodes fire when their inputs arrive.

Replaces the linear DAG executor. Supports cyclic graphs naturally:
data flowing back through edges triggers re-firing.

Key concepts:
- NodeInstance: per-node persistent state + pending inputs
- Entry nodes: fire at startup with no required inputs
- StopExecution: abort signal a node may raise to stop the run
- Step counter: increments when iterOut fires (once per iteration)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..components.bases import FireList
from ..errors import get_bus
from ..graph_def import GraphDefinition
from ..llm.call import _current_node_usage
from ..server.serialization import _current_node_transport
from ..standard.actions import ACTION_NAMES
from ..standard.node_io import get_required_inputs
from ..standard.wire_types import is_list_type
from ..state import broadcast

# Import node handlers from builtin_nodes (built-in node classes + registry)
from .builtin_nodes import NODE_HANDLERS
from .hooks import HookRunner, extract_node_hooks, merge_hooks, safe_serialize
from .scope_analysis import GRAPH_SCOPE_ID, ScopeForest, analyze_scopes

log = logging.getLogger("agentcanvas.graph-executor")

# C.5 Dynamic Fire-List: spawner.execute() may return a FireList; the engine
# fires each spec sequentially through ``_fire_dynamic_children``. Children
# are ephemeral (no entry in ``self.nodes`` / ``adjacency`` / ``scope_forest``)
# and these structural node types are not allowed as children — they would
# either (a) be no-ops outside the static topology, or (b) corrupt scope
# state. See ``components/bases.py:FireSpec`` and the design at
# ``.claude/plans/foamy-kindling-puppy.md``.
_DYNAMIC_FIRELIST_FORBIDDEN_CHILD_TYPES = frozenset(
    {
        "iterIn",
        "iterOut",
        "graphIn",
        "graphOut",
    }
)


def _apply_declarative_aggregator(child_results: list[dict], recipe: dict[str, Any]) -> dict:
    """Collapse children's output dicts using a declarative recipe.

    Used by server-mode spawners that return FireList over HTTP — the
    framework-side proxy doesn't have the original ``aggregate()`` method,
    so the spawner attaches a recipe to ``FireList.aggregator`` and the
    engine applies it framework-side. Recipe shapes documented on
    ``FireList.aggregator``.
    """
    kind = recipe.get("kind", "")
    if kind == "passthrough_last":
        return dict(child_results[-1]) if child_results else {}
    if kind == "passthrough_index":
        idx = int(recipe.get("index", 0))
        if 0 <= idx < len(child_results):
            return dict(child_results[idx])
        return {}
    if kind == "merge_all":
        merged: dict[str, Any] = {}
        for cr in child_results:
            if isinstance(cr, dict):
                merged.update(cr)
        return merged
    if kind == "rename":
        # Wrap a passthrough recipe and rename keys post-hoc.
        inner = _apply_declarative_aggregator(child_results, recipe.get("inner", {}))
        rename_map = recipe.get("map", {})
        return {rename_map.get(k, k): v for k, v in inner.items()}
    raise ValueError(
        f"Unknown FireList.aggregator kind {kind!r}; supported: "
        f"passthrough_last / passthrough_index / merge_all / rename"
    )


# ── Per-scope execution state (multi-scope iteration) ──


@dataclass
class _ScopeState:
    """Runtime state per iteration scope.

    For single-scope graphs (the legacy shape), there is exactly one
    entry keyed by the outermost iter_in's id, plus the synthetic
    GRAPH_SCOPE_ID entry for graph-scope nodes (entry nodes and sinks
    outside any author scope).
    """

    scope_id: str  # iter_in node id, or GRAPH_SCOPE_ID for the graph scope
    parent_scope_id: str | None = None
    step_counter: int = 0
    step_budget: int = 500
    terminated: bool = False
    member_node_ids: set[str] = field(default_factory=set)
    graphout_node_ids: list[str] = field(default_factory=list)
    iter_out_id: str | None = None  # the iterOut node belonging to this scope


# ── Node Instance ──


@dataclass
class NodeInstance:
    """A live node in the dataflow graph with persistent state."""

    id: str
    type: str
    config: dict = field(default_factory=dict)
    label: str = ""
    state: dict = field(default_factory=dict)  # persistent across firings
    pending_inputs: dict = field(default_factory=dict)  # port → value, cleared after firing
    # iterIn only: one slot per declared port. Writers (run-start init
    # edges, iterOut at iter boundary) all land here via
    # ``_write_iterin_slot``. At iterIn fire time the slots become the
    # ``fire_inputs`` dict; after the fire, slots for ports with
    # ``persist=False`` are cleared; persist=True slots keep their value
    # until the next write. Replaces the old ``init_port_cache`` +
    # ``pending_inputs`` split.
    port_slots: dict = field(default_factory=dict)


class NodeErrorAggregate(RuntimeError):
    """Raised at the end of a run when one or more nodes failed mid-run.

    A failed node (exception, or an ``{"error": ...}`` result from e.g.
    a server-mode proxy) starves its downstream but does not stop the
    dataflow loop — without this, the run would finish indistinguishable
    from a clean completion (the ``status="completed"``/``step_count=0``
    eval pathology). The executor records every node error and convicts
    the run here, AFTER the verdict stage, so final-side metrics are
    still collected on the error path.
    """


# ── iterIn helpers ──


def _iterin_port_names(iterin_node: NodeInstance) -> set[str]:
    """Return the set of port names declared in an iterIn node's config.

    Reads the unified ``ports`` list introduced by the per-port persist
    refactor. Legacy configs with ``init_ports`` / ``loop_ports`` must be
    rejected at graph load by the validator (version gate), not silently
    accepted here.
    """
    ports_cfg = iterin_node.config.get("ports") or []
    return {p["name"] for p in ports_cfg if isinstance(p, dict) and "name" in p}


def _iterin_persist_map(iterin_node: NodeInstance) -> dict[str, bool]:
    """Map each iterIn port name to its persist flag (default True)."""
    ports_cfg = iterin_node.config.get("ports") or []
    return {
        p["name"]: bool(p.get("persist", True))
        for p in ports_cfg
        if isinstance(p, dict) and "name" in p
    }


# ── Stop Signal ──


class StopExecution(Exception):
    """Raised by a node to cleanly stop the dataflow loop.

    The engine's own halt decision is the iterOut-boundary stop check
    (the Decide phase); this exception is the escape hatch for node code
    that must abort the run mid-iteration.
    """

    def __init__(self, reason: str = "done"):
        self.reason = reason
        super().__init__(reason)


# ── Graph Executor ──


class GraphExecutor:
    """Execute a graph by dataflow firing — nodes fire when inputs arrive. Cycles supported."""

    def __init__(
        self,
        logger: Any = None,
        env_panel_overrides: dict[str, Any] | None = None,
        server_url_overrides: dict[str, str] | None = None,
    ) -> None:
        self.nodes: dict[str, NodeInstance] = {}
        self.edges: list[dict] = []
        self.adjacency: dict[str, list[dict]] = {}  # source_id → outgoing edges
        self.step_counter: int = 0
        self.step_budget: int = 500
        self.terminated: bool = False
        # ── Multi-scope iteration (additive; 0/1-scope graphs unchanged)
        # scope_state holds per-scope step counters, budgets, stop state.
        # Always includes a synthetic GRAPH_SCOPE_ID entry for nodes outside
        # any author scope. For single-scope graphs there's also one entry
        # keyed by the outermost iter_in id, whose step_counter mirrors
        # self.step_counter.
        self.scope_state: dict[str, _ScopeState] = {}
        self.scope_forest: ScopeForest | None = None
        self._outermost_scope_id: str = GRAPH_SCOPE_ID  # set in run()
        # Final side (ADR: two-sided iterOut). ``_final_ready`` collects the
        # after-loop nodes made ready by the root scope's final_* emission;
        # ``_final_emitted`` guards exactly-once emission per scope.
        self._final_ready: list[str] = []
        self._final_emitted: set[str] = set()
        self._flatten_map: Any = None  # FlattenMap for error tracing
        # State containers (visible shared state)
        self.containers: dict[str, Any] = {}  # container_id → StateContainer
        self._access_grant_index: dict[str, set] = {}  # node_id → {container_ids}
        self._graph_state_id: str | None = None  # well-known "graph_state" container
        # Read-only previews of nodeset-owned (subprocess-local) containers,
        # received via the /call response piggyback (proxy.record path) and
        # merged into the nav_step broadcast. nodeset_name → {cid: {label, states}}.
        self._subprocess_container_previews: dict[str, dict] = {}
        self._hook_runner: HookRunner = HookRunner([])
        # In-memory checkpoint storage: step -> {container_id: snapshot}
        self._checkpoints: dict[int, dict[str, Any]] = {}
        self.max_checkpoints: int = 100
        # Execution logger (optional — captures per-node I/O + timing)
        self._logger: Any = logger  # ExecutionLogger | None
        # Per-runner env panel overrides (ADR-028) — worker-pool eval binds
        # tagged env-panel proxies here so this runner resolves
        # "env_habitat" to its own worker subprocess rather than the global
        # singleton. Empty/None = pass-through to the global registry.
        self._env_panel_overrides: dict[str, Any] = env_panel_overrides or {}
        # Per-runner server URL overrides (ADR-028 PB-1.5) — worker-pool eval
        # binds tagged subprocess URLs here so in-graph proxy node calls
        # (env_habitat__step_native, ...) route to this worker's own env
        # subprocess instead of the URL baked into the proxy class closure
        # at registry-load time. Empty/None = use the baked URL (canvas Play
        # / single-worker bit-identical behaviour).
        self._server_url_overrides: dict[str, str] = server_url_overrides or {}

    def get_env_panel(self, name: str) -> Any:
        """Resolve an env panel, checking per-runner overrides before the global registry.

        Used when a node or orchestration step needs to talk to a nodeset's
        BaseEnvPanel (ADR-025). Worker-pool eval (ADR-028) binds tagged
        proxies via ``env_panel_overrides`` so unqualified lookups route to
        this runner's own env subprocess.
        """
        if name in self._env_panel_overrides:
            return self._env_panel_overrides[name]
        from ..components.env_panel import get_env_panel as _global_get_env_panel

        return _global_get_env_panel(name)

    def get_server_url(self, nodeset_name: str) -> str | None:
        """Resolve the server URL for an in-graph proxy call (ADR-028 PB-1.5).

        Returns the per-runner override if present (worker-pool eval) or
        ``None`` to indicate "fall back to the URL baked into the proxy
        class closure" (canvas Play / single-worker eval). Called from
        ``server/proxy.py:_make_execute`` at every proxy fire.
        """
        return self._server_url_overrides.get(nodeset_name)

    async def run(
        self,
        graph: GraphDefinition,
        session: Any,
        execution_id: str | None = None,
        step_delay_ms: int = 200,
        stop_event: asyncio.Event | None = None,
        pause_event: asyncio.Event | None = None,
        global_hooks: list[Any] | None = None,
        step_budget_override: int | None = None,
    ) -> None:
        """Main graph execution loop.

        ``step_budget_override`` (when set) wins over the graph's authored
        ``step_budget``. Used by the eval batch resolver chain to push an
        env-supplied per-episode value (e.g. HM-EQA's
        ``int(sqrt(scene_size) * 3)``) without mutating the shared graph
        object across worker coroutines.

        Structure (the ``# ───`` banners below mark the phases):
        pre-loop (BUILD steps 1-7, then ENTRY DISCOVERY) → in-loop (the
        firing loop; the iterOut boundary closes each iteration in four
        phases: record → settle → decide → hand off) → after-loop (the
        verdict stage) → finalise. Narrated walkthrough with diagrams:
        docs/pages/developer-guide/design-docs/graph-executor.html (Part I).
        """

        # ─── pre-loop · BUILD — compile the static graph into live state ───

        # BUILD 1/7 — merge hooks from 3 sources: global → graph → node (R3)
        node_hooks = extract_node_hooks(graph.nodes)
        merged = merge_hooks(global_hooks or [], graph.hooks, node_hooks)
        self._hook_runner = HookRunner(merged)

        # BUILD 2/7 — flatten composite nodes before execution (ADR-dataflow-001)
        from .flatten import flatten_graph

        graph, self._flatten_map = flatten_graph(graph)

        # BUILD 3/7 — state containers + access grants from the graph definition
        if graph.containers:
            from .state_containers import build_containers

            self.containers = build_containers(graph.containers)
            for ag in graph.access_grants:
                self._access_grant_index.setdefault(ag.node_id, set()).add(ag.container_id)
            log.info(
                "State containers: %d containers, %d access grants",
                len(self.containers),
                len(graph.access_grants),
            )
            # If a container with the well-known id "graph_state" exists,
            # set up the convenience binding.  Access is still gated by
            # explicit access grants — no auto-inject.
            if "graph_state" in self.containers:
                self._graph_state_id = "graph_state"

        # BUILD 4/7 — resolve the step budget. Order: explicit override →
        # graph authored value → framework default. The eval batch runner
        # computes the override via its resolver chain (env-dynamic → API
        # override).
        from ..config import DEFAULT_STEP_BUDGET as _DEFAULT_STEP_BUDGET

        if step_budget_override is not None:
            self.step_budget = step_budget_override
        elif graph.step_budget is not None:
            self.step_budget = graph.step_budget
        else:
            self.step_budget = _DEFAULT_STEP_BUDGET
        self.edges = [e.to_dict() for e in graph.edges]
        node_defs = graph.nodes

        if not node_defs:
            log.error("Empty graph — nothing to execute")
            return

        # BUILD 5/7 — node instances: the static definition becomes
        # per-run mutable state (pending_inputs buffer + persistent state)
        for nd in node_defs:
            self.nodes[nd.id] = NodeInstance(
                id=nd.id,
                type=nd.type,
                config=nd.config,
                label=nd.label,
            )

        # BUILD 6/7 — per-scope execution state. Always includes a
        # synthetic GRAPH_SCOPE_ID entry; for single-scope graphs there is
        # also one entry keyed by the outermost author scope's iter_in id.
        # Multi-scope graphs add one entry per author scope. The root
        # scope's stop signal halts the entire run; a non-root scope's stop
        # only ends that inner loop and lets the outer one continue.
        self.scope_forest, scope_errors = analyze_scopes(graph)
        if scope_errors:
            for err in scope_errors:
                log.warning("Scope analysis: %s", err)
        # Always create the synthetic graph-scope entry — holds graph-scope
        # nodes (run-start entry nodes, after-loop sinks).
        self.scope_state[GRAPH_SCOPE_ID] = _ScopeState(
            scope_id=GRAPH_SCOPE_ID,
            parent_scope_id=None,
            step_budget=self.step_budget,  # graph-level cap also applies here
            member_node_ids=set(self.scope_forest.graph_scope_node_ids),
        )
        # Per-author-scope state; resolve step_budget with fallback chain
        # (scope's own iter_in.step_budget → parent scope's step_budget →
        # graph step_budget).
        for scope_id, info in self.scope_forest.scopes.items():
            resolved_budget = info.step_budget
            if resolved_budget is None:
                # Walk up parent chain looking for a budget; fall back to graph
                p = info.parent_scope_id
                while p is not None and resolved_budget is None:
                    p_info = self.scope_forest.scopes.get(p)
                    if p_info is None:
                        break
                    resolved_budget = p_info.step_budget
                    p = p_info.parent_scope_id
            if resolved_budget is None:
                resolved_budget = self.step_budget
            self.scope_state[scope_id] = _ScopeState(
                scope_id=scope_id,
                parent_scope_id=info.parent_scope_id,
                step_budget=resolved_budget,
                member_node_ids=set(info.member_node_ids),
                graphout_node_ids=list(info.graphout_node_ids),
                iter_out_id=info.iter_out_id,
            )
        # The "outermost" scope id — used for backward compat: self.step_counter
        # and self.terminated mirror this scope's state. For 0-scope graphs
        # this stays GRAPH_SCOPE_ID; for single-scope graphs it's the sole
        # author scope's id; for multi-scope graphs it's the (single) root.
        if len(self.scope_forest.root_scope_ids) == 1:
            self._outermost_scope_id = self.scope_forest.root_scope_ids[0]
        elif len(self.scope_forest.root_scope_ids) > 1:
            # Multiple peer roots — pick the first deterministically; legacy
            # self.step_counter mirrors only its counter (multi-root graphs
            # are an advanced shape; consumers should read scope_state directly).
            self._outermost_scope_id = self.scope_forest.root_scope_ids[0]
        else:
            self._outermost_scope_id = GRAPH_SCOPE_ID

        # BUILD 7/7 — adjacency: outgoing edges indexed by source id, so
        # routing a fired node's outputs is a single lookup in the loop
        for edge in self.edges:
            src = edge.get("source", "")
            self.adjacency.setdefault(src, []).append(edge)

        # ─── pre-loop · ENTRY DISCOVERY — prime the ready-queue, arm the loop ───

        # A node is an entry node if: explicitly marked OR (has no required
        # inputs AND has no incoming edges). Nodes with incoming edges to
        # optional ports should wait for data even if no required ports exist.
        incoming_targets = set()
        for edge in self.edges:
            tgt = edge.get("target", "")
            if tgt:
                incoming_targets.add(tgt)

        # Entry discovery: queue at run-start iff all three structural
        # conditions hold:
        #   1. type != "iterIn" (iterIn fires from init edges / iterOut transfer)
        #   2. no incoming edges
        #   3. no required input ports
        # validate_graph_connectivity (graph_def.py) rejects required-but-
        # unwired ports at load time, so condition 3 normally only excludes
        # well-formed leaf consumers — never silently drops an entry node.
        # System Log: ready-time stamps for queue_wait_ms (set on enqueue,
        # popped on fire in _fire_node). Run-scoped — reset on each run().
        self._t_ready: dict[str, float] = {}
        ready_queue: list[str] = []
        for node in self.nodes.values():
            if node.type == "iterIn":
                continue
            if node.id in incoming_targets:
                continue
            if self._get_required_ports_for_node(node):
                continue
            self._enqueue(ready_queue, node.id)

        log.info(
            "GraphExecutor: %d nodes, %d edges, entry_nodes=%s, step_budget=%d",
            len(self.nodes),
            len(self.edges),
            [self.nodes[nid].label or nid for nid in ready_queue],
            self.step_budget,
        )

        stop = stop_event or asyncio.Event()
        pause = pause_event or asyncio.Event()
        if not pause.is_set():
            pause.set()

        session._status = "running"
        # Node failures recorded during this run — convicts the run at the
        # finalise stage (see NodeErrorAggregate).
        self.node_errors: list[dict] = []
        _suppress = getattr(getattr(session, "principles", None), "suppress_nav_events", False)
        if not _suppress:
            await broadcast(session._ws("nav_status", {"status": "running", "step": 0}))

        # run_start signal fires before any node fires, so state
        # containers with lifetime="run" reset to a clean baseline.
        self.broadcast_signal("run_start", {"graph_name": graph.name})

        # GraphStart hook
        if self._hook_runner.has_hooks():
            await self._hook_runner.run_hooks(
                "GraphStart",
                payload={
                    "graph_name": graph.name,
                    "node_count": len(self.nodes),
                },
            )

        # Safety: max total firings to prevent infinite loops
        max_firings = self.step_budget * len(self.nodes) * 3
        total_firings = 0

        # ─── in-loop · the firing loop — pop, guard, fire, route ───
        try:
            while ready_queue and total_firings < max_firings:
                # Pause/stop check
                await pause.wait()
                if stop.is_set():
                    break

                node_id = ready_queue.pop(0)
                node = self.nodes.get(node_id)
                if node is None:
                    continue

                # Guard 1 — scope barrier. Once an inner scope is marked
                # terminated (by the iterOut boundary's decide check at the
                # end of the final inner iter), forbid any further fire of nodes belonging
                # to that scope until the scope is re-entered. Without this
                # guard, in-flight inner-body nodes that were enqueued
                # before the scope stopped (e.g. ``move_to_pose`` already
                # running, or ``episode_info`` triggered by it) fire one
                # extra time AFTER the stop and leak per-iter outputs through any
                # cross-scope wires (post-flatten direct edges from inner
                # producers to outer consumers, OR direct graphOut→outer
                # wires when graphOut is preserved), satisfying outer
                # iterOut's required gates and causing outer to cycle
                # without inner actually re-entering.
                #
                # Exemption set — scope-entry cascade nodes that MUST be
                # allowed to fire so the next outer iter can reopen this
                # scope: ``iterIn`` (existing re-entry path below clears
                # the terminated flag; its init slots re-capture
                # outer-supplied scope-entry args), ``graphIn`` (ferries
                # outer→inner values across the boundary into iterIn's
                # init side). Body nodes and iterOut stay barred — they
                # have nothing useful to do between scope terminate and
                # scope re-entry.
                _n_scope_id = self._scope_of(node_id)
                _n_scope = (
                    self.scope_state.get(_n_scope_id)
                    if _n_scope_id and _n_scope_id != GRAPH_SCOPE_ID
                    else None
                )
                if (
                    _n_scope is not None
                    and _n_scope.terminated
                    and _n_scope.parent_scope_id is not None
                    and node.type not in ("iterIn", "graphIn")
                ):
                    node.pending_inputs = {}
                    continue

                # Guard 2 — iterIn step_start. iterIn marks the start of each
                # iteration — emit step_start before the node fires so
                # lifetime="step" states see a fresh slate for the upcoming
                # iteration.
                # Multi-scope re-entry: an inner scope's iter_in firing
                # while its scope is marked terminated indicates outer just
                # entered a new outer iter and is re-invoking inner. Reset
                # inner scope's terminated flag and step_counter — inner
                # gets a fresh loop. (Root-scope iter_in re-firing after
                # termination would only happen if main loop didn't break;
                # we still allow the reset for symmetry, though in practice
                # root termination breaks the loop above.)
                if node.type == "iterIn":
                    _ii_scope = self.scope_state.get(node.id)
                    if _ii_scope is not None and _ii_scope.terminated:
                        if _ii_scope.parent_scope_id is not None:
                            # Inner scope re-entry from outer iter — reset
                            log.info(
                                "Scope re-entry: %s (parent=%s) — resetting terminated/counter",
                                node.id,
                                _ii_scope.parent_scope_id,
                            )
                            _ii_scope.terminated = False
                            _ii_scope.step_counter = 0
                            # Re-arm the final side: an inner scope emits
                            # final_* once per termination, i.e. once per
                            # outer iteration.
                            self._final_emitted.discard(node.id)
                            # Reset per-fire state on body nodes that hold
                            # transient counters (only those tracked via
                            # state['_scoped_reset']=True)? — out of scope
                            # for v1; user tests must use cumulative counters.
                        else:
                            # Root scope: should not happen (main loop breaks
                            # on root termination). Defensive: skip the fire.
                            node.pending_inputs = {}
                            continue
                    # step_start signal — add scope_id (additive, single-scope
                    # readers using just `step` keep working).
                    _next_step = (
                        _ii_scope.step_counter + 1
                        if _ii_scope is not None
                        else self.step_counter + 1
                    )
                    self.broadcast_signal(
                        "step_start",
                        {"step": _next_step, "scope_id": node.id},
                    )

                # Fire the node
                _error_from_exception = False
                try:
                    result = await self._fire_node(node, session)
                except StopExecution as e:
                    log.info(
                        "StopExecution from %s: %s at step %d", node.id, e.reason, self.step_counter
                    )
                    self.terminated = True
                    break
                except Exception as e:
                    origin = self._flatten_map.trace(node.id) if self._flatten_map else node.id
                    # Surface to user via Report tab; result still gets {"error": ...}
                    # so downstream nodes can react to the failure.
                    get_bus().from_exception(
                        e,
                        source="node",
                        code="NODE_EXEC_FAIL",
                        scope={
                            "node_id": node.id,
                            "node_type": node.type,
                            "origin": origin,
                            "step": self.step_counter,
                            "execution_id": getattr(session, "_execution_id", None),
                        },
                        title=f"Node {node.id} ({node.type}) failed",
                    )
                    # Strict mode (AGENTCANVAS_STRICT_ERRORS=1): re-raise after
                    # bus emit so the outer execution loop fails fast instead of
                    # quietly continuing with result={"error": ...}. Designed for
                    # smoke / eval runs where any node failure should be visible.
                    if os.environ.get("AGENTCANVAS_STRICT_ERRORS", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    ):
                        raise
                    result = {"error": str(e)}
                    _error_from_exception = True

                # Error-shaped result — the node reported failure by returning
                # {"error": ...} instead of its declared ports (server-mode
                # proxies surface HTTP failures this way; the exception path
                # above converts to the same shape). Routing would silently
                # drop it: downstream starves and the run drains away as if
                # completed. Record it for the end-of-run conviction, and put
                # returned errors on the bus (the exception path already
                # emitted NODE_EXEC_FAIL). Nodes that legitimately declare an
                # ``error`` output port (e.g. env_libero tools) are exempt.
                if (
                    isinstance(result, dict)
                    and "error" in result
                    and not self._declares_error_output(node)
                ):
                    self.node_errors.append(
                        {
                            "node_id": node.id,
                            "node_type": node.type,
                            "step": self.step_counter,
                            "error": str(result["error"]),
                        }
                    )
                    if not _error_from_exception:
                        get_bus().emit(
                            severity="error",
                            source="node",
                            code="NODE_RESULT_ERROR",
                            title=f"Node {node.id} ({node.type}) returned an error result",
                            message=str(result["error"]),
                            scope={
                                "node_id": node.id,
                                "node_type": node.type,
                                "step": self.step_counter,
                                "execution_id": getattr(session, "_execution_id", None),
                            },
                        )

                total_firings += 1

                # ── iterOut boundary — fires once per iteration; four phases:
                # record → settle → decide → hand off ──
                if node.type == "iterOut":
                    # Resolve which scope this iterOut belongs to. The scope
                    # is keyed by the paired iterIn's id (== scope_id).
                    _io_scope_id = node.config.get("pairedWith", "") or self._outermost_scope_id
                    _io_scope = self.scope_state.get(_io_scope_id)
                    _is_root_scope = _io_scope_id == self._outermost_scope_id

                    # ── boundary phase 1/4 · record — counters, checkpoint, log, nav_step ──
                    # Advance per-scope counter. For backward compat,
                    # self.step_counter mirrors the OUTERMOST scope's counter
                    # (single-scope graphs unchanged).
                    if _io_scope is not None:
                        _io_scope.step_counter += 1
                        _scope_step = _io_scope.step_counter
                    else:
                        _scope_step = self.step_counter + 1
                    if _is_root_scope:
                        self.step_counter += 1
                        session._current_step = self.step_counter

                    _parent_scope_id = _io_scope.parent_scope_id if _io_scope is not None else None
                    log.info(
                        "Iter: scope=%s step=%d (parent=%s, root=%s)",
                        _io_scope_id,
                        _scope_step,
                        _parent_scope_id,
                        _is_root_scope,
                    )

                    self.broadcast_signal(
                        "step_end",
                        {"step": _scope_step, "scope_id": _io_scope_id},
                    )

                    # Checkpoint all containers AFTER iter boundary
                    # Semantics: snapshot = state ready for step N+1 to begin
                    # Only checkpoint on root-scope iterations (single-scope
                    # graphs unchanged; nested inner scopes don't checkpoint)
                    if self.containers and _is_root_scope:
                        self._checkpoints[self.step_counter] = {
                            cid: c.checkpoint() for cid, c in self.containers.items()
                        }
                        if len(self._checkpoints) > self.max_checkpoints:
                            oldest = min(self._checkpoints)
                            del self._checkpoints[oldest]

                    # Flush execution log entries to JSONL at iteration boundary
                    if self._logger:
                        self._logger.flush()

                    # Broadcast consolidated nav_step from all output viewer data
                    # Only on root-scope iter (matches pre-refactor: one
                    # nav_step per outer iteration).
                    if _is_root_scope:
                        await self._broadcast_step(session)

                    # Edges FROM iterOut are final-side only (``final_*``
                    # handles) — they emit once at scope termination via
                    # ``_emit_final_side``, never per-iteration, so iterOut's
                    # own outputs are NOT propagated to adjacency here.

                    # ── boundary phase 2/4 · settle — drain this iter's leftover sinks ──
                    # Settle BEFORE the stop check, so in-loop viewer /
                    # telemetry sinks scheduled in this iter's wave still
                    # emit on the terminal step.
                    # iterIn is excluded — it advances to next iter and is
                    # correctly fired only via the pairedWith handoff after
                    # the stop check.
                    # Multi-scope: settle drain restricted to nodes belonging
                    # to the same scope as the just-fired iterOut. Outer-scope
                    # / peer-scope nodes are not drained by inner iter_out.
                    _max_settle = 64  # safety cap against pathological queues
                    _settle_n = 0
                    while ready_queue and _settle_n < _max_settle:
                        # Skip iterIn — next iter, handled by the pairedWith
                        # handoff after the stop check.
                        _next_idx = None
                        for _i, _nid in enumerate(ready_queue):
                            _n = self.nodes.get(_nid)
                            if _n is None or _n.type == "iterIn":
                                continue
                            # Multi-scope: only drain nodes in the same scope
                            # as the iterOut that triggered this settle pass.
                            # Single-scope graphs: every body node is in the
                            # outermost scope, so this matches today's behaviour.
                            # Exception — ROOT boundaries also drain graph-scope
                            # nodes: dead-end sinks are never on a path to any
                            # iterOut, so scope analysis leaves them in the
                            # graph scope; without this they miss the terminal
                            # step entirely (the run exits with them still
                            # queued). Mid-loop this only fires them earlier
                            # than the main loop would have. Inner boundaries
                            # keep the strict filter — the run continues and
                            # the main loop drains them.
                            _n_scope = self._scope_of(_nid)
                            if (
                                _io_scope is not None
                                and _n_scope != _io_scope_id
                                and not (_is_root_scope and _n_scope == GRAPH_SCOPE_ID)
                            ):
                                continue
                            _next_idx = _i
                            break
                        if _next_idx is None:
                            break  # nothing to settle in this scope
                        _nid = ready_queue.pop(_next_idx)
                        _n = self.nodes[_nid]
                        try:
                            _r = await self._fire_node(_n, session)
                        except Exception:
                            log.exception("settle: error firing %s", _nid)
                            _n.pending_inputs = {}
                            _settle_n += 1
                            continue
                        _n.pending_inputs = {}
                        # Multi-scope: graphOut nodes inside a non-graph
                        # scope BUFFER value into state["latched_value"]
                        # instead of propagating outward. Mirrors the same
                        # suppression in the standard propagation block;
                        # without it, every inner iter would prematurely
                        # propagate graphOut to outer downstream.
                        if _n.type == "graphOut" and self._scope_of(_nid) != GRAPH_SCOPE_ID:
                            if isinstance(_r, dict):
                                _n.state["latched_value"] = _r.get("value")
                            _settle_n += 1
                            continue
                        # Mirror the propagation block below (not extracted
                        # to keep main-loop changes minimal).
                        if isinstance(_r, dict):
                            for _edge in self.adjacency.get(_nid, []):
                                _tgt_id = _edge.get("target", "")
                                if not _tgt_id:
                                    continue
                                _tgt = self.nodes.get(_tgt_id)
                                if _tgt is None:
                                    continue
                                _sh = _edge.get("sourceHandle", "default")
                                _th = _edge.get("targetHandle", _sh)
                                if _sh in _r:
                                    self._route_value_to_port(_tgt, _th, _r[_sh])
                                elif _sh == "default":
                                    self._route_value_to_port(_tgt, _th, _r)
                                if self._is_ready(_tgt):
                                    self._enqueue(ready_queue, _tgt_id)
                        _settle_n += 1

                    # ── boundary phase 3/4 · decide — stop input, then budget check ──
                    # Read the just-fired iterOut's own ``stop`` input. The
                    # stop signal is structurally bound to its scope (it sits
                    # on the scope's own iterOut), and this check runs exactly
                    # once per iteration, after the settle drain and before
                    # the handoff. Unwired stop = budget-only loop.
                    # Termination (stop or budget) emits the terminal
                    # iteration's values exactly once on the final_* handles.
                    _stop = isinstance(result, dict) and bool(result.get("stop"))
                    if _stop:
                        log.info(
                            "Stop: iterOut %s stop=True at scope=%s step=%d",
                            node_id,
                            _io_scope_id or "(graph)",
                            _scope_step,
                        )
                        if _io_scope is not None:
                            _io_scope.terminated = True
                        # Final side: emit the terminal iteration's values
                        # exactly once on the final_* handles.
                        self._emit_final_side(
                            node,
                            result,
                            _io_scope_id,
                            ready_queue=None if _is_root_scope else ready_queue,
                        )
                        if _is_root_scope:
                            self.terminated = True
                            break

                    # Per-scope step_budget exhaust check.
                    _scope_budget = (
                        _io_scope.step_budget if _io_scope is not None else self.step_budget
                    )
                    if _scope_step >= _scope_budget:
                        log.info(
                            "Step budget (%d) exhausted for scope=%s",
                            _scope_budget,
                            _io_scope_id or "(graph)",
                        )
                        if _io_scope is not None:
                            _io_scope.terminated = True
                        self._emit_final_side(
                            node,
                            result if isinstance(result, dict) else {},
                            _io_scope_id,
                            ready_queue=None if _is_root_scope else ready_queue,
                        )
                        if _is_root_scope:
                            self.terminated = True
                            break
                        # Inner-scope budget exhaust: don't break root loop;
                        # propagate inner graphOut latches and let outer continue
                        self._propagate_graphout_latches(_io_scope, ready_queue)
                        # Skip the iterIn re-queue below (scope is done)
                        continue

                    if step_delay_ms > 0 and _is_root_scope:
                        await asyncio.sleep(step_delay_ms / 1000)

                    # If this scope was just stopped (inner-scope stop above,
                    # without breaking the root loop), propagate graphOut
                    # latch + skip the iterIn re-queue.
                    if _io_scope is not None and _io_scope.terminated:
                        self._propagate_graphout_latches(_io_scope, ready_queue)
                        continue

                    # ── boundary phase 4/4 · hand off — carry → paired iterIn, next iteration ──
                    # iterIn slots are prefixed with "iterout_" for
                    # iterOut writes (always-prefix synthesis). Each iterOut
                    # output key ``X`` maps to iterIn slot ``iterout_<X>``.
                    # ``stop`` is not loop-carried; final_* handles never fire here.
                    paired_id = node.config.get("pairedWith", "")
                    paired = self.nodes.get(paired_id) if paired_id else None
                    if paired and paired.type == "iterIn":
                        slot_names = _iterin_port_names(paired)
                        for key, val in result.items():
                            slot_key = f"iterout_{key}"
                            if slot_key in slot_names:
                                self._write_iterin_slot(paired, slot_key, val)
                        self._enqueue(ready_queue, paired.id)

                    # Standard propagation (line ~750 below) is skipped for
                    # iterOut — adjacency was already propagated above (before
                    # the settle drain) so latch buffers see the final iter
                    # value. Clear pending_inputs and continue.
                    node.pending_inputs = {}
                    continue

                # ── ordinary path — consume · special cases · route · enqueue ──
                # Clear fired node's inputs (consumed)
                node.pending_inputs = {}

                # iterIn: after fire, clear slots for ports with persist=False
                # so they don't re-emit next iteration. persist=True slots
                # stay populated until the next write by a matching writer
                # (run-start init edge, or iterOut at iter boundary).
                # Ordering — fire → clear → re-enqueue — persist=False
                # semantics: emit once on the step of the write, then gone.
                if node.type == "iterIn":
                    persist_map = _iterin_persist_map(node)
                    for slot_name, keep in persist_map.items():
                        if not keep:
                            node.port_slots.pop(slot_name, None)

                # Multi-scope: graphOut nodes inside a non-graph scope BUFFER
                # their value into ``state["latched_value"]`` instead of
                # propagating out immediately. ``_propagate_graphout_latches``
                # flushes the latch to outer-scope downstream when the
                # owning scope terminates. graphOuts in the graph scope (eval
                # graph metric harvest) propagate normally as before.
                if node.type == "graphOut":
                    _po_scope = self._scope_of(node_id)
                    if _po_scope != GRAPH_SCOPE_ID and isinstance(result, dict):
                        node.state["latched_value"] = result.get("value")
                        if total_firings <= 5:
                            log.info(
                                "After %s: queue=%s",
                                node.label or node.id,
                                [self.nodes[nid].label or nid for nid in ready_queue],
                            )
                        continue  # skip standard propagation block below

                # Propagate outputs to downstream nodes
                for edge in self.adjacency.get(node_id, []):
                    target_id = edge.get("target", "")
                    target = self.nodes.get(target_id)
                    if target is None:
                        continue

                    src_handle = edge.get("sourceHandle", "default")
                    tgt_handle = edge.get("targetHandle", src_handle)

                    # Route specific port value — the _route_value_to_port
                    # helper applies LIST[T] coercion (ADR-027) so scalar
                    # producers feeding a LIST[T] consumer are auto-wrapped
                    # and fan-in concatenates in edge declaration order.
                    if isinstance(result, dict) and src_handle in result:
                        self._route_value_to_port(target, tgt_handle, result[src_handle])
                    elif isinstance(result, dict) and src_handle == "default":
                        self._route_value_to_port(target, tgt_handle, result)

                    # Check if target is now ready to fire
                    is_ready = self._is_ready(target)
                    if is_ready:
                        self._enqueue(ready_queue, target_id)

                if total_firings <= 5:
                    log.info(
                        "After %s: queue=%s",
                        node.label or node.id,
                        [self.nodes[nid].label or nid for nid in ready_queue],
                    )

            # ─── after-loop · verdict stage — fallback emission + band drain ───
            # If the loop ended without a boundary final-side emission
            # (drained queue, max_firings, user stop), reconstruct the last
            # completed iteration's values from the paired iterIn's slots
            # and emit best-effort; then drain the after-loop band — the
            # nodes fed by the root iterOut's final_* handles (evaluate,
            # graphOut chains). This is the run-end verdict stage.
            self._emit_final_fallback()
            await self._after_loop_pass(session)

            # ─── conviction — surface node failures at end of run ───
            # Runs AFTER the verdict stage so final-side metrics are already
            # collected; the raise routes through the error path below and
            # the run finishes with status="error" instead of masquerading
            # as completed (both stages are idempotent on the re-entry).
            if self.node_errors:
                _n_err = len(self.node_errors)
                _head = "; ".join(
                    f"{e['node_id']}@step{e['step']}: {e['error']}" for e in self.node_errors[:3]
                )
                raise NodeErrorAggregate(
                    f"{_n_err} node error(s) during run: {_head}" + ("; ..." if _n_err > 3 else "")
                )

            # ─── finalise — clean exit ───
            session._status = "done"
            metrics = None
            # Find metrics from any node that stored them (generic — any step node)
            for node in self.nodes.values():
                if node.state.get("metrics"):
                    metrics = node.state["metrics"]
            if session._metrics:
                metrics = session._metrics

            # run_end signal — fires after the loop finishes cleanly.
            # lifetime="run" states clear here.
            self.broadcast_signal(
                "run_end",
                {"step": self.step_counter, "terminated": self.terminated},
            )

            if not _suppress:
                await broadcast(
                    session._ws(
                        "nav_complete",
                        {"step": self.step_counter, "metrics": metrics},
                    )
                )

            # GraphComplete hook
            if self._hook_runner.has_hooks():
                await self._hook_runner.run_hooks(
                    "GraphComplete",
                    payload={
                        "step": self.step_counter,
                        "terminated": self.terminated,
                    },
                )

            # Final flush of any remaining log entries
            if self._logger:
                self._logger.flush()

        except Exception as e:
            # ─── finalise — error path (verdict first, then error reporting) ───
            # Best-effort after-loop stage first: a final-side evaluate can
            # still emit fresh metrics even when the main loop crashed, so
            # eval harvest reports the actual final env step count instead
            # of a stale pre-crash value. Tolerant of cascading failures
            # (the env may be torn down already).
            try:
                self._emit_final_fallback()
                await self._after_loop_pass(session)
            except Exception as _fpe:
                log.warning("after_loop_pass on error path raised: %s", _fpe)
            get_bus().from_exception(
                e,
                source="graph",
                code="NODE_ERRORS" if isinstance(e, NodeErrorAggregate) else "GRAPH_CRASH",
                scope={
                    "step": self.step_counter,
                    "execution_id": getattr(session, "_execution_id", None),
                },
                title=(
                    f"Run finished with node errors at step {self.step_counter}"
                    if isinstance(e, NodeErrorAggregate)
                    else f"Graph execution crashed at step {self.step_counter}"
                ),
            )
            session._status = "error"
            # run_end also fires on the error path so lifetime="run"
            # states reset even when the loop crashes.
            self.broadcast_signal(
                "run_end",
                {"step": self.step_counter, "error": str(e)},
            )
            if not _suppress:
                await broadcast(session._ws("nav_status", {"status": "error", "error": str(e)}))

            # GraphError hook — fires on unhandled exceptions
            if self._hook_runner.has_hooks():
                await self._hook_runner.run_hooks("GraphError", payload={"error": str(e)})

            # Flush log entries even on error
            if self._logger:
                self._logger.flush()

    def _enqueue(self, ready_queue: list[str], node_id: str) -> None:
        """Append ``node_id`` to the ready queue if absent, stamping its
        ready-time so ``_fire_node`` can derive ``queue_wait_ms`` (time spent
        waiting in the queue before firing). Stamps only on a real enqueue
        transition; a node already queued keeps its original ready-time.
        Terminal-band enqueues (final side / latch flush) bypass this helper,
        so those firings simply report ``queue_wait_ms=None``."""
        if node_id in ready_queue:
            return
        ready_queue.append(node_id)
        self._t_ready[node_id] = time.perf_counter()

    def _declares_error_output(self, node: NodeInstance) -> bool:
        """True when the node's handler legitimately declares an ``error``
        output port (e.g. env_libero tools, navgpt navigate) — for those,
        an ``error`` key in the result is data, not a failure report."""
        node_cls = NODE_HANDLERS.get(node.type)
        if node_cls is None:
            return False
        resolver = getattr(node_cls, "_resolve_ports", None)
        if callable(resolver):
            try:
                _ins, outs = resolver(node.config or {})
                return any(getattr(p, "name", None) == "error" for p in outs)
            except Exception:
                return True  # can't resolve — don't misconvict the node
        return any(getattr(p, "name", None) == "error" for p in getattr(node_cls, "output_ports", []))

    async def _fire_node(self, node: NodeInstance, session: Any) -> dict:
        """Execute a node's handler with its pending inputs and persistent state."""
        # System Log: pop the ready-time stamped at enqueue; queue_wait is the
        # gap from "became ready" to "fires now" (None for terminal-band fires).
        _t_ready_map = getattr(self, "_t_ready", None)
        t_ready = _t_ready_map.pop(node.id, None) if _t_ready_map is not None else None
        queue_wait_ms = (time.perf_counter() - t_ready) * 1000 if t_ready is not None else None
        node_cls = NODE_HANDLERS.get(node.type)
        if node_cls is None:
            log.warning("No handler for node type: %s (id=%s)", node.type, node.id)
            return {}

        # Instantiate BaseCanvasNode subclass with per-node config
        instance = node_cls()
        instance.config = node.config
        instance.node_id = node.id

        # Create a minimal ctx-like object that handlers can use
        ctx = _NodeStateProxy(node.state, self.step_counter, session)
        # ADR-028 PB-1.5: expose the executor so in-graph proxy nodes can
        # resolve per-runner server URL overrides at call time (worker-pool
        # eval routes each worker's proxy calls to its own env subprocess).
        ctx._executor = self

        # Inject granted state containers (access granted via access grants).
        # C.5: dynamic-firelist children inherit grants from their spawner.
        # Ephemeral child node ids (``{spawner.id}::dyn{i}``) never appear in
        # ``_access_grant_index``, so without this fallback the child would
        # see no containers — which breaks any LMP child that needs the
        # per-episode runtime container shared with its spawner.
        _grant_lookup_id = getattr(node, "_dynamic_parent_id", None) or node.id
        connected = self._access_grant_index.get(_grant_lookup_id, set())
        if connected:
            ctx._containers = {
                cid: self.containers[cid] for cid in connected if cid in self.containers
            }
            # Cross-nodeset container-access prototype (faces A/C): granted cids
            # NOT homed in the executor are homed in a server subprocess. Inject
            # a RemoteContainerProxy that brokers through the executor's own
            # internal endpoint to that subprocess. Lazy imports keep this off
            # the hot path + avoid an import cycle. (A home node reaching a
            # sub-owned container = face A; a proxy node getting the cid here is
            # what later lets its subprocess reach it = face C.)
            missing = [c for c in connected if c not in self.containers]
            if missing:
                import os as _os

                exec_id = getattr(session, "_execution_id", None)
                reg = None
                try:
                    from ..state import get_services

                    reg = get_services().workspace_component_registry
                except Exception:
                    reg = None
                if reg is not None and exec_id:
                    from ..config import resolve_executor_url

                    base_url = _os.environ.get("AGENTCANVAS_EXECUTOR_URL") or resolve_executor_url()
                    from ..server.remote_container import RemoteContainerProxy

                    for cid in missing:
                        if reg.get_container_home_url(cid):
                            ctx._containers[cid] = RemoteContainerProxy(base_url, exec_id, cid)
                        else:
                            # Was a silent drop — the node would just see no
                            # container under this id. Surface it loudly.
                            log.error(
                                "node %s granted container %r but it is neither "
                                "executor-home nor a known sub-home — not injected.",
                                node.id,
                                cid,
                            )
                else:
                    log.error(
                        "node %s has cross-process container grant(s) %s but "
                        "cannot inject a RemoteContainerProxy (%s).",
                        node.id,
                        missing,
                        "no execution_id" if reg is not None else "no component registry",
                    )

        # Convenience binding: the well-known "graph_state" container is
        # still exposed as ctx._graph_state, but only when this node has
        # an explicit access grant to it.  No auto-inject.
        if (
            self._graph_state_id
            and self._graph_state_id in connected
            and self._graph_state_id in self.containers
        ):
            ctx._graph_state = self.containers[self._graph_state_id]

        # PreNodeExecute hook — can block or modify inputs
        if self._hook_runner.has_hooks():
            pre_result = await self._hook_runner.run_hooks(
                "PreNodeExecute",
                node_type=node.type,
                node_id=node.id,
                payload={
                    "node_id": node.id,
                    "inputs": safe_serialize(node.pending_inputs),
                },
            )
            if pre_result.action == "block":
                log.info("Hook blocked node %s (%s)", node.id, node.type)
                return {}
            if pre_result.action == "modify" and pre_result.modified_data:
                node.pending_inputs.update(pre_result.modified_data)

        # iterIn: fire_inputs = port_slots (one slot per port, keyed by
        # unprefixed port name). Writers (run-start init edges, iterOut)
        # have all landed here via ``_write_iterin_slot``.
        # After the fire, persist=False slots are cleared; persist=True
        # slots are left in place until next write.
        fire_inputs = dict(node.port_slots) if node.type == "iterIn" else node.pending_inputs

        exec_error: Exception | None = None
        # Per-node LLM usage bucket — populated by every llm_complete /
        # vlm_complete call reached during this execute(). See
        # ``app.llm.call._current_node_usage``. The ContextVar is set
        # for ALL node firings (cheap), so any future node that calls
        # the LLM helpers automatically gets accounting without code
        # changes.
        usage_bucket: dict = {
            "calls": 0,
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "usd_cost": 0.0,
        }
        transport_bucket: dict = {
            "calls": 0,
            "rtt_ms": 0.0,
            "req_bytes": 0,
            "resp_bytes": 0,
            "serialize_ms": 0.0,
            "deserialize_ms": 0.0,
        }
        usage_token = _current_node_usage.set(usage_bucket)
        transport_token = _current_node_transport.set(transport_bucket)
        t0 = time.perf_counter()
        try:
            result = await instance.forward(fire_inputs, ctx)
        except Exception as e:
            exec_error = e
            result = {"error": str(e)}
        finally:
            _current_node_usage.reset(usage_token)
            _current_node_transport.reset(transport_token)
            if usage_bucket["calls"] > 0:
                # Round usd_cost to 6 dp so log entries stay readable
                usage_bucket["usd_cost"] = round(usage_bucket["usd_cost"], 6)
                instance._log_buffer.append({"key": "usage", "value": usage_bucket})
            if transport_bucket["calls"] > 0:
                instance._log_buffer.append({"key": "transport", "value": transport_bucket})
        duration_ms_total = (time.perf_counter() - t0) * 1000
        # System Log P2: split server-mode transport out of compute. A proxy
        # node's forward() time includes the HTTP round-trip; subtract it so
        # compute_ms is the node's own work (≈0 for a pure proxy). Local nodes
        # have no transport bucket entries → compute_ms == duration_ms_total.
        if transport_bucket["calls"] > 0:
            transport_ms = round(transport_bucket["rtt_ms"], 2)
            transfer_bytes = transport_bucket["req_bytes"] + transport_bucket["resp_bytes"]
            compute_ms = max(0.0, duration_ms_total - transport_bucket["rtt_ms"])
        else:
            transport_ms = None
            transfer_bytes = None
            compute_ms = duration_ms_total

        # Live per-node usage event — same node_id routing as viewer_data,
        # so the canvas card can show last-call tokens/latency as a gauge.
        if usage_bucket["calls"] > 0 and getattr(ctx, "session", None):
            await broadcast(
                ctx.session._ws(
                    "llm_usage",
                    {
                        "node_id": node.id,
                        "step": getattr(ctx, "step", 0),
                        "usage": {
                            **usage_bucket,
                            "duration_ms": round(duration_ms_total, 1),
                        },
                    },
                )
            )

        # C.5 Dynamic Fire-List dispatch.
        # A DynamicFireListNode subclass returns a FireList sentinel from
        # ``execute()`` instead of a port-output dict. The engine fires each
        # ``FireSpec`` sequentially via ``_fire_dynamic_children`` (which
        # recurses through this same ``_fire_node`` so children get the full
        # log/hook/ctx machinery), then calls ``instance.aggregate(...)`` to
        # collapse the children's outputs back into the spawner's declared
        # ports. The aggregated dict is merged with the spawner's direct
        # ``spawner_outputs`` (aggregated wins on collision) and replaces
        # ``result`` for downstream hook / log / propagation. ``duration_ms_total``
        # above intentionally measures the spawner's own ``execute()`` only;
        # each child has its own log entry with its own ``duration_ms``.
        if isinstance(result, FireList) and exec_error is None:
            fire_list_obj = result
            child_results = await self._fire_dynamic_children(
                spawner=node,
                fire_list=fire_list_obj,
                session=session,
            )
            try:
                # Declarative aggregator (FireList.aggregator) wins when set —
                # used by server-mode proxies that don't have the original
                # ``aggregate()`` method. Local DynamicFireListNode subclasses
                # leave aggregator empty and rely on ``instance.aggregate``.
                if fire_list_obj.aggregator:
                    aggregated = _apply_declarative_aggregator(
                        child_results, fire_list_obj.aggregator
                    )
                else:
                    aggregator_fn = getattr(instance, "aggregate", None)
                    if aggregator_fn is None:
                        raise TypeError(
                            f"Spawner {node.id!r} ({type(instance).__name__}) "
                            f"returned FireList without a declarative "
                            f"``aggregator`` and has no ``aggregate()`` "
                            f"method. Either subclass DynamicFireListNode "
                            f"and implement aggregate(), or set "
                            f"FireList.aggregator to a declarative recipe."
                        )
                    aggregated = aggregator_fn(child_results)
            except Exception as e:
                exec_error = e
                aggregated = {"error": str(e)}
            if not isinstance(aggregated, dict):
                exec_error = TypeError(
                    f"Aggregator must return dict, got "
                    f"{type(aggregated).__name__} from node {node.id!r}"
                )
                aggregated = {"error": str(exec_error)}
            # aggregated wins on key collision with spawner_outputs.
            result = {**fire_list_obj.spawner_outputs, **aggregated}

        # PostNodeExecute hook — fires on both success and error
        if self._hook_runner.has_hooks():
            post_result = await self._hook_runner.run_hooks(
                "PostNodeExecute",
                node_type=node.type,
                node_id=node.id,
                payload={
                    "node_id": node.id,
                    "outputs": safe_serialize(result),
                    "error": str(exec_error) if exec_error else None,
                },
            )
            if post_result.action == "modify" and post_result.modified_data and not exec_error:
                result = post_result.modified_data

        # Build port wire type mapping for log enrichment.  Honours
        # configurable-port overrides (_resolve_ports) so LIST[T] shows up
        # in the log panel even when the default schema is a scalar T.
        _port_wt: dict[str, str] = {}
        if node_cls is not None:
            _resolver = getattr(node_cls, "_resolve_ports", None)
            if callable(_resolver):
                try:
                    _in_ports, _out_ports = _resolver(node.config)
                except Exception:
                    _in_ports = list(getattr(node_cls, "input_ports", []))
                    _out_ports = list(getattr(node_cls, "output_ports", []))
            else:
                _in_ports = list(getattr(node_cls, "input_ports", []))
                _out_ports = list(getattr(node_cls, "output_ports", []))
            for p in list(_in_ports) + list(_out_ports):
                if hasattr(p, "name") and hasattr(p, "wire_type"):
                    _port_wt[p.name] = p.wire_type

        # Log node firing — captures post-hook outputs and voluntary inner log.
        # For dynamic-firelist children, ``_dynamic_parent_id`` /
        # ``_dynamic_index`` were attached to the ephemeral NodeInstance in
        # ``_fire_dynamic_children``; they propagate into the log entry so
        # downstream trace readers can reconstruct the spawner→child hierarchy.
        if self._logger:
            entry = self._logger.log_node(
                step=self.step_counter,
                node_id=node.id,
                node_type=node.type,
                node_label=node.label or node.id,
                duration_ms=duration_ms_total,
                inputs=node.pending_inputs,
                outputs=result,
                inner_log=instance.log(),
                port_wire_types=_port_wt,
                error=str(exec_error) if exec_error else None,
                parent_node_id=getattr(node, "_dynamic_parent_id", None),
                dynamic_index=getattr(node, "_dynamic_index", None),
                queue_wait_ms=queue_wait_ms,
                compute_ms=compute_ms,
                transport_ms=transport_ms,
                transfer_bytes=transfer_bytes,
            )
            # Broadcast exec_log WS event (suppressed in eval mode)
            _suppress_log = getattr(
                getattr(session, "principles", None), "suppress_nav_events", False
            )
            if not _suppress_log:
                await broadcast(
                    session._ws(
                        "exec_log",
                        {
                            "step": entry.step,
                            "node_id": entry.node_id,
                            "node_type": entry.node_type,
                            "node_label": entry.node_label,
                            "duration_ms": entry.duration_ms,
                            "queue_wait_ms": entry.queue_wait_ms,
                            "compute_ms": entry.compute_ms,
                            "error": entry.error,
                            "has_inner_log": bool(entry.inner_log),
                        },
                    )
                )

        # Re-raise if execute() failed (after hooks had a chance to observe)
        if exec_error is not None:
            raise exec_error

        # Runner-level metrics — exposed via the status API
        # (``LoopRunner.get_status``). Set when any node emits
        # ``{"done": True, "metrics": ...}`` (typically the env evaluate node).
        if isinstance(result, dict) and result.get("done") and result.get("metrics"):
            session._metrics = result["metrics"]

        # graphOut: snapshot pending_inputs into ``state["_last_inputs"]``.
        # This is the canonical run-output channel — read by both
        # ``_broadcast_step`` (for the nav_step WS event) and
        # ``BatchEvalRunner._collect_metrics`` (for ``aggregate_by_task``).
        # When a graph runs at top level, every graphOut fires and is
        # snapshotted; when used as a composite, ``flatten_graph`` rewires
        # graphOut nodes through to the parent so this branch becomes inert.
        if node.type == "graphOut":
            node.state["_last_inputs"] = dict(node.pending_inputs)

        return result

    async def _fire_dynamic_children(
        self,
        spawner: NodeInstance,
        fire_list: FireList,
        session: Any,
    ) -> list[dict]:
        """Sequentially fire each FireSpec from a DynamicFireListNode spawner.

        Children are ephemeral NodeInstances — they are NOT registered in
        ``self.nodes``, NOT in ``adjacency`` / ``scope_forest``, and NEVER
        enter the dataflow ready_queue. Each child runs through the existing
        ``_fire_node`` so it gets the full hook / log / ctx machinery (and so
        its log entry carries provenance back to this spawner via
        ``_dynamic_parent_id`` / ``_dynamic_index``).

        Children DO NOT auto-wire to each other: this method collects each
        child's result dict and returns the ordered list. The spawner's
        :meth:`aggregate` is responsible for any cross-child dataflow (and
        in practice, most cross-child state lives in a shared runtime
        container, not on wires).

        Forbidden child types (control / boundary): see
        ``_DYNAMIC_FIRELIST_FORBIDDEN_CHILD_TYPES``.

        Sequential semantics: if a child raises, subsequent children are NOT
        fired; the exception propagates to the spawner. Nested ``FireList``
        (child itself returns one) is rejected as ``NotImplementedError`` —
        upgrading to nested subgraphs is option "C" in the design, deferred.
        """
        results: list[dict] = []
        for idx, spec in enumerate(fire_list.specs):
            if spec.node_type in _DYNAMIC_FIRELIST_FORBIDDEN_CHILD_TYPES:
                raise ValueError(
                    f"DynamicFireListNode {spawner.id!r} cannot dispatch a "
                    f"child of type {spec.node_type!r}: control / boundary "
                    f"node types are forbidden as dynamic children "
                    f"(forbidden set: "
                    f"{sorted(_DYNAMIC_FIRELIST_FORBIDDEN_CHILD_TYPES)})"
                )

            child_id = f"{spawner.id}::dyn{idx}"
            child = NodeInstance(
                id=child_id,
                type=spec.node_type,
                config=dict(spec.config),
                label=spec.label or f"{spec.node_type}#{idx}",
            )
            child.pending_inputs = dict(spec.inputs)
            # Provenance — consumed by:
            #   * the access-grant lookup (so child inherits spawner's grants)
            #   * the logger (so the log entry carries parent_node_id /
            #     dynamic_index for Phase 2 hierarchical trace)
            child._dynamic_parent_id = spawner.id  # type: ignore[attr-defined]
            child._dynamic_index = idx  # type: ignore[attr-defined]

            child_result = await self._fire_node(child, session)

            if isinstance(child_result, FireList):
                raise NotImplementedError(
                    f"Nested FireList returned from child {child_id!r} "
                    f"(type={spec.node_type!r}) is not supported in C.5. "
                    f"Promote the dynamic structure into the static graph, "
                    f"or escalate to the full subgraph primitive (option C "
                    f"in the design)."
                )

            if spec.capture_outputs is not None and isinstance(child_result, dict):
                child_result = {k: v for k, v in child_result.items() if k in spec.capture_outputs}

            results.append(child_result if isinstance(child_result, dict) else {})

        return results

    def _get_required_ports_for_node(self, node: NodeInstance) -> set[str]:
        """Resolve the set of required input ports for a specific node instance.

        Combines class-level declarations (``get_required_inputs``) with
        instance-resolved ports from ``_resolve_ports(config)``, so dynamic
        nodes like ``llmCall`` or ``iterIn`` can mark a config-declared port
        ``required: true`` and the graph executor will respect it in both
        entry discovery and readiness checks.
        """
        required: set[str] = set(get_required_inputs(node.type))
        node_cls = NODE_HANDLERS.get(node.type)
        if node_cls is None:
            return required
        resolver = getattr(node_cls, "_resolve_ports", None)
        if callable(resolver):
            try:
                input_ports, _ = resolver(node.config)
            except Exception:
                input_ports = []
            for p in input_ports:
                if not getattr(p, "optional", True):
                    name = getattr(p, "name", "")
                    if name:
                        required.add(name)
        return required

    def _get_input_port_wire_type(self, node: NodeInstance, port_name: str) -> str | None:
        """Resolve the wire type of a consumer port on a live NodeInstance.

        Honours configurable-port nodes (``_resolve_ports``) so instance-level
        ``config.ports`` overrides are picked up (ADR-024).
        """
        node_cls = NODE_HANDLERS.get(node.type)
        if node_cls is None:
            return None
        resolver = getattr(node_cls, "_resolve_ports", None)
        if callable(resolver):
            try:
                input_ports, _ = resolver(node.config)
            except Exception:
                input_ports = list(getattr(node_cls, "input_ports", []))
        else:
            input_ports = list(getattr(node_cls, "input_ports", []))
        for p in input_ports:
            if getattr(p, "name", None) == port_name:
                return getattr(p, "wire_type", None)
        return None

    def _write_iterin_slot(self, iterin_node: NodeInstance, name: str, value: Any) -> None:
        """Unified write path for iterIn port slots.

        Both writer sites use this: the iterOut→iterIn iter-boundary
        transfer, and canvas edges targeting iterIn's init side (run-start
        init edges). Last write wins; writes never race because the writers
        fire at disjoint times.

        LIST[T] fan-in concatenation (the scalar → list append logic in
        ``_route_value_to_port``) is deliberately NOT applied here:
        persist=True on a LIST[T] port means "carry the last written list",
        not "accumulate across iterations".
        """
        iterin_node.port_slots[name] = value

    def _route_value_to_port(self, target: NodeInstance, tgt_handle: str, value: Any) -> None:
        """Assign ``value`` to ``target.pending_inputs[tgt_handle]`` with
        LIST[T] coercion and fan-in concatenation (ADR-027).

        Rules (non-iterIn targets):
        - Non-list consumer port: write-through (overwrite prior value).
        - ``LIST[T]`` consumer port:
            * incoming scalar → append to the existing list (or start a
              new length-1 list)
            * incoming list → extend (flatten fan-in of list producers)
          Fan-in order is determined by edge declaration order, because the
          adjacency iteration is stable.

        iterIn special-case: the port surface is one unified ``ports`` list;
        ``tgt_handle`` is the port name directly (no prefix). Writes go to
        ``port_slots[name]`` via ``_write_iterin_slot`` with overwrite
        semantics — LIST[T] fan-in is suppressed so persist=True ports don't
        accumulate unbounded.
        """
        if target.type == "iterIn":
            self._write_iterin_slot(target, tgt_handle, value)
            return
        declared = self._get_input_port_wire_type(target, tgt_handle)
        if declared and is_list_type(declared):
            existing = target.pending_inputs.get(tgt_handle)
            if not isinstance(existing, list):
                existing = []
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
            target.pending_inputs[tgt_handle] = existing
            return
        target.pending_inputs[tgt_handle] = value

    def _scope_of(self, node_id: str) -> str:
        """Look up the innermost scope id for a node (multi-scope helper).

        Returns ``GRAPH_SCOPE_ID`` for nodes outside any author scope or
        when the scope_forest hasn't been built yet (defensive fallback for
        non-multi-scope code paths).
        """
        if self.scope_forest is None:
            return GRAPH_SCOPE_ID
        return self.scope_forest.node_to_scope.get(node_id, GRAPH_SCOPE_ID)

    def _emit_final_side(
        self,
        iter_out: NodeInstance,
        values: dict,
        scope_id: str,
        ready_queue: list | None = None,
    ) -> None:
        """Emit the final side of a terminating scope's iterOut — once.

        Routes the terminal iteration's collected values along the iterOut's
        outgoing ``final_<name>`` edges, plus the constant ``final_stop=True``
        (the canonical after-loop trigger). Targets that become ready are
        enqueued onto ``ready_queue`` when given (inner-scope termination —
        the outer loop keeps running and consumes them) or collected into
        ``self._final_ready`` (root scope — drained by ``_after_loop_pass``).

        Graph-scope ``graphOut`` targets are routed-to and snapshotted into
        ``state["_last_inputs"]`` (the harvest channel) but never fired.

        Exactly-once per scope: guarded by ``self._final_emitted``.
        """
        if scope_id in self._final_emitted:
            return
        self._final_emitted.add(scope_id)

        payload: dict[str, Any] = {
            f"final_{k}": v for k, v in (values or {}).items() if k != "stop"
        }
        payload["final_stop"] = True

        sink = self._final_ready if ready_queue is None else ready_queue
        for edge in self.adjacency.get(iter_out.id, []):
            src_handle = edge.get("sourceHandle", "")
            if src_handle not in payload:
                continue
            target_id = edge.get("target", "")
            target = self.nodes.get(target_id)
            if target is None:
                continue
            tgt_handle = edge.get("targetHandle", src_handle)
            self._route_value_to_port(target, tgt_handle, payload[src_handle])
            if target.type == "graphOut":
                if self._scope_of(target_id) == GRAPH_SCOPE_ID:
                    # Eval-output sink: snapshot the harvest channel.
                    if target.pending_inputs:
                        target.state["_last_inputs"] = dict(target.pending_inputs)
                else:
                    # Composite return bridge (inner-scope graphOut): write
                    # the latch; ``_propagate_graphout_latches`` flushes it
                    # to outer-scope consumers right after this emission.
                    target.state["latched_value"] = payload[src_handle]
                continue
            if self._is_ready(target) and target_id not in sink:
                sink.append(target_id)
        log.info(
            "Final side: iterOut %s emitted %d handle(s) for scope=%s",
            iter_out.id,
            len(payload),
            scope_id or "(graph)",
        )

    def _emit_final_fallback(self) -> None:
        """Best-effort final-side emission when the loop ended without a
        boundary emission — error path, drained queue, max_firings, or a
        user stop. Reconstructs the last completed iteration's values from
        the paired iterIn's ``iterout_*`` slots (written by the most recent
        handoff) and emits them. No-op for DAG graphs (no author scope) and
        for scopes that already emitted.
        """
        root_id = self._outermost_scope_id
        if root_id == GRAPH_SCOPE_ID or root_id in self._final_emitted:
            return
        root = self.scope_state.get(root_id)
        iter_out = self.nodes.get(root.iter_out_id or "") if root else None
        iter_in = self.nodes.get(root_id)
        if iter_out is None or iter_in is None:
            return
        values: dict[str, Any] = {}
        for port in iter_out.config.get("ports") or []:
            name = port.get("name") if isinstance(port, dict) else None
            if name:
                values[name] = iter_in.port_slots.get(f"iterout_{name}")
        self._emit_final_side(iter_out, values, root_id, ready_queue=None)

    async def _after_loop_pass(self, session: Any) -> None:
        """Drain the after-loop stage once the dataflow loop has ended.

        Seeds are the nodes made ready by the root scope's final-side
        emission (``self._final_ready``). The drain fires a ready node,
        routes its outputs, and enqueues downstream consumers as they
        become ready — until quiescent. A node whose upstream after-loop
        node has not fired yet is re-queued, so a chain (e.g. ``evaluate ->
        adjudicator -> graphOut``) resolves in dependency order without a
        topological sort.

        Restricted to graph-scope nodes: the loop is over, so re-firing a
        loop-body node — or any ``iterIn`` / ``iterOut`` — is meaningless
        and barred.

        graphOut targets are routed-to and re-snapshotted into
        ``_last_inputs`` (the harvest channel ``_collect_metrics`` reads)
        but not themselves fired — inner-scope graphOuts are skipped
        entirely, their latch flush being scope-bound.

        Tolerant of execution errors: a node that raises (e.g. env already
        torn down on the exception path) is logged and skipped.
        """
        seeds = list(self._final_ready)
        if not seeds:
            return
        log.info("After-loop pass: %d seed node(s): %s", len(seeds), seeds)

        ready_queue: list[str] = list(seeds)
        fired: set[str] = set()
        # Bounded — a permanently-not-ready node (a missing required input)
        # is re-queued, but the cap guarantees termination.
        cap = max(len(self.nodes) * 4, 64)
        n_iter = 0
        while ready_queue and n_iter < cap:
            n_iter += 1
            node_id = ready_queue.pop(0)
            if node_id in fired:
                continue
            node = self.nodes.get(node_id)
            if node is None:
                continue
            # The loop is over — never fire loop-control / loop-body nodes,
            # and never re-enter an author (loop) scope.
            if node.type in ("iterIn", "iterOut"):
                continue
            if self._scope_of(node_id) != GRAPH_SCOPE_ID:
                continue
            # Dependency order without a topo sort: a node whose required
            # inputs are not yet latched is re-queued behind whatever will
            # produce them. The cap bounds a never-satisfiable re-queue.
            if not self._is_ready(node):
                ready_queue.append(node_id)
                continue
            try:
                # Fire via the standard path so logging / hooks / state
                # containers behave exactly as for an in-loop fire.
                result = await self._fire_node(node, session)
            except Exception as e:
                log.warning(
                    "post-loop fire of %s (%s) raised — skipping: %s",
                    node_id,
                    node.type,
                    e,
                )
                node.pending_inputs = {}
                fired.add(node_id)
                continue
            fired.add(node_id)
            node.pending_inputs = {}
            if not isinstance(result, dict):
                continue
            for edge in self.adjacency.get(node_id, []):
                target_id = edge.get("target", "")
                target = self.nodes.get(target_id)
                if target is None:
                    continue
                # Inner-scope graphOut: scope-bound latch flush — leave alone.
                if target.type == "graphOut" and self._scope_of(target_id) != GRAPH_SCOPE_ID:
                    continue
                src_handle = edge.get("sourceHandle", "default")
                tgt_handle = edge.get("targetHandle", src_handle)
                if src_handle in result:
                    self._route_value_to_port(target, tgt_handle, result[src_handle])
                elif src_handle == "default":
                    self._route_value_to_port(target, tgt_handle, result)
                if target.type == "graphOut":
                    # Graph-scope graphOut: re-snapshot the harvest channel.
                    # Not fired — the snapshot IS the harvest.
                    if target.pending_inputs:
                        target.state["_last_inputs"] = dict(target.pending_inputs)
                    continue
                if (
                    target_id not in fired
                    and target_id not in ready_queue
                    and self._is_ready(target)
                ):
                    ready_queue.append(target_id)

    def _propagate_graphout_latches(
        self,
        scope: _ScopeState,
        ready_queue: list,
    ) -> None:
        """When an inner scope terminates, propagate its graphOut nodes'
        latched values to outer-scope downstream nodes.

        graphOut nodes inside an inner scope buffer their incoming value in
        ``state["latched_value"]`` (set in the propagation block when the
        scope is non-graph) instead of immediately propagating outward.
        On scope termination we walk those graphOut nodes, route their
        latched value through outgoing edges, and queue any downstream
        target that becomes ready. This gives graphOut "function return
        value" semantics: outer sees the FINAL value, not every iteration's.

        For root-scope or graph-scope graphOuts (e.g. eval_graph metric
        harvest), values were already propagated normally in the standard
        propagation block — this method is a no-op for those.
        """
        if scope.scope_id == GRAPH_SCOPE_ID:
            return
        for pout_id in scope.graphout_node_ids:
            pnode = self.nodes.get(pout_id)
            if pnode is None:
                continue
            latched = pnode.state.get("latched_value")
            if latched is None:
                # No value was ever buffered (no inner-iter wrote to graphOut)
                continue
            for edge in self.adjacency.get(pout_id, []):
                tgt_id = edge.get("target", "")
                tgt = self.nodes.get(tgt_id)
                if tgt is None:
                    continue
                tgt_handle = edge.get("targetHandle", "value")
                self._route_value_to_port(tgt, tgt_handle, latched)
                if self._is_ready(tgt) and tgt_id not in ready_queue:
                    ready_queue.append(tgt_id)

    def _is_ready(self, node: NodeInstance) -> bool:
        """Check if a node is ready to fire.

        - If the node has required ports: all must have data.
        - If all ports are optional: fires when ANY data arrives.

        Required ports are resolved per-instance so configurable-port nodes
        (``_resolve_ports``) can mark instance ports ``required=True``.

        iterIn special-case: readiness is driven by ``port_slots`` (the
        unified slot dict) — not ``pending_inputs``. Required ports on
        iterIn check against ``port_slots``; the all-optional branch fires
        iterIn whenever any slot is populated.
        """
        if node.type == "iterIn":
            required = self._get_required_ports_for_node(node)
            if required:
                return all(port_name in node.port_slots for port_name in required)
            return len(node.port_slots) > 0
        # iterOut special-case: a bare ``stop`` signal does not constitute
        # an iteration. The boundary fires only once at least one loop-carry
        # value has arrived — otherwise an early stop producer (e.g. an
        # outer-body done flag emitted at iteration start) would fire the
        # boundary prematurely and advance the scope counter on no data.
        if node.type == "iterOut" and set(node.pending_inputs) <= {"stop"}:
            return False
        required = self._get_required_ports_for_node(node)
        if required:
            return all(port_name in node.pending_inputs for port_name in required)
        # All-optional: ready if any input has arrived
        return len(node.pending_inputs) > 0

    def broadcast_signal(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """Fan a framework or user signal out to every live state container.

        Signals are the declarative lifetime mechanism for state containers.
        Each state's ``reset_on`` list subscribes to zero or more signals;
        when the signal fires, the state clears to its ``initial_value``.

        Canonical signals emitted by the framework:

        * ``run_start`` / ``run_end`` (GraphExecutor, at run start / teardown)
        * ``step_start`` / ``step_end`` (GraphExecutor at IterIn/IterOut)
        * ``episode_reset`` (env panels via the env panel router)

        Any node or nodeset env panel can emit custom signals by calling
        this method on the running executor.  See
        ``docs/design-docs/state-containers.html`` for the full
        contract.
        """
        payload = payload or {}
        for c in self.containers.values():
            c.on_signal(name, payload)

    def record_subprocess_containers(self, nodeset_name: str, previews: dict) -> None:
        """Record read-only previews of a server-mode nodeset's owned containers.

        Called from the server proxy when a ``/call`` response piggybacks owned
        containers (``server_app.call_function``). Stored per nodeset and merged
        into the next ``nav_step`` broadcast (see ``_broadcast_step``). Display
        only — never a cross-process state-access path.
        """
        if previews:
            self._subprocess_container_previews[nodeset_name] = previews

    def restore_step(self, step_number: int) -> bool:
        """Restore all containers to a previous checkpoint and prepare re-entry."""
        if step_number not in self._checkpoints:
            return False
        snapshot = self._checkpoints[step_number]
        for cid, data in snapshot.items():
            if cid in self.containers:
                self.containers[cid].from_checkpoint(data)
        self.step_counter = step_number
        # Prune future checkpoints (they are now invalid)
        to_remove = [s for s in self._checkpoints if s > step_number]
        for s in to_remove:
            del self._checkpoints[s]
        return True

    def get_checkpoints(self) -> list:
        """List available checkpoint step numbers."""
        return sorted(self._checkpoints.keys())

    async def _broadcast_step(self, session: Any) -> None:
        """Broadcast a consolidated nav_step from ``graphOut`` sinks.

        Viewer nodes (imageViewer, textScroll, etc.) now emit their own
        per-node ``viewer_data`` WS events inside execute(); this method
        collects from all ``graphOut`` nodes — keyed by ``config.portName`` —
        for the NavigatePage / EvalPage global store. Recognised port names
        (rgb / depth / action / state / done / metrics / response) feed the
        consolidated payload; other names are ignored here (graph-level
        outputs that the UI doesn't visualise still flow into eval metrics
        via ``_collect_metrics``).
        """
        import numpy as np

        from ..standard.wire_types import depth_to_base64, image_to_base64

        # Gather (portName → last value) from every graphOut sink.
        merged: dict[str, Any] = {}
        for n in self.nodes.values():
            if n.type != "graphOut":
                continue
            last = n.state.get("_last_inputs")
            if not last:
                continue
            port_name = (n.config or {}).get("portName") or ""
            if not port_name:
                continue
            # GraphOut has a single input ``value``; ``pending_inputs`` keys
            # it as such. Fall back to scanning all keys if absent.
            if "value" in last:
                merged[port_name] = last["value"]
            else:
                for v in last.values():
                    if v is not None:
                        merged[port_name] = v
                        break

        # Don't bail when there's no graphOut viewer data if there are state
        # containers to surface: a loop whose only graphOut is the post-loop
        # verdict (e.g. explore_eqa) produces no per-step graphOut, but its
        # nodeset-owned containers must still stream live to the State panel.
        if not merged and not (self.containers or self._subprocess_container_previews):
            return

        broadcast_data: dict[str, Any] = {"step": self.step_counter}

        # RGB/depth: convert numpy arrays to base64
        rgb = merged.get("rgb")
        if isinstance(rgb, np.ndarray):
            broadcast_data["rgb_base64"] = image_to_base64(rgb)
        elif isinstance(rgb, str) and len(rgb) > 100:
            broadcast_data["rgb_base64"] = rgb

        depth = merged.get("depth")
        if isinstance(depth, np.ndarray):
            broadcast_data["depth_base64"] = depth_to_base64(depth)
        elif isinstance(depth, str) and len(depth) > 100:
            broadcast_data["depth_base64"] = depth

        # Action (integer for VLN-CE, string viewpoint ID for MP3D)
        action = merged.get("action")
        if action is not None:
            try:
                action_int = int(action)
                broadcast_data["action"] = action_int
                broadcast_data["action_name"] = ACTION_NAMES.get(action_int, "")
            except (ValueError, TypeError):
                broadcast_data["action"] = action
                broadcast_data["action_name"] = str(action)

        # State (pose dict with position/orientation) — accepted under
        # portName ``state`` or ``pose``.
        state = merged.get("state") or merged.get("pose")
        if isinstance(state, dict):
            broadcast_data["position"] = state.get("position", [])
            broadcast_data["orientation"] = state.get("orientation", [])

        # Done + metrics
        broadcast_data["done"] = merged.get("done", False)
        if merged.get("metrics"):
            broadcast_data["metrics"] = merged["metrics"]

        # LLM response
        if merged.get("response"):
            broadcast_data["response"] = merged["response"]

        # State container snapshots. Home (executor-process) containers are
        # tagged owner="home"; nodeset-owned (subprocess) containers arrive via
        # the /call response piggyback (record_subprocess_containers) tagged
        # with their nodeset name. Both shapes: {label, owner, states}.
        if self.containers or self._subprocess_container_previews:
            containers_payload: dict[str, Any] = {
                cid: {"label": c.label, "owner": "home", "states": c.get_preview()}
                for cid, c in self.containers.items()
            }
            for nodeset_name, previews in self._subprocess_container_previews.items():
                for cid, entry in previews.items():
                    containers_payload[cid] = {
                        "label": entry.get("label", cid),
                        "owner": nodeset_name,
                        "states": entry.get("states", {}),
                    }
            broadcast_data["containers"] = containers_payload

        if not getattr(getattr(session, "principles", None), "suppress_nav_events", False):
            await broadcast(session._ws("nav_step", broadcast_data))


class _NodeStateProxy:
    """Bridges the old ctx-based handler signature with per-node state.

    Handlers read/write attributes like ctx.raw_obs, ctx.rnn_states.
    This proxy redirects those to the node's persistent state dict.

    State containers are available via ``ctx.containers`` — a dict of
    ``{container_id: StateContainer}`` for containers connected to this
    node via state edges.
    """

    _INTERNAL_ATTRS = (
        "_state",
        "_containers",
        "_graph_state",
        "_executor",
        "step",
        "session",
    )

    def __init__(self, state: dict, step: int, session: Any):
        self._state = state
        self._containers: dict[str, Any] = {}
        self._graph_state: Any = None
        # ADR-028 PB-1.5: back-reference to the executing GraphExecutor
        # so in-graph proxy nodes can resolve per-runner server URL
        # overrides at call time. Assigned in ``_execute_node`` right after
        # ctx construction.
        self._executor: Any = None
        self.step = step
        self.session = session

    @property
    def containers(self) -> dict[str, Any]:
        """State containers this node holds an access grant to (ADR-dataflow-004).

        Keyed by container id. A node sees only the containers it was
        explicitly granted — there is no implicit/global access.
        """
        return self._containers

    @property
    def graph_state(self) -> Any:
        """The well-known ``graph_state`` container — bound only when this
        node holds an explicit grant to it (no auto-inject since
        ADR-dataflow-004); otherwise ``None``."""
        return self._graph_state

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._state.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL_ATTRS:
            super().__setattr__(name, value)
        else:
            self._state[name] = value
