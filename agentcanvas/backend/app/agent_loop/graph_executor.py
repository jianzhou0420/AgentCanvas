"""Dataflow execution engine — nodes fire when their inputs arrive.

Replaces the linear DAG executor. Supports cyclic graphs naturally:
data flowing back through edges triggers re-firing.

Key concepts:
- NodeInstance: per-node persistent state + pending inputs
- Entry nodes: fire at startup with no required inputs
- StopExecution: abort signal a node may raise to stop the run
- Step counter: increments when iterOut fires (once per iteration)

``GraphExecutor.run`` is a scheduling loop (pop → guard → fire → pivot or
deliver) over three internal seams, each written once and called at fixed
points:

- ``_fire`` — the one error policy. Every fire site (main loop, boundary
  settle drain, after-loop pass) goes through it, so a node failure is put
  on the bus, recorded for the end-of-run conviction and starves its
  consumers the same way wherever it happens.
- ``_deliver`` — the one delivery rule: route a fired node's outputs along
  its edges, enqueue the consumers that are ready. The final side, the
  after-loop pass and the graphOut latch flush are the same call.
- ``pivots`` (``agent_loop/pivots.py``) — iterIn slot / persist / handoff,
  iterOut decide / final, graphOut latch / snapshot. The loop calls them
  through ``_iter_boundary`` and ``_after_fire``; it holds no pivot state.

There is one executor instance per run (ADR-executor-003); the seams are
methods and helpers of it, not a second engine.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..components.bases import FireList
from ..config import strict_errors_enabled
from ..errors import get_bus
from ..graph_def import GraphDefinition
from ..standard.telemetry import _current_node_transport, _current_node_usage, new_usage_bucket
from ..standard.wire_types import is_list_type
from ..state import broadcast
from .builtin_nodes import CATALOG
from .hooks import HookRunner, extract_node_hooks, merge_hooks, safe_serialize
from .pivots import (
    LoopCarry,
    final_payload,
    graphout_last,
    graphout_latch,
    graphout_latched,
    graphout_snapshot,
    iterout_port_names,
    stop_requested,
)
from .scope_analysis import GRAPH_SCOPE_ID, ScopeForest, analyze_scopes

log = logging.getLogger("agentcanvas.graph-executor")

# ``_deliver``: an edge whose source handle is absent from the payload
# routes nothing (distinct from routing ``None``).
_NOT_ROUTED = object()

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
    # iterIn only — the loop-carried slots and their persist rule
    # (``pivots.LoopCarry``); built from ``config.ports`` on construction.
    carry: LoopCarry | None = None

    def __post_init__(self) -> None:
        if self.type == "iterIn" and self.carry is None:
            self.carry = LoopCarry(self.config)


class NodeErrorAggregate(RuntimeError):
    """Raised at the end of a run when one or more nodes failed mid-run.

    A failed node (exception, or an ``{"error": ...}`` result from e.g.
    a server-mode proxy) starves its downstream but does not stop the
    dataflow loop — without this, the run would finish indistinguishable
    from a clean completion (the ``status="completed"``/``step_count=0``
    eval pathology). ``_fire`` records every node error — in the main
    loop, the boundary settle drain and the after-loop pass alike — and
    the run is convicted here, AFTER the verdict stage, so final-side
    metrics are still collected on the error path.
    """


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
        on_event: Any = None,
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
        # Optional SDK-level run-event sink (Graph.run(on_event=...)). A plain
        # callable taking one event dict; sync or async. Surfaces the per-node
        # lifecycle (already dispatched to shell hooks) as native Python events
        # so an in-process run is observable without tailing logs. See _emit.
        self._on_event: Any = on_event

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

    async def _emit(self, kind: str, **fields: Any) -> None:
        """Deliver one run event to the ``on_event`` sink, if one is set.

        Best-effort and defensive: a sync *or* async callback is supported
        (an awaitable return is awaited), and any exception it raises is logged,
        never propagated — an observability hook must not be able to break the
        run. ``kind`` is one of ``graph_start`` / ``node_start`` /
        ``node_finish`` / ``node_error`` / ``graph_complete`` / ``graph_error``;
        ``fields`` carries that kind's payload (``node_id``, ``node_type``,
        ``inputs``, ``outputs``, ``error``, ``duration_ms``, …). ``step`` is
        stamped from the current step counter. Values are passed **live** (not
        copied/serialised) so a debugger sees the real objects; a callback must
        not mutate them.
        """
        cb = self._on_event
        if cb is None:
            return
        event = {"kind": kind, "step": self.step_counter, **fields}
        try:
            r = cb(event)
            if inspect.isawaitable(r):
                await r
        except Exception:
            log.warning("on_event callback raised on %r event", kind, exc_info=True)

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
        scheduling loop; the iterOut boundary — ``_iter_boundary`` — closes
        each iteration in four phases: record → settle → decide → hand off)
        → after-loop (the verdict stage) → finalise. The loop fires through
        ``_fire`` (the one error policy) and delivers through
        ``_after_fire`` / ``_deliver`` (the one delivery rule). Narrated
        walkthrough with diagrams:
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
        await self._emit("graph_start", graph_name=graph.name, node_count=len(self.nodes))

        # Safety: max total firings to prevent infinite loops
        max_firings = self.step_budget * len(self.nodes) * 3
        total_firings = 0

        # ─── in-loop · the scheduling loop — pop, guard, fire, pivot or deliver ───
        # The loop only schedules. The work happens in three seams called
        # at fixed points: ``_fire`` (the one error policy),
        # ``_iter_boundary`` (the iterOut pivot: record → settle → decide →
        # hand off) and ``_after_fire`` → ``_deliver`` (the one delivery
        # rule).
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

                # Fire the node. A failed fire yields an ``{"error": ...}``
                # result (or raises under strict errors); StopExecution is
                # the node-side halt and ends the run here.
                try:
                    result = await self._fire(node, session)
                except StopExecution as e:
                    log.info(
                        "StopExecution from %s: %s at step %d", node.id, e.reason, self.step_counter
                    )
                    self.terminated = True
                    break

                total_firings += 1

                # ── iterOut boundary — the pivot closes this iteration ──
                if node.type == "iterOut":
                    if await self._iter_boundary(node, result, ready_queue, session, step_delay_ms):
                        break
                    continue

                # ── ordinary path — consume · pivot cases · deliver ──
                self._after_fire(node, result, ready_queue)

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
            await self._emit(
                "graph_complete", terminated=self.terminated, metrics=metrics
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
            await self._emit("graph_error", error=str(e))

            # Flush log entries even on error
            if self._logger:
                self._logger.flush()

    # ── The seams — written once, called by the scheduling loop at fixed points ──

    async def _fire(self, node: NodeInstance, session: Any) -> dict:
        """Fire one node under the one error policy.

        Every fire site — the main loop, the boundary's settle drain, the
        after-loop pass — goes through here, so a node failure means the
        same thing wherever it happens: it is put on the error bus (the
        Report tab), recorded in ``node_errors`` for the end-of-run
        conviction (``NodeErrorAggregate``), and the fire yields an
        ``{"error": ...}`` result — which routes to no declared port, so
        downstream starves rather than consuming garbage. Under
        ``AGENTCANVAS_STRICT_ERRORS`` an exception is re-raised after the
        bus emit instead, so smoke / eval runs fail fast. A node that
        legitimately declares an ``error`` output port (env_libero tools,
        navgpt navigate) is exempt: its ``error`` key is data.
        ``StopExecution`` is not an error and passes through to the caller.
        """
        from_exception = False
        try:
            result = await self._fire_node(node, session)
        except StopExecution:
            raise
        except Exception as e:
            self._report_node_error(node, session, exc=e)
            if strict_errors_enabled():
                raise
            result = {"error": str(e)}
            from_exception = True
        if isinstance(result, dict) and "error" in result and not self._declares_error_output(node):
            if not from_exception:
                self._report_node_error(node, session, message=str(result["error"]))
            self.node_errors.append(
                {
                    "node_id": node.id,
                    "node_type": node.type,
                    "step": self.step_counter,
                    "error": str(result["error"]),
                }
            )
        return result

    def _report_node_error(
        self, node: NodeInstance, session: Any, *, exc: Exception | None = None, message: str = ""
    ) -> None:
        """Put one node failure on the error bus: the exception itself
        (``NODE_EXEC_FAIL``) or the message of a returned error result
        (``NODE_RESULT_ERROR``)."""
        scope = {
            "node_id": node.id,
            "node_type": node.type,
            "origin": self._flatten_map.trace(node.id) if self._flatten_map else node.id,
            "step": self.step_counter,
            "execution_id": getattr(session, "_execution_id", None),
        }
        if exc is not None:
            get_bus().from_exception(
                exc,
                source="node",
                code="NODE_EXEC_FAIL",
                scope=scope,
                title=f"Node {node.id} ({node.type}) failed",
            )
            return
        get_bus().emit(
            severity="error",
            source="node",
            code="NODE_RESULT_ERROR",
            title=f"Node {node.id} ({node.type}) returned an error result",
            message=message,
            scope=scope,
        )

    def _deliver(
        self, src_id: str, payload: dict, queue: list[str], *, terminal: bool = False
    ) -> None:
        """Route one fired node's outputs along its outgoing edges and
        enqueue the consumers that are ready — the one delivery rule.

        For every edge out of ``src_id`` the value is
        ``payload[sourceHandle]`` — the whole payload when the handle is
        ``"default"``, nothing when the handle is absent (so an
        error-shaped result routes nowhere). The write goes through
        ``_route_value_to_port`` (LIST[T] coercion, the iterIn slot
        write). A target that is ready afterwards is enqueued;
        ``_enqueue`` dedupes and stamps the ready-time.

        ``terminal`` marks the after-loop band (the final side, the
        after-loop pass): a ``graphOut`` reached there is routed-to and
        recorded — graph scope: the harvest snapshot; inner scope: the
        latch — but never fired. In-loop a graphOut fires like any node.
        """
        for edge in self.adjacency.get(src_id, []):
            target = self.nodes.get(edge.get("target", ""))
            if target is None:
                continue
            src_handle = edge.get("sourceHandle", "default")
            tgt_handle = edge.get("targetHandle", src_handle)
            if src_handle in payload:
                value = payload[src_handle]
            elif src_handle == "default":
                value = payload
            else:
                value = _NOT_ROUTED
            if value is not _NOT_ROUTED:
                self._route_value_to_port(target, tgt_handle, value)
            if terminal and target.type == "graphOut":
                if value is _NOT_ROUTED:
                    continue
                if self._scope_of(target.id) == GRAPH_SCOPE_ID:
                    if target.pending_inputs:
                        graphout_snapshot(target)
                else:
                    graphout_latch(target, value)
                continue
            if self._is_ready(target):
                self._enqueue(queue, target.id)

    def _after_fire(self, node: NodeInstance, result: Any, queue: list[str]) -> None:
        """A body node has fired in-loop: consume its inputs, apply the
        pivot cases, deliver. An iterIn clears its ``persist=False`` slots;
        an inner-scope graphOut latches its value instead of delivering
        (``_propagate_graphout_latches`` flushes the latch when the scope
        terminates); everything else is delivered along its edges."""
        node.pending_inputs = {}
        if node.carry is not None:
            node.carry.after_fire()
        if node.type == "graphOut" and self._scope_of(node.id) != GRAPH_SCOPE_ID:
            if isinstance(result, dict):
                graphout_latch(node, result.get("value"))
            return
        self._deliver(node.id, result if isinstance(result, dict) else {}, queue)

    async def _iter_boundary(
        self,
        node: NodeInstance,
        result: Any,
        ready_queue: list[str],
        session: Any,
        step_delay_ms: int,
    ) -> bool:
        """The iterOut boundary — closes one iteration of its scope in four
        phases: record → settle → decide → hand off. Returns ``True`` when
        the root scope terminated (stop input, budget exhaust, or a
        StopExecution during the settle drain) and the scheduling loop must
        halt. An inner scope's termination flushes its graphOut latches and
        lets the outer loop continue.

        Edges FROM the iterOut are final-side only (``final_*`` handles):
        they emit once at scope termination via ``_emit_final_side``, never
        per iteration, so the iterOut's own result is never delivered here.
        """
        # The scope is keyed by the paired iterIn's id (== scope_id).
        io_scope_id = node.config.get("pairedWith", "") or self._outermost_scope_id
        io_scope = self.scope_state.get(io_scope_id)
        is_root = io_scope_id == self._outermost_scope_id

        # ── phase 1/4 · record — counters, checkpoint, log, nav_step ──
        # Advance the per-scope counter. For backward compat,
        # self.step_counter mirrors the OUTERMOST scope's counter
        # (single-scope graphs unchanged).
        if io_scope is not None:
            io_scope.step_counter += 1
            scope_step = io_scope.step_counter
        else:
            scope_step = self.step_counter + 1
        if is_root:
            self.step_counter += 1
            session._current_step = self.step_counter

        log.info(
            "Iter: scope=%s step=%d (parent=%s, root=%s)",
            io_scope_id,
            scope_step,
            io_scope.parent_scope_id if io_scope is not None else None,
            is_root,
        )
        self.broadcast_signal("step_end", {"step": scope_step, "scope_id": io_scope_id})

        # Checkpoint all containers AFTER the iter boundary — the snapshot
        # is the state step N+1 begins from. Root-scope iterations only
        # (nested inner scopes don't checkpoint).
        if self.containers and is_root:
            self._checkpoints[self.step_counter] = {
                cid: c.checkpoint() for cid, c in self.containers.items()
            }
            if len(self._checkpoints) > self.max_checkpoints:
                del self._checkpoints[min(self._checkpoints)]

        # Flush execution log entries to JSONL at the iteration boundary.
        if self._logger:
            self._logger.flush()

        # One consolidated nav_step per root iteration, from every
        # graphOut's harvest snapshot.
        if is_root:
            await self._broadcast_step(session)

        # ── phase 2/4 · settle — drain this iter's leftover sinks ──
        if await self._settle_drain(io_scope_id, io_scope, is_root, ready_queue, session):
            return True

        # ── phase 3/4 · decide — stop input, then budget check ──
        # The stop signal is structurally bound to its scope (it sits on
        # the scope's own iterOut) and is read exactly once per iteration,
        # after the settle drain and before the handoff. Unwired stop =
        # budget-only loop. Termination (stop or budget) emits the terminal
        # iteration's values exactly once on the final_* handles.
        if stop_requested(result):
            log.info(
                "Stop: iterOut %s stop=True at scope=%s step=%d",
                node.id,
                io_scope_id or "(graph)",
                scope_step,
            )
            if io_scope is not None:
                io_scope.terminated = True
            self._emit_final_side(
                node, result, io_scope_id, ready_queue=None if is_root else ready_queue
            )
            if is_root:
                self.terminated = True
                return True

        scope_budget = io_scope.step_budget if io_scope is not None else self.step_budget
        if scope_step >= scope_budget:
            log.info(
                "Step budget (%d) exhausted for scope=%s", scope_budget, io_scope_id or "(graph)"
            )
            if io_scope is not None:
                io_scope.terminated = True
            self._emit_final_side(
                node,
                result if isinstance(result, dict) else {},
                io_scope_id,
                ready_queue=None if is_root else ready_queue,
            )
            if is_root:
                self.terminated = True
                return True
            # Inner-scope budget exhaust: flush the inner graphOut latches
            # and let the outer loop continue; no handoff — the scope is done.
            self._propagate_graphout_latches(io_scope, ready_queue)
            return False

        if step_delay_ms > 0 and is_root:
            await asyncio.sleep(step_delay_ms / 1000)

        # An inner scope stopped above without halting the root loop: flush
        # its graphOut latches and skip the handoff.
        if io_scope is not None and io_scope.terminated:
            self._propagate_graphout_latches(io_scope, ready_queue)
            return False

        # ── phase 4/4 · hand off — carry → paired iterIn, next iteration ──
        # The fired bundle lands in the paired iterIn's ``iterout_<x>``
        # slots (``stop`` is not loop-carried; final_* handles never fire
        # here) and the iterIn is queued for the next iteration.
        paired_id = node.config.get("pairedWith", "")
        paired = self.nodes.get(paired_id) if paired_id else None
        if paired is not None and paired.carry is not None:
            paired.carry.handoff(result if isinstance(result, dict) else {})
            self._enqueue(ready_queue, paired.id)
        node.pending_inputs = {}
        return False

    async def _settle_drain(
        self,
        io_scope_id: str,
        io_scope: _ScopeState | None,
        is_root: bool,
        ready_queue: list[str],
        session: Any,
    ) -> bool:
        """Boundary phase 2/4: fire the just-closed iteration's leftover
        sinks (viewer / telemetry nodes queued behind the iterOut) BEFORE
        the stop check, so they still emit on the terminal step.

        Restricted to nodes of the iterOut's own scope — outer / peer-scope
        nodes are not drained by an inner boundary — with one exception: a
        ROOT boundary also drains graph-scope nodes. Dead-end sinks are
        never on a path to any iterOut, so scope analysis leaves them in
        the graph scope; without this they would miss the terminal step
        entirely (mid-loop it merely fires them earlier than the main loop
        would). iterIn is never drained — it advances to the next iteration
        and is fired only via the handoff after the stop check.

        Bounded by ``max_settle`` fires. Returns ``True`` when a drained
        node raised ``StopExecution`` — the run halts exactly as it does
        for a StopExecution in the main loop.
        """
        max_settle = 64  # safety cap against pathological queues
        settled = 0
        while ready_queue and settled < max_settle:
            next_idx = None
            for i, nid in enumerate(ready_queue):
                cand = self.nodes.get(nid)
                if cand is None or cand.type == "iterIn":
                    continue
                cand_scope = self._scope_of(nid)
                if (
                    io_scope is not None
                    and cand_scope != io_scope_id
                    and not (is_root and cand_scope == GRAPH_SCOPE_ID)
                ):
                    continue
                next_idx = i
                break
            if next_idx is None:
                return False  # nothing to settle in this scope
            nid = ready_queue.pop(next_idx)
            cand = self.nodes[nid]
            try:
                r = await self._fire(cand, session)
            except StopExecution as e:
                log.info(
                    "StopExecution from %s during settle: %s at step %d",
                    nid,
                    e.reason,
                    self.step_counter,
                )
                self.terminated = True
                return True
            self._after_fire(cand, r, ready_queue)
            settled += 1
        return False

    def _enqueue(self, ready_queue: list[str], node_id: str) -> None:
        """Append ``node_id`` to the ready queue if absent, stamping its
        ready-time so ``_fire_node`` can derive ``queue_wait_ms`` (time spent
        waiting in the queue before firing). Stamps only on a real enqueue
        transition; a node already queued keeps its original ready-time.
        Every enqueue — in-loop, final side, after-loop, latch flush — goes
        through here."""
        if node_id in ready_queue:
            return
        ready_queue.append(node_id)
        self._t_ready[node_id] = time.perf_counter()

    def _declares_error_output(self, node: NodeInstance) -> bool:
        """True when the node's handler legitimately declares an ``error``
        output port (e.g. env_libero tools, navgpt navigate) — for those,
        an ``error`` key in the result is data, not a failure report."""
        ports = CATALOG.ports_of(node.type, node.config)
        return ports is not None and ports.wire_type("error", "out") is not None

    async def _fire_node(self, node: NodeInstance, session: Any) -> dict:
        """Execute a node's handler with its pending inputs and persistent state."""
        # System Log: pop the ready-time stamped at enqueue; queue_wait is the
        # gap from "became ready" to "fires now" (None for terminal-band fires).
        _t_ready_map = getattr(self, "_t_ready", None)
        t_ready = _t_ready_map.pop(node.id, None) if _t_ready_map is not None else None
        queue_wait_ms = (time.perf_counter() - t_ready) * 1000 if t_ready is not None else None
        node_cls = CATALOG.get(node.type)
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

        # iterIn fires with its loop-carried slots (``pivots.LoopCarry``):
        # the writers (run-start init edges, the iterOut handoff) have all
        # landed there; ``_after_fire`` applies the persist rule afterwards.
        fire_inputs = node.carry.fire_inputs() if node.carry is not None else node.pending_inputs

        await self._emit(
            "node_start",
            node_id=node.id,
            node_type=node.type,
            node_label=node.label or node.id,
            inputs=fire_inputs,
            parent_node_id=getattr(node, "_dynamic_parent_id", None),
        )

        exec_error: Exception | None = None
        # Per-node LLM usage bucket — populated by every llm_complete /
        # vlm_complete call reached during this execute(). See
        # ``app.standard.telemetry``. The ContextVar is set
        # for ALL node firings (cheap), so any future node that calls
        # the LLM helpers automatically gets accounting without code
        # changes.
        usage_bucket: dict = new_usage_bucket()
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

        # Build port wire type mapping for log enrichment — instance ports,
        # so LIST[T] shows up in the log panel even when the default schema
        # is a scalar T.
        _port_wt: dict[str, str] = {}
        _ports = CATALOG.ports_of(node.type, node.config)
        if _ports is not None:
            for p in _ports.inputs + _ports.outputs:
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

        # Native run event — terminal per-fire signal (start already emitted).
        # Fires with the post-hook ``result`` on success, or the error before
        # it is re-raised below, so an ``on_event`` sink sees every fire.
        _ev = {
            "node_id": node.id,
            "node_type": node.type,
            "node_label": node.label or node.id,
            "inputs": node.pending_inputs,
            "duration_ms": duration_ms_total,
            "parent_node_id": getattr(node, "_dynamic_parent_id", None),
        }
        if exec_error is not None:
            await self._emit("node_error", error=str(exec_error), **_ev)
        else:
            await self._emit("node_finish", outputs=result, **_ev)

        # Re-raise if execute() failed (after hooks had a chance to observe)
        if exec_error is not None:
            raise exec_error

        # Runner-level metrics — exposed via the status API
        # (``LoopRunner.get_status``). Set when any node emits
        # ``{"done": True, "metrics": ...}`` (typically the env evaluate node).
        if isinstance(result, dict) and result.get("done") and result.get("metrics"):
            session._metrics = result["metrics"]

        # graphOut: record the harvest snapshot — the canonical run-output
        # channel, read by ``_broadcast_step`` (the nav_step WS event) and
        # ``graph_run`` (outputs / metrics harvest). When a graph runs at
        # top level, every graphOut fires and is snapshotted; when used as
        # a composite, ``flatten_graph`` rewires graphOut nodes through to
        # the parent so this branch becomes inert.
        if node.type == "graphOut":
            graphout_snapshot(node)

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
        """The required input ports of a specific node instance — the
        catalog's one rule (instance ports via ``_resolve_ports(config)``,
        else the class-level lists), so dynamic nodes like ``llmCall`` or
        ``iterIn`` can mark a config-declared port ``required: true`` and
        the executor respects it in both entry discovery and readiness
        checks."""
        return CATALOG.required_inputs(node.type, node.config)

    def _get_input_port_wire_type(self, node: NodeInstance, port_name: str) -> str | None:
        """Wire type of a consumer port on a live NodeInstance — instance
        ports, so ``config.ports`` overrides are picked up (ADR-024)."""
        return CATALOG.wire_type(node.type, port_name, "in", node.config)

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
        ``tgt_handle`` is the port name directly (no prefix). The write is
        a ``LoopCarry`` slot write — overwrite semantics, LIST[T] fan-in
        suppressed so persist=True ports don't accumulate unbounded.
        """
        if target.carry is not None:
            target.carry.write(tgt_handle, value)
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

        Delivery is terminal (``_deliver(terminal=True)``): a graph-scope
        ``graphOut`` target is snapshotted (the harvest channel), an
        inner-scope one is latched for ``_propagate_graphout_latches``;
        neither is fired.

        Exactly-once per scope: guarded by ``self._final_emitted``.
        """
        if scope_id in self._final_emitted:
            return
        self._final_emitted.add(scope_id)

        payload = final_payload(values)
        sink = self._final_ready if ready_queue is None else ready_queue
        self._deliver(iter_out.id, payload, sink, terminal=True)
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
        if iter_in.carry is None:
            return
        values = iter_in.carry.last_carried(iterout_port_names(iter_out.config))
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

        Delivery is terminal: a graph-scope graphOut target is routed-to
        and re-snapshotted (the harvest channel ``graph_run`` reads) but
        not fired.

        Fires through ``_fire`` — the one error policy: a node that fails
        here (e.g. the env already torn down on the exception path) is put
        on the bus and recorded like any other failure, and its
        error-shaped result starves its consumers.
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
                result = await self._fire(node, session)
            except StopExecution as e:
                # The loop is already over — nothing to stop.
                log.info("StopExecution from %s after the loop — ignored: %s", node_id, e.reason)
                result = {}
            fired.add(node_id)
            node.pending_inputs = {}
            self._deliver(
                node_id, result if isinstance(result, dict) else {}, ready_queue, terminal=True
            )

    def _propagate_graphout_latches(
        self,
        scope: _ScopeState,
        ready_queue: list,
    ) -> None:
        """When an inner scope terminates, propagate its graphOut nodes'
        latched values to outer-scope downstream nodes.

        graphOut nodes inside an inner scope latch their incoming value
        (``pivots.graphout_latch``, from ``_after_fire`` when the scope is
        non-graph) instead of immediately delivering outward.
        On scope termination we walk those graphOut nodes, route their
        latched value through outgoing edges, and queue any downstream
        target that becomes ready. This gives graphOut "function return
        value" semantics: outer sees the FINAL value, not every iteration's.

        For root-scope or graph-scope graphOuts (e.g. eval_graph metric
        harvest), values were already delivered in-loop — this method is a
        no-op for those. graphOut's one output handle is ``value``.
        """
        if scope.scope_id == GRAPH_SCOPE_ID:
            return
        for pout_id in scope.graphout_node_ids:
            pnode = self.nodes.get(pout_id)
            if pnode is None:
                continue
            latched = graphout_latched(pnode)
            if latched is None:
                # No value was ever buffered (no inner-iter wrote to graphOut)
                continue
            self._deliver(pout_id, {"value": latched}, ready_queue)

    def _is_ready(self, node: NodeInstance) -> bool:
        """Check if a node is ready to fire.

        - If the node has required ports: all must have data.
        - If all ports are optional: fires when ANY data arrives.

        Required ports are resolved per-instance so configurable-port nodes
        (``_resolve_ports``) can mark instance ports ``required=True``.

        iterIn special-case: readiness is driven by its loop-carried slots
        (``LoopCarry.ready``) — not ``pending_inputs``.
        """
        if node.carry is not None:
            return node.carry.ready(self._get_required_ports_for_node(node))
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

        from ..standard.wire_types import (
            DISCRETE_ACTION,
            depth_to_base64,
            image_to_base64,
            serialize_for_display,
        )

        # Gather (portName → last value) from every graphOut sink.
        merged: dict[str, Any] = {}
        for n in self.nodes.values():
            if n.type != "graphOut":
                continue
            last = graphout_last(n)
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

        # Action — rendered by the DISCRETE_ACTION wire type's display rule
        # (an int gets the standard action name, anything else is shown as
        # is); the engine itself carries no action vocabulary.
        action = merged.get("action")
        if action is not None:
            broadcast_data.update(serialize_for_display(DISCRETE_ACTION, action))

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
