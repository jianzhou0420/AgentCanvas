"""Graph definition models -- the universal data structure for composable node graphs.

A graph is pure JSON-serializable data: node references + wires + config.
No Python objects, no live state.  The executor creates live instances at
runtime by looking up handlers in NODE_HANDLERS[node.type].

Mirrors the frontend ``types.ts``: GraphDefinition / NodeDef / EdgeDef.

Used by
-------
- ``GraphExecutor``  -- receives a ``GraphDefinition``, builds live nodes
- ``flatten_graph()``   -- recursively expands composite nodes
- ``graphs_api.py``     -- CRUD for saved graphs in ``workspace/graphs/``
- ``BaseCanvasNode.children`` -- subgraph inside composite nodes
- ``RunRequest``        -- validated at the API boundary

last updated: 2026-03-31 01:00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# HookDef — a shell-command hook attached to a graph execution event
# ---------------------------------------------------------------------------


@dataclass
class HookDef:
    """A hook attached to a graph execution event.

    Hooks run as shell subprocesses.  The event payload is sent as JSON on
    stdin; the hook writes a JSON action response to stdout.

    Supported events: ``PreNodeExecute``, ``PostNodeExecute``,
    ``GraphStart``, ``GraphComplete``, ``GraphError``.

    ``match_node_type`` accepts ``"*"`` (all), an exact node type string, or
    a prefix glob like ``"env_habitat__*"`` (matched via ``startswith``).
    """

    event: str
    command: str
    match_node_type: str = "*"
    match_node_id: str | None = None  # exact node instance match (node-level hooks)
    timeout_ms: int = 1000
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event": self.event,
            "command": self.command,
            "match_node_type": self.match_node_type,
            "timeout_ms": self.timeout_ms,
            "enabled": self.enabled,
        }
        if self.match_node_id is not None:
            d["match_node_id"] = self.match_node_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HookDef:
        return cls(
            event=d["event"],
            command=d["command"],
            match_node_type=d.get("match_node_type", "*"),
            match_node_id=d.get("match_node_id"),
            timeout_ms=d.get("timeout_ms", 1000),
            enabled=d.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# StateDef — a single named state entry inside a container
# ---------------------------------------------------------------------------


@dataclass
class StateDef:
    """A single named state entry inside a :class:`ContainerDef`.

    Defines the reducer behaviour (``type``) and what data the state holds
    (``value_type``).  The ``config`` dict carries type-specific options
    (e.g. ``max_size`` for accumulators).

    ``lifetime`` declares when the state clears — orthogonal to the reducer:

    * ``"forever"`` (default) — never auto-clears
    * ``"step"`` — clears at every IterOut ``step_end`` signal
    * ``"episode"`` — clears on the framework ``episode_reset`` signal
      (fired by env panels on episode change)
    * ``"run"`` — clears on ``run_end`` (after the loop finishes)
    * ``"custom"`` — explicit signal list via ``reset_on``
    """

    type: str  # reducer: "accumulator" | "lastWrite" | "counter"
    value_type: str = "ANY"  # data shape from STATE_VALUE_TYPES
    config: dict[str, Any] = field(default_factory=dict)
    lifetime: str = "forever"  # "step" | "episode" | "run" | "forever" | "custom"
    reset_on: list[str] = field(default_factory=list)  # only honored when lifetime == "custom"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "value_type": self.value_type}
        if self.config:
            d["config"] = self.config
        if self.lifetime != "forever":
            d["lifetime"] = self.lifetime
        if self.reset_on:
            d["reset_on"] = list(self.reset_on)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StateDef:
        return cls(
            type=d.get("type", "lastWrite"),
            value_type=d.get("value_type", "ANY"),
            config=d.get("config", {}),
            lifetime=d.get("lifetime", "forever"),
            reset_on=list(d.get("reset_on", [])),
        )


# ---------------------------------------------------------------------------
# ContainerDef — a dict of named states, visible on the canvas
# ---------------------------------------------------------------------------


@dataclass
class ContainerDef:
    """A state container — dict of named states rendered on the canvas.

    Each container groups related states (e.g. "Navigation State" might hold
    ``action_history``, ``active_plan``, ``step_count``).  Nodes connect to
    containers via :class:`AccessGrantDef` to gain read/write access.
    """

    id: str
    label: str = ""
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    states: dict[str, StateDef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "position": self.position,
            "states": {name: sd.to_dict() for name, sd in self.states.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContainerDef:
        raw_states = d.get("states", {})
        states = {
            name: StateDef.from_dict(sd) if isinstance(sd, dict) else sd
            for name, sd in raw_states.items()
        }
        return cls(
            id=d["id"],
            label=d.get("label", ""),
            position=d.get("position", {"x": 0.0, "y": 0.0}),
            states=states,
        )


# ---------------------------------------------------------------------------
# AccessGrantDef — node → container access grant (not an edge)
# ---------------------------------------------------------------------------


@dataclass
class AccessGrantDef:
    """An access grant giving a node read/write permission on a container.

    Access grants are **not** wires.  They carry no data and do not trigger
    firing.  They define which nodes may call ``container.read()`` /
    ``container.write()`` at execution time.  Rendered as dashed violet
    lines on the canvas, but separate from the data wire system.
    """

    id: str
    node_id: str
    container_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "container_id": self.container_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccessGrantDef:
        return cls(
            id=d["id"],
            node_id=d["node_id"],
            container_id=d["container_id"],
        )


# ---------------------------------------------------------------------------
# EdgeDef
# ---------------------------------------------------------------------------


@dataclass
class EdgeDef:
    """A typed wire between two node ports."""

    id: str
    source: str
    target: str
    sourceHandle: str = ""
    targetHandle: str = ""

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "sourceHandle": self.sourceHandle,
            "targetHandle": self.targetHandle,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeDef:
        return cls(
            id=d["id"],
            source=d["source"],
            target=d["target"],
            sourceHandle=d.get("sourceHandle", ""),
            targetHandle=d.get("targetHandle", ""),
        )


# ---------------------------------------------------------------------------
# NodeDef
# ---------------------------------------------------------------------------


@dataclass
class NodeDef:
    """A node reference in a graph -- pure data, no live instance.

    The *type* field references a registered ``node_type`` (e.g. ``"envStep"``).
    The executor resolves it to a handler at runtime via
    ``NODE_HANDLERS[type]``.

    If *subgraph* is set this is a **composite node** containing a nested
    ``GraphDefinition``.
    """

    id: str
    type: str
    label: str = ""
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    config: dict[str, Any] = field(default_factory=dict)
    subgraph: GraphDefinition | None = None

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "position": self.position,
            "config": self.config,
        }
        if self.subgraph is not None:
            d["subgraph"] = self.subgraph.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeDef:
        sub_raw = d.get("subgraph")
        return cls(
            id=d["id"],
            type=d.get("type", "unknown"),
            label=d.get("label", ""),
            position=d.get("position", {"x": 0.0, "y": 0.0}),
            config=d.get("config", {}),
            subgraph=GraphDefinition.from_dict(sub_raw) if sub_raw else None,
        )


# ---------------------------------------------------------------------------
# GraphDefinition
# ---------------------------------------------------------------------------


@dataclass
class GraphDefinition:
    """A composable node graph -- nodes + edges, fully JSON-serializable.

    This is the universal container at every nesting depth:

    * Root canvas graph
    * Agent loop inner graph
    * Composite node subgraph
    * Saved template in ``workspace/graphs/``

    **Nodes** are references (type + config), not live instances.
    **Edges** are typed wires (source port -> target port).
    """

    name: str = ""
    description: str = ""
    nodes: list[NodeDef] = field(default_factory=list)
    edges: list[EdgeDef] = field(default_factory=list)

    # State containers (visible shared state on canvas).  Nodes gain
    # read/write access via ``access_grants`` — not wires.  A container
    # with the well-known id ``"graph_state"`` plays the role of the
    # optional graph-level blackboard (no more auto-inject: every node
    # that wants it must have its own AccessGrantDef).
    containers: list[ContainerDef] = field(default_factory=list)
    access_grants: list[AccessGrantDef] = field(default_factory=list)

    # Execution hooks (shell commands fired at lifecycle events)
    hooks: list[HookDef] = field(default_factory=list)

    # Loop-specific (present on agent loops, absent on simple composites).
    # ``step_budget`` is the per-episode iteration cap. The framework's
    # resolver chain (see eval_batch.py) lets the env override this per
    # episode (e.g. HM-EQA's scene-adaptive ``int(sqrt(scene_size) * 3)``);
    # an explicit ``step_budget`` on the eval API request takes precedence
    # over both. ``None`` means "let the framework default
    # (``DEFAULT_STEP_BUDGET``) decide" — used when the graph wants to
    # delegate fully to the env hook.
    step_budget: int | None = 500

    # Eval-graph flag: when True (default), the graph is required to declare
    # at least one ``graphOut`` node — each ``graphOut``'s last-fire snapshot
    # (``state["_last_inputs"]``) is the source of truth for
    # ``BatchEvalRunner._collect_metrics``. ``graphOut.config.portName``
    # becomes the metric key; the ``"metrics"`` portName is special-cased
    # to flatten a metrics-dict payload. Demo / playground graphs that
    # don't produce metrics opt out with ``eval_graph: false``.
    eval_graph: bool = True

    # Identity / provenance
    kind: str = "graph"  # "graph" (openable template) | "node" (draggable composite archive)
    group: str = ""  # user-defined group for organizing graph nodes (e.g. "history", "planning")
    presetId: str | None = None

    # -- helpers -------------------------------------------------------------

    @property
    def node_ids(self) -> list[str]:
        """All node IDs in this graph (non-recursive)."""
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> NodeDef | None:
        """Lookup a node by ID.  Returns ``None`` if not found."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "step_budget": self.step_budget,
            "eval_graph": self.eval_graph,
        }
        # State containers — only serialize if present (backward compat)
        if self.containers:
            d["containers"] = [c.to_dict() for c in self.containers]
        if self.access_grants:
            d["access_grants"] = [ag.to_dict() for ag in self.access_grants]
        if self.kind != "graph":
            d["kind"] = self.kind
        if self.group:
            d["group"] = self.group
        if self.presetId is not None:
            d["presetId"] = self.presetId
        if self.hooks:
            d["hooks"] = [h.to_dict() for h in self.hooks]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphDefinition:
        """Deserialize from a plain dict (JSON-parsed graph file or API payload).

        Graphs without ``containers`` / ``access_grants`` default to empty
        lists.

        Also synthesises each iterIn's ``config.ports`` from its authored
        initPorts + paired iterOut writer (the iterIn port surface is
        auto-derived, not user-authored).
        """
        nodes = [NodeDef.from_dict(n) for n in d.get("nodes", [])]
        containers = [ContainerDef.from_dict(c) for c in d.get("containers", [])]
        access_grants = [AccessGrantDef.from_dict(g) for g in d.get("access_grants", [])]

        # Backward-compat: accept the legacy ``maxIterations`` key from
        # pre-refactor graph files. New name is ``step_budget``; old key
        # wins if both are present so unmigrated files keep their authored
        # cap. Emit a one-shot DeprecationWarning so authors know to update.
        if "step_budget" in d:
            step_budget = d["step_budget"]
        elif "maxIterations" in d:
            import warnings as _warnings

            _warnings.warn(
                "Graph field 'maxIterations' is deprecated; rename to 'step_budget'.",
                DeprecationWarning,
                stacklevel=2,
            )
            step_budget = d["maxIterations"]
        else:
            step_budget = 500

        graph = cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            nodes=nodes,
            edges=[EdgeDef.from_dict(e) for e in d.get("edges", [])],
            containers=containers,
            access_grants=access_grants,
            hooks=[HookDef.from_dict(h) for h in d.get("hooks", [])],
            step_budget=step_budget,
            eval_graph=bool(d.get("eval_graph", True)),
            kind=d.get("kind", "graph"),
            group=d.get("group", ""),
            presetId=d.get("presetId"),
        )
        _synthesize_iterin_ports(graph)
        return graph

    # -- convenience constructors -------------------------------------------

    @classmethod
    def empty(cls, name: str = "", description: str = "") -> GraphDefinition:
        """Create an empty graph with no nodes or edges."""
        return cls(name=name, description=description)


# ---------------------------------------------------------------------------
# iterIn port synthesis
# ---------------------------------------------------------------------------
#
# iterIn has no user-authored ``ports``.  Its surface is the union of its
# own ``initPorts`` (run-start left side) and the paired ``iterOut.ports``
# (per-iter writer), plus any ``targetHandle`` names from direct canvas edges
# into the iterIn.  ``persist`` lives on each writer's port entry; merged by
# OR (if either writer marks a name persist=true → persist=true).


def _synthesize_iterin_ports(graph: GraphDefinition) -> None:
    """Populate each iterIn node's ``config.ports`` from its writers.

    Runs recursively on composite subgraphs.  Safe to call more than once;
    always overwrites ``config.ports`` with the freshly-computed list.
    """
    nodes_by_id = {n.id: n for n in graph.nodes}
    edges_by_target: dict[str, list] = {}
    for e in graph.edges:
        edges_by_target.setdefault(e.target, []).append(e)

    for node in graph.nodes:
        # Recurse into subgraphs before handling this level.
        if node.subgraph is not None:
            _synthesize_iterin_ports(node.subgraph)

        if node.type != "iterIn":
            continue

        # Always-prefix synthesis: initPorts.<X> → iterIn handle "init_<X>",
        # iterOut.<X> → iterIn handle "iterout_<X>". No cross-writer merging
        # — each writer owns its own iterIn namespace.
        synthesised: list[dict] = []
        seen: set[str] = set()

        def _emit_port(
            port: dict,
            origin: str,
            prefix: str,
            synthesised: list[dict],
            seen: set[str],
        ) -> None:
            name = port.get("name")
            if not name:
                return
            handle = f"{prefix}_{name}"
            if handle in seen:
                return  # defensive: same writer shouldn't declare duplicates
            seen.add(handle)
            # Default persist depends on origin: init ports are one-shot
            # (Step 0 only) by default; iterOut-transferred ports carry
            # across iterations by default.
            default_persist = origin == "iterOut"
            synthesised.append(
                {
                    "name": handle,
                    "wire_type": port.get("wire_type") or "ANY",
                    "persist": bool(port.get("persist", default_persist)),
                    "origin": origin,
                    "writer_name": name,  # original name on the writer node
                }
            )

        # iterIn's own authored init ports (two-sided model — the left/input
        # side). Authored under ``config["initPorts"]`` to avoid the rejected
        # legacy ``init_ports``/``loop_ports`` keys. Prefix "init_".
        for p in node.config.get("initPorts", []):
            if isinstance(p, dict):
                _emit_port(p, "init", "init", synthesised, seen)

        # Paired iterOut writer (prefix "iterout_").
        paired_out_id = node.config.get("pairedWith")
        paired_out = nodes_by_id.get(paired_out_id) if paired_out_id else None
        if paired_out is not None and paired_out.type == "iterOut":
            for p in paired_out.config.get("ports", []):
                if isinstance(p, dict):
                    _emit_port(p, "iterOut", "iterout", synthesised, seen)

        # Direct canvas edges targeting iterIn — these use their own targetHandle
        # (already prefixed or not) as the iterIn port name. Treat as "init"
        # origin (they're run-start seeds).
        for e in edges_by_target.get(node.id, []):
            if not e.targetHandle:
                continue
            if e.targetHandle in seen:
                continue
            seen.add(e.targetHandle)
            synthesised.append(
                {
                    "name": e.targetHandle,
                    "wire_type": "ANY",
                    "persist": False,
                    "origin": "init",
                    "writer_name": e.targetHandle,
                }
            )

        if node.config is None:
            node.config = {}
        node.config["ports"] = synthesised


# ---------------------------------------------------------------------------
# Edge wire-type compatibility (ADR-027)
# ---------------------------------------------------------------------------


def validate_edge_wire_type(source_type: str | None, target_type: str | None) -> tuple[bool, str]:
    """Check whether an edge from a port typed ``source_type`` may connect
    to a port typed ``target_type``.

    Rules (ADR-027):

    * ``ANY`` on either side is always allowed (escape hatch).
    * Equal types always allowed.
    * ``T`` → ``LIST[T]`` allowed (executor auto-wraps scalar to ``[scalar]``).
    * ``LIST[T]`` → ``T`` **rejected** (would be lossy).
    * Different inner types rejected.

    Returns ``(ok, reason)``.  ``reason`` is empty when ``ok`` is True.
    Unknown/missing types default to allowed — schema-less edges fall through
    to runtime behaviour unchanged.
    """
    # Avoid circular imports — wire_types lives under ``standard``.
    from .standard.wire_types import canonical_wire_type, is_list_type, unwrap_list

    if not source_type or not target_type:
        return True, ""
    if source_type == "ANY" or target_type == "ANY":
        return True, ""

    # Resolve deprecated names (e.g. ``ACTION`` → ``DISCRETE_ACTION``) so a
    # legacy port and a migrated port still connect during the migration sweep.
    source_type = canonical_wire_type(source_type)
    target_type = canonical_wire_type(target_type)

    if source_type == target_type:
        return True, ""

    src_is_list = is_list_type(source_type)
    tgt_is_list = is_list_type(target_type)

    if not src_is_list and tgt_is_list:
        # T → LIST[U] allowed only when T == U.
        if source_type == unwrap_list(target_type):
            return True, ""
        return (
            False,
            f"type mismatch: {source_type} → {target_type}",
        )

    if src_is_list and not tgt_is_list:
        return (
            False,
            f"lossy connection rejected: {source_type} → {target_type}",
        )

    # Both list or both scalar with different inner types.
    return False, f"type mismatch: {source_type} → {target_type}"


# ---------------------------------------------------------------------------
# Edge wire-type validation (ADR-027 — enforcement, 2026-06-13)
# ---------------------------------------------------------------------------
#
# ``validate_edge_wire_type`` (above) defined the compatibility rules but had
# no caller — types were declared, never enforced. This wires the rules into
# the load-time path so a shape mismatch is rejected before an (expensive)
# eval run, not discovered mid-run.
#
# A port type resolves to ``None`` ("unresolved") when its node type is not in
# ``NODE_HANDLERS`` or the handle is unknown; ``validate_edge_wire_type`` treats
# ``None`` as allowed, so coverage degrades safely. ``NODE_HANDLERS`` holds the
# builtins unconditionally and each nodeset's proxy classes once that nodeset is
# loaded (``register_node`` → ``NODE_HANDLERS[node_type] = cls``). So the SAME
# resolver covers nodeset env↔method edges too — but only after the nodeset is
# loaded. Run the check after ``ensure_nodesets_for_graph`` for whole-graph
# coverage; before load it sees builtins only.


def _resolve_port_type(node: NodeDef, handle: str, direction: str) -> str | None:
    """Best-effort wire-type of a port on any registered node.

    ``direction`` is ``"in"`` (consumer/target) or ``"out"`` (producer/source).
    Resolves builtin / iterIn / iterOut nodes always, and nodeset nodes once
    their nodeset is loaded (proxy class registered in ``NODE_HANDLERS``).
    Returns ``None`` for not-yet-loaded nodeset nodes and unknown handles —
    ``None`` means "unresolved → allowed".
    """
    from .agent_loop.builtin_nodes import NODE_HANDLERS

    cfg_ports = (node.config or {}).get("ports") or []
    if node.type in ("iterIn", "iterOut"):
        # iterOut's final-side source handles (``final_<name>``) mirror the
        # loop-carry port's type; ``step``/``final_stop`` are control signals.
        base = handle
        if node.type == "iterOut" and direction == "out" and handle.startswith("final_"):
            base = handle[len("final_") :]
        for p in cfg_ports:
            if isinstance(p, dict) and p.get("name") == base:
                return p.get("wire_type") or "ANY"
        if handle in ("step", "final_stop", "stop"):
            return "ANY"
        return None

    cls = NODE_HANDLERS.get(node.type)
    if cls is None:
        return None
    resolver = getattr(cls, "_resolve_ports", None)
    if callable(resolver):
        try:
            ins, outs = resolver(node.config or {})
        except Exception:
            ins = list(getattr(cls, "input_ports", []))
            outs = list(getattr(cls, "output_ports", []))
    else:
        ins = list(getattr(cls, "input_ports", []))
        outs = list(getattr(cls, "output_ports", []))
    for p in ins if direction == "in" else outs:
        if getattr(p, "name", None) == handle:
            return getattr(p, "wire_type", None)
    return None


def wire_type_report(graph: GraphDefinition, type_resolver=None) -> dict:
    """Whole-graph wire-type report (non-raising) — the shared core.

    Used by the standalone CLI (``app.tools.validate_graph``), the dev
    ``POST /api/graphs/validate`` endpoint, and the run/eval warn pass.

    ``type_resolver(node, handle, direction) -> str | None`` resolves a port's
    wire type; defaults to ``_resolve_port_type`` (builtins + iterIn/iterOut
    always; nodeset nodes once their nodeset is loaded into ``NODE_HANDLERS``).
    An edge with an unresolved end is counted as ``skipped`` (not an error) —
    coverage degrades safely.

    Returns ``{errors, checked, skipped, total_edges, unresolved_node_types}``.
    ``unresolved_node_types`` names the node types whose ports could not be
    resolved (typically nodesets not loaded in this process) so callers can
    tell the user what coverage they are missing and why.
    """
    resolve = type_resolver or _resolve_port_type
    by_id = {n.id: n for n in graph.nodes}
    errors: list[str] = []
    checked = 0
    skipped = 0
    unresolved: set[str] = set()
    for e in graph.edges:
        s = by_id.get(e.source)
        t = by_id.get(e.target)
        if s is None or t is None:
            continue
        src_type = resolve(s, e.sourceHandle or "", "out")
        tgt_type = resolve(t, e.targetHandle or "", "in")
        if src_type is None:
            unresolved.add(s.type)
        if tgt_type is None:
            unresolved.add(t.type)
        if src_type is None or tgt_type is None:
            skipped += 1
            continue
        checked += 1
        ok, reason = validate_edge_wire_type(src_type, tgt_type)
        if not ok:
            errors.append(
                f"Edge '{e.id}' {reason}: {s.type}.{e.sourceHandle} → {t.type}.{e.targetHandle}"
            )
    return {
        "errors": errors,
        "checked": checked,
        "skipped": skipped,
        "total_edges": len(graph.edges),
        "unresolved_node_types": sorted(unresolved),
    }


def validate_edge_wire_types(graph: GraphDefinition, type_resolver=None) -> list[str]:
    """Return wire-type mismatch errors for the graph's data edges.

    Thin wrapper over :func:`wire_type_report` kept for callers that only want
    the error list (the run/eval warn pass, ``validate_graph_connectivity``).
    """
    return wire_type_report(graph, type_resolver)["errors"]


# ---------------------------------------------------------------------------
# Graph connectivity validation (Phase 3)
# ---------------------------------------------------------------------------


def validate_graph_connectivity(graph: GraphDefinition, type_resolver=None) -> None:
    """Raise ``ValueError`` if any required input port lacks an incoming edge.

    Required-port detection mirrors the executor's runtime helper
    (``GraphExecutor._get_required_ports_for_node``): for each node, ask
    its handler class for the per-instance port schema via ``_resolve_ports``
    when available, otherwise fall back to the class-level ``input_ports``.
    A port is required iff ``getattr(p, "optional", True)`` is False.

    Catches a class of author errors at load time — required-but-unwired ports
    that would otherwise silently never satisfy and leave the node parked
    forever in the graph executor.
    """
    incoming: dict[str, set[str]] = {}
    for e in graph.edges:
        incoming.setdefault(e.target, set()).add(e.targetHandle or "")

    # Local import to avoid a circular dependency at module import time.
    from .agent_loop.builtin_nodes import NODE_HANDLERS

    errors: list[str] = []
    for node in graph.nodes:
        cls = NODE_HANDLERS.get(node.type)
        if cls is None:
            # Nodeset / unknown type — runtime will surface unresolved handlers.
            continue
        resolver = getattr(cls, "_resolve_ports", None)
        instance_inputs: list = []
        if callable(resolver):
            try:
                instance_inputs, _ = resolver(node.config or {})
            except Exception:
                instance_inputs = []
        if instance_inputs:
            required = {p.name for p in instance_inputs if not getattr(p, "optional", True)}
        else:
            required = {
                p.name for p in getattr(cls, "input_ports", []) if not getattr(p, "optional", True)
            }
        wired = incoming.get(node.id, set())
        missing = sorted(required - wired)
        if missing:
            errors.append(
                f"Node '{node.id}' (type={node.type}) has required input "
                f"port(s) {missing} with no incoming edge."
            )

    # iterIn's ``config.ports`` is synthesised at load from its own initPorts
    # + paired iterOut + direct canvas edges (see ``_synthesize_iterin_ports``).
    # So writer-existence is trivially satisfied.  Remaining checks: pairedWith
    # integrity, legacy-schema rejection, and downstream-handle validity.
    nodes_by_id = {n.id: n for n in graph.nodes}
    outgoing_by_source: dict[str, list] = {}
    for e in graph.edges:
        outgoing_by_source.setdefault(e.source, []).append(e)

    # Removed node types are rejected with migration hints instead of
    # failing later with an unknown-node-type error:
    #   * ``initialize`` (removed 2026-06-10, ADR-dataflow-008) — its
    #     run-start role lives on iterIn's init side.
    #   * ``termination`` (removed 2026-06-11, two-sided iterOut) — its
    #     halt role lives on iterOut's ``stop`` input port.
    for node in graph.nodes:
        if node.type == "initialize":
            errors.append(
                f"Node '{node.id}' has type 'initialize', which has been "
                f"removed. Author its ports on the paired iterIn's "
                f"config.initPorts and wire the seeds into the iterIn's "
                f"init_<name> handles."
            )
        elif node.type == "termination":
            errors.append(
                f"Node '{node.id}' has type 'termination', which has been "
                f"removed. Wire the done/stop signal (BOOL) into the loop's "
                f"iterOut 'stop' input port instead; the engine checks it "
                f"once per iteration at the iterOut boundary."
            )

    for node in graph.nodes:
        if node.type != "iterIn":
            continue

        if "init_ports" in node.config or "loop_ports" in node.config:
            errors.append(
                f"iterIn '{node.id}' uses the legacy init_ports/loop_ports "
                f"schema. Migrate to version 3: drop iterIn's own port list; "
                f"ports are synthesised from initPorts + paired iterOut."
            )
            continue

        paired_out_id = node.config.get("pairedWith", "")
        if paired_out_id:
            paired_out = nodes_by_id.get(paired_out_id)
            if paired_out is None or paired_out.type != "iterOut":
                errors.append(
                    f"iterIn '{node.id}' pairedWith='{paired_out_id}' does not "
                    f"point to an iterOut node."
                )

        port_names = {
            p["name"]
            for p in (node.config.get("ports") or [])
            if isinstance(p, dict) and "name" in p
        }
        for e in outgoing_by_source.get(node.id, []):
            sh = e.sourceHandle or ""
            if sh and sh != "step" and sh not in port_names:
                errors.append(
                    f"Edge '{e.id}' from iterIn '{node.id}' references unknown "
                    f"handle '{sh}' — valid handles are {sorted(port_names)} (plus 'step')."
                )

    # iterOut must declare its own ``config.ports``. It has no class-level
    # defaults (ADR-031 cleanup) and the frontend port-list editor is the
    # single source of truth. An empty / missing list would mean the pivot
    # transfers nothing — almost always an author error.
    incoming_by_target: dict[str, list] = {}
    for e in graph.edges:
        incoming_by_target.setdefault(e.target, []).append(e)
    for node in graph.nodes:
        if node.type != "iterOut":
            continue
        ports_cfg = node.config.get("ports")
        if not (isinstance(ports_cfg, list) and len(ports_cfg) > 0):
            errors.append(
                f"{node.type} '{node.id}' must declare a non-empty config.ports (ADR-031)."
            )
            continue

        paired_id = node.config.get("pairedWith", "")
        if paired_id:
            paired = nodes_by_id.get(paired_id)
            if paired is None or paired.type != "iterIn":
                errors.append(
                    f"{node.type} '{node.id}' pairedWith='{paired_id}' does "
                    f"not point to an iterIn node."
                )

        io_port_names = {p["name"] for p in ports_cfg if isinstance(p, dict) and "name" in p}
        # Incoming edges: loop-carry ports + the ``stop`` halt input.
        for e in incoming_by_target.get(node.id, []):
            th = e.targetHandle or ""
            if th and th != "stop" and th not in io_port_names:
                errors.append(
                    f"Edge '{e.id}' into iterOut '{node.id}' references unknown "
                    f"handle '{th}' — valid handles are "
                    f"{sorted(io_port_names)} (plus 'stop')."
                )
        # Outgoing edges: final-side handles only. These emit exactly once,
        # at scope termination — per-iteration taps belong on body nodes.
        allowed_final = {f"final_{n}" for n in io_port_names} | {"final_stop"}
        for e in outgoing_by_source.get(node.id, []):
            sh = e.sourceHandle or ""
            if sh and sh not in allowed_final:
                errors.append(
                    f"Edge '{e.id}' from iterOut '{node.id}' references "
                    f"handle '{sh}' — edges from iterOut are final-side only; "
                    f"valid handles are {sorted(allowed_final)}."
                )

    # Eval graphs must declare at least one graphOut node — every graphOut's
    # last-fire snapshot is harvested by ``BatchEvalRunner._collect_metrics``
    # and keyed by its ``config.portName``. Demo / playground graphs opt out
    # via ``eval_graph: false`` at the graph level.
    if graph.eval_graph:
        graphout_count = sum(1 for n in graph.nodes if n.type == "graphOut")
        if graphout_count < 1:
            errors.append(
                "eval graph must declare at least one graphOut node, "
                "found 0. Set 'eval_graph: false' on the graph if it does "
                "not produce metrics."
            )

    # Multi-scope topology check (additive — single-scope graphs return
    # zero errors). Surfaces cross-author-scope wires that bypass
    # graphIn/graphOut, duplicate pairedWith, unpaired iter_in, etc.
    # Also writes derivative ``nested_scope_ids`` onto outer iter_in
    # configs (UI hint for the canvas hierarchical config panel).
    forest = None
    try:
        from .agent_loop.scope_analysis import analyze_scopes

        forest, scope_errors = analyze_scopes(graph)
        errors.extend(scope_errors)
    except Exception as e:
        # Defensive: a bug in scope_analysis must not break the legacy
        # validation path. Log the analyzer error but don't propagate.
        import logging

        logging.getLogger("agentcanvas.graph-def").warning(
            "scope_analysis raised during validation: %r",
            e,
        )

    # After-loop band rules (two-sided iterOut). The band = the downstream
    # closure of the ROOT scope's final_* edges — the verdict stage fed
    # exactly once at termination. Two structural guarantees:
    #   1. Purity — band nodes take inputs only from the root iterOut's
    #      final side or from other band nodes. Anything the verdict needs
    #      must ride the pivot (be a loop-carry port exposed as final_*);
    #      an in-loop edge into the band would make the band node fireable
    #      mid-loop, re-opening the last-write-wins metric bug class.
    #   2. Verdict source — in an eval loop graph, graph-scope graphOut
    #      nodes may only be fed from the band (or the final side
    #      directly): a graphOut fed from the loop body would report
    #      whichever iteration happened to write last, not the terminal one.
    if forest is not None and forest.root_scope_ids:
        from .agent_loop.scope_analysis import GRAPH_SCOPE_ID as _GS

        root_iterout_ids = {
            forest.scopes[rid].iter_out_id
            for rid in forest.root_scope_ids
            if rid in forest.scopes and forest.scopes[rid].iter_out_id
        }
        # Band closure: BFS from final-edge targets of root iterOuts.
        band: set[str] = set()
        frontier = [e.target for rio in root_iterout_ids for e in outgoing_by_source.get(rio, [])]
        while frontier:
            nid = frontier.pop()
            if nid in band:
                continue
            band.add(nid)
            frontier.extend(e.target for e in outgoing_by_source.get(nid, []))

        for nid in sorted(band):
            n = nodes_by_id.get(nid)
            if n is None:
                continue
            n_scope = forest.node_to_scope.get(nid, _GS)
            if n_scope != _GS:
                errors.append(
                    f"After-loop node '{nid}' (downstream of a root iterOut's "
                    f"final side) lies inside loop scope '{n_scope}' — the "
                    f"after-loop band must be graph-scope."
                )
            for e in incoming_by_target.get(nid, []):
                if e.source in band or e.source in root_iterout_ids:
                    continue
                errors.append(
                    f"Edge '{e.id}' feeds after-loop node '{nid}' from "
                    f"'{e.source}', which is not on the final side. Verdict "
                    f"inputs must ride the pivot: expose the value as an "
                    f"iterOut port and wire from its final_<name> handle."
                )

        # Rule scope: only the harvest-critical verdict ports. Streaming
        # graphOuts (rgb / pose / action frames for the nav UI and replay)
        # legitimately ride per-iteration wires; composite archives
        # (kind="node") expose loop-internal return latches by design.
        if graph.eval_graph and graph.kind != "node":
            root_set = set(forest.root_scope_ids)
            _VERDICT_PORTS = {"metrics", "success"}
            for n in graph.nodes:
                if n.type != "graphOut":
                    continue
                if (n.config or {}).get("portName") not in _VERDICT_PORTS:
                    continue
                n_scope = forest.node_to_scope.get(n.id, _GS)
                if n_scope != _GS and n_scope not in root_set:
                    continue  # nested-scope graphOut = composite latch, exempt
                for e in incoming_by_target.get(n.id, []):
                    if e.source in band or e.source in root_iterout_ids:
                        continue
                    errors.append(
                        f"Edge '{e.id}' feeds eval-output graphOut '{n.id}' "
                        f"from '{e.source}' inside the loop — eval outputs "
                        f"must come from the after-loop band (final side) so "
                        f"they reflect the terminal iteration."
                    )

    # Edge wire-type compatibility (ADR-027 enforcement). Appended last so
    # shape mismatches surface alongside connectivity errors in one raise.
    errors.extend(validate_edge_wire_types(graph, type_resolver))

    if errors:
        raise ValueError("Graph connectivity validation failed:\n  - " + "\n  - ".join(errors))
