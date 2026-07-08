"""Code-first construction surface for AgentCanvas graphs (PoC).

A thin, ergonomic wrapper over the existing pure-dataclass graph model
(:mod:`app.graph_def`) and the in-process engine
(:class:`app.agent_loop.graph_executor.GraphExecutor`).  The point: let a
user *build a graph in Python and run it in-process*, LangGraph-style, without
authoring graph JSON or standing up the FastAPI backend / canvas GUI::

    from app.code_first import Graph

    g = Graph(name="demo", eval_graph=False)
    src = g.add("const_source", value=7)
    inc = g.add("increment")
    out = g.graph_out("result")
    src.out("value") >> inc.in_("x")     # or g.connect(src.out("value"), inc.in_("x"))
    inc.out("y")     >> out.in_("value")

    result = g.run()
    print(result["result"])              # 8

Nothing here is new machinery — every method assembles the same
``GraphDefinition`` / ``NodeDef`` / ``EdgeDef`` dataclasses the canvas emits,
so a code-built graph round-trips to JSON (:meth:`Graph.to_dict` /
:meth:`Graph.save`) and opens unchanged in the canvas GUI.

Scope (PoC): pure-Python nodes registered via
``app.agent_loop.builtin_nodes.register_node`` and the builtin control /
boundary nodes.  Env / GPU nodesets (which need the workspace registry +
server-mode subprocesses) are out of scope for this first slice.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph_def import AccessGrantDef, ContainerDef, EdgeDef, GraphDefinition, NodeDef

# ── Runtime session ──────────────────────────────────────────────────────


class DefaultSession:
    """Minimal ``session`` satisfying ``GraphExecutor.run()`` for headless runs.

    The executor only ever touches this handful of attributes/methods (writes
    ``_status`` / ``_current_step`` / ``_metrics``, reads ``_metrics`` /
    ``principles``, and calls ``_ws(...)``).  With no WebSocket clients
    connected the broadcast layer is a silent no-op, so a headless run has no
    UI side effects — the results are read back off the executor and this
    session (see :class:`RunResult`).
    """

    def __init__(self, execution_id: str = "code-first") -> None:
        self._status: str = "idle"
        self._current_step: int = 0
        self._metrics: dict[str, Any] | None = None
        self._execution_id: str = execution_id
        self.principles: Any = None

    def _ws(self, msg_type: str, data: Any = None) -> Any:
        from .models import WSMessage

        return WSMessage(type=msg_type, data=data, execution_id=self._execution_id)


# ── Port references + node handles ───────────────────────────────────────


@dataclass
class PortRef:
    """A reference to one port on one node — the endpoint of a wire.

    ``a.out("value") >> b.in_("x")`` is sugar for
    ``graph.connect(a.out("value"), b.in_("x"))``.
    """

    node: NodeHandle
    port: str
    direction: str  # "out" | "in"

    def __rshift__(self, other: PortRef) -> EdgeDef:
        return self.node._graph.connect(self, other)


class NodeHandle:
    """A live handle to a node added to a :class:`Graph`.

    Wraps the underlying :class:`app.graph_def.NodeDef` and knows its owning
    graph so port refs can wire themselves.
    """

    def __init__(self, graph: Graph, node_def: NodeDef) -> None:
        self._graph = graph
        self._def = node_def

    @property
    def id(self) -> str:
        return self._def.id

    @property
    def type(self) -> str:
        return self._def.type

    def out(self, port: str) -> PortRef:
        return PortRef(self, port, "out")

    def in_(self, port: str) -> PortRef:
        return PortRef(self, port, "in")

    def set(self, **config: Any) -> NodeHandle:
        """Merge extra config into this node; returns self for chaining."""
        self._def.config.update(config)
        return self

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"NodeHandle(id={self._def.id!r}, type={self._def.type!r})"


# ── Run result ───────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """What an in-process run produced.

    ``outputs`` maps each ``graphOut`` node's ``portName`` to its harvested
    value (the graphOut's last-fire snapshot, ``state["_last_inputs"]``; a
    single incoming handle is unwrapped to its bare value, multiple handles
    stay a dict).  ``metrics`` is whatever the run wrote to
    ``session._metrics``.  ``executor`` is the raw
    :class:`GraphExecutor` for power users (node states, ``step_counter``,
    ``terminated``).
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    executor: Any = None
    session: Any = None

    def __getitem__(self, key: str) -> Any:
        return self.outputs[key]

    def node_state(self, node_id: str) -> dict[str, Any]:
        """Persistent state dict of a node after the run."""
        inst = self.executor.nodes.get(node_id)
        return dict(inst.state) if inst is not None else {}

    @classmethod
    def _from_run(cls, executor: Any, session: Any) -> RunResult:
        outputs: dict[str, Any] = {}
        for inst in executor.nodes.values():
            if inst.type != "graphOut":
                continue
            port_name = (inst.config or {}).get("portName") or inst.id
            last = inst.state.get("_last_inputs") or {}
            if len(last) == 1:
                outputs[port_name] = next(iter(last.values()))
            else:
                outputs[port_name] = dict(last)
        return cls(
            outputs=outputs,
            metrics=dict(getattr(session, "_metrics", None) or {}),
            executor=executor,
            session=session,
        )


# ── The builder ──────────────────────────────────────────────────────────


class Graph:
    """Code-first builder over :class:`app.graph_def.GraphDefinition`.

    Add nodes with :meth:`add` (or the :meth:`graph_in` / :meth:`graph_out`
    boundary conveniences), wire them with :meth:`connect` (or ``>>``), then
    :meth:`run` in-process.  The underlying dataclass graph is available via
    :attr:`definition` and serialises with :meth:`to_dict` / :meth:`save`.
    """

    def __init__(
        self,
        name: str = "",
        *,
        description: str = "",
        eval_graph: bool = False,
        step_budget: int | None = 500,
    ) -> None:
        self._def = GraphDefinition(
            name=name,
            description=description,
            eval_graph=eval_graph,
            step_budget=step_budget,
        )
        self._type_counts: dict[str, int] = {}
        self._edge_n: int = 0

    # -- construction -------------------------------------------------------

    def add(self, node_type: str, id: str | None = None, **config: Any) -> NodeHandle:
        """Add a node of ``node_type`` (a registered ``node_type`` key).

        ``id`` auto-generates as ``"<type>_<n>"`` when omitted.  Extra kwargs
        become the node's ``config``.
        """
        if id is None:
            n = self._type_counts.get(node_type, 0)
            self._type_counts[node_type] = n + 1
            id = f"{node_type}_{n}"
        if any(nd.id == id for nd in self._def.nodes):
            raise ValueError(f"duplicate node id: {id!r}")
        node_def = NodeDef(id=id, type=node_type, label=id, config=dict(config))
        self._def.nodes.append(node_def)
        return NodeHandle(self, node_def)

    def graph_in(self, port_name: str, id: str | None = None, wire_type: str = "ANY") -> NodeHandle:
        """Add a ``graphIn`` boundary node exposing ``port_name``."""
        return self.add("graphIn", id=id or f"in_{port_name}", portName=port_name, wireType=wire_type)

    def graph_out(self, port_name: str, id: str | None = None, wire_type: str = "ANY") -> NodeHandle:
        """Add a ``graphOut`` sink whose ``port_name`` keys the run output."""
        return self.add(
            "graphOut", id=id or f"out_{port_name}", portName=port_name, wireType=wire_type
        )

    def connect(self, src: PortRef, dst: PortRef, id: str | None = None) -> EdgeDef:
        """Wire an output port to an input port."""
        if src.direction != "out":
            raise ValueError(f"connect() source must be an .out() port, got {src.direction!r}")
        if dst.direction != "in":
            raise ValueError(f"connect() target must be an .in_() port, got {dst.direction!r}")
        edge = EdgeDef(
            id=id or f"e{self._edge_n}",
            source=src.node.id,
            target=dst.node.id,
            sourceHandle=src.port,
            targetHandle=dst.port,
        )
        self._edge_n += 1
        self._def.edges.append(edge)
        return edge

    # -- shared state (containers + access grants) -------------------------

    def container(
        self,
        id: str,
        *,
        label: str = "",
        states: dict[str, Any] | None = None,
        position: dict[str, float] | None = None,
    ) -> str:
        """Add a state container (canvas blackboard). ``states`` follows the
        graph-JSON shape ``{name: {type, value_type, config, lifetime}}``.
        Returns the container id for use in :meth:`grant`.
        """
        self._def.containers.append(
            ContainerDef.from_dict(
                {
                    "id": id,
                    "label": label,
                    "position": position or {"x": 0.0, "y": 0.0},
                    "states": states or {},
                }
            )
        )
        return id

    def grant(self, node: NodeHandle | str, container_id: str, id: str | None = None) -> AccessGrantDef:
        """Grant a node read/write access to a container (not a wire)."""
        node_id = node.id if isinstance(node, NodeHandle) else node
        ag = AccessGrantDef(id=id or f"ag_{node_id}", node_id=node_id, container_id=container_id)
        self._def.access_grants.append(ag)
        return ag

    # -- validation / execution --------------------------------------------

    @property
    def definition(self) -> GraphDefinition:
        """The underlying (mutable) ``GraphDefinition``."""
        return self._def

    def validate(self) -> None:
        """Synthesise iterIn ports then raise on any connectivity/wire error.

        Mirrors what the JSON load path (``GraphDefinition.from_dict``) does
        for authored graphs; direct construction here bypasses it, so we run
        it explicitly.
        """
        from .graph_def import _synthesize_iterin_ports, validate_graph_connectivity

        _synthesize_iterin_ports(self._def)
        validate_graph_connectivity(self._def)

    def run(
        self,
        *,
        session: Any = None,
        validate: bool = False,
        step_delay_ms: int = 0,
        step_budget_override: int | None = None,
    ) -> RunResult:
        """Run the graph in-process and return a :class:`RunResult`.

        No FastAPI, no scheduler, no GPU admission — this drives
        ``GraphExecutor.run()`` directly, exactly as the executor unit tests
        do.  Set ``validate=True`` to fail fast on connectivity/wire errors.
        """
        from .agent_loop.graph_executor import GraphExecutor
        from .graph_def import _synthesize_iterin_ports

        _synthesize_iterin_ports(self._def)
        if validate:
            from .graph_def import validate_graph_connectivity

            validate_graph_connectivity(self._def)

        executor = GraphExecutor()
        sess = session or DefaultSession()
        asyncio.run(
            executor.run(
                self._def,
                sess,
                step_delay_ms=step_delay_ms,
                step_budget_override=step_budget_override,
            )
        )
        return RunResult._from_run(executor, sess)

    # -- serialisation (round-trip to canvas JSON) -------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canvas graph-JSON dict (iterIn ports synthesised)."""
        from .graph_def import _synthesize_iterin_ports

        _synthesize_iterin_ports(self._def)
        return self._def.to_dict()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the graph JSON to ``path`` (loadable by the canvas / backend)."""
        p = Path(path)
        p.write_text(self.to_json())
        return p

    @classmethod
    def from_definition(cls, definition: GraphDefinition) -> Graph:
        """Wrap an existing ``GraphDefinition`` (e.g. loaded from JSON)."""
        g = cls(name=definition.name)
        g._def = definition
        return g

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Graph(name={self._def.name!r}, "
            f"nodes={len(self._def.nodes)}, edges={len(self._def.edges)})"
        )
