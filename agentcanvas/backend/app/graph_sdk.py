"""Graph SDK construction surface for AgentCanvas graphs (PoC).

A thin, ergonomic wrapper over the existing pure-dataclass graph model
(:mod:`app.graph_def`) and the in-process engine
(:class:`app.agent_loop.graph_executor.GraphExecutor`).  The point: let a
user *build a graph in Python and run it in-process*, LangGraph-style, without
authoring graph JSON or standing up the FastAPI backend / canvas GUI::

    from app.graph_sdk import Graph

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

Pure-Python nodes (registered via
``app.agent_loop.builtin_nodes.register_node``) and the builtin control /
boundary nodes run with a bare :meth:`Graph.run`.  Env / GPU **nodeset**
graphs (``mapgpt__*``, ``env_mp3d__*``, …) run too — pass
``load_nodesets=True`` (or leave it ``"auto"``): :meth:`Graph.run` scans the
workspace registry and auto-loads every nodeset the graph needs
(``registry.ensure_nodesets_for_graph``), spawning server-mode subprocesses
for env nodesets exactly as the backend does, then tears down what it started.
The env nodeset's own conda env (e.g. ``ac-mp3d``) and data must be present;
a real multi-episode env run is still an *experiment* and belongs behind
``/experiment:run``.

Ergonomic sugar for the fiddly bits: :meth:`Graph.loop` (iterIn/iterOut
episode loop + carry wiring), :meth:`Graph.hook` (lifecycle shell hooks),
:meth:`Graph.composite` (nested subgraph nodes).  The inverse direction —
compiling an existing ``GraphDefinition`` back into a standalone builder
script — is :meth:`Graph.to_code` (see :mod:`app.graph_sdk_codegen`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

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

    def __init__(self, execution_id: str = "Graph SDK") -> None:
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


class NodeProxy(NodeHandle):
    """Base for the GENERATED typed node handles under ``agentcanvas/nodes/``
    (emitted by :func:`generate_node_stubs`). A generated subclass declares its
    ``node_type`` and — for the IDE — ``Literal``-typed ``out`` / ``in_``
    overrides, so a human gets port autocomplete and an agent gets a real
    symbol to reference. At runtime it *is* a :class:`NodeHandle`; pass the
    subclass to :meth:`Graph.add` to get a typed handle::

        from agentcanvas.nodes import env_habitat
        reset = g.add(env_habitat.Reset)     # typed handle
        reset.out("episode_id")              # autocompletes; typo → red squiggle
    """

    node_type: str = ""


_NP = TypeVar("_NP", bound=NodeProxy)


# ── Loop (iterIn / iterOut episode loop) ─────────────────────────────────


@dataclass
class _LoopSpec:
    name: str
    wire_type: str
    carried: bool  # True → written back each step by iterOut; False → seed-only


def _norm_loop_spec(entry: Any, *, carried: bool) -> _LoopSpec:
    """Accept ``"name"`` or ``("name", "WIRE_TYPE")`` loop-port declarations."""
    if isinstance(entry, (tuple, list)):
        name = entry[0]
        wire_type = entry[1] if len(entry) > 1 else "ANY"
    else:
        name, wire_type = entry, "ANY"
    return _LoopSpec(name=name, wire_type=wire_type, carried=carried)


class Loop:
    """Handle to an iterIn/iterOut episode loop (see :meth:`Graph.loop`).

    Collapses the verbose loop wiring into four intent-named calls:

    * :meth:`seed` — a run-start value into the loop (``iterIn.init_<name>``).
    * :meth:`feed` — hand a carried value to a consumer; wires **both** the
      run-start (``init_<name>``) and carried (``iterout_<name>``) sides so the
      consumer sees the seed on step 0 and the carried value thereafter.
    * :meth:`carry` — a per-step value written back for the next step
      (``iterOut.<name>``).
    * :meth:`stop` — a termination signal into ``iterOut.stop``.

    Plus :meth:`final` for the after-loop side (``iterOut.final_<name>``).
    """

    def __init__(self, graph: Graph, iter_in: NodeHandle, iter_out: NodeHandle, specs: list[_LoopSpec]) -> None:
        self._graph = graph
        self.iter_in = iter_in
        self.iter_out = iter_out
        self._specs = {s.name: s for s in specs}

    def _spec(self, name: str) -> _LoopSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown loop port {name!r}; declared: {sorted(self._specs)}") from None

    def seed(self, name: str, src: PortRef) -> EdgeDef:
        self._spec(name)
        return self._graph.connect(src, self.iter_in.in_(f"init_{name}"))

    def feed(self, name: str, dst: PortRef) -> list[EdgeDef]:
        s = self._spec(name)
        edges = [self._graph.connect(self.iter_in.out(f"init_{name}"), dst)]
        if s.carried:
            edges.append(self._graph.connect(self.iter_in.out(f"iterout_{name}"), dst))
        return edges

    def carry(self, name: str, src: PortRef) -> EdgeDef:
        s = self._spec(name)
        if not s.carried:
            raise ValueError(
                f"loop port {name!r} is init-only (declared in init=[...]); it cannot carry back"
            )
        return self._graph.connect(src, self.iter_out.in_(name))

    def stop(self, src: PortRef) -> EdgeDef:
        return self._graph.connect(src, self.iter_out.in_("stop"))

    def final(self, name: str = "stop") -> PortRef:
        return self.iter_out.out(f"final_{name}")


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


# ── Run events (observability stream) ─────────────────────────────────────


@dataclass
class RunEvent:
    """One lifecycle event from an in-process :meth:`Graph.run` — delivered to
    its ``on_event`` callback as it happens. The per-node lifecycle the
    executor already dispatches to shell hooks, surfaced as a native Python
    object so a run is observable without tailing logs.

    ``kind`` is one of:

    * ``"graph_start"`` — run began (``graph_name`` / ``node_count``).
    * ``"node_start"`` — a node is about to fire (``node_id`` / ``node_type`` /
      ``inputs``).
    * ``"node_finish"`` — a node fired OK (adds ``outputs`` / ``duration_ms``).
    * ``"node_error"`` — a node's ``forward`` raised (adds ``error`` /
      ``duration_ms``); the run then re-raises.
    * ``"graph_complete"`` — run finished cleanly (``terminated`` / ``metrics``).
    * ``"graph_error"`` — run crashed (``error``).

    ``inputs`` / ``outputs`` are the **live** engine dicts (not copies) — read
    them, don't mutate. ``parent_node_id`` is set for dynamic-firelist children.
    The callback may be sync or async and runs inside the executor's event loop;
    keep it cheap. To just collect the stream, pass ``on_event=events.append``.
    """

    kind: str
    step: int = 0
    node_id: str | None = None
    node_type: str | None = None
    node_label: str | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float | None = None
    parent_node_id: str | None = None
    graph_name: str | None = None
    node_count: int | None = None
    terminated: bool | None = None
    metrics: dict[str, Any] | None = None

    @classmethod
    def _from_dict(cls, d: dict) -> RunEvent:
        """Build from the executor's raw event dict, ignoring unknown keys."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def __str__(self) -> str:  # notebook-friendly one-liner
        bits = [self.kind]
        if self.node_type:
            bits.append(self.node_type + (f"#{self.node_id}" if self.node_id else ""))
        bits.append(f"step={self.step}")
        if self.duration_ms is not None:
            bits.append(f"{self.duration_ms:.1f}ms")
        if self.error:
            bits.append(f"error={self.error!r}")
        elif self.outputs:
            bits.append("out=[" + ", ".join(map(str, self.outputs)) + "]")
        return "  ".join(bits)


# ── Eval result (batch evaluation) ───────────────────────────────────────


@dataclass
class EvalResult:
    """What a batch eval produced (see :meth:`Graph.eval`).

    ``metrics`` is the aggregate (mean over completed episodes, e.g.
    ``success_rate`` / ``spl``); ``episodes`` is the per-episode breakdown;
    ``by_task`` groups the aggregate by selector/task; ``run`` is the raw
    :class:`app.agent_loop.eval_batch.EvalRun` for power users.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    by_task: dict[str, Any] = field(default_factory=dict)
    run: Any = None

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]

    @classmethod
    def _from_run(cls, run: Any) -> EvalResult:
        eps = [
            {
                "index": e.episode_index,
                "episode_id": e.episode_id,
                "scene": e.scene_id,
                "status": e.status,
                "steps": e.step_count,
                "metrics": dict(e.metrics),
                "error": e.error,
            }
            for e in run.episodes
        ]
        return cls(
            metrics=dict(run.aggregate_metrics or {}),
            episodes=eps,
            by_task=dict(run.aggregate_by_task or {}),
            run=run,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        done = sum(1 for e in self.episodes if e.get("status") == "completed")
        return f"EvalResult(episodes={len(self.episodes)}, completed={done}, metrics={self.metrics})"


# ── Validation ────────────────────────────────────────────────────────────


class GraphValidationError(ValueError):
    """Raised by :meth:`Graph.run` / :meth:`Graph.eval` (with ``check=True``,
    the default) when the graph references a node type or port that does not
    resolve — the class of typo that otherwise fails silently (an unknown
    node type no-ops; a misspelled port handle never delivers). Subclasses
    ``ValueError`` so existing ``except ValueError`` handlers still catch it.
    """


def _node_ports(cls: Any, config: dict) -> tuple[list, list]:
    """Resolve a handler's (input_ports, output_ports) for a given config —
    honouring a dynamic ``_resolve_ports(config)`` classmethod when present
    (graphIn/graphOut/iterIn/…), else the class-level port lists. Falls back
    to the class lists if the resolver raises."""
    resolver = getattr(cls, "_resolve_ports", None)
    if callable(resolver):
        try:
            ins, outs = resolver(config or {})
            return list(ins), list(outs)
        except Exception:
            pass
    return list(getattr(cls, "input_ports", [])), list(getattr(cls, "output_ports", []))


# ── Introspection: the node catalog ───────────────────────────────────────
#
# "What node types exist, and what are their ports?" — answered from a one-off
# workspace scan, WITHOUT spawning any env server. ``scan_all`` reads each
# nodeset's ``get_tools()`` (static node classes), so server nodesets
# (env_habitat / model_sam / …) contribute their metadata too. The result is
# cached; pass ``refresh=True`` after adding a nodeset. This is the programmatic
# surface that both a human (in a REPL / notebook) and a coding agent use to
# discover node types + ports instead of grepping the nodeset files.

_CATALOG: tuple | None = None  # (registry, index, ns_by_type) — lazily scanned once


def _env_name_from_python(python: str | None) -> str | None:
    """Best-effort conda env name from a ``.../envs/<name>/bin/python`` path."""
    if not python:
        return None
    parts = Path(python).parts
    if "envs" in parts:
        i = parts.index("envs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _build_catalog(refresh: bool = False) -> tuple:
    global _CATALOG
    if _CATALOG is not None and not refresh:
        return _CATALOG
    from .agent_loop.builtin_nodes import NODE_HANDLERS
    from .components.registry import WorkspaceComponentRegistry
    from .config import get_settings

    # Builtins (+ any node registered in-process) first; nodeset scan overlays.
    index: dict[str, tuple] = {nt: (None, cls) for nt, cls in NODE_HANDLERS.items()}
    ns_by_type: dict[str, Any] = {}

    s = get_settings()
    reg = WorkspaceComponentRegistry(s.workspace_dir, active_dir=s.active_workspace_dir or None)
    reg.scan_all()  # discovery reads get_tools() per nodeset — no env server spawned
    for name, inst in reg._discovered_nodesets.items():
        try:
            tools = inst.get_tools()
        except Exception:
            tools = []
        for t in tools:
            nt = getattr(t, "node_type", None) or getattr(t, "name", None)
            if not nt:
                continue
            index[nt] = (name, t if isinstance(t, type) else type(t))
            ns_by_type[nt] = inst
    _CATALOG = (reg, index, ns_by_type)
    return _CATALOG


def _is_server_ns(ns: Any) -> bool:
    return ns is not None and getattr(type(ns), "server_python", None) is not None


def catalog(
    *,
    nodeset: str | None = None,
    server: bool | None = None,
    kind: str | None = None,
    builtins: bool = True,
    refresh: bool = False,
) -> list[str]:
    """Sorted node-type keys available to the SDK, from a one-off workspace scan
    (metadata only — no env server is spawned). Filters: ``nodeset`` name;
    ``server`` (``True`` runs in a subprocess, ``False`` in-process); ``kind``
    (``block`` / ``control`` / ``composite``); ``builtins`` (include the
    graphIn / graphOut / iterIn / llmCall / … builtins). ``refresh=True``
    re-scans after a nodeset was added."""
    _, index, ns_by_type = _build_catalog(refresh=refresh)
    out = []
    for nt, (nsname, cls) in index.items():
        if nsname is None and not builtins:
            continue
        if nodeset is not None and nsname != nodeset:
            continue
        if server is not None and _is_server_ns(ns_by_type.get(nt)) != server:
            continue
        if kind is not None and getattr(cls, "kind", "block") != kind:
            continue
        out.append(nt)
    return sorted(out)


def _port_dict(p: Any) -> dict:
    return {
        "name": getattr(p, "name", ""),
        "type": getattr(p, "wire_type", "ANY"),
        "optional": bool(getattr(p, "optional", False)),
        "description": getattr(p, "description", ""),
    }


def describe(node_type: str, *, refresh: bool = False) -> dict:
    """Structured metadata for one node type — where it runs and its shape::

        {node_type, nodeset, server, env, kind, category, description,
         inputs:  [{name, type, optional, description}, ...],
         outputs: [{name, type, optional, description}, ...],
         config:  [{name, type, label, default}, ...]}

    Raises ``KeyError`` (with a did-you-mean hint) on an unknown type. Reads
    static class metadata only — no env server is spawned."""
    import difflib

    _, index, ns_by_type = _build_catalog(refresh=refresh)
    if node_type not in index:
        hint = difflib.get_close_matches(node_type, sorted(index), n=3)
        raise KeyError(
            f"unknown node type {node_type!r}" + (f"; did you mean {hint}?" if hint else "")
        )
    nsname, cls = index[node_type]
    ins, outs = _node_ports(cls, {})
    ns = ns_by_type.get(node_type)
    server_python = getattr(type(ns), "server_python", None) if ns is not None else None
    ui = getattr(cls, "ui_config", None)
    cfg_fields = getattr(ui, "config_fields", []) if ui is not None else []
    return {
        "node_type": node_type,
        "nodeset": nsname,
        "server": server_python is not None,
        "env": _env_name_from_python(server_python),
        "kind": getattr(cls, "kind", "block"),
        "category": getattr(cls, "category", ""),
        "description": getattr(cls, "description", "") or getattr(cls, "display_name", ""),
        "inputs": [_port_dict(p) for p in ins],
        "outputs": [_port_dict(p) for p in outs],
        "config": [
            {
                "name": getattr(f, "name", ""),
                "type": getattr(f, "field_type", ""),
                "label": getattr(f, "label", ""),
                "default": getattr(f, "default", None),
            }
            for f in cfg_fields
        ],
    }


def nodesets(*, refresh: bool = False) -> list[dict]:
    """List discovered nodesets as ``[{name, server, env, node_types}, ...]``
    (metadata only — no env server is spawned)."""
    reg, index, _ = _build_catalog(refresh=refresh)
    out = []
    for name, inst in reg._discovered_nodesets.items():
        sp = getattr(type(inst), "server_python", None)
        out.append(
            {
                "name": name,
                "server": sp is not None,
                "env": _env_name_from_python(sp),
                "node_types": sorted(nt for nt, (nn, _) in index.items() if nn == name),
            }
        )
    return sorted(out, key=lambda d: d["name"])


# ── Typed stub generation ─────────────────────────────────────────────────
#
# Turn the introspection catalog into real, importable, IDE-typed symbols:
# one module per nodeset under ``agentcanvas/nodes/``, each node a NodeProxy
# subclass with its ``node_type`` baked in and ``Literal``-typed ports. This
# is the "no strings" surface — ``g.add(env_habitat.Reset)`` instead of
# ``g.add("env_habitat__reset")`` — and it imports nothing heavy (the stub
# only depends on ``agentcanvas``, never habitat/torch). Generated files are
# regeneratable; treat them as build output (gitignored), not source.


def _one_line(s: str, limit: int = 160) -> str:
    s = " ".join(str(s).split()).replace("\\", "\\\\").replace('"', "'")
    return (s[: limit - 1] + "…") if len(s) > limit else s


def _stub_class_name(node_type: str) -> str:
    suffix = node_type.split("__", 1)[1] if "__" in node_type else node_type
    parts = [p for p in suffix.replace("-", "_").split("_") if p]
    name = "".join(w[:1].upper() + w[1:] for w in parts) or "Node"
    if name[0].isdigit():
        name = "N" + name
    if keyword.iskeyword(name):
        name += "_"
    return name


def _emit_nodeset_module(nodeset_name: str, nodes: list[dict]) -> str:
    """Return the source of one generated stub module (pure — no I/O)."""
    out = [
        "# GENERATED by agentcanvas.generate_node_stubs — do not edit.",
        f"# nodeset: {nodeset_name}. Regenerate after nodeset changes.",
        "from __future__ import annotations",
        "",
        "from typing import Literal",
        "",
        "from agentcanvas import NodeProxy, PortRef",
        "",
    ]
    seen: set[str] = set()
    names: list[str] = []
    for nd in nodes:
        cname = _stub_class_name(nd["node_type"])
        base, k = cname, 2
        while cname in seen:
            cname, k = f"{base}{k}", k + 1
        seen.add(cname)
        names.append(cname)
        where = ("server · " + (nd.get("env") or "?")) if nd.get("server") else "local"
        out += [
            "",
            f"class {cname}(NodeProxy):",
            f'    """{nd["node_type"]} — {_one_line(nd.get("description") or nd["node_type"])}  [{where}]"""',
            f"    node_type = {nd['node_type']!r}",
        ]
        outs = [p["name"] for p in nd.get("outputs", [])]
        ins = [p["name"] for p in nd.get("inputs", [])]
        if outs:
            out += [
                f"    def out(self, port: Literal[{', '.join(map(repr, outs))}]) -> PortRef:",
                "        return super().out(port)",
            ]
        if ins:
            out += [
                f"    def in_(self, port: Literal[{', '.join(map(repr, ins))}]) -> PortRef:",
                "        return super().in_(port)",
            ]
    out += ["", "", f"__all__ = {names!r}", ""]
    return "\n".join(out)


def generate_node_stubs(dst: str | Path | None = None, *, refresh: bool = False) -> list[str]:
    """Emit typed node stubs — one module per nodeset — so
    ``from agentcanvas.nodes import env_habitat`` then ``g.add(env_habitat.Reset)``
    gives IDE autocomplete on node types *and* ports. ``dst`` defaults to the
    installed ``agentcanvas`` package's ``nodes/`` dir. Regenerate after nodeset
    changes. Returns the written file paths (sorted)."""
    if dst is None:
        import importlib.util

        spec = importlib.util.find_spec("agentcanvas")
        locs = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
        if not locs:
            raise RuntimeError("cannot locate the 'agentcanvas' package — pass dst= explicitly")
        dst = Path(locs[0]) / "nodes"
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    modules: list[str] = []
    for ns in nodesets(refresh=refresh):
        types = ns.get("node_types") or []
        modname = re.sub(r"\W", "_", ns["name"])
        if not types or not modname or modname[0].isdigit() or keyword.iskeyword(modname):
            continue
        src = _emit_nodeset_module(ns["name"], [describe(t) for t in types])
        (dst / f"{modname}.py").write_text(src)
        written.append(str(dst / f"{modname}.py"))
        modules.append(modname)

    modules.sort()
    init = ["# GENERATED by agentcanvas.generate_node_stubs — do not edit.", ""]
    init += [f"from . import {m}" for m in modules]
    init += ["", f"__all__ = {modules!r}", ""]
    (dst / "__init__.py").write_text("\n".join(init))
    written.append(str(dst / "__init__.py"))
    return sorted(written)


# ── The builder ──────────────────────────────────────────────────────────


class Graph:
    """Graph SDK builder over :class:`app.graph_def.GraphDefinition`.

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

    @overload
    def add(self, node_type: type[_NP], id: str | None = ..., **config: Any) -> _NP: ...
    @overload
    def add(self, node_type: str | type, id: str | None = ..., **config: Any) -> NodeHandle: ...

    def add(self, node_type: Any, id: str | None = None, **config: Any) -> NodeHandle:
        """Add a node by ``node_type`` — a registered type key *or* a node class.

        ``node_type`` may be:

        * a ``str`` — the registered ``node_type`` key (e.g. ``"env_habitat__reset"``);
        * a **node class** — any class carrying a ``node_type`` attribute: a
          ``BaseCanvasNode`` subclass you defined, or a generated
          :class:`NodeProxy` (``from agentcanvas.nodes import env_habitat``).
          Passing a class gives a real symbol (autocomplete / go-to-def / typo
          = error); a ``NodeProxy`` subclass additionally returns a typed
          handle whose ``out`` / ``in_`` autocomplete the port names.

        ``id`` auto-generates as ``"<type>_<n>"`` when omitted; extra kwargs
        become the node's ``config``.
        """
        proxy_cls: type[NodeProxy] | None = None
        if isinstance(node_type, type):
            if issubclass(node_type, NodeProxy):
                proxy_cls = node_type
            nt = getattr(node_type, "node_type", "") or ""
            if not nt:
                raise ValueError(
                    f"{node_type.__name__} has no 'node_type' attribute — cannot add it"
                )
            node_type = nt
        if id is None:
            n = self._type_counts.get(node_type, 0)
            self._type_counts[node_type] = n + 1
            id = f"{node_type}_{n}"
        if any(nd.id == id for nd in self._def.nodes):
            raise ValueError(f"duplicate node id: {id!r}")
        node_def = NodeDef(id=id, type=node_type, label=id, config=dict(config))
        self._def.nodes.append(node_def)
        return proxy_cls(self, node_def) if proxy_cls is not None else NodeHandle(self, node_def)

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

    # -- loop / hooks / composites -----------------------------------------

    def loop(
        self,
        *,
        carry: Any = (),
        init: Any = (),
        id: str = "iter_in",
        out_id: str = "iter_out",
        version: int = 3,
    ) -> Loop:
        """Add a paired iterIn/iterOut episode loop and return a :class:`Loop`.

        ``carry`` and ``init`` are iterables of ``"name"`` or
        ``("name", "WIRE_TYPE")``:

        * ``carry`` — values written back each step (``iterOut``) and re-entered
          on the next step.  Seeded on step 0, then the carry supersedes.
        * ``init`` — seed-only values that persist across steps unchanged (no
          carry-back), e.g. an instruction string.

        The correct ``persist`` flags are set automatically (init-only ports
        persist; carried ports do not persist on the init side, only via the
        ``iterOut`` carry), so the loop can't accidentally starve on iter 1+.
        Wire it up through the returned handle's :meth:`Loop.seed` /
        :meth:`Loop.feed` / :meth:`Loop.carry` / :meth:`Loop.stop`.
        """
        specs_init = [_norm_loop_spec(e, carried=False) for e in init]
        specs_carry = [_norm_loop_spec(e, carried=True) for e in carry]
        specs = specs_init + specs_carry
        init_ports = [
            {"name": s.name, "wire_type": s.wire_type, "persist": not s.carried} for s in specs
        ]
        ii = self.add("iterIn", id=id, version=version, pairedWith=out_id, initPorts=init_ports)
        out_ports = [
            {"name": s.name, "wire_type": s.wire_type, "persist": True} for s in specs_carry
        ]
        io = self.add("iterOut", id=out_id, pairedWith=id, ports=out_ports)
        return Loop(self, ii, io, specs)

    def hook(
        self,
        event: str,
        command: str,
        *,
        match_node_type: str = "*",
        match_node_id: str | None = None,
        timeout_ms: int = 1000,
        enabled: bool = True,
    ) -> Any:
        """Attach a lifecycle shell hook (``GraphStart`` / ``PreNodeExecute`` /
        ``PostNodeExecute`` / ``GraphComplete`` / ``GraphError``).
        """
        from .graph_def import HookDef

        h = HookDef(
            event=event,
            command=command,
            match_node_type=match_node_type,
            match_node_id=match_node_id,
            timeout_ms=timeout_ms,
            enabled=enabled,
        )
        self._def.hooks.append(h)
        return h

    def composite(
        self,
        id: str,
        subgraph: Graph | GraphDefinition,
        *,
        type: str = "compositeNode",
        label: str = "",
        **config: Any,
    ) -> NodeHandle:
        """Add a composite node wrapping a nested graph.

        ``flatten_graph`` expands it into the parent before execution
        (``GraphExecutor.run`` BUILD 2/7), so composites run in-process.
        """
        from .graph_def import NodeDef, _synthesize_iterin_ports

        sub = subgraph.definition if isinstance(subgraph, Graph) else subgraph
        _synthesize_iterin_ports(sub)
        if any(nd.id == id for nd in self._def.nodes):
            raise ValueError(f"duplicate node id: {id!r}")
        nd = NodeDef(id=id, type=type, label=label or id, config=dict(config), subgraph=sub)
        self._def.nodes.append(nd)
        return NodeHandle(self, nd)

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

    def _check_resolved(self) -> None:
        """Post-load structural check (default-on via ``run`` / ``eval``'s
        ``check``): every node type resolves to a registered handler, every
        wired port exists on its node, and every required input is fed. Runs
        *after* nodesets are loaded, so it covers server-proxy node types too.
        Aggregates every problem into one :class:`GraphValidationError` with
        did-you-mean hints. Conservative — a node whose ports can't be
        resolved, a composite (expanded later at run), and the machine-wired
        ``iterIn`` / ``iterOut`` are skipped for the port checks, so it never
        false-positives on a well-formed graph.
        """
        import difflib

        from .agent_loop.builtin_nodes import NODE_HANDLERS

        errors: list[str] = []
        known = sorted(NODE_HANDLERS)
        nodes_by_id = {n.id: n for n in self._def.nodes}
        _generated = {"iterIn", "iterOut"}  # ports synthesised by loop(), not user-typed

        incoming: dict[str, set[str]] = {}
        for e in self._def.edges:
            incoming.setdefault(e.target, set()).add(e.targetHandle or "")

        ins_of: dict[str, list] = {}
        outs_of: dict[str, list] = {}
        for n in self._def.nodes:
            cls = NODE_HANDLERS.get(n.type)
            if cls is None:
                if getattr(n, "subgraph", None):
                    continue  # composite — its internals are validated when flattened
                hint = difflib.get_close_matches(n.type, known, n=1)
                tip = f" — did you mean {hint[0]!r}?" if hint else ""
                errors.append(f"node {n.id!r}: unknown node type {n.type!r}{tip}")
                continue
            if n.type in _generated:
                continue
            ins_of[n.id], outs_of[n.id] = _node_ports(cls, n.config or {})

        for e in self._def.edges:
            src = nodes_by_id.get(e.source)
            outs = outs_of.get(e.source)
            if src is not None and outs and e.sourceHandle:
                names = {p.name for p in outs}
                if e.sourceHandle not in names:
                    hint = difflib.get_close_matches(e.sourceHandle, sorted(names), n=1)
                    tip = f" — did you mean {hint[0]!r}?" if hint else ""
                    errors.append(
                        f"edge {e.id!r}: {src.type}.{e.sourceHandle!r} is not an output "
                        f"port (outputs: {sorted(names)}){tip}"
                    )
            dst = nodes_by_id.get(e.target)
            ins = ins_of.get(e.target)
            if dst is not None and ins and e.targetHandle:
                names = {p.name for p in ins}
                if e.targetHandle not in names:
                    hint = difflib.get_close_matches(e.targetHandle, sorted(names), n=1)
                    tip = f" — did you mean {hint[0]!r}?" if hint else ""
                    errors.append(
                        f"edge {e.id!r}: {dst.type}.{e.targetHandle!r} is not an input "
                        f"port (inputs: {sorted(names)}){tip}"
                    )

        for nid, ins in ins_of.items():
            required = {p.name for p in ins if not getattr(p, "optional", True)}
            missing = sorted(required - incoming.get(nid, set()))
            if missing:
                errors.append(
                    f"node {nid!r} (type={nodes_by_id[nid].type}): required input "
                    f"port(s) {missing} have no incoming edge"
                )

        if errors:
            raise GraphValidationError(
                "graph validation failed ({} issue{}):\n  - {}".format(
                    len(errors), "" if len(errors) == 1 else "s", "\n  - ".join(errors)
                )
            )

    def run(
        self,
        *,
        session: Any = None,
        validate: bool = False,
        check: bool = True,
        on_event: Any = None,
        load_nodesets: bool | str = "auto",
        worker_count: int = 1,
        teardown_nodesets: bool = True,
        registry: Any = None,
        step_delay_ms: int = 0,
        step_budget_override: int | None = None,
    ) -> RunResult:
        """Run the graph in-process and return a :class:`RunResult`.

        Drives ``GraphExecutor.run()`` directly — no FastAPI, no scheduler, no
        GPU-admission gate (that lives in ``JobScheduler``, not the executor).
        Set ``validate=True`` to fail fast on connectivity/wire errors before
        loading. ``check`` (default ``True``) runs *after* nodesets load and
        raises :class:`GraphValidationError` — with a did-you-mean hint — if any
        node type or wired port fails to resolve; it turns the two silent
        typo classes (unknown node type → no-op, misspelled handle → never
        delivered) into a fail-fast. Pass ``check=False`` for a dry run.

        ``on_event`` (optional) is a callback ``(RunEvent) -> None`` fired as the
        run proceeds — ``graph_start``, per-node ``node_start`` / ``node_finish``
        / ``node_error``, then ``graph_complete`` (or ``graph_error``). It makes
        an in-process run observable — which node fired when, with what inputs /
        outputs — without tailing logs. Sync or async; keep it cheap (it runs in
        the executor's loop); a raise inside it is logged, never breaks the run.
        To collect the whole stream, pass ``on_event=events.append``.

        ``load_nodesets`` controls whether the workspace registry is scanned
        and the graph's nodesets auto-loaded before the run (mutating the
        global ``NODE_HANDLERS`` the executor reads):

        * ``"auto"`` (default) — load iff the graph has any nodeset node type
          (a ``type`` containing ``"__"``).  Pure-Python / builtin graphs run
          untouched; ``mapgpt``-style graphs pull their nodesets in.
        * ``True`` / ``False`` — force on / off.

        Env nodesets spawn server-mode subprocesses (in their own conda env);
        with ``teardown_nodesets`` (default), whatever this run *started* is
        unloaded afterwards so no idle server is left behind.  Pass an existing
        ``registry`` to reuse a warm one (its pre-loaded nodesets are left
        alone).
        """
        from .agent_loop.graph_executor import GraphExecutor
        from .graph_def import _synthesize_iterin_ports

        _synthesize_iterin_ports(self._def)
        if validate:
            from .graph_def import validate_graph_connectivity

            validate_graph_connectivity(self._def)

        sess = session or DefaultSession()

        # Wrap the user callback: the executor emits raw event dicts; hand the
        # SDK user a typed RunEvent. None → executor's on_event stays off.
        _sink = None
        if on_event is not None:

            def _sink(ev: dict, _cb: Any = on_event) -> Any:
                return _cb(RunEvent._from_dict(ev))

        async def _drive() -> Any:
            reg = None
            started: list[str] = []
            if self._wants_nodesets(load_nodesets):
                reg, started = await self._load_nodesets(worker_count=worker_count, registry=registry)
            try:
                if check:
                    self._check_resolved()
                executor = GraphExecutor(on_event=_sink)
                await executor.run(
                    self._def,
                    sess,
                    step_delay_ms=step_delay_ms,
                    step_budget_override=step_budget_override,
                )
                return executor
            finally:
                if reg is not None and teardown_nodesets and started:
                    for ns in started:
                        # best-effort cleanup
                        with contextlib.suppress(Exception):
                            await reg.unload_nodeset(ns)

        executor = asyncio.run(_drive())
        return RunResult._from_run(executor, sess)

    # -- nodeset loading (env / GPU nodeset graphs) ------------------------

    def _wants_nodesets(self, flag: bool | str) -> bool:
        if flag == "auto":
            return any("__" in nd.type for nd in self._def.nodes)
        return bool(flag)

    async def _load_nodesets(self, *, worker_count: int = 1, registry: Any = None) -> tuple[Any, list[str]]:
        """Scan the workspace registry and auto-load this graph's nodesets.

        Returns ``(registry, started)`` where ``started`` is the list of
        nodesets this call actually loaded (so the caller can tear only those
        down).  Raises if any required nodeset fails to load.
        """
        reg = registry
        if reg is None:
            from .components.registry import WorkspaceComponentRegistry
            from .config import get_settings

            s = get_settings()
            reg = WorkspaceComponentRegistry(s.workspace_dir, active_dir=s.active_workspace_dir or None)
            reg.scan_all()
        result = await reg.ensure_nodesets_for_graph(self._def, worker_count=worker_count)
        if result.get("failed"):
            raise RuntimeError(
                f"nodeset load failed: {result['failed']} "
                f"(unknown={result.get('unknown')}, loaded={result.get('loaded')})"
            )
        if result.get("unknown"):
            import logging

            logging.getLogger(__name__).warning(
                "graph_sdk: unknown nodesets (nodes will no-op): %s", result["unknown"]
            )
        return reg, list(result.get("loaded", []))

    def _detect_env_nodeset(self, registry: Any = None) -> str:
        """The graph's env nodeset name (for eval episode placement)."""
        if registry is not None and hasattr(registry, "detect_env_nodesets_for_graph"):
            try:
                envs = registry.detect_env_nodesets_for_graph(self._def)
                if envs:
                    return envs[0] if isinstance(envs, (list, tuple)) else str(envs)
            except Exception:  # pragma: no cover - fall back to the prefix heuristic
                pass
        for nd in self._def.nodes:
            if "__" in nd.type and nd.type.split("__")[0].startswith("env_"):
                return nd.type.split("__")[0]
        return ""

    # -- batch evaluation --------------------------------------------------

    def eval(
        self,
        *,
        episodes: int = 1,
        dataset: str = "R2R",
        split: str = "val_unseen",
        start_index: int = 0,
        episode_indices: list[int] | None = None,
        env_nodeset: str | None = None,
        worker_count: int = 1,
        step_budget: int | None = None,
        run_id: str | None = None,
        load_nodesets: bool | str = "auto",
        teardown_nodesets: bool = True,
        registry: Any = None,
        check: bool = True,
    ) -> EvalResult:
        """Batch-evaluate this (``eval_graph=True``) graph over N episodes.

        Drives the same :class:`~app.agent_loop.eval_batch.BatchEvalRunner` the
        backend eval API uses — one ``GraphExecutor`` run per episode, metrics
        harvested off the ``metrics`` graphOut and averaged over completed
        episodes — but in-process, with no FastAPI / scheduler.  Auto-loads the
        graph's nodesets first (env panel + server); tears them down after.

        ``episodes`` runs ``start_index .. start_index+episodes-1`` of
        ``dataset``/``split`` (or the explicit ``episode_indices`` list).
        Returns an :class:`EvalResult` (``.metrics`` aggregate, ``.episodes``
        per-episode, ``.by_task``).

        This is a real env run (spawns the env server, uses GPU + the LLM) — an
        *experiment*; a large sweep belongs behind ``/experiment:run``.
        """
        from datetime import datetime, timezone

        n = len(episode_indices) if episode_indices else episodes

        async def _drive() -> Any:
            reg, started = None, []
            if self._wants_nodesets(load_nodesets):
                reg, started = await self._load_nodesets(worker_count=worker_count, registry=registry)
            try:
                if check:
                    self._check_resolved()
                from .agent_loop.eval_batch import (
                    BatchEvalRunner,
                    EvalConfig,
                    EvalRun,
                    EvalStatus,
                )

                cfg = EvalConfig(
                    graph_name=self._def.name or "Graph SDK",
                    env_nodeset=env_nodeset or self._detect_env_nodeset(reg),
                    selectors={"dataset": dataset, "split": split},
                    dataset=dataset,
                    split=split,
                    episode_count=n,
                    start_episode_index=start_index,
                    episode_indices=episode_indices,
                    worker_count=worker_count,
                    step_budget=step_budget,
                )
                now = datetime.now(timezone.utc)
                run = EvalRun(
                    run_id=run_id or f"graphsdk-{now.strftime('%Y%m%d-%H%M%S')}",
                    config=cfg,
                    status=EvalStatus.pending,
                    created_at=now.isoformat(),
                )
                await BatchEvalRunner(run, self._def).execute()
                return run
            finally:
                if reg is not None and teardown_nodesets and started:
                    for ns in started:
                        # best-effort cleanup
                        with contextlib.suppress(Exception):
                            await reg.unload_nodeset(ns)

        return EvalResult._from_run(asyncio.run(_drive()))

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

    def to_code(self, *, func_name: str = "build", var: str = "g") -> str:
        """Compile this graph into a standalone ``graph_sdk`` builder script.

        The inverse of authoring by hand: emits runnable Python that rebuilds
        the same topology (nodes, wires, containers, grants, hooks) via this
        very API — a self-contained reconstruction, like
        ``mapgpt_mp3d_sdk.py`` but generated.  Round-trips: the emitted
        script's graph is semantically identical to this one.
        """
        from .graph_sdk_codegen import graph_to_code

        return graph_to_code(self._def, func_name=func_name, var=var)

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
