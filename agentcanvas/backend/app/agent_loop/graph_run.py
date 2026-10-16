"""Graph run — prepare a graph against the registry, run it once, read the outcome.

Every entry point that runs a graph (canvas Play, the eval API, the eval
subprocess, the Graph SDK) is an adapter over this module:

* :func:`prepare_graph` — the one "parse → validate → ensure nodesets →
  wire-type check → detect env → collect hooks" pipeline. Failure and
  unknown-nodeset policies are parameters, so each adapter states its
  strictness instead of re-implementing the sequence.
* :class:`GraphRun` — one GraphExecutor built for one run of one graph.
  ``run()`` drives it; ``outcome()`` harvests what the run produced, also
  mid-run or after a timeout, so a caller never reads executor internals.
* :class:`RunSession` — the executor-facing session contract. The executor
  writes ``_status`` / ``_current_step`` / ``_metrics``, reads ``_metrics``
  and ``principles``, and calls ``_ws(...)``. Headless runs use it as-is;
  the canvas ``LoopRunner`` subclasses it to add pause / stop / status.
* :func:`harvest_outputs` / :func:`harvest_metrics` — the graphOut harvest,
  in one place, with one scope rule.

Deletion test: remove this module and the pipeline grows back in five
callers (it did — this module is their consolidation).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..graph_def import (
    GraphDefinition,
    HookDef,
    validate_edge_wire_types,
    validate_graph_connectivity,
)
from .scope_analysis import GRAPH_SCOPE_ID

if TYPE_CHECKING:
    from ..models import WSMessage

log = logging.getLogger("agentcanvas.graph-run")

Policy = Literal["raise", "warn"]


class NodesetLoadError(RuntimeError):
    """A nodeset the graph needs failed to load, or is unknown to the registry."""


# ── Session contract ──


@dataclass
class ExecutionPrinciples:
    """Configure execution behavior for batch eval mode.

    When set on the session, modifies execution behavior. When None
    (default), no behavioral changes — the canvas path is unchanged.
    """

    no_pause: bool = True  # ignore pause events in eval
    collect_metrics: bool = True  # collect per-episode metrics from env
    suppress_nav_events: bool = True  # suppress ALL nav_* WS events
    source_tag: str = "eval"  # tag on all WS events for frontend routing


class RunSession:
    """What ``GraphExecutor.run()`` needs from its ``session`` argument.

    The executor writes ``_status`` / ``_current_step`` / ``_metrics``, reads
    ``_metrics`` / ``principles``, and calls ``_ws(...)`` to build broadcast
    messages. With no WebSocket clients connected the broadcast layer is a
    silent no-op, so a headless run has no UI side effects.
    """

    def __init__(
        self,
        execution_id: str | None = None,
        principles: ExecutionPrinciples | None = None,
    ) -> None:
        self._status: str = "idle"
        self._current_step: int = 0
        self._metrics: dict[str, Any] | None = None
        self._execution_id: str | None = execution_id
        self.principles: ExecutionPrinciples | None = principles

    def _ws(self, msg_type: str, data: Any = None) -> WSMessage:
        """Create a WSMessage tagged with execution_id and optional source."""
        # Lazy: ``models`` pulls in pydantic, which the stdlib-only SDK import
        # tier must not require (see test_dependency_boundary).
        from ..models import WSMessage

        source = self.principles.source_tag if self.principles else None
        return WSMessage(type=msg_type, data=data, execution_id=self._execution_id, source=source)


# ── Prepare ──


@dataclass
class PreparedGraph:
    """A graph whose nodesets are loaded and whose env is known."""

    graph: GraphDefinition
    env_nodeset: str  # "" when the graph has no env nodeset with eval metadata
    loaded: list[str] = field(default_factory=list)
    already_loaded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    wire_warnings: list[str] = field(default_factory=list)
    global_hooks: list[HookDef] = field(default_factory=list)


async def detect_env_nodeset(registry: Any, graph: GraphDefinition) -> str:
    """The graph's env nodeset: the first candidate with eval metadata.

    One rule for every caller. With a registry, candidates come from
    ``detect_env_nodesets_for_graph`` and the first one whose
    ``get_eval_metadata_for_nodeset`` is non-empty wins. Without a registry
    (SDK runs that skipped loading) the ``env_`` node-type prefix is the
    only evidence available, so the first such prefix wins. ``""`` = none.
    """
    if registry is not None and hasattr(registry, "detect_env_nodesets_for_graph"):
        for ns_name in registry.detect_env_nodesets_for_graph(graph):
            metadata = await registry.get_eval_metadata_for_nodeset(ns_name)
            if metadata:
                return ns_name
        return ""
    for nd in graph.nodes:
        if "__" in nd.type and nd.type.split("__")[0].startswith("env_"):
            return nd.type.split("__")[0]
    return ""


async def prepare_graph(
    registry: Any,
    graph: GraphDefinition,
    *,
    worker_count: int = 1,
    shared_only: bool = False,
    on_failed: Policy = "raise",
    on_unknown: Policy = "warn",
    validate: bool = True,
) -> PreparedGraph:
    """Make ``graph`` runnable against ``registry``.

    Steps, in order: connectivity validation (``validate``), nodeset
    auto-load (all nodesets, or only ``parallelism="shared"`` ones when
    ``shared_only`` — the parent side of subprocess eval), the failure /
    unknown policies, the whole-graph wire-type check (warn-only, pending
    ADR-027 staged enforcement), env-nodeset detection, and the registry's
    global hooks.

    ``on_failed`` / ``on_unknown``: ``"raise"`` raises
    :class:`NodesetLoadError`; ``"warn"`` logs and continues (unknown node
    types no-op in the executor).
    """
    if validate:
        validate_graph_connectivity(graph)

    if shared_only:
        result = await registry.ensure_shared_nodesets_for_graph(graph)
    else:
        result = await registry.ensure_nodesets_for_graph(graph, worker_count=worker_count)

    loaded = list(result.get("loaded") or [])
    already = list(result.get("already_loaded") or [])
    failed = list(result.get("failed") or [])
    unknown = list(result.get("unknown") or [])
    if loaded:
        log.info("auto-loaded nodesets for %s: %s", graph.name or "<graph>", loaded)
    if failed:
        msg = f"failed to load nodesets: {failed}"
        if on_failed == "raise":
            raise NodesetLoadError(msg)
        log.warning("%s (continuing; their nodes will no-op)", msg)
    if unknown:
        msg = f"unknown nodesets (not in workspace scan): {unknown}"
        if on_unknown == "raise":
            raise NodesetLoadError(msg)
        log.warning("%s (continuing; their nodes will no-op)", msg)

    # Whole-graph wire-type check, now that nodeset proxy classes are in
    # NODE_HANDLERS (the pre-load validate_graph_connectivity only saw
    # builtins). Warn-only: it surfaces nodeset env↔method shape mismatches
    # without aborting the run, pending a clean sweep before this is
    # promoted to a hard error (ADR-027 staged enforcement).
    wire_warnings = validate_edge_wire_types(graph)
    if wire_warnings:
        log.warning(
            "wire-type mismatches (post-nodeset-load, warn-only):\n  - %s",
            "\n  - ".join(wire_warnings),
        )

    env_nodeset = await detect_env_nodeset(registry, graph)
    hooks = list(registry.get_global_hooks()) if hasattr(registry, "get_global_hooks") else []

    return PreparedGraph(
        graph=graph,
        env_nodeset=env_nodeset,
        loaded=loaded,
        already_loaded=already,
        failed=failed,
        unknown=unknown,
        wire_warnings=wire_warnings,
        global_hooks=hooks,
    )


# ── Harvest ──


def _graph_scope_graphouts(executor: Any) -> list[Any]:
    """The graph-scope ``graphOut`` nodes of a (possibly mid-run) executor.

    Inner-scope graphOut nodes preserved by ``flatten_graph`` (when a
    composite has its own scope) are scope-internal latches carrying
    per-iter cross-scope return values, not run outputs; they are skipped
    so the harvest stays faithful to the author's output declaration.
    """
    nodes = getattr(executor, "nodes", None)
    if not nodes:
        return []
    out = []
    for node in nodes.values():
        if node.type != "graphOut":
            continue
        if hasattr(executor, "_scope_of"):
            n_scope = executor._scope_of(node.id)
            if n_scope and n_scope != GRAPH_SCOPE_ID:
                continue
        out.append(node)
    return out


def harvest_outputs(executor: Any) -> dict[str, Any]:
    """``{portName: value}`` from every graph-scope graphOut's last fire.

    A graphOut with a single incoming handle unwraps to that bare value;
    multiple handles stay a dict. A graphOut that never fired is absent.
    Falls back to the node id when no ``portName`` is configured.
    """
    outputs: dict[str, Any] = {}
    for node in _graph_scope_graphouts(executor):
        port_name = (node.config or {}).get("portName") or node.id
        last = node.state.get("_last_inputs") or {}
        if not last:
            continue
        if len(last) == 1:
            outputs[port_name] = next(iter(last.values()))
        else:
            outputs[port_name] = dict(last)
    return outputs


def harvest_metrics(
    executor: Any, session_metrics: dict[str, Any] | None = None
) -> dict[str, float]:
    """Numeric metrics from the graph's graphOut sinks, plus session metrics.

    Two contributions per graph-scope ``graphOut``:

    * ``portName == "metrics"`` — the value is treated as a metrics dict (or
      JSON-stringified dict, since server-mode evaluate nodes cross the HTTP
      boundary on a TEXT-typed wire) and its numeric fields are flattened in.
    * Any other portName carrying a numeric/bool scalar — included as
      ``metrics[portName] = float(value)``.

    ``session_metrics`` (what a node emitted via ``{"done": True,
    "metrics": ...}``) is merged last and wins on key conflict.
    """
    metrics: dict[str, float] = {}
    for node in _graph_scope_graphouts(executor):
        snapshot = node.state.get("_last_inputs") or {}
        if not snapshot:
            continue
        value = snapshot.get("value")
        if value is None:
            # Permissive fallback — accept any non-None pending input
            for v in snapshot.values():
                if v is not None:
                    value = v
                    break
        port_name = (node.config or {}).get("portName") or ""

        if port_name == "metrics":
            if isinstance(value, str) and value:
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    value = None
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float, bool)):
                        metrics[k] = float(v)
        elif port_name and isinstance(value, (int, float, bool)):
            metrics[port_name] = float(value)
    if session_metrics and isinstance(session_metrics, dict):
        for k, v in session_metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
    return metrics


# ── Run ──


@dataclass
class RunOutcome:
    """What one run of one graph produced."""

    status: str  # "done" | "error" | "running" (harvested mid-run) | "idle"
    step_count: int
    outputs: dict[str, Any]
    metrics: dict[str, float]
    session_metrics: dict[str, Any]
    node_errors: list[dict]
    executor: Any
    session: Any

    @property
    def errored(self) -> bool:
        return self.status == "error"

    def error_summary(self, head: int = 3) -> str:
        """One line naming the first ``head`` node errors, or the status."""
        if self.node_errors:
            first = "; ".join(
                f"{err['node_id']}@step{err['step']}: {err['error']}"
                for err in self.node_errors[:head]
            )
            return f"{len(self.node_errors)} node error(s) during run: {first}" + (
                "; ..." if len(self.node_errors) > head else ""
            )
        return f"graph execution ended with status={self.status}"


class GraphRun:
    """One GraphExecutor, built for one run of one graph.

    Construct, then ``await run(...)``. ``outcome()`` reads results off the
    executor and session at any time — after a clean finish, after an
    exception, or after the caller's own timeout cancelled ``run()`` — so no
    caller needs to know the executor's internals.
    """

    def __init__(
        self,
        graph: GraphDefinition,
        *,
        session: RunSession | None = None,
        execution_id: str | None = None,
        logger: Any = None,
        env_panel_overrides: dict[str, Any] | None = None,
        server_url_overrides: dict[str, str] | None = None,
        on_event: Any = None,
    ) -> None:
        from .graph_executor import GraphExecutor

        self.graph = graph
        self.session: RunSession = session if session is not None else RunSession(execution_id)
        if execution_id is not None:
            self.session._execution_id = execution_id
        self.executor = GraphExecutor(
            logger=logger,
            env_panel_overrides=env_panel_overrides,
            server_url_overrides=server_url_overrides,
            on_event=on_event,
        )

    async def run(
        self,
        *,
        step_delay_ms: int = 0,
        global_hooks: list[HookDef] | None = None,
        step_budget_override: int | None = None,
        stop_event: asyncio.Event | None = None,
        pause_event: asyncio.Event | None = None,
    ) -> RunOutcome:
        await self.executor.run(
            graph=self.graph,
            session=self.session,
            execution_id=self.session._execution_id,
            step_delay_ms=step_delay_ms,
            stop_event=stop_event,
            pause_event=pause_event,
            global_hooks=global_hooks,
            step_budget_override=step_budget_override,
        )
        return self.outcome()

    def outcome(self) -> RunOutcome:
        session_metrics = dict(getattr(self.session, "_metrics", None) or {})
        return RunOutcome(
            status=getattr(self.session, "_status", "idle"),
            step_count=int(getattr(self.session, "_current_step", 0) or 0),
            outputs=harvest_outputs(self.executor),
            metrics=harvest_metrics(self.executor, session_metrics),
            session_metrics=session_metrics,
            node_errors=list(getattr(self.executor, "node_errors", None) or []),
            executor=self.executor,
            session=self.session,
        )


async def run_graph_once(
    graph: GraphDefinition,
    *,
    session: RunSession | None = None,
    execution_id: str | None = None,
    logger: Any = None,
    env_panel_overrides: dict[str, Any] | None = None,
    server_url_overrides: dict[str, str] | None = None,
    on_event: Any = None,
    step_delay_ms: int = 0,
    global_hooks: list[HookDef] | None = None,
    step_budget_override: int | None = None,
    stop_event: asyncio.Event | None = None,
    pause_event: asyncio.Event | None = None,
) -> RunOutcome:
    """Build a :class:`GraphRun`, run it, return its :class:`RunOutcome`."""
    gr = GraphRun(
        graph,
        session=session,
        execution_id=execution_id,
        logger=logger,
        env_panel_overrides=env_panel_overrides,
        server_url_overrides=server_url_overrides,
        on_event=on_event,
    )
    return await gr.run(
        step_delay_ms=step_delay_ms,
        global_hooks=global_hooks,
        step_budget_override=step_budget_override,
        stop_event=stop_event,
        pause_event=pause_event,
    )
