"""Node-error conviction — ``{"error": ...}`` results end the run loudly.

A failed node (e.g. a server-mode proxy surfacing an HTTP 500) returns
``{"error": "..."}`` instead of its declared port dict. The routing
block only forwards declared handles, so nothing reaches downstream —
consumers starve and the queue drains. Before 2026-07-04 the run then
finished indistinguishable from a clean completion: at eval level this
manifested as episodes marked ``status="completed"`` with
``step_count=0`` and empty metrics (run ``20260516_101057``: 11/100
episodes silently dropped, SR computed over the 89 survivors).

Now the executor records every error-shaped result into
``node_errors`` and convicts the run at the finalise stage by raising
``NodeErrorAggregate`` — AFTER the after-loop verdict stage, so
final-side metrics are still collected. The executor's outer except
block swallows it by design (its error surface is ``session._status``
plus the error bus), so callers observe ``_status == "error"`` — the
eval layer converts that into ``episode.status = "error"``. Nodes that
legitimately declare an ``error`` output port are exempt.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..components.bases import BaseCanvasNode, PortDef
from ..graph_def import GraphDefinition, _synthesize_iterin_ports
from .builtin_nodes import register_node
from .graph_executor import GraphExecutor
from .test_executor_scopes import _edge, _node, _StubSession


class _ErrorNode(BaseCanvasNode):
    """Test-only node whose forward reports failure the way the
    server-mode proxy does: an ``{"error": ...}`` result instead of the
    declared ``out`` port."""

    node_type = "_test_error_node"
    display_name = "Error Node (test)"
    category = "control"
    icon = "AlertTriangle"
    input_ports = [PortDef("trigger", "ANY", optional=True)]
    output_ports = [PortDef("out", "ANY")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"error": "boom (synthetic node failure)"}


class _ErrorPortNode(BaseCanvasNode):
    """Test-only node that legitimately declares an ``error`` OUTPUT
    port (like env_libero tools) — its ``error`` key is data, not a
    failure report, and must not convict the run."""

    node_type = "_test_error_port_node"
    display_name = "Error-Port Node (test)"
    category = "control"
    icon = "AlertTriangle"
    input_ports = [PortDef("trigger", "ANY", optional=True)]
    output_ports = [PortDef("out", "ANY"), PortDef("error", "TEXT")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"out": "ok", "error": ""}


register_node(_ErrorNode)
register_node(_ErrorPortNode)


def _run_convicted(graph: GraphDefinition) -> GraphExecutor:
    """Drive a graph expected to end convicted; return the executor.

    The executor swallows the internal ``NodeErrorAggregate`` by design
    — conviction is observable as ``session._status == "error"``.
    """
    _synthesize_iterin_ports(graph)
    exe = GraphExecutor()
    sess = _StubSession()
    asyncio.run(exe.run(graph, sess, step_delay_ms=0))
    assert sess._status == "error"
    return exe


def _dag_graph(err_type: str = "_test_error_node") -> GraphDefinition:
    """seed → err → consumer, plain DAG."""
    return GraphDefinition(
        name="error_dag",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node("err", err_type),
            _node("consumer", "_test_fire_counter"),
        ],
        edges=[
            _edge("e0", "seed", "err", sh="value", th="trigger"),
            _edge("e1", "err", "consumer", sh="out", th="trigger"),
        ],
    )


def _loop_graph(loop_iters: int = 3) -> GraphDefinition:
    """Loop whose body chain contains a failing node: the error starves
    the iterOut's stop input, so the loop only ends on step budget."""
    return GraphDefinition(
        name="error_loop",
        eval_graph=False,
        step_budget=loop_iters,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "driver", "wire_type": "ANY", "persist": True}],
            ),
            _node("err", "_test_error_node"),
            _node("gate", "_test_fire_counter", done_after=loop_iters),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
        ],
        edges=[
            _edge("e_seed", "seed", "iter_in", sh="value", th="init_driver"),
            _edge("e_drive", "iter_in", "err", sh="init_driver", th="trigger"),
            _edge("e_err_gate", "err", "gate", sh="out", th="trigger"),
            _edge("e_stop", "gate", "iter_out", sh="done", th="stop"),
            _edge("e_v", "gate", "iter_out", sh="value", th="v"),
        ],
    )


def test_error_result_convicts_the_run() -> None:
    exe = _run_convicted(_dag_graph())
    # Downstream still starves (routing semantics unchanged) ...
    assert (exe.nodes["consumer"].state.get("total_fires") or 0) == 0
    # ... but the failure is recorded and the run ends convicted,
    # naming the culprit node.
    assert exe.node_errors and exe.node_errors[0]["node_id"] == "err"


def test_error_in_loop_body_convicts_at_run_end() -> None:
    exe = _run_convicted(_loop_graph())
    # gate (downstream of the failing node) never fires — the loop
    # drains on step budget — and every failing firing was recorded.
    assert (exe.nodes["gate"].state.get("total_fires") or 0) == 0
    assert len(exe.node_errors) >= 1


def test_declared_error_port_is_not_convicted() -> None:
    """An ``error`` key emitted by a node that declares an ``error``
    output port is ordinary data — the run completes cleanly."""
    graph = _dag_graph(err_type="_test_error_port_node")
    _synthesize_iterin_ports(graph)
    exe = GraphExecutor()
    sess = _StubSession()
    asyncio.run(exe.run(graph, sess, step_delay_ms=0))  # must not raise
    assert sess._status == "done"
    assert not exe.node_errors
    # The declared ``out`` port routed normally.
    assert exe.nodes["consumer"].state.get("total_fires") == 1
