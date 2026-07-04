"""What happens downstream when a node's result is ``{"error": ...}``.

A failed node (e.g. a server-mode proxy surfacing an HTTP 500) returns
``{"error": "..."}`` instead of its declared port dict. The routing
block only forwards declared handles, so nothing reaches downstream —
consumers stay unready, the queue drains, and the run finishes as if
it had completed.

At eval level this manifested as episodes marked ``status="completed"``
with ``step_count=0`` and empty metrics (run ``20260516_101057``:
11/100 episodes silently dropped, SR computed over the 89 survivors).

These tests pin the CURRENT silent behavior. They are the exposure
half of the fix: when the executor learns to surface node-error
results explicitly, flip the assertions alongside the change.
"""

from __future__ import annotations

from typing import Any

from ..components.bases import BaseCanvasNode, PortDef
from ..graph_def import GraphDefinition
from .builtin_nodes import register_node
from .test_executor_scopes import _edge, _node, _run


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


register_node(_ErrorNode)


def _dag_graph() -> GraphDefinition:
    """seed → err → consumer, plain DAG."""
    return GraphDefinition(
        name="error_dag",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node("err", "_test_error_node"),
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


def test_error_result_currently_starves_downstream_silently() -> None:
    exe = _run(_dag_graph())
    # The error node fired, but nothing was routed onward: the consumer
    # never became ready and the run finished without any error signal.
    assert (exe.nodes["consumer"].state.get("total_fires") or 0) == 0


def test_error_in_loop_body_currently_stalls_the_scope() -> None:
    exe = _run(_loop_graph())
    # gate (downstream of the failing node) never fires, so stop never
    # arrives; nothing advances and the run just drains away. This is
    # the unit-level shape of the "completed, step_count=0, metrics={}"
    # eval pathology.
    assert (exe.nodes["gate"].state.get("total_fires") or 0) == 0
