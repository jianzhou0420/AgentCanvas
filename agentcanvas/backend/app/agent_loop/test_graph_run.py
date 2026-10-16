"""Tests for ``graph_run`` — the one prepare-and-run pipeline every entry
point adapts.

``prepare_graph`` is exercised against a fake registry (its interface is
four methods); ``GraphRun`` and ``BatchEvalRunner`` run a real pure-Python
graph through the real executor and are asserted through ``RunOutcome`` /
``EpisodeResult`` only — no executor internals.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ..components.bases import BaseCanvasNode, PortDef
from ..graph_def import GraphDefinition
from .builtin_nodes import register_node
from .eval_batch import BatchEvalRunner, EvalConfig, EvalRun, EvalStatus
from .graph_run import (
    GraphRun,
    NodesetLoadError,
    RunSession,
    detect_env_nodeset,
    harvest_outputs,
    prepare_graph,
    run_graph_once,
)

# ── a tiny pure-Python graph ─────────────────────────────────────────────


class _Verdict(BaseCanvasNode):
    """Emits a scalar ``success`` and a ``metrics`` dict — the two graphOut
    shapes the harvest understands."""

    node_type = "graph_run_test_verdict"
    input_ports: list = []
    output_ports = [PortDef("success", "ANY", ""), PortDef("metrics", "ANY", "")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {"success": True, "metrics": {"spl": 0.5, "note": "ignored"}}


class _Consumer(BaseCanvasNode):
    """One required input — unwired, it fails connectivity validation."""

    node_type = "graph_run_test_consumer"
    input_ports = [PortDef("x", "ANY", "")]
    output_ports: list = []

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        return {}


register_node(_Verdict)
register_node(_Consumer)


def _verdict_graph() -> GraphDefinition:
    from ..graph_sdk import Graph

    g = Graph(name="graph-run-test")
    v = g.add(_Verdict.node_type, id="v")
    s = g.graph_out("success")
    m = g.graph_out("metrics")
    g.connect(v.out("success"), s.in_("value"))
    g.connect(v.out("metrics"), m.in_("value"))
    return GraphDefinition.from_dict(g.to_dict())


# ── prepare_graph ────────────────────────────────────────────────────────


class _FakeRegistry:
    """The four registry methods ``prepare_graph`` crosses."""

    def __init__(
        self,
        *,
        loaded: list[str] | None = None,
        failed: list[str] | None = None,
        unknown: list[str] | None = None,
        env_candidates: list[str] | None = None,
        metadata: dict[str, dict] | None = None,
        hooks: list | None = None,
    ) -> None:
        self._result = {
            "loaded": loaded or [],
            "already_loaded": [],
            "failed": failed or [],
            "unknown": unknown or [],
        }
        self._env_candidates = env_candidates or []
        self._metadata = metadata or {}
        self._hooks = hooks or []
        self.calls: list[tuple[str, int]] = []

    async def ensure_nodesets_for_graph(
        self, graph: GraphDefinition, worker_count: int = 1
    ) -> dict:
        self.calls.append(("all", worker_count))
        return dict(self._result)

    async def ensure_shared_nodesets_for_graph(self, graph: GraphDefinition) -> dict:
        self.calls.append(("shared", 1))
        return dict(self._result)

    def detect_env_nodesets_for_graph(self, graph: GraphDefinition) -> list[str]:
        return list(self._env_candidates)

    async def get_eval_metadata_for_nodeset(self, name: str) -> dict:
        return self._metadata.get(name, {})

    def get_global_hooks(self) -> list:
        return list(self._hooks)


def test_prepare_routes_full_vs_shared_and_threads_worker_count() -> None:
    graph = _verdict_graph()
    reg = _FakeRegistry()
    asyncio.run(prepare_graph(reg, graph, worker_count=3))
    asyncio.run(prepare_graph(reg, graph, shared_only=True))
    assert reg.calls == [("all", 3), ("shared", 1)]


def test_prepare_failed_policy_raise_or_warn() -> None:
    graph = _verdict_graph()
    reg = _FakeRegistry(failed=["env_x"])
    with pytest.raises(NodesetLoadError, match="failed to load nodesets"):
        asyncio.run(prepare_graph(reg, graph, on_failed="raise"))
    prepared = asyncio.run(prepare_graph(reg, graph, on_failed="warn"))
    assert prepared.failed == ["env_x"]


def test_prepare_unknown_policy_raise_or_warn() -> None:
    graph = _verdict_graph()
    reg = _FakeRegistry(unknown=["model_y"])
    with pytest.raises(NodesetLoadError, match="unknown nodesets"):
        asyncio.run(prepare_graph(reg, graph, on_unknown="raise"))
    prepared = asyncio.run(prepare_graph(reg, graph, on_unknown="warn"))
    assert prepared.unknown == ["model_y"]


def test_prepare_env_is_first_candidate_with_metadata_and_hooks_come_along() -> None:
    graph = _verdict_graph()
    reg = _FakeRegistry(
        loaded=["env_b"],
        env_candidates=["model_a", "env_b"],
        metadata={"env_b": {"env_name": "b"}},
        hooks=["hook-1"],
    )
    prepared = asyncio.run(prepare_graph(reg, graph))
    assert prepared.env_nodeset == "env_b"
    assert prepared.loaded == ["env_b"]
    assert prepared.global_hooks == ["hook-1"]
    assert prepared.graph is graph


def test_prepare_no_env_metadata_means_no_env() -> None:
    graph = _verdict_graph()
    reg = _FakeRegistry(env_candidates=["model_a"])
    assert asyncio.run(prepare_graph(reg, graph)).env_nodeset == ""


def test_prepare_validate_rejects_an_unwired_required_port() -> None:
    from ..graph_sdk import Graph

    g = Graph(name="dangling")
    g.add(_Consumer.node_type, id="c")  # required ``x`` never wired
    graph = GraphDefinition.from_dict(g.to_dict())
    with pytest.raises(ValueError, match="required input"):
        asyncio.run(prepare_graph(_FakeRegistry(), graph, validate=True))
    asyncio.run(prepare_graph(_FakeRegistry(), graph, validate=False))  # caller's call


def test_detect_env_without_registry_uses_env_prefix() -> None:
    graph = _verdict_graph()
    assert asyncio.run(detect_env_nodeset(None, graph)) == ""
    graph.nodes[0].type = "env_fake__step"
    assert asyncio.run(detect_env_nodeset(None, graph)) == "env_fake"


# ── GraphRun / RunOutcome ────────────────────────────────────────────────


def test_graph_run_outcome_harvests_outputs_and_metrics() -> None:
    outcome = asyncio.run(run_graph_once(_verdict_graph(), execution_id="t1"))
    assert outcome.status == "done"
    assert not outcome.errored
    assert outcome.node_errors == []
    assert outcome.outputs["success"] is True
    assert outcome.outputs["metrics"] == {"spl": 0.5, "note": "ignored"}
    # numeric harvest: scalar port + flattened metrics dict, non-numeric dropped
    assert outcome.metrics == {"success": 1.0, "spl": 0.5}
    assert outcome.session._execution_id == "t1"


def test_graph_run_keeps_the_caller_session() -> None:
    sess = RunSession("mine")
    gr = GraphRun(_verdict_graph(), session=sess)
    outcome = asyncio.run(gr.run())
    assert outcome.session is sess
    assert sess._status == "done"
    assert gr.outcome().status == "done"  # harvest is repeatable


def test_outcome_before_run_is_idle_and_empty() -> None:
    gr = GraphRun(_verdict_graph())
    outcome = gr.outcome()
    assert outcome.status == "idle"
    assert outcome.outputs == {} and outcome.metrics == {}
    assert outcome.error_summary() == "graph execution ended with status=idle"


def test_harvest_outputs_unwraps_single_handle_keeps_multi() -> None:
    gr = GraphRun(_verdict_graph())
    asyncio.run(gr.run())
    out = harvest_outputs(gr.executor)
    assert out["success"] is True  # one handle → bare value
    inst = next(
        n for n in gr.executor.nodes.values() if (n.config or {}).get("portName") == "success"
    )
    inst.state["_last_inputs"] = {"a": 1, "b": 2}
    assert harvest_outputs(gr.executor)["success"] == {"a": 1, "b": 2}


# ── BatchEvalRunner as an adapter ────────────────────────────────────────


def test_batch_eval_runner_pure_graph_run_dir_and_on_episode(tmp_path: Path) -> None:
    run = EvalRun(
        run_id="t-run",
        config=EvalConfig(graph_name="graph-run-test", episode_count=2, episode_indices=[4, 9]),
    )
    seen: list[int] = []
    run_dir = tmp_path / "pool" / "t-run"
    runner = BatchEvalRunner(
        run, _verdict_graph(), run_dir=run_dir, on_episode=lambda ep: seen.append(ep.episode_index)
    )
    asyncio.run(runner.execute())

    assert run.status == EvalStatus.completed
    assert [ep.episode_index for ep in run.episodes] == [4, 9]
    assert all(ep.status == "completed" for ep in run.episodes)
    assert run.episodes[0].metrics == {"success": 1.0, "spl": 0.5}
    assert run.aggregate_metrics == {"success": 1.0, "spl": 0.5}
    assert seen == [4, 9]  # on_episode fires once per episode, in order
    # artefacts land under the caller's run_dir, not a repo-relative path
    assert json.loads((run_dir / "graph.json").read_text())["name"] == "graph-run-test"
    ep = json.loads((run_dir / "episodes" / "ep0004" / "episode.json").read_text())
    assert ep["status"] == "completed" and ep["metrics"] == {"success": 1.0, "spl": 0.5}


def test_batch_eval_runner_default_run_dir_follows_pool_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTCANVAS_EVAL_RUNS_DIR", str(tmp_path / "slot-pool"))
    run = EvalRun(run_id="t-pool", config=EvalConfig(graph_name="g", episode_count=1))
    BatchEvalRunner(run, _verdict_graph())
    assert (tmp_path / "slot-pool" / "t-pool" / "graph.json").exists()
