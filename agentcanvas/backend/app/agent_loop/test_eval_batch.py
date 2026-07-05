"""Pure-logic tests for the batch-eval layer.

Covers the episode bookkeeping and metric plumbing that every eval run
relies on: contiguous worker partitioning, run/episode summaries
(completed vs error accounting), metric aggregation (global and
per-task), and ``_collect_metrics`` — the graphOut harvest that turns
node state into ``EpisodeResult.metrics`` (including the server-mode
JSON-string path and the inner-scope skip).

The subprocess/worker orchestration paths (``execute``,
``_run_one_episode``) are intentionally out of scope here — they need
env workers and are exercised by integration smoke runs; the
node-error → episode conviction logic they contain is engine-tested in
``test_node_error_routing``.
"""

from __future__ import annotations

from typing import Any

import pytest

from .eval_batch import (
    BatchEvalRunner,
    EpisodeResult,
    EvalConfig,
    EvalRun,
    EvalStatus,
    _partition_contiguous,
)
from .graph_executor import NodeInstance

# ── worker partitioning ─────────────────────────────────────────────────


def test_partition_even_split() -> None:
    assert _partition_contiguous(6, 3) == [[0, 1], [2, 3], [4, 5]]


def test_partition_remainder_goes_to_leading_workers() -> None:
    # numpy array_split distribution: first (total % workers) get +1.
    assert _partition_contiguous(7, 3) == [[0, 1, 2], [3, 4], [5, 6]]


def test_partition_more_workers_than_episodes() -> None:
    chunks = _partition_contiguous(2, 4)
    assert chunks == [[0], [1], [], []]  # trailing workers exit immediately


def test_partition_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="worker_count"):
        _partition_contiguous(5, 0)


def test_partition_is_contiguous_and_complete() -> None:
    chunks = _partition_contiguous(23, 5)
    flat = [i for chunk in chunks for i in chunk]
    assert flat == list(range(23))  # nothing lost, order preserved


# ── run / episode summaries ─────────────────────────────────────────────


def _ep(idx: int, status: str, metrics: dict | None = None, **kw: Any) -> EpisodeResult:
    return EpisodeResult(episode_index=idx, status=status, metrics=metrics or {}, **kw)


def _run_with(*episodes: EpisodeResult) -> EvalRun:
    return EvalRun(
        run_id="r1",
        config=EvalConfig(graph_name="g"),
        status=EvalStatus.running,
        episodes=list(episodes),
        total_episodes=len(episodes),
    )


def test_summary_counts_terminal_and_error_episodes() -> None:
    run = _run_with(
        _ep(0, "completed", {"success": 1.0}),
        _ep(1, "error"),
        _ep(2, "pending"),
    )
    s = run.to_summary()
    # completed_count = every TERMINAL episode (clean + error) — drives
    # the progress bar; error_count is the broken subset. Neither is SR.
    assert s["completed_count"] == 2
    assert s["error_count"] == 1
    assert s["total_episodes"] == 3
    assert s["status"] == "running"


def test_episode_summary_carries_fields_and_rounds_elapsed() -> None:
    run = _run_with()
    ep = _ep(4, "completed", {"success": 1.0}, elapsed_sec=1.2345, worker_id=2)
    s = run.to_episode_summary(ep)
    assert s["episode_index"] == 4
    assert s["elapsed_sec"] == 1.2
    assert s["worker_id"] == 2
    assert s["status"] == "completed"


# ── aggregation ─────────────────────────────────────────────────────────


def test_aggregate_means_over_completed_with_metrics_only() -> None:
    agg = BatchEvalRunner._compute_aggregate(
        [
            _ep(0, "completed", {"success": 1.0, "spl": 0.8}),
            _ep(1, "completed", {"success": 0.0, "spl": 0.2}),
            _ep(2, "error", {"success": 1.0}),  # excluded: errored
            _ep(3, "completed"),  # excluded: no metrics
        ]
    )
    assert agg == {"success": 0.5, "spl": 0.5}


def test_aggregate_missing_key_averages_over_present_episodes() -> None:
    agg = BatchEvalRunner._compute_aggregate(
        [
            _ep(0, "completed", {"success": 1.0, "extra": 4.0}),
            _ep(1, "completed", {"success": 0.0}),
        ]
    )
    assert agg["success"] == 0.5
    assert agg["extra"] == 4.0  # averaged over the 1 episode carrying it


def test_aggregate_empty_when_nothing_completed() -> None:
    assert BatchEvalRunner._compute_aggregate([_ep(0, "error")]) == {}


def test_task_key_precedence_task_id_then_canonical_json() -> None:
    assert BatchEvalRunner._episode_task_key(_ep(0, "completed", selectors={"task_id": "t1"})) == "t1"
    # Same selectors, different insertion order → same canonical bucket.
    k1 = BatchEvalRunner._episode_task_key(_ep(0, "completed", selectors={"a": 1, "b": 2}))
    k2 = BatchEvalRunner._episode_task_key(_ep(1, "completed", selectors={"b": 2, "a": 1}))
    assert k1 == k2
    assert BatchEvalRunner._episode_task_key(_ep(0, "completed")) == "_default"


def test_aggregate_by_task_partitions_before_averaging() -> None:
    out = BatchEvalRunner._compute_aggregate_by_task(
        [
            _ep(0, "completed", {"success": 1.0}, selectors={"task_id": "pick"}),
            _ep(1, "completed", {"success": 0.0}, selectors={"task_id": "pick"}),
            _ep(2, "completed", {"success": 1.0}, selectors={"task_id": "move"}),
        ]
    )
    assert out["pick"] == {"success": 0.5}
    assert out["move"] == {"success": 1.0}


# ── graphOut metric harvest ─────────────────────────────────────────────


class _FakeExecutor:
    def __init__(self, nodes: dict[str, NodeInstance], scopes: dict[str, str] | None = None):
        self.nodes = nodes
        self._scopes = scopes or {}

    def _scope_of(self, node_id: str) -> str:
        return self._scopes.get(node_id, "")


class _FakeRunner:
    """Just enough LoopRunner surface for ``_collect_metrics``."""

    def __init__(
        self,
        nodes: dict[str, NodeInstance],
        scopes: dict[str, str] | None = None,
        metrics: dict | None = None,
    ):
        self._executor = _FakeExecutor(nodes, scopes)
        self._metrics = metrics


def _graphout(nid: str, port_name: str, snapshot: dict) -> NodeInstance:
    return NodeInstance(
        id=nid,
        type="graphOut",
        config={"portName": port_name},
        state={"_last_inputs": snapshot},
    )


def _collect(runner: _FakeRunner) -> dict[str, float]:
    # ``_collect_metrics`` never touches self — call unbound to avoid
    # BatchEvalRunner.__init__'s run-dir setup.
    return BatchEvalRunner._collect_metrics(None, runner)  # type: ignore[arg-type]


def test_collect_flattens_metrics_dict_and_scalar_ports() -> None:
    metrics = _collect(
        _FakeRunner(
            {
                "m": _graphout("m", "metrics", {"value": {"success": True, "spl": 0.8, "note": "x"}}),
                "s": _graphout("s", "score", {"value": 0.7}),
            }
        )
    )
    assert metrics == {"success": 1.0, "spl": 0.8, "score": 0.7}  # non-numeric dropped


def test_collect_parses_json_string_metrics_from_server_mode() -> None:
    metrics = _collect(
        _FakeRunner({"m": _graphout("m", "metrics", {"value": '{"success": 1, "spl": 0.5}'})})
    )
    assert metrics == {"success": 1.0, "spl": 0.5}


def test_collect_ignores_unparseable_metrics_string() -> None:
    assert _collect(_FakeRunner({"m": _graphout("m", "metrics", {"value": "not json"})})) == {}


def test_collect_skips_inner_scope_graphouts() -> None:
    metrics = _collect(
        _FakeRunner(
            {
                "outer": _graphout("outer", "success", {"value": 1.0}),
                "inner": _graphout("inner", "success", {"value": 0.0}),
            },
            scopes={"inner": "some_loop_scope"},  # outer stays graph-scope ("")
        )
    )
    assert metrics == {"success": 1.0}


def test_collect_falls_back_to_first_non_none_input() -> None:
    metrics = _collect(
        _FakeRunner({"g": _graphout("g", "success", {"other": None, "whatever": 1.0})})
    )
    assert metrics == {"success": 1.0}


def test_collect_runner_metrics_merge_wins() -> None:
    metrics = _collect(
        _FakeRunner(
            {"g": _graphout("g", "success", {"value": 0.0})},
            metrics={"success": 1.0, "extra": 2.0, "skipme": "str"},
        )
    )
    assert metrics == {"success": 1.0, "extra": 2.0}  # runner._metrics overrides
