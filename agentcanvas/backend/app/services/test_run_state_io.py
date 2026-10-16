"""The run files through their interface: JobSpec / RunSummary round trips,
tolerant reads, the parent-side finish paths, atomic writes."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..agent_loop.eval_config import EvalConfig
from .run_state_io import (
    JobSpec,
    RunSummary,
    Scheduling,
    atomic_write_json,
    is_done,
    mark_aborted,
    read_shared_urls,
    read_summary,
    touch_done,
    write_shared_urls,
)


def _spec(**eval_fields) -> JobSpec:
    cfg = EvalConfig(graph_name="g", env_nodeset="env_x", **eval_fields)
    return JobSpec(
        eval=cfg,
        graph={"nodes": [{"id": "a", "type": "t_a"}, {"id": "b", "type": "t_b"}], "edges": []},
        scheduling=Scheduling(marginal_vram_mb=512, exclusive_gpu=True, priority="high"),
        active_workspace_dir="/tmp/overlay",
        run_id="r1",
        created_at="2026-10-17T00:00:00",
    )


# ── JobSpec ─────────────────────────────────────────────────────────────


def test_jobspec_round_trips_through_disk(tmp_path: Path) -> None:
    spec = _spec(
        selectors={"split": "val_unseen", "task_id": 3},
        episode_indices=[4, 9],
        episode_selectors=[{"task_id": 1}, {"task_id": 2}],
        step_budget=50,
        per_step_budget_sec=2.5,
    )
    spec.write(tmp_path)
    back = JobSpec.read(tmp_path)
    assert back == spec
    assert back.node_types == ["t_a", "t_b"]
    # The on-disk shape keeps the historical top-level keys.
    raw = json.loads((tmp_path / "spec.json").read_text())
    assert set(raw) == {"run_id", "created_at", "eval", "scheduling", "graph", "active_workspace_dir"}
    assert raw["eval"]["env_nodeset"] == "env_x"


def test_jobspec_read_tolerates_missing_fields(tmp_path: Path) -> None:
    """An older spec with a sparse eval block and no scheduling takes the
    EvalConfig / Scheduling defaults — stated once, in the dataclasses."""
    atomic_write_json(
        tmp_path / "spec.json",
        {"eval": {"graph_name": "g", "episode_count": 3, "stale_key": 1}, "graph": {"nodes": []}},
    )
    spec = JobSpec.read(tmp_path)
    assert spec is not None
    assert spec.eval == EvalConfig(graph_name="g", episode_count=3)
    assert spec.eval.split == "val_unseen" and spec.eval.worker_count == 1
    assert spec.scheduling == Scheduling()
    assert spec.active_workspace_dir is None and spec.run_id == ""


def test_jobspec_missing_is_none(tmp_path: Path) -> None:
    assert JobSpec.read(tmp_path) is None
    assert RunSummary.read(tmp_path) is None
    assert read_summary(tmp_path) is None
    assert read_shared_urls(tmp_path) == {}
    assert is_done(tmp_path) is False


# ── RunSummary ──────────────────────────────────────────────────────────


class _Status(str, Enum):
    running = "running"


@dataclass
class _Episode:
    episode_index: int
    status: str = "completed"
    episode_id: str = "ep"
    scene_id: str = "scene"
    instruction: str = "go"
    metrics: dict = field(default_factory=lambda: {"success": 1.0})
    step_count: int = 3
    elapsed_sec: float = 1.5
    error: str | None = None
    selectors: dict = field(default_factory=dict)
    worker_id: int = 0


@dataclass
class _Run:
    run_id: str
    config: EvalConfig
    status: _Status = _Status.running
    episodes: list = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    aggregate_by_task: dict = field(default_factory=dict)
    total_episodes: int = 0
    created_at: str = "t0"
    finished_at: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None


def test_summary_from_run_round_trips_and_counts(tmp_path: Path) -> None:
    run = _Run(
        run_id="r1",
        config=EvalConfig(graph_name="g", env_nodeset="env_x", dataset="R2R"),
        episodes=[_Episode(0), _Episode(1, status="error", error="boom"), _Episode(2, status="running")],
        aggregate_metrics={"success": 0.5},
        total_episodes=3,
        elapsed_sec=12.345,
    )
    RunSummary.from_run(run).write(tmp_path)
    back = RunSummary.read(tmp_path)
    assert back is not None
    assert back.status == "running" and back.config == run.config
    assert [e["episode_index"] for e in back.episodes] == [0, 1, 2]
    assert back.to_dict() == RunSummary.from_run(run).to_dict()
    page = back.page_summary()
    assert page["completed_count"] == 2 and page["error_count"] == 1
    assert page["dataset"] == "R2R" and page["env_nodeset"] == "env_x"
    assert page["elapsed_sec"] == 12.3


def test_initial_summary_carries_the_spec(tmp_path: Path) -> None:
    spec = _spec(split="val_seen")
    RunSummary.initial(spec, status="pending").write(tmp_path)
    back = RunSummary.read(tmp_path)
    assert back is not None
    assert back.status == "pending" and back.run_id == "r1" and back.created_at == spec.created_at
    assert back.config.env_nodeset == "env_x" and back.config.split == "val_seen"
    assert back.episodes == [] and back.total_episodes == 0 and not back.is_terminal


def test_finish_appends_error_and_stamps_finished_at(tmp_path: Path) -> None:
    summary = RunSummary.initial(_spec())
    summary.error = "first"
    summary.finish("error", "second").write(tmp_path)
    back = RunSummary.read(tmp_path)
    assert back is not None
    assert back.status == "error" and back.is_terminal
    assert back.error == "first\nsecond" and back.finished_at


def test_summary_read_tolerates_old_files(tmp_path: Path) -> None:
    atomic_write_json(
        tmp_path / "summary.json",
        {"run_id": "old", "config": {"graph_name": "g"}, "status": "completed", "extra": 1},
    )
    back = RunSummary.read(tmp_path)
    assert back is not None
    assert back.run_id == "old" and back.status == "completed"
    assert back.config.graph_name == "g" and back.config.split == "val_unseen"
    assert back.episodes == [] and back.elapsed_sec == 0.0


def test_mark_aborted_promotes_running(tmp_path: Path) -> None:
    RunSummary.initial(_spec()).write(tmp_path)
    mark_aborted(tmp_path)
    after = RunSummary.read(tmp_path)
    assert after is not None and after.status == "aborted"
    assert "[aborted]" in (after.error or "")


def test_mark_aborted_skips_terminal(tmp_path: Path) -> None:
    RunSummary.initial(_spec(), status="completed").write(tmp_path)
    mark_aborted(tmp_path)
    assert read_summary(tmp_path)["status"] == "completed"  # untouched


# ── the other files ─────────────────────────────────────────────────────


def test_shared_urls_round_trip(tmp_path: Path) -> None:
    urls = {"vlm_prismatic": "http://127.0.0.1:34453"}
    write_shared_urls(tmp_path, urls)
    assert read_shared_urls(tmp_path) == urls


def test_done_flag(tmp_path: Path) -> None:
    assert not is_done(tmp_path)
    touch_done(tmp_path)
    assert is_done(tmp_path)
    # Idempotent — calling again is fine.
    touch_done(tmp_path)
    assert is_done(tmp_path)


def test_atomic_writes_no_partial_files(tmp_path: Path) -> None:
    """Concurrent writers must never leave the reader seeing a torn file."""
    target = tmp_path / "summary.json"
    payload_a = {"x": "a" * 5000}
    payload_b = {"y": "b" * 5000}

    def write_loop(payload: dict, n: int) -> None:
        for _ in range(n):
            atomic_write_json(target, payload)

    threads = [
        threading.Thread(target=write_loop, args=(payload_a, 50)),
        threading.Thread(target=write_loop, args=(payload_b, 50)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = json.loads(target.read_text())
    assert data in (payload_a, payload_b)
