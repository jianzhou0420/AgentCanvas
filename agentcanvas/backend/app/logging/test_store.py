"""The execution-log store on a tmp_path: addressing both run families,
the tolerant reader, listing, and the aggregations."""

from __future__ import annotations

import json
from pathlib import Path

from .store import GRAPH_FILE, LOG_FILE, LogDir, LogStore


def _write_log(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def _entry(step: int, node_type: str = "t", **extra) -> dict:
    return {
        "timestamp": f"2026-10-17T00:00:{step:02d}",
        "source": "canvas",
        "step": step,
        "node_id": f"n{step}",
        "node_type": node_type,
        "duration_ms": 10.0,
        **extra,
    }


def _layout(tmp_path: Path) -> LogStore:
    store = LogStore(tmp_path / "runs", tmp_path / "eval_runs")
    canvas = store.canvas_run_dir("c1")
    _write_log(canvas / LOG_FILE, [_entry(1), _entry(2, error="x")])
    (canvas / "assets").mkdir()
    (canvas / "assets" / "f.png").write_bytes(b"png")
    (canvas / GRAPH_FILE).write_text(json.dumps({"name": "canvas-graph"}))
    run = store.eval_run_dir("20261017_000000")
    run.mkdir(parents=True)
    (run / GRAPH_FILE).write_text(json.dumps({"name": "eval-graph"}))
    for idx in (0, 3):
        ep = store.episode_dir("20261017_000000", idx)
        _write_log(ep / LOG_FILE, [_entry(1, "env__step", compute_ms=5.0, queue_wait_ms=1.0)])
        (ep / "assets").mkdir()
        (ep / "assets" / "frame.jpg").write_bytes(b"jpg")
    return store


# ── addressing ──────────────────────────────────────────────────────────


def test_composite_ids_round_trip():
    assert LogStore.episode_execution_id("r", 7) == "r_ep0007"
    assert LogStore.split_execution_id("r_ep0007") == ("r", 7)
    assert LogStore.split_execution_id("run_with_ep_in_name") is None
    assert LogStore.split_execution_id("canvas-abc") is None
    assert LogStore.episode_subdir("/x", 12) == Path("/x/episodes/ep0012")


def test_locate_canvas_then_episode(tmp_path: Path):
    store = _layout(tmp_path)
    assert store.locate("c1") == LogDir(store.canvas_run_dir("c1"), "canvas")
    ep = store.locate("20261017_000000_ep0003")
    assert ep is not None and ep.source == "eval"
    assert ep.log_path == store.episode_dir("20261017_000000", 3) / LOG_FILE
    assert store.locate("20261017_000000_ep0001") is None  # no such episode
    assert store.locate("nope") is None


def test_asset_and_graph_paths(tmp_path: Path):
    store = _layout(tmp_path)
    assert store.asset_path("c1", "f.png") == store.canvas_run_dir("c1") / "assets" / "f.png"
    assert store.asset_path("20261017_000000_ep0000", "frame.jpg") is not None
    assert store.asset_path("c1", "missing.png") is None
    # A canvas run owns its graph; an episode shares the run-level one.
    assert store.read_graph("c1") == {"name": "canvas-graph"}
    assert store.read_graph("20261017_000000_ep0003") == {"name": "eval-graph"}
    assert store.read_graph("nope") is None


def test_resolve_run_and_log_files(tmp_path: Path):
    store = _layout(tmp_path)
    run_dir, source = store.resolve_run("20261017_000000")
    assert source == "eval"
    assert [p.parent.name for p in store.run_log_files(run_dir, source)] == ["ep0000", "ep0003"]
    canvas_dir, source = store.resolve_run("c1")
    assert source == "canvas" and store.run_log_files(canvas_dir, source) == [canvas_dir / LOG_FILE]
    assert store.resolve_run("nope") is None


# ── reading ─────────────────────────────────────────────────────────────


def test_read_entries_is_tolerant(tmp_path: Path):
    p = tmp_path / LOG_FILE
    p.write_text('{"a": 1}\n\nnot json\n[1, 2]\n{"b": 2}\n')
    assert LogStore.read_entries(p) == [{"a": 1}, {"b": 2}]
    assert LogStore.read_entries(tmp_path / "missing.jsonl") == []


def test_entries_by_execution_id(tmp_path: Path):
    store = _layout(tmp_path)
    assert [e["step"] for e in store.entries("c1")] == [1, 2]
    assert store.entries("nope") == []


def test_resource_samples_keep_the_tail(tmp_path: Path):
    store = _layout(tmp_path)
    run_dir = store.eval_run_dir("20261017_000000")
    _write_log(run_dir / "system.jsonl", [{"t": i} for i in range(5)])
    assert [r["t"] for r in store.resource_samples(run_dir, cap=2)] == [3, 4]
    assert store.resource_samples(store.canvas_run_dir("c1")) == []


# ── listing ─────────────────────────────────────────────────────────────


def test_list_executions_covers_both_families(tmp_path: Path):
    store = _layout(tmp_path)
    rows = store.list_executions()
    ids = {r["execution_id"]: r["source"] for r in rows}
    assert ids == {
        "c1": "canvas",
        "20261017_000000_ep0000": "eval",
        "20261017_000000_ep0003": "eval",
    }
    assert all({"size_bytes", "modified"} <= set(r) for r in rows)
    assert store.canvas_run_ids() == ["c1"]
    assert LogStore(tmp_path / "none", tmp_path / "none").list_executions() == []


# ── aggregation ─────────────────────────────────────────────────────────


def test_summarize():
    entries = [_entry(1, "a"), _entry(3, "b", error="boom"), _entry(2, "a")]
    s = LogStore.summarize("x", entries)
    assert s["total_firings"] == 3 and s["error_count"] == 1 and s["total_steps"] == 3
    assert s["node_types_fired"] == ["a", "b"]
    assert s["started_at"] == entries[0]["timestamp"] and s["ended_at"] == entries[-1]["timestamp"]
    assert LogStore.summarize("x", []) == {"execution_id": "x", "total_firings": 0}


def test_timing_aggregates_compute_queue_transport_and_usage(tmp_path: Path):
    p = tmp_path / LOG_FILE
    _write_log(
        p,
        [
            _entry(
                1,
                "llm",
                compute_ms=100.0,
                queue_wait_ms=2.0,
                inner_log=[
                    {"key": "usage", "value": {"total_tokens": 50, "usd_cost": 0.01, "calls": 1}}
                ],
            ),
            _entry(2, "llm", compute_ms=300.0, queue_wait_ms=4.0),
            _entry(3, "proxy", compute_ms=1.0, transport_ms=20.0, transfer_bytes=1000),
            _entry(4, "old"),  # no compute_ms → duration_ms counts
        ],
    )
    t = LogStore.timing([p])
    by = {row["node_type"]: row for row in t["node_timing"]}
    assert [row["node_type"] for row in t["node_timing"]] == ["llm", "old", "proxy"]
    assert by["llm"]["count"] == 2 and by["llm"]["compute_ms"]["total"] == 400.0
    assert by["llm"]["queue_wait_ms"]["mean"] == 3.0
    assert by["proxy"]["transport_ms"]["total"] == 20.0 and by["proxy"]["transfer_bytes"] == 1000
    assert by["old"]["compute_ms"]["total"] == 10.0 and by["old"]["queue_wait_ms"]["mean"] is None
    assert t["totals"] == {
        "firings": 4,
        "compute_ms": 411.0,
        "tokens": 50,
        "usd_cost": 0.01,
        "llm_calls": 1,
        "transport_ms": 20.0,
        "transfer_bytes": 1000,
    }
    empty = LogStore.timing([])
    assert empty["node_timing"] == [] and empty["totals"]["firings"] == 0
