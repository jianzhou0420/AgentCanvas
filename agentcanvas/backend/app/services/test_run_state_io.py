"""Round-trip + atomic-write tests for run_state_io."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .run_state_io import (
    atomic_write_json,
    initial_running_summary,
    is_done,
    mark_aborted,
    read_shared_urls,
    read_spec,
    read_summary,
    touch_done,
    write_shared_urls,
    write_spec,
)


def test_spec_round_trip(tmp_path: Path) -> None:
    spec = {"run_id": "abc", "eval": {"graph_name": "g"}, "graph": {"nodes": []}}
    write_spec(tmp_path, spec)
    assert read_spec(tmp_path) == spec


def test_shared_urls_round_trip(tmp_path: Path) -> None:
    urls = {"vlm_prismatic": "http://127.0.0.1:34453"}
    write_shared_urls(tmp_path, urls)
    assert read_shared_urls(tmp_path) == urls


def test_read_missing_returns_none_or_empty(tmp_path: Path) -> None:
    assert read_spec(tmp_path) is None
    assert read_summary(tmp_path) is None
    assert read_shared_urls(tmp_path) == {}
    assert is_done(tmp_path) is False


def test_done_flag(tmp_path: Path) -> None:
    assert not is_done(tmp_path)
    touch_done(tmp_path)
    assert is_done(tmp_path)
    # Idempotent — calling again is fine.
    touch_done(tmp_path)
    assert is_done(tmp_path)


def test_mark_aborted_promotes_running(tmp_path: Path) -> None:
    summary = initial_running_summary("rid", {"graph_name": "g"}, "2026-05-07T00:00:00")
    atomic_write_json(tmp_path / "summary.json", summary)
    mark_aborted(tmp_path)
    after = read_summary(tmp_path)
    assert after["status"] == "aborted"
    assert "[aborted]" in (after["error"] or "")


def test_mark_aborted_skips_terminal(tmp_path: Path) -> None:
    summary = initial_running_summary("rid", {}, "2026-05-07T00:00:00")
    summary["status"] = "completed"
    atomic_write_json(tmp_path / "summary.json", summary)
    mark_aborted(tmp_path)
    assert read_summary(tmp_path)["status"] == "completed"  # untouched


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
    # While they write, repeatedly parse the target.
    for _ in range(200):
        if target.exists():
            data = json.loads(target.read_text())  # raises if torn
            assert data in ({"x": payload_a["x"]}, {"y": payload_b["y"]})
    for t in threads:
        t.join()
