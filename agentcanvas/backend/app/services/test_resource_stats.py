"""ResourceStatsTracker unit tests.

Attribution and calibration math are exercised with injected fake
samples / process tables — no nvidia-smi, no /proc walk, no live registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from .resource_stats import (
    RAM_FLOOR_MB,
    SAFETY_MARGIN_MB,
    CalibrationStore,
    ResourceStatsTracker,
    _attribute_pid,
    _shared_server_pids,
    estimate_shared_mb,
    estimate_tree_mb,
)


class _FakeServer:
    def __init__(self, pid: int) -> None:
        self._pid = pid


class _FakeNodeset:
    def __init__(self, source_file: str) -> None:
        self._source_file = source_file


class _FakeRegistry:
    def __init__(self, servers: dict[str, int], local_loaded: set[str] | None = None) -> None:
        self._auto_servers = {k: _FakeServer(pid) for k, pid in servers.items()}
        self._discovered_nodesets: dict[str, _FakeNodeset] = {}
        self._local_loaded = local_loaded or set()

    def is_nodeset_loaded(self, name: str) -> bool:
        return name in self._local_loaded or name in self._auto_servers

    def get_server_url(self, name: str) -> str | None:
        return f"http://x:1/{name}" if name in self._auto_servers else None


def _sample(
    gpu_procs: list[dict],
    used_total: int = 0,
    mem_used: int = 8192,
    mem_total: int = 32768,
) -> dict:
    return {
        "ts": "2026-07-05T00:00:00",
        "gpus": [{"mem_used_mb": used_total, "mem_total_mb": 24576}],
        "gpu_procs": gpu_procs,
        "mem_used_mb": mem_used,
        "mem_total_mb": mem_total,
    }


# ── attribution helpers ──


def test_attribute_pid_walks_ancestor_chain() -> None:
    # 500 (worker) → 400 (auto_host, setsid'd but PPID intact) → 300 (job root)
    parents = {500: 400, 400: 300, 300: 1}
    assert _attribute_pid(500, {300: "run_a"}, {}, parents) == ("job", "run_a")
    assert _attribute_pid(500, {999: "run_a"}, {}, parents) is None


def test_attribute_pid_reaches_server_ancestor() -> None:
    # A shared server's forked child attributes to the server, not external.
    parents = {201: 200, 200: 1}
    assert _attribute_pid(201, {}, {200: "model_ram"}, parents) == ("shared", "model_ram")


def test_attribute_pid_survives_cycles() -> None:
    parents = {10: 20, 20: 10}  # corrupt table must not hang
    assert _attribute_pid(10, {300: "r"}, {}, parents) is None


def test_shared_server_pids_collapses_tags() -> None:
    reg = _FakeRegistry({"env_habitat#0": 111, "env_habitat#1": 112, "model_ram": 113})
    assert _shared_server_pids(reg) == {
        111: "env_habitat",
        112: "env_habitat",
        113: "model_ram",
    }
    assert _shared_server_pids(None) == {}


# ── observe → window → sediment flow ──


def test_observe_attributes_and_sediments(tmp_path: Path) -> None:
    calib = tmp_path / "resource_calibration.json"
    tracker = ResourceStatsTracker(calib)
    registry = _FakeRegistry({"model_ram": 113})

    tracker.note_job_started(
        "run_a", graph_name="smartway_ce", worker_count=4, shared_nodesets=("model_ram",)
    )

    sample = _sample(
        gpu_procs=[
            {"pid": 113, "mem_mb": 10000},  # shared singleton
            {"pid": 500, "mem_mb": 3000},  # descendant of job root 300
            {"pid": 999, "mem_mb": 400},  # external (no ancestor match)
        ],
        used_total=13934,
        mem_used=9000,
    )
    # pid → (ppid, rss_mb): serves both ancestry and the RAM usage source.
    table = {500: (300, 1200), 300: (1, 800), 113: (1, 4000), 999: (1, 300)}
    with (
        patch("app.services.resource_stats._latest_sample", return_value=sample),
        patch("app.services.resource_stats._proc_table", return_value=table),
    ):
        tracker.observe(jobs={"run_a": 300}, registry=registry)

    snap = tracker.snapshot()
    vram = snap["resources"]["vram"]
    assert vram["shared_mb"] == {"model_ram": 10000}
    assert vram["jobs_mb"] == {"run_a": 3000}
    assert vram["external_mb"] == 13934 - 13000
    ram = snap["resources"]["ram"]
    assert ram["shared_mb"] == {"model_ram": 4000}
    assert ram["jobs_mb"] == {"run_a": 1200 + 800}  # worker + job root RSS
    assert ram["external_mb"] == 9000 - 4000 - 2000

    with patch("app.services.resource_stats._source_hash", return_value="hash1"):
        tracker.note_job_finished("run_a", registry)

    data = json.loads(calib.read_text())
    ns = data["nodesets"]["model_ram"]
    assert ns["vram"]["peak_mb_max"] == 10000
    assert ns["vram"]["worker_obs"] == [[4, 10000]]
    assert ns["ram"]["peak_mb_max"] == 4000
    graph = data["graphs"]["smartway_ce"]["by_worker_count"]["4"]
    assert graph["vram"]["tree_peak_mb_max"] == 3000
    assert graph["ram"]["tree_peak_mb_max"] == 2000
    assert graph["n_obs"] == 1


def test_observe_keeps_peak_not_last(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    tracker.note_job_started("r", graph_name="g", worker_count=1, shared_nodesets=())
    for gpu_mem, rss in ((5000, 2500), (2000, 1000)):  # load spike then settle
        sample = _sample([{"pid": 500, "mem_mb": gpu_mem}], used_total=gpu_mem)
        table = {500: (300, rss), 300: (1, 0)}
        with (
            patch("app.services.resource_stats._latest_sample", return_value=sample),
            patch("app.services.resource_stats._proc_table", return_value=table),
        ):
            tracker.observe(jobs={"r": 300}, registry=None)
    tracker.note_job_finished("r", None)
    data = json.loads((tmp_path / "c.json").read_text())
    entry = data["graphs"]["g"]["by_worker_count"]["1"]
    assert entry["vram"]["tree_peak_mb_max"] == 5000
    assert entry["ram"]["tree_peak_mb_max"] == 2500


def test_observe_without_sampler_is_noop(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    with patch("app.services.resource_stats._latest_sample", return_value=None):
        tracker.observe(jobs={}, registry=None)  # must not raise
    assert tracker.snapshot() == {}


def test_finish_unknown_run_is_noop(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    tracker.note_job_finished("never-started", None)  # must not raise
    assert not (tmp_path / "c.json").exists()


def test_zero_vram_resident_server_is_calibrated_at_zero(tmp_path: Path) -> None:
    # A CPU-only shared server (in the registry, absent from gpu_procs)
    # must sediment a 0 VRAM observation — "measured 0" ≠ "never observed"
    # — while its RAM side records the real RSS.
    calib = tmp_path / "c.json"
    tracker = ResourceStatsTracker(calib)
    registry = _FakeRegistry({"env_adapter": 77})
    tracker.note_job_started("r", graph_name="g", worker_count=1, shared_nodesets=("env_adapter",))
    table = {300: (1, 100), 77: (1, 900)}
    with (
        patch("app.services.resource_stats._latest_sample", return_value=_sample([], 0)),
        patch("app.services.resource_stats._proc_table", return_value=table),
    ):
        tracker.observe(jobs={"r": 300}, registry=registry)
    with patch("app.services.resource_stats._source_hash", return_value="h"):
        tracker.note_job_finished("r", registry)
    data = json.loads(calib.read_text())
    assert data["nodesets"]["env_adapter"]["vram"]["peak_mb_max"] == 0
    assert data["nodesets"]["env_adapter"]["ram"]["peak_mb_max"] == 900
    assert data["graphs"]["g"]["by_worker_count"]["1"]["vram"]["tree_peak_mb_max"] == 0


def test_local_inprocess_shared_nodeset_calibrates_at_zero(tmp_path: Path) -> None:
    # env_adapter case: parallelism="shared" but loaded in-process (no
    # server PID). Must sediment 0 on every resource so it never blocks
    # estimate coverage.
    calib = tmp_path / "c.json"
    tracker = ResourceStatsTracker(calib)
    registry = _FakeRegistry({}, local_loaded={"env_adapter"})
    tracker.note_job_started("r", graph_name="g", worker_count=1, shared_nodesets=("env_adapter",))
    with (
        patch("app.services.resource_stats._latest_sample", return_value=_sample([], 0)),
        patch("app.services.resource_stats._proc_table", return_value={300: (1, 50)}),
    ):
        tracker.observe(jobs={"r": 300}, registry=registry)
    with patch("app.services.resource_stats._source_hash", return_value="h"):
        tracker.note_job_finished("r", registry)
    data = json.loads(calib.read_text())
    assert data["nodesets"]["env_adapter"]["vram"]["peak_mb_max"] == 0
    assert data["nodesets"]["env_adapter"]["ram"]["peak_mb_max"] == 0


def test_no_samples_window_sediments_nothing(tmp_path: Path) -> None:
    # Sampler down for the whole run → no blind zeros in calibration.
    calib = tmp_path / "c.json"
    tracker = ResourceStatsTracker(calib)
    tracker.note_job_started("r", graph_name="g", worker_count=1, shared_nodesets=())
    with patch("app.services.resource_stats._latest_sample", return_value=None):
        tracker.observe(jobs={"r": 300}, registry=None)
    tracker.note_job_finished("r", None)
    assert not calib.exists()


# ── measured free ──


def test_measured_free_per_resource(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    sample = _sample([], used_total=4000, mem_used=8192, mem_total=32768)
    with patch("app.services.resource_stats._latest_sample", return_value=sample):
        assert tracker.measured_free_mb("vram") == 24576 - 4000
        assert tracker.measured_free_mb("ram") == 32768 - 8192 - RAM_FLOOR_MB
    # No GPU rows → vram unmeasurable, ram still measured.
    cpu_only = dict(sample, gpus=[])
    with patch("app.services.resource_stats._latest_sample", return_value=cpu_only):
        assert tracker.measured_free_mb("vram") is None
        assert tracker.measured_free_mb("ram") == 32768 - 8192 - RAM_FLOOR_MB


# ── calibration store ──


def test_store_ewma_and_hash_invalidation(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "c.json")
    store.update_nodeset("ns", peaks={"vram": 1000, "ram": 500}, workers=1, source_hash="h1")
    store.update_nodeset("ns", peaks={"vram": 2000, "ram": 700}, workers=2, source_hash="h1")
    entry = store.data["nodesets"]["ns"]
    assert entry["n_obs"] == 2
    assert entry["vram"]["peak_mb_max"] == 2000
    assert entry["ram"]["peak_mb_max"] == 700
    assert 1000 < entry["vram"]["peak_mb_ewma"] < 2000  # EWMA, not last-write

    # Source hash change → statistics reset on EVERY resource.
    store.update_nodeset("ns", peaks={"vram": 500}, workers=1, source_hash="h2")
    entry = store.data["nodesets"]["ns"]
    assert entry["n_obs"] == 1
    assert entry["vram"]["peak_mb_max"] == 500
    assert entry["source_hash"] == "h2"
    assert "ram" not in entry


def test_store_migrates_v1_vram_only(tmp_path: Path) -> None:
    # A pre-rename vram_calibration.json (v1, flat fields) is adopted as
    # the VRAM slice; the RAM slice starts empty.
    legacy = tmp_path / "vram_calibration.json"
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "nodesets": {
                    "model_ram": {
                        "source_hash": "h1",
                        "n_obs": 3,
                        "peak_mb_ewma": 9800.0,
                        "peak_mb_max": 10000,
                        "worker_obs": [[4, 10000]],
                    }
                },
                "graphs": {
                    "g": {
                        "by_worker_count": {
                            "4": {
                                "n_obs": 3,
                                "tree_peak_mb_ewma": 2900.0,
                                "tree_peak_mb_max": 3000,
                            }
                        }
                    }
                },
            }
        )
    )
    store = CalibrationStore(tmp_path / "resource_calibration.json")
    ns = store.data["nodesets"]["model_ram"]
    assert ns["source_hash"] == "h1"
    assert ns["vram"]["peak_mb_max"] == 10000
    assert "ram" not in ns
    graph = store.data["graphs"]["g"]["by_worker_count"]["4"]
    assert graph["vram"]["tree_peak_mb_max"] == 3000


def test_store_survives_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{not json")
    store = CalibrationStore(path)
    assert store.data["nodesets"] == {}
    store.update_graph("g", worker_count=2, tree_peaks={"vram": 100, "ram": 50})
    store.save()
    assert json.loads(path.read_text())["graphs"]["g"]["by_worker_count"]["2"]["ram"]


# ── estimator (pure functions operate on one resource's flattened view) ──


def _graph_entry(points: dict[int, int]) -> dict:
    return {
        "by_worker_count": {
            str(wc): {"tree_peak_mb_max": peak, "n_obs": 1} for wc, peak in points.items()
        }
    }


def test_tree_estimate_bases() -> None:
    assert estimate_tree_mb(None, 1) is None
    # exact worker_count → measured
    assert estimate_tree_mb(_graph_entry({2: 600}), 2) == (600, "measured")
    # single point → proportional scaling
    assert estimate_tree_mb(_graph_entry({1: 300}), 4) == (1200, "scaled")
    # two points → linear fit: 300 = a+b, 900 = a+3b → a=0, b=300 → wc=4 → 1200
    assert estimate_tree_mb(_graph_entry({1: 300, 3: 900}), 4) == (1200, "fitted")


def test_tree_fit_clamps_negative_slope() -> None:
    # Decreasing observations must not extrapolate downward.
    mb, basis = estimate_tree_mb(_graph_entry({1: 900, 3: 300}), 10)
    assert basis == "fitted"
    assert mb >= 0


def test_shared_estimate_hash_and_loaded() -> None:
    entry = {"source_hash": "h1", "peak_mb_max": 2000, "worker_obs": [[1, 2000]]}
    assert estimate_shared_mb(None, 1, False, None) is None
    # stale hash → uncalibrated
    assert estimate_shared_mb(entry, 1, False, "h2") is None
    # not loaded → full base
    assert estimate_shared_mb(entry, 1, False, "h1") == (2000, "measured")
    # loaded → residency already measured; marginal growth only
    assert estimate_shared_mb(entry, 1, True, "h1") == (0, "measured, loaded")


def test_shared_estimate_worker_slope() -> None:
    entry = {
        "source_hash": "h1",
        "peak_mb_max": 2200,
        "worker_obs": [[1, 2000], [3, 2200]],  # slope 100/worker
    }
    mb, basis = estimate_shared_mb(entry, 5, False, "h1")
    assert basis == "fitted"
    assert mb == 2400  # 1900 + 100x5
    # loaded: only the growth beyond the observed base
    mb_loaded, _ = estimate_shared_mb(entry, 5, True, "h1")
    assert mb_loaded == 200


def test_shared_estimate_hint_fallback() -> None:
    entry = {"source_hash": "h1", "peak_mb_max": 2000, "worker_obs": [[1, 2000]]}
    # No calibration at all → hint prices it.
    assert estimate_shared_mb(None, 1, False, None, hint_mb=600) == (600, "hint")
    # Stale hash retires the measurement, but the preset persists.
    assert estimate_shared_mb(entry, 1, False, "h2", hint_mb=600) == (600, "hint")
    # Measurement wins over hint when valid.
    assert estimate_shared_mb(entry, 1, False, "h1", hint_mb=600) == (2000, "measured")
    # Loaded residency is already measured → preset marginal is 0.
    assert estimate_shared_mb(None, 1, True, None, hint_mb=600) == (0, "hint, loaded")


# ── tracker.estimate (per-resource assembly) ──


def test_estimate_full_coverage_and_max_workers(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    tracker._store.update_graph("g", worker_count=1, tree_peaks={"vram": 1000, "ram": 2000})
    tracker._store.update_nodeset(
        "ns", peaks={"vram": 2000, "ram": 500}, workers=1, source_hash="h1"
    )

    free_sample = {
        "gpus": [{"mem_used_mb": 4000, "mem_total_mb": 12000}],
        "mem_used_mb": 8192,
        "mem_total_mb": 32768,
    }
    with patch("app.services.resource_stats._latest_sample", return_value=free_sample):
        result = tracker.estimate(
            "g", 2, [{"name": "ns", "loaded": False, "source_hash": "h1"}]
        )
    vram = result["resources"]["vram"]
    # tree scaled 1000x2 + singleton 2000 + margin
    assert vram["estimate_mb"] == 2000 + 2000 + SAFETY_MARGIN_MB["vram"]
    assert vram["uncalibrated"] == []
    assert vram["measured_free_mb"] == 8000
    # free 8000: 1000w + 2000 + 1500 ≤ 8000 → w ≤ 4.5 → 4
    assert vram["max_workers"] == 4
    ram = result["resources"]["ram"]
    assert ram["estimate_mb"] == 4000 + 500 + SAFETY_MARGIN_MB["ram"]
    assert ram["measured_free_mb"] == 32768 - 8192 - RAM_FLOOR_MB  # 20480
    # 2000w + 500 + 1024 ≤ 20480 → w ≤ 9.47 → 9
    assert ram["max_workers"] == 9
    # combined = min over measurable resources
    assert result["max_workers"] == 4


def test_estimate_hint_completes_coverage(tmp_path: Path) -> None:
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    tracker._store.update_graph("g", worker_count=1, tree_peaks={"vram": 1000, "ram": 1500})
    infos = [
        {
            "name": "sam",
            "loaded": False,
            "source_hash": None,
            "hints": {"vram": 600, "ram": 300},
        }
    ]
    with patch("app.services.resource_stats._latest_sample", return_value=None):
        result = tracker.estimate("g", 1, infos)
    vram = result["resources"]["vram"]
    assert vram["estimate_mb"] == 1000 + 600 + SAFETY_MARGIN_MB["vram"]
    assert vram["used_hint"] is True
    assert vram["breakdown"]["shared"]["sam"] == {"mb": 600, "basis": "hint"}
    ram = result["resources"]["ram"]
    assert ram["estimate_mb"] == 1500 + 300 + SAFETY_MARGIN_MB["ram"]


def test_estimate_partial_coverage_is_null_per_resource(tmp_path: Path) -> None:
    # Migrated-v1 shape: vram calibrated, ram never seen → the ram side
    # must stay null (honesty rule is per resource).
    tracker = ResourceStatsTracker(tmp_path / "c.json")
    tracker._store.update_graph("g", worker_count=1, tree_peaks={"vram": 1000})
    with patch("app.services.resource_stats._latest_sample", return_value=None):
        result = tracker.estimate(
            "g", 1, [{"name": "never_seen", "loaded": False, "source_hash": None}]
        )
    vram = result["resources"]["vram"]
    assert vram["estimate_mb"] is None  # a missing singleton could be 10 GB
    assert vram["known_mb"] == 1000
    assert vram["uncalibrated"] == ["never_seen"]
    ram = result["resources"]["ram"]
    assert ram["estimate_mb"] is None
    assert ram["uncalibrated"] == ["graph-tree", "never_seen"]
    assert result["max_workers"] is None
