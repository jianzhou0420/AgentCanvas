"""Unit tests for env_vlnverse._metrics (numpy-only).

Includes a parity test against the NavHarness original ``calc_ndtw`` —
extracted from mllm_utils.py source via ast (that module imports cv2/torch/
matplotlib at module level, far too heavy to import here) — skipped when the
NavHarness checkout is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parent
_PKG = "envvlnverse_under_test"

NAVHARNESS_MLLM_UTILS = Path(
    "/home/xunyi/Desktop/Projects/NavHarness/navharness/model/basemodel/mllm/mllm_utils.py"
)


def _load(name: str):
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[_PKG] = pkg
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, PKG_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_metrics = _load("_metrics")
calc_ndtw = _metrics.calc_ndtw
EpisodeMetricsAccumulator = _metrics.EpisodeMetricsAccumulator


# ==================== calc_ndtw ====================


def test_ndtw_identical_trajs_is_one():
    traj = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 1.0, 0.0]]
    assert np.isclose(calc_ndtw(traj, traj), 1.0, atol=1e-12)


def test_ndtw_far_apart_near_zero():
    gt = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    pred = [[1000.0, 1000.0, 0.0], [1001.0, 1000.0, 0.0]]
    assert calc_ndtw(pred, gt) < 1e-6


def test_ndtw_empty_returns_zero():
    assert calc_ndtw([], [[0, 0, 0]]) == 0.0
    assert calc_ndtw([[0, 0, 0]], []) == 0.0


def test_ndtw_hand_computed_single_pair():
    # One pred point 1 m from one gt point: DTW = 1, ndtw = exp(-1/(3*1)).
    val = calc_ndtw([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    assert np.isclose(val, np.exp(-1.0 / 3.0), atol=1e-12)


def test_ndtw_asymmetric_lengths_in_range():
    pred = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    gt = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    v = calc_ndtw(pred, gt)
    assert 0.0 < v <= 1.0


def test_ndtw_uses_xy_only():
    gt = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    pred = [[0.0, 0.0, 99.0], [1.0, 0.0, -50.0]]  # z wildly off, xy identical
    assert np.isclose(calc_ndtw(pred, gt), 1.0, atol=1e-12)


def test_ndtw_parity_with_navharness_original():
    if not NAVHARNESS_MLLM_UTILS.exists():
        pytest.skip("NavHarness checkout not present")
    # Extract just calc_ndtw from the (heavy-importing) upstream module.
    tree = ast.parse(NAVHARNESS_MLLM_UTILS.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calc_ndtw"
    )
    ns: dict = {"np": np}
    exec(  # noqa: S102 — executing vendored-source function for parity check
        compile(ast.Module(body=[fn], type_ignores=[]), str(NAVHARNESS_MLLM_UTILS), "exec"),
        ns,
    )
    upstream = ns["calc_ndtw"]

    rng = np.random.default_rng(42)
    for _ in range(5):
        n, m = int(rng.integers(1, 15)), int(rng.integers(1, 15))
        pred = rng.uniform(-5, 5, size=(n, 3)).tolist()
        gt = rng.uniform(-5, 5, size=(m, 3)).tolist()
        assert abs(calc_ndtw(pred, gt) - upstream(pred, gt)) < 1e-6


# ==================== EpisodeMetricsAccumulator ====================


def test_success_with_stop():
    acc = EpisodeMetricsAccumulator(initial_distance=5.0)
    # Approach the goal over 3 moves + terminal STOP (pre-action dists).
    acc.record_step([0.0, 0.0, 1.2], 5.0, False, moved_distance=2.0)
    acc.record_step([2.0, 0.0, 1.2], 3.5, True, moved_distance=2.0)
    acc.record_step([4.0, 0.0, 1.2], 1.5, False, moved_distance=1.0)
    acc.record_step([4.8, 0.0, 1.2], 1.0, False, moved_distance=0.0)  # STOP
    acc.mark_stop()
    acc.set_end_reason("stop")
    m = acc.final_metrics("ep1", current_dist_to_goal=1.0, reference_path=[[5.0, 0.0, 1.2]])

    assert m["success"] == 1
    assert m["stop_success"] == 1
    assert m["called_stop"] is True
    assert m["oracle_success"] == 1
    assert m["steps_taken"] == 4
    assert m["moves"] == 3  # STOP not counted as a move
    assert m["path_length"] == 5.0
    assert np.isclose(m["spl"], 5.0 / max(5.0, 5.0))
    # first index with dist <= 3.0 is 2, total = 3
    assert np.isclose(m["success_efficiency"], 2.0 / 3.0)
    assert np.isclose(m["colrate"], 1.0 / 4.0)
    assert m["end_reason"] == "stop"
    assert m["distance_to_goal"] == 1.0
    assert m["initial_distance"] == 5.0
    assert 0.0 < m["ndtw"] <= 1.0


def test_failure_zeroes_success_family():
    acc = EpisodeMetricsAccumulator(initial_distance=12.0)
    acc.record_step([0.0, 0.0, 1.2], 12.0, False, moved_distance=2.0)
    acc.record_step([2.0, 0.0, 1.2], 10.0, False, moved_distance=2.0)
    m = acc.final_metrics("ep2", current_dist_to_goal=10.0, reference_path=[[12.0, 0.0, 1.2]])
    assert m["success"] == 0
    assert m["spl"] == 0.0
    assert m["stop_success"] == 0
    assert m["success_efficiency"] == 0
    assert m["oracle_success"] == 0
    assert m["colrate"] == 0.0


def test_oracle_success_without_final_success():
    acc = EpisodeMetricsAccumulator(initial_distance=6.0)
    acc.record_step([0.0, 0.0, 1.2], 6.0, False, moved_distance=4.0)
    acc.record_step([4.0, 0.0, 1.2], 2.0, False, moved_distance=4.0)  # walked past goal
    m = acc.final_metrics("ep3", current_dist_to_goal=6.5, reference_path=[])
    assert m["success"] == 0
    assert m["oracle_success"] == 1


def test_spl_penalizes_long_path():
    acc = EpisodeMetricsAccumulator(initial_distance=4.0)
    acc.record_step([0.0, 0.0, 1.2], 4.0, False, moved_distance=8.0)  # detour
    acc.record_step([1.0, 0.0, 1.2], 3.5, False, moved_distance=2.0)
    m = acc.final_metrics("ep4", current_dist_to_goal=2.0, reference_path=[])
    assert np.isclose(m["spl"], 4.0 / 10.0)


def test_empty_episode_fallbacks():
    acc = EpisodeMetricsAccumulator(initial_distance=2.0)
    m = acc.final_metrics("ep5", current_dist_to_goal=2.0, reference_path=[[0, 0, 0]])
    # dists falls back to [initial_distance]; already within 3.0 -> oracle hit
    assert m["success"] == 1
    assert m["oracle_success"] == 1
    assert m["steps_taken"] == 0
    assert m["colrate"] == 0.0
    assert m["ndtw"] == 0.0  # empty traj
    assert m["spl"] == 0.0  # path_length == 0 branch
    # upstream quirk: first == 0 -> `first and total` falsy -> efficiency 0
    assert m["success_efficiency"] == 0


def test_success_efficiency_first_step_quirk():
    # Success from step 0 (first == 0) keeps efficiency 0 — upstream-faithful.
    acc = EpisodeMetricsAccumulator(initial_distance=2.5)
    acc.record_step([0.0, 0.0, 1.2], 2.5, False, moved_distance=1.0)
    acc.record_step([1.0, 0.0, 1.2], 1.5, False, moved_distance=1.0)
    m = acc.final_metrics("ep6", current_dist_to_goal=1.0, reference_path=[])
    assert m["success"] == 1
    assert m["success_efficiency"] == 0


def test_ndtw_exception_swallowed_to_zero():
    acc = EpisodeMetricsAccumulator(initial_distance=5.0)
    acc.record_step([0.0, 0.0, 1.2], 5.0, False, moved_distance=1.0)
    # reference_path=None raises inside calc_ndtw -> caught -> 0.0
    m = acc.final_metrics("ep7", current_dist_to_goal=4.0, reference_path=None)
    assert m["ndtw"] == 0.0
