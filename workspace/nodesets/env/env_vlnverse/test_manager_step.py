"""Manager-level step contract tests over a stubbed backend (no Isaac).

Pins what the pure-logic tests cannot see (code-review 2026-07-20):
  - the discrete action sign contract (2=LEFT raises heading by +15°) — a
    flipped sign would silently mirror every discrete-agent trajectory;
  - pre-action accumulator recording (``dists[0] == initial_distance``);
  - STOP semantics: terminated (not truncated) + upstream end_reason string.

Needs the real package ``__init__`` (``app.components``), so it runs only
when the backend is importable (``PYTHONPATH=<repo>/agentcanvas/backend``)
— the same requirement pytest collection already has here; the module-level
importorskip just makes a lean-env skip explicit rather than a collect error.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("app.components", reason="needs backend on PYTHONPATH")

import env_vlnverse as pkg  # noqa: E402  (pytest basedir = workspace/nodesets/env)


class _FakeView:
    """Duck-typed RenderedView — the manager only reads .rgb/.depth."""

    def __init__(self) -> None:
        self.rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.depth = np.ones((8, 8), dtype=np.float32)


class _FakeBackend:
    """SimBackend stand-in: fixed frames, no process, records nothing."""

    def start(self, sim_cfg):  # pragma: no cover — not used by these tests
        pass

    def load_scene(self, scene_id, scene_usd_path):
        raise AssertionError("stub tests must never trigger a scene load")

    def capture_panoramic(self, position, heading, num_views=12, warmup_steps=10):
        return [_FakeView() for _ in range(max(1, int(num_views)))]

    def shutdown(self):
        pass


def _episode() -> dict:
    return pkg._normalize_episode(
        {
            "episode_id": "synthetic_0",
            "scene_id": "vlnverse/synthetic_scene",
            "scan": "synthetic_scene",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [1.0, 0.0, 0.0, 0.0],  # identity → heading 0
            "instruction": {"instruction_text": "go forward"},
            "goals": {"position": [2.0, 0.0, 0.0], "radius": 3.0},
            "reference_path": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        }
    )


def _manager() -> "pkg.VLNVerseEnvManager":
    """Fresh (non-singleton) manager seated on the synthetic episode, with
    the scene marked current so the lazy-ensure path never touches disk."""
    mgr = pkg.VLNVerseEnvManager()
    mgr._backend = _FakeBackend()
    mgr._episodes = [_episode()]
    mgr._dataset, mgr._split = "fine", "val_unseen"
    mgr._kin.use_occupancy_collision = False  # no freemap in stub land
    mgr._seat_episode_unlocked(0, load_scene=False)
    mgr._current_scene = "synthetic_scene"
    return mgr


def test_discrete_turn_signs():
    mgr = _manager()
    h0 = mgr._kin.agent_heading
    r = mgr.step(2)  # LEFT
    assert "error" not in r
    assert mgr._kin.agent_heading - h0 == pytest.approx(np.radians(15.0))
    mgr.step(3)  # RIGHT undoes it
    assert mgr._kin.agent_heading == pytest.approx(h0)


def test_forward_records_pre_action_distance():
    mgr = _manager()
    initial = mgr._acc.initial_distance
    assert initial == pytest.approx(2.0)  # goal 2 m ahead, z pinned equally
    r = mgr.step(1)  # FWD 0.25 m
    assert "error" not in r and not r["terminated"]
    # Upstream _record_pre semantics: dists[0] is the PRE-move distance.
    assert mgr._acc.dists[0] == pytest.approx(initial)
    assert mgr._kin.current_dist_to_goal() == pytest.approx(initial - 0.25)


def test_stop_is_terminated_not_truncated():
    mgr = _manager()
    r = mgr.step(0)
    assert r["terminated"] is True
    assert r["truncated"] is False
    assert mgr._acc.end_reason == "stop_called"
    assert r["metrics"]["called_stop"] is True
