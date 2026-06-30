"""Unit tests for the orphaned-canvas-guard reaper decision.

Covers ``state.canvas_guard_orphan_decision`` — the pure logic that decides
when a leaked canvas ``ExecutionGuard`` (frontend closed a paused run without
``/run/stop``) should be reaped so eval admission is no longer starved.
"""

from __future__ import annotations

from .state import CANVAS_ORPHAN_GRACE_SEC, ExecutionMode, canvas_guard_orphan_decision

CANVAS = ExecutionMode.canvas.value
EVAL = ExecutionMode.eval.value
IDLE = ExecutionMode.idle.value


def test_not_canvas_mode_never_reaps():
    # idle/eval guard is irrelevant to this reaper; timer stays cleared
    assert canvas_guard_orphan_decision(IDLE, 0, None, 100.0) == (False, None)
    assert canvas_guard_orphan_decision(EVAL, 0, 50.0, 100.0) == (False, None)


def test_canvas_with_clients_resets_timer():
    # a connected WS client means someone is watching → not orphaned
    assert canvas_guard_orphan_decision(CANVAS, 1, None, 100.0) == (False, None)
    # even if a timer was running, reconnect clears it
    assert canvas_guard_orphan_decision(CANVAS, 2, 50.0, 100.0) == (False, None)


def test_first_orphan_tick_starts_timer():
    assert canvas_guard_orphan_decision(CANVAS, 0, None, 100.0) == (False, 100.0)


def test_within_grace_keeps_timer():
    now = 100.0 + CANVAS_ORPHAN_GRACE_SEC - 1.0
    assert canvas_guard_orphan_decision(CANVAS, 0, 100.0, now) == (False, 100.0)


def test_past_grace_reaps_and_resets():
    now = 100.0 + CANVAS_ORPHAN_GRACE_SEC + 1.0
    assert canvas_guard_orphan_decision(CANVAS, 0, 100.0, now) == (True, None)


def test_exactly_grace_boundary_reaps():
    # boundary is inclusive (>=) so the run isn't held one tick longer
    now = 100.0 + CANVAS_ORPHAN_GRACE_SEC
    assert canvas_guard_orphan_decision(CANVAS, 0, 100.0, now) == (True, None)


def test_reconnect_then_disconnect_restarts_clock():
    # disconnect starts timer
    reap, since = canvas_guard_orphan_decision(CANVAS, 0, None, 10.0)
    assert (reap, since) == (False, 10.0)
    # reconnect clears it
    reap, since = canvas_guard_orphan_decision(CANVAS, 1, since, 12.0)
    assert (reap, since) == (False, None)
    # disconnect again starts a fresh clock from the later time, not the old one
    reap, since = canvas_guard_orphan_decision(CANVAS, 0, since, 40.0)
    assert (reap, since) == (False, 40.0)
    # still within grace measured from 40.0, not 10.0
    reap, since = canvas_guard_orphan_decision(CANVAS, 0, since, 40.0 + CANVAS_ORPHAN_GRACE_SEC - 1)
    assert reap is False
