"""V0 — deterministic guards. Zero model calls, zero env steps.

Fired by the pre_action hook at the tool boundary. A tripped guard BLOCKS the
action (it is not executed) and returns forced feedback; consecutive trips
escalate (the wrapper preempts the sub-session) — never wait for the budget to
burn down (the "10% hit the 401-step cap" signature).

Both guards are coordinate-free: stall is frame-hash similarity, no-progress
is state-block bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eharness.frames import FrameLog, hamming


@dataclass
class GuardConfig:
    stall_window: int = 2        # two identical post-move views = you moved and the world did not respond — flag AT ONCE
    stall_max_dist: int = 5      # …(hamming ≤ this) after move actions = stall
    no_progress_steps: int = 60  # env steps without advance_subgoal → warn
    escalate_after: int = 3      # consecutive guard trips → preempt sub-session


@dataclass
class GuardState:
    move_hashes: list[int] = field(default_factory=list)  # views after moves
    steps_at_last_advance: int = 0
    consecutive_trips: int = 0
    trips_total: int = 0


class Guards:
    def __init__(self, frames: FrameLog, config: GuardConfig | None = None) -> None:
        self.frames = frames
        self.cfg = config or GuardConfig()
        self.state = GuardState()

    # ── bookkeeping (post_observation) ──

    def note_move_view(self, frame_hash: int) -> None:
        # a genuinely CHANGED view = the agent got unstuck — the escalation
        # counter resets on real progress, never on merely passing the guard
        if (self.state.move_hashes
                and hamming(frame_hash, self.state.move_hashes[-1])
                > self.cfg.stall_max_dist):
            self.state.consecutive_trips = 0
        self.state.move_hashes.append(frame_hash)
        del self.state.move_hashes[:-self.cfg.stall_window]

    def note_advance(self, steps_taken: int) -> None:
        self.state.steps_at_last_advance = steps_taken

    # ── checks (pre_action) — return a message to inject, or None ──

    def check_stall(self, *, corrective: bool = False) -> str | None:
        """corrective=True → the incoming action already changes something
        (starts with a turn / different waypoint): let it EXECUTE — a guard
        that blocks the cure is a deadlock generator (found live 2026-08-05:
        the wall-stuck loop where every corrective move got vetoed against
        stale history). On a trip the streak is cleared: one stall, one
        warning, then the next action runs and is judged by its outcome."""
        if corrective:
            return None
        h = self.state.move_hashes
        if len(h) < self.cfg.stall_window:
            return None
        if all(hamming(h[i], h[i + 1]) <= self.cfg.stall_max_dist
               for i in range(len(h) - 1)):
            self.state.move_hashes.clear()
            return (f"GUARD(stall): your last {self.cfg.stall_window} moves "
                    "left the view essentially unchanged — you moved and the "
                    "world did not respond. You are blocked. ")
        return None

    def check_no_progress(self, steps_taken: int) -> str | None:
        gone = steps_taken - self.state.steps_at_last_advance
        if gone >= self.cfg.no_progress_steps:
            self.state.steps_at_last_advance = steps_taken  # rearm, don't spam
            return (f"GUARD(no-progress): {gone} env steps since the last "
                    "sub-instruction advance. Re-read the current "
                    "sub-instruction; if this area is exhausted, record it as "
                    "a dead end and go back to your last confirmed point.")
        return None

    def trip(self) -> bool:
        """Record a trip; True = escalate (preempt) now."""
        self.state.consecutive_trips += 1
        self.state.trips_total += 1
        return self.state.consecutive_trips >= self.cfg.escalate_after
