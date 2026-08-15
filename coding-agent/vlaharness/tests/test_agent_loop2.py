"""State-machine tests for the v2 loop — fake toolset, scripted model replies.

These exist because the expensive part of this project is a 100-episode run, and
every bug that has cost one so far was a control-flow bug reachable from a
fake robot in under a second: a failed call read as consent, a motor command
sent down the text channel, a fallback that replayed the whole mission, an
unstick manoeuvre that re-triggered itself.

Run:  python -m pytest vlaharness/tests/test_agent_loop2.py -q
      (or plain `python vlaharness/tests/test_agent_loop2.py` for a summary)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vlaharness import agent_loop2 as AL  # noqa: E402
from vlaharness.agent_judge2 import (UNRESOLVED_STOP, VERIFIED_FAILURE,  # noqa: E402
                                     VERIFIED_SUCCESS, adjudicate)
from vlaharness.agent_state2 import EvidenceState  # noqa: E402

BASE = {"sub_instruction": "x", "stop_reason": "policy_stop", "steps_used": 20,
        "net_displacement_m": 3.0, "steps_taken_total": 20, "steps_remaining": 300,
        "_actions": ["MOVE_FORWARD"] * 20, "_frames_leg": ["f"], "_leg_frame_steps": [0],
        "_frame_hash": 1}


class FakeTools:
    """A robot that does exactly what the test says and records what was asked."""

    step_budget = 400
    max_steps = 50

    def __init__(self, out=None, hashes=None):
        self.episode_over = False
        self.end_reason = None
        self.out = dict(out or BASE)
        self.hashes = list(hashes or [])
        self.log: list[tuple] = []
        self.policy_env_steps = self.harness_drive_steps = 0
        self.verification_steps = self.render_only_observations = 0
        self._d = 5.0

    def current_views(self):
        return {"_frame_front": "f", "_frame_left": "l",
                "_frame_back": "b", "_frame_right": "r"}

    def look_around(self, n):
        self.log.append(("look", n))
        return {"_sweep": ["s"] * n, "_sweep_deg": list(range(0, 360, 360 // n))}

    def probe_distance(self):
        return self._d

    def verifying(self, on):
        self.log.append(("verifying", on))

    def _bundle(self):
        o = dict(self.out)
        if self.hashes:
            o["_frame_hash"] = self.hashes.pop(0)
        return o

    def execute(self, s):
        self.log.append(("execute", s))
        return self._bundle()

    def drive(self, a):
        self.log.append(("drive", list(a)))
        return self._bundle()

    def finish(self):
        self.log.append(("finish",))
        self.episode_over = True
        return {"stopped": True}


def run(boot, reviews, verifies, *, out=None, hashes=None, ablate=frozenset(),
        max_turns=6):
    """Drive one episode with scripted replies. Returns (tools, trace)."""
    it_r, it_v = iter(reviews), iter(verifies)
    AL.bootstrap = lambda *a, **k: (boot, "raw", "", {})
    AL.review = lambda *a, **k: (next(it_r, {"decision": "finish",
                                             "receipt": {"claim": "done"}}), "raw", "", {})
    AL.verify = lambda *a, **k: (next(it_v, {"close_enough": True, "relation_holds": True,
                                             "target_visible": True}), "raw", "", {})
    t = FakeTools(out, hashes)
    return t, AL.run_agent_episode2(t, "walk to the sink", max_turns=max_turns, ablate=ablate)


BOOT = {"terminate": "the sink", "clauses": ["cross the room", "stop at the sink"],
        "state": {}, "next_instruction": "walk to the sink"}


# ── termination is typed, and a forced stop is not a pass ────────────

def test_verified_success():
    _t, tr = run(BOOT, [{"decision": "finish", "receipt": {"claim": "at the sink"}}],
                 [{"close_enough": True, "relation_holds": True, "target_visible": True}])
    assert tr["outcome"] == VERIFIED_SUCCESS
    assert tr["forced_finish"] is False


def test_veto_allowance_is_unresolved_not_success():
    """The old loop finished `forced=True` here and filed it beside real passes."""
    _t, tr = run(BOOT, [{"decision": "finish", "receipt": {"claim": "maybe"}}],
                 [{"close_enough": False, "relation_holds": False, "target_visible": False,
                   "next_instruction": "walk further to the sink"}] * 4)
    assert tr["outcome"] == UNRESOLVED_STOP
    assert tr["counts"]["unverified_stops"] == 1


def test_failed_verification_never_passes():
    """`None` from the verifier must not become success. This was fail-open."""
    _t, tr = run(BOOT, [{"decision": "finish", "receipt": {"claim": "?"}}], [None])
    assert tr["outcome"] == UNRESOLVED_STOP
    assert tr["counts"]["parse_failures"] >= 1


def test_failed_review_does_not_open_the_arrival_gate():
    t, _tr = run(BOOT, [None, {"decision": "finish", "receipt": {"claim": "ok"}}],
                 [{"close_enough": True, "relation_holds": True}])
    assert sum(1 for e in t.log if e[0] == "execute") >= 2, "a failed review must not finish"


# ── the arrival gate needs the mission to be nearly over ─────────────

def test_arrival_gate_requires_near_goal_and_terminal_clause():
    t, _ = run(BOOT, [{"decision": "continue", "receipt": {"claim": "in the room"},
                       "next_instruction": "keep going to the sink",
                       "state": {"near_goal": False}}] * 3,
               [{"close_enough": True, "relation_holds": True}])
    assert sum(1 for e in t.log if e[0] == "execute") >= 3, "gate fired too early"


def test_arrival_gate_fires_on_terminal_clause():
    t, tr = run(BOOT, [{"decision": "continue", "receipt": {"claim": "at the sink"},
                        "next_instruction": "step to the sink",
                        "state": {"near_goal": True,
                                  "progress": [{"clause": 0, "status": "done"},
                                               {"clause": 1, "status": "done"}]}}],
                [{"close_enough": True, "relation_holds": True}])
    assert tr["outcome"] == VERIFIED_SUCCESS
    assert sum(1 for e in t.log if e[0] == "execute") == 1


# ── no silent mission replay ─────────────────────────────────────────

def test_missing_instruction_never_replays_the_mission():
    """371 dispatches in the no-drive ablation were this bug."""
    t, tr = run(BOOT, [{"decision": "continue", "receipt": {"claim": "x"},
                        "next_instruction": None, "state": {}}] * 6,
                [{"close_enough": True, "relation_holds": True}])
    sent = [e[1] for e in t.log if e[0] == "execute"]
    assert sent.count("walk to the sink") <= 1, f"mission replayed: {sent}"
    assert tr.get("ended_dispatching")


def test_motor_command_never_enters_the_text_channel():
    t, _ = run(BOOT, [{"decision": "continue", "receipt": {"claim": "x"},
                       "next_instruction": "MOVE_FORWARD", "state": {}}] * 3,
               [{"close_enough": True, "relation_holds": True}])
    assert "MOVE_FORWARD" not in [e[1] for e in t.log if e[0] == "execute"]


# ── wedged means not moving AND not seeing anything new ──────────────

def test_turning_into_a_new_view_is_not_wedged():
    stuck = {**BASE, "net_displacement_m": 0.0}
    t, _ = run(BOOT, [{"decision": "continue", "receipt": {"claim": "x"},
                       "next_instruction": "turn toward the sink", "state": {}}] * 5,
               [{"close_enough": True, "relation_holds": True}],
               # Views that genuinely differ: >6 differing bits each step.
               # (1 and 2**40 differ in only two bits — under the threshold
               # they ARE the same view, which is the point of the metric.)
               out=stuck, hashes=[0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                  0xAAAAAAAAAAAAAAAA, 0x5555555555555555,
                                  0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0])
    assert not any(e[0] == "drive" for e in t.log), "unstick fired on a changing view"


def test_not_moving_and_not_seeing_anything_new_is_wedged():
    stuck = {**BASE, "net_displacement_m": 0.0}
    t, _ = run(BOOT, [{"decision": "continue", "receipt": {"claim": "x"},
                       "next_instruction": "walk to the sink area", "state": {}}] * 6,
               [{"close_enough": True, "relation_holds": True}],
               out=stuck, hashes=[7] * 8)
    assert any(e[0] == "drive" for e in t.log), "unstick never fired on a frozen view"


# ── corrections are micro and bounded ────────────────────────────────

def test_verify_correction_is_at_most_two_actions():
    t, _ = run(BOOT, [{"decision": "finish", "receipt": {"claim": "?"}}],
               [{"close_enough": False, "relation_holds": False,
                 "step": ["TURN_LEFT"] * 12}] * 5)
    for e in t.log:
        if e[0] == "drive":
            assert len(e[1]) <= AL.MICRO_ACTIONS, f"burst too long: {e[1]}"


def test_forward_can_be_ablated_out_of_corrections():
    t, _ = run(BOOT, [{"decision": "finish", "receipt": {"claim": "?"}}],
               [{"close_enough": False, "relation_holds": False,
                 "step": ["MOVE_FORWARD", "MOVE_FORWARD"]}] * 4,
               ablate=frozenset({"forward"}))
    for e in t.log:
        if e[0] == "drive":
            assert all(a.startswith("TURN") for a in e[1]), e


def test_drive_ablation_removes_every_motor_path():
    stuck = {**BASE, "net_displacement_m": 0.0}
    t, _ = run(BOOT, [{"decision": "drive", "actions": ["TURN_LEFT"] * 6,
                       "receipt": {"claim": "x"}, "next_instruction": "go left", "state": {}}] * 6,
               [{"close_enough": False, "relation_holds": False, "step": ["TURN_LEFT"]}] * 4,
               out=stuck, hashes=[3] * 8, ablate=frozenset({"drive"}))
    assert not any(e[0] == "drive" for e in t.log)


# ── evidence gate on negative facts ──────────────────────────────────

def test_negative_belief_without_evidence_is_refused():
    s = EvidenceState(mission="m", clauses=["a"], step_budget=400)
    ok, bad = s.propose({"beliefs": [{"claim": "the kitchen is not through that door",
                                      "kind": "negative"}]}, evidence_pool=set(), swept=False)
    assert not ok and bad and not s.active("negative")


def test_negative_belief_with_a_cited_frame_is_accepted():
    s = EvidenceState(mission="m", clauses=["a"], step_budget=400)
    ok, _ = s.propose({"beliefs": [{"claim": "no door on that wall", "kind": "negative",
                                    "evidence_ids": ["obs_001_left"]}]},
                      evidence_pool={"obs_001_left"}, swept=False)
    assert ok and len(s.active("negative")) == 1


def test_beliefs_are_retired_by_id_not_by_rewriting_prose():
    s = EvidenceState(mission="m", clauses=["a"], step_budget=400)
    s.propose({"beliefs": [{"claim": "bar ahead", "kind": "observed"}]},
              evidence_pool=set(), swept=True)
    bid = s.beliefs[0].id
    s.propose({"contradict": [bid]}, evidence_pool=set(), swept=True)
    assert s.beliefs[0].status == "contradicted" and not s.active("observed")


def test_render_is_bounded():
    s = EvidenceState(mission="m" * 300, terminate="t" * 200, clauses=["c"] * 8,
                      step_budget=400)
    for i in range(60):
        s.propose({"beliefs": [{"claim": f"thing {i} " + "x" * 300, "kind": "observed"}]},
                  evidence_pool=set(), swept=True)
        s.record(asked="q" * 300, telemetry={"net_displacement_m": 0.1, "steps_used": 5},
                 actions=["TURN_LEFT"], decision="continue")
    assert len(s.render()) < 8000, len(s.render())


# ── the arrival rule, against the measured confusion matrix ──────────

def test_adjudicate_matches_the_calibration():
    # the 17 false rejections: close, swept, landmark not nameable
    assert adjudicate({"close_enough": True, "relation_holds": False,
                       "target_visible": False}, swept=True)[0] == VERIFIED_SUCCESS
    # the 20 false accepts: visible but far
    assert adjudicate({"close_enough": False, "relation_holds": False,
                       "target_visible": True}, swept=True)[0] == VERIFIED_FAILURE
    # a named visible contradiction always wins
    assert adjudicate({"close_enough": True, "relation_holds": True,
                       "contradiction": "bathroom hallway"}, swept=True)[0] == VERIFIED_FAILURE
    assert adjudicate(None, swept=True)[0] == UNRESOLVED_STOP


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  ok    {n}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {n}: {e}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  ERROR {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
