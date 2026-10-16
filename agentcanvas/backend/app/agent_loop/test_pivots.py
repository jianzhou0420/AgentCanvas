"""The loop pivots through their interface — no executor, no session."""

from __future__ import annotations

from dataclasses import dataclass, field

from .pivots import (
    LoopCarry,
    final_payload,
    graphout_last,
    graphout_latch,
    graphout_latched,
    graphout_snapshot,
    iterout_port_names,
    stop_requested,
)


def _carry(**ports: bool) -> LoopCarry:
    """A LoopCarry over ``{port name: persist flag}``."""
    return LoopCarry(
        {"ports": [{"name": n, "wire_type": "ANY", "persist": p} for n, p in ports.items()]}
    )


# ── LoopCarry: the six per-port configurations of IterInNode ───────────


def test_run_constant_persists_across_fires():
    """C1 — persist=T, init writer only: the same value every iteration."""
    c = _carry(init_instruction=True)
    c.write("init_instruction", "go left")
    for _ in range(3):
        assert c.fire_inputs() == {"init_instruction": "go left"}
        c.after_fire()


def test_one_shot_emits_once():
    """C2 — persist=F, init writer only: emitted on the step of the write, then gone."""
    c = _carry(init_seed=False)
    c.write("init_seed", 7)
    assert c.ready(set())
    assert c.fire_inputs() == {"init_seed": 7}
    c.after_fire()
    assert not c.ready(set())
    assert c.fire_inputs() == {}


def test_loop_carried_is_fresh_each_iteration():
    """C3 — persist=F, iterOut writer only: each iteration sees the last handoff and nothing older."""
    c = _carry(iterout_pose=False)
    assert c.handoff({"pose": 1, "stop": False}) == ["iterout_pose"]
    assert c.fire_inputs() == {"iterout_pose": 1}
    c.after_fire()
    assert c.fire_inputs() == {}
    c.handoff({"pose": 2})
    assert c.fire_inputs() == {"iterout_pose": 2}


def test_seeded_then_refreshed():
    """P1 — both writers, persist=T: the seed stays, the carry refreshes."""
    c = _carry(init_x=True, iterout_x=True)
    c.write("init_x", "seed")
    assert c.fire_inputs() == {"init_x": "seed"}
    c.after_fire()
    c.handoff({"x": "iter1"})
    assert c.fire_inputs() == {"init_x": "seed", "iterout_x": "iter1"}
    c.after_fire()
    c.handoff({"x": "iter2"})
    assert c.fire_inputs() == {"init_x": "seed", "iterout_x": "iter2"}


def test_pure_feedback_carries_until_rewritten():
    """P5 — iterOut writer only, persist=T: nothing on iter 0, then carried."""
    c = _carry(iterout_hidden=True)
    assert not c.ready(set())
    c.handoff({"hidden": "h1"})
    c.after_fire()
    assert c.fire_inputs() == {"iterout_hidden": "h1"}


def test_handoff_writes_only_declared_slots_and_never_stop():
    c = _carry(iterout_v=True)
    assert c.handoff({"v": 1, "stop": True, "other": 2}) == ["iterout_v"]
    assert c.fire_inputs() == {"iterout_v": 1}
    assert c.port_names == {"iterout_v"}


def test_write_is_last_write_wins_no_fan_in():
    c = _carry(init_items=True)
    c.write("init_items", [1])
    c.write("init_items", [2])
    assert c.fire_inputs() == {"init_items": [2]}


def test_ready_rule():
    c = _carry(init_a=True, init_b=True)
    assert not c.ready(set())
    c.write("init_a", 1)
    assert c.ready(set())
    assert not c.ready({"init_a", "init_b"})
    c.write("init_b", 2)
    assert c.ready({"init_a", "init_b"})


def test_last_carried_reads_the_last_handoff():
    c = _carry(iterout_pose=True, iterout_rgb=True)
    c.handoff({"pose": "p2", "rgb": "img2"})
    c.after_fire()
    c.handoff({"pose": "p3"})
    assert c.last_carried(["pose", "rgb", "never"]) == {
        "pose": "p3",
        "rgb": "img2",
        "never": None,
    }


def test_carry_from_an_unsynthesised_config_is_empty():
    c = LoopCarry({})
    assert c.port_names == set()
    assert not c.ready(set())
    assert c.handoff({"x": 1}) == []


# ── iterOut ─────────────────────────────────────────────────────────────


def test_iterout_port_names_skips_malformed_entries():
    cfg = {"ports": [{"name": "a"}, {"wire_type": "X"}, "junk", {"name": "b"}]}
    assert iterout_port_names(cfg) == ["a", "b"]
    assert iterout_port_names(None) == []


def test_stop_requested_is_the_truthy_stop_input():
    assert stop_requested({"stop": True})
    assert not stop_requested({"stop": 0})
    assert not stop_requested({"v": 1})
    assert not stop_requested(None)


def test_final_payload_mirrors_values_and_adds_final_stop():
    assert final_payload({"pose": 1, "stop": True}) == {"final_pose": 1, "final_stop": True}
    assert final_payload(None) == {"final_stop": True}


# ── graphOut ────────────────────────────────────────────────────────────


@dataclass
class _Node:
    state: dict = field(default_factory=dict)
    pending_inputs: dict = field(default_factory=dict)


def test_graphout_snapshot_is_a_copy_of_the_pending_inputs():
    n = _Node(pending_inputs={"value": 3})
    assert graphout_last(n) == {}
    graphout_snapshot(n)
    n.pending_inputs["value"] = 4
    assert graphout_last(n) == {"value": 3}


def test_graphout_latch_round_trip():
    n = _Node()
    assert graphout_latched(n) is None
    graphout_latch(n, 9)
    assert graphout_latched(n) == 9
