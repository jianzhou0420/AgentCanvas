"""Semantic tests for the state-container subsystem.

Covers the three axes the design docs promise (ADR-dataflow-002/-004):

* **Reducers** — how writes combine (accumulator / lastWrite / counter);
* **Lifetime** — when a state clears (``lifetime`` expands to the
  ``reset_on`` signal list at build time; ``on_signal`` honours it);
* **Keyed partitions** — one container serving N concurrent workers,
  cloned per key, evicted per key.

Plus the build-time guards: raw wire payload types rejected for
graph-level containers (ADR-026), unknown reducers falling back to
lastWrite, and checkpoint round-trips that skip opaque slots.
"""

from __future__ import annotations

from typing import Any

import pytest

from ..graph_def import ContainerDef, StateDef
from .state_containers import (
    AccumulatorState,
    CounterState,
    LastWriteState,
    _resolve_reset_on,
    build_container,
    build_containers,
)


def _container(**states: StateDef):
    return build_container(ContainerDef(id="c", label="C", states=dict(states)))


# ── reducers ────────────────────────────────────────────────────────────


def test_accumulator_appends_and_trims_to_max_size() -> None:
    s = AccumulatorState(name="h", max_size=3)
    for i in range(5):
        s.write(i)
    assert s.read() == [2, 3, 4]  # oldest trimmed


def test_accumulator_without_max_size_keeps_everything() -> None:
    s = AccumulatorState(name="h")
    for i in range(5):
        s.write(i)
    assert s.read() == [0, 1, 2, 3, 4]


def test_last_write_keeps_only_most_recent() -> None:
    s = LastWriteState(name="v")
    s.write("a")
    s.write("b")
    assert s.read() == "b"


def test_counter_sums_writes_from_zero_default() -> None:
    s = CounterState(name="n")
    s.write(2)
    s.write(3)
    assert s.read() == 5


def test_clear_restores_initial_value_isolated_copy() -> None:
    s = AccumulatorState(name="h", initial_value=["seed"])
    s.write("x")
    s.clear()
    assert s.read() == ["seed"]
    # clear() must hand out a COPY of initial_value — mutating the live
    # value must not corrupt the pristine initial for the next clear.
    s.write("y")
    s.clear()
    assert s.read() == ["seed"]


# ── lifetime → reset_on expansion ───────────────────────────────────────


@pytest.mark.parametrize(
    ("lifetime", "expected"),
    [
        ("forever", []),
        ("step", ["step_end"]),
        ("episode", ["episode_reset"]),
        ("run", ["run_end"]),
    ],
)
def test_lifetime_expands_to_signals(lifetime: str, expected: list[str]) -> None:
    sdef = StateDef(type="lastWrite", lifetime=lifetime)
    assert _resolve_reset_on(sdef) == expected


def test_custom_lifetime_honours_explicit_reset_on() -> None:
    sdef = StateDef(type="lastWrite", lifetime="custom", reset_on=["my_signal"])
    assert _resolve_reset_on(sdef) == ["my_signal"]


def test_unknown_lifetime_falls_back_to_forever() -> None:
    sdef = StateDef(type="lastWrite", lifetime="fortnight")
    assert _resolve_reset_on(sdef) == []


def test_on_signal_clears_only_matching_states() -> None:
    c = _container(
        per_step=StateDef(type="accumulator", lifetime="step"),
        per_episode=StateDef(type="accumulator", lifetime="episode"),
        keep=StateDef(type="accumulator", lifetime="forever"),
    )
    for name in ("per_step", "per_episode", "keep"):
        c.write(name, "x")
    c.on_signal("step_end")
    assert c.read("per_step") == []  # cleared at the step boundary
    assert c.read("per_episode") == ["x"]
    assert c.read("keep") == ["x"]
    c.on_signal("episode_reset")
    assert c.read("per_episode") == []
    assert c.read("keep") == ["x"]  # forever never auto-clears


# ── container access ────────────────────────────────────────────────────


def test_undeclared_state_raises_keyerror_naming_declared() -> None:
    c = _container(known=StateDef(type="lastWrite"))
    with pytest.raises(KeyError, match="known"):
        c.write("unknown", 1)
    with pytest.raises(KeyError, match="undeclared"):
        c.read("unknown")


def test_keyed_partitions_are_isolated_and_lazily_cloned() -> None:
    c = _container(hist=StateDef(type="accumulator"))
    c.write("hist", "a", key="ep1")
    c.write("hist", "b", key="ep2")
    assert c.read("hist", key="ep1") == ["a"]
    assert c.read("hist", key="ep2") == ["b"]
    # The non-keyed template stays untouched by keyed writes.
    assert c.read("hist") == []


def test_keyed_read_of_never_written_key_yields_fresh_initial() -> None:
    c = _container(hist=StateDef(type="accumulator"))
    got = c.read("hist", key="ghost")
    assert got == []
    # ... and it is a copy: mutating it must not leak into the container.
    got.append("junk")
    assert c.read("hist", key="ghost") == []


def test_evict_drops_one_key_leaving_siblings() -> None:
    c = _container(hist=StateDef(type="accumulator"))
    c.write("hist", "a", key="ep1")
    c.write("hist", "b", key="ep2")
    c.evict("ep1")
    assert c.read("hist", key="ep1") == []  # back to fresh initial
    assert c.read("hist", key="ep2") == ["b"]


def test_on_signal_fans_to_keyed_partitions() -> None:
    c = _container(hist=StateDef(type="accumulator", lifetime="episode"))
    c.write("hist", "a", key="ep1")
    c.on_signal("episode_reset")
    assert c.read("hist", key="ep1") == []


# ── checkpoint round-trip ───────────────────────────────────────────────


def test_checkpoint_roundtrip_including_keyed() -> None:
    c = _container(hist=StateDef(type="accumulator"), n=StateDef(type="counter"))
    c.write("hist", "a")
    c.write("n", 2)
    c.write("hist", "k", key="ep1")
    snap = c.checkpoint()

    c2 = _container(hist=StateDef(type="accumulator"), n=StateDef(type="counter"))
    c2.from_checkpoint(snap)
    assert c2.read("hist") == ["a"]
    assert c2.read("n") == 2
    assert c2.read("hist", key="ep1") == ["k"]


class _Opaque:
    """Deliberately non-checkpointable value (deepcopy raises)."""

    def __deepcopy__(self, memo: dict) -> Any:
        raise TypeError("opaque handle cannot be copied")


def test_checkpoint_skips_opaque_slots_instead_of_raising() -> None:
    c = _container(ok=StateDef(type="lastWrite"), opaque=StateDef(type="lastWrite"))
    c.write("ok", "fine")
    c.write("opaque", _Opaque())
    snap = c.checkpoint()  # must not raise
    assert snap["ok"] == "fine"
    assert "opaque" not in snap


# ── build-time guards ───────────────────────────────────────────────────


def test_raw_wire_payload_value_type_rejected_for_graph_containers() -> None:
    cdef = ContainerDef(id="c", states={"img": StateDef(type="lastWrite", value_type="IMAGE")})
    with pytest.raises(ValueError, match="raw wire payloads"):
        build_container(cdef)


def test_allow_opaque_relaxes_value_type_check_for_nodeset_containers() -> None:
    cdef = ContainerDef(id="c", states={"img": StateDef(type="lastWrite", value_type="IMAGE")})
    c = build_container(cdef, allow_opaque=True)  # nodeset-level: by reference
    c.write("img", object())  # holds anything, never serialized


def test_unknown_reducer_falls_back_to_last_write() -> None:
    c = _container(v=StateDef(type="not_a_reducer"))
    c.write("v", 1)
    c.write("v", 2)
    assert c.read("v") == 2  # lastWrite semantics


def test_initial_value_and_max_size_flow_from_config() -> None:
    c = _container(
        h=StateDef(type="accumulator", config={"max_size": 2}),
        n=StateDef(type="counter", config={"initial_value": 10}),
    )
    for i in range(4):
        c.write("h", i)
    assert c.read("h") == [2, 3]
    c.write("n", 5)
    assert c.read("n") == 15


def test_build_containers_builds_all() -> None:
    out = build_containers(
        [
            ContainerDef(id="a", states={"x": StateDef(type="lastWrite")}),
            ContainerDef(id="b", states={"y": StateDef(type="counter")}),
        ]
    )
    assert set(out) == {"a", "b"}


# ── preview ─────────────────────────────────────────────────────────────


def test_preview_summarizes_lists_strings_and_keyed_buckets() -> None:
    c = _container(
        hist=StateDef(type="accumulator"),
        note=StateDef(type="lastWrite"),
        khist=StateDef(type="accumulator"),
    )
    for i in range(5):
        c.write("hist", i)
    c.write("note", "n" * 200)
    c.write("khist", "k", key="ep1")
    preview = c.get_preview()
    assert preview["hist"]["size"] == 5
    assert preview["note"]["preview"].endswith("...")  # truncated at 100
    # Keyed states are summarized across live keys: key count + a
    # bounded sample of ONE key (so size reflects the sample, not the
    # untouched template).
    assert preview["khist"]["keys"] == 1
    assert preview["khist"]["size"] == 1
    assert preview["khist"]["preview"].startswith("[ep1] ")
