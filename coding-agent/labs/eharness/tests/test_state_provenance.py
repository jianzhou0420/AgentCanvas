"""§14.11 — StateBlock three-layer provenance + §14.2 STATE versioning.

ObservedFact (harness-measured, carries frame/map_version/confidence) /
ModelBelief (tool-parameter writes, revocable, carries who/frame/step) /
VerifiedMilestone (judge-confirmed segment) — and the render must SHOW the
difference, because before this split a wrong model claim read exactly as
reliable as a measured fact.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eharness.state_block import StateBlock


def _fresh(**kw) -> StateBlock:
    sb = StateBlock(**kw)
    sb.sub_instructions = ["walk past the bar", "turn left at the pool",
                           "stop by the corner of the bar"]
    sb.terminate = "the corner of the bar"
    return sb


def test_model_writes_are_beliefs_with_provenance():
    sb = _fresh()
    acc = sb.apply_update({"landmark": "red sofa", "dead_end": "left hallway is a dead end",
                           "surroundings": "open lounge"},
                          source="sub:step", frame=7, step=42)
    assert any("landmark" in a for a in acc) and any("dead end" in a for a in acc)
    lm = sb.landmarks["red sofa"]
    assert lm["origin"] == "model" and lm["frame"] == 7 and lm["step"] == 42
    nf = sb.negative_facts[0]
    assert nf["origin"] == "model" and nf["by"] == "sub:step" and nf["step"] == 42
    assert sb.belief_meta["surroundings"]["frame"] == 7
    text = sb.render()
    assert "YOUR BELIEFS — unverified, revocable:" in text
    assert "landmarks you named: red sofa" in text
    assert "dead ends you claimed (unverified" in text


def test_harness_facts_carry_frame_and_map_version():
    sb = _fresh()
    assert sb.bind_landmark("bar counter", frame=3, map_version=17, confidence=0.81)
    meta = sb.landmarks["bar counter"]
    assert meta["origin"] == "harness" and meta["map_version"] == 17
    assert meta["confidence"] == 0.81 and meta["frame"] == 3
    sb.add_journey("crossed the lounge")
    text = sb.render()
    assert "OBSERVED — measured by the harness:" in text
    assert "landmarks (observed): bar counter" in text
    # the harness landmark must NOT appear under the model's claims
    belief_part = text.split("YOUR BELIEFS")[1]
    assert "bar counter" not in belief_part


def test_verified_milestone_vs_claimed_advance():
    sb = _fresh()
    # model claims segment 1 done → belief-grade advance
    sb.apply_update({"advance_subgoal": True, "progress_note": "past the bar"},
                    source="sub:goto", frame=5, step=20)
    assert sb.done[0]["verified"] is False
    assert "(done (claimed))" in sb.render()
    # judge confirms segment 2 → VerifiedMilestone (wrapper appends verified=True)
    sb.done.append({"idx": 1, "text": sb.sub_instructions[1],
                    "evidence": "judge-evidence: pool on the left", "frame": None,
                    "verified": True})
    sb.cursor = 2
    text = sb.render()
    assert "VERIFIED milestones: segment [2] ✓" in text
    assert "(done ✓)" in text and "(done (claimed))" in text


def test_revoke_dead_end_leaves_audit_trail():
    sb = _fresh()
    sb.apply_update({"dead_end": "corridor on the right is blocked"},
                    source="sub:step", frame=2, step=8)
    before = sb.snapshot()
    acc = sb.apply_update({"revoke_dead_end": "corridor on the right"},
                          source="sub:face", frame=9, step=30)
    assert any("revoked" in a for a in acc)
    nf = sb.negative_facts[0]
    assert nf["revoked"]["by"] == "sub:face" and nf["revoked"]["frame"] == 9
    # monotone: the ENTRY never leaves — only its status changed
    StateBlock.assert_monotone(before, sb.snapshot())
    text = sb.render()
    assert "corridor on the right" not in text.split("dead ends you claimed")[0] \
        or "revoked" in text
    # a revoked claim is out of the active claimed list
    assert "dead ends you claimed" not in text  # it was the only one, now revoked


def test_revoke_matches_substring_and_only_once():
    sb = _fresh()
    sb.apply_update({"dead_end": "left door leads nowhere"}, source="sub:step")
    sb.apply_update({"dead_end": "left stair is roped off"}, source="sub:step")
    sb.apply_update({"revoke_dead_end": "left door"}, source="sub:step")
    revoked = [f for f in sb.negative_facts if f.get("revoked")]
    assert len(revoked) == 1 and revoked[0]["fact"].startswith("left door")


def test_legacy_state_json_migrates():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        p.write_text(json.dumps({
            "instruction": "walk",
            "sub_instructions": ["a", "b"],
            "cursor": 2,
            "landmarks": {"old sofa": {"frame": 1, "by": "sub:step", "t": 0}},
            "done": [
                {"idx": 0, "text": "a", "evidence": "judge-evidence: seen",
                 "frame": None},
                {"idx": 1, "text": "b", "evidence": "walked it", "frame": 4},
            ],
            "negative_facts": [{"id": "n1", "fact": "old dead end",
                                "frame": 2, "by": "sub:step",
                                "verified": False, "place": ""}],
        }))
        sb = StateBlock.load(p)
    assert sb.landmarks["old sofa"]["origin"] == "model"
    assert sb.done[0]["verified"] is True      # judge-evidence prefix
    assert sb.done[1]["verified"] is False     # model-claimed
    sb.render()                                # renders without KeyError


def test_state_version_line_monotone():
    sb = _fresh()
    assert sb.render().startswith("[STATE]")           # v0: classic header
    sb.state_version = 3
    t3 = sb.render()
    assert t3.startswith("[STATE v3 — supersedes every earlier STATE block]")
    sb.state_version += 1
    assert sb.render().startswith("[STATE v4")
    # purity: same state → same text
    assert sb.render() == sb.render()


def test_wrapper_inject_bumps_version():
    from eharness.wrapper import HarnessedToolset

    class R:
        def __init__(self):
            self.content = []

    import tempfile
    from pathlib import Path

    w = object.__new__(HarnessedToolset)
    w.inject_state = True
    w.state = _fresh()
    w.state.path = Path(tempfile.mkdtemp(prefix="sbver_")) / "state.json"
    r = R()
    w._append_state_render(r)
    # review P2: an UNCHANGED render is not re-issued — a full-history
    # executor got a ~2.6KB duplicate per turn for zero information. The
    # second injection is a one-line stub citing the still-law version.
    w._append_state_render(r)
    assert w.state.state_version == 1
    assert "[STATE v1 — supersedes" in r.content[0]["text"]
    assert r.content[-1]["text"].startswith("[STATE v1 unchanged")
    # a real state change mints a new version and a full render again
    w.state.journey.append("crossed the lobby")
    w._append_state_render(r)
    assert w.state.state_version == 2
    assert "[STATE v2 — supersedes" in r.content[-1]["text"]
    # review P2: the shown version is PERSISTED — a bridge restart must not
    # regress it and reissue a duplicate "supersedes" number
    reloaded = type(w.state).load(w.state.path)
    assert reloaded.state_version == 2
    # solo path: no injection, version untouched → classic header preserved
    w2 = object.__new__(HarnessedToolset)
    w2.inject_state = False
    w2.state = _fresh()
    r2 = R()
    w2._append_state_render(r2)
    assert r2.content == [] and w2.state.state_version == 0


def test_render_shrinks_but_keeps_groups():
    sb = _fresh()
    for i in range(20):
        sb.add_journey(f"leg {i} of the walk went fine and was uneventful")
        sb.apply_update({"landmark": f"landmark number {i}"},
                        source="sub:step", frame=i, step=i)
    sb.bind_landmark("measured doorway", frame=1, map_version=2)
    text = sb.render()
    assert len(text) <= StateBlock.STATE_MAX_CHARS + 400
    assert "OBSERVED — measured by the harness:" in text
    assert "YOUR BELIEFS — unverified, revocable:" in text


def test_forward_cap_exempts_stop_batches():
    # review P1: the wrapper-level pace cap truncated step([1]*7+[0]) to
    # [1]*6 BEFORE _is_stop_intent or the env saw the STOP — the terminal
    # decision was silently discarded. The inner toolset already exempts
    # STOP batches; the wrapper cap must too.
    import inspect

    from eharness.wrapper import HarnessedToolset
    src = inspect.getsource(HarnessedToolset.execute)
    assert '0 not in args["actions"]' in src


def test_absorb_places_keeps_both_same_named_places():
    # review P1: record_visit dedupes consecutive same-named places, and the
    # unconditional visited[-1]["id"] stamp then overwrote the PREVIOUS
    # place's id — the monotone set lost a paid-for entry.
    from types import SimpleNamespace

    from eharness.wrapper import HarnessedToolset

    class P:
        def __init__(self, i, n):
            self.index, self._n = i, n

        def name(self):
            return self._n

    w = object.__new__(HarnessedToolset)
    w.state = _fresh()
    w._emit = lambda *a, **k: None
    same = "a place you could not make out"
    w.inner = SimpleNamespace(places=SimpleNamespace(
        places=[P(1, same), P(2, same)],
        recognition_question=lambda: ""))
    w._absorb_places()
    ids = [v["id"] for v in w.state.visited]
    assert ids == ["place1", "place2"], ids
    # …and re-running is idempotent (both known now)
    w._absorb_places()
    assert [v["id"] for v in w.state.visited] == ["place1", "place2"]


def test_revoke_picks_best_match_not_first_substring():
    # review P2, scenario A: 'bar' must revoke the recent WRONG bar-counter
    # fact, not the older still-TRUE stairs fact that merely mentions a bar.
    sb = _fresh()
    sb.negative_facts.append({"fact": "the stairs by the bar are a dead end"})
    sb.negative_facts.append({"fact": "bar counter path is blocked"})
    sb.apply_update({"revoke_dead_end": "bar"}, source="model")
    assert not sb.negative_facts[0].get("revoked")
    assert sb.negative_facts[1].get("revoked")
    # scenario B: a natural revocation sentence must hit the fact it names,
    # never a terse generic fact that happens to be a substring of it.
    sb2 = _fresh()
    sb2.negative_facts.append({"fact": "dead end"})
    sb2.negative_facts.append({"fact": "the upstairs hallway is a dead end"})
    sb2.apply_update(
        {"revoke_dead_end": "the upstairs hallway is not a dead end after all"},
        source="model")
    assert not sb2.negative_facts[0].get("revoked")
    assert sb2.negative_facts[1].get("revoked")
    # a purely-generic key still works on a purely-generic fact
    sb3 = _fresh()
    sb3.negative_facts.append({"fact": "dead end"})
    sb3.apply_update({"revoke_dead_end": "dead end"}, source="model")
    assert sb3.negative_facts[0].get("revoked")
    # …and a key matching nothing revokes nothing (safe no-op)
    sb4 = _fresh()
    sb4.negative_facts.append({"fact": "the kitchen is blocked"})
    sb4.apply_update({"revoke_dead_end": "upstairs balcony"}, source="model")
    assert not sb4.negative_facts[0].get("revoked")


def test_sink_cold_keeps_provenance():
    # review P2: sinking a harness-OBSERVED landmark must not strip its
    # origin — it re-rendered as a revocable model claim.
    import tempfile
    from pathlib import Path

    sb = _fresh()
    for i in range(13):
        sb.bind_landmark(f"observed landmark {i}", frame=i, map_version=i,
                         confidence=0.9)
    sunk = sb.sink_cold(Path(tempfile.mkdtemp(prefix="sink_")) / "cold.md",
                        keep_landmarks=12)
    assert sunk == 1
    stub = sb.landmarks["observed landmark 0"]
    assert stub.get("sunk") is True and stub.get("origin") == "harness"
    text = sb.render()
    head, _, beliefs = text.partition("YOUR BELIEFS")
    assert "observed landmark 0" not in beliefs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
