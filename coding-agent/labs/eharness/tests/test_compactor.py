"""Offline self-test for the context organ's size discipline.

Run:  python -m eharness.tests.test_compactor

The case that matters here killed a real episode: one extra sentence per tool
result took a run from 3 compactions to 21 and from a 13.1k-token peak to 16.3k,
because the L2 cut is required to keep 24 messages and 24 fat messages are still
fat. The 9B then emitted malformed tool-call XML three times and the episode
died. A count is not a budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CA = Path(__file__).resolve().parents[2]
for _p in (str(_CA), str(_CA / "harnesses" / "mini")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eharness.compactor import SoloModel, SoloModelConfig

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name + (f"  [{detail}]" if detail else ""))


class _Fake(SoloModel):
    """Only the cut is under test — no model, no harness, no network."""

    def __init__(self, budget: int, tail: int = 24):
        self.config = SoloModelConfig(model_name="test", tail_msgs=tail,
                                      tail_budget=budget, compact_at=100)
        self._h = _FakeHarness()

    def _emit_cb(self, *a, **k):
        pass

    def _boundary_message(self, n):
        return {"role": "user", "content": f"[{n} dropped]"}


class _FakeState:
    def snapshot(self):
        return {"landmarks": set(), "negative_facts": set(), "visited": set()}

    def assert_monotone(self, a, b):
        for k in a:
            assert a[k] <= b[k], k


class _FakeFrames:
    def keyframes(self):
        return []


class _FakeHarness:
    state = _FakeState()
    frames = _FakeFrames()


def turns(n: int, chars: int) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * chars})
    return msgs


print("── 1. a fat tail is cut deeper than the count alone would ──────────")
thin, fat = turns(30, 200), turns(30, 1400)
m = _Fake(budget=2000)
kept_thin, kept_fat = m._l2_cut(thin), m._l2_cut(fat)
check("both are cut at all", len(kept_thin) < len(thin) and len(kept_fat) < len(fat),
      f"{len(kept_thin)} / {len(kept_fat)}")
check("the FAT run keeps fewer messages than the thin one",
      len(kept_fat) < len(kept_thin), f"fat {len(kept_fat)} vs thin {len(kept_thin)}")
check("and the fat tail actually lands under budget",
      m._estimate(kept_fat) < 2000 * 2.5, f"{m._estimate(kept_fat)} tokens")

print()
print("── 2. the count still binds when messages are small ────────────────")
m2 = _Fake(budget=10**9)
kept = m2._l2_cut(turns(30, 50))
check("with an unlimited budget the tail is the configured count",
      len(kept) <= 2 + 1 + 24 + 1, f"{len(kept)} messages")

print()
print("── 3. tool_use / tool_result pairs are never split ─────────────────")
for budget in (500, 1500, 4000, 10**9):
    kept = _Fake(budget=budget)._l2_cut(turns(30, 900))
    roles = [x.get("role") for x in kept]
    orphan = any(roles[i] == "tool" and roles[i - 1] not in ("assistant", "tool")
                 for i in range(1, len(roles)))
    check(f"budget {budget}: no orphaned tool result", not orphan, str(roles[:6]))

print()
print("── 4. the cut never eats the head ──────────────────────────────────")
kept = _Fake(budget=1)._l2_cut(turns(30, 2000))
check("system + task survive any budget",
      kept[0]["role"] == "system" and kept[1]["role"] == "user",
      str([x.get("role") for x in kept[:3]]))
check("and something of the conversation survives too", len(kept) > 3, str(len(kept)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("compactor self-test: all passed")
