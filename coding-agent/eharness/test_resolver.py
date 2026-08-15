"""§16.4/§16.5 — the ONE InstructionResolver: conservative fallback, no
frame-0 arming, one record for every executor.

The convicted bug: the old fallback put parts[:-1] on the route and
parts[-1] in terminate, so "Walk to the kitchen." produced ZERO route
segments and near_stop_armed held from the first frame — the episode
started with its termination condition live.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eharness import resolver as rv
from eharness.state_block import StateBlock

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def armed(rec: rv.InstructionRecord) -> bool:
    s = StateBlock()
    s.instruction = rec.instruction
    s.sub_instructions = rec.segments
    s.terminate = rec.terminate
    return s.near_stop_armed


SINGLE = "Walk to the kitchen."
MULTI = ("Go past the couch, walk between the bar and the chairs, "
         "stop at the corner of the bar")

print("── fallback is conservative, never armed at frame 0 ────────────────")
rec = rv.resolve(SINGLE)                     # no model configured
check("single sentence → ONE actionable segment (the whole instruction)",
      rec.segments == [SINGLE] and rec.source == "fallback",
      str(rec.segments))
check("…terminate is the safe clause, not the swallowed last leg",
      rec.terminate == rv.FALLBACK_TERMINATE)
check("…and near_stop is NOT armed at frame 0", not armed(rec))

rec2 = rv.resolve(MULTI, transport=lambda s, u: (_ for _ in ()).throw(
    TimeoutError("splitter timed out")))
check("transport timeout → fallback, reason recorded",
      rec2.source == "fallback" and "transport" in rec2.fallback_reason
      and rec2.segments == [MULTI], rec2.fallback_reason)
check("…not armed either", not armed(rec2))

rec3 = rv.resolve(MULTI, transport=lambda s, u: "sure! here's my parse: [not json")
check("invalid JSON → fallback", rec3.source == "fallback"
      and rec3.fallback_reason == "invalid JSON")

rec4 = rv.resolve(MULTI, transport=lambda s, u:
                  '{"segments": [], "terminate": "stop at the bar corner"}')
check("ZERO segments from the LLM is a REJECTION, not an armed route",
      rec4.source == "fallback" and rec4.segments == [MULTI]
      and not armed(rec4), rec4.fallback_reason)

rec5 = rv.resolve(MULTI, transport=lambda s, u:
                  '{"segments": ["go past the couch"], "terminate": ""}')
check("missing terminate is rejected too", rec5.source == "fallback")

print()
print("── a good parse passes through, terminate separate ─────────────────")
GOOD = ('{"segments": ["go past the couch", '
        '"walk between the bar and the chairs"], '
        '"terminate": "stop when you reach the corner of the bar", '
        '"landmarks": ["bar counter", "lounge chairs"]}')
rec6 = rv.resolve(MULTI, transport=lambda s, u: GOOD)
check("segments preserved in order", rec6.segments[0] == "go past the couch"
      and len(rec6.segments) == 2 and rec6.source == "llm")
check("terminate is its own condition", "stop when" in rec6.terminate)
check("detector-friendly landmarks ride along",
      rec6.landmarks == ["bar counter", "lounge chairs"])
check("not armed until the segments are actually done", not armed(rec6))
s6 = StateBlock()
s6.sub_instructions, s6.terminate = rec6.segments, rec6.terminate
s6.cursor = 2
check("…armed exactly when the cursor clears the route", s6.near_stop_armed)

SIX = json.dumps({"segments": [f"leg {i}" for i in range(1, 7)],
                  "terminate": "stop at the end"})
rec6b = rv.resolve(MULTI, transport=lambda s, u: SIX)
check("6-segment parse: falls back losslessly, NEVER truncates a leg",
      (rec6b.source == "fallback" and rec6b.segments == [MULTI])
      or len(rec6b.segments) == 6,
      f"source={rec6b.source}, {len(rec6b.segments)} segs")

print()
print("── strict parsing: raw_decode, not a greedy regex ──────────────────")
rec7 = rv.resolve(MULTI, transport=lambda s, u:
                  'thinking... {"segments": ["a"], "terminate": "stop at x"} '
                  'and later {"unrelated": true}')
check("first JSON object wins, trailing braces ignored",
      rec7.source == "llm" and rec7.segments == ["a"])

print()
print("── one record, written once, read everywhere ───────────────────────")
tdir = Path(tempfile.mkdtemp(prefix="rec_"))
rv.save_record(tdir, rec6)
loaded = rv.load_record(tdir)
check("round-trip keeps every field",
      loaded is not None and loaded.segments == rec6.segments
      and loaded.terminate == rec6.terminate
      and loaded.landmarks == rec6.landmarks
      and loaded.version == rv.RECORD_VERSION)
calls = {"n": 0}


def counting_transport(s, u):
    calls["n"] += 1
    return GOOD


again = rv.resolve_or_load(tdir, MULTI, transport=counting_transport)
check("an existing record wins — no second parse",
      calls["n"] == 0 and again.segments == rec6.segments)
stale = rv.resolve_or_load(tdir, "a DIFFERENT instruction",
                           transport=counting_transport)
check("…but a stale record for another instruction is ignored",
      calls["n"] == 1 and stale.instruction == "a DIFFERENT instruction")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("instruction resolver: all passed")
