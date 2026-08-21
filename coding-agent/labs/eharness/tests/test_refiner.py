"""§17.7 — the Refiner, text-only. No Habitat, no RGB/depth/map, no real
model call: every case injects the raw model reply and tests the CONTRACT
— prompt content, parsing, fallback, and the driver's single execution
point. Semantic quality of a real model's rewrite is deliberately NOT
tested (§17.5: no semantic validator exists, so none is tested)."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eharness import refiner as rf

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


BAR = ("Go straight past the pool. Walk between the bar and chairs. Stop "
       "when you get to the corner of the bar. That's where you will wait.")
BAR_REFINED = ("Go straight past the pool. Walk between the bar and the "
               "chairs. Stop when you reach the far corner of the bar.")

print("── the canonical Harness SKILL.md carries the §17 rules ────────────")
skill = rf.load_skill_text()
check("Harness-local SKILL.md found and non-trivial", len(skill) > 500,
      f"{len(skill)} chars from {rf.SKILL_PATH}")
check("skill is no longer stored at the repository root",
      not Path(__file__).resolve().parents[3].joinpath("skill.md").exists())
check("fixed JSON contract, no scratchpad escape hatch",
      '"instruction"' in skill and '"clarified"' in skill
      and "scratchpad" not in skill.lower().replace("no scratchpad", ""))
check("keep-input-language rule present", "input language" in skill.lower())
check("the far-corner traversal example is in the prompt",
      "far corner" in skill)
check("the painting-on-your-left keep rule is in the prompt",
      "painting" in skill)
check("the filler example ('where you will wait') is in the prompt",
      "where you will wait" in skill)
system, user = rf.build_prompt(skill, BAR)
check("build_prompt pins the output contract in the system prompt",
      "ONE JSON object" in system and user == BAR)

print()
print("── parse: the fixed contract, and nothing clever ───────────────────")
good = json.dumps({"instruction": BAR_REFINED, "clarified": True,
                   "note": "corner → far corner, entailed by prior "
                           "between-traversal"})
r = rf.parse(good, BAR)
check("bar/chairs: refined text adopted, clarified recorded",
      r.instruction == BAR_REFINED and r.clarified
      and not r.fallback_reason)
check("note is audit-only — never merged into the instruction",
      "entailed" not in r.instruction and r.note.startswith("corner"))

HALL = "Turn right at the hallway. Wait by the door."
r2 = rf.parse(json.dumps({"instruction": HALL, "clarified": False}), HALL)
check("hallway door: text-insufficient → unchanged, clarified false",
      r2.instruction == HALL and not r2.clarified and not r2.fallback_reason)

PAINT = ("Walk down the corridor. You will pass a painting on your left. "
         "Stop at the double doors.")
r3 = rf.parse(json.dumps({"instruction": PAINT, "clarified": False}), PAINT)
check("painting cue: a route cue passes through intact",
      "painting on your left" in r3.instruction)

# "That's where you wait": removable ONLY when the previous sentence already
# carries the full stop endpoint (the bar case above); when it does not,
# the model must keep it — parse passes either through, the RULE lives in
# the prompt (checked above), and the fallback keeps everything.
WAIT_KEEP = "Walk toward the shelves. That's where you wait."
r4 = rf.parse(json.dumps({"instruction": WAIT_KEEP, "clarified": False}),
              WAIT_KEEP)
check("'that's where you wait' with no prior endpoint survives verbatim",
      r4.instruction == WAIT_KEEP)

HINDI = "सीढ़ियों से ऊपर जाओ और अंत में कमरे में रुको।"
r5 = rf.parse(json.dumps({"instruction": HINDI, "clarified": False}), HINDI)
check("multilingual text passes through byte-identical (no translation)",
      r5.instruction == HINDI)

print()
print("── every failure falls back to the ORIGINAL, recorded ──────────────")
for label, raw, reason in [
    ("empty output", "", "empty output"),
    ("whitespace output", "   \n", "empty output"),
    ("invalid JSON", "Sure! The cleaned instruction is: go past the pool",
     "invalid JSON"),
    ("empty instruction field",
     json.dumps({"instruction": "", "clarified": True}),
     "empty instruction field"),
    ("None (call failed)", None, "empty output"),
]:
    rr = rf.parse(raw, BAR)
    check(f"{label} → original text, reason recorded",
          rr.instruction == BAR and rr.fallback_reason == reason,
          rr.fallback_reason)

print()
print("── refine_episode: one call, one audit record ──────────────────────")
tdir = Path(tempfile.mkdtemp(prefix="ref_"))
calls = {"n": 0}


def fake_call(system_p: str, user_p: str) -> str:
    calls["n"] += 1
    assert "ONE JSON object" in system_p and user_p == BAR
    return good


res = rf.refine_episode(BAR, live_dir=tdir, call=fake_call, model="test-model")
rec = json.loads((tdir / "instruction_refinement.json").read_text())
check("exactly one model call", calls["n"] == 1)
check("audit record: original + refined + note + model",
      rec["original"] == BAR and rec["instruction"] == BAR_REFINED
      and rec["clarified"] and rec["model"] == "test-model")
check("audit record carries measured refiner latency",
      isinstance(rec["duration_ms"], int) and rec["duration_ms"] >= 0)
res2 = rf.refine_episode(BAR, live_dir=tdir, call=None, model="x")
check("adapter without a refiner → recorded fallback, never an error",
      res2.instruction == BAR
      and res2.fallback_reason == "adapter has no refiner")


def exploding(s, u):
    raise RuntimeError("provider down")


res3 = rf.refine_episode(BAR, live_dir=tdir, call=exploding, model="x")
check("a crashing call falls back whole, reason recorded",
      res3.instruction == BAR and res3.fallback_reason.startswith("call:"))

print()
print("── driver: refine runs ONCE, before the resolver, same gate ────────")
driver_src = Path(__file__).resolve().parents[2].joinpath("driver.py").read_text()
i_refine = driver_src.find("refine_episode")
i_resolve = driver_src.find("resolve_or_load")
check("refine sits BEFORE the resolver in run_episode",
      0 < i_refine < i_resolve)
seg = driver_src[i_refine - 1200:i_resolve]
check("both ride the same eharness-line gate",
      driver_src.count('adapter.name == "eharness" or flag_on('
                       'cfg["extra"].get("eh_bridge"))') == 2)
check("the refined text replaces `instruction` for everything downstream",
      "instruction = _ref.instruction" in seg)
check("adapters without the method are a fallback, not a crash",
      'hasattr(adapter, "refine_instruction")' in seg)

from eharness.eharness_agent import EharnessAdapter  # noqa: E402

check("mini adapter implements the refine protocol",
      callable(getattr(EharnessAdapter, "refine_instruction", None)))
sdk_src = Path(__file__).resolve().parents[2].joinpath(
    "harnesses", "claude_sdk.py").read_text()
check("SDK refine is tool-less, single-turn, no MCP",
      "max_turns=1" in sdk_src and "refine_instruction" in sdk_src
      and "mcp_servers" not in inspect.getsource(  # the refine method itself
          __import__("harnesses.claude_sdk", fromlist=["ClaudeSdkAdapter"])
          .ClaudeSdkAdapter.refine_instruction))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("instruction refiner: all passed")
