from __future__ import annotations

"""The outer agent's calls, v2 — schema-checked, three-state, evidence-citing.

What changed from ``agent_judge`` and why, all of it forced by the n=100 numbers:

**Verification asked the wrong question.** R2R-CE scores success as *within 3 m
of the goal*. The verifier was asked "can you see and name the termination
landmark". Those are not the same test, and the confusion matrix shows the cost
of conflating them — 63% agreement, 61% precision, 65% recall:

    accepted 51  →  31 真 / 20 误收   误收 NE 中位 6.96 m, 7 集 >8 m
    否决     49  →  17 误拒 / 32 真   误拒 NE 全部 <3.0 m, 中位 1.80 m

The 17 false rejections are the sharpest signal in the run: the robot was
standing a median 1.8 m from the goal, had already been shown the full 360°
sweep, and the verifier still said no. So the single `satisfied` boolean is
split into the three questions it was silently averaging — *can you see it*,
*does the spatial relation hold*, *are you close enough* — and *the harness*
combines them. The model reports observations; the arrival rule stays in code
where it can be recalibrated against the logs without re-running anything.

**A failed verification claimed success.** `parse_failed → satisfied: True`
fails open on the one irreversible gate. Now: strict schema check → retry on the
same model → retry on a cheap backup → `UNKNOWN`, which the terminal policy
records as an *unverified* stop, never as a verified one.

**Corrections were open-loop.** Up to 12 blind actions per burst, and episodes
that used them scored 41.2% where the policy alone scored 64.7% on the same
episodes. Corrections are now micro closed-loop: at most `MICRO_ACTIONS` per
burst, re-observe between bursts, and forward motion is off unless explicitly
enabled — turning is free of positional cost, walking is not.

**Beliefs had no provenance.** The review now returns a receipt: what it claims,
which frames support it, what it did *not* establish. Evidence ids are the frame
handles the event log wrote, so a claim resolves to pixels after the run.

last updated: 2026-08-10
"""

import asyncio
import base64
import io
import json
import os
from typing import Any

from vlaharness.judge import _extract_obj
from vlaharness.state import MOTOR, compact_actions

BACKUP_MODEL = "claude-haiku-4-5"
MAX_IMAGE_B64 = 12_000_000
QUALITY_LADDER = (92, 85, 78, 70, 62)
# One correction burst. Small enough that the robot is re-observed before it can
# walk out of a success radius it may already be inside.
MICRO_ACTIONS = 2


# ── asking, with the failure made visible and the schema enforced ─────

async def _ask_once(system: str, content: Any, model: str
                    ) -> tuple[str, str, dict[str, Any]]:
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                                  ResultMessage, TextBlock, ThinkingBlock)

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    options = ClaudeAgentOptions(
        model=model, system_prompt=system, allowed_tools=[],
        permission_mode="bypassPermissions", max_turns=4, setting_sources=[], env=env,
        thinking={"type": "adaptive", "display": "summarized"})
    said: list[str] = []
    thought: list[str] = []
    meta: dict[str, Any] = {}
    async with ClaudeSDKClient(options=options) as client:
        if isinstance(content, str):
            await client.query(content)
        else:
            async def stream():
                yield {"type": "user", "message": {"role": "user", "content": content},
                       "parent_tool_use_id": None}

            await client.query(stream())
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        said.append(b.text)
                    elif isinstance(b, ThinkingBlock):
                        thought.append(getattr(b, "thinking", "") or "")
            elif isinstance(msg, ResultMessage):
                use = getattr(msg, "usage", None) or {}
                meta = {"is_error": getattr(msg, "is_error", None),
                        "subtype": getattr(msg, "subtype", None),
                        "num_turns": getattr(msg, "num_turns", None),
                        "cost_usd": getattr(msg, "total_cost_usd", None),
                        "api_ms": getattr(msg, "duration_api_ms", None),
                        "in_tok": use.get("input_tokens"),
                        "out_tok": use.get("output_tokens"),
                        "cache_read_tok": use.get("cache_read_input_tokens"),
                        "cache_write_tok": use.get("cache_creation_input_tokens"),
                        "ctx_tok": (use.get("input_tokens") or 0)
                                   + (use.get("cache_read_input_tokens") or 0)
                                   + (use.get("cache_creation_input_tokens") or 0)}
                break
    return "\n".join(said), "\n".join(t for t in thought if t.strip()), meta


def ask(system: str, content: Any, model: str, *, require: tuple[str, ...] = (),
        backup: str | None = BACKUP_MODEL) -> tuple[dict[str, Any] | None, str, str, dict]:
    """Ask, validate, escalate. Returns (parsed|None, raw, thinking, diagnostics).

    ``None`` means the answer never arrived in a usable form after every retry.
    Callers must branch on that explicitly — the one thing they may not do is
    treat it as a value.
    """
    attempts: list[dict[str, Any]] = []
    plan = [("primary", model), ("retry", model)]
    if backup:
        plan.append(("backup", backup))
    meta: dict[str, Any] = {}
    raw = ""
    thinking = ""
    for label, m in plan:
        try:
            raw, thinking, meta = asyncio.run(_ask_once(system, content, m))
        except Exception as exc:  # noqa: BLE001
            raw, thinking, meta = "", "", {"exception": repr(exc)[:200]}
        meta["attempt"] = label
        got = _extract_obj(raw) if raw.strip() else None
        missing = [k for k in require if got is not None and k not in got]
        if got is not None and not missing and not meta.get("is_error"):
            meta["failed"] = False
            meta["attempts"] = attempts + [{"attempt": label, "ok": True}]
            return got, raw, thinking, meta
        attempts.append({"attempt": label, "ok": False,
                         "why": ("empty" if not raw.strip() else
                                 "unparseable" if got is None else
                                 f"missing {missing}")})
    meta["failed"] = True
    meta["attempts"] = attempts
    return None, raw, thinking, meta


# ── payload packing (unchanged mechanics, kept local) ─────────────────

def _encode(b64: str, quality: int) -> tuple[str, int] | None:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return data, len(data)
    except Exception:
        return None


def _pack(items: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {"n_requested": len(items)}
    for q in QUALITY_LADDER:
        enc = [(lab, _encode(b, q)) for lab, b in items]
        enc = [(lab, g) for lab, g in enc if g]
        total = sum(g[1] for _, g in enc)
        if total <= MAX_IMAGE_B64:
            report.update(quality=q, bytes_b64=total, dropped=0, n_sent=len(enc))
            blocks: list[dict[str, Any]] = []
            for lab, (data, _n) in enc:
                blocks += [{"type": "text", "text": f"[{lab}]"},
                           {"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": data}}]
            return blocks, report
    report.update(quality=QUALITY_LADDER[-1], bytes_b64=0, dropped=len(items), n_sent=0)
    return [], report


CURRENT = (("FRONT", "_frame_front"), ("LEFT", "_frame_left"),
           ("BACK", "_frame_back"), ("RIGHT", "_frame_right"))


def handle_map(handles: Any) -> dict[str, str]:
    """Frame handles → view key, matched by the filename suffix the event log
    writes (``obs_003_front``), not by position.

    Positional zipping is wrong here and was: ``frames(full=True)`` writes the
    segment frames first and the four current views after, so the lists are
    different lengths and the first current view would be labelled with a
    segment frame's id — an evidence handle pointing at the wrong picture, which
    is worse than no handle at all.
    """
    if isinstance(handles, dict):
        return {k: v for k, v in handles.items() if v}
    out: dict[str, str] = {}
    for h in (handles or []):
        for tag, key in CURRENT:
            if str(h).endswith("_" + tag.lower()):
                out[key] = str(h)
    return out


def current_items(views: dict[str, Any], handles: Any = None) -> list[tuple[str, str]]:
    hm = handle_map(handles)
    out = []
    for tag, key in CURRENT:
        if views.get(key):
            h = hm.get(key)
            out.append((f"CURRENT VIEW — {tag}" + (f"  id={h}" if h else ""), views[key]))
    return out


def sweep_items(look: dict[str, Any], handles: list[str] | None = None
                ) -> list[tuple[str, str]]:
    frames = look.get("_sweep") or []
    degs = look.get("_sweep_deg") or []
    items = []
    for i, b64 in enumerate(frames):
        d = degs[i] if i < len(degs) else None
        where = ("straight ahead" if d == 0 else "directly behind" if d == 180 else
                 f"{d}° LEFT" if (d or 0) < 180 else f"{360 - (d or 0)}° RIGHT")
        h = handles[i] if handles and i < len(handles) else None
        items.append((f"SWEEP {i + 1}/{len(frames)} — {where}" + (f"  id={h}" if h else ""), b64))
    return items


def segment_items(out: dict[str, Any], handles: list[str] | None = None
                  ) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seg = out.get("_frames_leg") or []
    steps = out.get("_leg_frame_steps") or []
    acts = out.get("_actions") or []
    n_steps = len(acts) or len(seg)
    base = out.get("steps_at_start") or 0
    for i, b64 in enumerate(seg):
        k = steps[i] if i < len(steps) else i
        act = MOTOR.get(acts[k], "?") if k < len(acts) else "?"
        when = "START" if i == 0 else ("END" if i == len(seg) - 1 else "during")
        segh = [x for x in (handles or []) if "_seg" in str(x)]
        h = segh[i] if i < len(segh) else None
        items.append((f"this dispatch {i + 1}/{len(seg)} · {when} · step {k + 1}/{n_steps} "
                      f"(episode step {base + k + 1}) · robot went {act}"
                      + (f"  id={h}" if h else ""), b64))
    items += current_items(out, handles)
    return items


# ── prompts ───────────────────────────────────────────────────────────

_TOOL = """\
Your only actuator is a small 2B navigation policy. You hand it one line of \
English; it drives the robot for up to 50 steps and hands control back. It sees \
its own camera and keeps its own visual history of the episode, but it has no \
idea what you told it last time and no idea how far through the mission it is. \
It cannot backtrack.

Measured over 900+ dispatches of this policy:
- Instructions of 4-12 words move the robot 93% of the time. 17+ word \
descriptions move it only 51% of the time. **Keep it short.**
- Naming something already in view can read as *already satisfied*, so it stops \
without moving. Aim at where you want it to END UP.
- Do not describe what lies beyond the destination. The policy walks to whatever \
the sentence describes last.

Turning costs no distance — the episode is scored on where the robot is \
standing, not where it is looking. Forward motion does cost.\
"""

BOOTSTRAP_SYSTEM = f"""\
You are directing a robot through a building to carry out one navigation \
mission. You cannot move or look around yourself.

{_TOOL}

You are shown four views from where the robot stands now — front, left, back, \
right — before it has moved. The back view matters: these missions often start \
with the robot facing away from where it must go.

Do three things.

**1. Split the mission into clauses** — the things it asks for, in order. These \
are not instructions to send; they are the checklist you will mark off, so that \
"arrived" can mean "the last one is done" rather than "we stopped somewhere".

**2. Identify the termination test** — the place the robot must END UP. Not a \
movement; the test for being finished.

**3. Write the first instruction and your opening beliefs.** A belief is a \
record, not a sentence: say what you claim, whether you *observed* it or \
*inferred* it, and cite the ids of the frames that support it.

Return ONLY this JSON object, no prose and no code fence:

{{
  "clauses": ["walk past the dining table", "go through the kitchen", "stop at the sink"],
  "terminate": "the place the robot must end up",
  "state": {{
    "beliefs": [{{"claim": "a kitchen counter is visible ahead-left",
                 "kind": "observed", "evidence_ids": ["obs_000_left"], "confidence": 0.8}}],
    "current_place": "where it is standing, in relative terms",
    "next_objective": "what has to happen next"
  }},
  "next_instruction": "the first line to send the policy — 4 to 12 words"
}}\
"""

REVIEW_SYSTEM = f"""\
You are directing a robot through a building. You cannot move or look around \
yourself; the images below are your only eyes.

{_TOOL}

## What you are looking at

Your **state block** — the mission's clauses and their status, your beliefs with \
their ids and evidence, and the harness's dispatch ledger, which is written from \
telemetry and which you cannot edit.

Then **the walk the dispatch you just sent actually did** — sampled from that \
segment only — and the **four views at the pose it ended in**. Every image has \
an `id=`. Cite those ids when you claim something.

These images are gone next turn. They are on disk, but they never re-enter your \
context. What survives is what you write into the state block.

## How to judge

**Read `stop_reason` first:**
- `policy_stop` with ≤2 steps and almost no displacement → the line did not read \
  as something to go and do. Re-aim at something visible.
- `policy_stop` with real displacement → **it believes it finished, and that \
  belief deserves weight.** Overriding it needs positive evidence you can SEE.
- `budget` → cut off mid-movement at the step cap. Not a decision. Never finish here.
- `driven` → those were the harness's own motor actions.

**Then judge the walk, not the endpoint.** Read the frames against the step \
numbers and actions: a segment that reads `R R R R F F` went somewhere the last \
frame alone will not show you.

## Writing state

Beliefs are typed records. `observed` means you can point at a frame. `inferred` \
means you reasoned it. **`negative` — "the goal is not that way" — is the \
dangerous one**: nothing ever revisits a ruled-out direction, so a wrong one can \
remove the correct route for the rest of the episode. A negative belief is \
**refused** unless you cite frame ids or the robot has just swept 360°.

You may `confirm`, `contradict` or `retire` earlier beliefs **by id**. Use this \
instead of restating everything: a belief you leave alone simply stays.

Mark clause progress as you go. That is what tells the harness whether you are \
on the mission's last leg.

## Deciding

- `continue` — send another instruction (4-12 words).
- `drive`    — take the wheel; return `actions` instead of an instruction.
- `finish`   — you believe the termination test is satisfied; it will be verified.

Recovering is not free. A correction issued somewhere that was actually correct \
walks the robot away from a mission it had already completed — measured, this \
has turned finished episodes into failures. When ambiguous, stopping beats \
wandering.

Return ONLY this JSON object, no prose and no code fence:

{{
  "receipt": {{
    "claim": "one sentence: what this dispatch accomplished",
    "evidence_ids": ["ids of the frames that show it"],
    "incidental": "anything else worth noting that you were not looking for",
    "not_done": "what this dispatch did NOT establish"
  }},
  "verdict": "done" | "not_started" | "partial" | "off_route",
  "decision": "continue" | "drive" | "finish",
  "next_instruction": "4-12 words, one movement with a destination; null otherwise",
  "actions": ["TURN_LEFT", "MOVE_FORWARD"],
  "state": {{
    "beliefs": [{{"claim": "...", "kind": "observed|inferred|negative",
                 "evidence_ids": ["..."], "confidence": 0.0}}],
    "confirm": ["b3"], "contradict": ["b7"], "retire": [],
    "progress": [{{"clause": 0, "status": "done"}}],
    "current_place": "...", "next_objective": "...", "near_goal": true
  }},
  "contradiction": "REQUIRED if overriding a policy_stop that moved: what you SEE that contradicts the mission. Empty if none."
}}\
"""

VERIFY_SYSTEM = """\
You are checking whether a robot has finished its navigation mission. The \
episode ends the moment this is accepted, and it is scored on distance to the \
goal at that instant. There is no undo.

**Report what you observe. You are not deciding.** The harness combines your \
three judgements below with the rest of the episode's evidence. Answer each one \
on its own — do not collapse them into a single feeling about whether you are \
"there".

**1. `target_visible`** — can you actually SEE the thing the termination test \
names, and say which frame it is in?

**2. `relation_holds`** — is the robot in the spatial relation the mission \
describes? "At the corner of the bar" means beside that corner, not looking at \
it from across the room. "Near the sink" means in the sink's part of the room.

**3. `close_enough`** — is the robot within a few metres of that place? This is \
the one that decides the score, and it is **not** the same as seeing it. \
Measured on 100 episodes: verifications were rejected while the robot stood a \
median of 1.8 m from the goal, having already swept the full circle — because \
the landmark could not be named from where it stood. **A landmark you cannot \
name may still be one you are standing next to.** Judge distance from scale, \
floor space and doorways, not from recognition.

## You can look around first, and it is free

`"look": true` gets you the full 360° in eight frames. No motor actions, no \
step budget, no change of position — the simulator renders those views. So "I \
cannot see it" is never an answer; it is a reason to sweep.

## A small correction, if it is genuinely a small one

`"step"` executes up to 2 raw actions and then re-observes and asks you again. \
Turning costs nothing positionally. **Forward motion can walk the robot out of \
a success radius it is already inside** — measured, episodes where the verifier \
drove scored 41% where the policy alone scored 65% on the same episodes. Ask for \
forward only when you can see the target and it is clearly further than a couple \
of metres.

Return ONLY this JSON object, no prose and no code fence:

{
  "target_visible": true or false,
  "which_frame": "id of the frame it is in, or empty",
  "relation_holds": true or false,
  "close_enough": true or false,
  "confidence": 0.0 to 1.0,
  "why": "one sentence naming what you see",
  "contradiction": "what you SEE that says this is NOT the described place; empty if none",
  "look": true or false,
  "step": ["TURN_LEFT"]
}\
"""


# ── the three calls ───────────────────────────────────────────────────

def bootstrap(mission: str, views: dict[str, Any], handles: dict[str, str],
              model: str) -> tuple[dict[str, Any] | None, str, str, dict]:
    blocks, report = _pack(current_items(views, handles))
    content = [{"type": "text", "text":
                f"MISSION:\n{mission.strip()}\n\nWhat the robot can see from where it "
                f"stands now — front, left, back and right, before it has moved:"},
               *blocks]
    got, raw, thinking, meta = ask(BOOTSTRAP_SYSTEM, content, model,
                                   require=("terminate", "next_instruction"))
    report["call_meta"] = meta
    return got, raw, thinking, report


def review(state, out: dict[str, Any], actions: list[str], handles: list[str],
           model: str) -> tuple[dict[str, Any] | None, str, str, dict]:
    tel = {k: v for k, v in out.items() if not k.startswith("_") and k != "_goal_m"}
    tel.pop("_goal_m", None)
    header = (
        f"{state.render()}\n\n"
        f"── THE DISPATCH THAT JUST FINISHED ──\n"
        f"You sent: {tel.get('sub_instruction')!r}\n"
        f"Motor trail: {compact_actions(actions) or '(nothing)'}\n"
        f"Telemetry: {json.dumps(tel, ensure_ascii=False)}\n\n"
        f"Below: the walk THIS dispatch did, then the four views at the pose it "
        f"ended in. Nothing from earlier dispatches — that is in the state block."
    )
    blocks, report = _pack(segment_items(out, handles))
    got, raw, thinking, meta = ask(REVIEW_SYSTEM, [{"type": "text", "text": header}, *blocks],
                                   model, require=("decision",))
    report["call_meta"] = meta
    return got, raw, thinking, report


def verify(state, views: dict[str, Any], handles, model: str, *,
           sweep: dict[str, Any] | None = None, can_look: bool = True
           ) -> tuple[dict[str, Any] | None, str, str, dict]:
    if sweep:
        items = sweep_items(sweep, handles)
        head = ("The full 360° sweep from where the robot is standing, in 45° steps. "
                "This cost nothing and the robot has not moved.")
        avail = ""
    else:
        items = current_items(views, handles)
        head = "The four views from where the robot is standing right now:"
        avail = ("\nYou have not swept yet. `\"look\": true` gets the full 360° free.\n"
                 if can_look else "\nYou have already swept the full 360°. Report.\n")
    blocks, report = _pack(items)
    content = [{"type": "text", "text":
                f"MISSION: {state.mission.strip()}\n"
                f"TERMINATION TEST: {state.terminate or '(none was identified)'}\n\n"
                f"{state.render()}\n{avail}\n{head}"}, *blocks]
    got, raw, thinking, meta = ask(VERIFY_SYSTEM, content, model,
                                   require=("close_enough", "relation_holds"))
    report["call_meta"] = meta
    return got, raw, thinking, report


# ── the arrival rule lives here, in code, not in the prompt ───────────

VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
VERIFIED_FAILURE = "VERIFIED_FAILURE"
UNRESOLVED_STOP = "UNRESOLVED_STOP"


def adjudicate(v: dict[str, Any] | None, *, swept: bool) -> tuple[str, str]:
    """Combine the verifier's three observations into a verdict. Returns
    (verdict, why).

    Calibrated against the 100-episode confusion matrix rather than intuition:

      * The 17 false rejections were all within 3 m, all after a full sweep, and
        all failed on *recognition* — so `close_enough` alone, once the robot has
        actually looked around, is sufficient. Requiring the landmark to be
        named is what threw those away.
      * The 20 false accepts sat at a median 6.96 m with 7 beyond 8 m — plausible
        pictures at the wrong distance. So `target_visible` on its own is never
        sufficient; distance has to be asserted too.
      * A named, visible contradiction still overrides everything. That rule was
        paid for by an episode that finished in a bathroom hallway.
    """
    if v is None:
        return UNRESOLVED_STOP, "verification never returned a usable answer"
    if str(v.get("contradiction") or "").strip():
        return VERIFIED_FAILURE, f"contradiction: {v['contradiction']}"
    close = bool(v.get("close_enough"))
    rel = bool(v.get("relation_holds"))
    seen = bool(v.get("target_visible"))
    if close and (rel or seen or swept):
        return VERIFIED_SUCCESS, f"close_enough (relation={rel} visible={seen} swept={swept})"
    if rel and seen:
        return VERIFIED_SUCCESS, "relation holds and target visible"
    if not close and not rel:
        return VERIFIED_FAILURE, "neither close enough nor in the described relation"
    return VERIFIED_FAILURE, f"insufficient: close={close} relation={rel} visible={seen}"
