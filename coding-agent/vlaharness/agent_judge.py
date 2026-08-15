from __future__ import annotations

"""The outer agent's three calls: bootstrap once, review after each dispatch,
verify before the episode may end.

What comes back from a dispatch is meant to read the way a finished subagent
reads: **10 frames uniformly sampled over the walk this dispatch just did, plus
the four views at the pose it stopped in** — 14 images. Sampled the same way the
2B policy subsamples its own visual history, but scoped to the segment, because
the segment is what is being judged. Where the robot has been across the whole
episode lives in the state block as prose, not as re-sent pixels.

Four views, not three. A robot standing where it was sent has the thing it came
for behind it as often as in front, and a 270-degree blind spot makes "the
landmark is not here" indistinguishable from "the landmark is not in frame" —
which is the single mistake that has cost this harness the most finished
episodes. The back view is free: a panorama is a render, not a move.

Native resolution AND near-lossless quality. The rig renders 720x640 and that is
what goes over — downscaling a navigation frame throws away exactly the
far-field detail a judgment about "did it reach the archway" rests on.

An earlier version of this file squeezed JPEG quality to fit a 1 MiB budget,
citing the SDK's stdio limit. That was wrong, and the correction is worth
recording because the wrong version *looked* well-founded — it had measured
byte counts in it. ``max_buffer_size`` (default 1 MiB) guards the SDK **reading
the CLI's stdout**. Images sent as a user message go the other way, over stdin,
and are never framed by that guard. Measured against the real transport, at the
default cap:

    13 frames, native, q95     3.12 MB    ok
    13 frames, native, q100    5.47 MB    ok
    13 frames, 1440px, q95     7.93 MB    ok
    26 frames, 1440px, q95    15.85 MB    ok

The limit is real, but it binds the *conversational* arm, where frames ride back
inside an MCP tool result and the CLI echoes them onto stdout. It never bound
this one.

So the ladder below is a backstop, not a budget: it starts near-lossless and
only steps down if a payload somehow passes a cap set an order of magnitude
above anything a dispatch produces. Image token cost depends on pixel
dimensions, not file size, so quality here is free in tokens — it costs only
transfer time (~1s per extra MB).

The frames themselves are written to disk at full quality by the event log and
never re-enter any context. This is the only place pixels are ever in the
window, and they leave with the session that saw them.

last updated: 2026-08-09
"""

import asyncio
import base64
import io
import json
import os
from typing import Any

from vlaharness.judge import _extract_obj
from vlaharness.state import MOTOR, compact_actions


# ── asking, with the failure mode made visible ───────────────────────
# A session can come back with no text at all — an API error, an overload, a
# turn that produced only thinking. The obvious `async for ... break on
# ResultMessage` loop cannot tell that apart from a real answer, so an empty
# reply becomes a parsed-nothing fallback and the fallback becomes a decision.
# Measured: one episode took three empty replies in a row, and the harness's
# arrival gate promoted the first of them to "finish". So: read is_error, retry
# once, and hand the caller a flag it must check.

async def _ask_once(system: str, content: Any, model: str
                    ) -> tuple[str, str, dict[str, Any]]:
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                                  ResultMessage, TextBlock, ThinkingBlock)

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    options = ClaudeAgentOptions(
        model=model, system_prompt=system, allowed_tools=[],
        permission_mode="bypassPermissions",
        # NOT 1. A session that needs a second turn to finish its answer dies
        # with subtype `error_max_turns` and returns no text at all — which is
        # indistinguishable, downstream, from a model that had nothing to say.
        # That is exactly how an episode ended on a verification that never ran.
        # There are no tools here, so extra turns cannot loop; they only buy the
        # model room to finish.
        max_turns=4, setting_sources=[], env=env,
        # Adaptive thinking, summarized. These are genuinely hard visual calls —
        # "did this segment carry out what I asked" against ten frames and a
        # motor trail — and the reasoning is also the only window anyone has
        # into WHY an episode went the way it did. Summarized rather than
        # omitted so it reaches the event log; it is archived, never replayed.
        thinking={"type": "adaptive", "display": "summarized"})

    said: list[str] = []
    # Captured for the log ONLY. It is never fed back into a later call — that
    # would put the previous turn's reasoning into this turn's context and undo
    # the whole O(1) arrangement. What crosses turns is the state block the
    # model wrote deliberately, not the working-out behind it.
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
                        # Without this the context length is unrecoverable: with
                        # prompt caching, `input_tokens` counts only what was
                        # neither cached nor being cached, and it sits at 2 while
                        # the 14 images — ~606 tokens each — land entirely here.
                        "cache_write_tok": use.get("cache_creation_input_tokens"),
                        "ctx_tok": (use.get("input_tokens") or 0)
                                   + (use.get("cache_read_input_tokens") or 0)
                                   + (use.get("cache_creation_input_tokens") or 0)}
                break
    return "\n".join(said), "\n".join(t for t in thought if t.strip()), meta


def ask(system: str, content: Any, model: str, retries: int = 1
        ) -> tuple[str, str, dict[str, Any]]:
    """Returns (text, thinking, diagnostics). `diagnostics['failed']` is the flag
    callers must check before treating a missing answer as an answer."""
    last: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            text, thinking, meta = asyncio.run(_ask_once(system, content, model))
        except Exception as exc:  # noqa: BLE001
            text, thinking, meta = "", "", {"exception": repr(exc)[:200]}
        meta["attempt"] = attempt + 1
        meta["thinking_chars"] = len(thinking)
        if text.strip() and not meta.get("is_error"):
            meta["failed"] = False
            return text, thinking, meta
        last = meta
        last["empty_reply"] = not text.strip()
    last["failed"] = True
    return "", "", last

# ── payload budgeting ────────────────────────────────────────────────
# A backstop against a pathological payload, not a budget — 13 native frames at
# q92 come to roughly 1 MB, and 15.85 MB was measured going over intact. Nothing
# a dispatch produces should ever reach this, and if something does, the report
# says so rather than quietly shrinking the evidence.
MAX_IMAGE_B64 = 12_000_000
QUALITY_LADDER = (92, 85, 78, 70, 62)


def _encode(b64: str, quality: int) -> tuple[str, int] | None:
    """Re-encode at NATIVE resolution. Only quality moves."""
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
    """Encode labelled frames under the transport cap. Returns (blocks, report).

    Quality is shared across the batch so the segment and the current views are
    judged at the same fidelity. If even the floor of the ladder does not fit,
    the OLDEST frames go first: items are laid down last-first, so what survives
    is the end of the segment and the views the robot is standing in — the
    evidence a decision actually rests on.
    """
    report: dict[str, Any] = {"n_requested": len(items)}
    chosen: list[tuple[str, str]] | None = None
    quality = QUALITY_LADDER[-1]

    for q in QUALITY_LADDER:
        encoded = [(label, _encode(b64, q)) for label, b64 in items]
        encoded = [(label, got) for label, got in encoded if got]
        total = sum(got[1] for _, got in encoded)
        if total <= MAX_IMAGE_B64:
            chosen, quality = [(label, got[0]) for label, got in encoded], q
            report.update(quality=q, bytes_b64=total, dropped=0)
            break

    if chosen is None:                          # floor quality still too big
        q = QUALITY_LADDER[-1]
        encoded = [(label, _encode(b64, q)) for label, b64 in items]
        encoded = [(label, got) for label, got in encoded if got]
        keep, total, dropped = [], 0, 0
        # Walk from the end so the current views and the newest route frames win.
        for label, got in reversed(encoded):
            if total + got[1] > MAX_IMAGE_B64:
                dropped += 1
                continue
            total += got[1]
            keep.insert(0, (label, got[0]))
        chosen, quality = keep, q
        report.update(quality=q, bytes_b64=total, dropped=dropped)

    blocks: list[dict[str, Any]] = []
    for label, data in chosen:
        blocks.append({"type": "text", "text": f"[{label}]"})
        blocks.append({"type": "image",
                       "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    report["n_sent"] = len(chosen)
    return blocks, report


CURRENT = (("FRONT", "_frame_front"), ("LEFT", "_frame_left"),
           ("BACK", "_frame_back"), ("RIGHT", "_frame_right"))


def current_items(views: dict[str, Any]) -> list[tuple[str, str]]:
    """The four views at the current pose. All free — a panorama is a render."""
    return [(f"CURRENT VIEW — {tag} (0 steps to obtain)", views[key])
            for tag, key in CURRENT if views.get(key)]


def sweep_items(look: dict[str, Any]) -> list[tuple[str, str]]:
    """A full circle at the current pose, labelled by heading offset."""
    frames = look.get("_sweep") or []
    degs = look.get("_sweep_deg") or []
    items = []
    for i, b64 in enumerate(frames):
        d = degs[i] if i < len(degs) else None
        if d == 0:
            where = "straight ahead"
        elif d == 180:
            where = "directly behind"
        elif d is None:
            where = f"view {i + 1}"
        else:
            where = f"{d}° to the LEFT" if d < 180 else f"{360 - d}° to the RIGHT"
        items.append((f"SWEEP {i + 1}/{len(frames)} — {where}", b64))
    return items


def frame_items(out: dict[str, Any]) -> list[tuple[str, str]]:
    """The 13: **this dispatch's** walk, then the three views it ended in.

    Segment-scoped, not episode-scoped. Sampling the whole episode instead looks
    reasonable and is quietly wrong: the question on the table is "how did the
    subtask I just sent go", and ten frames spread over the whole route give that
    subtask a share of the pictures proportional to its share of the episode. A
    four-step correction on turn eight got one frame out of ten, nine of which
    showed ground already judged on earlier turns and already compressed into the
    state block. Episode-scale memory is the state block's job; these frames are
    the subtask's own transcript, the way a subagent returns its own work.
    """
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
        items.append((f"this dispatch {i + 1}/{len(seg)} · {when} · "
                      f"step {k + 1} of {n_steps} in this dispatch "
                      f"(episode step {base + k + 1}) · robot went {act}", b64))
    items += current_items(out)
    return items


# ── prompts ──────────────────────────────────────────────────────────

_TOOL = """\
Your only actuator is a small 2B navigation policy. You hand it one line of \
English; it drives the robot for up to 50 steps and hands control back. It sees \
its own camera and keeps its own visual history of the episode, but it has no \
idea what you told it last time and no idea how far through the mission it is. \
It cannot backtrack.

Measured over 744 dispatches of this policy:
- About 30% of dispatches come back with the robot not having moved at all, at \
every instruction length. This is a property of the policy, not of your wording.
- Instructions of 4-12 words have the lowest no-movement rate (26-31%). Both \
bare 2-word commands and 17+ word descriptions sit near 40%.
- Naming something the robot can already see can read as *already satisfied*, \
so it stops without moving. Aim at where you want it to END UP.
- Do not describe what lies beyond the destination. The policy walks to whatever \
the sentence describes last: "walk to the archway" stopped correctly at 38 \
steps; the same line with the room beyond the archway appended ran to the step \
cap and overshot by 9 metres.

You may also take the wheel yourself with raw motor actions. MOVE_FORWARD is \
**0.25 m**; TURN_LEFT / TURN_RIGHT are **15°**, so a right angle is six turns \
and a half turn is twelve. Up to 12 actions at a time, and you are blind while \
they run.

**Turning does not move the robot.** A turn changes only which way it faces, so \
it cannot make the mission's score worse — the episode is scored on where the \
robot is standing, not where it is looking. Turning to look costs step budget \
and nothing else. MOVE_FORWARD does move it, and that can cost. Use this when the policy is wedged — measured, ten consecutive \
dispatches with ten different phrasings and zero displacement — or for a small \
obvious move that needs no navigation. Do not try to walk a route this way.\
"""

BOOTSTRAP_SYSTEM = f"""\
You are directing a robot through a building to carry out one navigation \
mission. You cannot move or look around yourself.

{_TOOL}

You are being shown four views from where the robot is standing right now, \
before it has moved: **front, left, back and right**. The back view matters — \
these missions sometimes start with the robot facing away from where it has to \
go, and a leg aimed at a wall dies immediately.

Do two things.

**1. Identify the termination test.** Most of these missions end with a sentence \
like "stop at the corner of the bar" or "wait by the stairs". That is not \
another movement to perform — it is the test for being finished, and it is what \
you will later have to verify. Pull it out as a place.

**2. Write the opening state block and the first instruction.** The state block \
is your entire memory of this episode. You rewrite all of it every turn; nothing \
carries over except what you write. Right now most of it is unknown, and saying \
so is correct — do not invent.

Return ONLY this JSON object, no prose and no code fence:

{{
  "terminate": "the place the robot must end up for this to count as done",
  "state": {{
    "progress":  "what of the mission is done so far — at this point, nothing",
    "here":      "where the robot is standing, from what you can see",
    "layout":    "the space as you understand it: what is where, relative to what",
    "ruled_out": "anything you can already tell is not the way",
    "remaining": "what still has to happen, in order"
  }},
  "next_instruction": "the first line to send the policy — one movement with a destination"
}}\
"""

REVIEW_SYSTEM = f"""\
You are directing a robot through a building. You cannot move or look around \
yourself; the images below are your only eyes.

{_TOOL}

## What you are looking at

First your **state block** — everything you knew at the end of last turn, \
written by you — and the **dispatch ledger**, which the harness writes from \
telemetry and you cannot edit.

Then **the walk the dispatch you just sent actually did** — uniformly sampled \
from that segment only, labelled with the step number within the dispatch and \
the motor action taken at each frame — and finally the **four views at the pose \
it is standing in now**: front, left, back and right. Those four are free; the \
simulator renders them without the robot moving.

These frames cover this dispatch and nothing before it. Where the robot has been \
earlier in the episode is in the state block, in your own words, because you \
already looked at those frames on an earlier turn and wrote down what mattered.

These images will be gone next turn. They are written to disk but they never \
come back into your context. Anything worth keeping has to go into the state \
block through your answer.

## How to judge the dispatch that just finished

**Read `stop_reason` first:**
- `policy_stop` with 2 or fewer steps and almost no displacement → the line did \
  not read as something to go and do. Not a refusal, not a completion. Re-aim at \
  something you can see, or drive.
- `policy_stop` with real displacement → **it believes it finished, and that \
  belief deserves weight.** This policy alone completes 57% of these missions and \
  its stop token was trained for exactly this moment. Overriding it needs \
  positive evidence the robot is somewhere the mission never described — \
  something you can SEE that contradicts it. "I cannot identify the landmark" is \
  NOT such evidence: these are three narrow views and it can be behind the robot. \
  Absence of confirmation is not contradiction.
- `budget` → cut off at the step cap mid-movement. Not a decision of any kind. \
  Re-issue. **Never finish here.**
- `driven` → those were your own motor actions, not the policy's.

**Then judge the walk, not the endpoint.** Read the sampled frames against the \
step numbers and actions. A robot can stop somewhere plausible having gone \
entirely the wrong way; the last frame will not show you that, but a segment \
that reads `R R R R F F` across four frames will.

## Rewriting the state block

Rewrite all five fields every turn. This is not an append — you are producing \
the version that replaces the old one, so carry forward what still matters, drop \
what has stopped mattering, and correct what you now know was wrong. Keep each \
field to roughly two sentences.

`layout` is the one that pays off: it is the spatial understanding you dispatch \
from next turn. Relations, not coordinates — "the bar runs left-to-right ahead, \
the pool room is behind us through the archway".

Anything you leave out of a field is forgotten. A field you omit entirely keeps \
its previous value.

## Deciding

Exactly one of:
- `continue` — send another instruction and keep going.
- `drive`    — take the wheel; return `actions` instead of an instruction.
- `finish`   — you believe the termination test in DONE WHEN is satisfied. You \
  will be asked to verify before the episode ends.

Recovering is not free and it is not the safe default. Every correction spends \
the step budget, and a correction issued somewhere that was actually correct \
walks the robot away from a mission it had already completed — measured, this \
exact mistake has turned finished episodes into failures. When the evidence is \
ambiguous, stopping beats wandering. When the budget runs low, finish at your \
best guess; an episode that never finishes scores zero.

Return ONLY this JSON object, no prose and no code fence:

{{
  "saw": "one sentence naming what you can actually identify in these frames",
  "verdict": "done" | "not_started" | "partial" | "off_route",
  "decision": "continue" | "drive" | "finish",
  "next_instruction": "one movement with a destination — for continue; null otherwise",
  "actions": ["TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD"],
  "state": {{
    "progress":  "...", "here": "...", "layout": "...",
    "ruled_out": "...", "remaining": "..."
  }},
  "near_goal": true or false,
  "contradiction": "REQUIRED if you are overriding a policy_stop that moved: name what you can SEE that contradicts the mission. Empty if you have none."
}}

`actions` is required when decision is `drive` and must be empty otherwise.\
"""

VERIFY_SYSTEM = """\
You are verifying, one last time, whether a robot has finished its navigation \
mission. The episode ends the moment you say yes, and it is scored on distance \
to the goal at that instant. There is no undo.

You are shown the mission, the termination test, the state block, and the three \
views from where the robot is standing right now.

Answer the narrow question: **does what you can see satisfy the termination \
test?** Name the thing. "I think we are probably near it" is a no.

## You can look around before answering, and it is free

You are given front, left, back and right. That is four fixed frames; a landmark \
can still sit between two of them, or be small in one.

**Set `"look": true` and you get the full 360° in eight frames, 45° apart.** It \
costs nothing at all — no motor actions, no step budget, no change of position. \
The simulator renders those views; the robot does not move to obtain them.

So "I cannot see the landmark" is never an answer here. It is a reason to sweep. \
Only conclude after you have looked at every direction the landmark could be in.

If, after sweeping, the thing is in view but a short distance off, you can return \
`step` — a few raw motor actions the harness executes directly. MOVE_FORWARD is \
0.25 m, TURN_LEFT / TURN_RIGHT are 15°. Unlike looking, forward motion **does** \
move the robot and can cost you distance to the goal; turning does not.

## The two asymmetries

Stopping in the wrong place scores zero, so do not rubber-stamp. But this robot \
cannot backtrack, and a rejection that sends it wandering away from a spot it \
had actually reached is the more expensive error — it has already cost finished \
episodes. So once you have looked: reject only if you can point at something you \
can SEE that says this is NOT the described place.

Return ONLY this JSON object, no prose and no code fence:

{
  "look": true to be shown the full 360° sweep before deciding — free, use it rather than guessing,
  "step": ["TURN_LEFT", "MOVE_FORWARD"] — raw actions for a small final adjustment, else [],
  "satisfied": true or false,
  "why": "one sentence — what you want to check, or what you see, or what contradicts",
  "next_instruction": "if not satisfied and not looking or stepping: the single English movement that would get there; null otherwise"
}

`next_instruction` goes to the navigation policy and must be an English \
sentence. Motor actions belong in `step` — the policy is a language model and \
cannot act on the word "MOVE_FORWARD".\
"""


# ── calls ────────────────────────────────────────────────────────────

def bootstrap(mission: str, views: dict[str, Any],
              model: str = "claude-sonnet-5"
              ) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """One call at episode start: termination test + opening state + first line."""
    blocks, report = _pack(current_items(views))
    content = [{"type": "text", "text":
                f"MISSION:\n{mission.strip()}\n\n"
                f"What the robot can see from where it stands now — front, left, "
                f"back and right, before it has moved a step:"},
               *blocks]
    raw, thinking, meta = ask(BOOTSTRAP_SYSTEM, content, model)
    report["call_meta"] = meta
    got = _extract_obj(raw)
    if not got:
        # The worst of the three to lose silently: with no answer there is no
        # termination test, and `verify` then guards an episode against a
        # condition of "(none was identified)" — it cannot reject anything, so
        # the last gate quietly stops being a gate. Flagged so the row shows it.
        return {"parse_failed": True}, raw, thinking, report
    return got, raw, thinking, report


def review(state, out: dict[str, Any], actions: list[str],
           model: str = "claude-sonnet-5"
           ) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """One call per dispatch. Fresh session; the 13 frames live only here."""
    telemetry = {k: v for k, v in out.items() if not k.startswith("_")}
    header = (
        f"{state.render()}\n\n"
        f"── THE DISPATCH THAT JUST FINISHED ──\n"
        f"You sent: {telemetry.get('sub_instruction')!r}\n"
        f"Motor trail: {compact_actions(actions) or '(nothing)'}\n"
        f"Telemetry: {json.dumps(telemetry, ensure_ascii=False)}\n\n"
        f"Below: the walk THIS dispatch did, then the three views at the pose it "
        f"ended in. Nothing from earlier dispatches — that is in the state block."
    )
    blocks, report = _pack(frame_items(out))
    content = [{"type": "text", "text": header}, *blocks]
    raw, thinking, meta = ask(REVIEW_SYSTEM, content, model)
    report["call_meta"] = meta
    got = _extract_obj(raw)
    if not got:
        # No answer is not an answer. `parse_failed` is what stops the harness's
        # arrival gate from promoting this to a termination — without it, a
        # silent API error reads as "the model declined to object".
        return ({"decision": "continue", "parse_failed": True,
                 "saw": f"<no usable review: {json.dumps(meta, default=str)}>"},
                raw, thinking, report)
    return got, raw, thinking, report


def verify(state, views: dict[str, Any], model: str = "claude-sonnet-5",
           sweep: dict[str, Any] | None = None,
           can_look: bool = True) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """The last gate before STOP fires — with a free 360° look available first.

    The failure this exists to stop is the verifier concluding "not there" from
    a handful of narrow views that simply do not contain the landmark. It can
    point the camera anywhere, at no cost: a panorama is a render, not a move,
    so it spends no env steps and gives up no distance to the goal.
    """
    if sweep:
        items = sweep_items(sweep)
        header = ("The full 360° sweep from where the robot is standing, in 45° "
                  "steps. This cost nothing and the robot has not moved.")
        avail = ""
    else:
        items = current_items(views)
        header = "The four views from where the robot is standing right now:"
        avail = ("\nYou have not swept yet. `\"look\": true` gets you the full 360° "
                 "in eight frames, for free.\n" if can_look else
                 "\nYou have already swept the full 360°. Decide.\n")
    blocks, report = _pack(items)
    content = [{"type": "text", "text":
                f"MISSION: {state.mission.strip()}\n"
                f"TERMINATION TEST: {state.terminate or '(none was identified)'}\n\n"
                f"{state.render()}\n{avail}\n{header}"},
               *blocks]
    raw, thinking, meta = ask(VERIFY_SYSTEM, content, model)
    report["call_meta"] = meta
    got = _extract_obj(raw)
    if not got:
        # A verifier that cannot run must not block termination — an episode
        # that never fires STOP scores zero, so the fallback accepts. But it is
        # marked, so a run can be audited for how many episodes ended on a
        # verification that never actually happened.
        return ({"satisfied": True, "parse_failed": True,
                 "why": f"<no usable verification: {json.dumps(meta, default=str)}>"},
                raw, thinking, report)
    return got, raw, thinking, report
