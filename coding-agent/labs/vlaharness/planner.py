from __future__ import annotations

"""The outer planner — a frontier model driving the 2B VLA as a tool.

Loop shape (one episode):

    decompose the instruction into its own clauses
      └─► execute(clause)            VLA rolls out; its STOP ends the rollout
            └─► planner reads the frames it stopped on + telemetry, then:
                  continue  → execute(next clause)
                  recover   → execute(a corrective instruction)
                  finish    → fire the env STOP

Three things the prompt enforces, all of them measured rather than assumed:

  * **Legs are authored, not sliced.** The rule-based L1 arm could only return
    verbatim spans and lost 17 points to the policy alone — a leg cut at the
    wrong seam ("wait the archway", "stove and stop near the sink") is worse
    than no decomposition. So the planner writes each leg as one movement with
    a destination, keeping R2R register and the instruction's landmarks.
  * **A ≤2-step, zero-displacement STOP is not a completion.** 21% of L1's 299
    dispatches came back that way: the STOP token was trained on whole
    instructions, so a short leg reads as already-satisfied. Advancing past one
    of these is how L1 killed its own routes; the prompt names the signature.
  * **Intent is checked against the traversed route, not the stopping frame.**
    A robot can stop somewhere plausible having got there the wrong way.

A manual tool-use loop rather than the SDK tool runner: the loop's terminal
condition is *simulator* state (the env can exhaust its step budget mid-rollout
and end the episode underneath the planner), not model state, and every call
has to land in the episode trace for later failure analysis.

last updated: 2026-08-06
"""

import json
import os
from typing import Any

PLANNER_SYSTEM = """\
You are navigating a building by directing a small trained navigation policy. \
You cannot move or look around yourself — the policy is your only actuator, and \
the images it returns are your only eyes.

## What the policy is, and how it fails

It is a 2B vision-language model trained on this exact benchmark. Inside one leg \
of a route it is better at low-level navigation than you are: it reads the scene, \
picks headings, avoids furniture. What it does not have is any memory across your \
dispatches, any ability to backtrack, or any idea which part of the route it is \
on. Each dispatch, it wakes up with only the text you hand it.

**Its STOP was trained on whole instructions, so a short leg often makes it stop \
without moving at all.** Measured on this benchmark: roughly one dispatch in five \
comes back having taken ≤2 steps with essentially zero displacement. That is not \
a refusal, and it is not "leg complete" — it means the text you sent did not read \
as something to go and do. Re-target it; do not advance past it.

## Your tools

`execute(sub_instruction)` runs the policy for up to 50 steps on one instruction \
and returns:
  · **this leg** — forward views from its start, middle and end
  · **left and right** at the pose where it stopped
  · **the route so far** — the whole episode's forward views, uniformly sampled
  · telemetry: steps used, net displacement, action histogram, guard flags, \
    steps remaining in the episode's hard budget

It never ends the episode.

`finish()` declares arrival and permanently ends the episode. Scored on whether \
the robot is within 3 metres of the instruction's endpoint at that moment.

## Choosing the next leg — this is not sentence-splitting

Do not chop the instruction into clauses and read them out in order. Decide, each \
time, what the robot should do **next given where it actually is**. That might be \
the next thing the instruction describes, or the same goal re-phrased because the \
last phrasing did not take, or a corrective move the instruction never mentions.

Write every leg as **one movement with a destination**: "walk down the hallway to \
the double doors", "turn right and cross the living room to the fireplace".

- Never dispatch a bare "stop", "continue", or "keep going" — a policy with no \
  memory cannot act on those.
- Never dispatch a leg with no target. "Go forward" is worse than "go forward \
  through the doorway ahead".
- Keep the instruction's own landmarks and its plain, imperative register. You \
  may repair the wording, complete an elliptical phrase, merge two fragments, or \
  re-aim a leg at something you can currently see. Do not invent landmarks that \
  are not in the instruction and not in front of you.

## Reading what came back — work through this every time

**1. Check `stop_reason` first.**
  · `policy_stop` with `steps_used` ≤ 2 and `net_displacement_m` ≈ 0 → the leg \
    did not read as actionable. **Do not advance.** Re-issue the same goal as a \
    concrete movement toward something visible, or turn to face it first.
  · `policy_stop` with real displacement → it believes it finished. Verify it.
  · `budget` → it was cut off at 50 steps mid-leg and was still moving. This is \
    not a decision of any kind. Re-dispatch to continue. **Never `finish()` here.**
  · `env_done` → the episode's own budget is gone; it is already over.

**2. Judge against the route, not the endpoint.** Look at the sampled route so \
far, not just the frame it stopped on. A robot can end up somewhere that looks \
plausible having got there entirely the wrong way, and the last frame alone will \
not show you that. Ask: does this path look like the route the instruction \
describes — the same rooms, in the same order?

**3. Say what you see, then act.** One or two sentences: name the landmarks you \
can actually identify, and whether they match the leg you dispatched. If you \
cannot name anything, say so — that is itself evidence you are off-route.

Then choose one of three:
  · **continue** — that leg is done and route remains → dispatch the next leg.
  · **recover** — lost, stuck, or somewhere the instruction never described → \
    dispatch a corrective movement, then resume the route.
  · **finish** — the robot is at the destination the instruction's final leg \
    names, and you can point to that landmark in the current views. Say which \
    landmark before you call it. If you cannot name it, you are not there.

## Do not thrash

Correcting has a cost and a risk. Every dispatch spends from a hard episode step \
budget, and an episode that runs out mid-correction is scored wherever it happens \
to be standing. Worse, over-correcting can walk a rollout that was going fine off \
its route entirely.

So: if two consecutive recoveries have not changed what you see, stop correcting \
and commit to the most likely direction. When `steps_remaining` gets low, stop \
exploring and finish at your best guess — an unfired `finish()` scores zero.
"""

TOOLS = [
    {
        "name": "execute",
        "description": (
            "Run the navigation policy for up to 50 steps on one sub-instruction. "
            "Returns this leg's forward views (start/middle/end), the left and right "
            "views at the stopping pose, the whole episode's route so far as sampled "
            "forward views, and telemetry (steps used, net displacement, action "
            "histogram, guard flags, steps remaining). Does NOT end the episode — "
            "the policy's own stop only ends this rollout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sub_instruction": {
                    "type": "string",
                    "description": (
                        "One movement with a destination for the policy to carry "
                        "out, phrased like the source instruction. Not a bare "
                        "'stop'/'continue', and not a leg with no target."
                    ),
                }
            },
            "required": ["sub_instruction"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": (
            "Declare the robot has reached the instruction's endpoint and end the "
            "episode. Irreversible; scored on distance to the goal at this moment."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def build_client(model: str):
    """Resolve credentials the way the SDK does, then fall back to the Claude
    Code OAuth profile on disk (which the SDK does not read on its own).

    OAuth tokens go on ``Authorization: Bearer`` plus the ``oauth-2025-04-20``
    beta header — they are not API keys and fail on ``x-api-key``.
    """
    import anthropic

    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(), "api_key"

    cred_path = os.path.expanduser("~/.claude/.credentials.json")
    if os.path.isfile(cred_path):
        with open(cred_path) as f:
            oauth = json.load(f).get("claudeAiOauth") or {}
        token = oauth.get("accessToken")
        if token:
            return (
                anthropic.Anthropic(
                    auth_token=token,
                    default_headers={"anthropic-beta": "oauth-2025-04-20"},
                ),
                f"oauth:{oauth.get('subscriptionType', '?')}",
            )
    raise RuntimeError(
        "no credentials — set ANTHROPIC_API_KEY or log in with Claude Code"
    )


def _tool_result_content(out: dict[str, Any]) -> list[dict[str, Any]]:
    """Frames as image blocks, everything else as one JSON text block."""
    frames = out.pop("_frames_front", None) or []
    left = out.pop("_frame_left", None)
    right = out.pop("_frame_right", None)

    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(out)}]
    for i, b64 in enumerate(frames):
        label = "final forward view" if i == len(frames) - 1 else f"forward view −{len(frames) - 1 - i}"
        content.append({"type": "text", "text": f"[{label}]"})
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
        )
    for label, b64 in (("left at stopping pose", left), ("right at stopping pose", right)):
        if b64:
            content.append({"type": "text", "text": f"[{label}]"})
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
            )
    return content


_SENT_END = (". ", "! ", "? ")
_CONNECTIVES = (", and then ", " and then ", ", then ", ", and ", " and ")


def split_clauses(instruction: str, max_words: int = 12) -> list[str]:
    """Deterministic clause split — the L1 control's decomposer.

    No model in the loop: sentences first, then long sentences broken at the
    connectives R2R instructions actually use. Spans are returned verbatim so
    the policy sees its own training phrasing, which is the same constraint the
    planner prompt puts on the agent arm.
    """
    text = instruction.strip()
    if not text:
        return []
    sentences, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= 2 and buf[-2:] in _SENT_END:
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    out: list[str] = []
    for s in sentences:
        if len(s.split()) <= max_words:
            out.append(s)
            continue
        parts, rest = [], s
        for conn in _CONNECTIVES:
            if conn in rest:
                head, _, tail = rest.partition(conn)
                if head.split() and tail.split():
                    parts = [head.strip(), tail.strip()]
                    break
        out.extend(parts or [s])
    return [c for c in out if c.split()]


def run_episode(
    client,
    tools_impl,
    instruction: str,
    *,
    model: str = "claude-sonnet-5",
    effort: str = "high",
    max_turns: int = 30,
    mode: str = "planner",
    live=None,
    legs: list[str] | None = None,
    log=None,
) -> dict[str, Any]:
    """Drive one episode. Returns the trace.

    ``mode`` selects the arm:
      ``whole``    the instruction goes to the policy once, its STOP honoured.
                   A parity check against the policy-alone baseline — if this
                   doesn't reproduce it, the harness itself changed something.
      ``clauses``  L1: legs are decided offline (by ``legs``, from either the
                   model decomposer or the rule splitter) and dispatched in
                   order, advancing on the policy's STOP. No model in the loop.
                   Isolates how much of any gain is the decomposition itself.
      ``planner``  L3: the frontier model decomposes, judges each stop, recovers.
    """
    trace: dict[str, Any] = {"turns": 0, "tool_calls": [], "planner_text": []}

    def _record(sub: str, out: dict[str, Any]) -> dict[str, Any]:
        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        trace["tool_calls"].append(
            {"name": "execute", "input": {"sub_instruction": sub}, "result": clean}
        )
        if live is not None:
            frames = out.get("_frames_front") or []
            from .live import shrink_for_view

            live.execute(sub, clean, shrink_for_view(frames[-1]) if frames else None)
        return clean

    if mode in ("whole", "clauses"):
        subs = [instruction] if mode == "whole" else (legs or split_clauses(instruction))
        subs = subs or [instruction]
        trace["legs"] = subs
        if mode == "clauses":
            if live is not None:
                live.plan("offline split: " + " | ".join(subs))
            if log is not None:
                log.emit("subgoal_parse", segments=subs)
        for sub in subs:
            if tools_impl.episode_over:
                break
            if log is not None:
                log.emit("tool_use", name="execute", input={"sub_instruction": sub})
            out = tools_impl.execute(sub)
            clean = _record(sub, out)
            if log is not None:
                log.emit("tool_result", name="execute", result=clean,
                         frames=log.frames(out) or None)
        if not tools_impl.episode_over:
            out = tools_impl.finish()
            trace["tool_calls"].append({"name": "finish", "input": {}})
            if log is not None:
                log.emit("tool_use", name="finish", input={})
                log.emit("tool_result", name="finish", result=out)
            if live is not None:
                live.finish()
        return trace

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Navigate this instruction:\n\n{instruction}\n\n"
                "Break it into its clauses and dispatch the first one."
            ),
        }
    ]

    for _ in range(max_turns):
        trace["turns"] += 1
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=PLANNER_SYSTEM,
            output_config={"effort": effort},
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            trace["error"] = "planner_refusal"
            break

        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace["planner_text"].append(block.text.strip())
                if live is not None:
                    live.plan(block.text)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # Planner ended its turn without acting; nudge once, then give up.
            if response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not ended the episode. Either execute the next "
                            "clause, execute a corrective instruction, or call finish()."
                        ),
                    }
                )
                continue
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            if tu.name == "execute":
                sub = tu.input.get("sub_instruction", "")
                out = tools_impl.execute(sub)
                _record(sub, out)
                results.append(
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": _tool_result_content(out)}
                )
            elif tu.name == "finish":
                out = tools_impl.finish()
                trace["tool_calls"].append({"name": "finish", "input": {}, "result": out})
                if live is not None:
                    live.finish()
                results.append(
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": json.dumps(out)}
                )
            else:
                results.append(
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": f"unknown tool {tu.name}", "is_error": True}
                )
        messages.append({"role": "user", "content": results})

        if tools_impl.episode_over:
            break

    # A planner that ran out of turns without finishing scores zero otherwise —
    # fire STOP where it stands so the episode is at least evaluated.
    if not tools_impl.episode_over:
        tools_impl.finish()
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": True})
        trace["forced_finish"] = True
        if live is not None:
            live.finish(forced=True)

    return trace
