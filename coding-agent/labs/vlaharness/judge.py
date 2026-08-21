from __future__ import annotations

"""The stateful L3 planner: route planning, and an O(1)-context judge.

The conversational L3 kept one session alive for the whole episode, so every
dispatch's images stayed in context forever. By dispatch 25 that is most of the
window, the useful signal is buried, and the episodes that need the most
recovery are exactly the ones whose context is worst.

Here the loop is inverted. **Every decision is a fresh session** carrying:

    · the rendered state block — route, segments done, places, landmarks,
      what has been ruled out, budget
    · only the CURRENT dispatch's frames, each labelled with the motor action
      the policy took at that moment
    · that dispatch's telemetry

Images never accumulate: the pixels are dropped the moment the judgment is
made, and what survives is prose the judge itself wrote into the state block.
That is the compression contract — the frames are evidence for one decision,
the state block is the memory.

Two calls, both structured, both single-turn:

    plan_route()  once per episode — legs, plus the termination condition as a
                  separate field. A route decomposed into N legs with the last
                  sentence used as leg N has no stopping test at all.
    judge()       after every dispatch — look back, decide, and propose the
                  belief update on the same call (no extra round trip).

last updated: 2026-08-07
"""

import asyncio
import json
import os
import re
from typing import Any

from vlaharness.state import MOTOR

PLAN_SYSTEM = """\
You are planning a route for a robot through a building.

The robot is driven by a small navigation policy that sees only its camera and \
one line of text at a time. It has no memory between lines, cannot backtrack, \
and has no idea how far through the route it is.

**You are also shown what the robot can see right now, from its starting pose.** \
Use it to choose *which* target each leg names and *what order* the legs go in — \
an instruction says "walk into the kitchen", and the views tell you whether the \
kitchen is straight ahead or through a doorway on the left.

**Use it to pick the target, not to pad the wording.** Legs stay short: 4 to 12 \
words. Do not append what you can see to the leg text, and above all do not \
describe what lies *beyond* the leg's endpoint — the policy walks to whatever \
the sentence describes last, so naming the room visible through the archway \
sends it past the archway. Measured: "Walk across the floor to the archway" \
stopped correctly at 38 steps; the same leg with the far room appended ran to \
the step cap and overshot by 9 metres.

Where the views and the instruction disagree, the instruction is the task.

Split the instruction into **legs** — the movements a person would make one at \
a time — and separate out the **termination condition**.

The termination condition is a *place*, not a movement: where the robot must \
end up for the task to count as done. Most instructions end with a sentence \
like "stop at the corner of the bar" or "wait by the stairs". That sentence is \
not another leg to walk; it is the test for being finished. Pull it out.

Rules for legs:
- Each leg is one movement with a destination: "walk down the hallway to the \
  double doors". Never a bare "continue" or "stop" — the policy has no memory \
  and cannot act on those.
- Keep the instruction's own landmarks and its plain imperative register. You \
  may repair broken wording, complete an elliptical phrase, or merge a fragment \
  into its neighbour. Do not invent landmarks that are not in the instruction.
- Few legs, not many — usually 2 to 4. A leg that is too short reads as \
  already-satisfied to the policy, which stops without moving and the route dies.
- **4 to 12 words each.** Measured over 744 dispatches: that band has the lowest \
  no-movement rate (26-31%); both 1-3 word commands and 17+ word descriptions sit \
  near 40%. Each leg names its own endpoint and stops there.

Return ONLY this JSON object, no prose and no code fence:
{"legs": ["...", "..."], "terminate": "..."}\
"""

JUDGE_SYSTEM = """\
You are directing a robot through a building. You cannot move or look around \
yourself. A small navigation policy is your only actuator, and the images below \
are your only eyes.

You have just sent it one instruction and it has come back. Your job now is to \
look at what actually happened and decide what to do next.

## What you are looking at

You get the state block — everything known about this episode so far, written \
by you on previous turns — and then the frames from the dispatch that just \
finished, each labelled with the motor action the policy took at that moment \
(F = forward, L = turn left, R = turn right, S = stop). Reading the frames \
alongside the actions tells you not just where it ended up but **how it got \
there**: whether it walked a corridor or spun on the spot, whether it went \
through the doorway or turned away from it.

The frames you are seeing now will be gone next turn. Anything worth keeping \
has to go into the state block through your answer.

## How to judge

**First, read `stop_reason`:**
- `policy_stop` with ≤2 steps and almost no displacement → the instruction did \
  not read as something to go and do. This is not a refusal and not a \
  completion; it happens to roughly one dispatch in five. **Do not advance the \
  route.** Re-aim the same goal at something visible: name a door, a corridor, \
  a piece of furniture you can see, or turn to face it first.
- `policy_stop` with real displacement → **it believes it finished, and that \
  belief deserves weight.** This policy alone completes 57% of these routes, and \
  its stop token was trained for exactly this moment. Overriding it needs \
  positive evidence that the robot is somewhere the route never described — \
  something you can SEE that contradicts the instruction. "I cannot identify \
  the landmark" is **not** such evidence: these are three narrow views, and the \
  landmark can be behind the robot, beside it, or simply out of frame. Absence \
  of confirmation is not contradiction.
- `budget` → it was cut off at the step cap mid-movement and was still going. \
  Not a decision of any kind. Re-issue to continue. **Never finish here.**
- `env_done` → the episode is already over.

**Then judge the movement, not the endpoint.** Compare what you asked for \
against the motor trail and the frames. A robot can stop somewhere that looks \
plausible having gone entirely the wrong way — the last frame alone will not \
show you that, but "R×12 F×2" will.

**Then decide.** Exactly one of:
- `continue` — that leg is genuinely done; move to the next leg.
- `recover` — lost, or somewhere the instruction never described. Send a \
  corrective movement for the policy to carry out, then resume.
- `drive` — **you take the wheel yourself.** Instead of an instruction, you \
  return a short list of raw motor actions and the harness executes them \
  directly, bypassing the policy.
- `finish` — the termination condition in the state block is satisfied. Only \
  choose this if you can **name the landmark from that condition in the frames \
  you are looking at right now**. If you cannot name it, you are not there.

## When to drive yourself

The policy is better than you at real navigation — crossing rooms, following \
corridors, avoiding furniture. Leave that to it. But there are poses where it \
will not move **for any wording**: measured, ten consecutive dispatches with ten \
different phrasings and zero displacement, while the same robot under the full \
original instruction simply walked on. When the actuator is wedged, asking it \
more politely is not a recovery.

`drive` is for exactly that, and for small obvious moves that need no \
navigation at all:

- **Unsticking.** Two dispatches in a row with no displacement → drive.
- **Aiming.** The thing you want is clearly in the left view but the policy \
  will not turn → turn it yourself, then hand back.
- **A last short hop.** The destination is a couple of metres straight ahead \
  and in frame → drive there rather than negotiating.

The motor space is coarse and you must count in it:
- `MOVE_FORWARD` = **0.25 m**. Four of them is one metre.
- `TURN_LEFT` / `TURN_RIGHT` = **15°**. A right angle is **six** turns; a \
  half turn is twelve.

So facing something in your left view takes roughly `["TURN_LEFT"] × 6`, and \
then a few `MOVE_FORWARD` to approach it. Up to 12 actions per drive. You are \
blind while driving — you only see the result afterwards — so drive in short \
bursts and look before continuing. Do not try to walk a whole route this way; \
hand back to the policy as soon as the pose is unstuck.

**Recovering is not free, and it is not the safe default.** Every correction \
spends the episode's step budget, and a correction issued at a place that was \
actually correct walks the robot away from a route it had already completed. \
Measured: this exact mistake has turned finished episodes into failures. When \
the evidence is ambiguous, **stopping is the better bet than wandering** — the \
policy got there on its own, and you cannot re-find a place this robot cannot \
backtrack to.

If the state block says recoveries are stacking up with no progress, stop \
correcting and commit. When the step budget runs low, finish at your best \
guess — an episode that never finishes scores zero.

## Your answer

Return ONLY this JSON object, no prose and no code fence:

{
  "saw": "one sentence naming what you can actually identify in these frames",
  "leg_verdict": "done" | "not_started" | "partial" | "off_route",
  "decision": "continue" | "recover" | "drive" | "finish",
  "next_instruction": "one movement with a destination — for continue/recover; null otherwise",
  "actions": ["TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD"],
  "current_place": "where the robot is, in relative terms — no coordinates",
  "surroundings": "what is around it",
  "landmarks": {"name of thing you identified": "where it is relative to the robot"},
  "ruled_out": ["things you now know are NOT here"],
  "lesson": "if something went wrong and you fixed it, one line — else empty",
  "near_goal": true or false,
  "contradiction": "REQUIRED when decision is 'recover' on the last leg: name what you can SEE that contradicts the route. Empty if you have none."
}

`actions` is REQUIRED when decision is `drive`, and must be empty otherwise. \
`saw` is not optional and must not be a guess. If you cannot identify anything, \
say exactly that — but remember it means your view is uninformative, not that \
the robot is lost.\
"""


def _extract_obj(text: str) -> dict[str, Any] | None:
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.+?)```", text, re.S)):
        m = re.search(r"\{.*\}", candidate, re.S)
        if not m:
            continue
        try:
            got = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(got, dict):
            return got
    return None


def _options(system: str, model: str):
    from claude_agent_sdk import ClaudeAgentOptions

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        allowed_tools=[],          # no tools — this session only judges
        permission_mode="bypassPermissions",
        max_turns=1,
        setting_sources=[],
        env=env,
    )


async def _ask(system: str, content: Any, model: str) -> str:
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock

    said: list[str] = []
    async with ClaudeSDKClient(options=_options(system, model)) as client:
        if isinstance(content, str):
            await client.query(content)
        else:
            async def stream():
                yield {"type": "user", "message": {"role": "user", "content": content},
                       "parent_tool_use_id": None}

            await client.query(stream())
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                said += [b.text for b in msg.content if isinstance(b, TextBlock)]
            elif isinstance(msg, ResultMessage):
                break
    return "\n".join(said)


# ── route planning ────────────────────────────────────────────────────

def plan_route(instruction: str, model: str = "claude-sonnet-5",
               views: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan from the instruction AND the starting view, not the text alone."""
    if views:
        content = [{"type": "text", "text":
                    f"INSTRUCTION:\n{instruction.strip()}\n\n"
                    f"What the robot can see from where it is standing now:"},
                   *_frame_blocks(views, [])]
    else:
        content = instruction.strip()
    try:
        raw = asyncio.run(_ask(PLAN_SYSTEM, content, model))
    except Exception as exc:  # noqa: BLE001
        raw = f"<error {exc!r}>"
    got = _extract_obj(raw) or {}
    legs = [str(x).strip() for x in (got.get("legs") or []) if str(x).strip()]
    if not legs:
        from vlaharness.planner import split_clauses

        legs = split_clauses(instruction) or [instruction.strip()]
    return {"legs": legs, "terminate": str(got.get("terminate") or "").strip(), "raw": raw}


# ── per-dispatch judgment ─────────────────────────────────────────────

def _frame_blocks(out: dict[str, Any], actions: list[str]) -> list[dict[str, Any]]:
    """This dispatch's frames, each labelled with the motor action taken there.

    The policy's k-th action is what it chose while looking at the k-th frame,
    so the pairing is exact — that is the whole point of showing them together.
    """
    from vlaharness.planner_sdk import _compress

    leg = out.get("_frames_leg") or []
    steps = out.get("_leg_frame_steps") or []
    items: list[tuple[str, str]] = []
    for i, b64 in enumerate(leg):
        k = steps[i] if i < len(steps) else None
        act = MOTOR.get(actions[k], "?") if (k is not None and k < len(actions)) else "?"
        when = ("start" if i == 0 else "end" if i == len(leg) - 1 else "during")
        pos = f"step {k + 1}/{len(actions)}" if k is not None else when
        items.append((f"{when} of this leg — {pos}, policy chose {act}", b64))
    for label, key in (("looking LEFT from where it stopped", "_frame_left"),
                       ("looking RIGHT from where it stopped", "_frame_right")):
        if out.get(key):
            items.append((label, out[key]))

    blocks: list[dict[str, Any]] = []
    total = 0
    for label, b64 in reversed(items):
        got = _compress(b64)
        if got is None:
            continue
        data, mime = got
        if total + len(data) > 600_000:
            continue
        total += len(data)
        blocks[:0] = [
            {"type": "text", "text": f"[{label}]"},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
        ]
    return blocks


def judge(state, out: dict[str, Any], actions: list[str],
          model: str = "claude-sonnet-5") -> tuple[dict[str, Any], str]:
    """One decision, one fresh session. Returns (parsed, raw_reply)."""
    from vlaharness.state import compact_actions

    telemetry = {k: v for k, v in out.items() if not k.startswith("_")}
    header = (
        f"{state.render()}\n\n"
        f"── THE DISPATCH THAT JUST FINISHED ──\n"
        f"You sent: {telemetry.get('sub_instruction')!r}\n"
        f"Motor trail: {compact_actions(actions) or '(nothing)'}\n"
        f"Telemetry: {json.dumps(telemetry, ensure_ascii=False)}\n\n"
        f"The frames below are from this dispatch, in order, each labelled with "
        f"the action the policy chose while looking at it."
    )
    content = [{"type": "text", "text": header}, *_frame_blocks(out, actions)]
    try:
        raw = asyncio.run(_ask(JUDGE_SYSTEM, content, model))
    except Exception as exc:  # noqa: BLE001
        return {"decision": "recover", "next_instruction": None,
                "saw": f"<judge error {exc!r}>"}, repr(exc)
    got = _extract_obj(raw)
    if not got:
        # An unreadable judgment must not silently become "finish".
        return {"decision": "recover", "next_instruction": None,
                "saw": "<unparseable judgment>"}, raw
    return got, raw
