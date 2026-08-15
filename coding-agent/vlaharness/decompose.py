from __future__ import annotations

"""L1's decomposer — a model, and nothing else.

L1 asks one question: how much of any gain is *time abstraction* rather than
*agency*? So the model gets exactly one job — cut the instruction into legs —
and is then removed from the loop entirely. It never sees an image, never sees
what the policy did, never gets to change its mind. The rollout that follows is
pure Python: dispatch leg, advance on STOP, dispatch next.

That is the only difference from the earlier rule-based L1. If a model-authored
split beats the rule-based one, the gain was in the wording. If it still loses
to L0, decomposition itself is the problem — and the value of the agent arm has
to live in verification and recovery, not in cutting.

Legs may be *rewritten*, not just sliced. The rule-based splitter could only
return verbatim spans, and some of the spans it produced were bad ("wait the
archway"). A model can fix the seam while keeping R2R phrasing.

last updated: 2026-08-07
"""

import asyncio
import json
import os
import re

DECOMPOSE_SYSTEM = """\
You break a robot navigation instruction into the legs a person would walk one \
at a time.

Each leg is handed, on its own, to a small navigation policy that sees only the \
robot's current camera views and that one leg of text. It has no memory of the \
route and no idea which leg it is on. Between legs it is not re-oriented — it \
simply gets the next string.

Write legs that survive those conditions:

- **One movement each.** A leg is a thing to do now: cross a room, go through a \
  door, turn and follow a corridor. If a sentence contains two movements, split \
  it. If a sentence is only a fragment of one movement, join it to its neighbour.
- **Phrase them like the source.** Keep the instruction's own landmarks and \
  wording. These policies are trained on this style of English, and paraphrase \
  into some other register degrades them. You may repair the text — fix a typo, \
  complete an elliptical clause ("wait the archway" → "stop at the archway"), \
  fold in a landmark the leg needs to be actionable — but do not invent \
  landmarks that are not in the instruction, and do not restate it in your own \
  voice.
- **Make each leg carry its own target.** A leg that says only "continue" or \
  "keep going" is useless to a policy with no memory. Name where it ends: \
  "walk down the hallway to the double doors".
- **Do not emit a bare stop.** A final leg of just "stop" or "wait there" gives \
  the policy nothing to navigate toward. Attach the stopping condition to the \
  movement that reaches it: "walk to the corner of the bar and stop there".
- **Few legs, not many.** Most instructions are 2–4 legs. Over-cutting is the \
  bigger risk: a leg too short looks already-satisfied to the policy, which \
  stops immediately and the route dies. When in doubt, cut less.

Return ONLY a JSON array of strings. No prose, no code fence, no commentary.\
"""


def _extract(text: str) -> list[str] | None:
    """Pull the JSON array out, tolerating a fence or stray prose around it."""
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.+?)```", text, re.S)):
        m = re.search(r"\[.*\]", candidate, re.S)
        if not m:
            continue
        try:
            got = json.loads(m.group(0))
        except Exception:
            continue
        legs = [str(x).strip() for x in got if str(x).strip()]
        if legs:
            return legs
    return None


async def _ask(instruction: str, model: str) -> tuple[list[str] | None, str]:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
    )

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=DECOMPOSE_SYSTEM,
        # No tools at all — this model's entire contribution is the split.
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=1,
        setting_sources=[],
        env=env,
    )
    said: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(instruction.strip())
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        said.append(b.text)
            elif isinstance(msg, ResultMessage):
                break
    text = "\n".join(said)
    return _extract(text), text


def decompose(instruction: str, model: str = "claude-sonnet-5",
              fallback=None) -> tuple[list[str], str]:
    """Return (legs, raw_reply). Falls back to the rule-based splitter if the
    model returns something unparseable — a dead decomposer must not silently
    become "one leg = the whole instruction", which is a different arm (L0)."""
    try:
        legs, raw = asyncio.run(_ask(instruction, model))
    except Exception as exc:  # noqa: BLE001
        legs, raw = None, f"<error {exc!r}>"
    if legs:
        return legs, raw
    if fallback is None:
        from vlaharness.planner import split_clauses as fallback  # noqa: PLC0415
    return fallback(instruction) or [instruction], raw
