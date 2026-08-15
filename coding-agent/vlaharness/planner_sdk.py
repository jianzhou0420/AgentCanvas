from __future__ import annotations

"""The L3 planner, driven through the Claude Agent SDK.

Why not the Messages API: the Claude Code OAuth credential is not an API key.
Put on ``Authorization: Bearer`` against ``/v1/messages`` it answers 429 on the
first request, for Sonnet and Opus alike — it is scoped to the Claude Code
harness, not to raw API traffic. The SDK talks to that harness, which is the
path the rest of this repo's subscription-billed runs already use (see
``harnesses/claude_sdk.py``). So the planner is an SDK session whose only tools
are the two VLA tools, exposed as an in-process MCP server.

Everything else is unchanged from the Messages-API version: the same system
prompt, the same two tools, the same rule that the policy's STOP is evidence
rather than a decision. Only the transport moved.

A stray ANTHROPIC_API_KEY would silently switch billing from the subscription
to the API, so it is scrubbed from the child environment.

last updated: 2026-08-06
"""

import asyncio
import json
import os
from typing import Any

from vlaharness.planner import PLANNER_SYSTEM

EXECUTE_DESC = (
    "Run the navigation policy for up to 50 steps on one sub-instruction — one "
    "movement with a destination, phrased like the source instruction. Returns "
    "the whole episode's route so far as sampled forward views, this leg's own "
    "start/middle/end views, the left and right views at the stopping pose, and "
    "telemetry (steps used, net displacement, action histogram, guard flags, "
    "steps remaining). Does NOT end the episode — the policy's own stop only ends this "
    "rollout."
)
FINISH_DESC = (
    "Declare the robot has reached the instruction's endpoint and end the episode. "
    "Irreversible; scored on distance to the goal at this moment."
)


class _Episode:
    """Whatever the tool handlers need for the episode currently in flight.

    The MCP tools are registered once per process but the toolset is rebuilt per
    episode, so the handlers read the live one off this holder rather than
    closing over a stale reference.
    """

    tools: Any = None
    trace: dict[str, Any] | None = None
    live: Any = None


_EP = _Episode()


# The SDK talks to the harness over stdio and rejects any single JSON message
# above 1 MiB. Five 512px PNGs are ~500 KB of base64 and blow straight through
# it — the session dies the moment the first execute returns, which reads from
# the outside exactly like a planner that dispatched one clause and gave up.
# So frames are re-encoded to JPEG at viewing size before they go over the wire.
VIEW_PX = 288
VIEW_QUALITY = 72
MAX_PAYLOAD_B = 600_000


def _compress(b64: str) -> tuple[str, str] | None:
    try:
        import base64 as _b64
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(_b64.b64decode(b64))).convert("RGB")
        if max(img.size) > VIEW_PX:
            s = VIEW_PX / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=VIEW_QUALITY)
        return _b64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception:
        return None


def _image_blocks(out: dict[str, Any]) -> list[dict[str, Any]]:
    """Three timescales, in the order the planner should read them: the route it
    has walked, then what this leg did, then where it is now facing.

    Priority matters — everything is squeezed under one payload cap, and if
    something has to be dropped it should be a middle frame of the route, not
    the view the stop judgment rests on. So items are laid down last-first and
    the cap bites the oldest.
    """
    items: list[tuple[str, str]] = []

    path = out.get("_frames_path") or []
    for i, b64 in enumerate(path):
        items.append((f"route so far {i + 1}/{len(path)}"
                      + (" — where it is now" if i == len(path) - 1 else ""), b64))

    leg = out.get("_frames_leg") or []
    for i, b64 in enumerate(leg):
        where = "start of this leg" if i == 0 else (
            "end of this leg" if i == len(leg) - 1 else f"during this leg {i}")
        items.append((where, b64))

    for label, key in (("looking left from where it stopped", "_frame_left"),
                       ("looking right from where it stopped", "_frame_right")):
        if out.get(key):
            items.append((label, out[key]))

    blocks: list[dict[str, Any]] = []
    total = 0
    for label, b64 in reversed(items):
        got = _compress(b64)
        if got is None:
            continue
        data, mime = got
        if total + len(data) > MAX_PAYLOAD_B:
            continue
        total += len(data)
        blocks[:0] = [
            {"type": "text", "text": f"[{label}]"},
            {"type": "image", "data": data, "mimeType": mime},
        ]
    return blocks


def build_tools():
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("execute", EXECUTE_DESC, {"sub_instruction": str})
    async def execute(args: dict[str, Any]) -> dict[str, Any]:
        sub = args.get("sub_instruction", "")
        out = await asyncio.to_thread(_EP.tools.execute, sub)
        clean = {k: v for k, v in out.items() if not k.startswith("_")}
        _EP.trace["tool_calls"].append(
            {"name": "execute", "input": {"sub_instruction": sub}, "result": clean}
        )
        if _EP.live is not None:
            from vlaharness.live import shrink_for_view

            frames = out.get("_frames_leg") or []
            _EP.live.execute(sub, clean, shrink_for_view(frames[-1]) if frames else None)
        if _EP.log is not None:
            _EP.log.emit("tool_use", name="execute", input={"sub_instruction": sub})
            _EP.log.emit("tool_result", name="execute", result=clean,
                         frames=_EP.log.frames(out) or None)
        return {"content": [{"type": "text", "text": json.dumps(clean)}, *_image_blocks(out)]}

    @tool("finish", FINISH_DESC, {})
    async def finish(args: dict[str, Any]) -> dict[str, Any]:
        out = await asyncio.to_thread(_EP.tools.finish)
        _EP.trace["tool_calls"].append({"name": "finish", "input": {}, "result": out})
        if _EP.live is not None:
            _EP.live.finish()
        if _EP.log is not None:
            _EP.log.emit("tool_use", name="finish", input={})
            _EP.log.emit("tool_result", name="finish", result=out)
        return {"content": [{"type": "text", "text": json.dumps(out)}]}

    return create_sdk_mcp_server(name="vla", version="1.0.0", tools=[execute, finish])


def build_options(model: str, effort: str, max_turns: int, cwd: str):
    from claude_agent_sdk import ClaudeAgentOptions

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return ClaudeAgentOptions(
        model=model,
        system_prompt=PLANNER_SYSTEM,
        mcp_servers={"vla": build_tools()},
        # Only the two VLA tools. The planner has no filesystem, no bash, no web —
        # its sole actuator is the policy, which is what the prompt tells it.
        allowed_tools=["mcp__vla__execute", "mcp__vla__finish"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        # Don't inherit the user's CLAUDE.md / settings into a benchmark run.
        setting_sources=[],
        cwd=cwd,
        env=env,
    )


async def _run(options, instruction: str) -> dict[str, Any]:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
    )

    trace = _EP.trace
    prompt = (
        f"Navigate this instruction:\n\n{instruction}\n\n"
        "Break it into its clauses and dispatch the first one."
    )
    if _EP.log is not None:
        _EP.log.emit("session_inputs", harness="vla", model=options.model,
                     system_prompt=options.system_prompt, first_prompt=prompt,
                     tools=list(options.allowed_tools or []), max_turns=options.max_turns)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                trace["turns"] += 1
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        think = getattr(block, "thinking", "") or ""
                        if think.strip() and _EP.log is not None:
                            _EP.log.emit("thinking", chars=len(think), text=think)
                    elif isinstance(block, TextBlock) and block.text.strip():
                        trace["planner_text"].append(block.text.strip())
                        if _EP.live is not None:
                            _EP.live.plan(block.text)
                        if _EP.log is not None:
                            _EP.log.emit("assistant_text", text=block.text)
            elif isinstance(msg, ResultMessage):
                trace["result"] = {
                    "is_error": getattr(msg, "is_error", None),
                    "num_turns": getattr(msg, "num_turns", None),
                    "cost_usd": getattr(msg, "total_cost_usd", None),
                }
                if _EP.log is not None:
                    _EP.log.emit("session_result", **trace["result"])
                # The episode can end mid-rollout underneath the planner, so a
                # session that returns while the episode is still open is not an
                # error — the caller fires the fallback STOP.
                break
    return trace


def run_episode_sdk(
    tools_impl,
    instruction: str,
    *,
    model: str = "claude-sonnet-5",
    effort: str = "high",
    max_turns: int = 30,
    live=None,
    log=None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Drive one episode through an SDK session. Same trace shape as the
    Messages-API planner, so both write identical result rows."""
    trace: dict[str, Any] = {"turns": 0, "tool_calls": [], "planner_text": []}
    _EP.tools, _EP.trace, _EP.live, _EP.log = tools_impl, trace, live, log

    options = build_options(model, effort, max_turns, cwd or os.getcwd())
    try:
        asyncio.run(_run(options, instruction))
    except Exception as exc:  # noqa: BLE001 — a dead session must still be scored
        trace["error"] = repr(exc)

    # A planner that ran out of turns, errored, or ended its turn without acting
    # scores zero otherwise — fire STOP where it stands so the episode is at
    # least evaluated.
    if not tools_impl.episode_over:
        tools_impl.finish()
        trace["tool_calls"].append({"name": "finish", "input": {}, "forced": True})
        trace["forced_finish"] = True
        if live is not None:
            live.finish(forced=True)
    return trace
