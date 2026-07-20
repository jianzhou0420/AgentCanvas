"""Claude Agent SDK adapter — Anthropic's closed scaffolding.

Session block ported verbatim from the legacy SDK driver, which keeps its
frozen copy for provenance at legacy/beta-coding-agent/run_episodes.py.
Auth rides the logged-in
Claude subscription; a stray ANTHROPIC_API_KEY would silently switch billing
to the API in headless mode, so prepare() strips it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from driver import (
    EpisodeContext, EventSink, SessionOutcome, is_rate_limited, json_safe,
)


def _tool_result_texts(block: Any) -> list[str]:
    content = block.content
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    texts.append("<image elided>")
    return texts


class ClaudeSdkAdapter:
    name = "claude-sdk"

    def __init__(self) -> None:
        self.inherent: dict[str, Any] = {
            "auth": "claude subscription",
            "thinking": "adaptive+summarized (4.6+/5) / enabled+budget (haiku); "
                        "effort per cell extra",
            "turn_cap": "hard (SDK max_turns)",
            # CC/Agent-SDK keeps only a recent window of tool_result images in
            # the outgoing request (old observe()/look_around() frames are
            # evicted) — undocumented but empirically bounded: image-heavy
            # episodes stay single-digit MB, far under the API's 32 MB cap.
            # Contrast mini's image_window=0 (full history). See docs
            # developer-guide/coding-agent/harness-notes.
            "vision_context": "recent-window auto (evicts old tool_result images; ~single-digit MB)",
        }

    def prepare(self, spec) -> None:  # noqa: ARG002 — no per-cell setup needed
        # Default: strip ANTHROPIC_API_KEY so sessions use the logged-in
        # subscription (a stray key silently switches billing to the metered API
        # in headless mode). Opt-in escape hatch: STD_SDK_USE_API=1 KEEPS the key
        # so the SDK runs on metered API billing — used to sidestep the
        # subscription's 5h window / peak throttle (same model + harness + effort,
        # only the auth/billing path differs). Records the path into `inherent`.
        if os.environ.get("STD_SDK_USE_API") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
            self.inherent["auth"] = "anthropic API key (metered billing, STD_SDK_USE_API=1)"
            print("[sdk] STD_SDK_USE_API=1 — keeping ANTHROPIC_API_KEY; sessions use METERED API billing")
        elif os.environ.pop("ANTHROPIC_API_KEY", None):
            print("[sdk] ANTHROPIC_API_KEY was set — removed so sessions use subscription auth")
        try:
            import claude_agent_sdk
            self.inherent["sdk_version"] = getattr(claude_agent_sdk, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass

    # Thinking config is model-family-specific. Claude 4.6+/5 models take
    # `adaptive`; pre-4.6 (haiku-4.5) need explicit `enabled` + budget_tokens
    # and think ONCE per turn (no interleaved thinking without the beta).
    @staticmethod
    def _thinking_config(ctx: EpisodeContext) -> dict[str, Any]:
        # An explicit think_budget (wp cells set it; --set think_budget=N
        # overrides) forces enabled thinking so reasoning blocks are
        # substantive rather than adaptive one-liners.
        budget = ctx.extra.get("think_budget")
        if budget:
            return {"type": "enabled", "budget_tokens": int(budget),
                    "display": "summarized"}
        if "haiku" in (ctx.model or ""):
            return {"type": "enabled", "budget_tokens": 4000,
                    "display": "summarized"}
        return {"type": "adaptive", "display": "summarized"}

    def _options(self, ctx: EpisodeContext) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        # persona ablation: keep the stock Claude Code system prompt and
        # append the briefing, instead of replacing it wholesale
        system_prompt: Any = (
            {"type": "preset", "preset": "claude_code", "append": ctx.briefing}
            if ctx.persona else ctx.briefing
        )
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={
                "env": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(ctx.bridge_path)],
                    "env": ctx.bridge_env(),
                }
            },
            tools=[],  # no built-in tools: vanilla ReAct over the env only
            # No filesystem settings: without this the CLI walks up from cwd
            # and injects the repo CLAUDE.md into every session.
            setting_sources=[],
            thinking=self._thinking_config(ctx),
            effort=ctx.extra.get("effort"),
            betas=ctx.extra.get("betas", []),
            # ONLY our bridge — never the user's global MCP config.
            strict_mcp_config=True,
            allowed_tools=(
                ["mcp__env__observe", "mcp__env__goto", "mcp__env__stop"]
                if ctx.wp
                else ["mcp__env__observe", "mcp__env__step"]
                if ctx.bare
                else ["mcp__env__observe", "mcp__env__step", "mcp__env__look_around"]
            ),
            permission_mode="bypassPermissions",
            # look_around() returns four images in one MCP message; the default
            # 1 MiB stdout buffer truncates it and kills the session mid-parse
            max_buffer_size=32 * 1024 * 1024,
            max_turns=ctx.max_turns,
            # empty model (Monitor UI run with the field blank) → SDK default,
            # the legacy driver's semantics
            model=ctx.model or None,
            cwd=str(ctx.workdir),
        )

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"options": json_safe(self._options(ctx))}

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        raw_path = ctx.raw_dir / f"episode_{ctx.index}.jsonl"
        result_msg: Any = None

        with raw_path.open("w") as raw:

            def record_raw(message: Any) -> None:
                raw.write(json.dumps(
                    {"type": type(message).__name__, "msg": json_safe(message)},
                    ensure_ascii=False) + "\n")
                raw.flush()

            async with ClaudeSDKClient(options=self._options(ctx)) as client:
                # The CLI starts reasoning before MCP servers finish connecting;
                # gate the prompt on the bridge reporting 'connected'.
                bridge_status: str | None = None
                for _ in range(60):
                    status = await client.get_mcp_status()
                    entries = status.get("mcpServers", []) if isinstance(status, dict) else []
                    bridge_status = next(
                        (e.get("status") for e in entries if e.get("name") == "env"), None
                    )
                    if bridge_status == "connected" or bridge_status in (
                        "failed", "needs-auth", "disabled",
                    ):
                        break
                    await asyncio.sleep(0.5)
                sink.emit("bridge_status", {"status": bridge_status})
                if bridge_status != "connected":
                    raise RuntimeError(f"env bridge not connected: {bridge_status}")

                await client.query(ctx.first_prompt)
                async for message in client.receive_response():
                    record_raw(message)
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                sink.emit("assistant_text", {"text": block.text})
                            elif isinstance(block, ThinkingBlock):
                                sink.emit("thinking", {"chars": len(block.thinking),
                                                       "text": block.thinking})
                            elif isinstance(block, ToolUseBlock):
                                sink.emit("tool_use", {"id": block.id, "name": block.name,
                                                       "input": block.input})
                    elif isinstance(message, UserMessage):
                        content = message.content
                        blocks = content if isinstance(content, list) else []
                        for block in blocks:
                            if isinstance(block, ToolResultBlock):
                                sink.emit("tool_result", {
                                    "tool_use_id": block.tool_use_id,
                                    "texts": _tool_result_texts(block)})
                    elif isinstance(message, SystemMessage):
                        if getattr(message, "subtype", None) == "init":
                            data = getattr(message, "data", {}) or {}
                            sink.emit("system_init", {"model": data.get("model"),
                                                      "tools": data.get("tools")})
                    elif isinstance(message, ResultMessage):
                        result_msg = message

        # The Agent SDK sets is_error=True even on a clean subtype="success"
        # result — the flag tracks the session, not the navigation outcome, so
        # an episode that called stop and reached the goal still comes back
        # is_error=True (observed: fable ep40). is_error alone therefore
        # over-flags. Score by the ENV terminal instead: both "success" (normal
        # return, whatever the nav result) and "error_max_turns" (clean
        # truncation, like mini's step_limit) are scored outcomes — only a
        # genuine execution error (error_during_execution, or a missing
        # subtype) is a broken session that propagates as error.
        subtype = getattr(result_msg, "subtype", None)
        result_text = str(getattr(result_msg, "result", "") or "")
        error = None
        if is_rate_limited(result_text):
            # subscription throttle returns subtype="success" is_error=True with a
            # "temporarily limiting requests" result — tag retryable so the driver
            # backs off and re-runs it, never scoring it as a navigation failure.
            error = "rate_limited"
        elif (getattr(result_msg, "is_error", False)
                and subtype not in ("error_max_turns", "success")):
            error = f"sdk result {subtype or 'is_error'}"
        return SessionOutcome(
            usage=json_safe(getattr(result_msg, "usage", None)),
            cost_usd=getattr(result_msg, "total_cost_usd", None),
            turns=getattr(result_msg, "num_turns", None),
            error=error,
            extra={"duration_ms": getattr(result_msg, "duration_ms", None),
                   "result_subtype": subtype},
        )
