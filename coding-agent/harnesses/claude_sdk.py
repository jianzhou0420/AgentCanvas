"""Claude Agent SDK adapter — Anthropic's closed scaffolding.

Session block ported verbatim from beta-coding-agent/run_episodes.py (the
legacy driver keeps its frozen copy for provenance). Auth rides the logged-in
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

from driver import BRIDGE_PATH, EpisodeContext, EventSink, SessionOutcome, json_safe


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
            "thinking": "adaptive+summarized (4.6+/5) / enabled+budget (haiku)",
            "turn_cap": "hard (SDK max_turns)",
        }

    def prepare(self) -> None:
        if os.environ.pop("ANTHROPIC_API_KEY", None):
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
        if "haiku" in (ctx.model or ""):
            return {"type": "enabled",
                    "budget_tokens": ctx.extra.get("think_budget", 4000),
                    "display": "summarized"}
        return {"type": "adaptive", "display": "summarized"}

    def _options(self, ctx: EpisodeContext) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=ctx.briefing,
            mcp_servers={
                "env": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(BRIDGE_PATH)],
                    "env": ctx.bridge_env(),
                }
            },
            tools=[],  # no built-in tools: vanilla ReAct over the env only
            # No filesystem settings: without this the CLI walks up from cwd
            # and injects the repo CLAUDE.md into every session.
            setting_sources=[],
            thinking=self._thinking_config(ctx),
            betas=ctx.extra.get("betas", []),
            # ONLY our bridge — never the user's global MCP config.
            strict_mcp_config=True,
            allowed_tools=(
                ["mcp__env__observe", "mcp__env__step"]
                if ctx.bare
                else ["mcp__env__observe", "mcp__env__step", "mcp__env__look_around"]
            ),
            permission_mode="bypassPermissions",
            # look_around() returns four images in one MCP message; the default
            # 1 MiB stdout buffer truncates it and kills the session mid-parse
            max_buffer_size=32 * 1024 * 1024,
            max_turns=ctx.max_turns,
            model=ctx.model,
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

        return SessionOutcome(
            usage=json_safe(getattr(result_msg, "usage", None)),
            cost_usd=getattr(result_msg, "total_cost_usd", None),
            turns=getattr(result_msg, "num_turns", None),
            error=("sdk result is_error" if getattr(result_msg, "is_error", False) else None),
            extra={"duration_ms": getattr(result_msg, "duration_ms", None)},
        )
