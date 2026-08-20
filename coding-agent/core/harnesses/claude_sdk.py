"""Claude Agent SDK adapter — Anthropic's closed scaffolding.

Session block ported verbatim from the legacy driver, frozen in git history
at d10591e:beta-coding-agent/run_episodes.py (gated by harnesses/mini/check_equivalence). Auth rides the logged-in
Claude subscription; a stray ANTHROPIC_API_KEY would silently switch billing
to the API in headless mode, so prepare() strips it by default.

To bill a run through the API instead (e.g. the Claude subscription plan is
about to run out), export CODING_AGENT_ALLOW_API_KEY=1 in the shell that
launches stdrun.py — prepare() then leaves ANTHROPIC_API_KEY in place and the
bundled Claude CLI subprocess picks it up at launch (there is no separate
api_key field on ClaudeAgentOptions; it's purely env-var-driven). Everything
else about the harness (prompt, tools, WP bridge) stays identical — only
billing changes, so results stay comparable to subscription-billed runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from core.driver import (EpisodeContext, EventSink, SessionOutcome,
                    is_rate_limited, json_safe)


def classify_outcome(result_msg: Any) -> str | None:
    """Map an SDK ResultMessage to the driver's error contract.

    The Agent SDK sets is_error=True even on a clean subtype="success"
    result — the flag tracks the session, not the navigation outcome, so
    an episode that called stop and reached the goal still comes back
    is_error=True (observed: fable ep40). is_error alone therefore
    over-flags. Score by the ENV terminal instead: "success" (normal
    return, whatever the nav result), "error_max_turns" (clean
    truncation, like mini's step_limit), and "error_max_budget_usd"
    (USD fuse tripped — same clean-truncation semantics) are scored
    outcomes — only a genuine execution error (error_during_execution,
    or a missing subtype) is a broken session that propagates as error.
    Keeping the fuse subtype OUT of this whitelist would make the driver
    retry the most expensive episodes — the opposite of a budget cap.
    """
    subtype = getattr(result_msg, "subtype", None)
    result_text = str(getattr(result_msg, "result", "") or "")
    if is_rate_limited(result_text):
        # subscription throttle returns subtype="success" is_error=True with a
        # "temporarily limiting requests" result — tag retryable so the driver
        # backs off and re-runs it, never scoring it as a navigation failure.
        return "rate_limited"
    if (getattr(result_msg, "is_error", False)
            and subtype not in ("error_max_turns", "error_max_budget_usd",
                                "success")):
        return f"sdk result {subtype or 'is_error'}"
    return None


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
            # CC/Agent-SDK does NOT evict old images per-turn (verified against
            # Claude Code source: applyToolResultBudget passes image tool_results
            # through as-is; the time-based clear is off by default). Images are
            # re-sent in FULL every turn, same as mini image_window=0. The only
            # backstop is full compaction (auto at ~167k tokens / reactive on a
            # media-size error), which drops the whole pre-boundary history and
            # replaces images with '[image]'. So request size grows with
            # accumulated frames until a compaction resets it — NOT a recent
            # window. See docs coding-agent/harness-notes + tmp/cc-internals.
            "vision_context": "full history re-sent each turn; compaction backstop ~167k tok (no per-turn eviction)",
            # §14.2: the SDK cannot edit its resident transcript, so the
            # 12-cycle context contract does NOT apply here — this is a
            # DIFFERENT experimental condition and must be labelled as such,
            # never passed off as the mini/solo cycle compiler.
            "context_control": False,
            "context_label": "full-history",
        }
        self._gateway = None  # litellm gateway (cross-API seats only)

    def prepare(self, spec) -> None:
        # cross-API seat: serve the foreign model through an owned litellm
        # gateway and point Claude Code at it — pure env-var wiring
        # (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN). Native cells skip this
        # branch and keep their exact historical call path.
        from core.api.proxy import gateway_for_spec
        self._gateway = gateway_for_spec(spec, tag="sdk")
        if self._gateway is not None:
            os.environ.update(self._gateway.anthropic_env())
            self.inherent["api_gateway"] = self._gateway.describe()
            self.inherent["auth"] = "litellm gateway (vendor API billing)"
            print(f"[sdk] api gateway :{self._gateway.port} — "
                  f"routes {self._gateway.describe()['routes']}")
        allow_api_key = os.environ.get("CODING_AGENT_ALLOW_API_KEY") == "1"
        if allow_api_key:
            if os.environ.get("ANTHROPIC_API_KEY"):
                print("[sdk] CODING_AGENT_ALLOW_API_KEY=1 — keeping ANTHROPIC_API_KEY, sessions bill via API")
                self.inherent["auth"] = "provider API key (litellm-free direct billing)"
            else:
                print("[sdk] CODING_AGENT_ALLOW_API_KEY=1 but ANTHROPIC_API_KEY is unset — "
                      "falling back to subscription auth")
        elif os.environ.pop("ANTHROPIC_API_KEY", None):
            print("[sdk] ANTHROPIC_API_KEY was set — removed so sessions use subscription auth "
                  "(export CODING_AGENT_ALLOW_API_KEY=1 to bill via API instead)")
        try:
            import claude_agent_sdk
            self.inherent["sdk_version"] = getattr(claude_agent_sdk, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass

    def finalize(self, run_dir) -> dict:  # noqa: ARG002
        # gateway teardown + env cleanup, so a later cell in the same batch
        # cannot silently inherit the redirected base URL
        if self._gateway is not None:
            self._gateway.stop()
            for var in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
                os.environ.pop(var, None)
            self._gateway = None
        return {}

    def refine_instruction(self, *, model: str, system: str, user: str,
                           extra: dict | None = None) -> str | None:  # noqa: ARG002
        """§17: one short tool-less SDK query (subscription auth, max_turns=1,
        no MCP, no settings) with the run's own nav model. Returns the raw
        assistant text; None on any failure — the caller falls back to the
        original instruction whole."""
        import asyncio as _aio

        async def _one() -> str:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                TextBlock,
            )
            opts = ClaudeAgentOptions(
                system_prompt=system, tools=[], setting_sources=[],
                max_turns=1, model=model or None,
                permission_mode="bypassPermissions")
            texts: list[str] = []
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(user)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for b in message.content:
                            if isinstance(b, TextBlock):
                                texts.append(b.text)
            return "\n".join(texts)

        try:
            # §20.1: 45 s fail-fast, same cap as the other two refine
            # transports — a wedged CLI costs one fallback, never minutes
            return _aio.run(_aio.wait_for(_one(), timeout=45))
        except Exception:  # noqa: BLE001 — a preprocessor never costs a run
            return None

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

        # user ruling 2026-08-14: ONE model per run — the in-episode judge
        # rides the nav model's own subscription channel, never a local
        # side-model (--set judge_model=… still overrides for ablation).
        _bridge_env = {**ctx.bridge_env(), "EH_EXECUTOR": "claude-sdk"}
        _judge_model = (str(ctx.extra.get("judge_model", ""))
                        or (ctx.model or ""))
        if _judge_model:
            _bridge_env["EH_JUDGE_MODEL"] = _judge_model

        return ClaudeAgentOptions(
            system_prompt=ctx.briefing,
            mcp_servers={
                "env": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(ctx.bridge_path)],
                    # EH_EXECUTOR: snapshot identity (§14.14) — the bridge's
                    # toolset stamps who is driving into live_snapshot.json
                    "env": _bridge_env,
                }
            },
            tools=[],  # no built-in tools: vanilla ReAct over the env only
            # No filesystem settings: without this the CLI walks up from cwd
            # and injects the repo CLAUDE.md into every session.
            setting_sources=[],
            thinking=self._thinking_config(ctx),
            # Without this line every sdk *_max cell silently runs at
            # default effort (source: 6d0bf63).
            effort=ctx.extra.get("effort"),
            betas=ctx.extra.get("betas", []),
            # ONLY our bridge — never the user's global MCP config.
            strict_mcp_config=True,
            allowed_tools=(
                # libero toolbox: atomic reads + scene locator + servo macros
                # (libero_bridge.py TOOLBOX surface); the locator is
                # get_objects (GT) or pixel_to_3d (vision) per toolbox_gt
                ["mcp__env__observe_third_person", "mcp__env__observe_wrist",
                 "mcp__env__get_state",
                 ("mcp__env__get_objects" if ctx.toolbox_gt
                  else "mcp__env__pixel_to_3d"),
                 "mcp__env__move_to", "mcp__env__gripper", "mcp__env__step"]
                if ctx.benchmark == "libero" and ctx.toolbox
                # libero bare/full share the two-tool surface (the full
                # condition adds sensors to the same tools, not new tools)
                else ["mcp__env__observe", "mcp__env__step"]
                if ctx.benchmark == "libero"
                else ["mcp__env__observe", "mcp__env__observe_waypoints",
                      "mcp__env__step", "mcp__env__goto", "mcp__env__stop"]
                if ctx.hybrid
                else ["mcp__env__observe", "mcp__env__goto", "mcp__env__stop"]
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
            # Per-episode USD fuse (hmeqa frozen: $18). The CLI checks between
            # API calls and ends the session with subtype error_max_budget_usd
            # — whitelisted below as a scored truncation, like error_max_turns.
            max_budget_usd=ctx.max_budget_usd,
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
        # §14.2 payload accounting — what the model ACTUALLY accumulated this
        # session, counted from the same stream record_raw sees. With
        # context_control=False the transcript only ever grows, so these are
        # totals, not a window.
        n_tool_results = 0
        n_images = 0
        n_state_blocks = 0
        init_tools: Any = None

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
                _deadline = float(ctx.extra.get("bridge_connect_timeout_s", 30))
                _t0 = time.monotonic()
                while True:
                    status = await client.get_mcp_status()
                    entries = status.get("mcpServers", []) if isinstance(status, dict) else []
                    bridge_status = next(
                        (e.get("status") for e in entries if e.get("name") == "env"), None
                    )
                    if bridge_status == "connected" or bridge_status in (
                        "failed", "needs-auth", "disabled",
                    ) or time.monotonic() - _t0 > _deadline:
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
                                texts = _tool_result_texts(block)
                                n_tool_results += 1
                                n_images += sum(1 for t in texts
                                                if t == "<image elided>")
                                n_state_blocks += sum(1 for t in texts
                                                      if "[STATE" in t)
                                sink.emit("tool_result", {
                                    "tool_use_id": block.tool_use_id,
                                    "texts": texts})
                    elif isinstance(message, SystemMessage):
                        if getattr(message, "subtype", None) == "init":
                            data = getattr(message, "data", {}) or {}
                            init_tools = data.get("tools")
                            sink.emit("system_init", {"model": data.get("model"),
                                                      "tools": data.get("tools")})
                    elif isinstance(message, ResultMessage):
                        result_msg = message

        error = classify_outcome(result_msg)
        # §14.2: the honest provider-payload manifest for a closed executor —
        # roles/images/STATE copies the session accumulated, the tool list the
        # model was actually shown, and the label that says this run is
        # full-history, NOT the 12-cycle contract. Telemetry only: a write
        # failure must never touch the SessionOutcome.
        try:
            manifest = {
                "label": "full-history",
                "context_control": False,
                "episode": ctx.index,
                "num_turns": getattr(result_msg, "num_turns", None),
                "usage": json_safe(getattr(result_msg, "usage", None)),
                "tool_results": n_tool_results,
                "images": n_images,
                "state_blocks": n_state_blocks,
                "system_init_tools": init_tools,
            }
            (ctx.raw_dir / f"context_manifest_{ctx.index}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1))
        except Exception:  # noqa: BLE001 — manifest is telemetry, not outcome
            pass
        return SessionOutcome(
            usage=json_safe(getattr(result_msg, "usage", None)),
            cost_usd=getattr(result_msg, "total_cost_usd", None),
            turns=getattr(result_msg, "num_turns", None),
            error=error,
            extra={"duration_ms": getattr(result_msg, "duration_ms", None)},
        )
