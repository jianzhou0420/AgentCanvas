"""OpenAI Codex CLI adapter — OpenAI's closed scaffolding.

Session block ported from beta-codex-agent/run_episodes.py. codex keeps its
built-in system prompt (that closed scaffolding is the thing under test); the
briefing rides as the one user prompt. Auth is the logged-in ChatGPT
subscription. Codex-specific wiring that was hard-won (2026-07-13):

- MCP tool calls are approval-gated and exec mode auto-cancels them; v0.142
  accepts only prompt|approve for default_tools_approval_mode ("auto" from
  the newer docs is silently invalid).
- no SDK-level turn cap exists: ctx.max_turns feeds the bridge broadcast /
  STOP gate only (recorded difference; hard caps = step budget + timeout).
- reasoning is usage-counted but unreadable (summary=[] + encrypted content).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from typing import Any

from driver import EpisodeContext, EventSink, SessionOutcome, json_safe


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _tool_result_texts(result: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(result, dict):
        return texts
    content = result.get("content")
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


class CodexCliAdapter:
    name = "codex"

    def __init__(self) -> None:
        self.inherent: dict[str, Any] = {
            "auth": "ChatGPT subscription (codex login)",
            "thinking": "reasoning tokens counted, content unreadable",
            "turn_cap": "broadcast only (no codex-side hard cap)",
            # codex CLI owns its own context/vision management; its image
            # retention policy is not audited here. See docs developer-guide/
            # coding-agent/harness-notes.
            "vision_context": "codex-managed (unaudited)",
        }

    def prepare(self, spec) -> None:  # noqa: ARG002 — no per-cell setup needed
        self.inherent["codex_version"] = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()

    def _argv(self, ctx: EpisodeContext) -> list[str]:
        env_table = ", ".join(
            f"{k} = {_toml_str(v)}" for k, v in ctx.bridge_env().items()
        )
        return [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "-c", f"model = {_toml_str(ctx.model)}",
            "-c", f"model_reasoning_effort = {_toml_str(ctx.extra.get('effort', 'medium'))}",
            # Display-layer knob: surfaces reasoning summaries if any exist.
            "-c", 'model_reasoning_summary = "detailed"',
            "-c", f"mcp_servers.env.command = {_toml_str(sys.executable)}",
            "-c", f"mcp_servers.env.args = [{_toml_str(str(ctx.bridge_path))}]",
            "-c", f"mcp_servers.env.env = {{ {env_table} }}",
            "-c", 'mcp_servers.env.default_tools_approval_mode = "approve"',
            # No AGENTS.md injection (the SDK cell's setting_sources=[] analog).
            "-c", "project_doc_max_bytes = 0",
        ]

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"options": {"argv": self._argv(ctx), "sandbox": "read-only",
                            "system_prompt_note": "<codex builtin>"}}

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        prompt = ctx.briefing + "\n\n" + ctx.first_prompt
        argv = self._argv(ctx) + [prompt]

        raw_path = ctx.raw_dir / f"episode_{ctx.index}.jsonl"
        stderr_path = ctx.raw_dir / f"episode_{ctx.index}.stderr.log"
        usage_totals: dict[str, int] = {}
        thread_id: str | None = None
        exit_code: int | None = None

        def handle_event(event: dict[str, Any]) -> None:
            nonlocal thread_id
            kind = event.get("type")
            if kind == "thread.started":
                thread_id = event.get("thread_id")
                sink.emit("system_init", {"thread_id": thread_id, "model": ctx.model})
                return
            if kind == "turn.completed":
                for key, value in (event.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage_totals[key] = usage_totals.get(key, 0) + int(value)
                return
            if kind in ("turn.failed", "error"):
                sink.emit("driver_error", {"error": json_safe(event)})
                return
            item = event.get("item") or {}
            item_type = item.get("type")
            if kind == "item.started" and item_type == "mcp_tool_call":
                sink.emit("tool_use", {"id": item.get("id"),
                                       "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                                       "input": item.get("arguments")})
                return
            if kind == "item.completed":
                if item_type == "mcp_tool_call":
                    texts = _tool_result_texts(item.get("result"))
                    if item.get("error"):
                        texts.append(json.dumps({"error": json_safe(item["error"])}))
                    sink.emit("tool_result", {"tool_use_id": item.get("id"), "texts": texts})
                elif item_type == "agent_message":
                    sink.emit("assistant_text", {"text": item.get("text", "")})
                elif item_type == "reasoning":
                    text = item.get("text") or ""
                    sink.emit("thinking", {"chars": len(text), "text": text})
                elif item_type == "command_execution":
                    # codex's own shell tool — can't be unmounted; sandbox is
                    # read-only and every use is on the record.
                    sink.emit("tool_use", {"id": item.get("id"), "name": "shell",
                                           "input": {"command": item.get("command")}})
                    sink.emit("tool_result", {
                        "tool_use_id": item.get("id"),
                        "texts": [str(item.get("aggregated_output", ""))[:4000]]})
                else:
                    sink.emit("driver_error", {"error": {"unhandled_item": json_safe(item)}})

        with raw_path.open("w") as raw, stderr_path.open("wb") as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_file,
                cwd=str(ctx.workdir),
                start_new_session=True,  # own PGID so timeout kill reaps MCP child
                # look_around returns four images in one JSONL event; the
                # default 64 KiB line limit would kill the stream mid-parse
                limit=32 * 1024 * 1024,
            )
            try:
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        sink.emit("driver_error", {
                            "error": {"nonjson_stdout": line[:300].decode(errors="replace")}})
                        continue
                    raw.write(json.dumps({"event": json_safe(event)},
                                         ensure_ascii=False) + "\n")
                    raw.flush()
                    handle_event(event)
                exit_code = await proc.wait()
            finally:
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()

        return SessionOutcome(
            usage=usage_totals,
            cost_usd=None,  # subscription auth — no per-token billing
            turns=None,     # codex "turns" ≠ LLM calls; bridge counts tool calls
            error=(f"codex exited {exit_code}" if exit_code else None),
            extra={"thread_id": thread_id, "exit_code": exit_code},
        )
