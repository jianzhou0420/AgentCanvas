"""OpenAI Codex CLI adapter — OpenAI's closed scaffolding.

Session block ported from the legacy driver (git history:
d10591e:beta-codex-agent/run_episodes.py). codex keeps its
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

from core.driver import EpisodeContext, EventSink, SessionOutcome, json_safe


def _toml_str(value: str) -> str:
    # TOML basic strings cannot carry raw newlines/tabs — escape them or a
    # crafted value splits the -c override into a second TOML statement
    return ('"' + value.replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\t", "\\t") + '"')


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
            # §14.2: codex owns its own history — retention count, role
            # structure and map residency are unproven in this repo, so runs
            # carry the unaudited label rather than claiming the 12-cycle
            # contract.
            "context_control": False,
            "context_label": "codex-managed-unaudited",
        }
        self._gateway = None  # litellm gateway (cross-API seats only)

    def prepare(self, spec) -> None:
        self.inherent["codex_version"] = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        # cross-API seat: an owned litellm gateway becomes a codex custom
        # model_provider (openai-chat wire). Native cells skip this branch
        # and keep their exact historical argv.
        from core.api.proxy import gateway_for_spec
        self._gateway = gateway_for_spec(spec, tag="codex")
        if self._gateway is not None:
            self.inherent["api_gateway"] = self._gateway.describe()
            self.inherent["auth"] = "litellm gateway (vendor API billing)"
            print(f"[codex] litellm gateway :{self._gateway.port} — "
                  f"{spec.model_id} -> {self._gateway.routes[0].litellm_model}")

    def finalize(self, run_dir) -> dict:  # noqa: ARG002
        if self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
        return {}

    def _argv(self, ctx: EpisodeContext) -> list[str]:
        # §16.6: the bridge's snapshot identity must say WHO is driving —
        # same pattern as claude_sdk._options ({**bridge_env, EH_EXECUTOR}).
        bridge_env = {**ctx.bridge_env(), "EH_EXECUTOR": "codex"}
        env_table = ", ".join(
            f"{k} = {_toml_str(v)}" for k, v in bridge_env.items()
        )
        return [
            "codex", "exec", "--json", "--skip-git-repo-check",
            # blank model = the CLI's own default; `-c model = ""` would 400
            *(["-c", f"model = {_toml_str(ctx.model)}"] if ctx.model else []),
            "-c", f"model_reasoning_effort = {_toml_str(ctx.extra.get('effort', 'medium'))}",
            # Display-layer knob: surfaces reasoning summaries if any exist.
            "-c", 'model_reasoning_summary = "detailed"',
            "-c", f"mcp_servers.env.command = {_toml_str(sys.executable)}",
            "-c", f"mcp_servers.env.args = [{_toml_str(str(ctx.bridge_path))}]",
            "-c", f"mcp_servers.env.env = {{ {env_table} }}",
            "-c", 'mcp_servers.env.default_tools_approval_mode = "approve"',
            # No AGENTS.md injection (the SDK cell's setting_sources=[] analog).
            "-c", "project_doc_max_bytes = 0",
            # cross-API seat: route this run through the owned litellm gateway
            *(self._gateway.codex_args() if self._gateway is not None else []),
        ]

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"options": {"argv": self._argv(ctx), "sandbox": "read-only",
                            "system_prompt_note": "<codex builtin>"}}

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        prompt = ctx.briefing + "\n\n" + ctx.first_prompt
        # `--` ends option parsing so a prompt starting with "-" cannot be
        # read as a flag
        argv = self._argv(ctx) + ["--", prompt]

        raw_path = ctx.raw_dir / f"episode_{ctx.index}.jsonl"
        stderr_path = ctx.raw_dir / f"episode_{ctx.index}.stderr.log"
        usage_totals: dict[str, int] = {}
        thread_id: str | None = None
        exit_code: int | None = None
        n_turns = 0
        n_tool_results = 0
        n_images = 0
        n_state_blocks = 0

        def handle_event(event: dict[str, Any]) -> None:
            nonlocal thread_id, n_turns, n_tool_results, n_images, n_state_blocks
            kind = event.get("type")
            if kind == "thread.started":
                thread_id = event.get("thread_id")
                sink.emit("system_init", {"thread_id": thread_id, "model": ctx.model})
                return
            if kind == "turn.completed":
                n_turns += 1
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
                    n_tool_results += 1
                    n_images += sum(1 for t in texts if t == "<image elided>")
                    n_state_blocks += sum(1 for t in texts if "[STATE" in t)
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

        # §16.6: the honest provider-payload manifest for a closed executor —
        # mirrors claude_sdk's, with the extra honesty codex demands: every
        # count comes from the EVENT STREAM, not the provider payload, and
        # codex "turns" are its own units, not LLM calls. Telemetry only: a
        # write failure must never touch the SessionOutcome.
        try:
            manifest = {
                "label": "codex-managed-unaudited",
                "context_control": False,
                "episode": ctx.index,
                "num_turns": n_turns,          # codex turn.completed events
                "usage": usage_totals or None,  # summed client-side; unaudited
                "tool_results": n_tool_results,
                "images": n_images,
                "state_blocks": n_state_blocks,
                "note": ("counts from the codex event stream, not the "
                         "resident provider payload — codex owns its own "
                         "transcript and no per-request audit exists here"),
            }
            (ctx.raw_dir / f"context_manifest_{ctx.index}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1))
        except Exception:  # noqa: BLE001 — manifest is telemetry, not outcome
            pass
        return SessionOutcome(
            usage=usage_totals,
            cost_usd=None,  # subscription auth — no per-token billing
            turns=None,     # codex "turns" ≠ LLM calls; bridge counts tool calls
            error=(f"codex exited {exit_code}" if exit_code else None),
            extra={"thread_id": thread_id, "exit_code": exit_code},
        )

    def refine_instruction(self, *, model: str, system: str, user: str,
                           extra: dict | None = None) -> str | None:
        """§17: one short text-only codex exec call with the run's own nav
        model — no MCP servers, no tools, read-only sandbox. Returns the
        agent's final message text, or None on ANY failure (= the caller's
        refinement_failed fallback to the original instruction).

        Signature matches the driver's refine protocol (keyword-only
        model/system/user/extra — EpisodeContext does not exist yet at the
        refine anchor)."""
        del extra  # codex needs no extra knobs for a one-shot text call
        argv = [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "-c", f"model = {_toml_str(model)}",
            "-c", 'model_reasoning_effort = "low"',
            "-c", "project_doc_max_bytes = 0",
            # a text-cleaning call must not inherit the operator's personal
            # MCP servers (~/.codex/config.toml defined codegraph — every
            # refine spun it up; review P2). The empty inline table replaces
            # the whole user mcp_servers table. codex's builtin shell tool
            # cannot be unmounted — the prompt asks for text only.
            "-c", "mcp_servers = {}",
            system + "\n\n" + user,
        ]
        try:
            # §20.1: 45 s fail-fast, aligned with the other two adapters
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=45)
            if proc.returncode != 0:
                return None
            texts: list[str] = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                item = event.get("item") or {}
                if (event.get("type") == "item.completed"
                        and item.get("type") == "agent_message"):
                    texts.append(str(item.get("text", "")))
            out = "\n".join(t for t in texts if t).strip()
            return out or None
        except Exception:  # noqa: BLE001 — refinement must never kill a run
            return None
