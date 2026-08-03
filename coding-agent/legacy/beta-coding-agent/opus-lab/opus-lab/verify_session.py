"""One-off proof: an opus-lab SDK session carries ONLY our system prompt + our
three env tools — no Claude Code persona, no CLAUDE.md, no user MCP/settings.

GPU-free by design: the system prompt and session config are fixed before any
tool call, so no live habitat auto_host is needed (we point the bridge at a dead
port). We build the SAME ClaudeAgentOptions the driver builds and dump:
  (1) the exact system prompt we pass (the entire --system-prompt value);
  (2) the spawned `claude` process argv — proving --system-prompt is present and
      --append-system-prompt is ABSENT (replace, not append);
  (3) the session's own system_init message (its tools / mcp / model / cwd / auth).

Run:  env -u ANTHROPIC_API_KEY python beta-coding-agent/opus-lab/verify_session.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, SystemMessage

import driver  # reuse SYSTEM_PROMPT, BRIDGE_PATH, _json_safe verbatim

DUMMY_INSTRUCTION = "Walk to the kitchen and stop by the sink."  # affects prompt text only
FLAGS = (
    "--system-prompt",
    "--append-system-prompt",
    "--setting-sources",
    "--mcp-config",
    "--strict-mcp-config",
    "--permission-mode",
    "--model",
)


async def main() -> None:
    os.environ.pop("ANTHROPIC_API_KEY", None)

    system_prompt = driver.SYSTEM_PROMPT.format(instruction=DUMMY_INSTRUCTION, budget=500)
    print("=" * 72)
    print("(1) SYSTEM PROMPT WE PASS  — this string IS the entire --system-prompt")
    print("=" * 72)
    print(system_prompt)
    print("=" * 72)

    workdir = Path(__file__).resolve().parent / "_verify_workdir"
    workdir.mkdir(exist_ok=True)
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={
            "env": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(driver.BRIDGE_PATH)],
                # dead port on purpose — the MCP handshake needs no habitat
                "env": {"HABITAT_SERVER_URL": "http://127.0.0.1:9999",
                        "HABITAT_STEP_BUDGET": "500"},
            }
        },
        tools=[],
        setting_sources=[],
        thinking={"type": "adaptive", "display": "summarized"},
        strict_mcp_config=True,
        allowed_tools=["mcp__env__observe", "mcp__env__step", "mcp__env__look_around"],
        permission_mode="bypassPermissions",
        max_buffer_size=32 * 1024 * 1024,
        max_turns=1,
        model="claude-opus-4-8",
        cwd=str(workdir),
    )

    async with ClaudeSDKClient(options=options) as client:
        # wait for the bridge MCP handshake (no habitat required for this)
        for _ in range(60):
            st = await client.get_mcp_status()
            ents = st.get("mcpServers", []) if isinstance(st, dict) else []
            s = next((e.get("status") for e in ents if e.get("name") == "env"), None)
            if s in ("connected", "failed", "needs-auth", "disabled"):
                break
            await asyncio.sleep(0.5)

        # (2) capture the spawned CLI's argv and check which flags are present
        ps = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True)
        cli_lines = [l for l in ps.stdout.splitlines() if "--system-prompt" in l]
        print("\n" + "=" * 72)
        print("(2) SPAWNED `claude` CLI — which system-prompt flags are present?")
        print("=" * 72)
        for flag in FLAGS:
            present = any(flag in l for l in cli_lines)
            print(f"  {flag:24s}: {'PRESENT' if present else 'absent'}")
        # show that the --system-prompt value begins with OUR text (first line)
        if cli_lines:
            head = cli_lines[0].split("--system-prompt", 1)[1][:90].replace("\n", " ")
            print(f"\n  --system-prompt value starts with: …{head!r}")

        # (3) dump the session's own init self-report. system_init is the first
        # message; a message cap guards against the dead-habitat tool errors
        # (max_turns=1 ends the turn anyway, so this never hangs).
        await client.query("Begin.")
        printed = False
        count = 0
        async for message in client.receive_response():
            count += 1
            if isinstance(message, SystemMessage) and getattr(message, "subtype", None) == "init":
                print("\n" + "=" * 72)
                print("(3) system_init — the session's OWN report of what it carries")
                print("=" * 72)
                print(json.dumps(driver._json_safe(message), indent=2, ensure_ascii=False))
                printed = True
                break
            if count > 30:
                break
        if not printed:
            print("\n(no system_init captured — session may need auth; check output above)")


if __name__ == "__main__":
    asyncio.run(main())
