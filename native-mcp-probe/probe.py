"""One real episode on the raw env_habitat toolface over native /mcp.

Standalone: no imports from coding-agent/. See README.md for the design.
Run from the agentcanvas env: python native-mcp-probe/probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "agentcanvas" / "backend"
DEFAULT_HABITAT_PYTHON = os.path.expanduser("~/miniforge3/envs/ac-vlnce/bin/python")

# The two tools the endpoint exposes — everything else on the nodeset stays
# reachable via /manifest + /call but never appears on /mcp.
MCP_TOOLS = "observe_egocentric,step_discrete"
AGENT_TOOLS = [
    "mcp__env__env_habitat__observe_egocentric",
    "mcp__env__env_habitat__step_discrete",
]

# Subscription auth is the line's convention; an inherited API key silently
# outranks the claude.ai login and stalls every model call.
for _var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
    os.environ.pop(_var, None)


# ── control plane (experimenter infra; the agent never sees these) ──

def panel_field(url: str, name: str, value: Any) -> None:
    requests.post(f"{url}/env-panel/field/{name}", json={"value": value},
                  timeout=600).raise_for_status()


def panel_action(url: str, name: str) -> None:
    requests.post(f"{url}/env-panel/action/{name}", json={"params": {}},
                  timeout=600).raise_for_status()


def call_fn(url: str, fn: str, inputs: dict, config: dict | None = None) -> dict:
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(f"{url}/call/{fn}", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["outputs"]


def spawn_server(python: str, port: int, log_path: Path) -> subprocess.Popen:
    cmd = [python, "-m", "app.server.auto_host",
           "--file", "../../workspace/nodesets/env/env_habitat.py",
           "--class", "EnvHabitatNodeSet",
           "--host", "127.0.0.1", "--port", str(port),
           "--mcp-tools", MCP_TOOLS]
    env = {**os.environ, "PYTHONPATH": f"{BACKEND}:{REPO_ROOT}"}
    log = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=BACKEND, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(url: str, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/manifest", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"auto_host at {url} not ready after {timeout_s}s")


def briefing(instruction: str) -> str:
    return (
        "You control a robot in an indoor scene via exactly two tools.\n\n"
        f"NAVIGATION INSTRUCTION: {instruction}\n\n"
        "Tools:\n"
        "- env_habitat__observe_egocentric: current first-person view (RGB "
        "image plus a JSON block with pose and metadata).\n"
        "- env_habitat__step_discrete: one discrete action per call — "
        "action 0 = STOP (permanently ends the episode; call it when you "
        "believe you are at the goal), 1 = move forward 0.25 m, "
        "2 = turn left 30 degrees, 3 = turn right 30 degrees.\n\n"
        "Observe, decide, act, observe again. One step per call; the episode "
        "hard-stops after 500 steps. Success requires ending with action 0 "
        "within 3 m of the goal. Do not narrate at length; navigate."
    )


async def run_agent(server_url: str, brief: str, model: str, max_turns: int,
                    transcript: Path) -> dict:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        system_prompt=brief,
        mcp_servers={"env": {"type": "http", "url": f"{server_url}/mcp"}},
        tools=[],
        setting_sources=[],
        strict_mcp_config=True,
        allowed_tools=AGENT_TOOLS,
        model=model,
        max_turns=max_turns,
    )
    counts = {"observe": 0, "step": 0}
    num_turns = 0
    result_text = None
    with transcript.open("w") as tf:
        def log(rec: dict) -> None:
            tf.write(json.dumps(rec) + "\n")
            tf.flush()

        async with ClaudeSDKClient(options=options) as client:
            await client.query("Begin. Follow your navigation instruction.")
            async for msg in client.receive_response():
                kind = type(msg).__name__
                if kind == "AssistantMessage":
                    num_turns += 1
                    for block in msg.content:
                        b = type(block).__name__
                        if b == "ToolUseBlock":
                            name = block.name.rsplit("__", 1)[-1]
                            counts["observe" if "observe" in name else "step"] += 1
                            log({"t": "tool", "name": block.name,
                                 "input": block.input})
                        elif b == "TextBlock":
                            log({"t": "text", "text": block.text})
                elif kind == "ResultMessage":
                    result_text = getattr(msg, "result", None)
                    log({"t": "result", "result": result_text})
    return {"num_turns": num_turns, "tool_calls": counts,
            "final_text": result_text}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--dataset", default="R2R-CE")
    ap.add_argument("--split", default="rand100")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--rgb", default="512")
    ap.add_argument("--port", type=int, default=9260)
    ap.add_argument("--server", default=None,
                    help="reuse a running auto_host (skips spawn/teardown)")
    ap.add_argument("--habitat-python", default=DEFAULT_HABITAT_PYTHON)
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / "runs" / f"ep{args.episode}_{stamp}"
    run_dir.mkdir(parents=True)

    proc = None
    url = args.server or f"http://127.0.0.1:{args.port}"
    if not args.server:
        print(f"[probe] spawning env_habitat auto_host on {url} "
              f"(/mcp narrowed to: {MCP_TOOLS})")
        proc = spawn_server(args.habitat_python, args.port,
                            run_dir / "server.log")
    try:
        wait_ready(url)
        panel_field(url, "dataset", args.dataset)
        panel_field(url, "split", args.split)
        panel_field(url, "episode_index", args.episode)
        panel_action(url, "play")
        ep = call_fn(url, "env_habitat__reset", {"trigger": "probe"},
                     {"rgb_resolution": args.rgb})
        instruction = ep["instruction"]
        print(f"[probe] episode {args.episode} ({ep.get('episode_id')}): "
              f"{instruction[:120]}")

        t0 = time.monotonic()
        agent = anyio.run(run_agent, url, briefing(instruction), args.model,
                          args.max_turns, run_dir / "transcript.jsonl")
        wall_s = round(time.monotonic() - t0, 1)

        metrics = call_fn(url, "env_habitat__evaluate", {})
        # env_habitat is stateful -> exclusive /mcp session. A fresh
        # initialize succeeding right after the agent disconnected proves the
        # SDK sent DELETE (the 600 s idle reaper can't have fired yet).
        session_released = None
        try:
            r = requests.post(
                f"{url}/mcp",
                headers={"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18",
                                 "capabilities": {},
                                 "clientInfo": {"name": "probe", "version": "0"}}},
                timeout=10)
            session_released = r.status_code == 200
        except requests.RequestException:
            pass

        result = {
            "episode": args.episode, "dataset": args.dataset,
            "split": args.split, "model": args.model,
            "episode_id": ep.get("episode_id"), "instruction": instruction,
            "metrics": metrics.get("metrics", {}),
            "success": metrics.get("success"), "spl": metrics.get("spl"),
            "agent": agent, "wall_s": wall_s,
            "session_released": session_released,
            "server": url, "mcp_tools": MCP_TOOLS,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[probe] done: success={result['success']} spl={result['spl']} "
              f"turns={agent['num_turns']} steps={agent['tool_calls']['step']} "
              f"wall={wall_s}s session_released={session_released}")
        print(f"[probe] artifacts: {run_dir}")
    finally:
        if proc is not None and not args.keep_server:
            os.killpg(proc.pid, signal.SIGTERM)
            print(f"[probe] server pgid {proc.pid} terminated")


if __name__ == "__main__":
    main()
