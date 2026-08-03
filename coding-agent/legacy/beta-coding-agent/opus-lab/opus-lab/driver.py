"""opus-lab driver — Agent SDK over the bare habitat MCP bridge.

Clean-room sibling of ``beta-coding-agent/run_episodes.py``, purpose-built for
the opus skill hill-climb. One deliberate difference: this driver holds the env
FIXED at the bare ``observe`` + ``step`` pair (``bridge.py``) and treats the
skill as the ONLY variable. The old driver conflated the two — ``--bare`` meant
"no tuned mechanisms AND no skill", so you could never run bare-tools + skill.
Here that is the whole point: no ``--skill`` = the pure bare baseline; with
``--skill`` = the same bare env plus the skill layered onto the system prompt.

Per episode: place it via the env-panel HTTP surface (the same field-push +
play path BatchEvalRunner uses), read the instruction driver-side, run ONE clean
Agent SDK session restricted to the two bridge tools, then read habitat's own
measures via ``env_habitat__evaluate`` — the same ruler as the verified
baselines.

Artifacts land under ``outputs/beta-coding-agent/{run_name}/`` (shared with the
old driver so the Coding-Agent Monitor tab browses opus-lab runs too):
    episode_{index}.jsonl   trajectory (tool calls, text, timings, final metrics)
    raw/episode_{index}.jsonl   full unfiltered SDK dump (with --raw-log)
    summary.json            per-episode records + running aggregate
    live_{index}/           spectator frames

Auth rides the logged-in Claude subscription; a stray ANTHROPIC_API_KEY would
silently switch billing to the API in headless mode, so the driver strips it
from the environment before any session starts.

Usage (agentcanvas env; habitat auto_host must already be up — see README):
    python opus-lab/driver.py --episodes 0                 # pure bare baseline
    python opus-lab/driver.py --episodes 0-9 --skill opus-nav
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = Path(__file__).resolve().parent / "bridge.py"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "beta-coding-agent"

SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera. Returns the RGB \
view, a depth image, and a text block with per-region depth in METRES (a 3x3 \
grid — use it to judge how far away what you see actually is), your world pose \
(position and heading), and how far you have walked / moved from the start.
- look_around(): scan in ONE call — eight labeled RGB views 45 degrees apart \
(ahead, ahead-right, right, behind-right, behind, behind-left, left, \
ahead-left), each with a compact metric-depth readout, covering a full circle; \
rotates 360 degrees and restores your heading (costs 24 turn steps).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees. Returns your \
running total forward distance walked.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."


# ── habitat auto_host HTTP helpers (driver-side; not visible to the agent) ──


def panel_field(server_url: str, name: str, value: Any) -> None:
    resp = requests.post(
        f"{server_url}/env-panel/field/{name}", json={"value": value}, timeout=600
    )
    resp.raise_for_status()


def panel_action(server_url: str, name: str) -> None:
    resp = requests.post(
        f"{server_url}/env-panel/action/{name}", json={"params": {}}, timeout=600
    )
    resp.raise_for_status()


def call_function(
    server_url: str, fn: str, inputs: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(f"{server_url}/call/{fn}", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["outputs"]


# ── trajectory capture ──


def _json_safe(obj: Any, _depth: int = 0) -> Any:
    """Recursively coerce any SDK message/block into JSON. Every dataclass field
    is kept (tagged with `_type`); base64 image blobs are elided to a marker so
    the raw dump stays readable (the frames themselves live in live_*/)."""
    if _depth > 12:
        return "<max-depth>"
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"_type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = _json_safe(getattr(obj, f.name), _depth + 1)
        return out
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, _depth + 1) for x in obj]
    if isinstance(obj, bytes):
        return f"<bytes {len(obj)}>"
    if isinstance(obj, str):
        # long, space-free string = base64 blob (image data) → elide; real prose
        # and JSON keep their spaces, so they pass through untouched.
        if len(obj) > 4000 and " " not in obj[:200]:
            return f"<blob {len(obj)} chars elided>"
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _tool_result_texts(block: ToolResultBlock) -> list[str]:
    """Extract text payloads from a tool result; image payloads are elided."""
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


def _parse_step_result(texts: list[str]) -> dict[str, Any] | None:
    for text in texts:
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and "steps_taken_total" in data:
            return data
    return None


_TOOL_SCHEMAS_CACHE: Any = None


async def _bridge_tool_schemas(env_overrides: dict[str, str] | None = None) -> Any:
    """The bridge's own tool definitions — name, description, input schema —
    introspected in-process from BRIDGE_PATH. This is the SAME module the CLI
    spawns as the MCP server, so the captured schemas are exactly what the model
    receives. ``env_overrides`` mirror any env vars that gate tool registration
    (e.g. the shared bridge's HABITAT_BARE) so the set matches the live
    subprocess. Cached across episodes; never raises (logging must not break a
    run)."""
    global _TOOL_SCHEMAS_CACHE
    if _TOOL_SCHEMAS_CACHE is not None:
        return _TOOL_SCHEMAS_CACHE
    overrides = env_overrides or {}
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        spec = importlib.util.spec_from_file_location("_bridge_introspect", BRIDGE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tools = await mod.mcp.list_tools()
        _TOOL_SCHEMAS_CACHE = _json_safe([
            {"name": getattr(t, "name", None),
             "description": getattr(t, "description", None),
             "input_schema": getattr(t, "inputSchema", None)}
            for t in tools
        ])
    except Exception as exc:  # noqa: BLE001 — logging must never break a run
        _TOOL_SCHEMAS_CACHE = {"error": f"tool-schema introspection failed: {exc!r}"}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return _TOOL_SCHEMAS_CACHE


# ── episode loop ──


async def run_episode(
    args: argparse.Namespace, url: str, index: int, run_dir: Path
) -> dict[str, Any]:
    # Blocking HTTP rides to_thread so parallel workers never stall the loop
    # (first play on a cold server can hold a scene load for ~30s).
    await asyncio.to_thread(panel_field, url, "episode_index", index)
    await asyncio.to_thread(panel_action, url, "play")

    reset_config = (
        {"rgb_resolution": str(args.rgb_resolution)} if args.rgb_resolution else None
    )
    ep = await asyncio.to_thread(
        call_function, url, "env_habitat__reset", {"trigger": "driver"}, reset_config
    )
    instruction = ep["instruction"]

    workdir = run_dir / f"workdir_{index}"
    workdir.mkdir(parents=True, exist_ok=True)

    briefing = SYSTEM_PROMPT.format(instruction=instruction, budget=args.step_budget)
    if getattr(args, "skill_text", None):
        briefing += (
            "\n\nYou have been equipped with the following navigation skill."
            " Follow its discipline exactly throughout the episode.\n\n"
            f'<skill name="{args.skill}">\n{args.skill_text}\n</skill>\n'
        )

    # System-prompt mode. Default (custom): our navigation briefing REPLACES the
    # CLI persona entirely (--system-prompt) and the first user message is just
    # "Begin". Preset (--builtin-system-prompt): leave the CLI's built-in Claude
    # Code system prompt untouched and deliver the SAME briefing as the first
    # user message — the vanilla coding-agent persona meeting an embodied task
    # the way a real user would hand it one. Caveat: in preset mode the briefing
    # text is fully visible here (it IS our first message), but the built-in
    # system prompt itself is injected by the CLI and is NOT visible driver-side
    # — only the API receives its bytes (capture it with api_capture_proxy.py).
    if getattr(args, "builtin_prompt", False):
        system_prompt_opt: Any = {"type": "preset", "preset": "claude_code"}
        first_prompt = f"{briefing}\n\n{FIRST_PROMPT}"
    else:
        system_prompt_opt = briefing
        first_prompt = FIRST_PROMPT

    # Thinking config is model-family-specific. Claude 4.6+/5 models (opus-4.6+,
    # sonnet-5, fable-5) take `adaptive`; "summarized" opts into a readable
    # reasoning summary (raw CoT is never returned on opus/fable). Pre-4.6 models
    # (haiku-4.5) do NOT accept `adaptive` — they take explicit `enabled` +
    # budget_tokens. opus is the target here; the haiku branch is a safety net.
    model_name = args.model or ""
    if "haiku" in model_name:
        thinking_config: dict[str, Any] = {
            "type": "enabled",
            "budget_tokens": args.think_budget,
            "display": "summarized",
        }
    else:
        thinking_config = {"type": "adaptive", "display": "summarized"}

    options = ClaudeAgentOptions(
        system_prompt=system_prompt_opt,
        mcp_servers={
            "env": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(BRIDGE_PATH)],
                "env": {
                    "HABITAT_SERVER_URL": url,
                    "HABITAT_STEP_BUDGET": str(args.step_budget),
                    "HABITAT_LIVE_DIR": str(run_dir / f"live_{index}"),
                },
            }
        },
        tools=[],  # no built-in tools: vanilla ReAct over the env only
        # No filesystem settings: without this the CLI walks up from cwd and
        # injects the repo CLAUDE.md into every session — token overhead plus
        # behavioral contamination of the vanilla claim.
        setting_sources=[],
        thinking=thinking_config,
        betas=args.betas,
        # ONLY our bridge — never the user's global MCP config. Without this,
        # sessions inherit whatever ~/.claude MCP servers connect first (Gmail,
        # x-mcp, ...), which pollutes the vanilla claim and starves the bridge
        # of its startup window under parallel session spawns.
        strict_mcp_config=True,
        allowed_tools=[
            "mcp__env__observe",
            "mcp__env__step",
            "mcp__env__look_around",
        ],
        permission_mode="bypassPermissions",
        max_buffer_size=32 * 1024 * 1024,
        max_turns=args.max_turns,
        model=args.model,
        cwd=str(workdir),
    )

    trajectory_path = run_dir / f"episode_{index}.jsonl"
    # Raw dump lives in a raw/ subdir, NOT run_dir/episode_{i}.raw.jsonl — the
    # latter matches the backend's non-recursive `episode_*.jsonl` globs and
    # `int("{i}.raw")` crashes episode enumeration (frontend 500).
    raw_path = run_dir / "raw" / f"episode_{index}.jsonl"
    tool_calls = {"observe": 0, "step": 0}
    last_step_result: dict[str, Any] | None = None
    result_msg: ResultMessage | None = None
    t0 = time.time()

    if args.raw_log:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_cm = raw_path.open("w") if args.raw_log else contextlib.nullcontext()
    with trajectory_path.open("w") as traj, raw_cm as raw_traj:

        def record(kind: str, payload: dict[str, Any]) -> None:
            traj.write(json.dumps({"t": round(time.time() - t0, 2), "kind": kind, **payload}) + "\n")
            traj.flush()  # live tail -f must see every event as it happens

        def record_raw(message: Any) -> None:
            # Full, unfiltered dump of every SDK message (all fields, every block
            # type) — the curated `traj` above stays as the frontend's feed.
            if raw_traj is None:
                return
            raw_traj.write(json.dumps(
                {"t": round(time.time() - t0, 2), "type": type(message).__name__,
                 "msg": _json_safe(message)}, ensure_ascii=False) + "\n")
            raw_traj.flush()

        record("episode_meta", {"index": index, "episode_id": ep.get("episode_id"),
                                "scene_id": ep.get("scene_id"), "instruction": instruction,
                                "skill": args.skill})
        # Full INPUT snapshot for this session — every knob we feed the model:
        # system prompt, first user message, model, and the ENTIRE options object
        # (thinking, tools, mcp, betas, cwd, …). The outputs are the events below
        # (thinking / assistant_text / tool_use / tool_result), so the two sides
        # together make the trajectory a complete I/O record. _json_safe elides
        # base64 blobs so this stays readable.
        tool_schemas = await _bridge_tool_schemas()
        record("session_inputs", {
            "model": args.model,
            "skill": args.skill,
            "builtin_system_prompt": bool(getattr(args, "builtin_prompt", False)),
            "system_prompt": _json_safe(system_prompt_opt),
            "first_prompt": first_prompt,
            "tool_schemas": tool_schemas,
            "options": _json_safe(options),
        })

        async with ClaudeSDKClient(options=options) as client:
            # The CLI starts reasoning before MCP servers finish connecting;
            # under parallel session spawns the model's first turn reliably
            # beats the bridge and sees zero tools. Gate the prompt on the
            # bridge reporting 'connected'.
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
            record("bridge_status", {"status": bridge_status})
            if bridge_status != "connected":
                raise RuntimeError(f"env bridge not connected: {bridge_status}")

            await client.query(first_prompt)
            async for message in client.receive_response():
                record_raw(message)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            record("assistant_text", {"text": block.text})
                        elif isinstance(block, ThinkingBlock):
                            record("thinking", {"chars": len(block.thinking),
                                                "text": block.thinking})
                        elif isinstance(block, ToolUseBlock):
                            short = block.name.rsplit("__", 1)[-1]
                            if short in tool_calls:
                                tool_calls[short] += 1
                            record("tool_use", {"id": block.id, "name": block.name,
                                                "input": block.input})
                elif isinstance(message, UserMessage):
                    content = message.content
                    blocks = content if isinstance(content, list) else []
                    for block in blocks:
                        if isinstance(block, ToolResultBlock):
                            texts = _tool_result_texts(block)
                            parsed = _parse_step_result(texts)
                            if parsed is not None:
                                last_step_result = parsed
                            record("tool_result", {"tool_use_id": block.tool_use_id,
                                                   "texts": texts})
                elif isinstance(message, SystemMessage):
                    if getattr(message, "subtype", None) == "init":
                        data = getattr(message, "data", {}) or {}
                        record("system_init", {"model": data.get("model"),
                                               "tools": data.get("tools")})
                elif isinstance(message, ResultMessage):
                    result_msg = message

        # Session ended — record the final ResultMessage (usage, cost, turns,
        # stop_reason, …) as its own trajectory event, not only in summary.json.
        if result_msg is not None:
            record("result", {"result": _json_safe(result_msg)})

        # Episode finished — evaluate while the trajectory file is still open
        # so the final metrics land in the log itself (fenced by a divider the
        # frontend renders). evaluate is driver-side; the agent never sees it.
        metrics_out = await asyncio.to_thread(
            call_function, url, "env_habitat__evaluate", {"trigger": "driver"}
        )
        metrics = metrics_out.get("metrics") or {}
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        record("episode_metrics", {"metrics": metrics})

    episode: dict[str, Any] = {
        "index": index,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "instruction": instruction,
        "metrics": metrics,
        "agent": {
            "tool_calls": tool_calls,
            "env_steps": (last_step_result or {}).get("steps_taken_total", 0),
            "end_reason": (last_step_result or {}).get("end_reason"),
            "called_stop": (last_step_result or {}).get("end_reason") == "stop_called",
            "num_turns": getattr(result_msg, "num_turns", None),
            "duration_ms": getattr(result_msg, "duration_ms", None),
            "usage": getattr(result_msg, "usage", None),
            "total_cost_usd": getattr(result_msg, "total_cost_usd", None),
            "is_error": getattr(result_msg, "is_error", None),
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    return episode


# ── aggregation & main ──


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"episode_count": len(episodes)}
    numeric: dict[str, list[float]] = {}
    for record in episodes:
        for key, value in (record.get("metrics") or {}).items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
        numeric.setdefault("env_steps", []).append(float(record["agent"]["env_steps"]))
    for key, values in numeric.items():
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    agg["stop_rate"] = round(
        sum(1 for r in episodes if r["agent"]["called_stop"]) / max(1, len(episodes)), 4
    )
    return agg


def parse_episodes(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return indices


async def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress must survive file redirection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:9200")
    parser.add_argument(
        "--server-urls",
        default=None,
        help="comma-separated auto_host URLs; one parallel worker per server "
        "(overrides --server-url)",
    )
    parser.add_argument("--dataset", default="R2R-CE")
    parser.add_argument("--split", default="rand100")
    parser.add_argument("--episodes", required=True, help='e.g. "0", "0-9", "0,3,7"')
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument(
        "--think-budget",
        type=int,
        default=4000,
        help="thinking budget_tokens for pre-4.6 models (haiku-4.5); "
        "ignored by 4.6+/5 models which use adaptive thinking",
    )
    parser.add_argument(
        "--betas",
        default="",
        help="comma-separated Anthropic beta headers passed to the CLI "
        "(e.g. interleaved-thinking-2025-05-14)",
    )
    parser.add_argument("--episode-timeout", type=int, default=2400, help="seconds")
    parser.add_argument(
        "--no-raw-log",
        dest="raw_log",
        action="store_false",
        help="disable the raw SDK-message dump (ON by default). The dump writes "
        "every SDK message (all fields, all block types) to raw/episode_{i}.jsonl; "
        "base64 image blobs elided. Note: opus/fable thinking is still only the "
        "summary — the SDK never receives raw CoT.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--rgb-resolution",
        type=int,
        default=None,
        help="RGB render size in px (blank = YAML default 224); applied via reset config",
    )
    parser.add_argument(
        "--skill",
        default=None,
        help="name of a skill dir under opus-lab/skills/; its SKILL.md body is "
        "appended to the system prompt. Omit for the pure bare baseline.",
    )
    parser.add_argument(
        "--builtin-system-prompt",
        dest="builtin_prompt",
        action="store_true",
        help="use the CLI's built-in Claude Code system prompt (preset) instead "
        "of our custom navigation system prompt; the navigation briefing is then "
        "delivered as the first user message. Tests the vanilla coding-agent persona.",
    )
    args = parser.parse_args()

    args.betas = [b.strip() for b in args.betas.split(",") if b.strip()]

    args.skill_text = None
    if args.skill:
        skill_path = SKILLS_DIR / args.skill / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        if text.startswith("---"):  # strip frontmatter
            text = text.split("---", 2)[2]
        args.skill_text = text.strip()
        print(f"[driver] skill '{args.skill}' loaded ({len(args.skill_text)} chars)")

    # Subscription auth: an inherited API key silently wins in headless mode.
    import os

    if os.environ.pop("ANTHROPIC_API_KEY", None):
        print("[driver] ANTHROPIC_API_KEY was set — removed so sessions use subscription auth")

    urls = [u.strip() for u in (args.server_urls or args.server_url).split(",") if u.strip()]
    for url in urls:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"[driver] {url} healthy: {health.json()['name']}")

    run_name = args.run_name or time.strftime("opuslab_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[driver] dataset={args.dataset} split={args.split} "
          f"workers={len(urls)} skill={args.skill or '<bare>'} -> {run_dir}")
    for url in urls:
        panel_field(url, "dataset", args.dataset)
        panel_field(url, "split", args.split)

    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in parse_episodes(args.episodes):
        queue.put_nowait(index)

    episodes: list[dict[str, Any]] = []
    summary_path = run_dir / "summary.json"
    write_lock = asyncio.Lock()

    async def flush_summary() -> None:
        async with write_lock:
            episodes.sort(key=lambda r: r["index"])
            summary = {
                "run_name": run_name,
                "config": {k: v for k, v in vars(args).items()},
                "servers": urls,
                "aggregate": aggregate([e for e in episodes if "error" not in e]),
                "episodes": episodes,
            }
            summary_path.write_text(json.dumps(summary, indent=2))

    async def worker(position: int, url: str) -> None:
        await asyncio.sleep(position * 2)  # stagger session spawns
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"[driver] episode {index} starting on {url}")
            try:
                episode = await asyncio.wait_for(
                    run_episode(args, url, index, run_dir), timeout=args.episode_timeout
                )
            except asyncio.TimeoutError:
                print(f"[driver] episode {index} TIMED OUT after {args.episode_timeout}s")
                episode = {"index": index, "error": "timeout", "metrics": {},
                           "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}}}
            except Exception as exc:  # noqa: BLE001 — one bad episode must not kill the run
                print(f"[driver] episode {index} FAILED: {exc!r}")
                episode = {"index": index, "error": repr(exc), "metrics": {},
                           "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}}}
            episodes.append(episode)
            await flush_summary()
            m = episode.get("metrics") or {}
            print(
                f"[driver] episode {index} done: success={m.get('success')} "
                f"spl={m.get('spl')} steps={episode['agent'].get('env_steps')} "
                f"stop={episode['agent'].get('called_stop')}"
            )

    await asyncio.gather(*(worker(i, url) for i, url in enumerate(urls)))

    print(f"[driver] run complete -> {summary_path}")
    print(json.dumps(aggregate([e for e in episodes if 'error' not in e]), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
