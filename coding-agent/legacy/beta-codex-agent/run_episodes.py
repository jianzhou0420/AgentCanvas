"""Vanilla codex-agent VLN driver — OpenAI Codex CLI over the habitat MCP bridge.

The third harness cell: beta-coding-agent measures Anthropic's closed
scaffolding (Claude Agent SDK), beta-react-harness the open ReAct loop
(mini-swe-agent) — this directory measures OpenAI's closed scaffolding by
driving the SAME env surface through `codex exec --json`. The bridge
(`beta-coding-agent/mcp_bridge.py`) is reused verbatim, so tool names,
schemas, prompts, clearance math, budget broadcast, and STOP gate are
byte-identical to the other two cells.

Per episode: place it via the env-panel HTTP surface, read the instruction
driver-side, run ONE fresh `codex exec` session with the bridge mounted as
an MCP server, then read habitat's own measures via ``env_habitat__evaluate``
— the same ruler as the verified baselines.

Artifacts land under ``outputs/beta-codex-agent/{run_name}/`` in the same
layout the Coding-Agent Monitor renders (episode_{i}.jsonl / raw/ / live_{i}/
/ summary.json).

Recorded differences vs the Claude SDK cell (kept honest, not hidden):
- codex's built-in tools (shell, etc.) cannot be removed; the sandbox is
  read-only and every command_execution is logged as a shell tool event.
- there is no SDK-level max_turns; --max-turns only feeds the bridge's
  turn-budget broadcast / STOP gate. Hard caps are the env step budget and
  --episode-timeout.
- auth is the logged-in ChatGPT subscription (codex login), no per-token
  billing; cost fields stay null.

Usage (agentcanvas env; habitat auto_host must already be up — see README):
    python beta-codex-agent/run_episodes.py --episodes 0
    python beta-codex-agent/run_episodes.py --episodes 0-9 --split rand100
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "beta-coding-agent" / "mcp_bridge.py"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "beta-codex-agent"

SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (RGB image plus \
a clearance readout: meters to the nearest obstacle in the left/center/right \
thirds of the view; 10.0 = open).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading (costs 24 \
turn steps).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions; each step() result reports \
roughly how many remain.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

BARE_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

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
    """Recursively coerce a codex event into JSON-safe form; base64 image blobs
    are elided to a marker so the raw dump stays readable (frames live in
    live_*/)."""
    if _depth > 12:
        return "<max-depth>"
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, _depth + 1) for x in obj]
    if isinstance(obj, str):
        # long, space-free string = base64 blob (image data) → elide; real prose
        # and JSON keep their spaces, so they pass through untouched.
        if len(obj) > 4000 and " " not in obj[:200]:
            return f"<blob {len(obj)} chars elided>"
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _tool_result_texts(result: Any) -> list[str]:
    """Extract text payloads from an MCP tool result; image payloads elided."""
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
    """The bridge's own tool definitions, introspected in-process from
    BRIDGE_PATH — the SAME module codex spawns as the MCP server, so the
    captured schemas are exactly what the model receives. Cached across
    episodes; never raises (logging must not break a run)."""
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


# ── codex exec invocation ──


def _toml_str(value: str) -> str:
    """Quote a value as a TOML basic string for a -c override."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_codex_argv(args: argparse.Namespace, url: str, live_dir: Path) -> list[str]:
    """One fresh session, our bridge as the only configured MCP server, model
    knobs pinned explicitly so runs don't inherit the user's ~/.codex tuning."""
    bridge_env = {
        "HABITAT_SERVER_URL": url,
        "HABITAT_STEP_BUDGET": str(args.step_budget),
        # bare = no turn-budget broadcast and no STOP gate (both keyed off
        # TURN_BUDGET>0 in the bridge); full stack gets max_turns so the model
        # can see its remaining calls. There is no codex-side hard turn cap.
        "HABITAT_TURN_BUDGET": "0" if args.bare else str(args.max_turns),
        "HABITAT_BARE": "1" if args.bare else "0",
        "HABITAT_LIVE_DIR": str(live_dir),
    }
    env_table = ", ".join(f"{k} = {_toml_str(v)}" for k, v in bridge_env.items())
    return [
        "codex", "exec", "--json", "--skip-git-repo-check",
        "-c", f"model = {_toml_str(args.model)}",
        "-c", f"model_reasoning_effort = {_toml_str(args.effort)}",
        # Display-layer knob: ask for reasoning summaries so any that exist
        # land as thinking events. In practice gpt-5.5 via exec returned
        # summary=[] + encrypted content in probes (2026-07-13) — reasoning
        # is usage-counted (reasoning_output_tokens) but not readable.
        "-c", 'model_reasoning_summary = "detailed"',
        "-c", f"mcp_servers.env.command = {_toml_str(sys.executable)}",
        "-c", f"mcp_servers.env.args = [{_toml_str(str(BRIDGE_PATH))}]",
        "-c", f"mcp_servers.env.env = {{ {env_table} }}",
        # MCP tool calls are approval-gated by default and exec mode
        # auto-cancels them ("user cancelled MCP tool call"); v0.142 accepts
        # only prompt|approve here — the newer docs' "auto" is silently invalid.
        "-c", 'mcp_servers.env.default_tools_approval_mode = "approve"',
        # No AGENTS.md injection: keeps the session hermetic even if the repo
        # grows one later (the Claude cell's setting_sources=[] analog).
        "-c", "project_doc_max_bytes = 0",
    ]


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

    base_prompt = BARE_SYSTEM_PROMPT if args.bare else SYSTEM_PROMPT
    briefing = base_prompt.format(instruction=instruction, budget=args.step_budget)
    if getattr(args, "skill_text", None) and not args.bare:
        briefing += (
            "\n\nYou have been equipped with the following navigation skill."
            " Follow its discipline exactly throughout the episode.\n\n"
            f'<skill name="{args.skill}">\n{args.skill_text}\n</skill>\n'
        )
    # codex keeps its built-in system prompt (the closed agent under test);
    # the briefing rides as the one user prompt — same delivery as opus-lab's
    # --builtin-system-prompt mode on the Claude side.
    prompt = briefing + "\n\n" + FIRST_PROMPT

    argv = build_codex_argv(args, url, run_dir / f"live_{index}") + [prompt]

    trajectory_path = run_dir / f"episode_{index}.jsonl"
    # Raw dump lives in a raw/ subdir, NOT run_dir/episode_{i}.raw.jsonl — the
    # latter matches the backend's non-recursive `episode_*.jsonl` globs and
    # `int("{i}.raw")` crashes episode enumeration (frontend 500).
    raw_path = run_dir / "raw" / f"episode_{index}.jsonl"
    stderr_path = run_dir / "raw" / f"episode_{index}.stderr.log"
    tool_calls: dict[str, int] = {"observe": 0, "step": 0, "look_around": 0, "shell": 0}
    last_step_result: dict[str, Any] | None = None
    usage_totals: dict[str, int] = {}
    thread_id: str | None = None
    exit_code: int | None = None
    t0 = time.time()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_cm = raw_path.open("w") if args.raw_log else contextlib.nullcontext()
    with trajectory_path.open("w") as traj, raw_cm as raw_traj, \
            stderr_path.open("wb") as stderr_file:

        def record(kind: str, payload: dict[str, Any]) -> None:
            traj.write(json.dumps({"t": round(time.time() - t0, 2), "kind": kind, **payload}) + "\n")
            traj.flush()  # live tail -f must see every event as it happens

        def record_raw(event: dict[str, Any]) -> None:
            if raw_traj is None:
                return
            raw_traj.write(json.dumps(
                {"t": round(time.time() - t0, 2), "event": _json_safe(event)},
                ensure_ascii=False) + "\n")
            raw_traj.flush()

        def handle_event(event: dict[str, Any]) -> None:
            nonlocal thread_id, last_step_result
            kind = event.get("type")
            if kind == "thread.started":
                thread_id = event.get("thread_id")
                record("system_init", {"thread_id": thread_id, "model": args.model})
                return
            if kind == "turn.completed":
                for key, value in (event.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage_totals[key] = usage_totals.get(key, 0) + int(value)
                return
            if kind in ("turn.failed", "error"):
                record("driver_error", {"error": _json_safe(event)})
                return
            item = event.get("item") or {}
            item_type = item.get("type")
            if kind == "item.started" and item_type == "mcp_tool_call":
                tool = item.get("tool") or "?"
                if tool in tool_calls:
                    tool_calls[tool] += 1
                record("tool_use", {"id": item.get("id"),
                                    "name": f"mcp__{item.get('server')}__{tool}",
                                    "input": item.get("arguments")})
                return
            if kind == "item.completed":
                if item_type == "mcp_tool_call":
                    texts = _tool_result_texts(item.get("result"))
                    if item.get("error"):
                        texts.append(json.dumps({"error": _json_safe(item["error"])}))
                    parsed = _parse_step_result(texts)
                    if parsed is not None:
                        last_step_result = parsed
                    record("tool_result", {"tool_use_id": item.get("id"), "texts": texts})
                elif item_type == "agent_message":
                    record("assistant_text", {"text": item.get("text", "")})
                elif item_type == "reasoning":
                    text = item.get("text") or ""
                    record("thinking", {"chars": len(text), "text": text})
                elif item_type == "command_execution":
                    # codex's own shell tool — can't be unmounted; sandbox is
                    # read-only and every use is on the record.
                    tool_calls["shell"] += 1
                    record("tool_use", {"id": item.get("id"), "name": "shell",
                                        "input": {"command": item.get("command")}})
                    record("tool_result", {"tool_use_id": item.get("id"),
                                           "texts": [str(item.get("aggregated_output", ""))[:4000]]})
                else:
                    record("driver_error", {"error": {"unhandled_item": _json_safe(item)}})

        record("episode_meta", {"index": index, "episode_id": ep.get("episode_id"),
                                "scene_id": ep.get("scene_id"), "instruction": instruction,
                                "skill": args.skill})
        tool_schemas = await _bridge_tool_schemas(
            {"HABITAT_BARE": "1" if args.bare else "0"}
        )
        record("session_inputs", {
            "model": args.model,
            "effort": args.effort,
            "skill": args.skill,
            "system_prompt": "<codex builtin>",
            "briefing": briefing,
            "first_prompt": FIRST_PROMPT,
            "tool_schemas": tool_schemas,
            "options": {"argv": argv[:-1], "codex_version": args.codex_version,
                        "sandbox": "read-only", "workdir": str(workdir)},
        })

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_file,
            cwd=str(workdir),
            start_new_session=True,  # own PGID so timeout kill reaps MCP child too
            # look_around returns four images in one JSONL event; the default
            # 64 KiB line limit would kill the stream mid-parse
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
                    record("driver_error", {"error": {"nonjson_stdout": line[:300].decode(errors="replace")}})
                    continue
                record_raw(event)
                handle_event(event)
            exit_code = await proc.wait()
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()

        record("result", {"result": {"exit_code": exit_code, "thread_id": thread_id,
                                     "usage": usage_totals}})

        # Episode finished — evaluate while the trajectory file is still open
        # so the final metrics land in the log itself. evaluate is driver-side;
        # the agent never sees it.
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
            "thread_id": thread_id,
            "exit_code": exit_code,
            "usage": usage_totals,
            "total_cost_usd": None,  # subscription auth — no per-token billing
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
    parser.add_argument(
        "--max-turns", type=int, default=100,
        help="feeds the bridge's turn-budget broadcast and STOP gate only; "
        "codex has no SDK-level turn cap (hard caps: step budget + timeout)",
    )
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument("--episode-timeout", type=int, default=2400, help="seconds")
    parser.add_argument(
        "--no-raw-log",
        dest="raw_log",
        action="store_false",
        help="disable the raw codex-event dump (ON by default; "
        "raw/episode_{i}.jsonl, base64 image blobs elided)",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--effort", default="medium",
        help="model_reasoning_effort (codex factory default: medium)",
    )
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
        help="name of a skill dir under beta-coding-agent/skills/; its SKILL.md "
        "body is appended to the briefing (recorded in summary config)",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="vanilla baseline: strip every tuned mechanism (clearance readout, "
        "turn-budget broadcast, STOP-confirmation gate, look_around) and use "
        "the bare observe/step briefing. Overrides --skill.",
    )
    args = parser.parse_args()

    args.skill_text = None
    if args.skill:
        skill_path = REPO_ROOT / "beta-coding-agent" / "skills" / args.skill / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        if text.startswith("---"):  # strip frontmatter
            text = text.split("---", 2)[2]
        args.skill_text = text.strip()
        print(f"[driver] skill '{args.skill}' loaded ({len(args.skill_text)} chars)")

    args.codex_version = subprocess.run(
        ["codex", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"[driver] {args.codex_version} model={args.model} effort={args.effort}")

    urls = [u.strip() for u in (args.server_urls or args.server_url).split(",") if u.strip()]
    for url in urls:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"[driver] {url} healthy: {health.json()['name']}")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    condition = "<bare>" if args.bare else (args.skill or "<mechanisms, no skill>")
    print(f"[driver] dataset={args.dataset} split={args.split} "
          f"workers={len(urls)} condition={condition} -> {run_dir}")
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
                "harness": "codex-cli",
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
