"""ReAct-harness VLN driver — mini-swe-agent over the habitat toolset.

The claude-SDK path's driver (beta-coding-agent/run_episodes.py) with the SDK
session swapped for mini-swe-agent's DefaultAgent; everything driver-side is
kept: per episode, place it via the env-panel HTTP surface, read the
instruction driver-side, run ONE fresh agent (own model + environment +
toolset), then read habitat's own measures via ``env_habitat__evaluate`` —
the same ruler as the verified baselines and the SDK runs.

Artifacts land under ``outputs/beta-react-harness/{run_name}/`` in the same
shape the SDK path produced (episode_{i}.jsonl curated events, live_{i}/
frames, summary.json), plus mini's own per-step trajectory dump under raw/.

Unlike the SDK path (subscription auth), litellm bills through the provider
API key — ANTHROPIC_API_KEY must be SET for anthropic/* models.

Usage (agentcanvas env; habitat auto_host must already be up — see README):
    python beta-react-harness/run_episodes.py --episodes 0 --model anthropic/claude-sonnet-5
    python beta-react-harness/run_episodes.py --episodes 0-9 --split rand100 --bare ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat sibling modules

from jinja2 import StrictUndefined, Template

from env import HabitatEnvironment
from model import NavToolsModel
from nav_agent import NavAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "beta-react-harness"

# Same text as the claude-SDK path's SYSTEM_PROMPT / BARE_SYSTEM_PROMPT, with
# jinja slots ({{ task }}, {{ step_budget }}) instead of str.format fields.
# check_equivalence.py asserts the rendered outputs are byte-identical.
SYSTEM_TEMPLATE = """\
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

"{{ task }}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {{ step_budget }} movement actions; each step() result reports \
roughly how many remain.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

BARE_SYSTEM_TEMPLATE = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Your task is to follow this navigation instruction to its endpoint:

"{{ task }}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {{ step_budget }} movement actions.
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


# ── episode loop (sync; one worker thread per env server) ──


def run_episode(args: argparse.Namespace, url: str, index: int, run_dir: Path) -> dict[str, Any]:
    panel_field(url, "episode_index", index)
    panel_action(url, "play")

    # reset re-arms only a done episode; the one just placed is read untouched.
    reset_config = (
        {"rgb_resolution": str(args.rgb_resolution)} if args.rgb_resolution else None
    )
    ep = call_function(url, "env_habitat__reset", {"trigger": "driver"}, reset_config)
    instruction = ep["instruction"]

    system_template = BARE_SYSTEM_TEMPLATE if args.bare else SYSTEM_TEMPLATE
    if getattr(args, "skill_text", None) and not args.bare:
        system_template += (
            "\n\nYou have been equipped with the following navigation skill."
            " Follow its discipline exactly throughout the episode.\n\n"
            f'<skill name="{args.skill}">\n{args.skill_text}\n</skill>\n'
        )

    env = HabitatEnvironment(
        server_url=url,
        bare=args.bare,
        step_budget=args.step_budget,
        # bare = no turn-budget broadcast and no STOP gate (both keyed off
        # turn_budget>0 in the toolset, exactly like the bridge's TURN_BUDGET).
        turn_budget=0 if args.bare else args.max_turns,
        live_dir=str(run_dir / f"live_{index}"),
    )
    set_cache_control = (
        "default_end"
        if any(s in (args.model or "").lower() for s in ["anthropic", "sonnet", "opus", "claude"])
        else None
    )
    model = NavToolsModel(
        model_name=args.model,
        tools=env.toolset.tool_schemas(),
        image_window=args.image_window,
        set_cache_control=set_cache_control,
        model_kwargs={"drop_params": True},
    )

    trajectory_path = run_dir / f"episode_{index}.jsonl"
    raw_traj_path = run_dir / "raw" / f"episode_{index}.traj.json"
    t0 = time.time()

    with trajectory_path.open("w") as traj:

        def record(kind: str, payload: dict[str, Any]) -> None:
            traj.write(
                json.dumps({"t": round(time.time() - t0, 2), "kind": kind, **payload}) + "\n"
            )
            traj.flush()  # live tail -f must see every event as it happens

        record("episode_meta", {
            "index": index,
            "episode_id": ep.get("episode_id"),
            "scene_id": ep.get("scene_id"),
            "instruction": instruction,
            "skill": args.skill,
        })
        rendered_system = Template(system_template, undefined=StrictUndefined).render(
            task=instruction, step_budget=args.step_budget
        )
        record("session_inputs", {
            "harness": "mini-swe-agent",
            "model": args.model,
            "skill": args.skill,
            "system_prompt": rendered_system,
            "first_prompt": FIRST_PROMPT,
            "tool_schemas": env.toolset.tool_schemas(),
            "agent_config": {
                "step_limit": args.max_turns,
                "cost_limit": args.cost_limit,
                "wall_time_limit_seconds": args.episode_timeout,
            },
            "model_config": {"image_window": args.image_window,
                             "set_cache_control": set_cache_control},
            "environment_config": env.config.model_dump(),
        })

        agent = NavAgent(
            model,
            env,
            system_template=system_template,
            instance_template=FIRST_PROMPT,
            step_limit=args.max_turns,
            cost_limit=args.cost_limit,
            wall_time_limit_seconds=args.episode_timeout,
            output_path=raw_traj_path,
            event_hook=record,
        )
        exit_info: dict[str, Any] = {}
        error: str | None = None
        try:
            exit_info = agent.run(task=instruction)
        except Exception as exc:  # noqa: BLE001 — run() already logged + saved
            error = repr(exc)
            record("driver_error", {"error": error})

        # Episode finished — evaluate while the trajectory file is still open
        # so the final metrics land in the log itself. Driver-side; the agent
        # never sees it.
        metrics: dict[str, Any] = {}
        try:
            metrics_out = call_function(url, "env_habitat__evaluate", {"trigger": "driver"})
            metrics = metrics_out.get("metrics") or {}
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
        except Exception as exc:  # noqa: BLE001
            record("driver_error", {"error": f"evaluate failed: {exc!r}"})
        record("episode_metrics", {"metrics": metrics})

    episode: dict[str, Any] = {
        "index": index,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "instruction": instruction,
        "metrics": metrics,
        "agent": {
            "tool_calls": env.toolset.calls_by_tool,
            "env_steps": env.toolset.steps_taken,
            "end_reason": env.toolset.end_reason,
            "called_stop": env.toolset.end_reason == "stop_called",
            "exit_status": exit_info.get("exit_status"),
            "num_llm_calls": agent.n_calls,
            "cost_usd": round(agent.cost, 4),
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    if error:
        episode["error"] = error
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
    sys.stdout.reconfigure(line_buffering=True)  # progress must survive redirection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:9200")
    parser.add_argument(
        "--server-urls", default=None,
        help="comma-separated auto_host URLs; one parallel worker per server",
    )
    parser.add_argument("--dataset", default="R2R-CE")
    parser.add_argument("--split", default="rand100")
    parser.add_argument("--episodes", required=True, help='e.g. "0", "0-9", "0,3,7"')
    parser.add_argument("--max-turns", type=int, default=100,
                        help="agent step_limit (LLM calls) — SDK max_turns parity")
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument("--cost-limit", type=float, default=5.0,
                        help="per-episode USD cap (litellm cost tracking)")
    parser.add_argument("--image-window", type=int, default=0,
                        help="keep only newest K camera frames in the API payload "
                        "(0 = keep all; mini has no other context management)")
    parser.add_argument("--episode-timeout", type=int, default=2400,
                        help="seconds; enforced in-loop via wall_time_limit_seconds")
    parser.add_argument("--model", required=True,
                        help="litellm model name, e.g. anthropic/claude-sonnet-5")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rgb-resolution", type=int, default=None,
                        help="RGB render size in px (blank = YAML default 224)")
    parser.add_argument(
        "--skill", default=None,
        help="skill dir under beta-coding-agent/skills/; SKILL.md body is "
        "appended to the system prompt (same mechanism as the SDK path)",
    )
    parser.add_argument(
        "--bare", action="store_true",
        help="vanilla ① baseline: bare observe/step prompt, no clearance readout, "
        "no turn-budget broadcast, no STOP gate, no look_around. Overrides --skill.",
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

    # litellm bills through the provider API key — the OPPOSITE of the SDK
    # path, which stripped ANTHROPIC_API_KEY to ride the subscription.
    if "anthropic" in args.model.lower() and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[driver] ANTHROPIC_API_KEY is not set — litellm needs it (API billing).")

    urls = [u.strip() for u in (args.server_urls or args.server_url).split(",") if u.strip()]
    for url in urls:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"[driver] {url} healthy: {health.json()['name']}")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[driver] dataset={args.dataset} split={args.split} "
          f"workers={len(urls)} -> {run_dir}")
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
                "harness": "mini-swe-agent",
                "config": {k: v for k, v in vars(args).items()},
                "servers": urls,
                "aggregate": aggregate([e for e in episodes if "error" not in e]),
                "episodes": episodes,
            }
            summary_path.write_text(json.dumps(summary, indent=2))

    async def worker(position: int, url: str) -> None:
        await asyncio.sleep(position * 2)  # stagger cold-server scene loads
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"[driver] episode {index} starting on {url}")
            try:
                # wall_time_limit_seconds enforces the budget in-loop; this
                # outer timeout is a backstop for a wedged HTTP call.
                episode = await asyncio.wait_for(
                    asyncio.to_thread(run_episode, args, url, index, run_dir),
                    timeout=args.episode_timeout + 600,
                )
            except asyncio.TimeoutError:
                print(f"[driver] episode {index} TIMED OUT (backstop)")
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
                f"stop={episode['agent'].get('called_stop')} "
                f"cost=${episode['agent'].get('cost_usd')}"
            )

    await asyncio.gather(*(worker(i, url) for i, url in enumerate(urls)))

    print(f"[driver] run complete -> {summary_path}")
    print(json.dumps(aggregate([e for e in episodes if "error" not in e]), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
