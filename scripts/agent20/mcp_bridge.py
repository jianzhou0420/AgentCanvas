"""MCP bridge — expose the habitat env nodeset to an external coding agent.

Stdio MCP server (FastMCP) forwarding a deliberately minimal toolset to a
running ``env_habitat`` auto_host subprocess (ADR-server-001 HTTP surface,
``POST /call/{fn}``). The agent sees exactly two tools:

- ``observe()``      -> egocentric RGB (base64 PNG passthrough as MCP image)
- ``step(actions)``  -> discrete actions 0-3, sanitized gym flags only

Episode selection and metric collection stay driver-side (run_episodes.py);
the agent never sees SR/SPL, reward, pose, depth, or panoramas.

One bridge process serves one agent session = one episode (the Agent SDK
spawns a fresh stdio server per session), so per-episode step accounting can
live in module globals.

Env vars:
    HABITAT_SERVER_URL   auto_host base URL (default http://127.0.0.1:9200)
    HABITAT_STEP_BUDGET  advisory low-level step budget echoed to the agent
                         (default 500 — habitat's MAX_EPISODE_STEPS truncates
                         authoritatively regardless)
    HABITAT_LIVE_DIR     optional dir for live spectating: every observe frame
                         lands as obs_NNNN.png (+ overwritten latest.png),
                         every step call appends a line to actions.log
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
LIVE_DIR = Path(os.environ["HABITAT_LIVE_DIR"]) if os.environ.get("HABITAT_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50

mcp = FastMCP("habitat-env")

_steps_taken = 0
_obs_count = 0
_episode_over = False
_end_reason: str | None = None
_t0 = time.time()


def _live_frame(png: bytes) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / f"obs_{_obs_count:04d}_step{_steps_taken:03d}.png").write_bytes(png)
    (LIVE_DIR / "latest.png").write_bytes(png)


def _live_log(entry: dict[str, Any]) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with (LIVE_DIR / "actions.log").open("a") as fh:
        fh.write(json.dumps({"t": round(time.time() - _t0, 1), **entry}) + "\n")


def _call(function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/call/{function_name}", json={"inputs": inputs}, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["outputs"]


@mcp.tool()
def observe() -> Image:
    """Look through the robot's forward-facing camera.

    Returns the current egocentric RGB view. Pure read — does not advance
    the simulator or consume step budget.
    """
    global _obs_count
    outputs = _call("env_habitat__observe_egocentric", {})
    png = base64.b64decode(outputs["rgb"])
    _obs_count += 1
    _live_frame(png)
    return Image(data=png, format="png")


@mcp.tool()
def step(actions: list[int]) -> dict[str, Any]:
    """Execute a sequence of movement actions, in order.

    Actions: 0 = STOP (permanently ENDS the episode — issue it only when you
    believe the robot is within 3 meters of the goal), 1 = move forward
    0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

    Executes sequentially and halts early if the episode ends. Returns how
    many actions ran, total steps taken, remaining budget, and whether the
    episode is over. The camera view changes after stepping — call observe()
    to see the result.
    """
    global _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more steps possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    bad = [a for a in actions if a not in (0, 1, 2, 3)]
    if bad:
        return {"error": f"invalid actions {bad}; valid: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT"}

    executed = 0
    for action in actions:
        outputs = _call("env_habitat__step_discrete", {"action": action})
        executed += 1
        _steps_taken += 1
        terminated = bool(outputs.get("terminated"))
        truncated = bool(outputs.get("truncated"))
        if terminated or truncated:
            _episode_over = True
            if action == 0:
                _end_reason = "stop_called"
            elif truncated:
                _end_reason = "step_budget_exhausted"
            else:
                _end_reason = "terminated"
            break

    result = {
        "executed": executed,
        "requested": len(actions),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
    }
    _live_log({"actions": actions, **result})
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
