"""MCP bridge — expose the habitat env nodeset to an external coding agent.

Stdio MCP server (FastMCP) forwarding a deliberately minimal toolset to a
running ``env_habitat`` auto_host subprocess (ADR-server-001 HTTP surface,
``POST /call/{fn}``). The agent sees exactly two tools:

- ``observe()``      -> egocentric RGB (base64 PNG passthrough as MCP image)
- ``step(actions)``  -> discrete actions 0-3, sanitized gym flags only
- ``look_around()``  -> four labeled views (ahead/right/behind/left) in one
                        call; rotates 360 degrees and restores heading

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
    HABITAT_TURN_BUDGET  the driver's max_turns; when set (>0), every step()
                         result reports tool calls used/remaining and injects
                         an escalating BUDGET_WARNING near exhaustion, so the
                         binding harness budget is environmental state rather
                         than something the model must track itself
    HABITAT_LIVE_DIR     optional dir for live spectating: every observe frame
                         lands as obs_NNNN.png (+ overwritten latest.png),
                         every step call appends a line to actions.log
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("HABITAT_TURN_BUDGET", "0"))
PANO_VIEW_PX = int(os.environ.get("HABITAT_PANO_VIEW_PX", "384"))
LIVE_DIR = Path(os.environ["HABITAT_LIVE_DIR"]) if os.environ.get("HABITAT_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50
# Vanilla-baseline switch: strip every tuned mechanism so the agent sees
# exactly observe(RGB) + step — the "bare" ① condition. look_around is
# withheld driver-side (removed from allowed_tools) and TURN_BUDGET=0 already
# disables the budget broadcast + STOP gate, so BARE only needs to drop the
# depth-derived clearance and keep the tool descriptions honest.
BARE = os.environ.get("HABITAT_BARE") == "1"

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not advance the simulator or "
    "consume step budget."
    if BARE else
    "Look through the robot's forward-facing camera.\n\n"
    "Returns the current egocentric RGB view plus a clearance readout: "
    "distance in meters to the nearest obstacle in the left/center/right "
    "thirds of the view (10.0 = open, 10 m or more). If the goal object is "
    "centered ahead, clearance \"center\" is your true distance to it. Pure "
    "read — does not advance the simulator or consume step budget."
)
_STEP_DESC = (
    "Execute a sequence of movement actions, in order.\n\n"
    "Actions: 0 = STOP (permanently ENDS the episode — issue it only when you "
    "believe the robot is within 3 meters of the goal), 1 = move forward "
    "0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.\n\n"
    "Executes sequentially and halts early if the episode ends. Returns how "
    "many actions ran, total steps taken, remaining budget, and whether the "
    "episode is over. The camera view changes after stepping — call observe() "
    "to see the result."
    + (
        ""
        if BARE else
        " Note: when plenty of budget remains, your FIRST STOP request is "
        "withheld pending a placement check — call step([0]) again to confirm "
        "and execute it."
    )
)

mcp = FastMCP("habitat-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_stop_armed = False  # first budget-rich STOP request is withheld; second executes
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn.

    The driver's max_turns is the budget that actually kills sessions, but the
    model has no way to observe it. Report it in every step result and escalate
    to explicit orders near exhaustion; a session that ends without STOP scores
    exactly zero.
    """
    if TURN_BUDGET <= 0:
        return {}
    remaining = max(0, TURN_BUDGET - _tool_calls)
    fields: dict[str, Any] = {
        "tool_calls_used": _tool_calls,
        "tool_calls_remaining": remaining,
    }
    if remaining <= 10:
        fields["BUDGET_WARNING"] = (
            f"CRITICAL — only {remaining} tool calls left before this session is "
            "killed. Execute your terminal stop protocol NOW: move to your best "
            "goal candidate (or stay right here if it scores best) and call "
            "step([0]). Ending without STOP scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is killed. "
            "Stop exploring new areas; commit to your best goal candidate, "
            "approach it, and STOP before the budget runs out."
        )
    return fields


def _clearance_m(depth_field: dict[str, Any] | None) -> dict[str, float] | None:
    """Metric free-space readout from the raw depth frame.

    Distance (m) to the nearest obstacle in the left/center/right thirds of
    the view, from an eye-level band (rows 40-65%) so floor and ceiling don't
    pollute it; 10th percentile per sector rides over stray pixels. VLN-CE
    depth is normalized [0,1] over 0-10 m, so 10.0 means "open, >= 10 m".
    """
    if not isinstance(depth_field, dict) or "__ndarray__" not in depth_field:
        return None
    try:
        arr = np.frombuffer(
            base64.b64decode(depth_field["__ndarray__"]),
            dtype=depth_field.get("dtype", "float32"),
        ).reshape(depth_field["shape"])
    except Exception:
        return None
    h, w = arr.shape[:2]
    band = arr[int(h * 0.40) : int(h * 0.65), :]
    sectors = {
        "left": band[:, : w // 3],
        "center": band[:, w // 3 : 2 * w // 3],
        "right": band[:, 2 * w // 3 :],
    }
    return {
        name: round(float(np.percentile(sector, 10)) * 10.0, 1)
        for name, sector in sectors.items()
    }


def _downscale(png: bytes, side: int) -> bytes:
    """Shrink a PNG so four panorama views fit one MCP message comfortably."""
    img = PILImage.open(BytesIO(png))
    if max(img.size) <= side:
        return png
    img = img.resize((side, side), PILImage.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    global _obs_count, _tool_calls
    _tool_calls += 1
    outputs = _call("env_habitat__observe_egocentric", {})
    png = base64.b64decode(outputs["rgb"])
    _obs_count += 1
    _live_frame(png)
    if BARE:
        return [Image(data=png, format="png")]
    status = {"clearance_m": _clearance_m(outputs.get("depth")), **_budget_fields()}
    return [Image(data=png, format="png"), json.dumps(status)]


def look_around() -> list:
    """Scan the surroundings in ONE call: rotates a full 360 degrees and
    returns four labeled views — ahead (0°), right (+90°), behind (+180°),
    left (+270°) — then restores the original heading exactly.

    Costs 24 low-level turn steps from the step budget but only one tool
    call. Use this at junctions and decision points instead of turning 15
    degrees at a time; your position and final heading are unchanged.
    """
    global _obs_count, _steps_taken, _episode_over, _end_reason, _tool_calls
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more steps possible"]

    content: list[Any] = []
    for label in ("ahead (0°)", "right (+90°)", "behind (+180°)", "left (+270°)"):
        outputs = _call("env_habitat__observe_egocentric", {})
        png = base64.b64decode(outputs["rgb"])
        _obs_count += 1
        _live_frame(png)
        clearance = _clearance_m(outputs.get("depth"))
        if clearance:
            label += (
                f" — clearance m L/C/R: {clearance['left']}/{clearance['center']}"
                f"/{clearance['right']}"
            )
        content.extend([label, Image(data=_downscale(png, PANO_VIEW_PX), format="png")])
        # rotate 90° right toward the next view; the 4th rotation restores
        # the original heading (4 x 90° = 360°)
        for _ in range(6):
            outputs = _call("env_habitat__step_discrete", {"action": 3})
            _steps_taken += 1
            if outputs.get("terminated") or outputs.get("truncated"):
                _episode_over = True
                _end_reason = "step_budget_exhausted"
                break
        if _episode_over:
            break

    status = {
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "heading_restored": not _episode_over,
        **_budget_fields(),
    }
    content.append(json.dumps(status))
    _live_log({"look_around": True, **status})
    return content


# look_around is a tuned mechanism (iteration 2) — register it only OUTSIDE the
# bare baseline. allowed_tools alone cannot withhold it: the driver runs with
# permission_mode="bypassPermissions", which lets the model call any REGISTERED
# tool regardless of the allowlist. Not registering it is the only reliable gate.
if not BARE:
    mcp.tool()(look_around)


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> dict[str, Any]:
    """Execute a sequence of movement actions, in order.

    Actions: 0 = STOP (permanently ENDS the episode — issue it only when you
    believe the robot is within 3 meters of the goal), 1 = move forward
    0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

    Executes sequentially and halts early if the episode ends. Returns how
    many actions ran, total steps taken, remaining budget, and whether the
    episode is over. The camera view changes after stepping — call observe()
    to see the result. Note: when plenty of budget remains, your FIRST STOP
    request is withheld pending a placement check — call step([0]) again to
    confirm and execute it.
    """
    global _steps_taken, _episode_over, _end_reason, _tool_calls, _stop_armed
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more steps possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    bad = [a for a in actions if a not in (0, 1, 2, 3)]
    if bad:
        return {"error": f"invalid actions {bad}; valid: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT"}

    # STOP confirmation gate: a budget-rich first STOP is withheld so the
    # agent verifies placement before committing (near-miss rush stops are
    # the dominant recoverable failure). Near budget exhaustion the gate is
    # open — salvage stops must never be blocked.
    remaining = max(0, TURN_BUDGET - _tool_calls) if TURN_BUDGET > 0 else 0
    if 0 in actions and not _stop_armed and TURN_BUDGET > 0 and remaining > 15:
        _stop_armed = True
        prefix = actions[: actions.index(0)]
        withheld = {
            "stop_withheld": True,
            "message": (
                "STOP not executed (first request). You still have "
                f"{remaining} tool calls — verify placement before committing: "
                "(1) look_around(); (2) re-read the instruction's FINAL clause "
                "word by word; (3) apply the placement rules (several similar "
                "spots -> the one farther along your travel direction; 'all "
                "the way / to the end' -> keep going until center clearance "
                "< 1.5 m; 'door/doorway' -> stand at the opening). Move if a "
                "better spot exists, then call step([0]) again to execute it."
            ),
        }
        if prefix:
            # run the movement part of the call; only the STOP is withheld
            result = _execute_actions(prefix)
            result.update(withheld)
            return result
        return {**withheld, **_budget_fields()}
    if 0 in actions:
        _stop_armed = True  # confirmed (or budget-critical) — let it through
    return _execute_actions(actions)


def _execute_actions(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason
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
        **_budget_fields(),
    }
    if not _episode_over and not BARE:
        # post-move clearance (pure read; no sim advance) — metric feedback
        # for the approach phase without an extra observe round-trip
        outputs = _call("env_habitat__observe_egocentric", {})
        clearance = _clearance_m(outputs.get("depth"))
        if clearance:
            result["clearance_m"] = clearance
    _live_log({"actions": actions, **result})
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
