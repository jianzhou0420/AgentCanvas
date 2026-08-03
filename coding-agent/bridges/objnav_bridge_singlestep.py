"""MCP bridge EXPERIMENT VARIANT — single-tool surface for the ObjectNav family.

Mechanism experiment (2026-07-22): the frozen bridge (objnav_bridge.py) forces
a strict observe→step alternation — step() returns only status JSON and the
docstring sends the model back to observe(), so one perception-action cycle
costs TWO harness turns, and measured runs show ~39% of the bill is the
observe half. This variant collapses the cycle:

- ONE registered tool: ``step(actions)`` — executes the actions, then returns
  the forward camera view AFTER the last action as an MCP image (+ status).
- ``step([])`` is legal: no simulator advance, just return the current view
  (this is how the agent sees its FIRST frame — the briefing tells it to open
  with step([])).
- observe() and look_around() do not exist here.

Everything else (action space, STOP gate wording, clearance, budget
broadcast, live spectating, env vars) is copied verbatim from the frozen
bridge so the ONLY variable is the merged cycle. Not part of any frozen
protocol — runs must use --nonstd naming.
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

SERVER_URL = os.environ.get("OBJNAV_SERVER_URL", "http://127.0.0.1:9200")
VERB_PREFIX = os.environ.get("OBJNAV_VERB_PREFIX", "env_objnav")
STEP_BUDGET = int(os.environ.get("OBJNAV_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("OBJNAV_TURN_BUDGET", "0"))
LIVE_DIR = Path(os.environ["OBJNAV_LIVE_DIR"]) if os.environ.get("OBJNAV_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50
BARE = os.environ.get("OBJNAV_BARE") == "1"

MIN_DEPTH_M = 0.5
MAX_DEPTH_M = 5.0

_STEP_DESC = (
    "Execute a sequence of movement actions, in order, then return the "
    "robot's forward-facing camera view (RGB image) after the last action.\n\n"
    "Actions: 0 = STOP (permanently ENDS the episode — issue it only when you "
    "believe the robot is within 0.5 meters of the target object), 1 = move "
    "forward 0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 degrees, "
    "4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 degrees "
    "(tilting changes only the camera pitch, not your position or heading).\n\n"
    "Executes sequentially and halts early if the episode ends. Calling "
    "step([]) with no actions returns the current view without moving — "
    "useful once at the very start; you never need a separate look call "
    "after that, every step already shows you the resulting view."
    + (
        ""
        if BARE else
        " Note: when plenty of budget remains, your FIRST STOP request is "
        "withheld pending a placement check — call step([0]) again to confirm "
        "and execute it."
    )
)

mcp = FastMCP("objnav-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_stop_armed = False
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
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
            "target-object candidate (or stay right here if it scores best) and "
            "call step([0]). Ending without STOP scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is killed. "
            "Stop exploring new areas; commit to your best target-object "
            "candidate, approach it, and STOP before the budget runs out."
        )
    return fields


def _clearance_m(depth_field: dict[str, Any] | None) -> dict[str, float] | None:
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
        name: round(
            MIN_DEPTH_M
            + float(np.percentile(sector, 10)) * (MAX_DEPTH_M - MIN_DEPTH_M),
            1,
        )
        for name, sector in sectors.items()
    }


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


def _current_view() -> tuple[Image, dict[str, Any] | None]:
    """Fetch the forward camera frame (pure read, no sim advance)."""
    global _obs_count
    outputs = _call(f"{VERB_PREFIX}__observe_egocentric", {})
    png = base64.b64decode(outputs["rgb"])
    _obs_count += 1
    _live_frame(png)
    clearance = None if BARE else _clearance_m(outputs.get("depth"))
    return Image(data=png, format="png"), clearance


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> list:
    """Execute movement actions, then return the camera view after the last
    action. step([]) returns the current view without moving."""
    global _steps_taken, _episode_over, _end_reason, _tool_calls, _stop_armed
    _tool_calls += 1
    if _episode_over:
        return [json.dumps(
            {"error": f"episode already over ({_end_reason}); no more steps possible"})]
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return [json.dumps(
            {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"})]
    bad = [a for a in actions if a not in (0, 1, 2, 3, 4, 5)]
    if bad:
        return [json.dumps(
            {"error": f"invalid actions {bad}; valid: 0=STOP 1=FORWARD 2=LEFT "
                      "3=RIGHT 4=LOOK_UP 5=LOOK_DOWN"})]

    if not actions:
        # peek: no sim advance, just the current frame (first-frame path)
        view, clearance = _current_view()
        status = {"executed": 0, "steps_taken_total": _steps_taken,
                  "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
                  "episode_over": False, **_budget_fields()}
        if clearance:
            status["clearance_m"] = clearance
        _live_log({"actions": [], **{k: v for k, v in status.items() if k != "clearance_m"}})
        return [view, json.dumps(status)]

    # STOP confirmation gate — verbatim from the frozen bridge
    remaining = max(0, TURN_BUDGET - _tool_calls) if TURN_BUDGET > 0 else 0
    if 0 in actions and not _stop_armed and TURN_BUDGET > 0 and remaining > 15:
        _stop_armed = True
        prefix = actions[: actions.index(0)]
        withheld = {
            "stop_withheld": True,
            "message": (
                "STOP not executed (first request). You still have "
                f"{remaining} tool calls — verify placement before committing: "
                "(1) look at the returned view and confirm the TARGET OBJECT "
                "itself is in it — not just the room it usually lives in; "
                "(2) check your distance: you must stand within 0.5 meters of "
                "it (if clearance \"center\" reads well above 0.7 with the "
                "target centered, walk closer first). Move if needed, then "
                "call step([0]) again to execute it."
            ),
        }
        if prefix:
            result = _execute_actions(prefix)
            result.update(withheld)
            return _with_view(result)
        return _with_view({**withheld, **_budget_fields()})
    if 0 in actions:
        _stop_armed = True
    return _with_view(_execute_actions(actions))


def _with_view(result: dict[str, Any]) -> list:
    """Append the post-action camera frame unless the episode is over."""
    if _episode_over:
        return [json.dumps(result)]
    view, clearance = _current_view()
    if clearance:
        result["clearance_m"] = clearance
    return [view, json.dumps(result)]


def _execute_actions(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason
    executed = 0
    for action in actions:
        outputs = _call(f"{VERB_PREFIX}__step_discrete", {"action": action})
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
    _live_log({"actions": actions, **result})
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
