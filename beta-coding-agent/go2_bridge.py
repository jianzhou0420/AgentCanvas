"""MCP bridge — expose a real Unitree Go2 to an external coding agent.

Structural copy of the frozen ``mcp_bridge.py``: stdio MCP server (FastMCP)
forwarding a deliberately minimal toolset over ``POST /call/{fn}``. Same tool
names, same action integers, same policy layer, so ``claude_sdk.py``'s hardcoded
``mcp__env__{observe,step,look_around}`` allowlist needs no edit.

Action magnitudes are full habitat parity: 0.25 m forward, 15 degree turns
(user decision 2026-07-20 night). Early free-gait tests condemned 15°/0.25 m
(±42% scatter), but under the pinned StaticWalk gait (the app's 常规 style)
with coast compensation they hold to about ±6% with <2% bias — see
go2_host.py's calibration notes. ``look_around`` uses habitat's 6 steps per
quarter turn.

  observe()       -> head-camera RGB
  step(actions)   -> discrete actions 0-3, sanitized flags only
  look_around()   -> four labeled views; rotates 360 degrees

The counterpart of habitat's auto_host is ``go2_host.py`` (same directory), which
is deployed to and runs on the ROBOT'S host machine — CycloneDDS is layer-2, so
the SDK must sit on the dog's wire. This process runs locally and speaks plain
HTTP to it, which is the whole point of the split: the remote box holds only an
SDK-to-HTTP shim, and everything the harness touches stays here.

WHAT DIFFERS FROM THE HABITAT BRIDGE, and why — the honest list:

- **No clearance readout.** habitat's ``clearance_m`` is computed from the depth
  frame; the Go2 head camera is RGB only. Rather than fake metric free-space
  from a monocular image, the readout is absent and the tool descriptions do not
  promise it. This is the one tuned mechanism that does not survive the port.
- **Distances are closed-loop measured, not nominal.** Calibrated on the real
  dog (2026-07-20, StaticWalk gait + coast compensation): 0.25 m → −0.7%,
  15° → +1.7%, both σ≈±6%. Step results report the measured values.
- **look_around does not exactly restore heading.** Four 90-degree turns still
  accumulate a small closed-loop tolerance each, so the result reports
  ``heading_restored: "approximate"``. In habitat it is exact.
- **Actions cost wall-clock.** Each one drives the real robot for roughly
  STEP_M / MAX_VX seconds plus settle. Batching via ``step([...])`` is not just a
  turn-budget saver here, it is the difference between minutes and tens of
  minutes.

Env vars:
    GO2_SERVER_URL    go2_host base URL (default http://10.12.76.41:9300)
    GO2_STEP_BUDGET   advisory step budget echoed to the agent (host truncates
                      authoritatively regardless)
    GO2_TURN_BUDGET   the driver's max_turns; when set (>0), every step() result
                      reports tool calls used/remaining and escalates near
                      exhaustion, same as habitat
    GO2_LIVE_DIR      optional dir for live spectating: every observe frame lands
                      as obs_NNNN.jpg (+ overwritten latest.jpg), every step call
                      appends a line to actions.log
    GO2_BARE          "1" strips the tuned mechanisms (STOP gate, look_around)
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

SERVER_URL = os.environ.get("GO2_SERVER_URL", "http://10.12.76.41:9300")
STEP_BUDGET = int(os.environ.get("GO2_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("GO2_TURN_BUDGET", "0"))
LIVE_DIR = Path(os.environ["GO2_LIVE_DIR"]) if os.environ.get("GO2_LIVE_DIR") else None
# A real robot cannot be handed a 50-action burst the operator cannot interrupt;
# habitat's 50 is safe because it is a simulator.
MAX_ACTIONS_PER_CALL = 12
BARE = os.environ.get("GO2_BARE") == "1"

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not move the robot or consume step "
    "budget."
)
_STEP_DESC = (
    "Execute a sequence of movement actions on a REAL robot, in order.\n\n"
    "Actions: 0 = STOP (permanently ENDS the episode — issue it only when you "
    "believe the robot is within 3 meters of the goal), 1 = move forward "
    "0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.\n\n"
    "Movement is closed-loop on the robot's own odometry and IMU, so distances "
    "and angles are held to a small tolerance rather than estimated; the result "
    "reports what was actually measured. "
    "Executes sequentially and halts early if the episode ends. Each action "
    "takes about a second of real time, so prefer batching several per call. "
    "Returns how many actions ran, total steps taken, remaining budget, and "
    "whether the episode is over. The camera view changes after stepping — call "
    "observe() to see the result."
    + (
        ""
        if BARE else
        " Note: when plenty of budget remains, your FIRST STOP request is "
        "withheld pending a placement check — call step([0]) again to confirm "
        "and execute it."
    )
)

mcp = FastMCP("go2-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_stop_armed = False
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn. Verbatim from
    the habitat bridge: the driver's max_turns is what actually kills sessions
    and the model cannot otherwise observe it."""
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


def _live_frame(jpg: bytes) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / f"obs_{_obs_count:04d}_step{_steps_taken:03d}.jpg").write_bytes(jpg)
    (LIVE_DIR / "latest.jpg").write_bytes(jpg)


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
    outputs = _call("env_go2__observe_egocentric", {})
    raw = outputs.get("rgb") or ""
    if not raw:
        return [json.dumps({"error": "camera returned no frame",
                            "detail": outputs.get("info")})]
    jpg = base64.b64decode(raw)
    _obs_count += 1
    _live_frame(jpg)
    if BARE:
        return [Image(data=jpg, format="jpeg")]
    status = _budget_fields()
    content: list[Any] = [Image(data=jpg, format="jpeg")]
    if status:
        content.append(json.dumps(status))
    return content


def look_around() -> list:
    """Scan the surroundings in ONE call: rotates a full 360 degrees and returns
    four labeled views — ahead (0°), right (+90°), behind (+180°), left (+270°).

    Costs 24 low-level turn steps and roughly half a minute of real time, but
    only one tool call. Use this at junctions and decision points instead of
    turning 15 degrees at a time. Your position is unchanged; the final heading
    returns to approximately the original one.
    """
    global _obs_count, _steps_taken, _episode_over, _end_reason, _tool_calls
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more steps possible"]

    content: list[Any] = []
    for label in ("ahead (0°)", "right (+90°)", "behind (+180°)", "left (+270°)"):
        outputs = _call("env_go2__observe_egocentric", {})
        raw = outputs.get("rgb") or ""
        if raw:
            jpg = base64.b64decode(raw)
            _obs_count += 1
            _live_frame(jpg)
            content.extend([label, Image(data=jpg, format="jpeg")])
        else:
            content.append(f"{label} — camera returned no frame")
        # rotate 90° right toward the next view; the 4th rotation brings the
        # heading back to approximately where it started (4 x 90° = 360°).
        for _ in range(6):
            outputs = _call("env_go2__step_discrete", {"action": 3})
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
        "heading_restored": "approximate" if not _episode_over else False,
        **_budget_fields(),
    }
    content.append(json.dumps(status))
    _live_log({"look_around": True, **status})
    return content


# Same gate as the habitat bridge: allowed_tools cannot withhold a tool from a
# driver running permission_mode="bypassPermissions", so BARE must not register it.
if not BARE:
    mcp.tool()(look_around)


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> dict[str, Any]:
    """Execute a sequence of movement actions on a REAL robot, in order.

    Actions: 0 = STOP (permanently ENDS the episode), 1 = move forward 0.25 m,
    2 = turn left 15 degrees, 3 = turn right 15 degrees. Closed-loop on the
    robot's odometry and IMU. Halts early if the episode ends.
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
                "spots -> the one farther along your travel direction; 'all the "
                "way / to the end' -> keep going until you are close to the wall; "
                "'door/doorway' -> stand at the opening). Move if a better spot "
                "exists, then call step([0]) again to execute it."
            ),
        }
        if prefix:
            result = _execute_actions(prefix)
            result.update(withheld)
            return result
        return {**withheld, **_budget_fields()}
    if 0 in actions:
        _stop_armed = True
    return _execute_actions(actions)


def _execute_actions(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason
    executed = 0
    measured: list[dict[str, Any]] = []   # per-action truth from the host
    for action in actions:
        try:
            outputs = _call("env_go2__step_discrete", {"action": action})
        except requests.RequestException as exc:
            # The host stops the dog on its own error paths; surface the break
            # rather than silently reporting a shorter successful run.
            _episode_over = True
            _end_reason = "host_unreachable"
            return {"error": f"robot host unreachable: {exc}", "executed": executed,
                    "requested": len(actions), "steps_taken_total": _steps_taken,
                    "episode_over": True, "end_reason": _end_reason}
        # The host reports a refusal (stale episode, invalid action) as
        # terminated+info.error. Surfacing it matters: swallowing it makes a
        # host that was never reset look exactly like a normal episode end,
        # which would silently score as a legitimate failed run.
        host_error = (outputs.get("info") or {}).get("error")
        if host_error:
            _episode_over = True
            _end_reason = "host_error"
            return {"error": f"robot host refused the action: {host_error}",
                    "hint": "the host may need env_go2__reset (driver-side)",
                    "executed": executed, "requested": len(actions),
                    "steps_taken_total": _steps_taken, "episode_over": True,
                    "end_reason": _end_reason}
        executed += 1
        _steps_taken += 1
        measured.append(outputs.get("info") or {})
        if bool(outputs.get("terminated")) or bool(outputs.get("truncated")):
            _episode_over = True
            if action == 0:
                _end_reason = "stop_called"
            elif outputs.get("truncated"):
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
        **_summarize(measured),
        **_budget_fields(),
    }
    _live_log({"actions": actions, **result})
    return result


def _summarize(measured: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-action host measurements into fields worth the model's context.

    Reports totals rather than a per-action list (a 12-action call would
    otherwise dump 12 dicts), and surfaces the two conditions the model should
    actually react to: control that fell back to open-loop, and moves that hit
    the timeout without reaching target — both mean "you moved less than you
    asked for", which changes what the next observe will show.
    """
    if not measured:
        return {}
    out: dict[str, Any] = {}
    turned = sum(m.get("actual_deg", 0.0) for m in measured)
    advanced = sum(m.get("actual_m", 0.0) for m in measured)
    if turned:
        out["turned_deg"] = round(turned, 1)
    if advanced:
        out["advanced_m"] = round(advanced, 3)
    degraded = [i for i, m in enumerate(measured) if not m.get("closed_loop", True)]
    stalled = [i for i, m in enumerate(measured) if m.get("timed_out")]
    if degraded:
        out["open_loop_actions"] = degraded
        out["WARNING"] = ("robot feedback unavailable for some actions — they ran "
                          "open-loop and moved substantially less than requested")
    if stalled:
        out["stalled_actions"] = stalled
        out["WARNING"] = ("some actions timed out before reaching target — the "
                          "robot may be blocked; re-observe before continuing")
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
