"""MCP bridge — expose the HM-EQA env nodeset to an external coding agent.

Structural copy of the ObjectNav bridge (objnav_bridge.py), re-pointed at
env_hmeqa's gym surface for the coding-agent EQA line (2026-07-29). The
benchmark is HM-EQA (Ren et al. 2024, explore-eqa): a multiple-choice
question about an HM3D building, answered by exploring it.

The agent sees the std-board tool surface with ONE structural difference —
the episode ends by ANSWERING, not by stopping at a goal:

- ``observe()``      -> egocentric RGB (base64 PNG passthrough as MCP image)
- ``step(actions)``  -> discrete actions 1-5 (no STOP — see answer(); 4/5 =
                        camera tilt, masked when HMEQA_TILT=0)
- ``answer(letter)`` -> permanently ends the episode with "A"/"B"/"C"/"D"
- ``look_around()``  -> four labeled views in one call (non-bare only)

Deliberate differences from the benchmark-native protocol (explore-eqa
teleports to TSDF-planned waypoints; there is no native discrete surface),
all surfaced to the model rather than hidden:

- discrete actions 1-5 (0.25 m forward / 30-degree turns / 30-degree
  camera tilt, pitch clamped to +-60) via ``env_hmeqa__step_discrete`` —
  magnitudes mirror the ObjectNav line so the bare toolface stays
  comparable across board lines. Tilt actions added 2026-07-29 (user
  decision B: the fixed -30 pitch made near-overhead ceiling fixtures
  structurally unobservable — see the ep4 smoke analysis); HMEQA_TILT=0
  masks them for the 1-3-only ablation
- the step budget is enforced HERE (the env has no cap on the discrete
  surface): once it runs out, movement is refused but observe()/answer()
  keep working — an out-of-budget episode can still be answered
- the camera is the benchmark's own: 640x480, hfov 120, STARTING tilted
  30 degrees DOWN (explore-eqa cfg/vlm_exp.yaml); tilt actions move it
  from there
- clearance (non-bare only) reads the TOP band of the frame (rows 0-25%),
  which straddles the horizon under the -30-degree tilt; depth here is raw
  metric meters (no ObjectNav-style normalization), capped at 10.0

Episode selection and metric collection stay driver-side. The agent's
answer travels through the tool-result channel: answer() returns a JSON
carrying ``steps_taken_total`` (so the driver's EventSink parses it as the
last step result) plus ``answer``; the driver reads it from there and calls
``env_hmeqa__evaluate`` with ``pred_letter``. The agent never sees the
ground truth.

One bridge process serves one agent session = one episode (the Agent SDK
spawns a fresh stdio server per session), so per-episode step accounting
can live in module globals.

Env vars:
    HMEQA_SERVER_URL    auto_host base URL (default http://127.0.0.1:9240)
    HMEQA_STEP_BUDGET   movement budget enforced by this bridge (default 500)
    HMEQA_TURN_BUDGET   the driver's max_turns; when set (>0), every step()
                        result reports tool calls used/remaining and injects
                        an escalating BUDGET_WARNING near exhaustion
    HMEQA_BARE          "1" = bare toolface (no clearance, no look_around)
    HMEQA_TILT          "1" (default) = camera-tilt actions 4/5 available;
                        "0" masks them (validation + description + briefing
                        all follow the driver-side flag)
    HMEQA_PANO_VIEW_PX  optional downscale for look_around views (0 = native)
    HMEQA_LIVE_DIR      optional dir for live spectating: every observe frame
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

SERVER_URL = os.environ.get("HMEQA_SERVER_URL", "http://127.0.0.1:9240")
STEP_BUDGET = int(os.environ.get("HMEQA_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("HMEQA_TURN_BUDGET", "0"))
# Camera-tilt actions 4/5 (the native −30° pitch makes near-overhead
# ceiling fixtures
# structurally unobservable, so the discrete surface gets LOOK_UP/LOOK_DOWN
# like the ObjectNav line). HMEQA_TILT=0 masks them — action validation,
# tool description, and the driver-side briefing all key off the same flag,
# so the agent's world stays self-consistent either way.
TILT = os.environ.get("HMEQA_TILT", "1") == "1"
PANO_VIEW_PX = int(os.environ.get("HMEQA_PANO_VIEW_PX", "0"))
LIVE_DIR = Path(os.environ["HMEQA_LIVE_DIR"]) if os.environ.get("HMEQA_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50
# Vanilla-baseline switch: strip every tuned mechanism so the agent sees
# exactly observe(RGB) + step + answer — the "bare" condition. look_around is
# withheld by not registering it, TURN_BUDGET=0 already disables the budget
# broadcast, so BARE only needs to drop the depth-derived clearance.
BARE = os.environ.get("HMEQA_BARE") == "1"

# HM-EQA's depth sensor is raw metric meters (habitat-sim DEPTH, no
# ObjectNav-style [0,1] normalization); cap the readout at 10 m = "open".
MAX_CLEARANCE_M = 10.0

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not advance the simulator or "
    "consume step budget."
    if BARE else
    "Look through the robot's forward-facing camera.\n\n"
    "Returns the current egocentric RGB view plus a clearance readout: "
    "distance in meters to the nearest obstacle in the left/center/right "
    "thirds of the view (10.0 = open, 10 m or more). Pure read — does not "
    "advance the simulator or consume step budget."
)
_TILT_DESC = (
    ", 4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 "
    "degrees (tilt changes the camera pitch only, not your position or "
    "heading; the camera starts tilted 30 degrees down)"
    if TILT else ""
)
_STEP_DESC = (
    "Execute a sequence of movement actions, in order.\n\n"
    "Actions: 1 = move forward 0.25 m, 2 = turn left 30 degrees, 3 = turn "
    f"right 30 degrees{_TILT_DESC}. There is NO stop action — the episode "
    "ends when you call answer().\n\n"
    "Executes sequentially and halts early if the movement budget runs out "
    "(you can still observe() and answer() after that). Returns how many "
    "actions ran, total steps taken, and remaining budget. The camera view "
    "changes after stepping — call observe() to see the result."
)
_ANSWER_DESC = (
    "Permanently END the episode by answering the multiple-choice question. "
    "Pass a single letter: \"A\", \"B\", \"C\" or \"D\". This is your one "
    "and only answer — call it once you have seen enough of the building to "
    "be confident. An episode that ends without answer() scores zero."
)

mcp = FastMCP("hmeqa-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_answer: str | None = None
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn.

    The driver's max_turns is the budget that actually kills sessions, but the
    model has no way to observe it. Report it in every step result and escalate
    to explicit orders near exhaustion; a session that ends without answer()
    scores exactly zero.
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
            f"CRITICAL — only {remaining} tool calls left before this session "
            "is killed. Call answer() with your best letter NOW. Ending "
            "without an answer scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is "
            "killed. Stop exploring; weigh the evidence you already have and "
            "call answer() before the budget runs out."
        )
    return fields


def _clearance_m(depth_field: dict[str, Any] | None) -> dict[str, float] | None:
    """Metric free-space readout from the raw depth frame.

    Distance (m) to the nearest obstacle in the left/center/right thirds of
    the view; 10th percentile per sector rides over stray pixels. HM-EQA's
    camera is tilted 30 degrees DOWN, so the near-horizon band is the TOP of
    the frame (rows 0-25%), not the eye-level band the level-camera bridges
    use. Depth is raw meters already — no normalization to undo.
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
    band = arr[: int(h * 0.25), :]
    sectors = {
        "left": band[:, : w // 3],
        "center": band[:, w // 3 : 2 * w // 3],
        "right": band[:, 2 * w // 3 :],
    }
    return {
        name: round(min(float(np.percentile(sector, 10)), MAX_CLEARANCE_M), 1)
        for name, sector in sectors.items()
    }


def _downscale(png: bytes, side: int) -> bytes:
    """Shrink a PNG so four panorama views fit one MCP message comfortably.
    side=0 disables the shrink (views stay at native render resolution)."""
    img = PILImage.open(BytesIO(png))
    if not side or max(img.size) <= side:
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
    outputs = _call("env_hmeqa__observe_egocentric", {})
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

    Costs 12 low-level turn steps from the step budget but only one tool
    call. Use this when deciding where to explore next instead of turning 30
    degrees at a time; your position and final heading are unchanged.
    """
    global _obs_count, _steps_taken, _tool_calls
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more steps possible"]
    if _steps_taken + 12 > STEP_BUDGET:
        return ["not enough movement budget left for look_around (needs 12 "
                f"turn steps, {max(0, STEP_BUDGET - _steps_taken)} remain); "
                "observe() and answer() still work"]

    content: list[Any] = []
    for label in ("ahead (0°)", "right (+90°)", "behind (+180°)", "left (+270°)"):
        outputs = _call("env_hmeqa__observe_egocentric", {})
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
        # the original heading (4 x 90° = 360°). Turns are 30°, so 90° = 3
        # turn steps.
        for _ in range(3):
            _call("env_hmeqa__step_discrete", {"action": 3})
            _steps_taken += 1

    status = {
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "heading_restored": True,
        **_budget_fields(),
    }
    content.append(json.dumps(status))
    _live_log({"look_around": True, **status})
    return content


# look_around is a tuned mechanism — register it only OUTSIDE the bare
# baseline. allowed_tools alone cannot withhold it: the driver runs with
# permission_mode="bypassPermissions", which lets the model call any REGISTERED
# tool regardless of the allowlist. Not registering it is the only reliable gate.
if not BARE:
    mcp.tool()(look_around)


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> dict[str, Any]:
    """Execute a sequence of movement actions, in order.

    Actions: 1 = move forward 0.25 m, 2 = turn left 30 degrees, 3 = turn
    right 30 degrees. There is NO stop action — the episode ends when you
    call answer().

    Executes sequentially and halts early if the movement budget runs out
    (you can still observe() and answer() after that). Returns how many
    actions ran, total steps taken, and remaining budget. The camera view
    changes after stepping — call observe() to see the result.
    """
    global _tool_calls
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more steps possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    valid = (1, 2, 3, 4, 5) if TILT else (1, 2, 3)
    bad = [a for a in actions if a not in valid]
    if bad:
        legend = ("1=FORWARD 2=LEFT 3=RIGHT 4=TILT_UP 5=TILT_DOWN" if TILT
                  else "1=FORWARD 2=LEFT 3=RIGHT")
        return {"error": f"invalid actions {bad}; valid: {legend} "
                         "(no STOP — end the episode with answer())"}
    return _execute_actions(actions)


def _execute_actions(actions: list[int]) -> dict[str, Any]:
    global _steps_taken
    executed = 0
    exhausted = False
    env_error: str | None = None
    tilt_deg: float | None = None
    for action in actions:
        if _steps_taken >= STEP_BUDGET:
            exhausted = True
            break
        outputs = _call("env_hmeqa__step_discrete", {"action": action})
        executed += 1
        _steps_taken += 1
        info = outputs.get("info") or {}
        if "tilt_deg" in info:
            tilt_deg = info["tilt_deg"]
        if outputs.get("terminated") or outputs.get("truncated"):
            env_error = str(info.get("error") or "env error")
            break

    result: dict[str, Any] = {
        "executed": executed,
        "requested": len(actions),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_budget_fields(),
    }
    if TILT and tilt_deg is not None:
        result["camera_tilt_deg"] = tilt_deg
    if exhausted or _steps_taken >= STEP_BUDGET:
        result["movement_budget_exhausted"] = True
        result["message"] = (
            "Movement budget exhausted — no more steps possible. You can "
            "still observe() from here; weigh the evidence you have and call "
            "answer() with your best letter."
        )
    if env_error:
        result["env_error"] = env_error
    if not _episode_over and not BARE and executed:
        # post-move clearance (pure read; no sim advance) — metric feedback
        # without an extra observe round-trip
        outputs = _call("env_hmeqa__observe_egocentric", {})
        clearance = _clearance_m(outputs.get("depth"))
        if clearance:
            result["clearance_m"] = clearance
    _live_log({"actions": actions, **result})
    return result


@mcp.tool(description=_ANSWER_DESC)
def answer(letter: str) -> dict[str, Any]:
    """Permanently END the episode by answering the multiple-choice question.

    Pass a single letter: "A", "B", "C" or "D". This is your one and only
    answer — call it once you have seen enough of the building to be
    confident. An episode that ends without answer() scores zero.
    """
    global _tool_calls, _episode_over, _end_reason, _answer
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); "
                         f"your answer was {_answer!r}"}
    clean = str(letter or "").strip().strip("\"'.()").upper()
    if clean not in ("A", "B", "C", "D"):
        return {"error": f"invalid answer {letter!r}; pass a single letter: "
                         "\"A\", \"B\", \"C\" or \"D\""}
    _episode_over = True
    _end_reason = "answer_submitted"
    _answer = clean
    # steps_taken_total makes this the EventSink's last parsed step result,
    # which is how the driver reads the answer back for evaluate().
    result = {
        "answer": clean,
        "answer_submitted": True,
        "episode_over": True,
        "end_reason": _end_reason,
        "steps_taken_total": _steps_taken,
        "message": f"Final answer {clean} recorded — the episode is over.",
    }
    _live_log(result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
