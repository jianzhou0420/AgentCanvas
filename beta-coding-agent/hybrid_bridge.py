"""Hybrid bridge — primitive actions AND the waypoint tool in ONE tool surface.

The Agent-selected Hybrid Interface condition: the model is handed BOTH the
low-level primitive controls (``step`` of 0-3 discrete actions) and the
waypoint selector (``goto`` a numbered candidate), AND two matching ways to
look — a forward egocentric camera (``observe``) and a numbered-waypoint
panorama (``observe_waypoints``). It decides for itself which lens to look
through and which control to move with, and may combine or switch them at any
point. No navigation workflow is injected — only the tool semantics (see
prompts.py HYBRID_SYSTEM_PROMPT).

Crucially this bridge runs in the classic SEPARATE (non-auto-observe) mode: a
move returns a text status only, and the agent must choose which observation to
take next — the forward view (to keep going with step) or the panorama (to pick
a waypoint for goto). That choice IS the interface choice, so it must be the
agent's, not something coupled automatically into every move result. This also
makes each path's observation byte-identical to its standalone R2R condition
(bare's observe = forward RGB; wp's observe = numbered panorama).

Deliberately a THIRD, self-contained stdio MCP server next to mcp_bridge.py
(bare/nav primitives) and wp_bridge.py (waypoints), which are left untouched so
the two standalone conditions stay frozen and byte-comparable. The pure,
stateless helpers (panorama annotation, waypoint normalization, geometry) are
imported from wp_bridge; all EPISODE STATE (step count, episode-over, the live
waypoint numbering) lives here in ONE shared set of globals so primitive steps
and waypoint moves advance the same episode coherently.

Tools the agent sees:
- ``observe()``            -> forward egocentric RGB. The lens for precise
                             primitive control. Does NOT arm goto().
- ``observe_waypoints()``  -> 4-view panorama [Left|Front|Right|Back] with
                             numbered green waypoint circles + a JSON of each
                             waypoint's direction/angle/distance. The lens for
                             choosing a waypoint; ARMS goto() with these numbers.
- ``step(actions)``        -> primitive discrete actions in order (0=STOP,
                             1=forward 0.25 m, 2=left 15 deg, 3=right 15 deg).
                             The view changes; observe() again to see it, and
                             the waypoint numbers go stale.
- ``goto(waypoint)``       -> walk to one numbered waypoint from the LATEST
                             observe_waypoints(); after arriving call
                             observe_waypoints() again for fresh numbers.
- ``stop()``               -> discrete action 0; permanently ends the episode.

The binding budget is the SAME 500 low-level step budget as the standalone
conditions (habitat truncates at MAX_EPISODE_STEPS regardless); there is NO
separate waypoint-move cap — a goto simply spends more of the 500 than a
primitive step, and the agent trades them freely.

Env vars (union of mcp_bridge.py's and wp_bridge.py's):
    HABITAT_SERVER_URL     habitat auto_host (default http://127.0.0.1:9200)
    HABITAT_WP_SERVER_URL  waypoint-predictor auto_host (default :9210)
    HABITAT_WP_PREDICT_FN  predict fn (default smartway_waypoint__predict)
    HABITAT_STEP_BUDGET    low-level step budget echoed to the agent (default 500)
    HABITAT_TURN_BUDGET    driver max_turns broadcast (0 = off, as in bare/wp)
    HABITAT_LIVE_DIR       live spectating dir (annotated frames + actions.log)
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image

# Ensure this file's own directory is importable so ``import wp_bridge`` resolves
# both when run as a script (SDK spawns `python hybrid_bridge.py`) and when the
# driver introspects the tool schemas via importlib.exec_module (which does not
# seed sys.path with the script dir).
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pure, stateless helpers reused from the frozen waypoint bridge (no episode
# state crosses over — importing only executes its module body, whose FastMCP
# instance and tools are never run here).
from wp_bridge import (  # noqa: E402
    N_PANO_VIEWS,
    _action_options,
    _annotate_strip,
    _direction_of,
    _norm_pi,
    _normalize_candidates,
)

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
WP_SERVER_URL = os.environ.get("HABITAT_WP_SERVER_URL", "http://127.0.0.1:9210")
PREDICT_FN = os.environ.get("HABITAT_WP_PREDICT_FN", "smartway_waypoint__predict")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("HABITAT_TURN_BUDGET", "0"))
LIVE_DIR = Path(os.environ["HABITAT_LIVE_DIR"]) if os.environ.get("HABITAT_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view — the lens for precise, low-level control. Pure read: "
    "does not advance the simulator or consume step budget. COMMITMENT: after "
    "this look your next action must be step() (or stop()); you cannot goto(), "
    "and you cannot look again without moving first."
)
_OBSERVE_WP_DESC = (
    "Scan a panorama and get your waypoint options. Returns a panoramic image "
    "of four views labeled Left / Front / Right / Back with numbered green "
    "circles marking the waypoints you can jump to, plus a JSON listing each "
    "waypoint's direction, angle (degrees left of heading; negative = right) "
    "and distance in meters. Pure read: does not advance the simulator or "
    "consume step budget. COMMITMENT: after this look your next action must be "
    "goto() (or stop()); you cannot step(), and you cannot look again without "
    "moving first."
)
_STEP_DESC = (
    "PRIMITIVE control: execute a sequence of low-level movement actions, in "
    "order. Actions: 0 = STOP (permanently ENDS the episode), 1 = move forward "
    "0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees. The lens for "
    "fine, exact positioning and tight turns. REQUIRES an immediately preceding "
    "observe() (the forward camera); it is not available right after "
    "observe_waypoints(). Returns how many actions ran, total steps taken, "
    "remaining budget, and whether the episode is over. Look again before your "
    "next move."
)
_GOTO_DESC = (
    "WAYPOINT control: walk to one numbered waypoint from the LATEST "
    "observe_waypoints() — the robot turns toward it and walks there in one "
    "call. The lens for covering distance quickly toward a visible landmark. "
    "REQUIRES an immediately preceding observe_waypoints() (it is not available "
    "right after observe()). Returns steps consumed, remaining budget, and "
    "whether the episode is over. Look again before your next move."
)
_STOP_DESC = (
    "Permanently END the episode, declaring you have reached the goal. Issue it "
    "only when you believe the robot is within 3 meters of the instruction's "
    "endpoint — stopping is irreversible. (Equivalent to step([0]).)"
)

mcp = FastMCP("habitat-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_prim_moves = 0          # primitive step() calls that ran (logging/analysis only)
_wp_moves = 0            # goto() calls that ran (logging/analysis only)
_episode_over = False
_end_reason: str | None = None
_last_candidates: list[dict[str, float]] | None = None  # armed only by observe_waypoints()
# Interface-commitment state machine. Looking picks a lens and COMMITS the next
# move to it; the move then clears the commitment. Enforces, structurally:
#   observe()            -> the only move allowed next is step()
#   observe_waypoints()  -> the only move allowed next is goto()
#   no two looks in a row -> you must MOVE between observations
# _pending is None right after a move (look required before the next move),
# "forward" after observe(), "waypoints" after observe_waypoints().
_pending: str | None = None
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — off (empty) in hybrid, since bare-style cells set
    HABITAT_TURN_BUDGET=0. Kept for parity with the two standalone bridges."""
    if TURN_BUDGET <= 0:
        return {}
    remaining = max(0, TURN_BUDGET - _tool_calls)
    fields: dict[str, Any] = {"tool_calls_used": _tool_calls, "tool_calls_remaining": remaining}
    if remaining <= 15:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls left before this session is killed. "
            "Commit to your best goal candidate and stop() NOW — ending without "
            "STOP scores ZERO."
        )
    return fields


def _step_fields() -> dict[str, Any]:
    return {
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
    }


def _call(function_name: str, inputs: dict[str, Any],
          config: dict[str, Any] | None = None, base: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(f"{base or SERVER_URL}/call/{function_name}", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["outputs"]


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


# ── observations ──


def _render_panorama() -> tuple[bytes, dict[str, Any]]:
    """Render the RGB-D panorama, predict waypoints, annotate the numbered strip,
    and ARM the shared ``_last_candidates``. Pure read — advances the frame
    counter, never the simulator or step budget."""
    global _obs_count, _last_candidates
    pano = _call(
        "env_habitat__observe_panorama", {"trigger": "hybrid"},
        config={"representation": "views_rgbd", "n_views": N_PANO_VIEWS},
    )
    views = pano.get("views") or []
    slim = [
        {"dir_id": v.get("dir_id"), "rgb_base64": v.get("rgb_base64"),
         "depth_base64": v.get("depth_base64")}
        for v in views
    ]
    pred = _call(PREDICT_FN, {"views": slim}, base=WP_SERVER_URL)
    _last_candidates = _normalize_candidates(pred.get("candidates"))
    png = _annotate_strip(views, _last_candidates)
    _obs_count += 1
    _live_frame(png)
    status: dict[str, Any] = {
        "waypoints": {
            str(i + 1): {
                "direction": _direction_of(c["angle"]),
                "angle_deg": round(math.degrees(_norm_pi(c["angle"])), 1),
                "distance_m": round(c["distance"], 2),
            }
            for i, c in enumerate(_last_candidates)
        },
        "action_options": _action_options(_last_candidates),
        **_step_fields(),
        **_budget_fields(),
    }
    if not _last_candidates:
        status["note"] = (
            "no reachable waypoints predicted here — use step() to reposition, "
            "or stop() if you are at the goal"
        )
    _live_log({"observe_waypoints": True, "num_waypoints": len(_last_candidates)})
    return png, status


def _capture_egocentric() -> bytes:
    """Render the current forward egocentric RGB. Pure read."""
    global _obs_count
    outputs = _call("env_habitat__observe_egocentric", {})
    png = base64.b64decode(outputs["rgb"])
    _obs_count += 1
    _live_frame(png)
    return png


# ── tools ──


def _already_looked_msg() -> str:
    nxt = "step()" if _pending == "forward" else "goto()"
    lens = "observe()" if _pending == "forward" else "observe_waypoints()"
    return (f"you already looked here (via {lens}) — you cannot take two looks in a row. "
            f"Make your move first ({nxt}, or stop()), then look again.")


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    """Forward egocentric camera (the primitive lens) — commits the next move to step()."""
    global _tool_calls, _pending
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more moves possible"]
    if _pending is not None:
        return [_already_looked_msg()]
    png = _capture_egocentric()
    _pending = "forward"
    _live_log({"observe": True})
    return [Image(data=png, format="png")]


@mcp.tool(description=_OBSERVE_WP_DESC)
def observe_waypoints() -> list:
    """Numbered-waypoint panorama (the waypoint lens) — arms and commits the next move to goto()."""
    global _tool_calls, _pending
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more moves possible"]
    if _pending is not None:
        return [_already_looked_msg()]
    png, status = _render_panorama()
    _pending = "waypoints"
    return [Image(data=png, format="png"), json.dumps(status)]


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason, _tool_calls, _last_candidates, _prim_moves, _pending
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more moves possible"}
    if _pending != "forward":
        if _pending == "waypoints":
            return {"error": "you looked at the waypoint panorama — its matching move is goto(). "
                             "For a primitive step(), observe() the forward camera first."}
        return {"error": "look before you move — call observe() (the forward camera) to commit to "
                         "a primitive step()."}
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
            _end_reason = ("stop_called" if action == 0
                           else "step_budget_exhausted" if truncated else "terminated")
            break
    _prim_moves += 1
    _pending = None          # move committed -> a look is required before the next move
    _last_candidates = None  # position changed -> panorama numbers are stale

    result: dict[str, Any] = {
        "interface": "primitive",
        "executed": executed,
        "requested": len(actions),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_step_fields(),
        **_budget_fields(),
    }
    if not _episode_over:
        result["note"] = (
            "moved — look again before your next move: observe() (forward camera) to "
            "step again, or observe_waypoints() to switch to a goto"
        )
    _live_log({"step": actions, "steps_taken_total": _steps_taken, "episode_over": _episode_over})
    return result


@mcp.tool(description=_GOTO_DESC)
def goto(waypoint: int) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason, _tool_calls, _last_candidates, _wp_moves, _pending
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more moves possible"}
    if _pending != "waypoints":
        if _pending == "forward":
            return {"error": "you looked at the forward camera — its matching move is step(). "
                             "To goto a waypoint, observe_waypoints() first."}
        return {"error": "look before you move — call observe_waypoints() to see the numbered "
                         "waypoints and commit to a goto()."}
    if _last_candidates is None:  # defensive; _pending=='waypoints' implies armed
        return {"error": "no armed waypoints — call observe_waypoints() first"}
    try:
        waypoint = int(waypoint)
    except (TypeError, ValueError):
        return {"error": f"invalid waypoint {waypoint!r}; expected an integer"}
    if not 1 <= waypoint <= len(_last_candidates):
        return {"error": (f"invalid waypoint {waypoint}; valid choices are 1-"
                          f"{len(_last_candidates)} from the LATEST observe_waypoints()")}

    cand = _last_candidates[waypoint - 1]
    outputs = _call("env_habitat__step_hightolow", {"angle": cand["angle"], "distance": cand["distance"]})
    info = outputs.get("info") or {}
    if isinstance(info.get("step_count"), (int, float)):
        _steps_taken = int(info["step_count"])
    _wp_moves += 1
    _pending = None          # move committed -> a look is required before the next move
    _last_candidates = None  # arrived somewhere new -> re-scan before next goto
    terminated = bool(outputs.get("terminated"))
    truncated = bool(outputs.get("truncated"))
    if terminated or truncated:
        _episode_over = True
        _end_reason = "step_budget_exhausted" if truncated else "terminated"

    result: dict[str, Any] = {
        "interface": "waypoint",
        "moved_to": waypoint,
        "direction": _direction_of(cand["angle"]),
        "distance_m": round(cand["distance"], 2),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_step_fields(),
        **_budget_fields(),
    }
    if not _episode_over:
        result["note"] = (
            "arrived — look again before your next move: observe_waypoints() to goto "
            "again, or observe() (forward camera) to switch to a primitive step"
        )
    _live_log({"goto": waypoint, "distance_m": result["distance_m"],
               "steps_taken_total": _steps_taken, "episode_over": _episode_over})
    return result


@mcp.tool(description=_STOP_DESC)
def stop() -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason, _tool_calls
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more moves possible"}
    _call("env_habitat__step_discrete", {"action": 0})
    _steps_taken += 1
    _episode_over = True
    _end_reason = "stop_called"
    result = {"interface": "stop", "stopped": True, "episode_over": True,
              "end_reason": _end_reason, **_step_fields()}
    _live_log(result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
