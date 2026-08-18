"""slam_r2r_02 bridge — frozen per-experiment copy (exp_workspace rule).

Arm: bare surface + get_pose/get_map/get_trajectory (map v2: no frontier,
floor-layered, fixed snapped axes) + slam_note on step.

Promoted 2026-08-17 from the probe bridge (slam_vlnce_bridge.py, retired):
same agent-facing surface — the bare observe/step of mcp_bridge.py (identical
action semantics and result fields) over SlamEnv's coarse-to-micro world,
with the three read-only SLAM query tools (get_pose / get_map /
get_trajectory) ALWAYS registered (registration is the gate; this folder IS
the map-v2 arm — no runtime switch).

Changes against the probe bridge (the integration, not the surface):
- The env peer is an env_slam_vlnce auto_host over HTTP (SLAMB_SERVER_URL),
  not a spawned cross-worktree JSON-lines subprocess — fleet lifecycle,
  /health name preflight and worker isolation come from the std machinery.
- STOP is forwarded to the env (env_slam_vlnce__step_discrete action 0), so
  stop state lives env-side and the driver scores via env_slam_vlnce__evaluate
  (full NE/SR/SPL/OSR/nDTW suite) — the probe's per-step report.json hot-write
  and runner-side stop-gated-geodesic rule are gone.
- The get_map JSON no longer carries the probe's vestigial "png": null field.

Env vars (own SLAMB_ prefix, per bridge_env()'s no-mixing rule):
  SLAMB_SERVER_URL     env auto_host base URL (this experiment's nodeset)
  SLAMB_STEP_BUDGET    coarse-action budget (default 500; env truncates too)
  SLAMB_LIVE_DIR       live-spectating dir (obs_* frames + actions.log)
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

SERVER_URL = os.environ.get("SLAMB_SERVER_URL", "http://127.0.0.1:9260")
STEP_BUDGET = int(os.environ.get("SLAMB_STEP_BUDGET", "500"))
LIVE_DIR = Path(os.environ["SLAMB_LIVE_DIR"]) if os.environ.get("SLAMB_LIVE_DIR") else None
VERB = "env_slam_vlnce"
MAX_ACTIONS_PER_CALL = 50

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not advance the simulator or "
    "consume step budget."
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
)

mcp = FastMCP("habitat-env")

_steps_taken = 0
_obs_count = 0
_episode_over = False
_end_reason: str | None = None
_collisions = 0
_last_slam = "ok"
_last_reanchors = 0
_t0 = time.time()


def _call(function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/call/{function_name}", json={"inputs": inputs}, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["outputs"]


def _live_frame(png: bytes) -> None:
    """Same live-spectating contract as mcp_bridge: obs_* frames + latest.png."""
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


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    global _obs_count
    png = base64.b64decode(_call(f"{VERB}__observe_egocentric", {})["rgb"])
    _obs_count += 1
    _live_frame(png)
    return [Image(data=png, format="png")]


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> list:  # bare `list` => FastMCP unstructured
    # path; a plain dict return still serializes (mcp_bridge.py's own note).
    global _steps_taken, _episode_over, _end_reason, _collisions
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more steps possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    bad = [a for a in actions if a not in (0, 1, 2, 3)]
    if bad:
        return {"error": f"invalid actions {bad}; valid: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT"}

    global _last_slam, _last_reanchors
    executed = 0
    collided = False
    slam_note: str | None = None
    for action in actions:
        outputs = _call(f"{VERB}__step_discrete", {"action": action})
        executed += 1
        info = outputs.get("info") or {}
        if action != 0:
            _steps_taken += 1
            if info.get("collision"):
                collided = True
                _collisions += 1
        # SLAM health transitions (instrumented arm only — bare has no
        # instruments to go stale, and its six-field surface stays frozen)
        slam_now = str(info.get("slam") or "")
        re_now = int(info.get("reanchors") or 0)
        if slam_now:
            if slam_now == "slam_lost" and _last_slam == "ok":
                slam_note = ("SLAM tracking lost — get_pose/get_map estimates "
                             "are stale until it recovers.")
            elif slam_now == "ok" and _last_slam == "slam_lost":
                slam_note = ("SLAM tracking recovered"
                             + (" and re-anchored — the map reference frame "
                                "may have shifted slightly."
                                if re_now > _last_reanchors else "."))
            _last_slam = slam_now
            _last_reanchors = max(_last_reanchors, re_now)
        if outputs.get("terminated"):
            _episode_over = True
            _end_reason = "stop_called"
            break
        if outputs.get("truncated"):
            _episode_over = True
            _end_reason = "step_budget_exhausted"
            break

    # Result fields mirror mcp_bridge's bare surface exactly (six fields, no
    # extras): collisions stay in actions.log, not on the surface.
    result = {
        "executed": executed,
        "requested": len(actions),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
    }
    if slam_note:
        result["slam_note"] = slam_note
    _live_log({"actions": actions, "collision": collided, **result})
    return result


# ── instrument tools: registered ONLY in the instrumented arm ──

def get_pose() -> list:
    """Read the robot's SLAM-estimated pose: position (x, z) in meters and
    heading yaw_deg, in a fixed frame anchored at your start pose (x = right
    and z = forward OF YOUR STARTING POSE; yaw_deg is 0 at your starting
    heading and INCREASES when you turn right). The same x/z coordinates
    label the gridlines on get_map(), so pose numbers place you on the map.
    Free — no step cost. slam="slam_lost" means the estimate is currently
    stale."""
    r = _call(f"{VERB}__get_pose", {})["pose"]
    _live_log({"get_pose": r})
    return [json.dumps(r)]


def get_map() -> list:
    """Read the occupancy map built automatically as you move
(it already contains an initial 360-degree scan of your start location).
Returns a top-down image cropped to what you have explored — UP on the image
is your STARTING heading (+z), right is +x. On it: white = walked-free
space, black = obstacle, gray = unexplored; the blue ARROW is you, pointing
your current heading; the blue LINE is your path so far; the "S" circle is
your start. Gridlines every 2 m are labeled on both axes with the same x/z
coordinates get_pose() reports (the 0-axes are drawn heavier), and the crop
window never pans or rescales between calls — it only grows in 2 m steps as
you explore, so coordinates stay put from map to map. The map shows YOUR
CURRENT FLOOR only; a "floor k/n" tag (top right) says which storey you are
on and how many have been seen — changing floors switches the map to that
floor's layer. Free — no step cost. SLAM estimates, not ground truth."""
    global _obs_count
    outputs = _call(f"{VERB}__get_map", {})
    png = base64.b64decode(outputs["map"])
    payload = dict(outputs.get("info") or {})
    # The frames viewer pairs the Nth image-returning tool call with the Nth
    # obs_* file in sorted order, so map frames must advance the SAME counter
    # as camera frames (sorted filenames == emission order). obs_ prefix gets
    # it listed; _map suffix keeps it distinguishable.
    _obs_count += 1
    if LIVE_DIR is not None:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        (LIVE_DIR / f"obs_{_obs_count:04d}_map.png").write_bytes(png)
    _live_log({"get_map": {"frontiers": len(payload.get("frontiers", []))}})
    return [Image(data=png, format="png"), json.dumps(payload)]


def get_trajectory() -> list:
    """Read your own path so far as (x, z) points in the same fixed frame as
    get_pose(), oldest first. Useful to check whether you are circling or
    which areas you already covered. Free — no step cost."""
    r = _call(f"{VERB}__get_trajectory", {})["trajectory"]
    _live_log({"get_trajectory": {"n_total": r.get("n_total")}})
    return [json.dumps(r)]

mcp.tool()(get_pose)
mcp.tool()(get_map)
mcp.tool()(get_trajectory)


if __name__ == "__main__":
    mcp.run(transport="stdio")
