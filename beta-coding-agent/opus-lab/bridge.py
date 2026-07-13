"""Minimal MCP bridge — the bare habitat toolset with the standard VLN
egocentric observation (RGB-D + proprioception).

Stdio MCP server (FastMCP) forwarding a small toolset to a running
``env_habitat`` auto_host subprocess (ADR-server-001 HTTP surface,
``POST /call/{fn}``):

- ``observe()``      -> the full egocentric observation: RGB, a depth image and
                        per-region metric depth, world pose (position + heading),
                        distance walked / displacement, camera intrinsics.
- ``look_around()``  -> eight labeled RGB views + compact per-view depth, one call
- ``step(actions)``  -> discrete actions 0-3, sanitized gym flags only

The observation set is exactly what a standard VLN-CE agent receives — RGB-D,
pose/odometry and intrinsics — and nothing that would be cheating: the agent
never sees SR/SPL, reward, the shortest-path sensor, or the oracle progress
sensor (those live in ``raw_obs``, which is deliberately NOT forwarded). Depth is
habitat's normalized [0,1] map rescaled to metres via MAX_DEPTH (habitat 0.1.7
default 10 m; vlnce_task.yaml sets only the depth W/H so the default applies).

This is the *clean-room* bridge for the opus skill hill-climb. Episode selection
and metric collection stay driver-side (``driver.py``). One bridge process serves
one agent session = one episode, so per-episode accounting lives in module
globals.

Env vars:
    HABITAT_SERVER_URL   auto_host base URL (default http://127.0.0.1:9200)
    HABITAT_STEP_BUDGET  advisory low-level step budget echoed to the agent
                         (default 500 — habitat's MAX_EPISODE_STEPS truncates
                         authoritatively regardless)
    HABITAT_LIVE_DIR     optional dir for live spectating: every observe RGB frame
                         lands as obs_NNNN.png (+ overwritten latest.png),
                         every step call appends a line to actions.log
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
LIVE_DIR = Path(os.environ["HABITAT_LIVE_DIR"]) if os.environ.get("HABITAT_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 50
# Habitat DEPTH_SENSOR normalizes to [0,1]; vlnce_task.yaml overrides only W/H, so
# habitat 0.1.7 defaults MIN_DEPTH=0, MAX_DEPTH=10, NORMALIZE_DEPTH=True hold.
_MAX_DEPTH_M = 10.0
_FORWARD_M = 0.25  # metres per forward (action 1) step

_OBSERVE_DESC = (
    "Look through the robot's forward-facing camera. Returns the standard VLN "
    "egocentric observation: (1) the RGB view; (2) a depth image (nearer = "
    "brighter); (3) a text block with per-region metric depth (a 3x3 grid in "
    "METRES — use it to tell how FAR the things you see are: a landmark that "
    "looks 'right there' is often several metres ahead, and you are only BETWEEN "
    "two objects once their depths are small and on opposite sides, not while "
    "they are still far ahead), your world pose (position x,y,z and heading in "
    "degrees), how far you have walked and your straight-line displacement from "
    "the start, and camera intrinsics. Pure read — does not advance the simulator "
    "or consume step budget."
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
_forward_steps = 0
_obs_count = 0
_episode_over = False
_end_reason: str | None = None
_t0 = time.time()
_start_pos: list[float] | None = None


def _live_frame(png: bytes, *, suffix: str = "", latest: bool = True) -> None:
    """Write one spectator frame. ``suffix`` distinguishes the paired depth frame
    (``_depth``) from its RGB frame under the SAME obs index — the backend sorts
    ``obs_*.png`` lexically, and '.' < '_' keeps the RGB tile before its depth
    tile and both before the next obs. ``latest`` only the RGB updates the live
    ``latest.png`` thumbnail."""
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"obs_{_obs_count:04d}_step{_steps_taken:03d}{suffix}.png"
    (LIVE_DIR / fname).write_bytes(png)
    if latest:
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


# ── egocentric perception decode (RGB-D + pose + intrinsics) ──


def _decode_depth_m(field: Any) -> "np.ndarray | None":
    """Env serializes depth as {__ndarray__: b64, dtype, shape}, normalized to
    [0,1]. Rescale to metres via MAX_DEPTH. None if depth is absent/unparseable."""
    if not isinstance(field, dict) or "__ndarray__" not in field:
        return None
    try:
        arr = np.frombuffer(
            base64.b64decode(field["__ndarray__"]), dtype=field.get("dtype", "float32")
        ).reshape(field["shape"])
        return arr.astype(np.float32) * _MAX_DEPTH_M
    except Exception:  # noqa: BLE001 — perception decode must never break a run
        return None


def _depth_png(depth_m: "np.ndarray") -> bytes:
    """Grayscale depth viz: nearer = brighter (255), farther = darker (0)."""
    norm = np.clip(depth_m / _MAX_DEPTH_M, 0.0, 1.0)
    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(gray, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _depth_grid(depth_m: "np.ndarray") -> list[list[float]]:
    """3x3 per-cell median depth in metres (rows top->bottom, cols left->right)."""
    h, w = depth_m.shape
    rb = [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
    cb = [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]
    return [
        [round(float(np.median(depth_m[r0:r1, c0:c1])), 1) for (c0, c1) in cb]
        for (r0, r1) in rb
    ]


def _nearest_ahead(depth_m: "np.ndarray") -> float:
    """Closest real obstacle in the central forward region, in metres. Uses the
    5th percentile over the middle cell so sparse depth==0 holes (habitat's
    'no return' marker) don't read as a 0 m collision."""
    h, w = depth_m.shape
    central = depth_m[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    valid = central[central > 0.1]
    if valid.size == 0:
        return float(_MAX_DEPTH_M)
    return round(float(np.percentile(valid, 5)), 1)


def _heading_deg(quat_xyzw: list[float]) -> float:
    """World heading (deg) from the agent's [x,y,z,w] quaternion. 0° ≈ facing -z,
    + turns toward +x. Derived from the rotated local-forward vector [0,0,-1]."""
    x, y, z, w = quat_xyzw
    fx = -2.0 * (x * z + w * y)
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return round(math.degrees(math.atan2(fx, -fz)), 1)


def _egocentric() -> dict:
    """One observe_egocentric call -> decoded RGB-D + pose + intrinsics. Captures
    the episode start position on first call (for displacement)."""
    global _start_pos
    out = _call("env_habitat__observe_egocentric", {})
    rgb_png = base64.b64decode(out["rgb"])
    depth_m = _decode_depth_m(out.get("depth"))
    pose = out.get("pose") or {}
    pos = list(pose.get("position") or [0.0, 0.0, 0.0])
    if _start_pos is None:
        _start_pos = list(pos)
    return {
        "rgb_png": rgb_png,
        "depth_m": depth_m,
        "pose": pose,
        "position": pos,
        "intrinsics": out.get("intrinsics"),
    }


def _spatial_text(ego: dict) -> str:
    pos = ego["position"]
    quat = (ego["pose"] or {}).get("orientation") or [0.0, 0.0, 0.0, 1.0]
    sx, sz = (_start_pos[0], _start_pos[2]) if _start_pos else (pos[0], pos[2])
    disp = math.hypot(pos[0] - sx, pos[2] - sz)
    lines = [
        f"pose: position (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}) | "
        f"heading {_heading_deg(quat)}°",
        f"odometry: walked {_forward_steps * _FORWARD_M:.1f} m forward so far "
        f"({_forward_steps} forward steps) | {disp:.1f} m straight-line from start",
    ]
    if ego["depth_m"] is not None:
        g = _depth_grid(ego["depth_m"])
        lines.append(
            "depth (metres, median per cell; ~10 = far / no return):\n"
            f"  top:    L={g[0][0]:<5} C={g[0][1]:<5} R={g[0][2]}\n"
            f"  middle: L={g[1][0]:<5} C={g[1][1]:<5} R={g[1][2]}   <- eye level\n"
            f"  bottom: L={g[2][0]:<5} C={g[2][1]:<5} R={g[2][2]}\n"
            f"  nearest obstacle straight ahead: {_nearest_ahead(ego['depth_m']):.1f} m"
        )
    intr = ego.get("intrinsics")
    if intr:
        lines.append(
            f"camera: {int(intr['width'])}x{int(intr['height'])} px, "
            f"fx={intr['fx']:.0f} fy={intr['fy']:.0f}, HFOV 90°"
        )
    return "\n".join(lines)


def _depth_line(depth_m: "np.ndarray | None") -> str:
    """Compact one-line depth for a look_around view: mid-row L/C/R + nearest ahead."""
    if depth_m is None:
        return ""
    g = _depth_grid(depth_m)
    return f"  [depth m — mid L/C/R {g[1][0]}/{g[1][1]}/{g[1][2]}; nearest ahead {_nearest_ahead(depth_m):.1f}]"


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    global _obs_count
    ego = _egocentric()
    _obs_count += 1
    _live_frame(ego["rgb_png"])
    content: list = [Image(data=ego["rgb_png"], format="png")]
    if ego["depth_m"] is not None:
        depth_png = _depth_png(ego["depth_m"])
        # second spectator frame so the Monitor shows depth inline and the frame
        # count matches the two image blocks (else the depth tile reads "frame
        # pending" and the cursor desyncs for every later observe).
        _live_frame(depth_png, suffix="_depth", latest=False)
        content.append(Image(data=depth_png, format="png"))
    content.append(_spatial_text(ego))
    return content


_LOOK_LABELS = [
    "ahead (0°)",
    "ahead-right (45°)",
    "right (90°)",
    "behind-right (135°)",
    "behind (180°)",
    "behind-left (225°)",
    "left (270°)",
    "ahead-left (315°)",
]
_LOOK_DESC = (
    "Scan your full surroundings in ONE call. Returns EIGHT egocentric RGB "
    "views at 45° increments, each labeled by its rightward turn from your "
    "current heading and annotated with a compact metric-depth readout (mid-row "
    "left/center/right depth and nearest obstacle ahead, in metres): ahead (0°), "
    "ahead-right (45°), right (90°), behind-right (135°), behind (180°), "
    "behind-left (225°), left (270°), ahead-left (315°). Each view spans a 90° "
    "field, so consecutive views overlap by 45° — nothing between them is missed. "
    "Rotates a full 360° and restores your original heading EXACTLY: your position "
    "and facing are unchanged afterward. Ends with your current pose. Costs 24 "
    "low-level turn steps (3 per view) but only one tool call. Use it at junctions "
    "and decision points instead of turning a little at a time."
)


@mcp.tool(description=_LOOK_DESC)
def look_around() -> list:
    """Eight labeled RGB+depth views 45° apart; full 360° rotation, heading restored."""
    global _obs_count, _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more steps possible"]

    content: list = []
    for label in _LOOK_LABELS:
        ego = _egocentric()
        _obs_count += 1
        _live_frame(ego["rgb_png"])
        content.extend(
            [label + _depth_line(ego["depth_m"]), Image(data=ego["rgb_png"], format="png")]
        )
        # rotate 45° right (3 × 15°) toward the next view; after the 8th view
        # the accumulated 8 × 45° = 360° restores the original heading.
        for _ in range(3):
            outputs = _call("env_habitat__step_discrete", {"action": 3})
            _steps_taken += 1
            if outputs.get("terminated") or outputs.get("truncated"):
                _episode_over = True
                _end_reason = "step_budget_exhausted"
                break
        if _episode_over:
            break

    status = {
        "views": sum(1 for c in content if not isinstance(c, str)),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "heading_restored": not _episode_over,
    }
    if not _episode_over:
        content.append(_spatial_text(_egocentric()))  # pose after heading restored
    content.append(json.dumps(status))
    _live_log({"look_around": True, **status})
    return content


@mcp.tool(description=_STEP_DESC)
def step(actions: list[int]) -> dict[str, Any]:
    global _steps_taken, _forward_steps, _episode_over, _end_reason
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
        if action == 1:
            _forward_steps += 1
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
        "forward_steps_total": _forward_steps,
        "distance_walked_m": round(_forward_steps * _FORWARD_M, 2),
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "end_reason": _end_reason,
    }
    _live_log({"actions": actions, **result})
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
