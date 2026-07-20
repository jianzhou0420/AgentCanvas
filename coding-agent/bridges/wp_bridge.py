"""WP bridge — waypoint-selection action space for an external coding agent.

Stdio MCP server (FastMCP) exposing the third action-space condition next to
bare/nav (both live in ``mcp_bridge.py``; this file is deliberately separate
so each action space is frozen and audited on its own). Ported from the
VLN-MME waypoint pipeline (``vlnce_baselines/common/base_il_trainer.py``):
a depth-based predictor proposes candidate waypoints, they are drawn as
numbered circles on a four-view panorama, and the model picks one by number.

The agent sees exactly three tools:

- ``observe()``       -> render a 12-view RGB-D panorama, send it to the
                         waypoint-predictor auto_host, draw the <=5 candidates
                         on a [Left|Front|Right|Back] strip (the VLN-MME
                         ``display_observation`` convention), return the
                         annotated image + a JSON of numbered options
- ``goto(waypoint)``  -> execute one candidate via env_habitat__step_hightolow
                         (rotate to its angle, walk its distance)
- ``stop()``          -> discrete action 0; permanently ends the episode

Two servers, both ADR-server-001 auto_hosts (``POST /call/{fn}``):
the habitat env (same one mcp_bridge talks to) and the waypoint predictor
(smartway_waypoint or opennav_waypoint nodeset — both emit counter-clockwise
radians ``2π − idx/120·2π`` and distances ``(idx+1)·0.25 m``, the exact
convention ``step_hightolow`` was verified against).

One bridge process serves one agent session = one episode, so per-episode
accounting lives in module globals (same contract as mcp_bridge.py).

Env vars (superset of mcp_bridge.py's):
    HABITAT_SERVER_URL     habitat auto_host (default http://127.0.0.1:9200)
    HABITAT_WP_SERVER_URL  waypoint-predictor auto_host (default :9210)
    HABITAT_WP_PREDICT_FN  predict function on that server (default
                           smartway_waypoint__predict; opennav_waypoint__predict
                           output is normalized transparently)
    HABITAT_STEP_BUDGET    advisory low-level step budget echoed to the agent
    HABITAT_TURN_BUDGET    driver max_turns broadcast (0 = off, as in bare)
    HABITAT_LIVE_DIR       live spectating dir (annotated frames + actions.log)
    HABITAT_WP_VIEW_PX     per-tile side of the returned strip (default 384)
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

SERVER_URL = os.environ.get("HABITAT_SERVER_URL", "http://127.0.0.1:9200")
WP_SERVER_URL = os.environ.get("HABITAT_WP_SERVER_URL", "http://127.0.0.1:9210")
PREDICT_FN = os.environ.get("HABITAT_WP_PREDICT_FN", "smartway_waypoint__predict")
STEP_BUDGET = int(os.environ.get("HABITAT_STEP_BUDGET", "500"))
TURN_BUDGET = int(os.environ.get("HABITAT_TURN_BUDGET", "0"))
# Waypoint-move budget: the decision-step cap (VLN-MME's ``max_step`` — one
# goto = one decision), NOT the low-level MOVE_FORWARD count. The episode
# truncates when it is exhausted, so a wandering agent is stopped without
# waiting for the 500-primitive budget.
WP_MAX_MOVES = int(os.environ.get("HABITAT_WP_MAX_MOVES", "30"))
WP_VIEW_PX = int(os.environ.get("HABITAT_WP_VIEW_PX", "384"))
LIVE_DIR = Path(os.environ["HABITAT_LIVE_DIR"]) if os.environ.get("HABITAT_LIVE_DIR") else None

N_PANO_VIEWS = 12
# dir_id -> strip slot: render_panorama_rgbd's dir i faces i*30 deg counter-
# clockwise, so 3 = Left, 0 = Front, 9 = Right, 6 = Back — the VLN-MME
# [Left|Front|Right|Back] display order at zero extra render cost.
STRIP_DIRS = ((3, "Left"), (0, "Front"), (9, "Right"), (6, "Back"))

_OBSERVE_DESC = (
    "Look around from where you stand. Returns a panoramic image of four "
    "views labeled Left / Front / Right / Back with numbered green circles "
    "marking the waypoints you can move to, plus a JSON listing each "
    "waypoint's direction, angle (degrees left of your heading; negative = "
    "right) and distance in meters. Pure read — does not advance the "
    "simulator or consume step budget."
)
_GOTO_DESC = (
    "Move to one numbered waypoint from the LATEST observe() result: the "
    "robot turns toward it and walks there. Moving invalidates the old "
    "numbers — call observe() again afterwards to see the new surroundings "
    "and waypoints. Returns steps consumed, remaining budget, and whether "
    "the episode is over."
)
_STOP_DESC = (
    "Permanently END the episode, declaring you have reached the goal. "
    "Issue it only when you believe the robot is within 3 meters of the "
    "instruction's endpoint — stopping is irreversible."
)

mcp = FastMCP("habitat-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_moves = 0  # waypoint moves executed (goto calls that ran)
_episode_over = False
_end_reason: str | None = None
_last_candidates: list[dict[str, float]] | None = None  # None = observe() required
_t0 = time.time()


def _move_fields() -> dict[str, Any]:
    """Waypoint-move budget broadcast — the binding budget in wp mode."""
    remaining = max(0, WP_MAX_MOVES - _moves)
    fields: dict[str, Any] = {"moves_used": _moves, "moves_remaining": remaining}
    if 0 < remaining <= 3:
        fields["MOVE_WARNING"] = (
            f"Only {remaining} waypoint move(s) left before the episode ends. "
            "If you are at the goal, call stop() now; otherwise make them count."
        )
    return fields


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — verbatim contract from mcp_bridge.py."""
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
            "killed. Move to your best goal candidate (or stay right here if it "
            "scores best) and call stop() NOW. Ending without stop scores ZERO."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is killed. "
            "Stop exploring new areas; commit to your best goal candidate, "
            "approach it, and stop() before the budget runs out."
        )
    return fields


def _call(function_name: str, inputs: dict[str, Any],
          config: dict[str, Any] | None = None, base: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"inputs": inputs}
    if config:
        body["config"] = config
    resp = requests.post(
        f"{base or SERVER_URL}/call/{function_name}", json=body, timeout=600
    )
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


# ── candidate handling ──


def _normalize_candidates(raw: Any) -> list[dict[str, float]]:
    """Predictor output -> ordered [{angle, distance}]; marker text is i+1.

    smartway_waypoint__predict: {"0": {angle, distance, rgb_base64}, ...}
    opennav_waypoint__predict:  {"slot": [angle_rad, distance_m], ...}
    Both emit dict entries in ascending angle-index order (the VLN-MME
    numbering); JSON object order survives the HTTP round trip, so response
    order is kept as-is.
    """
    out: list[dict[str, float]] = []
    if not isinstance(raw, dict):
        return out
    for value in raw.values():
        if isinstance(value, dict) and "angle" in value and "distance" in value:
            out.append({"angle": float(value["angle"]), "distance": float(value["distance"])})
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            out.append({"angle": float(value[0]), "distance": float(value[1])})
    return out


def _norm_pi(angle: float) -> float:
    """Normalize to [-pi, pi) — counter-clockwise positive (left)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _direction_of(angle: float) -> str:
    """Quadrant label, verbatim from VLN-MME get_action_options
    (vlnce_baselines/utils.py:275)."""
    a = angle % (2 * math.pi)
    if a <= math.pi / 4 or a >= 7 * math.pi / 4:
        return "Front"
    if a <= 3 * math.pi / 4:
        return "Left"
    if a <= 5 * math.pi / 4:
        return "Back"
    return "Right"


def _action_options(candidates: list[dict[str, float]]) -> dict[str, list[int]]:
    options: dict[str, list[int]] = {"Left": [], "Front": [], "Right": [], "Back": []}
    for i, cand in enumerate(candidates):
        options[_direction_of(cand["angle"])].append(i + 1)
    return options


# ── panorama annotation (VLN-MME display_observation, PIL port) ──


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def _annotate_strip(views: list[dict[str, Any]],
                    candidates: list[dict[str, float]]) -> bytes:
    """[Left|Front|Right|Back] hstack with numbered waypoint markers.

    Angle→x mapping is the VLN-MME formula (vlnce_baselines/utils.py:216-232):
    for counter-clockwise angle θ normalized to [-pi, pi),
    ``x = w_single * (1.5 - 2θ/π)`` — Front centers on tile 2, Left on tile 1,
    Right on tile 3, Back wraps across the strip ends onto tile 4.
    """
    by_dir = {v.get("dir_id"): v for v in views}
    tiles: list[PILImage.Image] = []
    for dir_id, _label in STRIP_DIRS:
        view = by_dir.get(dir_id)
        if view is None or not view.get("rgb_base64"):
            raise ValueError(f"panorama view dir_id={dir_id} missing")
        tiles.append(
            PILImage.open(BytesIO(base64.b64decode(view["rgb_base64"]))).convert("RGB")
        )

    w_single, h = tiles[0].size
    w_full = w_single * 4
    label_h = max(20, h // 14)
    canvas = PILImage.new("RGB", (w_full, h + label_h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w_single, label_h))

    draw = ImageDraw.Draw(canvas)
    label_font = _font(int(label_h * 0.7))
    for i, (_dir_id, label) in enumerate(STRIP_DIRS):
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            ((i + 0.5) * w_single - (bbox[2] - bbox[0]) / 2,
             (label_h - (bbox[3] - bbox[1])) / 2 - bbox[1]),
            label, fill=(0, 0, 0), font=label_font,
        )

    radius = max(12, h // 20)
    num_font = _font(int(radius * 1.15))
    y = label_h + h // 2
    for i, cand in enumerate(candidates):
        theta = _norm_pi(cand["angle"])
        x = w_single * (1.5 - 2 * theta / math.pi)
        if x < 0:
            x += w_full
        x = min(max(x, radius), w_full - radius)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0, 255, 0), outline=(0, 128, 0), width=2,
        )
        text = str(i + 1)
        bbox = draw.textbbox((0, 0), text, font=num_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2 - bbox[0],
             y - (bbox[3] - bbox[1]) / 2 - bbox[1]),
            text, fill=(255, 0, 0), font=num_font,
        )

    if h > WP_VIEW_PX:
        scale = WP_VIEW_PX / h
        canvas = canvas.resize(
            (int(canvas.width * scale), int(canvas.height * scale)), PILImage.LANCZOS
        )
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ── tools ──


@mcp.tool(description=_OBSERVE_DESC)
def observe() -> list:
    global _obs_count, _tool_calls, _last_candidates
    _tool_calls += 1
    if _episode_over:
        return [f"episode already over ({_end_reason}); no more moves possible"]

    pano = _call(
        "env_habitat__observe_panorama", {"trigger": "wp"},
        config={"representation": "views_rgbd", "n_views": N_PANO_VIEWS},
    )
    views = pano.get("views") or []
    # forward only what the predictor reads — drops depth_raw_base64 (~4 MB)
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
        "steps_taken_total": _steps_taken,
        **_move_fields(),
        **_budget_fields(),
    }
    if not _last_candidates:
        status["note"] = (
            "no reachable waypoints predicted here; if you believe you are at "
            "the goal call stop(), otherwise observe() again after the next move"
        )
    _live_log({"observe_wp": True, "num_waypoints": len(_last_candidates)})
    return [Image(data=png, format="png"), json.dumps(status)]


@mcp.tool(description=_GOTO_DESC)
def goto(waypoint: int) -> dict[str, Any]:
    global _steps_taken, _episode_over, _end_reason, _tool_calls, _last_candidates, _moves
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more moves possible"}
    if _last_candidates is None:
        return {"error": "no current waypoint set — call observe() first"}
    try:
        waypoint = int(waypoint)
    except (TypeError, ValueError):
        return {"error": f"invalid waypoint {waypoint!r}; expected an integer"}
    if not 1 <= waypoint <= len(_last_candidates):
        return {
            "error": (
                f"invalid waypoint {waypoint}; valid choices are 1-"
                f"{len(_last_candidates)} from the LATEST observe()"
            )
        }

    cand = _last_candidates[waypoint - 1]
    outputs = _call(
        "env_habitat__step_hightolow",
        {"angle": cand["angle"], "distance": cand["distance"]},
    )
    info = outputs.get("info") or {}
    if isinstance(info.get("step_count"), (int, float)):
        _steps_taken = int(info["step_count"])
    _moves += 1
    terminated = bool(outputs.get("terminated"))
    truncated = bool(outputs.get("truncated"))
    if terminated or truncated:
        _episode_over = True
        _end_reason = "step_budget_exhausted" if truncated else "terminated"
    elif _moves >= WP_MAX_MOVES:
        # decision-step budget spent — truncate like habitat's step cap
        _episode_over = True
        _end_reason = "wp_move_budget_exhausted"
    _last_candidates = None  # position changed; numbers are stale

    result: dict[str, Any] = {
        "moved_to": waypoint,
        "direction": _direction_of(cand["angle"]),
        "distance_m": round(cand["distance"], 2),
        "steps_taken_total": _steps_taken,
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_move_fields(),
        **_budget_fields(),
    }
    if not _episode_over:
        result["note"] = "call observe() to see the new surroundings and waypoints"
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
    result = {
        "stopped": True,
        "steps_taken_total": _steps_taken,
        "episode_over": True,
        "end_reason": _end_reason,
    }
    _live_log(result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
