"""MCP bridge — expose the LIBERO manipulation env nodeset to an external
coding agent.

The manipulation line (2026-08-03): the same two-tool minimal surface as the
habitat-r2r std line, re-embodied on a Franka Panda arm in LIBERO (Liu et
al. 2023, arXiv:2306.03310) via env_libero's gym surface. Where the nav
lines' step() executes discrete primitives, here each action IS the env's
native action space — one 7-D continuous control tick — so the bridge adds
no discretization of its own:

- ``observe()``      -> fixed third-person agentview RGB (base64 PNG
                        passthrough as MCP image)
- ``step(actions)``  -> ordered sequence of 7-number actions
                        [dx, dy, dz, droll, dpitch, dyaw, gripper], every
                        value in [-1, 1]; one action = one OSC control tick

There is NO terminal tool: LIBERO detects task success from the scene state
itself (the wrapper's done flag), so the episode ends on task success or
budget exhaustion — never by agent declaration. step() reports
``task_success`` the moment the env raises it; after that (or after the
budget) movement is refused but observe() keeps working.

Action semantics (calibrated 2026-08-03 on libero_object task 0, the
installed robosuite; magnitudes are stated in the tool description so the
model plans in physical units):

- translation: sustained full-scale (+-1) command moves the EE ~1 cm/tick
  (first tick from rest ~0.4 cm — impedance ramp-up)
- rotation: full-scale command ~5 deg/tick (delta axis-angle)
- gripper: +1 = close, -1 = open (empirically verified 2026-06-28, see
  env_libero); full actuation takes ~12 ticks — the value must be HELD
  across those ticks
- frame: +x from the robot base toward the workspace (and the camera),
  +y = the robot's left, +z = up; in the delivered agentview image the arm
  enters from the top, +x runs toward the bottom edge, +y toward the left
  edge (camera sits across the workspace facing the robot; render is the
  180-degree-flipped agentview, matching VLA training preprocessing)

The step budget mirrors the env's own per-episode cap (_SUITE_MAX_STEPS =
2500): the env truncates there anyway, so the bridge budget exists to
REPORT remaining ticks, and to enforce a lower cap under --nonstd.

Episode selection and metric collection stay driver-side (panel task_id /
episode_index cascade + env_libero__evaluate). The agent never sees the
success ground truth beyond the env's own terminal flag.

One bridge process serves one agent session = one episode (the Agent SDK
spawns a fresh stdio server per session), so per-episode step accounting
can live in module globals.

The non-bare ("full") toolface is the SENSOR rung of the interface ladder
(2026-08-03, after the fable ep0 anatomy: the bare wall is the depth/height
DoF — single view + no proprio makes fingertip height and along-camera-axis
position guesswork). Everything added is a sensor or feedback readout — no
skill, no planner, no task logic:

- observe() also returns the WRIST camera view and a proprio readout
  (EE position + gripper opening; both native to the env's obs bundle)
- step() reports the MEASURED EE displacement over the call (commanded vs
  achieved — a shortfall reveals a stall, the go2 line's precedent) plus
  the post-move proprio
- with LIBERO_AUTO_OBSERVE=1 the step() result also carries the resulting
  views, so observe() is a first-look only (the nav lines' auto-observe)

The TOOLBOX surface (2026-08-04, user direction: max out the tool surface
first, attribute later — "先跑通") replaces observe() with atomic per-view /
per-sensor reads and adds ground-truth + servo macros over the env's own
frozen VoxPoser-era nodes (env_libero__observe_objects / __step_ee_pose).
Every tool is independent — one tool, one job:

- observe_third_person() / observe_wrist()  — one RGB view each
- get_state()    — EE position + gripper opening (pure read)
- get_objects()  — the simulator's ground-truth scene readout: per-object
                   3D center + size (privileged; the point of this rung)
- move_to(x,y,z) — closed-loop OSC servo to a world position, current
                   orientation and gripper command held
- gripper(cmd)   — "close"/"open" actuation + resulting opening readout
- step(actions)  — the native 7-D tick surface, unchanged (escape hatch)

Env vars:
    LIBERO_SERVER_URL     auto_host base URL (default http://127.0.0.1:9250)
    LIBERO_STEP_BUDGET    control-tick budget reported/enforced by this bridge
                          (default 2500 = the env's own cap)
    LIBERO_TURN_BUDGET    the driver's max_turns; when set (>0), every step()
                          result reports tool calls used/remaining and injects
                          an escalating BUDGET_WARNING near exhaustion
    LIBERO_BARE           "1" = bare toolface (agentview + step only: no wrist
                          view, no proprio, no measured-movement feedback)
    LIBERO_AUTO_OBSERVE   "1" = step() results carry the post-move views
                          (observe() becomes a first-look only)
    LIBERO_TOOLBOX        "1" = the loaded toolbox surface above (replaces
                          observe(); implies non-bare; AUTO_OBSERVE ignored
                          — move results are JSON, views are on-demand)
    LIBERO_TOOLBOX_GT     "1" (default) = privileged get_objects readout;
                          "0" = non-privileged pixel_to_3d depth
                          backprojection instead (condition
                          libero_toolbox_vision)
    LIBERO_LIVE_DIR       optional dir for live spectating: every view the
                          agent is shown lands as its own obs group —
                          obs_NNNN_stepSSSS.png (third-person, + overwritten
                          latest.png) and obs_NNNN_stepSSSS_wrist.png — one
                          group per labeled image block in the tool result
                          (the Coding-Agent Monitor's pairing contract);
                          every step call appends a line to actions.log
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from mcp.server.fastmcp import FastMCP, Image

SERVER_URL = os.environ.get("LIBERO_SERVER_URL", "http://127.0.0.1:9250")
STEP_BUDGET = int(os.environ.get("LIBERO_STEP_BUDGET", "2500"))
TURN_BUDGET = int(os.environ.get("LIBERO_TURN_BUDGET", "0"))
LIVE_DIR = Path(os.environ["LIBERO_LIVE_DIR"]) if os.environ.get("LIBERO_LIVE_DIR") else None
MAX_ACTIONS_PER_CALL = 100
MAX_PIXELS_PER_CALL = 100  # pixel_to_3d batch cap (one depth render serves all)
# Vanilla-baseline switch: bare = observe(agentview RGB) + step only. The
# non-bare extras are sensors and feedback only (wrist view, proprio,
# measured EE movement) — the manipulation analog of the nav bridges'
# depth-derived clearance.
BARE = False  # BAKED (libero_tb arm; was env LIBERO_*)
# Auto-observe (nav-line precedent): step() results carry the post-move
# views, observe() is a first-look only. The briefing and this flag are
# driven off the same cells knob so they can never disagree.
AUTO_OBSERVE = False  # BAKED (libero_tb arm; was env LIBERO_*)
# Loaded-toolbox surface: atomic per-view/per-sensor reads +
# GT scene readout + servo macros, replacing observe(). Cells drive it via
# the toolbox knob (condition libero_toolbox); implies non-bare, and the
# step() result stays JSON-only (no auto-observe image attach).
TOOLBOX = True  # BAKED (libero_tb arm; was env LIBERO_*)
if TOOLBOX:
    AUTO_OBSERVE = False
# GT switch within the toolbox:
# "1" (default) = the privileged get_objects scene readout; "0" = the
# non-privileged locator instead — pixel_to_3d depth backprojection
# (camera geometry + depth buffer only, never sim object state).
TOOLBOX_GT = True  # BAKED (libero_tb arm; was env LIBERO_*)

_OBSERVE_DESC = (
    "Look through the fixed third-person camera. Returns the current RGB "
    "view of the workspace (the robot arm enters from the top of the "
    "image). Pure read — does not advance the simulation or consume the "
    "tick budget."
    if BARE else
    "Look at the workspace through BOTH cameras.\n\n"
    "Returns two RGB views — the fixed third-person camera (the robot arm "
    "enters from the top of the image) and the wrist camera looking out "
    "along the gripper — plus a proprio readout: the end-effector's "
    "position in meters and the gripper opening in millimeters."
    + (" You only need this for your FIRST look — every step() result "
       "carries the updated views." if AUTO_OBSERVE else "")
    + " Pure read — does not advance the simulation or consume the tick "
    "budget."
)
_STEP_DESC = (
    "Execute a sequence of control ticks, in order. Each action is a list "
    "of 7 numbers [dx, dy, dz, droll, dpitch, dyaw, gripper], every value "
    "in [-1, 1].\n\n"
    "dx/dy/dz move the end-effector: +x away from the robot toward the "
    "bottom of the camera image, +y to the robot's left (the left of the "
    "image), +z up. A sustained full-scale (±1) command moves about 1 cm "
    "per tick. droll/dpitch/dyaw rotate the gripper about those same axes "
    "(about 5 degrees per tick at full scale). gripper: +1 closes, -1 "
    "opens — actuation takes about 12 ticks, so HOLD the value across the "
    "sequence while it completes. Repeat an action to travel, e.g. ten "
    "copies of [0,0,-1,0,0,0,-1] descends ~10 cm with the gripper open.\n\n"
    "Executes sequentially; halts early on task success or when the tick "
    "budget runs out (you can still observe() after that). Returns how "
    "many ticks ran, total ticks taken, remaining budget, and whether the "
    "environment has detected task success."
    + ("" if BARE else
       " Also reports the end-effector's MEASURED movement in cm over the "
       "call and the current proprio readout — trust the measured movement "
       "over the commanded amount; a shortfall means the arm stalled on an "
       "obstacle.")
    + (" The resulting camera views are attached to the result, so no "
       "separate observe() is needed." if AUTO_OBSERVE else
       " The scene changes after stepping — check with "
       "observe_third_person(), observe_wrist() or get_state()." if TOOLBOX
       else
       " The camera view changes after stepping — call observe() to see "
       "the result.")
)
if TOOLBOX:
    _STEP_DESC = (
        "Low-level escape hatch — usually move_to()/gripper() serve better; "
        "use this for what they cannot express (rotating the wrist, sub-cm "
        "nudges, custom motions).\n\n" + _STEP_DESC
    )

mcp = FastMCP("libero-env")

_steps_taken = 0
_obs_count = 0
_tool_calls = 0
_episode_over = False
_end_reason: str | None = None
_t0 = time.time()


def _budget_fields() -> dict[str, Any]:
    """Turn-budget broadcast — one tool call ≈ one harness turn.

    The driver's max_turns is the budget that actually kills sessions, but the
    model has no way to observe it. Report it in every step result and escalate
    to explicit orders near exhaustion."""
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
            "is killed. Finish the task NOW or it scores zero."
        )
    elif remaining <= 20:
        fields["BUDGET_WARNING"] = (
            f"Only {remaining} tool calls remain before this session is "
            "killed. Move decisively toward completing the task."
        )
    return fields


def _ndarr(field: Any) -> np.ndarray | None:
    """Decode an auto_host ndarray marker ({__ndarray__, dtype, shape})."""
    if not isinstance(field, dict) or "__ndarray__" not in field:
        return None
    try:
        return np.frombuffer(
            base64.b64decode(field["__ndarray__"]),
            dtype=field.get("dtype", "float32"),
        ).reshape(field["shape"])
    except Exception:
        return None


def _proprio(state_field: dict[str, Any] | None) -> dict[str, Any] | None:
    """EE position + gripper width from the 8-D proprio state (non-bare only).

    state = eef_pos(3) + axis_angle(3) + gripper_qpos(2); the two finger
    joints mirror each other, so opening width = q6 - q7 (~77 mm open,
    ~0 closed)."""
    arr = _ndarr(state_field)
    if arr is None or arr.size < 8:
        return None
    return {
        "ee_pos_m": [round(float(v), 3) for v in arr[:3]],
        "gripper_open_mm": round(float(arr[6] - arr[7]) * 1000, 1),
    }


def _quat_xyzw_to_rpy_deg(q: np.ndarray) -> list[float]:
    """xyzw quaternion → world-frame roll/pitch/yaw in degrees (ZYX)."""
    x, y, z, w = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [round(math.degrees(v), 1) for v in (roll, pitch, yaw)]


def _live_frames(agent_png: bytes | None, wrist_png: bytes | None) -> None:
    """Dump the captured views for the Coding-Agent Monitor / live spectating.

    Each view gets its OWN obs index: the monitor groups on-disk frames by
    obs_<NNNN> and pairs one group per LABELED image block in the tool
    result, so a two-view capture must write two groups — and conversely a
    capture whose images are NOT attached to a result must write none, or
    the monitor's group cursor desyncs. The _wrist tag is cosmetic (only
    _depth means anything to the monitor)."""
    global _obs_count
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if agent_png is not None:
        _obs_count += 1
        (LIVE_DIR / f"obs_{_obs_count:04d}_step{_steps_taken:04d}.png").write_bytes(agent_png)
        (LIVE_DIR / "latest.png").write_bytes(agent_png)
    if wrist_png is not None:
        _obs_count += 1
        (LIVE_DIR / f"obs_{_obs_count:04d}_step{_steps_taken:04d}_wrist.png").write_bytes(wrist_png)


def _live_log(entry: dict[str, Any]) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with (LIVE_DIR / "actions.log").open("a") as fh:
        fh.write(json.dumps({"t": round(time.time() - _t0, 1), **entry}) + "\n")


def _call(function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{SERVER_URL}/call/{function_name}", json={"inputs": inputs}, timeout=600
    )
    resp.raise_for_status()
    return resp.json()["outputs"]


def _capture_obs() -> tuple[bytes | None, bytes | None, dict[str, Any] | None]:
    """Pure read of the current obs bundle: (agentview PNG, wrist PNG,
    proprio). No sim advance — the env node returns its cached last obs."""
    outputs = _call("env_libero__observe_egocentric", {})
    agent = outputs.get("agentview_image")
    wrist = outputs.get("wrist_image")
    return (
        base64.b64decode(agent) if isinstance(agent, str) else None,
        base64.b64decode(wrist) if isinstance(wrist, str) else None,
        _proprio(outputs.get("state")),
    )


def observe() -> list:
    global _tool_calls
    _tool_calls += 1
    agent_png, wrist_png, proprio = _capture_obs()
    if agent_png is None:
        return [json.dumps({"error": "no observation available — env returned no image"})]
    if BARE:
        _live_frames(agent_png, None)
        return [Image(data=agent_png, format="png")]
    _live_frames(agent_png, wrist_png)
    status = {"proprio": proprio, **_budget_fields()}
    content: list[Any] = ["third-person view:", Image(data=agent_png, format="png")]
    if wrist_png is not None:
        content += ["wrist view:", Image(data=wrist_png, format="png")]
    content.append(json.dumps(status))
    return content


if not TOOLBOX:  # toolbox replaces observe() with atomic per-view reads
    mcp.tool(description=_OBSERVE_DESC)(observe)


@mcp.tool(description=_STEP_DESC)
def step(actions: list[list[float]]) -> list:  # bare `list` => FastMCP unstructured path
    # (handles both the [views, json] auto-observe return and the plain dict;
    # a structured annotation can't serialize the Image content — same reason
    # observe() is annotated -> list)
    """Execute a sequence of 7-number control ticks, in order.

    Each action is [dx, dy, dz, droll, dpitch, dyaw, gripper], every value
    in [-1, 1]. One action = one control tick (~1 cm translation / ~5 deg
    rotation at full scale; gripper +1 closes / -1 opens over ~12 ticks).
    Halts early on task success or budget exhaustion."""
    global _tool_calls
    _tool_calls += 1
    if _episode_over:
        return {"error": f"episode already over ({_end_reason}); no more ticks possible"}
    if not actions:
        return {"error": "empty action list"}
    if len(actions) > MAX_ACTIONS_PER_CALL:
        return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
    for i, action in enumerate(actions):
        if not isinstance(action, (list, tuple)) or len(action) != 7:
            return {"error": f"action {i} is not a 7-number list: {action!r}"}
        bad = [v for v in action
               if not isinstance(v, (int, float)) or not math.isfinite(v)
               or not -1.0 <= float(v) <= 1.0]
        if bad:
            return {"error": f"action {i} has values outside [-1, 1]: {action!r}"}
    return _execute_actions([[float(v) for v in a] for a in actions])


def _execute_actions(actions: list[list[float]]) -> Any:
    global _steps_taken, _episode_over, _end_reason
    requested = len(actions)
    remaining = STEP_BUDGET - _steps_taken
    truncated_to_budget = requested > remaining
    if truncated_to_budget:
        actions = actions[:remaining]

    # Pre-move proprio for the measured-movement readout (non-bare only;
    # pure read, no sim advance).
    pre_proprio: dict[str, Any] | None = None
    if not BARE:
        try:
            _, _, pre_proprio = _capture_obs()
        except Exception:  # noqa: BLE001 — feedback must not block movement
            pre_proprio = None

    prev_steps = _steps_taken
    outputs = _call("env_libero__step_continuous", {"action": json.dumps(actions)})
    env_err = (outputs.get("info") or {}).get("error")
    if env_err:
        return {"error": f"env error: {env_err}"}
    # The env manager's step counter is authoritative (it early-breaks the
    # chunk on success / its own cap, so len(actions) is only an upper bound).
    _steps_taken = int(outputs.get("step_index") or prev_steps)
    success = bool(outputs.get("success"))
    env_done = bool(outputs.get("terminated")) or bool(outputs.get("truncated"))

    if success:
        _episode_over = True
        _end_reason = "task_success"
    elif env_done or _steps_taken >= STEP_BUDGET:
        _episode_over = True
        _end_reason = "budget_exhausted"

    result: dict[str, Any] = {
        "executed": _steps_taken - prev_steps,
        "requested": requested,
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "task_success": success,
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_budget_fields(),
    }
    if success:
        result["message"] = (
            "TASK COMPLETE — the environment has detected success. The "
            "episode is over; no further actions are needed."
        )
    elif _episode_over:
        result["message"] = (
            "Tick budget exhausted — no more movement possible. The episode "
            "ends here; you can still observe() the final state."
        )
    _live_log({"actions": len(actions), **result})
    return _with_feedback(result, pre_proprio)


def _with_feedback(result: dict[str, Any], pre_proprio: dict[str, Any] | None) -> Any:
    """Attach the non-bare sensor feedback: measured EE movement + post-move
    proprio, and (auto-observe) the resulting camera views so the agent never
    spends a separate observe() turn. A finished episode carries no view
    (nothing left to act on), but keeps the final proprio readout."""
    if BARE:
        return result
    try:
        agent_png, wrist_png, post = _capture_obs()
    except Exception:  # noqa: BLE001 — feedback must not lose the step result
        return result
    if post is not None:
        if pre_proprio is not None:
            result["ee_moved_cm"] = [
                round((a - b) * 100, 1)
                for a, b in zip(post["ee_pos_m"], pre_proprio["ee_pos_m"])
            ]
        result["proprio"] = post
    if not AUTO_OBSERVE or _episode_over or agent_png is None:
        return result
    result["new_view"] = (
        "the attached images are the camera views AFTER these actions"
    )
    # Text labels before each image, exactly like observe(): the model sees
    # which view is which, and the monitor's viewpoint detection splits the
    # result into one tile per LABELED image (unlabeled consecutive images
    # would be misread as an RGB+depth pair of ONE viewpoint).
    _live_frames(agent_png, wrist_png)
    content: list[Any] = ["third-person view:", Image(data=agent_png, format="png")]
    if wrist_png is not None:
        content += ["wrist view:", Image(data=wrist_png, format="png")]
    content.append(json.dumps(result))
    return content


# ── toolbox surface ──────────────────────────────────────
# Atomic tools over the env's frozen VoxPoser-era nodes. Registered only
# under LIBERO_TOOLBOX=1; step() above is shared by all surfaces. Every
# mover returns JSON carrying steps_taken_total so the driver's EventSink
# env-step accounting keeps working unchanged.

# The last commanded gripper state — move_to must HOLD it every servo tick
# (the env's step_ee_pose sends a gripper command with every substep, and
# releasing mid-transport drops the object). Episodes start open
# (num_steps_wait settles with the gripper-open dummy action).
_gripper_closed = False


def _absorb_move(outputs: dict[str, Any], prev_steps: int) -> dict[str, Any]:
    """Shared bookkeeping for the macro movers (move_to / gripper): sync the
    bridge step counter to the env manager's authoritative step_index, fold
    in success / budget terminal state, and build the common result fields."""
    global _steps_taken, _episode_over, _end_reason
    _steps_taken = int(outputs.get("step_index") or prev_steps)
    success = bool(outputs.get("success"))
    env_done = bool(outputs.get("terminated")) or bool(outputs.get("truncated"))
    if success:
        _episode_over = True
        _end_reason = "task_success"
    elif env_done or _steps_taken >= STEP_BUDGET:
        _episode_over = True
        _end_reason = "budget_exhausted"
    result: dict[str, Any] = {
        "ticks_used": _steps_taken - prev_steps,
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "task_success": success,
        "episode_over": _episode_over,
        "end_reason": _end_reason,
        **_budget_fields(),
    }
    if success:
        result["message"] = (
            "TASK COMPLETE — the environment has detected success. The "
            "episode is over; no further actions are needed."
        )
    elif _episode_over:
        result["message"] = (
            "Tick budget exhausted — no more movement possible. The episode "
            "ends here; you can still look at the final state."
        )
    return result


if TOOLBOX:

    @mcp.tool(description=(
        "Look through the fixed third-person camera across the workspace "
        "(the robot arm enters from the top of the image). Returns one RGB "
        "image. Pure read — does not advance the simulation."))
    def observe_third_person() -> list:
        global _tool_calls
        _tool_calls += 1
        agent_png, _, _ = _capture_obs()
        if agent_png is None:
            return [json.dumps({"error": "no observation available — env returned no image"})]
        _live_frames(agent_png, None)
        return [Image(data=agent_png, format="png")]

    @mcp.tool(description=(
        "Look through the wrist camera, mounted on the gripper looking out "
        "along it — a close-up of whatever is under the end-effector. "
        "Returns one RGB image. Pure read — does not advance the simulation."))
    def observe_wrist() -> list:
        global _tool_calls
        _tool_calls += 1
        _, wrist_png, _ = _capture_obs()
        if wrist_png is None:
            return [json.dumps({"error": "no wrist view available — env returned no image"})]
        _live_frames(None, wrist_png)
        return [Image(data=wrist_png, format="png")]

    @mcp.tool(description=(
        "Read the robot's full proprioception (encoder-level, always "
        "available): end-effector world pose — position in meters plus "
        "orientation as world-frame roll/pitch/yaw in degrees (pointing "
        "straight down reads roll ~180, pitch ~0) — the 7 arm joint "
        "angles in radians (base to wrist), and the gripper opening in "
        "millimeters (~77 fully open, ~0 fully closed). Pure read — does "
        "not advance the simulation."))
    def get_state() -> list:  # bare list => unstructured path (single-encoded JSON text)
        global _tool_calls
        _tool_calls += 1
        outputs = _call("env_libero__observe_egocentric", {})
        proprio = _proprio(outputs.get("state"))
        if proprio is None:
            return {"error": "no proprio state available"}
        full: dict[str, Any] = dict(proprio)
        quat = _ndarr(outputs.get("ee_quat_xyzw"))
        if quat is not None and quat.size == 4:
            full["ee_rpy_deg"] = _quat_xyzw_to_rpy_deg(quat)
        joints = _ndarr(outputs.get("joint_pos"))
        if joints is not None:
            full["joint_pos_rad"] = [round(float(v), 3) for v in joints]
        # No steps_taken_total here: a pure read must never become the
        # sink's last_step_result (it carries no end_reason and would
        # blank the episode record's terminal fields).
        return {
            **full,
            "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
            **_budget_fields(),
        }

    if TOOLBOX_GT:

        @mcp.tool(description=(
            "Read the simulator's ground-truth scene state: every task object's "
            "3D center and size in meters (world frame: +x away from the robot, "
            "+y the robot's left, +z up — the same frame move_to uses), plus "
            "your end-effector position and whether the gripper is open. This "
            "is exact — trust it over pixel estimates. Pure read — does not "
            "advance the simulation."))
        def get_objects() -> list:  # bare list => unstructured path
            global _tool_calls
            _tool_calls += 1
            snap = _call("env_libero__observe_objects", {}).get("snapshot") or {}
            if snap.get("error"):
                return {"error": snap["error"]}
            objects: dict[str, Any] = {}
            for name, d in (snap.get("object_pcs") or {}).items():
                pc = np.asarray(d.get("pc") or [], dtype=np.float32)
                if pc.size == 0:
                    continue
                mn, mx = pc.min(axis=0), pc.max(axis=0)
                objects[name] = {
                    "center_m": [round(float(v), 3) for v in (mn + mx) / 2],
                    "size_m": [round(float(v), 3) for v in (mx - mn)],
                }
            return {
                "objects": objects,
                "ee_pos_m": [round(float(v), 3) for v in (snap.get("ee_pos") or [])],
                "gripper_open": bool(snap.get("gripper_open")),
            }

    else:

        @mcp.tool(description=(
            "Convert pixels of your LATEST camera image into the 3D world "
            "positions of the visible surfaces at those pixels (depth "
            "backprojection — camera geometry only, no privileged state). "
            'camera is "third_person" or "wrist"; points is a list of '
            "[x, y] pixels (x = column 0-255 left to right, y = row 0-255 "
            "top to bottom, exactly as you see the image), up to "
            f"{MAX_PIXELS_PER_CALL} per call. Batch them: probing a small "
            "grid across an object in ONE call and averaging the returns "
            "that cluster at the same height locates its center far better "
            "than a single click (single clicks land on whatever feature "
            "you aimed at and are biased a few cm). Points are in the same "
            "world frame move_to uses, and are the VISIBLE surface — "
            "clicking an object's top from above returns its TOP; the body "
            "extends below. Pure read — does not advance the simulation."))
        def pixel_to_3d(camera: str, points: list[list[int]]) -> list:  # bare list => unstructured path
            global _tool_calls
            _tool_calls += 1
            cam = str(camera).strip().lower()
            if cam not in ("third_person", "agentview", "wrist"):
                return {"error": f'camera must be "third_person" or "wrist", got {camera!r}'}
            if not isinstance(points, (list, tuple)) or not points:
                return {"error": "points must be a non-empty list of [x, y] pairs"}
            if (len(points) == 2
                    and all(isinstance(v, (int, float)) for v in points)):
                points = [points]  # tolerate a bare [x, y]
            if len(points) > MAX_PIXELS_PER_CALL:
                return {"error": f"too many points in one call (max {MAX_PIXELS_PER_CALL})"}
            outputs = _call("env_libero__pixel_to_3d", {
                "camera": "wrist" if cam == "wrist" else "agentview",
                "points": [list(p) for p in points],
            })
            if outputs.get("error"):
                return {"error": outputs["error"]}
            results = outputs.get("results") or []
            return {"results": [
                ({"pixel": list(map(int, p)), "point_m": r.get("point"),
                  "depth_m": r.get("depth_m")}
                 if "point" in r else
                 {"pixel": list(p) if isinstance(p, (list, tuple)) else p,
                  "error": r.get("error")})
                for p, r in zip(points, results)
            ]}

    # Two-phase servo (tuned on an oracle smoke run): the env's
    # step_ee_pose alone advances its bounded OSC goal only 5 mm/substep and
    # the arm tracks ~1.2 mm/tick, so a single 100-substep call covers barely
    # 12 cm. The COARSE phase instead drives full-scale native ticks straight
    # at the target (~1 cm/tick, the calibrated rate) with a measured-progress
    # stall check per round, then hands the last couple of cm to step_ee_pose
    # for the precise closed-loop landing (which also re-locks orientation).
    _COARSE_TOL_M = 0.025      # hand off to the fine servo below this
    _COARSE_TICK_M = 0.010     # calibrated full-scale translation per tick
    _COARSE_MAX_ROUNDS = 14
    _COARSE_CHUNK_TICKS = 40
    _STALL_FRACTION = 0.3      # measured/commanded below this = blocked

    def _ee_pos_now() -> list[float] | None:
        try:
            _, _, p = _capture_obs()
        except Exception:  # noqa: BLE001
            return None
        return None if p is None else p["ee_pos_m"]

    @mcp.tool(description=(
        "Servo the end-effector to a world position [x, y, z] in meters. "
        "Keeps going until it arrives (within ~1 cm) or genuinely stalls on "
        "an obstacle — reached=false means BLOCKED, not out of steam. The "
        "current gripper command is HELD throughout, so a grasped object "
        "travels with the arm; the wrist orientation is held too. Moves "
        "straight toward the target — route AROUND obstacles yourself "
        "(go UP, across, then DOWN). Advances the simulation, ~1-2 ticks "
        "per cm. Returns the measured final position."))
    def move_to(x: float, y: float, z: float) -> list:  # bare list => unstructured path
        global _tool_calls
        _tool_calls += 1
        if _episode_over:
            return {"error": f"episode already over ({_end_reason}); no more movement possible"}
        vals = [x, y, z]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            return {"error": f"target must be three finite numbers, got {vals!r}"}
        target = np.asarray([float(v) for v in vals], dtype=np.float64)
        g = 1.0 if _gripper_closed else -1.0
        prev_steps = _steps_taken
        stalled = False
        outputs: dict[str, Any] = {}

        # Coarse phase: full-scale native ticks straight at the target.
        pos_list = _ee_pos_now()
        if pos_list is None:
            return {"error": "could not read EE position"}
        pos = np.asarray(pos_list, dtype=np.float64)
        for _ in range(_COARSE_MAX_ROUNDS):
            err = target - pos
            dist = float(np.linalg.norm(err))
            if dist < _COARSE_TOL_M:
                break
            n = max(3, min(_COARSE_CHUNK_TICKS, int(dist / _COARSE_TICK_M)))
            unit = (err / dist).tolist()
            outputs = _call("env_libero__step_continuous", {
                "action": json.dumps([[*unit, 0.0, 0.0, 0.0, g]] * n),
            })
            env_err = (outputs.get("info") or {}).get("error")
            if env_err:
                return {"error": f"env error: {env_err}"}
            if (bool(outputs.get("success")) or bool(outputs.get("terminated"))
                    or bool(outputs.get("truncated"))):
                break
            new_list = _ee_pos_now()
            if new_list is None:
                break
            new_pos = np.asarray(new_list, dtype=np.float64)
            moved = float(np.linalg.norm(new_pos - pos))
            pos = new_pos
            if moved < max(0.005, _STALL_FRACTION * n * _COARSE_TICK_M):
                stalled = True
                break

        # Fine phase: precise closed-loop landing at the current orientation
        # (read via the same mat2quat path step_ee_pose compares against).
        env_live = not (bool(outputs.get("success")) or bool(outputs.get("terminated"))
                        or bool(outputs.get("truncated")))
        if env_live and not stalled:
            snap = _call("env_libero__observe_objects", {}).get("snapshot") or {}
            quat = snap.get("ee_quat")
            if not quat or len(quat) != 4:
                return {"error": "could not read current EE orientation"}
            outputs = _call("env_libero__step_ee_pose", {
                "action": json.dumps([*target.tolist(), *[float(q) for q in quat],
                                      1.0 if _gripper_closed else 0.0]),
            })
            env_err = (outputs.get("info") or {}).get("error")
            if env_err:
                return {"error": f"env error: {env_err}"}

        result = _absorb_move(outputs, prev_steps)
        final_pos = _ee_pos_now()
        if final_pos is not None:
            result["ee_pos_m"] = final_pos
            residual_cm = float(np.linalg.norm(
                target - np.asarray(final_pos, dtype=np.float64))) * 100
            result["reached"] = residual_cm <= 1.5
            if not result["reached"] and not result["task_success"]:
                result["note"] = (
                    f"did NOT reach the target — stopped {residual_cm:.1f} cm "
                    "short" + (" (the arm stalled against something; back off "
                               "and approach differently)" if stalled else "")
                )
        else:
            result["reached"] = bool(outputs.get("converged"))
        _live_log({"move_to": [round(float(v), 3) for v in vals], **{
            k: result[k] for k in ("reached", "ticks_used", "steps_taken_total", "task_success")
            if k in result}})
        return result

    @mcp.tool(description=(
        'Actuate the gripper: "close" or "open" (takes ~20 simulation '
        "ticks). Returns the resulting opening in millimeters — after "
        '"close", a width near 0 means the fingers closed on AIR (you '
        "grasped nothing); a width near the object's size means you are "
        "holding it. The command persists: move_to keeps it applied until "
        "you change it. Advances the simulation and consumes ticks."))
    def gripper(command: str) -> list:  # bare list => unstructured path
        global _tool_calls, _gripper_closed
        _tool_calls += 1
        if _episode_over:
            return {"error": f"episode already over ({_end_reason}); no more movement possible"}
        cmd = str(command).strip().lower()
        if cmd not in ("close", "open"):
            return {"error": f'command must be "close" or "open", got {command!r}'}
        g = 1.0 if cmd == "close" else -1.0
        prev_steps = _steps_taken
        outputs = _call("env_libero__step_continuous", {
            "action": json.dumps([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g]] * 20),
        })
        env_err = (outputs.get("info") or {}).get("error")
        if env_err:
            return {"error": f"env error: {env_err}"}
        _gripper_closed = cmd == "close"
        result = _absorb_move(outputs, prev_steps)
        try:
            _, _, post = _capture_obs()
        except Exception:  # noqa: BLE001 — readout must not lose the actuation result
            post = None
        if post is not None:
            result["gripper_open_mm"] = post["gripper_open_mm"]
            if cmd == "close":
                result["holding"] = post["gripper_open_mm"] > 5.0
        _live_log({"gripper": cmd, **{
            k: result[k] for k in ("gripper_open_mm", "holding", "steps_taken_total", "task_success")
            if k in result}})
        return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
