"""Go2 host — the auto_host stand-in for a real Unitree Go2.

Serves the same HTTP surface the habitat bridge consumes (``POST /call/{fn}``
returning ``{"outputs": {...}}``), so ``go2_bridge.py`` is a near-copy of the
frozen ``mcp_bridge.py`` rather than a new protocol.

Runs ON THE ROBOT'S HOST MACHINE, not here: CycloneDDS is a layer-2 protocol
and the Go2 requires the host to sit on its ``192.168.123.x`` wire, so every
line that touches ``SportClient`` has to live next to the dog. Nothing else
does — this file has zero agentcanvas imports and is the only piece deployed.

Deliberately NOT an env nodeset and NOT on the server-mode path (ADR-server-001):
the backend's auto_host dials ``localhost`` unconditionally, so a cross-machine
nodeset would need the never-exercised ``workspace/servers/*.yaml`` route. This
sidesteps that entirely.

Functions (mirroring env_habitat's verb names so the bridge diff stays small):

  env_go2__reset               stand up, balance, zero the step counter
  env_go2__step_discrete       0=STOP 1=FORWARD 2=TURN_LEFT 3=TURN_RIGHT
  env_go2__observe_egocentric  head-camera RGB (base64 JPEG)

Motion safety — the reason this file is hand-written rather than manifest-derived:

  * only Move / StopMove / StandUp / BalanceStand are reachable. The 41-method
    SportClient surface (BackFlip, HandStand, FrontJump, Dance…) is not wired to
    any action integer, so no agent output can reach it;
  * velocities are clamped to MAX_VX / MAX_VYAW regardless of what is asked;
  * ``Move`` is ``_CallNoReply`` — a velocity setpoint with an onboard watchdog,
    not a displacement. Each discrete action re-issues it at CMD_HZ for a
    computed duration and then calls StopMove(), so a dropped packet or a crashed
    host leaves the dog stationary rather than running;
  * StopMove() is issued in a finally-block on every path, including exceptions.

Motion is CLOSED-LOOP on the robot's own SportModeState feed (yaw from IMU,
position from onboard odometry); open-loop time-integrated velocity survives
only as the fallback when that feed goes stale. Every step's info reports what
was measured, never what was commanded.

Usage (in the ``unitree`` conda env on the robot host)::

    python go2_host.py --iface enp6s0 --port 9300

Env vars: GO2_IFACE, GO2_PORT, GO2_STEP_BUDGET, GO2_DRY_RUN
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ── motion constants ──
# Full habitat parity: 0.25 m / 15° (user decision 2026-07-20 night). The early
# free-gait calibration condemned these (±42% scatter — one gait cycle was
# longer than the step) and moved to 0.5/30; re-tested under the pinned
# StaticWalk gait with coast compensation they hold to ±6% with <2% bias, so
# parity with the simulator benchmarks wins back.
STEP_M = float(os.environ.get("GO2_STEP_M", "0.25"))
TURN_DEG = float(os.environ.get("GO2_TURN_DEG", "15"))
# Set from the remote-controller baseline measured 2026-07-20, NOT from caution:
# 30 s of hand-driven operation peaked at 1.23 m/s and 2.38 rad/s. These sit at
# roughly half that, which is well inside the gait's normal operating band.
MAX_VX = float(os.environ.get("GO2_MAX_VX", "0.6"))      # m/s
# 1.0, down from 1.5 (2026-07-20 night): the classic (常规) gait style the
# operator chose has a speed envelope, and commands above it appear to make the
# firmware auto-upgrade the style to 灵动 — the exact thing pinning is meant to
# prevent. 1.0 rad/s is operator-verified to stay in the classic gait.
MAX_VYAW = float(os.environ.get("GO2_MAX_VYAW", "1.0"))  # rad/s
CMD_HZ = 50.0            # Move is a watchdog'd setpoint; re-issue at this rate
SETTLE_S = 0.35          # let the gait actually stop before the next observe
STEP_BUDGET = int(os.environ.get("GO2_STEP_BUDGET", "500"))
DRY_RUN = os.environ.get("GO2_DRY_RUN") == "1"

# ── closed-loop control ──
# The floors are the load-bearing constants here, and they are counter-intuitive.
#
# Diagnosis history worth keeping, because two plausible theories were wrong:
# commanding 24 turns of a nominal 15 deg first produced only ~165 deg of the
# requested 360. It was NOT an acceleration ramp (closing the loop changed
# nothing) and NOT a stopped sport service or wrong motion mode (30 s of passive
# observation while the operator hand-drove the robot PERFECTLY showed mode and
# gait_type pinned at 0 and CheckMode pinned at 'mcf' the whole time — those
# fields simply do not report locomotion on this firmware).
#
# The cause was speed: the remote peaked at 2.38 rad/s while we commanded at most
# 0.5 and as little as 0.15. Those rates sit in a dead band where the gait barely
# engages, so the robot shuffled instead of walking. Setting a low ceiling for
# safety made the motion LESS controllable, not more — hence floors high enough
# to guarantee the gait actually engages.
YAW_TOL = float(os.environ.get("GO2_YAW_TOL_DEG", "2.0")) * 3.141592653589793 / 180.0
# Coast compensation: after StopMove the body keeps rotating a highly stable
# ~+2.2 deg (StaticWalk gait, measured over 24+24 steps 2026-07-20 night), so
# the controller aims short by this much and the coast carries it to target.
YAW_COAST = float(os.environ.get("GO2_YAW_COAST_DEG", "2.2")) * 3.141592653589793 / 180.0
# Same idea for forward: the body coasts a stable ~+2 cm past the break point
# (StaticWalk; measured +7.8% at 0.25 m and +1.9% at 0.5 m — same absolute).
POS_COAST = float(os.environ.get("GO2_POS_COAST_M", "0.02"))
POS_TOL = float(os.environ.get("GO2_POS_TOL_M", "0.03"))
K_YAW, K_X = 2.0, 1.5     # proportional gains on remaining error
MIN_VYAW, MIN_VX = 0.6, 0.25
CTRL_TIMEOUT_S = float(os.environ.get("GO2_CTRL_TIMEOUT_S", "6.0"))
# How long to wait for the WiFi DDS feed to come back — at step start and after
# a mid-step dropout — before giving up on closed-loop. Deliberately much longer
# than CTRL_TIMEOUT_S: waiting is free and safe (the dog holds still), whereas
# both alternatives are bad (open-loop moves blind, abandoning loses the step).
STALE_WAIT_S = float(os.environ.get("GO2_STALE_WAIT_S", "20.0"))

ACTION_STOP, ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

_sport: Any = None
_video: Any = None
_state: dict[str, Any] = {}   # latest SportModeState_ fields (feedback source)
_steps = 0
_episode_over = False


# ── robot lifecycle ──


def init_robot(iface: str) -> None:
    """Bring up DDS, the two clients, and the state subscriber that closes the
    control loop. Imports are function-local so --dry-run works on a machine
    without the SDK (schema/protocol testing off-robot)."""
    global _sport, _video, _sub
    if DRY_RUN:
        print("[go2_host] DRY_RUN — no DDS, no robot", flush=True)
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    ChannelFactoryInitialize(0, iface)
    _sport = SportClient()
    # 2 s, not the SDK-example 10 s: a command the firmware rejects (StopMove
    # returning 3104 was reproducible here) otherwise blocks the whole control
    # loop for a full 10 s per call, which is most of a step's wall-clock.
    _sport.SetTimeout(2.0)
    _sport.Init()
    _video = VideoClient()
    _video.SetTimeout(3.0)
    _video.Init()

    def _on_state(msg: Any) -> None:
        _state["yaw"] = msg.imu_state.rpy[2]
        _state["pos"] = (msg.position[0], msg.position[1])
        _state["mode"] = msg.mode
        _state["body_height"] = msg.body_height
        _state["t"] = time.time()

    _sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    _sub.Init(_on_state, 10)
    for _ in range(50):          # ~1 s for the first frame; feedback is required
        if "yaw" in _state:
            break
        time.sleep(0.02)
    print(f"[go2_host] robot clients up on {iface}; "
          f"feedback={'LIVE' if 'yaw' in _state else 'MISSING (open-loop fallback)'}",
          flush=True)


_sub: Any = None


def _rc(label: str, code: Any) -> None:
    """Log a non-zero SDK return code.

    Every SportClient method returns a status code and the first version of this
    file dropped all of them, which made a REJECTED command look exactly like a
    successful one — the reason a reset that never left mode 0 went unnoticed.
    """
    if code:
        print(f"[go2_host] WARN {label} returned code {code}", flush=True)


def _fresh_state() -> dict | None:
    """Feedback if it is live, else None. A stale subscriber is worse than no
    subscriber: it would close the loop on a frozen yaw and spin forever."""
    if DRY_RUN or "yaw" not in _state:
        return None
    return _state if (time.time() - _state.get("t", 0.0)) < 0.5 else None


def _await_state(timeout_s: float) -> dict | None:
    """Wait up to ``timeout_s`` for live feedback. Over WiFi the DDS feed drops
    out in bursts (measured 2026-07-20: 3 of 8 steps hit one), and an instant
    None check turned every burst into a lost step; a short wait rides most of
    them out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = _fresh_state()
        if st is not None:
            return st
        if DRY_RUN:
            return None
        time.sleep(0.05)
    return _fresh_state()


def _signed(v: float, lo: float, hi: float) -> float:
    """Magnitude clamped into [lo, hi], sign preserved."""
    return math.copysign(min(hi, max(lo, abs(v))), v)


def _drive_open(vx: float, vyaw: float, duration: float) -> None:
    """Open-loop fallback: hold a clamped setpoint for ``duration``, then stop.

    Only used when feedback is missing. Known to undershoot badly (~0.46x on
    turns) because the acceleration ramp eats the window — kept so a dead
    subscriber degrades to "moves too little" rather than "does not move".
    """
    vx, vyaw = _signed(vx, 0.0, MAX_VX), _signed(vyaw, 0.0, MAX_VYAW)
    if DRY_RUN:
        time.sleep(min(duration, 0.05))
        return
    period, deadline = 1.0 / CMD_HZ, time.time() + duration
    try:
        while time.time() < deadline:
            _sport.Move(vx, 0.0, vyaw)
            time.sleep(period)
    finally:
        _rc("StopMove", _sport.StopMove())
        time.sleep(SETTLE_S)


def _turn(delta_rad: float) -> dict:
    """Rotate by ``delta_rad`` using yaw feedback. Returns what actually happened.

    Proportional control on the remaining angle, with the yaw unwrapped
    incrementally so a pass through +/-pi does not read as a full reverse
    rotation. Reports the post-settle measurement, including coast-through after
    StopMove, so the caller never has to assume the command was obeyed.
    """
    _rc("StaticWalk", _sport.StaticWalk())  # re-assert 常规 — StopMove drops it
    st = _await_state(STALE_WAIT_S)
    if st is None:
        _drive_open(0.0, math.copysign(MAX_VYAW, delta_rad), abs(delta_rad) / MAX_VYAW)
        return {"closed_loop": False, "requested_deg": math.degrees(delta_rad)}

    period, deadline = 1.0 / CMD_HZ, time.time() + CTRL_TIMEOUT_S
    # aim short of the target; the post-StopMove coast covers the difference
    aim = delta_rad - math.copysign(min(YAW_COAST, abs(delta_rad) / 2), delta_rad)
    prev, turned = st["yaw"], 0.0
    exit_reason, stale_recoveries = "timeout", 0
    try:
        while time.time() < deadline:
            now = _fresh_state()
            if now is None:                       # feedback died mid-move
                # Same recovery as _forward: the dog is already stopping (no
                # re-issued Move), so wait the dropout out. The unwrap across
                # the gap stays valid because the dog holds still through it.
                _rc("StopMove", _sport.StopMove())
                t_pause = time.time()
                now = _await_state(STALE_WAIT_S)
                if now is None:
                    exit_reason = "feedback_stale"
                    break
                deadline += time.time() - t_pause
                stale_recoveries += 1
            step = (now["yaw"] - prev + math.pi) % (2 * math.pi) - math.pi
            turned += step
            prev = now["yaw"]
            remaining = aim - turned
            if abs(remaining) <= YAW_TOL:
                exit_reason = "reached"
                break
            _sport.Move(0.0, 0.0, _signed(K_YAW * remaining, MIN_VYAW, MAX_VYAW))
            time.sleep(period)
    finally:
        _rc("StopMove", _sport.StopMove())
        time.sleep(SETTLE_S)

    end = _fresh_state()
    if end is not None:                            # fold in the coast-through
        turned += (end["yaw"] - prev + math.pi) % (2 * math.pi) - math.pi
    return {"closed_loop": True,
            "requested_deg": round(math.degrees(delta_rad), 1),
            "actual_deg": round(math.degrees(turned), 1),
            "exit_reason": exit_reason, "stale_recoveries": stale_recoveries,
            "timed_out": exit_reason == "timeout"}


def _forward(dist_m: float) -> dict:
    """Advance ``dist_m`` using odometry feedback. Returns what actually happened.

    Progress is the displacement projected onto the heading at step start —
    signed, so backward slip subtracts. The first version used the Euclidean
    norm of the position delta, which cannot distinguish "walked forward 0.5 m"
    from "skidded sideways 0.5 m"; the projection can, and the rejected
    perpendicular component comes back as ``lateral_m`` so a crabbing gait shows
    up in the data instead of masquerading as a distance shortfall.
    """
    _rc("StaticWalk", _sport.StaticWalk())  # re-assert 常规 — StopMove drops it
    st = _await_state(STALE_WAIT_S)
    if st is None:
        _drive_open(MAX_VX, 0.0, dist_m / MAX_VX)
        return {"closed_loop": False, "requested_m": dist_m}

    period, deadline = 1.0 / CMD_HZ, time.time() + CTRL_TIMEOUT_S
    x0, y0 = st["pos"]
    hx, hy = math.cos(st["yaw"]), math.sin(st["yaw"])

    def _project(pos: tuple) -> tuple:
        dx, dy = pos[0] - x0, pos[1] - y0
        return dx * hx + dy * hy, -dx * hy + dy * hx

    moved = lateral = 0.0
    exit_reason, stale_recoveries = "timeout", 0
    try:
        while time.time() < deadline:
            now = _fresh_state()
            if now is None:
                # Feedback burst-dropout mid-step. Not commanding IS stopping
                # (Move is watchdog'd), so pause and wait for the feed instead
                # of abandoning the step — the dropouts were the entire scatter
                # in the 2026-07-20 calibration once the bias was fixed. The
                # wait does not consume motion budget: the dog is stationary,
                # so the deadline slides by however long the dropout lasted.
                _rc("StopMove", _sport.StopMove())
                t_pause = time.time()
                now = _await_state(STALE_WAIT_S)
                if now is None:
                    exit_reason = "feedback_stale"
                    break
                deadline += time.time() - t_pause
                stale_recoveries += 1
            moved, lateral = _project(now["pos"])
            # Break at target minus the coast allowance and aim the P-term one
            # POS_TOL past that, keeping the tolerance band beyond the break
            # point (band on the near side measured -7.5%; no coast allowance
            # measured +7.8% at 0.25 m — both systematic, both compensated).
            if moved >= dist_m - POS_COAST:
                exit_reason = "reached"
                break
            _sport.Move(_signed(K_X * (dist_m - POS_COAST + POS_TOL - moved),
                                MIN_VX, MAX_VX), 0.0, 0.0)
            time.sleep(period)
    finally:
        _rc("StopMove", _sport.StopMove())
        time.sleep(SETTLE_S)

    end = _fresh_state()
    if end is not None:
        moved, lateral = _project(end["pos"])
    return {"closed_loop": True, "requested_m": round(dist_m, 3),
            "actual_m": round(moved, 3), "lateral_m": round(lateral, 3),
            "exit_reason": exit_reason, "stale_recoveries": stale_recoveries,
            "timed_out": exit_reason == "timeout"}


# ── the three functions ──


def fn_reset(_inputs: dict) -> dict:
    """Idempotent ensure-live: stand the dog up and zero the counters.

    Does not choose an episode — episode selection is driver-side, exactly as
    in the habitat bridge where the driver owns run_episodes.py.
    """
    global _steps, _episode_over
    _steps, _episode_over = 0, False
    if DRY_RUN:
        return {"step_budget": STEP_BUDGET, "ready": True, "dry_run": True}
    _rc("StandUp", _sport.StandUp())
    time.sleep(1.5)
    _rc("BalanceStand", _sport.BalanceStand())
    time.sleep(0.5)
    # Pin the gait style so it is part of the episode's initial conditions.
    # StaticWalk = the app's 常规 style (mapped empirically 2026-07-20 night by
    # driving 90° under each style API while the operator read the app label;
    # ClassicWalk is the app's 经典, FreeWalk its 灵动). The operator chose 常规
    # for its statically-stable three-feet-down gait. Without pinning, the
    # style silently carries over from whatever ran last.
    _rc("StaticWalk", _sport.StaticWalk())
    time.sleep(0.3)
    # Wait for a genuinely fresh frame before returning. Measured 2026-07-20: the
    # first step after reset silently fell back to open-loop because the cached
    # state was still older than the staleness window at that moment.
    for _ in range(50):
        if _fresh_state() is not None:
            break
        time.sleep(0.02)
    st = _fresh_state()
    # Report the post-reset pose instead of asserting readiness: on the real dog
    # BalanceStand returned 0 while mode stayed 0, so "the call succeeded" and
    # "the robot is in a controllable state" are not the same claim.
    return {"step_budget": STEP_BUDGET, "ready": True,
            "feedback": st is not None,
            "mode": st.get("mode") if st else None,
            "body_height": round(st["body_height"], 3) if st else None}


def fn_step_discrete(inputs: dict) -> dict:
    """One discrete action. Same four output ports as env_habitat's step nodes.

    ``truncated`` is enforced here because a real robot has no
    MAX_EPISODE_STEPS to truncate on its behalf.
    """
    global _steps, _episode_over
    action = int(inputs.get("action", ACTION_FORWARD))
    if action not in ACTION_NAMES:
        return {"reward": 0.0, "terminated": False, "truncated": False,
                "info": {"error": f"invalid action {action}"}}
    if _episode_over:
        return {"reward": 0.0, "terminated": True, "truncated": False,
                "info": {"error": "episode already over"}}

    if action == ACTION_STOP:
        if not DRY_RUN:
            _rc("StopMove", _sport.StopMove())
        _episode_over = True
        return {"reward": 0.0, "terminated": True, "truncated": False,
                "info": {"action_name": "STOP", "step_count": _steps}}

    if DRY_RUN:
        measured: dict = {"closed_loop": False, "dry_run": True}
    elif action == ACTION_FORWARD:
        measured = _forward(STEP_M)
    else:
        sign = 1.0 if action == ACTION_TURN_LEFT else -1.0
        measured = _turn(sign * math.radians(TURN_DEG))

    _steps += 1
    truncated = _steps >= STEP_BUDGET
    _episode_over = truncated
    return {"reward": 0.0, "terminated": False, "truncated": truncated,
            "info": {"action_name": ACTION_NAMES[action], "step_count": _steps,
                     **measured}}


def fn_observe_egocentric(_inputs: dict) -> dict:
    """Head-camera frame. Pure read — never advances anything.

    No depth port: the Go2 head camera is RGB only. The habitat bridge's
    clearance readout is depth-derived and therefore simply absent here rather
    than faked from a monocular guess.
    """
    if DRY_RUN:
        return {"rgb": "", "format": "jpeg", "info": {"dry_run": True}}
    code, data = _video.GetImageSample()
    if code != 0:
        return {"rgb": "", "format": "jpeg", "info": {"error": f"video code {code}"}}
    return {"rgb": base64.b64encode(bytes(data)).decode(), "format": "jpeg",
            "info": {"step_count": _steps}}


FUNCTIONS = {
    "env_go2__reset": fn_reset,
    "env_go2__step_discrete": fn_step_discrete,
    "env_go2__observe_egocentric": fn_observe_egocentric,
}


# ── HTTP surface ──


class Handler(BaseHTTPRequestHandler):
    """Single-threaded by construction (HTTPServer, not ThreadingHTTPServer) —
    serializing requests is a feature here: two overlapping Move commands to one
    robot is exactly the race we do not want."""

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            # feedback is reported here, not just at startup: losing the state
            # subscriber silently downgrades every action to open-loop (which
            # undershoots badly) while the process stays up and healthy-looking.
            # Observed twice — once as a startup race, once as a mid-run WiFi
            # dropout — so a caller that only checks "status" learns nothing.
            st = _fresh_state()
            self._send(200, {
                "status": "ok",
                "dry_run": DRY_RUN,
                "feedback": st is not None,
                "closed_loop": st is not None,
                "turn_deg": TURN_DEG,
                "step_m": STEP_M,
                "max_vyaw": MAX_VYAW,
                "max_vx": MAX_VX,
                "mode": st.get("mode") if st else None,
            })
        elif self.path == "/manifest":
            self._send(200, {"name": "env_go2", "functions": list(FUNCTIONS)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/call/"):
            self._send(404, {"error": "not found"})
            return
        name = self.path[len("/call/"):]
        fn = FUNCTIONS.get(name)
        if fn is None:
            self._send(404, {"error": f"unknown function {name}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, {"outputs": fn(body.get("inputs") or {})})
        except Exception as exc:  # never leave the dog moving on an error path
            if _sport is not None and not DRY_RUN:
                try:
                    _sport.StopMove()
                except Exception:
                    pass
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[go2_host] {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", default=os.environ.get("GO2_IFACE", "enp6s0"),
                    help="network interface wired to the Go2 (192.168.123.x)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GO2_PORT", "9300")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    init_robot(args.iface)
    print(f"[go2_host] serving on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
