from __future__ import annotations

"""VlaToolSet — the two-server tool surface a planner drives the VLA through.

The planner never sees a raw simulator. It sees two tools:

    execute(sub_instruction)  roll the VLA out on one clause, then hand back
                              the frames it stopped on plus telemetry
    finish()                  fire the env's STOP and end the episode

The load-bearing design decision is that ``execute`` does NOT let the policy end
the episode. Upstream's STOP token was trained on *whole* instructions; under a
sub-instruction its semantics are undefined — it may fire immediately (the short
clause looks already-satisfied) or never (it is waiting for a route's end). So
the rollout treats STOP as a *signal*: it stops the rollout and reports
``policy_stopped``, and the planner decides whether that means "clause done,
continue", "we're lost, recover", or "we've actually arrived, finish".

What comes back is chosen for a planner that is a VLM but has no odometry: the
last few front views (what the policy was looking at when it decided to stop)
plus the left/right views at the stopping pose, and symbols it cannot infer from
pixels — net displacement, action histogram, unparseable-generation count.

Servers (both AgentCanvas auto_hosts, ``POST /call/{fn}``):
    env_habitat      simulator, episodes, metrics
    policy_vla_nav   the 2B VLA

last updated: 2026-08-06
"""

import base64
import io
import math
from collections import Counter
from typing import Any

import requests

# render_panorama with n_views=4 yields yaw 0/90/180/270 counter-clockwise;
# +90 is left. Same convention wp_bridge uses at n_views=12 (dir 3 = Left).
DIR_FRONT, DIR_LEFT, DIR_BACK, DIR_RIGHT = 0, 1, 2, 3

# Upstream's evaluator rig (argparse defaults of
# evaluation_qwen3_vl_omega_current3_actionformer.py, which override the yaml).
RIG = {"rgb_resolution": "720x640", "rgb_hfov": "110", "rgb_height_m": "0.5"}

STOP_ACTION_ID = 0
# The harness's own actuator. VLN-CE discrete space: 0.25 m per FORWARD,
# 15° per turn — so a right angle is six turns.
DRIVE_IDS = {"MOVE_FORWARD": 1, "FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
FORWARD_STEP_M = 0.25
TURN_ANGLE_DEG = 15


def _sample_idx(n: int, k: int) -> list[int]:
    """Indices of k uniformly spaced frames, always keeping first and last."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    seen, out = set(), []
    for i in range(k):
        j = round(i * (n - 1) / (k - 1))
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def _sample(frames: list, k: int) -> list:
    return [frames[i] for i in _sample_idx(len(frames), k)]


class VlaToolSet:
    """One episode's worth of tool state. Construct fresh per episode."""

    def __init__(
        self,
        env_url: str,
        policy_url: str,
        *,
        max_steps: int = 50,
        leg_frames: int = 3,
        path_frames: int = 6,
        frame_px: int = 512,
        step_budget: int = 400,
        rig: dict[str, str] | None = None,
        observe_at_end: bool = False,
        back_view: bool = True,
    ) -> None:
        self.env_url = env_url
        self.policy_url = policy_url
        self.max_steps = max_steps
        self.leg_frames = leg_frames
        self.path_frames = path_frames
        self.frame_px = frame_px
        self.step_budget = step_budget
        # A rollout that ends on the step cap took its last action *after* its
        # last observation, so the "current" views are one step stale. Arms that
        # show the caller a current three-view re-observe once at the end.
        self.observe_at_end = observe_at_end
        # Ablation axis: dropping the back view restores the 270-degree
        # blind spot the earlier arms judged arrival through.
        self.back_view = back_view
        # The sensor rig this episode runs under. Defaults to the upstream
        # evaluator's; overridable so the same arm can be re-measured on a
        # different camera without touching any other code path.
        self.rig = dict(rig or RIG)

        # Per-episode accounting
        self.steps_taken = 0
        self.episode_over = False
        self.end_reason: str | None = None
        self.calls: list[dict[str, Any]] = []
        # Every forward view the policy saw this episode, in order. The planner
        # judges "did this path actually carry out what I asked" against the
        # traversed route, not against the single frame it stopped on — a robot
        # can stop somewhere that looks right having got there the wrong way.
        self.trail: list[str] = []
        # The action chosen at each of those views, same indexing. Lets a caller
        # label a sampled route frame with what the robot did there, across the
        # whole episode rather than just the current dispatch.
        self.action_trail: list[str] = []
        # Four separate step ledgers. One counter conflating them made
        # `dispatches=3, turns=1` rows that nobody could read, and hid how much
        # of an episode's budget verification was spending.
        self.policy_env_steps = 0        # steps the 2B policy caused
        self.harness_drive_steps = 0     # steps the harness drove
        self.verification_steps = 0      # of those, spent during verification
        self.render_only_observations = 0  # panoramas: no env step at all
        self._verifying = False          # tags drive steps to the right ledger

    # ── plumbing ──────────────────────────────────────────────────────

    def _call(self, base: str, fn: str, inputs=None, config=None, timeout=900):
        body: dict[str, Any] = {"inputs": inputs or {}}
        if config:
            body["config"] = config
        r = requests.post(f"{base}/call/{fn}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["outputs"]

    def panel_field(self, name: str, value: Any) -> None:
        requests.post(
            f"{self.env_url}/env-panel/field/{name}", json={"value": value}, timeout=600
        ).raise_for_status()

    def panel_action(self, name: str) -> None:
        requests.post(
            f"{self.env_url}/env-panel/action/{name}", json={"params": {}}, timeout=600
        ).raise_for_status()

    # ── observation ───────────────────────────────────────────────────

    def _three_views(self) -> tuple[str, str, str]:
        """front / left / right at the current pose, as base64 PNG.

        One panorama render at a fixed position — geometrically identical to
        upstream's three fixed sensors, because the sensor offset [0, h, 0] is
        purely vertical, so rotating the agent's yaw and rotating the sensor's
        yaw give the same camera extrinsics.
        """
        out = self._call(
            self.env_url,
            "env_habitat__observe_panorama",
            {"trigger": "vla"},
            {"representation": "views_rgbd", "n_views": 4},
        )
        views = {v["dir_id"]: v for v in out.get("views", [])}
        missing = [d for d in (DIR_FRONT, DIR_LEFT, DIR_RIGHT) if d not in views]
        if missing:
            raise RuntimeError(f"panorama missing dirs {missing}")
        return (
            views[DIR_FRONT]["rgb_base64"],
            views[DIR_LEFT]["rgb_base64"],
            views[DIR_RIGHT]["rgb_base64"],
        )

    def _panorama(self, n_views: int) -> list[dict[str, Any]]:
        """n evenly spaced views at the current pose, ordered counter-clockwise
        from straight ahead. Costs no env steps and does not move the robot —
        `observe_panorama` renders, it does not act. Looking around is free.
        """
        out = self._call(
            self.env_url,
            "env_habitat__observe_panorama",
            {"trigger": "vla"},
            {"representation": "views_rgbd", "n_views": n_views},
        )
        self.render_only_observations += 1
        views = sorted(out.get("views", []), key=lambda v: v["dir_id"])
        if len(views) < n_views:
            raise RuntimeError(f"panorama returned {len(views)} of {n_views} views")
        return views

    def probe_distance(self) -> float | None:
        """Current geodesic distance to the goal — INSTRUMENTATION ONLY.

        ``env_habitat__evaluate`` reads habitat's measurement cache and does not
        step the simulator, so this is free and side-effect free. It is oracle
        information: it must never reach the model, and nothing in the control
        flow may branch on it. It exists so a question like "did that correction
        move the robot closer or further from the goal" has an answer instead of
        a guess — which is exactly the question the first verification-step
        analysis could not settle.
        """
        try:
            m = self._call(self.env_url, "env_habitat__evaluate",
                           {"trigger": "probe"}).get("metrics", {})
            d = m.get("distance_to_goal")
            return float(d) if d is not None else None
        except Exception:
            return None

    @staticmethod
    def dhash(b64: str, size: int = 8) -> int | None:
        """64-bit difference hash of a frame. Local, free, no model.

        The stall detector needs to tell "turned and saw something new" from
        "the view did not change", and net displacement cannot: a pure rotation
        moves zero metres whether it revealed a new corridor or spun against a
        wall. Habitat does not surface collisions through step_discrete, so
        frame similarity is the available substitute.
        """
        try:
            from PIL import Image

            img = (Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
                   .resize((size + 1, size), Image.Resampling.LANCZOS))
            px = list(img.getdata())
            bits = 0
            for r in range(size):
                row = px[r * (size + 1):(r + 1) * (size + 1)]
                for c in range(size):
                    bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
            return bits
        except Exception:
            return None

    @staticmethod
    def hamming(a: int | None, b: int | None) -> int | None:
        return None if (a is None or b is None) else bin(a ^ b).count("1")

    def _pose(self) -> list[float] | None:
        try:
            out = self._call(self.env_url, "env_habitat__observe_camera_pose", {})
            pos = out.get("position")
            return [float(v) for v in pos] if pos else None
        except Exception:
            return None

    def _shrink(self, b64: str) -> str:
        """Downscale a frame before it goes to the planner — the rig renders
        720x640, which is more resolution than a judgment call needs and costs
        image tokens on every planner turn."""
        if not self.frame_px:
            return b64
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            if max(img.size) <= self.frame_px:
                return b64
            scale = self.frame_px / max(img.size)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS,
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return b64

    # ── episode lifecycle ─────────────────────────────────────────────

    def reset_episode(self, index: int) -> dict[str, Any]:
        self.panel_field("episode_index", index)
        self.panel_action("play")
        info = self._call(self.env_url, "env_habitat__reset", {"trigger": index}, self.rig)
        self._call(
            self.policy_url,
            "policy_vla_nav__reset",
            {"instruction": info.get("instruction") or ""},
        )
        self.steps_taken = 0
        self.episode_over = False
        self.end_reason = None
        self.calls = []
        self.trail = []
        self.action_trail = []
        self.policy_env_steps = 0
        self.harness_drive_steps = 0
        self.verification_steps = 0
        self.render_only_observations = 0
        self._verifying = False
        return info

    def metrics(self) -> dict[str, Any]:
        # env_habitat__evaluate nests task metrics under "metrics"; only
        # success/spl are echoed at the top level (and as TEXT).
        return self._call(self.env_url, "env_habitat__evaluate", {"trigger": "eval"}).get(
            "metrics", {}
        )

    # ── the two tools ─────────────────────────────────────────────────

    def execute(self, sub_instruction: str) -> dict[str, Any]:
        """Roll the VLA out on one clause. Never ends the episode.

        Returns telemetry plus the frames the policy stopped on. The policy's
        STOP breaks the rollout and is reported as ``policy_stopped`` — the
        decision about what it means belongs to the caller.
        """
        if self.episode_over:
            return {"error": f"episode already over ({self.end_reason})"}

        # Validate emptiness but pass the string through UNMODIFIED — the
        # dataset's instructions carry a trailing space, and a control arm that
        # silently strips it is no longer byte-identical to the policy-alone
        # baseline it is supposed to be compared against.
        if not (sub_instruction or "").strip():
            return {"error": "empty sub_instruction"}

        start_pose = self._pose()
        steps_at_start = self.steps_taken
        front_history: list[str] = []
        actions: list[str] = []
        unrecognized = 0
        stop_reason = "budget"
        front = left = right = back = None

        for _ in range(self.max_steps):
            front, left, right = self._three_views()
            act = self._call(
                self.policy_url,
                "policy_vla_nav__act",
                {
                    "rgb_front": front,
                    "rgb_left": left,
                    "rgb_right": right,
                    "instruction": sub_instruction,
                },
            )
            actions.append(act["action"])
            front_history.append(front)
            self.trail.append(front)
            self.action_trail.append(act["action"])
            if not act.get("recognized", True):
                unrecognized += 1

            if act["stop"]:
                # Signal, not control flow — the episode stays alive.
                stop_reason = "policy_stop"
                break

            out = self._call(
                self.env_url,
                "env_habitat__step_discrete",
                {"action": int(act["action_id"])},
            )
            self.steps_taken += 1
            self.policy_env_steps += 1
            if out.get("terminated") or out.get("truncated"):
                self.episode_over = True
                self.end_reason = "env_budget_exhausted"
                stop_reason = "env_done"
                break
            if self.steps_taken >= self.step_budget:
                self.episode_over = True
                self.end_reason = "step_budget_exhausted"
                stop_reason = "env_done"
                break

        if self.observe_at_end and not self.episode_over:
            # One extra render so "current" really is current — after a budget
            # stop the robot has moved since `front/left/right` were taken —
            # and four-way, so the caller is not judging arrival through a
            # 270-degree blind spot. Free: a render, not a move.
            by = {v["dir_id"]: v["rgb_base64"] for v in self._panorama(4)}
            front, left, back, right = (by[DIR_FRONT], by[DIR_LEFT],
                                        by[DIR_BACK], by[DIR_RIGHT])

        end_pose = self._pose()
        net = None
        if start_pose and end_pose:
            net = round(math.dist(start_pose, end_pose), 2)

        hist = dict(Counter(actions))
        # STOP is a decision, not a step: it moves nothing and the env never
        # sees it. Counting it in `steps_used` while `steps_taken_total`
        # excludes it gave two different meanings to the word "step" in one
        # telemetry dict, off by one exactly when the policy stopped itself.
        env_actions = [a for a in actions if a != "STOP"]
        result = {
            "sub_instruction": sub_instruction,
            "stop_reason": stop_reason,
            "policy_stopped": stop_reason == "policy_stop",
            "steps_used": len(env_actions),
            "generations": len(actions),
            "net_displacement_m": net,
            "action_hist": hist,
            "unrecognized": unrecognized,
            "steps_at_start": steps_at_start,
            "steps_taken_total": self.steps_taken,
            "steps_remaining": max(0, self.step_budget - self.steps_taken),
            "episode_over": self.episode_over,
        }
        # Deterministic guards the planner shouldn't have to infer from pixels.
        flags = []
        if net is not None and net < 0.3 and len(actions) >= 8:
            flags.append("no_net_progress")
        if len(hist) == 1 and len(actions) >= 8 and "MOVE_FORWARD" not in hist:
            flags.append("spinning_in_place")
        if unrecognized:
            flags.append("unparseable_generations")
        result["flags"] = flags

        # Three views of what just happened, at three timescales:
        #   _frames_leg   what THIS dispatch did, start → middle → end
        #   _frame_l/r    orientation at the stopping pose
        #   _frames_path  the whole episode so far, uniformly sampled — this is
        #                 what the planner checks intent against
        self.calls.append({k: v for k, v in result.items()})
        # Step index of each leg frame, so a caller can pair every frame with
        # the motor action the policy chose while looking at it.
        return self._bundle(result, front_history, actions, left, right, back=back)

    def _bundle(self, result: dict[str, Any], front_history: list[str],
                actions: list[str], left, right, front=None, back=None) -> dict[str, Any]:
        leg_idx = _sample_idx(len(front_history), self.leg_frames)
        # The whole episode's route, uniformly sampled. Each sampled frame keeps
        # its step index and the action taken there, so a caller can show the
        # traversed trajectory as "step k of N, robot went forward" rather than
        # as an unanchored pile of pictures.
        path_idx = _sample_idx(len(self.trail), self.path_frames)
        if front is None and front_history:
            front = front_history[-1]
        return {
            **result,
            "_frames_leg": [self._shrink(front_history[i]) for i in leg_idx],
            "_leg_frame_steps": leg_idx,
            "_actions": actions,
            "_frame_front": self._shrink(front) if front else None,
            "_frame_left": self._shrink(left) if left else None,
            "_frame_back": self._shrink(back) if (back and self.back_view) else None,
            "_frame_right": self._shrink(right) if right else None,
            "_frames_path": [self._shrink(self.trail[i]) for i in path_idx],
            "_path_steps": path_idx,
            "_path_actions": [self.action_trail[i] if i < len(self.action_trail) else "?"
                              for i in path_idx],
            "_path_len": len(self.trail),
            "_frame_hash": self.dhash(front) if front else None,
            "_ledger": {"policy_env_steps": self.policy_env_steps,
                        "harness_drive_steps": self.harness_drive_steps,
                        "verification_steps": self.verification_steps,
                        "render_only_observations": self.render_only_observations},
        }

    def current_views(self) -> dict[str, Any]:
        """Front / left / back / right at the current pose.

        Four, not three. A robot standing at a goal it has just walked to has
        the thing it came for *behind* it as often as in front, and a planner
        given a 270-degree blind spot cannot tell "the landmark is not here"
        from "the landmark is not in these three frames" — which is precisely
        the mistake that has cost this harness finished episodes. The back view
        is free: the panorama is a render, not a move.
        """
        views = self._panorama(4)
        by = {v["dir_id"]: v["rgb_base64"] for v in views}
        front = by[DIR_FRONT]
        return {"_frames_leg": [self._shrink(front)],
                "_leg_frame_steps": [0], "_actions": [],
                "_frame_front": self._shrink(front),
                "_frame_left": self._shrink(by[DIR_LEFT]),
                "_frame_back": self._shrink(by[DIR_BACK]) if self.back_view else None,
                "_frame_right": self._shrink(by[DIR_RIGHT])}

    def look_around(self, n_views: int = 8) -> dict[str, Any]:
        """A full 360-degree sweep at the current pose. Costs nothing.

        No env steps, no pose change, no distance to the goal given up. This is
        what "I cannot see the landmark" should escalate to — never a turn, and
        never a verdict.
        """
        views = self._panorama(n_views)
        step = 360 // n_views
        out: dict[str, Any] = {"_sweep": [], "_sweep_deg": []}
        for v in views:
            deg = (v["dir_id"] * step) % 360
            out["_sweep"].append(self._shrink(v["rgb_base64"]))
            out["_sweep_deg"].append(deg)
        return out

    def verifying(self, on: bool) -> None:
        """Tag subsequent drive steps as verification spend."""
        self._verifying = bool(on)

    def drive(self, actions: list[str], max_actions: int = 12) -> dict[str, Any]:
        """The harness drives the robot directly, bypassing the policy.

        There are poses where this policy will not move for any wording — ep3
        of the stateful run spent ten dispatches and ten phrasings at a standstill
        while the baseline, holding the whole instruction, simply walked on. When
        the actuator is wedged, asking it more politely is not a recovery. This
        is: the outer agent takes the wheel for a few steps, changes the pose,
        and hands control back.
        """
        if self.episode_over:
            return {"error": f"episode already over ({self.end_reason})"}

        seq = [str(a).upper().strip() for a in (actions or [])][:max_actions]
        seq = [a for a in seq if a in DRIVE_IDS]
        if not seq:
            return {"error": "no valid actions — use MOVE_FORWARD / TURN_LEFT / TURN_RIGHT"}

        start_pose = self._pose()
        steps_at_start = self.steps_taken
        front_history: list[str] = []
        taken: list[str] = []
        left = right = None
        stop_reason = "driven"

        for name in seq:
            front, left, right = self._three_views()
            front_history.append(front)
            self.trail.append(front)
            out = self._call(self.env_url, "env_habitat__step_discrete",
                             {"action": DRIVE_IDS[name]})
            self.steps_taken += 1
            self.harness_drive_steps += 1
            if self._verifying:
                self.verification_steps += 1
            taken.append("MOVE_FORWARD" if name in ("FORWARD", "MOVE_FORWARD") else name)
            self.action_trail.append(taken[-1])
            if out.get("terminated") or out.get("truncated"):
                self.episode_over = True
                self.end_reason = "env_budget_exhausted"
                stop_reason = "env_done"
                break
            if self.steps_taken >= self.step_budget:
                self.episode_over = True
                self.end_reason = "step_budget_exhausted"
                stop_reason = "env_done"
                break

        back = None
        if left is None or (self.observe_at_end and not self.episode_over):
            if self.observe_at_end and not self.episode_over:
                by = {v["dir_id"]: v["rgb_base64"] for v in self._panorama(4)}
                front, left, back, right = (by[DIR_FRONT], by[DIR_LEFT],
                                            by[DIR_BACK], by[DIR_RIGHT])
            else:
                front, left, right = self._three_views()
            front_history.append(front)

        end_pose = self._pose()
        net = round(math.dist(start_pose, end_pose), 2) if (start_pose and end_pose) else None
        result = {
            "driver": "harness",
            "sub_instruction": " ".join(taken),
            "stop_reason": stop_reason,
            "policy_stopped": False,
            "steps_used": len(taken),
            "net_displacement_m": net,
            "action_hist": dict(Counter(taken)),
            "unrecognized": 0,
            "steps_at_start": steps_at_start,
            "steps_taken_total": self.steps_taken,
            "steps_remaining": max(0, self.step_budget - self.steps_taken),
            "episode_over": self.episode_over,
            "flags": [],
        }
        self.calls.append(dict(result))
        return self._bundle(result, front_history, taken, left, right, back=back)

    def finish(self) -> dict[str, Any]:
        """Declare arrival — fires the env's STOP. Irreversible."""
        if self.episode_over:
            return {"error": f"episode already over ({self.end_reason})"}
        self._call(
            self.env_url, "env_habitat__step_discrete", {"action": STOP_ACTION_ID}
        )
        self.steps_taken += 1
        self.episode_over = True
        self.end_reason = "planner_stop"
        return {"stopped": True, "steps_taken_total": self.steps_taken}
