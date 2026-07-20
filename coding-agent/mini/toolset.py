"""Nodeset-as-toolset — expose an auto_host nodeset to a ReAct agent as LLM tools.

The conceptual wrapper the harness rides on: a ``NodesetToolSet`` turns a
running auto_host nodeset (ADR-server-001 HTTP surface, ``POST /call/{fn}``)
into (a) a list of tool schemas the model layer declares to the LLM and (b) an
``execute(name, args)`` entry the environment routes parsed tool calls through.
Adding another nodeset later = another subclass (or config-declared instance)
plus a whitelist entry in the run condition; the invocation path stays this one.

``HabitatToolSet`` is the first instance: the agent-facing toolset of the
claude-SDK path (``coding-agent/bridges/mcp_bridge.py``) ported verbatim — same
tool descriptions and input schemas, same clearance readout, same turn-budget
broadcast, same STOP confirmation gate, same live-spectating artifacts — minus
the MCP subprocess. Per-episode state that lived in bridge module globals
lives on the instance (one toolset per episode). ``check_equivalence.py``
verifies the port byte-for-byte against the bridge.
"""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

MAX_ACTIONS_PER_CALL = 50


def png_part(png: bytes) -> dict[str, Any]:
    """OpenAI-style image content part (litellm converts per provider)."""
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


@dataclass
class ToolResult:
    """One tool call's outcome.

    ``content`` is what the model sees (OpenAI content parts, images included);
    ``info`` is the machine-readable side for the environment (episode_over,
    end_reason, step counters) and the curated trajectory log.
    """

    content: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class NodesetToolSet:
    """Base: HTTP forwarding + tool registry. Subclasses register tools."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.calls_by_tool: dict[str, int] = {}
        self._handlers: dict[str, Callable[..., ToolResult]] = {}
        self._schemas: list[dict[str, Any]] = []

    def _register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., ToolResult],
    ) -> None:
        self._schemas.append(
            {"name": name, "description": description, "input_schema": input_schema}
        )
        self._handlers[name] = handler
        self.calls_by_tool.setdefault(name, 0)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Neutral {name, description, input_schema} dicts, declaration order."""
        return [dict(s) for s in self._schemas]

    def tool_names(self) -> list[str]:
        return [s["name"] for s in self._schemas]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            err = {"error": f"unknown tool '{name}'; available: {self.tool_names()}"}
            return ToolResult(content=[text_part(json.dumps(err))], info=err)
        self.calls_by_tool[name] += 1
        return handler(**args)

    def _call(self, function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{self.server_url}/call/{function_name}", json={"inputs": inputs}, timeout=300
        )
        resp.raise_for_status()
        return resp.json()["outputs"]


# ── habitat toolset — mcp_bridge.py port ──

# Tool descriptions: byte-identical to the bridge's (verified by
# check_equivalence.py). BARE strips the tuned mechanisms exactly as the
# bridge's HABITAT_BARE=1 does.

OBSERVE_DESC_BARE = (
    "Look through the robot's forward-facing camera. Returns the current "
    "egocentric RGB view. Pure read — does not advance the simulator or "
    "consume step budget."
)
OBSERVE_DESC_FULL = (
    "Look through the robot's forward-facing camera.\n\n"
    "Returns the current egocentric RGB view plus a clearance readout: "
    "distance in meters to the nearest obstacle in the left/center/right "
    "thirds of the view (10.0 = open, 10 m or more). If the goal object is "
    "centered ahead, clearance \"center\" is your true distance to it. Pure "
    "read — does not advance the simulator or consume step budget."
)
STEP_DESC_BASE = (
    "Execute a sequence of movement actions, in order.\n\n"
    "Actions: 0 = STOP (permanently ENDS the episode — issue it only when you "
    "believe the robot is within 3 meters of the goal), 1 = move forward "
    "0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.\n\n"
    "Executes sequentially and halts early if the episode ends. Returns how "
    "many actions ran, total steps taken, remaining budget, and whether the "
    "episode is over. The camera view changes after stepping — call observe() "
    "to see the result."
)
STEP_DESC_STOP_NOTE = (
    " Note: when plenty of budget remains, your FIRST STOP request is "
    "withheld pending a placement check — call step([0]) again to confirm "
    "and execute it."
)
# The bridge registers look_around via its raw docstring (FastMCP keeps the
# indentation), so the port must carry the same raw string.
LOOK_AROUND_DESC = (
    "Scan the surroundings in ONE call: rotates a full 360 degrees and\n"
    "    returns four labeled views — ahead (0°), right (+90°), behind (+180°),\n"
    "    left (+270°) — then restores the original heading exactly.\n"
    "\n"
    "    Costs 24 low-level turn steps from the step budget but only one tool\n"
    "    call. Use this at junctions and decision points instead of turning 15\n"
    "    degrees at a time; your position and final heading are unchanged.\n"
    "    "
)

OBSERVE_SCHEMA = {"properties": {}, "title": "observeArguments", "type": "object"}
LOOK_AROUND_SCHEMA = {"properties": {}, "title": "look_aroundArguments", "type": "object"}
STEP_SCHEMA = {
    "properties": {
        "actions": {"items": {"type": "integer"}, "title": "Actions", "type": "array"}
    },
    "required": ["actions"],
    "title": "stepArguments",
    "type": "object",
}


class HabitatToolSet(NodesetToolSet):
    """The env_habitat toolset — observe / step (+ look_around when not bare)."""

    def __init__(
        self,
        server_url: str,
        *,
        bare: bool = False,
        step_budget: int = 500,
        turn_budget: int = 0,
        pano_view_px: int = 0,
        live_dir: Path | None = None,
    ) -> None:
        super().__init__(server_url)
        self.bare = bare
        self.step_budget = step_budget
        self.turn_budget = turn_budget
        self.pano_view_px = pano_view_px
        self.live_dir = live_dir

        self.steps_taken = 0
        self.episode_over = False
        self.end_reason: str | None = None
        self._obs_count = 0
        self._tool_calls = 0
        self._stop_armed = False  # first budget-rich STOP request is withheld
        self._t0 = time.time()

        self._register(
            "observe",
            OBSERVE_DESC_BARE if bare else OBSERVE_DESC_FULL,
            OBSERVE_SCHEMA,
            self._tool_observe,
        )
        if not bare:
            self._register(
                "look_around", LOOK_AROUND_DESC, LOOK_AROUND_SCHEMA, self._tool_look_around
            )
        self._register(
            "step",
            STEP_DESC_BASE + ("" if bare else STEP_DESC_STOP_NOTE),
            STEP_SCHEMA,
            self._tool_step,
        )

    # ── tools ──

    def _tool_observe(self, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        outputs = self._call("env_habitat__observe_egocentric", {})
        png = base64.b64decode(outputs["rgb"])
        self._obs_count += 1
        self._live_frame(png)
        if self.bare:
            return ToolResult(content=[png_part(png)], info={"kind": "observe"})
        status = {"clearance_m": self._clearance_m(outputs.get("depth")), **self._budget_fields()}
        return ToolResult(
            content=[png_part(png), text_part(json.dumps(status))],
            info={"kind": "observe", **status},
        )

    def _tool_look_around(self, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            msg = f"episode already over ({self.end_reason}); no more steps possible"
            return ToolResult(content=[text_part(msg)], info={"kind": "look_around", "error": msg})

        content: list[dict[str, Any]] = []
        for label in ("ahead (0°)", "right (+90°)", "behind (+180°)", "left (+270°)"):
            outputs = self._call("env_habitat__observe_egocentric", {})
            png = base64.b64decode(outputs["rgb"])
            self._obs_count += 1
            self._live_frame(png)
            clearance = self._clearance_m(outputs.get("depth"))
            if clearance:
                label += (
                    f" — clearance m L/C/R: {clearance['left']}/{clearance['center']}"
                    f"/{clearance['right']}"
                )
            content.extend([text_part(label), png_part(self._downscale(png, self.pano_view_px))])
            # rotate 90° right toward the next view; the 4th rotation restores
            # the original heading (4 x 90° = 360°)
            for _ in range(6):
                outputs = self._call("env_habitat__step_discrete", {"action": 3})
                self.steps_taken += 1
                if outputs.get("terminated") or outputs.get("truncated"):
                    self.episode_over = True
                    self.end_reason = "step_budget_exhausted"
                    break
            if self.episode_over:
                break

        status = {
            "kind": "look_around",
            "steps_taken_total": self.steps_taken,
            "steps_remaining_approx": max(0, self.step_budget - self.steps_taken),
            "episode_over": self.episode_over,
            "heading_restored": not self.episode_over,
            **self._budget_fields(),
        }
        content.append(text_part(json.dumps({k: v for k, v in status.items() if k != "kind"})))
        self._live_log({"look_around": True, **status})
        if self.episode_over:
            status["end_reason"] = self.end_reason
        return ToolResult(content=content, info=status)

    def _tool_step(self, actions: Any = None, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            return self._json_result(
                {"error": f"episode already over ({self.end_reason}); no more steps possible"}
            )
        if not isinstance(actions, list) or not actions:
            return self._json_result({"error": "empty action list"})
        if len(actions) > MAX_ACTIONS_PER_CALL:
            return self._json_result(
                {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
            )
        bad = [a for a in actions if a not in (0, 1, 2, 3)]
        if bad:
            return self._json_result(
                {"error": f"invalid actions {bad}; valid: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT"}
            )

        # STOP confirmation gate: a budget-rich first STOP is withheld so the
        # agent verifies placement before committing (near-miss rush stops are
        # the dominant recoverable failure). Near budget exhaustion the gate is
        # open — salvage stops must never be blocked.
        remaining = max(0, self.turn_budget - self._tool_calls) if self.turn_budget > 0 else 0
        if 0 in actions and not self._stop_armed and self.turn_budget > 0 and remaining > 15:
            self._stop_armed = True
            prefix = actions[: actions.index(0)]
            withheld = {
                "stop_withheld": True,
                "message": (
                    "STOP not executed (first request). You still have "
                    f"{remaining} tool calls — verify placement before committing: "
                    "(1) look_around(); (2) re-read the instruction's FINAL clause "
                    "word by word; (3) apply the placement rules (several similar "
                    "spots -> the one farther along your travel direction; 'all "
                    "the way / to the end' -> keep going until center clearance "
                    "< 1.5 m; 'door/doorway' -> stand at the opening). Move if a "
                    "better spot exists, then call step([0]) again to execute it."
                ),
            }
            if prefix:
                # run the movement part of the call; only the STOP is withheld
                result = self._execute_actions(prefix)
                result.update(withheld)
                return self._json_result(result)
            return self._json_result({**withheld, **self._budget_fields()})
        if 0 in actions:
            self._stop_armed = True  # confirmed (or budget-critical) — let it through
        return self._json_result(self._execute_actions(actions))

    # ── ported helpers ──

    def _json_result(self, result: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[text_part(json.dumps(result))], info={"kind": "step", **result})

    def _execute_actions(self, actions: list[int]) -> dict[str, Any]:
        executed = 0
        for action in actions:
            outputs = self._call("env_habitat__step_discrete", {"action": action})
            executed += 1
            self.steps_taken += 1
            terminated = bool(outputs.get("terminated"))
            truncated = bool(outputs.get("truncated"))
            if terminated or truncated:
                self.episode_over = True
                if action == 0:
                    self.end_reason = "stop_called"
                elif truncated:
                    self.end_reason = "step_budget_exhausted"
                else:
                    self.end_reason = "terminated"
                break

        result = {
            "executed": executed,
            "requested": len(actions),
            "steps_taken_total": self.steps_taken,
            "steps_remaining_approx": max(0, self.step_budget - self.steps_taken),
            "episode_over": self.episode_over,
            "end_reason": self.end_reason,
            **self._budget_fields(),
        }
        if not self.episode_over and not self.bare:
            # post-move clearance (pure read; no sim advance) — metric feedback
            # for the approach phase without an extra observe round-trip
            outputs = self._call("env_habitat__observe_egocentric", {})
            clearance = self._clearance_m(outputs.get("depth"))
            if clearance:
                result["clearance_m"] = clearance
        self._live_log({"actions": actions, **result})
        return result

    def _budget_fields(self) -> dict[str, Any]:
        """Turn-budget broadcast — one tool call ≈ one harness turn (bridge parity:
        the binding limit is the agent's step_limit, reported here as env state)."""
        if self.turn_budget <= 0:
            return {}
        remaining = max(0, self.turn_budget - self._tool_calls)
        fields: dict[str, Any] = {
            "tool_calls_used": self._tool_calls,
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

    @staticmethod
    def _clearance_m(depth_field: dict[str, Any] | None) -> dict[str, float] | None:
        """Metric free-space readout from the raw depth frame (bridge port).

        Distance (m) to the nearest obstacle in the left/center/right thirds of
        the view, from an eye-level band (rows 40-65%) so floor and ceiling don't
        pollute it; 10th percentile per sector rides over stray pixels. VLN-CE
        depth is normalized [0,1] over 0-10 m, so 10.0 means "open, >= 10 m".
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
        band = arr[int(h * 0.40) : int(h * 0.65), :]
        sectors = {
            "left": band[:, : w // 3],
            "center": band[:, w // 3 : 2 * w // 3],
            "right": band[:, 2 * w // 3 :],
        }
        return {
            name: round(float(np.percentile(sector, 10)) * 10.0, 1)
            for name, sector in sectors.items()
        }

    @staticmethod
    def _downscale(png: bytes, side: int) -> bytes:
        """Shrink a PNG so four panorama views stay context-friendly.
        side=0 disables the shrink (views stay at native render resolution)."""
        img = PILImage.open(BytesIO(png))
        if not side or max(img.size) <= side:
            return png
        img = img.resize((side, side), PILImage.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _live_frame(self, png: bytes) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        (self.live_dir / f"obs_{self._obs_count:04d}_step{self.steps_taken:03d}.png").write_bytes(png)
        (self.live_dir / "latest.png").write_bytes(png)

    def _live_log(self, entry: dict[str, Any]) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        with (self.live_dir / "actions.log").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **entry}) + "\n")


# ── waypoint toolset — wp_bridge.py port ──
#
# In-process mirror of bridges/wp_bridge.py's action space: a
# depth-based predictor proposes ≤5 candidate waypoints, they are drawn as
# numbered circles on a [Left|Front|Right|Back] strip, and the agent picks one
# by number (goto) or stops. Tool descriptions + input schemas are byte-
# identical to the bridge (verified by check_equivalence.py), as are the
# geometry helpers and the annotated strip. Unlike HabitatToolSet this talks to
# TWO auto_hosts: the habitat env (self.server_url) and the waypoint predictor
# (self.wp_server_url) — hence the _call2 that carries a base + config, matching
# the bridge's own _call. Per-episode state lives on the instance (one toolset
# per episode), exactly as the bridge kept it in module globals.

# Descriptions: byte-identical to wp_bridge.py's _OBSERVE_DESC / _GOTO_DESC /
# _STOP_DESC (checked by check_equivalence.py).
WP_OBSERVE_DESC = (
    "Look around from where you stand. Returns a panoramic image of four "
    "views labeled Left / Front / Right / Back with numbered green circles "
    "marking the waypoints you can move to, plus a JSON listing each "
    "waypoint's direction, angle (degrees left of your heading; negative = "
    "right) and distance in meters. Pure read — does not advance the "
    "simulator or consume step budget."
)
WP_GOTO_DESC = (
    "Move to one numbered waypoint from the LATEST observe() result: the "
    "robot turns toward it and walks there. Moving invalidates the old "
    "numbers — call observe() again afterwards to see the new surroundings "
    "and waypoints. Returns steps consumed, remaining budget, and whether "
    "the episode is over."
)
WP_STOP_DESC = (
    "Permanently END the episode, declaring you have reached the goal. "
    "Issue it only when you believe the robot is within 3 meters of the "
    "instruction's endpoint — stopping is irreversible."
)

# FastMCP generates these from the tool signatures; the port hardcodes the
# exact shapes (title "<name>Arguments", "Waypoint" capitalized) so the schemas
# the model sees are byte-equal across paths.
WP_OBSERVE_SCHEMA = {"properties": {}, "title": "observeArguments", "type": "object"}
WP_GOTO_SCHEMA = {
    "properties": {"waypoint": {"title": "Waypoint", "type": "integer"}},
    "required": ["waypoint"],
    "title": "gotoArguments",
    "type": "object",
}
WP_STOP_SCHEMA = {"properties": {}, "title": "stopArguments", "type": "object"}

N_PANO_VIEWS = 12
# dir_id -> strip slot: render_panorama_rgbd's dir i faces i*30° counter-
# clockwise, so 3 = Left, 0 = Front, 9 = Right, 6 = Back — the VLN-MME
# [Left|Front|Right|Back] display order at zero extra render cost.
STRIP_DIRS = ((3, "Left"), (0, "Front"), (9, "Right"), (6, "Back"))


class WaypointToolSet(NodesetToolSet):
    """The env_habitat waypoint toolset — observe / goto / stop."""

    def __init__(
        self,
        server_url: str,
        *,
        wp_server_url: str,
        wp_max_moves: int = 40,
        predict_fn: str = "smartway_waypoint__predict",
        turn_budget: int = 0,
        pano_view_px: int = 384,
        live_dir: Path | None = None,
    ) -> None:
        super().__init__(server_url)
        self.wp_server_url = wp_server_url
        self.wp_max_moves = wp_max_moves
        self.predict_fn = predict_fn
        self.turn_budget = turn_budget
        self.pano_view_px = pano_view_px
        self.live_dir = live_dir

        self.steps_taken = 0
        self.episode_over = False
        self.end_reason: str | None = None
        self._obs_count = 0
        self._tool_calls = 0
        self._moves = 0  # waypoint moves executed (goto calls that ran)
        self._last_candidates: list[dict[str, float]] | None = None
        self._t0 = time.time()

        self._register("observe", WP_OBSERVE_DESC, WP_OBSERVE_SCHEMA, self._tool_observe)
        self._register("goto", WP_GOTO_DESC, WP_GOTO_SCHEMA, self._tool_goto)
        self._register("stop", WP_STOP_DESC, WP_STOP_SCHEMA, self._tool_stop)

    # ── two-server HTTP (env + predictor); mirrors wp_bridge._call ──

    def _call2(
        self, function_name: str, inputs: dict[str, Any],
        config: dict[str, Any] | None = None, base: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"inputs": inputs}
        if config:
            body["config"] = config
        resp = requests.post(
            f"{base or self.server_url}/call/{function_name}", json=body, timeout=600
        )
        resp.raise_for_status()
        return resp.json()["outputs"]

    # ── tools ──

    def _tool_observe(self, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            msg = f"episode already over ({self.end_reason}); no more moves possible"
            return ToolResult(content=[text_part(msg)], info={"kind": "observe", "error": msg})

        pano = self._call2(
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
        pred = self._call2(self.predict_fn, {"views": slim}, base=self.wp_server_url)
        self._last_candidates = self._normalize_candidates(pred.get("candidates"))

        png = self._annotate_strip(views, self._last_candidates, self.pano_view_px)
        self._obs_count += 1
        self._live_frame(png)

        status: dict[str, Any] = {
            "kind": "observe",
            "waypoints": {
                str(i + 1): {
                    "direction": self._direction_of(c["angle"]),
                    "angle_deg": round(math.degrees(self._norm_pi(c["angle"])), 1),
                    "distance_m": round(c["distance"], 2),
                }
                for i, c in enumerate(self._last_candidates)
            },
            "action_options": self._action_options(self._last_candidates),
            "steps_taken_total": self.steps_taken,
            **self._move_fields(),
            **self._budget_fields(),
        }
        if not self._last_candidates:
            status["note"] = (
                "no reachable waypoints predicted here; if you believe you are at "
                "the goal call stop(), otherwise observe() again after the next move"
            )
        self._live_log({"observe_wp": True, "num_waypoints": len(self._last_candidates)})
        public = {k: v for k, v in status.items() if k != "kind"}
        return ToolResult(content=[png_part(png), text_part(json.dumps(public))], info=status)

    def _tool_goto(self, waypoint: Any = None, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            return self._json_result(
                {"kind": "goto", "error": f"episode already over ({self.end_reason}); "
                                          "no more moves possible"})
        if self._last_candidates is None:
            return self._json_result(
                {"kind": "goto", "error": "no current waypoint set — call observe() first"})
        try:
            waypoint = int(waypoint)
        except (TypeError, ValueError):
            return self._json_result(
                {"kind": "goto", "error": f"invalid waypoint {waypoint!r}; expected an integer"})
        if not 1 <= waypoint <= len(self._last_candidates):
            return self._json_result({
                "kind": "goto",
                "error": (f"invalid waypoint {waypoint}; valid choices are 1-"
                          f"{len(self._last_candidates)} from the LATEST observe()"),
            })

        cand = self._last_candidates[waypoint - 1]
        outputs = self._call2(
            "env_habitat__step_hightolow",
            {"angle": cand["angle"], "distance": cand["distance"]},
        )
        info = outputs.get("info") or {}
        if isinstance(info.get("step_count"), (int, float)):
            self.steps_taken = int(info["step_count"])
        self._moves += 1
        terminated = bool(outputs.get("terminated"))
        truncated = bool(outputs.get("truncated"))
        if terminated or truncated:
            self.episode_over = True
            self.end_reason = "step_budget_exhausted" if truncated else "terminated"
        elif self._moves >= self.wp_max_moves:
            # decision-step budget spent — truncate like habitat's step cap
            self.episode_over = True
            self.end_reason = "wp_move_budget_exhausted"
        self._last_candidates = None  # position changed; numbers are stale

        result: dict[str, Any] = {
            "kind": "goto",
            "moved_to": waypoint,
            "direction": self._direction_of(cand["angle"]),
            "distance_m": round(cand["distance"], 2),
            "steps_taken_total": self.steps_taken,
            "episode_over": self.episode_over,
            "end_reason": self.end_reason,
            **self._move_fields(),
            **self._budget_fields(),
        }
        if not self.episode_over:
            result["note"] = "call observe() to see the new surroundings and waypoints"
        self._live_log({"goto": waypoint, "distance_m": result["distance_m"],
                        "steps_taken_total": self.steps_taken, "episode_over": self.episode_over})
        return self._json_result(result)

    def _tool_stop(self, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            return self._json_result(
                {"kind": "stop", "error": f"episode already over ({self.end_reason}); "
                                          "no more moves possible"})
        self._call2("env_habitat__step_discrete", {"action": 0})
        self.steps_taken += 1
        self.episode_over = True
        self.end_reason = "stop_called"
        result = {
            "kind": "stop",
            "stopped": True,
            "steps_taken_total": self.steps_taken,
            "episode_over": True,
            "end_reason": self.end_reason,
        }
        self._live_log(result)
        return self._json_result(result)

    # ── helpers ──

    def _json_result(self, result: dict[str, Any]) -> ToolResult:
        public = {k: v for k, v in result.items() if k != "kind"}
        return ToolResult(content=[text_part(json.dumps(public))], info=result)

    def _move_fields(self) -> dict[str, Any]:
        """Waypoint-move budget broadcast — the binding budget in wp mode."""
        remaining = max(0, self.wp_max_moves - self._moves)
        fields: dict[str, Any] = {"moves_used": self._moves, "moves_remaining": remaining}
        if 0 < remaining <= 3:
            fields["MOVE_WARNING"] = (
                f"Only {remaining} waypoint move(s) left before the episode ends. "
                "If you are at the goal, call stop() now; otherwise make them count."
            )
        return fields

    def _budget_fields(self) -> dict[str, Any]:
        """Turn-budget broadcast — verbatim contract from wp_bridge.py. Inert
        while turn_budget<=0, which is the case for every wp cell (bare=True)."""
        if self.turn_budget <= 0:
            return {}
        remaining = max(0, self.turn_budget - self._tool_calls)
        fields: dict[str, Any] = {
            "tool_calls_used": self._tool_calls,
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

    # ── geometry (verbatim from wp_bridge.py; asserted equal in check_equivalence) ──

    @staticmethod
    def _normalize_candidates(raw: Any) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        if not isinstance(raw, dict):
            return out
        for value in raw.values():
            if isinstance(value, dict) and "angle" in value and "distance" in value:
                out.append({"angle": float(value["angle"]), "distance": float(value["distance"])})
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                out.append({"angle": float(value[0]), "distance": float(value[1])})
        return out

    @staticmethod
    def _norm_pi(angle: float) -> float:
        """Normalize to [-pi, pi) — counter-clockwise positive (left)."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _direction_of(angle: float) -> str:
        a = angle % (2 * math.pi)
        if a <= math.pi / 4 or a >= 7 * math.pi / 4:
            return "Front"
        if a <= 3 * math.pi / 4:
            return "Left"
        if a <= 5 * math.pi / 4:
            return "Back"
        return "Right"

    @staticmethod
    def _action_options(candidates: list[dict[str, float]]) -> dict[str, list[int]]:
        options: dict[str, list[int]] = {"Left": [], "Front": [], "Right": [], "Back": []}
        for i, cand in enumerate(candidates):
            options[WaypointToolSet._direction_of(cand["angle"])].append(i + 1)
        return options

    @staticmethod
    def _font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            try:
                return ImageFont.load_default(size)
            except TypeError:
                return ImageFont.load_default()

    @staticmethod
    def _annotate_strip(views: list[dict[str, Any]],
                        candidates: list[dict[str, float]],
                        view_px: int = 384) -> bytes:
        """[Left|Front|Right|Back] hstack with numbered waypoint markers.

        Angle→x is the VLN-MME formula (vlnce_baselines/utils.py:216-232): for
        counter-clockwise θ normalized to [-pi, pi), ``x = w_single·(1.5 - 2θ/π)``
        — Front centers on tile 2, Left on tile 1, Right on tile 3, Back wraps
        across the strip ends onto tile 4. Verbatim port of wp_bridge._annotate_strip.
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
        label_font = WaypointToolSet._font(int(label_h * 0.7))
        for i, (_dir_id, label) in enumerate(STRIP_DIRS):
            bbox = draw.textbbox((0, 0), label, font=label_font)
            draw.text(
                ((i + 0.5) * w_single - (bbox[2] - bbox[0]) / 2,
                 (label_h - (bbox[3] - bbox[1])) / 2 - bbox[1]),
                label, fill=(0, 0, 0), font=label_font,
            )

        radius = max(12, h // 20)
        num_font = WaypointToolSet._font(int(radius * 1.15))
        y = label_h + h // 2
        for i, cand in enumerate(candidates):
            theta = WaypointToolSet._norm_pi(cand["angle"])
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

        if h > view_px:
            scale = view_px / h
            canvas = canvas.resize(
                (int(canvas.width * scale), int(canvas.height * scale)), PILImage.LANCZOS
            )
        buf = BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()

    def _live_frame(self, png: bytes) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        (self.live_dir / f"obs_{self._obs_count:04d}_step{self.steps_taken:03d}.png").write_bytes(png)
        (self.live_dir / "latest.png").write_bytes(png)

    def _live_log(self, entry: dict[str, Any]) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        with (self.live_dir / "actions.log").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **entry}) + "\n")
