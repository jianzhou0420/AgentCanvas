"""Nodeset-as-toolset — expose an auto_host nodeset to a ReAct agent as LLM tools.

The conceptual wrapper the harness rides on: a ``NodesetToolSet`` turns a
running auto_host nodeset (ADR-server-001 HTTP surface, ``POST /call/{fn}``)
into (a) a list of tool schemas the model layer declares to the LLM and (b) an
``execute(name, args)`` entry the environment routes parsed tool calls through.
Adding another nodeset later = another subclass (or config-declared instance)
plus a whitelist entry in the run condition; the invocation path stays this one.

``HabitatToolSet`` is the first instance: the agent-facing toolset of the
claude-SDK path (``beta-coding-agent/mcp_bridge.py``) ported verbatim — same
tool descriptions and input schemas, same clearance readout, same turn-budget
broadcast, same STOP confirmation gate, same live-spectating artifacts — minus
the MCP subprocess. Per-episode state that lived in bridge module globals
lives on the instance (one toolset per episode). ``check_equivalence.py``
verifies the port byte-for-byte against the bridge.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image as PILImage

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
