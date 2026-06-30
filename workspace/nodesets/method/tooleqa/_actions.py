"""ToolEQA discrete-string → free-pose JSON action adapter.

Translates the LLM's GoNextPointTool string output into the JSON action
shape expected by `env_hmeqa__step` (free-pose teleport). Mirrors the
upstream constants from `src/runs/eqa_modeling.py:go_next_point` (which
in turn lifts from explore-eqa's run_vlm_exp.py).

Action alphabet (verbatim from
`third_party/zz_just_for_refer/tooleqa/src/tools/go_next_point.py:11-15`):

    move_forward  — 0.5 meters forward in the agent's heading direction
    turn_left     — 45 degree left turn (no translation)
    turn_right    — 45 degree right turn (no translation)
    turn_around   — 180 degree turn (no translation)

env_hmeqa__step accepts JSON TEXT shape:
    {"position_normal": [x, y], "angle": float}
where position is in the normal frame (x, y planar; floor_height appended
inside env), and angle is yaw in radians.
"""

from __future__ import annotations

import json
import math
from typing import Any

# Upstream constants — see eqa_modeling.py:go_next_point branches
_FORWARD_STEP_M = 0.5
_TURN_RAD = math.radians(45.0)
_TURN_AROUND_RAD = math.radians(180.0)

VALID_DIRECTIONS = ("move_forward", "turn_left", "turn_right", "turn_around")


def discrete_to_pose_json(
    direction: str,
    current_position_normal: list[float] | tuple[float, ...],
    current_angle: float,
) -> str:
    """Translate one discrete command into env_hmeqa__step JSON.

    Args:
        direction: One of move_forward / turn_left / turn_right / turn_around.
        current_position_normal: Current 3-vector or 2-vector (x, y[, z]) in
            normal frame. Only x/y are used; floor_height is added inside env.
        current_angle: Current yaw in radians (normal frame).

    Returns:
        JSON string suitable for env_hmeqa__step's `action` input.
    """
    direction = (direction or "").strip().lower().replace("-", "_").replace(" ", "_")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction {direction!r}; expected one of {VALID_DIRECTIONS}")

    cur_x = float(current_position_normal[0])
    cur_y = float(current_position_normal[1])
    cur_angle = float(current_angle)

    if direction == "move_forward":
        # In the normal frame, +x is east, +y is north (planar). The agent's
        # heading is the yaw angle. Standard convention: forward = (cos, sin).
        new_x = cur_x + _FORWARD_STEP_M * math.cos(cur_angle)
        new_y = cur_y + _FORWARD_STEP_M * math.sin(cur_angle)
        new_angle = cur_angle
    elif direction == "turn_left":
        new_x, new_y = cur_x, cur_y
        new_angle = cur_angle + _TURN_RAD
    elif direction == "turn_right":
        new_x, new_y = cur_x, cur_y
        new_angle = cur_angle - _TURN_RAD
    else:  # turn_around
        new_x, new_y = cur_x, cur_y
        new_angle = cur_angle + _TURN_AROUND_RAD

    # Wrap angle to (-pi, pi] for cleanliness; env doesn't strictly require this.
    new_angle = math.atan2(math.sin(new_angle), math.cos(new_angle))

    return json.dumps({"position_normal": [new_x, new_y], "angle": new_angle})


def parse_action_buffer(buffer: dict[str, Any] | None) -> tuple[str, str, bool]:
    """Read toolbox's `_pending_*` buffer.

    Returns:
        (action_json_or_empty, final_answer_or_empty, done_flag)
    """
    if not buffer:
        return "", "", False
    action_json = str(buffer.get("action_json", "") or "")
    final_answer = str(buffer.get("final_answer", "") or "")
    done = bool(buffer.get("done", False)) or bool(final_answer)
    return action_json, final_answer, done
