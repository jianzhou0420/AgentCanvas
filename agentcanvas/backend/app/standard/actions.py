"""Standard action definitions — the discrete action space for VLN agents.

All nodes and handlers use these constants. Never hardcode action integers.
"""

# ── Action Constants ──
from __future__ import annotations

ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3

ACTION_NAMES: dict[int, str] = {
    ACTION_STOP: "STOP",
    ACTION_FORWARD: "FORWARD",
    ACTION_TURN_LEFT: "TURN_LEFT",
    ACTION_TURN_RIGHT: "TURN_RIGHT",
}

ACTION_FROM_NAME: dict[str, int] = {v: k for k, v in ACTION_NAMES.items()}


def parse_action_from_text(text: str) -> int:
    """Parse an action integer from free-form text (e.g., LLM response).

    Looks for action keywords in the text (case-insensitive).
    Returns ACTION_FORWARD as default if no match.
    """
    upper = text.upper()
    # Check in specificity order (TURN_LEFT before LEFT to avoid false matches)
    if "STOP" in upper:
        return ACTION_STOP
    if "TURN_LEFT" in upper or "LEFT" in upper:
        return ACTION_TURN_LEFT
    if "TURN_RIGHT" in upper or "RIGHT" in upper:
        return ACTION_TURN_RIGHT
    if "FORWARD" in upper:
        return ACTION_FORWARD
    return ACTION_FORWARD  # default
