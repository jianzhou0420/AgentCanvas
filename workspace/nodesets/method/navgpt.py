"""NavGPT nodeset — nodes for NavGPT-CE graphs.

ActionOutputFormatSkill: system prompt with action format rules
ParseActionNode:  regex-based extraction of FORWARD/LEFT/RIGHT/STOP from LLM text
FormatHistoryNode: reads action_history from graph_state, formats as prompt text

Load:   POST /api/components/nodesets/navgpt/load
Unload: POST /api/components/nodesets/navgpt/unload

last updated: 2026-03-31 23:00
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_ACTION_OUTPUT_FORMAT = """\
# Action Output Format

Strict formatting rules to ensure reliable action parsing.

## Rules

1. End your response with exactly ONE action keyword on its own line:
   FORWARD, LEFT, RIGHT, or STOP
2. Do NOT add text after the action keyword.
3. Do NOT output multiple actions — one action per reasoning step.
4. If uncertain, reason more before committing to an action.

## Correct Examples

Thought: I see the kitchen ahead and my sub-goal is to enter it.
FORWARD

Thought: The doorway is to my left, I need to turn.
LEFT

## Incorrect Examples

Do not output multiple actions, numeric codes, text after action, or bury the action in text."""

_ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
_ACTION_FROM_KEYWORD: dict[str, int] = {
    "STOP": 0,
    "FORWARD": 1,
    "LEFT": 2,
    "TURN_LEFT": 2,
    "RIGHT": 3,
    "TURN_RIGHT": 3,
}


# ═══════════════════════════════════════════════════════════════════════
# Action Output Format (system prompt skill)
# ═══════════════════════════════════════════════════════════════════════


class ActionOutputFormatSkill(BaseCanvasNode):
    """Strict formatting rules — end with FORWARD/LEFT/RIGHT/STOP on its own line."""

    node_type: ClassVar[str] = "navgpt__action_output_format"
    display_name: ClassVar[str] = "Action Output Format"
    description: ClassVar[str] = (
        "Strict formatting rules — end with FORWARD/LEFT/RIGHT/STOP on its own line"
    )
    category: ClassVar[str] = "skill"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "AlignLeft"
    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Skill guidance text"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        self._self_log("format_length", len(_ACTION_OUTPUT_FORMAT))
        return {"text": _ACTION_OUTPUT_FORMAT}


# ═══════════════════════════════════════════════════════════════════════
# Parse Action
# ═══════════════════════════════════════════════════════════════════════


class ParseActionNode(BaseCanvasNode):
    """Parse FORWARD/LEFT/RIGHT/STOP from LLM response text."""

    node_type: ClassVar[str] = "navgpt__parse_action"
    display_name: ClassVar[str] = "Parse Action"
    description: ClassVar[str] = "Extract discrete navigation action from LLM response text"
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "GitBranch"
    input_ports: ClassVar[list] = [
        PortDef("response", "TEXT", "LLM response text to parse"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("action", "ACTION", "Parsed action (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)"),
        PortDef("action_name", "TEXT", "Action keyword"),
        PortDef("thought", "TEXT", "Extracted thought/reasoning text"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import re

        response = str(inputs.get("response", ""))
        upper = response.upper()

        # Extract thought (text before the action keyword)
        thought = response.strip()
        thought_match = re.search(
            r"(?:Thought|Reasoning)\s*:\s*(.+?)(?:\n(?:Action|FORWARD|LEFT|RIGHT|STOP)|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Parse action — check last line first, then scan full text
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        action = 1  # default FORWARD
        if lines:
            last_upper = lines[-1].upper()
            for keyword, code in _ACTION_FROM_KEYWORD.items():
                if keyword in last_upper:
                    action = code
                    break
            else:
                # Fallback: scan full text
                if "STOP" in upper:
                    action = 0
                elif "LEFT" in upper:
                    action = 2
                elif "RIGHT" in upper:
                    action = 3
                elif "FORWARD" in upper:
                    action = 1

        action_name = _ACTION_NAMES.get(action, "FORWARD")

        # Persist to graph_state if available
        if ctx and hasattr(ctx, "graph_state") and ctx.graph_state:
            import json

            entry = json.dumps(
                {
                    "step": getattr(ctx, "step", 0),
                    "action": action,
                    "action_name": action_name,
                    "thought": thought[:120],
                }
            )
            ctx.graph_state.write("action_history", entry)
            ctx.graph_state.write("step", 1)

        self._self_log("parsed_action", action)
        self._self_log("action_name", action_name)
        self._self_log("thought", thought[:200] if thought else "")
        self._self_log("response_length", len(str(inputs.get("response", ""))))
        return {"action": action, "action_name": action_name, "thought": thought}


# ═══════════════════════════════════════════════════════════════════════
# Format History
# ═══════════════════════════════════════════════════════════════════════


class FormatHistoryNode(BaseCanvasNode):
    """Read action_history from graph_state and format into prompt-ready text.

    Reads directly from ``ctx.graph_state.read("action_history")`` — no
    data-edge input needed.  Wire a trigger port to control when it fires.
    """

    node_type: ClassVar[str] = "navgpt__format_history"
    display_name: ClassVar[str] = "Format History"
    description: ClassVar[str] = "Read action history from graph state and format as text"
    category: ClassVar[str] = "processing"
    kind: ClassVar[str] = "block"
    icon: ClassVar[str] = "ClipboardList"
    input_ports: ClassVar[list] = [
        # Pure trigger: any value fires the node (it reads history from graph
        # state, not from this port). ANY so a POSE / metrics / text trigger
        # all wire in without a false type mismatch.
        PortDef("trigger", "ANY", "Any data to trigger re-fire", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("history", "TEXT", "Formatted history text"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField(
                "max_entries", "slider", label="Max entries", default=15, min=5, max=50, step=5
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import json

        # Read from graph_state directly
        entries: list = []
        if ctx and hasattr(ctx, "graph_state") and ctx.graph_state:
            raw = ctx.graph_state.read("action_history")
            if isinstance(raw, list):
                # Accumulator stores strings — parse each JSON entry
                for item in raw:
                    if isinstance(item, str):
                        try:
                            entries.append(json.loads(item))
                        except (json.JSONDecodeError, TypeError):
                            entries.append({"action_name": item})
                    elif isinstance(item, dict):
                        entries.append(item)

        if not entries:
            return {"history": "(no actions yet)"}

        max_entries = int(self.config.get("max_entries", 15))
        recent = entries[-max_entries:]

        lines: list[str] = []
        if len(entries) > max_entries:
            lines.append(f"(...{len(entries) - max_entries} earlier steps omitted)")

        for entry in recent:
            if isinstance(entry, dict):
                step = entry.get("step", "?")
                name = entry.get("action_name", "?")
                thought = entry.get("thought", "")
                if len(thought) > 120:
                    thought = thought[:117] + "..."
                line = f"Step {step}: {name}"
                if thought:
                    line += f" — {thought}"
                lines.append(line)
            else:
                lines.append(str(entry))

        self._self_log("entry_count", len(entries))
        self._self_log("formatted_length", sum(len(s) for s in lines))
        return {"history": "\n".join(lines)}


# ═══════════════════════════════════════════════════════════════════════
# NodeSet
# ═══════════════════════════════════════════════════════════════════════


class NavgptNodeSet(BaseNodeSet):
    name = "navgpt"
    description = "NavGPT-CE utility nodes — action parsing and history formatting"

    def get_tools(self) -> list:
        return [
            ActionOutputFormatSkill(),
            ParseActionNode(),
            FormatHistoryNode(),
        ]
