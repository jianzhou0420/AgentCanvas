from __future__ import annotations

"""Open-Nav method nodeset.

Faithful AgentCanvas port of:

    Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in
    Continuous Environment with Open-Source LLMs
    Qiao, Lyu, Wang, Wang, Li, Zhang, Tan, Wu — ICRA 2025
    arXiv: https://arxiv.org/abs/2409.18794
    Upstream: https://github.com/YanyuanQiao/Open-Nav @ 3a8dcef (MIT)
    Re-fetch: workspace/nodesets/_upstream/open-nav/fetch_upstream.sh

This nodeset implements the **method-side** of Open-Nav: prompts, parsers,
ensemble + fusion + tie-break logic, and history accumulation. It does
NOT touch the simulator (use ``env_habitat`` for that), the waypoint
predictor (use ``env_habitat_opennav_waypoint``), or the scene perception
models RAM/SpatialBot (use ``env_habitat_opennav_perception``).

All prompts and parsers below are copied character-for-character from the
reference implementation; each constant cites the source file:line it was
copied from. The instruction-parse cache (``cache_files/{dataset}/...``)
from the reference is intentionally **dropped** — every episode re-runs
Stage 1 (action detection + landmark detection). This is a deliberate
faithfulness/simplicity tradeoff approved at the ralplan checkpoint.

Companion files:

    workspace/nodesets/method/opennav_waypoint/     — frozen TRM_net
    workspace/nodesets/method/opennav_perception.py   — RAM + SpatialBot
    workspace/graphs/opennav_habitat.json             — wired graph

last updated: 2026-04-15
"""

import json
import logging
import random
import re
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

log = logging.getLogger("agentcanvas.opennav")


# ═══════════════════════════════════════════════════════════════════════
# Verbatim prompt templates — copied character-for-character from
# Open-Nav/vlnce_baselines/common/navigator/prompts.py
# ═══════════════════════════════════════════════════════════════════════

# prompts.py:2-5 — Stage 1a
_ACTION_DETECTION_SYSTEM = "You are an action decomposition expert. Your task is to detect all actions in the given navigation instruction. You need to ensure the integrity of each action. Your answer must consist ONLY of a series of labled action phrases without begin sentence."
_ACTION_DETECTION_USER = 'Can you decompose actions in the instruction "{}"? Actions: '

# prompts.py:8-11 — Stage 1b
_LANDMARK_DETECTION_SYSTEM = "You are a landmark extraction expert. Your task is to detect all landmarks in the given navigation instruction. You need to ensure the integrity of each landmarks. Your answer must consist ONLY of a series of labled landmark phrases without other sentences."
_LANDMARK_DETECTION_USER = 'Can you extract landmarks in the instruction "{}"? Landmarks: '

# prompts.py:18-22 — history compression (per-step observation summary)
_OBSERVATION_SUMMARY_SYSTEM = "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible.                                             You ONLY need to summarize in a single paragraph."
_OBSERVATION_SUMMARY_USER = 'Given Environment Description "{}", Summarization:'

# prompts.py:25-29 — history compression (per-step thought summary)
_THOUGHT_SUMMARY_SYSTEM = 'You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible.                                             You ONLY need to summarize the what actions you did and what landmarks you passed in "Thought" using a single paragraph. Do NOT include Direction information. '
_THOUGHT_SUMMARY_USER = 'Given Thought Process "{}", Summarization:'

# prompts.py:32-43 — Stage 2
_COMPLETION_ESTIMATION_SYSTEM = 'You are a completion estimation expert. Your task is to estimate what actions in the instruction have been executed based on navigation history and landmarks.                 All actions in the instruction are given following the temporal order. Your answer includes two parts: "Thought" and "Executed Actions". You need to use "Thought" and "Executed Actions" without any other symbols.                 In the "Thought", you must follow procedures to analyze as detailed as possible what actions have been executed:                 (1) What given landmarks of actions have appeared in the navigation history?                 (2) Analyze the direction change at each step in the navigation history.                 (3) Estimate each action in the instruction based on each step in the navigation history to check their completion.                 (4) You must estimate actions in order. This means that if action 1 is not completed, you can not completed actions 2.                 In the "Executed Actions", you must only write down actions that have been executed without other words.                 You must strictly refer original actions in the given instruction to estimate.'
_COMPLETION_ESTIMATION_USER = 'Given Navigation History "{}" and Landmarks in the instruction "{}", estimate what actions in instruction "{}" have been executed.'

# prompts.py:46-75 — Stage 3 (decision)
_NAVIGATOR_SYSTEM = 'You are a navigation agent who follows instruction to move in an indoor environment with the least action steps.             I will give you one instruction and tell you landmarks. I will also give you navigation history and estimation of executed actions for reference.             You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you.             Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint.             In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0             Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you.             If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position.             You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations.             If you feel struggling to find the landmark or execute the action, you can try to execute the subsequent action and find the subsequent landmark.             Your answer includes two parts: "Thought" and "Prediction". In the "Thought", you should think as detailed as possible following procedures:             (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after "Prediction" without any other words or characters .             (2) Check whether the latest executed action has been completed by comparing current environment and landmark in the latest executed action.             (3) Determine the action you should execute and landmark you should reach now. If the latest executed action have not been completed,             you should continue to execute it. Otherwise, you should execute the next action in the given instruction.             (4) Analyze which direction in the current environment is most suitable to execute the action you decide and explain your reason.             (5) Predict moving to which direction viewpoint based on your thought process.             (6) The "Thought" you predicted should be a single paragraph.             (7) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the "Prediction".             (8) If you want to make a left turn, you usually need to select a viewpoint ID between 1 and 5. If you want to make a right turn, you usually need to select a viewpoint ID between 7 and 11. However, the viewpoint ID you predict must be within the Current Environment.            (9) Your output after "Prediction" must be one of the number in Candidate Viewpoint IDs List without any other words.             Then, please make decision on the next viewpoint in the "Prediction".             Your decision is very important, must make it very carefully.             You need to double check the output in "Prediction:". The output must be in the Candidate Viewpoint IDs without any other words.             You also need to double check the output in "Thought". The output must be a single paragraph'
_NAVIGATOR_USER = 'Candidate Viewpoint IDs List: [{}] Step {} Instruction: {} ({}) Landmarks: {} Navigation History: {}             Estimation of Executed Actions: {} Current Environment: {} -> Thought: ... Prediction: ...             Your output after "Prediction" must be one of the number in Candidate Viewpoint IDs List without any other words.             Your output after "Thought" must be a single paragraph about why you choose this viewpoint id. '

# prompts.py:78-82 — Stage 3 (fusion)
_THOUGHT_FUSION_SYSTEM = "You are a thought fusion expert. Your task is to fuse given thought processes                     into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words."
_THOUGHT_FUSION_USER = "Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{}, Fused thought: "

# prompts.py:85-90 — Stage 3 (tie-break)
_DECISION_TEST_SYSTEM = "You are a decision testing expert. Your task is to evaluate the feasibility of each movement                         prediction based on thought process and environment. Then, you will make a final decision about direction viewpoint ID without other words.                             The answer should only be a number and within the candidate list."
_DECISION_TEST_USER = "The candidate list: {}. Can you help me make a final decision? The Observation: {}, Navigation Instruction: {}, {}, Final Decision: "

# prompts.py:14-15 — DIRECTIONS lookup table used to prefix summarised
# observations in save_history. The "Font Left" typo is verbatim from the
# reference source; preserved to keep LLM-visible history strings
# character-for-character consistent with reference navigator_log.log.
_DIRECTIONS = [
    "Front, range(left 15 to right 15)",
    "Font Left, range(left 15 to left 45)",
    "Left, range(left 45 to left 75)",
    "Left, range(left 75 to left 105)",
    "Rear Left, range(left 105 to left 135)",
    "Rear Left, range(left 135 to left 165)",
    "Back, range(left 165 to right 165)",
    "Rear Right, range(right 135 to right 165)",
    "Right, range(right 105 to right 135)",
    "Right, range(right 75 to right 105)",
    "Front Right, range(right 45 to right 75)",
    "Front Right, range(right 15 to right 45)",
]


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 prompt assemblers — produce {system, user} for the llmCall node
# ═══════════════════════════════════════════════════════════════════════


class DetectActionsPromptNode(BaseCanvasNode):
    """Stage 1a — assemble ACTION_DETECTION prompt for the LLM call."""

    node_type: ClassVar[str] = "opennav__detect_actions_prompt"
    display_name: ClassVar[str] = "Open-Nav: Detect Actions Prompt"
    description: ClassVar[str] = "Stage 1a — decompose instruction into atomic actions"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "List"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("instruction", "TEXT", "Raw navigation instruction")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        user = _ACTION_DETECTION_USER.format(instruction)
        self._self_log("instruction_len", len(instruction))
        return {"system": _ACTION_DETECTION_SYSTEM, "user": user}


class DetectLandmarksPromptNode(BaseCanvasNode):
    """Stage 1b — assemble LANDMARK_DETECTION prompt for the LLM call."""

    node_type: ClassVar[str] = "opennav__detect_landmarks_prompt"
    display_name: ClassVar[str] = "Open-Nav: Detect Landmarks Prompt"
    description: ClassVar[str] = "Stage 1b — extract landmarks from instruction"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("instruction", "TEXT", "Raw navigation instruction")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        user = _LANDMARK_DETECTION_USER.format(instruction)
        return {"system": _LANDMARK_DETECTION_SYSTEM, "user": user}


# ═══════════════════════════════════════════════════════════════════════
# Step-cap derivation (6 vs 8 from action count)
# ═══════════════════════════════════════════════════════════════════════


class ComputeStepCapNode(BaseCanvasNode):
    """Derive ``step_length`` from parsed action count.

    Mirrors ``base_il_trainer_llm.py:394`` — episodes truncate at 6 steps
    if the parsed action list has ≤ 6 lines, else 8 steps. This is the
    sole reason Open-Nav's absolute SR stays well below supervised SOTA;
    we replicate it verbatim.
    """

    node_type: ClassVar[str] = "opennav__compute_step_cap"
    display_name: ClassVar[str] = "Open-Nav: Compute Step Cap"
    description: ClassVar[str] = "step_length = 6 if len(actions.split('\\n')) <= 6 else 8"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Hash"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("threshold", "text", label="Threshold", default=6),
            ConfigField("short_cap", "text", label="Short cap", default=6),
            ConfigField("long_cap", "text", label="Long cap", default=8),
        ],
    )
    input_ports = [PortDef("actions", "TEXT", "Stage 1a output (action list)")]
    output_ports = [PortDef("step_length", "ANY", "Episode step cap")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        actions = str(inputs.get("actions", ""))
        cfg = self.config or {}
        threshold = int(cfg.get("threshold", 6))
        short_cap = int(cfg.get("short_cap", 6))
        long_cap = int(cfg.get("long_cap", 8))
        action_count = len(actions.split("\n"))
        step_length = short_cap if action_count <= threshold else long_cap
        self._self_log("action_count", action_count)
        self._self_log("step_length", step_length)
        return {"step_length": step_length}


# ═══════════════════════════════════════════════════════════════════════
# Per-direction observation formatter (RAM tags + SpatialBot caption)
# ═══════════════════════════════════════════════════════════════════════


class FormatObservationNode(BaseCanvasNode):
    """Fuse per-direction perception into the NAVIGATOR's Current Environment.

    Mirrors ``spatialNavigator.py`` ``observe_environment`` output shape:

        Direction {i} Direction Viewpoint ID: {i} in Step ID: {t} Elevation: Eye Level
        Scene Description: {SpatialBot caption}
        Scene Objects: {RAM tags};

    Inputs come from the perception nodeset (per-candidate dicts keyed by
    direction id); only candidate directions returned by the waypoint
    predictor are formatted (the rest are dropped).
    """

    node_type: ClassVar[str] = "opennav__format_observation"
    display_name: ClassVar[str] = "Open-Nav: Format Observation"
    description: ClassVar[str] = "Fuse RAM tags + SpatialBot captions per candidate direction"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Eye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("candidates", "ANY", "{dir_id: (angle, distance)} from waypoint predictor"),
        PortDef("tags", "ANY", "{dir_id: 'tag tag tag'} from RAM"),
        PortDef("captions", "ANY", "{dir_id: 'spatial caption'} from SpatialBot"),
        PortDef("step", "ANY", "Current step index (0-based)"),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "Concatenated per-direction observation string"),
        PortDef("candidate_ids", "TEXT", "Comma-separated candidate viewpoint id list"),
        PortDef(
            "blocks",
            "ANY",
            "{dir_id: per-direction observation block} for downstream per-vp selection",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        candidates = inputs.get("candidates") or {}
        tags = inputs.get("tags") or {}
        captions = inputs.get("captions") or {}
        step = int(inputs.get("step") or 0)

        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = {}
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = {}
        if isinstance(captions, str):
            try:
                captions = json.loads(captions)
            except Exception:
                captions = {}

        parts: list[str] = []
        ids: list[str] = []
        blocks: dict[str, str] = {}
        for dir_id in sorted(candidates.keys(), key=lambda x: int(x)):
            cap = str(captions.get(dir_id, "")).strip()
            tag = str(tags.get(dir_id, "")).strip()
            # Single-line format matches reference observe_view exactly:
            # "Direction {i} Direction Viewpoint ID: {i} in Step ID: {t} "
            # "Elevation: Eye Level Scene Description: ... Scene Objects: ...; "
            block = (
                f"Direction {dir_id} Direction Viewpoint ID: {dir_id} in Step ID: {step} "
                f"Elevation: Eye Level "
                f"Scene Description: {cap} "
                f"Scene Objects: {tag}; "
            )
            parts.append(block)
            ids.append(str(dir_id))
            blocks[str(dir_id)] = block

        observation = "".join(parts)
        candidate_ids = ", ".join(ids)
        self._self_log("num_candidates", len(ids))
        self._self_log("observation_chars", len(observation))
        return {"observation": observation, "candidate_ids": candidate_ids, "blocks": blocks}


class SelectDirectionObservationNode(BaseCanvasNode):
    """Pick the single-direction block for the chosen ``next_vp``.

    The reference ``save_history`` (`spatialNavigator.py:40-43`) summarises
    only the observation for the selected VP, not the whole per-step
    concatenation. This node bridges the decision output and the
    observation-summary LLM call:

        blocks[next_vp] → observation + direction_id (parsed from block)
    """

    node_type: ClassVar[str] = "opennav__select_direction_observation"
    display_name: ClassVar[str] = "Open-Nav: Select Direction Observation"
    description: ClassVar[str] = "Pick per-direction block for next_vp (+ parse direction_id)"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("blocks", "ANY", "{dir_id: per-direction block} from format_observation"),
        PortDef("next_vp", "TEXT", "Chosen viewpoint id"),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "Single-direction observation block (empty if not found)"),
        PortDef("direction_id", "ANY", "Integer direction id of next_vp"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        blocks = inputs.get("blocks") or {}
        next_vp = str(inputs.get("next_vp", "")).strip()
        if isinstance(blocks, str):
            try:
                blocks = json.loads(blocks)
            except Exception:
                blocks = {}

        observation = str(blocks.get(next_vp, "") or "")
        try:
            direction_id = int(next_vp)
        except (TypeError, ValueError):
            direction_id = 0
        self._self_log("next_vp", next_vp)
        self._self_log("direction_id", direction_id)
        self._self_log("block_chars", len(observation))
        return {"observation": observation, "direction_id": direction_id}


# ═══════════════════════════════════════════════════════════════════════
# History review (joins nav_history into a flat string for prompts)
# ═══════════════════════════════════════════════════════════════════════


class ReviewHistoryNode(BaseCanvasNode):
    """Read ``nav_history`` from the state container and flatten to a string.

    Mirrors ``spatialNavigator.py:56-59`` — joins ``"Step {i+1} Observation:
    {obs} Thought: {thought}"`` with `` -> ``. Returns ``"Step 0 start
    position. "`` if the history is empty (matches reference behaviour).
    """

    node_type: ClassVar[str] = "opennav__review_history"
    display_name: ClassVar[str] = "Open-Nav: Review History"
    description: ClassVar[str] = "Flatten nav_history container to '... -> ...' string"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "BookOpen"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports: list = []
    output_ports = [PortDef("history_traj", "TEXT", "Flattened history string")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        entries: list = []
        if ctx and getattr(ctx, "graph_state", None):
            raw = ctx.graph_state.read("nav_history")
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        entries.append(item)
                    elif isinstance(item, str):
                        try:
                            entries.append(json.loads(item))
                        except Exception:
                            entries.append({"observation": item, "thought": ""})

        if not entries:
            return {"history_traj": "Step 0 start position. "}

        parts = [
            f"Step {i + 1} Observation: {e.get('observation', '')} Thought: {e.get('thought', '')}"
            for i, e in enumerate(entries)
        ]
        history_traj = " -> ".join(parts)
        self._self_log("history_steps", len(entries))
        return {"history_traj": history_traj}


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — completion estimation prompt + parser
# ═══════════════════════════════════════════════════════════════════════


class CompletionEstimationPromptNode(BaseCanvasNode):
    """Assemble the COMPLETION_ESTIMATION user prompt."""

    node_type: ClassVar[str] = "opennav__estimate_completion_prompt"
    display_name: ClassVar[str] = "Open-Nav: Estimate Completion Prompt"
    description: ClassVar[str] = "Stage 2 — estimate executed actions from history + landmarks"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "CheckSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("history_traj", "TEXT", "Flattened nav history"),
        PortDef("landmarks", "TEXT", "Stage 1b output"),
        PortDef("instruction", "TEXT", "Raw instruction"),
    ]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        user = _COMPLETION_ESTIMATION_USER.format(
            str(inputs.get("history_traj", "")),
            str(inputs.get("landmarks", "")),
            str(inputs.get("instruction", "")),
        )
        return {"system": _COMPLETION_ESTIMATION_SYSTEM, "user": user}


class ParseExecutedActionsNode(BaseCanvasNode):
    """Split COMPLETION_ESTIMATION output on ``"Executed Actions:"``.

    Mirrors ``spatialNavigator.py:62-70`` — the LLM is instructed to emit
    ``"Thought: ...\\nExecuted Actions: ..."``; we parse with a plain
    string split, no regex. Falls back to the full text if the marker is
    missing.
    """

    node_type: ClassVar[str] = "opennav__parse_executed_actions"
    display_name: ClassVar[str] = "Open-Nav: Parse Executed Actions"
    description: ClassVar[str] = "Split LLM output on 'Executed Actions:'"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Filter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("llm_out", "TEXT", "LLM completion estimation output")]
    output_ports = [
        PortDef("executed_actions", "TEXT", "Plain-string executed action list"),
        PortDef("thought", "TEXT", "Pre-marker thought text"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        text = str(inputs.get("llm_out", ""))
        marker = "Executed Actions:"
        if marker in text:
            thought_part, executed_part = text.split(marker, 1)
            executed = executed_part.strip()
            thought = thought_part.replace("Thought:", "").strip()
        else:
            executed = text.strip()
            thought = ""
        self._self_log("executed_chars", len(executed))
        return {"executed_actions": executed, "thought": thought}


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — NAVIGATOR prompt + parser
# ═══════════════════════════════════════════════════════════════════════


class NavigatorPromptNode(BaseCanvasNode):
    """Assemble the NAVIGATOR user prompt.

    Format mirrors ``spatialNavigator.py`` call site:
        NAVIGATOR['user'].format(candidate_ids, step, instruction,
                                 instruction, landmarks, history_traj,
                                 executed_actions, observation)
    """

    node_type: ClassVar[str] = "opennav__navigator_prompt"
    display_name: ClassVar[str] = "Open-Nav: Navigator Prompt"
    description: ClassVar[str] = "Stage 3a — assemble decision-making prompt"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Compass"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("candidate_ids", "TEXT", "Candidate viewpoint id list"),
        PortDef("step", "ANY", "Current step index"),
        PortDef("instruction", "TEXT", "Raw instruction"),
        PortDef("landmarks", "TEXT", "Stage 1b output"),
        PortDef("history_traj", "TEXT", "Flattened nav history"),
        PortDef("executed_actions", "TEXT", "Stage 2 output"),
        PortDef("observation", "TEXT", "Per-direction observation block"),
    ]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        user = _NAVIGATOR_USER.format(
            str(inputs.get("candidate_ids", "")),
            int(inputs.get("step") or 0),
            str(inputs.get("instruction", "")),
            str(inputs.get("instruction", "")),
            str(inputs.get("landmarks", "")),
            str(inputs.get("history_traj", "")),
            str(inputs.get("executed_actions", "")),
            str(inputs.get("observation", "")),
        )
        return {"system": _NAVIGATOR_SYSTEM, "user": user}


class ParseNavigatorOutputNode(BaseCanvasNode):
    """Verbatim parser from ``spatialNavigator.py:89-91``.

        pred_thought = decision_reasoning.split("Prediction:")[0].strip()
        pred_vp = decision_reasoning.split("Prediction:")[1].strip() \\
            .replace('"','').replace("'","").replace("\\n","") \\
            .replace(".","").replace("*","")

    Returns ``("", "")`` if the marker is missing — the upstream
    ``test_decisions`` fallback handles that case.
    """

    node_type: ClassVar[str] = "opennav__parse_navigator_output"
    display_name: ClassVar[str] = "Open-Nav: Parse Navigator Output"
    description: ClassVar[str] = "Verbatim string-split parser on 'Prediction:'"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Filter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("llm_out", "TEXT", "Single NAVIGATOR sample")]
    output_ports = [
        PortDef("pred_vp", "TEXT", "Predicted viewpoint id (raw string)"),
        PortDef("pred_thought", "TEXT", "Pre-Prediction thought"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        decision_reasoning = str(inputs.get("llm_out", ""))
        marker = "Prediction:"
        if marker not in decision_reasoning:
            self._self_log("parse_error", "no_prediction_marker")
            return {"pred_vp": "", "pred_thought": decision_reasoning.strip()}
        pred_thought = decision_reasoning.split(marker)[0].strip()
        pred_vp = (
            decision_reasoning.split(marker)[1]
            .strip()
            .replace('"', "")
            .replace("'", "")
            .replace("\n", "")
            .replace(".", "")
            .replace("*", "")
        )
        self._self_log("pred_vp", pred_vp)
        return {"pred_vp": pred_vp, "pred_thought": pred_thought}


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — group predictions, fuse, test, fallback
# ═══════════════════════════════════════════════════════════════════════


class GroupPredictionsNode(BaseCanvasNode):
    """Group N navigator samples by predicted viewpoint id.

    Mirrors ``spatialNavigator.py:98-110`` — builds ``matched_dict`` with
    ``"; ".join(["Thought i: t" for i, t in enumerate(value)])`` per group.
    The fusion call itself is one ``llmCall`` per group, fanned out from
    here.
    """

    node_type: ClassVar[str] = "opennav__group_predictions"
    display_name: ClassVar[str] = "Open-Nav: Group Predictions"
    description: ClassVar[str] = "Group navigator samples by predicted vp"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Group"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("preds", "ANY", "List of predicted vp strings"),
        PortDef("thoughts", "ANY", "List of per-sample thoughts (parallel)"),
    ]
    output_ports = [
        PortDef("groups", "ANY", "{vp_id: '; '-joined thoughts}"),
        PortDef("group_count", "ANY", "Number of unique groups"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        preds = inputs.get("preds") or []
        thoughts = inputs.get("thoughts") or []
        if isinstance(preds, str):
            try:
                preds = json.loads(preds)
            except Exception:
                preds = [preds]
        if isinstance(thoughts, str):
            try:
                thoughts = json.loads(thoughts)
            except Exception:
                thoughts = [thoughts]

        matched: dict[str, list[str]] = {}
        for pred, thought in zip(preds, thoughts, strict=False):
            key = str(pred).strip()
            matched.setdefault(key, []).append(str(thought))

        groups: dict[str, str] = {}
        for key, value in matched.items():
            groups[key] = "; ".join(f"Thought {i + 1}: {t}" for i, t in enumerate(value))

        self._self_log("group_count", len(groups))
        return {"groups": groups, "group_count": len(groups)}


async def _internal_llm_call(
    profile_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 512
) -> str:
    """Run a single LLM call against a profile — used by Stage 3 fan-out.

    Wraps ``app.llm.call.llm_complete`` so the Stage 3 ensemble + fusion
    + tie-break can issue N LLM calls per node firing without blowing
    the graph up into N parallel ``llmCall`` nodes.
    """
    from app.llm.call import get_llm_config, llm_complete

    cfg = get_llm_config(profile_name or "")
    if cfg is None:
        log.warning("opennav: LLM profile '%s' not found", profile_name)
        return ""
    out = await llm_complete(
        config=cfg,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return out or ""


def _parse_navigator(text: str) -> tuple[str, str]:
    """Verbatim parser from spatialNavigator.py:89-91 (returns (vp, thought))."""
    if "Prediction:" not in text:
        return "", text.strip()
    pred_thought = text.split("Prediction:")[0].strip()
    pred_vp = (
        text.split("Prediction:")[1]
        .strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\n", "")
        .replace(".", "")
        .replace("*", "")
    )
    return pred_vp, pred_thought


class OpenNavDecisionNode(BaseCanvasNode):
    """Full Stage 3 — ensemble + fusion + tie-break + fallback in one node.

    Internally issues:

      - ``num_samples`` (default 3) NAVIGATOR calls (temperature=0)
      - one THOUGHT_FUSION call per unique predicted vp
      - one DECISION_TEST call iff ≥ 2 unique vps survive fusion
      - falls back to ``random.choice`` after ``error_threshold`` parse errors

    Folding this into a single Python node (rather than 3 + N graph
    ``llmCall`` nodes) is what lets the static graph faithfully express
    the dynamic-fan-out fusion step. All N+M+1 LLM calls still happen and
    each one runs through the same ``app.llm.call.llm_complete`` path the
    built-in ``llmCall`` node uses — no observability is lost beyond the
    fact that they appear as a single node firing in the graph view.
    """

    node_type: ClassVar[str] = "opennav__decision"
    display_name: ClassVar[str] = "Open-Nav: Decision (ensemble+fuse+test)"
    description: ClassVar[str] = (
        "Stage 3 — N navigator samples → group → fuse → tie-break → fallback"
    )
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Compass"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("llm_profile", "text", label="LLM profile", default=""),
            ConfigField("num_samples", "text", label="Navigator samples", default=3),
            ConfigField(
                "navigator_outer_retries", "text", label="Navigator outer retries", default=2
            ),
            ConfigField("decision_test_retries", "text", label="DECISION_TEST retries", default=2),
            ConfigField("error_threshold", "text", label="Error threshold", default=2),
            ConfigField("max_tokens", "text", label="max_tokens", default=512),
        ],
    )
    input_ports = [
        PortDef("candidate_ids", "TEXT", "Candidate viewpoint id list"),
        PortDef("step", "ANY", "Current step index"),
        PortDef("instruction", "TEXT", "Raw instruction"),
        PortDef("landmarks", "TEXT", "Stage 1b output"),
        PortDef("history_traj", "TEXT", "Flattened nav history"),
        PortDef("executed_actions", "TEXT", "Stage 2 output"),
        PortDef("observation", "TEXT", "Per-direction observation block"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]}"),
    ]
    output_ports = [
        PortDef("next_vp", "TEXT", "Final chosen viewpoint id"),
        PortDef("decision_thought", "TEXT", "Fused thought for next_vp"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        profile = str(cfg.get("llm_profile", ""))
        num_samples = int(cfg.get("num_samples", 3))
        navigator_outer_retries = int(cfg.get("navigator_outer_retries", 2))
        decision_test_retries = int(cfg.get("decision_test_retries", 2))
        error_threshold = int(cfg.get("error_threshold", 2))
        max_tokens = int(cfg.get("max_tokens", 512))

        candidates = inputs.get("candidates") or {}
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = {}

        # Cross-step error_number accumulator — lives in graph_state so
        # consecutive failures trigger the `random.choice` fallback at
        # step-level (reference base_il_trainer_llm.py:369,448).
        prev_error_number = 0
        if ctx and getattr(ctx, "graph_state", None):
            raw = ctx.graph_state.read("opennav_error_number")
            if isinstance(raw, int):
                prev_error_number = raw
            elif isinstance(raw, str) and raw.isdigit():
                prev_error_number = int(raw)

        navigator_user = _NAVIGATOR_USER.format(
            str(inputs.get("candidate_ids", "")),
            int(inputs.get("step") or 0),
            str(inputs.get("instruction", "")),
            str(inputs.get("instruction", "")),
            str(inputs.get("landmarks", "")),
            str(inputs.get("history_traj", "")),
            str(inputs.get("executed_actions", "")),
            str(inputs.get("observation", "")),
        )

        # 1) Stage 3a - outer 2x retry wrapping N-sample navigator batch.
        # Reference spatialNavigator.py:76-93 resets `effective_prediction,
        # thought_list = [], []` at the top of each outer iteration, so
        # only the LAST iteration's samples are retained. We replicate
        # that behaviour verbatim (6 total LLM calls, last 3 kept).
        samples: list[str] = []
        for _retry in range(max(1, navigator_outer_retries)):
            samples = []
            for _ in range(num_samples):
                text = await _internal_llm_call(
                    profile, _NAVIGATOR_SYSTEM, navigator_user, max_tokens=max_tokens
                )
                samples.append(text)

        preds: list[str] = []
        thoughts: list[str] = []
        step_errors = 0
        for s in samples:
            vp, thought = _parse_navigator(s)
            if not vp:
                step_errors += 1
            preds.append(vp)
            thoughts.append(thought)

        # 2) Stage 3b — group by predicted vp, run THOUGHT_FUSION per group.
        matched: dict[str, list[str]] = {}
        for pred, thought in zip(preds, thoughts, strict=False):
            if not pred:
                continue
            matched.setdefault(pred, []).append(thought)

        fused: dict[str, str] = {}
        for key, value in matched.items():
            multi = "; ".join(f"Thought {i + 1}: {t}" for i, t in enumerate(value))
            fused_text = await _internal_llm_call(
                profile,
                _THOUGHT_FUSION_SYSTEM,
                _THOUGHT_FUSION_USER.format(multi),
                max_tokens=max_tokens,
            )
            fused[key] = fused_text

        # Pre-filter malformed keys (reference spatialNavigator.py:114-116
        # pops entries where ``len(fused_key) > 2``). Leaves only single/
        # double-digit numeric keys for the tie-break prompt.
        fused = {k: v for k, v in fused.items() if len(k) <= 2}

        next_vp = ""
        decision_thought = ""
        error_number = prev_error_number

        # 3) Stage 3c — single survivor → return; otherwise DECISION_TEST
        # with up to `decision_test_retries` retries (reference
        # spatialNavigator.py:126-131).
        if len(fused) == 1:
            next_vp = next(iter(fused.keys()))
            decision_thought = next(iter(fused.values()))
        elif len(fused) > 1:
            tie_prompt_user = _DECISION_TEST_USER.format(
                list(fused.keys()),
                str(inputs.get("observation", "")),
                str(inputs.get("instruction", "")),
                "; ".join(f"vp {k}: {v}" for k, v in fused.items()),
            )
            for _ in range(max(1, decision_test_retries)):
                tie_text = await _internal_llm_call(
                    profile,
                    _DECISION_TEST_SYSTEM,
                    tie_prompt_user,
                    max_tokens=max_tokens,
                )
                match = re.search(r"(\d+)", tie_text or "")
                if match and match.group(1) in fused:
                    next_vp = match.group(1)
                    decision_thought = fused.get(next_vp, "")
                    break
            if not next_vp:
                error_number += 1

        # 4) Empty-fused → treat as decision error.
        if not fused and not next_vp:
            error_number += 1

        # 5) Fallback path — verbatim from spatialNavigator.py:141-150.
        if not next_vp and error_number >= error_threshold:
            valid = {k: v for k, v in fused.items() if len(k) < 2}
            if valid:
                next_vp, decision_thought = random.choice(list(valid.items()))
            elif candidates:
                next_vp = random.choice(list(candidates.keys()))
                decision_thought = "(random fallback over candidates)"
            error_number = 0  # reference resets after fallback fires
            self._self_log("fallback", "random_choice")

        if not next_vp and candidates:
            next_vp = next(iter(candidates.keys()))
            decision_thought = "(first-candidate guard)"

        # 6) Reset error_number on a clean decision — matches
        # base_il_trainer_llm.py:448 (`error_number = 0` after a
        # successful step).
        if next_vp and decision_thought and not decision_thought.startswith("("):
            error_number = 0

        if ctx and getattr(ctx, "graph_state", None):
            ctx.graph_state.write("opennav_error_number", error_number)

        self._self_log("num_samples", num_samples)
        self._self_log("outer_retries", navigator_outer_retries)
        self._self_log("preds", preds)
        self._self_log("prev_error_number", prev_error_number)
        self._self_log("step_errors", step_errors)
        self._self_log("error_number", error_number)
        self._self_log("group_count", len(fused))
        self._self_log("next_vp", next_vp)
        return {"next_vp": str(next_vp), "decision_thought": decision_thought}


class TestDecisionNode(BaseCanvasNode):
    """Pick a final viewpoint from the fused groups; fall back on errors.

    Mirrors ``spatialNavigator.py:112-151``:

        if error_number >= 2:
            error_number = 0
            if fused_pred_thought and all(len(key) < 2 for key in fused_pred_thought):
                next_vp, _ = random.choice(list(fused_pred_thought.items()))
            else:
                next_vp, _ = random.choice(list(observe_dict.items()))

    If exactly one unique vp survives fusion, it is returned without a
    tie-break LLM call. Otherwise, the user supplies an upstream
    ``llmCall(DECISION_TEST)`` result via the ``tie_break`` input.
    """

    node_type: ClassVar[str] = "opennav__test_decision"
    display_name: ClassVar[str] = "Open-Nav: Test Decision"
    description: ClassVar[str] = "Tie-break + 2-error random fallback"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Shuffle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("error_threshold", "text", label="Error threshold", default=2),
        ],
    )
    input_ports = [
        PortDef("groups", "ANY", "Output of opennav__group_predictions"),
        PortDef("fused", "ANY", "{vp_id: fused_thought} from THOUGHT_FUSION calls"),
        PortDef("tie_break", "TEXT", "Optional DECISION_TEST llm output (for >1 group)"),
        PortDef("candidates", "ANY", "{dir_id: (angle, distance)} from waypoint predictor"),
    ]
    output_ports = [
        PortDef("next_vp", "TEXT", "Final chosen viewpoint id (string)"),
        PortDef("decision_thought", "TEXT", "Fused thought for next_vp"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        error_threshold = int(cfg.get("error_threshold", 2))

        groups = inputs.get("groups") or {}
        fused = inputs.get("fused") or {}
        candidates = inputs.get("candidates") or {}
        tie_break_raw = str(inputs.get("tie_break", "") or "").strip()

        if isinstance(groups, str):
            try:
                groups = json.loads(groups)
            except Exception:
                groups = {}
        if isinstance(fused, str):
            try:
                fused = json.loads(fused)
            except Exception:
                fused = {}
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = {}

        # Per-step error counter on graph_state (LastWrite reducer).
        error_number = 0
        if ctx and getattr(ctx, "graph_state", None):
            raw = ctx.graph_state.read("opennav_error_number")
            if isinstance(raw, int):
                error_number = raw
            elif isinstance(raw, str) and raw.isdigit():
                error_number = int(raw)

        next_vp = ""
        decision_thought = ""

        # Single unique vp → return immediately.
        if len(fused) == 1:
            next_vp = next(iter(fused.keys()))
            decision_thought = next(iter(fused.values()))
        elif len(fused) > 1 and tie_break_raw:
            # Parse trailing digits from the DECISION_TEST output.
            match = re.search(r"(\d+)", tie_break_raw)
            if match and match.group(1) in fused:
                next_vp = match.group(1)
                decision_thought = fused.get(next_vp, "")
            else:
                error_number += 1

        # Fallback: ≥ 2 errors → random.choice
        if not next_vp and error_number >= error_threshold:
            error_number = 0
            valid = {k: v for k, v in fused.items() if len(k) < 2}
            if valid:
                next_vp, decision_thought = random.choice(list(valid.items()))
            elif candidates:
                next_vp = random.choice(list(candidates.keys()))
                decision_thought = "(random fallback over candidates)"
            self._self_log("fallback", "random_choice")

        # Final guard.
        if not next_vp and candidates:
            next_vp = next(iter(candidates.keys()))
            decision_thought = "(first-candidate guard)"

        if ctx and getattr(ctx, "graph_state", None):
            ctx.graph_state.write("opennav_error_number", error_number)

        self._self_log("next_vp", next_vp)
        self._self_log("error_number", error_number)
        self._self_log("group_count", len(fused))
        return {"next_vp": str(next_vp), "decision_thought": decision_thought}


# ═══════════════════════════════════════════════════════════════════════
# Resolve next_vp → (angle, distance) for env_habitat__step_hightolow
# ═══════════════════════════════════════════════════════════════════════


class ResolveActionNode(BaseCanvasNode):
    """Look up ``(angle, distance)`` for the chosen viewpoint id.

    The waypoint predictor returns ``{dir_id: [angle_rad, distance_m]}``.
    This node pulls the entry for ``next_vp`` and exposes the two scalars
    as state-typed ports so the downstream ``env_habitat__step_hightolow``
    can wire them directly.
    """

    node_type: ClassVar[str] = "opennav__resolve_action"
    display_name: ClassVar[str] = "Open-Nav: Resolve Action"
    description: ClassVar[str] = "Lookup (angle, distance) for next_vp"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "ArrowRight"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("next_vp", "TEXT", "Chosen direction id string"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]}"),
    ]
    output_ports = [
        PortDef("angle", "ANY", "Angle in radians"),
        PortDef("distance", "ANY", "Distance in metres"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        next_vp = str(inputs.get("next_vp", "")).strip()
        candidates = inputs.get("candidates") or {}
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = {}

        entry = candidates.get(next_vp)
        if entry is None and candidates:
            entry = next(iter(candidates.values()))
        angle, distance = 0.0, 0.0
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            angle = float(entry[0])
            distance = float(entry[1])
        elif isinstance(entry, dict):
            angle = float(entry.get("angle", 0.0))
            distance = float(entry.get("distance", 0.0))
        self._self_log("next_vp", next_vp)
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)
        return {"angle": angle, "distance": distance}


# ═══════════════════════════════════════════════════════════════════════
# History summary prompts + nav_history append
# ═══════════════════════════════════════════════════════════════════════


class ObservationSummaryPromptNode(BaseCanvasNode):
    """Assemble the OBSERVATION_SUMMARY prompt for the per-step compressor."""

    node_type: ClassVar[str] = "opennav__observation_summary_prompt"
    display_name: ClassVar[str] = "Open-Nav: Observation Summary Prompt"
    description: ClassVar[str] = "Compress observation block before history append"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Minimize2"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("observation", "TEXT", "Per-step observation string")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        user = _OBSERVATION_SUMMARY_USER.format(str(inputs.get("observation", "")))
        return {"system": _OBSERVATION_SUMMARY_SYSTEM, "user": user}


class ThoughtSummaryPromptNode(BaseCanvasNode):
    """Assemble the THOUGHT_SUMMARY prompt for the per-step compressor."""

    node_type: ClassVar[str] = "opennav__thought_summary_prompt"
    display_name: ClassVar[str] = "Open-Nav: Thought Summary Prompt"
    description: ClassVar[str] = "Compress decision thought before history append"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Minimize2"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("thought", "TEXT", "Decision thought string")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        user = _THOUGHT_SUMMARY_USER.format(str(inputs.get("thought", "")))
        return {"system": _THOUGHT_SUMMARY_SYSTEM, "user": user}


class AppendHistoryNode(BaseCanvasNode):
    """Append ``{step, viewpoint, observation, thought}`` to ``nav_history``.

    Writes to the graph state container (``Accumulator`` reducer) so the
    next step's ``ReviewHistoryNode`` can read the updated trajectory.
    """

    node_type: ClassVar[str] = "opennav__append_history"
    display_name: ClassVar[str] = "Open-Nav: Append History"
    description: ClassVar[str] = "Append step entry to nav_history container"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Plus"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports = [
        PortDef("step", "ANY", "Current step index"),
        PortDef("viewpoint", "TEXT", "Chosen viewpoint id"),
        PortDef("observation_summary", "TEXT", "Compressed observation"),
        PortDef("thought_summary", "TEXT", "Compressed thought"),
        PortDef("direction_id", "ANY", "Direction id (0..11) for DIRECTIONS prefix (optional)"),
    ]
    output_ports = [PortDef("entry", "TEXT", "JSON-encoded history entry (for downstream wires)")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        observation_summary = str(inputs.get("observation_summary", ""))
        direction_id_raw = inputs.get("direction_id")
        # Prefix with DIRECTIONS[direction_id] to mirror reference
        # save_history: "Direction {direction} " + summary. The LLM uses
        # this prefix to reason about direction change in history.
        if direction_id_raw is not None and observation_summary:
            try:
                did = int(direction_id_raw)
                if 0 <= did < len(_DIRECTIONS):
                    observation_summary = f"Direction {_DIRECTIONS[did]} " + observation_summary
            except (TypeError, ValueError):
                pass

        entry = {
            "step": int(inputs.get("step") or 0),
            "viewpoint": str(inputs.get("viewpoint", "")),
            "observation": observation_summary,
            "thought": str(inputs.get("thought_summary", "")),
        }
        if ctx and getattr(ctx, "graph_state", None):
            ctx.graph_state.write("nav_history", entry)
        self._self_log("entry_step", entry["step"])
        self._self_log("entry_viewpoint", entry["viewpoint"])
        return {"entry": json.dumps(entry)}


# ═══════════════════════════════════════════════════════════════════════
# Step counter + step-cap termination (reference step_length enforcement)
# ═══════════════════════════════════════════════════════════════════════


class IncrementNode(BaseCanvasNode):
    """Integer +1 increment — feeds ``step`` from one iteration to the next.

    The graph executor's ``iterOut`` transfers loop_port values back to
    the paired ``iterIn`` at iteration end. We use this node to route
    ``iter_in.step + 1`` into ``iter_out.step`` so ``iter_in.step`` holds
    a monotonically-increasing 1-based step counter next iteration
    (matching reference ``current_step += 1``).
    """

    node_type: ClassVar[str] = "opennav__increment"
    display_name: ClassVar[str] = "Open-Nav: Increment"
    description: ClassVar[str] = "Integer +1 increment"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Plus"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="slate")
    input_ports = [PortDef("value", "ANY", "Input integer")]
    output_ports = [PortDef("next", "ANY", "value + 1")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            v = int(inputs.get("value") or 0)
        except (TypeError, ValueError):
            v = 0
        return {"next": v + 1}


class StepCapCompareNode(BaseCanvasNode):
    """Compare ``step`` against ``step_length`` and OR with habitat ``done``.

    Reference ``base_il_trainer_llm.py:450-451``: `if current_step ==
    step_length: dones[0] = True`. We return ``done = step >= step_length
    OR habitat_done`` so either termination path ends the loop.
    """

    node_type: ClassVar[str] = "opennav__step_cap_compare"
    display_name: ClassVar[str] = "Open-Nav: Step Cap Compare"
    description: ClassVar[str] = "done = step >= step_length OR habitat_done"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Flag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="slate")
    input_ports = [
        PortDef("step", "ANY", "Current 1-based step counter"),
        PortDef("step_length", "ANY", "Episode cap (6 or 8)"),
        PortDef("habitat_done", "BOOL", "done signal from env_habitat step"),
    ]
    output_ports = [PortDef("done", "BOOL", "Combined termination signal")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            step = int(inputs.get("step") or 0)
        except (TypeError, ValueError):
            step = 0
        try:
            cap = int(inputs.get("step_length") or 0)
        except (TypeError, ValueError):
            cap = 0
        habitat_done = bool(inputs.get("habitat_done"))
        cap_hit = cap > 0 and step >= cap
        done = bool(cap_hit or habitat_done)
        self._self_log("step", step)
        self._self_log("step_length", cap)
        self._self_log("cap_hit", cap_hit)
        self._self_log("habitat_done", habitat_done)
        self._self_log("done", done)
        return {"done": done}


# ═══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ═══════════════════════════════════════════════════════════════════════


class OpenNavNodeSet(BaseNodeSet):
    """Open-Nav method nodeset — prompts, parsers, ensemble + fusion logic."""

    name = "opennav"
    description = (
        "Open-Nav (ICRA 2025) method-side nodes: spatial-temporal CoT prompts, "
        "string-split parsers, 3-sample ensemble + fusion + tie-break decision policy. "
        "Pairs with env_habitat + opennav_waypoint + opennav_perception."
    )

    def get_tools(self) -> list:
        return [
            DetectActionsPromptNode(),
            DetectLandmarksPromptNode(),
            ComputeStepCapNode(),
            FormatObservationNode(),
            SelectDirectionObservationNode(),
            ReviewHistoryNode(),
            CompletionEstimationPromptNode(),
            ParseExecutedActionsNode(),
            NavigatorPromptNode(),
            ParseNavigatorOutputNode(),
            GroupPredictionsNode(),
            TestDecisionNode(),
            OpenNavDecisionNode(),
            ResolveActionNode(),
            ObservationSummaryPromptNode(),
            ThoughtSummaryPromptNode(),
            AppendHistoryNode(),
            IncrementNode(),
            StepCapCompareNode(),
        ]
