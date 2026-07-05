"""Three-Step Nav — verbatim prompt assets + parsers.

All system/user prompt strings below are copied character-for-character from
the upstream Three-Step Nav reference, each constant citing its source
`file:line`. Keeping them here (a sidecar, not the entry module) mirrors the
`smartway_mono/_prompts.py` layout and keeps the node bodies in `__init__.py`
thin.

Upstream: https://github.com/ZoeyZheng0/3-step-Nav @ 5cdbdcf
    vlnce_baselines/common/navigator/prompts.py            (MapGPT navigator, decompose, summaries)
    vlnce_baselines/common/navigator/decision_agent.py     (the judge / back-check)
    vlnce_baselines/common/navigator/navigation_capabilities.py (capability descriptions)

Prompt text *is* the experiment — do not paraphrase.
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════
# Step 1 — instruction decomposition (prompts.py:8-14, 17-20)
# ═══════════════════════════════════════════════════════════════════════

ACTION_DETECTION_SYSTEM = "You are an action decomposition expert. Your task is to decompose the whole instruction into a series of sub-instructions and all actions in the given navigation instruction. You need to ensure the integrity of each action. You need to make sure the sub-instructions are complete, and include the details of the current environment if it is mentioned in the instruction.                 Your answer must consist ONLY of a series of labled action phrases without begin sentence.                 For each sub-instruction, it should involve at least one action, and all the description of the environment related to the same location.                 A typical answer should involve 3 to 8 sub-instructions."
ACTION_DETECTION_USER = 'Can you decompose actions in the instruction "{}"? Actions: '

LANDMARK_DETECTION_SYSTEM = "You are a landmark extraction expert. Your task is to detect all landmarks in the given navigation instruction. You need to ensure the integrity of each landmarks. Your answer must consist ONLY of a series of labeled landmark phrases without other sentences."
LANDMARK_DETECTION_USER = 'Can you extract landmarks in the instruction "{}"? Landmarks: '

# ═══════════════════════════════════════════════════════════════════════
# Per-step history compression (prompts.py:27-31, 34-38)
# ═══════════════════════════════════════════════════════════════════════

OBSERVATION_SUMMARY_SYSTEM = "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible.                                             You ONLY need to summarize in a single paragraph."
OBSERVATION_SUMMARY_USER = 'Given Environment Description "{}", Summarization:'

THOUGHT_SUMMARY_SYSTEM = 'You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible.                                             You ONLY need to summarize the what actions you did and what landmarks you passed in "Thought" using a single paragraph. Do NOT include Direction information. '
THOUGHT_SUMMARY_USER = 'Given Thought Process "{}", Summarization:'

# ═══════════════════════════════════════════════════════════════════════
# DIRECTIONS lookup (prompts.py:23-24) — the "Font Left" typo at index 1 is
# verbatim from upstream; preserved so LLM-visible history strings stay
# character-identical to the reference logs.
# ═══════════════════════════════════════════════════════════════════════

DIRECTIONS = [
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
# Step 2 — MapGPT-style image navigator (prompts.py:86-117). Single call;
# folds the completion estimate ("Completion Estimation: Yes/No") into the
# navigator output, which gates the Step-3 judge.
# ═══════════════════════════════════════════════════════════════════════

MAPGPT_NAVIGATOR_SYSTEM = "You are an embodied robot that navigates in the real world.             You need to explore between some places marked with IDs and ultimately find the destination to stop.             I will give you one instruction and tell you landmarks. I will also give you navigation history for reference.             You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you.             Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint.             Each direction viewpoint has an image that you can see.             In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0             Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you.             If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position.             You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations.             For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress.             If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination.             Your answer includes four parts: \"Thought\", \"Distance\", \"Prediction\" and \"Completion Estimation\". In the \"Thought\", you should think as detailed as possible following procedures:             (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after \"Prediction\" without any other words or characters .             (2) Analyze which direction in the current environment is most suitable to execute the instruction and explain your reason.             (3) You need to combine 'Instruction', 'Landmarks', your past 'Navigation History', 'Current Environment', and the provided images to think about what to do next, and complete your thinking into 'Thought'.             (4) Predict moving to which direction viewpoint based on your thought process.             (5) The \"Thought\" you predicted should be a single paragraph.             (6) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the \"Prediction\".             (7) If you want to make a left turn, you usually need to select a viewpoint ID between 1 and 5. If you want to make a right turn, you usually need to select a viewpoint ID between 7 and 11. However, the viewpoint ID you predict must be within the Current Environment.            (8) Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words.             Then, please make decision on the next viewpoint in the \"Prediction\".             Your decision is very important, must make it very carefully.             You need to double check the output in \"Prediction:\". The output must be in the Candidate Viewpoint IDs without any other words.             You also need to double check the output in \"Thought\". The output must be a single paragraph.             After finished all the above steps, you need to estimate the completion of the instruction based on the 'Instruction', 'Next instruction', 'Landmarks', your past 'Navigation History', 'Current Environment', and the provided images.             Please think carefully about the 'Distance' when you estimate the completion of the instruction. If your current distance to the destination is very far, you should answer 'No'.             If your current distance to the destination is close and you think you are ready to walk towards the landmarks of next instruction, you should answer 'Yes'."
MAPGPT_NAVIGATOR_USER = "Candidate Viewpoint IDs List: [{}] Instruction: {} Landmarks: {} Navigation History: {} Next instruction: {}             Current Environment: {} -> Thought: ... Distance: ... Prediction: ... Completion Estimation: ... "


def build_navigator_user(
    candidate_ids: str,
    instruction: str,
    landmarks: str,
    history_traj: str,
    next_instruction: str,
    observation: str,
) -> str:
    """Mirror move_to_next_vp_single (spatialNavigator.py:120) format order:
    (observe_dict.keys(), instruction, landmarks, history_traj,
     next_instruction, observation).
    """
    return MAPGPT_NAVIGATOR_USER.format(
        candidate_ids, instruction, landmarks, history_traj, next_instruction, observation
    )


def parse_navigator(text: str) -> tuple[str, str, str]:
    """Verbatim parser from move_to_next_vp_single (spatialNavigator.py:141-194).

    Returns (pred_vp, pred_thought, completion_estimation). pred_vp is the
    numeric direction id (regex-extracted); "" if no Prediction marker.
    """
    decision_reasoning = str(text).replace("**", "")
    if "Prediction:" not in decision_reasoning:
        return "", decision_reasoning.strip(), "Unknown"

    # Use the LAST "Prediction:" (models sometimes output multiple in thinking)
    parts = decision_reasoning.split("Prediction:")
    if len(parts) > 1:
        pred_thought = "Prediction:".join(parts[:-1]).strip()
        remaining_text = parts[-1].strip()
    else:
        pred_thought = ""
        remaining_text = decision_reasoning.strip()

    if "Completion Estimation:" in remaining_text:
        pred_vp = (
            remaining_text.split("Completion Estimation:")[0]
            .strip()
            .replace('"', "")
            .replace("'", "")
            .replace("\n", "")
            .replace(".", "")
            .replace("*", "")
        )
        completion_est = remaining_text.split("Completion Estimation:")[1].strip()
    else:
        pred_vp = (
            remaining_text.replace('"', "")
            .replace("'", "")
            .replace("\n", "")
            .replace(".", "")
            .replace("*", "")
        )
        completion_est = "Unknown"

    numeric_match = re.search(r"(?:Direction|Viewpoint)?\s*(\d+)", pred_vp)
    if numeric_match:
        pred_vp = numeric_match.group(1)
    else:
        digit_match = re.search(r"\d+", pred_vp)
        if digit_match:
            pred_vp = digit_match.group()

    # The completion estimate is returned RAW (post the response-wide "**"
    # strip + .strip(), exactly as upstream). The judge gate is a strict
    # ``== "Yes"`` (base_il_trainer_llm.py:728) — "Yes.", "yes" etc. do NOT
    # fire the judge upstream, so no normalisation here (grill 2026-07-02).
    return pred_vp, pred_thought, completion_est


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — the judge / back-check (decision_agent.py). The eval loop uses
# make_informed_decision_with_capture (base_il_trainer_llm.py:764), NOT the
# dead Open_Nav.judge() / JUDGE_PROMPT. We reproduce that path:
#   system   = JUDGE_SYSTEM
#   user     = enhanced_prompt + "\n\n" + JUDGE_FORMAT_INSTRUCTIONS
# The per-capability "understanding" blocks (descriptions dict, lines 182-235)
# AND the AST "logic analysis" meta-lines are both reproduced verbatim. The
# logic lines are byte-faithful to upstream's *runtime* output: at run time
# analyze_capability_logic() does ``ast.parse(inspect.getsource(staticmethod))``
# on a still-class-indented source, which raises ``IndentationError`` and falls
# into the ``except`` → every capability's block degenerates to empty
# ``Operations:`` / ``Has conditions: False`` / ``Complexity: unknown``
# (verified by static reproduction of the AST path, 2026-06-22). So the upstream
# judge never sees real operation lists — only this constant degenerate block.
# ═══════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are a code-aware navigation decision agent that understands implementation details."
)

# NavigationDecisionParser.get_format_instructions (decision_agent.py:96-99)
JUDGE_FORMAT_INSTRUCTIONS = """Format your response as:
Decision: [Continue/Stay/Backtrack/Look Around]
Confidence: [0-10]
Reasoning: [Your detailed reasoning]"""

# understand_capability tool output (decision_agent.py:352-364) over the
# static descriptions dict (navigation_capabilities.py:182-235).
_CAP_UNDERSTANDING = {
    "continue": """
Capability: Continue to Next Instruction
Purpose: Progress to the next sub-instruction when current is completed
Effects: Increments action index, Resets navigation history, Clears visual memory, Starts fresh for new sub-goal
When to use: When current landmarks have been found and instruction is satisfied
Side effects: Loses context from current instruction
""",
    "stay": """
Capability: Stay with Current Instruction
Purpose: Continue working on the current sub-instruction
Effects: Maintains current action index, Preserves navigation history, Continues accumulating evidence, Increments attempt counter
When to use: When current instruction is not yet satisfied
Side effects: May lead to loops if stuck
""",
    "backtrack": """
Capability: Backtrack to Previous Position
Purpose: Undo last movement and return to previous position
Effects: Reverses last physical movement, Removes last history entry, Removes last visual observation, Provides opportunity to try different path
When to use: When current path seems wrong or dead-end reached
Side effects: Loses progress, may increase total steps
""",
    "look_around": """
Capability: Look Around Comprehensively
Purpose: Gather detailed information from all viewpoints
Effects: Analyzes all available viewpoints, Builds spatial map, Identifies all visible landmarks, Provides comprehensive scene understanding
When to use: When confused or need more information to decide
Side effects: Takes additional time, no physical movement
""",
}

# analyze_capability_logic tool output (decision_agent.py:366-374). At run time
# the AST parse raises IndentationError on the class-indented staticmethod source
# (analyze_capability_logic returns ``{'error': ...}``), so every capability
# block degenerates to this exact constant (verified 2026-06-22). The ``{cap}``
# token is the capability name, matching capability_analysis's dict keys.
_CAP_LOGIC = {
    cap: (
        f"\nLogic analysis for {cap}:\n"
        "- Operations: \n"
        "- Has conditions: False\n"
        "- Complexity: unknown\n"
    )
    for cap in ("continue", "stay", "backtrack", "look_around")
}

# Compact in-context examples embedded in make_informed_decision_with_capture
# (decision_agent.py:722-742) — verbatim.
_JUDGE_EXAMPLES = """## In-Context Learning Examples:

### CONTINUE (Resets history, moves to next sub-instruction):
- "Walk through doorway" done → "Turn left": Doorway passed, ready for turn = Continue (9/10)
- "Go upstairs" done → "Find door #2": At top, doors visible = Continue (8/10)
- "Exit room" done → "Go to kitchen": Outside room, hallway ahead = Continue (9/10)

### STAY (Preserves context, continues current):
- "Find fireplace room": In living room, no fireplace yet = Stay (6/10)
- "Pass 3 doors": Passed 2/3 doors = Stay (7/10)
- "Reach hallway end": Midway through = Stay (7/10)

### BACKTRACK (Reverses last action):
- "Turn right": Turned left instead = Backtrack (9/10)
- "Blue wall room": Entered white room = Backtrack (8/10)
- "Glass door": Went through wood door = Backtrack (9/10)

### LOOK AROUND (Explores viewpoints, no movement):
- "Find piano room": Multiple rooms, unclear = Look Around (5/10)
- "Go to kitchen": Multiple paths available = Look Around (4/10)
- "Find stairs down": Large area, not visible = Look Around (5/10)
"""


def build_judge_user(
    current_action: str,
    landmarks: str,
    history_traj: str,
    num_images: int,
    descriptions: list[str] | None = None,
) -> str:
    """Reproduce make_informed_decision_with_capture's enhanced_prompt
    (decision_agent.py:703-754) + "\\n\\n" + format instructions, including the
    (degenerate) AST logic-analysis lines and the prepended "Image sequence
    descriptions" block (decision_agent.py:766-784), byte-faithful to upstream."""
    enhanced_prompt = f"""
I have analyzed the implementation of each navigation capability:

CONTINUE Implementation:
{_CAP_UNDERSTANDING["continue"]}
{_CAP_LOGIC["continue"]}

STAY Implementation:
{_CAP_UNDERSTANDING["stay"]}
{_CAP_LOGIC["stay"]}

BACKTRACK Implementation:
{_CAP_UNDERSTANDING["backtrack"]}
{_CAP_LOGIC["backtrack"]}

LOOK_AROUND Implementation:
{_CAP_UNDERSTANDING["look_around"]}
{_CAP_LOGIC["look_around"]}

{_JUDGE_EXAMPLES}
Given this understanding of what each action actually does in the code,
and considering the current context:
- Instruction: {current_action}
- Landmarks to find: {landmarks if landmarks else "None"}
- History: {history_traj}
- Images analyzed: {num_images}

What is the most appropriate navigation decision?
Consider the actual code effects, not just the conceptual purpose.
Match your situation to the examples above.
"""
    # Prepend the per-image "Image sequence descriptions" block exactly as
    # make_informed_decision_with_capture does (decision_agent.py:766-773):
    # final_prompt = descriptions_text + "\n" + enhanced_prompt.
    final_prompt = enhanced_prompt
    if descriptions:
        descriptions_text = "\n\nImage sequence descriptions:\n"
        for i, desc in enumerate(descriptions):
            descriptions_text += f"Image {i}: {desc}\n"
        final_prompt = descriptions_text + "\n" + enhanced_prompt
    return final_prompt + "\n\n" + JUDGE_FORMAT_INSTRUCTIONS


def parse_judge(text: str) -> tuple[str, float, str]:
    """Verbatim NavigationDecisionParser.parse (decision_agent.py:59-93).

    Returns (decision_str, confidence, reasoning). decision_str is the raw
    token (Continue/Stay/Backtrack/Look Around); default "Stay".
    """
    text = str(text).replace("**", "").strip()

    decision = "Stay"
    if "Decision:" in text:
        decision_line = text.split("Decision:")[1].split("\n")[0].strip()
        decision = decision_line.split(",")[0].strip()

    confidence = 5.0
    if "Confidence:" in text:
        try:
            conf_line = text.split("Confidence:")[1].split("\n")[0].strip()
            conf_str = re.findall(r"\d+\.?\d*", conf_line)[0]
            confidence = float(conf_str)
            confidence = max(0, min(10, confidence))
        except Exception:
            pass

    if "Reasoning:" in text:
        reasoning = text.split("Reasoning:")[1].strip()
    elif "Thought:" in text:
        reasoning = text.split("Thought:")[1].strip()
    else:
        reasoning = text

    return decision, confidence, reasoning


def string_to_decision(decision_str: str) -> str:
    """Verbatim _string_to_decision (decision_agent.py:834-844). Returns one
    of "continue" / "stay" / "backtrack" / "look_around"."""
    s = decision_str.strip().upper()
    if "CONTINUE" in s or "YES" in s:
        return "continue"
    if "BACKTRACK" in s:
        return "backtrack"
    if "LOOK" in s or "AROUND" in s:
        return "look_around"
    return "stay"


def apply_decision_rules(
    decision: str,
    confidence: float,
    enabled_abilities: list[str],
    current_action_idx: int,
    total_actions: int,
    num_images: int,
) -> str:
    """Verbatim _apply_decision_rules (decision_agent.py:897-941). `decision`
    in/out are canonical lowercase ("continue"/"stay"/"backtrack"/"look_around").
    """
    if decision not in enabled_abilities:
        decision = string_to_decision(enabled_abilities[0])

    if confidence < 5 and "look_around" in enabled_abilities:
        return "look_around"
    if confidence < 5 and "stay" in enabled_abilities:
        return "stay"

    if (
        decision == "continue"
        and current_action_idx == total_actions - 1
        and "stay" in enabled_abilities
    ):
        return "stay"

    if num_images < 2 and "stay" in enabled_abilities:
        return "stay"

    return decision
