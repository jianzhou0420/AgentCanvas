"""SmartWay (IROS 2025) prompt assembly — verbatim helpers + constants.

Source (verbatim from upstream SmartWay-Code @ daa2dd8):

    vlnce_baselines/GPT/one_stage_prompt_manager.py    OneStagePromptManager class
    vlnce_baselines/common/prompt.py                   per-step prompt strings
    vlnce_baselines/common/base_il_trainer.py:167-204  make_equiv_action

Re-fetch upstream: workspace/nodesets/_upstream/smartway-code/fetch_upstream.sh

Paper: https://arxiv.org/abs/2503.10069 (Shi, Li, Lyu, Xia, Dayoub, Qiao, Wu).
Author-relationship note: this is an Adelaide Qi Wu group paper; per repo
``vln-methods.md`` § 3.2 it lives as a side-experiment port, not PortBench v1.

This module contains only paradigm-independent pieces — verbatim string
constants and pure helper functions. All mutable per-episode state lives
in the AgentCanvas ``graph_state`` container, set/read by the canvas nodes
in ``__init__.py``. The upstream class's instance attributes
(``self.history``, ``self.planning``, ``self.backtrack`` …) map to
named state entries with explicit lifetimes — see ``smartway_ce.json``.
"""

from __future__ import annotations

import math
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
# Verbatim constants from upstream
# ═══════════════════════════════════════════════════════════════════════

# one_stage_prompt_manager.py:20 — initial planning entry; the prompt
# template reads ``self.planning[i][-1]``, so this is the "Previous Planning"
# the LLM sees on step 0.
DEFAULT_PLANNING: str = "Navigation has just started, with no planning yet."

# one_stage_prompt_manager.py:234 — shown as "History" on step 0.
INIT_HISTORY: str = "The navigation has just begun, with no history."

# one_stage_prompt_manager.py:110 — synthetic return-option phrase used
# both in the action_prompts list and as the latch trigger in
# ``make_history`` (line 183: ``if last_action == "..." → self.backtrack = True``).
# DO NOT change the wording — the latch lookup is exact-string match.
RETURN_PHRASE: str = "Move back to last position in an opposite direction"

# one_stage_prompt_manager.py:205-230 — the eight components of
# ``task_description``. Concatenated in :build_task_description() below.
_BACKGROUND = "You are an embodied robot that navigates in the real world."

_BACKGROUND_SUPP = (
    "You need to explore between some places marked with IDs and ultimately"
    " find the destination to stop."
    " At each step, a series of images corresponding to the places you have"
    " observed will be provided to you."
)

_INSTR_DES = (
    "'Instruction' is a global, step-by-step detailed guidance that describes"
    " the correct navigation path. However, some steps may have already been"
    " executed. Your goal is to determine which parts of the 'Instruction'"
    " have **already been completed** and which remain **to be executed**."
)

_HISTORY_DESC = (
    "'History' represents the places you have already explored along with"
    " their corresponding images. It includes both correct movements according"
    " to the 'Instruction' and some past mistaken explorations.  \n"
    "        **You must use 'History' to verify whether an instruction step"
    " has already been completed.** Seeing an object in an image does **not**"
    " mean you have not yet passed it; instead, confirm whether the object"
    " was previously observed **from a past location** before assuming that"
    " step remains incomplete."
)

_OPTION_DESC = (
    "'Action options' are the set of available actions at this step. Each"
    " action corresponds to a place, an image, and detected objects. After"
    " moving for a while, a 'stop' option will appear. Additionally, you have"
    " an option to return to the previous location if you believe you have"
    " made a mistake or need to explore an alternative path."
)

_PRE_PLANNING = (
    "'Previous Planning' records prior multi-step navigation strategies."
    " **Your goal is to refine and update this plan rather than discard it"
    " unless a critical mistake has occurred.**"
)

_REQUIREMENT = (
    "For each provided image, **analyze it in conjunction with 'Instruction'"
    " and 'History'** to determine:  \n"
    "        1. What **parts of the instruction have already been executed**?  \n"
    "        2. What **steps remain to be executed**?  \n"
    "        3. Whether your **current position is still aligned with the"
    " instruction** or if you have deviated.  \n"
    "        Your reasoning must be based on **actual past movements, not"
    " just object visibility in images**."
)

_RETURN_POLICY = (
    "If you **detect a navigation error**, or if your current path does not"
    " align well with the 'Instruction', you should consider returning to the"
    " last position using the 'Move back to last position in an opposite"
    " direction' action.  \n        "
)

_THOUGHT = (
    "Your response must be in JSON format with three fields:  \n"
    "        1. 'Thought': Explain your reasoning by integrating 'Instruction',"
    " 'History', 'Previous Planning', and 'Action options'. Clearly state"
    " **which steps are completed and which remain**, and ensure your next"
    " move continues executing the instruction correctly.  \n"
    "        2. 'New Planning': Update your multi-step path planning based on"
    " 'Previous Planning' and your reasoning in 'Thought'. Do not modify"
    " completed steps; only refine future steps.  \n"
    "        3. 'Action': Choose a single capital letter corresponding to an"
    ' action from \'Action options\'. Example: `"Action": "A"`.'
)


def build_task_description() -> str:
    """Verbatim ``task_description`` from ``make_r2r_json_prompts`` (line 232).

    The task description is **static** — it doesn't depend on episode state.
    Built once and reused every step. The seven components are concatenated
    with newlines in the same order as upstream.
    """
    return (
        f"{_BACKGROUND} {_BACKGROUND_SUPP}\n"
        f"{_INSTR_DES}\n"
        f"{_HISTORY_DESC}\n"
        f"{_PRE_PLANNING}\n"
        f"{_OPTION_DESC}\n"
        f"{_RETURN_POLICY}\n"
        f"{_REQUIREMENT}\n"
        f"{_THOUGHT}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Pure helper functions
# ═══════════════════════════════════════════════════════════════════════


def get_action_concept_backtrack(rel_heading: float) -> str:
    """Heading delta (rad) → English direction phrase.

    Verbatim port of ``OneStagePromptManager.get_action_concept_backtrack``
    (one_stage_prompt_manager.py:23-41). Six 60° buckets covering [0, 360);
    the off-by-half "turn around" bucket at 150-210° matches upstream.
    """
    deg = math.degrees(rel_heading) % 360.0
    if 0 <= deg < 30 or deg >= 330:
        return "go forward"
    if 30 <= deg < 90:
        return "turn slight left"
    if 90 <= deg < 150:
        return "turn sharp left"
    if 150 <= deg <= 210:
        return "turn around"
    if 210 < deg < 270:
        return "turn sharp right"
    if 270 <= deg < 330:
        return "turn slight right"
    # Should be unreachable after the modulo above.
    raise AssertionError(f"Unexpected angle deg={deg:.2f}")


def make_action_prompts(
    candidates_dict: dict[int, dict[str, Any]],
    res_list: list[str],
    node_indices: list[int],
    *,
    t: int,
    last_backtrack: bool,
) -> list[str]:
    """Build per-candidate action prompt strings.

    Mirrors the inner loop of ``make_action_prompt_backtrackv2``
    (one_stage_prompt_manager.py:98-113) — the per-candidate phrase building.

    For each candidate ``cc`` with angle/distance:

    * If this is the trailing synthetic backtrack entry (``j == K-1`` AND
      ``last_backtrack=False`` AND ``t != 0``), emit the verbatim
      ``RETURN_PHRASE``.
    * Otherwise emit
      ``f"{direction} to Place {node_idx} which is corresponding to Image"
      f" {node_idx}, and this image contains objects such as {tags}."``

    ``node_idx`` is the **episode-global** index assigned when the candidate's
    waypoint UUID was appended to ``nodes_list`` (upstream:
    ``node_index = nodes_list[0].index(waypoint_id)``). It must be stable
    across steps because ``make_history`` writes ``last_action`` (which
    contains this Place ID) into the history string the LLM reads next
    step. Per-step indices ``j`` would make history references meaningless.
    """
    n = len(candidates_dict)
    out: list[str] = []
    for j, cc in candidates_dict.items():
        direction = get_action_concept_backtrack(float(cc["angle"]))
        is_synthetic_return = (
            not last_backtrack and j == n - 1 and t != 0 and bool(cc.get("type") == "return")
        )
        if is_synthetic_return:
            out.append(RETURN_PHRASE)
        else:
            tag = res_list[j] if 0 <= j < len(res_list) else ""
            node_idx = node_indices[j]
            out.append(
                f"{direction} to Place {node_idx} which is corresponding to Image {node_idx},"
                f" and this image contains objects such as {tag}."
            )
    return out


def prepend_stop_options(
    action_prompts: list[str],
    *,
    t: int,
) -> tuple[list[str], list[str]]:
    """Mirror ``make_action_options_backtrack`` (line 148-165).

    When ``t >= 2``, prepend ``'stop'`` to ``action_prompts`` so the LLM
    can choose to stop after at least two real steps. Letter-prefix every
    entry with A, B, C, …. Returns
    ``(letter_prefixed_options, just_letters)``.
    """
    prompts = list(action_prompts)
    if t >= 2:
        prompts = ["stop", *prompts]
    full = [f"{chr(j + 65)}. {prompts[j]}" for j in range(len(prompts))]
    letters = [chr(j + 65) for j in range(len(prompts))]
    return full, letters


def assemble_prompt(
    *,
    instruction: str,
    history: str,
    planning_latest: str,
    action_options: list[str],
    t: int,
) -> str:
    """Verbatim port of the ``prompt`` f-string in ``make_r2r_json_prompts``
    (one_stage_prompt_manager.py:242-245).
    """
    hist_field = INIT_HISTORY if t == 0 else history
    return (
        f"Instruction: {instruction}\n"
        f"History: {hist_field}\n"
        f"Previous Planning:\n{planning_latest}\n"
        f"Action options (step {t}): {action_options}"
    )


def parse_json_action(
    json_output: dict[str, Any],
    only_options: list[str],
    *,
    t: int,
) -> int:
    """Verbatim port of ``parse_json_action`` (one_stage_prompt_manager.py:294-309).

    Returns ``output_index`` per upstream semantics:

    * If ``json_output["Action"]`` is a letter in ``only_options``,
      ``output_index = only_options.index(letter)``.
    * Else fallback to ``0``.
    * After this, if ``t >= 2``, ``output_index -= 1`` to absorb the
      prepended ``'stop'`` (so ``-1`` means STOP, ``0+`` indexes into the
      original candidate list).
    """
    try:
        output = str(json_output.get("Action", "")).strip().rstrip(".").upper()
        output_index = only_options.index(output) if output in only_options else 0
    except Exception:
        output_index = 0

    if t >= 2:
        output_index -= 1

    return output_index


def parse_json_planning(json_output: dict[str, Any]) -> str:
    """Verbatim port of ``parse_json_planning`` (line 284-291).

    Returns the new planning text. ``"No plans currently."`` on parse
    failure. Caller appends this to the ``planning`` state's history.
    """
    try:
        return str(json_output["New Planning"])
    except Exception:
        return "No plans currently."
