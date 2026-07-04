"""Verbatim AO-Planner prompts + assemblers/parsers.

All prompt text below is quoted character-for-character from AO-Planner
@ 719f42a1 (`llm/grounded_sam_Gemini.py::query_llm` for the VLM#1 proposer;
`llm/prompting/prompt_manager.py::make_graph_baseline_prompts` for the VLM#2
PathAgent). Both upstream schemas are PROSE-ONLY — there is no literal
``{...}`` example in either prompt, so none is added here (faithfulness).
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Initial state strings (verbatim) ──────────────────────────────────────
# planning[-1] seed (zero_shot_agent rollout init) and the step-0 history line.
DEFAULT_PLANNING = "Navigation has just started, with no planning yet."
INIT_HISTORY = "The navigation has just begun, with no history."
# Upstream sentinel appended when a step yields no 'New Planning'
# (zero_shot_agent.py:818-821). The port previously appended "".
NO_PLANNING_FALLBACK = "No planning in last step."

# dir_id 0/1/2/3 → AO-Planner direction words (front/backward/left/right order
# is the upstream Options wording; the actual mapping to panorama bins is set
# by the graph wiring / aggregator).
DIRECTIONS = ["front", "backward", "left", "right"]


# ══════════════════════════════════════════════════════════════════════
# VLM#1 — waypoint proposer (grounded_sam_Gemini.py::query_llm)
# ══════════════════════════════════════════════════════════════════════

_PROP_BACKGROUND = (
    "You are a robot and need to identify potential 'Waypoints' and corresponding "
    "'Paths' in the environment from the current observed image."
)
_PROP_WAYPOINT_DEF = (
    "'Waypoints' refer to locations that can be reached and meet the following "
    "conditions. 1. They are on the ground and maintain a reasonable distance from "
    "obstacles to avoid collisions. 2. Ideally, they occupy crucial positions at the "
    "center of different regions and can be connected to various regions. 3. Select "
    "the most representative waypoints (up to 3), preferably not too close to each other."
)
_PROP_WAYPOINT_OUTPUT = (
    "Some position candidates on the ground are annotated with IDs in the image. You "
    "need to select some of them and provide the IDs as your selected 'Waypoints'."
)
_PROP_PATH_OUTPUT = (
    "For these 'Waypoints', you also need to select some positions that need to be "
    "passed through to reach each selected waypoint. For each path, you can start from "
    "any of the points in the bottom row of the image. You need to ensure that "
    "connecting the selected positions in order can form some shortest 'Paths' that "
    "lead to the 'Waypoints' while navigating around obstacles to avoid collisions."
)
_PROP_OUTPUT_REQUIREMENT = (
    "You should return a JSON object that has the fields 'Waypoints' (a list recording "
    "waypoints) and 'Paths' (a list recording paths to each waypoint)."
)


def _prop_instr_des(instruction: str) -> str:
    return (
        f"'Instruction': '{instruction}', is a step-by-step detailed guidance for "
        "navigation, but you might have already executed some of the commands. If key "
        "information from the 'Instruction', such as scene descriptions, landmarks, and "
        "objects, appears in the observed image, select the corresponding waypoint and path."
    )


def build_proposer_system(instruction: str = "") -> str:
    """Assemble the VLM#1 proposer system prompt (verbatim separators).

    With instruction:    bg waypoint_def \\n\\n wp_out path_out \\n\\n instr_des \\n\\n out_req
    Without instruction: bg waypoint_def \\n\\n wp_out path_out \\n\\n out_req
    """
    head = f"{_PROP_BACKGROUND} {_PROP_WAYPOINT_DEF}\n\n{_PROP_WAYPOINT_OUTPUT} {_PROP_PATH_OUTPUT}"
    if instruction:
        return f"{head}\n\n{_prop_instr_des(instruction)}\n\n{_PROP_OUTPUT_REQUIREMENT}"
    return f"{head}\n\n{_PROP_OUTPUT_REQUIREMENT}"


def parse_proposal(text: str) -> dict[str, Any]:
    """Parse the proposer's {Waypoints, Paths} JSON block.

    Mirrors AO-Planner `parse_results` (fenced or bare json -> json.loads), then
    normalizes the two Paths shapes gpt-5-mini emits (bucket-C, model-forced):
    upstream-style list-of-lists, or list-of-objects {"Waypoint": id, "Path":
    [id,...]}. Returns {"waypoints": [int...], "paths": [[int...]...]} with `paths`
    aligned 1:1 to `waypoints` (object-form routes mapped by their Waypoint id,
    list-form by position). Empty on parse failure.
    """
    if not text:
        return {"waypoints": [], "paths": []}
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {"waypoints": [], "paths": []}
    if not isinstance(data, dict):
        return {"waypoints": [], "paths": []}

    def _ints(seq: Any) -> list[int]:
        """Extract integer IDs, tolerant of the shapes gpt-5-mini actually emits:
        bare ints, numeric strings (``"6"``), ``"id_2"``-style labels (first
        integer), or element objects like ``{"id": 2}``.

        Bucket-C, MODEL-FORCED (2026-06-17) — this is NOT decorative: the unified
        gpt-5-mini proposer returns string/object IDs, unlike upstream's Gemini
        bare list-of-ints, so without this the Waypoints and Paths collapse and
        D-2 multi-hop never fires (verified on run 20260617_165734). Re-filed from
        the earlier mistaken D-6 'strict' trim. Pure labels with no digit
        (``"bottom_center"``) have no grid id and are dropped.
        """
        out: list[int] = []
        if not isinstance(seq, list):
            seq = [seq]
        for x in seq:
            if isinstance(x, dict):
                for k in ("id", "ID", "Id", "index", "Index", "waypoint", "Waypoint"):
                    if k in x:
                        x = x[k]
                        break
                else:
                    continue
            try:
                out.append(int(x))
                continue
            except (TypeError, ValueError):
                pass
            mm = re.search(r"-?\d+", str(x))
            if mm:
                out.append(int(mm.group()))
        return out

    waypoints = _ints(data.get("Waypoints") or data.get("waypoints"))
    paths_raw = data.get("Paths") or data.get("paths") or []
    # gpt-5-mini emits Paths two ways (bucket-C, model-forced): upstream-style
    # list-of-lists [[id,...],...], or list-of-objects [{"Waypoint": id, "Path":
    # [id,...]}]. Map object-form routes to their waypoint id; keep list-form
    # positional. Both are aligned to `waypoints` below (object order != waypoint
    # order in practice, so positional pairing alone mis-associates routes).
    route_by_wp: dict[int, list[int]] = {}
    positional: list[list[int]] = []
    if isinstance(paths_raw, list):
        for p in paths_raw:
            if isinstance(p, dict):
                route = p.get("Path") or p.get("path") or p.get("Route") or p.get("route") or []
                r = _ints(route) if isinstance(route, list) else _ints([route])
                wp = _ints([p.get("Waypoint", p.get("waypoint"))])
                if wp:
                    route_by_wp[wp[0]] = r
                else:
                    positional.append(r)
            elif isinstance(p, list):
                positional.append(_ints(p))
            else:
                positional.append(_ints([p]))
    # Upstream count-mismatch branch (llm/utils.py:43-56): for list-form Paths
    # whose count differs from Waypoints, ignore the Waypoints list and use each
    # path's LAST point as its destination ("use path[-1] as waypoint instead").
    # Object-form routes (gpt-5-mini) carry their own Waypoint id, so the branch
    # only applies when every route is positional.
    if not route_by_wp and positional and len(waypoints) != len(positional):
        kept = [r for r in positional if r]
        return {"waypoints": [r[-1] for r in kept], "paths": kept}
    # One route per waypoint: by waypoint id (object form) else positional.
    paths: list[list[int]] = []
    for i, w in enumerate(waypoints):
        if w in route_by_wp:
            paths.append(route_by_wp[w])
        elif i < len(positional):
            paths.append(positional[i])
        else:
            paths.append([])
    return {"waypoints": waypoints, "paths": paths}


# ══════════════════════════════════════════════════════════════════════
# VLM#2 — PathAgent (prompt_manager.py::make_graph_baseline_prompts)
# ══════════════════════════════════════════════════════════════════════

_PA_BACKGROUND = "You are an embodied robot that navigates in the real world."
_PA_BACKGROUND_SUPP = (
    "You need to explore between some locations marked with IDs and ultimately find "
    "the destination to stop. At each step, a series of images corresponding to the "
    "locations you have explored and have observed will be provided to you."
)
_PA_INSTR_DES = (
    "'Instruction' is a global, step-by-step detailed guidance, but you might have "
    "already executed some of the commands. You need to carefully discern the commands "
    "that have not been executed yet."
)
_PA_HISTORY = (
    "'History' represents the places you have explored in previous steps along with "
    "their corresponding images. It may include the correct landmarks mentioned in the "
    "'Instruction' as well as some past erroneous explorations."
)
_PA_PRE_PLANNING = (
    "'Previous Planning' records previous long-term multi-step planning info that you "
    "can refer to now."
)
_PA_OPTION = (
    "'Options' are some navigable location IDs with some observed images from front, "
    "backward, left, and right views. You need to select one location from the set as "
    "your next move. These IDs are also marked in the provided images."
)
# NOTE: upstream `requirement` and `thought` carry a trailing space — preserved.
_PA_REQUIREMENT = (
    "For each provided image of the environments, you should combine the 'Instruction' "
    "and carefully examine the relevant information, such as scene descriptions, "
    "landmarks, and objects. You need to align 'Instruction' with 'History' to estimate "
    "your instruction execution progress. "
)
_PA_DIST_REQUIRE = (
    "If you can already see the destination, estimate the distance between you and it. "
    "If the distance is far, continue moving and try to stop within 1 meter of the "
    "destination."
)
_PA_THOUGHT = (
    "Your answer should be JSON format and must include three fields: 'Thought', 'New "
    "Planning' and 'Action'. You need to combine 'Instruction', your past 'History', "
    "'Options', and the provided images to think about what to do next and why, and "
    "complete your thinking into 'Thought'. "
)
_PA_NEW_PLANNING = (
    "Based on your 'Previous Planning' and current 'Thought', you also need to update "
    "your new multi-step planning to 'New Planning'."
)
_PA_ACTION = (
    "Place only the ID of the chosen location in 'Action'. If you think you have arrived "
    "at the destination, place 'Stop' into 'Action'."
)


def build_task_description() -> str:
    """Assemble the PathAgent system prompt (verbatim \\n joins)."""
    return (
        f"{_PA_BACKGROUND} {_PA_BACKGROUND_SUPP}\n{_PA_INSTR_DES}\n{_PA_HISTORY}\n"
        f"{_PA_PRE_PLANNING}\n{_PA_OPTION}\n{_PA_REQUIREMENT}\n{_PA_DIST_REQUIRE}\n"
        f"{_PA_THOUGHT}\n{_PA_NEW_PLANNING}\n{_PA_ACTION}"
    )


def options_line(action_space_text: str, t: int) -> str:
    """The verbatim Options content item (prompt_manager.py:65/75)."""
    return f"Options (step {t}): Locations {{{action_space_text}}}\n"


def assemble_pathagent_prompt(
    instruction: str,
    planning_latest: str,
    history: str,
    action_space_text: str,
    t: int,
) -> str:
    """Build the PathAgent user prompt text (verbatim layout).

    Step 0:  Instruction / History: <init> / Previous Planning / Options
             (one text block, then the option images — upstream user_content[0])
    Step >0: Instruction / Previous Planning / History:\\n
             History images follow, then the Options line rides the FIRST
             option image's label (BuildImagesNode) so the content order is
             upstream's exactly: text → history images → Options text → option
             images (prompt_manager.py:69-79).
    """
    if t == 0:
        return (
            f"Instruction: {instruction}\nHistory: {INIT_HISTORY}\n"
            f"Previous Planning: {planning_latest}\n{options_line(action_space_text, t)}"
        )
    return f"Instruction: {instruction}\nPrevious Planning: {planning_latest}\nHistory:\n"


_DIRECTION_PHRASES: dict[str, str] = {
    "front": "move forward",
    "left": "turn left",
    "backward": "turn around",
    "right": "turn right",
}


def image_label(direction: str, node_id_text: str, i: int) -> str:
    """Per-image option caption (verbatim): '({direction}) Locations {ids} in Image {i}:'."""
    return f"({direction}) Locations {{{node_id_text}}} in Image {i}:"


def history_image_label(direction: str, action_id: int, t: int) -> str:
    """Per-history-image caption matching upstream make_graph_history format."""
    phrase = _DIRECTION_PHRASES.get(direction, "move")
    return (
        f"Step {t}, {phrase} towards the scene in the image below and "
        f"proceed along path to location {action_id}:"
    )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    try:
        d = json.loads(cleaned)
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_pathagent(text: str) -> dict[str, Any]:
    """Parse the PathAgent {Thought, New Planning, Action} JSON.

    Action handling mirrors upstream `parse_num` (`re.findall(r'\\d+', action)[0]`,
    zero_shot_agent.py:838-851): the FIRST integer is the chosen ID; STOP is
    signalled only by the *absence* of any digit (an exact 'Stop' has none, and a
    no-digit / format-error reply is treated as STOP via upstream's except path).
    So 'Stop in 2 m' moves to id 2, exactly like upstream — no 'stop' substring
    check (that was the §3E divergence; fixed 2026-06-17).

    `new_planning` falls back to the upstream sentinel when the key is absent
    (NO_PLANNING_FALLBACK), not "". `ok` is False when the reply did not parse as
    JSON at all, so the caller can leave `planning` untouched (upstream carries
    the prior planning on a total parse failure).
    """
    parsed = _extract_json(text)
    ok = bool(parsed)
    thought = str(parsed.get("Thought", parsed.get("thought", "")))
    np_raw = parsed.get("New Planning", parsed.get("new_planning"))
    new_planning = str(np_raw) if np_raw not in (None, "") else NO_PLANNING_FALLBACK
    action_raw = str(parsed.get("Action", parsed.get("action", "")))
    digits = re.findall(r"\d+", action_raw)
    is_stop = len(digits) == 0
    action_id = int(digits[0]) if digits else -1
    return {
        "thought": thought,
        "new_planning": new_planning,
        "action_id": action_id,
        "is_stop": is_stop,
        "ok": ok,
    }
