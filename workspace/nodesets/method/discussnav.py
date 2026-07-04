"""DiscussNav nodeset — multi-LLM discussion VLN port (R2R / MP3D).

Port of *Long et al., "Discuss Before Moving: Visual Language Navigation
via Multi-Expert Discussions" (ICRA 2024)*. Method-side reasoning only — no
env imports, no model loads. Vision lives in ``model_ram`` (RAM tagger,
server-mode) and ``navgpt_mp3d_tools`` (InstructBLIP, local-mode).

Pipeline per step (matches upstream ``DiscussNav.py:507-523`):

    1. Vision_Perception — RAM tag + InstructBLIP describe per direction
    2. Completion_Estimation — single GPT-4 call, history-aware
    3. DiscussNav_Agent.pred_vp — GPT-4 with n=5, retry until 5 valid
    4. Decision_Testing.thought_fusion — per-unique-vp GPT-4 (sequential)
    5. Decision_Testing.test_decisions — single deciding GPT-4 if multi-vp
    6. env step → next viewpoint
    7. Completion_Estimation.save_history — 2 sequential summarisation
       GPT-4 calls; append to nav_history state container

All prompt strings are verbatim from upstream with file:line citations.

Load:   POST /api/components/nodesets/discussnav/load
Unload: POST /api/components/nodesets/discussnav/unload

last updated: 2026-05-10
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

log = logging.getLogger("agentcanvas.discussnav")


# ══════════════════════════════════════════════════════════════════════
# Verbatim prompts — copied byte-for-byte from upstream DiscussNav.py
# ══════════════════════════════════════════════════════════════════════

# DiscussNav.py:106 (Instruction_Analysis_Experts.detect_actions)
_SYS_DETECT_ACTIONS = (
    "You are an action decomposition expert. Your task is to detect all "
    "actions in the given navigation instruction. You need to ensure the "
    "integrity of each action."
)

# DiscussNav.py:115 (Instruction_Analysis_Experts.detect_landmarks)
_SYS_DETECT_LANDMARKS = (
    "You are a landmark extraction expert. Your task is to detect all "
    "landmarks in the given navigation instruction. You need to ensure "
    "the integrity of each landmarks."
)

# DiscussNav.py:275-282 (Completion_Estimation_Experts.estimate_completion)
_SYS_ESTIMATE_COMPLETION = (
    "You are a completion estimation expert. Your task is to estimate "
    "what actions in the instruction have been executed based on "
    "navigation history and landmarks.                 "
    "All actions in the instruction are given following the temporal "
    'order. Your answer includes two parts: "Thought" and "Executed '
    'Actions".                 '
    'In the "Thought", you must follow procedures to analyze as '
    "detailed as possible what actions have been executed:                 "
    "(1) What given landmarks of actions have appeared in the navigation "
    "history?                 "
    "(2) Analyze the direction change at each step in the navigation "
    "history.                 "
    "(3) Estimate each action in the instruction based on each step in "
    "the navigation history to check their completion.                 "
    'In the "Executed Actions", you must only write down actions that '
    "have been executed without other words.                 "
    "You must strictly refer original actions in the given instruction "
    "to estimate."
)

# DiscussNav.py:298-313 (DiscussNav_Agent.pred_vp pred_definition)
_SYS_PRED_VP = (
    "You are a navigation agent who follows instruction to move in an "
    "indoor environment with the least action steps.             "
    "I will give you one instruction and tell you landmarks. I will "
    "also give you navigation history and estimation of executed "
    "actions for reference.             "
    "You can observe current environment by scene descriptions, scene "
    "objects and possible existing landmarks in different directions "
    "around you.             "
    "Each direction contains navigable viewpoints you can move to. "
    "Your task is to predict moving to which navigable viewpoint.             "
    "Note that environment direction that contains more landmarks "
    "mentioned in the instruction is usually the better choice for you.             "
    "If you are required to go up stairs, you need to move to direction "
    "with higher position. If you are required to go down stairs, you "
    "need to move to direction with lower position.             "
    "You are encouraged to move to new viewpoints to explore environment "
    "while avoid revisiting accessed viewpoints in non-essential situations.             "
    "If you feel struggling to find the landmark or execute the action, "
    "you can try to execute the subsequent action and find the subsequent landmark.             "
    'Your answer includes two parts: "Thought" and "Prediction". In '
    'the "Thought", you should think as detailed as possible following '
    "procedures:             "
    "(1) Check whether the latest executed action has been completed by "
    "comparing current environment and landmark in the latest executed action.             "
    "(2) Determine the action you should execute and landmark you "
    "should reach now. If the latest executed action have not been completed,             "
    "you should continue to execute it. Otherwise, you should execute "
    "the next action in the given instruction.             "
    "(3) Analyze which direction in the current environment is most "
    "suitable to execute the action you decide and explain your reason.             "
    "(4) Predict moving to which navigable viewpoint based on your "
    "thought process.             "
    "Then, please make decision on the next viewpoint in the "
    '"Prediction". You must only answer next viewpoint ID in the '
    '"Prediction" without other words.             '
    "Your decision is very important, must make it very carefully."
)

# DiscussNav.py:374-375 (Decision_Testing_Experts.thought_fusion)
_SYS_THOUGHT_FUSION = (
    "You are a thought fusion expert. Your task is to fuse given thought "
    "processes                     into one thought. You need to reserve "
    "key information related to actions, landmarks, direction changes. "
    "You should only answer fused thought without other words."
)

# DiscussNav.py:391-392 (Decision_Testing_Experts.test_decisions)
_SYS_TEST_DECISIONS = (
    "You are a decision testing expert. Your task is to evaluate the "
    "feasibility of each movement                     prediction based "
    "on thought process and environment. Then, you will make a final "
    "decision about navigation viewpoint ID without other words."
)

# DiscussNav.py:242 / 250 (Completion_Estimation_Experts.summarize_*)
_SYS_SUMMARIZE_OBS_OR_THOUGHT = (
    "You are a trajectory summary expert. Your task is to simplify "
    "{kind} as short and clear as possible."
)

# DiscussNav.py:237-238 (Completion_Estimation_Experts.summarize_observation)
_DIRECTION_LABELS_12 = [
    "Front, range(right 0 to right 30)",
    "Font Right, range(right 30 to right 60)",  # sic — typo preserved verbatim
    "Right, range(right 60 to right 90)",
    "Right, range(right 90 to right 120)",
    "Rear Right, range(right 120 to right 150)",
    "Rear Right, range(right 150 to right 180)",
    "Rear Left, range(left 180 to left 150)",
    "Rear Left, range(left 150 to left 120)",
    "Left, range(left 120 to left 90)",
    "Left, range(left 90 to left 60)",
    "Front Left, range(left 60 to left 30)",
    "Front Left, range(left 30 to left 0)",
]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _direction_bucket_12(heading_rad: float) -> int:
    """Map a heading-radian (relative to current pose) to the 0..11 bucket
    DiscussNav uses (12 directions x 30° each, starting at front=0)."""
    import math

    twopi = 2.0 * math.pi
    h = heading_rad % twopi
    bucket = round(h / (twopi / 12.0)) % 12
    return bucket


def _encode_rgb_b64(arr) -> str:
    from PIL import Image

    pil = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ══════════════════════════════════════════════════════════════════════
# Init: parse instruction-decompose responses → (actions, landmarks, step_budget)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavInitDecomposeNode(BaseCanvasNode):
    """Combine the two init-LLM responses into (actions, landmarks, step_budget).

    Upstream ``DiscussNav.py:494-506``: actions/landmarks come from two
    separate ``gpt_response`` calls; ``step_length`` is computed from
    the action count (``5 if len(actions.split('\\n')) <= 5 else 7``).

    The two LLMCall nodes that feed this node use the system prompts
    ``_SYS_DETECT_ACTIONS`` and ``_SYS_DETECT_LANDMARKS`` and the user
    templates::

        Can you decompose actions in the instruction "{instruction}"? Actions:
        Can you extract landmarks in the instruction "{actions}"? Landmarks:
    """

    node_type: ClassVar[str] = "discussnav__init_decompose"
    display_name: ClassVar[str] = "DiscussNav: Init Decompose"
    description: ClassVar[str] = "Combine actions + landmarks LLM responses; emit step budget"
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "ListOrdered"
    input_ports: ClassVar[list] = [
        PortDef("actions_response", "TEXT", "Response of detect_actions LLM call"),
        PortDef("landmarks_response", "TEXT", "Response of detect_landmarks LLM call"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("actions", "TEXT", "Decomposed action list (newline-separated)"),
        PortDef("landmarks", "TEXT", "Extracted landmarks"),
        PortDef("step_budget", "TEXT", "Step length (5 short / 7 long)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        actions = str(inputs.get("actions_response", "")).strip()
        landmarks = str(inputs.get("landmarks_response", "")).strip()

        # DiscussNav.py:506
        n_actions = len([ln for ln in actions.split("\n") if ln.strip()])
        step_budget = 5 if n_actions <= 5 else 7

        self._self_log("n_actions", n_actions)
        self._self_log("step_budget", step_budget)
        self._self_log("actions_preview", actions[:200])
        self._self_log("landmarks_preview", landmarks[:200])

        return {
            "actions": actions,
            "landmarks": landmarks,
            "step_budget": str(step_budget),
        }


# ══════════════════════════════════════════════════════════════════════
# Panorama → views base64 (feeds opennav_perception__tag_panorama)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavPanoramaToViewsNode(BaseCanvasNode):
    """Select & encode ONLY the navigable-direction eye-level views.

    L2 fidelity fix (2026-06-16): mirrors upstream ``observe_view``
    (``DiscussNav.py:161-204``), which captions only the views toward
    *navigable* viewpoints — not the whole 36-tile panorama. For each distinct
    12-direction bucket that holds a navigable vp, this picks the single
    eye-level (elevation≈0) tile whose absolute heading matches that direction,
    encodes it to base64 PNG, and tags it ``dir_id = bucket`` (0..11).

    Output ``view_tiles`` (and the parallel ``dir_ids``) therefore carry
    ~2-5 tiles aligned to the aggregator's 12-direction buckets — not 36 in
    arbitrary view-index order. This (a) cuts InstructBLIP/RAM work ~10x, the
    bottleneck that timed out 5-worker runs, and (b) fixes the alignment bug
    where the aggregator indexed ``caps[d]`` into a 36-list whose first 12 were
    the look-DOWN (-30°) row, feeding down-tilted captions to eye-level slots.

    Heading frames: ``view_meta.heading_deg`` is ABSOLUTE (env_mp3d
    __init__.py:952); ``navigable_json`` headings are RELATIVE to the agent
    (radians). The agent's absolute heading (``agent_heading_deg``, from
    observe_navigable) bridges them: target_abs = agent + bucket·30°.

    base64 PNG encoding is required because ``model_ram`` / ``model_instructblip``
    run server-mode in separate conda envs — numpy IMAGE arrays don't cross the
    HTTP boundary.
    """

    node_type: ClassVar[str] = "discussnav__panorama_to_views"
    display_name: ClassVar[str] = "DiscussNav: Panorama → Views"
    description: ClassVar[str] = "Select navigable-direction eye-level views, encode to base64"
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Grid3x3"
    input_ports: ClassVar[list] = [
        PortDef("views", "LIST[IMAGE]", "Per-view panorama images from env_mp3d"),
        PortDef("view_meta", "TEXT", "Per-view metadata JSON aligned 1:1 with views"),
        PortDef("navigable_json", "TEXT", "JSON {vp: {heading, elevation, distance}} (relative rad)"),
        PortDef("agent_heading_deg", "TEXT", "Agent's current absolute heading (deg)"),
        PortDef("instruction", "TEXT", "Navigation instruction (enables the stairs look-down branch)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("view_tiles", "ANY", "List of {dir_id, rgb_base64}; navigable eye-level + (stairs) look-down tiles"),
        PortDef("dir_ids", "LIST[TEXT]", "bucket id per tile ('b' eye-level, 'bd' look-down), parallel to view_tiles"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="pink")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:

        raw_views = inputs.get("views")
        views = list(raw_views) if isinstance(raw_views, list) else []
        try:
            view_meta = json.loads(str(inputs.get("view_meta", "[]")))
            if not isinstance(view_meta, list):
                view_meta = []
        except (json.JSONDecodeError, TypeError):
            view_meta = []
        try:
            navigable = json.loads(str(inputs.get("navigable_json", "{}"))) or {}
            if not isinstance(navigable, dict):
                navigable = {}
        except (json.JSONDecodeError, TypeError):
            navigable = {}
        try:
            agent_heading_deg = float(inputs.get("agent_heading_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            agent_heading_deg = 0.0

        if not views or not view_meta or not navigable:
            self._self_log("error", f"views={len(views)} meta={len(view_meta)} nav={len(navigable)}")
            return {"view_tiles": [], "dir_ids": []}

        # Distinct 12-direction buckets that hold a navigable viewpoint.
        nav_buckets: set[int] = set()
        for _vp, info in navigable.items():
            try:
                rel = float(info.get("heading", info.get("heading_rad", 0.0)))
            except (TypeError, ValueError):
                continue
            nav_buckets.add(_direction_bucket_12(rel))

        # Eye-level (elevation≈0) tiles only.
        eye_level = [
            (i, m) for i, m in enumerate(view_meta)
            if isinstance(m, dict) and abs(float(m.get("elevation_deg", 99))) <= 15.0
        ]
        if not eye_level:
            self._self_log("error", "no eye-level tiles in view_meta")
            return {"view_tiles": [], "dir_ids": []}

        def _circ(a: float, b: float) -> float:
            d = abs((a - b) % 360.0)
            return min(d, 360.0 - d)

        view_tiles: list[dict] = []
        dir_ids: list[str] = []
        for b in sorted(nav_buckets):
            target_abs = (agent_heading_deg + b * 30.0) % 360.0
            best_i = min(
                eye_level,
                key=lambda im: _circ(float(im[1].get("heading_deg", 0.0)), target_abs),
            )[0]
            if best_i >= len(views):
                continue
            view_tiles.append({"dir_id": str(b), "rgb_base64": _encode_rgb_b64(views[best_i])})
            dir_ids.append(str(b))

        # Stairs look-down (DiscussNav.py:189-200): when the instruction mentions
        # stairs, additionally emit the -30° tile per navigable direction, tagged
        # dir_id=f"{b}d". Upstream RAM-tags the look-down only (no caption); here
        # the tile rides the same list so RAM tags it, and the aggregator uses
        # only its tag (the look-down caption is computed-but-discarded).
        instr = str(inputs.get("instruction", "")).lower()
        if ("stair" in instr) or ("the steps" in instr):
            down_level = [
                (i, m) for i, m in enumerate(view_meta)
                if isinstance(m, dict) and -45.0 <= float(m.get("elevation_deg", 99)) <= -15.0
            ]
            for b in sorted(nav_buckets):
                if not down_level:
                    break
                target_abs = (agent_heading_deg + b * 30.0) % 360.0
                di = min(
                    down_level,
                    key=lambda im: _circ(float(im[1].get("heading_deg", 0.0)), target_abs),
                )[0]
                if di >= len(views):
                    continue
                view_tiles.append({"dir_id": f"{b}d", "rgb_base64": _encode_rgb_b64(views[di])})
                dir_ids.append(f"{b}d")

        self._self_log("n_navigable_dirs", len(view_tiles))
        self._self_log("dir_ids", ",".join(dir_ids))
        return {"view_tiles": view_tiles, "dir_ids": dir_ids}


# ══════════════════════════════════════════════════════════════════════
# Observation aggregator: tags + captions + nav_json → DiscussNav-format obs + manifest
# ══════════════════════════════════════════════════════════════════════


class DiscussNavObservationAggregatorNode(BaseCanvasNode):
    """Fold per-direction tags + captions into the DiscussNav Observation text.

    Output ``observation`` matches the upstream string exactly per direction
    (``DiscussNav.py:185-187``)::

        Direction {d} {elev_flag} Navigable Viewpoint ID: {vp} [(Passed Area)]
        Elevation: Eye Level Scene Description: {caption} Scene Objects: {tags};

    Output ``manifest`` is a JSON list of valid ``{direction_id, vp,
    elevation_flag, passed}`` records — used by the parser to validate
    the LLM's predicted vp string.

    Reads ``nav_history`` from the state container to mark already-visited
    viewpoints as ``(Passed Area)``.
    """

    node_type: ClassVar[str] = "discussnav__observation_aggregator"
    display_name: ClassVar[str] = "DiscussNav: Observation Aggregator"
    description: ClassVar[str] = "Build per-direction observation text + vp manifest"
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "AlignLeft"
    input_ports: ClassVar[list] = [
        PortDef("tags_per_dir", "LIST[TEXT]", "RAM tag strings, aligned 1:1 with dir_ids"),
        PortDef("captions_per_dir", "LIST[TEXT]", "InstructBLIP descriptions, aligned 1:1 with dir_ids"),
        PortDef("dir_ids", "LIST[TEXT]", "12-direction bucket id per tag/caption (from panorama_to_views)"),
        PortDef("navigable_json", "TEXT", "JSON {vp: {heading_rad, elevation_rad, distance}}"),
        PortDef("instruction", "TEXT", "Navigation instruction (for stairs detection)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("observation", "TEXT", "DiscussNav observation block (12 directions)"),
        PortDef("manifest", "TEXT", "JSON manifest of valid (direction, vp) pairs"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        tags_list = inputs.get("tags_per_dir") or []
        caps_list = inputs.get("captions_per_dir") or []
        dir_ids = inputs.get("dir_ids") or []
        nav_raw = inputs.get("navigable_json") or "{}"
        instruction = str(inputs.get("instruction", ""))

        # captions/tags arrive aligned 1:1 with dir_ids (the navigable-direction
        # buckets panorama_to_views selected) — NOT positionally by bucket. Map
        # them back to their 12-direction bucket so direction d gets ITS view.
        cap_by_bucket: dict[int, str] = {}
        tag_by_bucket: dict[int, str] = {}
        down_tag_by_bucket: dict[int, str] = {}
        for i, raw_b in enumerate(dir_ids):
            rb = str(raw_b)
            if rb.endswith("d"):
                # stairs look-down tile (RAM tag only; its caption is discarded) — :194
                try:
                    b = int(rb[:-1])
                except (TypeError, ValueError):
                    continue
                if i < len(tags_list):
                    down_tag_by_bucket[b] = tags_list[i]
                continue
            try:
                b = int(rb)
            except (TypeError, ValueError):
                continue
            if i < len(caps_list):
                cap_by_bucket[b] = caps_list[i]
            if i < len(tags_list):
                tag_by_bucket[b] = tags_list[i]

        stairs = ("stair" in instruction.lower()) or ("the steps" in instruction.lower())

        try:
            navigable = json.loads(nav_raw) if isinstance(nav_raw, str) else (nav_raw or {})
        except json.JSONDecodeError:
            navigable = {}

        # Bucket navigable vps by 12-direction
        bucketed: dict[int, list[tuple[str, dict]]] = {i: [] for i in range(12)}
        for vp, info in navigable.items():
            try:
                # env_mp3d navigable_json uses keys "heading"/"elevation" (radians;
                # __init__.py:1464-1468) — NOT "heading_rad" (stale docstring). Reading
                # the wrong key collapsed every vp into direction 0 (manifest of 1).
                bucket = _direction_bucket_12(
                    float(info.get("heading", info.get("heading_rad", 0.0)))
                )
            except (TypeError, ValueError):
                continue
            bucketed[bucket].append((vp, info))

        # Read visited vps from state (set of vp ids)
        visited: set[str] = set()
        if ctx and hasattr(ctx, "graph_state") and ctx.graph_state:
            try:
                hist = ctx.graph_state.read("nav_history") or []
                if isinstance(hist, list):
                    for entry in hist:
                        if isinstance(entry, dict) and "viewpoint" in entry:
                            visited.add(str(entry["viewpoint"]))
            except KeyError:
                pass

        lines: list[str] = []
        manifest: list[dict] = []
        for d in range(12):
            cands = bucketed.get(d, [])
            if not cands:
                continue
            vp, info = cands[0]
            try:
                rel_elev = float(info.get("elevation", info.get("elevation_rad", 0.0)))
            except (TypeError, ValueError):
                rel_elev = 0.0
            if rel_elev < -0.1:
                elev_flag = "(lower position indicates down stairs)"
            elif rel_elev > 0.1:
                elev_flag = "(higher position indicates up stairs)"
            else:
                elev_flag = ""
            passed = " (Passed Area)" if vp in visited else ""
            tag = tag_by_bucket.get(d, "")
            cap = cap_by_bucket.get(d, "")
            view_obs = f"Scene Description: {cap} Scene Objects: {tag}; "
            line = (
                f"Direction {d} {elev_flag} Navigable Viewpoint ID: {vp}{passed} "
                f"Elevation: Eye Level " + view_obs
            )
            if stairs and d in down_tag_by_bucket:
                # DiscussNav.py:199 — append the look-down RAM tags for this direction
                line += f"Elevation: Look Down Scene Objects: {down_tag_by_bucket[d]}; "
            lines.append(line)
            manifest.append(
                {
                    "direction": d,
                    "vp": vp,
                    "passed": bool(passed),
                    "elev_flag": elev_flag,
                    # per-direction observation line — upstream save_history summarises
                    # only the CHOSEN direction's string (make_action return value),
                    # not the whole panorama (DiscussNav.py:256-257, :349-361).
                    "line": line,
                }
            )

        if not lines:
            lines.append("(no navigable directions visible)")

        observation = " ".join(lines)
        # Stairs look-down (DiscussNav.py:189-200) IS handled: when `stairs`,
        # each direction with a navigable look-down tile gets an appended
        # `Elevation: Look Down Scene Objects: {tags};` block above.

        self._self_log("n_directions", len(manifest))
        self._self_log("n_visited", len(visited))
        self._self_log("observation_length", len(observation))

        return {
            "observation": observation,
            "manifest": json.dumps(manifest),
        }


# ══════════════════════════════════════════════════════════════════════
# History reader: format nav_history as DiscussNav `history_traj`
# ══════════════════════════════════════════════════════════════════════


class DiscussNavHistoryReaderNode(BaseCanvasNode):
    """Read nav_history state container; emit ``history_traj`` text.

    Mirrors ``Completion_Estimation_Experts.review_history``
    (``DiscussNav.py:267-271``) — joins entries as
    ``Step N Observation: ... Thought: ...`` separated by `` -> ``.

    Returns ``"Step 0 start position."`` when the history is empty
    (matches DiscussNav.py:512 fallback).
    """

    node_type: ClassVar[str] = "discussnav__history_reader"
    display_name: ClassVar[str] = "DiscussNav: History Reader"
    description: ClassVar[str] = "Read nav_history state, format as history_traj text"
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "ClipboardList"
    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Optional sequencing trigger", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("history_traj", "TEXT", "Formatted history trajectory"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        entries: list[dict] = []
        if ctx and hasattr(ctx, "graph_state") and ctx.graph_state:
            try:
                raw = ctx.graph_state.read("nav_history") or []
            except KeyError:
                raw = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        entries.append(item)

        if not entries:
            traj = "Step 0 start position."
        else:
            parts = []
            for idx, item in enumerate(entries):
                obs = item.get("observation", "")
                thought = item.get("thought", "")
                parts.append(f"Step {idx + 1} Observation: {obs} Thought: {thought}")
            traj = " -> ".join(parts)

        self._self_log("entry_count", len(entries))
        self._self_log("traj_length", len(traj))
        return {"history_traj": traj}


# ══════════════════════════════════════════════════════════════════════
# pred_vp — sample n predictions, retry until enough valid (break_flag)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavPredVpNode(BaseCanvasNode):
    """Faithful port of ``DiscussNav_Agent.pred_vp`` (``DiscussNav.py:292-347``).

    - Applies the upstream ``estimate_completion`` slice — keep only the text
      after ``"Executed Actions"`` (``:286-288``) — to the raw estimation.
    - Formats the verbatim ``input_info`` (``:314-315``) with the live
      ``ctx.step`` as ``current_step`` and the ``_SYS_PRED_VP`` system prompt.
    - Samples ``num_predictions`` responses and **retries the whole batch up
      to ``num_retry`` times** until ``num_predictions`` valid predictions
      land (``:322,:343-345``). A prediction is valid iff it parses to a
      32-hex vp present in the observation manifest (``:327-338``).
    - ``break_flag`` is True when it still can't — upstream then breaks the
      episode (``:516-517``); here it is OR-ed into the horizon stop.

    Sampling uses concurrent single-shot calls (model-agnostic; equivalent to
    upstream's batched ``n`` at temperature 1) so models without an ``n``
    parameter still work.
    """

    node_type: ClassVar[str] = "discussnav__pred_vp"
    display_name: ClassVar[str] = "DiscussNav: pred_vp"
    description: ClassVar[str] = "Sample + retry candidate viewpoints (DiscussNav_Agent.pred_vp)"
    category: ClassVar[str] = "llm"
    icon: ClassVar[str] = "Dices"
    input_ports: ClassVar[list] = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("actions", "TEXT", "Decomposed actions"),
        PortDef("landmarks", "TEXT", "Extracted landmarks"),
        PortDef("history_traj", "TEXT", "Formatted history trajectory"),
        PortDef("estimation", "TEXT", "Raw completion-estimation response"),
        PortDef("observation", "TEXT", "DiscussNav observation block"),
        PortDef("manifest", "TEXT", "JSON manifest of valid (direction, vp) pairs"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("predictions", "LIST[TEXT]", "Validated predicted viewpoint IDs"),
        PortDef("thoughts", "LIST[TEXT]", "Extracted thoughts, parallel to predictions"),
        PortDef("break_flag", "BOOL", "True iff < num_predictions valid (upstream break)"),
        PortDef("thinking", "TEXT", "First thought (viewer)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "profile",
                "select",
                "Model",
                default="",
                options=[{"value": "__DYNAMIC_PROFILES__", "label": ""}],
            ),
            ConfigField(
                "num_predictions", "slider", "Samples per step (upstream 5)",
                default=5, min=1, max=10, step=1,
            ),
            ConfigField(
                "num_retry", "slider", "Retries until enough valid (upstream 5)",
                default=5, min=1, max=10, step=1,
            ),
            ConfigField(
                "temperature", "slider", "Temperature", default=1.0, min=0.0, max=2.0, step=0.1
            ),
            ConfigField(
                "max_tokens", "slider", "Max tokens", default=1024, min=64, max=4096, step=64
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from app.llm import get_llm_config, llm_complete

        instruction = str(inputs.get("instruction") or "")
        actions = str(inputs.get("actions") or "")
        landmarks = str(inputs.get("landmarks") or "")
        history_traj = str(inputs.get("history_traj") or "Step 0 start position.")
        estimation_raw = str(inputs.get("estimation") or "")
        observation = str(inputs.get("observation") or "")
        manifest_raw = inputs.get("manifest") or "[]"
        try:
            manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else manifest_raw
        except json.JSONDecodeError:
            manifest = []
        valid_vps = {str(m.get("vp")) for m in manifest if isinstance(m, dict)}

        # Upstream estimate_completion returns only the "Executed Actions" slice
        # (DiscussNav.py:286-288) — the plain llmCall hands us the full response.
        if "Executed Actions" in estimation_raw:
            estimation = estimation_raw.split("Executed Actions", 1)[1].strip()
        else:
            estimation = estimation_raw

        current_step = int(getattr(ctx, "step", 0) or 0)
        # Verbatim input_info (DiscussNav.py:314-315)
        input_info = (
            f"Step {current_step} Instruction: {instruction} ({actions}) "
            f"Landmarks: {landmarks} Navigation History: {history_traj}             "
            f"Estimation of Executed Actions: {estimation} "
            f"Current Environment: {observation} -> Thought: ... Prediction: ..."
        )

        target_n = int(self.config.get("num_predictions", 5))
        num_retry = int(self.config.get("num_retry", 5))
        temp = float(self.config.get("temperature", 1.0))
        max_tokens = int(self.config.get("max_tokens", 1024))
        cfg = get_llm_config(self.config.get("profile", ""))

        preds: list[str] = []
        thoughts: list[str] = []
        if cfg is not None:
            for attempt in range(num_retry):
                preds, thoughts = [], []
                results = await asyncio.gather(
                    *[
                        llm_complete(
                            cfg,
                            [{"role": "user", "content": input_info}],
                            system_prompt=_SYS_PRED_VP,
                            max_tokens=max_tokens,
                            temperature=temp,
                        )
                        for _ in range(target_n)
                    ],
                    return_exceptions=True,
                )
                for resp in results:
                    text = "" if isinstance(resp, BaseException) else str(resp or "")
                    if "Prediction:" not in text:
                        continue
                    head, _, tail = text.partition("Prediction:")
                    thought = head.strip()
                    # DiscussNav.py:331 — strip quotes / dots / asterisks / newlines
                    pred_vp = (
                        tail.strip()
                        .replace('"', "")
                        .replace("'", "")
                        .replace("\n", "")
                        .replace(".", "")
                        .replace("*", "")
                        .strip()
                    )
                    m = re.search(r"[a-fA-F0-9]{32}", pred_vp)
                    if m:
                        pred_vp = m.group(0)
                    if len(pred_vp) != 32:
                        continue
                    if valid_vps and pred_vp not in valid_vps:
                        continue
                    preds.append(pred_vp)
                    thoughts.append(thought)
                self._self_log(f"attempt_{attempt}_valid", len(preds))
                if len(preds) >= target_n:
                    break

        # break_flag stays True until we collected num_predictions valid (upstream :321-345)
        break_flag = len(preds) < target_n
        self._self_log("current_step", current_step)
        self._self_log("n_valid", len(preds))
        self._self_log("break_flag", break_flag)
        return {
            "predictions": preds,
            "thoughts": thoughts,
            "break_flag": break_flag,
            "thinking": thoughts[0] if thoughts else "",
        }


# ══════════════════════════════════════════════════════════════════════
# Thought fusion (variable-cardinality sequential fan-out)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavThoughtFusionNode(BaseCanvasNode):
    """Group predictions by unique vp; fuse per-vp thoughts via GPT-4.

    Mirrors ``Decision_Testing_Experts.thought_fusion`` (``DiscussNav.py:
    364-382``). For each unique predicted vp, calls litellm sequentially
    with ``_SYS_THOUGHT_FUSION`` (gpt-4, temperature=1) — single-vp case
    skips the LLM call entirely (returns the lone thought).

    Output ``fused_json`` is ``{vp: fused_thought}``. ``unique_count`` is
    the number of unique vps after grouping.
    """

    node_type: ClassVar[str] = "discussnav__thought_fusion"
    display_name: ClassVar[str] = "DiscussNav: Thought Fusion"
    description: ClassVar[str] = "Group predictions by vp; fuse thoughts (sequential GPT-4)"
    category: ClassVar[str] = "llm"
    icon: ClassVar[str] = "Combine"
    input_ports: ClassVar[list] = [
        PortDef("predictions", "LIST[TEXT]", "Validated predicted viewpoints"),
        PortDef("thoughts", "LIST[TEXT]", "Per-prediction thoughts, parallel"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("fused_json", "TEXT", "JSON {vp: fused_thought}"),
        PortDef("unique_count", "TEXT", "Number of unique predicted vps"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "profile",
                "select",
                "Model",
                default="",
                options=[{"value": "__DYNAMIC_PROFILES__", "label": ""}],
            ),
            ConfigField(
                "temperature", "slider", "Temperature", default=1.0, min=0.0, max=2.0, step=0.1
            ),
            ConfigField(
                "max_tokens", "slider", "Max tokens", default=512, min=64, max=2048, step=64
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from app.llm import get_llm_config, llm_complete

        preds = inputs.get("predictions") or []
        thoughts = inputs.get("thoughts") or []

        # Group thoughts by unique vp (preserve first-seen order)
        order: list[str] = []
        groups: dict[str, list[str]] = {}
        for vp, th in zip(preds, thoughts, strict=False):
            vp = str(vp)
            if vp not in groups:
                groups[vp] = []
                order.append(vp)
            groups[vp].append(str(th))

        if not order:
            return {"fused_json": "{}", "unique_count": "0"}

        cfg = get_llm_config(self.config.get("profile", ""))
        temp = float(self.config.get("temperature", 1.0))
        max_tokens = int(self.config.get("max_tokens", 512))

        fused: dict[str, str] = {}
        for vp in order:
            ts = groups[vp]
            # Upstream calls the fusion LLM for every unique vp key, even a
            # single-thought group (DiscussNav.py:371-378) — no short-circuit.
            # DiscussNav.py:372
            multiple = "; ".join([f"Thought {idx + 1}: {t}" for idx, t in enumerate(ts)])
            user = (
                "Can you help me fuse the thoughts leading to the same movement "
                f"direction? The thoughts are :{multiple}, Fused thought: "
            )
            text: str | None = None
            if cfg is not None:
                text = await llm_complete(
                    cfg,
                    [{"role": "user", "content": user}],
                    system_prompt=_SYS_THOUGHT_FUSION,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
            fused[vp] = text or ts[0]

        self._self_log("unique_count", len(order))
        self._self_log("fused_keys", list(fused.keys()))
        return {"fused_json": json.dumps(fused), "unique_count": str(len(order))}


# ══════════════════════════════════════════════════════════════════════
# Decision test (single-vp passthrough, multi-vp GPT-4)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavDecisionTestNode(BaseCanvasNode):
    """Pick a final vp from fused predictions.

    Mirrors ``Decision_Testing_Experts.test_decisions`` (``DiscussNav.py:
    384-405``). Single-vp case returns immediately. Multi-vp case calls
    litellm with ``_SYS_TEST_DECISIONS`` (gpt-4, temperature=0), retries
    up to 3x on length / membership failures.
    """

    node_type: ClassVar[str] = "discussnav__decision_test"
    display_name: ClassVar[str] = "DiscussNav: Decision Test"
    description: ClassVar[str] = "Pick final vp (single-vp passthrough or GPT-4 with retry)"
    category: ClassVar[str] = "llm"
    icon: ClassVar[str] = "Crosshair"
    input_ports: ClassVar[list] = [
        PortDef("fused_json", "TEXT", "JSON {vp: fused_thought}"),
        PortDef("observation", "TEXT", "Current observation text"),
        PortDef("instruction", "TEXT", "Navigation instruction"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("next_vp", "TEXT", "Final selected viewpoint ID"),
        PortDef("final_thought", "TEXT", "Thought associated with final_vp"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "profile",
                "select",
                "Model",
                default="",
                options=[{"value": "__DYNAMIC_PROFILES__", "label": ""}],
            ),
            ConfigField(
                "max_tokens", "slider", "Max tokens", default=128, min=32, max=512, step=32
            ),
            ConfigField(
                "temperature",
                "slider",
                "Temperature (upstream 0.0; gpt-5 family needs 1.0)",
                default=0.0,
                min=0.0,
                max=2.0,
                step=0.1,
            ),
            ConfigField(
                "max_retries",
                "slider",
                "Max retries on length / membership fail",
                default=3,
                min=1,
                max=5,
                step=1,
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from app.llm import get_llm_config, llm_complete

        try:
            fused = json.loads(str(inputs.get("fused_json") or "{}"))
        except json.JSONDecodeError:
            fused = {}
        observation = str(inputs.get("observation") or "")
        instruction = str(inputs.get("instruction") or "")

        if not fused:
            self._self_log("error", "fused empty")
            return {"next_vp": "", "final_thought": ""}

        keys = list(fused.keys())
        if len(keys) == 1:
            vp = keys[0]
            self._self_log("decision_path", "single-vp")
            return {"next_vp": vp, "final_thought": str(fused[vp])}

        # DiscussNav.py:389
        joined = "; ".join([f"Navigation Viewpoint ID: {k} Thought: {v}" for k, v in fused.items()])
        user = (
            f"Can you help me make a final decision? The Observation: {observation}, "
            f"Navigation Instruction: {instruction}, {joined}, Final Decision: "
        )

        cfg = get_llm_config(self.config.get("profile", ""))
        max_tokens = int(self.config.get("max_tokens", 128))
        max_retries = int(self.config.get("max_retries", 3))
        temp = float(self.config.get("temperature", 0.0))

        chosen = ""
        last_cand = ""
        if cfg is not None:
            for attempt in range(max_retries):
                resp = await llm_complete(
                    cfg,
                    [{"role": "user", "content": user}],
                    system_prompt=_SYS_TEST_DECISIONS,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                cand = (resp or "").strip()
                m = re.search(r"[a-fA-F0-9]{32}", cand)
                if m:
                    cand = m.group(0)
                last_cand = cand  # upstream returns the LAST attempt on exhaustion
                if len(cand) != 32:
                    self._self_log(f"retry_{attempt}_length", cand[:64])
                    continue
                if cand not in fused:
                    self._self_log(f"retry_{attempt}_not_in_keys", cand)
                    continue
                chosen = cand
                break

        if chosen:
            self._self_log("decision_path", "multi-vp-llm")
        else:
            # Mirror upstream test_decisions (DiscussNav.py:395-405): on retry
            # exhaustion return the LAST attempted candidate (possibly invalid),
            # NOT a fall-back to the first vp. Upstream then does
            # fused_pred_thought[next_vp], which KeyErrors on an invalid id; we
            # use .get to avoid crashing the node — the only safe deviation from
            # the literal upstream bug.
            chosen = last_cand
            self._self_log("decision_path", "mirror-upstream-last")
        return {"next_vp": chosen, "final_thought": str(fused.get(chosen, ""))}


# ══════════════════════════════════════════════════════════════════════
# History writer (sequential summarisations + state append)
# ══════════════════════════════════════════════════════════════════════


class DiscussNavHistoryWriterNode(BaseCanvasNode):
    """Summarise observation + thought; append to nav_history state.

    Mirrors ``Completion_Estimation_Experts.save_history`` (``DiscussNav.py:
    256-265``). Two sequential litellm calls (gpt-4, temperature=1):

    1. summarize_observation — uses observation + landmarks
       (DiscussNav.py:235-246; the per-direction-label prefix is computed
       from the chosen vp's bucket and prepended verbatim).
    2. summarize_thought — uses the chosen-vp thought.

    Then appends ``{viewpoint, observation, thought}`` to ``nav_history``.
    """

    node_type: ClassVar[str] = "discussnav__history_writer"
    display_name: ClassVar[str] = "DiscussNav: History Writer"
    description: ClassVar[str] = "Summarise obs+thought; append to nav_history"
    category: ClassVar[str] = "llm"
    icon: ClassVar[str] = "BookPlus"
    input_ports: ClassVar[list] = [
        PortDef("next_vp", "TEXT", "Final chosen viewpoint ID"),
        PortDef("observation", "TEXT", "Current observation block"),
        PortDef("final_thought", "TEXT", "Thought associated with chosen vp"),
        PortDef("manifest", "TEXT", "Manifest from observation_aggregator (for direction lookup)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("history_text", "TEXT", "Latest entry preview (for viewer + downstream)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "profile",
                "select",
                "Model",
                default="",
                options=[{"value": "__DYNAMIC_PROFILES__", "label": ""}],
            ),
            ConfigField(
                "max_tokens", "slider", "Max tokens", default=384, min=64, max=1024, step=32
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from app.llm import get_llm_config, llm_complete

        next_vp = str(inputs.get("next_vp") or "").strip()
        observation = str(inputs.get("observation") or "")
        thought = str(inputs.get("final_thought") or "")
        manifest_raw = inputs.get("manifest") or "[]"
        try:
            manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else manifest_raw
        except json.JSONDecodeError:
            manifest = []

        # Find the direction-label + per-direction line for the chosen vp
        # (DiscussNav.py:236). Upstream summarises ONLY the chosen direction's
        # observation string (make_action return value), not the whole panorama.
        direction_id = -1
        chosen_line = ""
        for m in manifest:
            if isinstance(m, dict) and str(m.get("vp")) == next_vp:
                try:
                    direction_id = int(m.get("direction", -1))
                except (TypeError, ValueError):
                    direction_id = -1
                chosen_line = str(m.get("line") or "")
                break
        direction_label = (
            _DIRECTION_LABELS_12[direction_id]
            if 0 <= direction_id < len(_DIRECTION_LABELS_12)
            else "Front, range(right 0 to right 30)"
        )

        # Per upstream summarize_observation: keep only the "Scene Description ..." slice
        # of the CHOSEN direction (fall back to the full block if the line is missing).
        src = chosen_line or observation
        if "Scene Description" in src:
            obs_slice = "Scene Description" + src.split("Scene Description", 1)[1]
        else:
            obs_slice = src

        cfg = get_llm_config(self.config.get("profile", ""))
        max_tokens = int(self.config.get("max_tokens", 384))

        async def _summarise(kind: str, payload: str) -> str:
            if cfg is None:
                return payload[:200]
            user = (
                f'Given Environment Description "{payload}", Summarization:'
                if kind == "environment description"
                else f'Given Thought Process "{payload}", Summarization:'
            )
            text = await llm_complete(
                cfg,
                [{"role": "user", "content": user}],
                system_prompt=_SYS_SUMMARIZE_OBS_OR_THOUGHT.format(kind=kind),
                max_tokens=max_tokens,
                temperature=1.0,
            )
            return text or payload[:200]

        # Two sequential summarisation calls
        obs_summary = await _summarise("environment description", obs_slice)
        thought_summary = await _summarise("navigation thought process", thought)

        full_obs = f"Direction {direction_label} {obs_summary}"

        entry = {"viewpoint": next_vp, "observation": full_obs, "thought": thought_summary}

        if ctx and hasattr(ctx, "graph_state") and ctx.graph_state:
            try:
                hist = ctx.graph_state.read("nav_history") or []
            except KeyError:
                hist = []
            if not isinstance(hist, list):
                hist = []
            # Copy-on-write so the state container sees a fresh list ref
            hist = [*list(hist), entry]
            ctx.graph_state.write("nav_history", hist)
            self._self_log("nav_history_len", len(hist))

        preview = (
            f"vp={next_vp[:8]}… dir={direction_id} "
            f'obs="{obs_summary[:80]}…" thought="{thought_summary[:80]}…"'
        )
        self._self_log("preview", preview)
        return {"history_text": preview}


# ══════════════════════════════════════════════════════════════════════
# Horizon stop — adaptive 5/7 step length + pred_vp break_flag
# ══════════════════════════════════════════════════════════════════════


class DiscussNavHorizonStopNode(BaseCanvasNode):
    """Reproduce upstream loop termination — ``range(step_length)`` + break.

    Upstream runs ``for current_step in range(step_length)`` with the adaptive
    ``step_length = 5 if len(actions) <= 5 else 7`` (``DiscussNav.py:506-507``)
    and breaks early when ``pred_vp`` cannot find enough valid predictions
    (``:516-517``). DiscussNav has no STOP action, so this — not the env's
    ``terminated`` (which is STOP-only) — is the real loop exit.

    Emits ``stop=True`` once ``ctx.step`` reaches ``step_budget - 1`` (so the
    loop runs exactly ``step_budget`` iterations, each of which moves) OR when
    ``pred_vp`` raised ``break_flag``. Single writer to ``iterOut.stop`` —
    folds both conditions to avoid multi-writer ambiguity on the stop port.
    """

    node_type: ClassVar[str] = "discussnav__horizon_stop"
    display_name: ClassVar[str] = "DiscussNav: Horizon Stop"
    description: ClassVar[str] = "Stop at the adaptive 5/7 budget or on pred_vp break"
    category: ClassVar[str] = "control"
    icon: ClassVar[str] = "Flag"
    input_ports: ClassVar[list] = [
        PortDef("step_budget", "TEXT", "Adaptive step length (5 short / 7 long)"),
        PortDef("break_flag", "BOOL", "pred_vp failed to find enough valid predictions"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("stop", "BOOL", "Loop halt signal (→ iterOut.stop)"),
    ]
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="rose")

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            sb = int(str(inputs.get("step_budget") or "7").strip())
        except (TypeError, ValueError):
            sb = 7
        sb = max(1, sb)
        step = int(getattr(ctx, "step", 0) or 0)
        broke = bool(inputs.get("break_flag"))
        stop = (step >= sb - 1) or broke
        self._self_log("step", step)
        self._self_log("step_budget", sb)
        self._self_log("break_flag", broke)
        self._self_log("stop", stop)
        return {"stop": stop}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class DiscussNavNodeSet(BaseNodeSet):
    """DiscussNav reasoning nodeset — multi-expert-discussion VLN port.

    Nine nodes covering the full DiscussNav per-step pipeline:

    - ``discussnav__init_decompose`` — fold init LLM responses (actions,
      landmarks) and compute the adaptive step_budget (5 / 7).
    - ``discussnav__panorama_to_views`` — encode env per-view images →
      base64 view-tile list (feeds ``model_ram__tag_panorama``).
    - ``discussnav__observation_aggregator`` — fold tags + captions →
      observation text + vp manifest (with per-direction lines).
    - ``discussnav__history_reader`` — read nav_history state; emit
      history_traj text.
    - ``discussnav__pred_vp`` — sample n predictions, retry until enough
      valid, surface break_flag (DiscussNav_Agent.pred_vp).
    - ``discussnav__thought_fusion`` — per-unique-vp fusion call.
    - ``discussnav__decision_test`` — pick final vp (single passthrough
      or LLM with retry).
    - ``discussnav__history_writer`` — summarise chosen direction + append
      nav_history.
    - ``discussnav__horizon_stop`` — adaptive 5/7 horizon + pred_vp break →
      iterOut.stop (DiscussNav has no STOP action).
    """

    name = "discussnav"
    display_name = "DiscussNav"
    description = "Multi-expert-discussion VLN — DiscussNav (Long et al., ICRA 2024) port"
    # LLM-heavy: up to num_predictions*num_retry pred_vp samples + fusion +
    # decision + 2 summarisations per step. Override default budget.
    default_per_step_budget_sec = 90.0

    def get_tools(self) -> list:
        return [
            DiscussNavInitDecomposeNode(),
            DiscussNavPanoramaToViewsNode(),
            DiscussNavObservationAggregatorNode(),
            DiscussNavHistoryReaderNode(),
            DiscussNavPredVpNode(),
            DiscussNavThoughtFusionNode(),
            DiscussNavDecisionTestNode(),
            DiscussNavHistoryWriterNode(),
            DiscussNavHorizonStopNode(),
        ]

    async def initialize(self) -> None:
        log.info("DiscussNav nodeset initialised")

    async def shutdown(self) -> None:
        pass
