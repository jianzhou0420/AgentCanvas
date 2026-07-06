"""Three-Step Nav method nodeset.

Faithful AgentCanvas port of:

    Three-Step Nav — global-local navigation with a back-check verify step
    (R2R-CE / RxR-CE, zero-shot LLM-VLN)
    arXiv: https://arxiv.org/abs/2604.26946
    Upstream: https://github.com/ZoeyZheng0/3-step-Nav @ 5cdbdcf

Three-Step Nav is a descendant of Open-Nav (its agent class literally is
``class Open_Nav``). It reuses Open-Nav's exact perception+waypoint substrate,
so this nodeset pairs with the SAME tool nodesets:

    env_habitat                         — VLN-CE simulator (server mode)
    opennav_waypoint                    — frozen TRM waypoint predictor (12-dir)
    model_ram / vlm_spatialbot          — RAM tags + SpatialBot-3B captions (FM wrappers)

This nodeset implements the **method-side reasoning** — the three steps:

  Step 1 (global) — decompose the instruction into sub-instructions
      (``detect_actions``) + per-sub-instruction landmarks (``extract_landmarks``).
      Runs once per episode (pre-loop); fed into the loop via iterIn init ports.

  Step 2 (local)  — per env step: a MapGPT-style **image** navigator
      (``navigator_prompt`` → ``llmCall`` w/ candidate tiles → ``parse_navigator``)
      picks the next direction id AND self-reports ``Completion Estimation: Yes/No``.

  Step 3 (verify) — when the navigator reports ``Yes``, a **judge / decision
      agent** (folded into ``judge_decide``, image-augmented over the chosen-path
      image sequence) returns continue / stay / backtrack / look-around, which
      advances or holds the sub-instruction pointer and (continue) resets
      per-sub-instruction history.

The per-step env action is ``env_habitat__step_hightolow (angle, distance)``;
stopping is decide-to-stop (no env STOP action — the loop ends via ``iter_out.stop``
and ``evaluate`` scores the final pose), mirroring SmartWay-CE.

Prompts are verbatim in ``_prompts.py`` (each constant cites its upstream
``file:line``). Default planner/judge LLM is **gpt-5-mini** (the run config's
upstream default is gpt-5; we follow the house VLN-port default — needs
``temperature=1.0`` + ``max_tokens=2000`` for the gpt-5 family).

Deliberate deviations from upstream (see the doc page §3 for the full buckets):
  • [bucket-C] The judge LLM is gated on ``Completion Estimation == "Yes"``
    exactly as upstream; when gated off the decision defaults to "stay" with no
    extra LLM call (upstream cost behaviour preserved).
  • [bucket-D] look-around: upstream re-observes all viewpoints then still
    moves to the navigator's pick; this graph already perceives every candidate
    direction each step, so look-around reduces to a normal move (no separate
    re-observe pass). backtrack reverses the last move via a per-episode
    move-stack (mirrors ``env_actions_history`` + ``_create_reverse_action``);
    upstream's backtrack branch additionally still executes the forward move in
    the same iteration (two env steps) — the graph emits one action per step
    (bucket B, config-dead under the faithful ``[continue, stay]`` abilities).
  • [bucket-D] The per-instruction action/landmark cache (``cache_files/...``)
    is dropped — every episode re-runs Step 1. Faithfulness/simplicity tradeoff.

Byte-fidelity oracle: ``tmp/verify/threestepnav_upstream_equiv.py`` imports the
REAL upstream modules (langchain/model clients stubbed; trainer/api methods
AST-extracted) and byte-compares this nodeset function-by-function — 79 checks.

Load:  POST /api/components/nodesets/threestepnav/load

last updated: 2026-07-02
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)
from workspace.nodesets.method.threestepnav._prompts import (
    ACTION_DETECTION_SYSTEM,
    ACTION_DETECTION_USER,
    DIRECTIONS,
    JUDGE_SYSTEM,
    LANDMARK_DETECTION_SYSTEM,
    LANDMARK_DETECTION_USER,
    MAPGPT_NAVIGATOR_SYSTEM,
    OBSERVATION_SUMMARY_SYSTEM,
    OBSERVATION_SUMMARY_USER,
    THOUGHT_SUMMARY_SYSTEM,
    THOUGHT_SUMMARY_USER,
    apply_decision_rules,
    build_judge_user,
    build_navigator_user,
    parse_judge,
    parse_navigator,
    string_to_decision,
)

log = logging.getLogger("agentcanvas.threestepnav")


# ── helpers ──────────────────────────────────────────────────────────────


def _read(gs: Any, key: str, default: Any) -> Any:
    if gs is None:
        return default
    try:
        v = gs.read(key)
        return v if v is not None else default
    except Exception:
        return default


def _write(gs: Any, key: str, value: Any) -> None:
    if gs is None:
        return
    with contextlib.suppress(Exception):
        gs.write(key, value)


def _as_dict(v: Any) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, list) else []
        except Exception:
            return []
    return []


def _views_by_dir(views: Any) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for v in _as_list(views):
        if isinstance(v, dict) and "dir_id" in v:
            out[str(v["dir_id"])] = v
    return out


def _candidate_order(keys: Any) -> list[str]:
    """Upstream candidate-dict insertion order: the TRM emits angle indexes in
    ascending clockwise order (Policy_ViewSelection.py:374 ``nonzero()``), which
    ``2π - idx/120·2π`` (:378) turns into DESCENDING angles, so
    ``construct_image_dicts`` inserts bin '0' first (the 330°-360° else-branch),
    then '11', '10', … '1' (base_il_trainer_llm.py:202-259)."""
    return sorted((str(k) for k in keys), key=lambda k: (int(k) != 0, -int(k)))


def _png_b64_to_jpeg_b64(b64: str) -> str | None:
    """Decode a lossless env tile and re-encode as JPEG (PIL default
    quality) — the byte chain upstream's VLM actually receives
    (image_to_base64, base_il_trainer_llm.py:64-83; judge re-encode,
    decision_agent.py:806-815)."""
    import base64
    import io

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _flatten_history(entries: list) -> str:
    """review_history (spatialNavigator.py:57-60) — ``"Step {i+1} Observation:
    {obs} Thought: {thought}"`` joined with `` -> ``; the empty-history fallback
    is the caller-side ``"Step 0 start position. "`` (base_il_trainer_llm.py:722)."""
    if not entries:
        return "Step 0 start position. "
    return " -> ".join(
        f"Step {i + 1} Observation: {e.get('observation', '')} Thought: {e.get('thought', '')}"
        for i, e in enumerate(entries)
    )


# ── chosen-path image-sequence descriptions ──────────────────────────────
# Mirrors base_il_trainer_llm.py:489-491 (initial seed) + :690-718 (per-step
# direction phrase) — the textual sequence the upstream judge always receives
# prepended to its prompt (the "Image sequence descriptions" block).

_INITIAL_DESC = "Initial position: Agent standing at start point looking forward"


def _direction_desc(angle_rad: float) -> str:
    """12-way phrase for the chosen move angle (base_il_trainer_llm.py:692-715)."""
    d = math.degrees(angle_rad)
    if -15 <= d <= 15:
        return "forward"
    if 15 < d <= 45:
        return "front-left (30°)"
    if 45 < d <= 75:
        return "left (60°)"
    if 75 < d <= 105:
        return "left (90°)"
    if 105 < d <= 135:
        return "back-left (120°)"
    if 135 < d <= 165:
        return "back-left (150°)"
    if d > 165 or d < -165:
        return "backward (180°)"
    if -165 <= d < -135:
        return "back-right (150°)"
    if -135 <= d < -105:
        return "back-right (120°)"
    if -105 <= d < -75:
        return "right (90°)"
    if -75 <= d < -45:
        return "right (60°)"
    if -45 <= d < -15:
        return "front-right (30°)"
    return ""


def _step_desc(step: int, angle_rad: float) -> str:
    return (
        f"Step {step}: Agent at previous position looking {_direction_desc(angle_rad)} "
        "towards chosen next viewpoint"
    )


def _build_descs(gs: Any, vbd: dict, pred_vp: str, step: int, fwd_angle: float) -> list[str]:
    """Accumulate the description list in lockstep with the chosen-image list:
    seed on the first step (iff the seed image is added), append the per-step
    phrase (iff the chosen tile is added). Reads the PRIOR state, like the
    image accumulator, so mono and decomp agree."""
    descs = list(_as_list(_read(gs, "chosen_descriptions", [])))
    chosen = list(_as_list(_read(gs, "chosen_images_b64", [])))
    if not chosen and str(vbd.get("0", {}).get("rgb_base64", "")):
        descs.append(_INITIAL_DESC)
    if str(vbd.get(str(pred_vp), {}).get("rgb_base64", "")):
        descs.append(_step_desc(step, fwd_angle))
    return descs


# ── stuck-direction handling ─────────────────────────────────────────────
# Mirrors base_il_trainer_llm.py:655-664 (filter stuck dirs out of the
# navigator's candidates) + :873-910 (detect a no-move step, blacklist the
# chosen direction, pop the failed image/history) + :786 (clear on continue).

_STUCK_THRESHOLD_M = 0.1  # position_diff < 0.1m ⇒ stuck (base_il_trainer_llm.py:887)


def _filter_stuck_keys(keys: list, stuck: Any) -> list:
    """Drop blacklisted direction ids; if every candidate is stuck, keep all
    (base_il_trainer_llm.py:659-662 "All directions are stuck! Using original")."""
    stuck_set = {str(s) for s in _as_list(stuck)}
    kept = [k for k in keys if str(k) not in stuck_set]
    return kept if kept else list(keys)


class StuckDetectNode(BaseCanvasNode):
    """Iteration-start stuck check. Reads the current agent position; if the
    PREVIOUS step's move advanced &lt; 0.1m, the previously-chosen direction is
    blacklisted and the failed chosen-image / description / history entry are
    popped (base_il_trainer_llm.py:873-910). Emits the live blacklist as data so
    ``format_observation`` / ``build_images`` exclude those directions from the
    navigator (base_il_trainer_llm.py:655-657). Sole writer of
    ``previous_position`` + ``stuck_directions``; corrects the post-move overshoot
    of ``judge_decide``'s pre-move appends to nav_history / chosen_images_b64 /
    chosen_descriptions."""

    node_type: ClassVar[str] = "threestepnav__stuck_detect"
    display_name: ClassVar[str] = "3-Step: Stuck Detect"
    description: ClassVar[str] = "Blacklist no-move directions + pop the failed step"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Ban"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="rose")
    input_ports = [
        PortDef("position", "ANY", "Current world-frame position [x, y, z]"),
        PortDef(
            "trigger", "ANY", "Per-step trigger (wire iter_in.step) — forces re-fire", optional=True
        ),
    ]
    output_ports = [PortDef("stuck_directions", "ANY", "Live blacklist of stuck direction ids")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        gs = getattr(ctx, "graph_state", None) if ctx else None
        pos_now = _as_list(inputs.get("position"))
        prev = _read(gs, "previous_position", None)
        last_vp = str(_read(gs, "last_chosen_vp", "") or "")
        stuck = list(_as_list(_read(gs, "stuck_directions", [])))

        prev_l = _as_list(prev)
        if prev_l and last_vp and len(pos_now) >= 3 and len(prev_l) >= 3:
            diff = math.dist(
                [float(pos_now[0]), float(pos_now[1]), float(pos_now[2])],
                [float(prev_l[0]), float(prev_l[1]), float(prev_l[2])],
            )
            if diff < _STUCK_THRESHOLD_M:
                if last_vp not in {str(s) for s in stuck}:
                    stuck.append(last_vp)
                # Pop the failed step's appends (base_il_trainer_llm.py:899-910).
                for key in ("chosen_images_b64", "chosen_descriptions", "nav_history"):
                    seq = list(_as_list(_read(gs, key, [])))
                    if seq:
                        seq.pop()
                        _write(gs, key, seq)
                self._self_log("stuck_blacklist", last_vp)

        if len(pos_now) >= 3:
            _write(
                gs, "previous_position", [float(pos_now[0]), float(pos_now[1]), float(pos_now[2])]
            )
        _write(gs, "stuck_directions", stuck)
        self._self_log("stuck_directions", stuck)
        return {"stuck_directions": stuck}


async def _internal_llm_call(
    profile: str, system_prompt: str, user_prompt: str, max_tokens: int = 512
) -> str:
    """Single text LLM call — used for the per-sub-instruction landmark
    fan-out (variable N) that a static graph can't express as fixed nodes."""
    from app.llm.call import get_llm_config, llm_complete

    cfg = get_llm_config(profile or "")
    if cfg is None:
        log.warning("threestepnav: LLM profile '%s' not found", profile)
        return ""
    out = await llm_complete(
        config=cfg,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=1.0,
    )
    return out or ""


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — instruction decomposition (pre-loop, episode-fixed)
# ═══════════════════════════════════════════════════════════════════════


class DetectActionsPromptNode(BaseCanvasNode):
    """Step 1a — assemble ACTION_DETECTION prompt (decompose into sub-instructions)."""

    node_type: ClassVar[str] = "threestepnav__detect_actions_prompt"
    display_name: ClassVar[str] = "3-Step: Detect Actions Prompt"
    description: ClassVar[str] = "Decompose instruction into sub-instructions (ACTION_DETECTION)"
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
        return {
            "system": ACTION_DETECTION_SYSTEM,
            "user": ACTION_DETECTION_USER.format(instruction),
        }


class ExtractLandmarksNode(BaseCanvasNode):
    """Step 1b — split sub-instructions + extract landmarks PER sub-instruction.

    Mirrors base_il_trainer_llm.py:566-586 — ``action_list = actions.split("\\n")``
    then ``get_landmarks(action)`` per non-empty action. The per-action landmark
    fan-out (variable N) is folded into this one node (one ``llmCall`` per
    sub-instruction internally) so the static graph can express it; this is the
    same dynamic-fan-out fold Open-Nav uses for its ensemble.
    """

    node_type: ClassVar[str] = "threestepnav__extract_landmarks"
    display_name: ClassVar[str] = "3-Step: Extract Landmarks"
    description: ClassVar[str] = "Per-sub-instruction landmark extraction (LANDMARK_DETECTION xN)"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("llm_profile", "text", label="LLM profile", default="gpt-5-mini"),
            ConfigField("max_tokens", "text", label="max_tokens", default=2000),
        ],
    )
    input_ports = [PortDef("actions", "TEXT", "ACTION_DETECTION output (newline-separated)")]
    output_ports = [
        PortDef("action_list", "ANY", "JSON list of sub-instruction strings"),
        PortDef("landmark_list", "ANY", "JSON list[list[str]] landmarks per sub-instruction"),
        PortDef("num_actions", "ANY", "len(action_list)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        profile = str(cfg.get("llm_profile", "gpt-5-mini"))
        max_tokens = int(cfg.get("max_tokens", 2000))

        actions = str(inputs.get("actions", ""))
        action_list = actions.split("\n")

        landmark_list: list = []
        for action in action_list:
            if action.strip():
                landmarks = await _internal_llm_call(
                    profile,
                    LANDMARK_DETECTION_SYSTEM,
                    LANDMARK_DETECTION_USER.format(action),
                    max_tokens=max_tokens,
                )
                landmark_list.append(landmarks.replace("- ", "").split("\n"))
            else:
                landmark_list.append("")

        self._self_log("num_actions", len(action_list))
        return {
            "action_list": action_list,
            "landmark_list": landmark_list,
            "num_actions": len(action_list),
        }


# ═══════════════════════════════════════════════════════════════════════
# Sub-instruction pointer → current / next sub-instruction + landmarks
# ═══════════════════════════════════════════════════════════════════════


class SelectSubinstructionNode(BaseCanvasNode):
    """Read ``current_action_idx`` from state; emit the current sub-instruction,
    its landmarks, and the next sub-instruction. Mirrors the per-step indexing
    in base_il_trainer_llm.py:619-666 (``action_list[current_action_idx]`` etc.).
    """

    node_type: ClassVar[str] = "threestepnav__select_subinstruction"
    display_name: ClassVar[str] = "3-Step: Select Sub-instruction"
    description: ClassVar[str] = "Pointer → current/next sub-instruction + landmarks"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Milestone"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports = [
        PortDef("action_list", "ANY", "Episode sub-instruction list (from iterIn init)"),
        PortDef("landmark_list", "ANY", "Episode landmark list (from iterIn init)"),
        PortDef(
            "trigger", "ANY", "Per-step trigger (wire iter_in.step) — forces re-fire", optional=True
        ),
    ]
    output_ports = [
        PortDef("current_action", "TEXT", "action_list[idx]"),
        PortDef("current_landmarks", "TEXT", "str(landmark_list[idx]) for the navigator"),
        PortDef("current_landmarks_join", "TEXT", "', '-joined landmarks for the judge"),
        PortDef("next_instruction", "TEXT", "action_list[idx+1] or 'Stop.'"),
        PortDef("action_idx", "ANY", "Current sub-instruction index"),
        PortDef("num_actions", "ANY", "len(action_list)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        action_list = _as_list(inputs.get("action_list"))
        landmark_list = _as_list(inputs.get("landmark_list"))
        gs = getattr(ctx, "graph_state", None) if ctx else None
        idx = int(_read(gs, "current_action_idx", 0) or 0)
        n = len(action_list)
        idx = max(0, min(idx, n - 1)) if n else 0

        current_action = action_list[idx] if idx < n else ""
        cur_lm = landmark_list[idx] if idx < len(landmark_list) else []
        current_landmarks = str(cur_lm)
        if isinstance(cur_lm, list):
            current_landmarks_join = ", ".join(str(x) for x in cur_lm)
        else:
            current_landmarks_join = str(cur_lm)
        next_instruction = action_list[idx + 1] if idx + 1 < n else "Stop."

        self._self_log("action_idx", idx)
        self._self_log("current_action", current_action[:120])
        return {
            "current_action": current_action,
            "current_landmarks": current_landmarks,
            "current_landmarks_join": current_landmarks_join,
            "next_instruction": next_instruction,
            "action_idx": idx,
            "num_actions": n,
        }


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — observation formatting + image navigator
# ═══════════════════════════════════════════════════════════════════════


class FormatObservationNode(BaseCanvasNode):
    """Fuse per-direction RAM tags + SpatialBot captions into the navigator's
    ``Current Environment`` string. Mirrors ``observe_view`` (api.py:280-286):

        Direction {i} Direction Viewpoint ID: {i} in Step ID: {t} Elevation: Eye Level
        Scene Description: {caption} Scene Objects: {tags};
    """

    node_type: ClassVar[str] = "threestepnav__format_observation"
    display_name: ClassVar[str] = "3-Step: Format Observation"
    description: ClassVar[str] = "Per-candidate RAM tags + SpatialBot captions → observation string"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Eye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} from waypoint predictor"),
        PortDef("tags", "ANY", "{dir_id: 'tag tag'} from RAM"),
        PortDef("captions", "ANY", "{dir_id: 'caption'} from SpatialBot"),
        PortDef(
            "stuck_directions",
            "ANY",
            "Blacklisted dir ids to exclude (from stuck_detect)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "Concatenated per-direction observation"),
        PortDef("candidate_ids", "TEXT", "Comma-separated candidate viewpoint ids"),
        PortDef("blocks", "ANY", "{dir_id: per-direction observation block}"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        candidates = _as_dict(inputs.get("candidates"))
        tags = _as_dict(inputs.get("tags"))
        captions = _as_dict(inputs.get("captions"))
        # Upstream current_step is incremented BEFORE the observe
        # (base_il_trainer_llm.py:600) — 1-based; ctx.step is 0-based.
        step = (int(getattr(ctx, "step", 0)) if ctx else 0) + 1

        # The observation TEXT covers every candidate (upstream passes the
        # UNFILTERED observe list to the prompt, base_il_trainer_llm.py:624,666);
        # only the candidate-ID list (and the images) drop stuck directions
        # (:656-657).
        parts: list[str] = []
        blocks: dict[str, str] = {}
        ordered = _candidate_order(candidates.keys())
        for dir_id in ordered:
            cap = str(captions.get(dir_id, "")).strip()
            tag = str(tags.get(dir_id, "")).strip()
            block = (
                f"Direction {dir_id} Direction Viewpoint ID: {dir_id} in Step ID: {step} "
                f"Elevation: Eye Level "
                f"Scene Description: {cap} "
                f"Scene Objects: {tag}; "
            )
            parts.append(block)
            blocks[str(dir_id)] = block
        ids = _filter_stuck_keys(ordered, inputs.get("stuck_directions"))

        self._self_log("num_candidates", len(ids))
        return {
            # Upstream formats the LIST of per-direction strings straight into
            # the prompt ("Current Environment: {}".format(list)) — reproduce
            # the list repr, not a plain concatenation. Likewise the candidate
            # slot renders upstream's ``filtered_observe_dict.keys()`` view
            # (spatialNavigator.py:120) — "dict_keys(['0', '11', …])".
            "observation": str(parts),
            "candidate_ids": str({k: None for k in ids}.keys()),
            "blocks": blocks,
        }


class ReviewHistoryNode(BaseCanvasNode):
    """Flatten ``nav_history`` to a string. Mirrors review_history
    (spatialNavigator.py:57-60) — ``"Step {i+1} Observation: {obs} Thought:
    {thought}"`` joined with `` -> `` (else ``"Step 0 start position. "``)."""

    node_type: ClassVar[str] = "threestepnav__review_history"
    display_name: ClassVar[str] = "3-Step: Review History"
    description: ClassVar[str] = "Flatten nav_history container to '... -> ...' string"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "BookOpen"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports = [
        PortDef(
            "trigger", "ANY", "Per-step trigger (wire iter_in.step) — forces re-fire", optional=True
        ),
    ]
    output_ports = [PortDef("history_traj", "TEXT", "Flattened history string")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        gs = getattr(ctx, "graph_state", None) if ctx else None
        entries = [e for e in _as_list(_read(gs, "nav_history", [])) if isinstance(e, dict)]
        self._self_log("history_steps", len(entries))
        return {"history_traj": _flatten_history(entries)}


class NavigatorPromptNode(BaseCanvasNode):
    """Assemble the MAPGPT_NAVIGATOR user prompt. Mirrors move_to_next_vp_single
    (spatialNavigator.py:120) format order."""

    node_type: ClassVar[str] = "threestepnav__navigator_prompt"
    display_name: ClassVar[str] = "3-Step: Navigator Prompt"
    description: ClassVar[str] = "Assemble MapGPT image-navigator prompt"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Compass"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("candidate_ids", "TEXT", "Candidate viewpoint id list"),
        PortDef("current_action", "TEXT", "Current sub-instruction"),
        PortDef("current_landmarks", "TEXT", "Current sub-instruction landmarks"),
        PortDef("history_traj", "TEXT", "Flattened nav history"),
        PortDef("next_instruction", "TEXT", "Next sub-instruction"),
        PortDef("observation", "TEXT", "Per-direction observation"),
    ]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        user = build_navigator_user(
            str(inputs.get("candidate_ids", "")),
            str(inputs.get("current_action", "")),
            str(inputs.get("current_landmarks", "")),
            str(inputs.get("history_traj", "")),
            str(inputs.get("next_instruction", "Stop.")),
            str(inputs.get("observation", "")),
        )
        return {"system": MAPGPT_NAVIGATOR_SYSTEM, "user": user}


class BuildImagesNode(BaseCanvasNode):
    """Decode per-candidate RGB tiles + emit ``Viewpoint {dir_id}:`` labels for
    the image navigator. Mirrors gpt_infer_with_images (api.py:154-162) — the
    label-before-image interleaving."""

    node_type: ClassVar[str] = "threestepnav__build_images"
    display_name: ClassVar[str] = "3-Step: Build Images"
    description: ClassVar[str] = "Per-candidate RGB tiles + 'Viewpoint N:' labels for the navigator"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Image"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("views", "ANY", "Panorama views [{dir_id, rgb_base64, ...}]"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} candidate directions"),
        PortDef(
            "stuck_directions",
            "ANY",
            "Blacklisted dir ids to exclude (from stuck_detect)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("images", "LIST[IMAGE]", "Per-candidate RGB tiles"),
        PortDef("image_labels", "LIST[TEXT]", "'Viewpoint {dir_id}:' — 1:1 with images"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        candidates = _as_dict(inputs.get("candidates"))
        vbd = _views_by_dir(inputs.get("views"))

        # Upstream sends the navigator JPEG re-encodes of the raw render
        # (generate_input → image_to_base64, base_il_trainer_llm.py:64-83:
        # PIL save(format="JPEG"), default quality). Decode the env's
        # lossless PNG tile → re-encode JPEG → emit raw-b64 STRINGS (the
        # llmCall rgb port passes b64 strings through verbatim; its config
        # sets image_mime="image/jpeg").
        images: list = []
        image_labels: list[str] = []
        kept = _filter_stuck_keys(
            _candidate_order(candidates.keys()), inputs.get("stuck_directions")
        )
        for dir_id in kept:
            b64 = str(vbd.get(dir_id, {}).get("rgb_base64", ""))
            if not b64:
                continue
            jpeg = _png_b64_to_jpeg_b64(b64)
            if jpeg is None:
                continue
            images.append(jpeg)
            image_labels.append(f"Viewpoint {dir_id}:")

        self._self_log("n_images", len(images))
        return {"images": images, "image_labels": image_labels}


class ParseNavigatorNode(BaseCanvasNode):
    """Verbatim parser from move_to_next_vp_single (spatialNavigator.py:141-194).
    Extracts ``pred_vp`` / ``pred_thought`` / ``completion_estimation``."""

    node_type: ClassVar[str] = "threestepnav__parse_navigator"
    display_name: ClassVar[str] = "3-Step: Parse Navigator"
    description: ClassVar[str] = "Parse Thought / Prediction / Completion Estimation"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Filter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [PortDef("llm_out", "TEXT", "Navigator LLM output")]
    output_ports = [
        PortDef("pred_vp", "TEXT", "Predicted viewpoint id"),
        PortDef("pred_thought", "TEXT", "Pre-Prediction thought"),
        PortDef("completion_estimation", "TEXT", "Yes / No / Unknown"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        pred_vp, pred_thought, completion = parse_navigator(str(inputs.get("llm_out", "")))
        self._self_log("pred_vp", pred_vp)
        self._self_log("completion_estimation", completion)
        return {
            "pred_vp": pred_vp,
            "pred_thought": pred_thought,
            "completion_estimation": completion,
        }


class SelectDirectionObservationNode(BaseCanvasNode):
    """Pick the single-direction observation block for ``pred_vp`` + parse its
    direction id. Mirrors save_history (spatialNavigator.py:40-43)."""

    node_type: ClassVar[str] = "threestepnav__select_direction_observation"
    display_name: ClassVar[str] = "3-Step: Select Direction Observation"
    description: ClassVar[str] = "Per-direction block for pred_vp (+ direction id)"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Crosshair"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("blocks", "ANY", "{dir_id: block} from format_observation"),
        PortDef("pred_vp", "TEXT", "Chosen viewpoint id"),
    ]
    output_ports = [
        PortDef("observation", "TEXT", "Single-direction observation block"),
        PortDef("direction_id", "ANY", "Integer direction id"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        blocks = _as_dict(inputs.get("blocks"))
        pred_vp = str(inputs.get("pred_vp", "")).strip()
        observation = str(blocks.get(pred_vp, "") or "")
        try:
            direction_id = int(pred_vp)
        except (TypeError, ValueError):
            direction_id = 0
        return {"observation": observation, "direction_id": direction_id}


class ObservationSummaryPromptNode(BaseCanvasNode):
    """Assemble the OBSERVATION_SUMMARY prompt (per-step observation compressor)."""

    node_type: ClassVar[str] = "threestepnav__observation_summary_prompt"
    display_name: ClassVar[str] = "3-Step: Observation Summary Prompt"
    description: ClassVar[str] = "Compress the chosen direction's observation"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Minimize2"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("observation", "TEXT", "Chosen-direction observation block")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        # Upstream summarises only from "Scene Description" onward, stripping the
        # "Direction {id} Direction Viewpoint ID: ... Eye Level " prefix
        # (save_history, spatialNavigator.py:42). No maxsplit — if the caption
        # itself contains "Scene Description" upstream truncates at the second
        # occurrence, and so do we.
        observation = str(inputs.get("observation", ""))
        if "Scene Description" in observation:
            observation = "Scene Description" + observation.split("Scene Description")[1]
        return {
            "system": OBSERVATION_SUMMARY_SYSTEM,
            "user": OBSERVATION_SUMMARY_USER.format(observation),
        }


class ThoughtSummaryPromptNode(BaseCanvasNode):
    """Assemble the THOUGHT_SUMMARY prompt (per-step thought compressor)."""

    node_type: ClassVar[str] = "threestepnav__thought_summary_prompt"
    display_name: ClassVar[str] = "3-Step: Thought Summary Prompt"
    description: ClassVar[str] = "Compress the navigator thought"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Minimize2"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [PortDef("thought", "TEXT", "Navigator thought")]
    output_ports = [
        PortDef("system", "TEXT", "System prompt"),
        PortDef("user", "TEXT", "User prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        # The navigator thought is formatted RAW into the prompt; upstream's
        # ``.replace("Thought: ", "")`` applies to the summary LLM's OUTPUT
        # (save_history, spatialNavigator.py:45-46) and lives downstream in
        # build_history_entry.
        thought = str(inputs.get("thought", ""))
        return {
            "system": THOUGHT_SUMMARY_SYSTEM,
            "user": THOUGHT_SUMMARY_USER.format(thought),
        }


class BuildHistoryEntryNode(BaseCanvasNode):
    """Build the per-step history entry ``{step, viewpoint, observation,
    thought}`` (DIRECTIONS-prefixed observation). Mirrors save_history
    (spatialNavigator.py:38-55). Emits the entry as data only — ``judge_decide``
    is the SOLE WRITER of ``nav_history`` (it appends / resets / pops per the
    decision)."""

    node_type: ClassVar[str] = "threestepnav__build_history_entry"
    display_name: ClassVar[str] = "3-Step: Build History Entry"
    description: ClassVar[str] = "Assemble the per-step nav_history entry (no state write)"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Plus"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("viewpoint", "TEXT", "Chosen viewpoint id"),
        PortDef("observation_summary", "TEXT", "Compressed observation"),
        PortDef("thought_summary", "TEXT", "Compressed thought"),
        PortDef("direction_id", "ANY", "Direction id (0..11) for DIRECTIONS prefix"),
    ]
    output_ports = [PortDef("entry", "TEXT", "JSON-encoded history entry")]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        # 1-based like upstream current_step (base_il_trainer_llm.py:600).
        step = (int(getattr(ctx, "step", 0)) if ctx else 0) + 1
        observation_summary = str(inputs.get("observation_summary", ""))
        did_raw = inputs.get("direction_id")
        # Upstream prepends the DIRECTIONS phrase unconditionally
        # (save_history, spatialNavigator.py:41-43).
        if did_raw is not None:
            try:
                did = int(did_raw)
                if 0 <= did < len(DIRECTIONS):
                    observation_summary = f"Direction {DIRECTIONS[did]} " + observation_summary
            except (TypeError, ValueError):
                pass
        # Upstream strips "Thought: " from the thought-summary LLM output
        # (save_history, spatialNavigator.py:45-46).
        entry = {
            "step": step,
            "viewpoint": str(inputs.get("viewpoint", "")),
            "observation": observation_summary,
            "thought": str(inputs.get("thought_summary", "")).replace("Thought: ", ""),
        }
        return {"entry": json.dumps(entry)}


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — the judge / back-check + sub-instruction pointer + env action.
# Single node: folds the (gated, image-augmented) decision-agent LLM call,
# the four-ability decision rules, the pointer/history state machine, and the
# next env action (move vs reverse-backtrack vs decide-to-stop).
# SOLE WRITER of: nav_history, current_action_idx, chosen_images_b64, move_stack.
# ═══════════════════════════════════════════════════════════════════════


class JudgeDecideNode(BaseCanvasNode):
    """The verify step. Gated on ``completion_estimation == "Yes"`` exactly as
    upstream (base_il_trainer_llm.py:728). When fired, runs the image-augmented
    decision agent (make_informed_decision) over the chosen-path image sequence,
    applies the verbatim decision rules, and drives:

      • continue → advance ``current_action_idx`` (+reset nav_history); at last
        sub-instruction → stop. (base_il_trainer_llm.py:778-790)
      • stay / look_around → keep pointer, append entry, move to ``pred_vp``.
      • backtrack → reverse the last move (move-stack pop) + drop the entry.
        (base_il_trainer_llm.py:807-828)
      • should_stop_navigation (idx==last AND continue/stay). (decision_agent.py:943-965)

    When the gate is off (completion != "Yes") the decision is "stay" with no
    extra LLM call. Emits the resolved env action ``(angle, distance)`` and the
    loop ``stop`` flag.
    """

    node_type: ClassVar[str] = "threestepnav__judge_decide"
    display_name: ClassVar[str] = "3-Step: Judge & Decide"
    description: ClassVar[str] = (
        "Back-check verify (continue/stay/backtrack/look-around) + pointer + env action"
    )
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Gavel"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("llm_profile", "text", label="Judge LLM profile", default="gpt-5-mini"),
            ConfigField(
                "enabled_abilities",
                "text",
                label="Enabled abilities (CSV)",
                default="continue,stay,backtrack,look_around",
            ),
            ConfigField("max_tokens", "text", label="max_tokens", default=2000),
        ],
    )
    input_ports = [
        PortDef("completion_estimation", "TEXT", "Yes / No / Unknown from the navigator"),
        PortDef("pred_vp", "TEXT", "Navigator's chosen direction id"),
        PortDef("pred_thought", "TEXT", "Navigator thought (fallback decision thought)"),
        PortDef("current_action", "TEXT", "Current sub-instruction"),
        PortDef("current_landmarks_join", "TEXT", "', '-joined current landmarks"),
        PortDef("action_idx", "ANY", "Current sub-instruction index"),
        PortDef("num_actions", "ANY", "Total sub-instructions"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} from waypoint predictor"),
        PortDef("views", "ANY", "Panorama views [{dir_id, rgb_base64, ...}]"),
        PortDef("entry", "TEXT", "JSON history entry from build_history_entry"),
    ]
    output_ports = [
        PortDef("next_vp", "TEXT", "Chosen viewpoint id (passthrough)"),
        PortDef("angle", "TEXT", "Env action angle (radians)"),
        PortDef("distance", "TEXT", "Env action distance (metres)"),
        PortDef("stop", "BOOL", "Terminate the episode loop"),
        PortDef("decision", "TEXT", "continue / stay / backtrack / look_around"),
        PortDef("decision_thought", "TEXT", "Judge reasoning (or navigator thought)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        cfg = self.config or {}
        profile = str(cfg.get("llm_profile", "gpt-5-mini"))
        max_tokens = int(cfg.get("max_tokens", 2000))
        enabled = [a.strip() for a in str(cfg.get("enabled_abilities", "")).split(",") if a.strip()]
        if not enabled:
            enabled = ["continue", "stay", "backtrack", "look_around"]

        gs = getattr(ctx, "graph_state", None) if ctx else None
        completion = str(inputs.get("completion_estimation", "")).strip()
        pred_vp = str(inputs.get("pred_vp", "")).strip()
        pred_thought = str(inputs.get("pred_thought", ""))
        current_action = str(inputs.get("current_action", ""))
        current_landmarks = str(inputs.get("current_landmarks_join", ""))
        idx = int(_read(gs, "current_action_idx", inputs.get("action_idx") or 0) or 0)
        num_actions = int(inputs.get("num_actions") or 1)
        candidates = _as_dict(inputs.get("candidates"))
        vbd = _views_by_dir(inputs.get("views"))

        try:
            entry = json.loads(str(inputs.get("entry", "{}")))
        except Exception:
            entry = {}

        # Forward move resolved from the navigator's pick.
        cand = candidates.get(pred_vp)
        fwd_angle, fwd_distance = 0.0, 0.0
        if isinstance(cand, (list, tuple)) and len(cand) >= 2:
            fwd_angle, fwd_distance = float(cand[0]), float(cand[1])
        elif isinstance(cand, dict):
            fwd_angle = float(cand.get("angle", 0.0))
            fwd_distance = float(cand.get("distance", 0.0))

        # Accumulate the chosen-path image sequence (seed with the initial
        # forward view; base_il_trainer_llm.py:487-491,687).
        chosen = list(_as_list(_read(gs, "chosen_images_b64", [])))
        if not chosen:
            seed = str(vbd.get("0", {}).get("rgb_base64", ""))
            if seed:
                chosen.append(seed)
        chosen_rgb = str(vbd.get(pred_vp, {}).get("rgb_base64", ""))
        if chosen_rgb:
            chosen.append(chosen_rgb)

        # Per-image descriptions (1:1 with chosen) — the judge's prepended
        # "Image sequence descriptions" block. Built from PRIOR state, like
        # chosen. Step phrase is 1-based like upstream current_step.
        step = (int(getattr(ctx, "step", 0)) if ctx else 0) + 1
        descs = _build_descs(gs, vbd, pred_vp, step, fwd_angle)

        prior_history = [e for e in _as_list(_read(gs, "nav_history", [])) if isinstance(e, dict)]
        move_stack = list(_as_list(_read(gs, "move_stack", [])))

        decision = "stay"
        reasoning = ""
        stop = False
        new_idx = idx
        new_history = [*prior_history, entry] if entry else list(prior_history)
        action_mode = "move"
        out_angle, out_distance = fwd_angle, fwd_distance
        clear_stuck = False

        if completion == "Yes":
            # Upstream reviews the history AFTER save_history appended this
            # step's entry (base_il_trainer_llm.py:683→722), so the judge's
            # History field INCLUDES the current step.
            judge_history = _flatten_history(new_history)
            judge_user = build_judge_user(
                current_action, current_landmarks, judge_history, len(chosen), descriptions=descs
            )
            raw = await self._judge_llm(profile, judge_user, chosen, max_tokens)
            dstr, confidence, reasoning = parse_judge(raw)
            # Upstream tags the judge reasoning (decision_agent.py:790).
            reasoning = f"[Code-Informed] {reasoning}"
            decision = string_to_decision(dstr)
            decision = apply_decision_rules(
                decision, confidence, enabled, idx, num_actions, len(chosen)
            )
            self._self_log("judge_confidence", confidence)

            if decision == "continue":
                if idx < num_actions - 1:
                    new_idx = idx + 1
                    new_history = []  # reset per-sub-instruction history
                    clear_stuck = True  # base_il_trainer_llm.py:786
                else:
                    stop = True  # continued past the last sub-instruction
            elif decision == "backtrack":
                # Reverse the last executed move (env_actions_history pop +
                # _create_reverse_action); drop the just-built entry + last image.
                action_mode = "reverse"
                last = move_stack.pop() if move_stack else [0.0, 0.0]
                rev = (float(last[0]) + math.pi + math.pi) % (2 * math.pi) - math.pi
                out_angle, out_distance = rev, float(last[1])
                new_history = list(prior_history)
                if chosen:
                    chosen.pop()
                if descs:
                    descs.pop()
            # stay / look_around → keep pointer, append entry, normal move.

            # should_stop_navigation (decision_agent.py:957-959)
            if idx == num_actions - 1 and decision in ("continue", "stay"):
                stop = True

        if stop:
            out_angle, out_distance = 0.0, 0.0

        # Push the forward move so a future backtrack can reverse it.
        if action_mode == "move" and not stop and out_distance > 0:
            move_stack.append([out_angle, out_distance])

        _write(gs, "nav_history", new_history)
        _write(gs, "current_action_idx", new_idx)
        _write(gs, "chosen_images_b64", chosen)
        _write(gs, "chosen_descriptions", descs)
        _write(gs, "move_stack", move_stack)
        _write(gs, "last_chosen_vp", pred_vp)  # base_il_trainer_llm.py:669
        if clear_stuck:
            _write(gs, "stuck_directions", [])

        self._self_log("completion", completion)
        self._self_log("decision", decision)
        self._self_log("action_mode", action_mode)
        self._self_log("idx", idx)
        self._self_log("new_idx", new_idx)
        self._self_log("stop", stop)
        return {
            "next_vp": pred_vp,
            "angle": f"{out_angle:.6f}",
            "distance": f"{out_distance:.6f}",
            "stop": stop,
            "decision": decision,
            "decision_thought": reasoning or pred_thought,
        }

    async def _judge_llm(
        self, profile: str, user_prompt: str, images_b64: list[str], max_tokens: int
    ) -> str:
        from app.llm.call import get_llm_config, vlm_complete

        cfg = get_llm_config(profile or "")
        if cfg is None:
            log.warning("threestepnav: judge LLM profile '%s' not found", profile)
            return ""
        # Mirror gpt_infer_with_images (api.py:154-177): "Viewpoint {i}:" before
        # each chosen-path image (i = sequence index), detail="high". Upstream
        # JPEG-re-encodes the chosen PIL images (decision_agent.py:806-815) —
        # reproduce that byte chain from the stored lossless tiles.
        jpegs = [j for j in (_png_b64_to_jpeg_b64(b) for b in images_b64) if j]
        labels = [f"Viewpoint {i}:" for i in range(len(jpegs))]
        out = await vlm_complete(
            cfg,
            user_prompt,
            jpegs,
            image_labels=labels,
            system_prompt=JUDGE_SYSTEM,
            max_tokens=max_tokens,
            temperature=1.0,
            detail="high",
            mime="image/jpeg",
        )
        return out or ""


class SelectCandidateViewsNode(BaseCanvasNode):
    """Filter panorama views down to the waypoint candidates, keyed by dir_id.

    Method glue extracted from ``opennav_perception`` (TODO #56), duplicated
    from ``opennav.SelectCandidateViewsNode`` so this nodeset stays
    self-contained: only candidate directions get the expensive
    RAM/SpatialBot passes; the FM wrappers (``model_ram__tag_views`` /
    ``vlm_spatialbot__caption_views``) consume the keyed dict as-is.
    Empty ``candidates`` → all views pass through (legacy semantics).
    """

    node_type: ClassVar[str] = "threestepnav__select_candidate_views"
    display_name: ClassVar[str] = "Three-Step: Select Candidate Views"
    description: ClassVar[str] = "views + candidates → {dir_id: view} for candidate dirs only"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Filter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64, depth_*} from panorama_rgbd"),
        PortDef("candidates", "ANY", "{dir_id: ...} from waypoint predictor"),
    ]
    output_ports = [
        PortDef("candidate_views", "ANY", "{dir_id: view dict} in panorama order"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        candidates = inputs.get("candidates") or {}
        keys = {str(k) for k in candidates.keys()} if candidates else None
        out: dict = {}
        for v in views:
            if not isinstance(v, dict):
                continue
            dir_id = str(v.get("dir_id"))
            if keys is None or dir_id in keys:
                out[dir_id] = v
        self._self_log("num_candidates", len(out))
        return {"candidate_views": out}


# ═══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ═══════════════════════════════════════════════════════════════════════


class ThreeStepNavNodeSet(BaseNodeSet):
    """Three-Step Nav — global decompose · local image-navigator · back-check verify."""

    name = "threestepnav"
    description = (
        "Three-Step Nav (arXiv 2604.26946) zero-shot VLN-CE: instruction decomposition, "
        "MapGPT-style image navigator with folded completion estimation, and a "
        "continue/stay/backtrack/look-around judge driving a sub-instruction pointer. "
        "Pairs with env_habitat + opennav_waypoint + model_ram + vlm_spatialbot."
    )
    # ~4 LLM calls/step (navigator + 2 summaries + gated judge) over 1024px tiles.
    default_per_step_budget_sec = 60.0

    def get_tools(self) -> list:
        return [
            DetectActionsPromptNode(),
            ExtractLandmarksNode(),
            SelectSubinstructionNode(),
            FormatObservationNode(),
            ReviewHistoryNode(),
            NavigatorPromptNode(),
            BuildImagesNode(),
            ParseNavigatorNode(),
            SelectDirectionObservationNode(),
            ObservationSummaryPromptNode(),
            ThoughtSummaryPromptNode(),
            BuildHistoryEntryNode(),
            JudgeDecideNode(),
            StuckDetectNode(),
            SelectCandidateViewsNode(),
        ]
