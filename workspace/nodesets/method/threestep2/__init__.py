"""Three-Step Nav — DECOMPOSED method nodeset (Phase-2 of the ``threestep`` mono).

The ``threestep`` monolith concentrated ALL four state writes
(``current_action_idx`` / ``nav_history`` / ``chosen_images_b64`` / ``move_stack``)
plus the gated judge LLM call, the four-ability decision rules, the pointer/history
state machine, and the env-action resolution into a single ``judge_decide`` node.
This nodeset splits that one fat node along the single-writer-per-state-key seam:

    judge_decide  →  judge_verdict     gated judge LLM + parse + decision rules → ``decision``
                                       (NO state write)
                     resolve_action    pure: decision + candidates + move_stack → angle/distance/stop
                                       (NO state write)
                     update_nav_state   SOLE WRITER of all four state keys; applies the
                                       decision to the pointer / history / images / move-stack

The other twelve method nodes are byte-identical to the mono — they are reused by
subclassing the mono classes and overriding only ``node_type`` (``threestep2__*``).

Ground-truth: ``workspace/nodesets/method/threestep`` (the validated mono). The
``test_equivalence.py`` byte-compares this decomp's judge split against the mono's
``judge_decide`` over hand-crafted scenarios — that is the Phase-2 Tier-1 gate.

Reader/writer contract (race-free by dependency order, no snapshot node needed):
``judge_verdict`` reads prior ``chosen_images_b64``; ``resolve_action`` reads prior
``move_stack``; both fire BEFORE ``update_nav_state`` (which depends on their outputs)
so they always see the pre-write state. ``update_nav_state`` writes last.

last updated: 2026-06-22
"""

from __future__ import annotations

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

# Reuse helpers, prompts, and the twelve unchanged nodes from the validated mono.
from workspace.nodesets.method.threestep import (
    BuildHistoryEntryNode,
    BuildImagesNode,
    DetectActionsPromptNode,
    ExtractLandmarksNode,
    FormatObservationNode,
    NavigatorPromptNode,
    ObservationSummaryPromptNode,
    ParseNavigatorNode,
    ReviewHistoryNode,
    SelectDirectionObservationNode,
    SelectSubinstructionNode,
    StuckDetectNode,
    ThoughtSummaryPromptNode,
    _as_dict,
    _as_list,
    _build_descs,
    _read,
    _views_by_dir,
    _write,
)
from workspace.nodesets.method.threestep._prompts import (
    JUDGE_SYSTEM,
    apply_decision_rules,
    build_judge_user,
    parse_judge,
    string_to_decision,
)

log = logging.getLogger("agentcanvas.threestep2")


# ── helpers (decomp-local) ───────────────────────────────────────────────


def _build_chosen(gs: Any, vbd: dict, pred_vp: str) -> list:
    """Recompute the chosen-path image sequence exactly as the mono does
    (base_il_trainer_llm.py:487-491,687): seed with the current-orientation
    view on the first step, then append the chosen direction's tile.

    Both ``judge_verdict`` and ``update_nav_state`` call this on the SAME prior
    ``chosen_images_b64`` (only ``update_nav_state`` writes it), so they agree."""
    chosen = list(_as_list(_read(gs, "chosen_images_b64", [])))
    if not chosen:
        seed = str(vbd.get("0", {}).get("rgb_base64", ""))
        if seed:
            chosen.append(seed)
    chosen_rgb = str(vbd.get(pred_vp, {}).get("rgb_base64", ""))
    if chosen_rgb:
        chosen.append(chosen_rgb)
    return chosen


def _fwd_angle(candidates: dict, pred_vp: str) -> float:
    """The navigator's forward-pick angle (radians) for the image description
    phrase — candidates[pred_vp][0], same as mono ``fwd_angle``."""
    cand = candidates.get(str(pred_vp))
    if isinstance(cand, (list, tuple)) and len(cand) >= 2:
        return float(cand[0])
    if isinstance(cand, dict):
        return float(cand.get("angle", 0.0))
    return 0.0


# ── 12 unchanged nodes — subclass with a fresh node_type (body identical) ──


class DSDetectActionsPromptNode(DetectActionsPromptNode):
    node_type: ClassVar[str] = "threestep2__detect_actions_prompt"


class DSExtractLandmarksNode(ExtractLandmarksNode):
    node_type: ClassVar[str] = "threestep2__extract_landmarks"


class DSSelectSubinstructionNode(SelectSubinstructionNode):
    node_type: ClassVar[str] = "threestep2__select_subinstruction"


class DSFormatObservationNode(FormatObservationNode):
    node_type: ClassVar[str] = "threestep2__format_observation"


class DSReviewHistoryNode(ReviewHistoryNode):
    node_type: ClassVar[str] = "threestep2__review_history"


class DSNavigatorPromptNode(NavigatorPromptNode):
    node_type: ClassVar[str] = "threestep2__navigator_prompt"


class DSBuildImagesNode(BuildImagesNode):
    node_type: ClassVar[str] = "threestep2__build_images"


class DSParseNavigatorNode(ParseNavigatorNode):
    node_type: ClassVar[str] = "threestep2__parse_navigator"


class DSSelectDirectionObservationNode(SelectDirectionObservationNode):
    node_type: ClassVar[str] = "threestep2__select_direction_observation"


class DSObservationSummaryPromptNode(ObservationSummaryPromptNode):
    node_type: ClassVar[str] = "threestep2__observation_summary_prompt"


class DSThoughtSummaryPromptNode(ThoughtSummaryPromptNode):
    node_type: ClassVar[str] = "threestep2__thought_summary_prompt"


class DSBuildHistoryEntryNode(BuildHistoryEntryNode):
    node_type: ClassVar[str] = "threestep2__build_history_entry"


class DSStuckDetectNode(StuckDetectNode):
    node_type: ClassVar[str] = "threestep2__stuck_detect"


# ═══════════════════════════════════════════════════════════════════════
# Step 3 split — judge verdict · action resolution · state update
# ═══════════════════════════════════════════════════════════════════════


class JudgeVerdictNode(BaseCanvasNode):
    """The gated judge LLM + four-ability decision. Mirrors the
    ``completion == "Yes"`` block of mono ``judge_decide`` (base_il_trainer_llm.py:728,
    decision_agent.make_informed_decision_with_capture). Reads the prior
    ``chosen_images_b64`` to build the chosen-path image sequence for the judge,
    but does NOT write any state."""

    node_type: ClassVar[str] = "threestep2__judge_verdict"
    display_name: ClassVar[str] = "3-Step₂: Judge Verdict"
    description: ClassVar[str] = "Gated judge LLM + decision rules → decision (no state write)"
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
        PortDef("history_traj", "TEXT", "Flattened nav history"),
        PortDef("action_idx", "ANY", "Current sub-instruction index"),
        PortDef("num_actions", "ANY", "Total sub-instructions"),
        PortDef("views", "ANY", "Panorama views [{dir_id, rgb_base64, ...}]"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} — for the image-desc phrase"),
    ]
    output_ports = [
        PortDef("decision", "TEXT", "continue / stay / backtrack / look_around"),
        PortDef("decision_thought", "TEXT", "Judge reasoning (or navigator thought)"),
        PortDef("confidence", "ANY", "Judge confidence (0..10) or None when gated off"),
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
        history_traj = str(inputs.get("history_traj", ""))
        idx = int(inputs.get("action_idx") or 0)
        num_actions = int(inputs.get("num_actions") or 1)
        vbd = _views_by_dir(inputs.get("views"))
        candidates = _as_dict(inputs.get("candidates"))
        step = int(getattr(ctx, "step", 0)) if ctx else 0

        # chosen-path image sequence + its descriptions (same prior state both
        # judge_verdict and update_nav_state read; only update_nav_state writes).
        chosen = _build_chosen(gs, vbd, pred_vp)
        descs = _build_descs(gs, vbd, pred_vp, step, _fwd_angle(candidates, pred_vp))

        decision = "stay"
        reasoning = ""
        confidence: Any = None
        if completion == "Yes":
            judge_user = build_judge_user(
                current_action, current_landmarks, history_traj, len(chosen), descriptions=descs
            )
            raw = await self._judge_llm(profile, judge_user, chosen, max_tokens)
            dstr, confidence, reasoning = parse_judge(raw)
            decision = string_to_decision(dstr)
            decision = apply_decision_rules(
                decision, confidence, enabled, idx, num_actions, len(chosen)
            )
            self._self_log("judge_confidence", confidence)

        self._self_log("completion", completion)
        self._self_log("decision", decision)
        return {
            "decision": decision,
            "decision_thought": reasoning or pred_thought,
            "confidence": confidence,
        }

    async def _judge_llm(
        self, profile: str, user_prompt: str, images_b64: list[str], max_tokens: int
    ) -> str:
        from app.llm.call import get_llm_config, vlm_complete

        cfg = get_llm_config(profile or "")
        if cfg is None:
            log.warning("threestep2: judge LLM profile '%s' not found", profile)
            return ""
        # Mirror gpt_infer_with_images (api.py:154-177): "Viewpoint {i}:" before
        # each chosen-path image (i = sequence index), detail="high".
        labels = [f"Viewpoint {i}:" for i in range(len(images_b64))]
        out = await vlm_complete(
            cfg,
            user_prompt,
            list(images_b64),
            image_labels=labels,
            system_prompt=JUDGE_SYSTEM,
            max_tokens=max_tokens,
            temperature=1.0,
            detail="high",
        )
        return out or ""


class ResolveActionNode(BaseCanvasNode):
    """Pure action resolver. Mirrors the env-action half of mono ``judge_decide``
    (base_il_trainer_llm.py:849-856 forward move; :807-828 backtrack reverse;
    decision_agent.py:957-959 should_stop_navigation; :788 continue-past-last).
    Reads the prior ``move_stack`` to compute a backtrack reverse but writes
    nothing — ``update_nav_state`` owns the pop."""

    node_type: ClassVar[str] = "threestep2__resolve_action"
    display_name: ClassVar[str] = "3-Step₂: Resolve Action"
    description: ClassVar[str] = "decision + candidates → angle / distance / stop (no state write)"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("decision", "TEXT", "continue / stay / backtrack / look_around"),
        PortDef("completion_estimation", "TEXT", "Yes / No / Unknown (gates stop logic)"),
        PortDef("pred_vp", "TEXT", "Navigator's chosen direction id"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} from waypoint predictor"),
        PortDef("action_idx", "ANY", "Current sub-instruction index"),
        PortDef("num_actions", "ANY", "Total sub-instructions"),
    ]
    output_ports = [
        PortDef("angle", "TEXT", "Env action angle (radians, 6dp)"),
        PortDef("distance", "TEXT", "Env action distance (metres, 6dp)"),
        PortDef("stop", "BOOL", "Terminate the episode loop"),
        PortDef("action_mode", "TEXT", "move / reverse"),
        PortDef("move_angle", "ANY", "Resolved move angle (float) for move_stack push"),
        PortDef("move_distance", "ANY", "Resolved move distance (float) for move_stack push"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        gs = getattr(ctx, "graph_state", None) if ctx else None
        decision = str(inputs.get("decision", "stay")).strip() or "stay"
        completion = str(inputs.get("completion_estimation", "")).strip()
        pred_vp = str(inputs.get("pred_vp", "")).strip()
        candidates = _as_dict(inputs.get("candidates"))
        idx = int(inputs.get("action_idx") or 0)
        num_actions = int(inputs.get("num_actions") or 1)
        move_stack = _as_list(_read(gs, "move_stack", []))

        # forward move resolved from the navigator's pick
        cand = candidates.get(pred_vp)
        fwd_angle, fwd_distance = 0.0, 0.0
        if isinstance(cand, (list, tuple)) and len(cand) >= 2:
            fwd_angle, fwd_distance = float(cand[0]), float(cand[1])
        elif isinstance(cand, dict):
            fwd_angle = float(cand.get("angle", 0.0))
            fwd_distance = float(cand.get("distance", 0.0))

        action_mode = "move"
        out_angle, out_distance = fwd_angle, fwd_distance
        stop = False

        if completion == "Yes":
            if decision == "continue":
                if idx >= num_actions - 1:
                    stop = True  # continued past the last sub-instruction
            elif decision == "backtrack":
                action_mode = "reverse"
                last = move_stack[-1] if move_stack else [0.0, 0.0]
                rev = (float(last[0]) + math.pi + math.pi) % (2 * math.pi) - math.pi
                out_angle, out_distance = rev, float(last[1])
            # should_stop_navigation (decision_agent.py:957-959)
            if idx == num_actions - 1 and decision in ("continue", "stay"):
                stop = True

        if stop:
            out_angle, out_distance = 0.0, 0.0

        self._self_log("action_mode", action_mode)
        self._self_log("stop", stop)
        return {
            "angle": f"{out_angle:.6f}",
            "distance": f"{out_distance:.6f}",
            "stop": stop,
            "action_mode": action_mode,
            "move_angle": out_angle,
            "move_distance": out_distance,
        }


class UpdateNavStateNode(BaseCanvasNode):
    """SOLE WRITER of ``current_action_idx`` / ``nav_history`` / ``chosen_images_b64``
    / ``move_stack``. Applies the verdict to the pointer-and-history state machine
    (base_il_trainer_llm.py:778-828) and pushes/pops the move-stack. Passes ``stop``
    through to ``iter_out`` so it is guaranteed to fire (and write) before the loop
    re-iterates."""

    node_type: ClassVar[str] = "threestep2__update_nav_state"
    display_name: ClassVar[str] = "3-Step₂: Update Nav State"
    description: ClassVar[str] = "Sole writer: pointer / history / images / move_stack"
    category: ClassVar[str] = "agent"
    icon: ClassVar[str] = "Save"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("decision", "TEXT", "continue / stay / backtrack / look_around"),
        PortDef("completion_estimation", "TEXT", "Yes / No / Unknown"),
        PortDef("action_mode", "TEXT", "move / reverse (from resolve_action)"),
        PortDef("stop", "BOOL", "Loop stop flag (from resolve_action)"),
        PortDef("move_angle", "ANY", "Resolved move angle (float)"),
        PortDef("move_distance", "ANY", "Resolved move distance (float)"),
        PortDef("pred_vp", "TEXT", "Navigator's chosen direction id"),
        PortDef("views", "ANY", "Panorama views [{dir_id, rgb_base64, ...}]"),
        PortDef("entry", "TEXT", "JSON history entry from build_history_entry"),
        PortDef("action_idx", "ANY", "Current sub-instruction index"),
        PortDef("num_actions", "ANY", "Total sub-instructions"),
        PortDef("candidates", "ANY", "{dir_id: [angle, distance]} — for the image-desc phrase"),
        PortDef("next_vp", "TEXT", "pred_vp passthrough for the action log", optional=True),
    ]
    output_ports = [
        PortDef("stop", "BOOL", "Terminate the episode loop (passthrough)"),
        PortDef("new_action_idx", "ANY", "Sub-instruction index after the update"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        gs = getattr(ctx, "graph_state", None) if ctx else None
        decision = str(inputs.get("decision", "stay")).strip() or "stay"
        completion = str(inputs.get("completion_estimation", "")).strip()
        action_mode = str(inputs.get("action_mode", "move")).strip() or "move"
        stop = bool(inputs.get("stop"))
        out_distance = float(inputs.get("move_distance") or 0.0)
        out_angle = float(inputs.get("move_angle") or 0.0)
        pred_vp = str(inputs.get("pred_vp", "")).strip()
        idx = int(inputs.get("action_idx") or 0)
        num_actions = int(inputs.get("num_actions") or 1)
        vbd = _views_by_dir(inputs.get("views"))
        candidates = _as_dict(inputs.get("candidates"))
        step = int(getattr(ctx, "step", 0)) if ctx else 0

        try:
            entry = json.loads(str(inputs.get("entry", "{}")))
        except Exception:
            entry = {}

        chosen = _build_chosen(gs, vbd, pred_vp)
        descs = _build_descs(gs, vbd, pred_vp, step, _fwd_angle(candidates, pred_vp))
        prior_history = [e for e in _as_list(_read(gs, "nav_history", [])) if isinstance(e, dict)]
        move_stack = list(_as_list(_read(gs, "move_stack", [])))

        new_idx = idx
        new_history = [*prior_history, entry] if entry else list(prior_history)
        clear_stuck = False

        if completion == "Yes":
            if decision == "continue":
                if idx < num_actions - 1:
                    new_idx = idx + 1
                    new_history = []  # reset per-sub-instruction history
                    clear_stuck = True  # base_il_trainer_llm.py:786
                # else: stop (idx unchanged) — handled by resolve_action
            elif decision == "backtrack":
                # reverse the last executed move: pop move-stack, drop entry + last image
                if move_stack:
                    move_stack.pop()
                new_history = list(prior_history)
                if chosen:
                    chosen.pop()
                if descs:
                    descs.pop()
            # stay / look_around → keep pointer, append entry, normal move

        # push the forward move so a future backtrack can reverse it
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

        self._self_log("idx", idx)
        self._self_log("new_idx", new_idx)
        self._self_log("stop", stop)
        return {"stop": stop, "new_action_idx": new_idx}


# ═══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ═══════════════════════════════════════════════════════════════════════


class ThreeStep2NodeSet(BaseNodeSet):
    """Three-Step Nav (decomposed) — judge_decide split into verdict · resolve · update."""

    name = "threestep2"
    description = (
        "Decomposed Three-Step Nav: same pipeline as the threestep mono, with the "
        "monolithic judge_decide split along single-writer-per-state-key into "
        "judge_verdict (gated judge LLM) + resolve_action (pure env action) + "
        "update_nav_state (sole state writer). Byte-equivalent to the mono (see "
        "test_equivalence.py). Pairs with env_habitat + opennav_waypoint + opennav_perception."
    )
    default_per_step_budget_sec = 60.0

    def get_tools(self) -> list:
        return [
            DSDetectActionsPromptNode(),
            DSExtractLandmarksNode(),
            DSSelectSubinstructionNode(),
            DSFormatObservationNode(),
            DSReviewHistoryNode(),
            DSNavigatorPromptNode(),
            DSBuildImagesNode(),
            DSParseNavigatorNode(),
            DSSelectDirectionObservationNode(),
            DSObservationSummaryPromptNode(),
            DSThoughtSummaryPromptNode(),
            DSBuildHistoryEntryNode(),
            JudgeVerdictNode(),
            ResolveActionNode(),
            UpdateNavStateNode(),
            DSStuckDetectNode(),
        ]
