"""HarnessedToolset — every organ that lives at the tool boundary (插桩点 ①).

Wraps any NodesetToolSet *instance* (Habitat / Waypoint / Hybrid) without
touching its code — the wrapper is additive, so ``check_equivalence.py`` and
every baseline cell stay byte-identical. One wrapper instance = one episode;
sub-sessions come and go around it (``begin_subgoal`` / ``end_subgoal``).

What happens on one ``execute(name, args)``:

  1. harness-own tools (recall / finish_subgoal) are answered locally;
  2. decision-coupled state fields are stripped off the args
     (the inner toolset must see byte-identical calls);
  3. pre_action  — V0 guards; a tripped guard BLOCKS the action and returns
     forced feedback; consecutive trips preempt the sub-session;
  4. stop interception — the pre_stop verification gate (V1 judge; a veto
     forces a look, never flips the outcome; second attempt always passes);
  5. forward to the inner toolset;
  6. post_observation — frame log + event-driven keyframe promotion +
     guard bookkeeping + heartbeat (harness-written, overwriting);
  7. state writes applied (with provenance) and echoed back to the model;
  8. the state render + guard notices are APPENDED TO THE TOOL RESULT —
     the one injection channel that works for every executor, closed loops
     included (implementation-plan v2 §2).

v1 divergence from spec §3.3, recorded honestly: sub-agent proposals are
committed directly with ``by="sub"`` provenance instead of staged-then-
committed by the planner. Monotone sets only grow, so a bogus landmark stays
(marked); rejected receipts steer the planner rather than rolling back state.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any, Callable

from eharness.frames import FrameLog
from eharness.guards import GuardConfig, Guards
from eharness.heartbeat import Heartbeat
from eharness.judge import judge as run_judge
from eharness.judge import judge_milestone as run_judge_milestone
from eharness.judge import judge_route as run_judge_route
from eharness.judge import judge_stop as run_judge_stop
from eharness.receipts import Receipt
from eharness.state_block import STATE_UPDATE_FIELDS, StateBlock

STATE_KEYS = tuple(STATE_UPDATE_FIELDS)

FINISH_SUBGOAL_DESC = (
    "End your CURRENT SUBGOAL (not the whole task) and report back. "
    "claim: reached / not_reached / blocked. what_i_see: one line describing "
    "the current view. not_done: what remains or where you gave up."
)
FINISH_SUBGOAL_SCHEMA = {
    "type": "object", "title": "finishSubgoalArguments",
    "properties": {
        "claim": {"type": "string", "enum": ["reached", "not_reached", "blocked"]},
        "what_i_see": {"type": "string"},
        "not_done": {"type": "string"},
    },
    "required": ["claim"],
}
RECALL_DESC = (
    "View stored keyframes from earlier in this episode. query: a landmark "
    "name (e.g. 'stairs'), a frame number, 'dead end', or 'segment N' to "
    "replay the filmstrip of a completed sub-instruction. Returns the "
    "matching images. Pure read — costs nothing."
)
RECALL_SCHEMA = {
    "type": "object", "title": "recallArguments",
    "properties": {"query": {"type": "string"}}, "required": ["query"],
}

_STATE_NOTE = (
    " You may ALSO fill any of progress_note / advance_subgoal / landmark / "
    "dead_end on this call to update the shared state (one short phrase each)."
)


def _goto_place(args: dict[str, Any]) -> Any:
    """The goto target's REAL identity. The DWP schema calls the field
    ``place``; the older wp surface called it ``waypoint``. Reading only the
    old name recorded every DWP goto as ``goto#None`` — and a `None != None`
    comparison then judged a REPEATED goto as corrective, exactly backwards."""
    return args.get("place", args.get("waypoint"))


def _png_from_part(part: dict[str, Any]) -> bytes | None:
    if not (isinstance(part, dict) and part.get("type") == "image_url"):
        return None
    url = (part.get("image_url") or {}).get("url", "")
    if "," in url:
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:  # noqa: BLE001
            return None
    return None


class HarnessedToolset:
    """Same duck type as NodesetToolSet: tool_schemas() + execute() (+ the
    counters mini's adapter reads), so it drops into HabitatEnvironment's
    place unchanged."""

    MOVE_TOOLS = ("step", "goto", "face")

    def __init__(
        self,
        inner: Any,                    # NodesetToolSet instance (untouched)
        *,
        state: StateBlock,
        live_dir: Path | None,
        judge_model: str | None = None,      # None → verification off (ablation)
        judge_kwargs: dict[str, Any] | None = None,
        guard_config: GuardConfig | None = None,
        emit: Callable[[str, dict], None] | None = None,   # EventSink passthrough
        subgoal_turn_cap: int = 30,
        inject_state: bool = True,   # False in solo: the model layer owns the
                                     # ephemeral state prefix instead (spec §2:
                                     # tool-result injection is for closed loops)
        auto_view: bool = True,      # imageless moves get their outcome view
                                     # attached (transient — dropped next turn)
        verify_moves: bool = True,   # judge reconciles expect vs outcome view
        reflect_every: int = 0,      # >0: BLOCK a move every N moves until the
                                     # model states whether the segment is done
                                     # and whether the stop condition holds
        store_refs: bool = False,    # §3.4: stored transcript keeps image_ref
                                     # handles; the payload compiler
                                     # materialises them (solo/mini path only —
                                     # the SDK bridge needs real images inline)
    ) -> None:
        self.inner = inner
        self.state = state
        self.live_dir = live_dir
        self.frames = FrameLog(live_dir)
        self.guards = Guards(self.frames, guard_config)
        self.heartbeat = Heartbeat()
        self.judge_model = judge_model
        self.judge_kwargs = judge_kwargs or {}
        self.inject_state = inject_state
        self.auto_view = auto_view
        self.verify_moves = verify_moves
        self.reflect_every = int(reflect_every or 0)
        self._emit = emit or (lambda kind, payload: None)
        self.subgoal_turn_cap = subgoal_turn_cap
        self.store_refs = bool(store_refs)
        self._map_n = 0                        # versioned model-map counter

        self._stop_attempts = 0
        self._verify_calls = 0
        self._motion_pending: list[str] = []   # motor record since last frame
        self._seg_start = 0                    # first frame of the open segment
        self._last_goto_wp: Any = None         # for corrective-goto detection
        self._crisis_start: int | None = None  # frame idx where a stall began
        self._stall_ref: int | None = None     # the stuck view's hash, for veer
        self._milestone_miss: dict[int, tuple[int, str]] = {}  # sub → (frames, why)
        self._single_fwd_streak = 0            # pace advisory bookkeeping
        self._moves_since_reflect = 0          # segment-completion check cadence
        self._reflect_pending = False          # a demanded answer is outstanding
        self._reflect_answers = 0              # refusals to answer in booleans
        self._arrival_last_check = -10         # frame idx of last arrival bell
        self._last_expect: str = ""            # this call's expect, for the check
        self._receipt: Receipt | None = None    # the in-flight subgoal
        self._steps_at_begin = 0
        self._pending_finish: dict[str, Any] | None = None

    # ── the NodesetToolSet surface mini's layers expect ──

    @property
    def calls_by_tool(self) -> dict[str, int]:
        return self.inner.calls_by_tool

    @property
    def steps_taken(self) -> int:
        return self.inner.steps_taken

    @property
    def end_reason(self):  # noqa: ANN201 — mirror inner's loose typing
        return self.inner.end_reason

    @property
    def episode_over(self) -> bool:
        return self.inner.episode_over

    def tool_names(self) -> list[str]:
        return [s["name"] for s in self.tool_schemas()]

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Inner schemas + decision-coupled state fields + harness tools."""
        schemas = copy.deepcopy(self.inner.tool_schemas())
        for s in schemas:
            s["input_schema"].setdefault("properties", {}).update(
                copy.deepcopy(STATE_UPDATE_FIELDS))
            s["description"] += _STATE_NOTE
        schemas.append({"name": "recall", "description": RECALL_DESC,
                        "input_schema": copy.deepcopy(RECALL_SCHEMA)})
        inner_names = {s2["name"] for s2 in self.inner.tool_schemas()}
        if ("look_around" not in inner_names and "step" in inner_names
                and "goto" not in inner_names):
            # §5.3: look_around is a BARE-surface synthesis only. The DWP
            # surface turns with face() and re-proposes from the fresh view;
            # a standing four-image scan tool on that surface was the thing
            # models burned 64 % of a step budget on.
            schemas.append({
                "name": "look_around",
                "description": (
                    "Scan your surroundings: four views (ahead / right / "
                    "behind / left), heading restored afterwards. Costs about "
                    "24 env steps — use when you need bearings, not every turn."),
                "input_schema": {"type": "object", "title": "lookAroundArguments",
                                 "properties": {}}})
        if "step" in inner_names:
            schemas.append({
                "name": "veer",
                "description": (
                    "When forward is blocked: YOU pick the side that looks "
                    "open; the harness makes the small turn (15°, one more "
                    "notch to 30° if still blocked) plus a forward step FOR "
                    "you and returns the outcome view."),
                "input_schema": {"type": "object", "title": "veerArguments",
                                 "properties": {"direction": {
                                     "type": "string",
                                     "enum": ["left", "right"],
                                     "description": "which side looks open"}},
                                 "required": ["direction"]}})
        if self._receipt is not None:  # only a sub-session may end a subgoal
            schemas.append({"name": "finish_subgoal",
                            "description": FINISH_SUBGOAL_DESC,
                            "input_schema": copy.deepcopy(FINISH_SUBGOAL_SCHEMA)})
        return schemas

    # ── subgoal lifecycle (session boundary hooks land here) ──

    def begin_subgoal(self, subgoal: str, budget: int) -> None:
        self._receipt = Receipt(subgoal=subgoal)
        self._steps_at_begin = self.inner.steps_taken
        self._pending_finish = None
        self.guards.state.consecutive_trips = 0
        self.state.delegation = {"subgoal": subgoal, "budget": budget, "used": 0}
        self.state.save()
        self.heartbeat.subgoal = subgoal
        self.heartbeat.steps_budget = budget
        self.heartbeat.steps_used = 0
        self.heartbeat.tool_calls = 0
        self.heartbeat.status = "running"
        self._write_heartbeat()

    def end_subgoal(self, end_reason: str) -> Receipt:
        r = self._receipt or Receipt(subgoal="?")
        fin = self._pending_finish or {}
        r.claim = fin.get("claim", r.claim)
        r.what_i_see = fin.get("what_i_see", "")[:200]
        r.not_done = fin.get("not_done", "")[:300]
        r.steps_used = self.inner.steps_taken - self._steps_at_begin
        r.guard_trips = self.guards.state.trips_total
        r.end_reason = end_reason
        # evidence = keyframes promoted during this subgoal's window
        r.evidence_frames = [f.idx for f in self.frames.keyframes()
                             if f.step >= self._steps_at_begin][-3:]
        if self.inner.episode_over:
            r.claim = "episode_over"
        elif end_reason == "subgoal_budget_exhausted":
            r.claim = "budget_exhausted"
        elif end_reason == "preempted":
            r.claim = "blocked"
        self._receipt = None
        self.state.delegation = None
        self.heartbeat.status = "done" if end_reason == "subgoal_done" else end_reason
        self._write_heartbeat()
        self._emit("receipt", r.to_dict())
        if self.live_dir is not None:
            r.append_to(self.live_dir / "receipts.jsonl")
        return r

    def _absorb_places(self) -> None:
        """Copy new places out of the geometry organ into monotone state."""
        pm = getattr(self.inner, "places", None)
        if pm is None:
            return
        for place in pm.places:
            name = place.name()
            vid = f"place{place.index}"
            known = next((v for v in self.state.visited if v.get("id") == vid), None)
            if known is None:
                # record_visit assigns its own id; adopt ours so the wording can
                # be improved later as the place fills up, without the rename
                # reading as a deletion. record_visit DEDUPES consecutive
                # same-named places (two empty rooms both read "a place you
                # could not make out") — stamping visited[-1] then overwrote
                # the PREVIOUS place's id and the monotone set lost an entry
                # (review P1); when nothing was appended, append it ourselves.
                n_before = len(self.state.visited)
                self.state.record_visit(name)
                if len(self.state.visited) > n_before:
                    self.state.visited[-1]["id"] = vid
                else:
                    self.state.visited.append(
                        {"id": vid, "place": name[:120], "frame": None})
                self.state.save()
            elif known["place"] != name:
                self.state.rename_visit(vid, name)
        q = pm.recognition_question()
        if q:
            self.state.pending_question = q   # asked once, never asserted
            self._emit("recognition", {"question": q})

    # ── the one entry point ──

    MAX_FWD_PER_CALL = 6   # 用户 2026-08-07: 一步走的太大——fwd×10 slams
                           # into walls; boldness is capped, not banned

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        args = dict(args or {})
        update = {k: args.pop(k) for k in STATE_KEYS if k in args}

        # forward-run cap: truncate the actions list after the Nth forward
        # (deterministic pace ceiling; the note tells the model to re-look).
        # STOP-carrying batches are EXEMPT, same as the inner toolset's cap
        # (depth_toolset audit P1): "walk up to it and stop" is placing the
        # terminal pose, and trimming it here erased the STOP before
        # _is_stop_intent or the env ever saw it — the episode then ran on
        # with the stop decision silently discarded (review P1).
        capped = False
        if (name == "step" and isinstance(args.get("actions"), list)
                and 0 not in args["actions"]):
            acts, fwd, cut = args["actions"], 0, None
            for i, a in enumerate(acts):
                if a == 1:
                    fwd += 1
                    if fwd > self.MAX_FWD_PER_CALL:
                        cut = i
                        break
            if cut is not None:
                args["actions"] = acts[:cut]
                capped = True

        # depth reflex — 预先避障 (用户 2026-08-08: 可以用深度图预先避障么).
        # A forward-leading batch checks the depth clearance FIRST; if the
        # center is nearly blocked, one turn notch toward the roomier side
        # is prepended. Sensor reading is egocentric-instantaneous (a bumper
        # for the reflex arc); the MODEL still sees RGB only — the organ has
        # its own receptor, the model's 口径 does not change.
        depth_note = None
        if (name == "step" and not self.inner.episode_over
                and isinstance(args.get("actions"), list)
                and args["actions"][:1] == [1]):
            clr = getattr(self.inner, "read_clearance", lambda: None)()
            if clr and clr.get("center", 10.0) < 0.5:
                left = clr.get("left", 10.0)
                right = clr.get("right", 10.0)
                side, act = (("left", 2) if left >= right else ("right", 3))
                args["actions"] = [act] + args["actions"]
                depth_note = (
                    f"[depth reflex: only {clr.get('center', 0):.1f}m of "
                    f"clearance straight ahead — nudged one notch {side} "
                    f"first (left {left:.1f}m / right {right:.1f}m open).]")

        if name == "recall":
            return self._tool_recall(str(args.get("query", "")))
        if (name == "look_around"
                and "look_around" not in self.inner.calls_by_tool):
            return self._tool_look_around_harness()
        if name == "veer":
            return self._tool_veer(str(args.get("direction", "")), update)
        if name == "finish_subgoal":
            return self._tool_finish_subgoal(args, update)

        # forced self-assessment — the organ this whole line of runs asked for.
        # Seven EP0 runs ended the same way: the model narrates "approaching
        # the bar corner" turn after turn, standing 0.6 m from it, and NEVER
        # asks whether it has arrived. near_goal was set 0 times, so the brake
        # and the bell were never even woken. An advisory version of this check
        # already existed and was read and ignored every time. So it BLOCKS:
        # the move does not happen until the question is answered. The harness
        # compels the question; the answer is entirely the model's — which is
        # the same bargain as the guards and the pre-stop gate.
        if (self.reflect_every and name in self.MOVE_TOOLS
                and not self.inner.episode_over
                and not self._reflect_pending
                and self._moves_since_reflect >= self.reflect_every
                and self.state.sub_instructions):
            self._reflect_pending = True
            self._moves_since_reflect = 0
            self._emit("reflect", {"cursor": self.state.cursor,
                                   "steps": self.inner.steps_taken})
            return self._blocked_result(self._reflection_demand())

        # the answer must be the two BOOLEANS, not prose. Asked in words, the
        # 9B replied "very close to the termination point, continuing forward"
        # — the one answer the demand forbids. A boolean has no such escape.
        if (self._reflect_pending and name in self.MOVE_TOOLS
                and not self.inner.episode_over
                and update.get("segment_done") is None
                and update.get("at_goal") is None):
            self._reflect_answers += 1
            if self._reflect_answers <= 2:      # never deadlock: after two
                                                # refusals the move goes through
                return self._blocked_result(
                    "[STILL UNANSWERED — this move was NOT executed.]\n"
                    "Prose like \"very close\" or \"continuing\" is not an "
                    "answer. Put the two booleans ON your next action: "
                    "segment_done=true/false and at_goal=true/false. Both are "
                    "plain fields you can set alongside actions."
                    + self._detector_ledger())

        # pre_action — V0 guards (move tools only; a look is never blocked).
        # A STOP is NEVER guarded. step([0]) is a stop, not a move, and a robot
        # about to stop is by definition not changing its view — which is
        # exactly the stall guard's trigger. Live evidence: the first STOP this
        # line of runs ever produced ("Reached bar counter corner - final goal
        # achieved") was refused by GUARD(stall) and never reached the brake.
        # The harness was vetoing the one action it exists to protect.
        if (name in self.MOVE_TOOLS and not self.inner.episode_over
                and not self._is_stop_intent(name, args)):
            # check_stall clears its hash window on a trip; snapshot the last
            # stuck view first so the probe outcome has something to differ from
            stale_ref = (self.guards.state.move_hashes[-1]
                         if self.guards.state.move_hashes else None)
            notice = (self.guards.check_stall(
                          corrective=self._is_corrective(name, args))
                      or self.guards.check_no_progress(self.inner.steps_taken))
            if notice and notice.startswith("GUARD(stall)"):
                notice += self._unstick_hint(name)
                clr = getattr(self.inner, "read_clearance", lambda: None)()
                if clr:
                    notice += (f" Depth says: left {clr.get('left', 10):.1f}m "
                               f"/ center {clr.get('center', 10):.1f}m / "
                               f"right {clr.get('right', 10):.1f}m open.")
                if self._crisis_start is None and self.frames.frames:
                    self._crisis_start = self.frames.frames[-1].idx
            if notice:
                escalate = self.guards.trip()
                self._emit("guard", {"notice": notice, "escalate": escalate})
                if self.frames.frames:   # span digests need the trip on record
                    self.frames.tag(self.frames.frames[-1].idx,
                                    "guard:" + notice.split(":")[0].lower(),
                                    promote=False)
                info: dict[str, Any] = {"kind": "guard", "guard": True}
                if escalate and self._receipt is not None:
                    notice += ("\nGUARD ESCALATION: this subgoal is preempted — "
                               "report with finish_subgoal(claim='blocked').")
                    info["subgoal_over"] = True
                    info["preempt"] = True
                    self.heartbeat.status = "preempted"
                elif escalate:
                    # solo: no sub-session to preempt — force the recovery move
                    # (solo-harness §6) and rearm so the guard can fire again
                    ckpt = self.state.checkpoint or "your last confirmed point"
                    notice += (f"\nGUARD ESCALATION: go back to {ckpt} and take "
                               "a clearly different branch. recall(<landmark>) "
                               "shows what it looked like.")
                    self.guards.state.consecutive_trips = 0
                self.heartbeat.guard_trips = self.guards.state.trips_total
                self._write_heartbeat()
                if notice.startswith("GUARD(stall)"):
                    # the stuck view, for veer's freed-check (check_stall
                    # cleared its window on the trip)
                    self._stall_ref = stale_ref
                return self._result([_text(notice)], info)

        # pre_stop — verification gate on any stop intent
        if self._is_stop_intent(name, args) and not self.inner.episode_over:
            blocked = self._pre_stop_gate()
            if blocked is not None:
                return blocked

        result = self.inner.execute(name, args)
        self.state.env_steps = self.inner.steps_taken

        # ── the persistent tier goes on the UNCUTTABLE channel ───────────
        # The place memory used to be appended to every tool result, i.e. to
        # the one channel the L2 cut is allowed to eat, and repeated once per
        # message instead of stated once per turn. A run with it on went from 3
        # compactions to 21, peaked at 16.3k tokens, and died on malformed
        # tool-call XML. The state block is the channel that is rendered every
        # turn and never cut — which is what "persistent" was supposed to mean.
        self._absorb_places()

        # motor record — harness-written, egocentric, no coordinates: the
        # commands the body itself issued ("fwd×3 L×1" / "goto#2"). Feeds
        # the just-did line, the frame motion field, spans, and revisit.
        motion = self._motion_text(name, args)
        if motion and not (isinstance(result.info, dict)
                           and result.info.get("error")):
            if name == "goto":
                self._last_goto_wp = _goto_place(args)
            self._motion_pending.append(motion)
            self.state.last_action = motion
            self.state.note_move(motion)
            self.state.save()

        # forward through the inner toolset's own withhold gate if it fired
        # (HabitatToolSet arms _stop_armed on the first budget-rich STOP): the
        # harness gate has already run, so a judged STOP confirms immediately.
        if (self._is_stop_intent(name, args)
                and isinstance(result.info, dict)
                and result.info.get("stop_withheld")):
            result = self.inner.execute(name, args)

        if capped:
            result.content.append(_text(
                f"[pace: forward run capped at {self.MAX_FWD_PER_CALL} steps "
                "per call — look at the outcome view, then continue if the "
                "way is still open.]"))
        if depth_note:
            result.content.append(_text(depth_note))

        # pace advisory (用户 2026-08-07: 开始的时候也要走的大胆一些) — three
        # single-step forwards in a row while the world keeps responding →
        # one mechanical nudge to batch; creeping is for turns and goals
        if name == "step" and args.get("actions") == [1]:
            self._single_fwd_streak += 1
            if self._single_fwd_streak >= 3:
                self._single_fwd_streak = 0
                result.content.append(_text(
                    "[pace: three single steps in a row and the way keeps "
                    "opening — batch 4-6 forward steps when nothing "
                    "important is near, 2-3 at medium distance; save "
                    "single steps for turns and the final approach.]"))
        elif name in self.MOVE_TOOLS:
            self._single_fwd_streak = 0

        # segment-completion reflection (用户 2026-08-07: 模型总是不去想自己
        # 到底有没有完成 subgoal) — every 6 moves without an advance, the
        # harness compels the QUESTION; the answer stays the model's
        if name in self.MOVE_TOOLS:
            self._moves_since_reflect += 1
            if update.get("segment_done") is not None or update.get("at_goal") is not None:
                self._reflect_answers = 0
                if update.get("segment_done") is True and not update.get("advance_subgoal"):
                    # an answered "yes" IS an advance — the model said it; the
                    # harness only keeps it from going unrecorded
                    self.state.apply_update(
                        {"advance_subgoal": True,
                         "progress_note": str(update.get("progress_note")
                                              or "answered: segment complete")},
                        source="reflect")
                if update.get("at_goal") is True:
                    self.state.near_goal = True
                self.state.save()
            self._reflect_pending = False   # the move that follows the demand
                                            # carries the answer; never block twice
            if update.get("advance_subgoal"):
                self._moves_since_reflect = 0
            elif (not self.reflect_every          # the blocking gate owns the
                                                  # cadence when it is on; this
                                                  # advisory version reset the
                                                  # SAME counter and starved it
                  and self._moves_since_reflect >= 6
                  and self.state.sub_instructions
                  and self.state.cursor < len(self.state.sub_instructions)):
                self._moves_since_reflect = 0
                cur = self.state.sub_instructions[self.state.cursor]
                result.content.append(_text(
                    f"[segment check: you are on segment "
                    f"[{self.state.cursor + 1}] '{cur}'. Have you COMPLETED "
                    "it? If yes, set advance_subgoal=true on your next "
                    "action; if not, say what remains in progress_note — "
                    "and update surroundings with what is around you now.]"))

        self._post_observation(name, result)
        self._apply_update(name, update, result)
        self._maybe_auto_arm(update, result)
        self._check_expectation(name, update, result)
        self._check_arrival(name, result)
        self._append_state_render(result)
        self._track_budget(result)
        return result

    # ── harness-own tools ──

    def _tool_recall(self, query: str) -> Any:
        from toolset import ToolResult, png_part, text_part  # local import: keep
        # eharness importable without the toolset module on path (unit tests)
        q = query.strip().lower()
        m = q.replace("segment", "seg").replace("seg ", "seg")
        if m.startswith("seg") and m[3:].strip().isdigit():
            n = int(m[3:].strip())
            for rec in self.frames.segments:
                if rec["seg"] == n and self.live_dir is not None:
                    p = self.live_dir / rec["png"]
                    content = [text_part(
                        f"segment {n}: {rec['label']} — route: "
                        f"{rec['route'] or '(no note)'} · moved: {rec['motion']}")]
                    if p.exists():
                        content.append(png_part(p.read_bytes()))
                    self._emit("recall", {"query": query, "segment": n})
                    return ToolResult(content=content,
                                      info={"kind": "recall", "segment": n})
            return ToolResult(content=[text_part(f"no segment {n} archived yet")],
                              info={"kind": "recall", "hits": 0})
        hits = self.frames.recall(query)
        content: list[dict[str, Any]] = []
        for fr in hits:
            png = self.frames.png_bytes(fr.idx)
            if png:
                content.append(text_part(
                    f"frame#{fr.idx} (step {fr.step}): {' · '.join(fr.events)}"))
                content.append(png_part(png))
        # landmark/negative-fact text hits ride along even without pixels
        q = query.lower()
        facts = [f["fact"] for f in self.state.negative_facts if q in f["fact"].lower()]
        lms = [n for n in self.state.landmarks if q in n.lower()]
        if facts or lms:
            content.append(text_part(json.dumps(
                {"landmarks": lms, "ruled_out": facts}, ensure_ascii=False)))
        if not content:
            content.append(text_part(f"no stored memory matches '{query}'"))
        self._emit("recall", {"query": query, "hits": [f.idx for f in hits]})
        return ToolResult(content=content, info={"kind": "recall", "hits": len(hits)})

    def _tool_finish_subgoal(self, args: dict[str, Any],
                             update: dict[str, Any]) -> Any:
        from toolset import ToolResult, text_part
        if self._receipt is None:
            return ToolResult(content=[text_part(
                "finish_subgoal is only available inside a delegated subgoal")],
                info={"kind": "finish_subgoal", "error": "no subgoal"})
        self._pending_finish = args
        if update:
            self._apply_update("finish_subgoal", update, None)
        return ToolResult(
            content=[text_part(json.dumps({"subgoal_closed": True,
                                           "claim": args.get("claim")}))],
            info={"kind": "finish_subgoal", "subgoal_over": True})

    def _tool_look_around_harness(self) -> Any:
        """look_around for surfaces that lack it (bare): harness-driven — the
        same rotation the full toolset performs, built from the inner step
        tool. Views are ingested as transient (auto) frames."""
        from toolset import ToolResult, text_part
        if self.inner.episode_over:
            return ToolResult(content=[text_part("episode already over")],
                              info={"kind": "look_around", "error": "over"})
        content: list[Any] = []
        # §5.2: collect the four quarter views, then hand the model ONE
        # labelled contact sheet — four loose images plus "type twelve 3s"
        # both burned context and pushed the harness's own angle bookkeeping
        # onto the model.
        views: list[dict[str, Any]] = []
        for heading_ccw in (0, 270, 180, 90):     # ahead → right → behind → left
            try:
                obs = self.inner.execute("observe", {})
            except Exception:  # noqa: BLE001
                break
            png = next((_png_from_part(q) for q in (obs.content or [])
                        if _png_from_part(q)), None)
            if png is not None:
                views.append({"heading_deg": heading_ccw,
                              "rgb_base64": base64.b64encode(png).decode()})
                self._ingest_png(png, "look_around", content,
                                 auto=True, move=False)
            r = self.inner.execute("step", {"actions": [3] * 6})
            if isinstance(r.info, dict) and r.info.get("episode_over"):
                content.append(_text("step budget ran out mid-scan"))
                break
        sheet = b""
        try:
            from eharness import depthmap as _dm
            sheet = _dm.compose_panorama(views)
        except Exception:  # noqa: BLE001 — composition must not kill the scan
            sheet = b""
        if sheet:
            content.append(_text(
                "[panorama — AHEAD · LEFT · BEHIND · RIGHT in one sheet. "
                "Pick the panel matching your current sub-instruction, TURN "
                "to face it (LEFT panel → step([2]*6); RIGHT → step([3]*6); "
                "BEHIND → step([3]*12)), then go forward.]"))
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64,"
                       + base64.b64encode(sheet).decode()}})
        else:
            content.append(_text("[panorama unavailable — the scan views "
                                 "were recorded as recallable frames]"))
        self._motion_pending.append("look-around")
        self.state.last_action = "look-around"
        self.state.save()
        info = {"kind": "look_around",
                "steps_taken_total": self.inner.steps_taken,
                "episode_over": self.inner.episode_over}
        content.append(_text(json.dumps(info)))
        self.heartbeat.tool_calls += 1
        self._write_heartbeat()
        return ToolResult(content=content, info=info)

    # ── pre_stop gate ──

    def _is_stop_intent(self, name: str, args: dict[str, Any]) -> bool:
        if name == "stop":
            return True
        if name == "step":
            actions = args.get("actions")
            return isinstance(actions, list) and 0 in actions
        return False

    def _pre_stop_gate(self) -> Any | None:
        """None = let the stop through; otherwise the blocking ToolResult.
        First stop attempt is judged (V1); a veto forces a look and returns
        feedback; the second attempt always passes (the judge may compel work,
        never the conclusion)."""
        from toolset import ToolResult, text_part
        self._stop_attempts += 1
        if self.judge_model is None or self._stop_attempts > 1:
            return None
        evidence = self.frames.recent_pngs(2)
        if not evidence:
            return None
        tail = self.state.goal_text
        # route first: proximity to the final target means nothing if the
        # instructed segments never happened (v2.9 EP0: stopped at "a bar
        # corner" 4.4m off, done=[] — it had never walked between the bar
        # and chairs)
        miss, m_why = self._verify_milestones()
        if miss is not None:
            sub_txt = self.state.route_segments[miss]
            self._emit("verdict", {"gate": "pre_stop", "supported": False,
                                   "reason": f"segment [{miss + 1}] unevidenced: {m_why}",
                                   "advice": f"complete it first: {sub_txt}"})
            return ToolResult(content=[text_part(
                "STOP withheld — the verifier disagrees.\n"
                f"Why: your walk shows no evidence of segment [{miss + 1}] "
                f"'{sub_txt}' — {m_why}\n"
                f"Suggestion: complete that segment first ({sub_txt}), THEN "
                "continue to the final goal and stop there.\n"
                "You had your own reason to stop; if after looking you still "
                "believe the route is done, request STOP again and it WILL "
                "execute — the call is yours, not the verifier's.")],
                info={"kind": "pre_stop", "stop_vetoed": True,
                      "milestone": miss, "reason": m_why,
                      **self._tag_stop_attempt()})
        # clean context means CLEAN: the judge sees the instruction tail and
        # the pixels — never the state block. Ep4/ep7 of the first 10-ep run
        # showed a state-fed judge citing the agent's own "task completed"
        # claim as evidence (verdict: "the task status indicates...").
        supported, reason, advice = run_judge_stop(
            f"The robot has walked the instructed route and is now within "
            f"3 meters (VERY close) of: '{tail}'",
            self.state.route_context() + self._detector_ledger(), evidence,
            model_name=self.judge_model, model_kwargs=self.judge_kwargs)
        self._verify_calls += 1
        self._emit("verdict", {"gate": "pre_stop", "supported": supported,
                               "reason": reason, "advice": advice})
        # second check, the human way (2026-08-05): does the robot's own
        # account of the walk match the instructed route? Text-only, runs
        # only when there is a story to check. The narrative is the OBJECT
        # under test here, not evidence — consistent with decontamination.
        if supported is not False:
            report = "\n".join(
                [f"[{d['idx'] + 1}] {d['text']} — via: {d.get('evidence') or '(no note)'}"
                 for d in self.state.done]
                + self.frames.span_digest(max_spans=4)
                + ([f"now: {self.state.current_place}"]
                   if self.state.current_place else []))
            if report.strip():
                r_ok, r_why = run_judge_route(
                    self.state.instruction, report,
                    model_name=self.judge_model, model_kwargs=self.judge_kwargs)
                self._verify_calls += 1
                self._emit("verdict", {"gate": "pre_stop_route",
                                       "supported": r_ok, "reason": r_why})
                if r_ok is False:
                    supported, reason = False, f"route mismatch: {r_why}"
                    advice = ("compare your journey line against the "
                              "instruction, then move to where its FINAL "
                              "clause is satisfied")
        if supported is not False:   # pass AND judge-failure both let it through
            return None
        # a veto must carry its reason AND a concrete suggestion (用户
        # 2026-08-06: 你否决有你的理由，那就给个建议，否则它怎么会知道) —
        # and it must say out loud that the final call stays with the robot
        return ToolResult(content=[text_part(
            "STOP withheld — the verifier disagrees.\n"
            f"Why: {reason}\n"
            f"Suggestion: {advice or 'study the newest view and re-read the final sub-instruction'}\n"
            "You had your own reason to stop; if after looking you still "
            "believe you are at the goal, request STOP again and it WILL "
            "execute — the call is yours, not the verifier's.")],
            info={"kind": "pre_stop", "stop_vetoed": True, "reason": reason,
                  "advice": advice, **self._tag_stop_attempt()})

    def _tag_stop_attempt(self) -> dict:
        """A vetoed STOP becomes a frame event, so ActionSpans (§12.4) list
        it explicitly instead of it vanishing between two snapshots."""
        if self.frames.frames:
            self.frames.tag(self.frames.frames[-1].idx, "stop:vetoed",
                            promote=False)
        return {}

    def _blocked_result(self, text: str):
        """A refused action: the model gets the demand, the world does not move.
        Same shape the guards use to block, so every downstream layer already
        knows how to handle it."""
        return self._result([_text(text)], {"kind": "reflect", "blocked": True})

    def _reflection_demand(self) -> str:
        """Two yes/no questions the model cannot answer with 'continuing'."""
        subs = self.state.sub_instructions
        i = min(self.state.cursor, len(subs) - 1)
        cur = subs[i] if subs else self.state.instruction
        term = self.state.terminate or "the instruction's endpoint"
        return (
            "[STOP AND ANSWER — this move was NOT executed.]\n"
            "You have moved several times without deciding anything. Before "
            "you move again, answer BOTH questions explicitly. \"Still "
            "approaching\" is not an answer to either.\n"
            f"  (1) Segment [{i + 1}] is: \"{cur}\". Is it COMPLETE — yes or "
            "no? If yes, set advance_subgoal=true on your next action.\n"
            f"  (2) The stop condition is: \"{term}\". Does it hold RIGHT NOW "
            "— yes or no? If yes, set near_goal=true and call step([0]); the "
            "verifier will check you, and a withheld first STOP can be "
            "re-requested and always goes through.\n"
            "If both are no, say in progress_note what specifically is still "
            "missing — not where you are heading."
            + self._detector_ledger()
        )

    def _detector_ledger(self) -> str:
        """What the DETECTOR has recorded, as bookkeeping for the judge.

        The milestone judge keeps failing the same way: it re-decides from
        pixels whether a landmark was passed, gets it wrong, and vetoes a
        correct STOP (three EP0 runs, most recently a near-perfect episode
        rejected for "has not passed the pool" when the register shows the pool
        at 0.88 m forty steps earlier). Perception is a coin-flip; a ledger is
        not. So the harness hands over what it actually SAW and when, and the
        judge rules on top of that instead of guessing again.

        Advisory, never coercive: absence of a ledger line proves nothing (the
        detector may be off or the phrase unmatched), so silence is stated as
        silence and the judge is told not to read it as a negative."""
        reg = getattr(getattr(self, "inner", None), "register", None)
        phrases = list(getattr(getattr(self, "inner", None), "phrases", []) or [])
        if reg is None or not phrases:
            return ""
        step = getattr(self.inner, "steps_taken", 0)
        lines = [reg.evidence_line(p, step) for p in phrases]
        seen_any = any(reg.ever_seen(p) for p in phrases)
        if not seen_any:
            return ("\n\nDETECTOR LEDGER: the object detector has recorded "
                    "nothing on this route — it may be off or the phrases may "
                    "not match. Treat this as NO INFORMATION, not as evidence "
                    "against the walk.")
        return ("\n\nDETECTOR LEDGER (what the robot's detector actually "
                "recorded, frame by frame — this is bookkeeping, not opinion; "
                "a landmark seen close and then not seen for many steps has "
                "been PASSED):\n- " + "\n- ".join(lines))

    def _verify_milestones(self) -> tuple[int | None, str]:
        """(first un-evidenced intermediate sub index, reason) — (None, '')
        when the route so far holds up. Every intermediate sub-instruction
        not yet done is checked against the walk's own frame evidence (用户
        2026-08-06: 分成了几段，每段至少得检查一下). A sub the judge finds
        evidenced is marked done ON THE SPOT — verification REPAIRS the
        bookkeeping instead of trusting it (v2.8: cursor lagged reality) or
        ignoring it (v2.9: stopped with done=[], sub-2 skipped). Evidenced
        is permanent; a miss re-checks only after 4 new frames."""
        subs = self.state.route_segments
        if not subs:
            return None, ""
        evidence = self.frames.evidence_pngs(8)
        if not evidence:
            return None, ""
        for i in range(len(subs)):
            if any(d.get("idx") == i for d in self.state.done):
                continue
            cached = self._milestone_miss.get(i)
            if cached is not None and len(self.frames.frames) - cached[0] < 4:
                return i, cached[1]
            # think-free: evidence questions measured verdict-equivalent with
            # or without the 45-70s chain (用户: 有些 verification 很耗时)
            ok, why = run_judge_milestone(
                subs[i], evidence, model_name=self.judge_model,
                model_kwargs={**self.judge_kwargs, "judge_think": False},
                route_context=self.state.route_context() + self._detector_ledger())
            self._verify_calls += 1
            self._emit("verdict", {"gate": "milestone", "sub": i,
                                   "supported": ok, "reason": why})
            if ok is True:
                # §14.11: a judge-confirmed segment is a VerifiedMilestone —
                # the flag is what separates it from a model-claimed advance
                # in the mission card and the VERIFIED group of the render.
                self.state.done.append({"idx": i, "text": subs[i],
                                        "evidence": f"judge-evidence: {why[:120]}",
                                        "frame": None, "verified": True})
                self.state.cursor = max(self.state.cursor, i + 1)
                self.state.save()
                self.state.add_journey(
                    f"segment [{i + 1}] got confirmed done from my own footage")
                self._emit("state_write",
                           {"update": {"advance_subgoal": "judge-evidence"},
                            "accepted": [f"sub {i + 1} evidenced by judge"]})
                self._milestone_miss.pop(i, None)
                continue
            if ok is False:
                self._milestone_miss[i] = (len(self.frames.frames), why)
                return i, why
            # ok is None — judge failure is never a verdict; don't block
        return None, ""

    def verify(self, claim: str) -> tuple[bool | None, str]:
        """On-demand V1 (the planner's verify tool)."""
        if self.judge_model is None:
            return None, "verification disabled"
        self._verify_calls += 1
        verdict = run_judge(claim, f"Instruction: {self.state.instruction}",
                            self.frames.recent_pngs(2),
                            model_name=self.judge_model,
                            model_kwargs=self.judge_kwargs)
        self._emit("verdict", {"gate": "on_demand", "claim": claim,
                               "supported": verdict[0], "reason": verdict[1]})
        return verdict

    def stop_action(self) -> tuple[str, dict[str, Any]]:
        """The mode-correct stop call (planner's finish path)."""
        return (("stop", {}) if "stop" in self.inner.calls_by_tool
                else ("step", {"actions": [0]}))

    # ── post_observation ──

    def _ingest_png(self, png: bytes, tool: str, out: list[Any], *,
                    auto: bool = False, move: bool = False) -> None:
        """One recorded frame's full pipeline: frame log + marker + guard
        bookkeeping + novelty + revisit advisory. ``auto`` frames are the
        transient after-move views — shown once, dropped from history by the
        model layer next turn (看完扔掉)."""
        fr = self.frames.record(
            png, step=self.inner.steps_taken, tool=tool,
            motion=" · ".join(self._motion_pending))
        self._motion_pending = []
        # §3.4: with refs on, the STORED transcript holds a handle instead of
        # the base64 — the payload compiler materialises the ones it keeps.
        # The image part just appended to `out` is swapped in place.
        if self.store_refs and fr.png_path:
            for k in range(len(out) - 1, -1, -1):
                if _png_from_part(out[k]) is not None:
                    out[k] = {"type": "image_ref", "ref": fr.png_path}
                    break
        out.append(_text(f"[frame#{fr.idx} auto]" if auto else f"[frame#{fr.idx}]"))
        if move:
            prev = (self.guards.state.move_hashes[-1]
                    if self.guards.state.move_hashes else None)
            self.guards.note_move_view(fr.hash)
            if (self._crisis_start is not None and prev is not None):
                from eharness.frames import hamming as _ham
                if _ham(fr.hash, prev) > self.guards.cfg.stall_max_dist:
                    self._resolve_crisis(fr, out)
        self.frames.maybe_promote_novel(fr)
        back = self.frames.check_revisit(fr)
        if back is not None:
            self.frames.tag(fr.idx, f"revisit:frame#{back.idx}", promote=False)
            trace = self.frames.motion_trace(back.idx, fr.idx)
            out.append(_text(
                f"[note: this view closely matches frame#{back.idx} from "
                f"earlier (step {back.step}) — you may be back at a place "
                f"you already visited. recall({back.idx}) to compare."
                + (f" Your moves since then: {trace}." if trace else "")
                + "]"))
            self._emit("revisit", {"frame": fr.idx, "matches": back.idx})
            self.state.add_journey(
                "found myself back at a spot I had already been")

    def _post_observation(self, name: str, result: Any) -> None:
        new_content: list[Any] = []
        recorded = False
        prev_text = ""
        for part in (result.content or []):
            new_content.append(part)
            png = _png_from_part(part)
            if png is None:
                prev_text = str(part.get("text", "")) if isinstance(part, dict) \
                    else ""
                continue
            # The accumulated-map render (IMAGE 2) is NOT a camera frame: it
            # must not enter the FrameLog (where it would vote into the stall
            # guard, get promoted as a "novel view", be eligible for the
            # uniform HISTORY sample as if it were an RGB — or be handed to a
            # JUDGE as gate evidence), and it must not get a [frame#N] marker
            # (which made the context assembler classify it as a frame and
            # never as the map). Its label is its identity; match the STABLE
            # HEAD only: §20.4 reworded the legend after "IMAGE 2 — " and the
            # old full-prefix match silently let map renders into the frame
            # log — the 70-ep run then showed judges ruling on map pixels
            # ("insufficient — only a schematic map-memory overlay", EP6/60)
            # or, worse, rubber-stamping a stop from the map's own SAM
            # markers (EP17). Same failure class as the 5173 tail-match.
            if prev_text.startswith("IMAGE 2"):
                # the map is not a camera frame — but with refs on it still
                # needs a versioned file so the compiler can re-materialise
                # whichever single latest map it keeps (§3.3 latest-map slot)
                if self.store_refs and self.live_dir is not None:
                    try:
                        (self.live_dir / "maps").mkdir(parents=True,
                                                       exist_ok=True)
                        self._map_n += 1
                        ref = f"maps/map_{self._map_n:05d}.png"
                        (self.live_dir / ref).write_bytes(png)
                        new_content[-1] = {"type": "image_ref", "ref": ref,
                                           "map": True}
                    except OSError:
                        pass
                prev_text = ""
                continue
            prev_text = ""
            self._ingest_png(png, name, new_content,
                             move=(name in self.MOVE_TOOLS))
            recorded = True
        # auto-view: a move that came back imageless (bare/wp separate mode)
        # immediately shows its outcome view — the user's "把每个动作之后的
        # 图片 temporarily 给他看一下" (2026-08-05). Feeds the stall guard on
        # EVERY move (before, an unlooking model left the guard blind), and
        # the model sees at once that the world did not respond.
        if (not recorded and self.auto_view and name in self.MOVE_TOOLS
                and not self.inner.episode_over
                and isinstance(result.info, dict)
                and not result.info.get("error")):
            try:
                obs = self.inner.execute("observe", {})
                apngs = [q for q in (obs.content or []) if _png_from_part(q)]
            except Exception:  # noqa: BLE001 — auto-view must never kill a move
                apngs = []
            if apngs:
                new_content.append(_text(
                    "[auto view after your moves — this is your CURRENT "
                    "view:]"))
                new_content.append(apngs[0])
                self._ingest_png(_png_from_part(apngs[0]), "auto_view",
                                 new_content, auto=True, move=True)
                recorded = True
        if recorded:
            result.content[:] = new_content
        pngs = [p for p in (result.content or []) if _png_from_part(p)]
        # a look after a move (classic alternation) is the move's outcome view
        if name == "observe" and pngs and self.frames.frames:
            self.guards.note_move_view(self.frames.frames[-1].hash)
        self.heartbeat.tool_calls += 1
        self.heartbeat.steps_used = self.inner.steps_taken - self._steps_at_begin
        self._write_heartbeat()

    def _apply_update(self, name: str, update: dict[str, Any],
                      result: Any) -> None:
        if not update:
            return
        frame = self.frames.last_frame().idx if self.frames.frames else None
        accepted = self.state.apply_update(
            update, source=f"sub:{name}", frame=frame,
            step=getattr(self.inner, "steps_taken", None))
        if accepted:
            self._emit("state_write", {"update": update, "accepted": accepted})
            self.heartbeat.last_note = "; ".join(accepted)
            if frame is not None:
                if update.get("landmark"):
                    self.frames.tag(frame, f"landmark:{update['landmark'][:40]}")
                if update.get("dead_end"):
                    self.frames.tag(frame, f"dead-end:{update['dead_end'][:40]}")
                if update.get("advance_subgoal") is True:
                    self.frames.tag(frame, f"subgoal:{self.state.cursor} done")
                    self.guards.note_advance(self.inner.steps_taken)
                    # completed segment → filmstrip into episode long-term
                    # memory; the live window is free to drop these frames
                    done = self.state.done[-1] if self.state.done else {}
                    name_png = self.frames.archive_segment(
                        self._seg_start, frame,
                        label=done.get("text", f"sub-instruction {self.state.cursor}"),
                        route=done.get("evidence", ""))
                    self._seg_start = frame + 1
                    if name_png:
                        self._emit("segment", {"png": name_png,
                                               "label": done.get("text", "")[:120]})
                        if result is not None:
                            result.content.append(_text(
                                f"[segment archived: '{done.get('text', '')[:60]}' "
                                f"→ recall('segment {len(self.frames.segments)}') "
                                "replays that stretch]"))
            if result is not None:
                result.content.append(_text("state updated: " + "; ".join(accepted)))
            if self._receipt is not None:
                for k, v in update.items():
                    self._receipt.proposes.setdefault(k, []).append(v)

    _ARM_WORDS = ("stop", "wait", "final goal", "destination", "arrive")
    _ARM_STOPSET = frozenset(
        "the a an to of and or you your i my when get at in on is it that "
        "this will be where which".split())

    def _maybe_auto_arm(self, update: dict[str, Any], result: Any) -> None:
        """The near-stop flag lights from the model's OWN WORDS (v2.8 EP0:
        the model wrote 'where I need to stop and wait', the judge confirmed
        the corner — and the flag stayed dark because a brand-new schema
        field was its only trigger). If an expect references stopping or
        shares ≥2 content words with the final sub-instruction, that IS the
        belief the user described — the harness arms the flag itself. The
        model can still disarm with near_goal=false."""
        if self.state.near_goal or not self.state.sub_instructions:
            return
        exp = update.get("expect")
        if not (isinstance(exp, str) and exp.strip()):
            return
        low = exp.lower()
        hit = any(w in low for w in self._ARM_WORDS)
        if not hit:
            final = self.state.goal_text.lower()
            words = {w.strip(".,!?'\"") for w in final.split()} - self._ARM_STOPSET
            hit = sum(1 for w in words if len(w) > 2 and w in low) >= 2
        if not hit:
            return
        self.state.near_goal = True
        self.state.save()
        self._emit("near_stop", {"source": "expect", "expect": exp.strip()[:120]})
        if result is not None:
            result.content.append(_text(
                "[⚑ NEAR-STOP armed: your own expectation references the "
                "final goal — from now on every move is arrival-checked. "
                "STOP the moment you and the verifier agree you are there.]"))

    def _check_expectation(self, name: str, update: dict[str, Any],
                           result: Any) -> None:
        """Move-outcome reconciliation (用户 2026-08-05: 走过来的流程和结果，
        每一步有验证吗). When the model stated an expectation on a MOVE and
        the move returned an outcome view, a clean-context judge immediately
        reconciles view vs expectation and the verdict rides back in the same
        result. The harness guarantees the check HAPPENS — that is what
        harnessing means here; the judgment itself is still model-made, and
        the drift signal (ep0: unstuck, then veered into the chairs instead
        of the bar-chair gap) surfaces the turn it occurs, not at STOP."""
        if (self.judge_model is None or not self.verify_moves
                or name not in self.MOVE_TOOLS):
            return
        exp = update.get("expect")
        if not (isinstance(exp, str) and exp.strip()):
            return
        pngs = self.frames.recent_pngs(1)
        if not pngs:
            return
        # move_check is the HIGH-FREQUENCY gate (every move with an expect) —
        # it runs think-free: measured 2026-08-06, thinking added 30-88s per
        # call for verdict-identical output, and this check fires dozens of
        # times per episode (用户: 中间 verification 很慢很慢). The terminal
        # gates (pre-stop, arrival, route) keep the deliberate slow judge.
        supported, reason = run_judge(
            f"The robot's stated expectation is now met: '{exp.strip()[:160]}'",
            f"The robot just executed: {self.state.last_action or name}. "
            "Judge ONLY from the image whether the expectation was met.",
            pngs, model_name=self.judge_model,
            model_kwargs={**self.judge_kwargs, "judge_think": False})
        self._verify_calls += 1
        self._emit("verdict", {"gate": "move_check", "supported": supported,
                               "reason": reason, "expect": exp.strip()[:120]})
        if supported is False:
            result.content.append(_text(
                f"[expectation check: NOT met — {reason} Reconcile before "
                "your next move: are you still on the instructed route?]"))
        elif supported is True:
            result.content.append(_text("[expectation check: met]"))
            self.state.add_journey(
                f"confirmed I was on track ({reason[:70].rstrip('.')})")

    def _check_arrival(self, name: str, result: Any) -> None:
        """The termination BELL — symmetric to the pre-stop BRAKE (user
        2026-08-06: the harness's job is the loop's termination condition,
        and we had only made stopping harder, never visible). On the FINAL
        sub-instruction, each move outcome view is put to the SAME strict
        judge the pre-stop gate uses; when it rules very-close, the model is
        told to consider STOP. Same criterion both directions — the bell
        cannot re-introduce the early stops the brake blocks."""
        # armed = the model's own near-goal belief OR every route segment
        # done (v3.1: the terminate clause is a CONDITION, not a leg — the
        # bell verifies it once the legs are walked). Decoupled from cursor
        # alone since v2.7 (v2.6 EP0: oracle 1.0 with a silent bell).
        if (self.judge_model is None or not self.verify_moves
                or name not in self.MOVE_TOOLS
                or not (self.state.sub_instructions or self.state.terminate)
                or not self.state.near_stop_armed
                or self.inner.episode_over):
            return
        fr = self.frames.last_frame()
        if fr is None or fr.idx - self._arrival_last_check < 3:
            return
        pngs = self.frames.recent_pngs(1)
        if not pngs:
            return
        self._arrival_last_check = fr.idx
        tail = self.state.goal_text
        # the bell fires per move once armed (cooldown 3) — think-free keeps
        # it at 0.7s; the BRAKE (judge_stop) stays the one deliberate judge
        supported, reason = run_judge(
            f"The robot has walked the instructed route and is now within "
            f"3 meters (VERY close) of: '{tail}'",
            self.state.route_context() + self._detector_ledger(), pngs,
            model_name=self.judge_model,
            model_kwargs={**self.judge_kwargs, "judge_think": False})
        self._verify_calls += 1
        self._emit("verdict", {"gate": "arrival", "supported": supported,
                               "reason": reason})
        if supported is True:
            # the bell rings only over a VERIFIED route — looking close to
            # the final target while an instructed segment never happened is
            # exactly the v2.9 wrong-corner stop
            miss, m_why = self._verify_milestones()
            if miss is not None:
                result.content.append(_text(
                    f"[arrival check: you LOOK very close to the final goal, "
                    f"BUT your walk shows no evidence of segment "
                    f"[{miss + 1}] '{self.state.route_segments[miss]}' "
                    f"({m_why}). Complete that segment first — stopping now "
                    "would fail the route.]"))
                return
            result.content.append(_text(
                f"[arrival check: the verifier judges you are VERY close to "
                f"the final goal — {reason} If you agree, STOP now; walking "
                "past the goal fails the episode just like stopping short.]"))

    def _append_state_render(self, result: Any) -> None:
        """The universal injection channel (works for closed loops too).

        §14.2: closed executors (Claude SDK) keep every old tool result, so
        the model sees many stale STATE copies at once. Bumping the version
        HERE — an explicit act, render() stays pure — gives each injected
        copy a monotone number and a "supersedes" line, so only the newest
        reads as law. The mini/solo path passes inject_state=False and its
        render keeps the classic bare [STATE] header.

        Two review fixes ride here: the bumped version is SAVED (a bridge
        restart used to reload version−1 and reissue the same "supersedes"
        number for a different state, review P2), and an UNCHANGED render is
        replaced by a one-line stub — on a full-history executor the full
        ~2.6 KB copy per turn stacked ~19 k tokens of duplicate state into a
        30-turn session for zero information (review P2)."""
        if not self.inject_state:
            return
        self.state.state_version += 1
        render = self.state.render()
        body = render.split("\n", 1)[-1]
        if body == getattr(self, "_last_injected_body", None):
            self.state.state_version -= 1     # no new copy was issued
            result.content.append(_text(
                f"[STATE v{self.state.state_version} unchanged — see the "
                "newest full STATE above]"))
            return
        self._last_injected_body = body
        self.state.save()                     # disk must hold the shown version
        result.content.append(_text(render))

    def _track_budget(self, result: Any) -> None:
        if self._receipt is None or self.state.delegation is None:
            return
        used = self.inner.steps_taken - self._steps_at_begin
        self.state.delegation["used"] = used
        budget = self.state.delegation["budget"]
        if used >= budget and not self.inner.episode_over:
            result.content.append(_text(
                f"SUBGOAL BUDGET EXHAUSTED ({used}/{budget} steps) — report "
                "with finish_subgoal now."))
            if isinstance(result.info, dict):
                result.info["subgoal_over"] = True
                result.info["subgoal_budget_exhausted"] = True

    @staticmethod
    def _motion_nl(trace: str) -> str:
        """Motor codes → a natural phrase (状态块不写动作代码)."""
        t = trace or ""
        if "veer-L" in t:
            return "veering a little left"
        if "veer-R" in t:
            return "veering a little right"
        if "look-around" in t:
            return "looking around and picking a new direction"
        if "L×" in t:
            return "turning left and moving on"
        if "R×" in t:
            return "turning right and moving on"
        return "changing direction"

    def _resolve_crisis(self, fr: Any, out: list[Any]) -> None:
        """The stuck→freed span collapses to ONE memory: the problem frame
        and the freeing move stay; every step in between is dropped from the
        model's history (用户 2026-08-05: 中间的解法步骤大概率可以删掉，
        把「遇到问题→怎么解开」这条记忆记下来就行)."""
        a = self._crisis_start
        self._crisis_start = None
        self._stall_ref = None
        for f in self.frames.frames[a + 1:fr.idx]:
            f.crisis = True
        freed_by = self.frames.motion_trace(max(0, fr.idx - 2), fr.idx)             or (self.frames.frames[fr.idx].motion if fr.idx < len(self.frames.frames) else "")
        nl = self._motion_nl(freed_by)
        lesson = ("I got stuck for a moment (my moves stopped changing the "
                  f"view) and got free by {nl}")
        self.state.add_lesson(lesson)
        # the story entry is harness-written: the model reads "difficulty
        # met and solved", it never has to narrate the solution itself
        self.state.add_journey(f"hit an obstacle and got free by {nl}")
        self.frames.tag(a, "crisis:start", promote=False)
        self.frames.tag(fr.idx, f"crisis:freed-by {freed_by}"[:60], promote=False)
        out.append(_text(f"[crisis resolved — {lesson}. The in-between "
                         "steps will be dropped from history.]"))
        self._emit("crisis", {"start": a, "end": fr.idx, "lesson": lesson})

    def _tool_veer(self, direction: str, update: dict[str, Any]) -> Any:
        """The unstick split of labor (用户 2026-08-06 final form): the MODEL
        picks the side — the harness cannot know which way is open — and the
        harness does the correcting: 15° that way + forward, one more notch
        to 30° if the view still hasn't changed. The model never composes
        the maneuver; a blind harness try-first ladder wasted a round on the
        wrong side half the time."""
        from toolset import ToolResult
        if direction not in ("left", "right"):
            return self._result(
                [_text("veer needs direction='left' or 'right'")],
                {"kind": "veer", "error": "bad_direction"})
        if self.inner.episode_over:
            return self._result([_text("episode already over")],
                                {"kind": "veer", "error": "over"})
        from eharness.frames import ahash as _ahash
        from eharness.frames import hamming as _ham
        turn = 2 if direction == "left" else 3
        side = "L" if direction == "left" else "R"
        ref = self._stall_ref
        if ref is None and self.guards.state.move_hashes:
            ref = self.guards.state.move_hashes[-1]
        out: list[Any] = []
        notches = 0
        freed = False
        png = None
        for _attempt in (1, 2):
            try:
                self.inner.execute("step", {"actions": [turn, 1]})
            except Exception:  # noqa: BLE001 — the cure must never kill the turn
                break
            notches += 1
            if self.inner.episode_over:
                break
            try:
                obs = self.inner.execute("observe", {})
                png = next((_png_from_part(p) for p in (obs.content or [])
                            if _png_from_part(p)), None)
            except Exception:  # noqa: BLE001
                png = None
            if png is None:
                break
            freed = (ref is None
                     or _ham(_ahash(png), ref) > self.guards.cfg.stall_max_dist)
            if freed:
                break
        motion = f"veer-{side}×{notches} fwd×{notches}"
        self._motion_pending.append(motion)
        self.state.last_action = motion
        self.state.note_move(motion)
        self.state.env_steps = self.inner.steps_taken
        self.state.save()
        if png is not None:
            msg = ("the view changed — you are moving again" if freed else
                   "view STILL unchanged; this side seems closed too — "
                   "try the other side or go back")
            out.append(_text(f"[veer {direction}: turned {15 * notches}° and "
                             f"stepped forward — {msg}. Outcome view:]"))
            # the image PART must be appended — _ingest_png records/marks and
            # (with refs on) swaps it, but never adds it (audit P1: the veer
            # outcome view was announced and then not shown)
            out.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64,"
                       + base64.b64encode(png).decode()}})
            self._ingest_png(png, "veer", out, auto=True, move=True)
            fr = self.frames.frames[-1] if self.frames.frames else None
            if freed and fr is not None and self._crisis_start is not None:
                self._resolve_crisis(fr, out)
        self._stall_ref = None
        if self.inner.episode_over:
            out.append(_text("[step budget ran out during the veer]"))
        self._apply_update("veer", update, None)
        info = {"kind": "veer", "direction": direction, "notches": notches,
                "episode_over": self.inner.episode_over}
        self._emit("veer", {"direction": direction, "notches": notches,
                            "episode_over": self.inner.episode_over})
        self.heartbeat.tool_calls += 1
        self.heartbeat.steps_used = self.inner.steps_taken - self._steps_at_begin
        self._write_heartbeat()
        return ToolResult(content=out, info=info)

    def _is_corrective(self, name: str, args: dict[str, Any]) -> bool:
        """Does this action already change something? Turn-first step batches
        and different-waypoint gotos are the cure, not the disease."""
        if name == "step":
            acts = args.get("actions")
            return bool(isinstance(acts, list) and acts
                        and acts[0] in (2, 3))
        if name == "goto":
            return _goto_place(args) != self._last_goto_wp
        if name == "face":
            return True     # turning to face elsewhere IS the cure — the
                            # stall guard must never block it (audit P1)
        return False

    @staticmethod
    def _unstick_hint(name: str) -> str:
        if name == "step":
            return ("Look at the view, pick the side that looks open, and "
                    "call veer('left') or veer('right') — the harness makes "
                    "the small turn + forward step for you. Never spin 90° "
                    "blind. (A turn-first step batch also works and will not "
                    "be blocked.)")
        return ("Pick a DIFFERENT numbered place than your last goto, or "
                "face() another direction — every action returns fresh options.")

    @staticmethod
    def _motion_text(name: str, args: dict[str, Any]) -> str:
        if name == "step":
            acts = args.get("actions")
            if not isinstance(acts, list):
                return ""
            c1, c2, c3 = acts.count(1), acts.count(2), acts.count(3)
            s = " ".join(f"{b}×{c}" for b, c in
                         (("fwd", c1), ("L", c2), ("R", c3)) if c)
            return s + (" STOP" if 0 in acts else "") if (s or 0 in acts) else ""
        if name == "goto":
            return f"goto#{_goto_place(args)}"
        if name == "look_around":
            return "look-around"
        if name == "face":
            d = args.get("direction")
            return f"face({d})"
        return ""

    # ── misc ──

    def _write_heartbeat(self) -> None:
        self.heartbeat.write(
            self.live_dir / "heartbeat.json" if self.live_dir else None)

    @staticmethod
    def _result(content: list, info: dict) -> Any:
        from toolset import ToolResult
        return ToolResult(content=content, info=info)

    @property
    def verify_calls(self) -> int:
        return self._verify_calls


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
