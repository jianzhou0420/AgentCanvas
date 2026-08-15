"""StateBlock — the harness's working memory, resident every turn.

The eleven-field structured state from agentic-loop-spec §6. Three properties
are load-bearing:

- **Disk is the source of truth** (``state.json`` under live_dir). Any loop's
  context only ever sees ``render()``; the file survives sub-session teardown,
  process boundaries (MCP bridge subprocesses), and — later — episodes.
- **Monotone retention** (spec §7): the three sets that cost env steps to earn
  — visited places, landmark bindings, negative facts — may only grow. The
  invariant is a real assert (``snapshot()`` / ``assert_monotone()``), fired by
  the pre_compact hook and after every receipt commit.
- **Single writer**: only the harness mutates state (``apply_update`` from the
  tool boundary, ``commit_receipt`` from the planner). Models *propose*.

No coordinates anywhere: places and landmarks are names + relative
descriptions + frame handles (see no-coordinates ruling, O2 §2.4).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Flat, optional, 9B-friendly: these ride on every action tool's arguments
# (decision-coupled writes — the same call that picks the action carries the
# belief update; zero extra LLM calls).
STATE_UPDATE_FIELDS: dict[str, dict] = {
    # Answers to the harness's blocking self-assessment. BOOLEANS, not prose:
    # asked the two questions in words, the 9B replied "very close to the
    # termination point, continuing forward" — the one answer the demand
    # explicitly forbade. A boolean cannot be answered with "approaching".
    "segment_done": {
        "type": "boolean",
        "description": "Answer to the harness's segment check: is the CURRENT "
                       "segment complete? Required after a [STOP AND ANSWER].",
    },
    "at_goal": {
        "type": "boolean",
        "description": "Answer to the harness's stop-condition check: does the "
                       "termination condition hold RIGHT NOW? Required after a "
                       "[STOP AND ANSWER].",
    },
    "progress_note": {
        "type": "string",
        "description": "One short line: where you are and how the current "
                       "sub-instruction is going.",
    },
    "answer": {
        "type": "string",
        "description": "Your answer to the '?' question in the state block, "
                       "if one is shown. The question stays until answered.",
    },
    "advance_subgoal": {
        "type": "boolean",
        "description": "Set true ONLY when the current sub-instruction is "
                       "completed and you are moving to the next one. When "
                       "you set it, ALSO fill progress_note with how you got "
                       "here (the route you took) — that becomes your record.",
    },
    "landmark": {
        "type": "string",
        "description": "A landmark you can now identify, e.g. 'the stairs' or "
                       "'red sofa in the living room'. One per call.",
    },
    "dead_end": {
        "type": "string",
        "description": "A ruled-out direction or area, e.g. 'hallway to the "
                       "left is a dead end'. Only after actually checking.",
    },
    "expect": {
        "type": "string",
        "description": "One short line: where these actions should land you "
                       "or what you expect to see next. Check it against the "
                       "view that comes back.",
    },
    "surroundings": {
        "type": "string",
        "description": "Your UNDERSTANDING of the environment around you "
                       "right now — one line, e.g. 'open lounge, pool behind "
                       "me, bar counter ahead-left, seating on my right'. "
                       "Update it whenever the scene changes.",
    },
    "near_goal": {
        "type": "boolean",
        "description": "Set true the moment you believe the FINAL goal is in "
                       "sight or close — it arms arrival verification on "
                       "every move (the loop's termination condition lights "
                       "up). Set false if you realize you were wrong.",
    },
    "revoke_dead_end": {
        "type": "string",
        "description": "Retract a recorded dead end that new evidence "
                       "disproves — give enough of its wording to match it. "
                       "The record stays, marked revoked with who and when.",
    },
}

# §14.11: writes are three-layered by PROVENANCE. Everything arriving through
# apply_update() is a MODEL BELIEF (revocable, rendered as "yours"); harness
# channels (record_visit / bind_landmark / add_journey / judge verdicts) are
# OBSERVED FACTS carrying frame/map_version/confidence; a judge-confirmed
# segment is a VERIFIED MILESTONE. Sources in this set count as harness even
# through apply_update (legacy migration uses it too).
HARNESS_SOURCES = ("harness", "survey", "sam", "judge")


@dataclass
class StateBlock:
    instruction: str = ""
    sub_instructions: list[str] = field(default_factory=list)   # 1 路段 legs
    terminate: str = ""       # the TERMINATION CONDITION — a point, not a leg
                              # (用户 2026-08-07: 导航最重要的是停止，最后一句
                              # 不是第三段路，是终止条件；拆解要结构化)
    cursor: int = 0                                             # 2 当前子指令
    done: list[dict[str, Any]] = field(default_factory=list)    # 3 已完成+凭据
    current_place: str = ""                                     # 5 相对描述
    pending_question: str = ""
    """A question the harness wants answered THIS turn — asked once, then gone.

    Deliberately not in the monotone set and deliberately not a statement. The
    path from an asserted memory ("you have been here") to an armed stop
    condition has no gate in it, so a false one is unrecoverable for a small
    model; a question the model can answer "no" to is not."""
    surroundings: str = ""    # 对环境的理解——我周围大概是什么 (用户 2026-08-07)
    landmarks: dict[str, dict[str, Any]] = field(default_factory=dict)  # 6
    visited: list[dict[str, Any]] = field(default_factory=list)         # 7
    negative_facts: list[dict[str, Any]] = field(default_factory=list)  # 8
    checkpoint: str = ""                                        # 9 最后确认点
    delegation: dict[str, Any] | None = None                    # 10 当前委派
    receipts_tail: list[dict[str, Any]] = field(default_factory=list)   # 11
    last_action: str = ""     # harness-written motor record ("fwd×3 L×1")
    expectation: str = ""     # the model's own prediction, held up to it
    lessons: list[str] = field(default_factory=list)  # stuck→freed memories
    recent_moves: list[str] = field(default_factory=list)  # motor trail (cap 6)
    journey: list[str] = field(default_factory=list)  # harness-written story
    near_goal: bool = False   # model-armed near-stop flag (termination lights up)
    env_steps: int = 0        # wrapper-updated, for the render only
    step_budget: int = 0
    # §14.11 provenance: when each BELIEF field was last written (frame/step).
    # Facts carry their provenance inline (landmarks/negative_facts entries).
    belief_meta: dict[str, dict] = field(default_factory=dict)
    # Monotone display version. Bumped EXPLICITLY by the wrapper each time the
    # state is injected into a tool result (render itself stays pure): on
    # full-history executors (Claude SDK keeps every old tool result) the
    # model sees many stale [STATE] copies at once, and the version line is
    # what tells it which one is law.
    state_version: int = 0
    # bookkeeping (not rendered)
    turn: int = 0
    path: Path | None = None

    # ── structured-route views (legacy states have no terminate) ──

    @property
    def route_segments(self) -> list[str]:
        """The walkable legs — what milestone checks verify. With a
        structured parse that is ALL of sub_instructions; legacy states
        (terminate empty) treat the last sub-instruction as the goal."""
        return self.sub_instructions if self.terminate else self.sub_instructions[:-1]

    @property
    def goal_text(self) -> str:
        """What the brake/bell verify against."""
        if self.terminate:
            return self.terminate
        return self.sub_instructions[-1] if self.sub_instructions else self.instruction

    def route_context(self) -> str:
        """The composed goal for terminal judges (用户 2026-08-07: 终止条件
        不是孤立的最后一段——它有前置：先走过 bar 和椅子之间，才轮到那个
        bar 的角。v3.1 EP0 stopped at a reception desk 6m off because every
        judge verified 'a bar corner' with no route context)."""
        segs = "; ".join(f"[{i + 1}] {s}"
                         for i, s in enumerate(self.sub_instructions))
        term = self.goal_text
        if not segs:
            return term
        return (f"TERMINATION: {term}\n"
                f"PRECONDITION — the route, in order: {segs}. The "
                f"termination point belongs to the scene of the FINAL "
                "segment and comes only AFTER the earlier segments are "
                "traversed. A look-alike structure encountered earlier "
                "along the route (e.g. a similar counter near the start) "
                "does NOT count.")

    @property
    def near_stop_armed(self) -> bool:
        """Termination condition is LIVE: model believes it's near, or every
        route segment is done."""
        if self.near_goal:
            return True
        segs = self.route_segments
        return bool(self.terminate or self.sub_instructions) and self.cursor >= len(segs)

    # ── persistence: disk is the source of truth ──

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if k != "path"}
        # the literal text the model sees each turn — for monitors; load()
        # ignores it (no matching attr), so the round-trip stays clean
        data["_rendered"] = self.render()
        data["_schema"] = 2
        # Atomic: a process killed mid-write used to leave half a JSON, and
        # load() has no fallback for that. Write beside, then replace.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: Path) -> "StateBlock":
        sb = cls(path=path)
        if path.exists():
            for k, v in json.loads(path.read_text()).items():
                if hasattr(sb, k) and k != "path":
                    setattr(sb, k, v)
        sb._migrate_provenance()
        return sb

    def _migrate_provenance(self) -> None:
        """Older state.json entries predate the three-layer split (§14.11):
        landmarks had no origin, done entries no verified flag. Migrate on
        load with the honest defaults — a model tool write is a belief, and
        only a judge-evidence entry counts as verified."""
        for meta in self.landmarks.values():
            if isinstance(meta, dict) and "origin" not in meta \
                    and not meta.get("sunk"):
                by = str(meta.get("by", ""))
                meta["origin"] = ("harness" if by in HARNESS_SOURCES
                                  else "model")
        for d in self.done:
            if isinstance(d, dict) and "verified" not in d:
                d["verified"] = str(d.get("evidence", "")
                                    ).startswith("judge-evidence:")

    # ── writes (harness-only; models propose) ──

    def apply_update(self, update: dict[str, Any], *, source: str,
                     frame: int | None = None,
                     step: int | None = None,
                     negative_verified: bool = False) -> list[str]:
        """Apply one decision-coupled proposal. Returns human-readable notes
        of what was accepted (fed back into the tool result so the model sees
        its write landed). Unknown keys are ignored, never fatal — a 9B that
        muddles the schema loses the write, not the episode.

        §14.11: everything landing here is a MODEL BELIEF unless the source
        is in HARNESS_SOURCES — each write carries who/frame/step, and the
        negative set gains an explicit revocation channel."""
        origin = "harness" if source in HARNESS_SOURCES else "model"

        def _meta(field_name: str) -> None:
            self.belief_meta[field_name] = {
                "by": source, "origin": origin, "frame": frame, "step": step}

        accepted: list[str] = []
        note = update.get("progress_note")
        if isinstance(note, str) and note.strip():
            self.current_place = note.strip()[:200]
            _meta("current_place")
            accepted.append("progress noted")
        lm = update.get("landmark")
        if isinstance(lm, str) and lm.strip():
            name = lm.strip()[:80]
            if name not in self.landmarks:
                self.landmarks[name] = {"frame": frame, "step": step,
                                        "by": source, "origin": origin,
                                        "t": round(time.time(), 1)}
                accepted.append(f"landmark '{name}' bound")
        de = update.get("dead_end")
        if isinstance(de, str) and de.strip():
            fact = de.strip()[:160]
            if all(f["fact"] != fact for f in self.negative_facts):
                # Wrong negative facts self-seal (O2 §6.1): store them, but
                # carry the verification mark so retrieval can caveat them.
                self.negative_facts.append(
                    {"id": f"n{len(self.negative_facts) + 1}",
                     "fact": fact, "frame": frame, "step": step,
                     "by": source, "origin": origin,
                     "verified": negative_verified,
                     # Which place this was true IN. A dead end is not a global
                     # fact — "the door on the left is a dead end" becomes a
                     # fluent, permanent, exactly-reversed instruction when the
                     # same corridor is re-entered from the other end.
                     "place": self.current_place[:80] or ""})
                accepted.append(f"dead end recorded ({'verified' if negative_verified else 'unverified'})")
        rv = update.get("revoke_dead_end")
        if isinstance(rv, str) and rv.strip():
            # §14.11: negative conclusions must be revocable. The entry never
            # leaves (monotone set is keyed on ids) — it is MARKED, with who,
            # at which frame, so the retraction itself is auditable.
            #
            # Matching (review P2 — first-match bidirectional substring
            # revoked the WRONG fact): exact casefold equality wins outright;
            # otherwise candidates are gated by informative-word overlap
            # (generic words like "dead"/"end"/"the" identify nothing) and
            # the best-scoring, most-recent one is revoked. A key that
            # matches nothing revokes nothing — a safe no-op beats striking
            # a true fact the body paid steps to establish.
            key = rv.strip().casefold()
            live = [f for f in self.negative_facts if not f.get("revoked")]
            _STOP = {"the", "a", "an", "is", "are", "was", "were", "not",
                     "no", "and", "of", "to", "by", "at", "in", "on",
                     "after", "all", "dead", "end"}

            def _inf(text: str) -> set:
                return {w for w in re.findall(r"[a-z0-9]+", text)
                        if w not in _STOP}

            key_inf = _inf(key)
            target = next((f for f in live
                           if f["fact"].casefold() == key), None)
            if target is None:
                need = min(2, len(key_inf)) or None
                scored = []
                for f in live:
                    fact_cf = f["fact"].casefold()
                    overlap = len(key_inf & _inf(fact_cf))
                    if key in fact_cf or (need and overlap >= need):
                        scored.append((overlap, f))
                if not scored and not key_inf:
                    # a purely-generic key ("dead end"): substring both ways
                    scored = [(0, f) for f in live
                              if key in f["fact"].casefold()
                              or f["fact"].casefold() in key]
                if scored:
                    best = max(s for s, _ in scored)
                    target = [f for s, f in scored if s == best][-1]
            if target is not None:
                target["revoked"] = {"by": source, "frame": frame,
                                     "step": step, "t": round(time.time(), 1)}
                accepted.append(f"dead end '{target['fact'][:40]}' revoked")
        sur = update.get("surroundings")
        if isinstance(sur, str) and sur.strip():
            self.surroundings = sur.strip()[:200]
            _meta("surroundings")
            accepted.append("surroundings updated")
        ans = update.get("answer")
        if isinstance(ans, str) and ans.strip() and self.pending_question:
            # the explicit ack (§12.8): an accepted answer — and nothing
            # else — clears the question. It also becomes part of the story.
            self.journey.append(
                (f"answered '{self.pending_question[:40]}': "
                 f"{ans.strip()[:60]}")[:120])
            del self.journey[:-10]
            self.ack_question()
            accepted.append("question answered")
        exp = update.get("expect")
        if isinstance(exp, str) and exp.strip():
            self.expectation = exp.strip()[:160]
            _meta("expectation")
            accepted.append("expectation noted")
        ng = update.get("near_goal")
        if isinstance(ng, bool) and ng != self.near_goal:
            self.near_goal = ng
            _meta("near_goal")
            accepted.append("NEAR-STOP " + ("armed — arrival checks active "
                            "on every move" if ng else "disarmed"))
        if update.get("advance_subgoal") is True:
            if self.cursor < len(self.sub_instructions):
                # a model-claimed advance is a BELIEF until the milestone
                # judge confirms it (§14.11) — _verify_milestones() appends
                # its entries with verified=True
                self.done.append({"idx": self.cursor,
                                  "text": self.sub_instructions[self.cursor],
                                  "evidence": (note or "")[:160], "frame": frame,
                                  "verified": origin == "harness"})
                self.checkpoint = self.current_place or self.checkpoint
                self.cursor += 1
                accepted.append(f"advanced to sub-instruction {self.cursor + 1}")
                self.add_journey(
                    f"finished segment [{self.cursor}/{len(self.sub_instructions)}]"
                    + (f" — {(note or '')[:60]}" if note else ""))
        if accepted:
            self.save()
        return accepted

    def add_lesson(self, text: str) -> None:
        """A crisis memory: what went wrong and what freed it. The steps in
        between are dropped from history — this line is what remains."""
        self.lessons.append(text[:160])
        del self.lessons[:-5]
        self.save()

    def add_journey(self, text: str) -> None:
        """One entry in the harness-WRITTEN story of the walk (用户
        2026-08-06: 模型对自己一连串动作在干什么并不清晰,而且不该让模型
        自己叙述——这是 harness 的活). Appended by the wrapper at event
        moments (verified-on-track, obstacle freed, revisit, advance);
        the model reads its own trajectory back as a story, never writes it."""
        self.journey.append(text[:120])
        del self.journey[:-10]
        self.save()

    def bind_landmark(self, name: str, *, frame: int | None = None,
                      map_version: int | None = None,
                      confidence: float | None = None,
                      by: str = "harness") -> bool:
        """§14.11 ObservedFact channel: a landmark the HARNESS (geometry/SAM/
        judge) established, with the evidence handles a model claim cannot
        carry — frame, map version, detector confidence. Renders under
        OBSERVED, not under the model's claims."""
        name = name.strip()[:80]
        if not name or name in self.landmarks:
            return False
        self.landmarks[name] = {
            "frame": frame, "by": by, "origin": "harness",
            **({"map_version": map_version} if map_version is not None else {}),
            **({"confidence": round(float(confidence), 2)}
               if confidence is not None else {}),
            "t": round(time.time(), 1)}
        self.save()
        return True

    def note_move(self, motion: str) -> None:
        self.recent_moves.append(motion)
        del self.recent_moves[:-6]

    def record_visit(self, place: str, frame: int | None = None) -> None:
        if place and (not self.visited or self.visited[-1]["place"] != place):
            # A STABLE ID, assigned once. The monotone set is keyed on it rather
            # than on the display string, so a place can later be renamed —
            # "gray couch on my left" → "the couch, in the room with the pool" —
            # without the rename reading as "one entry deleted, one added" and
            # tripping the retention assertion. Nothing is ever removed; only
            # the words are allowed to improve.
            self.visited.append({"id": f"v{len(self.visited) + 1}",
                                 "place": place[:120], "frame": frame})
            self.save()

    def rename_visit(self, vid: str, place: str) -> bool:
        """Improve the wording of a place already recorded. Monotone-safe."""
        for v in self.visited:
            if v.get("id") == vid:
                v["place"] = place[:120]
                self.save()
                return True
        return False

    # ── monotone retention (spec §7) ──

    def snapshot(self) -> dict[str, set]:
        # Keyed on ids where they exist (entries restored from an older run may
        # not have one, and their string is then the id). What must never be
        # lost is the ENTRY, not the sentence describing it.
        return {
            "visited": {v.get("id") or v["place"] for v in self.visited},
            "landmarks": set(self.landmarks),
            "negative_facts": {f.get("id") or f["fact"] for f in self.negative_facts},
        }

    @staticmethod
    def assert_monotone(before: dict[str, set], after: dict[str, set]) -> None:
        """The compaction contract: pixels may go, prose may be rewritten —
        a place you visited or a fact you paid steps for never leaves."""
        for key in ("visited", "landmarks", "negative_facts"):
            lost = before[key] - after[key]
            if lost:
                raise AssertionError(
                    f"monotone retention violated: {key} lost {sorted(lost)!r}")

    # ── render (the ONLY thing any model context ever contains) ──

    STATE_MAX_CHARS = 3000

    def render(self, *, max_chars: int = STATE_MAX_CHARS) -> str:
        """PURE — same state in, same text out, nothing consumed (§12.8).

        The old render cleared pending_question as it printed it, so any
        monitoring render or bookkeeping save() (which embeds a render) ate
        the question before the model's turn ever saw it; and it truncated
        the joined string at 2400 chars, amputating whatever happened to
        render last — landmarks, ruled-out, checkpoint — mid-word.

        Now: four labelled sections; MISSION and OPEN ITEMS are never cut;
        the bounded lists in HARNESS FACTS / MODEL BELIEF shrink their
        windows level by level until the whole block fits, with an overflow
        count instead of a silent amputation. Clearing the question is an
        explicit act — the model answers via the `answer` state field and
        apply_update() acknowledges it."""
        # shrink levels: (journey, done, lessons, landmarks, neg_facts, moves)
        for j_n, d_n, l_n, lm_n, nf_n, mv in (
                (6, 3, 3, 6, 4, True), (4, 2, 2, 4, 3, True),
                (3, 1, 1, 3, 2, False), (2, 1, 1, 2, 1, False)):
            text = self._render_at(j_n, d_n, l_n, lm_n, nf_n, mv)
            if len(text) <= max_chars:
                return text
        return text     # smallest level; MISSION+OPEN are worth going over for

    def _render_at(self, j_n: int, d_n: int, l_n: int, lm_n: int,
                   nf_n: int, moves: bool) -> str:
        # §14.2: on full-history executors many stale STATE copies coexist in
        # the transcript — the version line names the one that is law. Version
        # 0 (mini/solo: only the newest survives compilation) keeps the plain
        # header, byte-identical to before.
        head = (f"[STATE v{self.state_version} — supersedes every earlier "
                "STATE block]" if self.state_version > 0 else "[STATE]")
        lines: list[str] = [head]
        # ── MISSION — never cut ──
        if self.terminate:
            lines.append(f"  TERMINATION — where/when to STOP: {self.terminate}")
        elif self.sub_instructions:
            lines.append(f"  FINAL GOAL: {self.sub_instructions[-1]}")
        # ALL route segments, marked — the full route card, not a window.
        # §14.11: a judge-confirmed segment reads "done ✓", a model-claimed
        # one "done (claimed)" — the two were indistinguishable before.
        verified_idx = {d.get("idx") for d in self.done if d.get("verified")}
        for i, si in enumerate(self.sub_instructions):
            mark = (("done ✓" if i in verified_idx else "done (claimed)")
                    if i < self.cursor
                    else "NOW ▸" if i == self.cursor
                    else "WATCH FOR" if i == self.cursor + 1 else "later")
            lines.append(f"  [{i + 1}/{len(self.sub_instructions)}] "
                         f"({mark}) {si}")
        if self.near_stop_armed:
            lines.append("  ⚑ NEAR-STOP ARMED — every move is arrival-checked;"
                         " STOP the moment you and the verifier agree you are"
                         " at the goal. Walking past fails like stopping short.")
        # ── OBSERVED — measured/harness-written (§14.11 layer 1) ──
        # NATURAL LANGUAGE (用户 2026-08-08: 人的状态块——我这一路在干嘛、有
        # 什么特殊情况、我大概在哪、周围是什么). The harness composes it; the
        # group header tells the model these lines are measured, not claimed.
        observed: list[str] = []
        if self.journey:
            omitted = max(0, len(self.journey) - j_n)
            observed.append("  What I've done on this walk: "
                            + "; then ".join(self.journey[-j_n:]) + "."
                            + (f" (+{omitted} earlier)" if omitted else ""))
        for d in self.done[-d_n:]:
            if d.get("evidence"):
                observed.append(f"  How segment [{d['idx'] + 1}] went: "
                                f"{d['evidence']}")
        if self.lessons:
            observed.append("  Unusual things so far: "
                            + " Also: ".join(self.lessons[-l_n:]))
        if self.last_action:
            recent = (" (before that: " + ", ".join(self.recent_moves[:-1])
                      + ")" if moves and len(self.recent_moves) > 1 else "")
            observed.append(f"  I just did: {self.last_action}{recent}.")
        if self.step_budget:
            observed.append(f"  env steps used: "
                            f"{self.env_steps}/{self.step_budget}")
        lm_names = list(self.landmarks)
        lm_fact = [n for n in lm_names
                   if self.landmarks[n].get("origin") == "harness"]
        lm_claim = [n for n in lm_names if n not in lm_fact]
        if lm_fact:
            omitted = max(0, len(lm_fact) - lm_n)
            observed.append("  landmarks (observed): "
                            + " | ".join(lm_fact[-lm_n:])
                            + (f" (+{omitted} more)" if omitted else ""))
        live_nf = [f for f in self.negative_facts if not f.get("revoked")]
        n_revoked = len(self.negative_facts) - len(live_nf)
        verified_nf = [f for f in live_nf if f.get("verified")]
        if verified_nf:
            observed.append("  ruled out (verified): " + " | ".join(
                f["fact"] for f in verified_nf[-nf_n:])
                + (f" ({n_revoked} revoked)" if n_revoked else ""))
        if self.checkpoint:
            observed.append(f"  last confirmed: {self.checkpoint}")
        if observed:
            lines.append("  OBSERVED — measured by the harness:")
            lines.extend(observed)
        # ── VERIFIED milestones (§14.11 layer 3) ──
        if verified_idx:
            lines.append("  VERIFIED milestones: " + ", ".join(
                f"segment [{i + 1}] ✓" for i in sorted(verified_idx))
                + " (judge-confirmed from your own footage)")
        # ── YOUR BELIEFS — the model's own claims, revocable (layer 2) ──
        lines.append("  YOUR BELIEFS — unverified, revocable:")
        place = self.current_place or "not noted yet"
        around = self.surroundings or "not noted yet"
        lines.append(f"  My belief — where I am: {place}. Around me: {around}.")
        if lm_claim:
            omitted = max(0, len(lm_claim) - lm_n)
            lines.append("  landmarks you named: " + " | ".join(lm_claim[-lm_n:])
                         + (f" (+{omitted} more)" if omitted else ""))
        unverified_nf = [f for f in live_nf if not f.get("verified")]
        if unverified_nf:
            lines.append("  dead ends you claimed (unverified — revoke via "
                         "`revoke_dead_end` if wrong): " + " | ".join(
                             f["fact"] for f in unverified_nf[-nf_n:]))
        if self.expectation:
            lines.append(f"  I expected this to: {self.expectation} — check "
                         "that against the current view.")
        # ── OPEN ITEMS — never cut; each waits for an explicit answer ──
        if self.pending_question:
            lines.append(f"  ? {self.pending_question} — answer via the "
                         "`answer` state field on your next action")
        return "\n".join(lines)

    def ack_question(self) -> None:
        """The ONLY way the pending question is cleared: the harness accepted
        an answer. Renders, saves, failed API calls and retries no longer
        consume it (§12.8)."""
        self.pending_question = ""

    def render_full(self) -> str:
        """Planner view: everything, still bounded (receipts tail rolls)."""
        parts = [self.render()]
        if self.done:
            parts.append("  completed: " + "; ".join(
                f"#{d['idx'] + 1} ({d['evidence'] or 'no evidence noted'})"
                for d in self.done[-6:]))
        if self.visited:
            parts.append("  visited: " + " -> ".join(
                v["place"] for v in self.visited[-8:]))
        for r in self.receipts_tail[-3:]:
            parts.append(f"  receipt[{r.get('subgoal', '?')[:60]}]: "
                         f"{r.get('claim')} ({r.get('verdict', 'unjudged')}) "
                         f"steps={r.get('steps_used')}"
                         + (f" not_done: {r['not_done'][:80]}" if r.get("not_done") else ""))
        return "\n".join(parts)

    # ── sinking (compaction = memory flow, spec §5.4; negative facts never sink) ──

    def sink_cold(self, memory_path: Path, *, keep_landmarks: int = 12) -> int:
        """Move cold landmark entries to the external memory document, leaving
        the name behind (the name IS the recall handle). Deterministic, no
        model call. Returns number of entries sunk."""
        if len(self.landmarks) <= keep_landmarks:
            return 0
        cold = list(self.landmarks)[:-keep_landmarks]
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open("a") as fh:
            for name in cold:
                fh.write(f"- landmark: {name} · {json.dumps(self.landmarks[name], ensure_ascii=False)}\n")
        # names stay (monotone set unaffected) — only the metadata sinks.
        # Provenance survives the sinking: dropping origin/by re-rendered a
        # measured harness fact as a revocable model claim (review P2).
        for name in cold:
            old = self.landmarks[name] if isinstance(
                self.landmarks[name], dict) else {}
            stub = {"sunk": True}
            for k in ("origin", "by"):
                if old.get(k) is not None:
                    stub[k] = old[k]
            self.landmarks[name] = stub
        self.save()
        return len(cold)
