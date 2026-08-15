from __future__ import annotations

"""EvidenceState — typed beliefs with provenance, replacing five free-text fields.

The previous state block (``agent_state.AgentState``) let the model rewrite five
prose fields wholesale. That fixed the truncation problem it was built for, but
it left a hole the measurements make concrete: **any sentence the model wrote
became durable state, with no evidence, no source, no age, and no way to retract
it except by remembering to rewrite the same field again.**

The asymmetry that makes this dangerous is specific:

  * A wrong *positive* belief ("the bar is ahead") is self-correcting — walking
    there disproves it.
  * A wrong *negative* belief ("the kitchen is not through that door") is not.
    Nothing ever revisits a ruled-out direction, so one bad sentence can remove
    the correct route from consideration for the rest of the episode.

So negative facts carry a higher burden of proof here: they must cite evidence
the harness actually recorded — a 360° sweep, a dispatch that went there and
came back, or a stall. A negative claim with no citation is refused, logged, and
does not enter state.

Everything else follows from wanting that to be checkable:

  * Every belief is a record, not a sentence: id, claim, kind, evidence, the turn
    it was created, the turn it was last confirmed, confidence, status.
  * Beliefs are *retired* or *contradicted* explicitly, by id. The model no
    longer has to reproduce a whole paragraph correctly to avoid losing a fact,
    and no longer silently loses one by forgetting to.
  * Evidence ids are frame handles the event log wrote to disk, so "why do you
    believe that" resolves to actual pixels after the run.

Rendering stays bounded by construction — caps on each list, oldest-first
eviction among *active* beliefs only, so a contradicted belief is dropped before
a live one.

last updated: 2026-08-10
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from vlaharness.state import compact_actions

KINDS = ("observed", "inferred", "negative")
STATUSES = ("active", "contradicted", "retired")

CLAIM_CAP = 160          # chars per belief
SHOWN = {"observed": 8, "inferred": 5, "negative": 6, "progress": 8}
LEDGER_SHOWN = 6
ASKED_SHOWN = 120

STALL_M = 0.3
STALL_WARN = 2
STALL_ACT = 3
# Two frames whose 64-bit difference hashes are this close are the same view.
# Used to tell "turned and saw something new" from "nothing changed".
SAME_VIEW_HAMMING = 6


@dataclass
class Belief:
    id: str = ""
    claim: str = ""
    kind: str = "observed"
    evidence_ids: list[str] = field(default_factory=list)
    created_turn: int = 0
    last_confirmed_turn: int = 0
    confidence: float = 0.5
    status: str = "active"

    def line(self) -> str:
        age = f"t{self.created_turn}"
        if self.last_confirmed_turn > self.created_turn:
            age += f"→t{self.last_confirmed_turn}"
        ev = f" ev:{','.join(self.evidence_ids[:3])}" if self.evidence_ids else ""
        mark = {"active": " ", "contradicted": "✗", "retired": "·"}[self.status]
        return f"  {mark}[{self.id}] {self.claim}  ({age} c={self.confidence:.1f}{ev})"


@dataclass
class EvidenceState:
    # ── harness-owned ────────────────────────────────────────────────
    mission: str = ""
    terminate: str = ""
    clauses: list[str] = field(default_factory=list)   # mission split for PROGRESS
    ledger: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    env_steps: int = 0
    step_budget: int = 400
    consecutive_stalls: int = 0
    consecutive_same_view: int = 0
    vetoes: int = 0
    path: str | None = None
    rejected_writes: list[str] = field(default_factory=list)

    # ── model-owned, but typed and gated ─────────────────────────────
    beliefs: list[Belief] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)  # per clause
    current_place: str = ""
    next_objective: str = ""
    near_goal: bool = False
    _seq: int = 0

    # ── harness writes ───────────────────────────────────────────────

    def record(self, *, asked: str, telemetry: dict[str, Any], actions: list[str],
               decision: str, driver: str = "policy",
               frame_hash: int | None = None, prev_hash: int | None = None) -> None:
        moved = telemetry.get("net_displacement_m")
        self.ledger.append({
            "n": len(self.ledger) + 1, "asked": asked, "by": driver,
            "did": compact_actions(actions), "steps": telemetry.get("steps_used"),
            "moved_m": moved, "stop_reason": telemetry.get("stop_reason"),
            "decision": decision,
            # Harness-only. Never rendered into the model's context.
            "_goal_m": telemetry.get("_goal_m"),
        })
        self.turn += 1
        self.env_steps = telemetry.get("steps_taken_total", self.env_steps)

        if driver != "policy":
            self.consecutive_stalls = 0
        else:
            self.consecutive_stalls = (
                self.consecutive_stalls + 1 if (moved is not None and moved < STALL_M) else 0
            )
        # A dispatch that changed neither position nor view is the strong wedged
        # signal; one that rotated into a genuinely new view is not, however
        # little it translated.
        if frame_hash is not None and prev_hash is not None:
            same = bin(frame_hash ^ prev_hash).count("1") <= SAME_VIEW_HAMMING
            self.consecutive_same_view = self.consecutive_same_view + 1 if same else 0
        else:
            self.consecutive_same_view = 0

    @property
    def wedged(self) -> bool:
        """Stuck for real: not moving AND not seeing anything new.

        Net displacement alone conflates a wall, an immediate STOP, a sensible
        turn in place, and lost pose telemetry. Requiring the view to be
        unchanged too is what separates "it rotated and found a doorway" from
        "it is grinding against the same wall".
        """
        return (self.consecutive_stalls >= STALL_ACT
                and self.consecutive_same_view >= STALL_ACT - 1)

    # ── model writes, gated ──────────────────────────────────────────

    def propose(self, update: Any, *, evidence_pool: set[str],
                swept: bool) -> tuple[list[str], list[str]]:
        """Apply a typed state update. Returns (accepted, rejected-with-reason).

        The gate is deliberately narrow: it checks provenance, not plausibility.
        A claim the harness cannot trace to something it recorded does not become
        durable state — most of all a negative one.
        """
        ok: list[str] = []
        bad: list[str] = []
        if not isinstance(update, dict):
            return ok, ["state update was not an object"]

        for raw in (update.get("beliefs") or [])[:12]:
            if not isinstance(raw, dict):
                continue
            claim = " ".join(str(raw.get("claim") or "").split())[:CLAIM_CAP]
            kind = str(raw.get("kind") or "observed").strip().lower()
            if not claim or kind not in KINDS:
                bad.append(f"malformed belief {str(raw)[:40]}")
                continue
            ev = [str(e) for e in (raw.get("evidence_ids") or []) if str(e) in evidence_pool]
            if kind == "negative" and not (ev or swept):
                # The load-bearing rule. A negative fact removes territory from
                # consideration permanently, so it needs something the harness
                # saw: a cited frame, or a full sweep taken this turn.
                bad.append(f"negative claim without evidence: {claim[:60]}")
                continue
            try:
                conf = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
            except Exception:
                conf = 0.5
            self._seq += 1
            self.beliefs.append(Belief(
                id=f"b{self._seq}", claim=claim, kind=kind, evidence_ids=ev[:4],
                created_turn=self.turn, last_confirmed_turn=self.turn,
                confidence=conf, status="active"))
            ok.append(f"+{kind}:b{self._seq}")

        by_id = {b.id: b for b in self.beliefs}
        for bid in (update.get("confirm") or [])[:12]:
            b = by_id.get(str(bid))
            if b and b.status == "active":
                b.last_confirmed_turn = self.turn
                b.confidence = min(1.0, b.confidence + 0.15)
                ok.append(f"confirm:{bid}")
        for bid in (update.get("contradict") or [])[:12]:
            b = by_id.get(str(bid))
            if b:
                b.status = "contradicted"
                ok.append(f"contradict:{bid}")
        for bid in (update.get("retire") or [])[:12]:
            b = by_id.get(str(bid))
            if b:
                b.status = "retired"
                ok.append(f"retire:{bid}")

        for item in (update.get("progress") or [])[:12]:
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("clause"))
            except Exception:
                continue
            st = str(item.get("status") or "").strip().lower()
            if st not in ("pending", "doing", "done") or not (0 <= i < len(self.clauses)):
                continue
            for p in self.progress:
                if p["clause"] == i:
                    p["status"] = st
                    break
            else:
                self.progress.append({"clause": i, "status": st})
            ok.append(f"clause{i}={st}")

        for fld in ("current_place", "next_objective"):
            v = " ".join(str(update.get(fld) or "").split())[:CLAIM_CAP]
            if v:
                setattr(self, fld, v)
                ok.append(fld)
        if isinstance(update.get("near_goal"), bool):
            self.near_goal = update["near_goal"]

        self.rejected_writes = (self.rejected_writes + bad)[-8:]
        return ok, bad

    # ── derived ──────────────────────────────────────────────────────

    @property
    def terminal_clause_active(self) -> bool:
        """Is the robot working on the last thing the mission asks for?

        The arrival gate must not fire on "walked into the kitchen" when the
        mission ends at the sink. With no clause finished, or the last one still
        pending while earlier ones are too, arrival is premature.
        """
        if not self.clauses:
            return True                      # nothing to be premature about
        done = {p["clause"] for p in self.progress if p["status"] == "done"}
        last = len(self.clauses) - 1
        return last in done or all(i in done for i in range(last))

    def active(self, kind: str) -> list[Belief]:
        return [b for b in self.beliefs if b.kind == kind and b.status == "active"]

    # ── rendering ────────────────────────────────────────────────────

    def render(self) -> str:
        L = [f"MISSION: {self.mission.strip()}",
             f"DONE WHEN: {self.terminate or '(not identified)'}", ""]

        if self.clauses:
            L.append("MISSION PROGRESS")
            st = {p["clause"]: p["status"] for p in self.progress}
            for i, c in enumerate(self.clauses):
                mark = {"done": "✓", "doing": "→", "pending": " "}.get(st.get(i, "pending"), " ")
                L.append(f"  {mark} [{i}] {c}")
            L.append("")

        L.append("BELIEFS — cite ids to confirm, contradict or retire them")
        for kind, label in (("observed", "OBSERVED"), ("inferred", "INFERRED"),
                            ("negative", "RULED OUT (needs evidence)")):
            act = self.active(kind)[-SHOWN[kind]:]
            if act:
                L.append(f" {label}")
                L += [b.line() for b in act]
        dead = [b for b in self.beliefs if b.status != "active"][-3:]
        if dead:
            L.append(" NO LONGER HELD")
            L += [b.line() for b in dead]
        if not self.beliefs:
            L.append("  (empty — nothing has been established yet)")
        L.append("")

        if self.current_place:
            L.append(f"WHERE I BELIEVE THE ROBOT IS: {self.current_place}")
        if self.next_objective:
            L.append(f"NEXT OBJECTIVE: {self.next_objective}")
        L.append("")

        if self.ledger:
            L.append("── DISPATCH LEDGER — harness-written from telemetry ──")
            skipped = len(self.ledger) - LEDGER_SHOWN
            if skipped > 0:
                L.append(f"({skipped} earlier dispatches on disk — ask by evidence id)")
            for d in self.ledger[-LEDGER_SHOWN:]:
                by = "" if d["by"] == "policy" else f" [{d['by']}]"
                asked = (d["asked"] or "")[:ASKED_SHOWN]
                L.append(f"{d['n']:>2}{by} {asked!r}")
                L.append(f"    {d['did']}  —  {d['steps']} steps, moved {d['moved_m']} m, "
                         f"ended {d['stop_reason']}  →  {d['decision']}")
            L.append("")

        left = self.step_budget - self.env_steps
        L.append(f"BUDGET: {self.env_steps} of {self.step_budget} env steps spent, {left} left.")
        if self.near_goal:
            L.append("NEAR-GOAL FLAG IS SET.")
        if self.rejected_writes:
            L.append("REFUSED STATE WRITES (no evidence): " + "; ".join(self.rejected_writes[-3:]))
        if self.consecutive_stalls >= STALL_WARN:
            L.append("")
            L.append(f"!! {self.consecutive_stalls} DISPATCHES WITHOUT MOVEMENT"
                     + (f", AND THE VIEW HAS NOT CHANGED IN {self.consecutive_same_view}."
                        if self.consecutive_same_view >= STALL_WARN else
                        " — but the view IS changing, so it is turning, not wedged."))
            L.append("   Aim at OPEN SPACE you can see rather than at something already in view.")
        return "\n".join(L)

    # ── persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        if not self.path:
            return
        d = asdict(self)
        d.pop("path", None)
        d["saved_at"] = time.strftime("%H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self.path)
