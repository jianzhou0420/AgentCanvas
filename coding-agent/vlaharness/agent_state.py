from __future__ import annotations

"""AgentState — a state block the outer agent REWRITES, not one it appends to.

This is the difference from ``NavState``. There, the model *proposed* additions
and the harness applied them monotonically: sets that only ever grew, plus a
segment log carrying one model-written sentence per dispatch. Replayed against
the real runs, that design fails in a specific way — the block outgrew its
character cap on the 4th judgment in the median episode, ``render`` truncated
from the tail, and what fell off the end was exactly the accumulated memory the
design existed to protect, while the unbounded per-segment prose survived.

So the contract here is inverted:

  * **The model rewrites its whole block every turn.** Five fields, each capped
    at write time. Nothing accumulates, so nothing has to be truncated later —
    the rendered block is bounded by construction and no judgment ever sees a
    cut-off state. Compaction is the model's job and it does it continuously,
    which is the only way it stays lossy in the right places.
  * **A field the model omits is kept, not cleared.** A malformed or partial
    answer must not be able to erase the episode's memory.
  * **The harness keeps a separate factual ledger** — one fixed-width line per
    dispatch, written from telemetry, that the model cannot touch or reword.
    Rewriting is the right call for beliefs and the wrong call for measurements.

No coordinates anywhere: places are names and relations.

last updated: 2026-08-09
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from vlaharness.state import compact_actions

# The model's five fields. Each is a full rewrite each turn.
BLOCK_FIELDS = ("progress", "here", "layout", "ruled_out", "remaining")
FIELD_CAP = 420          # chars per field — 5 × 420 ≈ 2.1 KB, the whole block
LEDGER_SHOWN = 6         # dispatches rendered; older ones stay on disk
ASKED_SHOWN = 120        # chars of the instruction echoed back in the ledger

# Being wedged is handled in two stages, and both thresholds live here so they
# cannot drift apart: at STALL_WARN the block tells the model it is stuck and
# what the measurements say about fixing it; at STALL_ACT the harness stops
# asking and turns the robot itself. Warn first, then act — the model gets one
# round to solve it with information before the harness spends steps.
STALL_M = 0.3            # displacement below this counts as "did not move"
STALL_WARN = 2           # dispatches without movement before the block shouts
STALL_ACT = 3            # ... before the harness takes the wheel (agent_loop)

_LABEL = {
    "progress":  "PROGRESS ",
    "here":      "HERE     ",
    "layout":    "LAYOUT   ",
    "ruled_out": "RULED OUT",
    "remaining": "REMAINING",
}


@dataclass
class AgentState:
    # ── harness-owned: the task and the measurements ──────────────────
    mission: str = ""
    terminate: str = ""                                  # the verification test
    ledger: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    env_steps: int = 0
    step_budget: int = 400
    consecutive_stalls: int = 0
    vetoes: int = 0                                      # failed verifications
    path: str | None = None

    # ── model-owned: rewritten wholesale every turn ───────────────────
    progress: str = ""      # what of the mission is actually done
    here: str = ""          # where the robot is now, spatially
    layout: str = ""        # the space learned so far — what is where
    ruled_out: str = ""     # where the goal is not
    remaining: str = ""     # what still has to happen
    near_goal: bool = False

    # ── harness writes ────────────────────────────────────────────────

    def record(self, *, asked: str, telemetry: dict[str, Any],
               actions: list[str], decision: str, driver: str = "policy") -> None:
        """One dispatch becomes one fixed-width factual line. No model prose
        here — this is the record the model cannot rewrite."""
        moved = telemetry.get("net_displacement_m")
        self.ledger.append({
            "n": len(self.ledger) + 1,
            "asked": asked,
            "by": driver,
            "did": compact_actions(actions),
            "steps": telemetry.get("steps_used"),
            "moved_m": moved,
            "stop_reason": telemetry.get("stop_reason"),
            "decision": decision,
        })
        self.turn += 1
        self.env_steps = telemetry.get("steps_taken_total", self.env_steps)
        # The stall counter measures ONE thing: the policy failing to move when
        # asked to. A harness burst is not a candidate — it is usually pure
        # rotation, whose displacement is zero *by design*, so counting it would
        # have the unstick manoeuvre register as another stall and the counter
        # would never clear. It resets instead: the pose was deliberately
        # changed, so the policy gets a fresh count from a fresh view.
        if driver != "policy":
            self.consecutive_stalls = 0
        else:
            self.consecutive_stalls = (
                self.consecutive_stalls + 1 if (moved is not None and moved < STALL_M) else 0
            )

    def rewrite(self, block: Any) -> list[str]:
        """Apply the model's rewritten state block. Returns which fields changed.

        Replace, don't merge — that is the whole point. But an *omitted* field
        keeps its old value: a truncated or malformed answer costs this turn's
        update, never the episode's memory.
        """
        changed: list[str] = []
        if not isinstance(block, dict):
            return changed
        for name in BLOCK_FIELDS:
            val = block.get(name)
            if val is None:
                continue
            text = " ".join(str(val).split())
            if not text:
                continue                       # empty ≠ "erase it"
            if len(text) > FIELD_CAP:
                text = text[:FIELD_CAP - 1] + "…"
            if text != getattr(self, name):
                setattr(self, name, text)
                changed.append(name)
        if isinstance(block.get("near_goal"), bool):
            self.near_goal = block["near_goal"]
        return changed

    # ── rendering: the agent's entire memory, bounded by construction ──

    def render(self) -> str:
        L = [f"MISSION: {self.mission.strip()}",
             f"DONE WHEN: {self.terminate or '(not identified)'}",
             "",
             "── STATE BLOCK — you wrote this; you will rewrite it this turn ──"]
        if any(getattr(self, f) for f in BLOCK_FIELDS):
            for name in BLOCK_FIELDS:
                L.append(f"{_LABEL[name]}: {getattr(self, name) or '—'}")
        else:
            L.append("(empty — this is the first dispatch)")
        L.append("")

        if self.ledger:
            L.append("── DISPATCH LEDGER — written by the harness from telemetry ──")
            skipped = len(self.ledger) - LEDGER_SHOWN
            if skipped > 0:
                L.append(f"({skipped} earlier dispatches not shown — they are in "
                         f"PROGRESS above if they mattered)")
            for d in self.ledger[-LEDGER_SHOWN:]:
                by = "" if d["by"] == "policy" else f" [{d['by']}]"
                asked = d["asked"] or ""
                if len(asked) > ASKED_SHOWN:      # keeps the block bounded even
                    asked = asked[:ASKED_SHOWN] + "…"   # for a runaway instruction
                L.append(f"{d['n']:>2}{by} {asked!r}")
                L.append(f"    {d['did']}  —  {d['steps']} steps, moved {d['moved_m']} m, "
                         f"ended {d['stop_reason']}  →  {d['decision']}")
            L.append("")

        left = self.step_budget - self.env_steps
        L.append(f"BUDGET: {self.env_steps} of {self.step_budget} env steps spent, {left} left.")
        if self.near_goal:
            L.append("NEAR-GOAL FLAG IS SET — the termination test may be satisfiable now.")
        if self.consecutive_stalls >= STALL_WARN:
            # 744 measured dispatches: ~30% return no movement at EVERY
            # instruction length, so re-wording is not a reliable fix. The band
            # that does help is 4-12 words (26-31% vs ~40% for 1-3 and 17+).
            L.append("")
            L.append(f"!! THE LAST {self.consecutive_stalls} DISPATCHES DID NOT MOVE THE ROBOT.")
            L.append("   Re-wording alone rarely fixes this. Aim at OPEN SPACE you can see")
            L.append("   rather than at something already in view, or take the wheel with")
            L.append("   `drive` and change the pose yourself.")
        if self.vetoes:
            L.append(f"NOTE: {self.vetoes} finish attempt(s) already failed verification.")
        return "\n".join(L)

    # ── persistence ───────────────────────────────────────────────────

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
