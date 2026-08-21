from __future__ import annotations

"""NavState — the VLA harness's working memory, resident every turn.

Same three properties as the eharness StateBlock, for the same reasons:

- **Disk is the source of truth** (``state.json`` under the episode's live dir).
  The planner's context only ever sees ``render()``.
- **Monotone retention.** The two sets that cost env steps to earn — places
  visited and landmark bindings — may only grow. Compaction may drop pixels and
  rewrite prose; it may not drop a fact the robot paid steps to learn.
- **Single writer.** Only the harness mutates state. The planner *proposes*, on
  the same call that picks the next action, so belief updates cost no extra
  round trip.

What differs from eharness is what the state is *for*. There, the model is the
actuator and the state keeps it oriented. Here the actuator is a 2B policy with
no memory at all, and the state exists so the planner can answer one question
after each rollout: **given everything this robot has done, is the route
actually being carried out?** So the fields lean toward evidence — what was
asked, what the motors did, what was seen, what was ruled out.

No coordinates anywhere: places are names, relative descriptions, and frame
handles.

last updated: 2026-08-07
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Compact motor alphabet — an action trail has to survive being read as text.
MOTOR = {"MOVE_FORWARD": "F", "TURN_LEFT": "L", "TURN_RIGHT": "R", "STOP": "S"}


def compact_actions(actions: list[str]) -> str:
    """'F F F L L F' → 'F×3 L×2 F'. Run-length keeps a 50-step leg readable."""
    out, run, prev = [], 0, None
    for a in [MOTOR.get(x, "?") for x in actions]:
        if a == prev:
            run += 1
        else:
            if prev is not None:
                out.append(prev if run == 1 else f"{prev}×{run}")
            prev, run = a, 1
    if prev is not None:
        out.append(prev if run == 1 else f"{prev}×{run}")
    return " ".join(out)


@dataclass
class NavState:
    instruction: str = ""
    legs: list[str] = field(default_factory=list)
    # The stopping condition is a POINT, not the last leg. Decomposing an
    # instruction into N legs and treating the final sentence as leg N is how
    # a route ends up with no termination test at all.
    terminate: str = ""

    cursor: int = 0                                             # which leg now
    done: list[dict[str, Any]] = field(default_factory=list)    # segment memory
    current_place: str = ""                                     # relative, no coords
    surroundings: str = ""                                      # what's around me
    landmarks: dict[str, str] = field(default_factory=dict)     # name → where seen
    visited: list[str] = field(default_factory=list)            # ordered places
    ruled_out: list[str] = field(default_factory=list)          # negative facts
    lessons: list[str] = field(default_factory=list)            # stuck → freed
    near_goal: bool = False

    recent_motor: list[str] = field(default_factory=list)       # last few trails
    consecutive_recoveries: int = 0
    consecutive_stalls: int = 0        # dispatches that barely moved the robot
    env_steps: int = 0
    step_budget: int = 400
    turn: int = 0
    path: str | None = None

    # ── views ─────────────────────────────────────────────────────────

    @property
    def current_leg(self) -> str:
        return self.legs[self.cursor] if 0 <= self.cursor < len(self.legs) else ""

    @property
    def legs_remaining(self) -> int:
        return max(0, len(self.legs) - self.cursor)

    # ── mutation (harness only) ───────────────────────────────────────

    def record_segment(self, *, asked: str, telemetry: dict[str, Any],
                       actions: list[str], verdict: str, saw: str) -> None:
        """One dispatch becomes one durable segment. This is the episode memory
        that survives the images being dropped from context."""
        trail = compact_actions(actions)
        self.done.append({
            "n": len(self.done) + 1,
            "asked": asked,
            "did": trail,
            "steps": telemetry.get("steps_used"),
            "moved_m": telemetry.get("net_displacement_m"),
            "stop_reason": telemetry.get("stop_reason"),
            "verdict": verdict,
            "saw": saw,
        })
        self.recent_motor = (self.recent_motor + [trail])[-4:]
        self.env_steps = telemetry.get("steps_taken_total", self.env_steps)
        self.turn += 1
        self.consecutive_recoveries = (
            self.consecutive_recoveries + 1 if verdict == "recover" else 0
        )
        moved = telemetry.get("net_displacement_m")
        self.consecutive_stalls = (
            self.consecutive_stalls + 1 if (moved is not None and moved < 0.3) else 0
        )

    def propose(self, update: dict[str, Any]) -> list[str]:
        """Apply one planner-proposed belief update. Returns what was accepted.

        Monotone fields only ever grow; a malformed proposal loses the write,
        never the episode.
        """
        notes: list[str] = []
        if not isinstance(update, dict):
            return notes

        place = str(update.get("current_place") or "").strip()
        if place:
            if place != self.current_place:
                self.current_place = place
                if place not in self.visited:            # monotone
                    self.visited.append(place)
                notes.append(f"place={place}")

        around = str(update.get("surroundings") or "").strip()
        if around:
            self.surroundings = around

        for name, where in (update.get("landmarks") or {}).items():
            name = str(name).strip()
            if name and name not in self.landmarks:      # monotone
                self.landmarks[name] = str(where).strip()
                notes.append(f"landmark+{name}")

        for fact in update.get("ruled_out") or []:
            fact = str(fact).strip()
            if fact and fact not in self.ruled_out:      # monotone
                self.ruled_out.append(fact)
                notes.append("ruled_out+")

        lesson = str(update.get("lesson") or "").strip()
        if lesson and lesson not in self.lessons:
            self.lessons.append(lesson)
            notes.append("lesson+")

        if isinstance(update.get("near_goal"), bool):
            self.near_goal = update["near_goal"]
        return notes

    def advance(self) -> None:
        self.cursor = min(self.cursor + 1, len(self.legs))

    # ── rendering: the planner's entire memory of the episode ──────────

    def render(self, max_chars: int = 3000) -> str:
        L = []
        L.append(f"INSTRUCTION: {self.instruction.strip()}")
        L.append(f"STOP WHEN: {self.terminate or '(not identified)'}")
        L.append("")
        L.append("ROUTE PLAN")
        for i, leg in enumerate(self.legs):
            mark = "→" if i == self.cursor else ("✓" if i < self.cursor else " ")
            L.append(f"  {mark} {i + 1}. {leg}")
        L.append("")

        if self.done:
            L.append("WHAT THE ROBOT HAS ALREADY DONE")
            for d in self.done[-8:]:
                L.append(f"  [{d['n']}] asked: {d['asked']}")
                L.append(f"      motors: {d['did']}  ({d['steps']} steps, "
                         f"moved {d['moved_m']} m, ended: {d['stop_reason']})")
                if d.get("saw"):
                    L.append(f"      seen: {d['saw']}")
                L.append(f"      → {d['verdict']}")
            if len(self.done) > 8:
                L.insert(-8 * 4, f"  (+{len(self.done) - 8} earlier segments elided)")
            L.append("")

        if self.current_place or self.surroundings:
            L.append("WHERE I BELIEVE THE ROBOT IS")
            if self.current_place:
                L.append(f"  {self.current_place}")
            if self.surroundings:
                L.append(f"  around it: {self.surroundings}")
            L.append("")

        if self.visited:
            L.append("PLACES PASSED THROUGH (in order): " + " → ".join(self.visited))
        if self.landmarks:
            L.append("LANDMARKS IDENTIFIED:")
            for k, v in list(self.landmarks.items())[-12:]:
                L.append(f"  · {k} — {v}")
        if self.ruled_out:
            L.append("RULED OUT: " + "; ".join(self.ruled_out[-6:]))
        if self.lessons:
            L.append("LEARNED THE HARD WAY: " + "; ".join(self.lessons[-4:]))
        L.append("")
        L.append(f"BUDGET: {self.env_steps} of {self.step_budget} env steps spent, "
                 f"{self.step_budget - self.env_steps} left.")
        if self.consecutive_stalls >= 2:
            # Grounded in 744 measured dispatches: about 30% come back with no
            # movement at EVERY instruction length, so this is not something
            # phrasing reliably fixes. What the data does show is a band —
            # 4-12 words is 26-31% no-movement, while 1-3 words and 17+ are
            # both ~40%. Beyond that, a robot that has not moved in several
            # tries is usually wedged, not misaddressed.
            L.append("")
            L.append(f"!! THE LAST {self.consecutive_stalls} DISPATCHES DID NOT MOVE THE ROBOT.")
            L.append("   Re-wording is unlikely to fix this on its own — this policy returns")
            L.append("   an immediate stop on roughly 30% of dispatches at every instruction")
            L.append("   length. Two things do help:")
            L.append("   · Keep the instruction in the 4-12 word band. Measured no-movement")
            L.append("     rate is 26-31% there, versus ~40% for both bare 2-word commands")
            L.append("     and 17+ word descriptions.")
            L.append("   · Aim at OPEN SPACE you can see, not at an object you have already")
            L.append("     named. An instruction describing something currently in view can")
            L.append("     read as already satisfied.")
            L.append("   If the next attempt also fails to move it, the robot is wedged.")
            L.append("   Stop trying to re-aim: send a plain movement toward whichever view")
            L.append("   shows the most open floor, or accept this position and move on.")
        elif self.consecutive_recoveries >= 2:
            L.append(f"WARNING: {self.consecutive_recoveries} recovery attempts in a row. "
                     f"Stop correcting and commit to the most likely direction.")
        if self.near_goal:
            L.append("NEAR-GOAL FLAG IS SET — the stopping condition may be satisfiable now.")

        text = "\n".join(L)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(state truncated)"
        return text

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
