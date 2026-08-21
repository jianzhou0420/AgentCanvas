"""Receipt — the only thing that crosses the sub→main context boundary.

Spec §4: claim + evidence (frame handles the parent can view, never prose-as-
oracle) + proposals (parent commits, sub proposes) + not_done (the raw material
of negative facts — "where I gave up" is free information) + cost telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Receipt:
    subgoal: str
    claim: str = "not_reached"     # reached | not_reached | blocked | budget_exhausted | episode_over
    what_i_see: str = ""           # one line, V1 quick-check only — not a fact source
    evidence_frames: list[int] = field(default_factory=list)   # recall handles
    proposes: dict[str, Any] = field(default_factory=dict)     # aggregated state proposals
    incidental: list[str] = field(default_factory=list)        # seen-but-not-asked
    not_done: str = ""
    steps_used: int = 0
    turns_used: int = 0
    guard_trips: int = 0
    end_reason: str = ""           # exit status of the sub-session
    verdict: str = ""              # filled by on_subagent_return (V1)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def append_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    def render(self) -> str:
        lines = [f"receipt: {self.claim} · subgoal: {self.subgoal[:80]}",
                 f"  steps {self.steps_used} · turns {self.turns_used} · "
                 f"guards {self.guard_trips} · end {self.end_reason}"]
        if self.what_i_see:
            lines.append(f"  sees: {self.what_i_see[:120]}")
        if self.not_done:
            lines.append(f"  not done: {self.not_done[:160]}")
        if self.incidental:
            lines.append("  incidental: " + " | ".join(s[:60] for s in self.incidental[:3]))
        if self.verdict:
            lines.append(f"  verdict: {self.verdict}")
        return "\n".join(lines)
