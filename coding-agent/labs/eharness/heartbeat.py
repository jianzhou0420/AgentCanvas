"""Heartbeat — harness-written, telemetry-derived, overwriting (2026-08-04 ruling).

Not written by any model: derived from tool telemetry after every tool call,
overwrites heartbeat.json (a slot, not a log), costs zero model tokens. The
watchdog reads it per call; the planner reads the latest render exactly once,
when it wakes. "Frequency" is therefore not a tunable — it rides tool calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Heartbeat:
    subgoal: str = ""
    steps_used: int = 0
    steps_budget: int = 0
    tool_calls: int = 0
    guard_trips: int = 0
    last_note: str = ""          # digest of the last accepted state proposal
    status: str = "idle"         # idle | running | done | preempted
    t: float = field(default_factory=time.time)

    def write(self, path: Path | None) -> None:
        self.t = time.time()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.__dict__, ensure_ascii=False))

    def render(self) -> str:
        return (f"[HEARTBEAT] {self.status} · subgoal: {self.subgoal[:60]} · "
                f"steps {self.steps_used}/{self.steps_budget} · "
                f"calls {self.tool_calls} · guards {self.guard_trips}"
                + (f" · {self.last_note[:60]}" if self.last_note else ""))
