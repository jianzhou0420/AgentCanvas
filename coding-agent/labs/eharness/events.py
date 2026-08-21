"""§14.7 — the unified NavigationEvent schema.

Every reflex the harness already had (collision brake, corridor collapse,
loop warning, episode end) and every detector this revision adds
(perception failure classes, registration drop, landmark sightings,
near-goal, topology change, held-goal invalidation) speaks ONE record
shape. Light geometric/change detectors run every primitive; the model is
only interrupted when an event says so — everything else surfaces at the
action endpoint as telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# severity ladder: info = worth telling the model at the endpoint;
# warning = the leg should stop so the model re-decides; critical = the
# world contradicted the plan (collision, perception loss) — stop now.
SEVERITIES = ("info", "warning", "critical")

# types that CANCEL the remaining leg when they fire mid-action (§14.5:
# depth/map failures fail CLOSED — a body that cannot see must not walk).
FAIL_CLOSED = ("perception_unavailable", "depth_invalid", "map_update_failed")


@dataclass
class NavigationEvent:
    type: str                 # collision | corridor_closed | loop |
    #                           perception_unavailable | depth_invalid |
    #                           map_update_failed | env_refused | episode_done |
    #                           new_opening | landmark_sighted | near_goal |
    #                           registration_drop | scene_transition |
    #                           held_invalidated | route_cancelled
    severity: str             # info | warning | critical
    frame_id: int             # sensor frame id (archive-independent, §14.8)
    map_version: int
    evidence: dict[str, Any] = field(default_factory=dict)
    requires_interrupt: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "severity": self.severity,
                "frame_id": self.frame_id, "map_version": self.map_version,
                "evidence": self.evidence,
                "requires_interrupt": self.requires_interrupt}
