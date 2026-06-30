"""Mock frontier-scoring tool (random placeholder scores).

Quarantined in ``other/``: it is neither a real method part nor a generic
model — a prototyping mock kept import-runnable for graph sketching. Promote
it to a method satellite (or delete it) when frontier scoring becomes real.

Load:  POST /api/components/nodesets/mock_frontier/load
"""

from __future__ import annotations

import json
import random
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef

# ── Tool definitions ──


class ScoreFrontierTool(BaseCanvasNode):
    node_type = "mock_frontier__score_frontier"
    display_name = "[Mock] Score Frontier"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    description = "[Mock] Score a list of frontiers by exploration priority. Returns random placeholder scores."
    category = "tool"
    icon = "BarChart"
    input_ports = [
        PortDef("frontiers", "TEXT", "JSON list of frontier objects from GetFrontier"),
    ]
    output_ports = [PortDef("result", "TEXT", "Scored frontiers result JSON")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        frontiers = inputs.get("frontiers", [])
        if isinstance(frontiers, str):
            frontiers = json.loads(frontiers)
        scored = []
        for f in frontiers:
            score = round(random.uniform(0.1, 1.0), 3)
            scored.append(
                {
                    **f,
                    "score": score,
                    "priority": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"result": json.dumps({"scored_frontiers": scored, "total": len(scored)})}


# ── NodeSet ──


class MockFrontierNodeSet(BaseNodeSet):
    name = "mock_frontier"
    description = "Mock frontier-scoring tool (quarantined prototype)"

    def get_tools(self) -> list:
        return [
            ScoreFrontierTool(),
        ]
