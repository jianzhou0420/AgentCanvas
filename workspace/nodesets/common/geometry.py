"""Geometry utility tools — deliberately cross-method (distance measurement).

Load:  POST /api/components/nodesets/geometry/load
"""

from __future__ import annotations

import json
import math
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef

# ── Tool definitions ──


class MeasureDistanceTool(BaseCanvasNode):
    node_type = "geometry__measure_distance"
    display_name = "Measure Distance"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    description = "Calculate Euclidean distance between two 3D points."
    category = "tool"
    icon = "Ruler"
    input_ports = [
        PortDef("point_a", "TEXT", "[x, y, z] coordinates of the first point"),
        PortDef("point_b", "TEXT", "[x, y, z] coordinates of the second point"),
    ]
    output_ports = [PortDef("result", "TEXT", "Distance measurement result JSON")]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        a = inputs["point_a"]
        b = inputs["point_b"]
        # Accept both JSON strings and lists
        if isinstance(a, str):
            a = json.loads(a)
        if isinstance(b, str):
            b = json.loads(b)
        dist = math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)))
        return {"result": json.dumps({"distance": round(dist, 4)})}


# ── NodeSet ──


class GeometryNodeSet(BaseNodeSet):
    name = "geometry"
    description = "Geometry utility tools"

    def get_tools(self) -> list:
        return [
            MeasureDistanceTool(),
        ]
