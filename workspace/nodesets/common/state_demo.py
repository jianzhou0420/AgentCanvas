"""state_demo — minimal demonstrator for nodeset-owned state containers.

A server-mode nodeset that **owns** one state container (``state_demo``) living
in its own subprocess. The single tool ``state_demo__bump`` writes to the
container by reference via ``ctx.containers`` on every fire — proving the
nodeset-level container path (declare → build-in-subprocess → by-reference
inject → live preview in the bottom State panel, Owner = "state_demo").

Load in server mode (forces the subprocess path; no special interpreter
needed)::

    POST /api/components/nodesets/state_demo/load?mode=server

Then drop ``state_demo__bump`` on a graph and run: the State panel shows the
``state_demo`` container with live ``fire_count`` (counter) and ``history``
(accumulator) values. (Nodeset-owned containers are built only in server mode;
in local mode this node is a no-op seed.)
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef
from app.graph_def import ContainerDef, StateDef


class BumpNode(BaseCanvasNode):
    """Seed node: bump the owned container's counter + append to its history."""

    node_type = "state_demo__bump"
    display_name = "State Demo Bump"
    description = "Increment fire_count and append to history in the owned container"
    category = "state_demo"
    icon = "Database"

    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("count", "TEXT", "Current fire_count as a string"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="indigo")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        containers = getattr(ctx, "containers", {}) or {}
        container = containers.get("state_demo")
        count = 0
        if container is not None:
            container.write("fire_count", 1)
            container.write("history", "bump")
            count = container.read("fire_count")
        return {"count": str(count)}


class StateDemoNodeSet(BaseNodeSet):
    """Owns a single nodeset-level state container for demonstration."""

    name: ClassVar[str] = "state_demo"
    description: ClassVar[str] = "Demonstrator for nodeset-owned state containers"
    # Owns a MUTABLE state container → must be replicated (per-worker private
    # subprocess copy). A `shared` server hosts one subprocess for all workers,
    # so the container would race under worker_count>1 — the #68 guardrail
    # warns about exactly that combination. (Was implicitly `shared` before.)
    parallelism: ClassVar[str] = "replicated"

    def get_tools(self) -> list:
        return [BumpNode()]

    def get_containers(self) -> list[ContainerDef]:
        return [
            ContainerDef(
                id="state_demo",
                label="State Demo",
                states={
                    "fire_count": StateDef(type="counter", value_type="METRICS"),
                    "history": StateDef(type="accumulator", value_type="TEXT"),
                },
            )
        ]
