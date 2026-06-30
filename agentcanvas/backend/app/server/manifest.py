"""Manifest protocol — dataclasses for the BaseServer ↔ AgentCanvas contract.

These types are independent of AgentCanvas internals so they can be used
by standalone server processes without importing the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortSchema:
    """Describes one input or output port of a server function."""

    name: str
    wire_type: str  # IMAGE, DEPTH, ACTION, POSE, TEXT, BOOL, METRICS
    description: str = ""
    optional: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wire_type": self.wire_type,
            "description": self.description,
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PortSchema:
        return cls(
            name=d["name"],
            wire_type=d["wire_type"],
            description=d.get("description", ""),
            optional=d.get("optional", False),
        )


@dataclass
class FunctionSchema:
    """Describes one callable function exposed by a server."""

    name: str
    description: str = ""
    input_ports: list[PortSchema] = field(default_factory=list)
    output_ports: list[PortSchema] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    ui_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "config_schema": self.config_schema,
            "ui_config": self.ui_config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FunctionSchema:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            input_ports=[PortSchema.from_dict(p) for p in d.get("input_ports", [])],
            output_ports=[PortSchema.from_dict(p) for p in d.get("output_ports", [])],
            config_schema=d.get("config_schema", {}),
            ui_config=d.get("ui_config", {}),
        )


@dataclass
class ServerManifest:
    """Complete manifest returned by ``GET /manifest``."""

    name: str
    version: str = "1.0"
    description: str = ""
    functions: list[FunctionSchema] = field(default_factory=list)
    # Nodeset-level (owned) state container schemas — plain dicts in the shape
    # of ``ContainerDef.to_dict()``. Carried so the canvas can display a
    # server-mode nodeset's owned containers at rest. Kept as raw dicts to keep
    # this module free of framework imports (its stated contract).
    containers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "functions": [f.to_dict() for f in self.functions],
            "containers": list(self.containers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ServerManifest:
        return cls(
            name=d["name"],
            version=d.get("version", "1.0"),
            description=d.get("description", ""),
            functions=[FunctionSchema.from_dict(f) for f in d.get("functions", [])],
            containers=list(d.get("containers", [])),
        )
