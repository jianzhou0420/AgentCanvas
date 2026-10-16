"""The node catalog: which node types exist in this process, and what ports
each *instance* of one has.

One :class:`NodeCatalog` holds every ``BaseCanvasNode`` subclass the
executor may fire — the framework builtins, workspace nodes, and the proxy
classes generated for hosted nodesets. Registration and eviction go
through it (``register`` / ``unregister``), so there is no cache to
invalidate; the registry is its client.

It also owns the one rule for "which ports does this node have": a node
class may override ``_resolve_ports(config)`` (ADR-dataflow-003) to derive
its port surface from the instance config (``llmCall``, ``iterIn``,
``graphOut``, the viewer sinks, …); :meth:`ports_of` asks it and falls
back to the class-level lists when there is no resolver or it raises.
Required ports, wire types and the serialized schema the frontend and the
SDK consume are all derived from that one answer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from .bases import BaseCanvasNode, PortDef, ui_config_to_dict

log = logging.getLogger("agentcanvas.catalog")


@dataclass
class Ports:
    """A node instance's port surface."""

    inputs: list[PortDef] = field(default_factory=list)
    outputs: list[PortDef] = field(default_factory=list)

    @property
    def required(self) -> set[str]:
        """Input ports the node will not fire without."""
        return {p.name for p in self.inputs if not getattr(p, "optional", True)}

    def wire_type(self, handle: str, direction: str) -> str | None:
        """Wire type of ``handle`` on the ``"in"`` or ``"out"`` side, or
        ``None`` when no such port exists."""
        for p in self.inputs if direction == "in" else self.outputs:
            if getattr(p, "name", None) == handle:
                return getattr(p, "wire_type", None)
        return None


class NodeCatalog:
    """Node types keyed by ``node_type``; last registration wins."""

    def __init__(self, classes: Iterable[type[BaseCanvasNode]] = ()):
        self._types: dict[str, type[BaseCanvasNode]] = {}
        for cls in classes:
            self.register(cls)

    # ── Registration ────────────────────────────────────────────────

    def register(self, cls: type[BaseCanvasNode]) -> None:
        """Add (or replace) a node class under its ``node_type``.

        ADR-028 PC-1: a ``batched=True`` class must name an actual input
        port as its ``batch_dim`` — fail fast here so misconfigured nodes
        never reach the runtime rendezvous tier.
        """
        if getattr(cls, "batched", False):
            batch_dim = getattr(cls, "batch_dim", "")
            if not batch_dim:
                raise ValueError(f"{cls.__name__}: batched=True requires a non-empty batch_dim")
            port_names = {p.name for p in getattr(cls, "input_ports", [])}
            if batch_dim not in port_names:
                raise ValueError(
                    f"{cls.__name__}: batch_dim={batch_dim!r} must name an "
                    f"input port, but input_ports has {sorted(port_names)}"
                )
        self._types[cls.node_type] = cls

    def unregister(self, node_types: Iterable[str]) -> list[str]:
        """Drop the given node types; unknown ones are ignored. Returns the
        types that were actually present."""
        removed: list[str] = []
        for nt in list(node_types):
            if self._types.pop(nt, None) is not None:
                removed.append(nt)
        return removed

    # ── Lookup ──────────────────────────────────────────────────────

    def get(self, node_type: str) -> type[BaseCanvasNode] | None:
        return self._types.get(node_type)

    def __contains__(self, node_type: object) -> bool:
        return node_type in self._types

    def __len__(self) -> int:
        return len(self._types)

    def types(self) -> list[str]:
        """Node types in registration order."""
        return list(self._types)

    def items(self) -> list[tuple[str, type[BaseCanvasNode]]]:
        return list(self._types.items())

    # ── Ports: the one rule ─────────────────────────────────────────

    def ports_of(self, node_type: str, config: dict | None = None) -> Ports | None:
        """The port surface of one node instance, or ``None`` for an
        unknown type. Honours the class's ``_resolve_ports(config)``; falls
        back to the class-level lists when it is absent or raises."""
        cls = self._types.get(node_type)
        if cls is None:
            return None
        return ports_for(cls, config)

    def required_inputs(self, node_type: str, config: dict | None = None) -> set[str]:
        """Input ports an instance will not fire without (empty for an
        unknown type)."""
        ports = self.ports_of(node_type, config)
        return ports.required if ports is not None else set()

    def wire_type(
        self, node_type: str, handle: str, direction: str, config: dict | None = None
    ) -> str | None:
        """Wire type of one port on an instance, ``None`` when the type or
        the handle is unknown."""
        ports = self.ports_of(node_type, config)
        return ports.wire_type(handle, direction) if ports is not None else None

    # ── Serialization ───────────────────────────────────────────────

    def schema(self, node_type: str) -> dict | None:
        """The node type's static description for the frontend palette and
        the SDK: identity, config schema, ``ui_config`` and the class-level
        port lists (instance ports come from :meth:`ports_of`)."""
        cls = self._types.get(node_type)
        if cls is None:
            return None
        return {
            "type": cls.node_type,
            "display_name": cls.display_name,
            "description": getattr(cls, "description", ""),
            "category": cls.category,
            "icon": getattr(cls, "icon", ""),
            "kind": getattr(cls, "kind", "block"),
            "ports_mode": getattr(cls, "ports_mode", "mirror"),
            "config_schema": getattr(cls, "config_schema", {}),
            "default_config": getattr(cls, "default_config", {}),
            "ui_config": ui_config_to_dict(getattr(cls, "ui_config", None)),
            "input_ports": [_port_to_dict(p) for p in getattr(cls, "input_ports", [])],
            "output_ports": [_port_to_dict(p) for p in getattr(cls, "output_ports", [])],
        }

    def schemas(self) -> list[dict]:
        """:meth:`schema` for every registered type, in registration order."""
        return [s for s in (self.schema(nt) for nt in self._types) if s is not None]


def ports_for(cls: type, config: dict | None = None) -> Ports:
    """The port surface of an instance of ``cls`` under ``config`` — the
    one rule, usable on a class that is not (yet) registered (the SDK's
    workspace scan). ``_resolve_ports(config)`` when the class has one and
    it answers; the class-level lists otherwise."""
    resolver = getattr(cls, "_resolve_ports", None)
    if callable(resolver):
        try:
            ins, outs = resolver(config or {})
            return Ports(list(ins), list(outs))
        except Exception:
            log.warning(
                "%s._resolve_ports raised; using class-level ports", cls.__name__, exc_info=True
            )
    return Ports(list(getattr(cls, "input_ports", [])), list(getattr(cls, "output_ports", [])))


def _port_to_dict(p: PortDef) -> dict:
    return {
        "name": p.name,
        "wire_type": p.wire_type,
        "description": p.description,
        "optional": p.optional,
    }
