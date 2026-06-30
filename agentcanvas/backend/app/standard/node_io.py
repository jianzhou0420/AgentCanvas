"""Standard node I/O interface — auto-generated from BaseCanvasNode subclasses.

Port declarations come from each node class's ``input_ports`` and
``output_ports`` ClassVars.  The ``NODE_IO_SCHEMA`` and ``REQUIRED_INPUTS``
dicts are built lazily from the ``NODE_HANDLERS`` registry so they always
stay in sync with the actual handler code.
"""

from __future__ import annotations

from typing import Any

# ── NodePort (kept for backward compat with existing consumers) ──


class NodePort:
    """Declares one input or output port on a node."""

    def __init__(self, name: str, wire_type: str, description: str = "", optional: bool = False):
        self.name = name
        self.wire_type = wire_type
        self.description = description
        self.optional = optional

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wire_type": self.wire_type,
            "description": self.description,
            "optional": self.optional,
        }


# ── Auto-generated from BaseCanvasNode port declarations ──


def _build_schema() -> tuple[dict[str, dict[str, list[NodePort]]], dict[str, list[str]]]:
    """Build NODE_IO_SCHEMA and REQUIRED_INPUTS from NODE_HANDLERS."""
    from ..agent_loop.builtin_nodes import NODE_HANDLERS

    schema: dict[str, dict[str, list[NodePort]]] = {}
    required: dict[str, list[str]] = {}
    seen: set = set()

    for node_type, cls in NODE_HANDLERS.items():
        # Avoid duplicating for aliases (multiple keys → same class)
        canon = cls.node_type
        if canon not in seen:
            seen.add(canon)
            schema[canon] = {
                "inputs": [
                    NodePort(p.name, p.wire_type, p.description, p.optional)
                    for p in cls.input_ports
                ],
                "outputs": [
                    NodePort(p.name, p.wire_type, p.description, p.optional)
                    for p in cls.output_ports
                ],
            }
            required[canon] = [p.name for p in cls.input_ports if not p.optional]

        # Also register under the alias key (for legacy lookups)
        if node_type != canon:
            schema[node_type] = schema[canon]
            required[node_type] = required[canon]

    return schema, required


# Lazy initialization — populated on first access
_schema_cache: dict[str, dict[str, list[NodePort]]] | None = None
_required_cache: dict[str, list[str]] | None = None


def _ensure_built() -> None:
    global _schema_cache, _required_cache
    if _schema_cache is None:
        _schema_cache, _required_cache = _build_schema()


@property
def _node_io_schema() -> dict[str, dict[str, list[NodePort]]]:
    _ensure_built()
    return _schema_cache  # type: ignore


# ── Public API (same interface as before) ──


def get_node_io_schema() -> dict[str, dict[str, list[NodePort]]]:
    """Return the full NODE_IO_SCHEMA dict."""
    _ensure_built()
    return _schema_cache  # type: ignore


def get_required_inputs_map() -> dict[str, list[str]]:
    """Return the full REQUIRED_INPUTS dict."""
    _ensure_built()
    return _required_cache  # type: ignore


def get_node_io(node_type: str) -> dict[str, list[NodePort]]:
    """Get the I/O schema for a node type."""
    _ensure_built()
    return _schema_cache.get(node_type, {"inputs": [], "outputs": []})  # type: ignore


def get_required_inputs(node_type: str) -> list[str]:
    """Get required input port names for a node type."""
    _ensure_built()
    return _required_cache.get(node_type, [])  # type: ignore


def invalidate_cache() -> None:
    """Clear the cached schema (call after registering new node types)."""
    global _schema_cache, _required_cache
    _schema_cache = None
    _required_cache = None


# ── Backward-compatible module-level dicts ──
# These are lazy proxies for code that reads NODE_IO_SCHEMA / REQUIRED_INPUTS directly.


class _LazyDict(dict):
    """Dict that auto-populates from BaseCanvasNode classes on first access."""

    def __init__(self, builder_key: str):
        super().__init__()
        self._builder_key = builder_key
        self._populated = False

    def _populate(self) -> None:
        if not self._populated:
            _ensure_built()
            if self._builder_key == "schema":
                self.update(_schema_cache)  # type: ignore
            else:
                self.update(_required_cache)  # type: ignore
            self._populated = True

    def __getitem__(self, key: str) -> Any:
        self._populate()
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._populate()
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self._populate()
        return super().__contains__(key)

    def items(self):
        self._populate()
        return super().items()

    def values(self):
        self._populate()
        return super().values()

    def keys(self):
        self._populate()
        return super().keys()


NODE_IO_SCHEMA: dict[str, dict[str, list[NodePort]]] = _LazyDict("schema")  # type: ignore
REQUIRED_INPUTS: dict[str, list[str]] = _LazyDict("required")  # type: ignore
