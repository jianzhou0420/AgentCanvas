"""AgentCanvas Server SDK — server-mode nodesets as canvas nodes.

Four classes serve different roles:

- :class:`ServerApp` — server-side: builds a FastAPI service that speaks
  the manifest protocol.  Used by server authors.
- :class:`AutoServerApp` — server-side: auto-generates a ServerApp from
  any BaseNodeSet by introspecting node ports (ADR-009).
- :class:`BaseServer` — framework-side: launches, monitors, and communicates
  with an external server process.
- :class:`ServerNodeSet` — combines BaseNodeSet + BaseServer: a loadable
  group of canvas-node tools backed by an external server.

All re-exports are lazy (PEP 562): importing a submodule such as
``app.server.serialization`` must not drag FastAPI in — a pip-installed
Graph SDK replays cassettes with only the ``server`` extra (httpx / msgpack /
numpy) installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .auto_server_app import AutoServerApp
    from .base_server import BaseServer
    from .manifest import FunctionSchema, PortSchema, ServerManifest
    from .nodeset import ServerNodeSet
    from .server_app import ServerApp, ServerFunction

__all__ = [
    "AutoServerApp",
    "BaseServer",
    "FunctionSchema",
    "PortSchema",
    "ServerApp",
    "ServerFunction",
    "ServerManifest",
    "ServerNodeSet",
]

_EXPORTS = {
    "AutoServerApp": ".auto_server_app",
    "BaseServer": ".base_server",
    "FunctionSchema": ".manifest",
    "PortSchema": ".manifest",
    "ServerApp": ".server_app",
    "ServerFunction": ".server_app",
    "ServerManifest": ".manifest",
    "ServerNodeSet": ".nodeset",
}


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(submodule, __name__), name)
