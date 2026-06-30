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
"""

from __future__ import annotations

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
