"""Entry point: ``python -m mcp_server``.

Bring up the backend (spawn or borrow), construct the FastMCP server,
run on stdio. Cleanup is handled by ``BackendManager``'s atexit +
SIGTERM/SIGINT hooks.
"""

from __future__ import annotations

import logging
import sys

from .backend_manager import BackendManager
from .server import build_server


def main() -> None:
    # Log to stderr — stdout is reserved for MCP JSON-RPC framing.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    backend = BackendManager()
    backend_url = backend.start()

    mcp = build_server(backend_url)
    # FastMCP.run() defaults to stdio when no transport is given.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
