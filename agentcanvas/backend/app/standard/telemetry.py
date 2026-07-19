"""Per-node-firing accounting ContextVars — shared executor/producer seam.

The graph executor sets each ContextVar to a fresh accumulator dict around
every ``BaseCanvasNode.execute()``; producer modules accumulate into whichever
bucket is active and the executor reads it back into the node's log entry.
ContextVars so concurrent firings don't cross-contaminate.

They live here — not in their producer modules (``app.llm.call``,
``app.server.serialization``) — so the executor can import them without
dragging litellm / numpy into a pure-local run; the producers re-export them.
"""

from __future__ import annotations

from contextvars import ContextVar

# Written by llm_complete / vlm_complete (app.llm.call).
_current_node_usage: ContextVar[dict | None] = ContextVar(
    "_current_node_usage", default=None
)

# Written by the server-proxy round-trip codec (app.server.serialization).
_current_node_transport: ContextVar[dict | None] = ContextVar(
    "_current_node_transport", default=None
)
