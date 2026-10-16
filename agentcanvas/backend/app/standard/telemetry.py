"""Per-node-firing accounting ContextVars — shared executor/producer seam.

The graph executor (and the server-side ``invoke``) set each ContextVar to
a fresh accumulator around every node firing; producer modules accumulate
into whichever bucket is active and the firing's owner reads it back into
the node's log entry. ContextVars so concurrent firings don't
cross-contaminate.

They live here — not in their producer modules (``app.llm.complete``,
``app.server.serialization``) — so the executor can import them without
dragging litellm / numpy into a pure-local run; the producers re-export them.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Written by every completion round trip (app.llm.complete).
_current_node_usage: ContextVar[dict | None] = ContextVar("_current_node_usage", default=None)

# Written by the server-proxy round-trip codec (app.server.serialization).
_current_node_transport: ContextVar[dict | None] = ContextVar(
    "_current_node_transport", default=None
)

# Written by ``BaseCanvasNode._self_log``. When set, a node's voluntary log
# entries land here instead of on the node instance — the server-side
# ``invoke`` uses it so a long-lived hosted node gets a fresh buffer per
# call (and concurrent calls never interleave); the executor leaves it
# unset and reads the per-firing instance's own buffer.
_current_node_log: ContextVar[list | None] = ContextVar("_current_node_log", default=None)


def new_usage_bucket() -> dict[str, Any]:
    """The per-firing LLM usage accumulator, in its canonical shape."""
    return {
        "calls": 0,
        "model": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "usd_cost": 0.0,
    }


def merge_usage(into: dict[str, Any], other: dict[str, Any] | None) -> None:
    """Fold one usage bucket into another (a hosted node's usage, carried
    back over the wire, into the proxy firing's bucket)."""
    if not other:
        return
    for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
        into[key] = int(into.get(key, 0)) + int(other.get(key, 0) or 0)
    into["usd_cost"] = float(into.get("usd_cost", 0.0)) + float(other.get("usd_cost", 0.0) or 0.0)
    if other.get("model") and not into.get("model"):
        into["model"] = other["model"]
