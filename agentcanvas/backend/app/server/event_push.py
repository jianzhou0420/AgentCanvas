"""The subprocess sink: envelopes pushed to the parent backend (Move 3, #54).

A server-mode nodeset (auto_host) and the eval runner each run in their own
subprocess, so their node failures and ``logging`` records would never reach
the canvas. :class:`PushSink` is the ``EventSink`` adapter for those
processes: it buffers envelopes and POSTs them to the parent's
``/api/internal/events`` (``api/execution/internal_events.py``), which
republishes each one on the parent's ``ErrorBus`` — the same envelope the
executor's local-node failures use, so the Report tab shows both alike.

Producers are the ordinary ones — ``get_bus().from_exception(...)`` from the
``/call`` error path or the eval subprocess's executor, and the one
``LogBridgeHandler`` (WARNING+ here). Delivery: an ``error`` envelope is
flushed at once; the rest go on the next flush — at every ``/call`` boundary
(``invoke``), at every episode boundary (the eval runner), or when the
buffer fills. The POST is synchronous and short (loopback, 3 s timeout);
a failed push is dropped — observability must never break execution.

Discovery: the parent URL is ``AGENTCANVAS_EXECUTOR_URL`` (set on each
auto_host subprocess by the registry) or the eval subprocess's
``--backend-url``. Without one, :func:`install_push_sink` installs nothing
and the process keeps its in-process bus. Python 3.8 subset: this module
runs under ``server_python``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional

from ..errors import ErrorEnvelope, EventSink, install_log_bridge, set_sink

log = logging.getLogger("agentcanvas.event-push")

_MAX_BUFFER = 50
_FLUSH_TIMEOUT = 3.0


class PushSink(EventSink):
    """Buffer envelopes, POST them to the parent in batches."""

    def __init__(self, executor_url: str, nodeset: Optional[str] = None) -> None:
        self.executor_url = executor_url.rstrip("/")
        self.nodeset = nodeset
        self._buffer: List[dict] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def publish(self, env: ErrorEnvelope) -> None:
        """Stamp the subprocess's identity on the envelope and buffer it;
        an ``error`` envelope is pushed at once."""
        scope = env.scope
        scope.setdefault("origin", "subprocess")
        if self.nodeset is not None:
            scope.setdefault("nodeset", self.nodeset)
        with self._lock:
            self._buffer.append(env.to_dict())
            due = len(self._buffer) >= _MAX_BUFFER
        if due or env.severity == "error":
            self.flush()

    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> None:
        """Send buffered envelopes to the parent. Cheap when empty; a failed
        POST drops the batch and never raises."""
        if getattr(self._local, "in_flush", False):  # reentrancy guard (httpx logs etc.)
            return
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()
        self._local.in_flush = True
        try:
            import httpx

            from ._loopback_proxy import loopback_httpx_kwargs

            with httpx.Client(timeout=_FLUSH_TIMEOUT, **loopback_httpx_kwargs()) as client:
                client.post(f"{self.executor_url}/api/internal/events", json={"events": batch})
        except Exception:
            log.debug("event push failed (dropped %d events)", len(batch), exc_info=True)
        finally:
            self._local.in_flush = False


def install_push_sink(
    executor_url: Optional[str],
    *,
    nodeset: Optional[str] = None,
    log_level: int = logging.WARNING,
) -> Optional[PushSink]:
    """Make this process publish to the parent at ``executor_url``: install
    the :class:`PushSink` as the process's sink and bridge ``log_level``+
    records into it. Returns ``None`` (nothing installed) without a URL."""
    if not executor_url:
        return None
    sink = PushSink(executor_url, nodeset=nodeset)
    set_sink(sink)
    install_log_bridge(level=log_level, sink=sink)
    return sink


__all__: List[Any] = ["PushSink", "install_push_sink"]
