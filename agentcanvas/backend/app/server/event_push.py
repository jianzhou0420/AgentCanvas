"""Subprocess → executor event push (Move 3, roadmap #54).

A server-mode nodeset runs in its own subprocess, so its node logs and handler
errors never reach the canvas — a handler exception is demoted to a swallowed
``{"error": ...}`` value (see ``proxy.py``). This module forwards structured
events back to the executor's ``/api/internal/events`` endpoint, which republishes
them on the ErrorBus so server-node logs/errors surface on the canvas exactly
like local-node ones.

Two producers:
- **explicit** — ``emit_event(...)`` from the ``/call`` error path (first-class
  errors with node/execution scope), and any node that wants to surface an event;
- **implicit** — :class:`SubprocessLogBridge`, a logging.Handler installed by the
  auto-host that forwards WARNING+ records.

Transport is JSON (events are lightweight text). Events are buffered and flushed
on a size threshold, on each ``/call`` response, and immediately on error. The
HTTP POST is synchronous (mirrors the container path) — acceptable because the
buffer is usually drained at call boundaries; a torrent of logs only costs a
batched round-trip every ``_MAX_BUFFER`` records.

Discovery: the executor base URL comes from ``AGENTCANVAS_EXECUTOR_URL`` (set on
each subprocess by ``registry._load_nodeset_as_server``). When unset (a
standalone ServerApp, or local mode), every call here is a no-op.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("agentcanvas.event-push")

_BUFFER: list = []
_LOCK = threading.Lock()
_MAX_BUFFER = 50
_FLUSH_TIMEOUT = 3.0
_local = threading.local()
# Never forward our own / the HTTP stack's logs — that would recurse.
_SKIP_LOGGERS = ("agentcanvas.event-push", "httpx", "httpcore", "urllib3")

_installed = False
_nodeset_name: str | None = None


def _executor_url() -> str | None:
    return os.environ.get("AGENTCANVAS_EXECUTOR_URL")


def emit_event(
    severity: str,
    message: str,
    *,
    code: str | None = None,
    node_id: str | None = None,
    step: int | None = None,
    nodeset: str | None = None,
    execution_id: str | None = None,
    details: dict | None = None,
) -> None:
    """Buffer one event for push to the executor. No-op without an executor URL."""
    if not _executor_url():
        return
    with _LOCK:
        _BUFFER.append(
            {
                "severity": severity,
                "message": str(message),
                "code": code,
                "node_id": node_id,
                "step": step,
                "nodeset": nodeset if nodeset is not None else _nodeset_name,
                "execution_id": execution_id,
                "details": details or {},
            }
        )
        due = len(_BUFFER) >= _MAX_BUFFER
    if due:
        flush()


def flush() -> None:
    """Send buffered events to the executor. Cheap (lock + len) when empty."""
    url = _executor_url()
    if not url:
        return
    if getattr(_local, "in_flush", False):  # reentrancy guard (httpx logs etc.)
        return
    with _LOCK:
        if not _BUFFER:
            return
        batch = _BUFFER[:]
        _BUFFER.clear()
    _local.in_flush = True
    try:
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        with httpx.Client(timeout=_FLUSH_TIMEOUT, **loopback_httpx_kwargs()) as client:
            client.post(f"{url.rstrip('/')}/api/internal/events", json={"events": batch})
    except Exception:
        # Observability must never break execution — drop on failure.
        log.debug("event push failed (dropped %d events)", len(batch), exc_info=True)
    finally:
        _local.in_flush = False


_LEVEL_TO_SEV = {
    logging.CRITICAL: "error",
    logging.ERROR: "error",
    logging.WARNING: "warning",
    logging.INFO: "info",
    logging.DEBUG: "debug",
}


class SubprocessLogBridge(logging.Handler):
    """Forward subprocess logging records to the executor as events."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_SKIP_LOGGERS):
            return
        try:
            sev = _LEVEL_TO_SEV.get(record.levelno, "info")
            emit_event(
                sev,
                record.getMessage(),
                code=f"SUBPROC_LOG_{record.levelname}",
            )
        except Exception:
            pass


def install_log_bridge(level: int = logging.WARNING, nodeset: str | None = None) -> bool:
    """Install the WARNING+ logging bridge once (no-op without an executor URL)."""
    global _installed, _nodeset_name
    if _installed or not _executor_url():
        return False
    _nodeset_name = nodeset
    logging.getLogger().addHandler(SubprocessLogBridge(level=level))
    _installed = True
    return True
