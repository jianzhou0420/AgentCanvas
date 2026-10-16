"""Error / log events: one envelope, one sink interface, two adapters.

Every subsystem that surfaces an event to the user — the executor, API
routes, the plugin lifecycle, the eval runner, a server-mode nodeset's
handler, a ``logging`` record — publishes an :class:`ErrorEnvelope` to the
process's :class:`EventSink` (``get_bus()``). Which sink that is depends on
the process:

- :class:`ErrorBus` — the parent backend's adapter: a bounded ring buffer
  (last 200, backfilled to the Report tab on reconnect) plus an async
  broadcast queue the WebSocket subscriber drains.
- ``server.event_push.PushSink`` — the subprocess adapter (auto_host
  nodeset servers, the eval subprocess): buffers envelopes and POSTs them
  to the parent's ``/api/internal/events``, which republishes them on the
  parent's ErrorBus. One envelope on both sides, so a subprocess node
  failure or an eval episode's error reaches the Report tab exactly like
  a local one.

:class:`LogBridgeHandler` is the one ``logging.Handler`` that turns records
into envelopes, over whichever sink the process has (INFO+ in the parent,
WARNING+ in a subprocess).

The envelope schema is mirrored in ``frontend/src/errors.ts``. This module
is imported by Python 3.8 subprocesses: runtime code stays on the 3.8
subset (annotations are strings under ``from __future__``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

Severity = Literal["error", "warning", "info", "debug"]
Source = Literal["node", "graph", "api", "ws", "frontend", "plugin", "eval", "log"]

_SEVERITIES = ("error", "warning", "info", "debug")
_RING_CAP = 200
_QUEUE_CAP = 500


@dataclass
class ErrorEnvelope:
    """Canonical schema for every event surfaced to the user."""

    id: str
    ts: str  # ISO-8601 UTC
    severity: Severity
    source: Source
    code: str
    title: str
    message: str
    scope: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ErrorEnvelope:
        """The wire form back into an envelope: an unknown severity reads as
        ``info``, a missing id / timestamp is minted here."""
        severity = d.get("severity")
        return cls(
            id=str(d.get("id") or _new_id()),
            ts=str(d.get("ts") or _now_iso()),
            severity=severity if severity in _SEVERITIES else "info",  # type: ignore[arg-type]
            source=d.get("source") or "node",
            code=str(d.get("code") or "EVENT"),
            title=str(d.get("title") or d.get("message") or d.get("code") or "event")[:80],
            message=str(d.get("message") or ""),
            scope=dict(d.get("scope") or {}),
            details=dict(d.get("details") or {}),
            hint=d.get("hint"),
        )


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    AsyncSub = Callable[[ErrorEnvelope], Awaitable[None]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _format_exc(exc: BaseException) -> str:
    # Three-argument form: the one-argument form is Python 3.10+.
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ── Sink interface ──


class EventSink(ABC):
    """Where a process's events go. ``publish`` is the one abstract method;
    ``from_exception`` / ``emit`` build the envelope and publish it;
    ``flush`` is for adapters that batch (the in-process bus has nothing
    to flush)."""

    @abstractmethod
    def publish(self, env: ErrorEnvelope) -> None:
        """Accept one envelope. Thread-safe; must never raise into the caller."""

    def flush(self) -> None:
        """Deliver anything buffered. No-op by default."""
        return None

    def from_exception(
        self,
        exc: BaseException,
        *,
        source: Source,
        code: str,
        scope: dict[str, Any] | None = None,
        hint: str | None = None,
        severity: Severity = "error",
        title: str | None = None,
    ) -> ErrorEnvelope:
        env = ErrorEnvelope(
            id=_new_id(),
            ts=_now_iso(),
            severity=severity,
            source=source,
            code=code,
            title=title or f"{type(exc).__name__}: {str(exc)[:80]}",
            message=str(exc),
            scope=dict(scope or {}),
            details={"traceback": _format_exc(exc)},
            hint=hint,
        )
        self.publish(env)
        return env

    def emit(
        self,
        *,
        severity: Severity,
        source: Source,
        code: str,
        title: str,
        message: str = "",
        scope: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> ErrorEnvelope:
        """Publish a non-exception envelope (warnings, info, custom events)."""
        env = ErrorEnvelope(
            id=_new_id(),
            ts=_now_iso(),
            severity=severity,
            source=source,
            code=code,
            title=title,
            message=message,
            scope=dict(scope or {}),
            details=dict(details or {}),
            hint=hint,
        )
        self.publish(env)
        return env


# ── Bus: the in-process adapter ──


class ErrorBus(EventSink):
    """Pub/sub + ring buffer. The parent backend's sink."""

    def __init__(self) -> None:
        self._ring: deque[ErrorEnvelope] = deque(maxlen=_RING_CAP)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[ErrorEnvelope] | None = None
        self._consumers: list[AsyncSub] = []

    # Loop binding (called once from FastAPI lifespan)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=_QUEUE_CAP)
        loop.create_task(self._dispatch_forever())

    def subscribe(self, sub: AsyncSub) -> None:
        """Register an async subscriber. Called for every published envelope."""
        self._consumers.append(sub)

    async def _dispatch_forever(self) -> None:
        assert self._queue is not None
        while True:
            env = await self._queue.get()
            for sub in list(self._consumers):
                try:
                    await sub(env)
                except Exception:
                    # Subscribers must not crash the bus. Use stdlib logger
                    # directly to avoid bridge recursion.
                    logging.getLogger("agentcanvas.errors").exception("ErrorBus subscriber failed")

    # Producer API

    def publish(self, env: ErrorEnvelope) -> None:
        """Thread-safe, loop-agnostic. Appends to ring + schedules broadcast."""
        with self._lock:
            self._ring.append(env)
        if self._loop is not None and self._queue is not None:
            # RuntimeError = loop closed (shutdown). Ring buffer still has it.
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._queue.put_nowait, env)

    # Consumer / inspection API

    def snapshot(self) -> list[ErrorEnvelope]:
        with self._lock:
            return list(self._ring)

    def clear(self) -> int:
        with self._lock:
            n = len(self._ring)
            self._ring.clear()
            return n

    def dismiss(self, env_id: str) -> bool:
        with self._lock:
            for i, env in enumerate(self._ring):
                if env.id == env_id:
                    del self._ring[i]
                    return True
            return False


_sink: EventSink | None = None


def get_bus() -> Any:
    """The process's sink — an :class:`ErrorBus` unless :func:`set_sink`
    installed another adapter (a subprocess's ``PushSink``). Typed ``Any``
    because the parent-only callers (the Report API, the lifespan) use the
    bus's ring and loop methods on it."""
    global _sink
    if _sink is None:
        _sink = ErrorBus()
    return _sink


def set_sink(sink: EventSink) -> None:
    """Install the process's sink (a subprocess, before anything publishes)."""
    global _sink
    _sink = sink


# ── logging.Handler bridge ──


_LEVEL_MAP: dict[int, Severity] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}

# Loggers never bridged: the sinks' own (recursion), and the HTTP / server
# stacks (noise, and the push adapter's own transport would recurse).
_BRIDGE_SKIP_PREFIXES = (
    "agentcanvas.errors",
    "agentcanvas.event-push",
    "uvicorn",
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "watchfiles",
)


class LogBridgeHandler(logging.Handler):
    """Forward Python log records to the process's sink as envelopes.

    The one bridge for every process: INFO+ in the parent backend, WARNING+
    in a subprocess. Records from ``_BRIDGE_SKIP_PREFIXES`` loggers are
    dropped to prevent feedback loops.
    """

    def __init__(self, sink: EventSink, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_BRIDGE_SKIP_PREFIXES):
            return
        severity = _LEVEL_MAP.get(record.levelno, "info")
        try:
            message = record.getMessage()
        except Exception:
            message = record.msg if isinstance(record.msg, str) else repr(record.msg)
        details: dict[str, Any] = {"logger": record.name}
        if record.exc_info:
            details["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        env = ErrorEnvelope(
            id=_new_id(),
            ts=_now_iso(),
            severity=severity,
            source="log",
            code=f"LOG_{record.levelname}",
            title=f"[{record.name}] {message[:80]}",
            message=message,
            scope={"logger": record.name, "fn": record.funcName, "line": record.lineno},
            details=details,
        )
        with contextlib.suppress(Exception):  # a handler never raises into logging
            self._sink.publish(env)


def install_log_bridge(
    level: int = logging.INFO, sink: EventSink | None = None
) -> LogBridgeHandler:
    """Attach a LogBridgeHandler over ``sink`` (default: the process's) to
    the root logger. Idempotent."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, LogBridgeHandler):
            return h
    handler = LogBridgeHandler(sink if sink is not None else get_bus(), level=level)
    root.addHandler(handler)
    return handler


__all__ = [
    "ErrorBus",
    "ErrorEnvelope",
    "EventSink",
    "LogBridgeHandler",
    "Severity",
    "Source",
    "get_bus",
    "install_log_bridge",
    "set_sink",
]


# Hot-path timing safety net: if the import is happening at request time
# something has gone weird — record so we notice.
_loaded_at = time.time()
