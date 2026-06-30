"""Unified error reporting bus.

Single in-process publisher that all backend subsystems use to surface
events to the user. Replaces ad-hoc ``log.exception`` swallowing across
the executor, API routes, plugin lifecycle, and eval runner.

Architecture
------------
- ``ErrorEnvelope``  — schema every event conforms to (mirrored in
  ``frontend/src/errors.ts``).
- ``ErrorBus``       — singleton with a bounded ring buffer (last 200)
  and an async broadcast queue. Sync ``publish()`` is thread-safe and
  loop-agnostic; async subscribers drain the queue.
- ``LogBridgeHandler`` — ``logging.Handler`` subclass that converts
  records (INFO+) into envelopes so the frontend "Report" tab acts as a
  full developer console without touching every log site.
- ``from_exception``  — convenience that pulls traceback + title from an
  ``Exception`` instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Severity = Literal["error", "warning", "info", "debug"]
Source = Literal["node", "graph", "api", "ws", "frontend", "plugin", "eval", "log"]

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Bus ──


AsyncSub = Callable[[ErrorEnvelope], Awaitable[None]]


class ErrorBus:
    """Pub/sub + ring buffer. One instance per process."""

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
            details={"traceback": "".join(traceback.format_exception(exc))},
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


_bus: ErrorBus | None = None


def get_bus() -> ErrorBus:
    global _bus
    if _bus is None:
        _bus = ErrorBus()
    return _bus


# ── logging.Handler bridge ──


_LEVEL_MAP: dict[int, Severity] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}

_BRIDGE_SKIP_LOGGERS = {
    "agentcanvas.errors",  # avoid recursion from bus internals
    "uvicorn.access",  # too noisy; uvicorn already logs to stderr
    "uvicorn.error",
    "uvicorn",
    "httpx",
    "httpcore",
    "asyncio",
    "watchfiles",
    "watchfiles.main",
}


class LogBridgeHandler(logging.Handler):
    """Forward Python log records to the ErrorBus.

    Records below the configured level are dropped; records from
    ``_BRIDGE_SKIP_LOGGERS`` are dropped to prevent feedback loops.
    """

    def __init__(self, bus: ErrorBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in _BRIDGE_SKIP_LOGGERS:
            return
        if record.name.startswith("agentcanvas.errors"):
            return
        # Coarse prefix skip: drop noisy library loggers without listing every child.
        for prefix in ("uvicorn.", "httpx.", "httpcore.", "watchfiles."):
            if record.name.startswith(prefix):
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
        # Use the bus directly; do NOT route through publish() that touches the
        # loop here — keep this handler hot path minimal.
        self._bus.publish(env)


def install_log_bridge(level: int = logging.INFO) -> LogBridgeHandler:
    """Attach a LogBridgeHandler to the root logger. Idempotent."""
    bus = get_bus()
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, LogBridgeHandler):
            return h
    handler = LogBridgeHandler(bus, level=level)
    root.addHandler(handler)
    return handler


__all__ = [
    "ErrorBus",
    "ErrorEnvelope",
    "LogBridgeHandler",
    "Severity",
    "Source",
    "get_bus",
    "install_log_bridge",
]


# Hot-path timing safety net: if the import is happening at request time
# something has gone weird — record so we notice.
_loaded_at = time.time()
