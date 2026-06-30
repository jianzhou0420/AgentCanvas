"""Internal reverse-channel: structured log/error events pushed by server-mode
subprocesses (Move 3, roadmap #54).

The executor (this process) receives batches from
``app/server/event_push.py::flush`` and republishes each on the ErrorBus, which
already fans out to the canvas over WebSocket as ``error_event``. This is what
makes a server-node's logs and (first-classed) errors visible on the canvas like
local-node events — closing the gap where a subprocess handler exception was
demoted to a swallowed ``{"error": ...}`` value.

JSON transport (events are lightweight text). Mounted under ``/api/internal``.
NOT a public API.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ...errors import get_bus

log = logging.getLogger("agentcanvas.internal-events")
router = APIRouter()

_SEVERITIES = {"error", "warning", "info", "debug"}


class PushEvent(BaseModel):
    severity: str = "info"
    message: str = ""
    code: str | None = None
    node_id: str | None = None
    step: int | None = None
    nodeset: str | None = None
    execution_id: str | None = None
    details: dict[str, Any] | None = None


class PushBatch(BaseModel):
    events: list[PushEvent] = []


@router.post("/events")
async def push_events(batch: PushBatch) -> dict:
    bus = get_bus()
    for ev in batch.events:
        severity = ev.severity if ev.severity in _SEVERITIES else "info"
        scope = {
            k: v
            for k, v in {
                "node_id": ev.node_id,
                "step": ev.step,
                "nodeset": ev.nodeset,
                "execution_id": ev.execution_id,
                "origin": "subprocess",
            }.items()
            if v is not None
        }
        title = (ev.message or ev.code or "subprocess event")[:80]
        bus.emit(
            severity=severity,  # type: ignore[arg-type]
            source="node",
            code=ev.code or "SUBPROC_LOG",
            title=title,
            message=ev.message,
            scope=scope,
            details=ev.details or {},
        )
    return {"ok": True, "count": len(batch.events)}
