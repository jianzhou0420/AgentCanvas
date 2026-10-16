"""Internal reverse-channel: envelopes pushed by subprocesses (Move 3, #54).

The parent backend receives batches from ``server.event_push.PushSink`` and
republishes each envelope on its ``ErrorBus`` — the same ``ErrorEnvelope``
that local events use, so a subprocess node failure, an eval episode's
error or a forwarded WARNING log reaches the Report tab like a local one.

JSON transport (events are lightweight text). Mounted under ``/api/internal``.
NOT a public API.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ...errors import ErrorEnvelope, get_bus

log = logging.getLogger("agentcanvas.internal-events")
router = APIRouter()


class PushBatch(BaseModel):
    """A list of envelope dicts (``ErrorEnvelope.to_dict``)."""

    events: list[dict[str, Any]] = []


@router.post("/events")
async def push_events(batch: PushBatch) -> dict:
    bus = get_bus()
    for raw in batch.events:
        env = ErrorEnvelope.from_dict(raw)
        env.scope.setdefault("origin", "subprocess")
        bus.publish(env)
    return {"ok": True, "count": len(batch.events)}
