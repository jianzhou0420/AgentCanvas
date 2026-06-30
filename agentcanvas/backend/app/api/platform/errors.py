"""Error/event endpoints — list and clear the in-memory ring buffer.

Used by the frontend Report tab on WebSocket reconnect to backfill any
events that were published while the socket was down.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...errors import get_bus

router = APIRouter()


@router.get("")
async def list_events() -> dict:
    bus = get_bus()
    return {"events": [e.to_dict() for e in bus.snapshot()]}


@router.delete("")
async def clear_events() -> dict:
    bus = get_bus()
    n = bus.clear()
    return {"cleared": n}


@router.delete("/{event_id}")
async def dismiss_event(event_id: str) -> dict:
    bus = get_bus()
    return {"dismissed": bus.dismiss(event_id)}
