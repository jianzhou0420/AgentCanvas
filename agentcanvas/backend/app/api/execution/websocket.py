"""WebSocket endpoint — thin handler, broadcast logic lives in state.py."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...state import register_ws_client, unregister_ws_client, ws_client_count

log = logging.getLogger("agentcanvas.ws")
router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    register_ws_client(ws)
    log.info(f"WebSocket connected. Total: {ws_client_count()}")
    try:
        await asyncio.gather(
            _handle_client(ws),
        )
    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        unregister_ws_client(ws)
        log.info(f"WebSocket cleaned up. Remaining: {ws_client_count()}")


async def _handle_client(ws: WebSocket) -> None:
    """Handle messages from the client (ping/pong)."""
    while True:
        try:
            data = await ws.receive_json()
            msg_type = data.get("type", "")
            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
            else:
                log.debug(f"Unknown client message: {msg_type}")
        except WebSocketDisconnect:
            raise
        except Exception:
            break
