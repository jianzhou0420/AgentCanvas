"""Generic env panel REST API.

Replaces the env-specific ``api/canvas/env.py``. One router serves every
nodeset that registered a ``BaseEnvPanel`` subclass — the env_habitat
panel, future MP3D panel, anything that wants a fixed UI panel.

Endpoints (all under ``/api/env-panels``):

- ``GET  /``                          — list registered env panels (schema for picker)
- ``GET  /{name}/state``              — initial state from ``on_load()``
- ``GET  /{name}/options/{field}``    — populate dynamic select options
- ``POST /{name}/field/{field}``      — apply a field change (body: ``{value}``)
- ``POST /{name}/action/{action}``    — invoke an action button (body: ``{params}``)

POST endpoints are guarded by ``ExecutionGuard`` for actions whose
``side_effect`` is ``run_start`` (i.e. would mutate env state mid-run);
``run_pause`` / ``run_stop`` actions are allowed during execution because
they are intended to interrupt it.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...components.env_panel import (
    BaseEnvPanel,
    get_env_panel,
    list_env_panels,
)
from ...state import ExecutionGuard, ExecutionMode

log = logging.getLogger("agentcanvas.env_panel_api")

router = APIRouter()


# ── Request models ──


class FieldChangeRequest(BaseModel):
    value: Any


class ActionRequest(BaseModel):
    params: dict[str, Any] = {}


# ── Helpers ──


def _get_or_404(name: str) -> BaseEnvPanel:
    panel = get_env_panel(name)
    if panel is None:
        available = [c.name for c in list_env_panels()]
        raise HTTPException(
            status_code=404,
            detail=f"No env panel '{name}'. Available: {available}",
        )
    return panel


def _require_idle() -> None:
    """Reject the call if a canvas/eval execution is currently active."""
    state = ExecutionGuard.current()
    if state["mode"] != ExecutionMode.idle.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot perform this action while {state['mode']} execution is active. "
                "Stop the current execution first."
            ),
        )


def _action_side_effect(panel: BaseEnvPanel, action_name: str) -> str:
    for a in panel.actions:
        if a.name == action_name:
            return a.side_effect
    return "none"


# ── GET endpoints ──


@router.get("")
async def list_all() -> list[dict]:
    """List every registered env panel with its display schema."""
    return [
        {
            "name": panel.info().name,
            "display_name": panel.info().display_name,
            "fields": panel.info().fields,
            "actions": panel.info().actions,
        }
        for panel in list_env_panels()
    ]


@router.get("/{name}/state")
async def get_state(name: str) -> dict:
    panel = _get_or_404(name)
    return await panel.on_load()


@router.get("/{name}/options/{field}")
async def get_options(name: str, field: str) -> list[dict]:
    panel = _get_or_404(name)
    return await panel.get_options(field)


# ── POST endpoints ──


def _forward_signal_if_any(result: dict) -> None:
    """If an action result carries a ``signal`` side_effect, broadcast it
    to the running executor's state containers.

    State containers declare their lifetime via ``reset_on`` signal
    subscriptions (see ``state_containers.py``).  This function is the
    single hand-off point between the env panel API and the live
    executor's signal bus.  If no run is active, the call is a safe
    no-op — the next run will rebuild fresh containers anyway.
    """
    if result.get("side_effect") != "signal":
        return
    signal_name = result.get("signal_name")
    if not signal_name:
        log.warning("env panel returned side_effect=signal without signal_name")
        return
    signal_payload = result.get("signal_payload") or {}
    # Lazy import to avoid circular deps.
    from ...agent_loop.loop_runner import get_loop_runner

    runner = get_loop_runner()
    executor = getattr(runner, "_executor", None)
    if executor is None or not getattr(executor, "containers", None):
        log.debug(
            "signal %s: no live executor with containers — skipping",
            signal_name,
        )
        return
    try:
        executor.broadcast_signal(signal_name, signal_payload)
        log.info("forwarded signal '%s' to %d containers", signal_name, len(executor.containers))
    except Exception:
        log.exception("failed to broadcast signal '%s'", signal_name)


@router.post("/{name}/field/{field}")
async def set_field(name: str, field: str, req: FieldChangeRequest) -> dict:
    panel = _get_or_404(name)
    _require_idle()  # field changes mutate env state — block during runs
    log.info("env panel %s field %s = %r", name, field, req.value)
    result = await panel.on_field_change(field, req.value)
    _forward_signal_if_any(result)
    return result


@router.post("/{name}/action/{action}")
async def invoke_action(name: str, action: str, req: ActionRequest) -> dict:
    panel = _get_or_404(name)
    side_effect = _action_side_effect(panel, action)
    # Only block actions that would mutate env state. Pause/stop must work
    # during execution; "none" / "signal" actions (e.g. reset) are caller's choice.
    if side_effect in ("run_start", "none", "signal"):
        _require_idle()
    log.info("env panel %s action %s (side_effect=%s)", name, action, side_effect)
    result = await panel.on_action(action, req.params)
    _forward_signal_if_any(result)
    return result
