"""Human-performance test API — interactive human driver over env_habitat.

Thin shell over ``services.human_runner.HumanRunner``. The browser owns the
control loop: start the env, load an episode, send one keypress = one discrete
action, then STOP to evaluate. Frames ride back inline as base64 PNG (data
URIs) so the loop stays a single request per action; the persisted trajectory
+ metrics live under ``outputs/human/{split}/`` (see HumanRunner).
"""

from __future__ import annotations

import asyncio

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...state import get_services

router = APIRouter()

# Errors the interactive endpoints turn into a clean 409 (recoverable — the UI
# shows the message and the user retries / restarts the session) rather than an
# opaque 500: bad state / bad input (RuntimeError, ValueError) and a dropped or
# failed call to the env auto_host (requests.RequestException).
_HANDLED = (RuntimeError, ValueError, requests.exceptions.RequestException)


class StartServerRequest(BaseModel):
    split: str = "rand100"


class LoadRequest(BaseModel):
    rgb_resolution: int = 512


class StepRequest(BaseModel):
    action: int  # 1 = FORWARD, 2 = TURN_LEFT, 3 = TURN_RIGHT


def _runner():
    runner = getattr(get_services(), "human_runner", None)
    if runner is None:
        raise HTTPException(503, "human runner not initialized")
    return runner


@router.post("/start-server")
async def start_server(req: StartServerRequest) -> dict:
    """Spawn (or return the already-live) env_habitat auto_host. Non-blocking:
    poll /server-status until state == 'ready' (~30 s cold scene load)."""
    return await asyncio.to_thread(_runner().start_server, req.split)


@router.post("/stop-server")
async def stop_server() -> dict:
    return await asyncio.to_thread(_runner().stop_server)


@router.get("/server-status")
async def server_status() -> dict:
    return _runner().server_status()


@router.post("/episode/{index}/load")
async def load_episode(index: int, req: LoadRequest) -> dict:
    try:
        return await asyncio.to_thread(
            _runner().load_episode, index, req.rgb_resolution
        )
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/step")
async def step(req: StepRequest) -> dict:
    try:
        return await asyncio.to_thread(_runner().step, req.action)
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/stop")
async def stop() -> dict:
    try:
        return await asyncio.to_thread(_runner().stop)
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/status")
async def status(split: str = "rand100") -> dict:
    """Per-episode tested/success records + aggregate (from summary.json)."""
    return _runner().status(split)
