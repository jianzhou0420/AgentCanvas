"""Internal reverse-channel + broker: server-mode subprocess nodes read/write
state containers regardless of which process homes them (faces A/B/C of the
cross-nodeset container-access prototype — see
docs/.../tmp/state-container-cross-nodeset-access.html §2.3).

NOT a public API. Called by ``app/server/remote_container.py::RemoteContainerProxy``.
Resolution per request:
  - container is executor-home (in the live ``GraphExecutor.containers``) →
    operate locally on the single source of truth (face B);
  - else it is sub-home (owned by a server nodeset) → the executor acts as a
    **broker** and forwards the op to that subprocess's ``/containers/{cid}/{op}``
    (faces A and C — a home node, or another subprocess, reaching it).

Transport is msgpack (``pack_body`` / ``unpack_body``) — ndarray/torch/PIL ride
raw-byte blobs, so state payloads cross without base64.

Prototype limits (deferred): no locking; cross-nodeset access to a *replicated*
nodeset's containers is not routable under ``worker_count>1`` (the home registry
resolves an untagged URL — see ``WorkspaceComponentRegistry._check_container_ownership``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from ...agent_loop.loop_runner import get_loop_runner
from ...server import serialization
from ...state import get_services

log = logging.getLogger("agentcanvas.internal-containers")
router = APIRouter()


async def _body(request: Request) -> dict:
    raw = await request.body()
    return serialization.unpack_body(raw) if raw else {}


def _respond(obj: dict) -> Response:
    return Response(
        content=serialization.pack_body(obj), media_type=serialization.MSGPACK_CONTENT_TYPE
    )


def _local_container(execution_id: str, container_id: str):
    """The executor-home container for ``container_id``, or None if it is not
    executor-home (then the caller brokers to the sub-home subprocess)."""
    runner = get_loop_runner()
    executor = getattr(runner, "_executor", None)
    if executor is None:
        return None
    live = getattr(runner, "_execution_id", None)
    containers = getattr(executor, "containers", None) or {}
    c = containers.get(container_id)
    if c is None:
        return None
    if execution_id and live and execution_id != live:
        raise HTTPException(409, f"stale execution_id {execution_id!r} (live {live!r})")
    return c


def _home_url(container_id: str) -> str:
    """Base URL of the sub-home subprocess for ``container_id``, or 404."""
    try:
        url = get_services().workspace_component_registry.get_container_home_url(container_id)
    except Exception:
        url = None
    if not url:
        raise HTTPException(404, f"no home for container {container_id!r}")
    return url


async def _forward(container_id: str, op: str, payload: dict) -> dict:
    from ...server.client import AutoHostClient, CallError

    # Async: these endpoints run in the executor's event loop; a blocking
    # client here would stall every other in-flight call while brokering.
    try:
        return await AutoHostClient(_home_url(container_id), timeout=30.0).container_op(
            container_id, op, payload
        )
    except CallError as e:
        raise HTTPException(e.status or 502, str(e)) from e


@router.post("/containers/{execution_id}/{container_id}/read")
async def container_read(execution_id: str, container_id: str, request: Request):
    body = await _body(request)
    c = _local_container(execution_id, container_id)
    if c is None:  # broker to sub-home (face A / C)
        return _respond(
            await _forward(container_id, "read", {"name": body.get("name"), "key": body.get("key")})
        )
    try:
        val = c.read(body.get("name"), key=body.get("key"))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return _respond({"value": val})


@router.post("/containers/{execution_id}/{container_id}/write")
async def container_write(execution_id: str, container_id: str, request: Request):
    body = await _body(request)
    c = _local_container(execution_id, container_id)
    if c is None:  # broker to sub-home (face A / C)
        return _respond(
            await _forward(
                container_id,
                "write",
                {"name": body.get("name"), "data": body.get("data"), "key": body.get("key")},
            )
        )
    try:
        c.write(body.get("name"), body.get("data"), key=body.get("key"))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return _respond({"ok": True})


@router.post("/containers/{execution_id}/{container_id}/evict")
async def container_evict(execution_id: str, container_id: str, request: Request):
    body = await _body(request)
    if body.get("key") is None:
        raise HTTPException(400, "evict requires 'key'")
    c = _local_container(execution_id, container_id)
    if c is None:  # broker: sub-home nodeset-wide key evict
        return _respond(await _forward(container_id, "evict", {"key": body.get("key")}))
    c.evict(body.get("key"))
    return _respond({"ok": True})
