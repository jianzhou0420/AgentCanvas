"""Read-only snapshot of currently-loaded shared nodesets.

Used as a fallback path: under normal operation the JobScheduler
already writes ``shared_urls.json`` into the run dir at submit time,
so the run subprocess does not need to call this endpoint. This
exists for tests and for run subprocesses that started without a
populated ``shared_urls.json`` (disaster recovery).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...state import get_services

router = APIRouter()


class SnapshotRequest(BaseModel):
    names: list[str] | None = None  # None = all loaded


class SnapshotResponse(BaseModel):
    urls: dict[str, str]


@router.post("/snapshot", response_model=SnapshotResponse)
def snapshot(req: SnapshotRequest) -> SnapshotResponse:
    registry = get_services().workspace_component_registry
    requested = set(req.names) if req.names else None

    urls: dict[str, str] = {}
    for name, server in registry._auto_servers.items():
        bare_name = name.split("#")[0]
        if requested is not None and bare_name not in requested:
            continue
        url = getattr(server, "url", None)
        if url:
            # When tagged, prefer the bare-name entry if it exists; else
            # take the first tag we encounter.
            urls.setdefault(bare_name, url)
    return SnapshotResponse(urls=urls)
