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
    urls = registry.server_urls()
    if req.names:
        requested = set(req.names)
        urls = {name: url for name, url in urls.items() if name in requested}
    return SnapshotResponse(urls=urls)
