"""System usage API — CPU / memory / GPU snapshots for the Monitor page.

Live machine telemetry is produced by the always-on ``ResourceSampler``
(``app/services/resource_sampler.py``); these endpoints just serve its latest
sample and short history. If the sampler isn't running (e.g. imported in a
context with no lifespan), ``/usage`` falls back to an inline snapshot so it
never 500s.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ...services.resource_sampler import get_sampler, sample_system
from ...services.system_runs import list_all_runs, run_detail

log = logging.getLogger("agentcanvas.system")
router = APIRouter()


@router.get("/usage")
def system_usage() -> dict[str, Any]:
    """Latest machine snapshot (cpu / mem / gpu + event-loop lag + ws clients)."""
    sampler = get_sampler()
    latest = sampler.latest() if sampler is not None else None
    return latest if latest is not None else sample_system()


@router.get("/history")
def system_history(n: int = 120) -> dict[str, Any]:
    """Recent resource samples (newest last) — lets the Monitor page backfill
    its Live sparklines after a refresh instead of starting from empty."""
    sampler = get_sampler()
    return {"samples": sampler.history(n) if sampler is not None else []}


@router.get("/runs")
def system_runs() -> dict[str, Any]:
    """Unified eval + canvas run list (run granularity) for the Run-view picker."""
    return {"runs": list_all_runs()}


@router.get("/runs/{run_id}")
def system_run_detail(run_id: str) -> dict[str, Any]:
    """Per-run System Log: node-timing breakdown + totals + run-local resource
    series (and the eval summary for eval runs)."""
    detail = run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return detail
