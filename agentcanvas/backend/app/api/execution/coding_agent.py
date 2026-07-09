"""Coding-Agent Monitor API — control + live-log surface for beta-coding-agent runs.

Thin shell over ``services.coding_agent_runner.CodingAgentRunner``. Live text
comes from the driver's per-episode trajectory JSONL (flushed per event, so
whole-file-with-offset polling is real-time); live images come from the
bridge's ``live_{i}/`` frame dumps.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...services.coding_agent_runner import OUTPUT_ROOT
from ...state import get_services

router = APIRouter()


class StartRequest(BaseModel):
    episodes: str = "0-9"
    split: str = "rand100"
    max_turns: int = 80
    model: str | None = None


def _runner():
    runner = getattr(get_services(), "coding_agent_runner", None)
    if runner is None:
        raise HTTPException(503, "coding-agent runner not initialized")
    return runner


def _episode_paths(run_name: str, index: int):
    runner = _runner()
    try:
        run_dir = runner.run_dir(run_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if run_dir is None or not run_dir.exists():
        raise HTTPException(404, f"unknown run: {run_name}")
    return run_dir / f"episode_{index}.jsonl", run_dir / f"live_{index}"


@router.post("/start")
async def start(req: StartRequest) -> dict:
    try:
        run_name = await asyncio.to_thread(
            _runner().start,
            episodes=req.episodes,
            split=req.split,
            max_turns=req.max_turns,
            model=req.model,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"run_name": run_name}


@router.post("/stop")
async def stop() -> dict:
    await asyncio.to_thread(_runner().stop)
    return {"ok": True}


@router.get("/status")
async def status() -> dict:
    return _runner().status()


@router.get("/runs")
async def list_runs() -> dict:
    """All run dirs under outputs/beta-coding-agent/ (UI- and CLI-launched alike)."""
    if not OUTPUT_ROOT.exists():
        return {"runs": []}
    runs = []
    for d in OUTPUT_ROOT.iterdir():
        if not d.is_dir():
            continue
        eps = sorted(int(p.stem.split("_")[1]) for p in d.glob("episode_*.jsonl")
                     if p.stem.split("_")[1].isdigit())
        has_summary = (d / "summary.json").exists()
        if not eps and not has_summary:
            continue  # e.g. auto_host log folders — not a run
        entry: dict = {"name": d.name, "mtime": d.stat().st_mtime, "episodes": eps}
        if has_summary:
            try:
                data = json.loads((d / "summary.json").read_text())
                agg = data.get("aggregate") or {}
                cfg = data.get("config") or {}
                entry.update(
                    success=agg.get("success"),
                    episode_count=agg.get("episode_count"),
                    model=cfg.get("model"),
                    skill=cfg.get("skill"),
                )
            except ValueError:
                pass
        runs.append(entry)
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": runs}


@router.get("/runs/{run_name}/summary")
async def run_summary(run_name: str) -> dict:
    """Per-episode outcomes for one run — same shape the live status uses."""
    runner = _runner()
    try:
        run_dir = runner.run_dir(run_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if run_dir is None or not run_dir.exists():
        raise HTTPException(404, f"unknown run: {run_name}")

    started = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("episode_*.jsonl")
                     if p.stem.split("_")[1].isdigit())
    episodes: list[dict] = []
    aggregate = None
    config: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            aggregate = data.get("aggregate")
            cfg = data.get("config") or {}
            config = {k: cfg.get(k) for k in ("split", "model", "skill", "max_turns", "episodes")}
            for e in data.get("episodes", []):
                m = e.get("metrics") or {}
                a = e.get("agent") or {}
                episodes.append(
                    {
                        "index": e.get("index"),
                        "success": m.get("success"),
                        "spl": m.get("spl"),
                        "distance_to_goal": m.get("distance_to_goal"),
                        "env_steps": a.get("env_steps"),
                        "called_stop": a.get("called_stop"),
                        "error": e.get("error"),
                    }
                )
        except ValueError:
            pass
    return {
        "run_name": run_name,
        "started_episodes": started,
        "episodes": episodes,
        "aggregate": aggregate,
        "config": config,
    }


@router.get("/runs/{run_name}/episode/{index}/textlog")
async def textlog(run_name: str, index: int, offset: int = 0) -> dict:
    jsonl_path, _ = _episode_paths(run_name, index)
    if not jsonl_path.exists():
        return {"lines": [], "next_offset": offset}
    raw = jsonl_path.read_text().splitlines()
    lines = []
    for line in raw[offset:]:
        try:
            lines.append(json.loads(line))
        except ValueError:
            continue
    return {"lines": lines, "next_offset": len(raw)}


@router.get("/runs/{run_name}/episode/{index}/frames")
async def frames(run_name: str, index: int) -> dict:
    _, live_dir = _episode_paths(run_name, index)
    if not live_dir.exists():
        return {"frames": []}
    names = sorted(p.name for p in live_dir.glob("obs_*.png"))
    return {"frames": names}


@router.get("/runs/{run_name}/episode/{index}/frame/{name}")
async def frame(run_name: str, index: int, name: str):
    if "/" in name or ".." in name or not name.endswith(".png"):
        raise HTTPException(400, "bad frame name")
    _, live_dir = _episode_paths(run_name, index)
    path = live_dir / name
    if not path.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(path, media_type="image/png")
