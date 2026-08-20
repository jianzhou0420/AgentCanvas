"""Coding-Agent Monitor API — control + live-log surface for coding-agent runs.

Thin shell over ``services.coding_agent_runner.CodingAgentRunner``. Live text
comes from the driver's per-episode trajectory JSONL (flushed per event, so
whole-file-with-offset polling is real-time); live images come from the
bridge's ``live_{i}/`` frame dumps.
"""

from __future__ import annotations

import asyncio
import json
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...services.coding_agent_runner import OUTPUT_ROOT, REPO_ROOT, display_aggregate
from ...state import get_services

router = APIRouter()

# Log sources the read endpoints can browse. All drivers write the same
# artifact layout (episode_{i}.jsonl + live_{i}/ + summary.json), so the same
# monitor surface serves all three; only the root differs. The control
# endpoints (start/stop/status) launch any of the three via the shared driver.
SOURCE_ROOTS = {
    "claude-sdk": OUTPUT_ROOT,                                # Agent SDK runs
    "mini-swe": OUTPUT_ROOT.parent / "beta-react-harness",    # mini-swe-agent harness
    "codex": OUTPUT_ROOT.parent / "beta-codex-agent",         # OpenAI Codex CLI harness
    "eharness": OUTPUT_ROOT.parent / "beta-eharness",         # embodied-harness shell
    "vla": OUTPUT_ROOT.parent / "beta-vla-harness",           # outer model driving the 2B VLA
    # ImagineVLN: single-turn MapGPT +/- world-model rollouts. Not a coding
    # harness -- it only borrows this artifact contract, so the monitor works
    # unchanged (see ImagineVLN/agent/run_mapgpt.py).
    "imagine": OUTPUT_ROOT.parent / "beta-imaginevln",
}


class StartRequest(BaseModel):
    episodes: str = "0-9"
    split: str = "rand100"
    max_turns: int = 200  # std-v2 frozen cap (cells.STD_FROZEN)
    model: str | None = None
    harness: str = "claude-sdk"   # claude-sdk | mini-swe | codex (SOURCE_ROOTS keys)
    condition: str = "bare"       # bare | wp | hybrid (MIP paper grid)
    tier: str = "default"         # effort tier: default | max
    extra: dict[str, str] = {}    # harness extra knobs (--set KEY=VAL), override tier's


def _runner():
    runner = getattr(get_services(), "coding_agent_runner", None)
    if runner is None:
        raise HTTPException(503, "coding-agent runner not initialized")
    return runner


def _source_root(source: str):
    root = SOURCE_ROOTS.get(source)
    if root is None:
        raise HTTPException(400, f"unknown source: {source!r}")
    return root


def _run_dir(source: str, run_name: str):
    # Dots are legal: every std-v1 cell name carries one (qwen3.5, opus-4.8,
    # gpt-5.5). Traversal stays blocked — "/" is not in the allowlist, and ".."
    # is rejected outright.
    if not run_name or ".." in run_name \
            or not all(c.isalnum() or c in "_-." for c in run_name):
        raise HTTPException(400, f"bad run name: {run_name!r}")
    run_dir = _source_root(source) / run_name
    if not run_dir.exists():
        raise HTTPException(404, f"unknown run: {run_name}")
    return run_dir


def _episode_paths(run_name: str, index: int, source: str = "claude-sdk"):
    run_dir = _run_dir(source, run_name)
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
            harness=req.harness,
            condition=req.condition,
            tier=req.tier,
            extra=req.extra,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
async def list_runs(source: str = "claude-sdk") -> dict:
    """All run dirs under the source's output root (UI- and CLI-launched alike)."""
    root = _source_root(source)
    if not root.exists():
        return {"runs": []}
    runs = []
    for d in root.iterdir():
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
                agg = display_aggregate(data.get("episodes", []))
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
async def run_summary(run_name: str, source: str = "claude-sdk") -> dict:
    """Per-episode outcomes for one run — same shape the live status uses."""
    run_dir = _run_dir(source, run_name)

    started = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("episode_*.jsonl")
                     if p.stem.split("_")[1].isdigit())
    episodes: list[dict] = []
    aggregate = None
    config: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            aggregate = display_aggregate(data.get("episodes", []))
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


@router.get("/runs/{run_name}/stats")
async def stats_report(run_name: str, source: str = "claude-sdk",
                       refresh: bool = False):
    """Self-contained statistics report (charts + tables after the metric
    block) for one run. The driver writes stats.html at run end; older runs
    are backfilled on demand here by invoking coding-agent/reporting/run_stats.py in
    this backend's interpreter (same env, has matplotlib)."""
    run_dir = _run_dir(source, run_name)
    path = run_dir / "stats.html"
    if refresh or not path.exists():
        if not (run_dir / "summary.json").exists():
            raise HTTPException(409, "run has no summary.json yet — still running?")
        script = REPO_ROOT / "coding-agent" / "reporting" / "run_stats.py"
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), str(run_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
        if not path.exists():
            raise HTTPException(
                500, f"stats generation failed: {out.decode(errors='replace')[-400:]}")
    return FileResponse(path, media_type="text/html")


@router.get("/runs/{run_name}/episode/{index}/textlog")
async def textlog(run_name: str, index: int, offset: int = 0, source: str = "claude-sdk") -> dict:
    jsonl_path, _ = _episode_paths(run_name, index, source)
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
async def frames(run_name: str, index: int, source: str = "claude-sdk") -> dict:
    _, live_dir = _episode_paths(run_name, index, source)
    if not live_dir.exists():
        return {"frames": []}
    # habitat bridges dump .png; the go2 bridge dumps camera .jpg, and the
    # real-robot harness dumps .jpeg — accept all three.
    names = sorted(
        p.name for p in live_dir.iterdir()
        if p.name.startswith("obs_") and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    return {"frames": names}


@router.get("/runs/{run_name}/episode/{index}/harness")
async def harness_state(run_name: str, index: int, source: str = "claude-sdk") -> dict:
    """eharness organ readout for one episode: the live state block, the
    harness-written heartbeat, the keyframe index, and receipts (two-tier
    runs). All files live under live_{i}/ and are written incrementally by
    the shell — reads are defensive because a poll can race a write."""
    import json as _json
    _, live_dir = _episode_paths(run_name, index, source)
    out: dict = {"state": None, "heartbeat": None, "depth": None,
                 "keyframes": [], "receipts": []}
    if not live_dir.exists():
        return out
    # `depth` is the geometry organ's readout (depth-waypoint runs): the
    # top-down map it built this look, the candidates it derived and the
    # landmark ledger. Written for humans watching the monitor — the model
    # never receives any of it.
    for key, fname in (("state", "state.json"), ("heartbeat", "heartbeat.json"),
                       ("depth", "depth.json")):
        p = live_dir / fname
        if p.exists():
            try:
                out[key] = _json.loads(p.read_text())
            except (ValueError, OSError):
                pass  # mid-write — next poll gets it
    kf_path = live_dir / "keyframes.jsonl"
    if kf_path.exists():
        latest: dict[int, dict] = {}   # append-log: last record per idx wins
        try:
            for line in kf_path.read_text().splitlines():
                try:
                    rec = _json.loads(line)
                    latest[rec["idx"]] = rec
                except (ValueError, KeyError):
                    continue
        except OSError:
            pass
        out["keyframes"] = [r for _, r in sorted(latest.items())
                            if r.get("keyframe")]
    seg_path = live_dir / "segments.jsonl"
    if seg_path.exists():
        try:
            out["segments"] = [_json.loads(line) for line
                               in seg_path.read_text().splitlines() if line.strip()]
        except (ValueError, OSError):
            out["segments"] = []
    else:
        out["segments"] = []
    rc_path = live_dir / "receipts.jsonl"
    if rc_path.exists():
        try:
            out["receipts"] = [_json.loads(line) for line
                               in rc_path.read_text().splitlines() if line.strip()]
        except (ValueError, OSError):
            pass
    return out


@router.get("/runs/{run_name}/episode/{index}/frame/{name}")
async def frame(run_name: str, index: int, name: str, source: str = "claude-sdk"):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if "/" in name or ".." in name or ext not in ("png", "jpg", "jpeg"):
        raise HTTPException(400, "bad frame name")
    _, live_dir = _episode_paths(run_name, index, source)
    path = live_dir / name
    if not path.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(path, media_type="image/png" if ext == "png" else "image/jpeg")


@router.get("/runs/{run_name}/episode/{index}/snapshot")
async def snapshot(run_name: str, index: int, source: str = "claude-sdk"):
    """The atomic live snapshot (§6 of the multiturn revision): one manifest
    naming the exact versioned files of ONE observation. The monitor reads
    this first and then fetches precisely these files — never a mix of
    latest.png generations. `consistent` is the backend's own check that
    every referenced file exists."""
    _, live_dir = _episode_paths(run_name, index, source)
    path = live_dir / "live_snapshot.json"
    if not path.exists():
        raise HTTPException(404, "no snapshot yet")
    try:
        snap = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        raise HTTPException(503, f"snapshot unreadable: {e}")
    missing = [k for k in ("rgb_file", "topdown_file", "accumulated_map_file",
                           "model_map_file", "depth_json_file")
               if snap.get(k) and not (live_dir / snap[k]).exists()]
    snap["consistent"] = not missing
    if missing:
        snap["missing"] = missing
    return snap
