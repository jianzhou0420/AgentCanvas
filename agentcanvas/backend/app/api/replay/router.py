"""Log replay API — env-agnostic timeline served from per-episode log.jsonl.

Endpoints:
    GET /api/replay/{run_id}/episodes
    GET /api/replay/{run_id}/episode/{episode_index}

Each eval episode owns a self-contained log dir
``outputs/eval_runs/{run_id}/episodes/ep{idx:04d}/`` — the parser reads
that episode's ``log.jsonl`` directly, no run-level splitting.

Frame bytes themselves are served by the logs assets endpoint, keyed by
the composite execution_id ``{run_id}_ep{idx:04d}``:
    GET /api/logs/{execution_id}/assets/{filename}

The frontend constructs that URL from the episode's ``execution_id`` +
each step's ``frame_url`` field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from app.api.execution import eval_storage
from app.api.execution.logs import _sanitize_id
from app.replay.interface import BaseReplayParser
from app.state import get_services

router = APIRouter()
log = logging.getLogger("agentcanvas.replay")


def _resolve_run_and_parser(run_id: str) -> tuple[dict, str, BaseReplayParser]:
    """Resolve run record + parser; raise HTTPException on any miss."""
    safe_id = _sanitize_id(run_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    run = eval_storage.load_run(safe_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {safe_id} not found")

    config = run.get("config") or {}
    env_nodeset = config.get("env_nodeset")
    if not env_nodeset:
        raise HTTPException(
            status_code=400,
            detail=f"Run {safe_id} has no env_nodeset in config",
        )

    registry = get_services().workspace_component_registry
    parser = registry.get_replay_parser(env_nodeset)
    if parser is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No replay parser registered for env_nodeset='{env_nodeset}'. "
                f"Add a replay_parser file next to the nodeset module."
            ),
        )
    return run, safe_id, parser


def _resolve_episode_log_path(run_id: str, episode_index: int) -> Path:
    """Locate one episode's ``log.jsonl`` under its self-contained dir."""
    log_path = (
        eval_storage.EVAL_RUNS_DIR / run_id / "episodes" / f"ep{episode_index:04d}" / "log.jsonl"
    )
    if not log_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"log.jsonl not found for run {run_id} episode {episode_index}",
        )
    return log_path


@router.get("/{run_id}/episodes")
async def list_episodes(run_id: str) -> dict[str, Any]:
    """Return episode summaries for a run, straight from summary.json.

    Each episode owns a self-contained log dir, so ``step_count`` here is
    eval-storage's authoritative value — there is no longer a parser
    re-derivation that can disagree. ``has_log`` flags whether the
    episode's ``log.jsonl`` is present (a crashed/aborted episode may
    have none).
    """
    safe_id = _sanitize_id(run_id)
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    run = eval_storage.load_run(safe_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {safe_id} not found")

    episodes_dir = eval_storage.EVAL_RUNS_DIR / safe_id / "episodes"
    episodes: list[dict[str, Any]] = []
    for stored in run.get("episodes") or []:
        idx = stored.get("episode_index", len(episodes))
        ep_log = episodes_dir / f"ep{idx:04d}" / "log.jsonl"
        episodes.append(
            {
                "episode_index": idx,
                "episode_id": stored.get("episode_id", ""),
                "instruction": stored.get("instruction", ""),
                "step_count": stored.get("step_count", 0),
                "metrics": stored.get("metrics", {}),
                "status": stored.get("status", ""),
                "scene_id": stored.get("scene_id", ""),
                "has_log": ep_log.is_file(),
            }
        )

    return {
        "run_id": safe_id,
        "env_nodeset": run.get("config", {}).get("env_nodeset"),
        "graph_name": run.get("config", {}).get("graph_name"),
        "episodes": episodes,
    }


@router.get("/{run_id}/episode/{episode_index}")
async def get_episode(run_id: str, episode_index: int) -> dict[str, Any]:
    """Return the full :class:`ReplayEpisode` timeline for a single episode."""
    run, safe_id, parser = _resolve_run_and_parser(run_id)
    log_path = _resolve_episode_log_path(safe_id, episode_index)

    try:
        episode = parser.parse(log_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # parser bugs surface here
        log.exception("Replay parser failed for run=%s ep=%d", safe_id, episode_index)
        raise HTTPException(status_code=500, detail=f"Parser error: {exc}") from exc

    # Parser works on a single-episode file and doesn't know run-context —
    # stamp the episode_index here.
    episode.episode_index = episode_index

    # Merge stored metrics in (parser leaves it empty by convention).
    # Match by episode_index field, not list position — runs with
    # start_episode_index > 0 have non-zero-based indices.
    stored = next(
        (e for e in run.get("episodes") or [] if e.get("episode_index") == episode_index),
        None,
    )
    if stored:
        if not episode.metrics:
            episode.metrics = stored.get("metrics", {}) or {}
        if not episode.instruction:
            episode.instruction = stored.get("instruction", "") or ""

    payload = episode.to_dict()
    # Backstop — parser.parse() may not set supports_smooth on its
    # ReplayEpisode (Generic doesn't, custom parsers may forget).
    # Source of truth is parser.supports_smooth_mode().
    payload["supports_smooth"] = parser.supports_smooth_mode()
    payload["run_id"] = safe_id
    # Composite id the frontend uses to fetch frame assets via
    # /api/logs/{execution_id}/assets/{filename}.
    payload["execution_id"] = f"{safe_id}_ep{episode_index:04d}"
    return payload


@router.get(
    "/{run_id}/episode/{episode_index}/step/{step_index}/frame",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_smooth_frame(
    run_id: str,
    episode_index: int,
    step_index: int,
    t: float = 0.5,
) -> Response:
    """Return a smooth-mode interpolated JPEG frame.

    ``t ∈ [0, 1]``: 0 = pose at ``step_index``, 1 = pose at ``step_index+1``.
    Frontend's smooth play loop requests N frames per step at evenly
    spaced ``t`` values to simulate continuous walking.
    """
    _, safe_id, parser = _resolve_run_and_parser(run_id)
    if not parser.supports_smooth_mode():
        raise HTTPException(
            status_code=404,
            detail=(
                f"smooth mode not supported by {type(parser).__name__} "
                f"(env_nodeset for run {safe_id})"
            ),
        )
    log_path = _resolve_episode_log_path(safe_id, episode_index)
    try:
        jpeg = await parser.get_smooth_frame(log_path, step_index, t)
    except (IndexError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception(
            "Smooth-frame render failed for run=%s ep=%d step=%d t=%.3f",
            safe_id,
            episode_index,
            step_index,
            t,
        )
        raise HTTPException(status_code=500, detail=f"render error: {exc}") from exc
    return Response(content=jpeg, media_type="image/jpeg")
