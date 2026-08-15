"""Human-performance test API — interactive human driver over env_habitat.

Thin shell over ``services.human_runner.HumanRunner``. The browser owns the
control loop: start the env, load an episode, send one keypress = one discrete
action, then STOP to evaluate. Frames ride back inline as base64 PNG (data
URIs) so the loop stays a single request per action; the persisted trajectory
+ metrics live under ``outputs/human/{split}/`` (see HumanRunner).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
    depth_resolution: int = 512   # match the camera; the YAML default is 256
    # The camera rig. Defaults live in human_runner (DEFAULT_CAM_HEIGHT_M /
    # DEFAULT_DEPTH_MAX_M / DEFAULT_DEPTH_NORMALIZE) and are restated here only
    # so the browser can override one without knowing the others.
    cam_height_m: float = 1.5     # RGB + DEPTH together; habitat's default 1.25
    depth_max_m: float = 100.0    # habitat's default 10 — past any scene here
    depth_normalize: bool = False  # False = true metres on the wire


class StepRequest(BaseModel):
    action: int  # 1 = FORWARD, 2 = TURN_LEFT, 3 = TURN_RIGHT


def _runner():
    runner = getattr(get_services(), "human_runner", None)
    if runner is None:
        raise HTTPException(503, "human runner not initialized")
    return runner


# ── depth-organ inspector ────────────────────────────────────────────────
# Drive by hand, watch what the geometry organ makes of the very same frame.
# This is the fastest way to judge a proposer: park the robot somewhere
# awkward and see whether the places it offers are the ones a person would
# take. Everything is computed on demand from the CURRENT observation — no
# state, no episode coupling, safe to call at any point in a session.
_DEPTH_ORGAN_PATH = "coding-agent"


def _depth_modules():
    """Import the organ from coding-agent/ lazily (it is not a backend dep)."""
    import sys
    from pathlib import Path

    # backend/app/api/execution/human.py → parents[5] is the repo root
    root = Path(__file__).resolve().parents[5] / _DEPTH_ORGAN_PATH
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eharness import depthmap as dm
    from eharness import landmarks as lm

    return dm, lm


class DepthAnalyzeRequest(BaseModel):
    phrases: list[str] = []          # landmark nouns for SAM 3; [] = skip it
    sam_url: str = "http://127.0.0.1:9220"
    # None = whatever the organ's own default is. A literal here is a second
    # place to remember: the grid moved from 6 m to 12 m in depthmap.py and this
    # copy went on pinning the live tab at 6, so every measurement taken through
    # this endpoint — the one surface a human can actually watch — described a
    # map nothing else was building.
    range_cap_m: float | None = None
    reset_map: bool = False          # forget the accumulated map
    sam_px: int = 1024               # upscale the frame before SAM 3 sees it


# The accumulated map is memory, so it lives across requests. Every analysis
# replays the actions taken since the last one: dead reckoning is exact and
# costs nothing, so a gap in the analysis sequence no longer destroys the map —
# it only means those frames contributed no evidence. The map is keyed to the
# SESSION, not the episode, because restarting the env teleports the body back
# to the start while episode_id and index stay the same.
#
# THIS ENDPOINT IS THE REFERENCE SURFACE FOR THE ACCUMULATED MAP. Every change
# to eharness/depthmap.py::AnchorMap must show up here, because this tab is the
# only place a human can watch the organ work on frames they chose themselves.
# An improvement that only exists on the agent path cannot be inspected, and an
# organ nobody can inspect is one nobody can distrust.
_MAP: dict = {"map": None, "last_step": None, "session": None, "steps": 0,
              "prev_depth": None, "prev_rgb": None, "rolled_m": 0.0,
              # The place currently being walked toward. Lives beside the map
              # because it is stored in the map's anchor frame — without a
              # pose there is nothing to hold a goal still against.
              "goal": None,
              # §7: the resident circle detector — trail-signal subset on the
              # human path (no waypoint-sector feed; revisit + circling +
              # map-growth still convict together)
              "loop": None}
_STEP_M = 0.25
_TURN_RAD = 0.2617993877991494      # 15°


@router.post("/depth-analysis")
async def depth_analysis(req: DepthAnalyzeRequest) -> dict:
    """Run the depth organ (and optionally SAM 3) on the current frame."""
    import base64 as _b64
    import math as _math

    import numpy as _np

    runner = _runner()
    url = getattr(runner, "_url", None)
    if not url:
        raise HTTPException(409, "no live env — start the server and load an episode")

    def _work() -> dict:
        dm, lm = _depth_modules()
        outputs = runner._call(url, "env_habitat__observe_egocentric", {})
        rgb_b64 = outputs.get("rgb")
        depth_field = outputs.get("depth")
        # The env states what its depth numbers mean, on the same observation
        # that carries them. Without this the organ falls back to inferring the
        # units from the frame ("max ≤ 1 must be normalised"), which is a coin
        # flip whenever the whole view is close range — and once habitat's 10 m
        # clip is off, being wrong is a 100× world, not a 10× one.
        units = outputs.get("depth_units") or {}
        scale_m = runner.scale_of(units)
        cap_m = float(req.range_cap_m or dm.RANGE_CAP_M)
        # Freshly proposed places. Which of them the driver is actually SHOWN is
        # settled further down, after the accumulated map has been moved to this
        # frame's pose — a commitment is held in the map's anchor frame, so it
        # cannot be converted back to a bearing until the pose is current. The
        # three pictures are therefore drawn at the end, from the final list;
        # drawing them here painted the pre-commitment candidates and the panel
        # disagreed with its own image.
        fresh, td = dm.propose_from_depth(depth_field,
                                          range_cap_m=cap_m,
                                          scale_m=scale_m)
        if td is None:
            return {"error": "no depth on this observation"}

        depth = dm.decode_depth(depth_field)
        metres, normalized = dm.to_metres(depth, scale_m)

        sightings = []
        detector = "off"
        misses: dict[str, list[str]] = {}
        sam_png = b""
        organ = None
        if req.phrases:
            organ = lm.LandmarkOrgan(req.sam_url)
            # SAM 3 resizes whatever it is given to its own working resolution,
            # so handing it a frame smaller than that throws away detail before
            # the model ever looks — a bar counter 10 m down a corridor is only
            # a few dozen pixels at 512 and simply is not there to be found.
            # Upscaling a small render adds no information; rendering the
            # episode at 1024 (the loader now does) is what actually helps, and
            # this keeps the input at SAM's scale either way.
            sam_input = lm.upscale_b64(rgb_b64, req.sam_px) if req.sam_px else rgb_b64
            sightings = organ.sightings(sam_input, depth_field, req.phrases,
                                        scale_m=scale_m)
            detector = ("ok" if organ.available
                        else f"unavailable: {organ.last_error[:140]}")
            # Which phrases the detector could not see under ANY of the words
            # tried for them. "SAM cannot find your bar, and here is everything
            # it was asked" is a different problem from "there is no bar here",
            # and the panel used to show the same blank for both.
            misses = {k: v for k, v in organ.misses.items()}
            if sightings:
                sam_png = lm.render_masks(sam_input, sightings)

        # ── accumulated map: replay the actions taken since the last look ────
        session = getattr(runner, "_session", None) or {}
        actions = list(session.get("actions") or [])
        # A step counter that goes BACKWARDS, or a different env URL, is the
        # unmistakable sign of a fresh life — start the map over rather than
        # draw two walks on top of each other.
        key = (session.get("episode_id"), session.get("index"), url)
        steps_now = int(session.get("step_count") or 0)
        rewound = steps_now < int(_MAP.get("steps") or 0)
        _MAP["steps"] = steps_now
        amap = _MAP["map"]
        rolled_m = 0.0
        if (amap is None or req.reset_map or _MAP["session"] != key or rewound
                or abs(float(getattr(amap, "cell", 0.10)) - 0.10) > 1e-9):
            # BACK to 10 cm cells (the 5 cm experiment made things WORSE in
            # practice: 4× thinner evidence per cell turned a fresh sweep
            # into sparse dots that read as a broken map, and the dotted
            # FREE broke the contiguous prefixes the 360° memory-candidate
            # scan needs). Smoothness comes from the LANCZOS pass; the
            # cell-size guard above migrates any live 5 cm map on first use.
            amap = dm.AnchorMap()
            _MAP.update({"map": amap, "session": key, "last_step": len(actions),
                         "prev_depth": None, "prev_rgb": None, "rolled_m": 0.0,
                         "goal": dm.WaypointGoal(), "loop": dm.LoopMonitor(),
                         # §21.6: fresh life, fresh candidate identity
                         "wreg": dm.WaypointRegistry(), "epoch": 0,
                         # the sweep below fuses the heading-0 view of THIS
                         # env step — the analysis that follows must not
                         # fuse the same frame a second time
                         "fused_step": steps_now})
            skipped = 0
            trans_expected = False       # standing at spawn — no translation
            # user ruling 2026-08-12 night: every FRESH map opens with the
            # zero-step ±60° sweep — the heading-up accumulated panel shows
            # the surroundings the moment the episode loads
            _MAP["sweep_seen"] = _sweep_frame0(
                amap, runner, url, organ, list(req.phrases or []),
                scale_m, dm)
        else:
            pending = actions[(_MAP["last_step"] or 0):]
            fwd_m = 0.0
            for a in pending:
                if isinstance(a, dict) and "turn" in a:
                    # a goto's quaternion rotation — real motion that never
                    # was a discrete primitive; skipping it rotated every
                    # post-goto wall by the missing angle
                    amap.odometry(float(a["turn"]), 0.0)
                elif a == 1:
                    amap.odometry(0.0, _STEP_M)
                    fwd_m += _STEP_M
                elif a == 2:
                    amap.odometry(_TURN_RAD, 0.0)
                elif a == 3:
                    amap.odometry(-_TURN_RAD, 0.0)
            # Did the body actually move? A commanded forward that a wall ate
            # still incremented the odometry, and that error is ONE-SIDED — it
            # only ever overstates progress, so it never averages out. Measured
            # against habitat's own pose on three episodes: 60 of 168 forwards
            # moved the body zero, one episode 40 of 56, and the accumulated
            # error tracked the loss almost exactly. Take the metres back before
            # anything is stamped, so the wedge lands at the corrected pose.
            #
            # This is why the "每步自动分析" checkbox matters: the test compares
            # THIS frame with the previous one, so analysing every step gives it
            # one command to judge at a time. Analyse every fifth step and it can
            # still catch "none of those five moved", but not which.
            # BOTH depth and RGB must call it still. Sliding along a flat
            # wall is invisible to depth (the wall looks the same from every
            # point along it) and obvious in RGB — measured, 8 slides in 168
            # forwards, 1.40 m of real travel that depth alone erased.
            if fwd_m > 0 and dm.frames_still(
                    _MAP.get("prev_depth"), depth,
                    prev_rgb=_MAP.get("prev_rgb"), rgb=rgb_b64,
                    scale_m=scale_m):
                amap.retract(fwd_m)
                rolled_m = fwd_m
                _MAP["rolled_m"] = float(_MAP.get("rolled_m") or 0.0) + fwd_m
            # Frames the driver walked through without analysing contribute no
            # evidence, but the POSE is still exact — replaying odometry is
            # lossless. Reported, not punished.
            skipped = max(0, len(pending) - 1)
            _MAP["last_step"] = len(actions)
            # §21.5: only a batch that actually WALKED may earn a
            # translation fix — pure turns (A/D, goto's quaternion
            # rotation) register read-only, or head-turning parallax gets
            # booked as body motion and the anchors smear
            trans_expected = fwd_m > 0 and not rolled_m
        _MAP["prev_depth"] = depth
        _MAP["prev_rgb"] = rgb_b64

        # Correct the dead reckoning against the map itself, then stamp.
        # register() no longer searches rotation: dead-reckoned heading error
        # measured 0.00° over 90 steps on all three episodes (habitat's turns
        # are exact), while letting the matcher rotate injected up to 15.5°.
        # IDEMPOTENT at a standstill (user report: same pose, same photo,
        # different DWP): the SAME env step is registered/fused exactly
        # once — re-analysing without moving is a pure read. Repeat fusing
        # kept deepening logodds and register kept nudging the pose, so
        # every extra click changed the candidates.
        loop_mon = _MAP.get("loop")
        if loop_mon is None:
            loop_mon = _MAP["loop"] = dm.LoopMonitor()
        if _MAP.get("fused_step") != steps_now:
            match = amap.register(td, translation_expected=trans_expected)
            drift = amap.disagreement(td)
            amap.fuse(td)
            _MAP["fused_step"] = steps_now
            loop_mon.note_growth(int((_np.abs(amap.logodds) > 0.5).sum()))
            for s in sightings:
                if s.mask is None:
                    continue
                xs, ys = dm.project_mask(s.mask, depth, td.floor_y,
                                         scale_m=scale_m)
                if xs.size:
                    amap.stamp_semantic(s.phrase, xs, ys,
                                        weight=float(s.score))
        else:
            match = float(amap.last_score)
            drift = 0.0
        loop_warn = loop_mon.assess(amap)
        # The pose is current, so the commitment can be read back now. This is
        # what stops the offered place from being a carrot on a stick: propose()
        # is memoryless and re-picks the furthest reachable point every frame, so
        # walking toward it moved it, and the panel showed the same "3.0 m ahead"
        # step after step. Held still, the metres count down and arriving is a
        # thing that happens.
        goal = _MAP.get("goal")
        if goal is None:
            goal = _MAP["goal"] = dm.WaypointGoal()
        # adopt=False: ANALYSIS is a read — auto-adopting the best fresh
        # place made the very next analyze show a held goal ("you were
        # already heading here") the human never chose, so two clicks at
        # the same pose disagreed. Adoption happens in the goto flow.
        waypoints = goal.apply(amap, td, fresh, adopt=False)
        # §global-dwp (user 2026-08-12 night): the Human tab gets the SAME
        # memory candidates the agent line gets — the corridor you saw on
        # the right is a numbered option here too (trusted-map + ≥2-updates
        # gates live inside; behind-view numbers appear on the map, not the
        # photo)
        try:
            waypoints = waypoints + dm.propose_from_memory(amap, waypoints)
        except Exception:  # noqa: BLE001 — memory must never break analyze
            pass
        # §21.6: same physical exit → same circle, here too. The registry
        # snaps matched candidates onto their standing anchor centre and
        # culls the ones whose fixed centre this frame cannot verify.
        wreg = _MAP.get("wreg")
        if wreg is None:
            wreg = _MAP["wreg"] = dm.WaypointRegistry()
        _MAP["epoch"] = int(_MAP.get("epoch") or 0) + 1
        try:
            waypoints = wreg.reconcile(amap, td, waypoints,
                                       epoch=int(_MAP["epoch"]))
        except Exception:  # noqa: BLE001 — identity must never break analyze
            pass
        held = bool(waypoints and waypoints[0].note == "you were already heading here")

        caption = dm.topdown_caption(td, waypoints)
        topdown = dm.render_topdown(td, waypoints, caption=caption)
        annotated = b""
        if isinstance(rgb_b64, str):
            annotated = dm.annotate_rgb(_b64.b64decode(rgb_b64), waypoints, td.floor_y)

        recall = amap.recall_sentence()
        # user 2026-08-15: ONE map on the Human tab — the anchor-fixed
        # GLOBAL alone (no local inset, no separate model-view duplicate;
        # the model's IMAGE 2 composite is the agent line's concern)
        map_model_png = b""
        # user 2026-08-15 (supersedes both the 08-12 heading-up ruling and
        # §21.7's composite HERE): the Human tab shows ONE map — the
        # anchor-fixed GLOBAL alone. The world holds still, the yellow
        # arrow moves and turns; render() content-crops so the frame hugs
        # the explored world; map_version/candidate_epoch ride the caption.
        _caption_text = (f"map_version {int(amap.updates)} · menu epoch "
            f"{int(_MAP.get('epoch') or 0)}\n" + amap.caption()
            + (f" · contradiction {drift:.2f} m" if drift else "")
            + (f"\nBLOCKED: the view did not change — {rolled_m:.2f} m of "
               "commanded forward was never walked, and has been taken back"
               if rolled_m else "")
            + (f"\n{skipped} move(s) happened without a look — pose replayed, "
               "no evidence gathered" if skipped else "")
            + (f"\nLOOP WARNING (score {loop_warn['loop_score']}): walked "
               f"{loop_warn['path_m']} m for {loop_warn['net_m']} m of net "
               "progress — the trail says this area was walked before"
               if loop_warn else "")
            + (f"\n{recall}" if recall else ""))
        map_png = amap.render(waypoints, caption=_caption_text)

        # Two profiles, two questions. `ahead_m` used to be the SIGHTLINE while
        # the sentence beside it quoted the PASSABLE distance, so the panel read
        # "正前方 2.85 m" directly above "something is blocking you about 1.0 m
        # ahead". Same for the roomiest bearing: read off the sightline it can
        # point at a crack the body cannot enter. Both now come off the profile
        # the proposer actually obeys, and the sightline is reported separately
        # under its own name.
        prof = dm.passable_range(td)
        walk_ahead = float(prof[len(td.bearings) // 2])
        best = float(prof.max())
        tied = [i for i, v in enumerate(prof) if float(v) >= best * 0.95]
        k = min(tied, key=lambda i: abs(float(td.bearings[i]))) if tied else 0
        bearing, widest = float(td.bearings[k]), float(prof[k])
        grid, total = td.grid, float(td.grid.size) or 1.0
        return {
            # max_m=None lets the ramp span what this frame actually holds. It
            # used to be pinned to the MAP's 6 m cap, so anything further away
            # was one flat blue — the picture could not show a 12 m corridor
            # even when the sensor measured it.
            "depth_png": _b64.b64encode(
                dm.render_depth(depth, scale_m=scale_m)).decode(),
            "topdown_png": _b64.b64encode(topdown).decode(),
            "map_png": _b64.b64encode(map_png).decode() if map_png else "",
            "sweep_seen": list(_MAP.get("sweep_seen") or []),
            "map_model_png": (_b64.b64encode(map_model_png).decode()
                              if map_model_png else ""),
            "sam_png": _b64.b64encode(sam_png).decode() if sam_png else "",
            "map_updates": amap.updates,
            # register() returns a hard 0.0 on its early-outs (too little on the
            # map, or too little in this wedge) — that is "no opinion", NOT a
            # bad match, and the two must not read the same. `map_last_score` is
            # the organ's own retained value, which is what `trusted` is judged
            # on; showing only the returned score put a 0.00 next to a green
            # trusted flag on the same panel.
            "map_match": round(match, 2),
            "map_no_opinion": match == 0.0,
            "map_last_score": round(float(amap.last_score), 2),
            "map_fixes": amap.fixes,
            "map_fix_m": round(_math.hypot(*amap.last_fix[:2]), 2),
            "map_drift_m": round(drift, 2),
            "map_skipped": skipped,
            # the drift-study organs, surfaced so the tab shows what actually ran
            "map_rolled_m": round(rolled_m, 2),
            "map_rolled_total_m": round(float(_MAP.get("rolled_m") or 0.0), 2),
            "map_turn_search": bool(dm.MATCH_TURN),
            "map_trusted": bool(amap.trusted),
            "map_walked_m": round(sum(
                _math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(amap.trail, amap.trail[1:])), 1),
            "map_recall": recall,
            "map_semantic": [
                {"phrase": r["phrase"], "distance_m": round(r["distance"], 2),
                 "bearing_deg": round(_math.degrees(r["bearing"]), 1),
                 "behind": r["behind"], "cells": r["cells"]}
                for r in amap.semantic_recall()
            ],
            "annotated_png": _b64.b64encode(annotated).decode() if annotated else "",
            "depth_units": (
                f"normalized[0,1] ×{scale_m or dm.DEPTH_FULL_RANGE_M:g} m"
                if normalized else "metres"
            ) + ("" if scale_m is not None else "  (INFERRED — env did not say)"),
            "depth_declared": bool(scale_m is not None),
            "depth_clip_m": units.get("max_depth_m"),
            "camera_height_m": units.get("camera_height_m"),
            "depth_shape": list(depth.shape),
            "depth_min_m": round(float(metres[metres > 0.05].min()), 2) if (metres > 0.05).any() else None,
            "depth_max_m": round(float(metres.max()), 2),
            "floor_below_camera_m": round(-td.floor_y, 2),
            "ahead_m": round(walk_ahead, 2),
            "sightline_m": round(td.ahead_m(), 2),
            "floor_blind_m": round(td.floor_blind_m, 2),
            "widest_bearing_deg": round(_math.degrees(bearing), 1),
            "widest_m": round(widest, 2),
            "free_pct": round(100 * float((grid == dm.FREE).sum()) / total),
            "open_pct": round(100 * float((grid == dm.OPEN).sum()) / total),
            "occupied_pct": round(100 * float((grid == dm.OCCUPIED).sum()) / total),
            "unknown_pct": round(100 * float((grid == dm.UNKNOWN).sum()) / total),
            "cell_m": dm.CELL_M,
            "range_cap_m": cap_m,
            # True while candidate 1 is a place already committed to rather than
            # a fresh pick — the distance on it should be counting down.
            "goal_held": held,
            "candidates": [
                {"n": i + 1, "kind": w.kind, "where": w.describe(),
                 "angle_deg": round(_math.degrees(w.angle), 1),
                 "distance_m": round(w.distance, 2),
                 "clearance_m": round(w.clearance, 2),
                 "squeeze_m": w.extras.get("squeeze"),
                 "stride_m": (round(w.stride_m, 2) if w.stride_m
                              else w.extras.get("stride_m")),
                 "verified_m": round(w.verified_m, 2),
                 "visible_m": round(w.visible_m, 2),
                 "confidence": w.confidence,
                 "merged_deg": w.extras.get("merged_bearings_deg"),
                 "boxed_in": bool(w.extras.get("boxed_in")),
                 # goto executes the SAFE STRIDE now, so this is what a click
                 # actually spends — quoting the aim over-billed every hop
                 "env_steps": int((w.stride_m or w.distance) / 0.25)}
                for i, w in enumerate(waypoints)
            ],
            "potential_regions": dm.potential_regions(td)[:3],
            "loop": loop_warn,
            "landmarks": [
                {"phrase": s.phrase, "bearing_deg": round(_math.degrees(s.bearing), 1),
                 "distance_m": round(s.distance, 2), "score": round(s.score, 2),
                 "pixels": s.pixels, "where": s.describe()}
                for s in sightings
            ],
            "detector": detector,
            "detector_misses": misses,
            "landmark_words": {s.phrase: s.matched_as for s in sightings
                               if getattr(s, "matched_as", "")},
            "surroundings": dm.surroundings_sentence(td),
        }

    try:
        return await asyncio.to_thread(_work)
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


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


_PHRASE_CACHE: dict[str, list[str]] = {}
_FIXED_KEYWORDS: dict[str, list[str]] | None = None
_FIXED_KEYWORDS_PATH = (
    Path(__file__).resolve().parents[5] / "coding-agent" / "bridges"
    / "keywords" / "rand100_keywords.json")


def _fixed_keywords() -> dict[str, list[str]]:
    """user 2026-08-15: the R2R + RxR rand100 keywords are FIXED — one
    sonnet parse per episode, extracted once to bridges/keywords/, so an
    episode load never spends an LLM call (and never needs ollama). Keyed
    by the exact instruction text; a whitespace-normalised key rides
    alongside for robustness."""
    global _FIXED_KEYWORDS
    if _FIXED_KEYWORDS is not None:
        return _FIXED_KEYWORDS
    table: dict[str, list[str]] = {}
    try:
        data = json.loads(_FIXED_KEYWORDS_PATH.read_text())
        for rows in (data.get("splits") or {}).values():
            for r in rows:
                marks = [str(p) for p in (r.get("landmarks") or []) if p]
                if not marks:
                    continue
                instr = str(r.get("instruction") or "")
                table[instr] = marks
                table[" ".join(instr.split())] = marks
    except Exception:  # noqa: BLE001 — no file, no fixed table
        pass
    _FIXED_KEYWORDS = table
    return table


def _episode_phrases(instruction: str) -> list[str]:
    """Landmark phrases for SAM 3 — the FIXED table first (no model call),
    then the resolver fallback for instructions outside rand100."""
    if instruction in _PHRASE_CACHE:
        return _PHRASE_CACHE[instruction]
    fixed = _fixed_keywords()
    phrases = list(fixed.get(instruction)
                   or fixed.get(" ".join(instruction.split())) or [])
    if not phrases:
        _depth_modules()          # puts coding-agent on sys.path
        try:
            import os as _os

            from eharness.resolver import resolve
            rec = resolve(instruction,
                          model_name="ollama_chat/" + _os.environ.get(
                              "EH_SPLIT_MODEL", "qwen3.5:9b-bf16-std"))
            phrases = list(rec.landmarks or [])
        except Exception:  # noqa: BLE001
            phrases = []
    if not phrases:
        _, lm = _depth_modules()
        phrases = lm.landmark_phrases([instruction])
    _PHRASE_CACHE[instruction] = phrases
    return phrases


def _sweep_frame0(amap, runner, url: str, organ, phrases: list[str],
                  scale_m, dm) -> list[str]:
    """User ruling 2026-08-12 night: a FRESH accumulated map opens with the
    zero-env-step ±60° sweep (same mechanism as the agent bootstrap):
    geometry from nine free renders (0, ±15…±60), SAM on the ±30/±60
    wings with bearings folded and masks rotated into the body frame. The
    heading-up panel is legible the moment the episode loads."""
    seen: list[str] = []
    try:
        pano = runner._call(url, "env_habitat__observe_panorama",  # noqa: SLF001
                            {"trigger": "human_sweep"},
                            {"representation": "views_rgbd", "n_views": 24})
    except Exception:  # noqa: BLE001 — a sweep must never block an analyze
        return seen
    picked = []
    for v in (pano.get("views") or []):
        h = float(v.get("heading_deg") or 0.0)
        sh = h if h <= 180.0 else h - 360.0
        if abs(sh) <= 60.1:
            picked.append((sh, v))
    if not picked:
        return seen
    picked.sort(key=lambda t: t[0])
    # PER-VIEW fusion — the merged-pano single fuse left singly-seen cells
    # below the ±0.5 known threshold and the load map knew 78% less than a
    # real out-and-back on the same pixels (A/B, EP3 start pose)
    dm.fuse_sweep_views(amap, picked, scale_m=scale_m,
                        range_cap_m=dm.RANGE_CAP_M)
    # the frontal view's floor plane, for wing-mask projection below
    td = None
    for sh, v in picked:
        if abs(sh) < 1.0:
            dep0 = dm.decode_panorama_depth(v, scale_m)
            if dep0 is not None:
                try:
                    td = dm.build_topdown(dep0, range_cap_m=dm.RANGE_CAP_M,
                                          scale_m=1.0)
                except Exception:  # noqa: BLE001
                    td = None
            break
    if organ is None or not phrases:
        return seen
    import math as _m
    for sh, v in picked:
        if abs(abs(sh) - 30.0) > 1.0 and abs(abs(sh) - 60.0) > 1.0:
            continue
        rgb = v.get("rgb_base64")
        dep = dm.decode_panorama_depth(v, scale_m)
        if not isinstance(rgb, str) or dep is None:
            continue
        try:
            sights = organ.sightings(rgb, dep, phrases, scale_m=1.0)
        except Exception:  # noqa: BLE001
            continue
        hr = _m.radians(sh)
        ch, s_h = _m.cos(hr), _m.sin(hr)
        for s in sights:
            seen.append(f"{s.phrase} {sh:+.0f}°")
            if s.mask is None or td is None:
                continue
            xs, ys = dm.project_mask(s.mask, dep, td.floor_y, scale_m=1.0)
            if xs.size:
                amap.stamp_semantic(s.phrase, xs * ch + ys * s_h,
                                    ys * ch - xs * s_h,
                                    weight=float(s.score))
    return seen


@router.post("/episode/{index}/load")
async def load_episode(index: int, req: LoadRequest) -> dict:
    try:
        out = await asyncio.to_thread(
            _runner().load_episode, index, req.rgb_resolution, req.depth_resolution,
            req.cam_height_m, req.depth_max_m, req.depth_normalize,
        )
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc
    # Per-episode SAM phrases from the ONE resolver (LLM, whole detector-
    # friendly nouns). The stoplist scraper split "kitchen area" into
    # "kitchen" + "area" and offered "little" as a keyword (user report
    # 2026-08-12 night) — a word list is not a landmark list. Advisory:
    # the box stays editable; scraper only as the offline fallback.
    try:
        out["suggested_phrases"] = await asyncio.to_thread(
            _episode_phrases, str(out.get("instruction", "")))
    except Exception:  # noqa: BLE001 — a phrase hint must never block a load
        out["suggested_phrases"] = []
    return out


@router.post("/step")
async def step(req: StepRequest) -> dict:
    try:
        return await asyncio.to_thread(_runner().step, req.action)
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc


class GotoRequest(BaseModel):
    place: int                       # 1-based, as shown on the annotated frame
    range_cap_m: float | None = None  # None = the organ's own cap, as /depth-analysis


@router.post("/goto")
async def goto(req: GotoRequest) -> dict:
    """Walk to one of the numbered places the depth organ is offering.

    Same shape as /step so the browser updates identically — the difference is
    that one call covers several metres along a line that was measured clear,
    which is the whole reason the geometric proposer exists."""

    def _offer(td, fresh):
        """The list the DRIVER is looking at, not the raw proposal.

        The held commitment is candidate 1 on the panel; without this it would
        not be candidate 1 here, and clicking the circle marked 1 would walk to
        a different place than the one drawn."""
        goal = _MAP.get("goal")
        amap = _MAP.get("map")
        if goal is None or amap is None:
            return fresh
        return goal.apply(amap, td, fresh)

    try:
        return await asyncio.to_thread(
            _runner().goto, req.place, req.range_cap_m, _offer)
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


@router.post("/clear")
async def clear() -> dict:
    """Wipe the live split for the next tester (archives first as a safety net).
    Returns the now-empty status plus `archived_to` (the saved snapshot name)."""
    try:
        return await asyncio.to_thread(_runner().clear)
    except _HANDLED as exc:
        raise HTTPException(409, str(exc)) from exc
