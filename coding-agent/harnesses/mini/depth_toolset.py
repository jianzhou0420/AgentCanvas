"""Depth-waypoint toolset — the habitat surface PLUS geometric goto().

A flexible action space, not a replacement one: the model keeps every primitive
it had (``step`` for fine alignment, turning, and STOP) and gains ``goto(place)``
for covering ground. Long stride when the way is open, single steps when the
last metre matters — the choice is the model's, which is the whole point.

The candidates come from ONE 90° depth frame through ``eharness.depthmap``.
Against the learned predictor this is better-informed on three counts that are
verifiable in-repo, not matters of taste:

  * range — the predictor's heatmap is 120 angles × 12 bins of 0.25 m, a hard
    3.00 m ceiling; measured free space regularly reaches 5–6 m;
  * scale — the predictor is fed ``depth_base64``, which env_habitat produces by
    PER-FRAME min-max normalisation to 8 bit, so "3 metres" is not expressible
    in its input at all;
  * safety — ``step_hightolow`` does not path-plan. It rotates once, then
    blind-walks ``int(d/0.25)`` forward primitives, sliding on collision. A
    learned peak carries no reachability guarantee; a point derived from
    measured free space does, and this toolset refuses to offer one that is not
    reachable in a straight line.

SAM 3 landmark grounding (``sam_url``) is advisory garnish: with the detector
off, missing or slow the toolset degrades to pure geometry and the episode
continues. Only ONE server is required — the depth organ runs in-process, so
unlike the wp toolset there is no second predictor host to keep alive.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

_CA = str(Path(__file__).resolve().parents[2])
if _CA not in sys.path:                     # coding-agent/ on the path
    sys.path.insert(0, _CA)

from eharness import depthmap as dm  # noqa: E402
from eharness.events import FAIL_CLOSED, NavigationEvent  # noqa: E402
from eharness.places import PlaceMemory  # noqa: E402
from eharness.landmarks import (  # noqa: E402
    LandmarkOrgan,
    LandmarkRegister,
    landmark_phrases,
)
from eharness.landmarks import surroundings_sentence as landmark_sentence  # noqa: E402

from toolset import HabitatToolSet, ToolResult, png_part, text_part  # noqa: E402

# The env's own primitives, for dead reckoning. Habitat's R2R-CE config: one
# MOVE_FORWARD is 0.25 m, one TURN is 15°.
STEP_M = 0.25
TURN_RAD = math.radians(15.0)
TURN_DEG = 15.0
# §14.8: with event-driven SAM (sam_intermediate_every=0) this is the
# low-frequency backstop — the detector still runs at least every Nth
# sensor frame even when no event asks for it.
SAM_LOW_CADENCE = 4
# mean |Δdepth| in metres between consecutive sensed frames above which the
# scene is considered to have visibly changed (doorway crossed, room
# entered) — one of the event triggers for the detector.
SCENE_CHANGE_M = 0.6

# §14.9: the tool-surface facts live in the CapabilityManifest — ONE copy,
# shared with the MCP bridge, the SDK allowlist, the prompts and the tests.
from eharness.capabilities import (  # noqa: E402
    DWP_STEP_DESC,
    FACE_DESC,
    FACE_SCHEMA,
    GOTO_DESC,
    GOTO_SCHEMA,
    OBSERVE_DESC_DWP,
)


def _turn_call(deg: float) -> str:
    """The step batch that turns ≈deg (left positive): 'step([2,2,2])'.
    face left the model surface (2026-08-12); every turn hint speaks in
    the verbs the model actually has."""
    n = max(1, min(12, int(round(abs(deg) / 15.0))))
    a = "2" if deg > 0 else "3"
    return "step([" + ",".join([a] * n) + "])"


class DepthWaypointToolSet(HabitatToolSet):
    """observe (annotated) + step (inherited) + goto (geometric waypoints)."""

    def __init__(
        self,
        server_url: str,
        *,
        instruction: str = "",
        segments: list[str] | None = None,
        terminate: str = "",
        bare: bool = True,
        step_budget: int = 500,
        turn_budget: int = 0,
        pano_view_px: int = 0,
        live_dir: Path | None = None,
        sam_url: str = "",
        landmark_every: int = 3,
        range_cap_m: float = dm.RANGE_CAP_M,
        views: int = 1,
        hide_observe: bool = True,
        max_fwd_per_step: int = 2,
        places: int = 0,
        sam_intermediate_every: int = 1,
        traj_archive: bool = True,
        executor: str = "",
        bootstrap_sweep: bool = True,
    ) -> None:
        super().__init__(server_url, bare=bare, step_budget=step_budget,
                         turn_budget=turn_budget, pano_view_px=pano_view_px,
                         live_dir=live_dir)
        self.range_cap_m = range_cap_m
        self.views = max(1, int(views))
        self.max_fwd_per_step = max(1, int(max_fwd_per_step))
        self._waypoints: list[dm.Waypoint] = []
        # §16.2: the candidate table's EPOCH — bumped every time the numbers
        # are re-proposed. It rides the artifact, the status json and every
        # goto receipt, so a monitor (or a test) can prove which table a
        # number referred to. The table itself lives ONLY on this instance;
        # bootstrap images and goto targets are the same object by design.
        self._candidate_epoch = 0
        self._last_td: dm.TopDown | None = None
        self._gotos = 0
        # The accumulated map. Fed from the harness's OWN commands (every
        # primitive it issues, every stride it executes) and corrected against
        # its own past observations — no simulator pose, nothing external. What
        # it buys the model is one sentence: where the things it has already
        # walked past are NOW, in its own egocentric words. That sentence is
        # the evidence the milestone judge kept having to guess at.
        self.amap = dm.AnchorMap()
        # §21.6: candidate identity. propose() stays the per-frame safety
        # generator; the registry gives its output standing anchor-frame
        # tracks, so the same physical exit keeps the same circle across
        # frames and heading changes instead of sliding with the depth.
        self.wreg = dm.WaypointRegistry()
        # The collapse operator. When on, the metric recall sentence is
        # REPLACED (not supplemented) by place-based memory — that substitution
        # IS the experiment: does a memory that asserts less survive odometry
        # drift better than one that is precise and sometimes confidently wrong.
        # See design/place-memory.html.
        self.places = PlaceMemory() if places else None
        self._prev_depth = None      # last frames, for the did-I-actually-move test
        self._prev_rgb = None        # RGB votes too — depth cannot see a wall-slide
        self._pending_fwd_m = 0.0    # metres the odometry has claimed but not confirmed

        self.phrases = landmark_phrases(segments or ([instruction] if instruction else []),
                                        terminate)
        self.landmarks = LandmarkOrgan(sam_url) if sam_url else None
        self.register = LandmarkRegister()
        self.landmark_every = max(1, landmark_every)
        # Detections from THIS frame only — empty on non-cadence frames. A
        # mask is pixels of one RGB frame; re-projecting it through a LATER
        # depth image at a LATER pose stamped phantom landmarks on two of
        # every three frames at the default cadence (§3 P0, stale mask).
        # History lives in the register ledger and the semantic layer, never
        # as a reusable mask.
        self._sightings: list = []
        self._looks = 0
        self._seen_at: dict[str, int] = {}   # phrase → look index last detected
        self._scale_m: float | None = None   # declared depth units, from the env
        # ── the micro-trajectory data plane (§4 of the revision plan) ──
        # Every executed primitive senses, maps and archives; the model only
        # ever sees the endpoint. sam_intermediate_every=1 runs the detector
        # on every intermediate frame too (correctness default; a latency
        # ablation may raise it — but a skipped frame is skipped, NEVER a
        # reused stale mask).
        # §14.8: 0 = event-driven (post-turn first frame, scene transition,
        # suspected landmark, and a SAM_LOW_CADENCE backstop); N>0 = every
        # Nth SENSOR frame (1 is the correctness upper bound). The cadence
        # counts sensor frames, never archive writes — recording must not
        # schedule the detector.
        self.sam_intermediate_every = max(0, int(sam_intermediate_every))
        self.traj_archive = bool(traj_archive)
        self._traj_n = 0                     # trajectory frame counter
        self._sensor_frame = 0               # §14.8: archive-independent
        self._sensed_at_step = -1            # env step the executor last sensed
        self._interrupt_reason = ""          # why a leg stopped early, if it did
        # §14.7: the per-action event log. Cleared by _begin_action; every
        # reflex and detector appends NavigationEvents; requires_interrupt
        # ones stop the leg, the rest surface in endpoint telemetry.
        self.events: list[NavigationEvent] = []
        self._events_consumed = 0            # interrupts already acted on
        self._current_action = ""            # high-level action id (identity)
        # §14.14 identity: who is driving this toolset — the in-process mini
        # path passes "mini", bridge processes inherit EH_EXECUTOR from the
        # adapter that spawned them
        self._executor = executor or os.environ.get("EH_EXECUTOR", "")
        # user ruling 2026-08-12 night: open the episode with a zero-step
        # ±60° LOOK before the first decision (see _bootstrap_sweep)
        self.bootstrap_sweep = bool(bootstrap_sweep)
        self._sense_failure: str | None = None   # §14.5 failure class
        self._yaw_pending_sam = False        # post-turn first frame → SAM
        self._sam_stats = {"calls": 0, "latency_s": 0.0,
                           "skips": {}}      # per-action detector accounting
        self._prev_reg_score: float | None = None
        self._leg_openings: list = []        # openings at leg start (topology)
        self._terminate = terminate          # near-goal phrase matching
        # §14.6: the persistent route intent — survives observations and
        # face() turns; dies only on arrival, cancellation by negative
        # evidence, or the model committing somewhere else entirely.
        self.route: dm.RouteCandidate | None = None
        self._route_history: list[dict] = []
        self._staging_of: dict[int, dict] = {}   # place idx → potential region
        # The model's chosen destination, held still in the anchor frame while
        # the body approaches it. Adopted in goto() — what the MODEL picked —
        # not on proposal; a commitment the model never made is not one.
        self.goal = dm.WaypointGoal()
        # §7: the resident circle detector — trail revisits, path-vs-net,
        # sector repetition, map growth. Deterministic; warnings reach the
        # model as telemetry text and map highlights, never as a locked door.
        self.loop = dm.LoopMonitor()

        # observe is re-registered (annotated view + candidate list); step and
        # look_around stay exactly as HabitatToolSet built them, so the model
        # keeps its primitives and the wrapper still recognises the surface.
        self._replace_tool("observe", OBSERVE_DESC_DWP, self._tool_observe)
        self._replace_tool("step", DWP_STEP_DESC, self._tool_step)
        self._register("goto", GOTO_DESC, GOTO_SCHEMA, self._tool_goto)
        self._register("face", FACE_DESC, FACE_SCHEMA, self._tool_face)
        # face is NOT on the model surface (user ruling 2026-08-12 evening:
        # goto + step only; step's 2/3 turns cover direction changes). The
        # handler stays callable — the human tab and internal probes use it.
        self._schemas = [s for s in self._schemas if s["name"] != "face"]
        if hide_observe:
            # Looking is not a decision, so it should not cost a turn. Every
            # action already comes back with a fresh view, the current places,
            # and what the detector sees — so `observe` is withdrawn from the
            # model's menu and survives only as the internal read the harness
            # calls for itself. Across three EP0 runs the model spent a third
            # of its turns re-looking instead of moving; this deletes that
            # failure mode rather than advising against it.
            self._schemas = [s for s in self._schemas if s["name"] != "observe"]

    def _replace_tool(self, name: str, description: str, handler) -> None:
        for schema in self._schemas:
            if schema["name"] == name:
                schema["description"] = description
                break
        self._handlers[name] = handler

    def set_route(self, segments: list[str], terminate: str = "",
                  landmarks: list[str] | None = None) -> None:
        """Adopt the route once the harness has split it (the env is built
        first). Detector-friendly phrases from the splitter win outright over
        words scraped from the instruction: SAM 3 finds nothing for "bar" and
        segments the counter immediately for "bar counter" — same frame, same
        model, measured on EP0. Scraping is only the fallback."""
        self._terminate = terminate or self._terminate
        if landmarks:
            self.phrases = [p for p in landmarks if p][:5]
            return
        phrases = landmark_phrases(segments, terminate)
        if phrases:
            self.phrases = phrases

    # ── §14.7: the per-action event plane ────────────────────────────────
    def _begin_action(self, action_id: str) -> None:
        """Reset the per-action planes: event log, detector accounting,
        failure class. Called once at the top of every model-facing verb."""
        self._current_action = action_id
        self.events = []
        self._events_consumed = 0
        self._sense_failure = None
        # every model-facing verb starts with a CLEAN interrupt plane — the
        # notches==0 yaw path never enters _execute_primitives, and a stale
        # reason from the previous action falsely aborted every subsequent
        # <15°-off goto with a fabricated what_happened (review P0)
        self._interrupt_reason = ""
        self._sam_stats = {"calls": 0, "latency_s": 0.0, "skips": {}}

    def _emit_event(self, type_: str, severity: str, *,
                    requires_interrupt: bool = False,
                    evidence: dict | None = None) -> NavigationEvent:
        ev = NavigationEvent(
            type=type_, severity=severity, frame_id=self._sensor_frame,
            map_version=int(self.amap.updates), evidence=evidence or {},
            requires_interrupt=requires_interrupt)
        self.events.append(ev)
        self._live_log({"event": ev.as_dict()})
        return ev

    def _take_interrupt(self) -> NavigationEvent | None:
        """The first not-yet-consumed event that demands the leg stop."""
        for i in range(self._events_consumed, len(self.events)):
            if self.events[i].requires_interrupt:
                self._events_consumed = i + 1
                return self.events[i]
        self._events_consumed = len(self.events)
        return None

    # ── §16.2/§16.7: the RESIDENT bootstrap ──────────────────────────────
    BOOT_SWEEP_DEG = 60.0            # look ±60° before the first decision
    BOOT_SWEEP_SAM_DEG = (30.0, 60.0)   # wing headings that also run SAM

    def _bootstrap_sweep(self) -> dict[str, Any]:
        """User ruling 2026-08-12 night: before the first decision, LOOK
        ±60° — 'turn left four notches mapping each, come back, turn right
        four notches mapping each, come back, all free'. observe_panorama
        renders at the pose WITHOUT stepping the simulator, so this is the
        honest zero-env-step version of that plan: the same pixels a
        physical out-and-back would capture, with no motion, no budget and
        no odometry to unwind.

        Geometry fuses ALL nine headings (0, ±15…±60 — build_topdown_pano
        rotates each wedge into the body frame); the detector runs on the
        frontal (the bootstrap sense already did it) plus the ±30/±60
        wings — adjacent 15° views share ~5/6 of their pixels, and five
        phrases × eight views would park the boot for minutes for no new
        evidence. Wing sightings fold their view heading into the bearing
        (ledger) and rotate their mask points into the body frame (semantic
        map), so 'kitchen area 40° to your right' is on the record before
        the model chooses its first move — the EP3 rescue: the kitchen was
        just right of the opening frame, and a frontal-only boot never saw
        it."""
        out: dict[str, Any] = {"views": 0, "sam_views": 0, "seen": []}
        if not self.bootstrap_sweep:
            return out
        try:
            pano = self._call2("env_habitat__observe_panorama",
                               {"trigger": "bootstrap_sweep"},
                               {"representation": "views_rgbd", "n_views": 24})
        except Exception:  # noqa: BLE001 — a sweep must never kill a boot
            return out
        picked: list[tuple[float, dict]] = []
        for v in (pano.get("views") or []):
            h = float(v.get("heading_deg") or 0.0)
            sh = h if h <= 180.0 else h - 360.0
            if abs(sh) <= self.BOOT_SWEEP_DEG + 0.1:
                picked.append((sh, v))
        if not picked:
            return out
        picked.sort(key=lambda t: t[0])
        # §21.2 provenance on the record: exactly which headings fed depth
        # geometry and which ran the detector — the three failure modes
        # ("stale PNG", "single-vote cell under threshold", "proposer never
        # saw the sweep map") become distinguishable from the artifact.
        out["depth_headings"] = [round(sh) for sh, _ in picked]
        out["sam_headings"] = []
        # frame-0 wings straight into the long-term map, PER VIEW — the
        # merged-pano single fuse left singly-seen cells below the known
        # threshold and the sweep map knew 78% less than a real
        # out-and-back (A/B on the EP3 start pose). No register: the map
        # is one wedge old, there is nothing to match against.
        fused = dm.fuse_sweep_views(self.amap, picked,
                                    scale_m=self._scale_m,
                                    range_cap_m=self.range_cap_m)
        if fused:
            self.loop.note_growth(int((np.abs(self.amap.logodds) > 0.5).sum()))
        out["views"] = len(picked)
        if self.landmarks is not None and self.phrases:
            for sh, v in picked:
                if not any(abs(abs(sh) - d) < 1.0
                           for d in self.BOOT_SWEEP_SAM_DEG):
                    continue
                rgb = v.get("rgb_base64")
                dep_m = dm.decode_panorama_depth(v, self._scale_m)
                if not isinstance(rgb, str) or dep_m is None:
                    continue
                self._sensor_frame += 1
                out["sam_views"] += 1
                out["sam_headings"].append(round(sh))
                try:
                    sights = self.landmarks.sightings(rgb, dep_m,
                                                      self.phrases,
                                                      scale_m=1.0)
                except Exception:  # noqa: BLE001 — semantics stay advisory
                    continue
                hr = math.radians(sh)
                ch, s_h = math.cos(hr), math.sin(hr)
                for s in sights:
                    b = hr + s.bearing
                    s.bearing = math.atan2(math.sin(b), math.cos(b))
                    s.frame_hint = f"bootstrap sweep, {sh:+.0f}°"
                    out["seen"].append(f"{s.phrase} {sh:+.0f}°")
                self.register.record(sights, self.steps_taken)
                # floor plane from the frontal frame the bootstrap sense
                # already built — same camera height in every wing view
                _floor = (self._last_td.floor_y
                          if self._last_td is not None else None)
                if _floor is not None:
                    for s in sights:
                        if s.mask is None:
                            continue
                        xs, ys = dm.project_mask(s.mask, dep_m, _floor,
                                                 scale_m=1.0)
                        if xs.size:
                            # view frame → body frame (view is rotated LEFT
                            # by sh): xb = x·cos h + y·sin h, yb = y·cos h − x·sin h
                            xb = xs * ch + ys * s_h
                            yb = ys * ch - xs * s_h
                            self.amap.stamp_semantic(s.phrase, xb, yb,
                                                     weight=float(s.score))
        self._live_log({"bootstrap_sweep": out})
        return out

    def bootstrap(self) -> dict[str, Any]:
        """The first look, taken by the SAME instance the model will drive.

        Frame 0 is sensed, fused into the AnchorMap, archived as a
        MicroObservation and counted as a sensor frame (§16.7 — the first
        frame is the localisation baseline, not a throwaway); candidates
        are proposed into THIS instance's table; the snapshot publishes;
        and ONE artifact (bootstrap.json + PNGs in the live dir) records
        the numbers with their candidate_epoch/map_version. Executors
        encode this artifact for their provider — they never re-derive it,
        so the number the model reads in the first image is a number
        goto() actually accepts (§16.2's split-brain fix).

        Works with SAM off and landmarks empty: geometry alone still
        yields current RGB + accumulated map + candidates. Zero env motion
        steps; exactly one sensor frame."""
        self._begin_action("bootstrap")
        self._sense_primitive("bootstrap", -1, 0, 0.0, False)
        cells_before = int((np.abs(self.amap.logodds) > 0.5).sum())
        # the ±60° zero-step sweep runs BETWEEN the frontal sense and the
        # candidate proposal, so the first candidates and IMAGE 2 already
        # carry the wings — and the MAP MEMORY lines already name whatever
        # the detector saw off-axis ("kitchen area 40° to your right")
        sweep = self._bootstrap_sweep()
        cells_after = int((np.abs(self.amap.logodds) > 0.5).sum())
        # §21.2 the FINALIZE: one publish, from the frozen frontal frame,
        # on the post-sweep planning view. _reuse_frame skips the second
        # env observe AND (via the already-sensed gate) any re-fuse, so
        # between here and the artifact write map_version cannot move —
        # the numbers the model reads ARE the numbers goto() accepts, on
        # the map the sweep just finished building. Pre-sweep candidates
        # never existed as a callable menu.
        view = self._tool_observe(_internal=True, _reuse_frame=True)
        parts = list(view.content)
        art: dict[str, Any] = {
            "candidate_epoch": self._candidate_epoch,
            "map_version": int(self.amap.updates),
            "proposal_map_version": int(self.amap.updates),
            "sensor_frame": self._sensor_frame,
            "trust": {"sweep_trust": bool(self.amap.sweep_trust),
                      "trusted": bool(self.amap.trusted)},
            "known_cells": {"before_sweep": cells_before,
                            "after_sweep": cells_after,
                            "published": int(
                                (np.abs(self.amap.logodds) > 0.5).sum())},
            "sweep": {"views": sweep.get("views", 0),
                      "sam_views": sweep.get("sam_views", 0),
                      "depth_headings": sweep.get("depth_headings", []),
                      "sam_headings": sweep.get("sam_headings", []),
                      "seen": sweep.get("seen", [])},
            "env_step": self.steps_taken,
            "candidates": [
                {"n": i + 1, "where": w.describe(), "kind": w.kind,
                 "distance_m": round(w.distance, 2),
                 "angle_deg": round(math.degrees(w.angle), 1),
                 "clear_beyond_m": round(w.continuation_m, 2),
                 **({"track_id": int(w.extras["track_id"])}
                    if w.extras.get("track_id") else {})}
                for i, w in enumerate(self._waypoints)],
            "texts": [str(p.get("text", "")) for p in parts
                      if isinstance(p, dict) and p.get("type") == "text"],
            "images": {},
        }
        if self.live_dir is not None:
            try:
                from eharness.wrapper import _png_from_part
                self.live_dir.mkdir(parents=True, exist_ok=True)
                names = iter(("current", "map"))
                for p in parts:
                    png = _png_from_part(p) if isinstance(p, dict) else None
                    if png is None:
                        continue
                    try:
                        key = next(names)
                    except StopIteration:
                        break
                    fn = f"bootstrap_{key}.png"
                    (self.live_dir / fn).write_bytes(png)
                    art["images"][key] = fn
                tmp = self.live_dir / "bootstrap.json.tmp"
                tmp.write_text(json.dumps(art, ensure_ascii=False, indent=1))
                os.replace(tmp, self.live_dir / "bootstrap.json")
            except Exception:  # noqa: BLE001 — a monitor file must not kill boot
                pass
        self._live_log({"bootstrap": True,
                        "candidates": len(art["candidates"]),
                        "candidate_epoch": art["candidate_epoch"],
                        "sensor_frame": art["sensor_frame"]})
        return {"artifact": art, "parts": parts}

    def opening_survey(self, n_views: int = 4, *,
                       skip_frontal: bool = False) -> dict:
        """One 360° look before the first move, recorded in the ledger.

        `observe_panorama` renders at the agent's pose without stepping the
        simulator, so a full turn of the world costs ZERO env steps — and the
        detector run on it establishes what was HERE at the start. That baseline
        is what the verifier has been missing: a near-perfect episode was vetoed
        for "has not passed the pool" when the pool had been 0.9 m away at the
        beginning and simply left behind. A ledger entry from step 0 makes that
        checkable instead of arguable."""
        if self.landmarks is None or not self.phrases:
            return {"views": 0, "seen": []}
        try:
            pano = self._call2("env_habitat__observe_panorama", {"trigger": "survey"},
                               {"representation": "views_rgbd", "n_views": n_views})
        except Exception:  # noqa: BLE001 — a survey must never end an episode
            return {"views": 0, "seen": [], "error": "panorama unavailable"}
        seen: list[str] = []
        for view in (pano.get("views") or []):
            rgb = view.get("rgb_base64")
            if not rgb:
                continue
            heading = float(view.get("heading_deg") or 0.0)
            # each panorama view carries its OWN depth (uint16, normalized×1000
            # — decode_panorama_depth knows the conversion); passing None here
            # made sightings() bail on its first line and the survey silently
            # produced nothing at all
            view_depth = dm.decode_panorama_depth(view)
            if view_depth is None:
                continue
            for s in self.landmarks.sightings(rgb, view_depth, self.phrases):
                # the sighting's bearing is inside its own view; fold in the
                # view's heading so the ledger speaks in the robot's frame —
                # WRAPPED to (−π, π], or a 270° view heading pushes a right-
                # side landmark past +155° and the sentence calls it "behind"
                b = math.radians(heading) + s.bearing
                s.bearing = math.atan2(math.sin(b), math.cos(b))
                s.frame_hint = f"opening survey, {heading:.0f}° from the start heading"
                self.register.record([s], 0)
                seen.append(f"{s.phrase} {heading:.0f}°")
        self._live_log({"opening_survey": True, "views": len(pano.get("views") or []),
                        "seen": seen})
        # the pictures themselves, for the model's first message: the frontal
        # view it is actually facing, then the rest of the turn labelled by
        # heading. Words about a room are no substitute for seeing it.
        # §16.2: when bootstrap() already produced the frontal (the RESIDENT
        # candidates), the survey must not re-derive a SECOND frontal with a
        # SECOND candidate table — skip_frontal keeps the panorama ledger
        # and the extra views only.
        parts: list = []
        if not skip_frontal:
            ego = self._call("env_habitat__observe_egocentric", {})
            frontal = ego.get("rgb")
            if isinstance(frontal, str):
                fp = base64.b64decode(frontal)
                # frame 0 keeps the declared-units contract too — the old
                # second observe call dropped depth_units and frame 0 fell
                # back to the depth.max() guess, which is exactly the 10×
                # misfire the port exists to prevent
                units0 = ego.get("depth_units") or {}
                scale0 = (float(units0["scale_m"])
                          if units0.get("known") and units0.get("scale_m")
                          else None)
                wp, td = dm.propose_from_depth(ego.get("depth"),
                                               range_cap_m=self.range_cap_m,
                                               scale_m=scale0)
                if td is not None:
                    fp = dm.annotate_rgb(fp, wp, td.floor_y)
                    self._waypoints, self._last_td = wp, td
                    self._candidate_epoch += 1
                parts.append(text_part(
                    "This is what you are facing right now, with the places "
                    "you can walk to numbered on it:"))
                parts.append(png_part(fp))
        for view in (pano.get("views") or []):
            rgb = view.get("rgb_base64")
            if not rgb:
                continue
            h = float(view.get("heading_deg") or 0.0)
            if abs(h) < 1e-6:
                continue                      # already shown as the frontal view
            label = ("behind you" if abs(h - 180) < 30 else
                     f"{h:.0f}° to your left" if h < 180 else
                     f"{360 - h:.0f}° to your right")
            parts.append(text_part(f"Looking {label}:"))
            parts.append(png_part(base64.b64decode(rgb)))
        return {"views": len(pano.get("views") or []), "seen": seen,
                "sentence": self._survey_sentence(), "parts": parts}

    def _survey_sentence(self) -> str:
        """The survey in one plain line, for the state block's first render.

        With `observe` withdrawn the model's FIRST action is otherwise blind —
        it has never seen the room when it decides. EP0 showed exactly that:
        turn 1 was a 45° turn into a wall, and the first frame in the log is
        already step 3. The survey has the answer before the first move, so it
        must be spoken before the first move."""
        parts = []
        for phrase in self.phrases:
            near = self.register.closest(phrase)
            if near is None:
                continue
            deg = math.degrees(near.bearing)
            if abs(deg) < 25:
                where = "straight ahead"
            elif abs(deg) > 155:
                where = "behind you"
            else:
                where = f"to your {'left' if deg > 0 else 'right'}"
            parts.append(f"the {phrase} {where}, about {near.distance:.1f} m away")
        if not parts:
            return ""
        return ("Before your first move I looked all the way round from here. "
                "I can see " + "; ".join(parts) + ".")

    def _call2(self, fn: str, inputs: dict, config: dict | None = None) -> dict:
        """_call with a node config — observe_panorama needs representation/n_views."""
        import requests
        body: dict = {"inputs": inputs}
        if config:
            body["config"] = config
        r = requests.post(f"{self.server_url}/call/{fn}", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["outputs"]

    # ── observe: the habitat view, with the geometry drawn on it ─────────
    def _tool_observe(self, _internal: bool = False,
                      _reuse_frame: bool = False, **_ignored: Any) -> ToolResult:
        if not _internal:
            # only a MODEL-initiated look spends a tool call; the automatic
            # look _augment attaches to every action is the harness's own
            # read, and counting it doubled the budget burn per action
            self._tool_calls += 1
            # identity only — no full _begin_action: an external observe is
            # its own action, but must not wipe another verb's event log
            self._current_action = "observe"
        if _reuse_frame and self._last_td is not None \
                and isinstance(self._prev_rgb, str):
            # §21.2 bootstrap finalize: publish from the FROZEN frame the
            # bootstrap sense already took — no second env observe, no
            # re-build, no chance for the menu to be generated from pixels
            # other than the ones in the artifact. The already_sensed gate
            # below then guarantees no re-register/re-fuse either, so
            # map_version is stable across the whole publish.
            outputs: dict[str, Any] = {}
            rgb_b64 = self._prev_rgb
            depth_arr = None
            td: dm.TopDown | None = self._last_td
        else:
            _reuse_frame = False
            outputs = self._call("env_habitat__observe_egocentric", {})
            rgb_b64 = outputs.get("rgb")
            # Units are DECLARED by the env, not guessed from depth.max(). The
            # heuristic breaks nose-to-wall (a metric frame all under 1 m reads
            # as normalized and inflates 10×); the human path was fixed months
            # before this one, exactly the kind of split §9 asked to close.
            units = outputs.get("depth_units") or {}
            self._scale_m = (float(units["scale_m"])
                             if units.get("known") and units.get("scale_m")
                             else None)
            depth_arr = dm.decode_depth(outputs.get("depth"))
            td = None
        try:
            if _reuse_frame:
                pass
            elif self.views > 1:
                # 360° from N free renders. observe_panorama is a pure read
                # (get_observations_at never steps the sim), so full coverage
                # costs zero env steps — only render time. Behind a knob
                # because the frontal condition is the ablation baseline.
                pano = self._call2("env_habitat__observe_panorama",
                                   {"trigger": "dwp"},
                                   {"representation": "views_rgbd",
                                    "n_views": self.views})
                td = dm.build_topdown_pano(pano.get("views") or [],
                                           range_cap_m=self.range_cap_m,
                                           scale_m=self._scale_m)
            elif depth_arr is not None:
                td = dm.build_topdown(depth_arr, range_cap_m=self.range_cap_m,
                                      scale_m=self._scale_m)
        except Exception:  # noqa: BLE001 — a broken frame must not end the episode
            td = None

        # §10.5: the map is brought fully up to date BEFORE anything is
        # proposed — pending-motion correction, then registration, then
        # fusion. The old order proposed from the raw frame first, so the
        # accumulated map never informed a single candidate.
        #
        # §4.2: when the primitive executor has ALREADY sensed this very env
        # step (per-primitive fusion), the endpoint observe must not vote the
        # same frame into the map a second time, and must not run the
        # detector twice on it — it reuses what the executor sensed and only
        # rebuilds the model-facing outputs.
        already_sensed = (self._sensed_at_step >= 0
                          and self._sensed_at_step == self.steps_taken)
        if td is not None and not already_sensed:
            # Did the body actually move? A commanded forward that a wall ate
            # still incremented the odometry, and that error is ONE-SIDED — it
            # only ever overstates progress, so it never averages out. Measured
            # on three episodes: 60 of 168 forwards moved the body zero, one
            # episode 40 of 56. Take the metres back before anything is stamped.
            # frames_still's tolerance is METRES; compare in metres, or the
            # same threshold means 2 mm on a metric frame and 2 cm on a
            # normalized one and the blocked-step detector changes species
            # with the rig configuration.
            depth_m = None
            if depth_arr is not None:
                # with DECLARED units convert to metres; with unknown units
                # compare RAW frames — per-frame re-guessing let a nose-to-
                # wall frame flip convention mid-pair and the blocked-step
                # rollback misfired (audit P2)
                depth_m = (dm.to_metres(depth_arr, self._scale_m)[0]
                           if self._scale_m is not None else depth_arr)
            if self._pending_fwd_m > 0 and dm.frames_still(
                    self._prev_depth, depth_m,
                    prev_rgb=self._prev_rgb, rgb=rgb_b64):
                self.amap.retract(self._pending_fwd_m)
                self._live_log({"blocked": round(self._pending_fwd_m, 2),
                                "note": "view unchanged — odometry rolled back"})
            self._pending_fwd_m = 0.0
            self._prev_depth = depth_m
            self._prev_rgb = rgb_b64
            self.amap.register(td)
            self.amap.fuse(td)
            self.loop.note_growth(int((np.abs(self.amap.logodds) > 0.5).sum()))
            self.loop.note_frame(dm.frame_signature(rgb_b64),
                                 (self.amap.px, self.amap.py))

        # Landmarks are advisory and slow (~2.5 s per phrase), so they run on
        # a cadence rather than every look. Detections belong to THIS frame
        # alone: mask, depth and pose all from the same instant, projected and
        # stamped here and never again (§3 P0 — the stale-mask fix). A frame
        # the executor already ran the detector on keeps those sightings.
        if not already_sensed:
            self._sightings = []
        if (not already_sensed
                and self.landmarks is not None and self.phrases
                and self._looks % self.landmark_every == 0):
            self._sightings = self.landmarks.sightings(
                rgb_b64, outputs.get("depth"), self.phrases,
                scale_m=self._scale_m)
            for s in self._sightings:
                self._seen_at[s.phrase] = self._looks
            self.register.record(self._sightings, self.steps_taken)
            for phrase in self.phrases:
                if not any(s.phrase == phrase for s in self._sightings):
                    self.register.record_miss(phrase)
            if td is not None and depth_arr is not None:
                for s in self._sightings:
                    if s.mask is None:
                        continue
                    xs, ys = dm.project_mask(s.mask, depth_arr, td.floor_y,
                                             scale_m=self._scale_m)
                    if xs.size:
                        self.amap.stamp_semantic(s.phrase, xs, ys,
                                                 weight=float(s.score))
        self._looks += 1

        # Propose on the PLANNING VIEW — the current wedge backed by trusted
        # memory (current OCCUPIED always wins; remembered floor fills what
        # the camera cannot currently see) — then keep the model's chosen
        # destination steady across frames instead of re-inventing it.
        pv = self.amap.planning_view(td) if td is not None else None
        fresh = dm.propose(pv) if pv is not None else []
        waypoints = (self.goal.apply(self.amap, pv, fresh, adopt=False)
                     if pv is not None else [])
        # §global-dwp: MEMORY candidates ride behind the fresh ones — the
        # corridor you SAW on the right is a numbered option even though
        # the camera no longer points at it. goto's turn-then-revalidate
        # re-earns them with fresh eyes; a wrong memory costs a look, not
        # a walk.
        try:
            mem = dm.propose_from_memory(self.amap, waypoints)
        except Exception:  # noqa: BLE001 — memory must never break a look
            mem = []
        waypoints = waypoints + mem
        # §21.6: same physical exit → same circle. The registry snaps
        # matched candidates onto their standing anchor centre (identity
        # from memory, executability re-earned from THIS frame) and culls
        # the ones whose fixed centre cannot be verified from here.
        try:
            waypoints = self.wreg.reconcile(self.amap, pv, waypoints,
                                            epoch=self._candidate_epoch + 1)
        except Exception:  # noqa: BLE001 — identity must never break a look
            pass

        png = base64.b64decode(rgb_b64) if isinstance(rgb_b64, str) else b""
        # §21.8: a number the model can CALL must be a circle it can SEE.
        # Fresh candidates that fail the shared projection gate (too close,
        # FOV edge, behind) leave the menu — their track survives in the
        # registry and comes back the moment the camera covers it. The
        # explicitly-remembered kind is the one sanctioned map-only entry
        # (§20.4's contract: a number only on the map is a remembered
        # place; goto turns you to it).
        hidden_commit = ""
        if png and td is not None:
            try:
                from PIL import Image as _PI
                _W, _H = _PI.open(BytesIO(png)).size
                kept: list[dm.Waypoint] = []
                for w in waypoints:
                    if w.kind == "remembered" or dm.is_waypoint_visible(
                            w, td.floor_y, _W, _H):
                        kept.append(w)
                    elif "already heading here" in (w.note or ""):
                        d = math.degrees(w.angle)
                        hidden_commit = (
                            "Your committed place is just outside this "
                            f"frame ({abs(d):.0f}° to your "
                            f"{'left' if d > 0 else 'right'}) — "
                            f"{_turn_call(d)} turns to face it.")
                waypoints = kept
            except Exception:  # noqa: BLE001
                pass
        self._waypoints, self._last_td = waypoints, td
        self._candidate_epoch += 1

        # user 2026-08-15: a numbered place the photo CANNOT show still gets
        # its bearing spoken — "place 3 is not in this photo, ~69° to your
        # left" — so the menu and the picture never silently disagree.
        off_camera: list[str] = []
        if png and td is not None:
            try:
                from PIL import Image as _PI
                _W2, _H2 = _PI.open(BytesIO(png)).size
                for i, w in enumerate(waypoints, 1):
                    if dm.is_waypoint_visible(w, td.floor_y, _W2, _H2):
                        continue
                    deg = math.degrees(w.angle)
                    side = ("almost straight ahead but below the frame"
                            if abs(deg) < 12 else
                            f"behind you on the {'left' if deg > 0 else 'right'}"
                            if abs(deg) >= 100 else
                            f"about {abs(deg):.0f}° to your "
                            f"{'left' if deg > 0 else 'right'}")
                    off_camera.append(
                        f"place {i} is NOT in this photo — it is {side}, "
                        f"{w.distance:.1f} m away; goto({i}) turns you to it")
            except Exception:  # noqa: BLE001
                pass

        if png and td is not None:
            png = dm.annotate_rgb(png, waypoints, td.floor_y)
        self._obs_count += 1
        if png:
            self._live_frame(png)
        self._write_depth_view(td, waypoints)

        around = [dm.surroundings_sentence(td)] if td is not None else []
        sentence = landmark_sentence(self._sightings)
        if sentence:
            around.append(sentence)
        # …and what is no longer in front of the camera. A landmark the robot
        # walked past stops existing for a model that only ever sees the current
        # frame, which is precisely how "have I passed the pool?" became a
        # coin flip. The map answers it from its own record.
        #
        # §4.2: PlaceMemory and the map recall are COMPLEMENTARY, not rivals.
        # "What place am I in" and "where are the things I have seen" answer
        # different questions; the old if/else meant enabling places silently
        # deleted the map's entire semantic memory from the model's context.
        if self.places is not None:
            event = self.places.observe(td, [s.phrase for s in self._sightings],
                                        (self.amap.px, self.amap.py))
            if event:
                self._live_log({"place_boundary": event,
                                "places": len(self.places.places)})
            # ONLY the current-place line rides the tool result. The walked
            # chain and the recognition question are persistent facts, so they
            # belong on the state block, which is rendered every turn and never
            # cut — the wrapper lifts them there (`_absorb_places`).
            here = self.places.here_sentence(self._sightings)
            if here:
                around.append(here)
        # Exclude from "you cannot see these NOW" anything detected within
        # the last cadence window: on a non-cadence frame the detector did
        # not run, and describing a landmark seen one look ago as unseen
        # invites the model to count it twice.
        recently = {p for p, lk in self._seen_at.items()
                    if self._looks - lk <= self.landmark_every}
        recall = self.amap.recall_sentence(
            exclude={s.phrase for s in self._sightings} | recently)
        if recall:
            around.append(recall)

        # Deterministic Map Telemetry (§10.6): read out of the arrays, never
        # out of a model. Same map state → same telemetry, zero API calls.
        if hidden_commit:
            around.append(hidden_commit)
        if off_camera:
            around.append("Off-camera places: " + "; ".join(off_camera) + ".")
        telem: dict[str, Any] = {}
        map_png = b""
        if pv is not None:
            prof, neck = dm.passable_profile(pv)
            telem = self._map_telemetry(pv, prof, neck, waypoints)
            # §21.7: ONE image, two truths — anchor-fixed global main map
            # (the world holds still; the arrow is the body) beside a small
            # heading-up local inset. map_version and menu epoch are
            # stamped in the shared caption so artifact, status json and
            # pixels can be proven to be the same publish.
            map_png = self.amap.render_composite(
                waypoints, current={s.phrase for s in self._sightings},
                loop=self.loop.last if self.loop.last
                and self.loop.last.get("warning") else None,
                caption=(f"map_version {int(self.amap.updates)} · "
                         f"menu epoch {self._candidate_epoch}"))
        status: dict[str, Any] = {
            "places_you_can_walk_to": {
                str(i + 1): {"where": w.describe(), "kind": w.kind,
                             "distance_m": round(w.distance, 2),
                             "angle_deg": round(math.degrees(w.angle), 1),
                             "safe_stride_m": round(w.stride_m or w.distance, 2),
                             "verified_ground_m": round(w.verified_m, 2),
                             "visible_clear_depth_m": round(w.visible_m, 2),
                             "clear_beyond_m": round(w.continuation_m, 2),
                             **({"track_id": int(w.extras["track_id"])}
                                if w.extras.get("track_id") else {}),
                             **({"staging_only": True}
                                if w.extras.get("short_verified_gateway")
                                else {}),
                             **({"confidence": w.confidence}
                                if w.confidence else {}),
                             **({"toward": telem["_toward"].get(i + 1)}
                                if telem.get("_toward", {}).get(i + 1) else {})}
                for i, w in enumerate(waypoints)
            },
            "around_you": " ".join(around),
            "candidate_epoch": self._candidate_epoch,
            "map_version": int(self.amap.updates),
            **({"map": telem["map"]} if telem else {}),
            **self._budget_fields(),
        }
        if not waypoints:
            # The organ is not blind here — it measured which bearing has the
            # most floor, it just cannot reach it in a straight line from where
            # the body is wedged. Saying "turn and look" while KNOWING the side
            # is the harness withholding what it has. Name the side; the model
            # still decides whether to take it.
            # Wedged in front of a chair with a way out on EITHER side, the
            # old note named one bearing — the roomiest — and said nothing about
            # the other. "No place is offered" then reads as "there is nowhere
            # to go", which is false and is the harness withholding what it
            # measured. List every opening it can see, say plainly that none is
            # enterable in a straight line FROM HERE, and give the turn for each.
            hint = ""
            if td is not None:
                ways = []
                for centre, width, reach in dm.openings(td)[:3]:
                    deg = math.degrees(centre)
                    if abs(deg) < 8:
                        ways.append(f"straight ahead ({reach:.1f} m of floor)"
                                    " — you already face it")
                        continue
                    side = "left" if deg > 0 else "right"
                    ways.append(f"about {abs(deg):.0f}° to your {side} "
                                f"({reach:.1f} m of floor that way) — "
                                f"{_turn_call(deg)} turns to it")
                if ways:
                    hint = (f" I can see {len(ways)} way(s) out of here, but none "
                            "of them is enterable in a straight line from this "
                            "exact spot — turn to one and the fresh view will "
                            "offer places along it: " + "; ".join(ways) + ".")
                else:
                    bearing, reach = td.widest()
                    deg = math.degrees(bearing)
                    if abs(deg) > 8 and reach > 1.0:
                        side = "left" if deg > 0 else "right"
                        hint = (f" The most open direction I can measure is about "
                                f"{abs(deg):.0f}° to your {side} ({reach:.1f} m of "
                                f"floor that way) — {_turn_call(deg)} turns to it.")
            status["note"] = (
                "No place is offered from this exact spot — what is in front of "
                "me is blocked or too tight, and I can only offer places the "
                "camera can see." + (hint or
                " step([2,2]) turns 30° left, step([3,3]) 30° right; every "
                "action returns the fresh view.")
            )
        # Two images, both explained, every turn (§10.6): the egocentric photo
        # and the accumulated memory, plus the deterministic readout. Nothing
        # is "thrown in without a role" — the labels are part of the contract.
        content: list = []
        if png:
            content.append(text_part(
                "IMAGE 1 — current egocentric RGB; numbered circles are the "
                "places you can walk to"))
            content.append(png_part(png))
        if map_png:
            # §20.4: the label IS the canonical legend — same constant the
            # system prompt teaches, so the words and the pixels never drift
            from eharness.capabilities import MAP_LEGEND
            content.append(text_part("IMAGE 2 — " + MAP_LEGEND))
            content.append(png_part(map_png))
        if telem:
            content.append(text_part(self._telemetry_text(telem)))
        content.append(text_part(json.dumps(status)))
        return ToolResult(content=content,
                          info={"kind": "observe", "map_image": bool(map_png),
                                **status})

    # ── step: inherited, plus a nudge back toward the measured option ────
    def _tool_step(self, actions: Any = None, **kw: Any) -> ToolResult:
        """HabitatToolSet.step, with one advisory bolted on.

        Live finding (EP0, 2026-08-05): given BOTH tools, a 9B took two gotos
        and then reverted to step([1]×8-9) for the rest of the episode, walked
        itself into a corner and burned 200 steps there. Flexibility costs
        discipline — the habit the model was trained into wins unless something
        keeps pointing at the better option. So when it pushes forward blind
        while a measured, guaranteed-clear stride is sitting on the table, the
        harness SAYS SO. It does not refuse the step: proposing is the organ's
        job, deciding stays the model's."""
        acts = list(actions or [])
        # step() stops being a way to COVER GROUND. Four EP0 runs showed the
        # advice above being delivered, read, and ignored: the model has an
        # overwhelming prior for step([1]*n) and never once tried goto, even
        # with a measured place on the table and the harness naming it. Advice
        # could not move that prior, so the mechanism moves instead — forward
        # runs are trimmed to a fine-alignment length and distance has to go
        # through the checked, body-aware verb. Turning is untouched, STOP is
        # untouched, and WHERE to go is still entirely the model's call: the
        # harness constrains the means, never the judgement.
        # STOP-carrying batches are executed EXACTLY as commanded (audit P1):
        # trimming "walk up to it and stop" to two forwards silently moved
        # the terminal pose 0.75 m short of where the model chose to stop —
        # and the [pace] advisory arrived after the episode had ended. The
        # pace cap exists to stop blind ground-covering; a terminal batch is
        # not covering ground, it is placing the STOP.
        if 0 in acts:
            trimmed, capped = list(acts), False
        else:
            trimmed, seen_fwd = [], 0
            for a in acts:
                if a == 1:
                    seen_fwd += 1
                    if seen_fwd > self.max_fwd_per_step:
                        continue
                trimmed.append(a)
            capped = len(trimmed) < len(acts)
        forward_heavy = acts.count(1) >= 3 and acts.count(1) >= len(acts) - 1
        self._fwd_runs = getattr(self, "_fwd_runs", 0) + 1 if forward_heavy else 0

        if 0 in trimmed:
            # §16.3: a STOP-carrying batch is MOTION + STOP, and the motion
            # is a body like any other — it walks through the unified
            # primitive executor (per-primitive sensing, fusion, archive,
            # reflex brakes), verbatim and uncapped (trimming would move the
            # STOP point — audit P1). Only the STOP itself keeps the base
            # path, where the confirmation gate and terminal accounting
            # live. If the walk is interrupted or goes blind, the STOP is
            # NOT executed — the model decides again from the new view.
            self._begin_action("step")
            self._tool_calls += 1
            if self.episode_over:
                return self._result({"kind": "step", "error":
                                     f"episode already over ({self.end_reason})"})
            bad = [a for a in trimmed if a not in (0, 1, 2, 3)]
            if bad or len(trimmed) > 50:
                return self._result({"kind": "step", "error":
                                     (f"invalid actions {bad}" if bad else
                                      "too many actions in one call (max 50)")})
            k0 = trimmed.index(0)
            pre, post = trimmed[:k0], trimmed[k0 + 1:]
            if pre:
                receipt = self._execute_primitives(pre, action_id="step")
                if receipt["interrupted"] or self.episode_over:
                    payload = {
                        "kind": "step", "executed": receipt["executed"],
                        "requested": len(trimmed),
                        "moved_m": receipt["moved_m"],
                        "stop_not_executed": True,
                        "what_happened": (
                            (receipt["reason"] or
                             f"episode ended ({self.end_reason})")
                            + " — the STOP was NOT executed; look at the "
                              "fresh view and decide again"),
                        "steps_taken_total": self.steps_taken,
                        "episode_over": self.episode_over,
                        "end_reason": self.end_reason,
                        **self._budget_fields(),
                    }
                    return self._augment(self._result(payload))
            # the walk completed cleanly — the STOP goes to the base gate.
            # tool_calls: the base counts one more; this call already
            # counted itself above, so hand the counter back one.
            self._tool_calls -= 1
            result = super()._tool_step(actions=[0], **kw)
            info = result.info if isinstance(result.info, dict) else {}
            if info.get("stop_withheld"):
                # withheld → the episode continues. The model MUST see that
                # its pre-walk WAS executed (review P1: the bare "STOP not
                # executed" receipt read as "nothing happened", and a
                # verbatim retry walked the whole pre a second time past
                # the chosen stop pose). Rebuild the payload with the
                # motion receipt merged in, exactly as the base gate does
                # for its own prefix.
                merged = {k: v for k, v in info.items() if k != "kind"}
                if pre:
                    merged.update({
                        "executed": receipt["executed"],
                        "requested": len(trimmed),
                        "moved_m": receipt["moved_m"],
                        "walked_before_stop_m": receipt["moved_m"],
                        "steps_taken_total": self.steps_taken,
                        "message": (
                            f"your {receipt['executed']} pre-STOP move(s) "
                            f"({receipt['moved_m']:.2f} m) WERE executed — "
                            "only the STOP itself is withheld. Do NOT "
                            "re-issue the walk; verify placement from the "
                            "fresh view, then call step([0]) alone to "
                            "confirm. " + str(merged.get("message", ""))),
                    })
                if post:
                    merged["ignored_after_stop"] = post
                result = self._result({"kind": "step", **merged})
                return self._augment(result)
            if pre or post:
                result.info = {**info, "walked_before_stop_m":
                               (receipt["moved_m"] if pre else 0.0),
                               **({"ignored_after_stop": post} if post else {})}
            return result          # STOP executed — terminal, nothing to add
        else:
            # every movement batch goes through the SAME primitive executor
            # goto uses (§4.4) — per-primitive sensing, actual-delta odometry,
            # per-frame fusion, archives, and reflex interrupts. There are no
            # longer two grades of perception depending on which verb moved
            # the body.
            self._tool_calls += 1
            if self.episode_over:
                return self._result({"kind": "step", "error":
                                     f"episode already over ({self.end_reason})"})
            bad = [a for a in trimmed if a not in (1, 2, 3)]
            if not trimmed or bad or len(trimmed) > 50:
                return self._result({"kind": "step", "error":
                                     ("empty action list" if not trimmed else
                                      f"invalid actions {bad}" if bad else
                                      "too many actions in one call (max 50)")})
            self._begin_action("step")
            receipt = self._execute_primitives(trimmed, action_id="step")
            payload: dict[str, Any] = {
                "kind": "step", "executed": receipt["executed"],
                "requested": receipt["requested"],
                "moved_m": receipt["moved_m"],
                "steps_taken_total": self.steps_taken,
                "episode_over": self.episode_over,
                "end_reason": self.end_reason,
                **self._budget_fields(),
            }
            if receipt["interrupted"]:
                payload["what_happened"] = receipt["reason"]
            noted = [e["type"] for e in receipt.get("events", [])
                     if not e["requires_interrupt"]]
            if noted:
                payload["noticed_on_the_way"] = noted
            result = self._result(payload)
        info = result.info if isinstance(result.info, dict) else {}
        if capped:
            result.content = list(result.content) + [text_part(
                f"[pace] step() carries at most {self.max_fwd_per_step} forward "
                f"moves ({0.25 * self.max_fwd_per_step:.1f} m) — it is for "
                "turning, lining up and the final metre. To cover ground, take "
                "one of the numbered places with goto(place): its safe stride "
                "is measured clear and wide enough for your body.")]
        if not forward_heavy:
            # Every action returns a fresh look — turns included. A model
            # whose observe verb was withdrawn is otherwise blind after a
            # turn, and the numbered places from BEFORE the turn stay
            # callable while pointing at the wrong world.
            if info.get("error"):
                return result
            return self._augment(result)

        # Fire on the FIRST blind push, not the second. Waiting for a second
        # one was too late in practice: by then the robot had walked itself
        # into clutter, the proposer had nothing left to offer, and the advice
        # that depends on having a candidate went silent exactly when it was
        # needed. So: if a measured stride is on the table, say so immediately;
        # if there is none, fall back on the bearing the map still knows is
        # open. The step is never refused — proposing is the organ's job.
        # The metres actually walked (trimmed count), not the metres asked
        # for — quoting the request as fact inflated the story by whatever
        # the pace cap ate. And rank places by VERIFIED ground: the aim can
        # legitimately point past proven floor, but "measured" in this advice
        # must mean measured.
        best = max(self._waypoints, key=lambda w: w.verified_m or w.distance,
                   default=None)
        covered = 0.25 * min(acts.count(1), trimmed.count(1))
        advice = ""
        if best is not None and (best.verified_m or best.distance) >= max(
                1.5, 1.5 * covered):
            advice = (
                f"You walked {covered:.1f} m forward blind. The last look "
                f"offered a measured place — {best.describe()} — whose safe "
                f"stride ({best.stride_m or best.distance:.1f} m) is checked "
                "clear and wide enough for your body; goto() covers that "
                "ground in one call and looks again. Take it if it goes "
                "where you want; if it does not, turn with step 2/3s toward "
                "the direction you mean and read the fresh view rather than "
                "pushing on."
            )
        elif best is None and self._last_td is not None:
            bearing, reach = self._last_td.widest()
            deg = math.degrees(bearing)
            if abs(deg) > 8 and reach > 1.0:
                side = "left" if deg > 0 else "right"
                notches = max(1, min(6, int(round(abs(deg) / 15.0))))
                turn = 2 if side == "left" else 3
                advice = (
                    f"You walked {covered:.1f} m forward blind and there is no "
                    "measured place ahead to walk to — you are pushing into "
                    f"clutter. The most open direction I can measure is about "
                    f"{abs(deg):.0f}° to your {side} ({reach:.1f} m of floor that "
                    f"way): {_turn_call(deg)} turns to it."
                )
        if advice:
            result.content = list(result.content) + [text_part(advice)]
            result.info = {**result.info, "harness_advice": advice}
        return self._augment(result)

    # ── the unified primitive executor (§4.4): ONE body for all motion ───
    def _execute_primitives(self, prims: list[int], *,
                            action_id: str) -> dict[str, Any]:
        """Walk a primitive list the way a body does: sense after EVERY
        primitive, integrate the ACTUAL displacement, map from every depth
        frame, archive the packet — and stop the leg early when the world
        says stop. The model never sees the intermediate frames; they are
        map evidence and episode video (§4.2/§4.3). Returns an execution
        receipt; self._interrupt_reason carries the structured cause when a
        leg ended before its command list did."""
        moved = 0.0
        collisions = 0
        executed = 0
        self._interrupt_reason = ""
        self._exec_last_td: dm.TopDown | None = None
        # topology baseline for THIS leg (review P2: a baseline from a
        # previous goto's pose re-reported every doorway as news). Fresh
        # current frame → use it; otherwise the first sensed frame of the
        # leg becomes the baseline and the watcher stays quiet until then.
        if self._sensed_at_step == self.steps_taken and self._last_td is not None:
            try:
                self._leg_openings = [(math.degrees(o.centre), float(o.reach))
                                      for o in dm.openings(self._last_td)]
                baseline_ready = True
            except Exception:  # noqa: BLE001
                self._leg_openings, baseline_ready = [], False
        else:
            self._leg_openings, baseline_ready = [], False
        for k, a in enumerate(prims):
            out = self._call("env_habitat__step_discrete", {"action": int(a)})
            info = out.get("info") if isinstance(out.get("info"), dict) else {}
            if out.get("error") or info.get("error"):
                # an env-side refusal (episode already done, say) executed
                # NOTHING — counting it as a step stamped phantom 0.25 m
                # into odometry (audit P2)
                if "already done" in str(out.get("error") or info.get("error")):
                    self.episode_over = True
                    self.end_reason = self.end_reason or "terminated"
                self._interrupt_reason = (
                    f"the environment refused the primitive: "
                    f"{out.get('error') or info.get('error')}")
                self._emit_event("env_refused", "critical",
                                 requires_interrupt=True,
                                 evidence={"error": str(
                                     out.get("error") or info.get("error"))[:200]})
                self._take_interrupt()
                break
            executed += 1
            if isinstance(info.get("step_count"), (int, float)):
                self.steps_taken = int(info["step_count"])
            else:
                self.steps_taken += 1
            if bool(out.get("terminated")) or bool(out.get("truncated")):
                self.episode_over = True
                self.end_reason = ("step_budget_exhausted"
                                   if out.get("truncated") else "terminated")
            delta = info.get("actual_translation_m")
            collided = bool(info.get("collided"))
            d = (float(delta) if isinstance(delta, (int, float))
                 else (STEP_M if a == 1 else 0.0))
            # odometry by what HAPPENED, not what was asked (§4.2): habitat's
            # turns are exact (measured), so commanded yaw is the truth for
            # 2/3; forwards use the env's own measured translation.
            if a == 1:
                self.amap.odometry(0.0, d)
                moved += d
            elif a in (2, 3):
                self.amap.odometry(TURN_RAD if a == 2 else -TURN_RAD, 0.0)
                # the baseline speaks in the CURRENT egocentric frame —
                # rotate it with the body or every turn re-frames old
                # openings as new ones
                shift = TURN_DEG if a == 2 else -TURN_DEG
                self._leg_openings = [(d0 - shift, r0)
                                      for d0, r0 in self._leg_openings]
                # arm the post-turn SAM once per YAW LEG, not per notch —
                # six notches of a face(90) are one turn, and running the
                # detector on every 15° slice cost ~15 s per quarter turn
                # for five near-identical frames (review P1)
                nxt = prims[k + 1] if k + 1 < len(prims) else None
                self._yaw_pending_sam = nxt not in (2, 3)
            if collided:
                collisions += 1
            td = self._sense_primitive(action_id, k, int(a), d, collided)
            if td is not None:
                self._exec_last_td = td
            # ── the fast-reflex layer (§4.5): cheap, deterministic, no LLM.
            # Everything below speaks NavigationEvent (§14.7); the first
            # requires_interrupt event ends the leg.
            if self.episode_over:
                self._emit_event("episode_done", "info",
                                 evidence={"end_reason": self.end_reason})
                break
            remaining = prims[k + 1:]
            remaining_fwd = sum(1 for x in remaining if x == 1)
            if self._sense_failure in FAIL_CLOSED:
                # §14.5 fail CLOSED: a body that cannot see must not walk.
                # The event was emitted inside _sense_primitive; here the
                # leg is cancelled and the cause named to the model. This
                # fires on the FINAL primitive too — with nothing left to
                # cancel it still arms the interrupt plane, so a goto whose
                # last yaw notch went blind refuses its walk (review P1).
                self._interrupt_reason = {
                    "perception_unavailable": (
                        "perception failed — no frame came back from the "
                        "environment; remaining moves cancelled (fail closed)"),
                    "depth_invalid": (
                        "the depth frame is invalid (empty or NaN) — "
                        "geometry is blind; remaining moves cancelled "
                        "(fail closed)"),
                    "map_update_failed": (
                        "the map could not be updated from the new frame — "
                        "remaining moves cancelled (fail closed)"),
                }[self._sense_failure]
                break
            if collided and d < 0.05:
                self._interrupt_reason = (
                    "collision — the body hit something and stopped moving")
                self._emit_event("collision", "critical",
                                 requires_interrupt=True,
                                 evidence={"primitive": k, "moved_m": round(d, 2)})
                self._take_interrupt()
                break
            if td is not None and remaining_fwd:
                ahead = dm._free_prefix(td, 0.0)
                if ahead < 0.4:
                    self._interrupt_reason = (
                        f"the verified corridor ahead has closed "
                        f"({ahead:.1f} m of proven floor left) — "
                        "remaining forward steps cancelled")
                    self._emit_event("corridor_closed", "warning",
                                     requires_interrupt=True,
                                     evidence={"ahead_m": round(ahead, 2)})
                    self._take_interrupt()
                    break
            # §14.7: topology change — a NEW opening worth a decision
            # appeared mid-leg (a side gateway coming into view). Info
            # only: it surfaces at the endpoint, it does not brake.
            if td is not None and remaining_fwd:
                if baseline_ready:
                    self._note_new_openings(td)
                else:
                    try:
                        self._leg_openings = [
                            (math.degrees(o.centre), float(o.reach))
                            for o in dm.openings(td)]
                        baseline_ready = True
                    except Exception:  # noqa: BLE001
                        pass
            # §4.5/§7: a loop warning interrupts the leg — the model should
            # re-decide with the warning in front of it, not finish a lap
            if remaining_fwd and (k % 4) == 3 \
                    and self.loop.assess(self.amap, tick=False) is not None:
                self._interrupt_reason = (
                    "loop warning — the trail says this area was walked "
                    "before; remaining steps cancelled so you can re-decide "
                    "with the map's evidence")
                self._emit_event("loop", "warning", requires_interrupt=True,
                                 evidence=dict(self.loop.last or {}))
                self._take_interrupt()
                break
            # landmark / near-goal / registration-drop events raised inside
            # _sense_primitive interrupt here, between primitives. This runs
            # on the FINAL primitive too: an event on the last yaw notch of
            # a goto must arm the interrupt plane so the walk leg pauses —
            # dropping it there was exactly the final-approach overshoot
            # §14.7 exists to prevent (review P2).
            ev = self._take_interrupt()
            if ev is not None:
                self._interrupt_reason = {
                    "landmark_sighted": (
                        f"'{ev.evidence.get('phrase', '?')}' just came into "
                        "view — leg paused so you can decide with it in front "
                        "of you"),
                    "near_goal": (
                        f"'{ev.evidence.get('phrase', '?')}' — a landmark of "
                        "your endpoint — is in view; leg paused for the "
                        "final approach decision"),
                    "registration_drop": (
                        "the map stopped recognising this view (registration "
                        "confidence dropped) — leg paused; look before "
                        "trusting remembered distances"),
                }.get(ev.type, f"{ev.type} — leg paused")
                break
        return {"executed": executed, "requested": len(prims),
                "moved_m": round(moved, 2), "collisions": collisions,
                "interrupted": bool(self._interrupt_reason),
                "reason": self._interrupt_reason,
                "events": [e.as_dict() for e in self.events]}

    def _note_new_openings(self, td: dm.TopDown) -> None:
        """Cheap per-primitive topology watch: an opening ≥2 m deep at a
        bearing no opening covered at leg start is NEWS (a doorway coming
        into view), recorded once per leg as an info event."""
        try:
            now = [(math.degrees(o.centre), float(o.reach))
                   for o in dm.openings(td)]
        except Exception:  # noqa: BLE001 — a watcher must never end a leg
            return
        for deg, reach in now:
            if reach < 2.0 or abs(deg) < 25.0:
                continue
            if any(abs(deg - d0) < 20.0 for d0, _ in self._leg_openings):
                continue
            if any(e.type == "new_opening"
                   and abs(e.evidence.get("bearing_deg", 999) - deg) < 20.0
                   for e in self.events):
                continue
            self._emit_event("new_opening", "info",
                             evidence={"bearing_deg": round(deg, 1),
                                       "reach_m": round(reach, 1)})

    # ── §14.4: yaw is REAL turn primitives, not a quaternion write ───────
    def _execute_yaw(self, deg: float, *, action_id: str) -> dict[str, Any]:
        """Turn the body the way the body turns: each 15° notch is one env
        turn primitive through the SAME executor as everything else — costed,
        sensed, fused, archived, interruptible. Only the sub-15° remainder
        is a rotate-only micro-alignment (zero env steps, logged as such,
        still sensed) — discrete Habitat cannot express it any other way.

        Returns a RECEIPT, not just a frame: {"td", "executed_notches",
        "residual_deg" (0.0 unless actually applied), "executed_deg",
        "requested_deg"} — the caller reports what HAPPENED, never what was
        asked (review P1: an interrupted face(90) claimed 90°/6 steps for a
        30° turn). td is None whenever the LAST sense of the yaw failed
        closed — a stale pre-turn frame must never validate the corridor
        the walk leg is about to trust (review P1)."""
        notches = int(abs(deg) // TURN_DEG)
        prim = 2 if deg > 0 else 3
        td: dm.TopDown | None = None
        executed = 0
        residual_applied = 0.0
        if notches:
            r = self._execute_primitives([prim] * notches, action_id=action_id)
            executed = int(r["executed"])
            td = self._exec_last_td
            if self.episode_over or self._interrupt_reason:
                if self._sense_failure in FAIL_CLOSED:
                    td = None
                return {"td": td, "executed_notches": executed,
                        "residual_deg": 0.0,
                        "executed_deg": math.copysign(TURN_DEG * executed, deg),
                        "requested_deg": float(deg)}
        residual = float(deg) - math.copysign(TURN_DEG * notches, deg)
        if abs(residual) >= 1.0:
            out = self._call("env_habitat__step_hightolow",
                             {"angle": math.radians(residual), "distance": 0.0})
            if bool(out.get("terminated")) or bool(out.get("truncated")):
                self.episode_over = True
                self.end_reason = ("step_budget_exhausted"
                                   if out.get("truncated") else "terminated")
            self.amap.odometry(math.radians(residual), 0.0)
            residual_applied = residual
            self._yaw_pending_sam = True
            self._live_log({"micro_yaw_deg": round(residual, 1)})
            td2 = self._sense_primitive(action_id, -1, 0, 0.0, False)
            if self._sense_failure in FAIL_CLOSED:
                td = None            # blind — do not fall back to stale frames
            elif td2 is not None:
                td = td2
            # events raised while sensing the residual (landmark, near-goal,
            # registration drop) must arm the interrupt plane HERE — leaking
            # them into the walk leg fired them one forward too late and
            # attributed the turn-phase sighting to the walk (review P2)
            ev = self._take_interrupt()
            if ev is not None and not self._interrupt_reason:
                self._interrupt_reason = f"{ev.type} while aligning — leg paused"
        return {"td": td, "executed_notches": executed,
                "residual_deg": residual_applied,
                "executed_deg": math.copysign(TURN_DEG * executed, deg)
                + residual_applied,
                "requested_deg": float(deg)}

    def _sense_primitive(self, action_id: str, prim_idx: int, commanded: int,
                         delta_m: float, collided: bool) -> "dm.TopDown | None":
        """Per-primitive perception: cached RGB-D read (no re-render), map
        registration/fusion from THIS depth, same-frame SAM on its own
        cadence, and one archived MicroObservation. Marks the env step as
        sensed so the endpoint observe does not double-fuse the same frame.

        §14.5 — failure is CLASSIFIED, not swallowed. self._sense_failure
        carries the class; FAIL_CLOSED classes make the executor cancel the
        remaining leg. Archive/SAM/RGB-only problems warn and continue —
        blindness stops the body, a broken notebook does not."""
        self._sense_failure = None
        try:
            outputs = self._call("env_habitat__observe_egocentric", {})
        except Exception as exc:  # noqa: BLE001 — classify, do not kill the process
            self._sense_failure = "perception_unavailable"
            self._emit_event("perception_unavailable", "critical",
                             requires_interrupt=True,
                             evidence={"error": str(exc)[:200]})
            return None
        self._sensor_frame += 1
        rgb_b64 = outputs.get("rgb")
        units = outputs.get("depth_units") or {}
        self._scale_m = (float(units["scale_m"])
                         if units.get("known") and units.get("scale_m") else None)
        depth_arr = dm.decode_depth(outputs.get("depth"))
        if (depth_arr is None or depth_arr.size == 0
                or bool(np.isnan(depth_arr).all())):
            # a frame arrived but its depth is unusable — geometry is blind
            self._sense_failure = "depth_invalid"
            self._emit_event("depth_invalid", "critical",
                             requires_interrupt=True,
                             evidence={"empty": depth_arr is None
                                       or depth_arr.size == 0})
            self._archive_packet(action_id=action_id, prim_idx=prim_idx,
                                 commanded=commanded, delta_m=delta_m,
                                 collided=collided, rgb_b64=rgb_b64,
                                 depth_arr=None)
            return None
        if not isinstance(rgb_b64, str):
            # RGB-only failure: geometry continues (depth is what walks the
            # body); semantics and the archive photo are what is lost
            self._emit_event("rgb_missing", "info")
        td = None
        try:
            td = dm.build_topdown(depth_arr, range_cap_m=self.range_cap_m,
                                  scale_m=self._scale_m)
        except Exception as exc:  # noqa: BLE001
            self._sense_failure = "map_update_failed"
            self._emit_event("map_update_failed", "critical",
                             requires_interrupt=True,
                             evidence={"stage": "build", "error": str(exc)[:200]})
        if td is not None:
            try:
                # §21.5: only a FORWARD primitive may earn a translation
                # fix. Pure turns (2/3), goto micro-yaws and standstill
                # senses (commanded 0) register READ-ONLY — Habitat turns
                # are exact and leave the body in place, and letting the
                # matcher hand a turn frame ±30 cm of px/py "correction"
                # is how head-turning smeared anchors across the map.
                self.amap.register(td,
                                   translation_expected=(commanded == 1))
                self.amap.fuse(td)
            except Exception as exc:  # noqa: BLE001
                self._sense_failure = "map_update_failed"
                self._emit_event("map_update_failed", "critical",
                                 requires_interrupt=True,
                                 evidence={"stage": "fuse",
                                           "error": str(exc)[:200]})
                td = None
        if td is not None:
            self.loop.note_growth(int((np.abs(self.amap.logodds) > 0.5).sum()))
            # §14.10: an independent VISUAL signal for the loop monitor —
            # a coarse frame signature tied to the pose it was taken at
            self.loop.note_frame(dm.frame_signature(rgb_b64),
                                 (self.amap.px, self.amap.py))
            # registration-confidence drop (§14.7): the map suddenly stops
            # recognising the world it built — worth re-deciding over
            score = float(self.amap.last_score)
            if (self.amap.updates > 5 and self._prev_reg_score is not None
                    and self._prev_reg_score >= dm.TRUST_MIN_SCORE
                    and score < 0.15):
                self._emit_event("registration_drop", "warning",
                                 requires_interrupt=True,
                                 evidence={"score": round(score, 2),
                                           "was": round(self._prev_reg_score, 2)})
            self._prev_reg_score = score
            depth_m = (dm.to_metres(depth_arr, self._scale_m)[0]
                       if self._scale_m is not None else depth_arr)
            scene_changed = self._scene_changed(self._prev_depth, depth_m)
            if scene_changed:
                self._emit_event("scene_transition", "info")
            self._prev_depth = depth_m
            self._prev_rgb = rgb_b64
            self._pending_fwd_m = 0.0        # odometry here is MEASURED
            # ── §14.8: SAM scheduling off SENSOR frames and EVENTS, never
            # the archive counter. N>0 keeps the fixed cadence; 0 (default)
            # runs post-turn first frame / scene transitions / a low-
            # frequency backstop, and logs every skip with its reason.
            run_sam, why = False, ""
            if self.landmarks is not None and self.phrases:
                if self.sam_intermediate_every > 0:
                    if self._sensor_frame % self.sam_intermediate_every == 0:
                        run_sam, why = True, "cadence"
                    else:
                        self._sam_skip("cadence_gap")
                elif self._yaw_pending_sam:
                    run_sam, why = True, "post_turn"
                elif scene_changed:
                    run_sam, why = True, "scene_transition"
                elif self._sensor_frame % SAM_LOW_CADENCE == 0:
                    run_sam, why = True, "low_cadence"
                else:
                    self._sam_skip("event_gate")
            if not run_sam:
                # the frame changed but the detector did not run: yesterday's
                # sightings are STALE for this pose — show none rather than
                # present an old detection as current (audit P2)
                self._sightings = []
            if run_sam:
                self._yaw_pending_sam = False
                t_sam = time.time()
                try:
                    sights = self.landmarks.sightings(
                        rgb_b64, outputs.get("depth"), self.phrases,
                        scale_m=self._scale_m)
                except Exception:  # noqa: BLE001 — semantics stay advisory
                    sights = []
                    self._sam_skip("detector_error")
                self._sam_stats["calls"] += 1
                self._sam_stats["latency_s"] = round(
                    self._sam_stats["latency_s"] + time.time() - t_sam, 2)
                self._sam_stats.setdefault("triggers", []).append(why)
                self.register.record(sights, self.steps_taken)
                for s in sights:
                    # §14.7: a landmark of the CURRENT route newly in view,
                    # or the terminate phrase itself — both are re-decide
                    # moments, not things to notice three metres later
                    fresh_sight = (s.phrase not in self._seen_at
                                   or self._looks - self._seen_at[s.phrase] > 6)
                    if fresh_sight:
                        # word-boundary match: "chair" inside "wheelchair
                        # ramp" is not the endpoint (review P2)
                        import re as _re
                        near_goal = bool(
                            self._terminate
                            and _re.search(
                                r"\b" + _re.escape(s.phrase.lower()) + r"\b",
                                self._terminate.lower()))
                        self._emit_event(
                            "near_goal" if near_goal else "landmark_sighted",
                            "warning", requires_interrupt=True,
                            evidence={"phrase": s.phrase,
                                      "bearing_deg": round(
                                          math.degrees(s.bearing), 1),
                                      "distance_m": round(float(s.distance), 1)})
                    if s.mask is None:
                        self._seen_at[s.phrase] = self._looks
                        continue
                    xs, ys = dm.project_mask(s.mask, depth_arr, td.floor_y,
                                             scale_m=self._scale_m)
                    if xs.size:
                        self.amap.stamp_semantic(s.phrase, xs, ys,
                                                 weight=float(s.score))
                    self._seen_at[s.phrase] = self._looks
                self._sightings = sights
            self._sensed_at_step = self.steps_taken
        self._archive_packet(action_id=action_id, prim_idx=prim_idx,
                             commanded=commanded, delta_m=delta_m,
                             collided=collided, rgb_b64=rgb_b64,
                             depth_arr=depth_arr)
        return td

    def _sam_skip(self, reason: str) -> None:
        sk = self._sam_stats["skips"]
        sk[reason] = sk.get(reason, 0) + 1

    @staticmethod
    def _scene_changed(prev: Any, now: Any) -> bool:
        """Did the view change more than walking one step explains?"""
        if prev is None or now is None or getattr(prev, "shape", None) is None:
            return False
        if prev.shape != now.shape:
            return True
        a, b = np.asarray(prev, np.float32), np.asarray(now, np.float32)
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if valid.sum() < 100:
            return False
        return float(np.abs(a[valid] - b[valid]).mean()) > SCENE_CHANGE_M

    def _archive_packet(self, *, action_id: str, prim_idx: int, commanded: int,
                        delta_m: float, collided: bool, rgb_b64: Any,
                        depth_arr: Any) -> None:
        """The trajectory archive (§4.3): PNG + NPZ + manifest line per
        primitive, addressable without step-glob guessing. Facts on disk;
        the MP4 anyone renders from them later is just a viewing copy."""
        if not self.traj_archive or self.live_dir is None:
            return
        try:
            base = self.live_dir / "trajectory"
            (base / "rgb").mkdir(parents=True, exist_ok=True)
            (base / "depth").mkdir(parents=True, exist_ok=True)
            self._traj_n += 1
            fid = self._traj_n
            rgb_f = dep_f = None
            if isinstance(rgb_b64, str):
                rgb_f = f"rgb/frame_{fid:06d}.png"
                (base / rgb_f).write_bytes(base64.b64decode(rgb_b64))
            if depth_arr is not None:
                dep_f = f"depth/frame_{fid:06d}.npz"
                np.savez_compressed(
                    base / dep_f, depth=depth_arr,
                    scale_m=np.float32(self._scale_m if self._scale_m else 0.0))
            rec = {"frame_id": fid, "action_id": action_id,
                   "primitive_index": prim_idx, "commanded": commanded,
                   "env_step": self.steps_taken,
                   "actual_delta_m": round(float(delta_m), 3),
                   "collided": bool(collided),
                   "map_version": int(self.amap.updates),
                   "rgb_file": rgb_f, "depth_file": dep_f,
                   "t": round(time.time() - self._t0, 2)}
            with (base / "manifest.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001 — the archive must never end a leg
            pass

    # ── goto: the geometric stride ───────────────────────────────────────
    def _augment(self, result: ToolResult) -> ToolResult:
        """Attach a fresh look to an action's result.

        The model no longer has an `observe` verb, so every action must hand
        back what the next decision needs: the view with the places drawn on
        it, the places themselves, and the sentences from the map and the
        detector. A look costs no env steps, so this is free in the only
        currency that matters."""
        if self.episode_over:
            return result
        try:
            view = self._tool_observe(_internal=True)
        except Exception as exc:  # noqa: BLE001 — never kill a move, but never
            # leave the PRE-move places on the table either: goto would then
            # happily execute a number pointing at the world as it was before
            # this action moved the body (review P1 — fail closed on data)
            self._waypoints = []
            self._last_td = None
            self._emit_event("perception_unavailable", "critical",
                             evidence={"stage": "endpoint_look",
                                       "error": str(exc)[:200]})
            result.content = list(result.content) + [text_part(
                "[look failed] the post-action view could not be captured — "
                "no places are current; act again to get a fresh view.")]
            return result
        parts = list(view.content)
        result.content = list(result.content) + parts
        result.info = {**result.info,
                       "places_you_can_walk_to": view.info.get("places_you_can_walk_to"),
                       "around_you": view.info.get("around_you")}
        return result

    def _tool_face(self, direction: Any = None, **_ignored: Any) -> ToolResult:
        """§5.2's one semantic turn, §14.4's honest cost: the yaw is REAL
        turn primitives through the unified executor — one env step per 15°
        notch, each one sensed and fused — plus a free sub-15° micro-
        alignment. The model still gets its one-call turn; the simulator
        gets the same six turns a body would spend."""
        self._tool_calls += 1
        if self.episode_over:
            return self._result({"kind": "face",
                                 "error": f"episode already over ({self.end_reason})"})
        named = {"left": 90.0, "right": -90.0, "back": 180.0, "behind": 180.0,
                 "around": 180.0}
        deg: float | None = None
        if isinstance(direction, str):
            key = direction.strip().lower()
            if key in named:
                deg = named[key]
            else:
                try:
                    deg = float(key.rstrip("°"))
                except ValueError:
                    deg = None
        elif isinstance(direction, (int, float)):
            deg = float(direction)
        if deg is None or not -180.0 <= deg <= 180.0:
            return self._result({
                "kind": "face",
                "error": (f"direction {direction!r} not understood — use "
                          "'left', 'right', 'back', or signed degrees in "
                          "[-180, 180] (left positive)")})
        # identity + a clean event/interrupt plane from here on — the <2°
        # no-op below still publishes snapshots via its augment look, and
        # they must not carry the PREVIOUS verb's action_id (review P2)
        self._begin_action("face")
        if abs(deg) < 2.0:
            # NEVER forward a ~zero yaw: env_habitat treats the exact
            # (angle=0, distance=0) pair as the canonical STOP no-op, so an
            # unguarded face(0) would silently END AND SCORE the episode
            # wherever the body stands (audit P0). Facing where you already
            # face is also not a turn — hand back the current view.
            return self._augment(self._result({
                "kind": "face", "turned_deg": 0.0,
                "note": "already facing that way — here is the view",
                **self._budget_fields()}))
        self.loop.note_choice(deg)
        yaw = self._execute_yaw(deg, action_id="face")
        self._waypoints = []          # heading changed; the numbers are stale
        # report what HAPPENED: an interrupted face(90) that executed two
        # notches turned 30°, and saying 90° put the model's heading belief
        # up to 165° off while the map quietly held the truth (review P1)
        self._live_log({"face": round(deg, 1),
                        "turned_deg": round(yaw["executed_deg"], 1),
                        "env_turn_steps": yaw["executed_notches"]})
        payload: dict[str, Any] = {
            "kind": "face", "turned_deg": round(yaw["executed_deg"], 1),
            "requested_deg": round(deg, 1),
            "env_turn_steps": yaw["executed_notches"],
            "steps_taken_total": self.steps_taken,
            "episode_over": self.episode_over,
            **self._budget_fields()}
        if self._interrupt_reason:
            payload["what_happened"] = (
                f"turn stopped after {yaw['executed_deg']:.0f}° of the "
                f"requested {deg:.0f}°: {self._interrupt_reason}")
        return self._augment(self._result(payload))

    def _tool_goto(self, place: Any = None, **_ignored: Any) -> ToolResult:
        self._tool_calls += 1
        if self.episode_over:
            return self._result({"kind": "goto",
                                 "error": f"episode already over ({self.end_reason})"})
        if not self._waypoints:
            return self._result({
                "kind": "goto",
                "error": ("no places are current from this pose — take any "
                          "action (step/face) and the result returns fresh "
                          "numbered places)")})
        try:
            place = int(place)
        except (TypeError, ValueError):
            return self._result({"kind": "goto",
                                 "error": f"invalid place {place!r}; expected an integer"})
        if not 1 <= place <= len(self._waypoints):
            return self._result({
                "kind": "goto",
                "error": (f"invalid place {place}; valid choices are 1-"
                          f"{len(self._waypoints)} from the latest view")})

        w = self._waypoints[place - 1]
        epoch_used = self._candidate_epoch      # §16.2: which table this was
        self._begin_action(f"goto#{place}")
        # Commit to the PLACE the model chose (anchor frame, from the pre-walk
        # pose): next observes keep offering it, counting down, until arrival.
        self.goal.adopt(self.amap, w)
        self.loop.note_choice(math.degrees(w.angle))
        # §14.6: taking the STAGING place of a potential region is taking a
        # ROUTE — remember the intent so it survives the walk, the turn at
        # the staging point, and every observation in between.
        self._maybe_adopt_route(place, w)
        # §10.2 P0: execute the SAFE STRIDE, not the aim.
        exec_m = float(min(w.distance, w.stride_m or w.distance))
        self._gotos += 1
        action_id = f"goto#{place}"

        # 1 · turn to face it — REAL turn primitives through the unified
        # executor (§14.4): each 15° notch is one env step, sensed and
        # fused; only the sub-15° remainder is a rotate-only micro-align.
        # A turn consumes a new vista before any forward is spent on it.
        deg_w = math.degrees(float(w.angle))
        if abs(deg_w) >= 1.0:
            yaw = self._execute_yaw(deg_w, action_id=action_id)
            td0 = yaw["td"]
            if self.episode_over or self._interrupt_reason:
                self._waypoints = []
                result = {
                    "kind": "goto", "walked_to": place,
                    "aimed_for": w.describe(),
                    "target_distance_m": round(w.distance, 2),
                    "turned_deg": round(yaw["executed_deg"], 1),
                    "executed_stride_m": 0.0, "moved_m": 0.0,
                    "what_happened": (
                        (f"stopped while turning to face it "
                         f"({yaw['executed_deg']:.0f}° of "
                         f"{deg_w:.0f}° done): " + self._interrupt_reason)
                        if self._interrupt_reason
                        else f"episode ended ({self.end_reason})"),
                    "steps_taken_total": self.steps_taken,
                    "episode_over": self.episode_over,
                    "end_reason": self.end_reason, **self._budget_fields(),
                }
                self._live_log({"goto": place, "turn_interrupted": True,
                                "turned_deg": round(yaw["executed_deg"], 1),
                                "reason": self._interrupt_reason})
                return self._augment(self._result(result))
        elif self._sensed_at_step == self.steps_taken and self._last_td is not None:
            # no rotation → the frame is bit-identical to the one the
            # pre-goto observe already fused; re-sensing it would vote the
            # same evidence twice and burn a full SAM round (audit P2)
            td0 = self._last_td
        else:
            td0 = self._sense_primitive(action_id, -1, 0, 0.0, False)
            if self._sense_failure in FAIL_CLOSED:
                td0 = None
            ev = self._take_interrupt()
            if ev is not None and not self._interrupt_reason:
                self._interrupt_reason = f"{ev.type} — leg paused"

        if self._interrupt_reason and not self.episode_over:
            # an event fired during the pre-walk sense (no-rotation path):
            # pause before the walk exactly as the yaw path does
            self._waypoints = []
            return self._augment(self._result({
                "kind": "goto", "walked_to": place, "aimed_for": w.describe(),
                "target_distance_m": round(w.distance, 2),
                "executed_stride_m": 0.0, "moved_m": 0.0,
                "what_happened": self._interrupt_reason,
                "steps_taken_total": self.steps_taken,
                "episode_over": self.episode_over,
                "end_reason": self.end_reason, **self._budget_fields()}))

        # 2 · corridor revalidation on the POST-rotation frame: the stride is
        # re-derived from what is actually ahead now, not from the pre-turn
        # proposal. Shorter corridor → shorter walk; none → zero forwards
        # and a structured interruption instead of a blind push. §14.5: a
        # frame that FAILED to arrive is not a licence to skip the check —
        # no td0 means no walk (fail closed), never a blind n_fwd.
        blind = td0 is None or self._sense_failure in FAIL_CLOSED
        if td0 is not None:
            ahead = dm._free_prefix(td0, 0.0)
            if ahead < exec_m - dm.CELL_M:
                exec_m = max(0.0, ahead - dm.CELL_M)
            self._leg_openings = [(math.degrees(o.centre), float(o.reach))
                                  for o in dm.openings(td0)]
        n_fwd = 0 if blind else int(exec_m / STEP_M)
        if n_fwd <= 0:
            self._waypoints = []
            result = {
                "kind": "goto", "walked_to": place, "aimed_for": w.describe(),
                "target_distance_m": round(w.distance, 2),
                "executed_stride_m": 0.0, "moved_m": 0.0,
                "what_happened": (
                    ("perception failed after the turn ("
                     + (self._sense_failure or "no frame")
                     + ") — refusing to walk blind; try again or pick "
                       "another action") if blind else
                    "after turning to face it, the verified corridor ahead is "
                    "too short to walk safely — the way the proposal saw is "
                    "not open from this exact pose. Pick another place or "
                    "turn further."),
                "steps_taken_total": self.steps_taken,
                "episode_over": self.episode_over,
                "end_reason": self.end_reason, **self._budget_fields(),
            }
            self._live_log({"goto": place, "target_m": round(w.distance, 2),
                            "stride_m": 0.0, "moved_m": 0.0,
                            "revalidation": ("blind" if blind else "failed")})
            return self._augment(self._result(result))

        # 3 · walk it one primitive at a time through the unified executor —
        # every 0.25 m senses, maps and archives; the leg stops early on
        # collision or a closing corridor (§4.2, §4.5).
        receipt = self._execute_primitives([1] * n_fwd, action_id=action_id)
        moved_m = receipt["moved_m"]
        self._waypoints = []          # position changed; the numbers are stale

        result = {
            "kind": "goto", "walked_to": place, "aimed_for": w.describe(),
            "candidate_epoch": epoch_used,
            "target_distance_m": round(w.distance, 2),
            "executed_stride_m": round(exec_m, 2),
            "moved_m": round(moved_m, 2),
            "steps_taken_total": self.steps_taken,
            "episode_over": self.episode_over, "end_reason": self.end_reason,
            **self._budget_fields(),
        }
        if exec_m < w.distance - 1e-6:
            result["note"] = (
                f"walked the safe stride ({exec_m:.1f} m) of the way toward a "
                f"place {w.distance:.1f} m off; if it is still the right way "
                "it will be offered again from the new view")
        if receipt["interrupted"]:
            result["what_happened"] = (
                f"I aimed for {w.describe()} but stopped after "
                f"{moved_m:.1f} m: {receipt['reason']}")
        self._live_log({"goto": place, "target_m": round(w.distance, 2),
                        "stride_m": round(exec_m, 2),
                        "moved_m": round(moved_m, 2),
                        "primitives": receipt["executed"],
                        "collisions": receipt["collisions"],
                        "interrupted": receipt["interrupted"],
                        "steps_taken_total": self.steps_taken})
        return self._augment(self._result(result))

    # ── §14.6: the route intent plane ────────────────────────────────────
    def _route_for_region(self, reg: dict) -> "dm.RouteCandidate | None":
        """§20.3: the route is planned BEFORE any staging number is
        published, on the accumulated map, and cached — the same object is
        later adopted verbatim in goto(), so the promise in the telemetry
        and the plan the body follows can never be two different things.
        Re-planned only when the map has genuinely grown (every 4 fusions)
        or the glimpse moved; an ACTIVE route for the same region wins."""
        best = None
        for acc in self.amap.potential_regions():
            gap = abs((acc["bearing_deg"] - reg["bearing_deg"] + 180.0)
                      % 360.0 - 180.0)
            if gap < 40.0:
                best = acc
                break
        if best is None:
            return None
        if (self.route is not None
                and self.route.frontier_id == best["id"]):
            return self.route
        cache = getattr(self, "_route_cache", None)
        if (cache is not None and cache[0] == best["id"]
                and self.amap.updates - cache[1] < 4):
            return cache[2]
        side = ("ahead" if abs(best["bearing_deg"]) < 12 else
                f"{abs(best['bearing_deg']):.0f}° "
                f"{'left' if best['bearing_deg'] > 0 else 'right'}")
        try:
            route = dm.plan_route(
                self.amap, best,
                intent=f"explore the open floor glimpsed {side}")
        except Exception:  # noqa: BLE001 — telemetry must never break a look
            route = None
        self._route_cache = (best["id"], int(self.amap.updates), route)
        return route

    def _maybe_adopt_route(self, place: int, w: dm.Waypoint) -> None:
        """goto(staging place of a glimpse) = taking a ROUTE. The intent is
        planned on the ACCUMULATED map (A* over inflated verified-FREE, §14.6)
        and survives every observation until arrival, cancellation by
        negative evidence, or the model committing somewhere else."""
        reg = self._staging_of.get(place)
        # supersede check FIRST, for every committed waypoint (review P1/P2):
        # walking somewhere unrelated abandons the intent whether or not the
        # new place stages a region of its own — and a stale route must not
        # survive a failed adoption below. Two guards keep honest routes
        # alive: past staging, the residual sub-metre staging bearing is
        # meaningless (compare against the REGION the intent is about), and
        # walking the route's own second leg is following it, not leaving it.
        if self.route is not None:
            pub = self.route.as_public(self.amap)
            past_staging = (self.route.state in ("staging", "leg2_open")
                            or pub["staging"]["distance_m"] < 1.0)
            ref_deg = (pub["region"]["bearing_deg"] if past_staging
                       else pub["staging"]["bearing_deg"])
            gap = abs((math.degrees(w.angle) - ref_deg + 180.0) % 360.0
                      - 180.0)
            if gap > 75.0:
                self._route_history.append(
                    {"intent": self.route.intent,
                     "superseded": f"model chose {w.describe()}"})
                self._live_log({"route_superseded": self.route.intent})
                self.route = None
        if reg is None:
            return
        # §20.3: adopt the VERY route the telemetry's staging label was
        # derived from — planning a second time here could disagree with
        # the number the model just read.
        route = reg.get("route") if isinstance(reg, dict) else None
        if route is not None:
            if self.route is not None and self.route.frontier_id != route.frontier_id:
                self._route_history.append(
                    {"intent": self.route.intent,
                     "superseded": f"replaced by: {route.intent}"})
            self.route = route
            self._live_log({"route_adopted": route.as_public(self.amap)})

    def _route_update(self) -> dict | None:
        """§14.6's state machine, run once per observation: verify or cancel
        leg 2 from the ACCUMULATED evidence, notice staging arrival, retire
        the route on completion. Negative evidence is recorded, not erased."""
        r = self.route
        if r is None:
            return None
        try:
            ev = dm.route_evidence(self.amap, r)
        except Exception:  # noqa: BLE001
            return None
        if ev["occupied_frac"] > dm.ROUTE_CANCEL_OCC:
            rec = {"intent": r.intent,
                   "cancelled": "the glimpsed region is now measured "
                                "OCCUPIED — the glimpse was wrong",
                   **{k: round(v, 2) for k, v in ev.items()}}
            self._route_history.append(rec)
            self._emit_event("route_cancelled", "info", evidence=rec)
            self.route = None
            return {"intent": r.intent, "state": "cancelled",
                    "why": rec["cancelled"]}
        if ev["free_frac"] > dm.ROUTE_OPEN_FREE:
            r.legs[1]["status"] = "verified"
            if r.state in ("leg1", "staging"):
                r.state = "leg2_open"
        pub = r.as_public(self.amap)
        if r.state == "leg1" and pub["staging"]["distance_m"] <= 1.0:
            r.state = "staging"
        if (pub["region"]["distance_m"] <= 1.5
                and ev["free_frac"] > dm.ROUTE_OPEN_FREE):
            self._route_history.append({"intent": r.intent, "done": True})
            self._live_log({"route_done": r.intent})
            self.route = None
            return {**pub, "state": "done"}
        pub["state"] = r.state
        return pub

    # ── deterministic map telemetry (§10.6) ──────────────────────────────
    def _map_telemetry(self, td: dm.TopDown, prof, neck,
                       waypoints: list[dm.Waypoint]) -> dict[str, Any]:
        """Measurements only — sources, ages and confidences, never a verdict.

        No best_direction, no probably_at_junction, no target_likely_ahead:
        the model decides what the numbers mean. Everything here is read
        straight out of arrays the harness already computed, so the same map
        state always produces the same telemetry and it costs no API call."""
        mid = len(td.bearings) // 2

        def deg(r: float) -> float:
            return round(math.degrees(r), 1)

        landmark_dir: dict[str, float] = {}
        marks: list[dict[str, Any]] = []
        for s in self._sightings:
            landmark_dir[s.phrase] = float(s.bearing)
            marks.append({"name": s.phrase, "bearing_deg": deg(s.bearing),
                          "distance_m": round(float(s.distance), 1),
                          "status": "visible"})
        for r in self.amap.semantic_recall():
            if r["phrase"] in landmark_dir:
                continue
            landmark_dir[r["phrase"]] = float(r["bearing"])
            marks.append({"name": r["phrase"], "bearing_deg": deg(r["bearing"]),
                          "distance_m": (round(float(r["distance"]), 1)
                                         if self.amap.trusted else None),
                          "status": "remembered",
                          **({"behind": True} if r["behind"] else {})})
        toward: dict[int, dict[str, float]] = {}
        for i, w in enumerate(waypoints, 1):
            # wrapped to [0, 180]: a remembered landmark behind the body sits
            # near ±π and an unwrapped difference printed impossible 200°+
            toward[i] = {
                p: round(abs((math.degrees(w.angle - b) + 180.0) % 360.0
                             - 180.0))
                for p, b in landmark_dir.items()}
        islands = dm._obstacle_islands(td)
        # §10.4's honest tail: an opening whose flank is too short to land a
        # candidate is still a WAY — "turn to face it, then look" beats
        # silently offering only one side of an obstacle.
        unwalked = []
        for o in dm.openings(td, prof, neck):
            if any(abs(math.degrees(w.angle - o.centre)) < 10.0
                   for w in waypoints):
                continue
            unwalked.append({"bearing_deg": deg(o.centre),
                             "verified_reach_m": round(float(o.reach), 1),
                             "walkable_from_here": False})
        # §2.4/§20.3: glimpsed-open regions. The STAGING number is earned by
        # PATH CONSISTENCY, never by bearing: the route is planned first
        # (A*/info-gain on the accumulated map) and a numbered place is
        # published as staging only if it actually lies on the verified
        # first leg — the angularly-nearest number routinely sat on the
        # wrong side of the occluder and "place N is the nearest staging"
        # sent the body to the bar's face instead of along its flank.
        pots = []
        self._staging_of = {}
        for ridx, r in enumerate(dm.potential_regions(td)[:3]):
            entry = {**r, "status": "unverified — floor glimpsed beyond "
                                    "an occluder; will not be walked blind"}
            if ridx == 0:            # plan for the principal glimpse only
                route = self._route_for_region(r)
                if route is not None:
                    n = dm.staging_place_for(route, waypoints, self.amap)
                    if n is not None:
                        entry["staging_place"] = n
                        self._staging_of[n] = {"region": r, "route": route}
                    else:
                        pub = route.as_public(self.amap)
                        hb = pub["staging"]["bearing_deg"]
                        entry["route_hint"] = {
                            "bearing_deg": hb,
                            "note": ("no numbered place lies on the safe "
                                     f"first leg from here — {_turn_call(hb)} "
                                     "turns toward it; read the fresh "
                                     "numbers")}
            pots.append(entry)
        route_pub = self._route_update()
        # a warning that fired MID-LEG is delivered here, once — re-assessing
        # would let its own cooldown suppress the very evidence the interrupt
        # promised the model (review P1)
        last_warn = self.loop.last
        if last_warn and last_warn.get("warning") \
                and not last_warn.get("delivered"):
            warn = last_warn
        else:
            warn = self.loop.assess(self.amap)
        if warn:
            warn["delivered"] = True
        return {
            **({"route": route_pub} if route_pub else {}),
            "map": {"match": round(float(self.amap.last_score), 2),
                    "trusted": bool(self.amap.trusted),
                    "looks": int(self.amap.updates)},
            **({"loop": warn} if warn else {}),
            "potential_regions": pots,
            "ahead": {"verified_free_m": round(float(prof[mid]), 1),
                      "clear_sightline_m": round(float(td.free_range[mid]), 1)},
            "obstacles": [
                {"bearing_deg": [deg(float(td.bearings[a])),
                                 deg(float(td.bearings[b]))],
                 "nearest_m": round(float(td.free_range[a:b + 1].min()), 1)}
                for a, b in islands],
            "unwalked_openings": unwalked,
            "landmarks": marks,
            "_toward": toward,
        }

    @staticmethod
    def _telemetry_text(t: dict[str, Any]) -> str:
        """The telemetry as template prose — mechanical, no free generation."""
        lines = ["CURRENT GEOMETRY"]
        a = t["ahead"]
        lines.append(f"  ahead: verified FREE {a['verified_free_m']} m; "
                     f"clear sightline to {a['clear_sightline_m']} m")
        for o in t["obstacles"]:
            lo, hi = o["bearing_deg"]
            lines.append(f"  obstacle across {lo}°..{hi}° of your view, "
                         f"nearest about {o['nearest_m']} m")
        for u in t.get("unwalked_openings", []):
            b = u["bearing_deg"]
            side = "left" if b > 0 else "right"
            lines.append(f"  a way {side} at {abs(b):.0f}° — only "
                         f"{u['verified_reach_m']} m verified from here; "
                         "turn to face it and look before walking")
        for r in t.get("potential_regions", []):
            b = r["bearing_deg"]
            side = ("ahead" if abs(b) < 12
                    else f"{abs(b):.0f}° to your {'left' if b > 0 else 'right'}")
            if r.get("staging_place"):
                stage = (f" — place {r['staging_place']} lies ON the safe "
                         "first leg toward it; take it to go and look")
            elif r.get("route_hint"):
                hb = r["route_hint"]["bearing_deg"]
                stage = (f" — no numbered place lies on the safe first leg "
                         f"from here; {_turn_call(hb)} turns toward it, "
                         "then read the fresh numbers")
            else:
                stage = (" — no direct staging point from here; turning "
                         "may help")
            lines.append(f"  POTENTIAL (unverified): open floor glimpsed "
                         f"{side}, nearest about {r['nearest_m']} m, beyond "
                         f"an occluder{stage}. The ground between is UNKNOWN "
                         "and will not be walked blind.")
        rt = t.get("route")
        if rt:
            if rt.get("state") == "cancelled":
                lines.append(f"  ROUTE cancelled: {rt.get('why', '')} "
                             "(recorded — that direction is settled unless "
                             "new evidence reopens it)")
            else:
                st, rg = rt["staging"], rt["region"]
                st_side = ("ahead" if abs(st["bearing_deg"]) < 12 else
                           f"{abs(st['bearing_deg']):.0f}° to your "
                           f"{'left' if st['bearing_deg'] > 0 else 'right'}")
                leg2 = rt["legs"][1]["status"] if len(rt.get("legs", [])) > 1 \
                    else "unverified"
                hint = {"leg1": f"staging point {st_side}, "
                                f"{st['distance_m']} m by verified floor",
                        "staging": "you are AT the staging point — face the "
                                   "region and look; leg 2 is planned from "
                                   "what you see",
                        "leg2_open": "leg 2 is now on verified floor — the "
                                     "way in is open",
                        "done": "route complete"}.get(rt["state"], rt["state"])
                lines.append(f"  ROUTE ({rt['state']}): {rt['intent']} — "
                             f"{hint}; region at {rg['distance_m']} m, "
                             f"leg 2 {leg2}. Only verified legs are walked; "
                             "UNKNOWN is never crossed blind.")
        if t["landmarks"] or not t["map"]["trusted"]:
            lines.append("MAP MEMORY")
        for m in t["landmarks"]:
            b = m["bearing_deg"]
            if m.get("behind"):
                where = "behind you"
            elif abs(b) < 12:
                where = "straight ahead"
            else:
                where = f"{abs(b):.0f}° to your {'left' if b > 0 else 'right'}"
            d = f", about {m['distance_m']} m" if m.get("distance_m") else ""
            lines.append(f"  {m['name']}: {where}{d} ({m['status']})")
        mp = t["map"]
        lines.append(f"  map registration: match {mp['match']} over "
                     f"{mp['looks']} looks"
                     + ("" if mp["trusted"]
                        else " — TREAT REMEMBERED DISTANCES AS ROUGH"))
        lw = t.get("loop")
        if lw:
            why = ", ".join(lw.get("signals", {}))
            sec = lw.get("repeated_sector_deg")
            lines.append(
                f"  LOOP WARNING (score {lw['loop_score']}): walked "
                f"{lw['path_m']} m for {lw['net_m']} m of net progress"
                + (f"; kept choosing the ~{sec:+d}° sector" if sec is not None
                   else "")
                + f" [{why}]. The trail on the map shows where you have "
                  "already been — pick a direction that leaves it.")
        return "\n".join(lines)

    # ── plumbing ─────────────────────────────────────────────────────────
    @staticmethod
    def _result(result: dict[str, Any]) -> ToolResult:
        public = {k: v for k, v in result.items() if k != "kind"}
        return ToolResult(content=[text_part(json.dumps(public))], info=result)

    def _write_depth_view(self, td: dm.TopDown | None,
                          waypoints: list[dm.Waypoint]) -> None:
        """Publish the organ's private geometry for the monitor at :5173.

        The model never sees any of this — it is written purely so a human can
        watch what the depth organ believed at each decision."""
        if self.live_dir is None or td is None:
            return
        try:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            seen = "; ".join(
                f"{s.phrase} {'L' if s.bearing > 0 else 'R'}"
                f"{abs(math.degrees(s.bearing)):.0f}° {s.distance:.1f}m"
                for s in self._sightings)
            png = dm.render_topdown(td, waypoints, caption=dm.topdown_caption(
                td, waypoints,
                extra=(f"obs {self._obs_count} · step {self.steps_taken} · "
                       f"goto ×{self._gotos}"
                       + (f"\nSAM3 sees: {seen}" if seen else ""))))
            if png:
                (self.live_dir / "topdown_latest.png").write_bytes(png)
                (self.live_dir / f"topdown_{self._obs_count:04d}.png").write_bytes(png)
            if self.places is not None:
                (self.live_dir / "places.txt").write_text(
                    f"obs {self._obs_count} · step {self.steps_taken}\n"
                    + self.places.readout()
                    + "\n\n给模型的句子：\n  "
                    + "\n  ".join(self.places.lines(self._sightings) or ["（无）"]),
                    encoding="utf-8")
                (self.live_dir / "places.json").write_text(
                    json.dumps(self.places.dump(), ensure_ascii=False, indent=1),
                    encoding="utf-8")
            amap_png = self.amap.render(waypoints, caption=self.amap.caption())
            if amap_png:
                (self.live_dir / "map_latest.png").write_bytes(amap_png)
                (self.live_dir / f"map_{self._obs_count:04d}.png").write_bytes(amap_png)
            # …and the EXACT picture the model receives as IMAGE 2, so the
            # monitor can show what the model sees, not a stand-in for it.
            # §21.7: that picture is now the dual-truth composite.
            model_png = self.amap.render_composite(
                waypoints, current={s.phrase for s in self._sightings},
                caption=(f"map_version {int(self.amap.updates)} · "
                         f"menu epoch {self._candidate_epoch}"))
            if model_png:
                (self.live_dir / "map_model_latest.png").write_bytes(model_png)
                (self.live_dir / f"map_model_{self._obs_count:04d}.png"
                 ).write_bytes(model_png)
            bearing, widest = td.widest()
            grid = td.grid
            total = float(grid.size) or 1.0
            payload = {
                "obs": self._obs_count,
                "steps_taken": self.steps_taken,
                "gotos": self._gotos,
                "map_looks": self.amap.updates,
                "map_match": round(self.amap.last_score, 2),
                "map_fixes": self.amap.fixes,
                "map_recall": self.amap.recall_sentence(),
                "depth_units": "normalized[0,1]→×10" if td.normalized_depth else "metres",
                "floor_below_camera_m": round(-td.floor_y, 2),
                "range_cap_m": self.range_cap_m,
                "cell_m": dm.CELL_M,
                "ahead_m": round(td.ahead_m(), 2),
                "widest_bearing_deg": round(math.degrees(bearing), 1),
                "widest_m": round(widest, 2),
                "free_pct": round(100 * float((grid == dm.FREE).sum()) / total),
                "occupied_pct": round(100 * float((grid == dm.OCCUPIED).sum()) / total),
                "unknown_pct": round(100 * float((grid == dm.UNKNOWN).sum()) / total),
                "candidates": [
                    {"n": i + 1, "kind": w.kind,
                     "angle_deg": round(math.degrees(w.angle), 1),
                     "target_m": round(w.distance, 2),
                     "stride_m": round(w.stride_m or w.distance, 2),
                     "verified_m": round(w.verified_m, 2),
                     "visible_m": round(w.visible_m, 2),
                     "confidence": w.confidence,
                     "clearance_m": round(w.clearance, 2),
                     "squeeze_m": w.extras.get("squeeze"),
                     "env_steps": int((w.stride_m or w.distance) / 0.25),
                     "where": w.describe()}
                    for i, w in enumerate(waypoints)
                ],
                "landmarks": [
                    {"phrase": s.phrase, "bearing_deg": round(math.degrees(s.bearing), 1),
                     "distance_m": round(s.distance, 2), "score": round(s.score, 2)}
                    for s in self._sightings
                ],
                "landmark_ledger": {
                    p: self.register.evidence_line(p, self.steps_taken)
                    for p in self.phrases
                },
                "detector": ("off" if self.landmarks is None else
                             "ok" if self.landmarks.available else
                             f"unavailable: {self.landmarks.last_error[:120]}"),
            }
            (self.live_dir / "depth.json").write_text(json.dumps(payload, indent=1))
            (self.live_dir / f"depth_{self._obs_count:04d}.json").write_text(
                json.dumps(payload, indent=1))
            # publish ONLY now — the versioned depth json above is the LAST
            # artifact of this observation. Publishing before it existed made
            # the snapshot's own exists-check bail on every single call and
            # live_snapshot.json was never written at all (review P0).
            self._publish_snapshot()
            # bounded retention for versioned monitor copies (audit P2): the
            # snapshot only ever references the CURRENT observation, so
            # anything 300 observations old is scrollback nobody can reach
            if self._obs_count % 50 == 0 and self._obs_count > 300:
                cutoff = self._obs_count - 300
                for pat in ("topdown_[0-9]*", "map_[0-9]*",
                            "map_model_[0-9]*", "depth_[0-9]*"):
                    for f in self.live_dir.glob(pat):
                        try:
                            if int(f.stem.split("_")[-1]) < cutoff:
                                f.unlink()
                        except (ValueError, OSError):
                            pass
        except Exception:  # noqa: BLE001 — telemetry must never end an episode
            pass

    def _publish_snapshot(self) -> None:
        """§6: the ATOMIC live snapshot. Called only after EVERY versioned
        file of this observation is on disk, then os.replace lands the
        manifest in one move — a monitor that reads the manifest first can
        never again stitch telemetry N to RGB N−1 and map N+1."""
        if self.live_dir is None:
            return
        try:
            snap = {
                "obs_id": self._obs_count,
                "env_step": self.steps_taken,
                "map_version": int(self.amap.updates),
                "gotos": self._gotos,
                # §14.14: snapshot IDENTITY — enough to tell which run,
                # episode, executor and high-level action this frame belongs
                # to, so the monitor can refuse to stitch strangers together
                "identity": {
                    "run_id": self.live_dir.parent.name,
                    "episode": self.live_dir.name,
                    "executor": self._executor,
                    "action_id": self._current_action,
                    "sensor_frame": self._sensor_frame,
                    "candidate_epoch": self._candidate_epoch,
                },
                "rgb_file": (f"obs_{self._obs_count:04d}_"
                             f"step{self.steps_taken:03d}.png"),
                "topdown_file": f"topdown_{self._obs_count:04d}.png",
                "accumulated_map_file": f"map_{self._obs_count:04d}.png",
                "model_map_file": f"map_model_{self._obs_count:04d}.png",
                "depth_json_file": f"depth_{self._obs_count:04d}.json",
                "t": round(time.time() - self._t0, 2),
            }
            for key in ("rgb_file", "topdown_file", "accumulated_map_file",
                        "model_map_file", "depth_json_file"):
                if not (self.live_dir / snap[key]).exists():
                    return          # a partial write earlier — do not publish
            import os as _os
            tmp = self.live_dir / "live_snapshot.json.tmp"
            tmp.write_text(json.dumps(snap, indent=1))
            _os.replace(tmp, self.live_dir / "live_snapshot.json")
        except Exception:  # noqa: BLE001 — telemetry must never end an episode
            pass

    def _live_log(self, entry: dict[str, Any]) -> None:
        if self.live_dir is None:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)
        with (self.live_dir / "actions.log").open("a") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **entry}) + "\n")
