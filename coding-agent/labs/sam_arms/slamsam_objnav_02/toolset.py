"""slamsam_objnav_02 toolset — ObjectNav (HM3D / MP3D / HM3D-OVON) on the
SLAM map, with SAM 3 as an EXTERNAL TOOL the model calls (user 2026-08-18:
"把 SAM 变成一个工具, 模型需要检测物体的时候, 再自动检测它返回的图 … 纯 ReAct
的模式 … pillow is not cushion, 严格地去检测, 就只有这一个词"; then "SAM 不可以
标到 SLAM 图上 … 置信度只有在 0.8 或者 0.85 以上才可以"; then "可以把 SAM 检测的
结果标到地图上, 但只有在它的 score 很高 (比如 0.85 以上) 的情况下再标" and
"最后没有贴得足够近 … 应该人为地把这个往前推, 离目标足够近").

Surface — jian's SLAM-instrument shape, auto-observe, plus ONE detector tool:
    step(actions)      primitives 0/1/2/3 (30° turns); the result carries the
                       photo from where the body now stands — and nothing
                       else: no detector watches the frames, no push while
                       walking. STOP (0) runs the FINAL PUSH first: if the
                       gated detector sees the target in the current view,
                       the harness turns to face it and walks it down (until
                       ~PUSH_STOP_M or blocked or out of sight, ≤ PUSH_MAX
                       primitives, all counted), THEN stops.
    detect_target()    run SAM 3 for the exact goal word on the CURRENT view
                       (the frame the last step showed); free. Only matches
                       scoring ≥ SAM_SCORE_THRESH (0.85 in the bridge) come
                       back at all (no synonyms: SAM_SYNONYMS=0). Hit → the
                       overlay image (what the detector matched, labelled
                       with distance) + per-instance dir_deg / dist_m /
                       score, AND those high-score matches are stamped on
                       the map at once. Miss → JSON only, nothing stamped.
    get_map()          the SLAM occupancy map (jian's integrate + frontier
                       renderer, map v1) with the high-score detection
                       patches, listed under `landmarks` (dir_deg / dist_m).
                       Patches never decay on this arm (a strict gate misses
                       real objects too often to let a miss erase a stamp);
                       the model judges.
    get_pose()         (x, z, yaw_deg) in the start-anchored frame
    get_trajectory()   the path so far, oldest first
No observe tool (auto-observe), no recall on this arm.

Underneath, per executed primitive: odometry from the env's measured motion
(actual_translation_m / actual_dy_m + commanded yaw), one RGB-D read,
OccupancyMap.integrate at that pose. The RGB-D of the LAST sensed frame is
kept with its pose so detect_target() runs on exactly the view the model is
looking at, fuses each mask with that frame's depth for bearing / distance,
and projects it through that pose onto the map.

Disk (live_dir): the frames the model saw (obs_NNNN_stepSSS.png,
obs_NNNN_map.png, bootstrap_current.png), the detector overlays it was shown
(det_NNNN_stepSSS.png) + ONE current map (map_latest.png).
Forked 2026-08-18 from exp_workspace/slamsam_ovon_01/toolset.py (the push arm).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# exp_workspace/slamsam_objnav_02 copy (2026-08-18 — see README.md).
# Execution code lives HERE (this file + slam_sidecar.py + slam/ +
# nodeset_ovon/ + nodeset_objnav/); only library code is imported from the
# shared tree: the mini NodesetToolSet base (harnesses/mini/toolset.py) and
# eharness.depthmap / eharness.landmarks (depth decode, mask projection, the
# SAM client + overlay painter).
_HERE = Path(__file__).resolve().parent
_CODING_AGENT = _HERE.parents[1]
for _p in (str(_CODING_AGENT), str(_CODING_AGENT / "harnesses" / "mini"),
           str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu  # noqa: E402

# the shared mini tool base (harnesses/mini/toolset.py) — loaded by path
# because this folder's own file is ALSO called toolset.py
_bspec = _ilu.spec_from_file_location(
    "_mini_toolset_base", _CODING_AGENT / "harnesses" / "mini" / "toolset.py")
_base = sys.modules.get("_mini_toolset_base")
if _base is None:
    _base = _ilu.module_from_spec(_bspec)
    sys.modules["_mini_toolset_base"] = _base   # dataclasses need it findable
    _bspec.loader.exec_module(_base)
MAX_ACTIONS_PER_CALL = _base.MAX_ACTIONS_PER_CALL
NodesetToolSet = _base.NodesetToolSet
ToolResult = _base.ToolResult
png_part = _base.png_part
text_part = _base.text_part

from eharness import depthmap as dm  # noqa: E402

STEP_M = 0.25
BLOCKED_M = 0.05        # a forward that moved less than this hit something
STAMP_VOTES = 2.0       # one gated (high-score) detection = a rendered map
                        # patch (CELL_SHOW_MIN 0.9 with score ≥ 0.85)
# the final push (user 2026-08-18: 人为地把这个往前推, 离目标足够近): on STOP,
# with the target detected in view, face it and walk it down before stopping
PUSH_STOP_M = 0.5       # close enough — stop pushing here
PUSH_MAX_STEPS = 8      # forward primitives at most (2 m)
PUSH_MAX_TURNS = 2      # face it with at most ±2 turns (±60°)


# ── the tool descriptions (single source: bridge + tests import these) ──

def step_desc(turn_deg: float, task: str = "objnav",
              push_stop_m: float = PUSH_STOP_M) -> str:
    return (
        "Execute a sequence of movement actions, in order. Actions: 0 = STOP "
        "(permanently ENDS the episode — issue it only when you believe you "
        "are within 1 meter of the target object), 1 = move forward "
        f"{STEP_M} m, 2 = turn left {turn_deg:g} degrees, 3 = turn right "
        f"{turn_deg:g} degrees. The result carries the camera view from where "
        "the robot now stands plus a short JSON: actions executed, metres "
        "moved, whether a forward step was blocked (the rest of that call is "
        "then cancelled), steps used and remaining, episode_over. Looking is "
        "free and automatic — there is no separate look action; a small turn "
        "is the cheapest way to look around. Nothing watches the frames for "
        "you: to check a view for the target object, call detect_target(). "
        "FINAL PUSH: when you issue STOP and the detector sees the target in "
        "your current view, the robot first turns to face it and walks up to "
        f"it (until about {push_stop_m:g} m, or blocked, or it leaves the "
        "view; a few steps at most, all counted) and only then stops — the "
        "JSON reports what it did under `final_push`. So STOP as soon as you "
        "are sure the object in front of you is the target; you need not "
        "shave off the last centimetres yourself."
    )


def detect_desc(goal: str, score_thresh: float = 0.85) -> str:
    g = goal or "the target object"
    return (
        f"Run an external object detector for the target object ('{g}') on "
        "the camera view you are looking at right now (the view the last "
        "step() result showed). Free — no step cost, call it as often as you "
        "like. It is asked for that exact word and nothing else — no "
        f"synonyms — and only matches it scores {score_thresh:g} or higher "
        "(out of 1) are returned at all; weaker ones are dropped silently. "
        "If it finds one or more, the result carries an image of your view "
        "with each match painted and labelled with its distance, plus a JSON "
        "list of instances (dir_deg relative to your heading, positive = to "
        "your RIGHT; dist_m from where you stand; score), and those "
        "high-score matches are stamped on the map at once (get_map shows "
        "each as a tinted patch carrying the target's name, listed under "
        "`landmarks` with direction and distance from wherever you then "
        "stand). If it finds nothing, the JSON says so and nothing is "
        "stamped. It matches by RESEMBLANCE and can still be wrong (asked "
        "for 'pillow' it may paint a sofa cushion): a match is a CANDIDATE, "
        "not a verdict — look at the painted region and decide whether it "
        f"really is a '{g}' in the plain sense of the word, the object "
        "itself, not a part of another piece of furniture, not a look-alike. "
        "Walk only to matches you accept. To close in on one: turn toward "
        "it, walk, call detect_target() again to re-read dist_m, until it is "
        "about 1 m in front of you, then STOP (the robot walks the last bit "
        "itself before stopping)."
    )


GET_MAP_DESC = (
    "Read the occupancy map built automatically as you move — free, no step "
    "cost. Returns a top-down image cropped to what you have explored — UP on "
    "the image is your STARTING heading (+z), right is +x. On it: white = "
    "walked-free space, black = obstacle, gray = unexplored; the blue ARROW "
    "is you, pointing your current heading; the blue LINE is your path so "
    "far; the \"S\" circle is your start; numbered green circles are "
    "frontiers — openings into unexplored space — where circle N is frontier "
    "\"FN\" in the accompanying JSON (each with dir_deg relative to your "
    "current heading, positive = to your right, distance in meters, and "
    "size); frontier ids are STABLE across calls. Faint gridlines every 2 m "
    "are labeled with the same x/z coordinates get_pose() reports, and a "
    "scale bar shows 2 m. Where detect_target() matched the TARGET OBJECT at "
    "high confidence, a tinted patch carrying its name is painted and listed "
    "under `landmarks` (dir_deg / dist_m from where you stand); patches stay "
    "— the map does not judge them, you do. SLAM estimates, not ground truth."
)

GET_POSE_DESC = (
    "Read the robot's SLAM-estimated pose: position (x, z) in meters and "
    "heading yaw_deg, in a fixed frame anchored at your start pose (x = right "
    "and z = forward OF YOUR STARTING POSE; yaw_deg is 0 at your starting "
    "heading and INCREASES when you turn right). The same x/z coordinates "
    "label the gridlines on get_map(), so pose numbers place you on the map. "
    "Free — no step cost."
)

GET_TRAJECTORY_DESC = (
    "Read your own path so far as (x, z) points in the same fixed frame as "
    "get_pose(), oldest first. Useful to check whether you are circling or "
    "which areas you already covered. Free — no step cost."
)
GET_POSE_SCHEMA = {"properties": {}, "title": "getPoseArguments",
                   "type": "object"}
GET_TRAJ_SCHEMA = {"properties": {}, "title": "getTrajectoryArguments",
                   "type": "object"}

GET_MAP_DESC_V2 = (
    "Read the top-down map your onboard SLAM builds as you move — free, no "
    "step cost. UP on the image is your STARTING heading. White = floor you "
    "have seen, black = obstacles, gray = unexplored; the blue arrow is you "
    "(pointing your current heading), the blue line is your path, the S "
    "circle is your start. The coordinate grid (2 m lines, labeled on both "
    "axes) stays FIXED between calls — the window only grows, never pans — "
    "so a place keeps its position on the map from one look to the next. "
    "There is no frontier list: unexplored space is the gray next to the "
    "white you have covered. A tinted patch with a name is where "
    "detect_target() matched the TARGET OBJECT at high confidence (also "
    "listed with direction and distance); patches stay — the map does not "
    "judge them, you do. SLAM estimates can drift slightly."
)


def get_map_desc(map_mode: str = "v1") -> str:
    return GET_MAP_DESC_V2 if str(map_mode) == "v2" else GET_MAP_DESC


STEP_SCHEMA = {
    "properties": {"actions": {"items": {"type": "integer"},
                               "title": "Actions", "type": "array"}},
    "required": ["actions"], "title": "stepArguments", "type": "object",
}
GET_MAP_SCHEMA = {"properties": {}, "title": "getMapArguments",
                  "type": "object"}
DETECT_SCHEMA = {"properties": {}, "title": "detectTargetArguments",
                 "type": "object"}


class LeanSlamToolSet(NodesetToolSet):
    """step + detect_target + get_map + get_pose + get_trajectory over any env
    that speaks the two habitat verbs (env_ovon / env_objnav)."""

    def __init__(
        self,
        server_url: str,
        *,
        verb: str = "env_habitat",
        phrases: list[str] | None = None,
        step_budget: int = 500,
        live_dir: Path | None = None,
        sam_url: str = "",
        turn_deg: float = 30.0,
        task: str = "objnav",
        sam_synonyms: int = 0,
        sam_score_thresh: float = 0.85,
        cam_height_fallback_m: float = 0.88,
        map_mode: str = "v1",
        final_push: bool = True,
        push_stop_m: float = PUSH_STOP_M,
        push_max_steps: int = PUSH_MAX_STEPS,
    ) -> None:
        super().__init__(server_url)
        self.verb = verb
        self.phrases = [str(p) for p in (phrases or []) if str(p).strip()]
        self.step_budget = int(step_budget)
        self.live_dir = Path(live_dir) if live_dir else None
        self.turn_deg = float(turn_deg)
        self.turn_rad = math.radians(self.turn_deg)
        self.task = task
        self.sam_synonyms = max(0, int(sam_synonyms))
        self.sam_score_thresh = float(sam_score_thresh)
        self.cam_h_fallback = float(cam_height_fallback_m)
        # "v1" = jian's frontier map (the smoked口径); "v2" = his slam_r2r_02
        # map (no frontier, 2 m-snapped grow-only window)
        self.map_mode = str(map_mode or "v1")
        self.final_push = bool(final_push)
        self.push_stop_m = float(push_stop_m)
        self.push_max_steps = int(push_max_steps)

        self.steps_taken = 0
        self.episode_over = False
        self.end_reason: str | None = None
        # odometry: theta left-positive, px right-of-start, py forward-of-start,
        # pz metres ABOVE the start floor (from the env's per-primitive
        # actual_dy_m — stairs; 0 when the env does not report it)
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.theta = 0.0
        self.slam: Any = None            # SlamMapSidecar, built on first depth
        self.hfov_deg = 90.0
        self.organ: Any = None           # LandmarkOrgan, if sam_url + goal
        self._sam_url = sam_url
        self.detect_stats = {"calls": 0, "hits": 0, "errors": 0, "seconds": 0.0,
                             "push_detects": 0}
        # the episode's own video: every frame the model saw, in order
        self.frames: list[dict] = []     # {idx, step, kind, path|png}
        self._obs_count = 0
        self._det_count = 0
        # the LAST sensed frame — what the model is looking at — kept whole
        # (with its pose) so detect_target() runs on it, fuses its masks with
        # its depth and projects them through its pose onto the map
        self._last_rgb_png: bytes | None = None
        self._last_rgb_b64: str | None = None
        self._last_depth_m: np.ndarray | None = None
        self._last_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._map_pool: ThreadPoolExecutor | None = None
        self._map_future: Future | None = None
        self._t0 = time.time()

        # the goal word: the ONE phrase the detector is ever asked for
        self.goal = self.phrases[0] if self.phrases else ""
        if sam_url and self.goal:
            from eharness.landmarks import LandmarkOrgan
            self.organ = LandmarkOrgan(sam_url, hfov_deg=self.hfov_deg,
                                       score_thresh=self.sam_score_thresh)

        self._register("step", step_desc(self.turn_deg, self.task,
                                         self.push_stop_m),
                       STEP_SCHEMA, self._tool_step)
        self._register("detect_target",
                       detect_desc(self.goal, self.sam_score_thresh),
                       DETECT_SCHEMA, self._tool_detect_target)
        self._register("get_map", get_map_desc(self.map_mode), GET_MAP_SCHEMA,
                       self._tool_get_map)
        self._register("get_pose", GET_POSE_DESC, GET_POSE_SCHEMA,
                       self._tool_get_pose)
        self._register("get_trajectory", GET_TRAJECTORY_DESC, GET_TRAJ_SCHEMA,
                       self._tool_get_trajectory)

    # ── pose helpers ────────────────────────────────────────────────────

    def _pose_ns(self) -> SimpleNamespace:
        return SimpleNamespace(px=self.px, py=self.py, theta=self.theta,
                               pz=self.pz)

    def _odom_turn(self, action: int) -> None:
        self.theta += self.turn_rad if action == 2 else -self.turn_rad

    def _odom_forward(self, d: float) -> None:
        if d > 1e-9:
            self.px += d * -math.sin(self.theta)
            self.py += d * math.cos(self.theta)

    # ── sensing: one RGB-D read → SLAM integrate; keep the frame ─────────

    @staticmethod
    def _depth_metres(depth: np.ndarray | None, units: dict) -> np.ndarray | None:
        if depth is None or depth.size == 0:
            return None
        if units and units.get("known"):
            if units.get("normalized"):
                lo = float(units.get("min_depth_m") or 0.0)
                hi = float(units.get("max_depth_m") or dm.DEPTH_FULL_RANGE_M)
                raw = depth.astype(np.float32)
                m = raw * (hi - lo) + lo
                # habitat CLIPS to [min, max] before normalising: a pixel
                # nearer than min_depth reads as exactly min_depth (ObjectNav /
                # OVON: 0.5 m). Left in, every near surface — the wall you
                # hug, the table edge, the door you face — is stamped as a
                # phantom obstacle shell 0.5 m out. A clipped pixel is NOT a
                # measurement: mark it invalid (0 → below the mapper's
                # depth_min, ignored by SAM projection too).
                if lo > 0.0:
                    m[raw <= 0.0] = 0.0
                return m
            return depth.astype(np.float32)
        return dm.to_metres(depth, None)[0]

    def _ensure_slam(self, depth_m: np.ndarray, intr: dict, units: dict) -> None:
        if self.slam is not None:
            return
        h, w = depth_m.shape[:2]
        try:
            fx = float(intr.get("fx") or 0.0)
            iw = float(intr.get("width") or w)
            if fx > 0:
                self.hfov_deg = math.degrees(2.0 * math.atan(iw / (2.0 * fx)))
        except Exception:  # noqa: BLE001
            pass
        cam_h = float(units.get("camera_height_m") or 0.0) if units else 0.0
        if not cam_h:
            cam_h = self.cam_h_fallback
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            f"_sidecar_{_HERE.name}", _HERE / "slam_sidecar.py")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        SlamMapSidecar = _mod.SlamMapSidecar
        # the semantic layer holds ONLY the goal word — fed by the gated
        # detector (score ≥ thresh) through _stamp, never by a stream
        self.slam = SlamMapSidecar(sam_phrases=list(self.phrases),
                                   cam_height_m=cam_h, rgb_hw=(h, w),
                                   hfov_deg=self.hfov_deg,
                                   map_mode=self.map_mode)
        if self.organ is not None:
            self.organ.hfov_deg = self.hfov_deg

    def _sense(self) -> bytes | None:
        """Read the current frame; integrate; keep it. Returns the PNG."""
        try:
            out = self._call(f"{self.verb}__observe_egocentric", {})
        except Exception:  # noqa: BLE001 — a lost read must not end the walk
            return None
        rgb_b64 = out.get("rgb")
        png = base64.b64decode(rgb_b64) if isinstance(rgb_b64, str) else None
        depth = dm.decode_depth(out.get("depth"))
        units = out.get("depth_units") or {}
        depth_m = self._depth_metres(depth, units)
        if depth_m is not None and depth_m.ndim == 2 \
                and not bool(np.isnan(depth_m).all()):
            try:
                self._ensure_slam(depth_m, out.get("intrinsics") or {}, units)
                self.slam.integrate(depth_m, self.px, self.py, self.theta,
                                    pz=self.pz)
            except Exception:  # noqa: BLE001 — the map must never brake a walk
                pass
            self._last_depth_m = depth_m
        else:
            self._last_depth_m = None
        if png is not None:
            self._last_rgb_png = png
            self._last_rgb_b64 = rgb_b64 if isinstance(rgb_b64, str) else None
        self._last_pose = (self.px, self.py, self.theta)
        return png

    # ── the detector: an external tool on the current frame ──────────────

    def _detect_now(self) -> list | None:
        """SAM 3 for the goal word on the last sensed frame, fused with that
        frame's depth for bearing / distance. Returns the sightings that
        clear the strict gate, or None when there is no frame / no
        detector. Does not stamp — the caller decides."""
        if self.organ is None or not self.goal:
            return None
        if self._last_rgb_b64 is None or self._last_depth_m is None:
            self._sense()                       # nothing looked at yet: look
        rgb_b64, depth_m = self._last_rgb_b64, self._last_depth_m
        if rgb_b64 is None or depth_m is None:
            return None
        t0 = time.time()
        try:
            sights = self.organ.sightings(rgb_b64, depth_m, [self.goal],
                                          scale_m=1.0, synonyms=self.sam_synonyms)
        except Exception:  # noqa: BLE001 — a dead detector is a plain miss
            self.detect_stats["errors"] += 1
            sights = []
        self.detect_stats["seconds"] += time.time() - t0
        # strict: the goal word itself (LandmarkOrgan already reports under
        # the asked-for phrase; with synonyms=0 nothing else was even tried)
        # and only matches at or above the score gate — the SAM host was
        # asked with the same threshold, this re-check is the contract
        return [s for s in sights
                if s.phrase == self.goal and s.mask is not None
                and float(s.score) >= self.sam_score_thresh]

    def _stamp(self, sights: list) -> None:
        """Project the gated masks through the last frame's depth + pose and
        stamp them on the map (user 2026-08-18: 只有在 score 很高的情况下再标).
        No decay on this arm."""
        if self.slam is None or not sights or self._last_depth_m is None:
            return
        px, py, th = (float(v) for v in self._last_pose)
        for s in sights:
            try:
                xs, ys = dm.project_mask(s.mask, self._last_depth_m, None,
                                         hfov_deg=self.hfov_deg, scale_m=1.0)
            except Exception:  # noqa: BLE001 — advisory layer, never fatal
                continue
            if getattr(xs, "size", 0):
                self.slam.stamp_points(s.phrase, xs, ys, px, py, th,
                                       float(s.score), votes=STAMP_VOTES)

    @staticmethod
    def _instances(sights: list) -> list[dict[str, Any]]:
        out = []
        for s in sorted(sights, key=lambda s: float(s.distance)):
            out.append({
                "dir_deg": round(-math.degrees(float(s.bearing)), 1),
                "dist_m": round(float(s.distance), 2),
                "width_deg": round(math.degrees(float(s.angular_width)), 1),
                "score": round(float(s.score), 2)})
        return out

    def _tool_detect_target(self, **_ignored: Any) -> ToolResult:
        if not self.goal:
            return self._json_only({"error": "no target word for this episode"},
                                   kind="detect_target")
        if self.organ is None:
            return self._json_only({"error": "detector unavailable on this "
                                             "run (no SAM server configured)"},
                                   kind="detect_target")
        self.detect_stats["calls"] += 1
        sights = self._detect_now()
        if sights is None:
            return self._json_only({"target": self.goal, "detected": False,
                                    "error": "no camera frame to detect on yet"},
                                   kind="detect_target")
        self._stamp(sights)
        instances = self._instances(sights)
        result: dict[str, Any] = {"target": self.goal,
                                  "detected": bool(instances),
                                  "instances": instances,
                                  "steps_taken_total": self.steps_taken}
        content: list[dict[str, Any]] = []
        if instances:
            self.detect_stats["hits"] += 1
            result["stamped_on_map"] = True
            result["note"] = (
                f"{len(instances)} candidate match(es) for '{self.goal}' in "
                "the current view (score ≥ "
                f"{self.sam_score_thresh:g}) — painted on the attached image "
                "(label = distance) and stamped on the map (get_map lists "
                "them under landmarks). dir_deg positive = to your right; "
                "dist_m from where you stand. Judge each from the overlay "
                f"before you walk to it: is the painted thing really a "
                f"'{self.goal}' in the plain sense of the word — the object "
                "itself, not a part of some other piece of furniture, not a "
                "look-alike? The detector matches by resemblance and can be "
                "wrong.")
            overlay = self._overlay_png(sights)
            if overlay is not None:
                idx = self._record_frame(
                    overlay, "detect",
                    filename=self._next_det_filename())
                result["frame"] = idx
                content.append(png_part(overlay))
            self._publish_map_async()
        else:
            result["note"] = (f"no '{self.goal}' at score ≥ "
                              f"{self.sam_score_thresh:g} in the current view. "
                              "If you can see one yourself, walk closer / face "
                              "it squarely and ask again.")
        content.append(text_part(json.dumps(result)))
        info = {"kind": "detect_target", "detected": bool(instances),
                "n": len(instances), "steps_taken_total": self.steps_taken}
        if instances:
            info["nearest_m"] = instances[0]["dist_m"]
        return ToolResult(content=content, info=info)

    def _overlay_png(self, sights: list) -> bytes | None:
        try:
            from eharness.landmarks import render_masks
            png = render_masks(self._last_rgb_b64 or "", sights)
            return png or None
        except Exception:  # noqa: BLE001 — the JSON still answers
            return None

    def _next_det_filename(self) -> str:
        self._det_count += 1
        return f"det_{self._det_count:04d}_step{self.steps_taken:03d}.png"

    # ── frames on disk (the video) ───────────────────────────────────────

    def _record_frame(self, png: bytes, kind: str,
                      filename: str | None = None) -> int:
        idx = len(self.frames)
        rec: dict[str, Any] = {"idx": idx, "step": self.steps_taken,
                               "kind": kind, "t": round(time.time() - self._t0, 1)}
        if self.live_dir is not None:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            if filename is None:
                self._obs_count += 1
                filename = (f"obs_{self._obs_count:04d}_map.png" if kind == "map"
                            else f"obs_{self._obs_count:04d}_step{self.steps_taken:03d}.png")
            (self.live_dir / filename).write_bytes(png)
            rec["path"] = filename
        else:
            rec["png"] = png
        self.frames.append(rec)
        return idx

    # ── the current map on disk: keep-latest, off the caller's thread ────

    def _publish_map_async(self) -> None:
        if self.slam is None or self.live_dir is None:
            return
        try:
            snap = self.slam.snapshot_for_render(self._pose_ns())
        except Exception:  # noqa: BLE001
            return
        if self._map_pool is None:
            self._map_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="lean-map")
        if self._map_future is not None and not self._map_future.done():
            self._map_future.cancel()
        live, slam = self.live_dir, self.slam

        def _work() -> None:
            try:
                png = slam.render_snapshot(snap)
                tmp = live / "map_latest.png.tmp"
                tmp.write_bytes(png)
                os.replace(tmp, live / "map_latest.png")
            except Exception:  # noqa: BLE001 — a monitor artifact, never fatal
                pass

        self._map_future = self._map_pool.submit(_work)

    # ── bootstrap: frame 0 as the first message's image ─────────────────

    def bootstrap(self) -> dict[str, Any]:
        """The opening look: one pure read at the seated start pose. Writes
        bootstrap_current.png + bootstrap.json (the SDK adapter's existing
        first-message contract) and records it as frame#0."""
        png = self._sense()
        art: dict[str, Any] = {"texts": [], "images": {}}
        if png is not None:
            self._record_frame(png, "view", filename="bootstrap_current.png")
            art["texts"] = [
                "IMAGE 1 — your opening view (frame#0): the camera view from "
                "where you start, facing your starting heading. Attached "
                "free — no steps were spent on it."]
            art["images"] = {"current": "bootstrap_current.png"}
        if self.live_dir is not None:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.live_dir / "bootstrap.json.tmp"
            tmp.write_text(json.dumps(art, ensure_ascii=False, indent=1))
            os.replace(tmp, self.live_dir / "bootstrap.json")
        self._publish_map_async()
        return art

    # ── one primitive: env step + odometry + sensing ─────────────────────

    def _exec_primitive(self, a: int) -> dict[str, Any]:
        """Send ONE primitive to the env and book it: steps, episode end,
        odometry (measured translation / climb, commanded yaw), the blocked
        flag (+ obstacle mark on the map), then one RGB-D read at the new
        pose. Returns {ok, err, blocked, d, png}. ok=False → nothing was
        executed (transport error or the env refused)."""
        rec: dict[str, Any] = {"ok": False, "err": None, "blocked": False,
                               "d": 0.0, "png": None}
        try:
            out = self._call(f"{self.verb}__step_discrete", {"action": a})
        except Exception as exc:  # noqa: BLE001
            rec["err"] = f"env step failed: {exc!r}"
            return rec
        info = out.get("info") if isinstance(out.get("info"), dict) else {}
        err = out.get("error") or info.get("error")
        if err:
            if "already done" in str(err).lower():
                self.episode_over = True
                self.end_reason = self.end_reason or "terminated"
            rec["err"] = str(err)
            return rec
        rec["ok"] = True
        if isinstance(info.get("step_count"), (int, float)):
            self.steps_taken = int(info["step_count"])
        else:
            self.steps_taken += 1
        if bool(out.get("terminated")) or bool(out.get("truncated")):
            self.episode_over = True
            self.end_reason = ("stop_called" if a == 0
                               else "step_budget_exhausted"
                               if out.get("truncated") else "terminated")
        if a == 0:
            return rec
        collided = bool(info.get("collided"))
        dy = info.get("actual_dy_m")
        if isinstance(dy, (int, float)) and abs(float(dy)) < 2.0:
            self.pz += float(dy)      # climbed / descended this primitive
        if a == 1:
            delta = info.get("actual_translation_m")
            d = (float(delta) if isinstance(delta, (int, float))
                 else (0.0 if collided else STEP_M))
            self._odom_forward(d)
            rec["d"] = d
            if d < BLOCKED_M:
                rec["blocked"] = True
                if self.slam is not None:
                    try:
                        self.slam.occ.mark_obstacle_ahead(
                            self.slam.pose_wc(self.px, self.py, self.theta,
                                              self.pz))
                    except Exception:  # noqa: BLE001
                        pass
        elif a in (2, 3):
            self._odom_turn(a)
        if self.episode_over:
            return rec
        rec["png"] = self._sense()
        return rec

    # ── the final push: face the detected target and walk it down ────────

    def _final_push(self) -> dict[str, Any] | None:
        """Called when the model issues STOP. If the gated detector sees the
        target in the current view: face the instance nearest to straight
        ahead (the one the model was told to stop in front of), then walk
        forward until close enough / blocked / out of sight / cap. Every
        primitive is a real env step (counted). Returns the ledger, or None
        when there was nothing to push toward."""
        if not self.final_push or self.organ is None or not self.goal:
            return None
        sights = self._detect_now()
        self.detect_stats["push_detects"] += 1
        if not sights:
            return None
        best = min(sights, key=lambda s: (abs(float(s.bearing)),
                                          float(s.distance)))
        d0 = float(best.distance)
        rec: dict[str, Any] = {"target": self.goal, "from_m": round(d0, 2),
                               "to_m": round(d0, 2), "turned": 0, "steps": 0,
                               "moved_m": 0.0, "stopped_because": ""}
        if d0 <= self.push_stop_m:
            rec["stopped_because"] = "already_close"
            return rec
        # face it: dir_deg positive = right = action 3
        dir_deg = -math.degrees(float(best.bearing))
        n_turn = int(round(dir_deg / self.turn_deg))
        n_turn = max(-PUSH_MAX_TURNS, min(PUSH_MAX_TURNS, n_turn))
        for _ in range(abs(n_turn)):
            r = self._exec_primitive(3 if n_turn > 0 else 2)
            if not r["ok"] or self.episode_over:
                rec["stopped_because"] = "env"
                return rec
            rec["turned"] += 1
        if n_turn:
            sights = self._detect_now() or []
            self.detect_stats["push_detects"] += 1
            if not sights:
                rec["stopped_because"] = "lost_sight_after_turn"
                return rec
            rec["to_m"] = round(min(float(s.distance) for s in sights), 2)
        for _ in range(self.push_max_steps):
            r = self._exec_primitive(1)
            if not r["ok"]:
                rec["stopped_because"] = "env"
                break
            rec["steps"] += 1
            rec["moved_m"] = round(rec["moved_m"] + float(r["d"]), 2)
            if self.episode_over:
                rec["stopped_because"] = "episode_over"
                break
            if r["blocked"]:
                rec["stopped_because"] = "blocked"
                break
            sights = self._detect_now() or []
            self.detect_stats["push_detects"] += 1
            if not sights:
                # too close to segment / below the camera / left the frame:
                # as far as the eyes can tell us — good enough
                rec["stopped_because"] = "lost_sight"
                break
            d = min(float(s.distance) for s in sights)
            rec["to_m"] = round(d, 2)
            if d <= self.push_stop_m:
                rec["stopped_because"] = "close_enough"
                break
        else:
            rec["stopped_because"] = "cap"
        return rec

    # ── tools ────────────────────────────────────────────────────────────

    def _tool_step(self, actions: Any = None, **_ignored: Any) -> ToolResult:
        if self.episode_over:
            return self._json_only({"error": f"episode already over "
                                             f"({self.end_reason}); no more "
                                             "steps possible",
                                    **self._status()})
        if not isinstance(actions, list) or not actions:
            return self._json_only({"error": "empty action list"})
        if len(actions) > MAX_ACTIONS_PER_CALL:
            return self._json_only({"error": f"too many actions in one call "
                                             f"(max {MAX_ACTIONS_PER_CALL})"})
        bad = [a for a in actions if a not in (0, 1, 2, 3)]
        if bad:
            return self._json_only({"error": f"invalid actions {bad}; valid: "
                                             "0=STOP 1=FORWARD 2=LEFT 3=RIGHT"})
        executed = 0
        moved = 0.0
        blocked = False
        png: bytes | None = None
        push: dict[str, Any] | None = None
        for a in actions:
            a = int(a)
            if a == 0:
                # the final push rides BEFORE the STOP primitive: face the
                # detected target and walk it down, then stop for real
                push = self._final_push()
                if push is not None:
                    moved += float(push.get("moved_m") or 0.0)
                    if push.get("stopped_because") == "blocked":
                        blocked = True
                    png = self._last_rgb_png    # the view it stopped at
                if self.episode_over:      # the push burnt the last steps
                    break
            r = self._exec_primitive(a)
            if not r["ok"]:
                if r["err"] and r["err"].startswith("env step failed"):
                    return self._json_only({"error": r["err"],
                                            "executed": executed,
                                            "requested": len(actions),
                                            **self._status()})
                break
            executed += 1
            if a == 0:
                break
            if r["blocked"]:
                blocked = True
            moved += float(r["d"])
            if self.episode_over:
                break
            png = r["png"]
            if blocked:
                break
        result: dict[str, Any] = {
            "executed": executed, "requested": len(actions),
            "moved_m": round(moved, 2), "blocked": blocked,
            **self._status(),
        }
        if blocked and executed < len(actions) and push is None:
            result["note"] = ("forward blocked — the remaining "
                              f"{len(actions) - executed} action(s) were cancelled")
        if push is not None:
            result["final_push"] = push
            result["note"] = (
                f"final push before STOP: the detector saw the '{self.goal}' "
                f"{push['from_m']} m ahead; the robot turned {push['turned']} "
                f"time(s) and walked {push['steps']} step(s) "
                f"({push['moved_m']} m) toward it, now about {push['to_m']} m "
                f"({push['stopped_because']}), then stopped.")
        content: list[dict[str, Any]] = []
        if png is None and not self.episode_over:
            png = self._sense()
        if png is not None:
            idx = self._record_frame(png, "view")
            result["frame"] = idx
            content.append(png_part(png))
        content.append(text_part(json.dumps(result)))
        self._publish_map_async()
        return ToolResult(content=content, info={"kind": "step", **result})

    def _tool_get_pose(self, **_ignored: Any) -> ToolResult:
        yaw = (-math.degrees(self.theta) + 180.0) % 360.0 - 180.0
        pose = {"x": round(self.px, 3), "z": round(self.py, 3),
                "yaw_deg": round(yaw, 1),
                "frame": ("x = right, z = forward of your start pose; yaw_deg "
                          "0 at your starting heading, increases turning right")}
        return ToolResult(content=[text_part(json.dumps(pose))],
                          info={"kind": "get_pose", **pose})

    def _tool_get_trajectory(self, **_ignored: Any) -> ToolResult:
        pts = ([[round(float(p[0]), 2), round(float(p[2]), 2)]
                for p in self.slam.track] if self.slam is not None else [])
        n = len(pts)
        if n > 400:   # thin, keep ends
            idx = np.linspace(0, n - 1, 400).round().astype(int)
            pts = [pts[i] for i in idx]
        traj = {"points": pts, "n_total": n,
                "frame": "same as get_pose(): x right, z forward of start"}
        return ToolResult(content=[text_part(json.dumps(traj))],
                          info={"kind": "get_trajectory", "n_total": n})

    def _tool_get_map(self, **_ignored: Any) -> ToolResult:
        if self.slam is None:
            return self._json_only({"error": "no map yet — take a step first"})
        try:
            png, payload = self.slam.render([], self._pose_ns())
        except Exception as exc:  # noqa: BLE001
            return self._json_only({"error": f"map render failed: {exc!r}"})
        idx = self._record_frame(png, "map")
        if self.live_dir is not None:
            try:
                tmp = self.live_dir / "map_latest.png.tmp"
                tmp.write_bytes(png)
                os.replace(tmp, self.live_dir / "map_latest.png")
            except Exception:  # noqa: BLE001
                pass
        payload = {"frame": idx, **payload}
        return ToolResult(content=[png_part(png), text_part(json.dumps(payload))],
                          info={"kind": "get_map", "frame": idx,
                                "frontiers": len(payload.get("frontiers") or []),
                                "landmarks": len(payload.get("landmarks") or [])})

    # ── helpers ─────────────────────────────────────────────────────────

    def _status(self) -> dict[str, Any]:
        return {"steps_taken_total": self.steps_taken,
                "steps_remaining_approx": max(0, self.step_budget - self.steps_taken),
                "episode_over": self.episode_over,
                "end_reason": self.end_reason}

    @staticmethod
    def _json_only(d: dict[str, Any], kind: str = "step") -> ToolResult:
        return ToolResult(content=[text_part(json.dumps(d))],
                          info={"kind": kind, **d})
