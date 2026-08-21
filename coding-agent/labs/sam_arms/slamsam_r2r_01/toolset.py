"""LeanSlamToolSet — the subtracted harness (user 2026-08-18: "我们的
harness 还是太重了 … 只有纯粹的 slam sam 地图 … 过往历史帧留着,1张当前
地图留在磁盘。剩下都不要").

Three tools and nothing else on the model's surface:

    step(actions)   primitives 0/1/2/3; the result carries the photo from
                    where the body now stands (tagged frame#N)
    get_map()       the SLAM occupancy map (jian's integrate rule + his
                    renderer) with the SAM landmark layer painted on it —
                    a PULLED instrument, free
    recall(...)     the frames the model already saw, by number / recent /
                    range — the episode's own video, addressable

What runs underneath, per executed primitive: odometry from the env's own
measured translation (+ commanded yaw), one RGB-D read, OccupancyMap
.integrate at that pose, one keep-latest async SAM job (results stamped at
their CAPTURE pose; a label the body walks up to and does not see decays —
"可覆盖错检测"). That is the whole organ list. There is NO state block, no
progress note, no subgoals, no judge, no guards, no heartbeat, no events
log, no keyframes, no reminders, no loop verdict, no bootstrap sweep, no
candidate menu, no instruction resolver. Everything the previous line grew
stays in depth_toolset.py / eharness_bridge.py, archived by non-use.

Disk (live_dir): the frames the model saw (obs_NNNN_stepSSS.png for views,
obs_NNNN_map.png for the maps it pulled, bootstrap_current.png for frame 0)
and ONE current map (map_latest.png, keep-latest async). Nothing else.

Env contract: ``{verb}__step_discrete`` (info.actual_translation_m,
info.collided, terminated/truncated), ``{verb}__observe_egocentric`` (rgb
PNG b64, depth ndarray, depth_units, intrinsics) — env_habitat (R2R/RxR,
15° turns) and env_ovon (30° turns, 640×480, hfov 79, depth normalised
over [0.5, 5] m) both speak it.
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

# exp_workspace/slamsam_r2r_01 copy (2026-08-18, forked from harnesses/mini/
# lean_toolset.py — see README.md). Execution code lives HERE (this file +
# slam_sidecar.py + slam/ + nodeset/); only library code is imported from
# the shared tree: the mini NodesetToolSet base (harnesses/mini/toolset.py)
# and eharness.depthmap / eharness.landmarks (depth decode, mask projection,
# the SAM client).
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
RECALL_MAX_TILES = 8    # filmstrip cap
RECALL_TILE_W = 320


# ── the three tool descriptions (single source: bridge + tests import these) ──

def step_desc(turn_deg: float, task: str = "vln") -> str:
    goal = ("within 3 meters of the instruction's endpoint" if task == "vln"
            else "within 1 meter of the target object")
    return (
        "Execute a sequence of movement actions, in order. Actions: 0 = STOP "
        f"(permanently ENDS the episode — issue it only when you believe you "
        f"are {goal}), 1 = move forward {STEP_M} m, 2 = turn left "
        f"{turn_deg:g} degrees, 3 = turn right {turn_deg:g} degrees. The "
        "result carries the camera view from where the robot now stands "
        "(tagged frame#N) plus a short JSON: actions executed, metres moved, "
        "whether a forward step was blocked (the rest of that call is then "
        "cancelled), steps used and remaining, episode_over. Looking is free "
        "and automatic — there is no separate look action; a small turn is the "
        "cheapest way to look around."
    )


GET_MAP_DESC = (
    "Read the top-down map your onboard SLAM builds as you move — free, no "
    "step cost. UP on the image is your STARTING heading. White = floor you "
    "have seen, black = obstacles, gray = unexplored; the blue arrow is you "
    "(pointing your current heading), the blue line is your path, the S "
    "circle is your start. Numbered green circles are FRONTIERS — openings "
    "into unexplored space — listed in the JSON as F<n> with direction "
    "relative to your heading (positive = right), distance and size; ids "
    "stay stable across calls. Tinted patches with a name are LANDMARKS a "
    "detector recognised as you moved (also listed with direction and "
    "distance) — estimates, not facts: a patch you walk up to and cannot see "
    "fades away on its own. SLAM estimates can drift slightly."
)

GET_MAP_DESC_V2 = (
    "Read the top-down map your onboard SLAM builds as you move — free, no "
    "step cost. UP on the image is your STARTING heading. White = floor you "
    "have seen, black = obstacles, gray = unexplored; the blue arrow is you "
    "(pointing your current heading), the blue line is your path, the S "
    "circle is your start. The coordinate grid (2 m lines, labeled on both "
    "axes) stays FIXED between calls — the window only grows, never pans — "
    "so a place keeps its position on the map from one look to the next. "
    "There is no frontier list: unexplored space is the gray next to the "
    "white you have covered. Tinted patches with a name are LANDMARKS a "
    "detector recognised as you moved (also listed with direction and "
    "distance) — estimates, not facts: a patch you walk up to and cannot see "
    "fades away on its own. SLAM estimates can drift slightly."
)


def get_map_desc(map_mode: str = "v1") -> str:
    return GET_MAP_DESC_V2 if str(map_mode) == "v2" else GET_MAP_DESC


RECALL_DESC = (
    "Look back at what you already saw — free. kind='frame' with id=N: the "
    "exact stored view tagged frame#N. kind='recent' (+ top_k, default 4): "
    "your last few views as one filmstrip. kind='range' with start=A, end=B: "
    "a filmstrip sampled from frames A..B. Frames are numbered in the order "
    "you saw them (camera views and pulled maps alike); the JSON lists which "
    "frame is which."
)

STEP_SCHEMA = {
    "properties": {"actions": {"items": {"type": "integer"},
                               "title": "Actions", "type": "array"}},
    "required": ["actions"], "title": "stepArguments", "type": "object",
}
GET_MAP_SCHEMA = {"properties": {}, "title": "getMapArguments",
                  "type": "object"}
RECALL_SCHEMA = {
    "properties": {
        "kind": {"type": "string", "enum": ["frame", "recent", "range"]},
        "id": {"type": "integer"},
        "top_k": {"type": "integer"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
    },
    "required": ["kind"], "title": "recallArguments", "type": "object",
}


class LeanSlamToolSet(NodesetToolSet):
    """step + get_map + recall over any env that speaks the two habitat verbs."""

    def __init__(
        self,
        server_url: str,
        *,
        verb: str = "env_habitat",
        phrases: list[str] | None = None,
        step_budget: int = 500,
        live_dir: Path | None = None,
        sam_url: str = "",
        turn_deg: float = 15.0,
        task: str = "vln",
        sam_synonyms: int = 0,
        cam_height_fallback_m: float = 1.25,
        map_mode: str = "v1",
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
        self.cam_h_fallback = float(cam_height_fallback_m)
        # "v1" = jian's frontier map (the smoked口径); "v2" = his slam_r2r_02
        # map (no frontier, 2 m-snapped grow-only window) — merged 2026-08-18
        self.map_mode = str(map_mode or "v1")

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
        self.organ: Any = None           # LandmarkOrgan, if sam_url + phrases
        self._sam_url = sam_url
        self._sam_pool: ThreadPoolExecutor | None = None
        self._sam_jobs: list[dict] = []
        self.sam_stats = {"calls": 0, "superseded": 0, "errors": 0}
        # the episode's own video: every frame the model saw, in order
        self.frames: list[dict] = []     # {idx, step, kind, path|png}
        self._obs_count = 0
        self._last_rgb_png: bytes | None = None
        self._map_pool: ThreadPoolExecutor | None = None
        self._map_future: Future | None = None
        self._t0 = time.time()

        if sam_url and self.phrases:
            from eharness.landmarks import LandmarkOrgan
            self.organ = LandmarkOrgan(sam_url, hfov_deg=self.hfov_deg)

        self._register("step", step_desc(self.turn_deg, self.task),
                       STEP_SCHEMA, self._tool_step)
        self._register("get_map", get_map_desc(self.map_mode), GET_MAP_SCHEMA,
                       self._tool_get_map)
        self._register("recall", RECALL_DESC, RECALL_SCHEMA, self._tool_recall)

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

    # ── sensing: one RGB-D read → SLAM integrate + SAM submit ────────────

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
                # nearer than min_depth reads as exactly min_depth (RxR-CE /
                # OVON: 0.5 m). Left in, every near surface — the wall you
                # hug, the table edge, the door you face — is stamped as a
                # phantom obstacle shell 0.5 m out, and the trail runs
                # through black (RxR EP0, 2026-08-18). A clipped pixel is
                # NOT a measurement: mark it invalid (0 → below the
                # mapper's depth_min, ignored by SAM projection too).
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
        self.slam = SlamMapSidecar(sam_phrases=list(self.phrases),
                                   cam_height_m=cam_h, rgb_hw=(h, w),
                                   hfov_deg=self.hfov_deg,
                                   map_mode=self.map_mode)
        if self.organ is not None:
            self.organ.hfov_deg = self.hfov_deg

    def _sense(self) -> bytes | None:
        """Read the current frame; integrate; submit SAM. Returns the PNG."""
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
            if self.organ is not None and isinstance(rgb_b64, str):
                self._sam_submit(rgb_b64, depth_m,
                                 (self.px, self.py, self.theta))
        self._sam_drain(wait=False)
        if png is not None:
            self._last_rgb_png = png
        return png

    # ── SAM: one keep-latest worker, stamps at capture pose, decay on miss ──

    def _sam_submit(self, rgb_b64: str, depth_m: np.ndarray,
                    pose: tuple[float, float, float]) -> None:
        if self._sam_pool is None:
            self._sam_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="lean-sam")
        kept = []
        for job in self._sam_jobs:
            if not job["future"].done() and job["future"].cancel():
                self.sam_stats["superseded"] += 1
            else:
                kept.append(job)
        self._sam_jobs = kept
        organ, phrases, hfov, syn = (self.organ, list(self.phrases),
                                     self.hfov_deg, self.sam_synonyms)

        def _work() -> tuple[list, list]:
            sights = organ.sightings(rgb_b64, depth_m, phrases, scale_m=1.0,
                                     synonyms=syn)
            projs = []
            for s in sights:
                if s.mask is None:
                    projs.append(None)
                    continue
                projs.append(dm.project_mask(s.mask, depth_m, None,
                                             hfov_deg=hfov, scale_m=1.0))
            return sights, projs

        self._sam_jobs.append({"future": self._sam_pool.submit(_work),
                               "pose": pose, "t0": time.time()})

    def _sam_drain(self, *, wait: bool) -> None:
        if not self._sam_jobs:
            return
        if wait:
            last = self._sam_jobs[-1]
            if not last["future"].done():
                budget = float(getattr(self.organ, "timeout", 60.0) or 60.0) + 30.0
                try:
                    last["future"].result(timeout=budget)
                except Exception:  # noqa: BLE001 — recorded by the merge below
                    pass
        remaining: list[dict] = []
        for job in self._sam_jobs:
            fut = job["future"]
            if not fut.done():
                remaining.append(job)
                continue
            try:
                sights, projs = fut.result()
            except Exception:  # noqa: BLE001 — semantics stay advisory
                self.sam_stats["errors"] += 1
                continue
            self.sam_stats["calls"] += 1
            self._absorb(sights, projs, job["pose"])
        self._sam_jobs = remaining

    def _absorb(self, sights: list, projs: list,
                pose: tuple[float, float, float]) -> None:
        if self.slam is None:
            return
        px, py, th = (float(v) for v in pose)
        for s, proj in zip(sights, projs):
            if proj is None:
                continue
            xs, ys = proj
            if getattr(xs, "size", 0):
                self.slam.stamp_points(s.phrase, xs, ys, px, py, th,
                                       float(s.score))
        seen = {s.phrase for s in sights}
        missed = [p for p in self.phrases if p not in seen]
        if missed:
            self.slam.decay_unseen_nearby(missed, px, py)

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

    def _frame_png(self, rec: dict) -> bytes | None:
        if "png" in rec:
            return rec["png"]
        if self.live_dir is not None and rec.get("path"):
            p = self.live_dir / rec["path"]
            if p.exists():
                return p.read_bytes()
        return None

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
        for a in actions:
            a = int(a)
            try:
                out = self._call(f"{self.verb}__step_discrete", {"action": a})
            except Exception as exc:  # noqa: BLE001
                return self._json_only({"error": f"env step failed: {exc!r}",
                                        "executed": executed,
                                        "requested": len(actions),
                                        **self._status()})
            info = out.get("info") if isinstance(out.get("info"), dict) else {}
            err = out.get("error") or info.get("error")
            if err:
                if "already done" in str(err).lower():
                    self.episode_over = True
                    self.end_reason = self.end_reason or "terminated"
                break
            executed += 1
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
                break
            collided = bool(info.get("collided"))
            dy = info.get("actual_dy_m")
            if isinstance(dy, (int, float)) and abs(float(dy)) < 2.0:
                self.pz += float(dy)      # climbed / descended this primitive
            if a == 1:
                delta = info.get("actual_translation_m")
                d = (float(delta) if isinstance(delta, (int, float))
                     else (0.0 if collided else STEP_M))
                self._odom_forward(d)
                moved += d
                if d < BLOCKED_M:
                    blocked = True
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
                break
            png = self._sense()
            if blocked:
                break
        result: dict[str, Any] = {
            "executed": executed, "requested": len(actions),
            "moved_m": round(moved, 2), "blocked": blocked,
            **self._status(),
        }
        if blocked and executed < len(actions):
            result["note"] = ("forward blocked — the remaining "
                              f"{len(actions) - executed} action(s) were cancelled")
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

    def _tool_get_map(self, **_ignored: Any) -> ToolResult:
        if self.slam is None:
            return self._json_only({"error": "no map yet — take a step first"})
        # the map is read on demand, so what the detector has finished for
        # the frame you are standing in must be on it before you look
        self._sam_drain(wait=True)
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

    def _tool_recall(self, kind: Any = None, id: Any = None, top_k: Any = None,
                     start: Any = None, end: Any = None,
                     **_ignored: Any) -> ToolResult:
        kind = str(kind or "").strip().lower()
        n = len(self.frames)
        if n == 0:
            return self._json_only({"error": "nothing stored yet"})
        if kind == "frame":
            try:
                i = int(id)
            except (TypeError, ValueError):
                return self._json_only({"error": "kind='frame' needs an integer id",
                                        "valid_range": [0, n - 1]})
            if not 0 <= i < n:
                return self._json_only({"error": "not_found",
                                        "valid_range": [0, n - 1]})
            rec = self.frames[i]
            png = self._frame_png(rec)
            if png is None:
                return self._json_only({"error": "frame file missing", "frame": i})
            meta = {"frame": i, "kind": rec["kind"], "step": rec["step"]}
            return ToolResult(content=[png_part(png), text_part(json.dumps(meta))],
                              info={"kind": "recall", "recall_kind": "frame",
                                    "frame": i})
        if kind == "recent":
            try:
                k = int(top_k) if top_k is not None else 4
            except (TypeError, ValueError):
                k = 4
            k = max(1, min(k, RECALL_MAX_TILES))
            views = [r for r in self.frames if r["kind"] == "view"][-k:]
            return self._sheet(views, "recent")
        if kind == "range":
            try:
                a = int(start) if start is not None else 0
                b = int(end) if end is not None else n - 1
            except (TypeError, ValueError):
                return self._json_only({"error": "kind='range' needs integer "
                                                 "start/end",
                                        "valid_range": [0, n - 1]})
            a, b = max(0, min(a, b)), min(n - 1, max(a, b))
            recs = [r for r in self.frames[a:b + 1] if r["kind"] == "view"]
            if len(recs) > RECALL_MAX_TILES:
                pick = np.linspace(0, len(recs) - 1, RECALL_MAX_TILES).round()
                recs = [recs[int(i)] for i in pick]
            return self._sheet(recs, "range")
        return self._json_only({"error": "kind must be one of frame / recent / "
                                         "range"})

    # ── helpers ─────────────────────────────────────────────────────────

    def _status(self) -> dict[str, Any]:
        return {"steps_taken_total": self.steps_taken,
                "steps_remaining_approx": max(0, self.step_budget - self.steps_taken),
                "episode_over": self.episode_over,
                "end_reason": self.end_reason}

    @staticmethod
    def _json_only(d: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[text_part(json.dumps(d))],
                          info={"kind": "step", **d})

    def _sheet(self, recs: list[dict], label: str) -> ToolResult:
        from PIL import Image, ImageDraw
        tiles: list[tuple[dict, Image.Image]] = []
        for r in recs:
            png = self._frame_png(r)
            if png is None:
                continue
            try:
                im = Image.open(io.BytesIO(png)).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            w, h = im.size
            nh = max(1, int(round(h * RECALL_TILE_W / max(1, w))))
            tiles.append((r, im.resize((RECALL_TILE_W, nh))))
        if not tiles:
            return self._json_only({"error": "no stored views in that range"})
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        th = max(t.size[1] for _, t in tiles) + 18
        sheet = Image.new("RGB", (cols * RECALL_TILE_W, rows * th), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (r, t) in enumerate(tiles):
            x, y = (i % cols) * RECALL_TILE_W, (i // cols) * th
            sheet.paste(t, (x, y + 18))
            draw.text((x + 4, y + 2), f"frame#{r['idx']} · step {r['step']}",
                      fill=(255, 255, 255))
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        listing = [{"frame": r["idx"], "step": r["step"]} for r, _ in tiles]
        return ToolResult(content=[png_part(buf.getvalue()),
                                   text_part(json.dumps({"kind": label,
                                                         "frames": listing}))],
                          info={"kind": "recall", "recall_kind": label,
                                "frames": [r["idx"] for r, _ in tiles]})
