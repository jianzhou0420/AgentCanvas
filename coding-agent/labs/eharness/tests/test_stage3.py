"""Stage 3 (subtractive-harness plan, 2026-08-16) offline regressions.

Run:  python -m eharness.tests.test_stage3.py    (or pytest)

Covers the new P0 contracts:
  P0-1  the model text is ONE compact facts block — WALKABLE PLACES /
        AHEAD / LANDMARK UPDATE (change-gated) / MAP WARNING (exception-
        gated); the image-payload side lives in test_toolset_contract.
  P0-2  per-move expectation judge OFF by default; arrival bell keeps its
        own switch; the every-6-moves advisory is out of the hot path.
  P0-3  recall is explicit — a numeric id is a FRAME (the old fuzzy path
        read it as a subgoal), not_found answers with the valid range,
        and detector sightings are indexed as landmark events.
  P0-4  remembered candidates: max 3, at most ONE per direction cluster
        (left flank / behind / right flank), no fillers.
"""
from __future__ import annotations

import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# direct-run parity with conftest.py: the mini executor's modules
_CA = Path(__file__).resolve().parents[2]
for _p in (str(_CA), str(_CA / "harnesses" / "mini")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eharness import depthmap as dm
from eharness.depthmap import AnchorMap, Waypoint, propose_from_memory
from eharness.frames import FrameLog
from eharness.state_block import StateBlock
from eharness.wrapper import HarnessedToolset

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (16, 16), (40, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


# ── P0-4 · remembered candidates: 3 max, one per direction cluster ───────
print("── P0-4 · memory candidates cluster by direction ──────────────────")
am = AnchorMap()
am.updates = 5                      # < 6 → trusted by the early-run rule


def _paint_free(bearing_deg: float, reach_m: float,
                half_deg: float = 12.0) -> None:
    for b in np.arange(bearing_deg - half_deg, bearing_deg + half_deg, 1.0):
        rad = math.radians(float(b))
        for r in np.arange(0.15, reach_m, am.cell / 2):
            i, j = am.cells(r * math.sin(rad), r * math.cos(rad))
            if 0 <= i < am.n and 0 <= j < am.n:
                am.logodds[i, j] = -2.0


# two LEFT-flank corridors (70°, 110°), one RIGHT (-80°), one BEHIND (175°)
_paint_free(70.0, 3.0)
_paint_free(110.0, 4.0)
_paint_free(-80.0, 3.5)
_paint_free(175.0, 3.0)
mem = propose_from_memory(am, [])
degs = sorted(round(math.degrees(w.angle)) for w in mem)


def _cluster(deg: float) -> str:
    return "behind" if abs(deg) >= 130 else ("left" if deg > 0 else "right")


clusters = [_cluster(math.degrees(w.angle)) for w in mem]
check("at most 3 remembered candidates", len(mem) <= 3, f"{len(mem)}: {degs}")
check("at most ONE candidate per direction cluster",
      len(clusters) == len(set(clusters)), f"{clusters}")
check("all three clusters are covered (two left corridors merged to one)",
      set(clusters) == {"left", "behind", "right"}, f"{degs}")
check("every memory candidate says so", all(w.kind == "remembered" for w in mem))

# ── P0-2 · judge frequency defaults ──────────────────────────────────────
print()
print("── P0-2 · per-move judges are off by default ──────────────────────")


class _Inner:
    def __init__(self) -> None:
        self.calls_by_tool: dict = {}
        self.steps_taken = 0
        self.end_reason = None
        self.episode_over = False

    def tool_schemas(self) -> list:
        return [{"name": "step", "description": "d",
                 "input_schema": {"type": "object", "properties": {}}}]

    def execute(self, name, args):  # noqa: ANN001
        raise AssertionError("recall tests never reach the inner toolset")


w = HarnessedToolset(_Inner(), state=StateBlock(), live_dir=None,
                     judge_model=None)
check("expectation judge (verify_moves) defaults OFF", w.verify_moves is False)
check("the arrival bell keeps its own switch, default ON",
      w.arrival_bell is True)
check("periodic reflect defaults off", w.reflect_every == 0)
_src = open("eharness/wrapper.py").read()
check("the every-6-moves advisory is out of the hot path",
      "segment check: you are on segment" not in _src)
check("judge thinking defaults off at every judge entry",
      open("eharness/judge.py").read().count('kw.pop("judge_think", False)') == 4)

# ── user 2026-08-18 · periodic full-instruction reminder ─────────────────
print()
print("── instruction reminder every N moves (attention insurance) ────────")
import json  # noqa: E402
from toolset import ToolResult as _TR, text_part as _tp  # noqa: E402


class _Stepper(_Inner):
    def execute(self, name, args):  # noqa: ANN001
        self.steps_taken += 1
        return _TR(content=[_tp(json.dumps({"steps_taken_total": self.steps_taken,
                                            "executed": 1}))],
                   info={"kind": name})


_st = StateBlock()
_st.instruction = "Walk past the pool. Stop at the far corner of the bar."
wr = HarnessedToolset(_Stepper(), state=_st, live_dir=None,
                      judge_model=None, guards_on=False, remind_every=3)


def _has_reminder(res) -> bool:
    return any("REMINDER — your full instruction" in str(p.get("text", ""))
               and _st.instruction in str(p.get("text", ""))
               for p in res.content if isinstance(p, dict))


_hits = [_has_reminder(wr.execute("step", {"actions": [1]})) for _ in range(7)]
check("reminder fires on the 3rd and 6th move only (every N, verbatim text)",
      _hits == [False, False, True, False, False, True, False], str(_hits))
wr0 = HarnessedToolset(_Stepper(), state=_st, live_dir=None,
                       judge_model=None, guards_on=False, remind_every=0)
check("remind_every=0 disables it",
      not any(_has_reminder(wr0.execute("step", {"actions": [1]}))
              for _ in range(6)))
check("bridge default is 6 (EH_REMIND_EVERY)",
      'REMIND_EVERY = int(os.environ.get("EH_REMIND_EVERY", "6")' in
      open("bridges/eharness_bridge.py").read())

# ── P0-3 · recall is explicit ────────────────────────────────────────────
print()
print("── P0-3 · recall(kind=…) resolves exactly ─────────────────────────")
for k in range(3):
    w.frames.record(_png(), step=k, tool="observe")
r = w.execute("recall", {"kind": "frame", "id": 1})
txt = " ".join(p.get("text", "") for p in r.content
               if isinstance(p, dict) and p.get("type") == "text")
imgs = sum(1 for p in r.content
           if isinstance(p, dict) and p.get("type") == "image_url")
check("kind='frame' id=1 returns exactly that frame + pixels",
      "frame#1" in txt and imgs == 1, txt[:60])
r = w.execute("recall", {"query": "2"})
check("legacy bare number now means FRAME, never subgoal",
      r.info.get("recall_kind") == "frame"
      and "frame#2" in " ".join(p.get("text", "") for p in r.content
                                if p.get("type") == "text"))
r = w.execute("recall", {"kind": "frame", "id": 99})
txt = r.content[0]["text"]
check("unknown frame id answers not_found + the valid range",
      "not_found" in txt and "0..2" in txt, txt)
r = w.execute("recall", {"kind": "segment", "id": 5})
check("unknown segment answers not_found + what exists",
      "not_found" in r.content[0]["text"], r.content[0]["text"])

w.frames.note_landmark("bar counter", frame_idx=1, bearing_deg=18.0,
                       distance_m=2.1, score=0.7)
r = w.execute("recall", {"kind": "landmark", "query": "Bars"})
txt = " ".join(p.get("text", "") for p in r.content
               if p.get("type") == "text")
imgs = sum(1 for p in r.content
           if isinstance(p, dict) and p.get("type") == "image_url")
check("normalised landmark query hits the recorded sighting",
      "bar counter" in txt and "frame#1" in txt and imgs == 1, txt[:80])
r = w.execute("recall", {"kind": "landmark", "query": "sofa"})
txt = r.content[-1]["text"]
check("a landmark never seen answers not_found + what WAS recorded",
      "not_found" in txt and "bar counter" in txt, txt[:90])
r = w.execute("recall", {"kind": "recent", "top_k": 2})
check("kind='recent' returns the newest stored views",
      any("frame#2" in str(p.get("text", "")) for p in r.content
          if isinstance(p, dict)), )

# sightings on a tool result index automatically, promote only on ENTER
print()
print("── P0-3 · SAM sightings become indexed landmark events ────────────")
from toolset import ToolResult  # noqa: E402  (mini path via conftest)


def _obs_result() -> ToolResult:
    import base64
    return ToolResult(
        content=[{"type": "image_url", "image_url": {
            "url": "data:image/png;base64,"
                   + base64.b64encode(_png()).decode()}}],
        info={"sightings": [{"phrase": "pool", "bearing_deg": -30.0,
                             "distance_m": 3.2, "score": 0.8}]})


n_ev0 = len(w.frames.landmark_events)
res = _obs_result()
w._post_observation("observe", res)
first_idx = w.frames.frames[-1].idx
res2 = _obs_result()
w._post_observation("observe", res2)
check("each sighting writes one landmark event",
      len(w.frames.landmark_events) == n_ev0 + 2,
      f"{n_ev0} → {len(w.frames.landmark_events)}")
check("landmark_hits finds the newest sighting frame",
      w.frames.landmark_hits("pool")[0]["frame"] == w.frames.frames[-1].idx)
tagged = [f.idx for f in w.frames.frames
          if any(e.startswith("landmark:pool") for e in f.events)]
check("only the ENTER frame is keyframe-tagged (no per-frame flood)",
      tagged == [first_idx], f"tagged {tagged}")

# ── P0-1 · the facts block ───────────────────────────────────────────────
print()
print("── P0-1 · facts block: change-gated, exception-gated ──────────────")
from depth_toolset import DepthWaypointToolSet  # noqa: E402


class _Shim:
    _staging_of: dict = {}
    _last_landmark_line = ""
    _side_words = staticmethod(DepthWaypointToolSet._side_words)
    _facts_text = DepthWaypointToolSet._facts_text


shim = _Shim()
wp_mem = Waypoint(angle=math.radians(72.0), distance=3.8, clearance=1.0,
                  kind="remembered", x_left=3.6, y_fwd=1.2, stride_m=3.0,
                  verified_m=3.8, continuation_m=0.0)
wp_vis = Waypoint(angle=math.radians(-8.0), distance=2.4, clearance=1.0,
                  kind="opening", x_left=-0.3, y_fwd=2.4, stride_m=2.0,
                  verified_m=2.4, continuation_m=1.1)
telem = {"ahead": {"verified_free_m": 2.3, "clear_sightline_m": 5.1},
         "landmarks": [{"name": "bar", "bearing_deg": 18.0,
                        "distance_m": 2.1, "status": "visible"}],
         "map": {"trusted": True, "match": 0.9, "looks": 5}}
txt1 = shim._facts_text([wp_vis, wp_mem], telem, [True, False], "")
check("place lines carry view status, bearing, stride, continuation",
      "1. visible" in txt1 and "clear another 1.1 m" in txt1
      and "2. remembered / off-camera" in txt1 and "72° left" in txt1
      and "re-checks with fresh eyes" in txt1, txt1.replace("\n", " | ")[:160])
check("no internal ids or versions leak into the model text",
      "track_id" not in txt1 and "map_version" not in txt1
      and "candidate_epoch" not in txt1)
check("LANDMARK UPDATE appears when the line is new",
      "LANDMARK UPDATE" in txt1)
txt2 = shim._facts_text([wp_vis, wp_mem], telem, [True, False], "")
check("…and stays SILENT while nothing changed",
      "LANDMARK UPDATE" not in txt2)
telem2 = {**telem, "landmarks": [{"name": "bar", "bearing_deg": -40.0,
                                  "distance_m": 1.0, "status": "visible"}]}
txt3 = shim._facts_text([wp_vis], telem2, [True], "")
check("…and returns the moment the landmark line changes",
      "LANDMARK UPDATE" in txt3)
check("a healthy trusted map raises no MAP WARNING",
      "MAP WARNING" not in txt1)
telem3 = {**telem, "map": {"trusted": False, "match": 0.2, "looks": 9}}
txt4 = shim._facts_text([wp_vis], telem3, [True], "")
check("an untrusted registration raises MAP WARNING",
      "MAP WARNING" in txt4 and "untrusted" in txt4)

# ── P1-1 · async SAM: submit early, join at need, stamp at capture pose ──
print()
print("── P1-1 · async semantic branch ───────────────────────────────────")
import base64  # noqa: E402
import threading  # noqa: E402
import time as _time  # noqa: E402

from eharness.landmarks import Sighting  # noqa: E402
from depth_toolset import DepthWaypointToolSet  # noqa: E402

_H = _W = 256
_CAM_H = 1.25


def _scene() -> np.ndarray:
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -_CAM_H / dy[below])
    x, y = dx * 3.0, dy * 3.0            # upright panel at z=3 m, right side
    band = (y > -_CAM_H + 0.3) & (y < -_CAM_H + 1.6) & (x > 0.5) & (x < 1.5)
    depth[band] = np.minimum(depth[band], 3.0)
    return depth


def _wire(depth: np.ndarray) -> dict:
    return {"__ndarray__": base64.b64encode(
        depth.astype(np.float32).tobytes()).decode(),
        "dtype": "float32", "shape": list(depth.shape)}


def _rgb64() -> str:
    buf = BytesIO()
    Image.new("RGB", (_W, _H), (90, 90, 90)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeEnv(DepthWaypointToolSet):
    def __init__(self, depth: np.ndarray, **kw):
        self.scene_depth = depth
        self._step_count = 0
        kw.setdefault("traj_archive", False)
        kw.setdefault("bootstrap_sweep", False)
        super().__init__("http://fake", **kw)

    def _call(self, fn: str, inputs: dict) -> dict:
        if fn.endswith("observe_egocentric"):
            return {"rgb": _rgb64(), "depth": _wire(self.scene_depth),
                    "depth_units": {"known": True, "scale_m": 1.0,
                                    "normalized": False}}
        if fn.endswith("step_hightolow"):
            self._step_count += int(float(inputs["distance"]) / 0.25)
            return {"info": {"step_count": self._step_count},
                    "terminated": False, "truncated": False}
        if fn.endswith("step_discrete"):
            a = int(inputs["action"])
            self._step_count += 1
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": 0.25 if a == 1 else 0.0,
                             "collided": False},
                    "terminated": a == 0, "truncated": False}
        raise AssertionError(f"unexpected env call {fn}")

    def _call2(self, fn: str, inputs: dict, config=None) -> dict:
        raise AssertionError(f"unexpected env call {fn}")


class _SlowSAM:
    """One detection with a mask, after a real delay — from any thread."""

    available = True
    last_error = ""
    timeout = 5.0

    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.calls = 0
        self.lock = threading.Lock()

    def sightings(self, rgb, depth, phrases, *, scale_m=None, synonyms=3):
        with self.lock:
            self.calls += 1
        _time.sleep(self.delay_s)
        mask = np.zeros((_H, _W), bool)
        mask[115:165, 150:190] = True
        return [Sighting(phrase=phrases[0], bearing=math.radians(-15.0),
                         distance=3.0, angular_width=0.3, score=0.9,
                         pixels=int(mask.sum()), mask=mask)]


def _drive(async_on: bool, delay_s: float) -> "_FakeEnv":
    ts = _FakeEnv(_scene(), instruction="find the bar counter", sam_url="",
                  sam_async=async_on, sam_intermediate_every=1,
                  landmark_every=1)
    ts.landmarks = _SlowSAM(delay_s)
    ts.phrases = ["bar counter"]
    ts._tool_observe()                    # frame 0 (observe path stays sync)
    ts._tool_step(actions=[1, 1])         # walk: primitives submit/drain
    return ts


def _sem_centroid(ts: "_FakeEnv") -> tuple[float, float] | None:
    g = ts.amap.sem.get("bar counter")
    if g is None or not (g > 0).any():
        return None
    ii, jj = np.nonzero(g > 0)
    w = g[ii, jj]
    return (float(np.average(ii, weights=w)), float(np.average(jj, weights=w)))


sync_ts = _drive(False, 0.0)
t0 = _time.time()
async_ts = _drive(True, 0.30)
stats = async_ts._sam_stats
check("async run produced current sightings at the endpoint (join rule)",
      len(async_ts._sightings) == 1)
check("the async machinery was actually exercised (late merge or supersede)",
      stats.get("merged_late", 0) > 0
      or async_ts._sam_stats["skips"].get("superseded", 0) > 0
      or stats.get("waited_ms", 0) > 0,
      f"merged_late={stats.get('merged_late', 0)} "
      f"superseded={stats['skips'].get('superseded', 0)} "
      f"waited_ms={stats.get('waited_ms', 0)}")
check("no job left in flight after the endpoint join",
      not async_ts._sam_jobs)
c_sync, c_async = _sem_centroid(sync_ts), _sem_centroid(async_ts)
check("semantic layer stamped in both modes",
      c_sync is not None and c_async is not None)
if c_sync and c_async:
    cell = async_ts.amap.cell
    dist_m = math.hypot(c_sync[0] - c_async[0],
                        c_sync[1] - c_async[1]) * cell
    check("late merges stamp at the CAPTURE pose (≈ sync map, not smeared "
          "down the walk)", dist_m < 0.35, f"centroid gap {dist_m:.2f} m")
sync_off = _FakeEnv(_scene(), instruction="x", sam_url="")
check("knob off → no pool, no jobs (the serial path untouched)",
      sync_off._sam_pool is None and not sync_off.sam_async)

# ── P1-3 · near-angle merge: two flanks of one far obstacle = ONE road ──
print()
print("── P1-3 · candidates closer than 25° merge into one ───────────────")


def _panel_scene(x_lo: float, x_hi: float, z: float,
                 wall_at: float) -> np.ndarray:
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -_CAM_H / dy[below])
    depth = np.minimum(depth, wall_at)
    x, y = dx * z, dy * z
    band = (y > -_CAM_H + 0.3) & (y < -_CAM_H + 1.6) & (x > x_lo) & (x < x_hi)
    depth[band] = np.minimum(depth[band], z)
    return depth


# a NARROW pillar far away: its two flanks sit well under 25° apart
td_far = dm.build_topdown(_panel_scene(-0.2, 0.2, 4.5, 6.0), scale_m=1.0)
far_cands = dm.propose(td_far)
far_degs = sorted(round(math.degrees(w.angle), 1) for w in far_cands)
merged = any(w.extras.get("merged_bearings_deg") for w in far_cands)
min_gap = min((abs(a - b) for a in far_degs for b in far_degs if a != b),
              default=999.0)
check("no two published candidates share one road (<25° apart)",
      min_gap >= dm.MIN_CANDIDATE_SEPARATION_DEG - 1e-6 or len(far_degs) <= 1,
      f"bearings {far_degs}, merged extras: {merged}")
# …while a NEAR obstacle's flanks (far apart) still yield one per side
td_near = dm.build_topdown(_panel_scene(-0.5, 0.5, 1.8, 6.0), scale_m=1.0)
near_degs = sorted(round(math.degrees(w.angle)) for w in dm.propose(td_near))
check("a near obstacle keeps BOTH side channels as separate candidates",
      any(d < -8 for d in near_degs) and any(d > 8 for d in near_degs),
      f"bearings {near_degs}")

# ── floor anchoring (2026-08-16, the kitchen-island bug) ────────────────
print()
print("── floor anchor · declared camera height beats the counter top ────")
_CAM = 1.25


def _counter_scene() -> np.ndarray:
    """A wide counter top (0.5 m high) fills the lower view out to x=+1.5;
    the true floor survives only on the right fringe. The self-estimator's
    dense low mode is the COUNTER (nearer → far more pixels); the declared
    height is the truth. Mirrors the live kitchen-island frame that
    reported floor 0.77 m under a 1.5 m camera."""
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    # true floor plane, 1.25 m below the camera
    depth[below] = np.minimum(depth[below], -_CAM / dy[below])
    # counter top: a horizontal plane 0.75 m below the camera
    with np.errstate(divide="ignore", invalid="ignore"):
        z_c = np.where(below, -0.75 / np.minimum(dy, -1e-6), np.inf)
    on_counter = below & np.isfinite(z_c) & (dx * z_c < 1.5) & (z_c < 5.0)
    depth[on_counter] = np.minimum(depth[on_counter], z_c[on_counter])
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


_cs = _counter_scene()
td_est = dm.build_topdown(_cs, scale_m=1.0)                 # self-estimated
td_anc = dm.build_topdown(_cs, scale_m=1.0, cam_height_m=_CAM)
check("the self-estimator IS hijacked by the counter (the failure mode)",
      td_est.floor_y > -1.0, f"estimated floor_y {td_est.floor_y:.2f}")
check("the declared height pins the window; the counter cannot enter it",
      abs(td_anc.floor_y + _CAM) < 0.05, f"floor_y {td_anc.floor_y:.2f}")
# THE danger the anchor kills: with the floor "at" counter height, the
# counter TOP satisfies the floor band and is laundered into FREE — the
# proposer would happily land a waypoint ON the counter. Anchored, the
# same cells are OBSTACLE.
_mid = td_est.grid.shape[1] // 2
_col_est = td_est.grid[:, _mid + 4]
_col_anc = td_anc.grid[:, _mid + 4]
check("hijacked floor launders the counter top into FREE (why it matters)",
      int((_col_est == dm.FREE).sum()) >= 15,
      f"{int((_col_est == dm.FREE).sum())} FREE cells ON the counter")
check("anchored floor reads the same cells as OBSTACLE",
      int((_col_anc == dm.OCCUPIED).sum()) >= 15
      and int((_col_anc == dm.FREE).sum())
      <= int(td_anc.floor_blind_m / dm.CELL_M) + 2,
      f"occupied {int((_col_anc == dm.OCCUPIED).sum())}, "
      f"free {int((_col_anc == dm.FREE).sum())}")

# the OTHER live failure (same day): habitat's navmesh floats the agent
# base above the visual floor — declared 1.25 m, PHYSICAL 1.41 m — so a
# hard floor := -declared misses the true plane by 0.16 m and an open
# living room read free 1%. The window refinement must absorb it.
_TRUE_H = 1.41


def _flat_scene(h: float) -> np.ndarray:
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    _, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                        np.arange(_W, dtype=np.float32))
    dy = -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -h / dy[below])
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


td_off = dm.build_topdown(_flat_scene(_TRUE_H), scale_m=1.0,
                          cam_height_m=_CAM)      # declared 1.25, true 1.41
check("navmesh base offset is absorbed (floor found at the TRUE plane)",
      abs(td_off.floor_y + _TRUE_H) < 0.05,
      f"declared {-_CAM} → refined {td_off.floor_y:.2f}")
_mid_o = td_off.grid.shape[1] // 2
_free_run = 0
for _k in range(td_off.grid.shape[0]):
    if td_off.grid[_k, _mid_o] == dm.FREE \
            or (_k * dm.CELL_M) <= td_off.floor_blind_m:
        _free_run = _k + 1
    else:
        break
check("…and the open floor reads FREE again",
      _free_run * dm.CELL_M >= 3.0, f"central prefix {_free_run * dm.CELL_M:.1f} m")

# ── DWP quality round (user 2026-08-16 evening) ─────────────────────────
print()
print("── DWP quality · bold fuse, fork coverage, landing band ───────────")


def _fork_scene() -> np.ndarray:
    """Two pillars at 1.6 m → three ways out (left / centre / right)."""
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -_CAM / dy[below])
    depth = np.minimum(depth, 7.0)
    for x_lo, x_hi, z in ((-1.2, -0.45, 1.6), (0.45, 1.2, 1.6)):
        x, y = dx * z, dy * z
        band = ((y > -_CAM + 0.1) & (y < -_CAM + 1.7)
                & (x > x_lo) & (x < x_hi))
        depth[band] = np.minimum(depth[band], z)
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


td_fork = dm.build_topdown(_fork_scene(), scale_m=1.0, cam_height_m=_CAM)
forks = dm.propose(td_fork)
fdegs = sorted(round(math.degrees(w.angle)) for w in forks)
check("a three-way fork yields THREE candidates (narrow gateways included)",
      len(forks) == 3 and any(d < -20 for d in fdegs)
      and any(abs(d) < 15 for d in fdegs) and any(d > 20 for d in fdegs),
      f"{fdegs}")
check("every fork candidate sits comfortably off the red",
      all(w.clearance >= 0.5 for w in forks),
      f"clearances {[round(w.clearance, 2) for w in forks]}")

# landing band: reach ~2.8 m used to land at 0.5·reach = 1.4 m — underfoot,
# below the photo's bottom edge; the vis floor pushes it past the blind ring
td_short = dm.build_topdown(_flat_scene(_CAM) * 0 + np.minimum(
    _flat_scene(_CAM), 3.0), scale_m=1.0, cam_height_m=_CAM)
cand_s = dm.propose(td_short)
# the visible band is FORWARD-depth based (the photo's bottom edge cuts
# on y_fwd): every landing's y_fwd must clear ~1.12× the blind ring
_vis_y = td_short.floor_blind_m * 1.12
check("landing rises past the floor-blind ring when the reach affords it",
      bool(cand_s) and all(w.y_fwd >= _vis_y - 0.02 for w in cand_s),
      f"y_fwd {[round(w.y_fwd, 2) for w in cand_s]} "
      f"(need ≥ {_vis_y:.2f})")

# bold fusion: ONE look paints the sightline corridor green on the
# accumulated map; verified consumers still see almost nothing; one
# obstacle return overturns the paint
am_bold = AnchorMap()
td_corr = dm.build_topdown(_flat_scene(_CAM), scale_m=1.0, cam_height_m=_CAM)
am_bold.fuse(td_corr)
_g = []
for _yy in np.arange(0.4, 8.5, 0.1):
    _X, _Y = am_bold.to_anchor(0.0, float(_yy))
    _i, _j = am_bold.cells(_X, _Y)
    _g.append(bool(am_bold.logodds[int(_i), int(_j)] < -0.5))
_g = np.array(_g)
_yy_scan = np.arange(0.4, 8.5, 0.1)
green_to = float(_yy_scan[_g][-1]) if _g.any() else 0.0
check("ONE fuse paints the open corridor green far past FUSE_MAX_M",
      green_to >= 6.0, f"green out to {green_to:.1f} m")
# user 2026-08-16: no SANDWICH — far verified-FREE (past the structural
# 4.5 m bound) joins the paint tier, so the corridor is green
# CONTINUOUSLY, not green-black-green
_band = _g[(_yy_scan >= 0.6) & (_yy_scan <= 8.0)]
check("…and the green is CONTINUOUS (no black gap mid-corridor)",
      bool(_band.all()), f"{int(_band.sum())}/{len(_band)} green")
# nearest-cluster recall (the two-sinks confusion): two clusters of one
# phrase — the sentence must speak about the NEAR one
am_two = AnchorMap()
_ii = np.array([0, 0, 1, 1]); _jj = np.array([0, 1, 0, 1])
for _dx_c, _dy_c in ((0.0, 0.0),):
    pass
am_two.stamp_semantic("sink", np.array([1.0, 1.0, 0.9, 0.9]),
                      np.array([1.0, 1.1, 1.0, 1.1]), weight=9.0)   # LEFT near
am_two.stamp_semantic("sink", np.array([-3.0, -3.0, -3.1, -3.1]),
                      np.array([3.0, 3.1, 3.0, 3.1]), weight=9.0)   # RIGHT far
_rec = [r for r in am_two.semantic_recall() if r["phrase"] == "sink"]
check("semantic recall speaks about the NEAREST cluster, not the centroid",
      bool(_rec) and _rec[0]["bearing"] > 0 and _rec[0]["distance"] < 2.5,
      f"bearing {math.degrees(_rec[0]['bearing']):+.0f}° "
      f"d={_rec[0]['distance']:.1f}" if _rec else "no recall")
# region merging (user 2026-08-16 final): fragments of ONE region within
# ~0.8 m fuse into a single cluster; instances rooms apart stay two
am_reg = AnchorMap()
am_reg.stamp_semantic("kitchen area", np.array([0.0, 0.1, 0.0, 0.1]),
                      np.array([2.0, 2.0, 2.1, 2.1]), weight=9.0)
am_reg.stamp_semantic("kitchen area", np.array([0.6, 0.7, 0.6, 0.7]),
                      np.array([2.0, 2.0, 2.1, 2.1]), weight=9.0)
_hit = am_reg._sem_mask(am_reg.sem["kitchen area"], 2.0)
check("nearby region fragments MERGE into one cluster",
      len(am_reg._sem_components(_hit)) == 1,
      f"{len(am_reg._sem_components(_hit))} clusters")
_hit2 = am_two._sem_mask(am_two.sem["sink"], 2.0)
check("instances rooms apart stay separate clusters",
      len(am_two._sem_components(_hit2)) == 2,
      f"{len(am_two._sem_components(_hit2))} clusters")
_, mem_prefix = dm.memory_profile(am_bold)
check("…but the paint feeds NO verified consumer (memory prefix stays tiny)",
      float(mem_prefix.max()) < 1.0, f"max prefix {mem_prefix.max():.2f} m")
# a cell that carries ONLY paint (floor rays invalidated → OPEN, no
# floor evidence) must stay in the paint tier and flip on one wall return
def _paint_only_scene() -> np.ndarray:
    f2 = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    _, vs2 = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dy2 = -(vs2 - _W / 2.0) / f2
    depth = np.full((_H, _W), 25.0, np.float32)
    depth[dy2 < -1e-6] = 0.0            # no floor returns at all
    return depth


am_p = AnchorMap()
am_p.fuse(dm.build_topdown(_paint_only_scene(), scale_m=1.0,
                           cam_height_m=_CAM))
_Xp, _Yp = am_p.to_anchor(0.0, 3.05)
_ip, _jp = am_p.cells(_Xp, _Yp)
paint_lo = float(am_p.logodds[int(_ip), int(_jp)])
check("pure sightline paint stays in the PAINT tier (green, not verified)",
      -1.5 < paint_lo < -0.5, f"logodds {paint_lo:.2f}")
am_p.fuse(dm.build_topdown(np.minimum(_flat_scene(_CAM), 3.0),
                           scale_m=1.0, cam_height_m=_CAM))
_wall_lo = max(
    float(am_p.logodds[int(_i2), int(_j2)])
    for _yy2 in np.arange(2.85, 3.25, 0.05)
    for _X2, _Y2 in [am_p.to_anchor(0.0, float(_yy2))]
    for _i2, _j2 in [am_p.cells(_X2, _Y2)])
check("one real wall return overturns the paint where the wall stands",
      _wall_lo > 0.0, f"max logodds near the wall {_wall_lo:.2f}")

# ── sightline candidates + SAM floor-expanse filter (user 三次裁定) ─────
print()
print("── deep-blue aiming · SAM floor-expanse rejection ─────────────────")


def _doorway_scene() -> np.ndarray:
    """The verified prefix dies at 1.5 m (no floor returns beyond — dark
    carpet through a doorway) while the sightline plainly reaches the
    room's far wall at 5.5 m: the two-bedrooms shape, distilled."""
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -_CAM / dy[below])
    depth = np.minimum(depth, 5.5)                     # room's far wall
    # floor returns vanish past 1.5 m (carpet eats the depth), the walls
    # and far structure stay visible — sight deep, verified short
    floor_far = below & (depth > 1.5) & (dy < -0.05)
    depth[floor_far] = 0.0
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


td_door = dm.build_topdown(_doorway_scene(), scale_m=1.0, cam_height_m=_CAM)
door_wp = dm.propose(td_door)
sl = [w for w in door_wp if w.kind == "sightline"]
check("a doorway pinch with deep floor beyond yields a SIGHTLINE aim",
      bool(sl), f"{[(w.kind, round(w.distance, 1)) for w in door_wp]}")
if sl:
    w0 = sl[0]
    check("…aimed INTO the blue: past verified, within sight, ≤ v+2.5",
          w0.verified_m + 0.75 <= w0.distance <= w0.verified_m + 2.51
          and w0.distance <= w0.visible_m,
          f"aim {w0.distance:.2f} v {w0.verified_m:.2f} "
          f"seen {w0.visible_m:.2f}")
    check("…and the blind-walkable stride never exceeds the proven prefix",
          w0.stride_m <= w0.verified_m + 1e-6,
          f"stride {w0.stride_m:.2f}")

# SAM floor-expanse: a mask that IS the walking surface is rejected with
# its reason on record; an elevated slab (counter/bed) survives
from eharness.landmarks import LandmarkOrgan  # noqa: E402


class _FloorSAM(LandmarkOrgan):
    def __init__(self, mode: str):
        super().__init__("http://fake")
        self.mode = mode
        self.available = True

    def segment(self, rgb_b64, word):  # noqa: ARG002
        mask = np.zeros((_H, _W), bool)
        if self.mode == "floor":
            mask[_H // 2 + 20:, :] = True          # the whole visible floor
        else:                                       # an elevated slab ahead
            mask[_H // 2 - 30:_H // 2 + 30, _W // 4:3 * _W // 4] = True
        import base64 as _b64
        from io import BytesIO as _BIO
        from PIL import Image as _PI
        buf = _BIO()
        _PI.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
        return [{"mask_b64": _b64.b64encode(buf.getvalue()).decode(),
                 "iou_score": 0.9}]


def _wire_depth(depth: np.ndarray) -> dict:
    return {"__ndarray__": base64.b64encode(
        depth.astype(np.float32).tobytes()).decode(),
        "dtype": "float32", "shape": list(depth.shape)}


_flat = _flat_scene(_CAM)
res_floor = _FloorSAM("floor")
got = res_floor.sightings(_rgb64(), _wire_depth(_flat), ["hallway"],
                          scale_m=1.0)
check("a floor-expanse mask is REJECTED (SAM's 'hallway = the floor' habit)",
      not got and any("floor-expanse" in n
                      for n in res_floor.misses.get("hallway", [])),
      str(res_floor.misses.get("hallway", []))[:90])
# elevated slab at counter height in the depth: reuse counter scene
_cs2 = _counter_scene()
res_slab = _FloorSAM("slab")
got2 = res_slab.sightings(_rgb64(), _wire_depth(_cs2), ["counter"],
                          scale_m=1.0)
check("an elevated slab (counter-height) still lands as a sighting",
      bool(got2), f"{len(got2)} sightings")

# ── keyword fidelity (user 2026-08-16 终裁: 不能自由发挥) ────────────────
print()
print("── landmark keywords stay VERBATIM to the instruction ─────────────")
from eharness.resolver import _faithful_phrase, _validate  # noqa: E402

_instr = ("Go straight past the pool. Walk between the bar and chairs. "
          "Stop when you get to the corner of the bar.")
for emb, want in (("swimming pool", "pool"), ("bar counter", "bar"),
                  ("lounge chairs", "chairs"), ("pool", "pool")):
    got = _faithful_phrase(emb, _instr)
    check(f"'{emb}' → instruction's own '{want}'", got == want, f"got {got!r}")
check("a stem match returns the instruction's word",
      _faithful_phrase("hallway", "walk down the hall to the end") == "hall")
check("a split compound returns the instruction's spelling",
      _faithful_phrase("bedroom", "enter the bed room on the left")
      == "bed room")
check("pure invention (never in the instruction) is dropped",
      _faithful_phrase("treadmill", _instr) is None)
_parsed = _validate({"segments": ["past the pool"], "terminate": "stop at "
                     "the bar", "landmarks": ["swimming pool",
                                              "lounge chairs", "bar"]},
                    _instr)
check("_validate maps every mark back to instruction words",
      _parsed is not None and _parsed[2] == ["pool", "chairs", "bar"],
      str(_parsed[2] if _parsed else None))
import json as _json  # noqa: E402
_tab = _json.loads((_CA / "bridges" / "keywords"
                    / "rand100_keywords.json").read_text())
_viol = sum(1 for rows in _tab["splits"].values() for r in rows
            for p in r["landmarks"]
            if _faithful_phrase(p, r["instruction"]) != p)
check("the FIXED table holds zero non-verbatim phrases (200 eps audited)",
      _viol == 0, f"{_viol} violations")

# ── FOV-edge phantom potential (2026-08-16, 右缘小暗黄) ─────────────────
print()
print("── a plain floor mints ZERO potential (no FOV-edge phantoms) ──────")
_td_plain = dm.build_topdown(_flat_scene(_CAM), scale_m=1.0,
                             cam_height_m=_CAM)
check("plain floor, nothing to occlude → zero POTENTIAL cells",
      int(_td_plain.potential.sum()) == 0,
      f"{int(_td_plain.potential.sum())} phantom cells")

# ── the fan-of-fingers bug (2026-08-16): needle glimpses must not paint ──
print()
print("── paint needs angular support: no 15°-pitch fan ──────────────────")


def _slit_scene() -> np.ndarray:
    """Deep sight only through a ±1.5° needle slit; walls at 3 m."""
    f = (_W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(_W, dtype=np.float32),
                         np.arange(_W, dtype=np.float32))
    dx, dy = (us - _W / 2.0) / f, -(vs - _W / 2.0) / f
    depth = np.full((_H, _W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -_CAM / dy[below])
    wall = np.abs(np.degrees(np.arctan2(dx, np.ones_like(dx)))) > 1.5
    depth[wall] = np.minimum(depth[wall], 3.0)
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


am_fan = AnchorMap()
_td_slit = dm.build_topdown(_slit_scene(), scale_m=1.0, cam_height_m=_CAM)
for _k in range(4):                    # four 15°-spaced looks, in place
    am_fan.fuse(_td_slit)
    am_fan.odometry(math.radians(15), 0.0)
_gg = am_fan.logodds < -0.5
_ii, _jj = np.nonzero(_gg)
_rr = np.hypot(am_fan.ox + (_jj - am_fan.centre) * am_fan.cell,
               am_fan.oy - (_ii - am_fan.centre) * am_fan.cell)
check("needle glimpses paint NO far fingers (the 15°-pitch fan bug)",
      int(((_rr > 4.5) & (_rr < 8.5)).sum()) == 0,
      f"{int(((_rr > 4.5) & (_rr < 8.5)).sum())} far cells")

print()
print("── map label placement · crowded landmarks stay readable ───────────")
# five labels all anchored within a few px of each other (the kitchen-corner
# case: sink/oven/counter names drew on top of one another)
_anch = [(100.0, 100.0), (103.0, 98.0), (98.0, 104.0),
         (101.0, 101.0), (99.0, 99.0)]
_szs = [(60.0, 11.0)] * 5
_pl = dm.place_labels(_anch, 256.0, 256.0, _szs)


def _lab_rect(p, s):
    return (p[0] - 2, p[1] - 1, p[0] + s[0] + 2, p[1] + s[1] + 1)


_rects = [_lab_rect(p, s) for p, s in zip(_pl, _szs)]
_overlaps = sum(1 for a in range(5) for b in range(a + 1, 5)
                if _rects[a][0] < _rects[b][2] and _rects[a][2] > _rects[b][0]
                and _rects[a][1] < _rects[b][3] and _rects[a][3] > _rects[b][1])
check("five same-spot labels place with ZERO pairwise overlap",
      _overlaps == 0, f"{_overlaps} overlapping pairs")
check("…and every label stays inside the image",
      all(r[0] >= 0 and r[1] >= 0 and r[2] <= 256 and r[3] <= 256
          for r in _rects))
_edge = dm.place_labels([(250.0, 250.0)], 256.0, 256.0, [(60.0, 11.0)])[0]
check("a corner anchor is clamped into frame, never cropped",
      _edge[0] + 60 + 3 <= 256 and _edge[1] + 11 + 2 <= 256, str(_edge))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("stage3 P0+P1-1 regressions: all passed")
