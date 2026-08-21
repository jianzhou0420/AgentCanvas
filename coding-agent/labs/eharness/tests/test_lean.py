"""The LEAN line's execution contract (2026-08-18), against a fake env.

step / get_map / recall over LeanSlamToolSet: odometry from measured motion,
one RGB-D read per primitive into the SLAM side-car, keep-latest SAM stamped
at capture pose (+ decay on a nearby miss), the seen frames on disk as the
recall video, ONE current map on disk — and NOTHING else on disk or on the
surface. Run: python eharness/tests/test_lean.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harnesses" / "mini"))

import importlib.util as _ilu  # noqa: E402

from eharness.landmarks import Sighting  # noqa: E402

# the implementation under test is the exp_workspace folder copy (dev/
# coding-agent-xunyi carries the shell as exp folders; slamsam_rxr_01 is the
# map-v1 rung of the step / get_map / recall shell)
_EXP = Path(__file__).resolve().parents[2] / "exp_workspace"


def _load(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_lt = _load("_slamsam_rxr_toolset", _EXP / "slamsam_rxr_01" / "toolset.py")
GET_MAP_DESC, RECALL_DESC, LeanSlamToolSet, step_desc = (
    _lt.GET_MAP_DESC, _lt.RECALL_DESC, _lt.LeanSlamToolSet, _lt.step_desc)

FAILS: list[str] = []
H = W = 256
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def _rays(hfov=90.0):
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(W, dtype=np.float32))
    return (us - W / 2.0) / f, -(vs - W / 2.0) / f


def scene(panels=(), wall_at=None, corridor=None) -> np.ndarray:
    dx, dy = _rays()
    depth = np.full((H, W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -CAM_H / dy[below])
    if corridor is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dx) > 1e-6, corridor / np.abs(dx), np.inf)
        depth = np.minimum(depth, np.where(np.isfinite(t), t, 25.0))
    if wall_at is not None:
        depth = np.minimum(depth, wall_at)
    for x_lo, x_hi, z in panels:
        x, y = dx * z, dy * z
        band = (y > -CAM_H + 0.3) & (y < -CAM_H + 1.6) & (x > x_lo) & (x < x_hi)
        depth[band] = np.minimum(depth[band], z)
    return depth


def depth_wire(depth: np.ndarray) -> dict:
    return {"__ndarray__": base64.b64encode(depth.astype(np.float32).tobytes()
                                            ).decode(),
            "dtype": "float32", "shape": list(depth.shape)}


def rgb_wire(shade: int = 90) -> str:
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (W, H), (shade, shade, shade)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEnv(LeanSlamToolSet):
    """The real toolset; only the HTTP boundary is replaced (env_habitat's
    per-primitive step_discrete contract + observe_egocentric)."""

    def __init__(self, depth: np.ndarray, *, block_after_fwd: int | None = None,
                 dy_per_fwd: float = 0.0, **kw):
        self.scene_depth = depth
        self.prims: list[int] = []
        self.observes = 0
        self._step_count = 0
        self._fwd = 0
        self.block_after_fwd = block_after_fwd
        self.dy_per_fwd = dy_per_fwd      # stairs: metres climbed per forward
        super().__init__("http://fake", **kw)

    def _call(self, fn: str, inputs: dict) -> dict:
        if fn.endswith("observe_egocentric"):
            self.observes += 1
            f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
            return {"rgb": rgb_wire(), "depth": depth_wire(self.scene_depth),
                    "depth_units": {"known": True, "scale_m": 1.0,
                                    "normalized": False,
                                    "camera_height_m": CAM_H},
                    "intrinsics": {"fx": f, "fy": f, "cx": W / 2, "cy": H / 2,
                                   "width": W, "height": H}}
        if fn.endswith("step_discrete"):
            a = int(inputs["action"])
            self.prims.append(a)
            self._step_count += 1
            moved, collided = 0.0, False
            if a == 1:
                self._fwd += 1
                if self.block_after_fwd is not None and self._fwd > self.block_after_fwd:
                    moved, collided = 0.0, True
                else:
                    moved = 0.25
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": moved,
                             "actual_dy_m": (self.dy_per_fwd if (a == 1 and moved > 0)
                                             else 0.0),
                             "collided": collided},
                    "terminated": a == 0, "truncated": False}
        raise AssertionError(f"unexpected env call {fn}")


class FakeSAM:
    """One detection with a mask on the upright panel, unless muted."""
    available = True
    last_error = ""
    timeout = 5.0

    def __init__(self, phrase="stove"):
        self.calls = 0
        self.muted = False
        self.phrase = phrase
        self.hfov_deg = 90.0

    def sightings(self, rgb, depth, phrases, *, scale_m=None, synonyms=3):
        self.calls += 1
        if self.muted:
            return []
        mask = np.zeros((H, W), bool)
        mask[115:165, 150:190] = True
        return [Sighting(phrase=phrases[0], bearing=math.radians(15.0),
                         distance=3.0, angular_width=0.3, score=0.9,
                         pixels=int(mask.sum()), mask=mask)]


print("── surface ────────────────────────────────────────────────────────")
ts = FakeEnv(scene(), phrases=[], sam_url="")
check("exactly three tools", ts.tool_names() == ["step", "get_map", "recall"],
      str(ts.tool_names()))
check("step text carries the env's turn size (15)",
      "turn left 15 degrees" in step_desc(15.0))
check("ovon step text: 30° turns + 1 m stop rule",
      "turn left 30 degrees" in step_desc(30.0, "objnav")
      and "within 1 meter" in step_desc(30.0, "objnav"))
check("get_map text names frontiers + landmarks + fading",
      "FRONTIERS" in GET_MAP_DESC and "LANDMARKS" in GET_MAP_DESC
      and "fades" in GET_MAP_DESC)
check("recall text names frame/recent/range",
      all(k in RECALL_DESC for k in ("kind='frame'", "kind='recent'",
                                     "kind='range'")))

print("── odometry + per-primitive sensing ───────────────────────────────")
ts = FakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=[], sam_url="")
r = ts.execute("step", {"actions": [1, 1, 2, 2, 1]})
body = json.loads(r.content[-1]["text"])
check("5 primitives → 5 senses (one RGB-D read each)", ts.observes == 5,
      str(ts.observes))
check("theta = +2 turns left", abs(ts.theta - 2 * math.radians(15)) < 1e-9)
exp_x = 0.25 * -math.sin(2 * math.radians(15))
exp_y = 0.5 + 0.25 * math.cos(2 * math.radians(15))
check("px/py from measured 0.25 m forwards along heading",
      abs(ts.px - exp_x) < 1e-6 and abs(ts.py - exp_y) < 1e-6,
      f"{ts.px:.3f},{ts.py:.3f}")
check("result JSON: executed/requested/moved/blocked/steps/frame",
      body["executed"] == 5 and body["requested"] == 5
      and abs(body["moved_m"] - 0.75) < 1e-6 and body["blocked"] is False
      and body["steps_taken_total"] == 5 and body["frame"] == 0
      and "steps_remaining_approx" in body and "end_reason" in body)
check("the result carries the photo (image + one JSON text)",
      r.content[0]["type"] == "image_url" and len(r.content) == 2)
check("SLAM side-car built + integrated 5 frames",
      ts.slam is not None and ts.slam.tick == 5,
      str(getattr(ts.slam, "tick", None)))
check("map is not empty (free cells painted)",
      ts.slam is not None and int((ts.slam.occ.grid != 0).sum()) > 50)
check("frame recorded in memory when no live_dir",
      len(ts.frames) == 1 and ts.frames[0]["kind"] == "view"
      and "png" in ts.frames[0])

print("── blocked forward cancels the rest ────────────────────────────────")
ts = FakeEnv(scene(), phrases=[], sam_url="", block_after_fwd=2)
r = ts.execute("step", {"actions": [1, 1, 1, 1, 1, 1]})
body = json.loads(r.content[-1]["text"])
check("third forward blocked → executed 3 of 6, blocked=True, note",
      body["executed"] == 3 and body["blocked"] is True and "cancelled" in
      body.get("note", ""), json.dumps(body))
check("moved_m counts only real motion (0.5)", abs(body["moved_m"] - 0.5) < 1e-6)
check("obstacle stamped ahead on the map",
      ts.slam is not None and int((ts.slam.occ.grid == 2).sum()) > 0)

print("── STOP ────────────────────────────────────────────────────────────")
ts = FakeEnv(scene(), phrases=[], sam_url="")
r = ts.execute("step", {"actions": [1, 0, 1]})
body = json.loads(r.content[-1]["text"])
check("STOP ends the episode; later actions not sent",
      body["episode_over"] is True and body["end_reason"] == "stop_called"
      and ts.prims == [1, 0] and body["executed"] == 2)
r2 = ts.execute("step", {"actions": [1]})
check("stepping after STOP is refused",
      "error" in json.loads(r2.content[-1]["text"]))

print("── get_map / recall / disk contract ───────────────────────────────")
tmp = Path(tempfile.mkdtemp(prefix="lean_"))
sam = FakeSAM()
ts = FakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=["stove"],
             sam_url="http://fake-sam", live_dir=tmp)
ts.organ = sam                       # swap the HTTP detector for the fake
art = ts.bootstrap()
check("bootstrap writes bootstrap.json + bootstrap_current.png, frame#0",
      (tmp / "bootstrap.json").exists() and (tmp / "bootstrap_current.png").exists()
      and art["images"].get("current") == "bootstrap_current.png"
      and ts.frames[0]["idx"] == 0 and ts.frames[0]["path"] == "bootstrap_current.png")
r0 = ts.execute("get_map", {})
check("get_map before any step still renders (frame-0 map)",
      r0.content[0]["type"] == "image_url")
ts.execute("step", {"actions": [1, 1]})
ts.execute("step", {"actions": [2, 2]})
r = ts.execute("get_map", {})
payload = json.loads(r.content[-1]["text"])
check("get_map returns image + JSON with frontiers/landmarks/frame",
      r.content[0]["type"] == "image_url" and "frontiers" in payload
      and "landmarks" in payload and isinstance(payload["frame"], int),
      json.dumps(payload)[:200])
check("SAM ran (keep-latest) and the map carries the stove patch",
      sam.calls >= 1 and any(
          str(lm.get("name", lm.get("phrase", ""))).startswith("stove")
          for lm in payload["landmarks"]),
      json.dumps(payload["landmarks"])[:200])
names = sorted(p.name for p in tmp.iterdir())
views = [n for n in names if n.startswith("obs_") and "_step" in n]
maps = [n for n in names if n.startswith("obs_") and n.endswith("_map.png")]
check("disk: 2 view frames + 2 pulled maps + map_latest + bootstrap pair",
      len(views) == 2 and len(maps) == 2 and "map_latest.png" in names
      and "bootstrap.json" in names and "bootstrap_current.png" in names,
      str(names))
extras = [n for n in names if not (n.startswith(("obs_", "bootstrap_"))
                                   or n in ("map_latest.png", "bootstrap.json",
                                            "map_latest.png.tmp"))]
check("disk: NOTHING else (no heartbeat/state/actions.log/keyframes/frames dir)",
      not extras, str(extras))
r = ts.execute("recall", {"kind": "frame", "id": 1})
check("recall frame#1 → that image + meta",
      r.content[0]["type"] == "image_url"
      and json.loads(r.content[-1]["text"])["frame"] == 1)
r = ts.execute("recall", {"kind": "recent", "top_k": 2})
meta = json.loads(r.content[-1]["text"])
check("recall recent → filmstrip of the last 2 VIEWS (maps excluded)",
      r.content[0]["type"] == "image_url" and len(meta["frames"]) == 2
      and all(ts.frames[f["frame"]]["kind"] == "view" for f in meta["frames"]),
      json.dumps(meta))
r = ts.execute("recall", {"kind": "range", "start": 0, "end": 99})
meta = json.loads(r.content[-1]["text"])
check("recall range clamps to stored frames", len(meta["frames"]) == 3,
      json.dumps(meta))
r = ts.execute("recall", {"kind": "frame", "id": 42})
check("recall unknown id → not_found + valid range",
      json.loads(r.content[-1]["text"]).get("error") == "not_found")

print("── revocable landmark: a nearby miss decays the patch ──────────────")
grid = ts.slam.sem.grids.get("stove")
_o, _cs = ts.slam.occ.origin, ts.slam.occ.cell_size


def _near(px, py, r=1.2):
    """max vote among the patch cells within r metres of (px, py)."""
    zi, xi = np.nonzero(grid > 0)
    d = np.hypot((xi - _o) * _cs - px, (zi - _o) * _cs - py)
    return float(grid[zi[d < r], xi[d < r]].max()) if (d < r).any() else 0.0


# walk up to the patch (it sits ~3 m ahead of the start) and look twice
ts.px, ts.py, ts.theta = 0.9, 2.6, 0.0
before, far_before = _near(0.9, 2.6), _near(-1.0, 3.6, 0.3)
sam.muted = True
ts.execute("step", {"actions": [2]})
ts.execute("step", {"actions": [3]})
ts._sam_drain(wait=True)
after, far_after = _near(0.9, 2.6), _near(-1.0, 3.6, 0.3)
check("two misses right next to the patch clear the nearby cells",
      before >= 0.89 and after == 0.0, f"{before:.2f} → {after:.2f}")
check("cells farther than 1.2 m are untouched (you can't disprove what you "
      "can't see)", far_before == far_after and far_before > 0,
      f"{far_before:.2f} → {far_after:.2f}")

print("── map v2 (jian slam_r2r_02, merged 2026-08-18) ─────────────────────")
sam2 = FakeSAM()
ts2 = FakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=["stove"],
              sam_url="http://fake-sam", map_mode="v2")
ts2.organ = sam2
ts2.execute("step", {"actions": [1, 1]})
r = ts2.execute("get_map", {})
pl = json.loads(r.content[-1]["text"])
check("v2 side-car: LayeredOccupancyMap on floor 0, grid grows",
      type(ts2.slam.occ).__name__ == "LayeredOccupancyMap"
      and ts2.slam.occ.current_floor == 0 and int((ts2.slam.occ.grid != 0).sum()) > 50)
check("v2 get_map: image + landmarks, NO frontiers key",
      r.content[0]["type"] == "image_url" and "frontiers" not in pl
      and "landmarks" in pl and "map_window_m" in pl, json.dumps(pl)[:200])
w1 = ts2.slam.map_window
ts2.execute("step", {"actions": [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]})
ts2.execute("get_map", {})
w2 = ts2.slam.map_window
check("v2 window is grow-only (never pans, never shrinks)",
      w1 is not None and w2 is not None
      and w2[0] <= w1[0] and w2[1] >= w1[1] and w2[2] <= w1[2] and w2[3] >= w1[3],
      f"{w1} → {w2}")
check("v1 default unchanged: OccupancyMap + frontiers on the surface",
      type(ts.slam.occ).__name__ == "OccupancyMap" and "frontiers" in payload)
# stairs: the env reports vertical motion → pz odometry → jian's dwell-gated
# floor registry opens floor 2 once the body has been level up there
ts3 = FakeEnv(scene(), phrases=[], sam_url="", map_mode="v2", dy_per_fwd=0.3)
ts3.execute("step", {"actions": [2] * 8})            # 8 level frames on floor 1
ts3.execute("step", {"actions": [1] * 10})           # climb 3.0 m
ts3.execute("step", {"actions": [2] * 10})           # 10 level frames up there
check("pz odometry integrates actual_dy_m (3.0 m climbed)",
      abs(ts3.pz - 3.0) < 1e-6, f"{ts3.pz:.2f}")
check("map v2 registers floor 2/2 after a dwell on the upper floor",
      ts3.slam._floor_label() == "floor 2/2"
      and len(ts3.slam.occ.layers) == 2 and ts3.slam.occ.current_floor == 1,
      str(ts3.slam._floor_label()))
traj, start_here = ts3.slam._floor_track()
check("v2 trail is filtered to the current floor and drops the S ring there",
      0 < len(traj) < len(ts3.slam.track) and not start_here,
      f"{len(traj)}/{len(ts3.slam.track)} start_here={start_here}")
r = ts3.execute("get_map", {})
check("upper-floor get_map renders", r.content[0]["type"] == "image_url")

print("── OVON depth units ─────────────────────────────────────────────────")
raw = np.array([[0.0, 0.5, 1.0]], np.float32)
m = LeanSlamToolSet._depth_metres(raw, {"known": True, "normalized": True,
                                        "min_depth_m": 0.5, "max_depth_m": 5.0,
                                        "scale_m": 4.5})
check("normalised [0.5,5] m: raw 0.5→2.75, 1→5.0 (raw 0 = clipped → invalid)",
      np.allclose(m, [[0.0, 2.75, 5.0]]), str(m))
m2 = LeanSlamToolSet._depth_metres(raw, {"known": True, "normalized": True,
                                         "min_depth_m": 0.0, "max_depth_m": 10.0,
                                         "scale_m": 10.0})
check("VLN-CE normalised [0,10] m: raw 0.5→5.0", np.allclose(m2, [[0, 5, 10]]))
check("min-clipped pixel (raw 0 with min_depth 0.5) is INVALID, not a 0.5 m wall",
      m[0, 0] == 0.0 and m[0, 1] == 2.75, str(m))
check("v2 map carries jian's floor tag",
      ts2.slam._floor_label() == "floor 1/1", str(ts2.slam._floor_label()))


print("── slamsam_ovon_01: instruments + the target push ─────────────────")
_ov = _load("_slamsam_ovon_toolset", _EXP / "slamsam_ovon_01" / "toolset.py")


class OvonFakeEnv(_ov.LeanSlamToolSet):
    def __init__(self, depth, **kw):
        self.scene_depth = depth
        self.prims = []
        self._step_count = 0
        super().__init__("http://fake", **kw)

    def _call(self, fn, inputs):
        if fn.endswith("observe_egocentric"):
            f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
            return {"rgb": rgb_wire(), "depth": depth_wire(self.scene_depth),
                    "depth_units": {"known": True, "scale_m": 1.0,
                                    "normalized": False,
                                    "camera_height_m": CAM_H},
                    "intrinsics": {"fx": f, "fy": f, "cx": W / 2, "cy": H / 2,
                                   "width": W, "height": H}}
        if fn.endswith("step_discrete"):
            a = int(inputs["action"]); self.prims.append(a); self._step_count += 1
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": 0.25 if a == 1 else 0.0,
                             "actual_dy_m": 0.0, "collided": False},
                    "terminated": a == 0, "truncated": False}
        raise AssertionError(fn)


osam = FakeSAM(); osam.muted = True
ov = OvonFakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=["stove"],
                 sam_url="http://fake-sam", turn_deg=30.0, task="objnav",
                 map_mode="v1")
ov.organ = osam
check("ovon surface = step / get_map / get_pose / get_trajectory (no recall)",
      ov.tool_names() == ["step", "get_map", "get_pose", "get_trajectory"],
      str(ov.tool_names()))
r = ov.execute("step", {"actions": [3, 1, 1]})
body = json.loads(r.content[-1]["text"])
check("no target yet → plain step result", "target" not in body and "target_alert" not in body)
p = json.loads(ov.execute("get_pose", {}).content[0]["text"])
check("get_pose: yaw increases turning RIGHT (30° right → +30)",
      abs(p["yaw_deg"] - 30.0) < 1e-6 and p["z"] > 0.3, json.dumps(p))
tj = json.loads(ov.execute("get_trajectory", {}).content[0]["text"])
check("get_trajectory: points oldest first, n_total = sensed frames",
      tj["n_total"] == 3 and len(tj["points"]) == 3, json.dumps(tj)[:120])
# the detector sees the goal → the running leg is cut, alert + map ride the result
osam.muted = False
r = ov.execute("step", {"actions": [1, 1, 1, 1, 1, 1]})
texts = [c["text"] for c in r.content if c["type"] == "text"]
body = json.loads(texts[0])
imgs = [c for c in r.content if c["type"] == "image_url"]
check("target push: leg cut short, TARGET DETECTED alert + target block",
      body["executed"] < 6 and "target_alert" in body and "target" in body
      and body["target"]["source"] == "seen"
      and "target detected" in body.get("note", ""), json.dumps(body)[:300])
check("target push: photo + map both attached (2 images)", len(imgs) == 2,
      str(len(imgs)))
d0 = body["target"]["dist_m"]
ov._sam_drain(wait=True)          # settle in-flight jobs before muting
osam.muted = True
r2 = ov.execute("step", {"actions": [1, 1]})
ov._sam_drain(wait=True)
b2 = json.loads([c["text"] for c in r2.content if c["type"] == "text"][0])
check("later results keep a `target` block and do NOT re-alert (throttled)",
      "target" in b2 and "target_alert" not in b2 and "note" not in b2,
      json.dumps(b2.get("target")))
gm = json.loads(ov.execute("get_map", {}).content[-1]["text"])
check("get_map payload carries the target block", "target" in gm, json.dumps(gm.get("target")))

print("── slamsam_objnav_02: SAM as an EXTERNAL TOOL (detect_target) ──────")
_o2 = _load("_slamsam_objnav_toolset", _EXP / "slamsam_objnav_02" / "toolset.py")


class ObjnavFakeEnv(_o2.LeanSlamToolSet):
    def __init__(self, depth, *, block_after_fwd=None, **kw):
        self.scene_depth = depth
        self.prims = []
        self._step_count = 0
        self._fwd = 0
        self.block_after_fwd = block_after_fwd
        self.observes = 0
        super().__init__("http://fake", **kw)

    def _call(self, fn, inputs):
        if fn.endswith("observe_egocentric"):
            self.observes += 1
            f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
            return {"rgb": rgb_wire(), "depth": depth_wire(self.scene_depth),
                    "depth_units": {"known": True, "scale_m": 1.0,
                                    "normalized": False,
                                    "camera_height_m": CAM_H},
                    "intrinsics": {"fx": f, "fy": f, "cx": W / 2, "cy": H / 2,
                                   "width": W, "height": H}}
        if fn.endswith("step_discrete"):
            a = int(inputs["action"]); self.prims.append(a); self._step_count += 1
            moved, collided = 0.0, False
            if a == 1:
                self._fwd += 1
                if self.block_after_fwd is not None and self._fwd > self.block_after_fwd:
                    collided = True
                else:
                    moved = 0.25
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": moved,
                             "actual_dy_m": 0.0, "collided": collided},
                    "terminated": a == 0, "truncated": False}
        raise AssertionError(fn)


class StrictFakeSAM(FakeSAM):
    """Records the phrases + synonyms it was asked for (the strictness
    contract); score / bearing / distance are settable, and `shrink_per_call`
    lets the object come closer with every look (the final push)."""

    def __init__(self, phrase="pillow", score=0.9):
        super().__init__(phrase)
        self.asked = []
        self.score = score
        self.bearing_deg = 15.0          # LEFT positive (Sighting contract)
        self.distance = 3.0
        self.shrink_per_call = 0.0

    def sightings(self, rgb, depth, phrases, *, scale_m=None, synonyms=3):
        self.asked.append((list(phrases), synonyms))
        out = super().sightings(rgb, depth, phrases, scale_m=scale_m,
                                synonyms=synonyms)
        for s in out:
            s.score = self.score
            s.bearing = math.radians(self.bearing_deg)
            s.distance = self.distance
        if out and self.shrink_per_call:
            self.distance = max(0.05, self.distance - self.shrink_per_call)
        return out


tmp2 = Path(tempfile.mkdtemp(prefix="objnav02_"))
dsam = StrictFakeSAM(); dsam.muted = True
o2 = ObjnavFakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=["pillow"],
                   sam_url="http://fake-sam", turn_deg=30.0, task="objnav",
                   map_mode="v1", sam_synonyms=0, sam_score_thresh=0.85,
                   live_dir=tmp2)
o2.organ = dsam
check("objnav_02 surface = step / detect_target / get_map / get_pose / get_trajectory",
      o2.tool_names() == ["step", "detect_target", "get_map", "get_pose",
                          "get_trajectory"], str(o2.tool_names()))
check("step text: nothing watches the frames, points at detect_target, names the FINAL PUSH",
      "detect_target" in _o2.step_desc(30.0) and "Nothing watches" in _o2.step_desc(30.0)
      and "FINAL PUSH" in _o2.step_desc(30.0) and "0.5 m" in _o2.step_desc(30.0))
check("detect text: exact word / no synonyms / 0.85 gate / positive = RIGHT / stamped",
      all(k in _o2.detect_desc("pillow", 0.85) for k in
          ("'pillow'", "no synonyms", "0.85", "RIGHT", "stamped on the map")))
check("get_map text: high-confidence patch, patches stay (no fading)",
      "high confidence" in _o2.GET_MAP_DESC and "patches stay" in _o2.GET_MAP_DESC
      and "fade" not in _o2.GET_MAP_DESC)
o2.bootstrap()
r = o2.execute("step", {"actions": [3, 1, 1, 1, 1, 1, 1]})
body = json.loads(r.content[-1]["text"])
check("no streaming: the detector is NOT run by step (0 SAM calls), leg runs whole",
      dsam.calls == 0 and body["executed"] == 7 and "target" not in body
      and "final_push" not in body, json.dumps(body)[:200])
check("side-car carries the goal word's semantic layer (fed only by the gate)",
      o2.slam is not None and o2.slam.sem is not None
      and list(o2.slam.sem.grids) == ["pillow"])
# a MISS: the model asks, the detector sees nothing → nothing stamped
r = o2.execute("detect_target", {})
b = json.loads(r.content[-1]["text"])
check("detect miss → JSON only, detected=false, no image, nothing stamped",
      b["detected"] is False and b["instances"] == [] and "stamped_on_map" not in b
      and all(c["type"] == "text" for c in r.content), json.dumps(b)[:200])
check("strict: asked for the ONE goal word with synonyms=0",
      dsam.asked == [(["pillow"], 0)], str(dsam.asked))
check("detect runs on the last sensed frame — no extra env read",
      o2.observes == 8, str(o2.observes))     # bootstrap + 7 primitives
gm = json.loads(o2.execute("get_map", {}).content[-1]["text"])
check("map after a miss: no landmarks (nothing stamped)", not gm.get("landmarks"),
      json.dumps(gm.get("landmarks")))
# a match UNDER the gate → dropped, not returned, not stamped
dsam.muted = False; dsam.score = 0.8
r = o2.execute("detect_target", {})
b = json.loads(r.content[-1]["text"])
check("score 0.8 < gate 0.85 → plain miss (not returned, no image)",
      b["detected"] is False and all(c["type"] == "text" for c in r.content),
      json.dumps(b)[:200])
gm = json.loads(o2.execute("get_map", {}).content[-1]["text"])
check("…and NOT stamped on the map", not gm.get("landmarks"), json.dumps(gm.get("landmarks")))
# a HIT at the gate: overlay + instances + stamped
dsam.score = 0.9
r = o2.execute("detect_target", {})
texts = [c["text"] for c in r.content if c["type"] == "text"]
imgs = [c for c in r.content if c["type"] == "image_url"]
b = json.loads(texts[0])
check("detect hit → overlay image + instances (dir_deg RIGHT-positive, dist, score) + stamped flag",
      b["detected"] is True and len(imgs) == 1 and len(b["instances"]) == 1
      and abs(b["instances"][0]["dir_deg"] + 15.0) < 1e-6      # bearing +15° left
      and abs(b["instances"][0]["dist_m"] - 3.0) < 1e-6
      and b["instances"][0]["score"] == 0.9 and b.get("stamped_on_map") is True,
      json.dumps(b)[:300])
gm = json.loads(o2.execute("get_map", {}).content[-1]["text"])
check("ONE high-score detection already renders on the map (landmarks lists 'pillow')",
      any(str(lm.get("name", "")).startswith("pillow") for lm in gm.get("landmarks", [])),
      json.dumps(gm.get("landmarks"))[:200])
check("get_map payload carries NO target block on this arm", "target" not in gm)
grid = o2.slam.sem.grids["pillow"]; peak_before = float(grid.max())
dsam.muted = True
o2.execute("detect_target", {})                    # a deliberate miss right here
check("no decay on this arm: a miss leaves the stamped patch untouched",
      float(grid.max()) == peak_before and peak_before >= 0.9, f"{peak_before:.2f}")
r = o2.execute("step", {"actions": [1, 1]})
body = json.loads(r.content[-1]["text"])
check("step after a hit still carries no target block (ask again instead)",
      "target" not in body and "final_push" not in body, json.dumps(body)[:200])
names2 = sorted(p.name for p in tmp2.iterdir())
dets = [n for n in names2 if n.startswith("det_")]
check("disk: one det_NNNN overlay for the one hit; views/maps/bootstrap/map_latest",
      len(dets) == 1 and any(n.startswith("obs_") for n in names2)
      and "map_latest.png" in names2 and "bootstrap.json" in names2, str(names2))

print("── slamsam_objnav_02: the FINAL PUSH on STOP ────────────────────────")
# (a) STOP with the target 2.0 m ahead, 40° to the right → turn once (round(40/30)=1),
#     walk forward while the detector still sees it and it is > 0.5 m, then STOP
psam = StrictFakeSAM(); psam.bearing_deg = -40.0; psam.distance = 2.0
psam.shrink_per_call = 0.25          # every look after a step: 0.25 m closer
o3 = ObjnavFakeEnv(scene(panels=[(0.5, 1.5, 3.0)]), phrases=["pillow"],
                   sam_url="http://fake-sam", turn_deg=30.0, task="objnav",
                   sam_score_thresh=0.85)
o3.organ = psam
o3.execute("step", {"actions": [1]})
n0 = len(o3.prims)
r = o3.execute("step", {"actions": [0]})
body = json.loads(r.content[-1]["text"])
fp = body.get("final_push") or {}
check("final push: turned right once toward the 40°-right target, then walked forward",
      fp.get("turned") == 1 and o3.prims[n0] == 3 and fp.get("steps", 0) >= 5
      and all(a == 1 for a in o3.prims[n0 + 1:-1]), json.dumps(fp))
check("final push: stopped at close_enough (≤ 0.5 m), then the STOP primitive went out",
      fp.get("stopped_because") == "close_enough" and fp.get("to_m", 9) <= 0.5
      and o3.prims[-1] == 0 and body["episode_over"] is True
      and body["end_reason"] == "stop_called", json.dumps(fp))
check("final push: moved_m + note + counted steps in the JSON",
      body["moved_m"] >= 1.0 and "final push before STOP" in body.get("note", "")
      and body["steps_taken_total"] == len(o3.prims), json.dumps(body)[:300])
check("final push: the pushed primitives were sensed + integrated (map ticks)",
      o3.slam.tick == len([a for a in o3.prims if a != 0]), str(o3.slam.tick))
# (b) STOP with nothing detected → plain STOP, no push, no extra primitives
qsam = StrictFakeSAM(); qsam.muted = True
o4 = ObjnavFakeEnv(scene(), phrases=["pillow"], sam_url="http://fake-sam",
                   turn_deg=30.0, task="objnav", sam_score_thresh=0.85)
o4.organ = qsam
o4.execute("step", {"actions": [1]})
r = o4.execute("step", {"actions": [0]})
body = json.loads(r.content[-1]["text"])
check("no detection at STOP → no push: prims = [1, 0], no final_push key",
      o4.prims == [1, 0] and "final_push" not in body and body["episode_over"] is True,
      json.dumps(body)[:200])
# (c) push meets an obstacle → stops pushing (blocked), still STOPs
bsam = StrictFakeSAM(); bsam.bearing_deg = 0.0; bsam.distance = 2.0
o5 = ObjnavFakeEnv(scene(), phrases=["pillow"], sam_url="http://fake-sam",
                   turn_deg=30.0, task="objnav", sam_score_thresh=0.85,
                   block_after_fwd=2)
o5.organ = bsam
r = o5.execute("step", {"actions": [0]})
body = json.loads(r.content[-1]["text"])
fp = body.get("final_push") or {}
check("push blocked after 2 free forwards → stopped_because=blocked, blocked=True, STOP sent",
      fp.get("stopped_because") == "blocked" and fp.get("steps") == 3
      and body["blocked"] is True and o5.prims[-1] == 0, json.dumps(fp))
# (d) already close (≤ 0.5 m) → no motion at all
csam = StrictFakeSAM(); csam.distance = 0.4; csam.bearing_deg = 0.0
o6 = ObjnavFakeEnv(scene(), phrases=["pillow"], sam_url="http://fake-sam",
                   turn_deg=30.0, task="objnav", sam_score_thresh=0.85)
o6.organ = csam
r = o6.execute("step", {"actions": [0]})
fp = (json.loads(r.content[-1]["text"]).get("final_push") or {})
check("already within 0.5 m → already_close, prims = [0]",
      fp.get("stopped_because") == "already_close" and o6.prims == [0], json.dumps(fp))
# (e) final_push=False bakes the plain STOP
o7 = ObjnavFakeEnv(scene(), phrases=["pillow"], sam_url="http://fake-sam",
                   turn_deg=30.0, task="objnav", sam_score_thresh=0.85,
                   final_push=False)
o7.organ = StrictFakeSAM()
r = o7.execute("step", {"actions": [0]})
check("final_push=False → plain STOP", o7.prims == [0]
      and "final_push" not in json.loads(r.content[-1]["text"]))

# the exp folder bridge registers the five tools with the baked knobs
os.environ.setdefault("OBJNAV_SERVER_URL", "http://127.0.0.1:1")   # reset fails fast → GOAL ""
_b2 = _ilu.spec_from_file_location("_slamsam_objnav_bridge_probe",
                                   _EXP / "slamsam_objnav_02" / "bridge.py")
_bm = _ilu.module_from_spec(_b2)
_b2.loader.exec_module(_bm)
import asyncio as _aio  # noqa: E402
_tools2 = _aio.run(_bm.mcp.list_tools())
check("objnav_02 bridge registers step/detect_target/get_map/get_pose/get_trajectory",
      sorted(t.name for t in _tools2) == ["detect_target", "get_map", "get_pose",
                                          "get_trajectory", "step"],
      str([t.name for t in _tools2]))
check("objnav_02 bridge bakes 30° turns, synonyms 0, gate 0.85, map v1, final push 0.5 m / 8",
      _bm.TURN_DEG == 30.0 and _bm.SAM_SYNONYMS == 0 and _bm.SAM_SCORE_THRESH == 0.85
      and _bm.MAP_MODE == "v1" and _bm.FINAL_PUSH is True and _bm.PUSH_STOP_M == 0.5
      and _bm.PUSH_MAX_STEPS == 8)
_dd = next(t.description for t in _tools2 if t.name == "detect_target")
_sd = next(t.description for t in _tools2 if t.name == "step")
check("bridge descriptions carry the 0.85 gate and the final push",
      "0.85" in _dd and "FINAL PUSH" in _sd)

print("── exp_workspace folder bridge (slamsam_rxr_01 = map v1) ───────────")
os.environ["EXP_INSTRUCTION"] = ("Go straight past the pool. Walk between the bar "
                                 "and chairs.  Stop when you get to the corner of "
                                 "the bar.  That's where you will wait.")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_slamsam_bridge_probe",
    Path(__file__).resolve().parents[2] / "exp_workspace" / "slamsam_rxr_01"
    / "bridge.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import asyncio  # noqa: E402

tools = asyncio.run(mod.mcp.list_tools())
tnames = [t.name for t in tools]
check("folder bridge registers step/get_map/recall only",
      sorted(tnames) == ["get_map", "recall", "step"], str(tnames))
sd = next(t.description for t in tools if t.name == "step")
gd = next(t.description for t in tools if t.name == "get_map")
check("folder bridge (rung 01) bakes 15° turns / 3 m stop / map v1 (frontiers)",
      "15 degrees" in sd and "3 meters" in sd and "FRONTIERS" in gd)
check("folder bridge phrases_for() = the episode's fixed keywords",
      mod.phrases_for(os.environ["EXP_INSTRUCTION"]) == ["pool", "bar", "chairs"],
      str(mod.phrases_for(os.environ["EXP_INSTRUCTION"])))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("lean contract: all passed")
