"""The toolset EXECUTION contract (§10.2 / §10.6 / §3 P0), against a mock env.

What the geometry suites cannot see is what actually crosses the wire: the
distance handed to step_hightolow, the images handed to the model, the frames
on which a SAM mask is allowed to stamp the semantic map. Each of those has
already been wrong while every geometry test was green — stride computed but
not executed, the map drawn only for humans, masks re-projected two frames
stale — so this suite drives the real DepthWaypointToolSet against a fake
env and asserts on the traffic itself.
"""
from __future__ import annotations

import base64
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harnesses" / "mini"))

from eharness import depthmap as dm  # noqa: E402
from eharness.landmarks import Sighting  # noqa: E402
from depth_toolset import DepthWaypointToolSet  # noqa: E402

FAILS: list[str] = []
H = W = 256
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def _rays():
    f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
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


def rgb_wire() -> str:
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (W, H), (90, 90, 90)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEnv(DepthWaypointToolSet):
    """The real toolset; only the HTTP boundary is replaced. Mirrors the
    §4 env contract: rotate-only step_hightolow (distance<0.25 walks zero
    forwards) and per-primitive step_discrete with measured motion."""

    def __init__(self, depth: np.ndarray, **kw):
        self.scene_depth = depth
        self.sent: list[dict] = []          # every step_hightolow request
        self.prims: list[int] = []          # every discrete primitive
        self._step_count = 0
        kw.setdefault("traj_archive", False)
        super().__init__("http://fake", **kw)

    def _call(self, fn: str, inputs: dict) -> dict:
        if fn.endswith("observe_egocentric"):
            return {"rgb": rgb_wire(), "depth": depth_wire(self.scene_depth),
                    "depth_units": {"known": True, "scale_m": 1.0,
                                    "normalized": False}}
        if fn.endswith("step_hightolow"):
            self.sent.append(dict(inputs))
            ksteps = int(float(inputs["distance"]) / 0.25)
            self._step_count += ksteps
            return {"info": {"step_count": self._step_count},
                    "terminated": False, "truncated": False}
        if fn.endswith("step_discrete"):
            a = int(inputs["action"])
            self.prims.append(a)
            self._step_count += 1
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": 0.25 if a == 1 else 0.0,
                             "collided": False},
                    # real env contract: STOP terminates the episode
                    "terminated": a == 0, "truncated": False}
        raise AssertionError(f"unexpected env call {fn}")

    def _call2(self, fn: str, inputs: dict, config=None) -> dict:
        raise AssertionError(f"unexpected env call {fn}")


class FakeSAM:
    """One detection, with a mask, every time it is asked."""

    available = True
    last_error = ""

    def __init__(self):
        self.calls = 0

    def sightings(self, rgb, depth, phrases, *, scale_m=None, synonyms=3):
        self.calls += 1
        mask = np.zeros((H, W), bool)
        # covers the upright panel at (0.5..1.5, z=3.0) in the stale-mask
        # scene — project_mask rightly keeps only points ABOVE the floor
        mask[115:165, 150:190] = True
        return [Sighting(phrase=phrases[0], bearing=math.radians(15.0),
                         distance=3.0, angular_width=0.3, score=0.9,
                         pixels=int(mask.sum()), mask=mask)]


print("── stride execution (§10.2 P0) ─────────────────────────────────────")
ts = FakeEnv(scene(), instruction="walk forward", sam_url="")
view = ts._tool_observe()
cands = view.info.get("places_you_can_walk_to") or {}
check("the mock env produces candidates", bool(cands), f"{len(cands)}")
far = None
for k, c in cands.items():
    check(f"candidate {k} carries the dual answer",
          all(f in c for f in ("safe_stride_m", "verified_ground_m",
                               "visible_clear_depth_m")),
          json.dumps({f: c.get(f) for f in ("distance_m", "safe_stride_m",
                                            "verified_ground_m")}))
    check(f"candidate {k}: stride never exceeds verified ground",
          c["safe_stride_m"] <= c["verified_ground_m"] + 1e-6,
          f"{c['safe_stride_m']} vs {c['verified_ground_m']}")
    if far is None:
        far = int(k)          # §2.1: the point IS mid-range; any candidate
                              # exercises the stride contract now
check("candidates exist to exercise the stride contract",
      far is not None,
      "; ".join(f"{c['distance_m']}m/stride {c['safe_stride_m']}m"
                for c in cands.values()))
if far is not None:
    aim = cands[str(far)]
    ts._tool_goto(place=far)
    check("the rotation went to the env as a rotate-only call (distance 0)",
          bool(ts.sent) and ts.sent[-1]["distance"] == 0.0,
          str(ts.sent[-1:]))
    n_fwd = ts.prims.count(1)
    check("the env received the SAFE STRIDE as primitives, not the aim",
          abs(n_fwd * 0.25 - aim["safe_stride_m"]) <= 0.25 + 1e-6,
          f"{n_fwd} forwards = {n_fwd * 0.25:.2f} m; "
          f"aim was {aim['distance_m']:.2f} m")
    check("…and the stride is within the pacing ceiling",
          n_fwd * 0.25 <= dm.MAX_STRIDE_M + 1e-6, f"{n_fwd * 0.25:.2f}")
    check("goto auto-observes afterwards (numbers refreshed)",
          bool(ts._waypoints), f"{len(ts._waypoints)} new candidates")
    # mid-range aiming can WALK the whole point in one stride — then the
    # commitment is adopted, arrived at, and correctly released. What must
    # be true is that the model's choice became a commitment at all.
    check("the model's chosen place became the held goal (then arrived)",
          ts.goal.adopted_m > 0,
          f"adopted {ts.goal.adopted_m:.1f} m, anchor now {ts.goal.anchor}")

print()
print("── the model payload (§10.6) ───────────────────────────────────────")
ts2 = FakeEnv(scene(panels=[(-0.4, 0.4, 2.0)]), instruction="x", sam_url="")
for _ in range(4):                      # a few looks so the map has content
    ts2._tool_observe()
view = ts2._tool_observe()
kinds = [p.get("type") for p in view.content]
n_img = sum(1 for p in view.content if "image" in str(p.get("type")))
texts = " ".join(p.get("text", "") for p in view.content
                 if p.get("type") == "text")
check("every observe carries BOTH images (RGB + accumulated map)",
      n_img >= 2, f"{n_img} images, parts {kinds}")
check("the images are explained, not thrown in",
      "IMAGE 1" in texts and "IMAGE 2" in texts
      # §21.7: the legend teaches the dual-truth composite
      and "TWO PANELS" in texts and "facing UP" in texts
      and "fixed to the WORLD" in texts)
# §20.4: the label IS the canonical legend — colour semantics included
check("IMAGE 2 carries the canonical map legend (colours + potential)",
      all(w in texts for w in ("GREEN", "RED", "AMBER", "BLUE-PURPLE"))
      and "never walked" in texts)
check("deterministic geometry readout rides along",
      "CURRENT GEOMETRY" in texts and "verified FREE" in texts)
check("map registration confidence is spoken",
      "map registration" in texts.lower() or "match" in texts)
telem1 = ts2._map_telemetry(ts2.amap.planning_view(ts2._last_td)
                            or ts2._last_td,
                            *dm.passable_profile(ts2._last_td),
                            ts2._waypoints)
telem2 = ts2._map_telemetry(ts2.amap.planning_view(ts2._last_td)
                            or ts2._last_td,
                            *dm.passable_profile(ts2._last_td),
                            ts2._waypoints)
check("same map state → same telemetry (no model in the loop)",
      json.dumps(telem1, sort_keys=True) == json.dumps(telem2, sort_keys=True))
check("telemetry never smuggles a decision",
      not any(k in json.dumps(telem1) for k in
              ("best_direction", "probably", "likely")))

# the audit's off-axis island: the flank too tight to land a candidate must
# still be SPOKEN — "turn to face it" — never silently dropped
ts_off = FakeEnv(scene(panels=[(0.6, 1.6, 2.0)]), instruction="x", sam_url="")
view_off = ts_off._tool_observe()
texts_off = " ".join(p.get("text", "") for p in view_off.content
                     if isinstance(p, dict) and p.get("type") == "text")
n_cand = len(view_off.info.get("places_you_can_walk_to") or {})
check("a flank with no landable candidate is named as a turn-first way",
      n_cand >= 2 or "turn to face it" in texts_off,
      f"{n_cand} candidates; turn-advice "
      f"{'present' if 'turn to face it' in texts_off else 'ABSENT'}")

print()
print("── stale masks stay dead (§3 P0) ───────────────────────────────────")
sam = FakeSAM()
ts3 = FakeEnv(scene(panels=[(0.5, 1.5, 3.0)]),
              instruction="find the bar counter", sam_url="")
ts3.landmarks = sam                     # detector on, cadence 2
ts3.phrases = ["bar counter"]
ts3.landmark_every = 2
ts3._tool_observe()                     # look 0: cadence frame — detects
votes_after_hit = float(ts3.amap.sem.get("bar counter", np.zeros(1)).sum())
check("a cadence frame stamps the detection",
      votes_after_hit > 0, f"votes {votes_after_hit:.1f}")
check("…and the frame's own sightings are visible", len(ts3._sightings) == 1)
ts3._tool_observe()                     # look 1: NOT a cadence frame
votes_after_skip = float(ts3.amap.sem["bar counter"].sum())
check("a non-cadence frame stamps NOTHING (no stale re-projection)",
      abs(votes_after_skip - votes_after_hit) < 1e-6,
      f"{votes_after_hit:.1f} → {votes_after_skip:.1f}")
check("…and carries no ghost sightings", len(ts3._sightings) == 0)
check("the detector really was skipped, not re-run", sam.calls == 1,
      f"{sam.calls} call(s)")

print()
print("── §9.4 · the micro-trajectory data plane ──────────────────────────")
import tempfile  # noqa: E402

tdir = Path(tempfile.mkdtemp(prefix="traj_"))
ts5 = FakeEnv(scene(), instruction="x", sam_url="", traj_archive=True,
              live_dir=tdir)
ts5._tool_observe()
fuses_before = ts5.amap.updates
prims_before = len(ts5.prims)
view = ts5._tool_goto(place=1)
n_fwd = ts5.prims[prims_before:].count(1)
n_turn = sum(1 for p in ts5.prims[prims_before:] if p in (2, 3))
check("a goto expands into forward primitives", n_fwd >= 4, f"{n_fwd} forwards")
# §14.4: the yaw is REAL turn primitives now — each one sensed and fused,
# plus at most one rotate-only micro-alignment look for the sub-15° residue.
check("EVERY primitive fused — real TURN primitives included (§14.4)",
      ts5.amap.updates - fuses_before in (n_fwd + n_turn, n_fwd + n_turn + 1),
      f"{ts5.amap.updates - fuses_before} fusions for {n_fwd} fwd + {n_turn} turn")
mani = (tdir / "trajectory" / "manifest.jsonl")
lines = [json.loads(l) for l in mani.read_text().splitlines()] if mani.exists() else []
check("trajectory manifest has EXACTLY one row per sensed packet",
      len(lines) == ts5.amap.updates - fuses_before,
      f"{len(lines)} rows for {ts5.amap.updates - fuses_before} fusions")
check("each row is fully addressed (action/primitive/step/map version)",
      bool(lines) and all(
          all(k in r for k in ("frame_id", "action_id", "primitive_index",
                               "env_step", "actual_delta_m", "collided",
                               "map_version", "rgb_file")) for r in lines))
rgbs = sorted((tdir / "trajectory" / "rgb").glob("frame_*.png"))
deps = sorted((tdir / "trajectory" / "depth").glob("frame_*.npz"))
check("unique RGB + depth files per packet, no step-glob guessing",
      len(rgbs) == len(lines) and len(deps) == len(lines),
      f"{len(rgbs)} png, {len(deps)} npz")
check("trail length agrees with the sum of actual deltas",
      abs(sum(r["actual_delta_m"] for r in lines) -
          sum(0.25 for p in ts5.prims[prims_before:] if p == 1)) < 1e-6)
# review P0: the snapshot used to be published BEFORE the versioned depth
# json existed, so its own exists-check bailed on every call and the file
# was never written at all
check("live_snapshot.json actually publishes",
      (tdir / "live_snapshot.json").exists())
if (tdir / "live_snapshot.json").exists():
    snap = json.loads((tdir / "live_snapshot.json").read_text())
    check("…and carries the §14.14 identity (episode / action / frame)",
          snap.get("identity", {}).get("episode") == tdir.name
          and "action_id" in snap.get("identity", {})
          and "sensor_frame" in snap.get("identity", {}),
          str(snap.get("identity")))

# mixed step batch: 4 primitives → 4 packets, turns included
ts6 = FakeEnv(scene(), instruction="x", sam_url="", traj_archive=True,
              live_dir=Path(tempfile.mkdtemp(prefix="traj_")))
ts6._tool_observe()
f0, p0 = ts6.amap.updates, len(ts6.prims)
ts6._tool_step(actions=[3, 3, 1, 1])
check("step([3,3,1,1]) → 4 primitives, 4 sensed packets",
      ts6.prims[p0:] == [3, 3, 1, 1]
      and ts6.amap.updates - f0 in (4, 5),
      f"prims {ts6.prims[p0:]}, fusions {ts6.amap.updates - f0}")

# collision mid-leg: the executor stops, odometry only counts what ran
class CollideEnv(FakeEnv):
    def _call(self, fn, inputs):
        if fn.endswith("step_discrete") and int(inputs["action"]) == 1 \
                and self.prims.count(1) >= 2:
            self.prims.append(1)
            self._step_count += 1
            return {"info": {"step_count": self._step_count,
                             "actual_translation_m": 0.0, "collided": True},
                    "terminated": False, "truncated": False}
        return super()._call(fn, inputs)

ts7 = CollideEnv(scene(), instruction="x", sam_url="")
ts7._tool_observe()
r = ts7._execute_primitives([1] * 8, action_id="goto#T")
check("a collision interrupts the leg (no blind continuation)",
      r["interrupted"] and r["executed"] == 3,
      f"executed {r['executed']}/8, reason: {r['reason'][:40]}")
check("phantom metres never enter odometry",
      abs(r["moved_m"] - 0.5) < 1e-6, f"moved {r['moved_m']} m")

print()
print("── declared units beat the guess ───────────────────────────────────")
ts4 = FakeEnv(scene(wall_at=0.8), instruction="x", sam_url="")
ts4._tool_observe()
td = ts4._last_td
check("a nose-to-wall metric frame is NOT inflated 10×",
      td is not None and td.ahead_m() < 2.0,
      f"ahead {td.ahead_m():.2f} m" if td is not None else "no grid")

print()
print("── §14.4 · face is REAL turn primitives ────────────────────────────")
ts8 = FakeEnv(scene(), instruction="x", sam_url="")
ts8._tool_observe()
p0, s0 = len(ts8.prims), ts8.steps_taken
ts8._tool_face(direction=90)
turns = [p for p in ts8.prims[p0:] if p in (2, 3)]
check("face(90) spends six 15° env turn primitives",
      turns == [2] * 6, f"prims {ts8.prims[p0:]}")
check("…each one counted against the step budget",
      ts8.steps_taken - s0 == 6, f"{ts8.steps_taken - s0} env steps")
check("…and no rotate-only channel was used for the notches",
      all(abs(float(s.get("angle", 0))) < math.radians(15.0) + 1e-6
          for s in ts8.sent), f"{len(ts8.sent)} hightolow calls")
ts8b = FakeEnv(scene(), instruction="x", sam_url="")
ts8b._tool_observe()
p0 = len(ts8b.prims)
f0 = ts8b.amap.updates
ts8b._tool_face(direction=40)
check("face(40) = two real notches + a sensed sub-15° micro-align",
      [p for p in ts8b.prims[p0:] if p in (2, 3)] == [2, 2]
      and len(ts8b.sent) == 1
      and abs(math.degrees(float(ts8b.sent[0]["angle"])) - 10.0) < 0.5,
      f"prims {ts8b.prims[p0:]}, residual "
      f"{[round(math.degrees(float(s['angle'])), 1) for s in ts8b.sent]}")
check("every yaw frame fused (2 notches + residual look ≥ 3 fusions)",
      ts8b.amap.updates - f0 >= 3, f"{ts8b.amap.updates - f0} fusions")

print()
print("── §14.5 · perception failure fails CLOSED ─────────────────────────")


class HttpFailEnv(FakeEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fail_after = 10**9

    def _call(self, fn, inputs):
        if fn.endswith("observe_egocentric") and self.prims and \
                self.prims.count(1) >= self.fail_after:
            raise RuntimeError("boom: env HTTP down")
        return super()._call(fn, inputs)


hf = HttpFailEnv(scene(), instruction="x", sam_url="")
hf._tool_observe()
hf.fail_after = 2
hf._begin_action("step")
r = hf._execute_primitives([1] * 6, action_id="step")
check("HTTP failure cancels the remaining forwards",
      r["interrupted"] and r["executed"] == 2 and hf.prims.count(1) == 2,
      f"executed {r['executed']}, walked {hf.prims.count(1)} fwd")
check("…with the perception_unavailable class named",
      "perception failed" in r["reason"], r["reason"][:60])
check("…and a critical NavigationEvent on the record",
      any(e["type"] == "perception_unavailable" and e["requires_interrupt"]
          for e in r["events"]))


class EmptyDepthEnv(FakeEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.break_after = 10**9

    def _call(self, fn, inputs):
        out = super()._call(fn, inputs)
        if fn.endswith("observe_egocentric") and \
                self.prims.count(1) >= self.break_after:
            out["depth"] = None
        return out


ed = EmptyDepthEnv(scene(), instruction="x", sam_url="")
ed._tool_observe()
ed.break_after = 1
ed._begin_action("step")
r = ed._execute_primitives([1] * 5, action_id="step")
check("EMPTY depth cancels the remaining forwards",
      r["interrupted"] and ed.prims.count(1) == 1,
      f"walked {ed.prims.count(1)} fwd, reason {r['reason'][:40]}")


class NanDepthEnv(EmptyDepthEnv):
    def _call(self, fn, inputs):
        out = FakeEnv._call(self, fn, inputs)
        if fn.endswith("observe_egocentric") and \
                self.prims.count(1) >= self.break_after:
            nan = np.full((H, W), np.nan, np.float32)
            out["depth"] = depth_wire(nan)
        return out


nd = NanDepthEnv(scene(), instruction="x", sam_url="")
nd._tool_observe()
nd.break_after = 1
nd._begin_action("step")
r = nd._execute_primitives([1] * 5, action_id="step")
check("all-NaN depth cancels the remaining forwards",
      r["interrupted"] and nd.prims.count(1) == 1
      and any(e["type"] == "depth_invalid" for e in r["events"]),
      f"walked {nd.prims.count(1)} fwd")

bx = FakeEnv(scene(), instruction="x", sam_url="")
bx._tool_observe()
_orig_build = dm.build_topdown
_calls = {"n": 0}


def _broken_build(*a, **k):
    _calls["n"] += 1
    if _calls["n"] >= 2:
        raise RuntimeError("boom: build exploded")
    return _orig_build(*a, **k)


dm.build_topdown = _broken_build
try:
    bx._begin_action("step")
    r = bx._execute_primitives([1] * 5, action_id="step")
finally:
    dm.build_topdown = _orig_build
check("a top-down build exception cancels the remaining forwards",
      r["interrupted"] and bx.prims.count(1) <= 2
      and any(e["type"] == "map_update_failed" for e in r["events"]),
      f"walked {bx.prims.count(1)} fwd, reason {r['reason'][:44]}")
check("…but the primitives already executed were still archived",
      r["executed"] >= 1, f"executed {r['executed']}")

# fail-closed at the goto boundary: no post-turn frame → ZERO forwards.
# Depending on the waypoint's angle the failure is caught either mid-turn
# (executor fail-closed interrupt) or at the revalidation gate ("refusing
# to walk blind") — both are correct; walking is not.
gf = HttpFailEnv(scene(), instruction="x", sam_url="")
gf._tool_observe()
gf.fail_after = 0            # every observe from now on fails
res = gf._tool_goto(place=1)
_wh = str(res.info.get("what_happened", "")) + str(res.info)
check("goto refuses to walk blind when the post-turn frame fails",
      gf.prims.count(1) == 0 and ("refusing to walk blind" in _wh
                                  or "fail closed" in _wh
                                  or "perception failed" in _wh),
      f"fwd {gf.prims.count(1)}")

print()
print("── §16.2/§16.7 · the RESIDENT bootstrap ────────────────────────────")
bt_dir = Path(tempfile.mkdtemp(prefix="boot_"))
bt = FakeEnv(scene(corridor=2.0, wall_at=9.0), instruction="x", sam_url="",
             traj_archive=True, live_dir=bt_dir)
boot = bt.bootstrap()
art = boot["artifact"]
check("bootstrap yields candidates from geometry alone (SAM off)",
      len(art["candidates"]) > 0 and len(boot["parts"]) >= 3,
      f"{len(art['candidates'])} candidates, {len(boot['parts'])} parts")
check("the artifact's numbers ARE the resident table's numbers",
      len(art["candidates"]) == len(bt._waypoints)
      and art["candidate_epoch"] == bt._candidate_epoch)
check("frame 0 is fused, archived and counted as a sensor frame",
      art["sensor_frame"] == 1 and bt.amap.updates >= 1
      and (bt_dir / "trajectory" / "manifest.jsonl").exists()
      and json.loads((bt_dir / "trajectory" / "manifest.jsonl")
                     .read_text().splitlines()[0])["action_id"] == "bootstrap")
check("bootstrap.json + both PNGs are on disk",
      (bt_dir / "bootstrap.json").exists()
      and (bt_dir / "bootstrap_current.png").exists()
      and (bt_dir / "bootstrap_map.png").exists())
from eharness.bootstrap import load_bootstrap_artifact  # noqa: E402

_ld = load_bootstrap_artifact(bt_dir)
check("the artifact loader returns text + images for the adapters",
      _ld is not None and len(_ld[1]) == 2 and "goto(n)" in _ld[0],
      f"{len(_ld[1]) if _ld else 0} pngs")
res_g = bt._tool_goto(place=1)
check("the first goto ACCEPTS the bootstrap's number (§16.2 split-brain)",
      res_g.info.get("error") is None
      and res_g.info.get("candidate_epoch") == art["candidate_epoch"],
      f"epoch {res_g.info.get('candidate_epoch')} vs {art['candidate_epoch']}")

print()
print("── bootstrap sweep · ±60° for zero env steps (2026-08-12 ruling) ───")


def _pano_depth_wire(depth: np.ndarray) -> str:
    from PIL import Image as _PI
    u16 = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    buf = BytesIO()
    _PI.fromarray(u16, mode="I;16").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class SweepEnv(FakeEnv):
    def _call2(self, fn, inputs, config=None):
        if fn.endswith("observe_panorama"):
            n = int((config or {}).get("n_views", 24))
            views = []
            for k in range(n):
                views.append({"heading_deg": k * 360.0 / n,
                              "rgb_base64": rgb_wire(),
                              "depth_raw_base64": _pano_depth_wire(
                                  self.scene_depth)})
            return {"views": views}
        raise AssertionError(f"unexpected env call {fn}")


sw_dir = Path(tempfile.mkdtemp(prefix="sweep_"))
sw = SweepEnv(scene(corridor=2.0, wall_at=9.0), instruction="x", sam_url="",
              traj_archive=False, live_dir=sw_dir)
boot_sw = sw.bootstrap()
known_sweep = int((np.abs(sw.amap.logodds) > 0.5).sum())
nf = FakeEnv(scene(corridor=2.0, wall_at=9.0), instruction="x", sam_url="",
             traj_archive=False)
nf.bootstrap()
known_frontal = int((np.abs(nf.amap.logodds) > 0.5).sum())
check("the sweep costs ZERO env steps and no primitives",
      sw.steps_taken == 0 and sw.prims == [], f"steps {sw.steps_taken}")
check("nine headings fused (0, ±15…±60)",
      boot_sw["artifact"]["sweep"]["views"] == 9,
      str(boot_sw["artifact"]["sweep"]))
check("the wings genuinely widen frame-0 (vs frontal-only boot)",
      known_sweep > 1.5 * known_frontal,
      f"{known_sweep} vs {known_frontal} known cells")
check("a sweepless env degrades to the frontal boot, never dies",
      nf.bootstrap()["artifact"]["sweep"]["views"] == 0)

sam_sw = FakeSAM()
sw2 = SweepEnv(scene(panels=((0.5, 1.5, 3.0),)), instruction="find the bar",
               sam_url="", traj_archive=False)
sw2.landmarks = sam_sw
sw2.phrases = ["bar counter"]
boot2 = sw2.bootstrap()
check("SAM runs on the ±30/±60 wings (4 views), bearings folded",
      boot2["artifact"]["sweep"]["sam_views"] == 4
      and sw2.register.closest("bar counter") is not None,
      str(boot2["artifact"]["sweep"]["seen"])[:60])

print()
print("── §21.2 · the atomic post-sweep finalize ──────────────────────────")
a21 = boot_sw["artifact"]
check("one publish: artifact / proposal / map pixels share ONE map_version",
      a21["map_version"] == a21["proposal_map_version"] == sw.amap.updates,
      f"v{a21['map_version']}")
check("trust provenance is on the record and the first frame IS trusted",
      a21["trust"]["sweep_trust"] and a21["trust"]["trusted"])
check("known-cells provenance separates frontal / sweep / published",
      a21["known_cells"]["before_sweep"] < a21["known_cells"]["after_sweep"]
      == a21["known_cells"]["published"],
      str(a21["known_cells"]))
check("depth headings enumerate the fan; SAM headings are explicit",
      a21["sweep"]["depth_headings"] == [-60, -45, -30, -15, 0, 15, 30, 45, 60]
      and set(boot2["artifact"]["sweep"]["sam_headings"]) == {-60, -30, 30, 60},
      str(a21["sweep"]["depth_headings"]))
check("fresh candidates carry stable track ids from birth "
      "(remembered ones are map-identity by construction)",
      all(c.get("track_id") for c in a21["candidates"]
          if c.get("kind") != "remembered")
      and any(c.get("track_id") for c in a21["candidates"]),
      str([(c.get("kind"), c.get("track_id")) for c in a21["candidates"]]))
_menu0 = [(c.get("track_id"), c["angle_deg"]) for c in a21["candidates"]]
_re = sw._tool_observe(_internal=True)
_menu1 = [(w.extras.get("track_id"), round(math.degrees(w.angle), 1))
          for w in sw._waypoints]
check("first-frame parity: a same-pose re-observe repeats the SAME menu "
      "(no A/D needed to wake the third point)", _menu0 == _menu1,
      f"{_menu0} vs {_menu1}")
check("the sweep pose is untouched: nine fusions, zero px/py/theta",
      (sw.amap.px, sw.amap.py, sw.amap.theta) == (0.0, 0.0, 0.0))

print()
print("── §16.3 · STOP batches walk through the executor ──────────────────")
sb = FakeEnv(scene(corridor=2.0, wall_at=9.0), instruction="x", sam_url="",
             traj_archive=False)
sb._tool_observe()
f0, p0 = sb.amap.updates, len(sb.prims)
res_s = sb._tool_step(actions=[1, 1, 0])
check("each pre-STOP forward is SENSED (no blind base path)",
      sb.amap.updates - f0 >= 2, f"{sb.amap.updates - f0} fusions for 2 fwd")
check("…and the STOP still executes through the base gate",
      sb.episode_over and sb.end_reason == "stop_called"
      and sb.prims[p0:] == [1, 1, 0], f"prims {sb.prims[p0:]}")
sf = HttpFailEnv(scene(), instruction="x", sam_url="")
sf._tool_observe()
sf.fail_after = 1
res_f = sf._tool_step(actions=[1, 1, 1, 0])
check("a failure mid-walk means the STOP is NOT executed (§16.8)",
      not sf.episode_over and res_f.info.get("stop_not_executed")
      and 0 not in sf.prims,
      f"episode_over={sf.episode_over}, wh={str(res_f.info.get('what_happened'))[:40]}")

print()
print("── review P0 · the interrupt plane resets per action ───────────────")
sp = CollideEnv(scene(), instruction="x", sam_url="")
sp._tool_observe()
sp._begin_action("step")
r = sp._execute_primitives([1] * 8, action_id="step")
check("(setup) the first leg really was interrupted", r["interrupted"])
res = sp._tool_face(direction=8)      # sub-15°: the notches==0 yaw path
wh = str(res.info.get("what_happened", ""))
check("a stale interrupt never leaks into the next sub-15° action",
      "collision" not in wh and res.info.get("turned_deg") == 8.0,
      f"what_happened={wh[:40]!r}, turned {res.info.get('turned_deg')}")

print()
print("── §14.8 · SAM cadence rides SENSOR frames, not the archive ────────")
sam2 = FakeSAM()
cs = FakeEnv(scene(panels=((0.5, 1.5, 3.0),)), instruction="find the bar",
             sam_url="", traj_archive=False, sam_intermediate_every=1)
cs.landmarks = sam2
cs.phrases = ["bar counter"]
cs._tool_observe()
c0 = sam2.calls
cs._begin_action("step")
cs._execute_primitives([1, 1], action_id="step")
check("cadence=1 with the archive OFF still runs SAM per primitive",
      sam2.calls - c0 == 2, f"{sam2.calls - c0} calls for 2 primitives")
sam3 = FakeSAM()
ev = FakeEnv(scene(panels=((0.5, 1.5, 3.0),)), instruction="find the bar",
             sam_url="", traj_archive=False, sam_intermediate_every=0)
ev.landmarks = sam3
ev.phrases = ["bar counter"]
ev._tool_observe()
c0 = sam3.calls
ev._begin_action("face")
ev._execute_yaw(30.0, action_id="face")
check("event-driven: the first post-turn frame runs the detector",
      sam3.calls > c0, f"{sam3.calls - c0} call(s) across a 30° face")
check("…and every skip is logged with a reason",
      sum(ev._sam_stats["skips"].values()) + ev._sam_stats["calls"] >= 2,
      f"calls {ev._sam_stats['calls']}, skips {ev._sam_stats['skips']}")

print()
print("── §14.7 · events surface at the endpoint ──────────────────────────")
nb = FakeEnv(scene(), instruction="x", sam_url="")
nb._tool_observe()
nb._begin_action("step")
r = nb._execute_primitives([1, 1], action_id="step")
check("a clean leg carries an (possibly empty) event list",
      isinstance(r.get("events"), list))
r2 = ts7._execute_primitives([1] * 3, action_id="step")  # CollideEnv again
check("the collision reflex speaks NavigationEvent",
      any(e["type"] == "collision" and e["severity"] == "critical"
          for e in r2.get("events", [])),
      f"events {[e['type'] for e in r2.get('events', [])]}")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("toolset contract: all passed")
