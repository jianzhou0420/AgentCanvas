"""§21 offline regressions — candidate identity, runway, visibility, turns.

Run:  python -m eharness.tests.test_registry.py

Synthetic depth only (same analytic fabric as test_depthmap); every case
encodes a §21 (2026-08-15 audit) contract:
  §21.5  pure turns register READ-ONLY (no pseudo-translation), and a
         ±60° round trip returns the pose exactly;
  §21.2  same-pose sweep provenance keeps `trusted` without a match score,
         and expires on the first real translation;
  §21.6  the registry pins a physical exit to ONE anchor circle — matched
         re-proposals snap back, occupied centres retire, unverifiable
         centres suspend without moving;
  §21.9  ordinary landings keep MIN_FORWARD_RUNWAY_M of verified prefix
         past the circle, or are demoted to short_verified_gateway;
  §21.8  the shared projection gate: callable ⇒ visible in the RGB, and
         annotate_rgb never edge-clamps an invisible point;
  §21.7  the global panel is anchor-fixed: turning the head moves no wall.
"""
from __future__ import annotations

import copy
import math
import sys

import numpy as np

from eharness.depthmap import (
    CELL_M,
    FOV_DEG,
    MIN_FORWARD_RUNWAY_M,
    AnchorMap,
    Waypoint,
    WaypointRegistry,
    annotate_rgb,
    build_topdown,
    is_waypoint_visible,
    project_waypoint_to_rgb,
    propose,
)

FAILS: list[str] = []
W = H = 256
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def render(corridor_half_width: float | None = None,
           wall_at: float | None = None,
           hfov_deg: float = 90.0) -> np.ndarray:
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    dx = (us - W / 2.0) / f
    dy = -(vs - H / 2.0) / f
    depth = np.full((H, W), 25.0, dtype=np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], (-CAM_H / dy[below]))
    if wall_at is not None:
        depth = np.minimum(depth, wall_at * np.ones_like(depth))
    if corridor_half_width is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dx) > 1e-6,
                         corridor_half_width / np.abs(dx), np.inf)
        depth = np.minimum(depth, np.where(np.isfinite(t), t, 25.0))
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


print("── §21.5 · pure turns are pose-read-only ───────────────────────────")
td = build_topdown(render(corridor_half_width=1.6, wall_at=6.0), scale_m=1.0)
am = AnchorMap()
for _ in range(4):
    am.fuse(td)
pose0 = (am.px, am.py, am.theta)
fixes0 = am.fixes
sc = am.register(td, translation_expected=False)
check("translation_expected=False measures a score but moves NOTHING",
      (am.px, am.py, am.theta) == pose0 and am.fixes == fixes0
      and am.last_fix == (0.0, 0.0, 0.0), f"score {sc:.2f}")
for _ in range(4):
    am.odometry(math.radians(15), 0.0)
for _ in range(8):
    am.odometry(math.radians(-15), 0.0)
for _ in range(4):
    am.odometry(math.radians(15), 0.0)
check("a 0°→+60°→−60°→0° round trip returns the pose EXACTLY",
      abs(am.px - pose0[0]) < 1e-12 and abs(am.py - pose0[1]) < 1e-12
      and abs(am.theta - pose0[2]) < 1e-9,
      f"theta residual {am.theta - pose0[2]:.2e}")

print()
print("── §21.2 · sweep trust provenance ──────────────────────────────────")
am2 = AnchorMap()
for _ in range(10):
    am2.fuse(td)
check("ten same-pose fusions WITHOUT provenance are not trusted "
      "(the first-frame bug reproduced)", not am2.trusted,
      f"updates {am2.updates}, score {am2.last_score:.2f}")
am2.sweep_trust = True
check("…and the sweep provenance alone restores trust", am2.trusted)
am2.odometry(0.0, 0.25)
check("the first real translation expires the provenance",
      not am2.sweep_trust and not am2.trusted)
am2.odometry(math.radians(15), 0.0)   # a later pure turn must not resurrect it
check("a pure turn neither restores nor extends it", not am2.sweep_trust)

print()
print("── §21.9 · forward runway ──────────────────────────────────────────")
td_open = build_topdown(render(corridor_half_width=4.0), scale_m=1.0)
ws_open = propose(td_open)
ok_all = bool(ws_open)
for w in ws_open:
    if w.extras.get("short_verified_gateway") or w.kind == "gateway":
        continue
    if w.continuation_m < MIN_FORWARD_RUNWAY_M - 1e-6:
        ok_all = False
check("open room: every ordinary landing keeps ≥ runway of verified prefix",
      ok_all, "; ".join(f"{w.distance:.1f}+{w.continuation_m:.1f}m"
                        for w in ws_open))
check("open room: the landing stays mid-range, not at the far edge",
      all(w.distance <= w.verified_m - MIN_FORWARD_RUNWAY_M + 1e-6
          for w in ws_open if not w.extras.get("short_verified_gateway")
          and w.kind != "gateway"))
td_wall = build_topdown(render(corridor_half_width=1.2, wall_at=1.9),
                        scale_m=1.0)
ws_wall = propose(td_wall)
check("wall-end: candidates exist, and any without runway is DEMOTED "
      "and says so",
      all(w.continuation_m >= MIN_FORWARD_RUNWAY_M - 1e-6
          or w.extras.get("short_verified_gateway")
          or w.kind == "gateway" or w.extras.get("boxed_in")
          for w in ws_wall),
      "; ".join(f"{w.distance:.1f}m cont {w.continuation_m:.1f} "
                f"{'SVG' if w.extras.get('short_verified_gateway') else ''}"
                for w in ws_wall))
_svg = [w for w in ws_wall if w.extras.get("short_verified_gateway")]
check("a demoted landing carries the staging wording",
      all("look again" in w.note for w in _svg),
      _svg[0].note[:60] if _svg else "none demoted (runway afforded)")

print()
print("── §21.8 · the shared projection gate ──────────────────────────────")


def _wp(x: float, y: float) -> Waypoint:
    return Waypoint(angle=math.atan2(x, y), distance=math.hypot(x, y),
                    clearance=1.0, kind="opening", x_left=x, y_fwd=y)


w_mid = _wp(0.0, 3.0)
w_near = _wp(0.0, 0.5)          # floor point projects below a 90° frame
w_behind = _wp(0.0, -1.0)
w_edge = _wp(3.2, 3.0)          # ~47° off-axis: outside a 45° half-FOV
check("3 m ahead is visible", is_waypoint_visible(w_mid, -CAM_H, W, H))
check("0.5 m underfoot projects below the frame — NOT visible",
      not is_waypoint_visible(w_near, -CAM_H, W, H))
check("behind the camera is not visible",
      not is_waypoint_visible(w_behind, -CAM_H, W, H))
check("the FOV edge is not visible",
      not is_waypoint_visible(w_edge, -CAM_H, W, H))
u, v, r, vis = project_waypoint_to_rgb(w_mid, -CAM_H, W, H)
check("the projection lands inside the frame with its number",
      vis and 0 <= u < W and 0 <= v - r - 15 and v + r < H,
      f"u {u:.0f} v {v:.0f} r {r}")
from io import BytesIO  # noqa: E402

from PIL import Image  # noqa: E402

_blank = BytesIO()
Image.new("RGB", (W, H), (7, 7, 7)).save(_blank, format="PNG")
_png0 = _blank.getvalue()
_png_inv = annotate_rgb(_png0, [w_near, w_behind, w_edge], -CAM_H)
check("annotate_rgb draws NOTHING for invisible points (no edge clamp)",
      np.array_equal(np.asarray(Image.open(BytesIO(_png_inv))),
                     np.asarray(Image.open(BytesIO(_png0)))))
_png_two = annotate_rgb(_png0, [w_near, w_mid], -CAM_H)
check("…while a visible point later in the list keeps its own number slot",
      not np.array_equal(np.asarray(Image.open(BytesIO(_png_two))),
                         np.asarray(Image.open(BytesIO(_png0)))))

print()
print("── §21.6 · the registry pins identity ──────────────────────────────")
am3 = AnchorMap()
for _ in range(3):
    am3.fuse(td_open)
reg = WaypointRegistry()
ws1 = propose(td_open)
out1 = reg.reconcile(am3, td_open, [copy.deepcopy(w) for w in ws1], epoch=1)
tids1 = [w.extras.get("track_id") for w in out1]
check("fresh candidates all receive track ids",
      len(out1) == len(ws1) and all(tids1), str(tids1))
out2 = reg.reconcile(am3, td_open, [copy.deepcopy(w) for w in ws1], epoch=2)
check("an identical re-proposal keeps the SAME ids at the SAME spots",
      [w.extras.get("track_id") for w in out2] == tids1
      and all(abs(a.x_left - b.x_left) < 1e-9 for a, b in zip(out1, out2)))
_shift = copy.deepcopy(ws1[0])
_shift.x_left += 0.30
_shift.angle = math.atan2(_shift.x_left, _shift.y_fwd)
_shift.distance = math.hypot(_shift.x_left, _shift.y_fwd)
out3 = reg.reconcile(am3, td_open, [_shift], epoch=3)
check("a 0.30 m-drifted re-sighting SNAPS BACK to the track's circle",
      len(out3) == 1 and out3[0].extras.get("track_id") == tids1[0]
      and abs(out3[0].x_left - out1[0].x_left) <= CELL_M + 1e-9,
      f"snapped to {out3[0].x_left:.2f} vs track {out1[0].x_left:.2f}")
t0 = reg.tracks[tids1[0]]
_i, _j = am3.cells(t0.ax, t0.ay)
am3.logodds[int(_i), int(_j)] = 2.0
reg.reconcile(am3, td_open, [], epoch=4)
check("a centre confirmed OCCUPIED retires the track (never slides it)",
      t0.retired == "occupied", t0.retired)
# an unverifiable centre SUSPENDS: same fresh waypoint, but the current
# frame is wall-limited so the far centre is UNKNOWN there
reg2 = WaypointRegistry()
w_far = _wp(0.0, 3.5)
reg2.reconcile(am3, td_open, [copy.deepcopy(w_far)], epoch=1)
_wf2 = copy.deepcopy(w_far)
_wf2.x_left += 0.3
out_s = reg2.reconcile(am3, td_wall, [_wf2], epoch=2)
t_far = next(iter(reg2.tracks.values()))
check("a centre the CURRENT frame cannot verify is suspended and off the "
      "menu — the circle does not move",
      out_s == [] and t_far.suspended and not t_far.retired,
      f"suspended={t_far.suspended} retired='{t_far.retired}'")

print()
print("── §21.7 · the global panel is anchor-fixed ────────────────────────")
am4 = AnchorMap()
for _ in range(4):
    am4.fuse(td)


def _occ_mask(png: bytes) -> np.ndarray:
    arr = np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(int)
    return (np.abs(arr - np.array([176, 68, 62])).sum(axis=2) < 30)


png_a = am4.render(caption="")
for _ in range(6):
    am4.odometry(math.radians(15), 0.0)
png_b = am4.render(caption="")
ma, mb = _occ_mask(png_a), _occ_mask(png_b)
check("turning the head 90° moves NO wall pixel in the global panel",
      ma.shape == mb.shape and bool((ma == mb).all()),
      f"wall px {int(ma.sum())} vs {int(mb.sum())}")
# stage3 P0-1: the dual-panel composite is RETIRED — the model's IMAGE 2 is
# the single anchor-fixed render, and remembered places draw as DASHED
# circles (distinguishable from a solid fresh circle at the same spot)
check("render_composite is gone from the map surface",
      not hasattr(am4, "render_composite"))
w_solid = _wp(0.0, 2.0)
w_mem = _wp(0.0, 2.0)
w_mem.kind = "remembered"
png_solid = am4.render([w_solid], caption="")
png_mem = am4.render([w_mem], caption="")
check("a remembered place renders differently (dashed) at the same spot",
      png_solid != png_mem)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("§21 registry/runway/visibility/turn regressions: all passed")
