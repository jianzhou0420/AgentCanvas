"""§10.7 behaviour table — the scenes the review reproduced the failures in.

Each scene is analytic (no noise, no simulator), so a failure here is a
geometry bug, never flakiness. These are the exact configurations from the
third review: the symmetric corridor that used to come out −15°, the central
obstacle that produced one ±45°/1.1 m shuffle instead of a way around each
side, and the resolution pair whose branch semantics used to flip.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eharness import depthmap as dm  # noqa: E402
from eharness.depthmap import FREE, build_topdown, propose  # noqa: E402

FAILS: list[str] = []
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def scene(panels=(), wall_at=None, corridor=None, px=512) -> np.ndarray:
    f = (px / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(px, dtype=np.float32),
                         np.arange(px, dtype=np.float32))
    dx, dy = (us - px / 2.0) / f, -(vs - px / 2.0) / f
    depth = np.full((px, px), 25.0, np.float32)
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


def cands(depth: np.ndarray) -> list[dm.Waypoint]:
    return propose(build_topdown(depth, scale_m=1.0))


def fmt(ws) -> str:
    return "; ".join(f"{math.degrees(w.angle):+.0f}°/{w.distance:.1f}m"
                     f"(v{w.verified_m:.1f})" for w in ws) or "none"


print("── symmetric scenes must come out straight ─────────────────────────")
ws = cands(scene())
best = min(ws, key=lambda w: abs(w.angle), default=None)
check("open expanse: a main candidate within 5° of straight ahead",
      best is not None and abs(math.degrees(best.angle)) < 5.0, fmt(ws))

ws = cands(scene(corridor=1.0, wall_at=6.0))
best = min(ws, key=lambda w: abs(w.angle), default=None)
check("symmetric corridor: |angle| < 5° (was −15°)",
      best is not None and abs(math.degrees(best.angle)) < 5.0, fmt(ws))

print()
print("── a central obstacle forks the way (§10.4) ────────────────────────")
for z in (1.2, 1.6, 2.0, 2.4, 2.8):
    ws = cands(scene(panels=[(-0.4, 0.4, z)]))
    signs = {int(np.sign(w.angle)) for w in ws if abs(w.angle) > 1e-3}
    check(f"obstacle at {z} m: one candidate each side",
          {-1, 1} <= signs, fmt(ws))

ws = cands(scene(panels=[(-4.0, -0.9, 2.5), (0.0, 0.6, 2.5)]))
signs = {int(np.sign(w.angle)) for w in ws if abs(w.angle) > 1e-3}
check("narrow left / wide right: BOTH branches kept",
      {-1, 1} <= signs, fmt(ws))
if {-1, 1} <= signs:
    left = next(w for w in ws if w.angle > 0)
    right = next(w for w in ws if w.angle < 0)
    check("…and each branch reports its own clearance",
          left.clearance > 0 and right.clearance > 0,
          f"left {2 * left.clearance:.1f} m wide, "
          f"right {2 * right.clearance:.1f} m wide")

print()
print("── the audit's off-axis island (shadow not straddling 0°) ──────────")
# An obstacle whose angular shadow sits entirely on one side of straight
# ahead: both ways around it share a bearing sign, so a side tag derived
# from sign(bearing) collapsed them. Sides must come from the FLANK. The
# tight flank near the FOV edge may honestly be too short to land a
# candidate — then the STRUCTURE must still name it (telemetry says "turn
# to face it"), never silently offer one side only.
d = scene(panels=[(0.6, 1.6, 2.0)])
td_off = build_topdown(d, scale_m=1.0)
import eharness.depthmap as _dm
prof_o, neck_o = _dm.passable_profile(td_off)
ops = _dm.openings(td_off, prof_o, neck_o)
op_sides = {o.side for o in ops if o.island >= 0}
check("both flanks exist as openings, tagged by flank",
      {-1, 1} <= op_sides,
      "; ".join(f"{math.degrees(o.centre):+.0f}°/side{o.side}/r{o.reach:.1f}"
                for o in ops))
ws_off = propose(td_off)
covered = {o.side for o in ops if o.island >= 0
           and any(abs(math.degrees(w.angle - o.centre)) < 10 for w in ws_off)}
uncovered = ({-1, 1} - covered)
check("every flank is either a candidate or an explicit unwalked opening",
      not uncovered or all(
          any(o.side == s and o.island >= 0 for o in ops) for s in uncovered),
      f"candidates {fmt(ws_off)}; uncovered flanks {sorted(uncovered)}")

print()
print("── resolution changes numbers, never branch semantics ──────────────")
by_res = {}
for px in (128, 256, 512):
    ws = cands(scene(panels=[(-0.4, 0.4, 2.0)], px=px))
    by_res[px] = sorted(int(np.sign(w.angle)) for w in ws if abs(w.angle) > 1e-3)
check("128² / 256² / 512² agree on the branch structure",
      by_res[128] == by_res[256] == by_res[512],
      " vs ".join(f"{k}:{v}" for k, v in by_res.items()))

print()
print("── §2.4 · the bar-occlusion scene: honest UNKNOWN + potential ──────")
# A counter to the front-right (under-gap 0.3 m, like real furniture): the
# floor behind it is glimpsed through the gap but no sightline ever opened
# the region. It must stay non-FREE, register as POTENTIAL, and the walkable
# candidates must stay on verified floor — staging points, not trespassers.
bar = scene(panels=[(0.5, 2.0, 2.5)])
td_bar = build_topdown(bar, scale_m=1.0)
behind = []
for r in (3.6, 4.2, 4.8):
    b = math.atan2(-1.2, r)      # through the bar's angular shadow
    i, j = td_bar.cell_of(r * math.sin(b), r * math.cos(b))
    behind.append(int(td_bar.grid[i, j]))
check("ground behind the occluder is never painted FREE",
      all(g != dm.FREE for g in behind), f"states {behind} (1=FREE)")
check("…but the glimpsed floor is remembered as POTENTIAL",
      td_bar.potential is not None and bool(td_bar.potential.any()),
      f"{int(td_bar.potential.sum())} cells")
regions = dm.potential_regions(td_bar)
check("potential_regions reports it on the correct side",
      bool(regions) and regions[0]["bearing_deg"] < 0,
      str(regions[:2]))
ws_bar = propose(td_bar)
check("candidates still land only on verified FREE (staging, not trespass)",
      bool(ws_bar) and all(
          td_bar.grid[td_bar.cell_of(w.x_left, w.y_fwd)] == dm.FREE
          for w in ws_bar), fmt(ws_bar))

# the reveal: once REAL evidence covers the region, the guess retires
am = dm.AnchorMap()
am.fuse(td_bar)
pot_before = float(am.potential.sum())
am.fuse(build_topdown(scene(), scale_m=1.0))     # the region seen properly
pot_after = float(am.potential.sum())
# Structural fusion only reaches FUSE_MAX_M, so ONE proper look retires the
# near guesses and honestly leaves the far ones until the body gets closer.
check("real evidence retires the potential hint (reveal → resolve)",
      pot_before > 0 and pot_after < pot_before * 0.5,
      f"votes {pot_before:.0f} → {pot_after:.0f}")

print()
print("── every executable candidate is walkable, everywhere ──────────────")
scenes = {
    "open": scene(), "corridor": scene(corridor=1.0, wall_at=6.0),
    "obstacle": scene(panels=[(-0.4, 0.4, 2.0)]),
    "asym": scene(panels=[(-4.0, -0.9, 2.5), (0.0, 0.6, 2.5)]),
}
for name, d in scenes.items():
    td = build_topdown(d, scale_m=1.0)
    for k, w in enumerate(propose(td), 1):
        li, lj = td.cell_of(w.verified_m * math.sin(w.angle),
                            w.verified_m * math.cos(w.angle))
        ok_land = td.inside(li, lj) and td.grid[li, lj] == FREE
        r, ok_prefix = 0.55, True
        while r < w.stride_m - 1e-9:
            r = min(r + dm.CELL_M, w.stride_m)
            pi, pj = td.cell_of(r * math.sin(w.angle), r * math.cos(w.angle))
            if not td.inside(pi, pj) or td.grid[pi, pj] != FREE:
                ok_prefix = False
                break
        check(f"{name} #{k}: FREE landing + FREE prefix + stride ≤ verified",
              ok_land and ok_prefix and w.stride_m <= w.verified_m + 1e-6,
              f"{math.degrees(w.angle):+.0f}° stride {w.stride_m:.1f} "
              f"verified {w.verified_m:.1f}")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("dwp scenes: all passed")
