"""Offline self-test for the depth geometry organ — synthetic depth, no env.

Run:  python -m eharness.test_depthmap

Every scene is rendered analytically, so a regression shows up as a wrong metre
rather than a flaky simulator. The cases encode the failures that actually cost
debugging time on the live env (see memory/habitat-depth-geometry-contract.md).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from eharness.depthmap import (
    CELL_M,
    FREE,
    OCCUPIED,
    MAX_WAYPOINT_M,
    MIN_WAYPOINT_M,
    RANGE_CAP_M,
    ROBOT_RADIUS_M,
    OPEN,
    _stride_cap,
    build_topdown,
    passable_range,
    project_mask,
    propose,
    surroundings_sentence,
    to_metres,
)

FAILS: list[str] = []
W = H = 256
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name + (f"  [{detail}]" if detail else ""))


def render(corridor_half_width: float | None = None, wall_at: float | None = None,
           hfov_deg: float = 90.0) -> np.ndarray:
    """Analytic depth for a flat floor, optional side walls and a front wall."""
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx = (us - W / 2.0) / f                       # right, per unit forward
    dy = -(vs - H / 2.0) / f                      # up, per unit forward
    depth = np.full((H, W), 25.0, dtype=np.float32)

    below = dy < -1e-6                            # floor: y = -CAM_H
    depth[below] = np.minimum(depth[below], (-CAM_H / dy[below]))
    if wall_at is not None:                       # plane z = wall_at
        depth = np.minimum(depth, wall_at * np.ones_like(depth))
    if corridor_half_width is not None:           # planes x = ±hw
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dx) > 1e-6, corridor_half_width / np.abs(dx), np.inf)
        depth = np.minimum(depth, np.where(np.isfinite(t), t, 25.0))
    return np.clip(depth, 0.0, 25.0).astype(np.float32)


# ── units: the trap that silently shrinks the world 10× ──────────────────
metric = np.array([[3.0, 7.5], [1.0, 9.0]], dtype=np.float32)
m1, norm1 = to_metres(metric)
check("units: raw metres pass through unscaled", not norm1 and abs(m1.max() - 9.0) < 1e-4)
m2, norm2 = to_metres(metric / 10.0)
check("units: normalized [0,1] is scaled to 0-10 m", norm2 and abs(m2.max() - 9.0) < 1e-4)

# …and why the guess is not good enough once habitat's 10 m clip comes off.
# A body with its nose against a wall returns a frame whose every pixel is
# under a metre; the self-detector reads that as normalized and multiplies by
# ten, putting a wall four metres away that is really 40 cm from the shins.
nose = np.array([[0.4, 0.9], [0.35, 0.8]], dtype=np.float32)   # true metres
_, guessed = to_metres(nose)
check("units: the GUESS misreads an all-close-range metric frame", guessed,
      "this is the failure the declared scale exists to remove")
told, told_norm = to_metres(nose, 1.0)
check("units: a declared scale of 1.0 keeps the metres it was given",
      not told_norm and abs(told.max() - 0.9) < 1e-6, f"{told.max():.2f} m")
told10, _ = to_metres(metric / 100.0, 100.0)
check("units: a declared normalized range is honoured, not assumed to be 10",
      abs(told10.max() - 9.0) < 1e-4, f"{told10.max():.2f} m")

# The far-validity ceiling used to be a hard 30 m, which is invisible while
# habitat clips at 10 and amputates exactly the long sightlines that removing
# the clip is FOR. Declared scale ⇒ no ceiling to trip over.
far = np.full((64, 64), 40.0, dtype=np.float32)
far[:32] = 2.0
td_far = build_topdown(far, range_cap_m=6.0, scale_m=1.0)
check("a 40 m sightline is not thrown away as an implausible reading",
      bool((td_far.free_range > 0).any()),
      f"deepest bearing {td_far.free_range.max():.1f} m")

# ── an empty corridor with a wall ahead ──────────────────────────────────
td = build_topdown(render(corridor_half_width=1.2, wall_at=4.0))
check("floor is self-calibrated to the camera height",
      abs(td.floor_y + CAM_H) < 0.12, f"{td.floor_y:.2f}")
check("front wall is measured at the right range",
      abs(td.ahead_m() - 4.0) < 0.35, f"{td.ahead_m():.2f} m")
check("occlusion behind the wall stays UNKNOWN, never FREE",
      td.grid[int(4.6 / CELL_M), td.n_lat // 2] != FREE)
check("the corridor floor in front of the wall is FREE",
      td.grid[int(2.0 / CELL_M), td.n_lat // 2] == FREE)
check("side walls register as obstacles",
      (td.grid == OCCUPIED).sum() > 20, str(int((td.grid == OCCUPIED).sum())))

wps = propose(td)
check("a walkable corridor yields candidates", len(wps) >= 1, f"{len(wps)}")
check("no candidate is closer than the minimum useful hop",
      all(w.distance >= 1.0 - 1e-6 for w in wps))
check("no candidate exceeds the depth range cap",
      all(w.distance <= RANGE_CAP_M + 1e-6 for w in wps))
check("no candidate is offered in a space narrower than the body",
      all(w.clearance >= ROBOT_RADIUS_M for w in wps))
check("no candidate is placed past the wall",
      all(w.y_fwd < 4.05 for w in wps), f"max y {max((w.y_fwd for w in wps), default=0):.2f}")

# ── open room: the point sits MID-RANGE of the verified reach (§2.1) ─────
# Not at the far edge of what happens to be visible: a circle on the horizon
# reads as a macro destination and buries the near/mid-range decision.
td_open = build_topdown(render(corridor_half_width=4.0))
wps_open = propose(td_open)
best_open = max(wps_open, key=lambda w: w.verified_m, default=None)
check("an open room lands the point near HALF its verified reach",
      best_open is not None
      and abs(best_open.distance - 0.5 * best_open.verified_m) < 0.6,
      (f"point {best_open.distance:.1f} m of verified "
       f"{best_open.verified_m:.1f} m" if best_open else "none"))
check("…and never retreats underfoot on shallow reaches",
      all(w.distance >= (0.5 if w.kind == 'gateway' else 1.0) - 1e-6
          for w in wps_open))

# ── a tight slot: strides must be shortened, not banned ──────────────────
check("stride cap: a roomy corridor gets the full pacing allowance",
      abs(_stride_cap(0.8) - MAX_WAYPOINT_M) < 1e-6, f"{_stride_cap(0.8)}")
check("stride cap: no hop ever exceeds the pacing ceiling",
      all(_stride_cap(c) <= MAX_WAYPOINT_M + 1e-6
          for c in (0.1, 0.25, 0.35, 0.5, 0.8, 2.0)))
check("stride cap: a squeeze shortens the leap", _stride_cap(0.28) <= 2.5)
check("stride cap: monotone in clearance",
      _stride_cap(0.2) <= _stride_cap(0.4) <= _stride_cap(0.8))
td_tight = build_topdown(render(corridor_half_width=0.45, wall_at=8.0))
for w in propose(td_tight):
    # The STRIDE is what a tight slot has to shorten — how far step_hightolow
    # may walk blind. The point itself may still mark the far end of the slot;
    # capping that as well is what pinned every proposal at the ceiling and made
    # it look identical step after step.
    check(f"tight slot keeps the {w.kind} STRIDE short",
          w.extras.get("stride_m", w.distance) <= 2.6,
          f"stride {w.extras.get('stride_m')} m toward a point at {w.distance:.1f} m")
    check(f"…and the {w.kind} point is still a place, not underfoot",
          w.distance >= MIN_WAYPOINT_M, f"{w.distance:.1f} m")

# ── a landmark layer is not bounded by the occupancy grid ────────────────
# Measured on R2R ep 7 with the un-clipped rig: SAM found doors at 11.7, 13.7
# and 15.4 m and project_mask dropped all three at `z < RANGE_CAP_M`, while the
# map kept "chairs at 2.8 m". Remembering a far door cannot walk the body into
# anything; a wrong FREE cell can. The two limits are not the same limit.
far_scene = np.full((64, 64), 13.0, dtype=np.float32)      # a door 13 m off
far_mask = np.zeros((64, 64), dtype=bool)
far_mask[20:40, 20:40] = True
_, ys_far = project_mask(far_mask, far_scene, floor_y=-1.5, scale_m=1.0)
check("a 13 m landmark survives projection", ys_far.size > 0,
      f"{ys_far.size} points" if ys_far.size else "all dropped")
check("…and it is remembered where it actually is",
      bool(ys_far.size) and abs(float(ys_far.mean()) - 13.0) < 0.6,
      f"{float(ys_far.mean()):.1f} m" if ys_far.size else "n/a")
_, ys_capped = project_mask(far_mask, far_scene, floor_y=-1.5, scale_m=1.0,
                            max_m=RANGE_CAP_M)
check("the old occupancy-grid cut is what threw it away", ys_capped.size == 0,
      f"{ys_capped.size} survived")

# …and the spill guard still holds, because it was never the absolute cut. A
# mask that leaks onto a wall far behind its object must still be trimmed.
spill = np.full((64, 64), 13.0, dtype=np.float32)
spill[20:26, 20:40] = 30.0                       # a fifth of the mask leaks
_, ys_spill = project_mask(far_mask, spill, floor_y=-1.5, scale_m=1.0)
check("mask spill onto a far wall is still trimmed",
      bool(ys_spill.size) and float(ys_spill.max()) < 20.0,
      f"deepest kept {float(ys_spill.max()):.1f} m" if ys_spill.size else "n/a")

# ── FREE means "I saw the floor", OPEN means "the ray got through" ───────
# The grid reaches 12 m but ground samples thin as 1/d³, so past a few metres a
# carved cell only proves a clear sightline. A sunken pool or a stairwell sits
# below the obstacle band and stops no ray — with FREE carrying both claims,
# the drop reads as walkable floor.
def _hall(px=512, cam_h=1.55, corridor=2.0, wall_at=None, hfov=90.0):
    """A corridor with no far wall: the floor runs on past where it can be seen."""
    f = (px / 2.0) / math.tan(math.radians(hfov) / 2.0)
    us, vs = np.meshgrid(np.arange(px, dtype=np.float32),
                         np.arange(px, dtype=np.float32))
    dx, dy = (us - px / 2.0) / f, -(vs - px / 2.0) / f
    d = np.full((px, px), 60.0, np.float32)
    below = dy < -1e-6
    d[below] = np.minimum(d[below], -cam_h / dy[below])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(dx) > 1e-6, corridor / np.abs(dx), np.inf)
    d = np.minimum(d, np.where(np.isfinite(t), t, 60.0))
    if wall_at is not None:
        d = np.minimum(d, wall_at)
    return d


td_h = build_topdown(_hall(), scale_m=1.0)
mid = td_h.n_lat // 2
col = td_h.grid[:, mid]
# Where the floor evidence FIRST runs out — not the furthest FREE cell. Ground
# density is noisy, so isolated FREE cells reappear past the boundary; a bound
# taken from the furthest one would pass no matter where the body may go.
first_open = min(((i + 1) * CELL_M for i, c in enumerate(col) if c == OPEN),
                 default=float("inf"))
check("far carved space is OPEN, not FREE", bool((td_h.grid == OPEN).any()),
      f"{int((td_h.grid == OPEN).sum())} cells are OPEN")
check("near space is still FREE", first_open > 2.0,
      f"floor evidence holds to {first_open:.1f} m")
check("the sightline still runs past where the floor was seen",
      td_h.ahead_m() > first_open + 1.0,
      f"sight {td_h.ahead_m():.1f} m vs floor {first_open:.1f} m")

# Under the feet there is no floor to SEE — the camera cannot look down that
# steeply. Marking that OPEN would ring the body in cells the proposer refuses
# to cross, so it is exempted by geometry, not by a tuned constant.
check("the optical blind spot is derived, not guessed",
      abs(td_h.floor_blind_m - 1.55 / math.tan(math.radians(45.0))) < 0.05,
      f"{td_h.floor_blind_m:.2f} m")
check("the ground right in front of the feet is not called unproven",
      td_h.grid[2, mid] == FREE, f"cell at 0.25 m is {td_h.grid[2, mid]}")

# The body must never be invited onto ground nobody has seen.
walk = passable_range(td_h)[len(td_h.bearings) // 2]
check("passable_range stops at the FIRST unproven cell, not the last proven one",
      walk <= first_open, f"walk {walk:.1f} m, floor evidence ends {first_open:.1f} m")
check("…and it is well short of the sightline",
      walk < td_h.ahead_m() - 1.0, f"walk {walk:.1f} vs sight {td_h.ahead_m():.1f}")
# §10.3 splits the candidate in two: the AIM (`distance`, may point past the
# proven floor at structure the sightline says continues — never at a wall)
# and the WALK (`stride_m` ≤ `verified_m`, executable blind). The executable
# part is what must stand on proven floor, every cell of it.
for w in propose(td_h):
    i, j = td_h.cell_of(w.x_left, w.y_fwd)
    check(f"the aim is never inside an obstacle ({w.kind})",
          td_h.grid[i, j] != OCCUPIED, f"cell state {td_h.grid[i, j]}")
    li, lj = td_h.cell_of(w.verified_m * math.sin(w.angle),
                          w.verified_m * math.cos(w.angle))
    check(f"the landing stands on proven floor ({w.kind})",
          td_h.grid[li, lj] == FREE, f"cell state {td_h.grid[li, lj]}")
    check(f"the stride never exceeds the proven ground ({w.kind})",
          w.stride_m <= w.verified_m + 1e-6,
          f"stride {w.stride_m:.1f} m, verified {w.verified_m:.1f} m")
    r = 0.55
    prefix_ok = True
    while r < w.stride_m - 1e-9:
        r = min(r + CELL_M, w.stride_m)
        pi, pj = td_h.cell_of(r * math.sin(w.angle), r * math.cos(w.angle))
        if td_h.grid[pi, pj] != FREE:
            prefix_ok = False
            break
    check(f"every cell of the executable prefix is FREE ({w.kind})",
          prefix_ok, f"checked to {w.stride_m:.1f} m")
    if w.distance > w.verified_m + 0.3:
        check("aiming past proven floor is said out loud, not smuggled",
              w.confidence in ("medium", "low")
              and w.visible_m >= w.distance,
              f"aim {w.distance:.1f} m, verified {w.verified_m:.1f} m, "
              f"sight {w.visible_m:.1f} m, confidence {w.confidence}")

# Where the line falls is set by the SENSOR, not by taste: ground samples per
# cell scale with resolution, so halving the depth grid halves how far the body
# may be sent. Worth knowing before anyone reverts depth to the YAML's 256.
reach = {}
for px in (256, 512):
    t = build_topdown(_hall(px=px), scale_m=1.0)
    reach[px] = float(passable_range(t)[len(t.bearings) // 2])
check("floor evidence — and so reach — scales with depth resolution",
      reach[512] > reach[256] + 1.0,
      f"256² → {reach[256]:.1f} m, 512² → {reach[512]:.1f} m")

# ── range_cap_m was a parameter that lied ────────────────────────────────
# Callers could pass any cap; every consumer downstream still took the lateral
# offset from the module constant, so the column index was off by half the grid
# width and the whole profile sheared — silently, because the shape was right.
for cap in (6.0, 12.0):
    t = build_topdown(_hall(wall_at=4.0), range_cap_m=cap, scale_m=1.0)
    check(f"cap {cap:g}: the grid is the size it was asked for",
          t.grid.shape == (int(cap / CELL_M), int(2 * cap / CELL_M)), str(t.grid.shape))
    check(f"cap {cap:g}: the wall is still measured at the right range",
          abs(t.ahead_m() - 4.0) < 0.35, f"{t.ahead_m():.2f} m")
    i, j = t.cell_of(0.0, 2.0)
    check(f"cap {cap:g}: straight ahead maps to the centre column",
          abs(j - t.n_lat // 2) <= 1, f"col {j} of {t.n_lat}")
    for w in propose(t):
        ii, jj = t.cell_of(w.x_left, w.y_fwd)
        check(f"cap {cap:g}: candidates land inside their own grid",
              t.inside(ii, jj) and t.grid[ii, jj] == FREE, f"({ii},{jj})")

# ── left / right must never be mirrored ──────────────────────────────────
lop = render(corridor_half_width=4.0)
f = (W / 2.0) / math.tan(math.radians(45.0))
us = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
lop[:, us[0] > W * 0.62] = np.minimum(lop[:, us[0] > W * 0.62], 1.5)  # blockage on the RIGHT
td_lop = build_topdown(lop)
third = max(1, len(td_lop.free_range) // 3)
right_open = float(np.median(td_lop.free_range[:third]))
left_open = float(np.median(td_lop.free_range[-third:]))
check("bearings are not mirrored: the blocked side reads as the right",
      right_open < left_open - 1.0, f"right {right_open:.1f} m vs left {left_open:.1f} m")
sentence = surroundings_sentence(td_lop)
check("the sentence names the OPEN side, and it is the left",
      "open to your left" in sentence, sentence)
# ties must not fling the "roomiest direction" to the edge of the field of view
td_even = build_topdown(render(corridor_half_width=5.0))
bearing_even, _ = td_even.widest()
check("an evenly open room reports straight ahead, not a FOV edge",
      abs(math.degrees(bearing_even)) < 20, f"{math.degrees(bearing_even):+.0f}°")

# ── degenerate input must never raise ────────────────────────────────────
for name, arr in (("all zeros (no return)", np.zeros((H, W), np.float32)),
                  ("all far", np.full((H, W), 20.0, np.float32)),
                  ("nose against a wall", np.full((H, W), 0.3, np.float32))):
    try:
        propose(build_topdown(arr))
        check(f"degenerate depth survives: {name}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"degenerate depth survives: {name}", False, repr(exc))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("depthmap self-test: all passed")
