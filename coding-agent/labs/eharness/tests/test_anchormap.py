"""Offline self-test for the accumulated map — synthetic wedges, no env.

Run:  python -m eharness.test_anchormap

Every case here is a bug that was REPORTED from the monitor before it was
understood, so each one states the symptom a human saw:

  * "这个 accumulated 的图好像是镜像相反的"  → mirror
  * "转向的时候后边走过的轨迹也跟着转"      → trail/grid rotating opposite ways
  * "漂移一下就没了"                        → resampling blur + drop-on-drift

The old rolling map failed cases 1-3 of this file. Keeping them means the next
rewrite has to earn its way past the same three complaints.
"""
from __future__ import annotations

import math
import sys
from io import BytesIO

import numpy as np

from eharness import depthmap as dm
from eharness.depthmap import (
    CELL_M,
    OCCUPIED,
    AnchorMap,
    build_topdown,
    project_mask,
    render_topdown,
)

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover
    PILImage = None

FAILS: list[str] = []
W = H = 256
CAM_H = 1.25
TURN = math.radians(15.0)
STEP = 0.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name + (f"  [{detail}]" if detail else ""))


def _rays():
    f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return (us - W / 2.0) / f, -(vs - H / 2.0) / f


def scene(panels=(), wall_at=None, corridor=None) -> np.ndarray:
    """Flat floor, optional side walls at x = ±corridor, a front wall, and
    upright panels given as (x_lo, x_hi, z) in metres."""
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


def red_halves(png: bytes) -> tuple[float, float]:
    img = np.asarray(PILImage.open(BytesIO(png)).convert("RGB")).astype(int)
    red = (img[:, :, 0] > 120) & (img[:, :, 1] < 110) & (img[:, :, 2] < 110)
    return float(red[:, : red.shape[1] // 2].sum()), float(red[:, red.shape[1] // 2 :].sum())


print("── 1. the accumulated map must agree with the per-frame map ────────")
# obstacle entirely on the LEFT, at 2 m
td_left = build_topdown(scene(panels=[(-3.0, -1.0, 2.0)]))
l0, r0 = red_halves(render_topdown(td_left, []))
check("per-frame top-down draws a left-hand obstacle on the left",
      l0 > r0, f"left {l0:.0f} vs right {r0:.0f}")
am = AnchorMap()
am.fuse(td_left)
l1, r1 = red_halves(am.render([], crop=False))
check("accumulated map draws the SAME obstacle on the same side (not mirrored)",
      l1 > r1, f"left {l1:.0f} vs right {r1:.0f}")

print()
print("── 2. trail and walls must swing the SAME way on a turn ────────────")
am = AnchorMap()
am.fuse(build_topdown(scene(panels=[(-0.4, 0.4, 2.0)])))     # wall straight ahead
for _ in range(4):
    am.odometry(0.0, STEP)                                   # walk 1 m
start_x, start_y = am.to_body(*am.trail[0][:2])
check("before turning, the start is straight BEHIND me",
      float(start_y) < -0.8 and abs(float(start_x)) < 0.2,
      f"x_left {float(start_x):+.2f} y_fwd {float(start_y):+.2f}")
for _ in range(6):
    am.odometry(TURN, 0.0)                                   # turn LEFT 90°
start_x, start_y = am.to_body(*am.trail[0][:2])
check("after turning LEFT 90°, the start is on my LEFT",
      float(start_x) > 0.8, f"x_left {float(start_x):+.2f}")
occ_i, occ_j = np.nonzero(am.logodds > 1.0)
X = am.ox + (occ_j - am.centre) * am.cell
Y = am.oy - (occ_i - am.centre) * am.cell
bx, by = am.to_body(X, Y)
far = np.hypot(bx, by) > 1.0
check("after the same turn, the wall that was ahead is on my RIGHT",
      float(np.median(bx[far])) < -0.5, f"median x_left {float(np.median(bx[far])):+.2f}")

print()
print("── 3. a hundred turns must not dissolve the map ────────────────────")
# The rolling map resampled the whole grid on every primitive; walking a square
# ten times was enough to blur a wall into nothing.
am = AnchorMap()
wall = build_topdown(scene(panels=[(-0.4, 0.4, 2.0)]))
am.fuse(wall)
mass0 = float((am.logodds > 1.0).sum())
for _ in range(10):                                  # ten full 360° spins in place
    for _ in range(24):
        am.odometry(TURN, 0.0)
mass1 = float((am.logodds > 1.0).sum())
check("240 turns in place leave the wall exactly where it was",
      mass1 == mass0, f"{mass0:.0f} → {mass1:.0f} cells")
back_x, back_y = am.to_body(*am.trail[0][:2])
check("ten full spins return the heading to where it started",
      abs(float(back_x)) < 0.05 and abs(float(back_y)) < 0.05,
      f"start now at {float(back_x):+.2f},{float(back_y):+.2f}")

print()
print("── 4. walking a corridor and coming back must close the loop ───────")
am = AnchorMap()
for k in range(12):
    am.fuse(build_topdown(scene(corridor=1.5, wall_at=6.0 - 0.25 * k)))
    am.odometry(0.0, STEP)
here = (am.px, am.py)
for _ in range(12):
    am.odometry(TURN, 0.0)                            # about-face
for _ in range(12):
    am.odometry(0.0, STEP)
check("odometry returns to the origin after out-and-back",
      math.hypot(am.px, am.py) < 0.1, f"{math.hypot(am.px, am.py):.2f} m off")
check("the trail records footsteps, not one entry per command",
      6 < len(am.trail) < 40, f"{len(am.trail)} points for 24 steps + 12 turns")

print()
print("── 5. traversed cells are floor, whatever a later ray says ─────────")
am = AnchorMap()
for _ in range(8):
    am.odometry(0.0, STEP)
i, j = am.cells(am.trail[4][0], am.trail[4][1])
am.logodds[i, j] = 5.0                                # pretend a stray hit lands there
am.fuse(build_topdown(scene()))
check("a cell the body stood on cannot be re-labelled wall",
      am.logodds[i, j] < 0, f"logodds {float(am.logodds[i, j]):+.1f}")

print()
print("── 6. registration corrects a slipped step, and only a little ──────")
# Habitat slides on collision: the harness commands 0.25 m and the body moves
# less. Build a map, then lie to the odometry and see it pulled back.
am = AnchorMap()
corridor = scene(corridor=1.2, wall_at=4.0)
for _ in range(6):
    am.fuse(build_topdown(corridor))
    am.odometry(0.0, 0.0)
am.px += 0.18                                          # a slip the odometry missed
score = am.register(build_topdown(corridor))
check("registration reports a match on a scene it has already mapped",
      score > 0.2, f"score {score:.2f}")
check("registration pulls the slipped pose back toward the truth",
      abs(am.px) < 0.18, f"|x| {abs(am.px):.2f} m (was 0.18)")
am2 = AnchorMap()
for _ in range(6):
    am2.fuse(build_topdown(corridor))
am2.px += 4.0                                          # far outside the search box
before = am2.px
am2.register(build_topdown(corridor))
check("registration refuses to teleport: a 4 m error is not silently 'fixed'",
      abs(am2.px - before) <= 0.30 + 1e-6, f"moved {am2.px - before:+.2f} m")

print()
print("── 7. the semantic layer remembers what is behind you ──────────────")
am = AnchorMap()
depth = scene(panels=[(1.0, 3.0, 2.0)])                # a thing on the RIGHT
td = build_topdown(depth)
am.fuse(td)
dx, dy = _rays()
mask = np.zeros((H, W), bool)
mask[(dx * 2.0 > 1.0) & (dx * 2.0 < 3.0) & (dy * 2.0 > -CAM_H + 0.3)
     & (dy * 2.0 < -CAM_H + 1.6)] = True
xs, ys = project_mask(mask, depth, td.floor_y)
check("a mask projects onto the floor where the geometry put the obstacle",
      xs.size > 50 and float(np.median(xs)) < -0.5,
      f"{xs.size} pts, median x_left {float(np.median(xs)) if xs.size else 0:+.2f}")
am.stamp_semantic("bar counter", xs, ys)
seen = am.semantic_recall()
check("the map can name what it stamped", bool(seen) and seen[0]["phrase"] == "bar counter")
check("and reports it on the RIGHT while it is still in front",
      seen[0]["bearing"] < -0.2, f"{math.degrees(seen[0]['bearing']):+.0f}°")
for _ in range(16):
    am.odometry(0.0, STEP)                             # walk 4 m past it
seen = am.semantic_recall()
check("after walking past, the landmark is remembered as BEHIND",
      seen[0]["behind"], f"{math.degrees(seen[0]['bearing']):+.0f}°")
check("and the recall sentence says so in plain egocentric words",
      "behind you" in am.recall_sentence(), am.recall_sentence())
check("a landmark the detector can SEE right now is left out of the recall",
      am.recall_sentence(exclude={"bar counter"}) == "",
      am.recall_sentence(exclude={"bar counter"}))
# A map that has stopped registering must stop quoting metres — a confidently
# wrong distance is the failure this whole project refuses to build on.
am.updates, am.last_score = 30, 0.05
check("a map that has lost registration says the side but not the distance",
      "m" not in am.recall_sentence().split("there:")[1].replace("remember", ""),
      am.recall_sentence())

print()
print("── 8. the map survives a long walk without leaving its box ─────────")
am = AnchorMap(half_m=6.0)
for k in range(200):
    am.fuse(build_topdown(scene(wall_at=4.0)))
    am.odometry(TURN if k % 40 == 39 else 0.0, STEP)
check("re-centring keeps the body inside the array over a 50 m walk",
      abs(am.px - am.ox) < am.half and abs(am.py - am.oy) < am.half,
      f"pose {am.px:.1f},{am.py:.1f} origin {am.ox:.1f},{am.oy:.1f}")
check("and the map still holds evidence afterwards",
      float((am.logodds < -0.5).sum()) > 100)
check("rendering a long walk does not raise", bool(am.render([])))
# What must survive the window sliding is the RELATIVE geometry. Stamp a door
# 2 m ahead, walk 5 m — past the re-centre threshold — and it must come out
# 3 m behind, in one piece. (Anything that scrolls out of the box is genuinely
# forgotten; the map is short-horizon by construction, not a world model.)
am = AnchorMap(half_m=6.0)
am.stamp_semantic("door", np.full(40, 0.0), np.full(40, 2.0))
for _ in range(20):
    am.odometry(0.0, STEP)
door = [r for r in am.semantic_recall() if r["phrase"] == "door"]
check("a landmark survives re-centring in one piece, in the right place",
      bool(door) and door[0]["behind"] and abs(door[0]["distance"] - 3.0) < 0.2,
      f"{door[0]['distance']:.2f} m behind={door[0]['behind']}" if door else "lost")

print()
print("── 9. the two things the drift study changed ───────────────────────")
# Measured on three R2R episodes against habitat's own pose (a ruler, never an
# input): dead-reckoned heading error was exactly 0.00° in all three, while
# letting the matcher rotate injected 15.5° / 2.0° / 13.0°. So rotation is off.
from eharness.depthmap import MATCH_TURN, frames_still, FRAME_STILL_M
check("the matcher does not search rotation by default", MATCH_TURN is False)
am = AnchorMap()
corridor = scene(corridor=1.2, wall_at=4.0)
for _ in range(6):
    am.fuse(build_topdown(corridor))
am.theta += math.radians(4.0)                    # a heading error to tempt it with
before = am.theta
am.register(build_topdown(corridor))
check("registration leaves the heading exactly alone",
      am.theta == before, f"{math.degrees(am.theta - before):+.2f}°")
am.register(build_topdown(corridor), allow_turn=True)
check("…unless explicitly asked to, which the ablation arm still needs",
      am.theta != before, f"{math.degrees(am.theta - before):+.2f}°")

# A blocked forward is detectable from the depth frame alone: over 168 commanded
# forwards it missed 0 of the 60 that moved the body zero.
still = scene(corridor=1.2, wall_at=2.0)
check("an identical frame reads as 'the body did not move'",
      frames_still(still, still.copy()))
check("a frame from 0.25 m further on does not",
      not frames_still(still, scene(corridor=1.2, wall_at=1.75)))
check("a missing frame is never mistaken for stillness — it is unknown",
      not frames_still(None, still) and not frames_still(still, None))
check("normalized vs metric frames are compared in metres, not raw units",
      frames_still(still / 10.0, still / 10.0))


# A blocked step is UNDONE, not walked backwards. Routing the undo through
# odometry() appended a second footstep, so the Human tab reported 14.2 m walked
# for 7.5 m of commands — every blocked step counted twice.
am = AnchorMap()
for _ in range(4):
    am.odometry(0.0, STEP)
walked = lambda m: sum(math.hypot(b[0]-a[0], b[1]-a[1])
                       for a, b in zip(m.trail, m.trail[1:]))
before, n_before = walked(am), len(am.trail)
am.odometry(0.0, STEP)          # a forward the wall ate
am.retract(STEP)                # …taken back
check("retract() returns the body to where it stood",
      abs(am.py - 4 * STEP) < 1e-9 and abs(am.px) < 1e-9,
      f"pose {am.px:+.3f},{am.py:+.3f}")
check("a blocked step adds NO distance to the walked total",
      abs(walked(am) - before) < 1e-9, f"{before:.2f} → {walked(am):.2f} m")
check("and adds no footstep to the trail",
      len(am.trail) == n_before, f"{n_before} → {len(am.trail)}")
check("retract ignores nonsense rather than moving backwards",
      (am.retract(0.0), am.retract(-1.0), abs(am.py - 4 * STEP) < 1e-9)[2])


print()
print("── 10. sliding along a wall must not read as 'did not move' ────────")
# ALLOW_SLIDING is on in R2R-CE, so a forward into an angled wall moves the body
# SIDEWAYS — and a flat wall looks identical in depth from every point along it.
# Depth alone erased 1.40 m of real travel over 168 forwards; RGB sees it.
import base64 as _b64
from io import BytesIO as _BIO
from eharness.depthmap import RGB_STILL, view_still

def _png(arr):
    img = PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")
    buf = _BIO(); img.save(buf, format="PNG")
    return _b64.b64encode(buf.getvalue()).decode()

wall = np.tile(np.arange(128, dtype=np.uint8)[None, :] * 2, (128, 1))  # textured wall
same, slid = _png(wall), _png(np.roll(wall, 12, axis=1))               # slid sideways
check("RGB calls an identical view still", view_still(same, same))
check("RGB sees a sideways slide that depth would call identical",
      not view_still(same, slid))
check("a missing RGB frame is never 'still' — it is unknown",
      not view_still(None, same) and not view_still(same, None))
flat = scene(corridor=1.2, wall_at=2.0)   # depth is bit-identical either way
check("depth alone calls the slide still (this is the hole)",
      frames_still(flat, flat.copy()))
check("with the RGB vote, the slide is no longer called still",
      not frames_still(flat, flat.copy(), prev_rgb=same, rgb=slid))
check("and a genuine block, where BOTH agree, still is",
      frames_still(flat, flat.copy(), prev_rgb=same, rgb=same))

print()
print("── 11. the RGB veto sits in the gap, not at the edge ──────────────")
from eharness.depthmap import RGB_STILL
# Measured: the slide that depth could not see moved RGB by 0.234; the largest
# frame-difference on a step that genuinely did not move was 0.0292. A veto
# anywhere between them catches the slide for free — at 0.03 it also vetoed
# genuine blocked steps and cost 0.16 m of mean position error.
check("the RGB veto is inside the measured gap, not on its lower edge",
      0.05 < RGB_STILL < 0.20, f"RGB_STILL = {RGB_STILL}")
check("it is still well below the slide it has to catch",
      RGB_STILL < 0.234, f"{RGB_STILL} vs slide 0.234")

print()
print("── 11b. the picture must show what the memory holds ────────────────")
# Structure stops at FUSE_MAX_M (4.5 m); a landmark is stamped as far as the
# detector can see one. The crop used to be computed from the structure layers
# alone, so a bar recognised at 6.6 m and a door at 13.0 m were written into the
# map and then cropped out of its image — precisely the far things worth drawing
# were the ones certain to fall outside the frame.
am = AnchorMap()
am.fuse(build_topdown(scene(corridor=1.5, wall_at=4.0)))
near_px = len(am.render([], crop=True))
am.stamp_semantic("door", np.full(60, 1.0), np.full(60, 13.0), weight=3.0)
i_far, j_far = am.cells(*am.to_anchor(1.0, 13.0))
check("a 13 m landmark is inside the grid at all",
      0 <= i_far < am.n and 0 <= j_far < am.n, f"cell ({i_far},{j_far}) of {am.n}")
rec = [r for r in am.semantic_recall() if r["phrase"] == "door"]
check("…and the memory holds it", bool(rec),
      f"{rec[0]['distance']:.1f} m" if rec else "not recalled")
# The crop has to reach it. Comparing PNG sizes is indirect; ask the renderer
# for the frame it chose by checking the picture grew to include that cell.
big = am.render([], crop=True)
check("…and the drawn map grew to include it", len(big) > near_px,
      f"{near_px} → {len(big)} bytes")

print()
print("── 11c. LoopMonitor: real circles convict, straight lines walk free ─")
lm = dm.LoopMonitor()
am_loop = dm.AnchorMap()
for lap in range(2):                       # a 2×2 m square, twice
    for side in range(4):
        for _ in range(8):
            am_loop.odometry(0.0, 0.25)
        am_loop.odometry(math.radians(90.0), 0.0)
        lm.note_choice(30.0)               # keeps picking the same sector
        lm.note_growth(100)                # …and the map stopped growing
warn = lm.assess(am_loop)
check("two laps of a square trigger a warning",
      warn is not None and warn["warning"],
      str({k: warn[k] for k in ("loop_score", "signals")} if warn else None))
check("…with the matched trail point named for the map ring",
      warn is not None and warn.get("revisit_trail_index") is not None)

lm2 = dm.LoopMonitor()
am_line = dm.AnchorMap()
for k in range(40):
    am_line.odometry(0.0, 0.25)            # 10 m dead straight
    lm2.note_choice(0.0 if k % 2 else 5.0)
    lm2.note_growth(40 * (k + 1))          # map grows as it walks
check("a straight walk never convicts", lm2.assess(am_line) is None,
      str(lm2.last))
am_back = dm.AnchorMap()
lm3 = dm.LoopMonitor()
for _ in range(20):
    am_back.odometry(0.0, 0.25)
am_back.odometry(math.radians(180.0), 0.0)
for k in range(6):
    am_back.odometry(0.0, 0.25)
    lm3.note_growth(500 + 40 * k)          # returning but map still growing
check("one ordinary look-back is not yet a loop",
      lm3.assess(am_back) is None, str(lm3.last))

print()
print("── 11d. §14.10: independent visual evidence + hysteresis ───────────")


def _sig(seed: int):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(16, 16)).astype(np.float32)
    m -= m.mean()
    return m / (abs(m).mean() or 1.0)


# the same square laps, but now every stored frame LOOKS different from the
# current view — a drifted trail claims a revisit the eyes cannot confirm.
lm4 = dm.LoopMonitor()
am4 = dm.AnchorMap()
n = 0
for lap in range(2):
    for side in range(4):
        for _ in range(8):
            am4.odometry(0.0, 0.25)
            n += 1
            lm4.note_frame(_sig(n), (am4.px, am4.py))   # all views distinct
        am4.odometry(math.radians(90.0), 0.0)
w4 = lm4.assess(am4)
check("spatial revisit without visual confirmation stays one signal",
      w4 is None or "revisit_visual" not in w4["signals"],
      str(lm4.last and lm4.last.get("signals")))

# same laps, and the SAME views recur where the trail recurs — the visual
# witness now agrees, and the warning carries it.
lm5 = dm.LoopMonitor()
am5 = dm.AnchorMap()
step_i = 0
for lap in range(2):
    for side in range(4):
        for k in range(8):
            am5.odometry(0.0, 0.25)
            lm5.note_frame(_sig(side * 100 + k), (am5.px, am5.py))
            step_i += 1
        am5.odometry(math.radians(90.0), 0.0)
w5 = lm5.assess(am5)
check("a revisit the eyes confirm adds the independent signal",
      w5 is not None and w5["signals"].get("revisit_visual") is True,
      str(w5 and w5["signals"]))

# perceptual alias: two similar-looking corridors FAR apart — matching
# views alone, with the trail nowhere near itself, must not convict.
lm6 = dm.LoopMonitor()
am6 = dm.AnchorMap()
for k in range(40):
    am6.odometry(0.0, 0.25)                 # 10 m straight
    lm6.note_frame(_sig(7), (am6.px, am6.py))   # every view identical
    lm6.note_growth(40 * (k + 1))
check("identical-looking corridors far apart do not convict",
      lm6.assess(am6) is None, str(lm6.last and lm6.last.get("signals")))

# hysteresis: the SAME physical loop must not warn on every assess — one
# warning, then a cooldown until the evidence goes away.
lm7 = dm.LoopMonitor()
am7 = dm.AnchorMap()
for lap in range(2):
    for side in range(4):
        for _ in range(8):
            am7.odometry(0.0, 0.25)
        am7.odometry(math.radians(90.0), 0.0)
        lm7.note_choice(30.0)
        lm7.note_growth(100)
first = lm7.assess(am7)
again = [lm7.assess(am7) for _ in range(3)]
check("the same loop warns ONCE, then holds its tongue",
      first is not None and all(a is None for a in again)
      and lm7.last.get("suppressed"),
      f"warnings {lm7.warnings}, last {lm7.last and lm7.last.get('signals')}")
check("…and the warning counter reflects one event, not four",
      lm7.warnings == 1, f"{lm7.warnings}")

print()
print("── 12. the offered place stops running away from you ───────────────")
# propose() is memoryless: it re-reads the frame and puts a point at the far end
# of what it can reach, so a step forward moves the point forward too. Measured
# down a plain corridor, the distance printed on the panel was 3.00 m on step 0
# and 3.00 m on step 7, while the point itself slid from 3.0 m to 4.7 m out in
# the world. Nothing on screen ever said the walking had accomplished anything.
am = dm.AnchorMap()
corridor = build_topdown(scene(corridor=2.0, wall_at=9.0))
am.fuse(corridor)
goal = dm.WaypointGoal()
first = goal.apply(am, corridor, dm.propose(corridor))
check("a place is offered to begin with", bool(first), f"{len(first)} candidates")
walked, body_d, world_d = 0.0, [first[0].distance], [first[0].distance]
for k in range(10):
    am.odometry(0.0, 0.25)
    walked += 0.25
    td_k = build_topdown(scene(corridor=2.0, wall_at=9.0 - 0.25 * (k + 1)))
    am.fuse(td_k)
    got = goal.apply(am, td_k, dm.propose(td_k))
    if not got:
        break
    body_d.append(got[0].distance)
    world_d.append(walked + got[0].distance)   # how far out in the world it sits

# 1. It must never become something underfoot. "怎么地也得有个两三米的距离"
# §2.1 mid-range aiming legitimately adopts ~2 m goals, so the "never nearer
# than 2.5 m" floor became the SCALED arrive radius: the place may be walked
# down to arrive_m() — a real arrival — and never past it into the feet.
check("the offered place is never walked past its arrival radius",
      min(body_d) >= goal.arrive_m() - 0.05,
      f"nearest {min(body_d):.2f} m (arrive {goal.arrive_m():.2f} m)")
# 2. The place must hold STILL in the world. The treadmill bug read the other
#    way round: the body-frame number sat at a constant 3.0 m while the world
#    position advanced in lockstep with the feet — walking never accomplished
#    anything on screen. In this dead-end corridor the aim pins to just short
#    of the end wall from the very first frame (there IS nowhere further), so
#    the world position must neither run ahead with the body nor slide back.
# The anti-treadmill shape under mid-range aiming: PLATEAUS. Within one
# commitment the world position is pinned (the body approaches it); a release
# steps it forward once. A treadmill instead advances world_d every single
# step with body_d constant — zero plateaus longer than one.
runs, cur = [], 1
for a, b in zip(world_d, world_d[1:]):
    if abs(b - a) < 0.15:
        cur += 1
    else:
        runs.append(cur)
        cur = 1
runs.append(cur)
check("and it advances in PLATEAUS, never step-for-step with the feet",
      max(runs) >= 3 and len(runs) <= 4,
      " → ".join(f"{v:.1f}" for v in world_d))
check("…never sliding back toward you",
      all(b >= a - 0.35 for a, b in zip(world_d, world_d[1:])),
      " → ".join(f"{v:.1f}" for v in world_d))
# 3. Within one commitment it approaches; the jumps are re-picks from further on.
check("inside one commitment the distance counts down",
      any(b < a - 0.1 for a, b in zip(body_d, body_d[1:])),
      " → ".join(f"{v:.1f}" for v in body_d))

# Arriving has to be a thing that happens, or the goal is just a slower carrot.
for _ in range(20):
    am.odometry(0.0, 0.25)
    if goal.held(am, corridor) is None:
        break
check("arriving releases the commitment", goal.anchor is None,
      "released" if goal.anchor is None else "still held past arrival")

# …and a commitment must never outlive its own truth: something in the way ends
# it, because a stale goal invites the body through whatever moved in.
#
# The obstruction is a PANEL, not `wall_at`. wall_at clamps the whole frame to
# one distance, which leaves no floor anywhere and no floor for estimate_floor
# to calibrate against — a degenerate image, not a blocked corridor, and it
# tested the height band far more than it tested this. A panel across the way
# with the floor still visible is what "something moved into the path" looks
# like.
am2 = dm.AnchorMap()
open_room = build_topdown(scene(corridor=4.0))
am2.fuse(open_room)
g2 = dm.WaypointGoal()
g2.apply(am2, open_room, dm.propose(open_room))
check("a goal was taken in the open room", g2.anchor is not None)
blocked = build_topdown(scene(corridor=4.0, panels=[(-1.5, 1.5, 1.2)]))
dropped = g2.held(am2, blocked) is None
check("something across the way in drops the commitment", dropped,
      "dropped" if dropped else "SURVIVED an obstacle in its path")
# The other half, and the easier one to get wrong: clutter that is NOT in the
# way must not cancel the plan. A goal dropped every time anything appeared
# anywhere would put the proposer back to re-picking from a standstill, which
# is the behaviour this whole mechanism exists to end.
g2.apply(am2, open_room, dm.propose(open_room))
beside = build_topdown(scene(corridor=4.0, panels=[(2.0, 3.0, 1.2)]))
kept = g2.held(am2, beside)
check("…but clutter off to the side does not", kept is not None,
      f"{kept.distance:.2f} m" if kept else "dropped for something not in the way")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("anchormap self-test: all passed")
