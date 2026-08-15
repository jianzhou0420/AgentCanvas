"""Offline self-test for the place memory — synthetic wedges, no env.

Run:  python -m eharness.test_places

Each case is one of the objections the adversarial review raised against the
design. They are here so the next person to touch the boundary rule has to walk
past the reason it is shaped this way.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from eharness.depthmap import build_topdown
from eharness.places import (
    MIN_PLACE_MOVES,
    PlaceMemory,
    free_width,
)
FAILS: list[str] = []
W = H = 256
CAM_H = 1.25


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name + (f"  [{detail}]" if detail else ""))


def scene(corridor=None, wall_at=None):
    """Flat floor, optional side walls at x = ±corridor, optional front wall."""
    f = (W / 2.0) / math.tan(math.radians(90.0) / 2.0)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx, dy = (us - W / 2.0) / f, -(vs - H / 2.0) / f
    depth = np.full((H, W), 25.0, np.float32)
    below = dy < -1e-6
    depth[below] = np.minimum(depth[below], -CAM_H / dy[below])
    if corridor is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dx) > 1e-6, corridor / np.abs(dx), np.inf)
        depth = np.minimum(depth, np.where(np.isfinite(t), t, 25.0))
    if wall_at is not None:
        depth = np.minimum(depth, wall_at)
    return depth


class FakeSighting:
    def __init__(self, phrase, bearing_deg):
        self.phrase = phrase
        self.bearing = math.radians(bearing_deg)


print("── 1. a pirouette is not a new place ───────────────────────────────")
# The review's sharpest objection: 90 degrees of FOV and 15 degrees per turn
# means six turn primitives rotate the entire visible set with the body
# standing still. A turnover-based boundary would open a new place for that.
pm = PlaceMemory()
open_room = build_topdown(scene(corridor=4.0))
for k in range(12):                       # spin in place, landmarks swap around
    seen = ["pool"] if k < 6 else ["bar"]     # complete turnover of what is visible
    ev = pm.observe(open_room, seen, (0.0, 0.0))
check("spinning on the spot opens no new place",
      len(pm.places) == 1, f"{len(pm.places)} places, events {pm.events}")
check("…but everything seen while spinning belongs to the one place",
      set(pm.places[0].seen) == {"pool", "bar"}, str(pm.places[0].seen))

print()
print("── 2. walking through a doorway does open one ──────────────────────")
pm = PlaceMemory()
wide = build_topdown(scene(corridor=4.0))
door = build_topdown(scene(corridor=0.6))      # a pinch across the body
for _ in range(MIN_PLACE_MOVES + 1):
    pm.observe(wide, ["pool"], (0.0, 0.0))
pm.observe(door, ["pool"], (0.0, 1.0))         # in the doorway
ev = pm.observe(wide, ["bar"], (0.0, 2.0))     # out the far side
check("a pinch then a release is a place boundary", ev != "", f"event {ev!r}")
check("and it is two places now", len(pm.places) == 2, f"{len(pm.places)}")
check("the first place kept its own contents only",
      set(pm.places[0].seen) == {"pool"}, str(pm.places[0].seen))

print()
print("── 3. a corridor stays ONE place ───────────────────────────────────")
# The review: a corridor is a place, not a sequence of places. Walking a long
# way in a constant-width space must not chop it up until the fallback bites.
pm = PlaceMemory()
corridor = build_topdown(scene(corridor=1.5))
for k in range(30):
    pm.observe(corridor, ["hallway"], (0.0, 0.25 * k))     # 7.5 m of walking
check("7.5 m of constant-width corridor is one place",
      len(pm.places) == 1, f"{len(pm.places)} places")
pm.observe(corridor, ["hallway"], (0.0, 12.0))
check("…but the open-plan fallback does eventually bite",
      len(pm.places) == 2, f"{len(pm.places)}")

print()
print("── 4. the sentences carry no metres, no step numbers ───────────────")
pm = PlaceMemory()
for _ in range(MIN_PLACE_MOVES + 1):
    pm.observe(wide, ["pool"], (0.0, 0.0))
pm.observe(door, ["pool"], (0.0, 1.0))
pm.observe(wide, ["bar"], (0.0, 2.0))
lines = pm.lines([FakeSighting("bar", -35)])
text = " ".join(lines)
check("something is actually said", bool(lines), text)
check("no metres anywhere in what the model sees",
      not any(ch.isdigit() for ch in text), text)
check("the walked line is ordinal, and names places by content",
      "walked through" in text and "the place with the pool" in text, text)
trip = pm.lines([FakeSighting("bar", -35), FakeSighting("bar", 40), FakeSighting("bar", 5)])[0]
check("the same noun is never listed twice", trip.count("the bar") == 1, trip)
check("side survives inside the current place",
      "on your right" in text, text)

print()
print("── 5. recognition is a QUESTION and never an assertion ─────────────")
# The path at_goal=True → near_goal → armed brake has no gate in it, so an
# asserted false memory is unrecoverable for a 9B. A question is not.
pm = PlaceMemory()
seq = [(["pool", "chairs"], 6), (["bar", "table"], 6), (["pool", "chairs"], 6)]
xy, out = 0.0, []
for phrases, n in seq:
    for _ in range(n):
        pm.observe(wide, phrases, (0.0, xy))
        xy += 0.3
    pm.observe(door, phrases, (0.0, xy)); xy += 0.3
    out.append(pm.observe(wide, phrases, (0.0, xy))); xy += 0.3
q = pm.recognition_question()
check("coming back to a place with the same contents is noticed",
      bool(q), q or "(nothing)")
check("and it is phrased as a question, not a claim",
      "?" in q and "same place" in q, q)
check("a single shared landmark is NOT enough to claim recognition",
      pm._match(pm.places[-1], need=2) is None or len(
          set(pm.places[-1].seen) & set(pm.places[0].seen)) >= 2)

print()
print("── 6. collapse emits an empty node, never no node ──────────────────")
pm = PlaceMemory()
for _ in range(MIN_PLACE_MOVES + 1):
    pm.observe(wide, [], (0.0, 0.0))        # detector found nothing at all
pm.observe(door, [], (0.0, 1.0))
pm.observe(wide, [], (0.0, 2.0))
check("a place with no recognised contents still exists as a node",
      len(pm.places) == 2, f"{len(pm.places)}")
check("and it says so rather than pretending", "could not make out" in pm.places[0].name(),
      pm.places[0].name())

print()
print("── 7. free_width measures ACROSS the body, not ahead ───────────────")
check("a wide room reads wide", free_width(build_topdown(scene(corridor=4.0))) > 3.0)
check("a doorway reads narrow", free_width(build_topdown(scene(corridor=0.6))) < 1.6)
# A wall straight ahead hides the probe row entirely, so the width reads 0 —
# and 0 must NOT be treated as a pinch, or every dead end would look like a door.
blocked = free_width(build_topdown(scene(corridor=4.0, wall_at=1.5)))
pm_w = PlaceMemory()
for _ in range(MIN_PLACE_MOVES + 2):
    pm_w.observe(build_topdown(scene(corridor=4.0, wall_at=1.5)), ["wall"], (0.0, 0.0))
check("a wall straight ahead is not mistaken for a doorway",
      len(pm_w.places) == 1, f"width {blocked:.2f} → {len(pm_w.places)} places")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("places self-test: all passed")
