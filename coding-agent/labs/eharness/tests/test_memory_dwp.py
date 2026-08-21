"""§global-dwp — DWP proposes from MEMORY, not just the current wedge.

A person who just saw a corridor on the right keeps it as an option after
turning away. Memory candidates come from the accumulated map's verified
FREE, live OUTSIDE the current 90° view, carry an honest provenance note
(visible_m=0, verified_m = the memory's proven prefix), and execution
still passes goto's post-turn revalidation — remembered floor proposes,
fresh eyes confirm.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harnesses" / "mini"))

from eharness import depthmap as dm  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def stamp(am, arr, x0, x1, y0, y1, val):
    xs, ys = np.meshgrid(np.arange(x0, x1, am.cell), np.arange(y0, y1, am.cell))
    i, j = am.cells(xs.ravel(), ys.ravel())
    ok = (i >= 0) & (i < am.n) & (j >= 0) & (j < am.n)
    arr[i[ok], j[ok]] = val


def seen_map(trusted: bool = True) -> dm.AnchorMap:
    """The body faces +y. A corridor it WALKED THROUGH runs off to the
    back-right (anchor +x, -y quadrant → body bearing ≈ −135°); a wall
    seals the current front. Only memory knows about the corridor."""
    am = dm.AnchorMap()
    # small free apron around the body
    stamp(am, am.logodds, -1.0, 1.0, -1.0, 1.0, -3.0)
    # the remembered corridor: 1.2 m wide, 5 m long, off back-right.
    # anchor x = -x_left, so body-right is anchor +x… corridor along the
    # diagonal (+x, −y) in anchor = bearing atan2(x_left, y_fwd) with
    # x_left negative, y negative → ≈ −135°.
    for t in np.arange(0.0, 5.0, 0.05):
        cx = 0.707 * t
        cy = -0.707 * t
        stamp(am, am.logodds, cx - 0.6, cx + 0.6, cy - 0.6, cy + 0.6, -3.0)
    am.updates = 6
    am.last_score = dm.TRUST_MIN_SCORE + 0.2 if trusted else 0.0
    return am


print("── the remembered corridor becomes a numbered option ───────────────")
am = seen_map()
mem = dm.propose_from_memory(am, [])
check("memory proposes the corridor the camera cannot see",
      len(mem) >= 1, f"{len(mem)} candidates")
if mem:
    w = mem[0]
    deg = math.degrees(w.angle)
    check("…at the remembered bearing (behind-right)",
          -170 < deg < -100, f"{deg:.0f}°")
    check("…with honest provenance: verified from MEMORY, visible 0, noted",
          w.verified_m >= dm.MEM_MIN_PREFIX_M and w.visible_m == 0.0
          and "from memory" in (w.note or "") and w.kind == "remembered",
          f"verified {w.verified_m}, note {w.note[:30]!r}")
    check("…aim is mid-range of the proven prefix, capped",
          dm.MIN_USEFUL_WAYPOINT_M <= w.distance <= dm.MEM_AIM_CAP_M,
          f"{w.distance:.1f} m of {w.verified_m:.1f} m proven")
    check("…and the spoken description says it is behind",
          "behind you" in w.describe(), w.describe()[:60])

print()
print("── honesty gates ───────────────────────────────────────────────────")
check("an UNTRUSTED map proposes nothing from memory",
      dm.propose_from_memory(seen_map(trusted=False), []) == [])
young = seen_map()
young.updates = 1
check("one look is a frame, not a memory (min updates gate)",
      dm.propose_from_memory(young, []) == [])
inview = dm.Waypoint(angle=math.radians(-30.0), distance=2.0, clearance=0.5,
                     kind="opening", x_left=-1.0, y_fwd=1.73)
check("the current wedge belongs to the fresh proposer (FOV guard)",
      all(abs(math.degrees(w.angle)) > dm.MEM_FOV_GUARD_DEG
          for w in dm.propose_from_memory(am, [inview])))
near_dup = dm.Waypoint(angle=math.radians(-130.0), distance=2.0,
                       clearance=0.5, kind="opening",
                       x_left=2.0 * math.sin(math.radians(-130.0)),
                       y_fwd=2.0 * math.cos(math.radians(-130.0)))
check("25° dedup applies against existing candidates",
      dm.propose_from_memory(am, [near_dup]) == [])

print()
print("── the toolset offers and EXECUTES a memory place ──────────────────")
from eharness.tests.test_toolset_contract import FakeEnv, scene  # noqa: E402

ts = FakeEnv(scene(corridor=2.0, wall_at=9.0), instruction="x", sam_url="")
ts._tool_observe()
# the body has (in this synthetic past) walked a corridor off back-right
for t in np.arange(0.0, 5.0, 0.05):
    stamp(ts.amap, ts.amap.logodds, 0.707 * t - 0.6, 0.707 * t + 0.6,
          -0.707 * t - 0.6, -0.707 * t + 0.6, -3.0)
ts.amap.last_score = dm.TRUST_MIN_SCORE + 0.2
view = ts._tool_observe()
places = view.info.get("places_you_can_walk_to") or {}
mem_ns = [int(k) for k, c in places.items() if "from memory" in
          str(c.get("where", ""))]
check("a memory place shows up in the numbered table",
      bool(mem_ns), f"places: {[(k, c['where'][:28]) for k, c in places.items()]}")
if mem_ns:
    n = mem_ns[0]
    p0 = len(ts.prims)
    res = ts._tool_goto(place=n)
    turns = [p for p in ts.prims[p0:] if p in (2, 3)]
    check("goto(memory place) turns for REAL through the executor",
          len(turns) >= 6, f"{len(turns)} turn notches")
    check("…then walks only what fresh eyes re-verified",
          res.info.get("error") is None
          and (res.info.get("moved_m", 0) > 0
               or "corridor" in str(res.info.get("what_happened", ""))
               or "blind" in str(res.info.get("what_happened", ""))),
          f"moved {res.info.get('moved_m')}, wh {str(res.info.get('what_happened'))[:40]}")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("memory dwp: all passed")
