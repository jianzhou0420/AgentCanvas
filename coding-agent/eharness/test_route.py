"""§14.6 — RouteCandidate: a glimpse becomes a PLANNED, cancellable route.

The bar scene, at the map level: verified FREE floor, an occluding counter,
and a POTENTIAL region glimpsed beyond it. The route planner must put its
staging point on walkable ground where a look BUYS something, keep the
second leg unverified until new evidence opens it, cancel the whole intent
when the glimpse turns out to be wall — and the intent must survive turning
the body, because "go look behind the bar" does not stop being the plan
when you face left for a moment.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harnesses" / "mini"))

from eharness import depthmap as dm  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    FAILS.append(name) if not ok else None
    print(("  ok  " if ok else "  FAIL") + " · " + name
          + (f"  [{detail}]" if detail else ""))


def stamp(am: dm.AnchorMap, arr: np.ndarray, x0, x1, y0, y1, val) -> None:
    xs, ys = np.meshgrid(np.arange(x0, x1, am.cell),
                         np.arange(y0, y1, am.cell))
    i, j = am.cells(xs.ravel(), ys.ravel())
    ok = (i >= 0) & (i < am.n) & (j >= 0) & (j < am.n)
    arr[i[ok], j[ok]] = val


def bar_map() -> tuple[dm.AnchorMap, dict]:
    """Verified corridor, a counter, and a glimpse of floor beyond it."""
    am = dm.AnchorMap()
    # proven FREE floor: a long room the body has looked down — the region
    # sits FAR (rim ≈5.8 m), so the start pose is outside STAGE_FAR_M and
    # the planner must actually walk a first leg
    stamp(am, am.logodds, -1.5, 1.5, -0.5, 7.5, -3.0)
    # the counter: OCCUPIED, off to the front-right, does not seal the room
    stamp(am, am.logodds, 0.6, 2.0, 5.0, 5.3, 3.0)
    # the glimpse: floor seen UNDER the counter — potential votes on
    # otherwise-UNKNOWN cells beyond it
    stamp(am, am.logodds, 0.8, 2.2, 5.8, 7.0, 0.0)
    stamp(am, am.potential, 0.8, 2.2, 5.8, 7.0, 1.0)
    regions = am.potential_regions()
    assert regions, "scene must yield a potential region"
    return am, regions[0]


print("── §14.6 · the accumulated potential layer is addressable ──────────")
am, region = bar_map()
check("one region, with cells / centroid / egocentric readout",
      region["cells"] > 50 and region["distance_m"] > 2.0
      and len(region["pts"]) > 10,
      f"cells {region['cells']}, at {region['distance_m']} m, "
      f"bearing {region['bearing_deg']}°")

print()
print("── §14.6 · plan_route: staging bought with information ─────────────")
route = dm.plan_route(am, region, intent="explore behind the counter")
check("a route is planned at all", route is not None)
assert route is not None
sx, sy = route.staging_anchor
si, sj = (int(v) for v in am.cells(sx, sy))
check("the staging point stands on verified FREE ground",
      float(am.logodds[si, sj]) < -0.5, f"logodds {am.logodds[si, sj]:.1f}")
check("leg 1 is verified, leg 2 is NOT (the glimpse stays a glimpse)",
      route.legs[0]["status"] == "verified"
      and route.legs[1]["status"] == "unverified",
      f"legs {[(l['status'], l['distance_m']) for l in route.legs]}")
check("leg 1 rides an actual path (A* polyline, not a bearing guess)",
      len(route.legs[0].get("path", [])) >= 2)

# §14.6: standing at the staging point must SEE more of the FRONTIER'S
# unknown than standing where the body started — that is what the point is
# FOR (generic unknown beyond mapped walls does not count)
from scipy.ndimage import distance_transform_edt  # noqa: E402

_rm = np.zeros((am.n, am.n), bool)
for (_x, _y) in region["pts"]:
    _i, _j = (int(v) for v in am.cells(_x, _y))
    _rm[_i, _j] = True
_d_region = distance_transform_edt(~_rm) * am.cell
r_info = int(round(dm.R_INFO_M / am.cell))
interesting = (((np.abs(am.logodds) <= 0.5) | (am.potential > 0.5))
               & (_d_region <= dm.R_INFO_M))


def gain_at(i: int, j: int) -> float:
    return float(interesting[max(0, i - r_info):i + r_info + 1,
                             max(0, j - r_info):j + r_info + 1].sum())


bi, bj = (int(v) for v in am.cells(am.px, am.py))
check("info gain at staging beats info gain at the start pose",
      gain_at(si, sj) > gain_at(bi, bj),
      f"staging {gain_at(si, sj):.0f} vs start {gain_at(bi, bj):.0f} cells")
check("landing clearance was demanded, not hoped for",
      route.evidence["staging_clearance_m"] >= dm.MIN_CLEARANCE_M,
      f"{route.evidence['staging_clearance_m']} m")

print()
print("── §14.6 · the second leg is EARNED or the route DIES ──────────────")
ev0 = dm.route_evidence(am, route)
check("before any new look the region is mostly unknown",
      ev0["unknown_frac"] > 0.5, f"unknown {ev0['unknown_frac']:.2f}")
# the turn-and-look at the staging point reveals real floor:
am2, region2 = bar_map()
route2 = dm.plan_route(am2, region2, intent="explore behind the counter")
stamp(am2, am2.logodds, 0.8, 2.2, 5.8, 7.0, -3.0)     # floor confirmed
ev_open = dm.route_evidence(am2, route2)
check("new FREE evidence opens leg 2", ev_open["free_frac"] > dm.ROUTE_OPEN_FREE,
      f"free {ev_open['free_frac']:.2f}")
# …or the glimpse was wrong and the region is wall:
am3, region3 = bar_map()
route3 = dm.plan_route(am3, region3, intent="explore behind the counter")
stamp(am3, am3.logodds, 0.8, 2.2, 5.8, 7.0, 3.0)      # solid after all
ev_dead = dm.route_evidence(am3, route3)
check("new OCCUPIED evidence cancels the route",
      ev_dead["occupied_frac"] > dm.ROUTE_CANCEL_OCC,
      f"occupied {ev_dead['occupied_frac']:.2f}")

print()
print("── §14.6 · the intent SURVIVES turning the body ────────────────────")
pub_before = route.as_public(am)
d_before = pub_before["staging"]["distance_m"]
am.odometry(math.radians(90.0), 0.0)                  # face left
pub_after = route.as_public(am)
check("a 90° turn changes the egocentric bearing, not the plan",
      abs(pub_after["staging"]["distance_m"] - d_before) < 0.05
      and abs(pub_after["staging"]["bearing_deg"]
              - pub_before["staging"]["bearing_deg"]) > 45.0,
      f"bearing {pub_before['staging']['bearing_deg']}° → "
      f"{pub_after['staging']['bearing_deg']}°, "
      f"distance {d_before} → {pub_after['staging']['distance_m']} m")
check("the route object itself is untouched by the turn",
      route.state == "leg1" and route.legs[1]["status"] == "unverified")

print()
print("── §20.3 · numbered staging must LIE ON leg-1, never bearing-nearest ─")


def flank_map() -> tuple[dm.AnchorMap, dict]:
    """A WIDE counter seals the frontal approach — the only safe leg runs
    up the LEFT flank. Staring at the glimpse is exactly wrong here."""
    am = dm.AnchorMap()
    stamp(am, am.logodds, -1.5, 1.5, -0.5, 7.5, -3.0)
    # a DEEP counter block: the whole frontal band sits > STAGE_FAR_M from
    # the region rim, so no front staging qualifies — only the left flank
    stamp(am, am.logodds, -0.2, 2.0, 2.9, 5.3, 3.0)
    stamp(am, am.logodds, 0.8, 2.2, 5.8, 7.0, 0.0)
    stamp(am, am.potential, 0.8, 2.2, 5.8, 7.0, 1.0)
    regions = am.potential_regions()
    assert regions
    return am, regions[0]


amS, regS = flank_map()
routeS = dm.plan_route(amS, regS, intent="explore behind the counter")
assert routeS is not None
_path = routeS.legs[0]["path"]


def _wp_at(x_left: float, y_fwd: float) -> dm.Waypoint:
    return dm.Waypoint(angle=math.atan2(x_left, y_fwd),
                       distance=math.hypot(x_left, y_fwd),
                       clearance=1.0, kind="opening",
                       x_left=x_left, y_fwd=y_fwd)


# wA sits ON the verified first leg (built from an actual path vertex);
# wB stares straight AT the glimpse — the old bearing-nearest winner —
# but stands off the safe leg, at the counter's face.
_mid = _path[max(1, len(_path) // 2)]
wA = _wp_at(*(float(v) for v in amS.to_body(_mid[0], _mid[1])))
_cx, _cy = regS["centroid_anchor"]
_norm = math.hypot(_cx, _cy) or 1.0
wB = _wp_at(*(float(v) for v in amS.to_body(4.5 * _cx / _norm,
                                            4.5 * _cy / _norm)))
chosen = dm.staging_place_for(routeS, [wB, wA], amS)
check("the on-leg place wins the staging label",
      chosen == 2, f"chosen {chosen} (1=at-the-glimpse, 2=on-leg)")
check("…and the at-the-glimpse place alone earns NO staging label",
      dm.staging_place_for(routeS, [wB], amS) is None)
d0 = routeS.as_public(amS)["staging"]["distance_m"]
_p1 = _path[1]
_ang = math.atan2(-_p1[0], _p1[1]) - amS.theta
amS.odometry(_ang, 0.0)
amS.odometry(0.0, 1.0)
d1 = routeS.as_public(amS)["staging"]["distance_m"]
check("walking leg-1 brings the staging CLOSER (no bar-face detour)",
      d1 < d0 - 0.5, f"{d0} → {d1} m")

print()
print("── §14.6 · the state machine on a live toolset ─────────────────────")
from depth_toolset import DepthWaypointToolSet  # noqa: E402

ts = object.__new__(DepthWaypointToolSet)
ts.amap, ts.route = am3, route3
ts._route_history, ts._events_log = [], []
ts.events, ts._events_consumed = [], 0
ts._sensor_frame = 0
ts._current_action = "observe"
ts._live_log = lambda *a, **k: None
pub = DepthWaypointToolSet._route_update(ts)
check("occupied evidence → state=cancelled + negative record kept",
      pub is not None and pub["state"] == "cancelled" and ts.route is None
      and ts._route_history and "cancelled" in ts._route_history[-1],
      f"{pub and pub.get('why', '')[:50]}")
check("…and the cancellation is a NavigationEvent",
      any(e.type == "route_cancelled" for e in ts.events))
ts2 = object.__new__(DepthWaypointToolSet)
ts2.amap, ts2.route = am2, route2
ts2._route_history = []
ts2.events, ts2._events_consumed = [], 0
ts2._sensor_frame = 0
ts2._current_action = "observe"
ts2._live_log = lambda *a, **k: None
pub2 = DepthWaypointToolSet._route_update(ts2)
check("free evidence → leg 2 verified, route stays alive",
      pub2 is not None and pub2["state"] in ("leg2_open", "done")
      and (ts2.route is None or ts2.route.legs[1]["status"] == "verified"))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("route candidate: all passed")
