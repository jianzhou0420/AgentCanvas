"""Depth-only waypoint proposal — the geometry organ.

ONE frontal depth frame (90° FOV) becomes an egocentric top-down free-space map,
and the free space becomes a handful of navigable waypoints expressed as
``(angle, distance)`` — bit-identical to what the learned waypoint predictor
emitted, so ``env_habitat__step_hightolow`` executes them unchanged.

Why geometry instead of the learned predictor (all verified in-repo):
  * ``smartway_waypoint``'s heatmap is 120 angles × 12 distance bins of 0.25 m —
    a HARD 3.00 m ceiling. It can never propose the 5.6 m stride this module
    measured at EP0's start.
  * it is fed ``depth_base64``, which env_habitat produces by PER-FRAME min-max
    normalisation to 8 bit — absolute scale destroyed, every view on its own
    ruler.
  * its RGB branch (``rgb_features``) is declared required and never sent, so it
    has always run as a depth model — on broken depth.
  * ``step_hightolow`` does NOT path-plan: it rotates, then blind-walks
    ``int(d/0.25)`` forward primitives, sliding on collision. A learned peak
    carries no reachability guarantee; a point derived from measured free space
    does. That guarantee is this module's whole reason to exist.

Doctrine: everything here is an ORGAN. The grid, the metres and the point cloud
never reach the model — it sees numbered circles on its own RGB plus a few
egocentric sentences. Nothing is expressed in a world frame; the map is always
robot-centred with the heading up, so no global coordinate is ever formed.

Verified contracts (probed live on :9200, 2026-08-05) — see
memory/habitat-depth-geometry-contract.md:
  * depth arrives as {"__ndarray__": b64, dtype, shape}; its UNITS ARE NOT
    STABLE — normalized [0,1] over 0-10 m on one env build, raw metres on
    another. Self-detect, never hardcode.
  * the `intrinsics` port describes the RGB sensor (512²), not the depth array
    (256²). Derive depth intrinsics from HFOV at the depth resolution.
  * depth 0.0 means MIN_DEPTH clip / no return, NOT an obstacle at 0 m.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np

try:  # pillow is already a harness dependency (frames.py)
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except Exception:  # pragma: no cover - pillow always present in practice
    PILImage = None
    ImageDraw = None
    ImageFont = None

# Pillow's built-in bitmap font has no CJK glyphs, so a Chinese caption burned
# into a map came out as a row of tofu boxes — which is exactly the caption a
# person reads. Fall back to whatever CJK face the machine has, then to the
# default; never fail, a missing font must not cost a render.
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
_FONT_CACHE: dict[int, Any] = {}


def caption_font(size: int = 12):
    if ImageFont is None:
        return None
    if size not in _FONT_CACHE:
        font = None
        for path in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:  # noqa: BLE001 - any unreadable face just moves on
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:  # noqa: BLE001
                font = None
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]

from scipy.ndimage import distance_transform_edt, median_filter

# ── the body, the map, the thresholds ────────────────────────────────────
DEPTH_FULL_RANGE_M = 10.0   # habitat's MAX_DEPTH when the frame is normalized
# How far the per-frame grid reaches. This was 6.0, and 6.0 was not a taste —
# it is where FLOOR evidence runs out. Ground samples thin as 1/d³ (a 10 cm cell
# subtends fewer pixels both across and along the ray), measured on R2R ep 7 at
# the 1.5 m rig: 9 floor points per cell at 4-6 m, 4 at 6-8 m, 3 at 8-10 m —
# straight through MIN_PTS_PER_CELL right about there.
#
# But FREE is not built from floor points. It is carved by ray-casting to the
# first OBSTACLE, and obstacle evidence does NOT thin the same way — a wall is
# vertical, so it keeps 10 points per cell out at 12-16 m. The old cap threw
# away structure the sensor measured perfectly well: on that frame, one bearing
# ran open to 7.35 m and got reported as a flat 6.00.
#
# So the grid now reaches 12 m and the two kinds of knowledge are kept APART —
# see FREE / OPEN below. Widening a grid that conflated them would have widened
# the confusion with it.
RANGE_CAP_M = 12.0
FLOOR_BAND_M = 0.10         # within this of the floor plane counts as ground seen
MIN_FLOOR_PTS = 4           # …and this many per cell make it evidence, not a speck
CELL_M = 0.10               # top-down grid resolution
ROBOT_RADIUS_M = 0.20       # the body the harness must respect
SAFETY_MARGIN_M = 0.10      # extra clearance demanded of a waypoint
MIN_CLEARANCE_M = ROBOT_RADIUS_M + SAFETY_MARGIN_M
# …and the floor below which the BODY genuinely does not fit. The difference
# between the two is a PREFERENCE, not a safety fact, and treating it as a veto
# made the harness route on the model's behalf: parked at a fork with a wide
# branch right and a tight one left, the tight branch measured 0.28 m of
# clearance — a 0.56 m passage for a 0.40 m body, 8 cm of slack per side — and
# was silently dropped, so the model was never told the fork existed. A squeeze
# the body fits through is the model's call; only "it does not fit" is ours.
TIGHT_CLEARANCE_M = ROBOT_RADIUS_M + 0.02
OBSTACLE_LO_M = 0.20        # height band above the floor that blocks the robot
OBSTACLE_HI_M = 1.80
MIN_PTS_PER_CELL = 4        # density threshold: one stray pixel is not a wall
MIN_WAYPOINT_M = 1.0        # closer than this is not worth a decision…
MIN_WAYPOINT_TIGHT_M = 0.5  # …unless nothing else is on offer at all
#   Wedged in front of a chair with a way out on either side, every candidate
#   past 1.0 m was blocked and the model was handed an empty list — told, in
#   effect, that there was nowhere to go, while two ways round the chair sat in
#   the measurement. When the choice is between a short hop and nothing, the
#   short hop is the decision.
# Ceiling on ONE hop. Not a safety limit — a pacing one: a 5.6 m leap crosses
# most of a room before the model gets to look again, and three EP0 runs showed
# it arriving somewhere it had never decided to go. Short hops keep judgement
# in the loop, which is the whole point of the harness.
# TWO different numbers that were one number, and conflating them is why the
# offered place never moved.
#
#   MAX_STRIDE_M   how far the body may walk BLIND in one go. step_hightolow
#                  does not path-plan: it turns once and issues int(d/0.25)
#                  forwards with sliding on collision, so a long leap down a
#                  narrow slot ends wedged in a corner. This is a safety limit
#                  on EXECUTION and it is real.
#
#   MAX_POINT_M    how far away the place being pointed AT may be. This is not
#                  a safety question at all — nothing walks there in one piece.
#
# They were the same constant (3.0) and the pointing inherited the walking's
# limit. Measured on R2R ep 7 step 0: the -22° branch is body-passable to
# 7.2 m, and the proposer offered a point at 3.0 m — the ceiling, not the
# geometry. Take a step and it offered 3.0 m again, and again, because the
# ceiling does not move when the body does. That is the "it predicts the same
# point after three or four steps" report: the point was never describing a
# PLACE, it was describing the cap.
#
# Pointing far is also what makes the difference legible: a target 7 m down a
# corridor stays the same target for many steps and visibly approaches, whereas
# a target at the stride limit is regenerated every step and can only ever look
# identical.
MAX_STRIDE_M = 3.0
MAX_POINT_M = 9.0
MAX_WAYPOINT_M = MAX_STRIDE_M      # back-compat alias; prefer the two above
# Closer than this and the place stops being a destination — it is underfoot.
# The commitment is re-taken from here rather than walked down to zero: "the
# point you are heading for should be a couple of metres out, at least".
REPROPOSE_M = 2.5
MAX_CANDIDATES = 3   # applied AFTER structure is extracted: the two sides of an
                     # obstacle island can never be squeezed out by two
                     # near-duplicate points from one wide opening (§10.4).
#   Three sectors (left / ahead / right) were an arbitrary cut of the field of
#   view, not a reading of it: with a 90° camera the middle "sector" overlaps
#   both others, so the model was routinely offered three points inside ONE
#   opening and asked to choose between them. Openings are what the depth frame
#   actually contains, and a 90° view rarely shows more than two.
OPEN_MIN_M = 1.2          # a bearing counts as an opening if floor reaches this far
OPEN_REL = 0.45           # …or this fraction of the deepest BODY-PASSABLE bearing
#   Anchored to `passable_range`, never to `free_range`: on the sightline
#   profile this term deleted a real 2.8 m branch because a crack elsewhere saw
#   6.0 m, and removing it entirely then merged a whole room into one opening
#   whose centre pointed at a chair. The reference has to be a direction the
#   body can actually use.
OPEN_MIN_DEG = 2.0        # below this is single-bin noise, nothing more
#   It used to be 8°, from when openings were read off the SIGHTLINE profile and
#   a narrow window really did mean a crack. On the body-passable profile the
#   question is already settled — `passable_range` only counts a bearing if the
#   clearance stays above the body radius the whole way — so angular width no
#   longer measures passability, it measures how precisely you must AIM. And
#   `goto` aims by writing the heading, so it can. Killing a 3.5° window that
#   the body can walk 2.55 m into was rejecting a real fork for the wrong reason.
OPEN_SEP_DEG = 18.0       # two candidates closer than this are the same opening
NMS_M = 0.8
FOV_DEG = 90.0
SELF_FOOTPRINT_M = 0.5      # the robot is already standing where it stands
FRAME_STILL_M = 0.002       # mean |Δdepth| below this ⇒ the body did not move
RGB_STILL = 0.10            # mean |Δgray|; see frames_still for where 0.10 came from

FREE, OCCUPIED, UNKNOWN = 1, 2, 0
# …and a fourth, because FREE was quietly carrying two different claims.
#
# A cell is carved by ray-casting: "nothing stopped the ray before here". Near
# the body that coincides with "there is floor here" — the ground is densely
# sampled and can be seen directly. Far out it does NOT: past roughly 8 m there
# are fewer than MIN_FLOOR_PTS ground points per cell, so the only thing the
# carve establishes is that the SIGHTLINE is clear.
#
# Those are not the same claim, and the difference is exactly where a body gets
# hurt. Negative obstacles — a sunken pool, a stairwell, the edge of a platform —
# sit BELOW the 0.2-1.8 m obstacle band, so nothing stops the ray, and with no
# floor samples to contradict it the drop reads as open ground. This episode's
# instruction is literally "go straight past the pool".
#
# OPEN is that second claim kept separate: you may look through it, reason about
# it, say "that direction runs deep" — but the proposer will not stand a
# waypoint in it, and passable_range stops at its edge. Widening the map without
# this split would have widened the lie.
OPEN = 3


# ── WHAT IN HERE IS SIMULATOR-ONLY ───────────────────────────────────────
# Several numbers below were MEASURED, which makes them trustworthy — and they
# were measured in habitat, which makes them trustworthy ABOUT HABITAT. Listed
# so nobody carries them onto a real robot by accident:
#
#   MATCH_TURN = False   justified by dead-reckoned heading error of 0.00° over
#                        90 steps. Habitat delivers exactly 15° per turn. A
#                        walking quadruped does not, and on hardware this must
#                        be re-argued from the IMU (which may well win it back)
#                        rather than inherited.
#   FRAME_STILL_M        a blocked step in habitat produces BIT-IDENTICAL frames
#   RGB_STILL            (median difference 0.0000). No real sensor does that.
#                        Both thresholds are sim calibrations. On a real Go2 the
#                        question they answer — "did the body actually move?" —
#                        is answered directly by the robot's own odometry/IMU
#                        (see bridges/go2_bridge.py), so this whole detector is
#                        a crutch for something habitat lacks, not a design
#                        element that has to survive transfer.
#   estimate_floor +     both assume the camera is LEVEL. A quadruped pitches
#   OBSTACLE_LO/HI_M     and rolls every gait cycle; the cloud must be rotated
#                        into gravity before the height band is sliced, or the
#                        floor lands in the obstacle band at range. Not handled
#                        here — habitat's camera never tilts.
#   MIN_PTS_PER_CELL = 4 tuned on dense, noiseless depth with exact intrinsics.
#
# What is NOT simulator-specific is the INTERFACE: numbered reachable waypoints
# in egocentric polar, plus sentences like "the pool is behind you on your
# right, about 3 m", plus `trusted` refusing to quote metres once registration
# fails. Nothing above this module knows how the map is built, so on hardware
# the mapper can be replaced wholesale (model_pyslam, or the robot's own
# estimator) without touching the harness.


# ── 0. decoding + units ──────────────────────────────────────────────────
def decode_depth(field_value: Any) -> np.ndarray | None:
    """The wire form env_habitat uses for a float array."""
    if not isinstance(field_value, dict) or "__ndarray__" not in field_value:
        return None
    try:
        arr = np.frombuffer(
            base64.b64decode(field_value["__ndarray__"]),
            dtype=field_value.get("dtype", "float32"),
        )
        return arr.reshape(field_value["shape"]).squeeze()
    except Exception:  # noqa: BLE001 — a sensor read must never kill a move
        return None


def to_metres(depth: np.ndarray,
              scale_m: float | None = None) -> tuple[np.ndarray, bool]:
    """Unit conversion — DECLARED if the caller knows, self-detecting if not.

    ``scale_m`` is metres per raw unit: 1.0 when habitat is configured with
    NORMALIZE_DEPTH False (raw metres on the wire), MAX_DEPTH when it is True
    (the frame is [0,1] across the whole range). A caller that has read the
    env's own config should pass it, because the fallback is a GUESS: it reads
    "max ≤ 1" as normalized, and a body with its nose against a wall produces a
    frame whose every pixel is under a metre. That misfire is a silent 10×,
    and it gets likelier the moment MAX_DEPTH stops being 10 — which is exactly
    what the un-clipped depth rig does. Guess only when nobody knows better.

    A hardcoded ×10 is not an option either: both conventions were observed on
    the same server within an hour (a resolution-override reset rebuilds the
    env and can flip it)."""
    if scale_m is not None:
        return depth.astype(np.float32) * float(scale_m), float(scale_m) != 1.0
    normalized = float(depth.max()) <= 1.0 + 1e-6
    scale = DEPTH_FULL_RANGE_M if normalized else 1.0
    return depth.astype(np.float32) * scale, normalized


# ── 1. depth → egocentric point cloud ────────────────────────────────────
def unproject(depth: np.ndarray, hfov_deg: float = FOV_DEG, *,
              scale_m: float | None = None):
    """(x_right, y_up, z_forward) in metres, camera frame, plus a validity mask.

    Intrinsics are derived from the DEPTH array's own shape — the env's
    `intrinsics` port describes the RGB sensor and is off by the resolution
    ratio."""
    h, w = depth.shape[:2]
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    d, _ = to_metres(depth, scale_m)
    # The far ceiling is a net under the GUESS, not a statement about the world:
    # a metric frame misread as normalized comes out ×10, and dropping those
    # pixels stops a 100 m phantom wall from being rasterised. When the scale
    # was declared there is nothing to catch, and keeping the old 30 m ceiling
    # would silently amputate exactly the long sightlines that removing
    # habitat's 10 m clip exists to reveal.
    ceiling = 1e4 if scale_m is not None else 3.0 * DEPTH_FULL_RANGE_M
    valid = (d > 0.05) & (d < ceiling)
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return (us - cx) * d / f, -(vs - cy) * d / f, d, valid


def estimate_floor(y: np.ndarray, z: np.ndarray, valid: np.ndarray) -> float:
    """Camera-frame height of the floor (negative — it is below the camera).

    Self-calibrated from the point cloud rather than read from a config: a real
    robot has no habitat YAML, and a sensor remount would silently invalidate a
    constant. The floor is the dense low mode of y among nearby points."""
    m = valid & (z < 4.0)
    if int(m.sum()) < 500:
        return -1.25
    ys = y[m]
    lo = float(np.percentile(ys, 0.5))
    low_half = ys[ys < np.percentile(ys, 40)]
    if low_half.size < 100:
        return lo
    hist, edges = np.histogram(low_half, bins=60, range=(lo, lo + 1.0))
    return float(edges[int(np.argmax(hist))] + 0.5 * (edges[1] - edges[0]))


# ── 2. point cloud → top-down free / occupied / unknown ──────────────────
@dataclass
class TopDown:
    """Robot-centred occupancy. Row = forward, col = lateral with LEFT positive
    (matching step_hightolow's counter-clockwise-positive angle). The robot is
    always at (row 0, centre col) facing up: no world frame is ever formed."""

    grid: np.ndarray                       # FREE / OPEN / OCCUPIED / UNKNOWN
    free_range: np.ndarray                 # metres of clear SIGHTLINE per bearing
    bearings: np.ndarray                   # bearing (rad, left +) of each bin
    clearance: np.ndarray                  # metres to the nearest real obstacle
    floor_y: float
    normalized_depth: bool
    # How far this grid reaches, and the distance inside which "no floor points"
    # means the OPTICS, not missing evidence.
    #
    # range_cap_m used to be a parameter that lied. Callers could pass 10 or 15
    # and every consumer downstream — cell_of, passable_range, the Voronoi
    # candidates, the renderer — went on using the module constant for the
    # lateral offset, so the column index was off by half the width and the
    # whole profile silently sheared. It read as a knob and behaved as a
    # decoration. It lives on the object now, and nothing computes a cell from
    # the global any more.
    range_cap_m: float = RANGE_CAP_M
    floor_blind_m: float = 0.0
    # §2.4: floor the camera actually SAW beyond the first occluder — cells
    # that stay UNKNOWN for execution (no verified way in) but carry real
    # evidence that open ground exists back there. Planning value, not
    # walkability: candidates never land here; telemetry and the maps name
    # it as somewhere worth going to LOOK at.
    potential: np.ndarray | None = None

    @property
    def n_fwd(self) -> int:
        return self.grid.shape[0]

    @property
    def n_lat(self) -> int:
        return self.grid.shape[1]

    def cell_of(self, x_left: float, y_fwd: float) -> tuple[int, int]:
        return (int(y_fwd / CELL_M), int((x_left + self.range_cap_m) / CELL_M))

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.n_fwd and 0 <= j < self.n_lat

    def ahead_m(self) -> float:
        return float(self.free_range[len(self.free_range) // 2])

    def widest(self) -> tuple[float, float]:
        """Roomiest bearing, with ties broken toward straight ahead.

        A bare argmax is badly behaved here: in an open room dozens of bearings
        all reach the range cap, and argmax silently returns the FIRST index —
        the extreme edge of the field of view — so the harness would announce
        "the roomiest direction is 45° to your right" about a room that is
        equally open everywhere. Among everything within 5 % of the best, take
        the bearing nearest the current heading."""
        best = float(self.free_range.max())
        near = np.nonzero(self.free_range >= best * 0.95)[0]
        k = int(near[np.argmin(np.abs(self.bearings[near]))])
        return float(self.bearings[k]), float(self.free_range[k])


def build_topdown(depth: np.ndarray, *, hfov_deg: float = FOV_DEG,
                  range_cap_m: float = RANGE_CAP_M,
                  scale_m: float | None = None) -> TopDown:
    """Rasterise obstacles, then carve free space by ray-casting.

    Obstacle points are binned with a DENSITY threshold first — an earlier
    version truncated each bearing at the 2nd percentile of its obstacle ranges
    and a single stray pixel amputated whole bearings, leaving 89 % of the map
    unknown. Everything past a ray's first hit stays UNKNOWN, never FREE:
    occlusion is not evidence of floor, and that distinction is what keeps the
    robot from being sent through a wall."""
    x, y, z, valid = unproject(depth, hfov_deg, scale_m=scale_m)
    floor_y = estimate_floor(y, z, valid)

    n_fwd = int(range_cap_m / CELL_M)
    n_lat = int(2 * range_cap_m / CELL_M)

    height = y - floor_y

    def _bin(sel: np.ndarray) -> np.ndarray:
        """Points passing `sel` → a count per cell, in this grid's own frame."""
        cx_left, cz = -x[sel], z[sel]
        keep = (cz > 0.05) & (cz < range_cap_m) & (np.abs(cx_left) < range_cap_m)
        cx_left, cz = cx_left[keep], cz[keep]
        out = np.zeros((n_fwd, n_lat), dtype=np.int32)
        if cx_left.size:
            ii = np.clip((cz / CELL_M).astype(np.int32), 0, n_fwd - 1)
            jj = np.clip(((cx_left + range_cap_m) / CELL_M).astype(np.int32),
                         0, n_lat - 1)
            np.add.at(out, (ii, jj), 1)
        return out

    occupied = _bin(valid & (height > OBSTACLE_LO_M)
                    & (height < OBSTACLE_HI_M)) >= MIN_PTS_PER_CELL
    # Ground actually SEEN, as opposed to ground merely not contradicted. This
    # is the evidence that separates FREE from OPEN.
    has_floor = _bin(valid & (np.abs(height) < FLOOR_BAND_M)) >= MIN_FLOOR_PTS

    # Right in front of the feet there is no floor to see and never was: the
    # camera looks down at most hfov/2, so the nearest ground it can image is
    # (camera height)/tan(hfov/2) — 1.55 m for this rig. Inside that radius,
    # "no floor points" is the OPTICS, not missing evidence, and marking it OPEN
    # would paint a blind ring around the body that the proposer then refuses to
    # cross. Geometry, not a tuned constant: raise the mast and the ring grows,
    # and this grows with it.
    floor_blind_m = abs(floor_y) / math.tan(math.radians(hfov_deg) / 2.0)
    ci = (np.arange(n_fwd, dtype=np.float32)[:, None] + 0.5) * CELL_M
    cj = (np.arange(n_lat, dtype=np.float32)[None, :] + 0.5) * CELL_M - range_cap_m
    # A STRIP in forward distance, not a disc. The image's bottom ROW cuts the
    # floor at planar z = |floor_y| / tan(vfov/2) for EVERY column, so the
    # unseeable region is "z closer than the blind distance", full width. The
    # old hypot() disc matched that strip only along the centre line; a cell at
    # 30° off axis with z = 1.2 m sits outside the disc, cannot hold a floor
    # pixel, stayed OPEN — and cut the verified-FREE prefix of every oblique
    # bearing at a constant 1.2 m. Straight ahead z equals r, which is why the
    # bug was invisible exactly where everyone looked.
    under_the_nose = ci <= floor_blind_m

    grid = np.full((n_fwd, n_lat), UNKNOWN, dtype=np.uint8)
    half = math.radians(hfov_deg) / 2.0
    n_bins = int(hfov_deg * 2) + 1                     # 0.5° rays
    bearings = np.linspace(-half, half, n_bins)
    # The first obstacle along each bearing, found by asking the OBSTACLES where
    # they are rather than by marching a ray out to meet them.
    #
    # Marching leaks. A ray is sampled at points; a cell is an area; and a ray
    # running diagonally past the corner of a wall slips between two samples and
    # reports open space behind solid geometry. Measured on a plain 2 m corridor:
    # the +45° bearing stopped at the wall, 2.85 m, and the -45° bearing — the
    # same wall, mirrored — ran clean through to 12.00 m, because the sample
    # nearest the corner landed one cell short of where the wall's cells begin.
    # A one-cell asymmetry in a symmetric room, and it scaled with the cap: the
    # escaped ray now paints ten metres of phantom floor instead of four.
    #
    # Every occupied cell instead claims the angular WIDTH it actually subtends
    # (its diagonal seen from the body) and writes its range into every bearing
    # it covers. A cell that covers no bin centre still covers bins — there is
    # no gap for a ray to find, because there are no rays.
    free_range = np.full(n_bins, range_cap_m, dtype=np.float32)
    oi, oj = np.nonzero(occupied)
    if oi.size:
        obx = (oj + 0.5) * CELL_M - range_cap_m
        oby = (oi + 0.5) * CELL_M
        ob = np.arctan2(obx, oby)
        orr = np.hypot(obx, oby)
        # Cells the body is standing on subtend most of the field of view and
        # would blank it; they are also not news — see SELF_FOOTPRINT_M.
        keep = (orr > 0.2) & (np.abs(ob) <= half + 0.05)
        ob, orr = ob[keep], orr[keep]
    if oi.size and ob.size:
        halfw = np.arctan2(CELL_M * 0.7071, np.maximum(orr, CELL_M))
        sb = (n_bins - 1) / (2 * half)
        lo = np.clip(np.floor((ob - halfw + half) * sb), 0, n_bins - 1).astype(np.int32)
        hi = np.clip(np.ceil((ob + halfw + half) * sb), 0, n_bins - 1).astype(np.int32)
        for s in range(int((hi - lo).max()) + 1):
            np.minimum.at(free_range, np.minimum(lo + s, hi), orr)

    # Fill from the PROFILE rather than by painting cells along each ray. Rays
    # are cast every 0.5°, so neighbouring rays sit r·0.0087 apart — 5 cm at 6 m,
    # where they overlap and paint a solid wedge, but 10.5 cm at 12 m, wider than
    # a cell. Past about 6 m the painted region broke into diagonal stripes with
    # unpainted cells between them, which is a sampling artefact of the drawing
    # method and nothing the sensor said. Invisible at the old cap; the first
    # thing you see at the new one.
    #
    # So ask the question the other way round: every cell knows its own bearing
    # and range, and a cell is carved iff it is inside the field of view and
    # nearer than that bearing's first obstacle. No gaps, no double-writes, no
    # dependence on ray order — and vectorised, so it costs less than the loop
    # it replaces.
    cell_b = np.arctan2(cj, ci)
    cell_r = np.hypot(cj, ci)
    idx = np.clip(np.rint((cell_b + half) / (2 * half) * (n_bins - 1)).astype(np.int32),
                  0, n_bins - 1)
    grid[(np.abs(cell_b) <= half) & (cell_r < free_range[idx])] = OPEN
    grid[occupied] = OCCUPIED
    grid[(grid == OPEN) & (has_floor | under_the_nose)] = FREE

    # Clearance measures distance to a REAL obstacle. Counting the unknown
    # frontier as wall starved the proposer — every candidate hugged the middle
    # of the visible wedge and nothing ever reached toward an opening.
    # OPEN is not an obstacle either: it is unproven floor, and the thing that
    # refuses to walk into it is passable_range, not this distance field.
    clearance = distance_transform_edt(grid != OCCUPIED) * CELL_M
    _, normalized = to_metres(depth, scale_m)
    # Floor seen where the carve never reached: glimpsed under a counter's
    # gap, over a table top, through a doorway past the first occluder. The
    # cell stays UNKNOWN — nobody verified a way IN — but the evidence is
    # real and §2.4 wants it kept as POTENTIAL, never laundered into FREE.
    potential = has_floor & (grid == UNKNOWN)
    return TopDown(grid=grid, free_range=free_range, bearings=bearings,
                   clearance=clearance, floor_y=floor_y,
                   normalized_depth=normalized, range_cap_m=float(range_cap_m),
                   floor_blind_m=float(floor_blind_m), potential=potential)


def decode_panorama_depth(view: dict, scale_m: float | None = None
                          ) -> np.ndarray | None:
    """Metres from a panorama view's ``depth_raw_base64``.

    The encoder writes ``uint16 = sensor_value × 1000`` — whatever the SENSOR
    hands it, which is normalized [0,1] on the clipped rig and raw metres on
    the un-clipped one. So divide the ×1000 back out to recover the sensor's
    own value, then apply the SAME declared-or-guessed unit conversion every
    other depth consumer uses. The old hardcoded /100 was the normalized case
    baked in (probed live: max 1000 ⇒ 10 m); it silently mis-scales the
    moment MAX_DEPTH stops being 10 — pass ``scale_m`` from the env's
    depth_units and nothing is guessed."""
    if PILImage is None:
        return None
    b64 = view.get("depth_raw_base64")
    if not b64:
        return None
    try:
        arr = np.asarray(PILImage.open(BytesIO(base64.b64decode(b64))))
    except Exception:  # noqa: BLE001
        return None
    if arr.ndim != 2:
        return None
    metres, _ = to_metres(arr.astype(np.float32) / 1000.0, scale_m)
    return metres


def build_topdown_pano(views: list[dict], *, hfov_deg: float = FOV_DEG,
                       range_cap_m: float = RANGE_CAP_M,
                       scale_m: float | None = None) -> TopDown | None:
    """Fuse several free renders into ONE robot-centred map.

    ``observe_panorama`` is a pure read — `sim.get_observations_at` renders at
    the agent's position without stepping the simulator — so 4 views at 90°
    spacing buy full 360° coverage for ZERO env steps. That matters because a
    frontal-only map runs 80-98 % UNKNOWN in furnished rooms, the proposer
    starves, and a model that keeps being offered nothing learns to ignore it.

    Still depth-only and still predictor-free: the same unprojection, the same
    height band, the same ray carve — just four wedges instead of one, each
    rotated into the robot frame by its own heading. Kept behind a knob so the
    frontal condition remains the ablation baseline."""
    n_fwd = int(range_cap_m / CELL_M)
    n_lat = int(2 * range_cap_m / CELL_M)
    counts = np.zeros((n_fwd, n_lat), dtype=np.int32)
    floors: list[float] = []
    per_view: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for view in views:
        depth = decode_panorama_depth(view, scale_m)
        if depth is None:
            continue
        x, y, z, valid = unproject(depth, hfov_deg)
        floor_y = estimate_floor(y, z, valid)
        floors.append(floor_y)
        # heading_deg is counter-clockwise (dir 3 of 12 = 90° = the robot's
        # LEFT, per the panorama strip's own mapping), matching our
        # left-positive bearing convention.
        yaw = math.radians(float(view.get("heading_deg") or 0.0))
        per_view.append((yaw, x, y, z, valid))

    if not per_view:
        return None
    floor_y = float(np.median(floors))

    for yaw, x, y, z, valid in per_view:
        height = y - floor_y
        obst = valid & (height > OBSTACLE_LO_M) & (height < OBSTACLE_HI_M)
        if not bool(obst.any()):
            continue
        # intra-view left-positive bearing, then rotate by the view's heading
        b = np.arctan2(-x[obst], z[obst]) + yaw
        r = np.hypot(x[obst], z[obst])
        keep = (r > 0.05) & (r < range_cap_m)
        b, r = b[keep], r[keep]
        ox_left, oz = r * np.sin(b), r * np.cos(b)
        # Points BEHIND the robot are dropped, not clipped. The grid's row 0 is
        # the robot's own position, so clipping a negative forward coordinate
        # into it stamps every rear wall directly in front of the agent — the
        # first version did exactly that and produced a map that was 0 % free
        # with "0.05 m ahead". Until the grid is re-centred to span
        # -range..+range, this path fuses the FORWARD halves of several views:
        # wider than one 90° wedge, and honest.
        inside = (np.abs(ox_left) < range_cap_m) & (oz > 0.05) & (oz < range_cap_m)
        ox_left, oz = ox_left[inside], oz[inside]
        if not ox_left.size:
            continue
        ii = np.clip((oz / CELL_M).astype(np.int32), 0, n_fwd - 1)
        jj = np.clip(((ox_left + range_cap_m) / CELL_M).astype(np.int32), 0, n_lat - 1)
        np.add.at(counts, (ii, jj), 1)

    occupied = counts >= MIN_PTS_PER_CELL
    # Rays now sweep the full circle, but the grid only extends forward, so the
    # carve still runs over the forward half — behind-the-robot obstacles are
    # kept as occupancy (they inform clearance) without inventing floor there.
    grid = np.full((n_fwd, n_lat), UNKNOWN, dtype=np.uint8)
    n_bins = 361
    bearings = np.linspace(-math.pi / 2, math.pi / 2, n_bins)
    free_range = np.full(n_bins, range_cap_m, dtype=np.float32)
    radii = np.arange(CELL_M / 2, range_cap_m, CELL_M / 2)
    for k, bearing in enumerate(bearings):
        xs = radii * math.sin(bearing)
        ys = radii * math.cos(bearing)
        ii = (ys / CELL_M).astype(np.int32)
        jj = ((xs + range_cap_m) / CELL_M).astype(np.int32)
        ok = (ii >= 0) & (ii < n_fwd) & (jj >= 0) & (jj < n_lat)
        ii, jj, rr = ii[ok], jj[ok], radii[ok]
        if not ii.size:
            continue
        hit = np.nonzero(occupied[ii, jj])[0]
        stop = int(hit[0]) if hit.size else len(ii)
        grid[ii[:stop], jj[:stop]] = FREE
        if hit.size:
            grid[ii[stop], jj[stop]] = OCCUPIED
            free_range[k] = float(rr[stop])
    grid[occupied] = OCCUPIED
    clearance = distance_transform_edt(grid != OCCUPIED) * CELL_M
    # No FREE/OPEN split on this path. The panorama fuses four wedges whose
    # ground samples arrive at four different headings, and a floor-evidence
    # count per cell across them is a separate piece of work; claiming the split
    # here without doing that work would mark most of a perfectly good panorama
    # OPEN and starve the proposer. Frontal is the arm that got the split.
    return TopDown(grid=grid, free_range=free_range, bearings=bearings,
                   clearance=clearance, floor_y=floor_y, normalized_depth=True,
                   range_cap_m=float(range_cap_m))


def frames_still(prev: np.ndarray | None, now: np.ndarray | None,
                 tol: float = FRAME_STILL_M, *, prev_rgb: Any = None,
                 rgb: Any = None, rgb_tol: float = RGB_STILL,
                 scale_m: float | None = None) -> bool:
    """Did the body actually move between these two depth frames?

    The cheapest useful proprioception there is, and it closes the LARGEST hole
    in the map. Measured over 168 commanded forwards on three R2R episodes
    (habitat pose used only to score it): 60 of them moved the body zero — one
    episode blocked 40 of 56 — and the harness credited 0.25 m for every one.
    That is not noise, it is a ONE-SIDED BIAS: a collision can only ever make a
    step short, never long, so the error grows monotonically with the number of
    collisions and no amount of averaging removes it.

    Mean absolute depth difference separates the two cases well: blocked steps
    peaked at 0.0017 (median 0.0000 — bit-identical frames) while real steps ran
    a median of 0.0124.

    DEPTH ALONE IS NOT ENOUGH, and the reason is geometric, not a threshold to
    tune. ``ALLOW_SLIDING`` is True in R2R-CE (vlnce_task.yaml:11), so a forward
    pressed against an angled wall slides the body sideways — and a flat wall
    looks IDENTICAL in depth from every point along it. That is the aperture
    problem: the one motion depth cannot see is the one sliding produces.
    Measured over 168 forwards: 8 steps moved the body a median of 0.169 m while
    depth called them still, erasing 1.40 m of real travel. An earlier version of
    this docstring called those false alarms "benign"; the measurement says they
    are not.

    So the RGB frame gets a vote too, and BOTH must agree before a step is
    called still. The wall is flat in depth but textured in RGB, and on the
    slide we caught, RGB moved 0.234 — eight times the largest value ever seen
    on a step that truly did not move (0.0292). Measured on the same 168 steps:

        depth only   6/60 blocked missed · 1 real step erased (0.12 m)
        RGB only     0/60 blocked missed · 2 real steps erased (0.46 m)
        BOTH         6/60 blocked missed · 0 real steps erased

    WHERE 0.10 COMES FROM, and why the first value was wrong. The veto was first
    set at 0.03 — the TOP of the "did not move" distribution — which is the worst
    place to put it: it maximises the number of genuinely blocked steps that get
    called "moved", and every one of those leaves the bias in. Scored against
    habitat's pose on two action plans (6 episodes, 348 forwards), by mean
    position error at the end:

        plan            depth only    ∧RGB 0.03    ∧RGB 0.10    ∧RGB 0.15
        turn-heavy        0.26 m       0.42 m       0.26 m       0.26 m
        wall-grinding     0.96 m       1.03 m       1.03 m       1.03 m

    The slide sits at 0.234 and the largest true-still value at 0.029, so the
    band between them is empty and the midpoint keeps the catch for free: 0.10
    costs nothing against depth alone where no slide occurs, and still removes
    the erasure where one does. Putting a threshold at the edge of a
    distribution rather than in the gap is what made the earlier value expensive.

    The intersection costs nothing against depth alone on the miss side and
    removes the erasure entirely — and erasing real travel is the worse error,
    because a missed block leaves a bias the scan matcher can still argue with,
    while an erased slide is a lie the map has no way to detect."""
    if prev is None or now is None or prev.shape != now.shape:
        return False
    a, _ = to_metres(prev, scale_m)
    b, _ = to_metres(now, scale_m)
    m = (a > 0.05) & (b > 0.05)
    if not m.any():
        return False
    if float(np.abs(a[m] - b[m]).mean()) > tol:
        return False
    if prev_rgb is not None or rgb is not None:
        return view_still(prev_rgb, rgb, rgb_tol)
    return True


def _gray256(rgb_b64: Any) -> np.ndarray | None:
    """A base64 PNG as a small grayscale array, for frame-to-frame comparison."""
    if PILImage is None or not isinstance(rgb_b64, str) or not rgb_b64:
        return None
    try:
        img = PILImage.open(BytesIO(base64.b64decode(rgb_b64))).convert("L")
    except Exception:  # noqa: BLE001
        return None
    return np.asarray(img.resize((256, 256)), dtype=np.float32) / 255.0


def view_still(prev_rgb: Any, rgb: Any, tol: float = RGB_STILL) -> bool:
    """The RGB half of the did-I-move test — see `frames_still`.

    Deliberately the crudest statistic that works (mean absolute difference on a
    256² grayscale). Phase correlation gives a direction too and separated the
    same cases, but a direction we do not yet use is not worth the FFT."""
    a, b = _gray256(prev_rgb), _gray256(rgb)
    if a is None or b is None:
        return False          # unknown is never "still"
    return float(np.abs(a - b).mean()) <= tol


def project_mask(mask: np.ndarray, depth: np.ndarray, floor_y: float | None = None,
                 *, hfov_deg: float = FOV_DEG, max_m: float | None = None,
                 scale_m: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A segmentation mask + the same depth frame → its footprint on the floor.

    This is the whole bridge between "SAM says these pixels are a bar counter"
    and "the map knows a bar counter stands there". The mask is resized to the
    DEPTH grid by nearest neighbour — SAM sees the RGB frame (1024²) and depth
    is 512², and indexing one with the other's coordinates is a silent shear.
    Returns body-frame ``(x_left, y_fwd)`` metres, empty if nothing survives.

    ``max_m`` DEFAULTS TO NO LIMIT, and used to default to RANGE_CAP_M — the
    OCCUPANCY grid's 6 m edge, which has no business bounding what a landmark
    layer may remember. Measured on R2R ep 7 with the un-clipped rig: SAM found
    three doors at 11.7, 13.7 and 15.4 m, all in the one bearing the depth map
    called open, and all three were thrown away on this line while the map kept
    "chairs at 2.8 m". Those doors are the far end of the corridor the
    instruction names — they are the "where to go", and remembering one cannot
    walk the body into a wall the way a wrong FREE cell can.

    Two things guarded against mask spill here and only ONE of them was doing
    the work. Segmentation masks leak — a few pixels of a "bar counter" land on
    the wall ten metres behind it — but the trim below cuts relative to THIS
    mask's own depth distribution, so it scales with however far the object is:
    a door at 13 m keeps its own surface and drops what shows through the
    doorway. The absolute cut was never the spill guard, only a range limit.

    The real ceiling is now AnchorMap's own extent (ANCHOR_HALF_M, 12 m around
    the body): a sighting past that lands outside the grid and is dropped when
    stamped. So the 15.4 m door still waits until the body is a few metres
    closer. That is the accumulated map's window, a separate decision from this
    one, and it is not silently widened here."""
    if mask is None or depth is None:
        return np.empty(0), np.empty(0)
    dh, dw = depth.shape[:2]
    if mask.shape[:2] != (dh, dw):
        ys = (np.arange(dh) * mask.shape[0] / dh).astype(int).clip(0, mask.shape[0] - 1)
        xs = (np.arange(dw) * mask.shape[1] / dw).astype(int).clip(0, mask.shape[1] - 1)
        mask = mask[np.ix_(ys, xs)]
    x, y, z, valid = unproject(depth, hfov_deg, scale_m=scale_m)
    if floor_y is None:
        floor_y = estimate_floor(y, z, valid)
    height = y - floor_y
    sel = (mask.astype(bool) & valid & (z > 0.05)
           & (height > 0.05) & (height < 2.4))
    if max_m is not None:
        sel &= z < max_m
    if not sel.any():
        return np.empty(0), np.empty(0)
    # Trim the tail. Segmentation masks leak: a few pixels of a "bar counter"
    # mask land on the wall ten metres behind it, and unfiltered they stretch
    # the landmark's footprint across the whole room. What a single view can
    # honestly claim is the surface it can see, so keep the near body of the
    # depth distribution and drop what trails off behind it.
    # THIS is the spill guard, and it is relative — see the docstring.
    zs = z[sel]
    cut = float(np.percentile(zs, 60)) + 0.5
    keep = zs <= cut
    return -x[sel][keep], zs[keep]


# ── 2b. many wedges → ONE map that holds still ───────────────────────────
# How much of the walk the map can hold. NOT the same knob as FUSE_MAX_M below,
# and the two were once described together as "widening the accumulated map",
# which is wrong in a way worth spelling out:
#
#   ANCHOR_HALF_M  how far around the body the grid extends. _recentre slides
#                  the array under the body and ZEROES what falls off the far
#                  edge, so this is simply how long a thing stays remembered.
#                  Widening it costs memory and nothing else — no frame's
#                  evidence changes, the same cells hold the same values, they
#                  just stop being thrown away so soon. Measured: 20 m half-width
#                  is 400² cells, 0.64 MB a layer, register 2.5 → 3.3 ms.
#
#   FUSE_MAX_M     how far out a SINGLE frame is allowed to stamp. Widening THIS
#                  is the risky one: a wall seen 10 m away is written wherever
#                  dead reckoning currently believes the body to be, so pose
#                  error lands directly in the wall's position and stays there.
#
# 12 m was also exactly where landmark memory kept getting cut off — SAM finds
# doors at 13-15 m and stamp_semantic dropped them for falling outside the grid.
# R2R trajectories run ~10 m, so 20 m of half-width holds a whole episode
# without forgetting its own start.
ANCHOR_HALF_M = 20.0
# The accumulated map is drawn at least this far around the body, however
# little has been mapped yet. It is the picture a person actually studies —
# a whole walk legible at once — and a crop tight to three steps of evidence
# rendered it smaller than the single-frame top-down sitting next to it.
MAP_MIN_HALF_M = 7.0
ANCHOR_GROW_M = 3.0          # re-centre before the body gets this close to the rim
LOGODDS_CLIP = 6.0
HIT_W, MISS_W = 2.0, 0.8     # an obstacle return outweighs an empty ray
FAR_W = 0.45                 # evidence past FAR_M counts less — it is one pixel wide
FAR_M = 3.5
FUSE_MAX_M = 4.5             # …and past THIS it is not stamped at all
TRAVERSED_FREE = -3.0        # you WALKED there; no later reading may call it wall
POTENTIAL_MAX_M = 10.0       # §2.4 hints may land this far out — advisory only
MATCH_SIGMA_M = 0.35         # width of the likelihood field's basin
MATCH_MAX_SHIFT_M = 0.30     # a correction, not a re-localisation
MATCH_MAX_TURN_DEG = 6.0
MATCH_MIN_POINTS = 40
MATCH_MIN_SCORE = 0.12       # below this the wedge matches nothing
MATCH_TURN = False           # measured harmful — see register()
SEM_MIN_CONF = 0.45          # one detection at SAM's own floor; see _sem_mask
TRUST_MIN_SCORE = 0.35       # below this the map stops quoting metres


class AnchorMap:
    """The accumulated map: every wedge stamped into ONE frame that holds still.

    The previous design kept the grid glued to the body ("the robot never moves —
    the MAP moves under it") and paid for it three times over. Convicted with a
    two-scene test before this rewrite:

      * every 15° turn ran ``scipy.ndimage.rotate`` over the whole float grid and
        every step ran a sub-pixel ``shift``. Both RESAMPLE. Interpolating a map
        into itself a hundred times is a low-pass filter with the robot's own
        path as its schedule: thin walls smear, then dissolve. That is the "漂移
        一下就没了" — most of it was never drift at all, it was blur.
      * the render flipped left-right on top of a grid whose columns were ALREADY
        in image order, so the accumulated map came out mirrored against the
        per-frame top-down that feeds it.
      * the trail rotated by ``+θ`` while the grid rotated by ``−θ``, so walls and
        footsteps swung opposite ways on every turn, and a ``(0,0)`` was appended
        on turns too, burying the real history under duplicate origins.

    So: pick an origin at the first look and never move it. Fusion transforms a
    few thousand POINTS (exact, ~0.3 ms) instead of resampling 57 600 cells. The
    body's pose lives in that frame, is dead-reckoned from the harness's own
    commands, and is corrected by matching each new wedge against what is already
    on the map — no localisation service, no simulator pose, nothing external.

    ON THE COORDINATE RULE. This class holds an (x, y, θ). It is the organ's
    private bookkeeping, born at the episode's first frame and dead at its last;
    it is not the world's frame, it cannot be compared across episodes, and it
    NEVER reaches the model — every sentence this map produces is egocentric
    ("the counter is 4 m behind you on your left"). A person walking a corridor
    keeps exactly this much: where they started relative to where they now stand.
    """

    def __init__(self, half_m: float = ANCHOR_HALF_M, cell_m: float = CELL_M):
        self.cell = cell_m
        self.half = half_m
        self.n = int(2 * half_m / cell_m)
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        # §2.4 POTENTIAL votes: floor glimpsed beyond an occluder. Real
        # evidence (a fused FREE/OCCUPIED reading, or standing there)
        # retires the guess — see fuse().
        self.potential = np.zeros((self.n, self.n), dtype=np.float32)
        self.sem: dict[str, np.ndarray] = {}     # phrase → confidence grid
        # pose of the body in the anchor frame: x right, y forward-at-birth,
        # theta counter-clockwise-positive (LEFT), all metres/radians.
        self.px = self.py = self.theta = 0.0
        self.ox = self.oy = 0.0                  # anchor point at the array centre
        self.trail: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        self.updates = 0
        self.dropped = 0
        self.fixes = 0                           # registrations that moved the pose
        self.last_score = 0.0
        self.last_fix = (0.0, 0.0, 0.0)
        self._field: np.ndarray | None = None    # cached likelihood field
        # §21.2 trust provenance: a same-pose fan sweep fused N wedges from
        # ONE known pose — that geometry needs no registration score to be
        # planning-usable, but pushing `updates` past the young-map grace
        # made `trusted` false exactly then. The flag says WHY the map is
        # trusted; it expires the moment the body actually translates
        # (odometry forward), after which registration is the honest source.
        self.sweep_trust = False

    # ── frame plumbing ───────────────────────────────────────────────────
    @property
    def centre(self) -> int:
        return self.n // 2

    def reset(self) -> None:
        self.logodds[:] = 0.0
        self.visited[:] = False
        self.potential[:] = 0.0
        self.sem.clear()
        self.px = self.py = self.theta = 0.0
        self.ox = self.oy = 0.0
        self.trail = [(0.0, 0.0, 0.0)]
        self.updates = 0
        self._field = None
        self.sweep_trust = False
        self.dropped += 1

    def to_anchor(self, x_left, y_fwd, pose=None):
        """Body-frame (left, forward) metres → anchor-frame (x, y) metres.

        At birth the body faces +y with its left toward −x, so a local point is
        ``(-x_left, y_fwd)`` rotated by the heading and offset by the position."""
        px, py, th = pose if pose is not None else (self.px, self.py, self.theta)
        u, v = -np.asarray(x_left, dtype=np.float64), np.asarray(y_fwd, dtype=np.float64)
        c, s = math.cos(th), math.sin(th)
        return u * c - v * s + px, u * s + v * c + py

    def to_body(self, x, y):
        """Anchor-frame → body-frame (left, forward). The inverse of `to_anchor`."""
        dx, dy = np.asarray(x) - self.px, np.asarray(y) - self.py
        c, s = math.cos(self.theta), math.sin(self.theta)
        u, v = dx * c + dy * s, -dx * s + dy * c
        return -u, v

    def cells(self, x, y):
        """Anchor metres → (row, col). Row up = +y, col right = +x, so the array
        is ALREADY in image orientation and the renderer must not flip it."""
        i = np.rint(self.centre - (np.asarray(y) - self.oy) / self.cell).astype(int)
        j = np.rint(self.centre + (np.asarray(x) - self.ox) / self.cell).astype(int)
        return i, j

    # ── ego-motion: dead reckoning, then a correction earned from the map ─
    def odometry(self, turn_rad: float, forward_m: float) -> None:
        """One commanded primitive. Turn first, then walk — the order
        ``step_hightolow`` itself uses."""
        self.theta += turn_rad
        if abs(forward_m) > 1e-9:
            self.px += forward_m * -math.sin(self.theta)
            self.py += forward_m * math.cos(self.theta)
            # a translation ends the same-pose sweep guarantee (§21.2) —
            # from here trust must be re-earned by registration. Turns do
            # not: the pose the sweep was taken from has not moved.
            self.sweep_trust = False
        self._note_pose()

    def retract(self, forward_m: float) -> None:
        """Take back metres the body was commanded to walk but never did.

        NOT the same as `odometry(0, -d)`, and the difference is not cosmetic:
        an undo is a correction of the last command, not a step backwards. Going
        through odometry() appended a second footstep, so the trail drew a spike
        out and back and the walked-distance readout counted a blocked step
        TWICE — the Human tab showed 14.2 m walked for 7.5 m of commands, which
        is how this was caught."""
        if forward_m <= 0:
            return
        self.px -= forward_m * -math.sin(self.theta)
        self.py -= forward_m * math.cos(self.theta)
        self._note_pose(footstep=False)
        # …and drop the footprint the cancelled step left behind, or a jammed
        # robot accumulates one duplicate trail point per blocked command.
        if len(self.trail) > 1 and math.hypot(
                self.trail[-1][0] - self.trail[-2][0],
                self.trail[-1][1] - self.trail[-2][1]) < 0.05:
            self.trail.pop()

    def _note_pose(self, footstep: bool = True) -> None:
        pose = (self.px, self.py, self.theta)
        # Record a footstep only when the BODY moved, and only when the move was
        # a COMMAND. The old map appended one per call, so a 90° turn buried six
        # duplicate origins; and a registration nudge is not a step — counting
        # those made a 5.5 m walk report as 8.5 m.
        if footstep and (not self.trail or math.hypot(
                pose[0] - self.trail[-1][0], pose[1] - self.trail[-1][1]) > 0.05):
            self.trail.append(pose)
        else:
            self.trail[-1] = pose
        if len(self.trail) > 2000:
            del self.trail[:-2000]
        self._stamp_traversed()
        if (abs(self.px - self.ox) > self.half - ANCHOR_GROW_M
                or abs(self.py - self.oy) > self.half - ANCHOR_GROW_M):
            self._recentre()

    def _stamp_traversed(self) -> None:
        """The disc the body occupies is floor, by proof of having stood on it."""
        i, j = self.cells(self.px, self.py)
        r = int(round(ROBOT_RADIUS_M / self.cell))
        i0, i1 = max(0, int(i) - r), min(self.n, int(i) + r + 1)
        j0, j1 = max(0, int(j) - r), min(self.n, int(j) + r + 1)
        if i0 < i1 and j0 < j1:
            self.visited[i0:i1, j0:j1] = True
            np.minimum(self.logodds[i0:i1, j0:j1], TRAVERSED_FREE,
                       out=self.logodds[i0:i1, j0:j1])

    def _recentre(self) -> None:
        """Slide the array back under the body by WHOLE cells — lossless, unlike
        the sub-pixel shift the rolling map ran on every single step."""
        i, j = self.cells(self.px, self.py)
        di, dj = self.centre - int(i), self.centre - int(j)
        if di == 0 and dj == 0:
            return
        for name in ("logodds", "visited", "potential"):
            setattr(self, name, np.roll(getattr(self, name), (di, dj), axis=(0, 1)))
        for phrase in self.sem:
            self.sem[phrase] = np.roll(self.sem[phrase], (di, dj), axis=(0, 1))

        def _blank(arr):
            if di > 0:
                arr[:di, :] = 0
            elif di < 0:
                arr[di:, :] = 0
            if dj > 0:
                arr[:, :dj] = 0
            elif dj < 0:
                arr[:, dj:] = 0

        _blank(self.logodds)
        _blank(self.visited)
        _blank(self.potential)
        for grid in self.sem.values():
            _blank(grid)
        # The array moved by (di, dj) cells, so the anchor point sitting at the
        # array centre moved with it — and it moved the OTHER way in metres,
        # because a row index grows downward while y grows up. Getting these two
        # signs wrong made the origin run away from the body instead of catching
        # up with it, and the next `cells()` indexed nine billion rows out.
        self.ox -= dj * self.cell
        self.oy += di * self.cell
        self._field = None

    # ── registration: correct the odometry against the map itself ────────
    def _likelihood(self) -> np.ndarray | None:
        """exp(−distance-to-nearest-wall / σ): a smooth field whose peak sits on
        the walls, so a wedge slid over it has a gradient to climb."""
        if self._field is not None:
            return self._field
        occ = self.logodds > 1.0
        if int(occ.sum()) < MATCH_MIN_POINTS:
            return None
        d = distance_transform_edt(~occ) * self.cell
        self._field = np.exp(-d / MATCH_SIGMA_M).astype(np.float32)
        return self._field

    @staticmethod
    def _wedge_points(td: TopDown) -> tuple[np.ndarray, np.ndarray]:
        """Occupied cell centres of one wedge, in body-frame metres."""
        rows, cols = np.nonzero(td.grid == OCCUPIED)
        half = td.n_lat * CELL_M / 2.0
        y = (rows + 0.5) * CELL_M
        x = (cols + 0.5) * CELL_M - half
        # Match like with like: points the map is never allowed to hold cannot
        # help align to it, they only add a constant to every candidate score.
        near = np.hypot(x, y) <= FUSE_MAX_M
        x, y = x[near], y[near]
        if x.size > 1500:                        # a subsample matches just as well
            k = np.linspace(0, x.size - 1, 1500).astype(int)
            x, y = x[k], y[k]
        return x, y

    def _score(self, field: np.ndarray, u: np.ndarray, v: np.ndarray,
               px: float, py: float, th: float) -> float:
        c, s = math.cos(th), math.sin(th)
        X, Y = u * c - v * s + px, u * s + v * c + py
        i = np.rint(self.centre - (Y - self.oy) / self.cell).astype(int)
        j = np.rint(self.centre + (X - self.ox) / self.cell).astype(int)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        if not ok.any():
            return 0.0
        return float(field[i[ok], j[ok]].sum()) / float(u.size)

    def register(self, td: TopDown, *, allow_turn: bool = MATCH_TURN,
                 translation_expected: bool = True) -> float:
        """Slide/rotate the new wedge a little to sit best on the accumulated map.

        This is what replaces "detect drift and throw the memory away". Habitat's
        error is not random — it is a collision that ate part of a commanded step,
        so the wedge is displaced by a few centimetres in a consistent direction
        and the map itself can say by how much. The search is BOUNDED
        (±30 cm, ±6°): beyond that a scene has genuinely changed and pretending
        otherwise is how a map invents a corridor. Returns the match quality in
        [0,1]; 0 means "no opinion, odometry stands".

        ROTATION IS OFF BY DEFAULT, and that was measured, not assumed. Driving
        three R2R episodes against habitat's own pose (as a RULER — it never
        enters the harness) showed dead-reckoned heading error of exactly 0.00°
        over 90 steps in all three: habitat's turns are exact, so heading is the
        one quantity odometry already gets perfectly right. Letting the matcher
        rotate could therefore only ever corrupt it, and it did — 15.5°, 2.0°
        and 13.0° of heading error, present in NO other arm. Searching a degree
        of freedom that is already correct is not conservative, it is a leak.

        TRANSLATION IS GATED THE SAME WAY (§21.5). Habitat's pure turns are
        exact AND leave the body where it stood — yet the matcher could
        still hand a turn-only frame up to ±30 cm of px/py "correction",
        mistaking parallax for body motion. The CALLER knows whether the
        commanded primitive could have translated; a pure turn (2/3, or a
        goto micro-yaw) passes translation_expected=False and the pose is
        READ-ONLY: the score is still measured (trust telemetry stays
        honest) but nothing moves. Never inferred from delta_m == 0 — a
        collision-eaten forward also reports zero displacement yet the body
        may genuinely have slid."""
        field = self._likelihood()
        if field is None or self.updates < 3:
            return 0.0
        x, y = self._wedge_points(td)
        if x.size < MATCH_MIN_POINTS:
            return 0.0
        u, v = -x.astype(np.float64), y.astype(np.float64)
        base = self._score(field, u, v, self.px, self.py, self.theta)
        if not translation_expected:
            # read-only registration: measure, report, touch nothing —
            # commanded heading is exact (measured, see above) and the body
            # did not translate, so there is no error for a fix to correct
            self.last_score, self.last_fix = base, (0.0, 0.0, 0.0)
            return base

        best = (base, 0.0, 0.0, 0.0)
        span_m, span_r = MATCH_MAX_SHIFT_M, math.radians(MATCH_MAX_TURN_DEG)
        for steps_m, steps_r in ((5, 4), (3, 3)):        # coarse, then fine
            dxs = np.linspace(-span_m, span_m, 2 * steps_m + 1)
            dys = dxs
            dts = (np.linspace(-span_r, span_r, 2 * steps_r + 1) if allow_turn
                   else np.zeros(1))
            cx, cy, ct = best[1], best[2], best[3]
            for dt in dts:
                for dx in dxs:
                    for dy in dys:
                        sc = self._score(field, u, v, self.px + cx + dx,
                                         self.py + cy + dy, self.theta + ct + dt)
                        if sc > best[0]:
                            best = (sc, cx + dx, cy + dy, ct + dt)
            span_m, span_r = span_m / 4.0, span_r / 4.0

        score, dx, dy, dt = best
        moved = math.hypot(dx, dy)
        if (score < MATCH_MIN_SCORE or score <= base * 1.02
                or moved > MATCH_MAX_SHIFT_M + 1e-6):
            self.last_score, self.last_fix = score, (0.0, 0.0, 0.0)
            return score
        self.px, self.py, self.theta = self.px + dx, self.py + dy, self.theta + dt
        self.last_score, self.last_fix = score, (dx, dy, math.degrees(dt))
        self.fixes += 1
        self._note_pose(footstep=False)
        return score

    # ── fusion: stamp the wedge, exactly, at the pose ────────────────────
    def fuse(self, td: TopDown) -> None:
        rows, cols = np.nonzero(td.grid != UNKNOWN)
        if not rows.size:
            return
        half = td.n_lat * CELL_M / 2.0
        y = (rows + 0.5) * CELL_M
        x = (cols + 0.5) * CELL_M - half
        # The wedge may reach 12 m; the MEMORY only keeps what was seen close
        # enough to be worth keeping. Out there one depth pixel is wider than a
        # grid cell, so far evidence lands as speckle that a later pass has to
        # argue with — and the user's ask was a good map of where we walked, not
        # a blurry one of everywhere we glanced. This bound guards POSE error,
        # not optics: a wall stamped from 10 m away lands wherever the dead
        # reckoning currently thinks the body is.
        near = np.hypot(x, y) <= FUSE_MAX_M
        rows, cols, x, y = rows[near], cols[near], x[near], y[near]
        if not rows.size:
            return
        X, Y = self.to_anchor(x, y)
        i, j = self.cells(X, Y)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        i, j = i[ok], j[ok]
        vals = td.grid[rows[ok], cols[ok]]
        # Far evidence is thin evidence: at 5 m one depth pixel spans ~4 cm and a
        # single grazing ray can paint a wall. Weight it down instead of trusting
        # it equally, which is what made distant furniture flicker in and out.
        w = np.where(np.hypot(x[ok], y[ok]) > FAR_M, FAR_W, 1.0).astype(np.float32)
        # OPEN cells are stamped as NEITHER, and that is a decision, not an
        # oversight. The accumulated map is what the harness later calls "where
        # I have been and what is around me"; writing unproven floor into it as
        # free space would launder a sightline into a memory of walkable ground,
        # and by the time it is read back the distinction is gone. A cell earns
        # its way in by having been seen as floor — or walked on.
        occ, free = vals == OCCUPIED, vals == FREE
        np.add.at(self.logodds, (i[occ], j[occ]), HIT_W * w[occ])
        np.add.at(self.logodds, (i[free], j[free]), -MISS_W * w[free])
        np.clip(self.logodds, -LOGODDS_CLIP, LOGODDS_CLIP, out=self.logodds)
        # No global decay. The old map multiplied EVERY cell by 0.995 per look,
        # so a room seen 200 looks ago had faded to a third of its evidence and
        # the far end of the walk quietly disappeared. Nothing about a wall
        # becomes less true because the robot walked on.
        np.minimum(self.logodds, TRAVERSED_FREE, where=self.visited,
                   out=self.logodds)
        # §2.4 POTENTIAL layer: floor glimpsed beyond an occluder votes into
        # its own grid — advisory only, never walked, so it may reach further
        # than structural fusion (pose error there misplaces a hint, not a
        # wall). Any REAL reading retires the guess for that cell.
        if td.potential is not None and bool(td.potential.any()):
            pr, pc = np.nonzero(td.potential)
            py_ = (pr + 0.5) * CELL_M
            px_ = (pc + 0.5) * CELL_M - half
            near_p = np.hypot(px_, py_) <= POTENTIAL_MAX_M
            if bool(near_p.any()):
                PX, PY = self.to_anchor(px_[near_p], py_[near_p])
                pi, pj = self.cells(PX, PY)
                okp = (pi >= 0) & (pi < self.n) & (pj >= 0) & (pj < self.n)
                np.add.at(self.potential, (pi[okp], pj[okp]), 1.0)
                np.clip(self.potential, 0.0, 20.0, out=self.potential)
        self.potential[(np.abs(self.logodds) > 0.5) | self.visited] = 0.0
        self.updates += 1
        self._field = None

    def disagreement(self, td: TopDown) -> float:
        """Metres of contradiction between what the map predicts along each
        bearing and what was just measured. Now a REPORT, not a trigger: with
        registration in place, a spike means the scene changed (a door opened, a
        person walked past), and the map should absorb it rather than commit
        suicide."""
        if self.updates < 3:
            return 0.0
        diffs = []
        occ = self.logodds > 1.0
        # Both sides must be read out to the SAME range or the comparison is
        # between two different questions: `measured` now runs to the wedge's
        # own cap, so the prediction has to as well.
        cap = float(td.range_cap_m)
        for k in range(0, len(td.bearings), max(1, len(td.bearings) // 24)):
            b = float(td.bearings[k])
            measured = float(td.free_range[k])
            pred = cap
            for r in np.arange(0.3, cap, self.cell):
                X, Y = self.to_anchor(r * math.sin(b), r * math.cos(b))
                i, j = self.cells(X, Y)
                if not (0 <= i < self.n and 0 <= j < self.n):
                    break
                if occ[i, j]:
                    pred = float(r)
                    break
            if pred < cap - 1e-6 or measured < cap - 1e-6:
                diffs.append(abs(pred - measured))
        return float(np.median(diffs)) if diffs else 0.0

    # ── the semantic layer ───────────────────────────────────────────────
    def stamp_semantic(self, phrase: str, x_left, y_fwd, weight: float = 1.0) -> None:
        """Paint a named thing onto the map at the body-frame points given.

        The map already knows a wall is there; this records that the wall is a
        *bar counter*. It is what turns "somewhere behind me" into "the counter I
        passed", which is the evidence the milestone judge kept having to guess."""
        x_left = np.atleast_1d(np.asarray(x_left, dtype=np.float64))
        y_fwd = np.atleast_1d(np.asarray(y_fwd, dtype=np.float64))
        if not x_left.size:
            return
        X, Y = self.to_anchor(x_left, y_fwd)
        i, j = self.cells(X, Y)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        if not ok.any():
            return
        grid = self.sem.setdefault(phrase, np.zeros((self.n, self.n), np.float32))
        # ONE vote per cell per view. Adding once per projected pixel made
        # confidence a function of how many pixels happened to land in a cell —
        # i.e. of how close the object was — so a big near blob outvoted a thing
        # seen clearly from twenty positions. What the layer should measure is
        # AGREEMENT ACROSS VIEWS, which is what survives mask spill.
        flat = np.unique(i[ok] * self.n + j[ok])
        grid.reshape(-1)[flat] += float(weight)
        np.clip(grid, 0.0, 40.0, out=grid)

    @staticmethod
    def _sem_mask(grid: np.ndarray, min_conf: float) -> np.ndarray:
        """Cells that carry this phrase, relative to its own best evidence.

        An absolute cut cannot work: after forty looks the mask's spill has been
        stamped often enough to pass any fixed threshold, and the landmark grows
        until it covers the room. A relative one keeps the part of the blob that
        repeated views agreed on and drops the fringe that appeared once."""
        top = float(grid.max())
        if top <= 0:
            return np.zeros_like(grid, dtype=bool)
        return grid >= max(min_conf, 0.35 * top)

    def semantic_recall(self, min_conf: float = SEM_MIN_CONF) -> list[dict]:
        """Where each named thing is NOW, relative to the body — including the
        ones behind it. Egocentric on the way out: the model never learns that
        the map has an origin."""
        out = []
        for phrase, grid in self.sem.items():
            ii, jj = np.nonzero(self._sem_mask(grid, min_conf))
            if not ii.size:
                continue
            w = grid[ii, jj]
            X = self.ox + (jj - self.centre) * self.cell
            Y = self.oy - (ii - self.centre) * self.cell
            cx = float(np.average(X, weights=w))
            cy = float(np.average(Y, weights=w))
            x_left, y_fwd = self.to_body(cx, cy)
            x_left, y_fwd = float(x_left), float(y_fwd)
            bx, by = self.to_body(X, Y)
            near = float(np.min(np.hypot(bx, by)))
            out.append({
                "phrase": phrase,
                "bearing": math.atan2(x_left, y_fwd),
                "distance": math.hypot(x_left, y_fwd),
                "nearest": near,
                "cells": int(ii.size),
                "behind": y_fwd < -0.3,
            })
        return sorted(out, key=lambda d: d["distance"])

    def recall_sentence(self, min_conf: float = SEM_MIN_CONF,
                        exclude: set[str] | None = None) -> str:
        """The line for things the robot has seen but cannot see NOW.

        Anything the detector is reporting in this very frame is excluded: the
        live sighting sentence already says where it is, and saying it twice —
        once as perception, once as "what you walked past" — reads as two
        different objects and invites the model to count them twice."""
        parts = []
        for r in self.semantic_recall(min_conf):
            if exclude and r["phrase"] in exclude:
                continue
            deg = math.degrees(r["bearing"])
            if r["behind"]:
                side = "behind you" if abs(deg) > 155 else (
                    f"behind you on your {'left' if deg > 0 else 'right'}")
            elif abs(deg) < 15:
                side = "straight ahead"
            else:
                side = f"to your {'left' if deg > 0 else 'right'}"
            # A metre from a map that has stopped registering is a CONFIDENTLY
            # WRONG number, which is worse than no number — it is the exact
            # failure class this project refuses to build on. When the last
            # match was poor, the recall keeps the side and drops the distance.
            parts.append(f"the {r['phrase']} {side}"
                         + (f", about {r['distance']:.1f} m" if self.trusted
                            else " (roughly — I have lost track of how far)"))
        if not parts:
            return ""
        return ("You cannot see these right now, but you have seen them and "
                "they are still there: " + "; ".join(parts[:4]) + ".")

    @property
    def trusted(self) -> bool:
        """Has the map earned the right to quote a distance?

        Before there is anything to register against, dead reckoning IS the
        truth and the answer is yes. After that it has to keep matching —
        UNLESS the evidence came from a same-pose fan sweep (§21.2), whose
        geometry is exact by construction and needs no match score. Without
        that provenance the nine sweep fusions pushed `updates` past the
        grace window with last_score still 0.0, and the first frame's
        planning view fell back to the frontal wedge — the 'two candidates
        until you press A/D' bug."""
        return (self.updates < 6 or self.sweep_trust
                or self.last_score >= TRUST_MIN_SCORE)

    # ── the planning view: current wedge, backed by memory ───────────────
    def potential_regions(self, min_cells: int = 8) -> list[dict]:
        """§14.6: the ACCUMULATED potential layer as addressable regions —
        long-term addressing of glimpses, not just the current frame's.
        Regions live in anchor metres (private); the dict also carries the
        egocentric readout for speaking about them."""
        mask = (self.potential > 0.5) & (np.abs(self.logodds) <= 0.5)
        if not mask.any():
            return []
        from scipy.ndimage import label as _cc_label
        lab, n = _cc_label(mask)
        out: list[dict] = []
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            if ii.size < min_cells:
                continue
            ys = (self.centre - ii) * self.cell + self.oy
            xs = (jj - self.centre) * self.cell + self.ox
            cx, cy = float(xs.mean()), float(ys.mean())
            bx, by = (float(v) for v in self.to_body(cx, cy))
            # ceiling division: a floor step let 201-599-cell regions emit
            # >200 pts and downstream head-cuts sampled only one side of
            # the region, skewing route_evidence fractions (review P2)
            step = -(-ii.size // 200)
            out.append({
                "id": int(abs(hash((round(cx, 1), round(cy, 1)))) % 10**8),
                "cells": int(ii.size),
                "centroid_anchor": (cx, cy),
                "pts": [(float(x), float(y))
                        for x, y in zip(xs[::step], ys[::step])],
                "bearing_deg": round(math.degrees(math.atan2(bx, by)), 1),
                "distance_m": round(math.hypot(bx, by), 1),
            })
        out.sort(key=lambda r: -r["cells"])
        return out

    def planning_view(self, td: TopDown) -> TopDown:
        """The grid the proposer should plan on (§10.5).

        Start from the current wedge — its OCCUPIED is authoritative and its
        FREE was seen this very frame. Then let TRUSTED memory fill in only
        what the camera cannot currently say (UNKNOWN cells): ground that was
        walked on or repeatedly fused as floor becomes FREE, remembered walls
        become OCCUPIED. OPEN stays OPEN — a sightline is direction evidence,
        not a landing. Memory reaches only to FUSE_MAX_M, the radius inside
        which it was ever allowed to hold structure, so nothing here is more
        confident than the fusion that fed it.

        This is what lets a gateway open into side space the body saw three
        steps ago — without ever overruling anything the current frame sees."""
        if td is None or self.updates < 3 or not self.trusted:
            return td
        grid = td.grid.copy()
        ii, jj = np.nonzero(grid == UNKNOWN)
        if not ii.size:
            return td
        y = (ii + 0.5) * CELL_M
        x = (jj + 0.5) * CELL_M - td.range_cap_m
        near = np.hypot(x, y) <= FUSE_MAX_M
        ii, jj, x, y = ii[near], jj[near], x[near], y[near]
        if not ii.size:
            return td
        X, Y = self.to_anchor(x, y)
        ai, aj = self.cells(X, Y)
        ok = (ai >= 0) & (ai < self.n) & (aj >= 0) & (aj < self.n)
        ii, jj, ai, aj = ii[ok], jj[ok], ai[ok], aj[ok]
        lo = self.logodds[ai, aj]
        free_mem = (lo < -1.5) | self.visited[ai, aj]
        occ_mem = lo > 1.0
        if not free_mem.any() and not occ_mem.any():
            return td
        grid[ii[free_mem], jj[free_mem]] = FREE
        grid[ii[occ_mem], jj[occ_mem]] = OCCUPIED
        clearance = distance_transform_edt(grid != OCCUPIED) * CELL_M
        return TopDown(grid=grid, free_range=td.free_range,
                       bearings=td.bearings, clearance=clearance,
                       floor_y=td.floor_y, normalized_depth=td.normalized_depth,
                       range_cap_m=td.range_cap_m,
                       floor_blind_m=td.floor_blind_m, potential=td.potential)

    # ── drawing ──────────────────────────────────────────────────────────
    _SEM_COLOURS = ((246, 130, 210), (130, 246, 190), (250, 176, 90),
                    (150, 170, 255), (230, 230, 120), (120, 230, 230))

    def render(self, waypoints: list[Waypoint] | None = None, *, scale: int = 3,
               caption: str = "", crop: bool = True,
               min_half_m: float = MAP_MIN_HALF_M) -> bytes:
        """The map as it is: origin fixed, walls where they were put, the body
        drawn as an arrow that turns. Nothing is mirrored — `cells()` already
        puts +x to the right and +y up, and the flip that used to sit here is
        exactly what made the accumulated map come out backwards."""
        if PILImage is None:
            return b""
        occ = self.logodds > 1.0
        free = self.logodds < -0.5
        rgb = np.full((self.n, self.n, 3), (26, 28, 34), dtype=np.uint8)
        t = np.clip(-self.logodds / LOGODDS_CLIP, 0.0, 1.0)[..., None]
        rgb[free] = (np.array((24, 66, 46)) + np.array((26, 84, 52)) * t)[free]
        rgb[(self.potential >= 2.0) & ~free & ~occ] = (110, 84, 36)  # §2.4 amber
        # Semantics go UNDER the structure, and only tint it. A solid overlay on
        # top hid exactly the thing the label is about: you could see that the
        # pool was there but no longer that its edge is a wall. Drawn first, the
        # walls and the walked path stay legible on top of the colour.
        for k, (phrase, grid) in enumerate(sorted(self.sem.items())):
            hit = self._sem_mask(grid, SEM_MIN_CONF)
            if hit.any():
                col = np.array(self._SEM_COLOURS[k % len(self._SEM_COLOURS)])
                rgb[hit] = (0.5 * rgb[hit] + 0.5 * col).astype(np.uint8)
        rgb[self.visited] = (68, 74, 118)
        rgb[occ] = (176, 68, 62)

        # Crop to what has actually been mapped, else the picture is 90 % void.
        i0, i1, j0, j1 = 0, self.n, 0, self.n
        if crop:
            # SEMANTICS COUNT AS MAPPED. This used to consider only the
            # structure layers, and structure stops at FUSE_MAX_M (4.5 m) while
            # a landmark is stamped as far as the detector can see one. So a bar
            # counter recognised at 6.6 m and a door at 13.0 m were both written
            # into the map correctly and then cropped out of its picture — the
            # memory knew, the image did not show it, and the only way to find
            # out was to read the recall sentence. Exactly the far things worth
            # drawing were the ones guaranteed to fall outside the frame.
            used_mask = occ | free | self.visited
            for _phrase, sgrid in self.sem.items():
                used_mask = used_mask | self._sem_mask(sgrid, SEM_MIN_CONF)
            used = np.nonzero(used_mask)
            if used[0].size:
                pad = int(1.0 / self.cell)
                i0 = max(0, int(used[0].min()) - pad)
                i1 = min(self.n, int(used[0].max()) + pad + 1)
                j0 = max(0, int(used[1].min()) - pad)
                j1 = min(self.n, int(used[1].max()) + pad + 1)
            # …and never crop tighter than this around the body. Three steps in,
            # the mapped region is a couple of metres across and a tight crop
            # renders it smaller than the single-frame top-down beside it — the
            # accumulated map is the one worth studying, and it was the one
            # being shrunk. A floor here keeps the scale steady between frames
            # too, so walking does not make the picture jump about.
            half = int(min_half_m / self.cell)
            bi, bj = self.cells(self.px, self.py)
            i0 = min(i0, max(0, int(bi) - half))
            i1 = max(i1, min(self.n, int(bi) + half + 1))
            j0 = min(j0, max(0, int(bj) - half))
            j1 = max(j1, min(self.n, int(bj) + half + 1))
            # Square it off, by growing the short side. A door 12 m ahead with
            # structure only out to 4.5 m makes a tall thin crop, and a tall
            # thin picture laid out beside a square one gets sized by its height
            # and ends up narrow — the opposite of the intent. Growing adds
            # unmapped space, which the map already draws as unmapped; shrinking
            # would drop something measured.
            side = max(i1 - i0, j1 - j0)
            di, dj = side - (i1 - i0), side - (j1 - j0)
            i0, i1 = max(0, i0 - di // 2), min(self.n, i1 + (di - di // 2))
            j0, j1 = max(0, j0 - dj // 2), min(self.n, j1 + (dj - dj // 2))
        img = PILImage.fromarray(rgb[i0:i1, j0:j1]).resize(
            ((j1 - j0) * scale, (i1 - i0) * scale), PILImage.NEAREST)
        draw = ImageDraw.Draw(img)

        def _px(x: float, y: float) -> tuple[float, float]:
            i, j = self.cells(x, y)
            return (float(j) - j0 + 0.5) * scale, (float(i) - i0 + 0.5) * scale

        if len(self.trail) > 1:
            draw.line([_px(p[0], p[1]) for p in self.trail],
                      fill=(126, 142, 232), width=2)
        for k, (phrase, grid) in enumerate(sorted(self.sem.items())):
            ii, jj = np.nonzero(self._sem_mask(grid, SEM_MIN_CONF))
            if not ii.size:
                continue
            cx = (float(jj.mean()) - j0 + 0.5) * scale
            cy = (float(ii.mean()) - i0 + 0.5) * scale
            draw.text((cx + 4, cy - 5), phrase[:18],
                      fill=self._SEM_COLOURS[k % len(self._SEM_COLOURS)])
        _wfont = caption_font(14)
        for n, w in enumerate(waypoints or [], 1):
            X, Y = self.to_anchor(w.x_left, w.y_fwd)
            x, y = _px(float(X), float(Y))
            colour = (255, 205, 70) if w.kind == "gateway" else (120, 210, 255)
            draw.ellipse([x - 11, y - 11, x + 11, y + 11],
                         outline=colour, width=3)
            draw.text((x - 4, y - 8), str(n), fill=(255, 255, 255),
                      font=_wfont)
        bx, by = _px(self.px, self.py)
        hx, hy = _px(self.px + 0.7 * -math.sin(self.theta),
                     self.py + 0.7 * math.cos(self.theta))
        draw.line([bx, by, hx, hy], fill=(250, 220, 90), width=3)
        draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(250, 220, 90))
        sx, sy = _px(self.trail[0][0], self.trail[0][1])
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], outline=(150, 210, 255),
                     width=2)

        if caption:
            pad, lh = 6, 12
            # Wrap, don't truncate. The readout is the only place the human sees
            # the registration numbers and the recall sentence, and the first
            # version clipped both at the image edge.
            width = max(28, (img.width - 2 * pad) // 6)
            lines: list[tuple[int, str]] = []
            for k, para in enumerate(caption.split("\n")):
                while len(para) > width:
                    cut = para.rfind(" ", 0, width)
                    cut = cut if cut > width // 2 else width
                    lines.append((k, para[:cut]))
                    para = para[cut:].lstrip()
                lines.append((k, para))
            strip = PILImage.new("RGB", (img.width, lh * len(lines) + 2 * pad),
                                 (12, 12, 16))
            d2 = ImageDraw.Draw(strip)
            font = caption_font(11)
            for k, (para, line) in enumerate(lines):
                d2.text((pad, pad + k * lh), line, font=font,
                        fill=(150, 210, 255) if para == 0 else (185, 185, 195))
            out = PILImage.new("RGB", (img.width, img.height + strip.height))
            out.paste(img, (0, 0))
            out.paste(strip, (0, img.height))
            img = out
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def caption(self) -> str:
        walked = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(self.trail, self.trail[1:]))
        fix = (f" · last fix {self.last_fix[0]:+.02f},{self.last_fix[1]:+.02f} m "
               f"{self.last_fix[2]:+.1f}°" if any(self.last_fix) else " · no fix")
        return (f"accumulated · {self.updates} looks · walked {walked:.1f} m · "
                f"match {self.last_score:.2f}{fix} · corrected {self.fixes}×")

    # ── the model-facing render (§10.6) ──────────────────────────────────
    MODEL_HALF_M = 8.0    # fixed crop half-width → the scale never jumps about
    MODEL_SCALE = 3

    def render_model(self, waypoints: list[Waypoint] | None = None, *,
                     current: set[str] | None = None,
                     loop: dict | None = None,
                     half_m: float | None = None,
                     scale: int | None = None,
                     caption: str = "", strip: bool = True) -> bytes:
        """The map drawn FOR THE MODEL, not for the operator.

        The human render is anchor-fixed: north stays north and the body
        wanders — right for someone studying a run, wrong for a VLM that must
        relate the map to the egocentric photo BESIDE it every single turn.
        Here the body sits at the centre facing up, always; left in this
        image is the body's left; the crop and scale never change; waypoint
        numbers are the same circles drawn on the RGB; a landmark the
        detector sees NOW is a solid dot, one merely remembered is a dashed
        ring. The legend therefore never has to be re-learned.

        Deterministic sampling, no interpolation: each output cell reads the
        anchor cell under it after rotating by the heading. Repeated renders
        do not blur anything because nothing is ever resampled into itself."""
        if PILImage is None:
            return b""
        # half_m/scale/caption exist for the HUMAN tab's heading-up view
        # (wider window, finer pixels, readout strip); the model path passes
        # nothing and keeps the fixed 8 m / ×3 contract unchanged.
        half_cells = int((half_m or self.MODEL_HALF_M) / self.cell)
        n = 2 * half_cells
        ys = (half_cells - np.arange(n, dtype=np.float64)[:, None] - 0.5) * self.cell
        xs = (np.arange(n, dtype=np.float64)[None, :] + 0.5 - half_cells) * self.cell
        c, s = math.cos(self.theta), math.sin(self.theta)
        X = xs * c - ys * s + self.px
        Y = xs * s + ys * c + self.py
        i, j = self.cells(X, Y)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        i_c = np.clip(i, 0, self.n - 1)
        j_c = np.clip(j, 0, self.n - 1)
        occ = (self.logodds[i_c, j_c] > 1.0) & ok
        free = (self.logodds[i_c, j_c] < -0.5) & ok
        vis = self.visited[i_c, j_c] & ok
        pot = (self.potential[i_c, j_c] >= 2.0) & ok & ~free & ~occ
        rgb = np.full((n, n, 3), (26, 28, 34), np.uint8)
        rgb[free] = (30, 78, 50)
        # §2.4: glimpsed-but-unverified floor is AMBER — visibly not the same
        # thing as walkable green, visibly not the same thing as unknown dark
        rgb[pot] = (128, 96, 38)
        for k, (phrase, sgrid) in enumerate(sorted(self.sem.items())):
            hit = self._sem_mask(sgrid, SEM_MIN_CONF)[i_c, j_c] & ok
            if hit.any():
                col = np.array(self._SEM_COLOURS[k % len(self._SEM_COLOURS)])
                rgb[hit] = (0.55 * rgb[hit] + 0.45 * col).astype(np.uint8)
        rgb[vis] = (68, 74, 118)
        rgb[occ] = (186, 70, 64)
        scale = scale or self.MODEL_SCALE
        img = PILImage.fromarray(rgb).resize((n * scale, n * scale),
                                             PILImage.NEAREST)
        draw = ImageDraw.Draw(img)

        def _px(ax: float, ay: float) -> tuple[float, float]:
            dx, dy = ax - self.px, ay - self.py
            xr = dx * c + dy * s
            yu = -dx * s + dy * c
            return ((half_cells + xr / self.cell) * scale,
                    (half_cells - yu / self.cell) * scale)

        if len(self.trail) > 1:
            draw.line([_px(p[0], p[1]) for p in self.trail],
                      fill=(126, 142, 232), width=2)
            # §7.2: the RECENT leg rides brighter, so "where I just walked"
            # and "where I walked long ago" stop looking identical
            recent = self.trail[-12:]
            if len(recent) > 1:
                draw.line([_px(p[0], p[1]) for p in recent],
                          fill=(188, 202, 255), width=4)
        if loop and loop.get("revisit_trail_index") is not None:
            k = int(loop["revisit_trail_index"])
            if 0 <= k < len(self.trail):
                rx, ry = _px(self.trail[k][0], self.trail[k][1])
                draw.ellipse([rx - 10, ry - 10, rx + 10, ry + 10],
                             outline=(255, 80, 80), width=3)
                draw.text((rx + 11, ry - 7), "revisit", fill=(255, 120, 120))
        sx, sy = _px(self.trail[0][0], self.trail[0][1])
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4],
                     outline=(150, 210, 255), width=2)
        for k, (phrase, sgrid) in enumerate(sorted(self.sem.items())):
            iiw, jjw = np.nonzero(self._sem_mask(sgrid, SEM_MIN_CONF))
            if not iiw.size:
                continue
            wgt = sgrid[iiw, jjw]
            ax = self.ox + (jjw - self.centre) * self.cell
            ay = self.oy - (iiw - self.centre) * self.cell
            x0, y0 = _px(float(np.average(ax, weights=wgt)),
                         float(np.average(ay, weights=wgt)))
            col = self._SEM_COLOURS[k % len(self._SEM_COLOURS)]
            if current and phrase in current:
                draw.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill=col)
            else:
                for a0 in range(0, 360, 45):        # a dashed ring: remembered
                    draw.arc([x0 - 6, y0 - 6, x0 + 6, y0 + 6], a0, a0 + 25,
                             fill=col, width=2)
            draw.text((x0 + 7, y0 - 6), phrase[:18], fill=col)
        _wfont = caption_font(14)
        for nw, w in enumerate(waypoints or [], 1):
            AX, AY = self.to_anchor(w.x_left, w.y_fwd)
            x0, y0 = _px(float(AX), float(AY))
            colour = (255, 205, 70) if w.kind == "gateway" else (120, 210, 255)
            draw.ellipse([x0 - 12, y0 - 12, x0 + 12, y0 + 12],
                         outline=colour, width=3)
            draw.text((x0 - 4, y0 - 8), str(nw), fill=(255, 255, 255),
                      font=_wfont)
        bx = by = half_cells * scale
        draw.line([bx, by, bx, by - 0.7 / self.cell * scale],
                  fill=(250, 220, 90), width=3)
        draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(250, 220, 90))

        if strip:
            cap = (f"map memory · {self.updates} looks · match "
                   f"{self.last_score:.2f}"
                   + ("" if self.trusted else " · DISTANCES UNTRUSTED")
                   + f" · you: centre, facing up · "
                   f"{2 * (half_m or self.MODEL_HALF_M):.0f} m across"
                   + (("\n" + caption) if caption else ""))
            pad, lh = 6, 12
            _cap_lines = cap.split("\n")
            _strip = PILImage.new("RGB",
                                  (img.width, lh * len(_cap_lines) + 2 * pad),
                                  (12, 12, 16))
            _d2 = ImageDraw.Draw(_strip)
            for _k, _line in enumerate(_cap_lines):
                _d2.text((pad, pad + _k * lh), _line[: img.width // 6],
                         font=caption_font(11), fill=(185, 185, 195))
            out = PILImage.new("RGB", (img.width, img.height + _strip.height))
            out.paste(img, (0, 0))
            out.paste(_strip, (0, img.height))
            img = out
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── §21.7: the dual-truth map — anchor-fixed global + heading-up local ─
    INSET_HALF_M = 2.5     # the local panel shows 5 m across
    INSET_SCALE = 4        # finer pixels: it is small and near-field

    def render_composite(self, waypoints: list[Waypoint] | None = None, *,
                         current: set[str] | None = None,
                         loop: dict | None = None,
                         caption: str = "") -> bytes:
        """ONE image, two truths (§21.7). LEFT: the long-term map fixed to
        the WORLD — walls, trail, landmarks and stable candidate circles
        stay at fixed pixels; the yellow robot moves through it and its
        arrow turns with the real heading, so turning your head no longer
        rotates the whole remembered world and loops are visible as loops.
        RIGHT: a small heading-up inset — you at the centre facing up,
        5 m across — that keeps left-in-the-map = left-in-the-photo for
        the near field. Same map_version/candidate_epoch, stamped once in
        the shared caption bar."""
        if PILImage is None:
            return b""
        g_png = self.render(waypoints, scale=3, caption="", crop=True)
        l_png = self.render_model(waypoints, current=current, loop=loop,
                                  half_m=self.INSET_HALF_M,
                                  scale=self.INSET_SCALE, strip=False)
        try:
            g = PILImage.open(BytesIO(g_png)).convert("RGB")
            loc = PILImage.open(BytesIO(l_png)).convert("RGB")
        except Exception:  # noqa: BLE001
            return g_png
        # keep the composite readable: global panel capped, inset as-is
        if g.width > 560:
            g = g.resize((560, int(g.height * 560 / g.width)),
                         PILImage.LANCZOS)
        label_h = 16
        h = max(g.height, loc.height) + label_h
        gap = 6
        out = PILImage.new("RGB", (g.width + gap + loc.width, h),
                           (16, 17, 21))
        d = ImageDraw.Draw(out)
        d.text((4, 2), "GLOBAL — fixed to the world; the arrow is you",
               font=caption_font(11), fill=(168, 176, 192))
        d.text((g.width + gap + 4, 2), "LOCAL — you centred, facing up",
               font=caption_font(11), fill=(168, 176, 192))
        out.paste(g, (0, label_h))
        out.paste(loc, (g.width + gap, label_h))
        if caption:
            pad, lh = 6, 12
            lines = caption.split("\n")
            strip = PILImage.new("RGB", (out.width, lh * len(lines) + 2 * pad),
                                 (12, 12, 16))
            d2 = ImageDraw.Draw(strip)
            for k, line in enumerate(lines):
                d2.text((pad, pad + k * lh), line[: out.width // 6],
                        font=caption_font(11), fill=(185, 185, 195))
            full = PILImage.new("RGB", (out.width, out.height + strip.height))
            full.paste(out, (0, 0))
            full.paste(strip, (0, out.height))
            out = full
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


# ── 3. free space → waypoint candidates ──────────────────────────────────
@dataclass
class Waypoint:
    """A proposal in the ONLY frame the harness admits: egocentric polar.
    `angle` is radians counter-clockwise-positive (left) and `distance` metres —
    exactly what env_habitat__step_hightolow consumes."""

    angle: float                   # where the PLACE is (radians, left +)
    distance: float                # how far the place is — the aiming target
    clearance: float
    kind: str                      # opening | gateway
    x_left: float = 0.0
    y_fwd: float = 0.0
    note: str = ""
    extras: dict = field(default_factory=dict)
    # §10.3: a candidate answers two different questions and must say which
    # number answers which. `distance` is the direction target — it may point
    # past the proven floor, at structure the sightline says continues.
    # `stride_m` is the only number the executor may walk blind, and it never
    # exceeds `verified_m`, the continuous FREE prefix along this bearing.
    stride_m: float = 0.0          # safe to execute in one blind go
    verified_m: float = 0.0        # ground actually seen, with clearance
    visible_m: float = 0.0         # clear sightline along this bearing
    confidence: str = ""           # high | medium | low
    # §21.9: verified passable prefix REMAINING past the landing on this
    # same bearing — the number that says the circle is a decision point
    # mid-route, not a wall-adjacent terminus. 0.0 = none measured.
    continuation_m: float = 0.0

    def as_candidate(self) -> dict:
        """The wire form the toolset speaks. `angle`/`distance` keep their old
        meaning for existing consumers; the contract fields ride alongside."""
        out = {"angle": self.angle, "distance": self.distance,
               "target_angle_deg": round(math.degrees(self.angle), 1),
               "target_distance_m": round(self.distance, 2),
               "safe_stride_m": round(self.stride_m or self.distance, 2),
               "verified_ground_m": round(self.verified_m, 2),
               "visible_clear_depth_m": round(self.visible_m, 2),
               "clear_beyond_m": round(self.continuation_m, 2)}
        if self.confidence:
            out["confidence"] = self.confidence
        if self.note:
            out["reason"] = self.note
        return out

    def describe(self) -> str:
        deg = math.degrees(self.angle)
        if abs(deg) < 12:
            side = "straight ahead"
        elif abs(deg) >= 100:
            # memory candidates can sit behind the camera — say so plainly
            side = f"behind you on the {'left' if deg > 0 else 'right'} ({abs(deg):.0f}°)"
        else:
            side = f"{'left' if deg > 0 else 'right'} {abs(deg):.0f}°"
        text = f"{side}, {self.distance:.1f} m away"
        width = 2 * self.clearance
        if self.kind == "gateway":
            text += f", the way around, about {width:.1f} m of room"
        elif width < 1.2:
            text += f", a narrow spot about {width:.1f} m wide"
        else:
            text += f", about {width:.1f} m of open space there"
        if self.distance > self.verified_m + 0.3 > 0.3:
            text += (f" — floor is proven for {self.verified_m:.1f} m and the "
                     f"view stays clear to {self.visible_m:.1f} m")
        elif self.continuation_m >= 0.5:
            # §21.9: say the landing is a mid-route point, not a terminus
            text += (f" — the way stays verified clear "
                     f"{self.continuation_m:.1f} m past it")
        return text + (f" ({self.note})" if self.note else "")


def _ray_min_clearance(td: TopDown, x_left: float, y_fwd: float) -> float:
    """Tightest squeeze anywhere along the straight path to a point."""
    r = math.hypot(x_left, y_fwd)
    n = max(3, int(2 * r / CELL_M))
    ts = np.linspace(0.0, 1.0, n)
    ii = np.clip((ts * y_fwd / CELL_M).astype(int), 0, td.n_fwd - 1)
    jj = np.clip(((ts * x_left + td.range_cap_m) / CELL_M).astype(int), 0, td.n_lat - 1)
    beyond = ts * r >= SELF_FOOTPRINT_M
    if not bool(beyond.any()):
        return float(td.clearance[ii[-1], jj[-1]])
    return float(np.min(td.clearance[ii[beyond], jj[beyond]]))


def _stride_cap(min_clearance: float) -> float:
    """How far it is safe to walk BLIND through a corridor that tight.

    step_hightolow does not path-plan — it rotates once, then walks
    int(d/0.25) forward primitives with sliding on collision. A 5.3 m stride
    down a 0.8 m-wide slot ended a live probe wedged in a corner with 0.4 m of
    floor left and no candidates at all. The tighter the squeeze, the shorter
    the leap the harness is willing to offer.

    Calibrated against live hops, not guessed: the body is 0.40 m across, so a
    0.30 m clearance (0.60 m of corridor) genuinely fits with room to spare and
    must not be throttled to a crawl — a stricter first table capped almost
    every stride at 2 m and gave away the whole advantage over the learned
    predictor's 3 m ceiling."""
    if min_clearance >= 0.50:      # 1.0 m of corridor — the full allowance
        return MAX_WAYPOINT_M
    if min_clearance >= 0.35:
        return min(MAX_WAYPOINT_M, 2.5)
    if min_clearance >= 0.25:
        return min(MAX_WAYPOINT_M, 2.0)
    return 1.5


def _reachable(td: TopDown, x_left: float, y_fwd: float,
               need: float = ROBOT_RADIUS_M) -> bool:
    """Straight-line walkability — the guarantee the learned predictor lacks.

    Tested against the clearance field rather than an isotropic dilation of the
    obstacles: dilating double-counts the body and seals corridors the robot
    genuinely fits through (it sealed EP0's wall/pool gap and returned zero
    candidates). The first half-metre is skipped because the robot is already
    standing there — EP0 starts 0.58 m from the pool coping, and treating that
    as a collision vetoes every ray."""
    r = math.hypot(x_left, y_fwd)
    n = max(3, int(2 * r / CELL_M))
    ts = np.linspace(0.0, 1.0, n)
    ii = np.clip((ts * y_fwd / CELL_M).astype(int), 0, td.n_fwd - 1)
    jj = np.clip(((ts * x_left + td.range_cap_m) / CELL_M).astype(int), 0, td.n_lat - 1)
    beyond = ts * r >= SELF_FOOTPRINT_M
    if not bool(beyond.any()):
        return True
    return bool(np.all(td.clearance[ii[beyond], jj[beyond]] >= need))


# Three generations of candidate extractors (Voronoi medial axis, per-bearing
# frontiers, free-range peaks) used to sit here uncalled. Deleted, not
# commented out: an implementation nobody runs is documentation that lies.


def passable_profile(td: TopDown, need: float = TIGHT_CLEARANCE_M,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per bearing: (verified reach, tightest squeeze inside that reach).

    Reach is the length of the CONTINUOUS VERIFIED-FREE prefix along the
    bearing: cells the camera actually saw floor on (or the body stood on),
    with `need` clearance the whole way. It stops at OPEN — a sightline is not
    ground — and it now also stops at UNKNOWN, which the old profile walked
    straight through: cells past the first obstacle, or outside the viewing
    wedge, are far from any *known* wall, so their clearance is generous and
    the clearance test alone let rays escape sideways out of the wedge. A
    sweep over central-obstacle scenes found 77 candidate endpoints on
    non-FREE cells, the first batch at ±44° — exactly the FOV edge, where half
    of every ray's cells are unobserved.

    The neck is the minimum clearance within the reach (past the body's own
    footprint) — the number corridor_bottleneck() walks out for one bearing,
    computed for every bearing in the pass the reach already makes.

    The default `need` is TIGHT_CLEARANCE_M — the same bar the proposer
    applies to a candidate. Two different bars made this profile and its
    consumer disagree about the same corridor (6.1 m passable, every candidate
    on it rejected), and the proposer walked its own suggestion back to 1.1 m."""
    n = len(td.bearings)
    reach = np.zeros(n, dtype=np.float32)
    neck = np.zeros(n, dtype=np.float32)
    cap = td.range_cap_m
    radii = np.arange(CELL_M, cap, CELL_M)
    for k, bearing in enumerate(td.bearings):
        xs = radii * math.sin(bearing)
        ys = radii * math.cos(bearing)
        ii = (ys / CELL_M).astype(np.int32)
        jj = ((xs + cap) / CELL_M).astype(np.int32)
        ok = (ii >= 0) & (ii < td.n_fwd) & (jj >= 0) & (jj < td.n_lat)
        if not ok.any():
            continue
        ii, jj, rr = ii[ok], jj[ok], radii[ok]
        cl = td.clearance[ii, jj]
        beyond = rr > SELF_FOOTPRINT_M
        bad = ((cl < need) | (td.grid[ii, jj] != FREE)) & beyond
        hit = np.nonzero(bad)[0]
        stop = int(hit[0]) if hit.size else len(rr)
        if stop:
            reach[k] = float(rr[stop - 1])
            seg = cl[:stop][beyond[:stop]]
            neck[k] = float(seg.min()) if seg.size else float(cl[:stop].min())
    return reach, neck


def _prof_at(td: TopDown, prof: np.ndarray, bearing: float) -> float:
    """A profile's value on an arbitrary bearing — nearest bin, no optimism."""
    half = float(td.bearings[-1])
    if half <= 0 or not len(prof):
        return 0.0
    k = int(round((bearing + half) / (2 * half) * (len(prof) - 1)))
    return float(prof[min(max(k, 0), len(prof) - 1)])


def _prefix_free(td: TopDown, bearing: float, upto: float,
                 need: float = TIGHT_CLEARANCE_M) -> bool:
    """Every cell the body will actually cross is verified floor with room.

    The executable-path contract: landing and prefix on FREE only. OPEN and
    UNKNOWN are not obstacles, but they are not somewhere to walk blind
    either. The profile already guarantees this along bin-centre bearings;
    this is the belt-and-braces check on the exact ray that will be walked."""
    r = SELF_FOOTPRINT_M
    while r < upto - 1e-9:
        r = min(r + CELL_M, upto)
        i, j = td.cell_of(r * math.sin(bearing), r * math.cos(bearing))
        if (not td.inside(i, j) or td.grid[i, j] != FREE
                or float(td.clearance[i, j]) < need):
            return False
    return True


def _free_prefix(td: TopDown, bearing: float,
                 need: float = TIGHT_CLEARANCE_M) -> float:
    """Length of the continuous verified-FREE prefix along ONE bearing —
    the single-ray form of passable_profile, for bearings that need not sit
    on a bin centre (held goals, route legs). Stops at the first cell that
    is not FREE or lacks `need` clearance, past the body's own footprint."""
    out = 0.0
    r = SELF_FOOTPRINT_M
    while r < td.range_cap_m:
        r += CELL_M
        i, j = td.cell_of(r * math.sin(bearing), r * math.cos(bearing))
        if (not td.inside(i, j) or td.grid[i, j] != FREE
                or float(td.clearance[i, j]) < need):
            break
        out = r
    return out


ISLAND_NEAR_M = 4.0     # a sight-blocker nearer than this can be an island…
ISLAND_RELIEF_M = 1.0   # …if BOTH flanks see at least this much past it
ISLAND_MIN_DEG = 3.0    # narrower is speckle, not a thing to walk around


def potential_regions(td: TopDown, *, min_cells: int = 12) -> list[dict]:
    """Clusters of §2.4 POTENTIAL evidence, egocentric on the way out.

    Deterministic connected components over the potential mask; each region
    reports where it lies (bearing, nearest range) and how much floor was
    glimpsed. min_cells keeps single-pixel speckle from becoming a place."""
    if td is None or td.potential is None or not td.potential.any():
        return []
    from scipy.ndimage import label as _cc_label
    lab, n = _cc_label(td.potential)
    out: list[dict] = []
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        if ii.size < min_cells:
            continue
        x = (jj + 0.5) * CELL_M - td.range_cap_m
        y = (ii + 0.5) * CELL_M
        bearing = math.atan2(float(x.mean()), float(y.mean()))
        out.append({"bearing_deg": round(math.degrees(bearing), 1),
                    "nearest_m": round(float(np.hypot(x, y).min()), 1),
                    "cells": int(ii.size)})
    out.sort(key=lambda r: -r["cells"])
    return out


# ── §14.6: RouteCandidate — a potential region becomes a PLANNED route ───
R_INFO_M = 2.5           # info-gain neighbourhood around a staging point
STAGE_NEAR_M = 0.4       # staging must land at least this close to walkable
STAGE_FAR_M = 3.0        # …and no further than this from the region rim
ROUTE_CANCEL_OCC = 0.5   # region majority-OCCUPIED → the glimpse was wrong
ROUTE_OPEN_FREE = 0.30   # region fraction verified FREE → leg 2 is real


@dataclass
class RouteCandidate:
    """§14.6: intent that OUTLIVES the observation that created it.

    A potential region is a glimpse; a RouteCandidate is a plan: walk the
    verified leg to a staging point chosen for what it will reveal, turn,
    look, and only then plan the second leg — or cancel the whole idea when
    the new evidence says the glimpse was wrong. Coordinates below are the
    map organ's private anchor frame and NEVER reach the model; everything
    spoken about a route is egocentric at speaking time."""
    intent: str
    frontier_id: int
    legs: list[dict]                  # [{status, distance_m, ...}, …]
    evidence: dict
    uncertainty: float
    created_map_version: int
    state: str = "leg1"               # leg1 | staging | leg2_open | done | cancelled
    staging_anchor: tuple[float, float] = (0.0, 0.0)
    region_centroid: tuple[float, float] = (0.0, 0.0)
    region_pts: list = field(default_factory=list)   # anchor xy samples

    def as_public(self, amap: Any) -> dict:
        """The egocentric readout — bearings/metres from the body NOW."""
        sx, sy = self.staging_anchor
        bx, by = (float(v) for v in amap.to_body(sx, sy))
        rx, ry = self.region_centroid
        gx, gy = (float(v) for v in amap.to_body(rx, ry))
        return {
            "intent": self.intent, "state": self.state,
            "staging": {"bearing_deg": round(math.degrees(
                math.atan2(bx, by)), 1),
                "distance_m": round(math.hypot(bx, by), 1)},
            "region": {"bearing_deg": round(math.degrees(
                math.atan2(gx, gy)), 1),
                "distance_m": round(math.hypot(gx, gy), 1)},
            "legs": [{k: v for k, v in leg.items() if k != "path"}
                     for leg in self.legs],
            "uncertainty": self.uncertainty,
        }


def _walkable_mask(amap: Any) -> np.ndarray:
    """Verified-FREE cells inflated by the body: where the CENTRE may go."""
    from scipy.ndimage import binary_dilation
    free = amap.logodds < -0.5
    obst = amap.logodds > 0.5
    r = max(1, int(round(MIN_CLEARANCE_M / amap.cell)))
    inflated = binary_dilation(obst, iterations=r)
    return free & ~inflated


def _dijkstra_field(walkable: np.ndarray, start: tuple[int, int],
                    cell_m: float) -> tuple[np.ndarray, dict]:
    """Uniform-cost search over the walkable mask. Returns metres-from-start
    (inf where unreachable) and a parent map for path reconstruction."""
    import heapq
    n0, n1 = walkable.shape
    # float64, deliberately: the heap carries python floats, and a float32
    # store rounds them by more than the staleness tolerance — cells then
    # look "already improved" on pop and the wavefront freezes ~0.8 m out
    dist = np.full((n0, n1), np.inf, dtype=np.float64)
    parent: dict = {}
    si, sj = start
    if not (0 <= si < n0 and 0 <= sj < n1):
        return dist, parent
    # the body stands where it stands — plan FROM here even if inflation
    # nibbled the exact start cell
    dist[si, sj] = 0.0
    heap = [(0.0, si, sj)]
    diag = math.sqrt(2.0) * cell_m
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j] + 1e-9:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if not (0 <= ii < n0 and 0 <= jj < n1):
                    continue
                if not walkable[ii, jj]:
                    continue
                nd = d + (diag if di and dj else cell_m)
                if nd < dist[ii, jj] - 1e-9:
                    dist[ii, jj] = nd
                    parent[(ii, jj)] = (i, j)
                    heapq.heappush(heap, (nd, ii, jj))
    return dist, parent


def plan_route(amap: Any, region: dict, *, intent: str = "") -> RouteCandidate | None:
    """§14.6: pick a STAGING point on verified-FREE ground by what standing
    there buys — path length, closeness to the frontier, expected
    information gain, landing clearance — and wrap it in a RouteCandidate
    whose second leg stays UNVERIFIED until the turn-and-look earns it.

    `region` comes from AnchorMap.potential_regions(). Returns None when no
    walkable staging point reaches the region's rim — an honest "not from
    here", never a blind leg."""
    from scipy.ndimage import distance_transform_edt
    walkable = _walkable_mask(amap)
    bi, bj = (int(v) for v in amap.cells(amap.px, amap.py))
    # the body's own disc is walkable by proof of standing on it
    r0 = max(1, int(round(ROBOT_RADIUS_M / amap.cell)))
    walkable[max(0, bi - r0):bi + r0 + 1, max(0, bj - r0):bj + r0 + 1] |= \
        amap.visited[max(0, bi - r0):bi + r0 + 1, max(0, bj - r0):bj + r0 + 1]
    dist, parent = _dijkstra_field(walkable, (bi, bj), amap.cell)

    region_mask = np.zeros_like(walkable)
    pts = region.get("pts") or []
    for (rx, ry) in pts:
        i, j = (int(v) for v in amap.cells(rx, ry))
        if 0 <= i < amap.n and 0 <= j < amap.n:
            region_mask[i, j] = True
    if not region_mask.any():
        return None
    d_region = distance_transform_edt(~region_mask) * amap.cell
    d_obst = distance_transform_edt(amap.logodds <= 0.5) * amap.cell
    unknown = np.abs(amap.logodds) <= 0.5

    cand = (walkable & np.isfinite(dist)
            & (d_region >= STAGE_NEAR_M) & (d_region <= STAGE_FAR_M)
            & (d_obst >= MIN_CLEARANCE_M))
    ii, jj = np.nonzero(cand)
    if ii.size == 0:
        return None
    if ii.size > 400:                      # cap the scoring loop
        sel = np.argsort(dist[ii, jj])[:400]
        ii, jj = ii[sel], jj[sel]
    # info gain: how much UNKNOWN-or-potential ground AT THIS FRONTIER sits
    # within sensing range of the staging point (m² a look could settle).
    # Gated to the region's neighbourhood on purpose — generic unknown
    # (the void beyond every mapped wall) is not what this route is for.
    r_info = int(round(R_INFO_M / amap.cell))
    gain = np.empty(ii.size, dtype=np.float32)
    interesting = (unknown | (amap.potential > 0.5)) & (d_region <= R_INFO_M)
    for k in range(ii.size):
        i0, i1 = max(0, ii[k] - r_info), ii[k] + r_info + 1
        j0, j1 = max(0, jj[k] - r_info), jj[k] + r_info + 1
        gain[k] = interesting[i0:i1, j0:j1].sum() * amap.cell * amap.cell
    score = (dist[ii, jj] + 0.6 * d_region[ii, jj].astype(np.float32)
             - 0.8 * gain)
    best = int(np.argmin(score))
    si, sj = int(ii[best]), int(jj[best])

    # reconstruct leg 1 (verified, on walkable ground the whole way)
    path_cells = [(si, sj)]
    while path_cells[-1] in parent:
        path_cells.append(parent[path_cells[-1]])
    path_cells.reverse()
    # cells → anchor metres, thinned
    path_xy = []
    for (i, j) in path_cells[::max(1, len(path_cells) // 12)] + [path_cells[-1]]:
        y = (amap.centre - i) * amap.cell + amap.oy
        x = (j - amap.centre) * amap.cell + amap.ox
        path_xy.append((round(float(x), 2), round(float(y), 2)))
    sx, sy = path_xy[-1]
    cx, cy = region["centroid_anchor"]
    leg2_m = math.hypot(cx - sx, cy - sy)
    return RouteCandidate(
        intent=intent or f"go look at the open floor glimpsed there",
        frontier_id=int(region.get("id", 0)),
        legs=[{"status": "verified", "distance_m": round(float(
            dist[si, sj]), 1), "path": path_xy},
            {"status": "unverified", "distance_m": round(leg2_m, 1)}],
        evidence={"potential_cells": int(region.get("cells", 0)),
                  "info_gain_m2": round(float(gain[best]), 1),
                  "staging_clearance_m": round(float(d_obst[si, sj]), 2)},
        uncertainty=round(1.0 / (1.0 + region.get("cells", 0) / 50.0), 2),
        created_map_version=int(amap.updates),
        staging_anchor=(float(sx), float(sy)),
        region_centroid=(float(cx), float(cy)),
        region_pts=[(float(x), float(y)) for x, y in pts[:200]],
    )


STAGING_MATCH_TOL_M = 0.6    # a numbered point this close to leg-1 IS on it
STAGING_MIN_PROGRESS_M = 0.3  # …and must actually advance along the leg


def staging_place_for(route: RouteCandidate, waypoints: list,
                      amap: Any) -> int | None:
    """§20.3: the number published as a route's staging must LIE ON the
    verified first leg. Bearing-nearest was a false promise — the point
    angularly closest to the glimpse routinely sat on the WRONG side of
    the occluder, and "place N is the nearest staging point" sent the body
    to the bar's face instead of along its flank. A place qualifies only
    if it advances ≥ STAGING_MIN_PROGRESS_M along leg-1, stays within
    STAGING_MATCH_TOL_M of the polyline, and does not overshoot the
    staging end. Returns the qualifying place number with the MOST
    progress, or None (the caller then says so honestly and offers the
    leg's bearing for a face())."""
    path = (route.legs[0].get("path") or []) if route.legs else []
    if len(path) < 2 or not waypoints:
        return None
    pts = np.asarray(path, dtype=np.float64)
    seg = np.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    best, best_prog = None, STAGING_MIN_PROGRESS_M
    for i, w in enumerate(waypoints, 1):
        wx, wy = (float(v) for v in amap.to_anchor(w.x_left, w.y_fwd))
        d_min, prog = float("inf"), 0.0
        for k in range(len(pts) - 1):
            ax, ay = pts[k]
            bx, by = pts[k + 1]
            abx, aby = bx - ax, by - ay
            l2 = abx * abx + aby * aby or 1e-12
            t = min(1.0, max(0.0, ((wx - ax) * abx + (wy - ay) * aby) / l2))
            d = math.hypot(wx - (ax + t * abx), wy - (ay + t * aby))
            if d < d_min:
                d_min = d
                prog = float(cum[k]) + t * math.sqrt(l2)
        if (d_min <= STAGING_MATCH_TOL_M
                and prog <= total + STAGING_MATCH_TOL_M
                and prog > best_prog):
            best, best_prog = i, prog
    return best


def route_evidence(amap: Any, route: RouteCandidate) -> dict:
    """What the accumulated map NOW says about the route's region."""
    occ = free = unk = 0
    for (x, y) in route.region_pts:
        i, j = (int(v) for v in amap.cells(x, y))
        if not (0 <= i < amap.n and 0 <= j < amap.n):
            continue
        lo = float(amap.logodds[i, j])
        if lo > 0.5:
            occ += 1
        elif lo < -0.5:
            free += 1
        else:
            unk += 1
    total = max(1, occ + free + unk)
    return {"occupied_frac": occ / total, "free_frac": free / total,
            "unknown_frac": unk / total, "sampled": total}


def _obstacle_islands(td: TopDown) -> list[tuple[int, int]]:
    """Free-standing obstacles as angular shadows: contiguous bearings whose
    SIGHT stops early while both flanks see clearly past them.

    This is the structural fork detector: a central island splits the walkable
    space into a left and a right gateway even when the passable-range valley
    is too shallow for the FORK_VALLEY ratio — which is exactly how the two
    ways around a chair kept collapsing into one ±45° point. Walls do not
    qualify: an interval touching the FOV edge has no flank on that side, and
    a wall across the whole view has no flanks at all."""
    fr = td.free_range
    n = len(fr)
    out: list[tuple[int, int]] = []
    k = 0
    while k < n:
        if float(fr[k]) >= ISLAND_NEAR_M:
            k += 1
            continue
        k2 = k
        while k2 + 1 < n and float(fr[k2 + 1]) < ISLAND_NEAR_M:
            k2 += 1
        blocked = float(fr[k:k2 + 1].min())
        right_flank, left_flank = fr[:k], fr[k2 + 1:]
        deg = math.degrees(float(td.bearings[k2]) - float(td.bearings[k]))
        if (right_flank.size and left_flank.size
                and float(right_flank.max()) >= blocked + ISLAND_RELIEF_M
                and float(left_flank.max()) >= blocked + ISLAND_RELIEF_M
                and deg >= ISLAND_MIN_DEG):
            out.append((k, k2))
        k = k2 + 1
    return out


@dataclass
class Opening:
    """One distinct way out of the current view.

    Iterates as the (centre, width, reach) tuple existing consumers unpack;
    the extra fields are what stops the proposer reusing the centre bearing's
    numbers for every other bearing in the run. `kind` is "opening" for a run
    that cleared the threshold on its own and "gateway" for a way around an
    obstacle island synthesised from the island's flank; `island` ties the two
    gateways of one island together so neither can squeeze the other out."""
    centre: float
    width: float
    reach: float
    lo: int = 0
    hi: int = 0
    kind: str = "opening"
    island: int = -1
    # Which FLANK of the island this opening is (−1 = lower-index/right,
    # +1 = higher-index/left), decided by ABUTMENT, not by sign(bearing):
    # an off-axis island puts both its flanks on the same side of 0°, and
    # tagging by bearing sign made the pair collapse in dedupe — the exact
    # §10.4 failure the island machinery exists to prevent.
    side: int = 0

    def __iter__(self):
        return iter((self.centre, self.width, self.reach))


MIN_USEFUL_WAYPOINT_M = 1.2   # "half the reach" may never retreat underfoot

# §21.9 landing contract: an ordinary DWP is an INTERMEDIATE decision
# point, so after the circle there must remain a stretch of the SAME
# bearing's verified-passable prefix still ahead — room to arrive, look and
# continue, not a wall at the nose. Landings that cannot afford it are not
# hidden; they are demoted to short_verified_gateway: honest "walk short,
# look again from there" staging, visibly not the same promise.
MIN_FORWARD_RUNWAY_M = 0.8


def _finish_waypoint(td: TopDown, w: Waypoint, *, landed_m: float,
                     reach_m: float | None = None) -> Waypoint:
    """Fill the dual answer onto a candidate whose LANDING is chosen.

    §2.1 of the revision plan retired sightline aiming: the point the model
    is shown IS the landing — mid-range of the verified-FREE reach — because
    a circle at the far edge of the visible world reads as a macro
    destination and buries the near/mid-range decision. What the eye can see
    still rides along as ``visible_m`` (words, not a circle), and the FULL
    verified reach is reported in ``verified_m`` so the model knows the
    corridor continues past the point."""
    w.verified_m = float(reach_m if reach_m is not None else landed_m)
    # §21.9: reach_m IS the verified-passable prefix on this bearing, so
    # what remains past the landing is the continuation — measured, not
    # asserted. Written before the demote decision so even a short gateway
    # reports its honest (small) number.
    w.continuation_m = round(max(0.0, w.verified_m - float(landed_m)), 2)
    w.visible_m = float(_prof_at(td, td.free_range, w.angle))
    squeeze = _ray_min_clearance(td, w.x_left, w.y_fwd)
    stride = min(_stride_cap(squeeze), float(landed_m))
    while stride > CELL_M and not _prefix_free(td, w.angle, stride):
        stride -= CELL_M                     # paranoia, not policy: see _prefix_free
    w.stride_m = round(float(stride), 2)
    w.extras["squeeze"] = round(squeeze, 2)
    w.extras["stride_m"] = w.stride_m        # legacy readers
    if not w.confidence:
        if squeeze < 0.25 or w.stride_m < 1.0:
            w.confidence = "low"
        elif w.stride_m < w.distance - 1e-6:
            w.confidence = "medium"
        else:
            w.confidence = "high"
    if w.stride_m < w.distance - 1e-6:
        w.note = ((w.note + "; " if w.note else "")
                  + f"walk about {w.stride_m:.1f} m of the way, then look again")
    return w


def passable_range(td: TopDown, need: float = TIGHT_CLEARANCE_M) -> np.ndarray:
    """How far the BODY can travel along each bearing, not how far the eye sees.
    The reach half of passable_profile(); kept for callers that only want it."""
    return passable_profile(td, need)[0]


def corridor_bottleneck(td: TopDown, bearing: float, upto: float) -> float:
    """The TIGHTEST clearance anywhere along this bearing, out to `upto`.

    "How close does this route pass to something" — which is the question, not
    "how much room is there where I stop". A point can sit in open floor and
    still be reached by threading past a chair, and the continuation past it
    matters too: obstacles are exactly where routes fork, so a path that skims
    one is a path that will need a decision it cannot see yet."""
    r, worst = 0.0, float("inf")
    while r < upto:
        r += CELL_M
        vx, vy = r * math.sin(bearing), r * math.cos(bearing)
        i, j = td.cell_of(vx, vy)
        if not td.inside(i, j):
            break
        if r > SELF_FOOTPRINT_M:          # already standing here
            worst = min(worst, float(td.clearance[i, j]))
    return 0.0 if worst == float("inf") else worst


FORK_VALLEY = 0.72       # a dip below this fraction of both peaks parts them
FORK_MIN_PEAK_M = 2.0    # …and each side has to be somewhere worth going


def _split_forks(run: np.ndarray, offset: int) -> list[tuple[int, int]]:
    """One contiguous walkable run → the separate WAYS it actually contains.

    A run of bearings that all clear the threshold is not necessarily one way
    out. Standing at the start of R2R ep 7 the profile reads 5.3 m at -20°,
    3.7 m across the middle, and 5.6 m at 0° — two corridors either side of the
    chairs, exactly the two the instruction means by "walk between the bar and
    chairs". Treated as a single run it collapsed to one opening centred at
    -22°, and the other way was never offered at all.

    So look for peaks with a real valley between them. The valley has to be
    clearly shallower than BOTH peaks — a gentle undulation along one corridor
    is not a fork, and splitting on it would offer the same way twice under two
    numbers, which is worse than offering it once.
    """
    n = len(run)
    if n < 5:
        return [(offset, offset + n)]
    cuts: list[int] = []
    # A local minimum deep enough to be a wall between two ways, and the peaks
    # on either side of it deep enough to be worth walking to.
    k = 1
    while k < n - 1:
        if run[k] <= run[k - 1] and run[k] <= run[k + 1]:
            left = float(run[:k].max()) if k else 0.0
            right = float(run[k + 1:].max()) if k + 1 < n else 0.0
            if (min(left, right) >= FORK_MIN_PEAK_M
                    and float(run[k]) < FORK_VALLEY * min(left, right)):
                cuts.append(k)
                k += 2          # do not cut again on the same trough
                continue
        k += 1
    if not cuts:
        return [(offset, offset + n)]
    out, start = [], 0
    for c in cuts:
        out.append((offset + start, offset + c + 1))
        start = c
    out.append((offset + start, offset + n))
    return [(a, b) for a, b in out if b - a >= 2]


GATEWAY_MIN_M = 0.6      # a flank counts as a way around if the body enters this far
ABS_SECTOR_MIN_M = 3.0   # a side fan verified this far is a way, whatever the front does
MIN_CANDIDATE_SEPARATION_DEG = 25.0   # closer than this is the same road (§2.2)


def _make_opening(td: TopDown, prof: np.ndarray, neck: np.ndarray,
                  s_lo: int, s_hi: int, *, smooth: np.ndarray | None = None,
                  kind: str = "opening", island: int = -1) -> Opening | None:
    """Choose one bearing to stand for a run of them, symmetric on purpose.

    The score is the old idea — most room along the way, depth breaking
    ties — evaluated on EVERY bin (the neck profile made that free), with
    near-ties settled by preference, not by scan order: nearest the run's own
    middle first, then nearest straight ahead, then the stable index. The old
    loop updated only on score > best, so in a perfectly symmetric corridor
    the first bin scanned — the right edge — won, and the proposer steered
    15° right of an empty centre line.

    Both score terms SATURATE, and both for the same reason: a preference
    only deserves to overrule the centre when it reflects structure. Past a
    metre of clearance, more room is not a better route — it just lets a
    stray far-corner cell tilt an empty room's scores past the tie tolerance.
    Past four metres of reach, deeper is not more promising either — and in
    an empty room the verified-floor boundary genuinely runs FURTHER on
    oblique bearings (the boundary is near-constant forward z, and
    r = z/cosθ), so an unsaturated depth bonus steers every open expanse
    toward its own FOV edge."""
    span = td.bearings[s_lo:s_hi]
    if len(span) < 2:
        return None
    width = float(abs(span[-1] - span[0]))
    if math.degrees(width) < OPEN_MIN_DEG and kind != "gateway":
        return None
    reach = prof[s_lo:s_hi]
    depth_term = (smooth if smooth is not None else prof)[s_lo:s_hi]
    score = np.minimum(neck[s_lo:s_hi], 1.0) + 0.05 * np.minimum(depth_term, 4.0)
    best = float(score.max())
    if best <= 0:
        return None
    tied = np.nonzero(score >= best - 0.05)[0]
    mid = 0.5 * (float(span[0]) + float(span[-1]))
    k = int(tied[np.lexsort((np.abs(span[tied]), np.abs(span[tied] - mid)))][0])
    return Opening(centre=float(span[k]), width=width,
                   reach=float(reach[k]), lo=s_lo, hi=s_hi,
                   kind=kind, island=island)


def openings(td: TopDown, prof: np.ndarray | None = None,
             neck: np.ndarray | None = None) -> list[Opening]:
    """The ways out of here, widest-and-deepest first.

    Structure comes from three detectors, in order of authority:
      1. threshold runs on the verified-passable profile — the base case;
      2. obstacle islands — an island's angular shadow CUTS any run it sits
         inside, regardless of how shallow the valley is relative to the
         peaks (§10.4: FORK_VALLEY is no longer the only fork judge);
      3. gateway synthesis — an island flank no run covers still gets an
         Opening, because "around the left of the chair" is a real way to go
         even when only 1.5 m of it is verified so far. Its candidate will be
         short, and says to look again from there.

    Iterating an Opening still yields (centre, width, reach)."""
    if prof is None or neck is None:
        prof, neck = passable_profile(td)
    if not len(prof):
        return []
    # STRUCTURE is read off a median-smoothed profile; SAFETY always reads the
    # raw one. The verified-floor boundary is a density threshold, so the raw
    # profile is ragged out where the samples thin — and those one-bin dips
    # are what _split_forks kept mistaking for walls, carving an empty room
    # into pieces whose centres then sat 20° off axis. A 2.5° median erases a
    # dip narrower than anything the body could not simply walk past, and a
    # real fork (an obstacle's angular shadow) is far wider than that.
    prof_s = median_filter(prof, size=5, mode="nearest")
    # Relative to the best BODY-passable direction, not the best sightline. On
    # the ray profile this cut deleted a real 2.8 m branch because some crack
    # elsewhere saw 6.0 m; on this profile the reference is itself somewhere the
    # robot can go, so "much shallower than the best way out" means what it says.
    # §2.3: the threshold is relative to the deepest way out — but it is
    # CAPPED at an absolute bar. A 9 m front used to raise the bar to 4 m and
    # silently swallow a 3–5 m side fan, and 4 m of verified side corridor is
    # a real direction for a body whatever the front is doing.
    thresh = max(OPEN_MIN_M,
                 min(OPEN_REL * float(prof_s.max()), ABS_SECTOR_MIN_M))
    runs, start = [], None
    for k in range(len(prof_s) + 1):
        is_open = k < len(prof_s) and float(prof_s[k]) >= thresh
        if is_open and start is None:
            start = k
        elif not is_open and start is not None:
            runs.append((start, k))
            start = None
    islands = _obstacle_islands(td)

    # 2. the structural cut: carve island interiors out of any run
    pieces: list[tuple[int, int]] = []
    for lo, hi in runs:
        cur = [(lo, hi)]
        for a, b in islands:
            nxt: list[tuple[int, int]] = []
            for c, d in cur:
                if a > c + 1 and b < d - 2:
                    nxt += [(c, a), (b + 1, d)]
                else:
                    nxt.append((c, d))
            cur = nxt
        pieces += cur

    out: list[Opening] = []
    for lo, hi in pieces:
        for s_lo, s_hi in _split_forks(prof_s[lo:hi], lo):
            # Only the SUB-piece that actually touches an island's shadow is
            # that island's flank; interior fork sub-pieces are their own
            # ways. Side by abutment: ends where the shadow starts → the
            # lower-index (right) flank; starts past it → the left flank.
            isl, side = next(((i, -1) for i, (a, b) in enumerate(islands)
                              if s_hi == a),
                             next(((i, +1) for i, (a, b) in enumerate(islands)
                                   if s_lo == b + 1), (-1, 0)))
            op = _make_opening(td, prof, neck, s_lo, s_hi, smooth=prof_s,
                               island=isl)
            if op is not None:
                op.side = side
                out.append(op)

    # 3. gateway synthesis for island flanks nothing covers yet
    for isl, (a, b) in enumerate(islands):
        for direction in (-1, +1):
            edge = a - 1 if direction < 0 else b + 1
            k = edge
            while 0 <= k < len(prof) and float(prof[k]) >= GATEWAY_MIN_M:
                k += direction
            lo, hi = (k + 1, a) if direction < 0 else (b + 1, k)
            if hi - lo < 2:
                continue
            claimed = next((o for o in out if not (o.hi <= lo or o.lo >= hi)), None)
            if claimed is not None:
                if claimed.island < 0:
                    claimed.island = isl
                    claimed.side = -1 if direction < 0 else +1
                continue
            op = _make_opening(td, prof, neck, lo, hi, smooth=prof_s,
                               kind="gateway", island=isl)
            if op is not None:
                op.side = -1 if direction < 0 else +1
                out.append(op)

    out.sort(key=lambda o: o.width * o.reach, reverse=True)   # wide AND deep first
    return out


def _candidate_for(td: TopDown, prof: np.ndarray, op: Opening,
                   op_rank: int = 0) -> Waypoint | None:
    """One point for one opening, landed on verified-FREE ground.

    Search the MIDDLE first, then outward inside this same opening. Only
    backing off along the centre bearing was not enough: pressed against a
    wall with the way out at −32°, the centre of that opening was itself
    unreachable and the proposer returned nothing at all — on the one frame
    where the model most needed to be told where the room was.

    Every offset bearing is measured with ITS OWN reach (§10.2). The old code
    computed the centre bearing's reach once and reused it for centre ± 4°,
    ± 8°…, so a sidestep inherited the centre's optimism and could land past
    its own floor — some of the 77 endpoints found on UNKNOWN got there
    exactly this way."""
    centre, width = op.centre, op.width
    min_r = MIN_WAYPOINT_TIGHT_M if op.kind == "gateway" else MIN_WAYPOINT_M
    offsets = [0.0]
    step = math.radians(4.0)
    k = 1
    while k * step <= width / 2:
        offsets += [k * step, -k * step]
        k += 1
    # Pass 1 wants the comfortable margin; pass 2 will accept a squeeze the
    # body still fits through, and SAYS it is a squeeze. An opening is never
    # dropped for being tight — that is the model's decision to make.
    for need in (MIN_CLEARANCE_M, TIGHT_CLEARANCE_M):
        for off in offsets:
            bearing = centre + off
            reach_b = _prof_at(td, prof, bearing)
            # §2.1: the point goes at the MID-RANGE of this bearing's own
            # verified reach — clamp(0.5·reach, floor, cap) — never at the
            # far edge of what happens to be visible. A fixed smaller
            # MAX_POINT_M cannot express "half of however open it is"; the
            # formula scales with the scene and stays monotone in reach.
            floor_r = (MIN_WAYPOINT_TIGHT_M if op.kind == "gateway"
                       else MIN_USEFUL_WAYPOINT_M)
            # §21.9 landing contract: the upper bound reserves safety margin
            # AND a forward runway — verified passable prefix left past the
            # landing, so the circle is a mid-route decision point and not a
            # wall-adjacent terminus. Near a wall the landing PULLS BACK to
            # afford the runway (the mid-range preference already yields
            # first to this cap); only when no useful landing can afford it
            # does the bearing fall back to the old margin-only bound and
            # the result is DEMOTED to short_verified_gateway staging —
            # never silently offered as an ordinary destination.
            r = min(max(0.5 * reach_b, floor_r),
                    reach_b - need - MIN_FORWARD_RUNWAY_M, MAX_POINT_M)
            short_runway = r < min_r
            if short_runway:
                r = min(max(0.5 * reach_b, floor_r),
                        reach_b - need - CELL_M, MAX_POINT_M)
            while r >= min_r:
                vx, vy = r * math.sin(bearing), r * math.cos(bearing)
                i, j = td.cell_of(vx, vy)
                # The landing must BE proven floor (§10.2: grid == FREE, not
                # merely "clearance looks fine" — clearance is a distance
                # field and is generous about cells nothing has ever seen),
                # must not sit in the route's own pinch, and the straight
                # line to it must fit the body.
                neck = corridor_bottleneck(td, bearing, r)
                if td.inside(i, j) and td.grid[i, j] == FREE \
                        and float(td.clearance[i, j]) >= need \
                        and float(td.clearance[i, j]) >= neck \
                        and _reachable(td, vx, vy, need=need - SAFETY_MARGIN_M
                                       if need > TIGHT_CLEARANCE_M
                                       else ROBOT_RADIUS_M):
                    mid = "the middle of" if abs(off) < 1e-6 else "inside"
                    tight = ("" if need > TIGHT_CLEARANCE_M else
                             f"; TIGHT — {2 * float(td.clearance[i, j]):.2f} m "
                             "across for a 0.40 m body, you would be squeezing")
                    if op.kind == "gateway":
                        side = "left" if bearing > 0 else "right"
                        note = (f"the way around on the {side}"
                                f"{tight or '; short — look again from there'}")
                    else:
                        note = (f"{mid} a {math.degrees(width):.0f}° "
                                f"opening{tight}")
                    w = Waypoint(angle=float(bearing), distance=float(r),
                                 clearance=float(td.clearance[i, j]),
                                 kind=op.kind, x_left=vx, y_fwd=vy, note=note)
                    if short_runway:
                        # §21.9: it exists, but it is a DIFFERENT promise —
                        # staging to look again from, not a through-route
                        w.extras["short_verified_gateway"] = True
                        w.confidence = "low"
                        w.note = ((w.note + "; " if w.note else "")
                                  + "SHORT VERIFIED GATEWAY — the proven "
                                    "floor ends just past this point; treat "
                                    "it as somewhere to look again from, "
                                    "not a through-route")
                    if op.island >= 0:
                        w.extras["island"] = op.island
                        # the FLANK, not sign(bearing) — see Opening.side
                        w.extras["side"] = op.side
                    w.extras["rank"] = op_rank
                    w.extras["sector_centre"] = float(centre)
                    return _finish_waypoint(td, w, landed_m=float(r),
                                            reach_m=float(reach_b))
                r -= CELL_M * 2  # back off along this bearing, then try the next
    return None


def _kept_partner(w: Waypoint, keep: list[Waypoint]) -> bool:
    """Is `w` the other flank of an island that already has a flank kept?"""
    isl, side = w.extras.get("island", -1), w.extras.get("side", 0)
    return isl >= 0 and side != 0 and any(
        p.extras.get("island", -1) == isl and p.extras.get("side", 0)
        not in (0, side) for p in keep)


def propose(td: TopDown, *, max_candidates: int = MAX_CANDIDATES) -> list[Waypoint]:
    """One point per DISTINCT opening; the cap comes after the structure.

    The old loop broke out as soon as two candidates existed, so whichever
    opening sorted first could spend both slots on near-duplicates while a
    real second branch was never even attempted (§10.4). Now every opening
    and gateway lands a candidate first; near-duplicates collapse; and only
    then is the list capped — with the guarantee that the two sides of an
    obstacle island are never collapsed into each other and a kept gateway
    pulls its partner in with it."""
    prof, neck = passable_profile(td)
    built: list[Waypoint] = []
    for rank, op in enumerate(openings(td, prof, neck)):
        w = _candidate_for(td, prof, op, op_rank=rank)
        if w is not None:
            built.append(w)

    # §2.2 — the FINAL dedup, on final waypoint bearings, after every
    # opening/gateway has landed its point. Two bearings closer than
    # MIN_CANDIDATE_SEPARATION_DEG are the same road to a body, island tags
    # notwithstanding. The representative is picked by: longer verified
    # reach, then wider squeeze, then nearer its own sector centre, then the
    # stable build order; what it absorbed is remembered for telemetry
    # ("this way also splits around an obstacle further in"), not drawn as a
    # second overlapping circle.
    order = {id(w): k for k, w in enumerate(built)}
    picked: list[Waypoint] = []
    for w in sorted(built, key=lambda x: (
            -round(x.verified_m, 1),
            -round(float(x.extras.get("squeeze", x.clearance)), 2),
            abs(x.angle - float(x.extras.get("sector_centre", x.angle))),
            order[id(x)])):
        clash = next((p for p in picked
                      if abs(math.degrees(p.angle - w.angle))
                      < MIN_CANDIDATE_SEPARATION_DEG), None)
        if clash is not None:
            clash.extras.setdefault("merged_bearings_deg", []).append(
                round(math.degrees(w.angle)))
            # the survivor INHERITS the absorbed road's standing: its better
            # sector rank (or the cap could drop the principal opening whose
            # representative happened to be the minor fork) and its island
            # pairing (audit P1)
            clash.extras["rank"] = min(clash.extras.get("rank", 0),
                                       w.extras.get("rank", 0))
            if clash.extras.get("island", -1) < 0 <= w.extras.get("island", -1):
                clash.extras["island"] = w.extras["island"]
                clash.extras["side"] = w.extras.get("side", 0)
            continue
        picked.append(w)
    picked.sort(key=lambda w: w.extras.get("rank", 0))   # sector importance
    if len(picked) > max_candidates:
        keep = picked[:max_candidates]
        for w in picked[max_candidates:]:
            if _kept_partner(w, keep):
                # diversity beats depth: displace the last kept candidate
                # that is NOT itself half of a kept island pair. The first
                # version demanded island < 0 — "not island-tagged at all" —
                # but the tag also lands on lone flanks and fork sub-pieces
                # whose partner never produced a candidate, so in cluttered
                # scenes there was no displaceable slot and the genuine
                # second way around a centred obstacle was silently dropped.
                for p in reversed(keep):
                    if p is not w and not _kept_partner(p, [q for q in keep
                                                           if q is not p]):
                        keep[keep.index(p)] = w
                        break
        picked = keep
    picked.sort(key=lambda w: w.angle, reverse=True)      # left → right
    if picked:
        # When nothing on offer is even a couple of metres out, the body is not
        # choosing a destination — it is boxed in and shuffling. Say so. The
        # geometry is honest either way (measured seven steps into R2R ep 7:
        # every bearing body-passable to at most 1.6 m while the eye ran to
        # 9.3 m — chairs at arm's length with the room visible past them), but
        # a 1.1 m shuffle described in the same words as a 6 m corridor reads as
        # a plan, and the reason it is short is the thing worth knowing.
        # Judged on VERIFIED ground, not on the aim: a far point past unproven
        # floor does not mean the body is free to move.
        if max(w.verified_m or w.distance for w in picked) < REPROPOSE_M:
            for w in picked:
                w.extras["boxed_in"] = True
                w.note = ((w.note + "; " if w.note else "")
                          + "hemmed in — this is only somewhere to shuffle to, "
                            "not a destination; look again from there")
        return picked
    # Never hand back an empty list: a boxed-in robot told "no options" has
    # nothing to reason about. Offer the single roomiest short hop that is
    # still safe, and if even that fails the caller says "turn and look".
    bearing, reach = td.widest()
    r = min(reach - MIN_CLEARANCE_M - CELL_M, 1.5)
    if r >= 0.6:
        vx, vy = r * math.sin(bearing), r * math.cos(bearing)
        i, j = td.cell_of(vx, vy)
        # Even the last resort keeps the executable contract: FREE landing,
        # FREE prefix. A desperate hop onto unproven ground is still a hop
        # onto unproven ground.
        if (td.inside(i, j) and td.grid[i, j] == FREE
                and _reachable(td, vx, vy) and _prefix_free(td, bearing, r)):
            w = Waypoint(angle=float(bearing), distance=float(r),
                         clearance=float(td.clearance[i, j]), kind="opening",
                         x_left=vx, y_fwd=vy, confidence="low",
                         note="the only bit of room I can find from here")
            return [_finish_waypoint(td, w, landed_m=float(r))]
    return []


def propose_from_depth(depth_field: Any, **kw) -> tuple[list[Waypoint], TopDown | None]:
    """Convenience: wire form → candidates. Never raises — a proposer that
    crashes must not end an episode."""
    depth = decode_depth(depth_field) if not isinstance(depth_field, np.ndarray) \
        else depth_field
    if depth is None or depth.ndim != 2:
        return [], None
    try:
        td = build_topdown(depth, **kw)
        return propose(td), td
    except Exception:  # noqa: BLE001
        return [], None


# ── 3b. staying committed to a place instead of re-choosing one ──────────
# ── §global-dwp (user idea 2026-08-12 evening): propose from MEMORY too ──
# The fresh proposer only sees the current 90° wedge; a person also offers
# themselves the corridor they SAW a moment ago on the right. Memory
# candidates come from the accumulated map's verified FREE — outside the
# current view, behind included — and goto's existing turn-then-revalidate
# machinery re-earns them with fresh eyes before a single forward is spent.
MEM_MIN_PREFIX_M = 2.0       # a remembered way must be worth the turn
MEM_MAX_CANDIDATES = 2       # appended AFTER the fresh ones, never instead
MEM_AIM_CAP_M = 4.0
MEM_FOV_GUARD_DEG = 50.0     # the current wedge belongs to the fresh proposer
MEM_STEP_DEG = 3.0
MEM_MAX_R_M = 6.0
MEM_MIN_UPDATES = 2          # one look is a frame, not a memory


def memory_profile(amap: Any, *, step_deg: float = MEM_STEP_DEG,
                   max_r: float = MEM_MAX_R_M
                   ) -> tuple[np.ndarray, np.ndarray]:
    """360° verified-FREE prefix from the ACCUMULATED map, body frame:
    for every bearing, how far could the body walk in a straight line over
    ground it has PROVEN free (with body clearance), regardless of where
    the camera points now."""
    from scipy.ndimage import distance_transform_edt
    free = amap.logodds < -0.5
    occ = amap.logodds > 0.5
    clear = distance_transform_edt(~occ) * amap.cell
    bearings = np.arange(-180.0, 180.0, step_deg)
    radii = np.arange(SELF_FOOTPRINT_M, max_r, amap.cell)
    b = np.radians(bearings)[:, None]
    x_left = radii[None, :] * np.sin(b)
    y_fwd = radii[None, :] * np.cos(b)
    xa, ya = amap.to_anchor(x_left, y_fwd)
    i, j = amap.cells(xa, ya)
    ok = (i >= 0) & (i < amap.n) & (j >= 0) & (j < amap.n)
    ii, jj = np.clip(i, 0, amap.n - 1), np.clip(j, 0, amap.n - 1)
    walkable = ok & free[ii, jj] & (clear[ii, jj] >= TIGHT_CLEARANCE_M)
    # prefix = contiguous walkable run from the body outward
    blocked = ~walkable
    first_block = np.where(blocked.any(axis=1),
                           blocked.argmax(axis=1), walkable.shape[1])
    prefix = np.where(first_block > 0,
                      radii[np.clip(first_block - 1, 0, len(radii) - 1)],
                      0.0)
    prefix[first_block == 0] = 0.0
    return bearings, prefix


def propose_from_memory(amap: Any, existing: list[Waypoint],
                        *, fov_guard_deg: float = MEM_FOV_GUARD_DEG
                        ) -> list[Waypoint]:
    """Numbered places OUTSIDE the current view, earned from accumulated
    evidence. Honesty contract: verified_m is the MEMORY's proven prefix,
    visible_m is 0 (the camera cannot see it from here), the note says it
    is a memory, and execution still passes goto's post-turn corridor
    revalidation — remembered floor proposes, fresh eyes confirm."""
    if not getattr(amap, "trusted", False) \
            or int(getattr(amap, "updates", 0)) < MEM_MIN_UPDATES:
        return []
    bearings, prefix = memory_profile(amap)
    good = (prefix >= MEM_MIN_PREFIX_M) & (np.abs(bearings) > fov_guard_deg)
    if not good.any():
        return []
    # contiguous runs over the circular bearing axis
    order = np.argsort(bearings)
    runs: list[list[int]] = []
    cur: list[int] = []
    for k in order:
        if good[k]:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    # wrap-around join (…,179°] + [-180°,…)
    if len(runs) > 1 and good[order[0]] and good[order[-1]]:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    cands: list[Waypoint] = []
    for run in runs:
        span = MEM_STEP_DEG * len(run)
        if span < OPEN_MIN_DEG:
            continue
        centre_k = max(run, key=lambda k: prefix[k])
        bdeg = float(bearings[centre_k])
        reach = float(prefix[centre_k])
        aim = min(max(0.5 * reach, MIN_USEFUL_WAYPOINT_M),
                  reach - 2 * CELL_M, MEM_AIM_CAP_M)
        if aim < MIN_USEFUL_WAYPOINT_M:
            continue
        b = math.radians(bdeg)
        w = Waypoint(angle=b, distance=aim, clearance=ROBOT_RADIUS_M,
                     kind="remembered",
                     x_left=aim * math.sin(b), y_fwd=aim * math.cos(b),
                     stride_m=round(aim, 2), verified_m=round(reach, 2),
                     visible_m=0.0, confidence="medium",
                     note="from memory — outside your current view; goto "
                          "turns you there and re-checks with fresh eyes "
                          "before walking")
        w.extras["stride_m"] = w.stride_m
        cands.append(w)
    # dedup against everything already on the table, then against each other
    kept: list[Waypoint] = []
    for w in sorted(cands, key=lambda w: -w.verified_m):
        deg = math.degrees(w.angle)
        others = existing + kept
        if any(abs((math.degrees(o.angle) - deg + 180.0) % 360.0 - 180.0)
               < MIN_CANDIDATE_SEPARATION_DEG for o in others):
            continue
        kept.append(w)
        if len(kept) >= MEM_MAX_CANDIDATES:
            break
    return kept


ARRIVE_M = REPROPOSE_M       # this close and it stops being a destination
GOAL_MAX_BEARING = math.radians(75.0)   # …turned further than this, it is behind you


class WaypointGoal:
    """Keeps the place last chosen, so walking toward it actually approaches it.

    `propose` is memoryless: it re-reads the frame and puts a point at the far
    end of whatever it can reach. Take a step and it re-reads and does it again,
    one step further out. Measured down a plain corridor, eight steps in a row:

        step 0   straight ahead, 3.00 m       step 4   straight ahead, 3.00 m
        step 2   straight ahead, 3.00 m       step 7   left 11°,       3.00 m

    Always 3.00 m — the pacing ceiling — while the point itself slid from 3.0 m
    to 4.7 m out in the world. The distance on the panel never changes because
    the target runs away exactly as fast as the body walks. It is a carrot on a
    stick: there is no approaching it, no arriving at it, and nothing on screen
    that says the last four steps accomplished anything.

    So hold the point still instead. It is stored in the ANCHOR frame — the
    accumulated map's private origin — and converted back to an egocentric
    bearing and range every frame, so the metres shrink as the body closes in:
    3.0, 2.75, 2.5 … and at ARRIVE_M it is reached and a fresh one is chosen.
    Nothing about this reaches the model as a coordinate; it sees the same
    "left 11°, 2.5 m away" it always saw. What changes is that the number now
    means something.

    The commitment is dropped the moment it stops being honest — reached,
    turned away from, no longer standing on proven floor, or no longer
    reachable in a straight line. A stale goal is worse than none: it would
    invite the body through whatever moved into the way.
    """

    __slots__ = ("anchor", "kind", "adopted_at", "adopted_m", "track_id")

    def __init__(self) -> None:
        self.anchor: tuple[float, float] | None = None
        self.kind: str = ""
        self.adopted_at: int = 0
        self.adopted_m: float = 0.0
        self.track_id: int = 0             # §21.6: the registry identity held

    def clear(self) -> None:
        self.anchor, self.kind, self.track_id = None, "", 0

    def adopt(self, amap: Any, w: Waypoint) -> None:
        self.anchor = tuple(float(v) for v in amap.to_anchor(w.x_left, w.y_fwd))
        self.kind = w.kind
        self.adopted_at = int(getattr(amap, "updates", 0))
        self.adopted_m = float(w.distance)
        self.track_id = int(w.extras.get("track_id", 0))

    def arrive_m(self) -> float:
        """The release radius SCALES with the commitment. Mid-range aiming
        (§2.1) legitimately adopts 2 m goals; a fixed 2.5 m radius released
        those the instant they were taken and the offered place went back to
        crawling forward with the feet — the exact treadmill the commitment
        exists to kill. 40 % of the adopted distance, floored at 0.8 m so
        arrival stays a real event, capped at the classic ARRIVE_M."""
        if self.adopted_m <= 0:
            return ARRIVE_M
        return min(ARRIVE_M, max(0.8, 0.4 * self.adopted_m))

    def held(self, amap: Any, td: TopDown) -> Waypoint | None:
        """The commitment as seen from where the body is NOW, or None if it has
        expired. Every rejection below is a way it could quietly become a lie."""
        if self.anchor is None or amap is None:
            return None
        x_left, y_fwd = (float(v) for v in amap.to_body(*self.anchor))
        dist = math.hypot(x_left, y_fwd)
        bearing = math.atan2(x_left, y_fwd)
        if dist <= self.arrive_m():
            # Close enough that it is no longer somewhere to go — and by now
            # the body has advanced, so the view reaches further than it did
            # when this was chosen. Re-choosing HERE is what makes the offered
            # place travel forward through the world instead of being walked
            # down to underfoot. The radius scales with the adopted distance —
            # see arrive_m().
            self.clear()
            return None
        if abs(bearing) > GOAL_MAX_BEARING or y_fwd <= 0.05:
            self.clear()                           # turned away from it
            return None
        i, j = td.cell_of(x_left, y_fwd)
        if not td.inside(i, j):
            self.clear()
            return None
        if td.grid[i, j] == OCCUPIED:
            self.clear()                           # something is standing there
            return None
        # The TARGET may sit on unproven ground — a destination several metres
        # off usually does, and demanding FREE there dropped the commitment
        # every few steps (measured: a 5.1 m goal released, then 1.1 m
        # offered twice in a row).
        #
        # The STRIDE may not. §2.5 of the revision plan: the executable part
        # of a held goal is re-derived from the CURRENT frame's continuous
        # verified-FREE prefix, exactly the bar fresh candidates meet. The
        # old "nothing OCCUPIED across the way in" scan let a stride cross
        # OPEN/UNKNOWN and then WROTE those metres into verified_m — a
        # fabricated number on the one field whose whole point is honesty.
        clear = float(td.clearance[i, j])
        prefix = _free_prefix(td, bearing)
        # WHY the prefix ended matters. Ended at OCCUPIED — something solid
        # stands ACROSS the straight line to the goal: the commitment as a
        # straight-line destination is dead, drop it (the model re-chooses,
        # or a route leg goes around). Ended at OPEN/UNKNOWN — merely
        # unverified: the place is still where it was, keep the MEMORY but
        # offer no executable held waypoint until the prefix returns.
        if prefix < dist - CELL_M:
            nr = max(prefix, SELF_FOOTPRINT_M) + CELL_M
            ii, jj = td.cell_of(nr * math.sin(bearing), nr * math.cos(bearing))
            if td.inside(ii, jj):
                g = td.grid[ii, jj]
                # OCCUPIED across the ray, or verified floor pinched below
                # body clearance by something solid beside it — either way a
                # REAL obstruction, not missing evidence.
                if g == OCCUPIED or (g == FREE and float(
                        td.clearance[ii, jj]) < TIGHT_CLEARANCE_M):
                    self.clear()
                    return None
        if prefix < MIN_WAYPOINT_TIGHT_M:
            return None
        near = min(dist, prefix)
        if not _reachable(td, x_left * near / dist, y_fwd * near / dist,
                          ROBOT_RADIUS_M):
            self.clear()                           # something moved into the way
            return None
        squeeze = _ray_min_clearance(td, x_left * near / dist,
                                     y_fwd * near / dist)
        stride = min(_stride_cap(squeeze), near)
        half = float(td.bearings[-1]) if len(td.bearings) else 0.0
        visible = (_prof_at(td, td.free_range, bearing)
                   if abs(bearing) <= half else 0.0)
        w = Waypoint(angle=bearing, distance=dist, clearance=clear,
                     kind=self.kind or "opening", x_left=x_left, y_fwd=y_fwd,
                     stride_m=round(float(stride), 2),
                     verified_m=round(float(prefix), 2),
                     visible_m=float(visible), confidence="medium",
                     note="you were already heading here")
        w.extras["stride_m"] = w.stride_m
        w.extras["squeeze"] = round(float(squeeze), 2)
        if self.track_id:
            w.extras["track_id"] = self.track_id   # identity survives holding
        return w

    def apply(self, amap: Any, td: TopDown, fresh: list[Waypoint], *,
              adopt: bool = True) -> list[Waypoint]:
        """Held goal first, then the freshly proposed places that are somewhere
        else. With `adopt` (the monitor's mode) an empty hand takes the best
        fresh place immediately; the agent toolset passes adopt=False and
        adopts only what the model actually chooses in goto() — a commitment
        the model never made is not a commitment."""
        held = self.held(amap, td)
        if held is None:
            if adopt and fresh:
                self.adopt(amap, fresh[0])
            return fresh
        # Drop fresh proposals that are really the same place seen again, or the
        # list becomes "here, and also here" and the second slot is wasted.
        others = [w for w in fresh
                  if abs(math.degrees(w.angle - held.angle))
                  >= MIN_CANDIDATE_SEPARATION_DEG
                  or math.hypot(w.x_left - held.x_left,
                                w.y_fwd - held.y_fwd) > 3.0]
        return [held, *others][:MAX_CANDIDATES]


# ── 3b′. the waypoint registry: anchor-frame candidate identity (§21.6) ──
@dataclass
class WaypointTrack:
    """One physical place's standing identity. The anchor centre is FIXED at
    birth — evidence, freshness and eligibility update; the circle does not
    move. A place that genuinely moved is a different place (new track)."""
    tid: int
    ax: float
    ay: float
    kind: str
    born_epoch: int
    last_epoch: int
    evidence: int = 1
    suspended: bool = False        # current frame cannot verify the centre
    retired: str = ""              # "" = alive; else the retire reason


class WaypointRegistry:
    """§21.6 — two-phase truth for candidates.

    ``propose()`` stays the per-frame SAFETY GENERATOR; this registry gives
    its output identity. Every fresh waypoint converts to anchor metres and
    either matches an existing live track (by anchor distance — never by
    bearing alone) or founds a new one. A match SNAPS the candidate onto the
    track's fixed centre and re-derives the executable numbers from the
    CURRENT frame at that centre — identity from memory, safety from now; a
    global point never skips the current-depth gate.

    Eligibility is dynamic, position is not:
      * centre confirmed OCCUPIED in the accumulated map → retired
      * left behind the body → retired (a re-found exit is a new track)
      * unseen for STALE_EPOCHS menus → retired quietly
      * centre currently UNKNOWN / unverifiable → SUSPENDED this round
        (dropped from the callable menu, never nudged to a new spot)
    """

    MATCH_M = 0.55                 # §21.6: anchor gate, mid of the 0.4-0.6 band
    BEHIND_M = 0.4
    STALE_EPOCHS = 60

    def __init__(self) -> None:
        self.tracks: dict[int, WaypointTrack] = {}
        self._next = 1

    def _match(self, ax: float, ay: float) -> WaypointTrack | None:
        best, best_d = None, self.MATCH_M
        for t in self.tracks.values():
            if t.retired:
                continue
            d = math.hypot(t.ax - ax, t.ay - ay)
            if d < best_d:
                best, best_d = t, d
        return best

    def reconcile(self, amap: Any, td: TopDown | None,
                  waypoints: list[Waypoint], *, epoch: int) -> list[Waypoint]:
        """Give this round's candidates their standing identity; cull the
        ones whose track cannot be verified from here. Returns the menu-
        eligible list (order preserved); every kept waypoint carries
        ``extras['track_id']``."""
        out: list[Waypoint] = []
        for w in waypoints:
            if w.kind == "remembered":
                out.append(w)          # map-identity by construction
                continue
            ax, ay = (float(v) for v in amap.to_anchor(w.x_left, w.y_fwd))
            t = self._match(ax, ay)
            if t is None:
                t = WaypointTrack(self._next, ax, ay, w.kind, epoch, epoch)
                self.tracks[self._next] = t
                self._next += 1
                w.extras["track_id"] = t.tid
                out.append(w)
                continue
            t.evidence += 1
            t.last_epoch = epoch
            w.extras["track_id"] = t.tid
            bx, by = (float(v) for v in amap.to_body(t.ax, t.ay))
            moved = math.hypot(bx - w.x_left, by - w.y_fwd)
            if moved <= CELL_M:        # same cell — nothing to re-derive
                t.suspended = False
                out.append(w)
                continue
            # snap onto the track centre, then re-earn the numbers from the
            # CURRENT frame at that centre
            ok = False
            if td is not None and by > 0.05:
                i, j = td.cell_of(bx, by)
                if td.inside(i, j) and td.grid[i, j] == FREE \
                        and float(td.clearance[i, j]) >= TIGHT_CLEARANCE_M:
                    w.x_left, w.y_fwd = bx, by
                    w.angle = float(math.atan2(bx, by))
                    w.distance = float(math.hypot(bx, by))
                    w.clearance = float(td.clearance[i, j])
                    prefix = _free_prefix(td, w.angle)
                    w.verified_m = round(float(prefix), 2)
                    w.continuation_m = round(
                        max(0.0, float(prefix) - w.distance), 2)
                    squeeze = _ray_min_clearance(td, bx, by)
                    w.stride_m = round(min(_stride_cap(squeeze),
                                           w.distance, float(prefix)), 2)
                    w.extras["squeeze"] = round(squeeze, 2)
                    w.extras["stride_m"] = w.stride_m
                    ok = True
            if ok:
                t.suspended = False
                out.append(w)
            else:
                # §21.6 rule 4: cannot verify the fixed centre from here —
                # suspend eligibility, never move the circle to keep it
                t.suspended = True
        self._cull(amap, epoch)
        return out

    def _cull(self, amap: Any, epoch: int) -> None:
        for t in self.tracks.values():
            if t.retired:
                continue
            i, j = amap.cells(t.ax, t.ay)
            if 0 <= int(i) < amap.n and 0 <= int(j) < amap.n \
                    and float(amap.logodds[int(i), int(j)]) > 1.0:
                t.retired = "occupied"
                continue
            bx, by = (float(v) for v in amap.to_body(t.ax, t.ay))
            if by < -self.BEHIND_M:
                t.retired = "behind"
                continue
            if epoch - t.last_epoch > self.STALE_EPOCHS:
                t.retired = "stale"

    def alive(self) -> list[WaypointTrack]:
        return [t for t in self.tracks.values() if not t.retired]


# ── 3c. the loop monitor: deterministic "you are going in circles" ───────
class LoopMonitor:
    """§7: a resident, deterministic circle detector over the map's OWN
    trail — no GT pose, no model call. One signal never convicts; a warning
    needs at least two of: (a) the body is back within WINDOW_M of a trail
    point it left ≥ MIN_GAP_M of walking ago; (b) recent path length far
    exceeds net displacement; (c) the recent waypoint choices keep taking
    the same angular sector; (d) the map has stopped growing while the legs
    keep spending. The output is evidence — matched point, counts, metres —
    for the model to read, never a locked door."""

    WINDOW_M = 1.2
    MIN_GAP_M = 3.0
    SECTOR_DEG = 30.0
    SIG_MATCH = 0.6          # §14.10: mean |Δsig| below this = same view
    COOLDOWN = 5             # assess calls a warning stays quiet after firing

    def __init__(self) -> None:
        self.sector_hist: list[int] = []
        self.growth_hist: list[int] = []
        self.warnings = 0
        self.last: dict | None = None
        # §14.10: an INDEPENDENT visual channel. The trail signals all read
        # the same dead-reckoned pose, so odometry drift moves them
        # together; coarse frame signatures tied to where they were taken
        # give the monitor one witness that never consulted the odometry.
        self.frames: list[tuple[float, float, Any]] = []
        self._cur_sig: Any = None
        self._cooldown = 0

    def note_choice(self, bearing_deg: float) -> None:
        self.sector_hist.append(int(round(bearing_deg / self.SECTOR_DEG)))
        del self.sector_hist[:-8]

    def note_growth(self, known_cells: int) -> None:
        self.growth_hist.append(int(known_cells))
        del self.growth_hist[:-10]

    def note_frame(self, sig: Any, pose_xy: tuple[float, float]) -> None:
        """A coarse signature of the CURRENT view at the CURRENT pose."""
        if sig is None:
            return
        self._cur_sig = sig
        x, y = float(pose_xy[0]), float(pose_xy[1])
        # dedupe only genuinely-stationary frames (turning in place): the
        # threshold must sit BELOW one forward step (0.25 m) or every step
        # replaces the last entry and the ring never grows past one point
        if self.frames and math.hypot(
                x - self.frames[-1][0], y - self.frames[-1][1]) < 0.15:
            self.frames[-1] = (x, y, sig)
            return
        self.frames.append((x, y, sig))
        del self.frames[:-400]

    def assess(self, amap: Any, *, tick: bool = True) -> dict | None:
        """tick=False for mid-leg (per-primitive) calls: the cooldown
        window is counted in OBSERVATIONS, not metres — otherwise a long
        leg's every-4-primitive checks burned the whole cooldown before
        the model ever saw the first warning (review P1)."""
        trail = getattr(amap, "trail", None)
        if not trail or len(trail) < 8:
            self.last = None
            return None
        pts = trail[-60:]
        segs = [math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(pts, pts[1:])]
        path = float(sum(segs))
        net = float(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
        cx, cy = trail[-1][0], trail[-1][1]
        revisit_at = None
        walked_back = 0.0
        for k in range(len(trail) - 2, -1, -1):
            a, b = trail[k], trail[k + 1]
            walked_back += math.hypot(b[0] - a[0], b[1] - a[1])
            if walked_back >= self.MIN_GAP_M and math.hypot(
                    trail[k][0] - cx, trail[k][1] - cy) <= self.WINDOW_M:
                revisit_at = k
                break
        sector_repeat = 0
        if self.sector_hist:
            last6 = self.sector_hist[-6:]
            sector_repeat = max(last6.count(s) for s in set(last6))
        stalled_map = (len(self.growth_hist) >= 4
                       and self.growth_hist[-1] - self.growth_hist[-4] < 25
                       and path > 4.0)
        # §14.10: the visual witness — only meaningful where the trail
        # already claims a revisit; a stored view from that spot that also
        # LOOKS like the current view is evidence odometry cannot fake
        # (and its absence keeps a drifted trail from convicting alone).
        revisit_visual = False
        if revisit_at is not None and self._cur_sig is not None:
            for (fx, fy, sig) in self.frames[:-3]:
                if (math.hypot(fx - cx, fy - cy) <= self.WINDOW_M
                        and sig is not None
                        and getattr(sig, "shape", None) == getattr(
                            self._cur_sig, "shape", None)):
                    try:
                        import numpy as _np
                        if float(_np.abs(_np.asarray(sig)
                                         - _np.asarray(self._cur_sig)
                                         ).mean()) < self.SIG_MATCH:
                            revisit_visual = True
                            break
                    except Exception:  # noqa: BLE001
                        pass
        signals = {
            "revisit": revisit_at is not None,
            "revisit_visual": revisit_visual,
            "circling": path > 4.0 and net < 0.35 * path,
            "sector_repeat": sector_repeat >= 4,
            "map_stalled": bool(stalled_map),
        }
        n = sum(signals.values())
        score = round(n / 5.0, 2)
        warning = n >= 2
        # §14.10 hysteresis: one physical loop is ONE warning, not one per
        # assess. After firing, the monitor stays quiet for COOLDOWN calls
        # unless the evidence first drops away (loop actually left).
        suppressed = False
        if warning and self._cooldown > 0:
            suppressed, warning = True, False
        out = {
            "loop_score": score,
            "signals": {k: v for k, v in signals.items() if v},
            "path_m": round(path, 1), "net_m": round(net, 1),
            "revisit_trail_index": revisit_at,
            "repeated_sector_deg": (int(max(set(self.sector_hist[-6:]),
                                            key=self.sector_hist[-6:].count)
                                        * self.SECTOR_DEG)
                                    if sector_repeat >= 4 else None),
            "warning": warning,
            **({"suppressed": True} if suppressed else {}),
        }
        if warning:
            self.warnings += 1
            self._cooldown = self.COOLDOWN
        elif tick:
            if n <= 1:
                self._cooldown = 0           # evidence gone — re-arm
            elif self._cooldown > 0:
                self._cooldown -= 1
        self.last = out
        return out if out["warning"] else None


def fuse_sweep_views(amap: Any, picked: list, *, scale_m: float | None = None,
                     range_cap_m: float = RANGE_CAP_M) -> int:
    """Fuse a fan of same-pose renders the way REAL turning would: ONE
    single-frame TopDown per heading, the pose rotated to that heading,
    one fuse() per view, pose restored exactly afterwards.

    The first sweep merged the fan into one wedge via build_topdown_pano
    and fused ONCE — every singly-seen cell sat at ±0.8 logodds, below the
    ±0.5 'known' threshold, and the sweep map knew 78% LESS than an actual
    out-and-back on the same pixels (A/B at the EP3 start pose: 417 vs
    1835 known cells, sign contradictions 8). Per-view fusion IS the
    honest equivalent of the user's 'turn left four notches mapping each'.

    `picked` is [(signed_heading_deg, view_dict), …]. Returns the number
    of views fused."""
    if not picked:
        return 0
    fused = 0
    theta0 = float(amap.theta)
    try:
        for sh, v in picked:
            dep = decode_panorama_depth(v, scale_m)
            if dep is None:
                continue
            try:
                td = build_topdown(dep, range_cap_m=range_cap_m, scale_m=1.0)
            except Exception:  # noqa: BLE001 — one bad view must not end a sweep
                continue
            amap.theta = theta0 + math.radians(float(sh))
            amap.fuse(td)
            fused += 1
    finally:
        amap.theta = theta0
    if fused:
        # §21.2 trust provenance: N wedges from ONE known pose — geometry
        # exact by construction, usable for planning without a match score.
        # Expires on the first real translation (see odometry()).
        amap.sweep_trust = True
    return fused


def frame_signature(rgb_b64: Any) -> "np.ndarray | None":
    """§14.10: a 16×16 contrast-normalised gray thumbnail — cheap enough to
    take every sensed frame, distinctive enough to say "this view again"."""
    g = _gray256(rgb_b64)
    if g is None:
        return None
    h, w = g.shape[:2]
    bh, bw = h // 16, w // 16
    if bh == 0 or bw == 0:
        return None
    m = np.asarray(g, np.float32)[:bh * 16, :bw * 16]
    m = m.reshape(16, bh, 16, bw).mean(axis=(1, 3))
    m = m - m.mean()
    s = float(np.abs(m).mean()) or 1.0
    return (m / s).astype(np.float32)


# ── 4. what the model sees ───────────────────────────────────────────────
def compose_panorama(views: list[dict], waypoints: list[Waypoint] | None = None,
                     floor_y: float | None = None) -> bytes:
    """Four directions as ONE readable contact sheet (§5.2).

    Panels are ordered AHEAD · LEFT · BEHIND · RIGHT, each labelled with its
    direction and heading; the AHEAD panel carries the same numbered
    candidate circles as the normal view, so a place seen here is a place
    goto(place) can take. The model never again receives four loose images
    with instructions to type twelve 3s — turning is face()'s job."""
    if PILImage is None or not views:
        return b""
    by_heading: dict[int, dict] = {}
    for v in views:
        by_heading[int(round(float(v.get("heading_deg") or 0.0))) % 360] = v
    order = [(0, "AHEAD"), (90, "LEFT"), (180, "BEHIND"), (270, "RIGHT")]
    panels: list[Any] = []
    for deg, label in order:
        v = by_heading.get(deg)
        if v is None or not v.get("rgb_base64"):
            continue
        png = base64.b64decode(v["rgb_base64"])
        if deg == 0 and waypoints and floor_y is not None:
            png = annotate_rgb(png, waypoints, floor_y)
        img = PILImage.open(BytesIO(png)).convert("RGB")
        if img.width > 320:
            img = img.resize((320, int(img.height * 320 / img.width)))
        strip = PILImage.new("RGB", (img.width, 18), (12, 12, 16))
        ImageDraw.Draw(strip).text(
            (6, 3), f"{label} ({deg:+d}°)" if deg else f"{label} (0°)",
            font=caption_font(12), fill=(240, 220, 130))
        cell = PILImage.new("RGB", (img.width, img.height + 18))
        cell.paste(strip, (0, 0))
        cell.paste(img, (0, 18))
        panels.append(cell)
    if not panels:
        return b""
    w = sum(p.width for p in panels) + 4 * (len(panels) - 1)
    h = max(p.height for p in panels)
    sheet = PILImage.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 4
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


WAYPOINT_EDGE_MARGIN_PX = 10   # §21.8: circle + number must clear the frame


def project_waypoint_to_rgb(w: Waypoint, floor_y: float, width: int,
                            height: int, *, hfov_deg: float = FOV_DEG
                            ) -> tuple[float, float, int, bool]:
    """§21.8 — THE one projection: (u, v, radius_px, fully_visible).

    Candidate table, RGB annotation, map and goto all consume the same
    verdict, so a number the model can call is a circle it can actually see.
    `fully_visible` demands the circle AND its number label sit inside the
    frame with a margin — a place too close (projects below the frame), at
    the FOV edge, or behind the camera is NOT visible, and per the callable
    contract it must then not be clamped into the picture as if it were."""
    if w.y_fwd <= 0.1:
        return 0.0, 0.0, 0, False
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = width / 2.0 + (-w.x_left) * f / w.y_fwd
    v = height / 2.0 - floor_y * f / w.y_fwd
    # user 2026-08-15: bigger rings — the old max(9, 26/d) was hard to
    # spot on a 512px frame
    r = max(14, int(38 / max(1.0, w.y_fwd)))
    m = WAYPOINT_EDGE_MARGIN_PX
    visible = (u - r - m >= 0 and u + r + m < width
               and v - r - 24 >= 0          # the number + backing sit above
               and v + r + m < height)
    return u, v, r, visible


def is_waypoint_visible(w: Waypoint, floor_y: float, width: int, height: int,
                        *, hfov_deg: float = FOV_DEG) -> bool:
    return project_waypoint_to_rgb(w, floor_y, width, height,
                                   hfov_deg=hfov_deg)[3]


def annotate_rgb(rgb_png: bytes, waypoints: list[Waypoint], floor_y: float,
                 *, hfov_deg: float = FOV_DEG) -> bytes:
    """Numbered circles where the waypoints actually stand on the floor.

    Only geometry crosses into the picture; no depth map, no grid, no metres in
    a world frame — the model reads its own first-person view with the options
    marked on it, exactly as the learned-predictor strip did.

    §21.8: NO off-frame clamping. The old edge-clamped ring + wedge was a
    debug indicator sold as a callable place — 'the list has a point but the
    picture shows it nowhere near there'. A waypoint that fails the shared
    projection gate simply is not drawn here; its number lives on the MAP
    (memory candidates), or the menu filter has already removed it."""
    if PILImage is None:
        return rgb_png
    try:
        img = PILImage.open(BytesIO(rgb_png)).convert("RGB")
    except Exception:  # noqa: BLE001
        return rgb_png
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = caption_font(18)
    for n, w in enumerate(waypoints, 1):
        u, v, r, visible = project_waypoint_to_rgb(w, floor_y, W, H,
                                                   hfov_deg=hfov_deg)
        if not visible:
            continue                      # numbering is preserved: n still counts
        colour = (255, 205, 70) if w.kind == "gateway" else (90, 230, 130)
        draw.ellipse([u - r, v - r, u + r, v + r], outline=colour, width=5)
        draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=colour)
        # the number gets a dark backing dot so it survives bright floors
        draw.ellipse([u - 11, v - r - 22, u + 11, v - r], fill=(12, 12, 16))
        draw.text((u - 5, v - r - 20), str(n), fill=(255, 255, 255),
                  font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_depth(depth: np.ndarray, *, max_m: float | None = None,
                 out_px: int = 512, scale_m: float | None = None) -> bytes:
    """The depth frame as a picture, for a human looking over the organ's
    shoulder. Near = warm, far = cool, no-return = black, so the invalid
    pixels that must never be read as "wall at 0 m" are visibly distinct from
    genuinely close surfaces.

    ``max_m`` is the colour ramp's far end. It used to be hardwired to
    RANGE_CAP_M — the MAP's cap — so every corridor longer than that
    metres rendered as one flat blue slab and the picture said nothing about
    what was actually out there. It is not the map's business how far the
    sensor can see. None now means "read it off this frame", so the ramp
    always spans the depth the sensor really returned, and the range is burned
    into the corner because a colour with no scale beside it is decoration.

    The 99th percentile, not the max: one stray far pixel through a doorway
    would otherwise compress the whole room into the warm end."""
    if PILImage is None:
        return b""
    metres, _ = to_metres(depth, scale_m)
    invalid = metres <= 0.05
    if max_m is None:
        real = metres[~invalid]
        max_m = float(np.percentile(real, 99)) if real.size else RANGE_CAP_M
    max_m = max(float(max_m), 0.5)
    t = np.clip(metres / max_m, 0.0, 1.0)
    r = np.clip(1.6 - 1.8 * t, 0, 1)
    g = np.clip(1.0 - np.abs(t - 0.5) * 1.9, 0, 1)
    b = np.clip(1.8 * t - 0.5, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[invalid] = 0.0
    img = PILImage.fromarray((rgb * 255).astype(np.uint8))
    if out_px and max(img.size) != out_px:
        # NEAREST on purpose: the depth sensor really is 256², and smoothing
        # the upscale would paint detail the sensor never measured.
        img = img.resize((out_px, out_px), PILImage.NEAREST)
    if ImageDraw is not None:
        real = metres[~invalid]
        far = float(real.max()) if real.size else 0.0
        near = float(real.min()) if real.size else 0.0
        draw = ImageDraw.Draw(img)
        label = (f"colour ramp: red {near:.1f} m → blue {max_m:.1f} m   ·   "
                 f"frame spans {near:.1f}–{far:.1f} m   ·   black = no return")
        draw.rectangle([0, img.size[1] - 18, img.size[0], img.size[1]],
                       fill=(0, 0, 0))
        draw.text((6, img.size[1] - 15), label, fill=(235, 235, 235),
                  font=caption_font(11))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def topdown_caption(td: TopDown, waypoints: list[Waypoint],
                    extra: str = "") -> str:
    """The readout that gets burned into the map image for the monitor.

    Two lines, because one had grown past the image width and was being cut off
    mid-number — and every distance here is now labelled walk-or-see. Quoting
    the sightline as "ahead" put "ahead 2.8m" on a picture whose panel said
    "能走 1.0m" about the same frame."""
    prof = passable_range(td)
    walk = float(prof[len(td.bearings) // 2])
    best = float(prof.max())
    tied = np.nonzero(prof >= best * 0.95)[0]
    k = int(tied[int(np.argmin(np.abs(td.bearings[tied])))]) if tied.size else 0
    bearing, widest = float(td.bearings[k]), float(prof[k])
    grid = td.grid
    total = float(grid.size) or 1.0
    side = "left" if math.degrees(bearing) > 0 else "right"
    body = int((( (grid == FREE) & (td.clearance < ROBOT_RADIUS_M) ).sum()))
    lines = [
        f"[绿=可走 黄=要挤 暗黄=过不去({body}格) 蓝=没见过地面 红=障碍]",
        (f"walk {walk:.1f}m | see {td.ahead_m():.1f}m | widest "
         f"{abs(math.degrees(bearing)):.0f}°{side} @ {widest:.1f}m | "
         f"free {100 * float((grid == FREE).sum()) / total:.0f}%"
         f" open {100 * float((grid == OPEN).sum()) / total:.0f}%"
         f" occ {100 * float((grid == OCCUPIED).sum()) / total:.0f}%"
         f" unk {100 * float((grid == UNKNOWN).sum()) / total:.0f}%"),
    ]
    for i, w in enumerate(waypoints, 1):
        sq = w.extras.get("squeeze")
        lines.append(f"{i} [{w.kind}] {w.describe()}"
                     + (f" | tightest {sq}m" if sq is not None else "")
                     + f" | {int(w.distance / 0.25)} steps")
    if not waypoints:
        lines.append("no candidate: blocked or too tight to offer anything")
    if extra:
        lines.extend(extra.split("\n"))
    return "\n".join(lines)


def render_topdown(td: TopDown, waypoints: list[Waypoint], *, scale: int = 3,
                   caption: str = "", square: bool = True) -> bytes:
    """The organ's private map, drawn for a HUMAN watching the monitor.

    Never shown to the model — it is metric, and the model navigates by
    landmark and topology. Rendered so the robot sits at the bottom centre
    facing up, and the image is mirrored at the end so the robot's left is the
    viewer's left.

    CROPPED TO WHAT IT ACTUALLY KNOWS, and square by default. The grid is
    n_fwd × 2·n_fwd — 12 m forward, 12 m to each side — so drawing every cell
    produces a 2:1 letterbox of which most is empty: the field of view is 90°,
    so data can only live in the wedge |x| ≤ y and the far corners are UNKNOWN
    by construction, never by measurement. On the human tab that shape ate the
    width the ACCUMULATED map needed, and the accumulated map is the one worth
    studying. So: find the cells that carry anything, keep the robot at the
    bottom centre, and pad the short side. Nothing is scaled anisotropically and
    nothing measured is cropped away — nothing is thrown out but blank."""
    if PILImage is None:
        return b""
    n_fwd, n_lat = td.grid.shape
    centre = n_lat // 2
    rows, cols = np.nonzero(td.grid != UNKNOWN)
    if rows.size:
        # …plus a margin, and always symmetric about the body: an off-centre
        # crop would put the robot somewhere other than the bottom middle and
        # quietly rotate a person's sense of which way is forward.
        pad = 3
        h = min(n_fwd, int(rows.max()) + 1 + pad)
        reach = max(centre - int(cols.min()), int(cols.max()) - centre) + pad
    else:
        h, reach = n_fwd, centre
    if square:
        # A 90° field of view is a wedge with a 45° half-angle, so its lateral
        # extent EQUALS its forward reach and the bounding box of anything it
        # can ever see is 2:1. Squaring therefore means choosing: crop the far
        # side corners, or extend forward into cells the sensor did not reach.
        #
        # Extend. Cropping would throw away measurements — usually the walls
        # furthest out to either side, which is exactly the context that makes a
        # corridor legible — whereas the space added ahead is UNKNOWN, which is
        # both true and what the rest of the map already draws for unknown.
        side = max(h, 2 * reach)
        h, reach = side, (side + 1) // 2
    j0, j1 = centre - reach, centre + reach
    view_fwd, view_lat = h, j1 - j0
    grid = np.full((view_fwd, view_lat), UNKNOWN, dtype=np.uint8)
    clearance = np.zeros((view_fwd, view_lat), dtype=np.float32)
    pot = np.zeros((view_fwd, view_lat), dtype=bool)
    # The window may now hang off the grid on any side; copy the overlap only.
    ci0, ci1 = 0, min(view_fwd, n_fwd)
    cj0, cj1 = max(0, j0), min(n_lat, j1)
    if ci1 > ci0 and cj1 > cj0:
        grid[ci0:ci1, cj0 - j0:cj1 - j0] = td.grid[ci0:ci1, cj0:cj1]
        clearance[ci0:ci1, cj0 - j0:cj1 - j0] = td.clearance[ci0:ci1, cj0:cj1]
        if td.potential is not None:
            pot[ci0:ci1, cj0 - j0:cj1 - j0] = td.potential[ci0:ci1, cj0:cj1]

    img = PILImage.new("RGB", (view_lat * scale, view_fwd * scale), (18, 18, 22))
    px = img.load()
    max_clear = 1.5
    for i in range(view_fwd):
        row = view_fwd - 1 - i
        for j in range(view_lat):
            cell = grid[i, j]
            clear = float(clearance[i, j])
            if cell == OCCUPIED:
                colour = (176, 68, 62)
            elif cell == FREE:
                # THREE tones, not one. Every free cell used to be drawn the
                # same green, so a 0.2 m crack the body cannot enter looked
                # exactly like a corridor — the map showed "walkable for a
                # point" while a person reads "walkable for this robot", and the
                # two disagree precisely where it matters. The body is 0.40 m
                # across; now the picture says so.
                if clear < ROBOT_RADIUS_M:
                    colour = (58, 58, 44)            # floor, but the body will not fit
                elif clear < MIN_CLEARANCE_M:
                    colour = (120, 116, 48)          # fits, but you would be squeezing
                else:
                    t = min(1.0, clear / max_clear)
                    colour = (int(26 + 34 * t), int(74 + 92 * t), int(52 + 58 * t))
            elif cell == OPEN:
                # Deliberately BLUE, not a dim green. The whole point of the
                # split is that this is a different KIND of claim — the ray got
                # through, the ground was never seen — and a dimmer shade of the
                # walkable colour would read as "walkable, a bit less sure".
                # A person glancing at this map has to be able to see where the
                # proposer stops being willing to send the body.
                colour = (42, 62, 96)
            elif pot[i, j]:
                # §2.4 AMBER: floor glimpsed beyond an occluder — a different
                # kind of claim again: worth going to LOOK at, never walkable
                colour = (110, 84, 36)
            else:
                colour = (28, 30, 38)
            for a in range(scale):
                for b in range(scale):
                    px[j * scale + b, row * scale + a] = colour
    # Mirror the CELLS first, then draw. Drawing before the flip mirrors the
    # digits too, so "2" and "3" came out backwards on the map — the markers
    # were in the right places but unreadable.
    img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    draw = ImageDraw.Draw(img)
    W = view_lat * scale

    def _px(w: Waypoint) -> tuple[int, int]:
        # Cell coordinates of the CROP, not of the full grid — the two differ by
        # j0 now, and using the old expression put every marker off to one side
        # by exactly the amount that was trimmed.
        x = int(((w.x_left + td.range_cap_m) / CELL_M - j0) * scale)
        y = int((view_fwd - 1 - w.y_fwd / CELL_M) * scale)
        return W - 1 - x, y          # mirrored to match the flipped cells

    rx, ry = W // 2, view_fwd * scale - 3
    for w in waypoints:
        x, y = _px(w)
        draw.line([rx, ry, x, y], fill=(84, 150, 240), width=2)
    for n, w in enumerate(waypoints, 1):
        x, y = _px(w)
        colour = (255, 205, 70) if w.kind == "gateway" else (120, 210, 255)
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=colour, width=3)
        draw.text((x - 3, y - 6), str(n), fill=(255, 255, 255))
    draw.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=(250, 220, 90))

    # The readout is burned INTO the image rather than shipped alongside it:
    # the monitor can already serve any png from the live dir, so a
    # self-contained picture needs no new endpoint and no backend restart to
    # become visible. (Mirrored after the flip so the text reads normally.)
    if caption:
        pad, line_h = 6, 12
        font = caption_font(11)
        # WRAPPED, not truncated. The picture is square now and therefore much
        # narrower than the old 2:1 letterbox, and a fixed `line[:120]` cut the
        # readout mid-number — the free/open/occ percentages simply vanished off
        # the right edge. Measure against the actual width instead of guessing a
        # character count, since the text is mixed Chinese and Latin and the two
        # are nowhere near the same width.
        probe = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))
        avail = img.width - 2 * pad

        def _wrap(text: str) -> list[str]:
            rows, cur = [], ""
            for ch in text:
                trial = cur + ch
                if cur and probe.textlength(trial, font=font) > avail:
                    rows.append(cur)
                    cur = ch
                else:
                    cur = trial
            rows.append(cur)
            return rows

        lines: list[tuple[str, bool]] = []
        for k, para in enumerate(caption.split("\n")):
            lines.extend((row, k == 0) for row in _wrap(para))
        strip = PILImage.new("RGB", (img.width, line_h * len(lines) + 2 * pad),
                             (12, 12, 16))
        d2 = ImageDraw.Draw(strip)
        for k, (line, is_head) in enumerate(lines):
            colour = (150, 210, 255) if is_head else (185, 185, 195)
            d2.text((pad, pad + k * line_h), line, font=font, fill=colour)
        out = PILImage.new("RGB", (img.width, img.height + strip.height))
        out.paste(img, (0, 0))
        out.paste(strip, (0, img.height))
        img = out

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def surroundings_sentence(td: TopDown) -> str:
    """One natural-language line for the state block's 我周围是什么 — the shape
    of the space, never a coordinate.

    "How far can I walk" and "how far can I see" are answered by two different
    profiles and this sentence used to give the SIGHTLINE for both. That was
    survivable while the grid stopped at 6 m and the two nearly coincided; with
    the grid at 12 m it is a straight lie — the eye reaches 7.4 m down a bearing
    where the body is invited 2.5 m, and the model reads "you can walk about
    7.4 m ahead" while every candidate it is offered stops far short. Saying
    both, and saying which is which, is also just more useful: "it stays open
    further on, I have not seen that ground yet" is an invitation to go look,
    which is what an unexplored direction should sound like."""
    sight = td.ahead_m()
    # EVERY comparison below is on the passable profile, not the sightline. A
    # sightline maximum is routinely a 0.2 m crack between two chairs — the very
    # failure passable_range was written for — so "the roomiest direction" read
    # off free_range points the body at a slot it cannot enter. Only `sight`
    # stays a sightline, and it is labelled as one in the text.
    prof = passable_range(td)
    ahead = float(prof[len(td.bearings) // 2])
    best = float(prof.max())
    near_best = np.nonzero(prof >= best * 0.95)[0]
    kb = int(near_best[np.argmin(np.abs(td.bearings[near_best]))])
    bearing, widest = float(td.bearings[kb]), float(prof[kb])
    # bearings run -45° (RIGHT) → +45° (LEFT), so the FIRST third of the array
    # is the robot's right. Getting this backwards told the model the room was
    # open to its left while every candidate pointed right.
    third = max(1, len(prof) // 3)
    right = float(np.percentile(prof[:third], 80))
    left = float(np.percentile(prof[-third:], 80))
    if ahead >= td.range_cap_m - 0.05:
        head = "The way ahead is open as far as you can see"
    elif ahead < 1.2:
        head = f"Something is blocking you about {ahead:.1f} m ahead"
    else:
        head = f"You can walk about {ahead:.1f} m straight ahead before something stops you"
    if sight > ahead + 1.5 and ahead < td.range_cap_m - 0.05:
        head += (f", and it stays open further on, out to about {sight:.1f} m — "
                 "you have not had a proper look at the ground that far yet")
    side = ""
    if right - left > 1.5:
        side = "; it is much more open to your right"
    elif left - right > 1.5:
        side = "; it is much more open to your left"
    elif widest > ahead + 1.5 and abs(math.degrees(bearing)) >= 8:
        # …and only when it is actually to a SIDE. The tie-break pulls the
        # roomiest bearing toward straight ahead, so this clause kept emitting
        # "the roomiest direction is 0° to your right" — which names a side that
        # does not exist and, at 0°, contradicts the clause before it.
        deg = math.degrees(bearing)
        side = (f"; the roomiest direction is {abs(deg):.0f}° to your "
                f"{'left' if deg > 0 else 'right'}")
    return head + side + "."
