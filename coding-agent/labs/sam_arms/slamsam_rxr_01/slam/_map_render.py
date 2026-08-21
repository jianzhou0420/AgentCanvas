"""Annotated agent-facing map render for env_slam_vlnce.

Replaces the probe's raw full-grid render (which drew the fixed 48 m window
at 1 px/cell — the explored area occupied ~1/5 of a mostly-gray frame, and
the 2026-08-17 instrumented smoke shows the model discarding it: "the map's
too zoomed out to pinpoint the bar clearly" / "I'll navigate visually
instead"). This renderer:

- CROPS to the explored bounding box (+``margin_m``, floor ``min_window_m``,
  square) and upscales NEAREST to ``out_px`` — content fills the frame at
  every stage of the episode;
- DRAWS the things the JSON alone could not make legible: the agent as a
  heading arrow, the trajectory polyline, an "S" start marker, frontier
  clusters as numbered circles (circle N = frontier "FN" in the JSON — the
  same numbered-circle idiom as the wp surface), faint 2 m gridlines with
  world-coordinate labels (so get_pose's x/z numbers land on pixels), and a
  scale bar.

Frame facts baked into the drawing (empirically verified against habitat):
the world is anchored at the episode's start pose, up on the image = +z =
the agent's STARTING heading, right = +x; yaw_deg increases turning right.

Palette: unknown gray / free white / obstacle black (probe-compatible),
frontier cells tinted light red under the numbered circles.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ._mapping import FREE, OBSTACLE, UNKNOWN

COL_UNKNOWN = (128, 128, 128)
COL_FREE = (255, 255, 255)
COL_OBSTACLE = (0, 0, 0)
COL_FRONTIER_CELL = (247, 213, 213)
COL_GRIDLINE = (108, 108, 108)
COL_LABEL = (60, 60, 60)
COL_TRAJ = (70, 130, 255)
COL_AGENT = (30, 60, 230)
COL_START = (150, 60, 200)
COL_FRONTIER = (30, 150, 50)


def _text_centered(draw: ImageDraw.ImageDraw, xy: tuple, text: str, fill: tuple) -> None:
    try:
        bbox = draw.textbbox((0, 0), text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # noqa: BLE001 — very old Pillow: rough fallback
        w, h = 6 * len(text), 11
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, fill=fill)


def render_annotated_map(
    grid: np.ndarray,
    fmask: np.ndarray,
    cell_size: float,
    origin: int,
    *,
    agent_xz: tuple | None,
    agent_yaw: float | None,
    trajectory: list,
    frontier_cells: list,
    out_px: int = 512,
    margin_m: float = 2.0,
    min_window_m: float = 8.0,
    grid_every_m: float = 2.0,
    semantic_layers: list | None = None,
    region_layers: list | None = None,
) -> tuple:
    """Render the cropped, annotated top-down map.

    ``frontier_cells``: [(number, (zi, xi))] — number N pairs with JSON id
    "FN". ``trajectory``: [[x, y, z], ...] world points (est_track).
    Returns (HxWx3 uint8 array, window_m float).
    """
    n = grid.shape[0]

    # ── crop window (cell space), square ──
    known = np.nonzero(grid != UNKNOWN)
    zi_pts = list(known[0])
    xi_pts = list(known[1])
    if agent_xz is not None:
        zi_pts.append(int(math.floor(agent_xz[1] / cell_size)) + origin)
        xi_pts.append(int(math.floor(agent_xz[0] / cell_size)) + origin)
    if not zi_pts:  # nothing integrated yet: center the floor window on start
        zi_pts = xi_pts = [origin]
    zi0, zi1 = min(zi_pts), max(zi_pts) + 1
    xi0, xi1 = min(xi_pts), max(xi_pts) + 1
    margin = int(round(margin_m / cell_size))
    zi0, zi1 = zi0 - margin, zi1 + margin
    xi0, xi1 = xi0 - margin, xi1 + margin
    side = max(zi1 - zi0, xi1 - xi0, int(round(min_window_m / cell_size)))
    # expand each axis symmetrically to the square side, then clamp
    zc, xc = (zi0 + zi1) // 2, (xi0 + xi1) // 2
    zi0, xi0 = zc - side // 2, xc - side // 2
    zi0 = max(0, min(zi0, n - side))
    xi0 = max(0, min(xi0, n - side))
    side = min(side, n)
    zi1, xi1 = zi0 + side, xi0 + side
    s = out_px / side  # px per cell

    def to_px(x: float, z: float) -> tuple:
        """World metres -> output pixel (col, row); up = +z."""
        cx = x / cell_size + origin
        cz = z / cell_size + origin
        return ((cx - xi0) * s, (zi1 - cz) * s)

    # ── base colors, cropped, flipped so +z is up ──
    crop = grid[zi0:zi1, xi0:xi1]
    fcrop = fmask[zi0:zi1, xi0:xi1]
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:] = COL_UNKNOWN
    img[crop == FREE] = COL_FREE
    img[crop == OBSTACLE] = COL_OBSTACLE
    img[fcrop] = COL_FRONTIER_CELL
    # ── §35 room-region pale fill (drawn FIRST, under everything; None =
    # byte-identical) — free cells of a named room take a light wash ──
    _sem_label_pts = []
    for room, rhit, rtint in (region_layers or []):
        rc = rhit[zi0:zi1, xi0:xi1] & (crop == FREE)
        if not rc.any():
            continue
        img[rc] = ((img[rc].astype(np.uint16) * 3
                    + np.asarray(rtint, np.uint16)) // 4).astype(np.uint8)
        zi_r, xi_r = np.nonzero(rc)
        _sem_label_pts.append((room, float(xi_r.mean()), float(zi_r.mean()),
                               (90, 90, 110)))

    # ── SAM semantic tint (phase-2 addition; None = byte-identical port) ──
    # semantic_layers: [(phrase, bool grid full-size, (r,g,b))] — cells tint
    # 50/50 over the base palette so occupancy stays readable underneath.
    for phrase, hit, tint in (semantic_layers or []):
        hc = hit[zi0:zi1, xi0:xi1]
        if not hc.any():
            continue
        img[hc] = (img[hc] // 2 + np.asarray(tint, np.uint8) // 2)
        zi_h, xi_h = np.nonzero(hc)
        _sem_label_pts.append((phrase, float(xi_h.mean()), float(zi_h.mean()),
                               tint))
    img = img[::-1]
    pil = Image.fromarray(img).resize((out_px, out_px), Image.NEAREST)
    draw = ImageDraw.Draw(pil)

    # ── gridlines every grid_every_m, labeled in world metres ──
    x_min, x_max = (xi0 - origin) * cell_size, (xi1 - origin) * cell_size
    z_min, z_max = (zi0 - origin) * cell_size, (zi1 - origin) * cell_size
    gx = math.ceil(x_min / grid_every_m) * grid_every_m
    while gx < x_max:
        col, _ = to_px(gx, 0.0)
        draw.line([(col, 0), (col, out_px)], fill=COL_GRIDLINE, width=1)
        draw.text((col + 3, out_px - 13), f"x={gx:g}", fill=COL_LABEL)
        gx += grid_every_m
    gz = math.ceil(z_min / grid_every_m) * grid_every_m
    while gz < z_max:
        _, row = to_px(0.0, gz)
        draw.line([(0, row), (out_px, row)], fill=COL_GRIDLINE, width=1)
        draw.text((3, row + 2), f"z={gz:g}", fill=COL_LABEL)
        gz += grid_every_m

    # ── trajectory polyline + start marker ──
    if trajectory:
        pts = [to_px(p[0], p[2]) for p in trajectory]
        if len(pts) >= 2:
            draw.line(pts, fill=COL_TRAJ, width=2)
        sx, sy = pts[0]
        draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], outline=COL_START, width=2)
        _text_centered(draw, (sx, sy), "S", COL_START)

    # ── frontier circles: circle N = JSON id "FN" ──
    for num, (fzi, fxi) in frontier_cells:
        fx = (fxi - origin + 0.5) * cell_size
        fz = (fzi - origin + 0.5) * cell_size
        cx, cy = to_px(fx, fz)
        r = 10
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COL_FRONTIER,
                     outline=(255, 255, 255), width=1)
        _text_centered(draw, (cx, cy), str(num), (255, 255, 255))

    # ── agent heading arrow ──
    if agent_xz is not None and agent_yaw is not None:
        ax, ay = to_px(agent_xz[0], agent_xz[1])
        dx, dy = math.sin(agent_yaw), -math.cos(agent_yaw)
        px, py = -dy, dx  # perpendicular
        tip = (ax + dx * 14, ay + dy * 14)
        left = (ax - dx * 6 + px * 7, ay - dy * 6 + py * 7)
        right = (ax - dx * 6 - px * 7, ay - dy * 6 - py * 7)
        draw.polygon([tip, left, right], fill=COL_AGENT, outline=(255, 255, 255))

    # ── SAM landmark names at their patch centroids (dark backing so they
    # survive the white floor; greedy downward nudge on overlap) ──
    _used_boxes: list = []
    for phrase, xh, zh, tint in _sem_label_pts:
        lx = (xh + 0.5) * s + 4
        ly = out_px - (zh + 0.5) * s - 5
        try:
            bb = draw.textbbox((0, 0), phrase)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:  # noqa: BLE001
            tw, th = 6 * len(phrase), 11
        lx = min(max(1, lx), out_px - tw - 3)
        ly = min(max(1, ly), out_px - th - 2)
        for _ in range(32):
            box = (lx - 2, ly - 1, lx + tw + 2, ly + th + 1)
            if not any(box[0] < b[2] and box[2] > b[0]
                       and box[1] < b[3] and box[3] > b[1]
                       for b in _used_boxes):
                break
            ly += th + 3
            if ly + th + 2 >= out_px:
                ly = 1
        _used_boxes.append((lx - 2, ly - 1, lx + tw + 2, ly + th + 1))
        draw.rectangle(_used_boxes[-1], fill=(28, 28, 32))
        draw.text((lx, ly), phrase, fill=tint)

    # ── scale bar (bottom-left) ──
    bar_px = (grid_every_m / cell_size) * s
    y0 = out_px - 24
    draw.line([(12, y0), (12 + bar_px, y0)], fill=COL_LABEL, width=3)
    draw.line([(12, y0 - 5), (12, y0 + 5)], fill=COL_LABEL, width=1)
    draw.line([(12 + bar_px, y0 - 5), (12 + bar_px, y0 + 5)], fill=COL_LABEL, width=1)
    draw.text((12 + bar_px / 2 - 10, y0 - 17), f"{grid_every_m:g} m", fill=COL_LABEL)

    return np.asarray(pil), side * cell_size


def render_annotated_map_v2(
    grid: np.ndarray,
    cell_size: float,
    origin: int,
    *,
    agent_xz: tuple | None,
    agent_yaw: float | None,
    trajectory: list,
    window_cells: tuple | None = None,
    floor_label: str | None = None,
    mark_start: bool = True,
    out_px: int = 512,
    margin_m: float = 2.0,
    min_window_m: float = 8.0,
    grid_every_m: float = 2.0,
    semantic_layers: list | None = None,
) -> tuple:
    """Map v2 render (SLAM-02): no frontier layer, stable axes.

    ``semantic_layers`` (OUR addition over jian's v2, 2026-08-18 — None =
    byte-identical to his): [(phrase, bool grid full-size, (r,g,b))] SAM
    landmark cells tinted 50/50 over the base palette, names at centroids
    with a dark backing and greedy downward nudge — the same overlay v1
    carries, so the SAM arm and the lean side-car render on v2 too.

    ``mark_start``: draw the "S" ring at trajectory[0]. Pass False on floors
    that do not contain the episode start — there trajectory[0] is only the
    point where the agent entered the floor.

    Differences from v1: no frontier circles/tint; the crop box is snapped
    OUTWARD to whole ``grid_every_m`` world multiples and unioned with
    ``window_cells`` (the previous call's box), so within an episode the
    window only grows in 2 m quanta and never pans — axis labels stay put
    between calls. The 0-axes are drawn heavier, every gridline is labeled
    on both axes, and an optional ``floor_label`` tag ("floor 2/2") is
    stamped top-right.

    Returns (HxWx3 uint8 array, window_m float, window_cells tuple) — pass
    the returned ``window_cells`` back in on the next call.
    """
    n = grid.shape[0]
    snap = max(1, int(round(grid_every_m / cell_size)))

    # ── explored bbox + margin, as in v1 ──
    known = np.nonzero(grid != UNKNOWN)
    zi_pts = list(known[0])
    xi_pts = list(known[1])
    if agent_xz is not None:
        zi_pts.append(int(math.floor(agent_xz[1] / cell_size)) + origin)
        xi_pts.append(int(math.floor(agent_xz[0] / cell_size)) + origin)
    if not zi_pts:
        zi_pts = xi_pts = [origin]
    margin = int(round(margin_m / cell_size))
    zi0, zi1 = min(zi_pts) - margin, max(zi_pts) + 1 + margin
    xi0, xi1 = min(xi_pts) - margin, max(xi_pts) + 1 + margin

    # ── snap bounds outward to whole grid_every_m world multiples ──
    def snap_lo(i: int) -> int:
        return origin + ((i - origin) // snap) * snap

    def snap_hi(i: int) -> int:
        return origin + -((origin - i) // snap) * snap  # ceil division

    zi0, zi1 = snap_lo(zi0), snap_hi(zi1)
    xi0, xi1 = snap_lo(xi0), snap_hi(xi1)

    # ── grow-only: union with the previous window ──
    if window_cells is not None:
        pz0, pz1, px0, px1 = window_cells
        zi0, zi1 = min(zi0, pz0), max(zi1, pz1)
        xi0, xi1 = min(xi0, px0), max(xi1, px1)

    # ── square + floor, expanding at the high end in snap quanta ──
    side = max(zi1 - zi0, xi1 - xi0, int(round(min_window_m / cell_size)))
    side = -(-side // snap) * snap
    zi1, xi1 = zi0 + side, xi0 + side
    # clamp to the grid, shifting the low bound in snap quanta if needed
    if zi1 > n:
        shift = -(-(zi1 - n) // snap) * snap
        zi0, zi1 = zi0 - shift, zi1 - shift
    if xi1 > n:
        shift = -(-(xi1 - n) // snap) * snap
        xi0, xi1 = xi0 - shift, xi1 - shift
    zi0, xi0 = max(0, zi0), max(0, xi0)
    side = min(zi1 - zi0, xi1 - xi0)
    zi1, xi1 = zi0 + side, xi0 + side
    s = out_px / side  # px per cell

    def to_px(x: float, z: float) -> tuple:
        """World metres -> output pixel (col, row); up = +z."""
        cx = x / cell_size + origin
        cz = z / cell_size + origin
        return ((cx - xi0) * s, (zi1 - cz) * s)

    # ── base colors, cropped, flipped so +z is up ──
    crop = grid[zi0:zi1, xi0:xi1]
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:] = COL_UNKNOWN
    img[crop == FREE] = COL_FREE
    img[crop == OBSTACLE] = COL_OBSTACLE
    _sem_label_pts = []
    for phrase, hit, tint in (semantic_layers or []):
        hc = hit[zi0:zi1, xi0:xi1]
        if not hc.any():
            continue
        img[hc] = (img[hc] // 2 + np.asarray(tint, np.uint8) // 2)
        zi_h, xi_h = np.nonzero(hc)
        _sem_label_pts.append((phrase, float(xi_h.mean()), float(zi_h.mean()),
                               tint))
    img = img[::-1]
    pil = Image.fromarray(img).resize((out_px, out_px), Image.NEAREST)
    draw = ImageDraw.Draw(pil)

    # ── gridlines every grid_every_m; 0-axes heavier; both axes labeled ──
    x_min, x_max = (xi0 - origin) * cell_size, (xi1 - origin) * cell_size
    z_min, z_max = (zi0 - origin) * cell_size, (zi1 - origin) * cell_size
    gx = math.ceil(x_min / grid_every_m) * grid_every_m
    while gx < x_max:
        col, _ = to_px(gx, 0.0)
        draw.line([(col, 0), (col, out_px)], fill=COL_GRIDLINE,
                  width=2 if gx == 0 else 1)
        draw.text((col + 3, out_px - 13), f"x={gx:g}", fill=COL_LABEL)
        gx += grid_every_m
    gz = math.ceil(z_min / grid_every_m) * grid_every_m
    while gz < z_max:
        _, row = to_px(0.0, gz)
        draw.line([(0, row), (out_px, row)], fill=COL_GRIDLINE,
                  width=2 if gz == 0 else 1)
        draw.text((3, row + 2), f"z={gz:g}", fill=COL_LABEL)
        gz += grid_every_m

    # ── trajectory polyline + start marker ──
    if trajectory:
        pts = [to_px(p[0], p[2]) for p in trajectory]
        if len(pts) >= 2:
            draw.line(pts, fill=COL_TRAJ, width=2)
        if mark_start:
            sx, sy = pts[0]
            draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], outline=COL_START,
                         width=2)
            _text_centered(draw, (sx, sy), "S", COL_START)

    # ── agent heading arrow ──
    if agent_xz is not None and agent_yaw is not None:
        ax, ay = to_px(agent_xz[0], agent_xz[1])
        dx, dy = math.sin(agent_yaw), -math.cos(agent_yaw)
        px, py = -dy, dx  # perpendicular
        tip = (ax + dx * 14, ay + dy * 14)
        left = (ax - dx * 6 + px * 7, ay - dy * 6 + py * 7)
        right = (ax - dx * 6 - px * 7, ay - dy * 6 - py * 7)
        draw.polygon([tip, left, right], fill=COL_AGENT, outline=(255, 255, 255))

    # ── SAM landmark names at their patch centroids (dark backing so they
    # survive the white floor; greedy downward nudge on overlap) ──
    _used_boxes: list = []
    for phrase, xh, zh, tint in _sem_label_pts:
        lx = (xh + 0.5) * s + 4
        ly = out_px - (zh + 0.5) * s - 5
        try:
            bb = draw.textbbox((0, 0), phrase)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:  # noqa: BLE001
            tw, th = 6 * len(phrase), 11
        lx = min(max(1, lx), out_px - tw - 3)
        ly = min(max(1, ly), out_px - th - 2)
        for _ in range(32):
            box = (lx - 2, ly - 1, lx + tw + 2, ly + th + 1)
            if not any(box[0] < b[2] and box[2] > b[0]
                       and box[1] < b[3] and box[3] > b[1]
                       for b in _used_boxes):
                break
            ly += th + 3
            if ly + th + 2 >= out_px:
                ly = 1
        _used_boxes.append((lx - 2, ly - 1, lx + tw + 2, ly + th + 1))
        draw.rectangle(_used_boxes[-1], fill=(28, 28, 32))
        draw.text((lx, ly), phrase, fill=tint)

    # ── floor tag (top-right) + scale bar (bottom-left) ──
    if floor_label:
        draw.text((out_px - 8 * len(floor_label) - 10, 8), floor_label,
                  fill=COL_LABEL)
    bar_px = (grid_every_m / cell_size) * s
    y0 = out_px - 24
    draw.line([(12, y0), (12 + bar_px, y0)], fill=COL_LABEL, width=3)
    draw.line([(12, y0 - 5), (12, y0 + 5)], fill=COL_LABEL, width=1)
    draw.line([(12 + bar_px, y0 - 5), (12 + bar_px, y0 + 5)], fill=COL_LABEL, width=1)
    draw.text((12 + bar_px / 2 - 10, y0 - 17), f"{grid_every_m:g} m", fill=COL_LABEL)

    return np.asarray(pil), side * cell_size, (zi0, zi1, xi0, xi1)
