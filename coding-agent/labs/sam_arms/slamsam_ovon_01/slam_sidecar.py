"""[exp_workspace/slamsam_ovon_01 frozen copy of harnesses/mini/slam_sidecar.py,
2026-08-18 — see README.md; do not edit after boards run, fork the folder.]

The SLAM map sidecar for the slamdwp surface (user 2026-08-17 深夜:
"SLAM 调图的整个流程是不变的 … SAM 又是 condition 在这个 SLAM 给的图上").

The FULL ported slamr2r mapping pipeline rides NEXT TO the untouched
AnchorMap: OccupancyMap.integrate fed per sensed primitive from
(depth, our odometry pose, camera intrinsics), the package's own annotated
renderer for get_map, and its semantic score-grid layer stamped from the
dwp line's ONE async SAM stream (capture-pose frozen upstream in
_absorb_sightings — no second detector, no double SAM bill).

AnchorMap keeps every DWP behavior byte-identical (candidates, goto,
remembered, facts); THIS object only answers get_map. The env_slam_vlnce
package cannot be imported wholesale here (its __init__ pulls habitat
0.3.3), so the three pure modules are loaded under a synthetic package
name with relative imports intact.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# exp_workspace copy (2026-08-18): the SLAM modules are THIS folder's own
# slam/ copies (jian's _mapping/_frontier/_map_render, map v2 merged, + our
# _semantic), never the shared workspace nodeset — one folder = one frozen
# experiment. Loaded under a folder-unique synthetic package name so two
# folders imported into one process never share module state.
_PKG_DIR = Path(__file__).resolve().parent / "slam"
_PKG_NAME = f"slamvl_{Path(__file__).resolve().parent.name}"


def _load_mods() -> dict:
    if f"{_PKG_NAME}._map_render" in sys.modules:
        return sys.modules
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(_PKG_NAME, None, is_package=True))
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG_NAME] = pkg
    for name in ("_mapping", "_frontier", "_map_render", "_semantic"):
        spec = importlib.util.spec_from_file_location(
            f"{_PKG_NAME}.{name}", _PKG_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PKG_NAME}.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules


class SlamMapSidecar:
    """occ-grid + semantic layer + renderer, in the anchor frame's world
    coordinates (x = right-of-start = our px, z = forward-at-start = our
    py; their cv convention adds y DOWN and yaw = −our θ)."""

    def __init__(self, *, sam_phrases: list[str], cam_height_m: float,
                 rgb_hw: tuple[int, int] = (512, 512),
                 hfov_deg: float = 90.0, map_mode: str = "v1") -> None:
        mods = _load_mods()
        self._mapping = mods[f"{_PKG_NAME}._mapping"]
        self._frontier = mods[f"{_PKG_NAME}._frontier"]
        self._render = mods[f"{_PKG_NAME}._map_render"]
        sem = mods[f"{_PKG_NAME}._semantic"]
        # map v2 (jian slam_r2r_02, merged 2026-08-18): per-floor layers +
        # the frontier-free, fixed-axes renderer. Our odometry carries no
        # height, so the layer registry stays on floor 0 — what v2 buys
        # here is the grow-only 2 m-snapped window (axis labels never pan
        # between get_map calls) and no frontier circles/list.
        self.map_mode = str(map_mode or "v1")
        if self.map_mode == "v2":
            self.occ = self._mapping.LayeredOccupancyMap(
                camera_height=cam_height_m)
        else:
            self.occ = self._mapping.OccupancyMap(camera_height=cam_height_m)
        self.map_window: tuple | None = None   # v2 grow-only crop state
        # consumption views only (grids / overlay_cells / landmarks_json);
        # its own worker is never used — stamps come from the dwp SAM stream
        self.sem = sem.SemanticSamLayer(
            "unused", list(sam_phrases or []), self.occ.n, self.occ.origin,
            self.occ.cell_size, cam_height_m) if sam_phrases else None
        h, w = rgb_hw
        f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.intrinsics = {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0,
                          "width": w, "height": h}
        self.cam_h = float(cam_height_m)
        self.track: list[list[float]] = []
        self.tick = 0
        # jian's stable frontier ids: the same opening keeps its number as
        # the map grows; an explored-away frontier's number retires
        self._freg: dict[int, tuple[float, float]] = {}
        self._fnext = 1

    # ── frames ───────────────────────────────────────────────────────────
    def pose_wc(self, px: float, py: float, theta: float,
                pz: float = 0.0) -> np.ndarray:
        """anchor pose → world-from-camera (cv: x right, y DOWN, z fwd).
        ``pz`` = metres the body has climbed above the start floor (from the
        env's actual_dy_m); in the y-down world that is T[1,3] = −pz — the
        value LayeredOccupancyMap keys its floor registry on."""
        psi = -theta                       # their yaw is right-positive
        c, s = math.cos(psi), math.sin(psi)
        T = np.eye(4)
        T[0, 0], T[0, 2] = c, s
        T[2, 0], T[2, 2] = -s, c
        T[0, 3], T[2, 3] = px, py
        T[1, 3] = -float(pz)
        return T

    # ── the SLAM pipeline verbatim: integrate every sensed frame ─────────
    def integrate(self, depth_m: Any, px: float, py: float,
                  theta: float, pz: float = 0.0) -> None:
        if depth_m is None:
            return
        try:
            self.occ.integrate(self.pose_wc(px, py, theta, pz),
                               np.asarray(depth_m, np.float64),
                               self.intrinsics)
            # track y = pose y (y-down world, −pz): jian's est_track shape,
            # so the v2 per-floor trajectory filter works unchanged
            self.track.append([float(px), -float(pz), float(py)])
            self.tick += 1
        except Exception:  # noqa: BLE001 — the sidecar must never brake a walk
            pass

    # ── semantic stamps from the dwp SAM stream (capture pose upstream) ──
    def stamp_points(self, phrase: str, xs_body: Any, ys_body: Any,
                     px: float, py: float, theta: float,
                     score: float) -> None:
        """Score-grid accumulation on the SLAM map's cells, fed with the
        dwp line's already-projected body-frame mask points (project_mask
        output, frozen at the CAPTURE pose by _absorb_sightings). Same
        capture-pose contract as their _stamp; per-point vote ≈ the
        sighting score, so two solid sightings cross CELL_SHOW_MIN."""
        if self.sem is None:
            return
        grid = self.sem.grids.get(phrase)
        if grid is None:
            grid = self.sem.grids[phrase] = np.zeros(
                (self.occ.n, self.occ.n), np.float32)
        try:
            xl = np.asarray(xs_body, np.float64)
            yf = np.asarray(ys_body, np.float64)
            if xl.size == 0:
                return
            # body (x_left, y_fwd) at capture (px, py, θ) → world (x, z):
            # x_right = −x_left; rotate by ψ=−θ (their yaw), translate.
            psi = -float(theta)
            c, s = math.cos(psi), math.sin(psi)
            xr = -xl
            xw = px + xr * c + yf * s
            zw = py + (-xr * s + yf * c)
            xs = (np.floor(xw / self.occ.cell_size).astype(int)
                  + self.occ.origin)
            zs = (np.floor(zw / self.occ.cell_size).astype(int)
                  + self.occ.origin)
            okb = ((xs >= 0) & (xs < self.occ.n)
                   & (zs >= 0) & (zs < self.occ.n))
            # user 2026-08-17 深夜 (the smeared EP0 map): project_mask hands
            # over FULL-resolution points (tens of thousands), and +score
            # per point saturated a cell to the 6.0 cap in ONE frame — a
            # single pool sighting painted half the map. Match the native
            # slamr2r vote口径 instead: one frame contributes at most ONE
            # score-weighted vote per cell (dedup the cell hits), so a
            # patch needs ~2 solid sightings to render (CELL_SHOW_MIN 0.9)
            # and stale spill never outvotes structure.
            if okb.any():
                cells = np.unique(zs[okb] * self.occ.n + xs[okb])
                grid.flat[cells] += float(score)
            np.clip(grid, 0.0, 6.0, out=grid)
        except Exception:  # noqa: BLE001 — advisory layer, never fatal
            pass

    # ── counter-evidence: a label you stood next to and did not see ──────
    DECAY_RADIUS_M = 1.2      # "standing right next to it"
    DECAY_VOTE = 1.0          # ≈ one sighting's worth; two misses clear a
                              # patch that two sightings built

    def decay_unseen_nearby(self, phrases: list, px: float, py: float
                            ) -> None:
        """user 2026-08-18: revocable landmark memory. For each phrase the
        detector did NOT report from this pose, take one vote off every
        stamped cell within DECAY_RADIUS_M of the body. Same score-grid,
        opposite sign — the patch that a wrong SAM call painted fades the
        moment the body walks up and looks. Never touches cells farther
        away (you can't disprove what you can't see)."""
        if self.sem is None:
            return
        try:
            n, cs, o = self.occ.n, self.occ.cell_size, self.occ.origin
            r_cells = max(1, int(round(self.DECAY_RADIUS_M / cs)))
            ci = int(math.floor(px / cs)) + o
            cj = int(math.floor(py / cs)) + o   # (x → column, z → row)
            zi0, zi1 = max(0, cj - r_cells), min(n, cj + r_cells + 1)
            xi0, xi1 = max(0, ci - r_cells), min(n, ci + r_cells + 1)
            if zi1 <= zi0 or xi1 <= xi0:
                return
            zz, xx = np.mgrid[zi0:zi1, xi0:xi1]
            disk = ((zz - cj) ** 2 + (xx - ci) ** 2) <= r_cells ** 2
            for p in phrases:
                g = self.sem.grids.get(p)
                if g is None:
                    continue
                sub = g[zi0:zi1, xi0:xi1]
                hit = disk & (sub > 0)
                if hit.any():
                    sub[hit] = np.maximum(0.0, sub[hit] - self.DECAY_VOTE)
        except Exception:  # noqa: BLE001 — advisory layer, never fatal
            pass

    def _floor_track(self) -> tuple[list, bool]:
        """jian's v2 rule: draw only the trajectory points on the CURRENT
        floor; the S ring only when that floor holds the episode start."""
        layers = getattr(self.occ, "layers", None)
        if not layers or not self.track:
            return list(self.track), True
        floor_y = float(self.occ.current_floor_y)
        merge = float(getattr(self.occ, "floor_merge_m", 1.0))
        traj = [p for p in self.track if abs(p[1] - floor_y) <= merge]
        start_here = bool(traj) and traj[0] is self.track[0]
        return traj, start_here

    def _floor_label(self) -> str | None:
        """jian's v2 'floor k/n' tag (LayeredOccupancyMap only)."""
        layers = getattr(self.occ, "layers", None)
        if not layers:
            return None
        return f"floor {int(self.occ.current_floor) + 1}/{len(layers)}"

    # ── frontiers: jian's stable-id contract on the sidecar grid ─────────
    def frontier_marks(self, agent_xz: tuple, yaw: float) -> tuple:
        """([(n, (zi, xi))], [row…]) — circle N pairs with JSON id "FN";
        ids stable across calls (centroid match ≤ 1.2 m), retired ids
        never reused. dir_deg relative to heading, positive = right."""
        try:
            mask = self._frontier.frontier_mask(self.occ.grid)
            comps = self._frontier.cluster_frontiers(mask)
        except Exception:  # noqa: BLE001
            return [], []
        cs = self.occ.cell_size
        # user 2026-08-17 深夜: the EP0 map came out as a 45-circle号码海 —
        # our depth-cone maps fray at the edges into many small frontier
        # islands. Keep the LARGEST openings only (jian's contract had no
        # cap because his SLAM grids don't fray); 8 matches the primmap
        # AnchorMap path.
        comps = sorted(comps, key=len, reverse=True)[:8]
        found: list[tuple[tuple, float, float, float]] = []
        for comp in comps:
            zi, xi = self._frontier.cluster_representative(comp)
            x, z = self.occ.cell_to_world(int(zi), int(xi))
            dx, dz = x - agent_xz[0], z - agent_xz[1]
            dist = math.hypot(dx, dz)
            rel = math.degrees(
                (math.atan2(dx, dz) - yaw + math.pi) % (2 * math.pi)
                - math.pi)
            found.append(((int(zi), int(xi)), x, z,
                          (dist, rel, len(comp) * cs * cs)))
        assigned: dict[int, tuple] = {}
        used: set[int] = set()
        for cell, x, z, stats in found:
            best, best_d = None, 1.2
            for fid, (fx, fz) in self._freg.items():
                if fid in used:
                    continue
                d = math.hypot(fx - x, fz - z)
                if d < best_d:
                    best, best_d = fid, d
            if best is None:
                best = self._fnext
                self._fnext += 1
            used.add(best)
            self._freg[best] = (x, z)
            assigned[best] = (cell, stats)
        for fid in [f for f in self._freg if f not in used]:
            del self._freg[fid]          # explored away: id retired
        marks, rows = [], []
        for fid in sorted(assigned):
            cell, (dist, rel, size) = assigned[fid]
            marks.append((fid, cell))
            rows.append({"id": f"F{fid}", "dir_deg": round(rel, 1),
                         "dist_m": round(dist, 2),
                         "size_m2": round(size, 2)})
        return marks, rows

    # ── get_map: THE slam render, waypoints riding the numbered channel ──
    # ── monitor artifact: snapshot on the caller's thread, draw elsewhere ──
    def snapshot_for_render(self, amap: Any) -> dict:
        """Copy everything the renderer reads (grid, frontier mask, semantic
        overlay, trajectory, pose) so a background thread can draw while
        the walk keeps integrating. Frontier circles are the current
        stable ids — same call the model's get_map makes."""
        yaw = -float(amap.theta)
        agent_xz = (float(amap.px), float(amap.py))
        sem_layers = None
        if self.sem is not None:
            sem_layers = [(p, np.array(h, copy=True), tint)
                          for p, h, tint in self.sem.overlay_cells()]
        if self.map_mode == "v2":
            traj, start_here = self._floor_track()
            return {"v2": True, "grid": np.array(self.occ.grid, copy=True),
                    "cell_size": self.occ.cell_size, "origin": self.occ.origin,
                    "agent_xz": agent_xz, "yaw": yaw,
                    "trajectory": [list(p) for p in traj],
                    "window": self.map_window, "sem_layers": sem_layers,
                    "floor_label": self._floor_label(),
                    "mark_start": start_here,
                    "tick": int(self.tick)}
        marks, _ = self.frontier_marks(agent_xz, yaw)
        return {"grid": np.array(self.occ.grid, copy=True),
                "fmask": np.array(self._frontier.frontier_mask(self.occ.grid),
                                  copy=True),
                "cell_size": self.occ.cell_size, "origin": self.occ.origin,
                "agent_xz": agent_xz, "yaw": yaw,
                "trajectory": [list(p) for p in self.track],
                "marks": list(marks), "sem_layers": sem_layers,
                "tick": int(self.tick)}

    def render_snapshot(self, snap: dict) -> bytes:
        """Draw a snapshot_for_render() copy — pure function of the copy,
        safe on any thread."""
        import io

        from PIL import Image
        if snap.get("v2"):
            # a monitor artifact must not advance the model's grow-only
            # window: render with the window as it stands, discard the new one
            arr, _, _ = self._render.render_annotated_map_v2(
                snap["grid"], snap["cell_size"], snap["origin"],
                agent_xz=snap["agent_xz"], agent_yaw=snap["yaw"],
                trajectory=snap["trajectory"], window_cells=snap["window"],
                semantic_layers=snap["sem_layers"],
                floor_label=snap.get("floor_label"),
                mark_start=bool(snap.get("mark_start", True)))
        else:
            arr, _ = self._render.render_annotated_map(
                snap["grid"], snap["fmask"], snap["cell_size"], snap["origin"],
                agent_xz=snap["agent_xz"], agent_yaw=snap["yaw"],
                trajectory=snap["trajectory"], frontier_cells=snap["marks"],
                semantic_layers=snap["sem_layers"])
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def render(self, waypoints: list, amap: Any) -> tuple[bytes, dict]:
        """The package's own annotated renderer. The dwp menu's numbers ride
        the frontier_cells channel (their numbered-circle mechanism), so
        circle N on this map IS place N in WALKABLE PLACES."""
        import io

        from PIL import Image
        yaw = -float(amap.theta)
        agent_xz = (float(amap.px), float(amap.py))
        if self.map_mode == "v2":
            # jian's map v2 contract: no frontier layer; the crop window is
            # snapped to 2 m multiples and only grows within the episode
            sem_layers = (self.sem.overlay_cells() if self.sem is not None
                          else None)
            traj, start_here = self._floor_track()
            arr, window_m, self.map_window = self._render.render_annotated_map_v2(
                self.occ.grid, self.occ.cell_size, self.occ.origin,
                agent_xz=agent_xz, agent_yaw=yaw, trajectory=traj,
                window_cells=self.map_window, semantic_layers=sem_layers,
                floor_label=self._floor_label(), mark_start=start_here)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            payload: dict = {"map_window_m": round(float(window_m), 1),
                             "orientation": ("up = your starting heading; "
                                             "right = right of start")}
            if self.sem is not None:
                payload["landmarks"] = self.sem.landmarks_json(agent_xz, yaw)
            return buf.getvalue(), payload
        frows: list = []
        if waypoints:
            marks = []
            for i, wp in enumerate(waypoints, 1):
                X, Y = amap.to_anchor(wp.x_left, wp.y_fwd)
                zi, xi = self.occ.world_to_cell(float(X), float(Y))
                if self.occ.in_bounds(zi, xi):
                    marks.append((i, (zi, xi)))
        else:
            # slamstep: no dwp menu — the numbered circles are jian's
            # stable-id FRONTIERS, exactly his get_map contract
            marks, frows = self.frontier_marks(agent_xz, yaw)
        sem_layers = self.sem.overlay_cells() if self.sem is not None else None
        arr, window_m = self._render.render_annotated_map(
            self.occ.grid, self._frontier.frontier_mask(self.occ.grid),
            self.occ.cell_size, self.occ.origin,
            agent_xz=agent_xz, agent_yaw=yaw,
            trajectory=self.track, frontier_cells=marks,
            semantic_layers=sem_layers)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        payload: dict = {"map_window_m": round(float(window_m), 1),
                         "orientation": ("up = your starting heading; "
                                         "right = right of start")}
        if frows:
            payload["frontiers"] = frows
        if self.sem is not None:
            payload["landmarks"] = self.sem.landmarks_json(agent_xz, yaw)
        return buf.getvalue(), payload
