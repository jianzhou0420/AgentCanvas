"""2D occupancy mapping from depth + camera pose (+ map rendering).

Ported 2026-08-17 from the slam-frontier probe worktree (slam_frontier/
mapping.py verbatim; render_map/save_map_png absorbed from its run.py — they
render this module's grid, so they live with it here).

Frame convention: poses are 4x4 world-from-camera in OpenCV camera axes
(x right, y DOWN, z forward), with the world anchored at the FIRST camera
frame — exactly what ORB-SLAM3 emits, and what _slam_env converts GT poses
to. The camera is level at ``camera_height`` above the floor, so in this
world frame the floor sits near y = +camera_height and the ground plane is
x-z.

Grid: fixed-size square, ``map_size_m`` across at ``cell_size`` resolution,
centered on the world origin (= the start position). Ternary cells:
0 unknown / 1 free / 2 obstacle, with obstacle-wins overwrite and free
carved by per-ray Bresenham from the agent cell to each measured endpoint.
"""

from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OBSTACLE = 0, 1, 2


class OccupancyMap:
    def __init__(
        self,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        camera_height: float = 1.25,
        obstacle_h_min: float = 0.15,
        obstacle_h_max: float = 1.4,
        depth_min: float = 0.3,
        depth_max: float = 4.5,
        px_stride: int = 4,
    ) -> None:
        self.cell_size = cell_size
        self.n = round(map_size_m / cell_size)
        self.origin = self.n // 2  # world (0,0) -> cell (origin, origin)
        self.grid = np.zeros((self.n, self.n), dtype=np.uint8)  # [zi, xi]
        self.camera_height = camera_height
        self.obstacle_h_min = obstacle_h_min
        self.obstacle_h_max = obstacle_h_max
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.px_stride = px_stride

    # ── coordinate helpers ──

    def world_to_cell(self, x: float, z: float) -> tuple:
        xi = int(np.floor(x / self.cell_size)) + self.origin
        zi = int(np.floor(z / self.cell_size)) + self.origin
        return zi, xi

    def cell_to_world(self, zi: int, xi: int) -> tuple:
        x = (xi - self.origin + 0.5) * self.cell_size
        z = (zi - self.origin + 0.5) * self.cell_size
        return x, z

    def in_bounds(self, zi: int, xi: int) -> bool:
        return 0 <= zi < self.n and 0 <= xi < self.n

    # ── integration ──

    def integrate(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
                  floor_ref_y: float | None = None) -> None:
        """Project one depth frame through the pose into the grid.

        ``floor_ref_y`` (map v2): camera world-y to use as the height
        reference, so obstacle/floor classification is relative to the floor
        UNDER the camera. Default None keeps the v1 behaviour (reference =
        the start floor, world y 0).
        """
        s = self.px_stride
        h, w = depth_m.shape
        fx = intrinsics["fx"] * w / intrinsics["width"]
        fy = intrinsics["fy"] * h / intrinsics["height"]
        cx = intrinsics["cx"] * w / intrinsics["width"]
        cy = intrinsics["cy"] * h / intrinsics["height"]

        vs, us = np.mgrid[0:h:s, 0:w:s]
        z = depth_m[::s, ::s].astype(np.float64)
        valid = (z > self.depth_min) & (z < self.depth_max)
        us, vs, z = us[valid], vs[valid], z[valid]
        if z.size == 0:
            return
        x_cam = (us - cx) / fx * z
        y_cam = (vs - cy) / fy * z
        pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
        pts_w = pose_wc @ pts_cam  # 4xN

        ref_y = 0.0 if floor_ref_y is None else floor_ref_y
        height_above_floor = self.camera_height - (pts_w[1] - ref_y)
        is_obstacle = (height_above_floor > self.obstacle_h_min) & \
                      (height_above_floor < self.obstacle_h_max)
        is_floor = (height_above_floor <= self.obstacle_h_min) & \
                   (height_above_floor > -0.5)
        keep = is_obstacle | is_floor
        if not keep.any():
            return
        xw, zw = pts_w[0][keep], pts_w[2][keep]
        obs_flag = is_obstacle[keep]

        a_zi, a_xi = self.world_to_cell(float(pose_wc[0, 3]), float(pose_wc[2, 3]))
        if self.in_bounds(a_zi, a_xi):
            self.grid[a_zi, a_xi] = FREE

        # Endpoint cells, deduplicated (obstacle wins within the frame)
        zi = np.floor(zw / self.cell_size).astype(int) + self.origin
        xi = np.floor(xw / self.cell_size).astype(int) + self.origin
        ib = (zi >= 0) & (zi < self.n) & (xi >= 0) & (xi < self.n)
        zi, xi, obs_flag = zi[ib], xi[ib], obs_flag[ib]

        seen: dict = {}
        for a, b, o in zip(zi, xi, obs_flag):  # noqa: B905 — py3.9 env
            key = (int(a), int(b))
            seen[key] = seen.get(key, False) or bool(o)

        for (ezi, exi), o in seen.items():
            for czi, cxi in _bresenham(a_zi, a_xi, ezi, exi)[:-1]:
                if self.in_bounds(czi, cxi) and self.grid[czi, cxi] != OBSTACLE:
                    self.grid[czi, cxi] = FREE
            if o:
                self.grid[ezi, exi] = OBSTACLE
            elif self.grid[ezi, exi] != OBSTACLE:
                self.grid[ezi, exi] = FREE

    def mark_obstacle_ahead(self, pose_wc: np.ndarray, dist_m: float = 0.3) -> None:
        """On collision: stamp an obstacle just ahead of the camera."""
        fwd = pose_wc[:3, 2]  # camera z axis in world
        p = pose_wc[:3, 3] + fwd * dist_m
        zi, xi = self.world_to_cell(float(p[0]), float(p[2]))
        if self.in_bounds(zi, xi):
            self.grid[zi, xi] = OBSTACLE

    # ── stats ──

    def explored_area_m2(self) -> float:
        return float((self.grid != UNKNOWN).sum()) * self.cell_size ** 2

    def free_area_m2(self) -> float:
        return float((self.grid == FREE).sum()) * self.cell_size ** 2


class LayeredOccupancyMap:
    """Per-floor occupancy layers (map v2).

    One OccupancyMap per registered floor. The camera's world-y picks the
    layer each frame: within ``floor_merge_m`` of a registered floor it
    joins that floor, otherwise a new floor is registered. Heights inside a
    frame are classified relative to the CURRENT camera height, so the
    plane under the agent is always the reference — v1 anchored everything
    to the start floor and misclassified other storeys.

    ``.grid`` exposes the current floor's grid, so single-grid callers
    (render, frontier, evaluate) work unchanged. Geometry constants
    (cell_size / n / origin) are shared by construction.
    """

    def __init__(
        self,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        camera_height: float = 1.25,
        floor_merge_m: float = 1.0,
        dwell_feeds: int = 8,
        dwell_span_m: float = 0.2,
    ) -> None:
        self._layer_kw = {"cell_size": cell_size, "map_size_m": map_size_m,
                          "camera_height": camera_height}
        self.floor_merge_m = floor_merge_m
        # Floor switches need DWELL: mid-stair heights would otherwise
        # register a fencepost "floor" every merge-threshold of climb (a
        # 3.6 m stair replay produced 4 floors without this). Only when the
        # camera height has been stable — span < dwell_span_m over the last
        # dwell_feeds frames — may the registry switch or register.
        self._dwell_feeds = dwell_feeds
        self._dwell_span_m = dwell_span_m
        self._y_hist: list = []
        # world is anchored at the start camera, so floor 0 sits at y = 0
        self.layers: list = [OccupancyMap(**self._layer_kw)]
        self.floor_ys: list = [0.0]
        self.current_floor = 0
        proto = self.layers[0]
        self.cell_size = proto.cell_size
        self.n = proto.n
        self.origin = proto.origin

    # ── floor registry ──

    def _update_floor(self, pose_y: float) -> None:
        self._y_hist.append(pose_y)
        if len(self._y_hist) > self._dwell_feeds:
            self._y_hist.pop(0)
        if len(self._y_hist) < self._dwell_feeds:
            return  # not enough history yet — stay on the current floor
        if max(self._y_hist) - min(self._y_hist) >= self._dwell_span_m:
            return  # climbing/descending — no switches mid-transit
        y_stable = sorted(self._y_hist)[len(self._y_hist) // 2]
        d, fid = min((abs(y_stable - fy), i) for i, fy in enumerate(self.floor_ys))
        if d <= self.floor_merge_m:
            self.current_floor = fid
            return
        self.floor_ys.append(y_stable)
        self.layers.append(OccupancyMap(**self._layer_kw))
        self.current_floor = len(self.layers) - 1

    @property
    def grid(self) -> np.ndarray:
        return self.layers[self.current_floor].grid

    @property
    def current_floor_y(self) -> float:
        return self.floor_ys[self.current_floor]

    # ── integration (mirrors OccupancyMap's surface) ──

    def integrate(self, pose_wc: np.ndarray, depth_m: np.ndarray, intrinsics: dict) -> None:
        pose_y = float(pose_wc[1, 3])
        self._update_floor(pose_y)
        self.layers[self.current_floor].integrate(
            pose_wc, depth_m, intrinsics, floor_ref_y=pose_y)

    def mark_obstacle_ahead(self, pose_wc: np.ndarray, dist_m: float = 0.3) -> None:
        self.layers[self.current_floor].mark_obstacle_ahead(pose_wc, dist_m)

    # ── coordinate helpers (pure geometry, layer-independent) ──

    def world_to_cell(self, x: float, z: float) -> tuple:
        return self.layers[0].world_to_cell(x, z)

    def cell_to_world(self, zi: int, xi: int) -> tuple:
        return self.layers[0].cell_to_world(zi, xi)

    def in_bounds(self, zi: int, xi: int) -> bool:
        return self.layers[0].in_bounds(zi, xi)

    # ── stats (all floors) ──

    def explored_area_m2(self) -> float:
        return float(sum(layer.explored_area_m2() for layer in self.layers))

    def free_area_m2(self) -> float:
        return float(sum(layer.free_area_m2() for layer in self.layers))


def _bresenham(z0: int, x0: int, z1: int, x1: int) -> list:
    """Integer line from (z0,x0) to (z1,x1), endpoints inclusive."""
    dz, dx = abs(z1 - z0), abs(x1 - x0)
    sz = 1 if z1 >= z0 else -1
    sx = 1 if x1 >= x0 else -1
    cells = []
    if dx >= dz:
        err = dx // 2
        z = z0
        for x in range(x0, x1 + sx, sx):
            cells.append((z, x))
            err -= dz
            if err < 0:
                z += sz
                err += dx
    else:
        err = dz // 2
        x = x0
        for z in range(z0, z1 + sz, sz):
            cells.append((z, x))
            err -= dx
            if err < 0:
                x += sx
                err += dz
    return cells


def render_map(grid: np.ndarray, fmask: np.ndarray | None,
               agent_cell: tuple | None,
               plan: list | None) -> np.ndarray:
    img = np.full((*grid.shape, 3), 128, dtype=np.uint8)  # unknown: gray
    img[grid == FREE] = (255, 255, 255)
    img[grid == OBSTACLE] = (0, 0, 0)
    if fmask is not None:
        img[fmask] = (220, 40, 40)
    if plan:
        for zi, xi in plan:
            img[zi, xi] = (40, 180, 60)
    if agent_cell is not None:
        zi, xi = agent_cell
        img[max(0, zi - 2):zi + 3, max(0, xi - 2):xi + 3] = (40, 80, 255)
    return img[::-1]  # flip so +z points up


def save_map_png(path: str, grid: np.ndarray, fmask: np.ndarray | None,
                 agent_cell: tuple | None,
                 plan: list | None) -> None:
    from PIL import Image

    Image.fromarray(render_map(grid, fmask, agent_cell, plan)).save(path)
