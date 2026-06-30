"""Vendored TSDFPlanner + geometry utilities from explore-eqa.

Sources (verbatim with minor tweaks) from upstream explore-eqa @ 18381da3:
  - src/tsdf.py
  - src/geom.py
Re-fetch upstream: workspace/nodesets/_upstream/explore-eqa/fetch_upstream.sh

Differences from upstream:
  1. `geom.py::get_scene_bnds` lives in the env nodeset
     (`workspace/nodesets/env/env_hmeqa/__init__.py::_get_scene_bnds`) because it
     needs a habitat pathfinder — irrelevant to the pure planner.
  2. `geom.py::get_cam_intr` also lives in the env nodeset.
  3. Module is self-contained: heavy imports (numba, scipy, skimage,
     sklearn, matplotlib) happen inside classes/functions, so loading
     the file from the agentcanvas env (Py 3.10) for component-registry
     scan does not require the full hmeqa env.
  4. No `from src.habitat import` — the planner never needed habitat.

Used by:
  `workspace/nodesets/explore_eqa.py` — wraps TSDFPlanner into
  canvas nodes (TSDFUpdate, FrontierScoringVLM, NextPosePlanner).

License:
  BSD 2-Clause (inherited from
  https://github.com/andyzeng/tsdf-fusion-python + Princeton / Allen Ren
  modifications in explore-eqa).

last updated: 2026-04-24
"""

from __future__ import annotations

import heapq
import logging
import math
import random

import numpy as np

log = logging.getLogger("agentcanvas.explore_eqa.tsdf")


# ══════════════════════════════════════════════════════════════════════
# Geometry helpers (vendored from explore-eqa/src/geom.py)
# ══════════════════════════════════════════════════════════════════════


def rigid_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def points_in_circle(center_x, center_y, radius, grid_shape):
    x, y = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing="ij")
    distance_matrix = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    inside = np.where(distance_matrix <= radius)
    return list(zip(inside[0], inside[1]))  # noqa: B905 — strict= unsupported on hmeqa env's Python 3.9


def run_dijkstra(grid, start, end):
    start = tuple(start[:2])
    end = tuple(end[:2])
    rows, cols = grid.shape
    directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]
    distance = np.full(grid.shape, np.inf)
    distance[start] = 0
    prev = {start: None}
    pq = [(0, start)]
    while pq:
        dist, current = heapq.heappop(pq)
        if current == end:
            break
        for dr, dc in directions:
            r, c = current[0] + dr, current[1] + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                new_dist = dist + math.sqrt(dr**2 + dc**2)
                if new_dist < distance[r, c]:
                    distance[r, c] = new_dist
                    heapq.heappush(pq, (new_dist, (r, c)))
                    prev[(r, c)] = current
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev.get(current)
    return path[::-1]


def find_normal(grid, x, y):
    import scipy.ndimage as ndimage

    sobel_y = ndimage.sobel(grid, axis=0)
    sobel_x = ndimage.sobel(grid, axis=1)
    Gx = sobel_x[x, y]
    Gy = sobel_y[x, y]
    if Gx == 0 and Gy == 0:
        normal = np.array([random.random(), random.random()])
    else:
        normal = np.array([-Gy, -Gx])
    normal = normal / np.linalg.norm(normal)
    return normal


def close_operation(array, structure=None):
    import scipy.ndimage as ndimage

    dilated = ndimage.binary_dilation(array, structure=structure)
    closed = ndimage.binary_erosion(dilated, structure=structure)
    return closed


def fps(points, n_samples):
    """Farthest-point-sampling subset of shape (n_samples, D)."""
    points = np.array(points)
    points_left = np.arange(len(points))
    sample_inds = np.zeros(n_samples, dtype="int")
    dists = np.ones_like(points_left, dtype=float) * float("inf")
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)
    for i in range(1, n_samples):
        if len(points_left) == 0:
            sample_inds = sample_inds[:i]
            break
        last_added = sample_inds[i - 1]
        dist_to_last = ((points[last_added] - points[points_left]) ** 2).sum(-1)
        dists[points_left] = np.minimum(dist_to_last, dists[points_left])
        selected = int(np.argmax(dists[points_left]))
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)
    return points[sample_inds]


# ══════════════════════════════════════════════════════════════════════
# TSDFPlanner (vendored from explore-eqa/src/tsdf.py)
# ══════════════════════════════════════════════════════════════════════


class TSDFPlanner:
    """Volumetric TSDF fusion + frontier-based semantic exploration.

    CPU-only. Maintains four voxel volumes (TSDF, color, semantic value,
    exploration mask) plus a 2D semantic-value weight accumulator.
    Frontiers are discovered on the 2D island (reachable free space),
    weighted by unexplored/unoccupied direction rates and the smoothed
    semantic value map, and sampled to select the next pose.
    """

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        floor_height_offset=0,
        pts_init=None,
        init_clearance=0,
    ):
        from numba import njit, prange  # noqa: F401 — njit for vox2world/cam2pix

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size
        self._color_const = 256 * 256

        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._val_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._weight_val_vol_cpu = np.zeros(self._vol_dim[:2]).astype(np.float32)
        self._explore_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0)
            .astype(int)
            .T
        )

        self.cam_pts_pre = TSDFPlanner.vox2world(
            self._vol_origin, self.vox_coords, self._voxel_size
        )

        self.min_height_voxel = int(floor_height_offset / self._voxel_size)

        coords_init = self.world2vox(pts_init)
        self.init_points = points_in_circle(
            coords_init[0],
            coords_init[1],
            int(init_clearance / self._voxel_size),
            self._vol_dim[:2],
        )

        self.target_point = None
        self.target_direction = None
        self.max_point = None
        self.candidates = np.empty((0, 2))

    # ── Pure-data state interface (for nodeset-owned container storage) ──
    #
    # The planner's entire instance state is numpy / list / scalar / None —
    # zero non-serializable handles. ``export_state`` / ``bind_state`` let a
    # nodeset hold the map as a set of *pure-data* named container states (no
    # live object) and rebuild a transient planner per verb call. ``__init__``
    # and every numerical method below are untouched, so the class API stays
    # backward compatible (e.g. ``tooleqa_explore`` keeps working).
    #
    # (state_name, instance_attr) — state_name is what the container declares.
    _STATE_FIELDS = (
        ("tsdf_vol", "_tsdf_vol_cpu"),
        ("weight_vol", "_weight_vol_cpu"),
        ("color_vol", "_color_vol_cpu"),
        ("val_vol", "_val_vol_cpu"),
        ("weight_val_vol", "_weight_val_vol_cpu"),
        ("explore_vol", "_explore_vol_cpu"),
        ("vol_bnds", "_vol_bnds"),
        ("vol_dim", "_vol_dim"),
        ("vol_origin", "_vol_origin"),
        ("voxel_size", "_voxel_size"),
        ("trunc_margin", "_trunc_margin"),
        ("color_const", "_color_const"),
        ("min_height_voxel", "min_height_voxel"),
        ("vox_coords", "vox_coords"),
        ("cam_pts_pre", "cam_pts_pre"),
        ("init_points", "init_points"),
        # exploration intermediates (reassigned during find_*/next_pose)
        ("candidates", "candidates"),
        ("target_point", "target_point"),
        ("target_direction", "target_direction"),
        ("max_point", "max_point"),
        ("cur_point", "cur_point"),
        ("island", "island"),
        ("unexplored", "unexplored"),
        ("unoccupied", "unoccupied"),
        ("occupied", "occupied"),
        ("unexplored_neighbors", "unexplored_neighbors"),
    )

    def export_state(self) -> dict:
        """Return the full planner state as a pure-data dict keyed by
        container-state name (numpy / list / scalar / None). Attributes set
        only after exploration (e.g. ``cur_point``) export as ``None`` until
        present. No method / live object is included."""
        return {name: getattr(self, attr, None) for name, attr in self._STATE_FIELDS}

    @classmethod
    def bind_state(cls, st: dict) -> "TSDFPlanner":
        """Rebuild a transient planner bound *by reference* to an existing
        pure-data state dict, skipping ``__init__`` (and its O(N) voxel-grid
        rebuild). In-place mutations land in the same arrays the caller writes
        back via ``export_state``."""
        obj = cls.__new__(cls)
        for name, attr in cls._STATE_FIELDS:
            setattr(obj, attr, st.get(name))
        return obj

    # ── Static numba kernels ──

    @staticmethod
    def vox2world(vol_origin, vox_coords, vox_size):
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(vol_origin, vox_coords, vox_size):
            vol_origin = vol_origin.astype(np.float32)
            vox_coords = vox_coords.astype(np.float32)
            cam_pts = np.empty_like(vox_coords, dtype=np.float32)
            for i in prange(vox_coords.shape[0]):
                for j in range(3):
                    cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
            return cam_pts

        return _kernel(vol_origin, vox_coords, vox_size)

    @staticmethod
    def cam2pix(cam_pts, intr):
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(cam_pts, intr):
            intr = intr.astype(np.float32)
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
            for i in prange(cam_pts.shape[0]):
                pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
                pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
            return pix

        return _kernel(cam_pts, intr)

    @staticmethod
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(tsdf_vol, dist, w_old, obs_weight):
            tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
            w_new = np.empty_like(w_old, dtype=np.float32)
            for i in prange(len(tsdf_vol)):
                w_new[i] = w_old[i] + obs_weight
                tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
            return tsdf_vol_int, w_new

        return _kernel(tsdf_vol, dist, w_old, obs_weight)

    def pix2cam(self, pix, intr):
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
        for i in range(cam_pts.shape[0]):
            cam_pts[i, 2] = 1
            cam_pts[i, 0] = (pix[i, 0] - cx) / fx * cam_pts[i, 2]
            cam_pts[i, 1] = (pix[i, 1] - cy) / fy * cam_pts[i, 2]
        return cam_pts

    def world2vox(self, pts):
        pts = pts - self._vol_origin
        coords = np.round(pts / self._voxel_size).astype(int)
        coords = np.clip(coords, 0, self._vol_dim - 1)
        return coords

    # ── Fusion ──

    def integrate_sem(self, sem_pix, radius=1.0, obs_weight=1.0):
        assert len(self.candidates) == len(sem_pix)
        for p_ind, p in enumerate(self.candidates):
            radius_vox = int(radius / self._voxel_size)
            pts = points_in_circle(p[0], p[1], radius_vox, self._vol_dim[:2])
            for pt in pts:
                w_old = self._weight_val_vol_cpu[pt[0], pt[1]].copy()
                self._weight_val_vol_cpu[pt[0], pt[1]] += obs_weight
                self._val_vol_cpu[pt[0], pt[1]] = (
                    w_old * self._val_vol_cpu[pt[0], pt[1]] + obs_weight * sem_pix[p_ind]
                ) / self._weight_val_vol_cpu[pt[0], pt[1]]

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        sem_im=None,
        w_new=None,
        obs_weight=1.0,
        margin_h=240,
        margin_w=120,
    ):
        im_h, im_w = depth_im.shape

        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0]
        )

        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        valid_pix_narrow = np.logical_and(
            pix_x >= margin_w,
            np.logical_and(
                pix_x < (im_w - margin_w),
                np.logical_and(
                    pix_y >= margin_h,
                    np.logical_and(pix_y < im_h, pix_z > 0),
                ),
            ),
        )
        depth_val_narrow = np.zeros(pix_x.shape)
        depth_val_narrow[valid_pix_narrow] = depth_im[
            pix_y[valid_pix_narrow], pix_x[valid_pix_narrow]
        ]

        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.maximum(-1, np.minimum(1, depth_diff / self._trunc_margin))
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

        depth_diff_narrow = depth_val_narrow - pix_z
        valid_pts_narrow = np.logical_and(
            depth_val_narrow > 0, depth_diff_narrow >= -self._trunc_margin
        )
        valid_vox_x_narrow = self.vox_coords[valid_pts_narrow, 0]
        valid_vox_y_narrow = self.vox_coords[valid_pts_narrow, 1]
        valid_vox_z_narrow = self.vox_coords[valid_pts_narrow, 2]
        if w_new is None:
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = TSDFPlanner.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            self._explore_vol_cpu[valid_vox_x_narrow, valid_vox_y_narrow, valid_vox_z_narrow] = 1

            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
                new_b * self._color_const + new_g * 256 + new_r
            )

        if sem_im is not None:
            old_sem = self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            new_sem = sem_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_sem = (w_old * old_sem + obs_weight * new_sem) / w_new
            self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_sem
        return w_new

    # ── Exploration ──

    def find_prompt_points_within_view(
        self,
        pts,
        im_w,
        im_h,
        cam_intr,
        cam_pose,
        # Defaults below mirror cfg/vlm_exp.yaml `visual_prompt:` block.
        height=0.4,
        cluster_threshold=1.0,
        num_prompt_points=3,
        num_max_unoccupied=300,
        min_points_for_clustering=3,
        point_min_dist=2,
        point_max_dist=10,
        cam_offset=0.6,  # paper cfg/vlm_exp.yaml line 63
        margin_h=100,
        margin_w=30,
        **kwargs,
    ):
        """Find unoccupied reachable points within the current view, cluster
        via FPS, and return pixel coords suitable for visual-prompting a VLM.

        Unlike the upstream reference this vendored variant does NOT produce
        matplotlib figures — the debug-plotting code was stripped so the
        planner can run headless inside a subprocess without leaking
        figures. Callers that want an annotated image should redraw in the
        node layer (see `FrontierScoringVLM` in explore_eqa.py).
        """
        import scipy.ndimage as ndimage

        cur_point = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts(pts, height=height)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        occupied = np.logical_not(unoccupied).astype(int)
        cam_pose = cam_pose @ np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ]
        )
        mask = self.get_current_view_mask(
            cam_intr, cam_pose, im_w, im_h, slack=0, margin_h=margin_h, margin_w=margin_w
        )

        unoccupied_in_view = np.multiply(unoccupied, mask)
        unoccupied_reachable_in_view = np.argwhere((island) & (unoccupied_in_view))

        if len(unoccupied_reachable_in_view) > 0:
            subsample_inds = np.random.choice(
                range(len(unoccupied_reachable_in_view)),
                min(num_max_unoccupied, len(unoccupied_reachable_in_view)),
                replace=False,
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[subsample_inds]
        else:
            unoccupied_reachable_in_view = np.empty((0, 2))

        unoccupied_reachable_in_view_new = np.empty((0, 2))
        for point in unoccupied_reachable_in_view:
            if not self.check_occupied_between(point, cur_point, occupied, threshold=1):
                unoccupied_reachable_in_view_new = np.concatenate(
                    (unoccupied_reachable_in_view_new, [point]), axis=0
                )
        unoccupied_reachable_in_view = unoccupied_reachable_in_view_new

        if len(unoccupied_reachable_in_view) > 0:
            dist_all = np.sqrt(
                (unoccupied_reachable_in_view[:, 0] - cur_point[0]) ** 2
                + (unoccupied_reachable_in_view[:, 1] - cur_point[1]) ** 2
            )
            mask_dist = (dist_all > point_min_dist / self._voxel_size) & (
                dist_all < point_max_dist / self._voxel_size
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[mask_dist]

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(unexplored, kernel, mode="constant", cval=0.0)
        frontiers_in_view = np.empty((0, 2))

        candidates_pre_cluster = np.concatenate(
            [frontiers_in_view, unoccupied_reachable_in_view], axis=0
        )

        if len(candidates_pre_cluster) < min_points_for_clustering:
            candidates_pix = np.empty((0, 2))
            self.candidates = np.empty((0, 2))
        else:
            clusters = fps(candidates_pre_cluster, num_prompt_points)
            clusters_new = np.empty((0, 2))
            for cluster in clusters:
                if len(clusters_new) == 0:
                    clusters_new = np.vstack((clusters_new, cluster))
                else:
                    clusters_array = np.array(clusters_new)
                    dist = np.sqrt(np.sum((clusters_array - cluster) ** 2, axis=1))
                    if np.min(dist) > cluster_threshold / self._voxel_size:
                        clusters_new = np.vstack((clusters_new, cluster))
            candidates = clusters_new
            self.candidates = candidates

            if len(candidates) > 0:
                candidates_cam = [
                    rigid_transform(
                        TSDFPlanner.vox2world(
                            self._vol_origin,
                            np.append(candidates[i], 0).reshape(1, 3),
                            self._voxel_size,
                        ),
                        np.linalg.inv(cam_pose),
                    )
                    for i in range(len(candidates))
                ]
                candidates_cam = np.concatenate(candidates_cam, axis=0)
                candidates_pix = TSDFPlanner.cam2pix(candidates_cam, cam_intr)
            else:
                candidates_pix = np.empty((0, 2))

        self.cur_point, self.island, self.unexplored = cur_point, island, unexplored
        self.unoccupied, self.occupied = unoccupied, occupied
        self.unexplored_neighbors = unexplored_neighbors

        return candidates_pix

    def find_next_pose(
        self,
        pts,
        angle,
        flag_no_val_weight=False,
        # Defaults below mirror cfg/vlm_exp.yaml `planner:` block.
        unexplored_T=0.2,  # paper cfg/vlm_exp.yaml line 41
        unoccupied_T=2.0,  # paper cfg/vlm_exp.yaml line 42
        val_T=0.5,
        val_dir_T=0.5,
        dist_T=10,
        min_dist_from_cur=0.5,
        max_dist_from_cur=3,
        frontier_spacing=1.5,
        frontier_min_neighbors=3,
        frontier_max_neighbors=4,
        max_unexplored_check_frontier=3.0,
        max_unoccupied_check_frontier=1.0,
        max_val_check_frontier=3.0,  # paper cfg/vlm_exp.yaml line 45
        smooth_sigma=5,
        eps=1,  # paper cfg/vlm_exp.yaml line 47
        **kwargs,
    ):
        """Pick the next (position, yaw) using semantic-value-weighted frontier sampling.

        Returns (next_point_normal, next_yaw, next_point_vox).
        (Upstream also returned a matplotlib figure — stripped here.)
        """
        import scipy.ndimage as ndimage
        from scipy.ndimage import gaussian_filter
        from sklearn.cluster import DBSCAN

        cur_point = self.world2vox(pts)
        if getattr(self, "cur_point", None) is not None:
            island = self.island
            unoccupied, occupied = self.unoccupied, self.occupied
            unexplored, unexplored_neighbors = (
                self.unexplored,
                self.unexplored_neighbors,
            )
        else:
            island, unoccupied = self.get_island_around_pts(pts, height=0.4)
            occupied = np.logical_not(unoccupied).astype(int)
            unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
            for point in self.init_points:
                unexplored[point[0], point[1]] = 0
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            unexplored_neighbors = ndimage.convolve(unexplored, kernel, mode="constant", cval=0.0)
        self.unexplored_neighbors = unexplored_neighbors
        self.unoccupied = unoccupied

        val_vol_2d = np.max(self._val_vol_cpu, axis=2).copy()
        val_vol_2d = gaussian_filter(val_vol_2d, sigma=smooth_sigma)

        frontiers = np.argwhere(
            island
            & (unexplored_neighbors >= frontier_min_neighbors)
            & (unexplored_neighbors <= frontier_max_neighbors)
        )

        if len(frontiers) > 10:
            db = DBSCAN(eps=eps, min_samples=2).fit(frontiers)
            labels = db.labels_
            frontiers_new = np.empty((0, 2))
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster = frontiers[labels == label]
                dist = np.sqrt(
                    (cluster[:, 0] - np.mean(cluster[:, 0])) ** 2
                    + (cluster[:, 1] - np.mean(cluster[:, 1])) ** 2
                )
                center = cluster[np.argmin(dist)]
                frontiers_new = np.append(frontiers_new, [center], axis=0)
            frontiers = frontiers_new.astype(int)

        point_type = "current"
        max_point = cur_point[:2]
        next_point = cur_point[:2]
        direction = np.random.rand(2) - 0.5
        direction = direction / np.linalg.norm(direction)
        max_num = 0
        path: list = []

        if self.target_point is None:
            frontiers_weight = np.empty(0)
            frontiers_new = np.empty((0, 2))
            for point in frontiers:
                normal = self.find_normal_into_space(point, unexplored, unexplored)

                max_pixel_check = int(max_unoccupied_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unoccupied_rate = (
                    np.sum(unoccupied[dir_pts[:, 0], dir_pts[:, 1]] == 1) / max_pixel_check
                )

                max_pixel_check = int(max_unexplored_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unexplored_rate = (
                    np.sum(unexplored[dir_pts[:, 0], dir_pts[:, 1]] == 1) / max_pixel_check
                )

                max_pixel_check = int(max_val_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                val_vol_2d_dir = val_vol_2d[dir_pts[:, 0], dir_pts[:, 1]]
                val_vol_2d_dir = val_vol_2d_dir[val_vol_2d_dir > 0]
                val = 0 if len(val_vol_2d_dir) == 0 else float(np.mean(val_vol_2d_dir))

                weight = np.exp(unexplored_rate / unexplored_T)
                weight *= np.exp(unoccupied_rate / unoccupied_T)
                if not flag_no_val_weight:
                    weight *= np.exp(val_vol_2d[point[0], point[1]] / val_T)
                    weight *= np.exp(val / val_dir_T)

                dist = (
                    np.sqrt((cur_point[0] - point[0]) ** 2 + (cur_point[1] - point[1]) ** 2)
                    * self._voxel_size
                )
                pts_angle = np.arctan2(normal[1], normal[0]) - np.pi / 2
                weight *= np.exp(-dist / dist_T)
                if (
                    dist < min_dist_from_cur / self._voxel_size
                    and np.abs(angle - pts_angle) < np.pi / 6
                ):
                    weight *= 1e-3

                frontiers_weight = np.append(frontiers_weight, weight)
                frontiers_new = np.concatenate((frontiers_new, [point]), axis=0)
            frontiers = frontiers_new.astype(int)

            if len(frontiers) > 0:
                point_type = "frontier"
                max_try = 50
                cnt_try = 0
                while True:
                    cnt_try += 1
                    if cnt_try > max_try:
                        point_type = "current"
                        break
                    frontiers_weight_red = frontiers_weight / np.mean(frontiers_weight)
                    frontier_ind = np.random.choice(
                        range(len(frontiers)),
                        p=frontiers_weight_red / np.sum(frontiers_weight_red),
                    )
                    max_point = frontiers[frontier_ind]
                    direction = self.find_normal_into_space(max_point, unexplored, unexplored)

                    next_point = np.array(max_point, dtype=float)
                    max_backtrack = int(frontier_spacing / self._voxel_size)
                    min_backtrack = 2
                    num_backtrack = 0
                    while True:
                        next_point -= direction
                        num_backtrack += 1
                        if num_backtrack >= max_backtrack:
                            break
                        if not self.check_within_bnds(next_point):
                            break
                        if (
                            occupied[int(next_point[0]), int(next_point[1])]
                            or not island[int(next_point[0]), int(next_point[1])]
                        ):
                            next_point += 2 * direction
                            break
                    next_point = np.round(next_point).astype(int)
                    if (
                        num_backtrack >= min_backtrack
                        and self.check_within_bnds(next_point)
                        and island[int(next_point[0]), int(next_point[1])]
                    ):
                        break

            if point_type == "current":
                max_point = cur_point[:2]
                next_point = cur_point[:2]
                direction = np.random.rand(2) - 0.5
                direction = direction / np.linalg.norm(direction)
        else:
            point_type = "commit"
            next_point = self.target_point.copy()
            direction = self.target_direction.copy()
            max_point = self.max_point.copy()

        log.debug("next_pose point_type=%s", point_type)

        dist = np.sqrt((next_point[0] - cur_point[0]) ** 2 + (next_point[1] - cur_point[1]) ** 2)
        if dist > max_dist_from_cur / self._voxel_size:
            self.target_point = next_point.copy()
            self.target_direction = direction.copy()
            self.max_point = max_point.copy()

            island_free = np.logical_not(island)
            path = run_dijkstra(island_free, cur_point, next_point)
            max_num = min(int(max_dist_from_cur / self._voxel_size), len(path) - 1)
            next_point = np.array(path[max_num])
            direction = max_point - next_point
            direction = direction / np.linalg.norm(direction)
        if dist <= max_dist_from_cur / self._voxel_size or (path and max_num == len(path) - 1):
            self.target_point = None
            self.target_direction = None
            self.max_point = None

        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2
        return next_point_normal, next_yaw, next_point

    # ── Support methods ──

    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        from skimage import measure

        cur_point = self.world2vox(pts)
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0,
            self._tsdf_vol_cpu[:, :, 0] < 0,
        )
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        fill_size = int(fill_dim / self._voxel_size)
        structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        unoccupied = close_operation(unoccupied, structuring_element_close)

        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]
        else:
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind
        return island, unoccupied

    def get_current_view_mask(
        self, cam_intr, cam_pose, im_w, im_h, slack=0, margin_h=0, margin_w=0
    ):
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]
        valid_pix = np.logical_and(
            pix_x >= -slack + margin_w,
            np.logical_and(
                pix_x < (im_w + slack - margin_w),
                np.logical_and(
                    pix_y >= -slack + margin_h,
                    np.logical_and(pix_y < im_h + slack, pix_z > 0),
                ),
            ),
        )
        valid_pix = valid_pix.reshape(self._vol_dim).astype(int)
        mask = np.max(valid_pix, axis=2)
        return mask

    def check_occupied_between(self, p1, p2, occupied, threshold):
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]]).astype(float)
        num_points = int(np.linalg.norm(direction))
        if num_points == 0:
            return False
        dir_norm = direction / np.linalg.norm(direction)
        points_between = (p1[:2] + dir_norm * np.arange(num_points + 1)[:, np.newaxis]).astype(int)
        points_occupied = np.sum(occupied[points_between[:, 0], points_between[:, 1]])
        return points_occupied > threshold

    def check_within_bnds(self, pts, slack=0):
        return not (
            pts[0] <= slack
            or pts[0] >= self._vol_dim[0] - slack
            or pts[1] <= slack
            or pts[1] >= self._vol_dim[1] - slack
        )

    def clip_2d_array(self, array):
        return array[
            (array[:, 0] >= 0)
            & (array[:, 0] < self._vol_dim[0])
            & (array[:, 1] >= 0)
            & (array[:, 1] < self._vol_dim[1])
        ]

    def find_normal_into_space(self, point, island, space, num_check=10):
        normal = find_normal(island.astype(int), point[0], point[1])
        dir_1 = (point + np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_2 = (point - np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_1 = self.clip_2d_array(dir_1)
        dir_2 = self.clip_2d_array(dir_2)
        dir_1_occupied = np.sum(space[dir_1[:, 0], dir_1[:, 1]])
        dir_2_occupied = np.sum(space[dir_2[:, 0], dir_2[:, 1]])
        direction = normal
        if dir_1_occupied < dir_2_occupied or (
            dir_1_occupied == dir_2_occupied and random.random() < 0.5
        ):
            direction *= -1
        return direction
