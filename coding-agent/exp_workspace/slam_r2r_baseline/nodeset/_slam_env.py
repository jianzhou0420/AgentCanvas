"""SlamEnv — perception middleware between habitat and an agentic VLN agent.

Ported 2026-08-17 from the slam-frontier probe worktree (slam_frontier/
slam_env.py verbatim; imports package-relative).

Design (rung 1 of the agentic-SLAM ladder):
- The agent keeps the coarse VLN-CE action space (FORWARD 0.25 m, LEFT/RIGHT
  15 deg, STOP) but every coarse action is decomposed into 5 micro-steps
  (0.05 m or 3 deg) executed in habitat. Each micro-step's RGB-D frame is
  auto-fed to ORB-SLAM3 and the occupancy grid as a side effect of motion —
  the agent has no code path to touch, skip, or reorder frames.
- The agent's only relationship with SLAM is read-only queries:
  get_pose / get_map / get_trajectory.
- The middleware never moves the robot on its own. Tracking loss is
  bookkeeping (status flag + re-anchor on recovery), never an action.
  The one disclosed exception is bootstrap(): a 360 deg scan used as
  episode initialization before the agent takes over.

Micro quantum 0.05 m / 3 deg sits well inside the validated stability
envelope (0.125 m / 6 deg at 512 px — probe 2026-08-16).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._env import HabitatEnv
from ._frontier import cluster_frontiers, cluster_representative, frontier_mask
from ._geom import angle_diff, relative_cv_pose, yaw_of
from ._mapping import LayeredOccupancyMap, OccupancyMap, render_map, save_map_png
from ._slam import TRACKING_OK, OrbSlam3Client

ACTIONS = {"forward": 1, "turn_left": 2, "turn_right": 3}
MICRO_PER_ACTION = 5


class SlamEnv:
    """Coarse actions in, auto-fed SLAM inside, read-only queries out."""

    def __init__(
        self,
        scene_path: str,
        img_size: int = 512,
        coarse_step_m: float = 0.25,
        coarse_turn_deg: float = 15.0,
        cell_size: float = 0.10,
        map_size_m: float = 48.0,
        pose_source: str = "slam",
        frontier_match_m: float = 1.5,
        map_mode: str = "v1",
    ) -> None:
        self.micro_step_m = coarse_step_m / MICRO_PER_ACTION      # 0.05 m
        self.micro_turn_deg = coarse_turn_deg / MICRO_PER_ACTION  # 3 deg
        self.env = HabitatEnv(scene_path, img_size=img_size,
                              step_size_m=self.micro_step_m,
                              turn_angle_deg=self.micro_turn_deg)
        # map v2 (SLAM-02): per-floor layers, height reference = current
        # camera; no frontier surface; grow-only snapped render window.
        self.map_mode = map_mode
        if map_mode == "v2":
            self.occ = LayeredOccupancyMap(cell_size=cell_size,
                                           map_size_m=map_size_m,
                                           camera_height=self.env.camera_height)
        else:
            self.occ = OccupancyMap(cell_size=cell_size, map_size_m=map_size_m,
                                    camera_height=self.env.camera_height)
        self.map_window: tuple | None = None  # v2 render crop, grow-only
        self.pose_source = pose_source
        self.slam: OrbSlam3Client | None = None
        self.tick = 0                      # micro-step counter = global clock
        self.status = "ok"
        self.last_pose: np.ndarray | None = None
        self.est_track: list = []
        self.gt_track: list = []
        self.lost_ticks = 0
        self.reanchors = 0
        self._anchor = np.eye(4)
        self._need_reanchor = False
        self._T0_inv: np.ndarray | None = None
        # Stable frontier ids (2026-08-18): clusters are recomputed per
        # get_map, but an opening keeps its id across calls — each current
        # representative is greedily matched to the nearest registered one
        # within ``frontier_match_m``; unmatched clusters draw fresh ids from
        # a monotonic counter (retired ids are never reused).
        self._frontier_match_m = frontier_match_m
        self._frontier_registry: list = []  # [(id, (wx, wz))]
        self._frontier_next_id = 1

    # ── episode control ──

    def reset(self, position: list | None, rotation_xyzw: list | None) -> None:
        self.env.reset(position, rotation_xyzw)
        self._T0_inv = np.linalg.inv(self.env.gt_camera_pose_cv())
        if self.pose_source == "slam":
            self.slam = OrbSlam3Client()
            self.slam.init(self.env.intrinsics(), fps=10.0)
        self._feed()  # anchor SLAM on the first frame

    def bootstrap(self, steps_per_feed: int = 2) -> dict:
        """Disclosed episode init: one full 360 deg in-place scan.

        Feeds SLAM every ``steps_per_feed`` micro-turns (default 2 -> 6 deg
        per frame, 60 frames), matching the validated v1 practice (all nine
        published runs bootstrapped at 6 deg/frame). Note: spin-induced yaw
        drift varies 5-20 deg run-to-run (RANSAC stochasticity, probe
        2026-08-17) — downstream consumers get an internally consistent
        (pose, map) pair either way. Frames render only on observe(), so
        intermediate un-observed steps skip nothing.
        """
        n = round(360.0 / (self.micro_turn_deg * steps_per_feed))
        for _ in range(n):
            for _ in range(steps_per_feed):
                self.env.step(2)
            self._feed()
        return {"feeds": n, "deg_per_feed": self.micro_turn_deg * steps_per_feed,
                "slam": self.status, "tick": self.tick}

    def close(self) -> None:
        if self.slam is not None:
            self.slam.close()
        self.env.close()

    # ── the only way the world moves ──

    def act(self, action: int | str, feed_every: int = 1) -> dict:
        """Execute one coarse action as 5 sim micro-steps.

        Default feeds SLAM on every micro-step (0.05 m / 3 deg per frame).
        ``feed_every`` sparsifies feeding (always feeds the last / colliding
        step); single-run probes showed no consistent accuracy win for
        sparser nav feeding, so dense stays the default.
        """
        a = ACTIONS.get(action, action) if isinstance(action, str) else int(action)
        if a not in (1, 2, 3):
            raise ValueError(f"unknown action {action!r}")
        done = 0
        collision = False
        for i in range(MICRO_PER_ACTION):
            collision = self.env.step(a)
            done += 1
            if (i + 1) % feed_every == 0 or i == MICRO_PER_ACTION - 1 or collision:
                self._feed()
            if collision:  # abort remaining micro-steps, report honestly
                if self.last_pose is not None:
                    self.occ.mark_obstacle_ahead(self.last_pose)
                break
        return {"action": a, "micro_steps_done": done, "collision": collision,
                "slam": self.status, "tick": self.tick}

    # ── auto-feed (structural invariant: every rendered frame, once, in order) ──

    def _feed(self) -> None:
        obs = self.env.observe()
        assert self._T0_inv is not None, "reset() before stepping"
        gt_pose = relative_cv_pose(self.env.gt_camera_pose_cv(), self._T0_inv)
        if self.slam is not None:
            res = self.slam.track(obs["rgb"], obs["depth"], t=self.tick * 0.1)
            raw = res["pose"] if res["state"] == TRACKING_OK else None
            if res["state"] == 1:  # NOT_INITIALIZED after a reset: origin moved
                self._need_reanchor = True
            if raw is not None and self._need_reanchor:
                self._anchor = (self.last_pose if self.last_pose is not None
                                else np.eye(4)) @ np.linalg.inv(raw)
                self._need_reanchor = False
                self.reanchors += 1
            pose = None if raw is None else self._anchor @ raw
            self.status = "ok" if raw is not None else "slam_lost"
        else:
            pose = gt_pose
        if pose is not None:
            self.last_pose = pose
            self.occ.integrate(pose, obs["depth"], self.env.intrinsics())
            self.est_track.append([float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])])
            self.gt_track.append([float(gt_pose[0, 3]), float(gt_pose[1, 3]),
                                  float(gt_pose[2, 3])])
        else:
            self.lost_ticks += 1
        self.tick += 1

    # ── read-only queries (no side effects, callable any time) ──

    def get_pose(self) -> dict:
        if self.last_pose is None:
            return {"slam": self.status, "tick": self.tick, "pose": None}
        p = self.last_pose
        return {"x": round(float(p[0, 3]), 3), "z": round(float(p[2, 3]), 3),
                "yaw_deg": round(math.degrees(yaw_of(p)), 1),
                "slam": self.status, "tick": self.tick}

    def _stable_frontier_ids(self, reps: list) -> list:
        """Match current cluster representatives (world coords) against the
        registry; return the id per rep, updating the registry in place."""
        pairs = []
        for ri, (wx, wz) in enumerate(reps):
            for ei, (_fid, (ox, oz)) in enumerate(self._frontier_registry):
                d = math.hypot(wx - ox, wz - oz)
                if d <= self._frontier_match_m:
                    pairs.append((d, ri, ei))
        pairs.sort()
        assigned: dict = {}
        used: set = set()
        for _d, ri, ei in pairs:
            if ri in assigned or ei in used:
                continue
            assigned[ri] = self._frontier_registry[ei][0]
            used.add(ei)
        ids = []
        registry = []
        for ri, (wx, wz) in enumerate(reps):
            fid = assigned.get(ri)
            if fid is None:
                fid = self._frontier_next_id
                self._frontier_next_id += 1
            ids.append(fid)
            registry.append((fid, (wx, wz)))
        self._frontier_registry = registry  # unmatched old ids retire
        return ids

    def get_map(self, png_path: str | None = None) -> dict:
        fmask = frontier_mask(self.occ.grid)
        clusters = cluster_frontiers(fmask)
        frontiers = []
        if self.last_pose is not None:
            px, pz = float(self.last_pose[0, 3]), float(self.last_pose[2, 3])
            yaw = yaw_of(self.last_pose)
            cells = [cluster_representative(comp) for comp in clusters]
            reps = [self.occ.cell_to_world(zi, xi) for zi, xi in cells]
            ids = self._stable_frontier_ids(reps)
            for fid, (zi, xi), (wx, wz), comp in zip(  # noqa: B905 — py3.9 env
                    ids, cells, reps, clusters):
                bearing = math.atan2(wx - px, wz - pz)
                frontiers.append({
                    "id": f"F{fid}",
                    "dir_deg": round(math.degrees(angle_diff(bearing, yaw)), 1),
                    "dist_m": round(math.hypot(wx - px, wz - pz), 2),
                    "size_cells": int(len(comp)),
                    # integration addition (2026-08-18): representative cell
                    # for the annotated render — stripped from the
                    # agent-facing payload by the manager
                    "cell": (zi, xi),
                })
            frontiers.sort(key=lambda f: f["dist_m"])
        agent_cell = None
        if self.last_pose is not None:
            agent_cell = self.occ.world_to_cell(float(self.last_pose[0, 3]),
                                                float(self.last_pose[2, 3]))
        if png_path is not None:
            save_map_png(png_path, self.occ.grid, fmask, agent_cell, None)
        return {
            "frontiers": frontiers,
            "dir_convention": "dir_deg relative to heading, positive = to the right",
            "free_m2": self.occ.free_area_m2(),
            "explored_m2": self.occ.explored_area_m2(),
            "png": png_path,
            "slam": self.status, "tick": self.tick,
        }

    def get_trajectory(self, max_points: int = 200) -> dict:
        pts = [[round(p[0], 2), round(p[2], 2)] for p in self.est_track]
        stride = max(1, len(pts) // max_points)
        return {"points": pts[::stride], "n_total": len(pts),
                "slam": self.status, "tick": self.tick}

    # ── eval-side helpers (not part of the agent surface) ──

    def map_image(self) -> np.ndarray:
        agent_cell = None
        if self.last_pose is not None:
            agent_cell = self.occ.world_to_cell(float(self.last_pose[0, 3]),
                                                float(self.last_pose[2, 3]))
        return render_map(self.occ.grid, frontier_mask(self.occ.grid), agent_cell, None)

    def metrics(self) -> dict:
        n = min(len(self.est_track), len(self.gt_track))
        ate = float("nan")
        if n:
            e = np.asarray(self.est_track[:n]) - np.asarray(self.gt_track[:n])
            ate = float(np.sqrt((e ** 2).sum(axis=1).mean()))
        return {
            "ticks": self.tick,
            "lost_ticks": self.lost_ticks,
            "tracking_ratio": (self.tick - self.lost_ticks) / max(1, self.tick),
            "reanchors": self.reanchors,
            "ate_rmse_m": ate,
            "free_m2": self.occ.free_area_m2(),
            "explored_m2": self.occ.explored_area_m2(),
            "collisions": self.env.collisions,
        }
