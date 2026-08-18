"""Pose-frame geometry helpers shared by the SLAM middleware.

Ported 2026-08-17 from the slam-frontier probe worktree: yaw_of / angle_diff
from its planner.py, relative_cv_pose from its run.py — the only pieces of
those modules SlamEnv consumes (the BFS planner + heading controller belong
to the standalone FBE baseline, which was not ported).

Yaw is measured in the mapping world frame (OpenCV camera axes, y down):
camera forward = pose[:3, 2], yaw = atan2(forward_x, forward_z).
"""

from __future__ import annotations

import numpy as np


def yaw_of(pose_wc: np.ndarray) -> float:
    fwd = pose_wc[:3, 2]
    return float(np.arctan2(fwd[0], fwd[2]))


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b in (-pi, pi]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def relative_cv_pose(T_cv: np.ndarray, T0_inv: np.ndarray) -> np.ndarray:
    return T0_inv @ T_cv
