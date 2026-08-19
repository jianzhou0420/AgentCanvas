"""Quaternion helpers — numpy reimplementation of NavHarness ``envs/_quat.py``.

Vendored from:
    NavHarness navharness/projects/internutopia_vln_extension/envs/_quat.py

Upstream backs these with ``scipy.spatial.transform.Rotation`` as drop-in
replacements for ``omni.isaac.core.utils.rotations``. Deviations here:

- scipy dependency dropped (nodeset dependency whitelist: numpy + stdlib).
  The math reproduces scipy's lowercase-``'xyz'`` behavior exactly —
  **extrinsic** x→y→z rotations, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
  (Upstream's docstring says "intrinsic" but the code passes lowercase
  ``'xyz'``, which scipy defines as extrinsic; we match the code, not the
  comment. For the yaw-only rotations this nodeset performs the two
  conventions coincide.)
- Same signatures and (w, x, y, z) ordering as upstream.

Parity with scipy is pinned by ``test_kinematics.py`` (skipped when scipy
is absent).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z) ordering."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def euler_angles_to_quat(rpy: Sequence[float], degrees: bool = False) -> np.ndarray:
    """Drop-in for ``omni.isaac.core.utils.rotations.euler_angles_to_quat``.

    Args:
        rpy: (roll, pitch, yaw) — extrinsic 'xyz' (scipy lowercase) order.
        degrees: interpret rpy in degrees if True.

    Returns:
        np.ndarray of shape (4,) ordered (w, x, y, z).
    """
    r, p, y = np.asarray(rpy, dtype=np.float64)
    if degrees:
        r, p, y = np.radians(r), np.radians(p), np.radians(y)
    qx = np.array([np.cos(r / 2.0), np.sin(r / 2.0), 0.0, 0.0])
    qy = np.array([np.cos(p / 2.0), 0.0, np.sin(p / 2.0), 0.0])
    qz = np.array([np.cos(y / 2.0), 0.0, 0.0, np.sin(y / 2.0)])
    # Extrinsic x→y→z composes as R = Rz @ Ry @ Rx ⇒ q = qz ⊗ qy ⊗ qx.
    return _quat_mul(qz, _quat_mul(qy, qx))


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion (normalized inside)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_euler_angles(quat_wxyz: Sequence[float], degrees: bool = False) -> np.ndarray:
    """Drop-in for ``omni.isaac.core.utils.rotations.quat_to_euler_angles``.

    Args:
        quat_wxyz: quaternion as (w, x, y, z).
        degrees: return result in degrees if True.

    Returns:
        np.ndarray of shape (3,) — (roll, pitch, yaw), extrinsic 'xyz'.
    """
    m = _quat_to_matrix(np.asarray(quat_wxyz, dtype=np.float64))
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll) ⇒ R[2,0] = -sin(pitch).
    sp = -m[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    if abs(sp) >= 1.0 - 1e-9:
        # Gimbal lock: roll and yaw degenerate; put everything in yaw
        # (matches scipy's convention of zeroing the first angle).
        pitch = np.pi / 2.0 if sp > 0 else -np.pi / 2.0
        roll = 0.0
        yaw = float(np.arctan2(-m[0, 1], m[1, 1]))
    else:
        pitch = float(np.arcsin(sp))
        roll = float(np.arctan2(m[2, 1], m[2, 2]))
        yaw = float(np.arctan2(m[1, 0], m[0, 0]))
    out = np.array([roll, pitch, yaw], dtype=np.float64)
    return np.degrees(out) if degrees else out
