"""
---------------------------------------------------------------------------
Jian's Rotation Utilities v1.0

This module provides small wrapper utilities for converting between common
3D pose / rotation representations using NumPy and SciPy
(scipy.spatial.transform.Rotation).

------------------------
# Representations
------------------------
- quat     : [x, y, z, w]                 quaternion (SciPy style, scalar last)
- euler    : [rx, ry, rz]                 3-angle rotation in 'XYZ' order
- matrix   : [3 x 3]                      orthonormal rotation matrix R
- axis     : [ax, ay, az]                 rotation vector = axis * angle (radians)

- HT       : [4 x 4]                      homogeneous transform (rotation + translation)
- RT       : (R, T)                       rotation matrix + translation vector
- PosEuler : [x, y, z, rx, ry, rz]        position + Euler
- PosQuat  : [x, y, z, qx, qy, qz, qw]    position + quaternion
- PosAxis  : [x, y, z, ax, ay, az]        position + axis-angle

------------------------
# Batch handling
------------------------
Most functions accept both single inputs (shape (...,)) and batched inputs
(shape (N, ...)), by adding/removing a batch dimension internally.
If you add new helpers, follow the same pattern.

------------------------
# ⚠️ Quaternion convention
------------------------
There are two common quaternion orderings:
- wxyz  (scalar first)  → used by transforms3d
- xyzw  (scalar last)   → used by SciPy
This file follows SciPy, so ALL quaternions are assumed to be [x, y, z, w].
If your source is wxyz, reorder before calling these functions.
"""

import numpy as np
from numpy import radians
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
# ------------------------------------------------------------
# region first layer
# ------------------------------------------------------------


def quat2axis(quats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Converts a batch of quaternions to axis-angle (rotation vector) format.
    Each output vector is the unit rotation axis scaled by the rotation angle (in radians).

    Args:
        quats (np.ndarray): Array of shape (N,4), each row = [x, y, z, w].
        eps (float): Small threshold to guard against divide-by-zero.

    Returns:
        np.ndarray: Array of shape (N,3), axis-angle vectors.
    """
    unsqueezed = False
    if quats.ndim == 1:
        unsqueezed = True
        quats = quats[None, :]
    # Ensure float array
    q = quats.astype(np.float64)

    # Clip w to valid range to avoid numerical issues with arccos
    w = np.clip(q[:, 3], -1.0, 1.0)

    # Compute angle: theta = 2 * arccos(w)
    theta = 2.0 * np.arccos(w)

    # Compute denominator = sin(theta/2) = sqrt(1 - w^2)
    den = np.sqrt(1.0 - w * w)

    # For very small angles, sin(theta/2) ~ 0 => axis is arbitrary; produce zero vector
    # So we avoid division by zero by clamping denominator and masking later
    inv_den = np.zeros_like(den)
    nonzero = den > eps
    inv_den[nonzero] = 1.0 / den[nonzero]

    # Compute axis components
    axes = q[:, :3] * inv_den[:, None]

    # Multiply axis by angle to get rotation vector
    rotvec = axes * theta[:, None]

    # For entries with den <= eps (i.e. near zero rotation), force rotvec = 0
    rotvec[~nonzero] = 0.0

    if unsqueezed:
        rotvec = rotvec[0, :]
    return rotvec


def quat2euler(q):
    return R.from_quat(q).as_euler("XYZ")


def quat2mat(q):
    return R.from_quat(q).as_matrix()


def axis2quat(rotvecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Converts a batch of axis-angle (rotation vector) inputs to quaternions.
    Each input row is a rotation vector (axis * angle), shape (N,3).
    Returns an array of quaternions shape (N,4) in [x, y, z, w] format.

    Args:
        rotvecs (np.ndarray): Array of shape (N,3), each row = axis * angle (radians).
        eps (float): Threshold below which we treat the rotation as zero.

    Returns:
        np.ndarray: Array of shape (N,4), quaternions [x, y, z, w].
    """
    unsqueezed = False

    # Compute rotation angles
    if rotvecs.ndim == 1:
        unsqueezed = True
        rotvecs = rotvecs[None, :]
    theta = np.linalg.norm(rotvecs, axis=1)  # (N,)
    half_theta = 0.5 * theta  # (N,)

    # Prepare output
    quats = np.zeros((rotvecs.shape[0], 4), dtype=np.float64)

    # For nonzero rotations, normalize axis
    nonzero = theta > eps
    axes = np.zeros_like(rotvecs)
    axes[nonzero] = rotvecs[nonzero] / theta[nonzero, None]

    # Compute sin and cos of half-angle
    sin_ht = np.sin(half_theta)  # (N,)
    cos_ht = np.cos(half_theta)  # (N,)

    # Fill quaternion: q = [axis*sin(theta/2), cos(theta/2)]
    quats[:, :3] = axes * sin_ht[:, None]
    quats[:, 3] = cos_ht

    # For very small angles, ensure quaternion is identity [0,0,0,1]
    quats[~nonzero, :] = np.array([0.0, 0.0, 0.0, 1.0])

    if unsqueezed:
        quats = quats[0, :]
    return quats


def euler2quat(euler):
    return R.from_euler("XYZ", euler).as_quat()


def mat2quat(mat):
    return R.from_matrix(mat).as_quat()


def mat2euler(mat):
    return R.from_matrix(mat).as_euler("XYZ")


def euler2mat(euler):
    return R.from_euler("XYZ", euler).as_matrix()


# ------------------------------------------------------------
# region second layer
# ------------------------------------------------------------


def RT2HT(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    assert R.shape[:-2] == T.shape[:-1], "R and T must have the same batch size"
    head_shape = R.shape[:-2]
    HT = np.zeros((*head_shape, 4, 4), dtype=R.dtype)
    HT[..., :3, :3] = R
    HT[..., :3, 3] = T
    HT[..., 3, 3] = 1.0
    return HT


def HT2PosEuler(T: np.ndarray, degrees=False) -> np.ndarray:
    out = np.hstack((T[..., :3, 3], mat2euler(T[..., :3, :3])))
    # radian to degree
    if degrees:
        out[..., 3:] = np.degrees(out[..., 3:])

    return out


def PosEuler2HT(PosEuler, degrees=False):
    HT = RT2HT(euler2mat(radians(PosEuler[..., 3:])), PosEuler[..., :3])
    return HT


def PosQuat2HT(PosQuat):
    HT = RT2HT(quat2mat(PosQuat[..., 3:]), PosQuat[..., :3])
    return HT


def HT2PosQuat(T):
    out = np.hstack((T[..., :3, 3], mat2quat(T[..., :3, :3])))
    return out


def PosAxis2PosQuat(PosAxis: np.ndarray) -> np.ndarray:
    Pos = PosAxis[..., :3]
    Axis = PosAxis[..., 3:6]

    quat = axis2quat(Axis)
    PosQuat = np.concatenate([Pos, quat], axis=-1)
    return PosQuat


def PosQuat2PosAxis(PosQuat: np.ndarray) -> np.ndarray:
    if PosQuat.ndim == 1:
        PosQuat = PosQuat[None, :]
    Pos = PosQuat[..., :3]
    quat = PosQuat[..., 3:7]

    Axis = quat2axis(quat)
    PosAxis = np.concatenate([Pos, Axis], axis=-1)
    return PosAxis


def HT2PosAxis(T: np.ndarray) -> np.ndarray:
    PosQuat = HT2PosQuat(T)
    PosAxis = PosQuat2PosAxis(PosQuat)
    return PosAxis


def PosAxis2HT(PosAxis: np.ndarray) -> np.ndarray:
    PosQuat = PosAxis2PosQuat(PosAxis)
    HT = PosQuat2HT(PosQuat)
    return HT


# endregion

if __name__ == "__main__":
    # test quat2euler
    euler = np.array(
        [[0, 0, 0], [15, 0, 0], [45, 0, 0]],
    )
    quat = euler2quat(np.radians(euler))
    print("quat", quat)
    axis = quat2axis(quat)
    print("axis", axis)
    quat = axis2quat(axis)
    print("quat", quat)

    test_posaxis = np.array([1, 2, 3, 0.1, 0.2, 0.3])
    test_posquat = PosAxis2PosQuat(test_posaxis)
    print("test_posquat", test_posquat)
