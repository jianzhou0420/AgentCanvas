"""
---------------------------------------------------------------------------
Jian's Rotation Utilities for PyTorch v1.0

A thin PyTorch-friendly wrapper around rotation/pose conversions, adapted from:
- PyTorch3D rotation conversions:
  https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
- 3d-diffuser-actor utilities:
  https://github.com/nickgkan/3d_diffuser_actor/blob/master/utils/pytorch3d_transforms.py

This file keeps the same math as PyTorch3D (quaternions are **real part first** = [w, x, y, z]),
but the “My Own API” section exposes functions with the more robotics-style order
**[x, y, z, w]** by default, and converts internally.

------------------------
Representations
------------------------
- quat_xyzw   : [x, y, z, w]  quaternion, scalar last (default for My Own API)
- quat_wxyz   : [w, x, y, z]  quaternion, scalar first (default for PyTorch3D parts)
- euler_XYZ   : [rx, ry, rz]  Euler angles in 'XYZ' (rotate X → Y → Z), radians unless noted
- matrix      : [3 x 3]       rotation matrix
- axis-angle  : [ax, ay, az]  rotation vector = axis * angle (radians)
- HT          : [4 x 4]       homogeneous transform (rotation + translation)
- eePose     : [x, y, z, qx, qy, qz, qw] end-effector pose (position + quaternion)

------------------------
My Own API (torch)
------------------------
These are the user-facing helpers that default to quat = [x, y, z, w], Euler = XYZ:
- quat2euler(q, order='xyzw')     → (..., 3) Euler XYZ, radians
- quat2mat(q, order='xyzw')       → (..., 3, 3)
- euler2quat(e, order='xyzw')     → (..., 4) [x, y, z, w]
- euler2mat(e)                    → (..., 3, 3)
- mat2quat(R, order='xyzw')       → (..., 4) [x, y, z, w]
- mat2euler(R)                    → (..., 3) Euler XYZ, radians
- RT2HT(R, T)                     → (..., 4, 4)
- HT2PosEuler(HT)                 → (..., 6) [x, y, z, roll, pitch, yaw] in degrees
- PosEuler2HT(pos_euler)          → (..., 4, 4) from [x, y, z, roll, pitch, yaw] in degrees
- eePose2HT(ee, order='xyzw')     → (..., 4, 4) from [x, y, z, qx, qy, qz, qw]
- HT2eePose(HT, order='xyzw')     → (..., 7)  to  [x, y, z, qx, qy, qz, qw]
- to_wxyz(q, order) / to_order(q, order) to swap quaternion ordering

Everything above accepts batched tensors: shapes like (N, 4), (N, 3, 3), (N, 4, 4), etc.

------------------------
Quaternion ordering
------------------------
There are two orders in this file:
1. My API (external): [x, y, z, w]  → pass order='xyzw'
2. PyTorch3D (internal): [w, x, y, z]

The helpers `to_wxyz(...)` and `to_order(...)` convert between them.
If you feed in quaternions from a different source, check the order first.

------------------------
Column vs row vectors
------------------------
Rotation matrices here follow the PyTorch3D convention: they are meant to act on
column vectors by post-multiplication. If your data is row-vectors, transpose R first.

------------------------
Batch handling
------------------------
All functions are written to broadcast over leading dimensions, e.g. (..., 4),
(..., 3, 3), (..., 4, 4). You can pass a single pose or a whole batch; PyTorch
broadcast rules will typically handle it.
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

Device = Union[str, torch.device]


# My Own API, default quat:xyzw and Euler:XYZ

# ------------------------------------------------------------
# region My own API
# ------------------------------------------------------------
# default is [x,y,z,w]


def quat2euler(q: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    q: (...,4) in [x,y,z,w] order, radians
    returns: (...,3) Euler angles XYZ, radians
    """
    q_wxyz = to_wxyz(q, order=order)  # (...,4) [w,x,y,z]
    R = quaternion_to_matrix(q_wxyz)  # (...,3,3)
    eulers = matrix_to_euler_angles(R, convention="XYZ")  # (...,3)
    return eulers


def quat2mat(q: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    q: (...,4) [x,y,z,w]
    returns: (...,3,3) rotation matrix
    """
    q_wxyz = to_wxyz(q, order=order)  # (...,4) [w,x,y,z]
    return quaternion_to_matrix(q_wxyz)


def euler2quat(e: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    e: (...,3) Euler angles XYZ, radians
    returns: (...,4) [x,y,z,w]
    """
    R = euler_angles_to_matrix(e, convention="XYZ")  # (...,3,3)
    q_wxyz = matrix_to_quaternion(R)  # (...,4) [w,x,y,z]
    quat = to_order(q_wxyz, order=order)  # (...,4) [x,y,z,w]
    return quat


def euler2mat(e: torch.Tensor) -> torch.Tensor:
    """
    e: (...,3) Euler angles XYZ, radians
    returns: (...,3,3)
    """
    return euler_angles_to_matrix(e, convention="XYZ")


def mat2quat(R: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,4) [x,y,z,w]
    """
    q_wxyz = matrix_to_quaternion(R)
    quat = to_order(q_wxyz, order=order)  # (...,4) [x,y,z,w]
    return quat


def mat2euler(R: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,3) Euler angles XYZ, radians
    """
    return matrix_to_euler_angles(R, convention="XYZ")


def RT2HT(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3), T: (...,3)
    returns: (...,4,4) homogeneous transform
    """
    shape = R.shape[:-2]
    HT = torch.eye(4, device=R.device, dtype=R.dtype).expand(*shape, 4, 4).clone()
    HT[..., :3, :3] = R
    HT[..., :3, 3] = T
    return HT


def HT2PosEuler(HT: torch.Tensor) -> torch.Tensor:
    """
    HT: (...,4,4)
    out: (...,6) [x,y,z,  α,β,γ] 其中 α,β,γ 单位为 degree
    """
    t = HT[..., :3, 3]
    R = HT[..., :3, :3]
    e = mat2euler(R)  # radians
    e_deg = torch.rad2deg(e)
    return torch.cat([t, e_deg], dim=-1)


def PosEuler2HT(pos_euler: torch.Tensor) -> torch.Tensor:
    """
    pos_euler: (...,6) [x,y,z, α,β,γ] degree
    返回 (...,4,4)
    """
    t = pos_euler[..., :3]
    e_rad = torch.deg2rad(pos_euler[..., 3:])
    R = euler2mat(e_rad)
    return RT2HT(R, t)


def eePose2HT(ee: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    ee: (...,7) [x,y,z, qx,qy,qz,qw]
    返回 (...,4,4)
    """
    t = ee[..., :3]
    q = ee[..., 3:]
    R = quat2mat(q, order=order)
    return RT2HT(R, t)


def HT2eePose(HT: torch.Tensor, order="xyzw") -> torch.Tensor:
    """
    HT: (...,4,4)
    返回 (...,7) [x,y,z, qx,qy,qz,qw]
    """
    t = HT[..., :3, 3]
    R = HT[..., :3, :3]
    q = mat2quat(R, order=order)  # (...,4) [x,y,z,w]
    return torch.cat([t, q], dim=-1)


def to_wxyz(q: torch.Tensor, order: str) -> torch.Tensor:
    if order == "xyzw":
        q_wxyz = _xyzw2wxyz(q)  # (...,4) [w,x,y,z]
    elif order == "wxyz":
        q_wxyz = q
    else:
        raise ValueError(f"Invalid order {order}.")
    return q_wxyz


def _wxyz2xyzw(q):
    """
    Convert quaternion from [x, y, z, w] to [w, x, y, z] format.
    """
    return torch.cat([q[..., 1:], q[..., 0:1]], dim=-1)


def _xyzw2wxyz(q):
    """
    Convert quaternion from [w, x, y, z] to [x, y, z, w] format.
    """
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)


def to_order(q: torch.Tensor, order: str) -> torch.Tensor:
    """
    receive [w,x,y,z] to required order
    """
    if order == "xyzw":
        q_order = _wxyz2xyzw(q)  # (...,4) [x,y,z,w]
    elif order == "wxyz":
        q_order = q
    else:
        raise ValueError(f"Invalid order {order}.")
    return q_order


# region: Copied from PyTorch3D,https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    return standardize_quaternion(out)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1), strict=False)
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def random_rotation(dtype: torch.dtype | None = None, device: Device | None = None) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device)[0]


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if not fast:
        return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


def matrix_to_axis_angle(matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if not fast:
        return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = 0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :] + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)
    # angles/2 are between [-pi/2, pi/2], thus sin_half_angles_over_angles
    # can't be zero
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


# endregion: Copied from PyTorch3D,


if __name__ == "__main__":

    def test_quat_euler_matrix():
        import torch

        # 示例数据
        q_wxyz = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # quaternion [x, y, z, w]
        euler_angles = torch.tensor(
            [[0.1, 0.2, 0.3]]
        )  # Euler angles [alpha, beta, gamma] in radians
        R_matrix = torch.tensor(
            [[[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]]]
        )  # Rotation matrix

        # 使用 scipy 进行转换

        def quat2euler_scipy(q):
            q_xyzw = _wxyz2xyzw(q)  # Convert to [x, y, z, w] order
            r = R.from_quat(q_xyzw.numpy())  # Convert quaternion to scipy Rotation object
            return r.as_euler("XYZ")

        def euler2quat_scipy(e):
            r = R.from_euler("XYZ", e.numpy())  # Convert Euler angles to scipy quaternion
            quat = r.as_quat()  # Get quaternion in [x, y, z, w] order
            # q_wxyz = _xyzw2wxyz(torch.tensor(quat))
            return quat

        def quat2mat_scipy(q):
            q_xyzw = _wxyz2xyzw(q)
            r = R.from_quat(q_xyzw.numpy())
            return torch.tensor(r.as_matrix())  # Convert to torch tensor

        def mat2quat_scipy(R_matrix):
            r = R.from_matrix(R_matrix.numpy())
            # q_wxyz = _xyzw2wxyz(torch.tensor(r.as_quat()))
            return r.as_quat()

        def quat2euler_torch(q):
            return quat2euler(q, order="wxyz")

        def euler2quat_torch(e):
            return mat2quat(euler2mat(e))

        def quat2mat_torch(q):
            return quat2mat(q, order="wxyz")  # Assuming quaternion_to_matrix is defined correctly

        def mat2quat_torch(R_matrix):
            return mat2quat(R_matrix)  # Assuming matrix_to_quaternion is defined correctly

        # 测试所有函数
        print("Testing quat2euler:")
        print("Scipy Result:", quat2euler_scipy(q_wxyz))
        print("Torch Result:", quat2euler_torch(q_wxyz))

        print("\nTesting euler2quat:")
        print("Scipy Result:", euler2quat_scipy(euler_angles))
        print("Torch Result:", euler2quat_torch(euler_angles))

        print("\nTesting quat2mat:")
        print("Scipy Result:", quat2mat_scipy(q_wxyz))
        print("Torch Result:", quat2mat_torch(q_wxyz))

        print("\nTesting mat2quat:")
        print("Scipy Result:", mat2quat_scipy(R_matrix))
        print("Torch Result:", mat2quat_torch(R_matrix))

    # numpy / scipy prints
    np.set_printoptions(precision=4, suppress=True)

    # torch prints
    torch.set_printoptions(precision=4, sci_mode=False)
    test_quat_euler_matrix()
