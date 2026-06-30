"""
all the normalizer below are for datasetwise computation
not for CUDA computation so assume all the input are numpy
"""

from __future__ import annotations

import json

import einops
import numpy as np
import torch

from workspace.nodesets.policy.policy_vla.models.z_utils.JianRotation import *  # using scipy where xyzw is the default quat order compatible with rlbench

# region 1. rotation


def normalize_quat2euler(rot: np.ndarray) -> np.ndarray:
    assert isinstance(rot, np.ndarray), "Input must be a numpy array"
    assert rot.shape[-1] == 4, "Input must be a quaternion tensor with last dimension of size 4"
    assert len(rot.shape) == 3, "Input must be a 3D tensor with shape (B, H, D)"
    B, H, D = rot.shape

    rot = einops.rearrange(rot, "b h d -> (b h) d")
    rot = quat2euler(rot) / 3.15
    rot = einops.rearrange(rot, "(b h) d -> b h d", b=B, h=H)
    return rot


def denormalize_quat2euler(rot: np.ndarray) -> np.ndarray:
    assert isinstance(rot, np.ndarray), "Input must be a torch tensor"
    assert rot.shape[-1] == 3, (
        "Input must be a normalized euler tensor with last dimension of size 3"
    )
    assert len(rot.shape) == 3, "Input must be a 3D tensor with shape (B, H, D)"

    B, H, D = rot.shape
    rot = rot * 3.15
    rot = einops.rearrange(rot, "b h d -> (b h) d")
    rot = euler2quat(rot)
    rot = einops.rearrange(rot, "(b h) d -> b h d", b=B, h=H)
    return rot


def quat2ortho6D(quat: np.ndarray) -> np.ndarray:
    assert isinstance(quat, np.ndarray), "Input must be a numpy array"
    assert quat.shape[-1] == 4, "Input must be a quaternion tensor with last dimension of size 4"
    assert len(quat.shape) == 3, "Input must be a 3D tensor with shape (B, H, D)"
    B, H, D = quat.shape
    quat = einops.rearrange(quat, "b h d -> (b h) d")
    matrix = quat2mat(quat)
    ortho6d = Ortho6D_numpy.get_ortho6d_from_rotation_matrix(matrix)
    ortho6d = einops.rearrange(ortho6d, "(b h) d -> b h d", b=B, h=H)
    return ortho6d


def ortho6d2quat(ortho6d: np.ndarray) -> np.ndarray:
    assert isinstance(ortho6d, np.ndarray), "Input must be a numpy array"
    assert ortho6d.shape[-1] == 6, "Input must be a ortho6d tensor with last dimension of size 6"
    assert len(ortho6d.shape) == 3, "Input must be a 3D tensor with shape (B, H, D)"
    B, H, D = ortho6d.shape
    ortho6d = einops.rearrange(ortho6d, "b h d -> (b h) d")
    matrix = Ortho6D_numpy.compute_rotation_matrix_from_ortho6d(ortho6d)
    quat = mat2quat(matrix)
    quat = einops.rearrange(quat, "(b h) d -> b h d", b=B, h=H)
    return quat


class Ortho6D_torch:
    """
    this class is just for code organization

    compute rotation matrix from ortho6d, cross product, normalize vector are
    copied from https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py

    """

    @staticmethod
    def compute_rotation_matrix_from_ortho6d(ortho6d):
        x_raw = ortho6d[:, 0:3]  # batch*3
        y_raw = ortho6d[:, 3:6]  # batch*3

        x = Ortho6D_torch.normalize_vector(x_raw)  # batch*3
        z = Ortho6D_torch.cross_product(x, y_raw)  # batch*3
        z = Ortho6D_torch.normalize_vector(z)  # batch*3
        y = Ortho6D_torch.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        return matrix

    @staticmethod
    def cross_product(u, v):
        batch = u.shape[0]
        # print (u.shape)
        # print (v.shape)
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
        return out

    @staticmethod
    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(
            v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8], device=v.device))
        )
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if return_mag == True:
            return v, v_mag[:, 0]
        else:
            return v

    @staticmethod  # modi
    def get_ortho6d_from_rotation_matrix(matrix):
        # The orhto6d represents the first two column vectors a1 and a2 of the
        # rotation matrix: [ | , |,  | ]
        #                  [ a1, a2, a3]
        #                  [ | , |,  | ]
        if isinstance(matrix, torch.Tensor):
            ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
        elif isinstance(matrix, np.ndarray):
            ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
        else:
            raise TypeError("Input must be a torch tensor or numpy array")
        return ortho6d


class Ortho6D_numpy:
    @staticmethod
    def normalize_vector(v, return_mag=False):
        v = np.asarray(v, dtype=np.float32)
        # 计算每行向量的 L2 范数
        v_mag = np.linalg.norm(v, axis=1)
        # 防止除以 0
        v_mag_safe = np.maximum(v_mag, 1e-8)
        v_norm = v / v_mag_safe[:, None]
        if return_mag:
            return v_norm, v_mag
        return v_norm

    @staticmethod
    def cross_product(u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        out = np.stack(
            [
                u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
            ],
            axis=1,
        )
        return out

    @staticmethod
    def compute_rotation_matrix_from_ortho6d(ortho6d):
        ortho6d = np.asarray(ortho6d, dtype=np.float32)
        x_raw = ortho6d[:, 0:3]  # (B,3)
        y_raw = ortho6d[:, 3:6]  # (B,3)

        # Gram–Schmidt
        x = Ortho6D_numpy.normalize_vector(x_raw)  # (B,3)
        z = Ortho6D_numpy.cross_product(x, y_raw)  # (B,3)
        z = Ortho6D_numpy.normalize_vector(z)  # (B,3)
        y = Ortho6D_numpy.cross_product(z, x)  # (B,3)

        x = x.reshape(-1, 3, 1)  # (B,3,1)
        y = y.reshape(-1, 3, 1)  # (B,3,1)
        z = z.reshape(-1, 3, 1)  # (B,3,1)
        matrix = np.concatenate((x, y, z), axis=2)  # (B,3,3)
        return matrix

    @staticmethod
    def get_ortho6d_from_rotation_matrix(matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        # transpose to makesure [r11,r21,r31, r12,r22,r32]
        ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
        return ortho6d


# region 2. joint position
# JP

JOINT_POSITION_LIMITS = [
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1],
]


def normalize_JP(JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        normalized_JP = 2 * (JP - lower_np) / (upper_np - lower_np) - 1

    elif isinstance(JP, torch.Tensor):
        lower_tensor = torch.tensor(lower, device=JP.device)
        upper_tensor = torch.tensor(upper, device=JP.device)
        # broadcasting
        normalized_JP = 2 * (JP - lower_tensor) / (upper_tensor - lower_tensor) - 1
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return normalized_JP


def denormalize_JP(norm_JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(norm_JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        JP = lower_np + (norm_JP + 1) / 2 * (upper_np - lower_np)

    elif isinstance(norm_JP, torch.Tensor):
        lower_t = torch.tensor(lower, device=norm_JP.device)
        upper_t = torch.tensor(upper, device=norm_JP.device)
        # broadcasting
        JP = lower_t + (norm_JP + 1) / 2 * (upper_t - lower_t)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return JP


# Pos


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: str | None = None):
    gripper_loc_bounds = json.load(open(path))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path))
        gripper_loc_bounds_min = (
            np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        )
        gripper_loc_bounds_max = (
            np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        )
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


gripper_loc_bounds = get_gripper_loc_bounds(
    "./assets/18_peract_tasks_location_bounds.json",
    buffer=0.04,
)

print("Gripper location bounds:", gripper_loc_bounds)


def normalize_pos(pos):
    pos_min = gripper_loc_bounds[0]
    pos_max = gripper_loc_bounds[1]
    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0


def denormalize_pos(pos):
    pos_min = gripper_loc_bounds[0]
    pos_max = gripper_loc_bounds[1]
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


if __name__ == "__main__":

    def validate_ortho6d():
        """
        验证 Ortho6D 的实现是否正确。
        1. 随机生成旋转矩阵
        2. 提取 Ortho6D
        3. 从 Ortho6D 恢复旋转矩阵
        4. 检查恢复的旋转矩阵与原始旋转矩阵的差异
        """

        # 设置随机种子以确保可重复性
        def random_rotation_matrix(batch):
            """生成一批均匀分布的旋转矩阵 (B,3,3)"""
            # 使用 QR 分解法
            A = np.random.randn(batch, 3, 3).astype(np.float32)
            Q, R = np.linalg.qr(A)
            # 确保右手系，determinant positive
            det = np.linalg.det(Q)
            Q[det < 0] *= -1
            return Q

        batch_size = 10
        np.random.seed(42)

        # Generate random ortho6d inputs
        ortho6d_np = np.random.randn(batch_size, 6).astype(np.float32)
        ortho6d_torch = torch.from_numpy(ortho6d_np)

        # Compute rotation matrices
        R_torch = Ortho6D_torch.compute_rotation_matrix_from_ortho6d(ortho6d_torch).cpu().numpy()
        R_np = Ortho6D_numpy.compute_rotation_matrix_from_ortho6d(ortho6d_np)

        # Compare rotation matrices
        diff_R = np.abs(R_torch - R_np)
        print("Max difference in rotation matrices:", diff_R.max())

        # Extract ortho6d back from rotation matrices
        ortho6d_torch_back = Ortho6D_torch.get_ortho6d_from_rotation_matrix(
            torch.from_numpy(R_np)
        ).numpy()
        ortho6d_np_back = Ortho6D_numpy.get_ortho6d_from_rotation_matrix(R_np)

        # Compare ortho6d vectors
        diff_o6d = np.abs(ortho6d_torch_back - ortho6d_np_back)
        print("Max difference in ortho6d vectors:", diff_o6d.max())
