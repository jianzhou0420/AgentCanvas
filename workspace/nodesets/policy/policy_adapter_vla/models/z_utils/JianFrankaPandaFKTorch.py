from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch import cos, sin
from torch import deg2rad as radians

from workspace.nodesets.policy.policy_adapter_vla.models.z_utils.JianRotationTorch import HT2eePose


def tensorfp32(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    else:
        raise TypeError(
            tensorfp32.__name__ + " only supports list, numpy.ndarray, or torch.Tensor types."
        )


class FrankaEmikaPanda_torch(nn.Module):
    """
    TODO: support batch process
    """

    def __init__(self):
        """
        PoseEuler: [x, y, z, Rx, Ry, Rz]
        """
        super().__init__()

        def rb(name, val):
            return self.register_buffer(name, val)  # 这一步太天才了

        d = tensorfp32(
            [
                0.333,
                0,
                0.316,
                0,
                0.384,
                0,
                0,
            ]
        )
        a = tensorfp32(
            [
                0,
                0,
                0,
                0.0825,
                -0.0825,
                0,
                0.088,
            ]
        )
        alpha = tensorfp32(
            [
                0,
                -math.pi / 2,
                math.pi / 2,
                math.pi / 2,
                -math.pi / 2,
                math.pi / 2,
                math.pi / 2,
            ]
        )

        PosEuler_ik = tensorfp32(
            [
                [-0.0001, -0.0347, -0.0752, -162.8693, 0.0033, 0.2122],
                [0.0, -0.0766, 0.0344, -72.9831, 0.2349, -178.696],
                [0.0333, 0.0266, -0.0412, -23.1133, 36.1784, -73.6655],
                [-0.0495, 0.0425, 0.0267, 67.4364, 35.2228, 105.8928],
                [-0.0012, 0.043, -0.109, -14.2746, -0.5962, -90.9492],
                [0.0425, 0.0152, 0.01, 2.82, -77.4572, 2.8245],
                [0.0136, 0.0117, 0.0787, 89.277, -45.0221, 88.992],
            ]
        )  # frame to link, constant

        bbox_link = tensorfp32(
            [
                [-0.05506498, 0.05506498, -0.07104017, 0.07104017, -0.13778356, 0.13778356],
                [-0.05553254, 0.05553254, -0.07103211, 0.07103211, -0.13867143, 0.13867143],
                [-0.06274896, 0.06274896, -0.06585597, 0.06585597, -0.12418943, 0.12418943],
                [-0.0627161, 0.0627161, -0.06663913, 0.06663913, -0.12511621, 0.12511621],
                [-0.06477058, 0.06477058, -0.05536535, 0.05536535, -0.17003113, 0.17003113],
                [-0.04413896, 0.04413896, -0.06654218, 0.06654218, -0.09111023, 0.09111023],
                [-0.02742341, 0.02742341, -0.04388235, 0.04388235, -0.07091156, 0.07091156],
            ]
        )

        bbox_other = tensorfp32(
            [
                [-0.0700, 0.0700, -0.0936, 0.0936, -0.1128, 0.1128],  # link base
                [-0.0314, 0.0314, -0.0459, 0.0459, -0.1023, 0.1023],  # gripper
                [-0.0121, 0.0121, -0.0105, 0.0105, -0.0278, 0.0278],  # left finger
                [-0.0120, 0.0120, -0.0105, 0.0105, -0.0277, 0.0277],  # right finger
            ]
        )

        PosEuler_ik_gripper_close = tensorfp32(
            [
                [0.0000, 0.0000, 0.1261, -89.9259, 45.0282, -179.1246],
                [0.0106, 0.0100, 0.1913, 3.4935, -3.589, 45.154],
                [-0.0076, -0.0084, 0.1913, 177.056, -2.8024, 134.9147],
            ]
        )

        PosEuler_ik_gripper_open = tensorfp32(
            [
                [0.0042, -0.0003, 0.1230, -89.9259, 45.0282, -179.1246],
                [0.044, 0.0380, 0.1913, 3.4935, -3.589, 45.154],
                [-0.03, -0.036, 0.1913, 177.056, -2.8024, 134.9147],
            ]
        )

        T_base = tensorfp32(
            [[1, 0, 0, -0.2677189], [0, 1, 0, -0.00628856], [0, 0, 1, 0.74968816], [0, 0, 0, 1]]
        )

        JP_offset = tensorfp32([0, 0, 0, radians(torch.tensor(-4)), 0, 0, 0, 0])  # link7 open1

        bbox_link_half = bbox_link[:, 1::2]

        T_last2eePose = tensorfp32(
            [
                [-0.7073, -0.7069, -0.0006, 0.0005],
                [0.7069, -0.7073, -0.0001, 0.0008],
                [-0.0004, -0.0005, 1.0000, 0.2174],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ]
        )  # TODO: refine this

        rb("d", d)
        rb("a", a)
        rb("alpha", alpha)
        rb("PosEuler_ik", PosEuler_ik)
        rb("bbox_link", bbox_link)
        rb("bbox_other", bbox_other)
        rb("PosEuler_ik_gripper_close", PosEuler_ik_gripper_close)
        rb("PosEuler_ik_gripper_open", PosEuler_ik_gripper_open)
        rb("T_base", T_base)
        rb("T_offset", None)
        rb("JP_offset", JP_offset)
        rb("bbox_link_half", bbox_link_half)
        rb("T_last2eePose", T_last2eePose)

    @staticmethod
    def dh_modified_transform(DH):
        alpha, a, theta, d = DH[..., 0, :], DH[..., 1, :], DH[..., 2, :], DH[..., 3, :]
        ct = cos(theta)
        st = sin(theta)

        ca = cos(alpha)
        sa = sin(alpha)

        t11 = ct
        t12 = -st
        t13 = torch.zeros_like(t11)
        t14 = a

        t21 = st * ca
        t22 = ct * ca
        t23 = -sa
        t24 = -d * sa

        t31 = st * sa
        t32 = ct * sa
        t33 = ca
        t34 = d * ca

        t41 = torch.zeros_like(t11)
        t42 = torch.zeros_like(t11)
        t43 = torch.zeros_like(t11)
        t44 = torch.ones_like(t11)

        row1 = torch.stack([t11, t12, t13, t14], dim=-1)
        row2 = torch.stack([t21, t22, t23, t24], dim=-1)
        row3 = torch.stack([t31, t32, t33, t34], dim=-1)
        row4 = torch.stack([t41, t42, t43, t44], dim=-1)
        T = torch.stack([row1, row2, row3, row4], dim=-2)

        return T

    def get_T_oi(self, theta):
        d = self.d
        a = self.a
        alpha = self.alpha

        other_shape = theta.shape[:-1]
        num_frame = theta.shape[-1]
        d = d.repeat(other_shape + (1,))
        a = a.repeat(other_shape + (1,))
        alpha = alpha.repeat(other_shape + (1,))
        DH = torch.stack([alpha, a, theta, d], dim=-2)

        T_i1_i = self.dh_modified_transform(DH)

        T_base = self.T_base.repeat(other_shape + (1, 1))

        T_cumulative = T_base.clone()
        T_oi = []
        for i in range(num_frame):
            T_cumulative = T_cumulative @ T_i1_i[..., i, :, :]
            T_oi.append(T_cumulative)
        T_oi = torch.stack(T_oi, dim=-3)
        return T_i1_i, T_oi

    def theta2PosQuat(self, theta, apply_offset: bool):  # assume no open
        T_eePose = self.theta2HT(theta, apply_offset)
        eePose = HT2eePose(T_eePose)
        return eePose

    def theta2HT(self, theta, apply_offset: bool):
        _, T_oi = self.get_T_oi(theta)
        other_shape = theta.shape[:-1]
        T_oi_last = T_oi[..., -1, :, :]
        T_last2eePose = self.T_last2eePose.repeat(other_shape + (1, 1))
        T_eePose = T_oi_last @ T_last2eePose
        if apply_offset:
            assert hasattr(self, "T_offset"), "T_offset must be set before applying offset."
            T_offset = self.T_offset.repeat(other_shape + (1, 1))
            T_eePose = T_eePose @ T_offset
        return T_eePose

    def set_T_base(self, T_base):
        """
        Set the base transformation matrix.
        :param T_base: The base transformation matrix.
        """
        self.T_base = tensorfp32(T_base)

    def set_T_offset(self, T_offset):
        """
        Set the offset transformation matrix.
        :param T_offset: The offset transformation matrix.
        """
        self.T_offset = tensorfp32(T_offset)

    def verify_data_integrity(self, eePose, JP):
        return NotImplementedError


if __name__ == "__main__":

    def test():
        from scripts.expForwardKinematics.ReconLoss.FrankaPandaFK import FrankaEmikaPanda

        franka_np = FrankaEmikaPanda()
        franka_torch = FrankaEmikaPanda_torch()
        theta = torch.randn(8, 7)
        for i in range(8):
            eePose_np = franka_np.theta2eePose(theta[i, :])
            print(eePose_np)
        eePose_torch = franka_torch.theta2PosQuat(theta)
        print(eePose_torch)
        theta_1 = theta.repeat(8, 1, 1)
        eePose_torch_1 = franka_torch.theta2PosQuat(theta_1)
        print(eePose_torch_1[-1])
        print(eePose_torch_1[5])

    test()
    # theta = np.array([-
