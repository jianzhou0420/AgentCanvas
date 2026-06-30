from __future__ import annotations

import math
from copy import deepcopy as copy
from math import cos, sin

import numpy as np
import open3d as o3d

from workspace.nodesets.policy.policy_vla.models.z_utils.JianRotation import (
    RT2HT,
    HT2PosQuat,
    PosEuler2HT,
    euler2mat,
)


class FrankaEmikaPanda:
    """
    TODO: support batch process
    """

    def __init__(self):
        """
        PoseEuler: [x, y, z, Rx, Ry, Rz]
        """
        self.d = [
            0.333,
            0,
            0.316,
            0,
            0.384,
            0,
            0,
        ]
        self.a = [
            0,
            0,
            0,
            0.0825,
            -0.0825,
            0,
            0.088,
        ]
        self.alpha = [
            0,
            -math.pi / 2,
            math.pi / 2,
            math.pi / 2,
            -math.pi / 2,
            math.pi / 2,
            math.pi / 2,
        ]

        self.PosEuler_ik = np.array(
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

        self.bbox_link = np.array(
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

        self.bbox_other = np.array(
            [
                [-0.0700, 0.0700, -0.0936, 0.0936, -0.1128, 0.1128],  # link base
                [-0.0314, 0.0314, -0.0459, 0.0459, -0.1023, 0.1023],  # gripper
                [-0.0121, 0.0121, -0.0105, 0.0105, -0.0278, 0.0278],  # left finger
                [-0.0120, 0.0120, -0.0105, 0.0105, -0.0277, 0.0277],  # right finger
            ]
        )

        self.PosEuler_ik_gripper_close = np.array(
            [
                [0.0000, 0.0000, 0.1261, -89.9259, 45.0282, -179.1246],
                [0.0106, 0.0100, 0.1913, 3.4935, -3.589, 45.154],
                [-0.0076, -0.0084, 0.1913, 177.056, -2.8024, 134.9147],
            ]
        )

        self.PosEuler_ik_gripper_open = np.array(
            [
                [0.0042, -0.0003, 0.1230, -89.9259, 45.0282, -179.1246],
                [0.044, 0.0380, 0.1913, 3.4935, -3.589, 45.154],
                [-0.03, -0.036, 0.1913, 177.056, -2.8024, 134.9147],
            ]
        )

        self.JP_offset = np.array([0, 0, 0, np.radians(-4), 0, 0, 0, 0])  # link7 open1

        self.bbox_link_half = self.bbox_link[:, 1::2]

        self.T_last2PosQuat = np.array(
            [
                [-0.7073, -0.7069, -0.0006, 0.0005],
                [0.7069, -0.7073, -0.0001, 0.0008],
                [-0.0004, -0.0005, 1.0000, 0.2174],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=float,
        )  # TODO: refine this

    @staticmethod
    def dh_modified_transform(alpha, a, theta, d):
        ct = cos(theta)
        st = sin(theta)

        ca = cos(alpha)
        sa = sin(alpha)

        t11 = ct
        t12 = -st
        t13 = 0
        t14 = a

        t21 = st * ca
        t22 = ct * ca
        t23 = -sa
        t24 = -d * sa

        t31 = st * sa
        t32 = ct * sa
        t33 = ca
        t34 = d * ca

        t41 = 0
        t42 = 0
        t43 = 0
        t44 = 1

        T = np.array(
            [[t11, t12, t13, t14], [t21, t22, t23, t24], [t31, t32, t33, t34], [t41, t42, t43, t44]]
        )
        return T

    def get_T_oi(self, theta):
        d = self.d
        a = self.a
        alpha = self.alpha

        T_i1i = []
        for i in range(7):
            T = self.dh_modified_transform(alpha[i], a[i], theta[i], d[i])
            T_i1i.append(T)

        T_cumulative = copy(self.T_base)
        T_oi = []
        for T in T_i1i:
            T_cumulative = T_cumulative @ T
            T_oi.append(copy(T_cumulative))

        T_i1i = np.array(T_i1i)
        T_oi = np.array(T_oi)
        return T_i1i, T_oi

    def get_T_ok(self, theta):
        assert len(theta) == 8
        open_gripper = theta[-1]
        theta = theta[:-1]
        T_ik = np.array(
            [
                RT2HT(euler2mat(np.radians(self.PosEuler_ik[i, 3:])), self.PosEuler_ik[i, :3])
                for i in range(7)
            ]
        )
        _, T_oi = self.get_T_oi(theta)
        T_ok = np.array([T_oi[i] @ T_ik[i] for i in range(7)])

        # base
        T_ok_base = copy(self.T_base)

        # gripper
        if open_gripper:
            T_ok_gripper = [
                T_oi[-1] @ PosEuler2HT(self.PosEuler_ik_gripper_close[i]) for i in range(3)
            ]
        else:
            T_ok_gripper = [
                T_oi[-1] @ PosEuler2HT(self.PosEuler_ik_gripper_open[i]) for i in range(3)
            ]

        T_ok_others = np.stack([T_ok_base, *T_ok_gripper])
        return T_ok, T_ok_others

    def get_obbox(self, T_ok, T_ok_others, color=[1, 0, 0], tolerance=0.01):
        if tolerance is not None:
            bbox = self.bbox_link + np.array([[-tolerance, tolerance] * 3])
            bbox_other = self.bbox_other + np.array([[-tolerance, tolerance] * 3])
        else:
            bbox = self.bbox_link
            bbox_other = self.bbox_other

        # link_bbox
        assert T_ok.shape[0] == bbox.shape[0]
        translation = T_ok[:, :3, 3]
        rotation = T_ok[:, :3, :3]
        link_obbox = []
        for i in range(bbox.shape[0]):
            s_obbox = o3d.geometry.OrientedBoundingBox(
                translation[i], rotation[i], bbox[i, 1::2] - bbox[i, ::2]
            )
            s_obbox.color = color
            link_obbox.append(s_obbox)

        # other bbox

        other_obbox = []
        for i in range(4):
            s_obbox = o3d.geometry.OrientedBoundingBox(
                T_ok_others[i][:3, 3],
                T_ok_others[i][:3, :3],
                bbox_other[i][1::2] - bbox_other[i][::2],
            )
            s_obbox.color = color
            other_obbox.append(s_obbox)

        return link_obbox, other_obbox

    def theta2obbox(self, theta):
        if len(theta) == 7:
            theta = np.hstack([theta, 1])
        T_ok, T_ok_others = self.get_T_ok(theta)
        obbox, other_bbox = self.get_obbox(T_ok, T_ok_others=T_ok_others, tolerance=0.005)
        return obbox, other_bbox

    def visualize_pcd(self, xyz, rgb, theta):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        bbox, _ = self.theta2obbox(theta)
        o3d.visualization.draw_geometries([pcd, *bbox], window_name="bbox", width=1920, height=1080)

    def theta2PosQuat(self, theta):  # assume no open
        _, T_oi = self.get_T_oi(theta)
        T_oi_last = T_oi[-1]
        T_PosQuat = T_oi_last @ self.T_last2PosQuat
        PosQuat = HT2PosQuat(T_PosQuat)
        return PosQuat

    def set_T_base(self, T_base):
        self.T_base = copy(T_base)
