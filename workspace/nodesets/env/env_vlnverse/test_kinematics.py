"""Unit tests for env_vlnverse._quat / ._kinematics (numpy-only, no Isaac).

Modules under test are imported through the synthetic package conftest.py
registers — not the real package, whose ``__init__.py`` pulls the
``app.components`` backend. Run with
``PYTHONPATH=<repo>/agentcanvas/backend:<repo>`` (pytest imports the real
package for conftest collection).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_quat = importlib.import_module("envvlnverse_under_test._quat")
_kinematics = importlib.import_module("envvlnverse_under_test._kinematics")

euler_angles_to_quat = _quat.euler_angles_to_quat
quat_to_euler_angles = _quat.quat_to_euler_angles
VLNVerseKinematics = _kinematics.VLNVerseKinematics


# ==================== quaternion helpers ====================


def test_yaw_quat_hand_pinned():
    # euler [0,0,pi/2] -> (w,x,y,z) = (cos(pi/4), 0, 0, sin(pi/4))
    q = euler_angles_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(q, [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-12)


@pytest.mark.parametrize("theta", [-3.0, -np.pi / 2, -0.3, 0.0, 0.7, np.pi / 2, 2.9])
def test_quat_heading_round_trip(theta):
    q = euler_angles_to_quat(np.array([0.0, 0.0, theta]))
    rpy = quat_to_euler_angles(q)
    assert abs(rpy[0]) < 1e-9 and abs(rpy[1]) < 1e-9
    assert np.isclose(rpy[2], theta, atol=1e-9)


def test_quat_parity_with_scipy():
    scipy_transform = pytest.importorskip("scipy.spatial.transform")
    R = scipy_transform.Rotation
    rng = np.random.default_rng(0)
    for _ in range(20):
        rpy = rng.uniform(-1.4, 1.4, size=3)  # stay clear of gimbal lock
        ours = euler_angles_to_quat(rpy)
        s = R.from_euler("xyz", rpy).as_quat()  # (x,y,z,w)
        theirs = np.array([s[3], s[0], s[1], s[2]])
        # q and -q are the same rotation
        assert np.allclose(ours, theirs, atol=1e-9) or np.allclose(ours, -theirs, atol=1e-9)
        # inverse direction
        back = quat_to_euler_angles(theirs)
        theirs_rpy = R.from_quat(s).as_euler("xyz")
        assert np.allclose(back, theirs_rpy, atol=1e-9)


# ==================== movement geometry ====================


def _kin_free() -> VLNVerseKinematics:
    k = VLNVerseKinematics(use_occupancy_collision=False)
    k.set_pose(np.array([0.0, 0.0, 1.2]), 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    return k


def test_straight_move_advances_along_heading():
    k = _kin_free()
    collision, dev = k.move(0.0, 0.0, 2.0)
    assert collision is False and dev == 0.0
    assert np.allclose(k.agent_position, [2.0, 0.0, 1.2], atol=1e-9)
    assert np.isclose(k.agent_heading, 0.0, atol=1e-9)


def test_heading_composition():
    k = _kin_free()
    k.move(np.pi / 2, 0.0, 1.0)  # face +y, advance 1
    assert np.isclose(k.agent_heading, np.pi / 2, atol=1e-9)
    assert np.allclose(k.agent_position, [0.0, 1.0, 1.2], atol=1e-9)
    k.move(np.pi / 2, 0.0, 1.0)  # face -x (heading pi), advance 1
    assert np.isclose(abs(k.agent_heading), np.pi, atol=1e-6)
    assert np.allclose(k.agent_position, [-1.0, 1.0, 1.2], atol=1e-9)


def test_elevation_is_degrees():
    # Upstream converts elevation via np.radians -> 90 means straight up.
    k = _kin_free()
    k.move(0.0, 90.0, 1.0)
    assert np.allclose(k.agent_position, [0.0, 0.0, 2.2], atol=1e-9)


# ==================== occupancy / freemap ====================
#
# Synthetic freemap convention (matches data/scene/*/freemap.npy):
#   row 0   = world-x coordinate of each column
#   col 0   = world-y coordinate of each row
#   interior: 0 obstacle, 1 reachable, 2 out-of-bounds
# Coordinates use a 0.125 m grid step (exactly representable in binary) so
# absolute world<->grid assertions below are exact, not approximate.

X0, Y0, STEP, N = 10.0, 20.0, 0.125, 12


def _make_freemap(interior: float = 1.0) -> np.ndarray:
    occ = np.full((N, N), interior, dtype=np.float64)
    occ[0, :] = X0 + STEP * np.arange(N)  # x axis along columns
    occ[:, 0] = Y0 + STEP * np.arange(N)  # y axis along rows
    return occ


def _kin_occ(tmp_path, occ: np.ndarray) -> VLNVerseKinematics:
    path = tmp_path / "freemap.npy"
    np.save(path, occ)
    k = VLNVerseKinematics(use_occupancy_collision=True)
    k.load_freemap(str(path))
    return k


def test_world_grid_mapping_hand_pinned(tmp_path):
    """Absolute mapping pin: cell (row i, col j) has world coords
    (X0 + j*STEP, Y0 + i*STEP), read back EXACTLY from the axis vectors.
    A convention regression (axis swap, offset) fails this loudly."""
    k = _kin_occ(tmp_path, _make_freemap())
    x, y = X0 + 5 * STEP, Y0 + 3 * STEP  # 10.625, 20.375 — exact doubles
    assert k.find_nearest_reachable(x, y) == (x, y, 0)


def test_free_path_no_collision_snaps_to_cell(tmp_path):
    k = _kin_occ(tmp_path, _make_freemap())
    k.set_pose(np.array([X0 + 2 * STEP + 0.01, Y0 + 2 * STEP + 0.01, 1.2]), 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.0, 0.0, 3 * STEP)  # intended lands ~cell (2, 5)
    assert collision is False
    assert dev <= np.hypot(STEP / 2, STEP / 2) + 1e-9  # snapped within half-cell
    # snapped exactly onto cell-center coords
    assert k.agent_position[0] in set((X0 + STEP * np.arange(N)).tolist())
    assert k.agent_position[1] in set((Y0 + STEP * np.arange(N)).tolist())


def test_wall_forces_snap_and_collision(tmp_path):
    occ = _make_freemap()
    occ[1:, 7:10] = 0.0  # 3-column wall at x = 10.875, 11.0, 11.125
    k = _kin_occ(tmp_path, occ)
    wall_x, mid_y = X0 + 8 * STEP, Y0 + 5 * STEP
    res = k.find_nearest_reachable(wall_x, mid_y)
    assert res is not None
    rx, ry, grid_d = res
    assert grid_d > 0
    assert np.isclose(abs(rx - wall_x), 2 * STEP, atol=1e-9)  # nearest free column
    # A move aimed into the wall: snap distance 0.25 m > threshold 0.1 -> collision
    k.set_pose(np.array([wall_x - 4 * STEP, mid_y, 1.2]), 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.0, 0.0, 4 * STEP)
    assert collision is True
    assert np.isclose(dev, 2 * STEP, atol=1e-9)
    assert k.collision_occurred is True


def test_no_reachable_cell_stays_put(tmp_path):
    occ = _make_freemap(interior=0.0)  # everything blocked
    k = _kin_occ(tmp_path, occ)
    start = np.array([X0 + 5 * STEP, Y0 + 5 * STEP, 1.2])
    k.set_pose(start, 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.5, 0.0, 1.0)
    assert collision is True and dev == 0.0
    assert np.allclose(k.agent_position, start)  # position unchanged...
    assert np.isclose(k.agent_heading, 0.5, atol=1e-9)  # ...but heading rotated


def test_missing_freemap_raises():
    k = VLNVerseKinematics(use_occupancy_collision=True)
    with pytest.raises(RuntimeError, match="freemap"):
        k.move(0.0, 0.0, 1.0)


# ==================== episode wiring ====================


def test_reset_from_episode_pose_and_geometry():
    k = VLNVerseKinematics(use_occupancy_collision=False)
    yaw = -0.6036719043591668 * 2  # arbitrary
    q = euler_angles_to_quat(np.array([0.0, 0.0, yaw]))
    ep = {
        "start_position": [-2.75, 2.98, 0.0],
        "start_rotation": q.tolist(),  # [w,x,y,z]
        "goals": {"position": [-3.71, -0.32, 0.0], "radius": 3.0},
        "reference_path": [[-2.75, 2.98, 0.0], [-2.68, 2.74, 0.0]],
    }
    k.collision_occurred = True
    k.reset_from_episode(ep)
    assert k.collision_occurred is False
    assert np.allclose(k.agent_position, [-2.75, 2.98, 1.2])  # z pinned to 1.2
    assert np.isclose(k.agent_heading, yaw, atol=1e-9)
    assert np.allclose(k.goal_position, [-3.71, -0.32, 1.2])
    assert len(k.reference_path) == 2
    assert np.isclose(k.reference_path[1][2], 1.2)  # path z lifted by +1.2
    d = k.current_dist_to_goal()
    assert np.isclose(d, np.linalg.norm(k.agent_position - k.goal_position))
