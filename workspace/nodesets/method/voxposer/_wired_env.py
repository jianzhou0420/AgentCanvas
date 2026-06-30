"""WiredEnv — read-only env contract for VoxPoser's ``LMP_interface``.

Backs the vendored ``LMP_interface`` with a GT scene snapshot delivered
over a canvas wire (``env_libero__observe_objects`` → ``snapshot``)
instead of a live simulator handle. The voxposer method nodeset runs
local-mode in the agentcanvas backend and cannot reach the LIBERO
subprocess synchronously — and it doesn't need to: planning never steps
the sim, so every env read during one ``plan_subtask`` call sees a
single static state. A frozen snapshot is semantically identical to the
live reads the old in-subprocess ``LiberoVoxPoserAdapter`` performed.

Read contract (everything ``LMP_interface`` calls during planning):
    workspace_bounds_min / workspace_bounds_max   (numpy 3-vec)
    get_object_names() -> list[str]
    get_3d_obs_by_name(name) -> (pc Nx3, normals Nx3)
    get_scene_3d_obs(ignore_robot, ignore_grasped_obj) -> (pc Nx3, None)
    get_ee_pos() / get_ee_quat() / get_ee_pose()
    get_last_gripper_action() -> float (VoxPoser convention: 1=closed)
    visualizer = None

Write surface: NONE. ``apply_action`` / ``reset_to_default_pose`` raise —
the decomposed path stubs the composer's ``execute`` to
``LMP_interface.plan`` (plan, don't drive) and its
``reset_to_default_pose`` to a lift-waypoint emitter, so the sim-driving
entry points are never reached. Raising here is the decoupling guard:
if vendored code ever tries to drive the sim from the backend process,
we want a loud failure, not a silent no-op (the abandoned v2 WiredEnv
buffered waypoints open-loop instead — SR=0).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class WiredEnv:
    """Snapshot-backed env for the VoxPoser LMP runtime, refreshed per subtask.

    Threading: one subtask plans synchronously inside one graph node's
    forward(). No concurrency concerns.
    """

    visualizer = None  # checked by LMP_interface for truthiness

    def __init__(
        self,
        *,
        workspace_bounds_min: np.ndarray,
        workspace_bounds_max: np.ndarray,
    ) -> None:
        self.workspace_bounds_min = np.asarray(workspace_bounds_min, dtype=np.float32)
        self.workspace_bounds_max = np.asarray(workspace_bounds_max, dtype=np.float32)
        self._object_names: list[str] = []
        self._object_pcs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._scene_pc: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._ee_pos = np.zeros(3, dtype=np.float32)
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Gripper convention: VoxPoser standard 1.0=closed, 0.0=open.
        # FROZEN-V1 PARITY: plan-time gripper state is pinned OPEN and never
        # updated from the snapshot. The verified in-subprocess path read
        # mgr._last_obs["robot0_gripper_qpos"], but _process_obs never emits
        # that key, so sync_last_grip_from_sim was a no-op and the adapter
        # kept its constructor value 0.0 (open) for the whole episode. The
        # snapshot's truthful physical state (closed after LIBERO's
        # 10-tick close-gripper reset settle) inverts every default
        # gripper_map ("keep current") and breaks grasp plans — verified
        # 2026-06-10 smoke: grasp approach planned closed-finger, bowl
        # pushed not grasped. So this seeds OPEN at episode start (correct
        # for the subtask-0 grasp). Cross-subtask continuity is then carried
        # forward by VoxPoserLMPRuntime.plan_subtask, which writes the last
        # COMMANDED gripper back here after each subtask (NOT the physical
        # snapshot read) — so a post-grasp lift keeps the object held.
        self._gripper = 0.0

    # ── snapshot wiring ────────────────────────────────────────────────

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Refresh from an ``env_libero__observe_objects`` snapshot dict.

        Expected keys: object_names, object_pcs ({name: {pc, normals}}),
        scene_pc, ee_pos, ee_quat, gripper_open, bounds_min, bounds_max.
        """
        bounds_min = snapshot.get("bounds_min")
        bounds_max = snapshot.get("bounds_max")
        if bounds_min is not None:
            self.workspace_bounds_min = np.asarray(bounds_min, dtype=np.float32)
        if bounds_max is not None:
            self.workspace_bounds_max = np.asarray(bounds_max, dtype=np.float32)

        self._object_names = [str(n) for n in (snapshot.get("object_names") or [])]

        new_pcs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, payload in (snapshot.get("object_pcs") or {}).items():
            pc = np.asarray(payload.get("pc") or [], dtype=np.float32).reshape(-1, 3)
            normals_payload = payload.get("normals")
            if normals_payload is not None:
                normals = np.asarray(normals_payload, dtype=np.float32).reshape(-1, 3)
            else:
                normals = np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (pc.shape[0], 1)
                )
            new_pcs[str(name)] = (pc, normals)
        self._object_pcs = new_pcs

        scene_pc = snapshot.get("scene_pc")
        self._scene_pc = (
            np.asarray(scene_pc, dtype=np.float32).reshape(-1, 3)
            if scene_pc is not None
            else np.zeros((0, 3), dtype=np.float32)
        )

        ee_pos = snapshot.get("ee_pos")
        if ee_pos is not None:
            self._ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
        ee_quat = snapshot.get("ee_quat")
        if ee_quat is not None:
            q = np.asarray(ee_quat, dtype=np.float32).reshape(-1)
            self._ee_quat = (
                q if q.shape[0] == 4
                else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )
        # snapshot["gripper_open"] is deliberately NOT consumed — see the
        # frozen-v1 parity note on self._gripper in __init__.

    # ── LMP_interface read contract ────────────────────────────────────

    def get_object_names(self) -> list[str]:
        return list(self._object_names)

    def get_3d_obs_by_name(self, query_name: str) -> tuple[np.ndarray, np.ndarray]:
        if query_name in self._object_pcs:
            return self._object_pcs[query_name]
        # Fuzzy fallback — LMPs may say "moka pot" for BDDL "moka_pot_1".
        target = query_name.lower().replace(" ", "_")
        for k, v in self._object_pcs.items():
            if target in k.lower() or k.lower() in target:
                return v
        raise ValueError(
            f"Object {query_name!r} not in snapshot "
            f"(have: {list(self._object_pcs.keys())})"
        )

    def get_scene_3d_obs(
        self,
        ignore_robot: bool = False,
        ignore_grasped_obj: bool = False,
    ) -> tuple[np.ndarray, None]:
        # observe_objects pre-applies ignore_robot when building scene_pc.
        # ignore_grasped_obj is ignored — a stale grasped object only makes
        # the avoidance map slightly conservative, not unsafe.
        return self._scene_pc, None

    def get_ee_pos(self) -> np.ndarray:
        return self._ee_pos.copy()

    def get_ee_quat(self) -> np.ndarray:
        return self._ee_quat.copy()

    def get_ee_pose(self) -> np.ndarray:
        return np.concatenate([self._ee_pos, self._ee_quat])

    def get_last_gripper_action(self) -> float:
        return float(self._gripper)

    # ── decoupling guard — no write surface ────────────────────────────

    def apply_action(self, pose8: Any) -> None:
        raise RuntimeError(
            "WiredEnv.apply_action: the voxposer method nodeset cannot drive "
            "the sim — execute() must be stubbed to LMP_interface.plan() "
            "(see VoxPoserLMPRuntime.plan_subtask)"
        )

    def reset_to_default_pose(self) -> None:
        raise RuntimeError(
            "WiredEnv.reset_to_default_pose: must be stubbed at the composer "
            "layer to a lift-waypoint emitter (see VoxPoserLMPRuntime.plan_subtask)"
        )
