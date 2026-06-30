"""VoxPoserLMPRuntime — episode-scoped LMP runtime around a WiredEnv.

Mirrors the retired in-subprocess ``libero/_voxposer/lmp_runtime.py`` but
swaps the live ``LiberoVoxPoserAdapter`` for a ``WiredEnv`` (GT snapshot
fed by the ``env_libero__observe_objects`` wire). The 7 LMPs themselves
are unchanged vendored VoxPoser code.

Key method — ``plan_subtask(subtask, snapshot)``: runs the composer LMP
with the real parse_query_obj / voxel-map LMPs against the snapshot, and
stubs ONLY the two sim-driving entry points:
  * ``execute(...)``            -> ``LMP_interface.plan(...)`` (plan, don't drive)
  * ``reset_to_default_pose()`` -> emit a single lift-to-home waypoint
The returned world-frame trajectory is followed closed-loop by the graph
via ``env_libero__step_pose``. This is the same plan/follow split the
verified in-subprocess ``plan_subtask`` used — only the snapshot
transport differs, which is semantically identical because planning
never steps the sim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import transforms3d

from ._wired_env import WiredEnv
from .interfaces import setup_LMP
from .utils import set_lmp_objects

# Lift altitude for the reset_to_default_pose stub is derived per-scene from
# the snapshot's workspace_bounds_max.z (just under the ceiling), NOT a fixed
# constant: LIBERO suites anchor the world frame differently (table-mounted
# libero_spatial ceiling z≈1.26 vs floor-mounted libero_object ceiling z≈0.35),
# so a hardcoded 1.20 made the floor-frame lift unreachable and dropped the
# object. See env_libero._workspace_bounds. This margin keeps the lift clear of
# the ceiling while staying inside the planner's reachable box.
_LIFT_MARGIN_BELOW_CEIL = 0.05

# Top-down grasp approach height (m above the object grasp point). The vendored
# path planner returns a smoothed diagonal descent that OVERSHOOTS the object in
# xy before snapping to the affordance target — at grasp height the open gripper
# then knocks the object aside and closes on empty space (diagnosed 2026-06-28,
# run 20260628_151303: soup pushed -0.244→-0.201, never lifted). A grasp on an
# open tabletop is reliably a vertical pinch, so for grasp subtasks we discard
# the overshooting path and descend straight down onto the (correctly-placed)
# affordance target instead.
_GRASP_APPROACH_M = 0.08


class _AttrDict(dict):
    """Dict supporting ``[k]`` and ``.k`` access (matches OmegaConf shape)."""

    __slots__ = ()

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _wrap_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({k: _wrap_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap_config(v) for v in obj]
    return obj


_LMP_TEMPLATE_CHAT = {
    "prompt_fname": "",
    "model": "gpt-4",
    "max_tokens": 512,
    "temperature": 0.0,
    "query_prefix": "# Query: ",
    "query_suffix": ".",
    "stop": ["# Query: ", "objects = "],
    "maintain_session": False,
    "include_context": True,
    "has_return": False,
    "return_val_name": "ret_val",
    "load_cache": True,
}


def _lmp_cfg(
    prompt_fname: str,
    *,
    has_return: bool = False,
    include_context: bool = True,
    stop: list[str] | None = None,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 512,
    load_cache: bool = True,
) -> dict[str, Any]:
    cfg = dict(_LMP_TEMPLATE_CHAT)
    cfg.update(
        {
            "prompt_fname": prompt_fname,
            "has_return": has_return,
            "include_context": include_context,
            "stop": list(stop) if stop is not None else ["# Query: ", "objects ="],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "load_cache": load_cache,
        }
    )
    return cfg


def build_general_config(
    *,
    map_size: int = 100,
    max_plan_iter: int = 1,
    controller_horizon: int = 1,
    controller_num_samples: int = 10000,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 512,
    load_cache: bool = True,
) -> dict[str, Any]:
    """Mirror of the retired lmp_runtime.build_general_config (env_name='libero')."""
    base = dict(model=model, max_tokens=max_tokens, temperature=temperature, load_cache=load_cache)
    return _wrap_config(
        {
            "env_name": "libero",
            "planner": {
                "stop_threshold": 0.001,
                "savgol_polyorder": 3,
                "savgol_window_size": 20,
                "obstacle_map_weight": 1,
                "max_steps": 300,
                "obstacle_map_gaussian_sigma": 10,
                "target_map_weight": 2,
                "stop_criteria": "no_nearby_equal",
                "target_spacing": 1,
                "max_curvature": 3,
                "pushing_skip_per_k": 5,
            },
            "controller": {
                "horizon_length": controller_horizon,
                "num_samples": controller_num_samples,
                "ee_local": "temperature",
                "ee_local_radius": 0.15,
            },
            "lmp_config": {
                "env": {
                    "map_size": map_size,
                    "num_waypoints_per_plan": 10000,
                    "max_plan_iter": max_plan_iter,
                    "visualize": False,
                },
                "lmps": {
                    "planner": _lmp_cfg("planner_prompt", **base),
                    "composer": _lmp_cfg("composer_prompt", include_context=False, **base),
                    "parse_query_obj": _lmp_cfg("parse_query_obj_prompt", has_return=True, **base),
                    "get_affordance_map": _lmp_cfg(
                        "get_affordance_map_prompt", has_return=True, **base
                    ),
                    "get_avoidance_map": _lmp_cfg(
                        "get_avoidance_map_prompt", has_return=True, **base
                    ),
                    "get_rotation_map": _lmp_cfg(
                        "get_rotation_map_prompt", has_return=True, **base
                    ),
                    "get_velocity_map": _lmp_cfg(
                        "get_velocity_map_prompt", has_return=True, **base
                    ),
                    "get_gripper_map": _lmp_cfg("get_gripper_map_prompt", has_return=True, **base),
                },
            },
        }
    )


class VoxPoserLMPRuntime:
    """Per-episode LMP runtime: holds the WiredEnv + 7 LMPs.

    Lifecycle:
      1. ``__init__`` — build WiredEnv from the initial snapshot, build all
         7 LMPs via ``setup_LMP``, prime object-name context.
      2. ``run_planner(instruction)`` — composer capture-stubbed; returns
         the ordered subtask strings.
      3. ``plan_subtask(subtask, snapshot)`` — refresh WiredEnv from the
         per-iter snapshot, plan ONE subtask, return the waypoint
         trajectory (the graph follows it closed-loop via step_pose).
    """

    def __init__(
        self,
        *,
        snapshot: dict[str, Any],
        map_size: int = 100,
        max_plan_iter: int = 1,
        controller_horizon: int = 1,
        controller_num_samples: int = 10000,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 512,
        load_cache: bool = True,
    ) -> None:
        bounds_min = snapshot.get("bounds_min")
        bounds_max = snapshot.get("bounds_max")
        if bounds_min is None or bounds_max is None:
            raise ValueError("snapshot must carry bounds_min/bounds_max")
        self._wired_env = WiredEnv(
            workspace_bounds_min=np.asarray(bounds_min, dtype=np.float32),
            workspace_bounds_max=np.asarray(bounds_max, dtype=np.float32),
        )
        self._wired_env.update_snapshot(snapshot)
        self._config = build_general_config(
            map_size=map_size,
            max_plan_iter=max_plan_iter,
            controller_horizon=controller_horizon,
            controller_num_samples=controller_num_samples,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            load_cache=load_cache,
        )
        self._lmps, self._lmp_env = setup_LMP(
            self._wired_env,
            self._config,
            debug=False,
        )
        self._planner_lmp = self._lmps["plan_ui"]
        self._composer_lmp = self._lmps["composer_ui"]
        set_lmp_objects(self._lmps, self._wired_env.get_object_names())
        # Executed-gripper state in DOWNSTREAM convention (1.0=closed, 0.0=open),
        # carried across subtasks so "move"/"back to default" hold what the last
        # grasp/release commanded. Episode starts with the gripper open.
        self._grip_state: float = 0.0

    # ── planner LMP ────────────────────────────────────────────────────

    def run_planner(self, instruction: str) -> dict[str, Any]:
        """Invoke planner LMP with the composer capture-stubbed.

        The planner LMP exec()s Python that calls ``composer("...")`` per
        subtask; the stub records the strings instead of running them.
        """
        captured: list[str] = []

        def _capture(arg: Any, *args: Any, **kwargs: Any) -> None:
            captured.append(str(arg))

        original = self._planner_lmp._variable_vars.get("composer")
        try:
            self._planner_lmp._variable_vars["composer"] = _ComposerCapture(_capture)
            self._planner_lmp(instruction)
        finally:
            self._planner_lmp._variable_vars["composer"] = original

        return {"subtasks": captured, "instruction": instruction}

    # ── per-subtask plan (the decomposed path) ─────────────────────────

    def plan_subtask(self, subtask: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Plan ONE subtask from the snapshot; return a waypoint trajectory.

        The real parse_query_obj / get_*_map LMPs run against the snapshot
        so the value maps are real; only the two sim-driving entry points
        are stubbed (execute -> plan, reset_to_default_pose -> lift
        waypoint). Single-execute-per-subtask is the LIBERO composer idiom;
        if a subtask emits several execute/reset calls, their trajectories
        concatenate in source order (last movable wins).
        """
        self._wired_env.update_snapshot(snapshot)
        set_lmp_objects(self._lmps, self._wired_env.get_object_names())

        captured: dict[str, Any] = {"trajectory": []}
        n_execute = [0]

        def _plan_stub(movable_obs_func, affordance_map=None, avoidance_map=None,
                       rotation_map=None, velocity_map=None, gripper_map=None):
            n_execute[0] += 1
            traj_world, object_centric = self._lmp_env.plan(
                movable_obs_func,
                affordance_map=affordance_map, avoidance_map=avoidance_map,
                rotation_map=rotation_map, velocity_map=velocity_map,
                gripper_map=gripper_map,
            )
            try:
                captured["movable_name"] = movable_obs_func()["name"]
            except Exception:  # noqa: BLE001 — name is diagnostic only
                captured["movable_name"] = None
            captured["object_centric"] = bool(object_centric)
            captured["trajectory"].extend(self._serialize_traj(traj_world))
            return None

        def _reset_stub():
            cur = self._wired_env.get_ee_pos()
            quat = self._wired_env.get_ee_quat()
            # Keep the gripper from the LAST waypoint emitted in THIS subtask
            # (e.g. closed after a grasp execute) rather than the stale
            # pre-subtask seed. A "back to default pose"/lift that follows a
            # grasp in the SAME subtask must hold the object, not release it —
            # reading get_last_gripper_action() here gave the constructor 0.0
            # (open) and dropped every grasped object on lift.
            if captured["trajectory"]:
                grip = float(captured["trajectory"][-1][7])
            else:
                grip = float(self._wired_env.get_last_gripper_action())
            # Mirror the retired adapter's reset_to_default_pose: lift toward
            # the (per-scene) workspace ceiling, keep xy + current orientation
            # + gripper. One waypoint for the follow loop to step_pose toward.
            # max() guards against a degenerate ceiling below the current EE.
            lift_z = max(
                float(cur[2]),
                float(self._wired_env.workspace_bounds_max[2]) - _LIFT_MARGIN_BELOW_CEIL,
            )
            captured["trajectory"].append([
                float(cur[0]), float(cur[1]), float(lift_z),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), grip,
            ])
            captured.setdefault("movable_name", "gripper")
            captured.setdefault("object_centric", False)
            return 0

        composer_vars = self._composer_lmp._variable_vars
        orig_exec = composer_vars.get("execute")
        orig_reset = composer_vars.get("reset_to_default_pose")
        error: str | None = None
        try:
            composer_vars["execute"] = _plan_stub
            if "reset_to_default_pose" in composer_vars:
                composer_vars["reset_to_default_pose"] = _reset_stub
            self._composer_lmp(subtask)
        except Exception as e:  # noqa: BLE001 — surfaced to the node
            error = f"{type(e).__name__}: {e}"
        finally:
            composer_vars["execute"] = orig_exec
            if orig_reset is not None:
                composer_vars["reset_to_default_pose"] = orig_reset

        trajectory = captured.get("trajectory", [])
        if error is None and not trajectory:
            error = f"composer produced no waypoints (n_execute={n_execute[0]})"

        # Gripper schedule — deterministic, from the composer's subtask STRING.
        # WHY not the vendored gripper_map: it uses the OPPOSITE convention to
        # our downstream 8-vec (gripper_map 1=open/0=closed vs step_pose
        # 1=closed/0=open) AND marks the "closed" region as a ~1 cm voxel ball
        # at the object that the planned path's last voxel routinely misses at
        # LIBERO's cm-scale objects. Net effect: the map-sampled gripper was
        # both inverted AND usually "open everywhere", so every grasp descended
        # with the fingers already (mis-)closed and never captured the object —
        # the soup never moved (diagnosed 2026-06-28, run 20260628_145644). v4
        # sidestepped this by driving the grasp through the dynamics controller
        # (contact-based close), not the gripper_map. The composer's subtask
        # intent IS the reliable signal, so we schedule the gripper from it.
        # Convention here is downstream: 1.0=closed, 0.0=open. prev_gripper_open
        # reflects the gripper at subtask START (seeds expand_for_settle's
        # cross-subtask flip detection).
        prev_gripper_open = bool(self._grip_state < 0.5)
        sub = subtask.lower()
        if trajectory:
            if "grasp" in sub or "pick up" in sub:
                # Top-down grasp: the planned path's LAST waypoint is the
                # affordance target (object centre, correctly placed). Replace
                # the overshooting diagonal descent with pre-grasp-above →
                # straight-down → close, so the open gripper never sweeps the
                # object laterally. expand_for_settle inserts the post-close
                # settle so the fingers actuate AT the object.
                tgt = [float(v) for v in trajectory[-1][:3]]
                quat = [float(v) for v in trajectory[-1][3:7]]
                pre_grasp = [tgt[0], tgt[1], tgt[2] + _GRASP_APPROACH_M, *quat, 0.0]
                at_object = [tgt[0], tgt[1], tgt[2], *quat, 1.0]
                trajectory = [pre_grasp, at_object]
                self._grip_state = 1.0
            elif (
                "open gripper" in sub or "release" in sub
                or "let go" in sub or "drop" in sub
            ):
                for wp in trajectory:
                    wp[7] = 0.0
                self._grip_state = 0.0
            elif "close gripper" in sub:
                for wp in trajectory:
                    wp[7] = 1.0
                self._grip_state = 1.0
            else:
                # move / transport / back-to-default-pose: hold the carried
                # state so a grasped object stays held through the lift and
                # the move to the placement target.
                for wp in trajectory:
                    wp[7] = self._grip_state
            captured["trajectory"] = trajectory
        return {
            "subtask": subtask,
            "trajectory": trajectory,
            "movable_name": captured.get("movable_name"),
            "object_centric": captured.get("object_centric", False),
            "prev_gripper_open": prev_gripper_open,
            "error": error,
        }

    @staticmethod
    def _serialize_traj(traj_world: list) -> list:
        """``traj_world`` elements ``(world_xyz[3], rotation[4], velocity, gripper)``
        → JSON-safe 8-vecs ``[x,y,z,qw,qx,qy,qz,gripper]`` (velocity dropped —
        the step_pose follow loop has its own OSC speed bound)."""
        out: list[list[float]] = []
        for wp in traj_world:
            xyz = np.asarray(wp[0], dtype=np.float32).reshape(-1)[:3]
            rot = np.asarray(wp[1], dtype=np.float32).reshape(-1)[:4]
            grip = float(np.asarray(wp[3]).reshape(-1)[0])
            out.append([
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]), grip,
            ])
        return out


class _ComposerCapture:
    """Substitute composer LMP that records subtask strings without firing."""

    def __init__(self, capture_fn) -> None:
        self._capture = capture_fn

    def __call__(self, arg, *args, **kwargs):
        self._capture(arg)


# transforms3d is imported here so its import failure surfaces at module load
# rather than at first composer LMP exec (the LMP namespace exposes
# ``euler2quat`` / ``quat2euler`` from transforms3d to the LLM).
_ = transforms3d
