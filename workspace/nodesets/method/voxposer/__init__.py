from __future__ import annotations

"""VoxPoser method nodeset — env-decoupled plan/follow for manipulation envs.

Method-side half of the decomposed VoxPoser graph
(``workspace/graphs/vla/unverified/voxposer_libero_decomposed.json``).
Runs local-mode in the agentcanvas backend; the ONLY env coupling is the
GT snapshot wire from ``env_libero__observe_objects`` (privileged
observation space) and the waypoint actions the graph sends to
``env_libero__step_ee_pose``. Any env exposing that observe/step pair can
host VoxPoser.

Architecture:
  voxposer__init               — one-shot per-episode setup; builds the
                                 LMP runtime + WiredEnv from the snapshot
  voxposer__planner_lmp        — one LLM call: instruction → subtask list
  voxposer__plan_subtask       — per outer iter: composer + voxel maps +
                                 path planner ONCE against a fresh
                                 snapshot; returns the waypoint trajectory
                                 WITHOUT driving the sim
  voxposer__expand_for_settle  — insert gripper-settle holds at flips
  voxposer__dispense_waypoint  — inner-loop cursor over the trajectory
  voxposer__plan_executor      — format one waypoint as step_ee_pose JSON
  voxposer__check_waypoint_done / voxposer__check_done — terminations

Planning is sound on a frozen snapshot because plan_subtask never steps
the sim — all env reads during one plan see a single static state. The
follow loop drives every waypoint closed-loop on the real env via
``env_libero__step_ee_pose`` (the abandoned v2 design buffered the whole
execute() open-loop instead — SR=0).

Module-level RUNTIME cache keys on a string handle (issued by
voxposer__init); other nodes look up the runtime by handle. This keeps
the heavy LMP runtime out of wire payloads.
"""

import logging
import secrets
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.voxposer")


# Module-level runtime cache. Keyed by opaque string handle (issued by
# voxposer__init). Cleared on episode_reset signal — see VoxPoserNodeSet.
_RUNTIMES: dict[str, Any] = {}  # value: VoxPoserLMPRuntime


def _new_handle() -> str:
    return secrets.token_hex(8)


def _resolve_runtime(handle: str | None):
    if not handle or handle not in _RUNTIMES:
        return None
    return _RUNTIMES[handle]


# ══════════════════════════════════════════════════════════════════════
# 1. voxposer__init
# ══════════════════════════════════════════════════════════════════════


class VoxPoserInitNode(BaseCanvasNode):
    node_type = "voxposer__init"
    display_name = "VoxPoser: Init"
    description = (
        "One-shot per-episode setup: build the 7-LMP runtime + WiredEnv "
        "from the GT scene snapshot (env observe_objects). Returns an opaque "
        "handle that downstream voxposer__* nodes resolve to the live runtime."
    )
    category = "method"
    icon = "Cpu"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model",
                "select",
                label="LLM model",
                options=[
                    {"value": "gpt-4", "label": "gpt-4"},
                    {"value": "gpt-4o", "label": "gpt-4o"},
                    {"value": "gpt-4-turbo", "label": "gpt-4-turbo"},
                ],
                default="gpt-4",
            ),
            ConfigField("temperature", "number", default=0.0),
            ConfigField("max_tokens", "number", default=512),
            ConfigField("map_size", "number", default=100),
            ConfigField("max_plan_iter", "number", default=1),
            ConfigField("controller_horizon", "number", default=1),
            ConfigField("controller_num_samples", "number", default=10000),
            ConfigField("load_cache", "toggle", default=True),
        ],
    )
    input_ports = [
        PortDef(
            "snapshot", "ANY",
            "GT scene snapshot from env observe_objects (object_names, "
            "object_pcs, scene_pc, ee_pos, ee_quat, gripper_open, bounds)",
        ),
    ]
    output_ports = [
        PortDef("runtime_handle", "TEXT", "Opaque runtime ID"),
        PortDef("error", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            from ._runtime import VoxPoserLMPRuntime
        except Exception as e:
            return {"runtime_handle": "", "error": f"runtime import failed: {e!r}"}

        snapshot = inputs.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("error"):
            err = (
                f"snapshot error: {snapshot.get('error')}"
                if isinstance(snapshot, dict)
                else "snapshot must be wired from env observe_objects"
            )
            self._self_log("error", err)
            return {"runtime_handle": "", "error": err}

        cfg = self.config or {}
        try:
            runtime = VoxPoserLMPRuntime(
                snapshot=snapshot,
                map_size=int(cfg.get("map_size", 100)),
                max_plan_iter=int(cfg.get("max_plan_iter", 1)),
                controller_horizon=int(cfg.get("controller_horizon", 1)),
                controller_num_samples=int(cfg.get("controller_num_samples", 10000)),
                model=str(cfg.get("model", "gpt-4")),
                temperature=float(cfg.get("temperature", 0.0)),
                max_tokens=int(cfg.get("max_tokens", 512)),
                load_cache=bool(cfg.get("load_cache", True)),
            )
        except Exception as e:
            log.exception("VoxPoser runtime build failed")
            return {"runtime_handle": "", "error": f"runtime build failed: {e!r}"}

        handle = _new_handle()
        _RUNTIMES[handle] = runtime
        n_objects = len(snapshot.get("object_names") or [])
        self._self_log("handle", handle)
        self._self_log("n_objects", n_objects)
        log.info("VoxPoser runtime built: handle=%s n_objects=%d", handle, n_objects)
        return {"runtime_handle": handle, "error": ""}


# ══════════════════════════════════════════════════════════════════════
# 2. voxposer__planner_lmp
# ══════════════════════════════════════════════════════════════════════


class VoxPoserPlannerLmpNode(BaseCanvasNode):
    node_type = "voxposer__planner_lmp"
    display_name = "VoxPoser: Planner LMP"
    description = (
        "Single LLM call: instruction → ordered list of sub-task strings. "
        "Composer LMP is monkey-patched to capture rather than execute."
    )
    category = "method"
    icon = "ListTree"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("runtime_handle", "TEXT"),
        PortDef("instruction", "TEXT"),
    ]
    output_ports = [
        PortDef("subtask_list", "LIST[TEXT]"),
        PortDef("subtask_count", "ANY"),
        PortDef("error", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        handle = str(inputs.get("runtime_handle") or "")
        runtime = _resolve_runtime(handle)
        if runtime is None:
            err = f"runtime_handle {handle!r} not found — voxposer__init must fire first"
            self._self_log("error", err)
            return {"subtask_list": [], "subtask_count": 0, "error": err}
        instruction = str(inputs.get("instruction") or "").strip()
        if not instruction:
            err = "instruction is empty"
            self._self_log("error", err)
            return {"subtask_list": [], "subtask_count": 0, "error": err}

        try:
            result = runtime.run_planner(instruction)
        except Exception as e:
            log.exception("planner LMP failed")
            return {"subtask_list": [], "subtask_count": 0, "error": f"planner LMP failed: {e!r}"}

        subtasks = list(result.get("subtasks") or [])
        self._self_log("instruction", instruction[:80])
        self._self_log("n_subtasks", len(subtasks))
        for i, s in enumerate(subtasks):
            self._self_log(f"subtask_{i}", s[:80])
        return {
            "subtask_list": subtasks,
            "subtask_count": len(subtasks),
            "error": "",
        }


# ══════════════════════════════════════════════════════════════════════
# 4. voxposer__plan_executor
# ══════════════════════════════════════════════════════════════════════


class VoxPoserPlanExecutorNode(BaseCanvasNode):
    node_type = "voxposer__plan_executor"
    display_name = "VoxPoser: Plan Executor"
    description = (
        "Format one waypoint for env step_ee_pose. Accepts either "
        "a raw 8-vec [x,y,z,qw,qx,qy,qz,gripper] (legacy) or a dict "
        "{'pose': [8-vec], 'hold_steps': int} (post-expand_for_settle)."
    )
    category = "method"
    icon = "ArrowRightCircle"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("waypoint", "ANY", "8-vec or {'pose', 'hold_steps'} dict; null = no-op"),
    ]
    output_ports = [
        PortDef("action", "TEXT", "JSON for env step_ee_pose: list (legacy) or object"),
        PortDef("has_action", "BOOL", "False when waypoint is null (no-op)"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import json as _json

        wp = inputs.get("waypoint")
        if wp is None:
            return {"action": "[]", "has_action": False}
        if isinstance(wp, dict):
            pose = wp.get("pose")
            try:
                arr = [float(x) for x in (pose or [])]
                hold = int(wp.get("hold_steps") or 0)
            except (TypeError, ValueError) as e:
                self._self_log("error", f"dict waypoint parse failed: {e!r}")
                return {"action": "[]", "has_action": False}
            if len(arr) != 8:
                self._self_log("error", f"pose length {len(arr)} != 8")
                return {"action": "[]", "has_action": False}
            return {
                "action": _json.dumps({"pose": arr, "hold_steps": hold}),
                "has_action": True,
            }
        try:
            arr = list(wp) if hasattr(wp, "__iter__") else [wp]
            if len(arr) != 8:
                self._self_log("error", f"waypoint length {len(arr)} != 8")
                return {"action": "[]", "has_action": False}
            return {"action": _json.dumps([float(x) for x in arr]), "has_action": True}
        except Exception as e:
            self._self_log("error", f"waypoint format failed: {e!r}")
            return {"action": "[]", "has_action": False}


# ══════════════════════════════════════════════════════════════════════
# 5. voxposer__check_done
# ══════════════════════════════════════════════════════════════════════


class VoxPoserCheckDoneNode(BaseCanvasNode):
    node_type = "voxposer__check_done"
    display_name = "VoxPoser: Check Done"
    description = "Termination: all sub-tasks complete OR episode success OR step budget."
    category = "method"
    icon = "Flag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("subtask_index", "ANY"),
        PortDef("subtask_count", "ANY"),
        PortDef("episode_success", "BOOL", optional=True),
        PortDef("step_index", "ANY", optional=True),
        PortDef("max_steps", "ANY", optional=True),
    ]
    output_ports = [
        PortDef("done", "BOOL"),
        PortDef("reason", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        try:
            idx = int(inputs.get("subtask_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        try:
            count = int(inputs.get("subtask_count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        success = bool(inputs.get("episode_success", False))
        try:
            step_idx = int(inputs.get("step_index", 0) or 0)
        except (TypeError, ValueError):
            step_idx = 0
        try:
            max_steps = int(inputs.get("max_steps", 0) or 0)
        except (TypeError, ValueError):
            max_steps = 0

        if success:
            return {"done": True, "reason": "episode_success"}
        if count > 0 and idx >= count:
            return {"done": True, "reason": "all_subtasks_complete"}
        if max_steps > 0 and step_idx >= max_steps:
            return {"done": True, "reason": "max_steps"}
        return {"done": False, "reason": ""}


# ══════════════════════════════════════════════════════════════════════
# 6. voxposer__plan_subtask     (multi-scope: outer-scope body)
# ══════════════════════════════════════════════════════════════════════


class VoxPoserPlanSubtaskNode(BaseCanvasNode):
    node_type = "voxposer__plan_subtask"
    display_name = "VoxPoser: Plan Subtask"
    description = (
        "Outer-scope body. Plans ONE subtask from a fresh GT snapshot "
        "(composer LMP + voxel maps + path planner) and returns the full "
        "world-frame waypoint trajectory — WITHOUT driving the sim. The "
        "graph's per-waypoint follow loop (dispense → step_ee_pose) then "
        "drives it closed-loop. Stateless across iters — caller (outer "
        "scope) drives subtask_index advancement."
    )
    category = "method"
    icon = "Brain"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("runtime_handle", "TEXT"),
        PortDef("subtask_list", "LIST[TEXT]"),
        PortDef(
            "subtask_index",
            "ANY",
            "Index of the subtask to plan (optional; defaults to 0 on iter 1 when iter_out hasn't fired yet)",
            optional=True,
        ),
        PortDef(
            "snapshot", "ANY",
            "Fresh GT scene snapshot from env observe_objects (re-fired per outer iter)",
        ),
    ]
    output_ports = [
        PortDef("trajectory", "LIST[ANY]",
                "List of 8-vec waypoints [x,y,z,qw,qx,qy,qz,gripper] for the follow loop"),
        PortDef("n_waypoints", "ANY"),
        PortDef("subtask_text", "TEXT"),
        PortDef("next_subtask_index", "ANY", "current_index + 1 (latched by outer iter_out)"),
        PortDef("movable_name", "TEXT"),
        PortDef("object_centric", "BOOL"),
        PortDef("prev_gripper", "BOOL",
                "Current env gripper (True=open) — seeds expand_for_settle first-flip detection"),
        PortDef("error", "TEXT"),
    ]

    @staticmethod
    def _null_out(idx: int, subtask: str = "", error: str = "") -> dict:
        return {
            "trajectory": [],
            "n_waypoints": 0,
            "subtask_text": subtask,
            "next_subtask_index": idx,
            "movable_name": "",
            "object_centric": False,
            "prev_gripper": True,
            "error": error,
        }

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        handle = str(inputs.get("runtime_handle") or "")
        runtime = _resolve_runtime(handle)
        subtask_list = list(inputs.get("subtask_list") or [])
        try:
            idx = int(inputs.get("subtask_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        if runtime is None:
            err = f"runtime_handle {handle!r} not found"
            self._self_log("error", err)
            return self._null_out(idx, error=err)
        if idx >= len(subtask_list):
            return self._null_out(idx)
        snapshot = inputs.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("error"):
            err = (
                f"snapshot error: {snapshot.get('error')}"
                if isinstance(snapshot, dict)
                else "snapshot must be wired from env observe_objects"
            )
            self._self_log("error", err)
            return self._null_out(idx, error=err)

        subtask = str(subtask_list[idx])
        self._self_log("subtask", subtask[:80])
        try:
            result = runtime.plan_subtask(subtask, snapshot)
        except Exception as e:
            log.exception("plan_subtask failed for subtask %r", subtask)
            return self._null_out(idx + 1, subtask, f"plan_subtask failed: {e!r}")
        traj = list(result.get("trajectory") or [])
        self._self_log("n_waypoints", len(traj))
        if result.get("error"):
            self._self_log("composer_error", result["error"])
        return {
            "trajectory": traj,
            "n_waypoints": len(traj),
            "subtask_text": subtask,
            "next_subtask_index": idx + 1,
            "movable_name": result.get("movable_name") or "",
            "object_centric": bool(result.get("object_centric", False)),
            "prev_gripper": bool(result.get("prev_gripper_open", True)),
            "error": result.get("error") or "",
        }


# ══════════════════════════════════════════════════════════════════════
# 7. voxposer__dispense_waypoint   (multi-scope: inner-scope body)
# ══════════════════════════════════════════════════════════════════════


class VoxPoserDispenseWaypointNode(BaseCanvasNode):
    node_type = "voxposer__dispense_waypoint"
    display_name = "VoxPoser: Dispense Waypoint"
    description = (
        "Inner-scope body. Cursor over a cached trajectory list; emits one "
        "waypoint per fire and `done=true` when exhausted. Cursor auto-resets "
        "when the input trajectory identity changes (= new outer iter)."
    )
    category = "method"
    icon = "ListChecks"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("trajectory", "LIST[ANY]"),
    ]
    output_ports = [
        PortDef("waypoint", "ANY", "Single 8-vec or null"),
        PortDef("cursor", "ANY"),
        PortDef("n_waypoints", "ANY"),
        PortDef("done", "BOOL", "True when cursor exhausted"),
        PortDef("progress", "TEXT", "'k/N waypoints'"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        traj = list(inputs.get("trajectory") or [])
        # Cheap identity hash: (length, first waypoint as tuple). New outer
        # iter brings a new trajectory → cursor resets.
        first = tuple(traj[0]) if traj else ()
        ident = (len(traj), first)
        last_ident = getattr(ctx, "last_trajectory_ident", None)
        cursor = int(getattr(ctx, "cursor", 0) or 0)
        if ident != last_ident:
            cursor = 0
            ctx.last_trajectory_ident = ident
        n = len(traj)
        if cursor >= n:
            ctx.cursor = cursor
            return {
                "waypoint": None,
                "cursor": cursor,
                "n_waypoints": n,
                "done": True,
                "progress": f"{cursor}/{n}",
            }
        waypoint = traj[cursor]
        cursor += 1
        ctx.cursor = cursor
        done = cursor >= n
        return {
            "waypoint": waypoint,
            "cursor": cursor,
            "n_waypoints": n,
            "done": done,
            "progress": f"{cursor}/{n}",
        }


# ══════════════════════════════════════════════════════════════════════
# 8. voxposer__check_waypoint_done   (multi-scope: inner-scope termination)
# ══════════════════════════════════════════════════════════════════════


class VoxPoserCheckWaypointDoneNode(BaseCanvasNode):
    node_type = "voxposer__check_waypoint_done"
    display_name = "VoxPoser: Check Waypoint Done"
    description = (
        "Inner-scope termination. Done when trajectory cursor exhausted OR "
        "episode_success already True (early-out — let outer scope close out)."
    )
    category = "method"
    icon = "Flag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("cursor_done", "BOOL"),
        PortDef(
            "episode_success", "BOOL", "Required (gates inner termination after episode_info fires)"
        ),
    ]
    output_ports = [
        PortDef("done", "BOOL"),
        PortDef("reason", "TEXT"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        if bool(inputs.get("episode_success", False)):
            return {"done": True, "reason": "episode_success"}
        if bool(inputs.get("cursor_done", False)):
            return {"done": True, "reason": "trajectory_exhausted"}
        return {"done": False, "reason": ""}


# ══════════════════════════════════════════════════════════════════════
# 9. voxposer__expand_for_settle   (graph-layer waypoint preprocessor)
# ══════════════════════════════════════════════════════════════════════


class VoxPoserExpandForSettleNode(BaseCanvasNode):
    node_type = "voxposer__expand_for_settle"
    display_name = "VoxPoser: Expand for Gripper Settle"
    description = (
        "Scan a raw 8-vec trajectory and insert post-motion 'hold' entries "
        "after every gripper transition. Restores v1 fat-node's "
        "_GRIPPER_TRANSITION_SETTLE behaviour at the graph layer: open-loop "
        "dispatch alone never gives MuJoCo enough physics ticks for the "
        "fingers to actually close on the object. Each detected flip emits "
        "a {pose, hold_steps>0} entry that env step_ee_pose "
        "interprets as 'skip OSC, force N zero-delta ticks with new gripper'."
    )
    category = "method"
    icon = "Wand"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "settle_steps",
                "number",
                default=60,
                placeholder="env ticks to hold pose + new gripper after flip (v1 uses 60)",
            ),
        ],
    )
    input_ports = [
        PortDef("trajectory", "LIST[ANY]", "List of raw 8-vec waypoints"),
        PortDef(
            "prev_gripper",
            "BOOL",
            "Current env gripper state (True=open) — seeds first-waypoint flip detection across subtasks",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "expanded",
            "LIST[ANY]",
            "List of {'pose': [8-vec], 'hold_steps': int}; hold_steps=0 for normal moves",
        ),
        PortDef("n_in", "ANY"),
        PortDef("n_out", "ANY"),
        PortDef("n_holds_inserted", "ANY"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        traj = list(inputs.get("trajectory") or [])
        try:
            settle = int(self.config.get("settle_steps", 60) or 60)
        except (TypeError, ValueError):
            settle = 60
        # Cross-fire carry: remember the LAST gripper command we emitted in
        # the previous outer iter. This is the only sound way to detect
        # subtask-boundary flips, because env_libero__get_ee_state's
        # gripper_open wire on iter_in_outer is persist=true and freezes at
        # episode start (multi-scope iterIn semantics).
        # Convention: 8-vec gripper bit follows VoxPoser (1.0=close, 0.0=open);
        # env port `gripper_open` is True when open (mapped to 0.0).
        prev_grip: float | None = getattr(ctx, "last_out_grip", None)
        if prev_grip is None:
            pg_in = inputs.get("prev_gripper")
            if pg_in is not None:
                prev_grip = 0.0 if bool(pg_in) else 1.0
        out: list[dict] = []
        holds = 0
        for wp in traj:
            try:
                arr = [float(x) for x in (wp or [])]
            except (TypeError, ValueError):
                continue
            if len(arr) != 8:
                self._self_log("skip_malformed", f"len={len(arr)}")
                continue
            cur_grip = arr[7]
            is_flip = (
                prev_grip is not None
                and ((cur_grip > 0.5) != (prev_grip > 0.5))
            )
            if is_flip:
                # Two-waypoint split on grip transition (mirror mono adapter):
                # first drive to pose with OLD grip held (no during-motion flip
                # that knocks the object lateral), then at pose flip + settle.
                arr_old = list(arr)
                arr_old[7] = float(prev_grip)
                out.append({"pose": arr_old, "hold_steps": 0})
                out.append({"pose": arr, "hold_steps": settle})
                holds += 1
            else:
                out.append({"pose": arr, "hold_steps": 0})
            prev_grip = cur_grip
        if prev_grip is not None:
            ctx.last_out_grip = prev_grip
        # Debug: dump per-waypoint gripper-bit sequence as O/C string + the
        # raw first waypoint's 8-vec, to verify what composer actually emits.
        grip_seq = "".join(
            "C" if (e.get("hold_steps") == 0 and e["pose"][7] >= 0.5)
            else ("c" if e.get("hold_steps") and e["pose"][7] >= 0.5
                  else ("o" if e.get("hold_steps") else "O"))
            for e in out
        )
        first_8 = out[0]["pose"] if out else None
        last_8 = out[-1]["pose"] if out else None
        self._self_log(
            "expanded",
            f"in={len(traj)} out={len(out)} holds={holds} last_grip={prev_grip} "
            f"grip_seq={grip_seq} first8={first_8} last8={last_8}",
        )
        return {
            "expanded": out,
            "n_in": len(traj),
            "n_out": len(out),
            "n_holds_inserted": holds,
        }


# ══════════════════════════════════════════════════════════════════════
# VoxPoserNodeSet
# ══════════════════════════════════════════════════════════════════════


class VoxPoserNodeSet(BaseNodeSet):
    """VoxPoser v2 — decomposed method nodeset (LIBERO).

    Local mode (no server_python). Method-side only — env data flows in
    via wires from env_libero__get_* nodes. The 7 LMPs share a Python
    namespace inside one run-start runtime (per-episode).
    """

    name = "voxposer"
    description = "VoxPoser v2 — decomposed method graph for LIBERO"
    parallelism = "shared"  # pure-functional method nodes, no per-worker state

    def __init__(self) -> None:
        super().__init__()

    def get_tools(self) -> list:
        return [
            VoxPoserInitNode(),
            VoxPoserPlannerLmpNode(),
            VoxPoserPlanExecutorNode(),
            VoxPoserCheckDoneNode(),
            VoxPoserPlanSubtaskNode(),
            VoxPoserDispenseWaypointNode(),
            VoxPoserCheckWaypointDoneNode(),
            VoxPoserExpandForSettleNode(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("VoxPoserNodeSet ready")

    async def shutdown(self) -> None:
        # Drop all cached runtimes.
        _RUNTIMES.clear()
