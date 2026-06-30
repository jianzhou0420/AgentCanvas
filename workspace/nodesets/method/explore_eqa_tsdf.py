"""ExploreEQA-TSDF — the voxel world-model server nodeset.

Split out of ``explore_eqa`` (2026-06-14): the TSDF voxel map is a stateful,
per-episode *world model* — the same shape as ``env_hmeqa``'s simulator, not
method reasoning. So it lives here as a first-class ``replicated`` server whose
map is held as **pure-data named numpy container states** (no live object), and
is driven by **stateless verb nodes**. The ``explore_eqa`` method nodeset now
runs stateless in the ``agentcanvas`` env and reaches these verbs over canvas
wires (env / VLM / world-model / method — four wired participants).

Why ``replicated`` (ADR-server-003): the map is per-worker stateful, so the
framework gives each worker its own subprocess + its own container instance —
no ``episode_id`` keying, no ``evict``, no hand-rolled isolation (and no race).

Why pure data, not opaque: ``TSDFPlanner``'s whole instance state is numpy /
list / scalar (``_explore_eqa_tsdf.py``, zero non-serializable handles). Each
verb binds a *transient* planner over the container's named states
(``TSDFPlanner.bind_state`` — skips ``__init__``/the O(N) grid rebuild), calls
the (unchanged) numerical method, and writes the state back via
``export_state``. The container therefore holds no live object.

Four verbs (old node → verb):
  explore_eqa_tsdf__integrate       — RGB-D + pose → fuse into the TSDF volume
  explore_eqa_tsdf__find_frontiers  — geometry → frontier *pixel* coords (the
                                      voxel ``candidates`` stay in the container
                                      for integrate_sem)
  explore_eqa_tsdf__integrate_sem   — per-frontier semantic value → semantic map
  explore_eqa_tsdf__next_pose       — frontier-weighted next pose → env step JSON

Server mode: dedicated ``hmeqa`` env (Python 3.9 + numba + open3d). Heavy imports
stay deferred (in ``_explore_eqa_tsdf.py``) so the ``agentcanvas`` parent env can
discover this class and read its ``parallelism`` ClassVar natively.

last updated: 2026-06-14
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar

import numpy as np

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)
from app.graph_def import ContainerDef, StateDef

log = logging.getLogger("agentcanvas.explore_eqa_tsdf")

_CONTAINER_ID = "explore_eqa_tsdf_map"

# The planner's pure-data state field names. MUST stay in sync with
# ``TSDFPlanner._STATE_FIELDS`` in ``_explore_eqa_tsdf.py``. Hardcoded (not
# imported) so this module loads at *discovery* time without ``workspace`` on
# sys.path — ``TSDFPlanner`` is imported lazily inside the verbs (at execution
# time ``workspace`` is importable; same pattern as explore_eqa/tooleqa).
_STATE_NAMES = (
    "tsdf_vol",
    "weight_vol",
    "color_vol",
    "val_vol",
    "weight_val_vol",
    "explore_vol",
    "vol_bnds",
    "vol_dim",
    "vol_origin",
    "voxel_size",
    "trunc_margin",
    "color_const",
    "min_height_voxel",
    "vox_coords",
    "cam_pts_pre",
    "init_points",
    "candidates",
    "target_point",
    "target_direction",
    "max_point",
    "cur_point",
    "island",
    "unexplored",
    "unoccupied",
    "occupied",
    "unexplored_neighbors",
)


def _TSDFPlanner():
    """Lazy import — workspace is on sys.path at node-execution time, not at
    module-discovery time (see ``_STATE_NAMES``)."""
    from workspace.nodesets.method._explore_eqa_tsdf import TSDFPlanner

    return TSDFPlanner


# Defaults mirror explore-eqa/cfg/vlm_exp.yaml (kept identical to the old nodes).
_DEFAULT_TSDF_VOXEL_SIZE = 0.1
_DEFAULT_INIT_CLEARANCE = 0.5
_DEFAULT_MARGIN_H_RATIO = 0.6
_DEFAULT_MARGIN_W_RATIO = 0.25
_DEFAULT_BLACK_PIXEL_RATIO = 0.5


# ══════════════════════════════════════════════════════════════════════
# Container access + pure-data state plumbing
# ══════════════════════════════════════════════════════════════════════


def _container(ctx):
    """Return the nodeset-owned container, injected by-reference in server mode.
    This nodeset is server-only (replicated); raise loudly otherwise."""
    containers = getattr(ctx, "containers", None) or {}
    c = containers.get(_CONTAINER_ID)
    if c is None:
        raise RuntimeError(
            f"explore_eqa_tsdf: owned container '{_CONTAINER_ID}' not injected — "
            "this nodeset is server-only; load it with mode=server."
        )
    return c


def _read_state(container) -> dict:
    """Read the full planner state dict from the container (keyless — replicated
    gives per-worker isolation, so no episode_id key)."""
    return {name: container.read(name) for name in _STATE_NAMES}


def _write_state(container, st: dict) -> None:
    """Write the FULL planner state dict back (used only on first build)."""
    for name, val in st.items():
        container.write(name, val)


# Per-verb write-back sets — each verb writes back ONLY the fields it modifies.
# Two-part fix for a write-write race: the graph now feeds a *per-step* ``order``
# token (iter_in.step) so the executor serializes integrate→find→integrate_sem→
# next_pose each step; AND each verb writes a disjoint field set so that even if
# two verbs overlap, integrate can never clobber find_frontiers's fresh
# ``candidates`` (which had made integrate_sem's len-assert fail ~4% of steps).
_WR_INTEGRATE = ("tsdf_vol", "weight_vol", "color_vol", "explore_vol")
_WR_FIND = (
    "candidates",
    "cur_point",
    "island",
    "unexplored",
    "unoccupied",
    "occupied",
    "unexplored_neighbors",
)
_WR_SEM = ("val_vol", "weight_val_vol")
_WR_NEXT = (
    "target_point",
    "target_direction",
    "max_point",
    "unexplored_neighbors",
    "unoccupied",
)


def _write_fields(container, planner, names) -> None:
    """Write back only ``names`` (the fields this verb modified)."""
    st = planner.export_state()
    for n in names:
        container.write(n, st[n])


def _get_or_build(container, tsdf_bnds, pose_normal, voxel_size, init_clearance):
    """Bind the worker's planner over container state; build + store it on first
    call. ``init_clearance * 2`` mirrors the old ``_get_or_build_planner``."""
    TSDFPlanner = _TSDFPlanner()
    if container.read("tsdf_vol") is not None:
        return TSDFPlanner.bind_state(_read_state(container))

    bnds = np.asarray(tsdf_bnds, dtype=np.float64)
    planner = TSDFPlanner(
        vol_bnds=bnds,
        voxel_size=float(voxel_size),
        floor_height_offset=0,
        pts_init=np.asarray(pose_normal, dtype=np.float64),
        init_clearance=float(init_clearance) * 2,
    )
    _write_state(container, planner.export_state())
    log.info(
        "TSDFPlanner built vol_dim=%s voxel_size=%s",
        planner._vol_dim.tolist(),
        voxel_size,
    )
    return planner


# ══════════════════════════════════════════════════════════════════════
# Verb 1: Integrate (RGB-D → TSDF volume)
# ══════════════════════════════════════════════════════════════════════


class TSDFIntegrateNode(BaseCanvasNode):
    """Integrate one RGB-D frame into the worker's TSDF volume."""

    node_type: ClassVar[str] = "explore_eqa_tsdf__integrate"
    display_name: ClassVar[str] = "TSDF: Integrate"
    description: ClassVar[str] = "RGB-D + camera pose → integrate into the TSDF volume"
    category: ClassVar[str] = "tool"
    icon: ClassVar[str] = "Box"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("voxel_size", "number", default=_DEFAULT_TSDF_VOXEL_SIZE),
        ConfigField("init_clearance", "number", default=_DEFAULT_INIT_CLEARANCE),
        ConfigField("margin_h_ratio", "number", default=_DEFAULT_MARGIN_H_RATIO),
        ConfigField("margin_w_ratio", "number", default=_DEFAULT_MARGIN_W_RATIO),
        ConfigField("black_pixel_ratio", "number", default=_DEFAULT_BLACK_PIXEL_RATIO),
    ]

    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "RGB observation"),
        PortDef("depth", "DEPTH", "Depth map (HxW float)"),
        PortDef("cam_pose", "ANY", "4x4 TSDF-frame extrinsic"),
        PortDef("cam_intr", "ANY", "3x3 intrinsics"),
        PortDef("pose_normal", "ANY", "3-vector normal-frame position"),
        PortDef("tsdf_bnds", "ANY", "3x2 TSDF bounds (only used on first call)"),
        PortDef("order", "ANY", "Pass-through ordering token (e.g. episode_id)"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("order", "ANY", "Pass-through ordering token for downstream verbs"),
        PortDef("ok", "BOOL", "Integration success"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        voxel_size = float(cfg.get("voxel_size", _DEFAULT_TSDF_VOXEL_SIZE))
        init_clearance = float(cfg.get("init_clearance", _DEFAULT_INIT_CLEARANCE))
        margin_h_ratio = float(cfg.get("margin_h_ratio", _DEFAULT_MARGIN_H_RATIO))
        margin_w_ratio = float(cfg.get("margin_w_ratio", _DEFAULT_MARGIN_W_RATIO))
        black_pixel_ratio = float(cfg.get("black_pixel_ratio", _DEFAULT_BLACK_PIXEL_RATIO))

        order = inputs.get("order")
        rgb = np.asarray(inputs.get("rgb"))
        depth = np.asarray(inputs.get("depth"))
        cam_pose = np.asarray(inputs.get("cam_pose"), dtype=np.float64)
        cam_intr = np.asarray(inputs.get("cam_intr"), dtype=np.float64)
        pose_normal = np.asarray(inputs.get("pose_normal"), dtype=np.float64)
        tsdf_bnds = inputs.get("tsdf_bnds")

        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if depth.ndim == 3:
            depth = depth.squeeze()

        container = _container(ctx)
        if tsdf_bnds is None and container.read("tsdf_vol") is None:
            self._self_log("error", "missing tsdf_bnds for first-call init")
            return {"order": order, "ok": False}

        planner = _get_or_build(container, tsdf_bnds, pose_normal, voxel_size, init_clearance)

        img_h = int(rgb.shape[0]) if rgb.ndim >= 2 else 480
        img_w = int(rgb.shape[1]) if rgb.ndim >= 2 else 640
        margin_h = int(margin_h_ratio * img_h)
        margin_w = int(margin_w_ratio * img_w)

        num_black = int(np.sum(np.sum(rgb, axis=-1) == 0)) if rgb.ndim == 3 else 0
        if num_black > black_pixel_ratio * img_w * img_h:
            self._self_log("skipped_black_frame", True)
            return {"order": order, "ok": False}

        def _integrate():
            return planner.integrate(
                color_im=rgb.astype(np.uint8),
                depth_im=depth.astype(np.float32),
                cam_intr=cam_intr,
                cam_pose=cam_pose,
                obs_weight=1.0,
                margin_h=margin_h,
                margin_w=margin_w,
            )

        try:
            await asyncio.to_thread(_integrate)
        except Exception as exc:
            log.exception("TSDF integrate failed")
            self._self_log("error", str(exc))
            return {"order": order, "ok": False}

        _write_fields(container, planner, _WR_INTEGRATE)
        self._self_log("integrated", True)
        return {"order": order, "ok": True}


# ══════════════════════════════════════════════════════════════════════
# Verb 2: FindFrontiers (geometry → frontier pixel coords)
# ══════════════════════════════════════════════════════════════════════


class FindFrontiersNode(BaseCanvasNode):
    """Find candidate frontier points within the current view; return their
    *pixel* coords for the method to label. The matching *voxel* ``candidates``
    are written into the container for ``integrate_sem`` to read back."""

    node_type: ClassVar[str] = "explore_eqa_tsdf__find_frontiers"
    display_name: ClassVar[str] = "TSDF: Find Frontiers"
    description: ClassVar[str] = (
        "Find frontier candidate points (returns pixel coords; stores voxel candidates)"
    )
    category: ClassVar[str] = "tool"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("num_prompt_points", "integer", default=3),
        ConfigField("img_width", "integer", default=640),
        ConfigField("img_height", "integer", default=480),
    ]

    input_ports: ClassVar[list] = [
        PortDef("pose_normal", "ANY", "3-vector normal-frame position"),
        PortDef("cam_intr", "ANY", "3x3 camera intrinsics"),
        PortDef("cam_pose", "ANY", "4x4 TSDF-frame camera extrinsic"),
        PortDef("order", "ANY", "Pass-through ordering token"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("candidates_pix", "ANY", "Frontier candidate pixel coords (N,2)"),
        PortDef(
            "candidates", "ANY", "Frontier VOXEL coords (N,2) — paired with sv into integrate_sem"
        ),
        PortDef("num_candidates", "ANY", "Number of frontier candidates"),
        PortDef("order", "ANY", "Pass-through ordering token"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        num_prompt = int(cfg.get("num_prompt_points", 3))
        img_w = int(cfg.get("img_width", 640))
        img_h = int(cfg.get("img_height", 480))

        order = inputs.get("order")
        pts_normal = np.asarray(inputs.get("pose_normal"), dtype=np.float64)
        cam_intr = np.asarray(inputs.get("cam_intr"), dtype=np.float64)
        cam_pose = np.asarray(inputs.get("cam_pose"), dtype=np.float64)

        container = _container(ctx)
        if container.read("tsdf_vol") is None:
            self._self_log("error", "no TSDF volume built yet")
            return {"candidates_pix": [], "num_candidates": 0, "order": order}

        planner = _TSDFPlanner().bind_state(_read_state(container))

        def _find():
            return planner.find_prompt_points_within_view(
                pts=pts_normal,
                im_w=img_w,
                im_h=img_h,
                cam_intr=cam_intr,
                cam_pose=cam_pose,
                num_prompt_points=num_prompt,
            )

        candidates_pix = await asyncio.to_thread(_find)
        # Voxel candidates ride the wire alongside sv (same firing) so
        # integrate_sem never reads a container value a later find desynced.
        cand_vox = np.asarray(getattr(planner, "candidates", np.empty((0, 2))))
        # Persist view caches for next_pose (candidates also kept for safety).
        _write_fields(container, planner, _WR_FIND)

        actual_n = len(candidates_pix)
        self._self_log("num_candidates", actual_n)
        return {
            "candidates_pix": np.asarray(candidates_pix),
            "candidates": cand_vox,
            "num_candidates": actual_n,
            "order": order,
        }


# ══════════════════════════════════════════════════════════════════════
# Verb 3: IntegrateSem (per-frontier semantic value → semantic map)
# ══════════════════════════════════════════════════════════════════════


class IntegrateSemNode(BaseCanvasNode):
    """Integrate per-frontier semantic value (sv = lsv*gsv, computed in the
    method core) into the TSDF semantic-value map."""

    node_type: ClassVar[str] = "explore_eqa_tsdf__integrate_sem"
    display_name: ClassVar[str] = "TSDF: Integrate Semantic"
    description: ClassVar[str] = "Integrate per-frontier semantic value into the TSDF map"
    category: ClassVar[str] = "tool"
    icon: ClassVar[str] = "MapPin"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    input_ports: ClassVar[list] = [
        PortDef("sem_pix", "ANY", "Per-frontier semantic value sv (length == num_candidates)"),
        PortDef("candidates", "ANY", "Voxel candidates from find_frontiers (paired with sem_pix)"),
        PortDef("num_candidates", "ANY", "Number of frontier candidates"),
        PortDef("skip", "BOOL", "True iff frontier scoring was skipped"),
        PortDef("order", "ANY", "Pass-through ordering token"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("ok", "BOOL", "Integrated semantic into TSDF"),
        PortDef("order", "ANY", "Pass-through ordering token"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        order = inputs.get("order")
        sv_in = inputs.get("sem_pix") or []
        n = int(inputs.get("num_candidates", 0) or 0)
        skip = bool(inputs.get("skip", False))

        container = _container(ctx)
        if skip or n == 0 or container.read("tsdf_vol") is None:
            self._self_log("skipped", True)
            return {"ok": False, "order": order}

        planner = _TSDFPlanner().bind_state(_read_state(container))
        sv = np.asarray(sv_in, dtype=np.float64)

        # Use the candidates passed ALONGSIDE sv on the wire (same find firing)
        # — NOT the container's, which a later/concurrent find write desyncs
        # from the in-flight sv (was the integrate_sem len-assert failure).
        cand_in = inputs.get("candidates")
        if cand_in is not None:
            planner.candidates = np.asarray(cand_in)

        _cand = getattr(planner, "candidates", None)
        _ncand = 0 if _cand is None else len(_cand)
        if _ncand != int(sv.shape[0]):
            # Should no longer happen with the wired candidates; skip safely.
            self._self_log("len_mismatch", {"candidates": _ncand, "sem_pix": int(sv.shape[0])})
            return {"ok": False, "order": order}

        def _integrate_sem():
            planner.integrate_sem(sem_pix=sv, radius=1.0, obs_weight=1.0)

        try:
            await asyncio.to_thread(_integrate_sem)
        except Exception as exc:
            log.exception("integrate_sem failed")
            self._self_log("error", str(exc))
            return {"ok": False, "order": order}

        _write_fields(container, planner, _WR_SEM)
        self._self_log("integrated_sem", True)
        return {"ok": True, "order": order}


# ══════════════════════════════════════════════════════════════════════
# Verb 4: NextPose (frontier-weighted next pose → env step JSON)
# ══════════════════════════════════════════════════════════════════════


class NextPoseNode(BaseCanvasNode):
    """Pick the next pose via frontier-weighted sampling; emit env step JSON."""

    node_type: ClassVar[str] = "explore_eqa_tsdf__next_pose"
    display_name: ClassVar[str] = "TSDF: Next Pose"
    description: ClassVar[str] = (
        "Frontier-weighted next-pose planner; emits env_hmeqa__step action JSON"
    )
    category: ClassVar[str] = "tool"
    icon: ClassVar[str] = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("min_random_init_steps", "integer", default=2),
    ]

    input_ports: ClassVar[list] = [
        PortDef("pose_normal", "ANY", "Current 3-vector normal-frame position"),
        PortDef("angle", "ANY", "Current yaw"),
        PortDef("step_index", "ANY", "Current step (used to gate random walk)"),
        PortDef("order", "ANY", "Pass-through ordering token"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("action", "TEXT", 'JSON: {"position_normal": [x, y], "angle": float}'),
        PortDef("next_pose_normal", "ANY", "2-vector (x, y) normal-frame position"),
        PortDef("next_angle", "ANY", "Yaw in radians"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        min_random_init_steps = int(cfg.get("min_random_init_steps", 2))

        pose_normal = np.asarray(inputs.get("pose_normal"), dtype=np.float64)
        angle = float(inputs.get("angle", 0.0) or 0.0)
        step_index = int(inputs.get("step_index", 0) or 0)

        container = _container(ctx)
        if container.read("tsdf_vol") is None:
            self._self_log("error", "no TSDF volume built yet")
            fallback_action = json.dumps(
                {
                    "position_normal": [float(pose_normal[0]), float(pose_normal[1])],
                    "angle": angle + 0.1,
                }
            )
            return {
                "action": fallback_action,
                "next_pose_normal": [float(pose_normal[0]), float(pose_normal[1])],
                "next_angle": angle + 0.1,
            }

        planner = _TSDFPlanner().bind_state(_read_state(container))

        def _plan():
            return planner.find_next_pose(
                pts=pose_normal,
                angle=angle,
                flag_no_val_weight=step_index < min_random_init_steps,
            )

        next_point_normal, next_yaw, _next_point_vox = await asyncio.to_thread(_plan)
        # Persist commit state (target_point/direction/max_point) for next step.
        _write_fields(container, planner, _WR_NEXT)

        px = float(next_point_normal[0])
        py = float(next_point_normal[1])
        nyaw = float(next_yaw)
        action = json.dumps({"position_normal": [px, py], "angle": nyaw})

        self._self_log("next_pose_normal", [px, py])
        self._self_log("next_angle", nyaw)
        return {
            "action": action,
            "next_pose_normal": [px, py],
            "next_angle": nyaw,
        }


# ══════════════════════════════════════════════════════════════════════
# NodeSet registration
# ══════════════════════════════════════════════════════════════════════


class ExploreEQATSDFNodeSet(BaseNodeSet):
    """ExploreEQA-TSDF world-model server — replicated, per-worker TSDF map held
    as pure-data named numpy container states (no live object)."""

    name: ClassVar[str] = "explore_eqa_tsdf"
    description: ClassVar[str] = (
        "TSDF voxel world-model server (integrate / find_frontiers / "
        "integrate_sem / next_pose) — replicated, pure-data container states"
    )
    # Stateful, per-worker world model → framework gives per-worker subprocess +
    # container instance (ADR-server-003). Read natively by registry._get_parallelism.
    parallelism: ClassVar[str] = "replicated"
    server_python: ClassVar[str] = os.environ.get(
        "HMEQA_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-hmeqa/bin/python")
    )

    def get_tools(self) -> list:
        return [
            TSDFIntegrateNode(),
            FindFrontiersNode(),
            IntegrateSemNode(),
            NextPoseNode(),
        ]

    def get_containers(self) -> list[ContainerDef]:
        # One container; the planner's full state as pure-data named numpy
        # states (lastWrite, ANY — non-opaque, in-process by reference). Keyless:
        # replicated gives per-worker isolation, so no episode_id key / no evict.
        states = {name: StateDef(type="lastWrite", value_type="ANY") for name in _STATE_NAMES}
        return [
            ContainerDef(
                id=_CONTAINER_ID,
                label="ExploreEQA TSDF Map",
                states=states,
            )
        ]
