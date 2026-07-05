from __future__ import annotations

"""AO-Planner — Affordances-Oriented Planning for continuous VLN (Chen et al., AAAI 2025).

Port of github.com/chen-judge/AO-Planner @ 719f42a1 onto env_habitat (VLN-CE).
Faithful two-VLM Visual Affordances Prompting (VAP) + pure-foundation-model
framing (the released code's supervised ETPNav TRM waypoint predictor +
ZeroShotGraphMap alignment are intentionally dropped — see
.claude/memory/nodeset/project_aoplanner_port.md).

This module is built incrementally:

  M2 (this commit) — the VAP geometry/overlay method nodes (per single view):
    aoplanner__sample_waypoints  (ground_mask_b64) -> candidate pixel grid
    aoplanner__annotate_markers  (image_b64, points[, labels]) -> annotated image
    aoplanner__project_waypoints (candidate_pixels, depth, intrinsics, heading)
                                 -> per-candidate relative-polar (angle, distance)

  M3 (next) — the VLM#1 proposer prompt/parse + the SmartWay-cloned decider
    (update_topology / build_action_options / assemble_prompt / build_images /
     parse_response / resolve_action / update_history) + a 4-view aggregator.

Geometry convention (verified against the repo's explore-eqa unprojection +
env_habitat, 2026-06-16): habitat camera is +X right / +Y up / -Z forward,
depth is planar Z-depth in metres (panorama depth_raw_base64 is 16-bit
millimetres -> /1000). step_hightolow yaw is about +Y; panorama heading_deg
uses the same +Y quaternion. The left/right SIGN of the in-view bearing is the
one thing that must be confirmed against a live env (M5 graph smoke); it is
exposed as the `bearing_sign` / `heading_sign` config knobs.

Load:  POST /api/components/nodesets/aoplanner/load

last updated: 2026-06-16
"""

import base64
import contextlib
import io
import json
import logging
import math
import os
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef
from workspace.nodesets.method.aoplanner import _prompts

log = logging.getLogger("agentcanvas.aoplanner")


def _read(gs: Any, key: str, default: Any) -> Any:
    if gs is None:
        return default
    try:
        v = gs.read(key)
        return v if v is not None else default
    except Exception:
        return default


def _write(gs: Any, key: str, value: Any) -> None:
    if gs is None:
        return
    with contextlib.suppress(Exception):
        gs.write(key, value)


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_DEFAULT_FONT_PATH = os.path.join(
    _REPO_ROOT, "data", "hm3d", "hmeqa", "Open_Sans", "static", "OpenSans-Regular.ttf"
)
_FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ══════════════════════════════════════════════════════════════════════
# decode helpers
# ══════════════════════════════════════════════════════════════════════


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


def _decode_mask(b64: str) -> np.ndarray:
    """Decode a base64 mask PNG (bool/grayscale) -> HxW bool array."""
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"))
    return arr > 127


def _union_sam_masks(result_json: str):
    """Union every mask_b64 in a model_sam result JSON -> HxW bool array (or None)."""
    try:
        data = json.loads(result_json)
    except Exception:
        return None
    out = None
    for mk in data.get("masks") or []:
        b64 = mk.get("mask_b64") if isinstance(mk, dict) else None
        if not b64:
            continue
        mask = _decode_mask(b64)
        out = mask if out is None else (out | mask)
    return out


def _decode_depth_m(
    b64: str,
    depth_is_mm: bool,
    depth_scale: float = 1.0,
    target_wh: tuple | None = None,
) -> np.ndarray:
    """Decode a base64 depth PNG -> HxW float32 metres.

    env_habitat observe_panorama emits depth_raw_base64 via encode_depth_raw_base64,
    which does ``clip(d*1000)`` -> uint16 "mm". For VLN-CE that ``d`` is the habitat
    DEPTH_SENSOR observation, which is NORMALIZED to [0,1] (NORMALIZE_DEPTH=True,
    MAX_DEPTH=10.0 — vlnce_task.yaml inherits habitat-lab defaults), NOT metres,
    despite the encoder's docstring. So the /1000 decode recovers the normalized
    value and the true metric depth needs an extra ``* depth_scale`` (= MAX_DEPTH
    = 10.0 for R2R-CE). Without it every distance is 10x too small and the agent
    crawls ~0.25 m/step (root cause of the M6 SR-0 run 20260616_233000). This is a
    LOCAL fix; the env encoder's "absolute metric depth" contract is wrong for ALL
    consumers (Open-Nav / discussnav / SpatialBot) — env-wide fix is a separate TODO.

    target_wh resizes the depth to the RGB/intrinsics resolution: the panorama RGB
    (and the candidate pixels sampled on its ground mask) are 224x224 but the
    DEPTH_SENSOR is 256x256, so indexing depth[v,u] with RGB-space (u,v) is
    misaligned ~12.5%. NEAREST avoids blending across depth discontinuities.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    arr = np.asarray(im).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if depth_is_mm:
        arr = arr / 1000.0
    if depth_scale != 1.0:
        arr = arr * float(depth_scale)
    if target_wh is not None:
        tw, th = int(target_wh[0]), int(target_wh[1])
        if (arr.shape[1], arr.shape[0]) != (tw, th):
            arr = np.asarray(
                Image.fromarray(arr, mode="F").resize((tw, th), Image.NEAREST), dtype=np.float32
            )
    return arr


def _load_font(size: int):
    from PIL import ImageFont

    for path in (_DEFAULT_FONT_PATH, _FALLBACK_FONT_PATH):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════
# Pure geometry — pixel -> relative-polar (testable, no canvas deps)
# ══════════════════════════════════════════════════════════════════════


def _rotate_vec(q: tuple, v: tuple) -> tuple:
    """Rotate vector v by unit quaternion q=[x,y,z,w] (Rodrigues, pure Python)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    return (
        vx + 2 * (qw * cx + qy * cz - qz * cy),
        vy + 2 * (qw * cy + qz * cx - qx * cz),
        vz + 2 * (qw * cz + qx * cy - qy * cx),
    )


def _qmul(q1: tuple, q2: tuple) -> tuple:
    """Quaternion product q1 * q2, both in [x,y,z,w] format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _heading_from_q(q: tuple) -> float:
    """Agent yaw from quaternion q=[x,y,z,w].

    Verbatim logic of upstream ``heading_from_quaternion`` (utils.py:843-849):
    rotate camera-forward [0,0,-1] to world frame, then atan2(x, -z).
    """
    fwd = _rotate_vec(q, (0.0, 0.0, -1.0))
    return math.atan2(fwd[0], -fwd[2])


def _project_pixel_world(
    u: float,
    v: float,
    d: float,
    fx: float,
    fy: float,
    cx_i: float,
    cy_i: float,
    position: list,
    agent_rotation: list,
    view_heading_deg: float = 0.0,
) -> tuple[float, float]:
    """Pixel + depth + camera pose → (angle_rad, distance_m) polar hop.

    Implements upstream ``pixel_to_world`` (utils.py:87-130) + ``_calculate_vp_rel_pos``
    (environments_llm.py:27-43) in the backend process.

    The view rotation = agent_rotation * Q_y(view_heading_deg) accounts for the
    panorama's per-view camera rotation before projecting to world frame.

    angle convention: ``agent_h - target_h`` matches existing ``bearing_sign=-1``
    (positive angle = turn left, negative = turn right — confirmed by smoke tests).
    """
    # Compose view rotation: agent_rotation * Q_y(view_heading_deg)
    view_rad = math.radians(view_heading_deg)
    view_q = (0.0, math.sin(view_rad / 2.0), 0.0, math.cos(view_rad / 2.0))
    view_rot = _qmul(tuple(agent_rotation), view_q)

    # Camera frame (habitat: +x right, +y up, -z forward; depth along -z)
    x_c = (u - cx_i) / fx * d
    y_c = -((v - cy_i) / fy * d)  # image y↓ → camera y↑
    z_c = -d

    # Camera → world frame using view rotation
    wx, _wy, wz = _rotate_vec(view_rot, (x_c, y_c, z_c))
    dx = wx  # relative to agent position
    dz = wz
    dist = math.hypot(dx, dz)

    # Polar relative to agent heading (agent_rotation, not view_rotation)
    target_h = math.atan2(dx, -dz)
    agent_h = _heading_from_q(tuple(agent_rotation))
    # agent_h - target_h (not target_h - agent_h) to match bearing_sign=-1 convention
    angle = agent_h - target_h
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle, dist


def _project_pixels_world(
    pixels: list,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx_i: float,
    cy_i: float,
    position: list,
    agent_rotation: list,
    view_heading_deg: float = 0.0,
    min_depth: float = 0.1,
    max_depth: float = 30.0,
) -> list:
    """World-coordinate pixel unprojection (upstream-faithful).

    Replaces the heading_sign/bearing_sign approximation when camera pose is
    available from ``observe_camera_pose``.
    """
    h, w = depth_m.shape[:2]
    out: list = []
    for idx, pt in enumerate(pixels):
        u = round(float(pt[0]))
        v = round(float(pt[1]))
        if not (0 <= u < w and 0 <= v < h):
            out.append(
                {"idx": idx, "u": u, "v": v, "angle": None, "distance": None, "valid": False}
            )
            continue
        d = float(depth_m[v, u])
        if not math.isfinite(d) or d < min_depth or d > max_depth:
            out.append(
                {"idx": idx, "u": u, "v": v, "angle": None, "distance": None, "valid": False}
            )
            continue
        angle, distance = _project_pixel_world(
            u, v, d, fx, fy, cx_i, cy_i, position, agent_rotation, view_heading_deg
        )
        out.append(
            {"idx": idx, "u": u, "v": v, "angle": angle, "distance": distance, "valid": True}
        )
    return out


def _project_pixels(
    pixels: list,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    view_heading_deg: float = 0.0,
    heading_sign: float = 1.0,
    bearing_sign: float = -1.0,
    min_depth: float = 0.1,
    max_depth: float = 30.0,
) -> list:
    """Unproject pixels to per-candidate relative-polar (angle, distance).

    Habitat camera frame (+X right, +Y up, -Z forward); depth is planar Z-depth:
        x_cam = (u - cx)/fx * d        (right = +X)
        y_cam = (cy - v)/fy * d        (up = +Y; image row v grows downward)
        z_cam = -d                     (forward = -Z)
    Ground-plane distance drops the vertical component (camera height cancels):
        distance = hypot(x_cam, d)
    In-view bearing off the optical axis toward +X:
        beta = atan2(x_cam, d)
    A +Y (CCW/left) step_hightolow yaw turns -Z forward toward -X, so a target at
    camera +X (right) needs a NEGATIVE yaw -> bearing_sign defaults to -1.
    view_heading_deg shares the same +Y quaternion as step_hightolow, so it adds
    directly (heading_sign=+1). Both signs are exposed for empirical reconciliation.

    Returns a list of {idx,u,v,angle,distance,valid}; invalid (no/clipped depth)
    entries carry valid=False (caller drops them).
    """
    h, w = depth_m.shape[:2]
    view_rad = math.radians(float(view_heading_deg))
    out: list = []
    for idx, pt in enumerate(pixels):
        u = round(float(pt[0]))
        v = round(float(pt[1]))
        if not (0 <= u < w and 0 <= v < h):
            out.append(
                {"idx": idx, "u": u, "v": v, "angle": None, "distance": None, "valid": False}
            )
            continue
        d = float(depth_m[v, u])
        if not math.isfinite(d) or d < min_depth or d > max_depth:
            out.append(
                {"idx": idx, "u": u, "v": v, "angle": None, "distance": None, "valid": False}
            )
            continue
        x_cam = (u - cx) / fx * d
        beta = math.atan2(x_cam, d)
        distance = math.hypot(x_cam, d)
        angle = heading_sign * view_rad + bearing_sign * beta
        out.append(
            {
                "idx": idx,
                "u": u,
                "v": v,
                "angle": float(angle),
                "distance": float(distance),
                "valid": True,
            }
        )
    return out


def _draw_markers(
    rgb: np.ndarray, points: list, labels: list, radius: int, font_size: int
) -> np.ndarray:
    """Burn numbered/filled markers onto an RGB array (lifted from
    explore_eqa.py:386-407, generalized off the A/B/C/D cap to arbitrary labels)."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    for i, pt in enumerate(points):
        px = round(float(pt[0]))
        py = round(float(pt[1]))
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(220, 30, 30),
            outline=(255, 255, 255),
            width=2,
        )
        label = str(labels[i]) if i < len(labels) else str(i)
        draw.text((px, py), label, font=font, fill=(255, 255, 255), anchor="mm")
    return np.asarray(pil)


# ══════════════════════════════════════════════════════════════════════
# Node: sample_waypoints
# ══════════════════════════════════════════════════════════════════════


class SampleWaypointsNode(BaseCanvasNode):
    """Lay a regular pixel grid over a ground mask and keep in-mask points.

    Mirrors AO-Planner ``sample_points`` (llm/grounded_sam_Gemini.py): a grid
    every ``grid_px`` pixels, keep points inside the navigable-ground mask, and
    prepend the agent's foot point ``(W//2, H-5)`` as candidate 0. The full grid
    is handed to VLM#1 (the waypoint proposer), which selects the affordances —
    this node does NOT reduce the set.
    """

    node_type: ClassVar[str] = "aoplanner__sample_waypoints"
    display_name: ClassVar[str] = "AO-Planner: Sample Waypoints"
    description: ClassVar[str] = "Grid-sample candidate pixels inside a navigable-ground mask"
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid3x3"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField("grid_px", "text", "Grid spacing in pixels (AO-Planner 50)", default="50"),
            ConfigField(
                "include_start",
                "boolean",
                "Prepend foot point (W//2, H-5) as candidate 0",
                default=True,
            ),
        ],
    )
    input_ports = [
        PortDef(
            "sam_result",
            "TEXT",
            "model_sam result JSON; all masks unioned into the ground mask "
            "(the composition path since the ground_mask eviction 2026-07-05)",
        ),
        PortDef(
            "ground_mask_b64",
            "TEXT",
            "Optional: base64 PNG of a pre-unioned ground mask (takes precedence)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "candidate_pixels",
            "TEXT",
            "JSON list of [u,v] pixel coords (index 0 = foot point if included)",
        ),
        PortDef("count", "ANY", "Number of candidate pixels"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        b64 = inputs.get("ground_mask_b64") or ""
        sam_result = inputs.get("sam_result") or ""
        cfg = getattr(self, "config", None) or {}
        grid_px = max(1, int(float(cfg.get("grid_px", 50) or 50)))
        include_start = bool(cfg.get("include_start", True))

        if b64:
            mask = _decode_mask(b64)
        elif sam_result:
            mask = _union_sam_masks(sam_result)
        else:
            mask = None
        if mask is None:
            return {"candidate_pixels": "[]", "count": 0}
        h, w = mask.shape[:2]
        pts: list = []
        if include_start:
            pts.append([w // 2, max(0, h - 5)])
        # Match upstream sample_points (grounded_sam_Gemini.py:174-177): rows counted
        # UP from the bottom, i/j from 1 (skips the u=0 column, the bottom row, and
        # the top row), so the candidate SET + id→pixel order mirror upstream
        # (was range(0,·) from the top — the §3E grid-enumeration divergence).
        height_num = h // grid_px
        width_num = w // grid_px
        for i in range(1, height_num):
            v = h - 1 - i * grid_px
            for j in range(1, width_num):
                u = j * grid_px
                if 0 <= v < h and 0 <= u < w and mask[v, u]:
                    pts.append([u, v])
        # Upstream early-abort (process_image:354-362): skip VLM#1 when candidates
        # are unusable — >40 means face-wall (excessive clutter), <=1 means only the
        # foot point (no navigable ground found). Return empty so the view is skipped.
        if len(pts) > 40 or len(pts) <= 1:
            self._self_log("n_candidates", 0)
            self._self_log("early_abort", f"count={len(pts)}")
            return {"candidate_pixels": "[]", "count": 0}
        self._self_log("n_candidates", len(pts))
        return {"candidate_pixels": json.dumps(pts), "count": len(pts)}


# ══════════════════════════════════════════════════════════════════════
# Node: annotate_markers
# ══════════════════════════════════════════════════════════════════════


class AnnotateMarkersNode(BaseCanvasNode):
    """Burn numbered markers onto an RGB image at supplied pixel coords.

    Generic visual-prompting overlay (AO-Planner ``vis_candidates`` /
    ``vis_ghost_nodes``): a filled circle + label at each point. Used both for
    the VAP candidate grid (fed to VLM#1) and the PathAgent ghost-node options
    (fed to VLM#2). Arbitrary integer/string labels — no A/B/C/D cap.
    """

    node_type: ClassVar[str] = "aoplanner__annotate_markers"
    display_name: ClassVar[str] = "AO-Planner: Annotate Markers"
    description: ClassVar[str] = (
        "Draw numbered markers on an RGB image at given pixel coords (visual prompting)"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Highlighter"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField("radius", "text", "Marker circle radius (px)", default="10"),
            ConfigField("font_size", "text", "Label font size (px)", default="20"),
            ConfigField(
                "skip_first",
                "boolean",
                "Skip points[0] (the foot anchor) — upstream vis_candidates never draws it",
                default=False,
            ),
            ConfigField(
                "label_start",
                "text",
                "First auto-label value (upstream grid labels start at 1)",
                default="0",
            ),
        ],
    )
    input_ports = [
        PortDef("image_b64", "TEXT", "Base64 PNG/JPEG RGB image"),
        PortDef("points", "TEXT", "JSON list of [u,v] pixel coords to mark"),
        PortDef("labels", "TEXT", "Optional JSON list of labels (default 0..N-1)", optional=True),
    ]
    output_ports = [
        PortDef("annotated_b64", "TEXT", "Base64 PNG of the annotated RGB image"),
        PortDef("annotated_image", "IMAGE", "Annotated RGB as np.uint8 array (for llmCall.rgb)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from PIL import Image

        img_b64 = inputs.get("image_b64") or ""
        if not img_b64:
            return {"annotated_b64": "", "annotated_image": None}
        try:
            points = json.loads(inputs.get("points") or "[]")
        except Exception:
            points = []
        labels_raw = inputs.get("labels")
        labels: list = []
        if labels_raw:
            try:
                labels = json.loads(labels_raw)
            except Exception:
                labels = []
        cfg = getattr(self, "config", None) or {}
        radius = max(1, int(float(cfg.get("radius", 10) or 10)))
        font_size = max(6, int(float(cfg.get("font_size", 20) or 20)))
        skip_first = bool(cfg.get("skip_first", False))
        label_start = int(float(cfg.get("label_start", 0) or 0))

        # Upstream vis_candidates(multi_start=True) draws only the GRID points,
        # labelled 1..N — the foot anchor (candidate 0) is inserted into the id
        # space afterwards and never shown to VLM#1 (grounded_sam_Gemini.py:346-348).
        if skip_first and points:
            points = points[1:]
        if not labels:
            labels = [str(label_start + i) for i in range(len(points))]

        rgb = _decode_rgb(img_b64)
        annotated = _draw_markers(rgb, points, labels, radius, font_size)
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        out = base64.b64encode(buf.getvalue()).decode("ascii")
        self._self_log("n_markers", len(points))
        return {"annotated_b64": out, "annotated_image": annotated}


# ══════════════════════════════════════════════════════════════════════
# Node: project_waypoints
# ══════════════════════════════════════════════════════════════════════


class ProjectWaypointsNode(BaseCanvasNode):
    """Unproject candidate pixels to per-candidate relative-polar (angle, distance).

    Pixel + depth + intrinsics + this view's heading -> the (angle, distance)
    polar that ``env_habitat__step_hightolow`` consumes (angle = yaw relative to
    the agent's CURRENT heading, distance = ground-plane metres). See the module
    docstring + ``_project_pixels`` for the convention; the bearing sign is
    config-flippable pending the M5 live-env check.
    """

    node_type: ClassVar[str] = "aoplanner__project_waypoints"
    display_name: ClassVar[str] = "AO-Planner: Project Waypoints"
    description: ClassVar[str] = (
        "Unproject candidate pixels to relative-polar (angle, distance) via depth + intrinsics"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Move3d"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField(
                "depth_is_mm",
                "boolean",
                "depth_b64 is 16-bit millimetres (panorama depth_raw)",
                default=True,
            ),
            ConfigField(
                "depth_scale",
                "text",
                "Multiply decoded depth (=MAX_DEPTH=10.0 to un-normalize VLN-CE [0,1] depth)",
                default="10.0",
            ),
            ConfigField(
                "heading_sign", "text", "Sign on view_heading_deg (+1 default)", default="1"
            ),
            ConfigField(
                "bearing_sign",
                "text",
                "Sign on in-view bearing (-1 default: +X=right=CW yaw)",
                default="-1",
            ),
            ConfigField("min_depth", "text", "Reject depth below (m)", default="0.1"),
            ConfigField("max_depth", "text", "Reject depth above (m)", default="30.0"),
        ],
    )
    input_ports = [
        PortDef("candidate_pixels", "TEXT", "JSON list of [u,v] pixel coords"),
        PortDef(
            "paths_pixels",
            "TEXT",
            "JSON list of routes (one per candidate) from parse_proposal (D-2)",
            optional=True,
        ),
        PortDef("depth_b64", "TEXT", "Base64 depth PNG (16-bit mm by default)", optional=True),
        PortDef(
            "depth", "ANY", "Raw depth HxW float metres (alternative to depth_b64)", optional=True
        ),
        PortDef(
            "intrinsics", "ANY", "Pinhole {fx,fy,cx,cy,width,height} (from observe_egocentric)"
        ),
        PortDef(
            "view_heading_deg",
            "ANY",
            "View yaw offset (deg) — used only when position/rotation absent",
            optional=True,
        ),
        PortDef(
            "position",
            "ANY",
            "World-frame agent position [x,y,z] from observe_camera_pose (enables world-coord projection)",
            optional=True,
        ),
        PortDef(
            "rotation",
            "ANY",
            "World-frame agent quaternion [x,y,z,w] from observe_camera_pose",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "candidates",
            "TEXT",
            "JSON list of {idx,u,v,angle,distance,path:[{angle,distance}...]} for valid candidates",
        ),
        PortDef("count", "ANY", "Number of valid candidates"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            pixels = json.loads(inputs.get("candidate_pixels") or "[]")
        except Exception:
            pixels = []
        if not pixels:
            return {"candidates": "[]", "count": 0}

        cfg = getattr(self, "config", None) or {}
        depth_is_mm = bool(cfg.get("depth_is_mm", True))
        depth_scale = float(cfg.get("depth_scale", 10.0) or 10.0)
        heading_sign = float(cfg.get("heading_sign", 1.0) or 1.0)
        bearing_sign = float(cfg.get("bearing_sign", -1.0) or -1.0)
        min_depth = float(cfg.get("min_depth", 0.1) or 0.1)
        max_depth = float(cfg.get("max_depth", 30.0) or 30.0)

        # Read intrinsics first: candidate pixels are sampled in the RGB/intrinsics
        # resolution (224x224), so resize the (256x256) depth to match before lookup.
        intr = inputs.get("intrinsics") or {}
        fx = float(intr.get("fx") or 0.0)
        fy = float(intr.get("fy") or fx)
        iw, ih = intr.get("width"), intr.get("height")
        target_wh = (int(iw), int(ih)) if (iw and ih) else None
        if fx <= 0:
            log.warning("project_waypoints: missing/invalid intrinsics fx; cannot unproject")
            return {"candidates": "[]", "count": 0}

        depth_b64 = inputs.get("depth_b64")
        raw_depth = inputs.get("depth")
        if depth_b64:
            depth_m = _decode_depth_m(
                depth_b64, depth_is_mm, depth_scale=depth_scale, target_wh=target_wh
            )
        elif raw_depth is not None:
            depth_m = np.asarray(raw_depth, dtype=np.float32)
            if depth_m.ndim == 3:
                depth_m = depth_m[..., 0]
            if depth_scale != 1.0:
                depth_m = depth_m * depth_scale
            if target_wh is not None and (depth_m.shape[1], depth_m.shape[0]) != target_wh:
                from PIL import Image as _Img

                depth_m = np.asarray(
                    _Img.fromarray(depth_m, mode="F").resize(target_wh, _Img.NEAREST),
                    dtype=np.float32,
                )
        else:
            return {"candidates": "[]", "count": 0}

        h, w = depth_m.shape[:2]
        cx = float(intr.get("cx") if intr.get("cx") is not None else w / 2.0)
        cy = float(intr.get("cy") if intr.get("cy") is not None else h / 2.0)
        # ── choose projection path ────────────────────────────────────────────
        pos_raw = inputs.get("position")
        rot_raw = inputs.get("rotation")
        use_world = (
            isinstance(pos_raw, (list, tuple))
            and len(pos_raw) == 3
            and isinstance(rot_raw, (list, tuple))
            and len(rot_raw) == 4
        )

        try:
            routes = json.loads(inputs.get("paths_pixels") or "[]")
        except Exception:
            routes = []
        if not isinstance(routes, list):
            routes = []

        out: list = []

        if use_world:
            # World-coordinate projection: pixel_to_world + _calculate_vp_rel_pos
            # (upstream utils.py:87-130 + environments_llm.py:27-43)
            position = [float(x) for x in pos_raw]
            agent_rotation = [float(x) for x in rot_raw]
            view_heading_deg = float(inputs.get("view_heading_deg") or 0.0)
            world_kw = dict(
                position=position,
                agent_rotation=agent_rotation,
                view_heading_deg=view_heading_deg,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            projected = _project_pixels_world(pixels, depth_m, fx, fy, cx, cy, **world_kw)
            for k, p in enumerate(projected):
                if not p["valid"]:
                    continue
                route_pix = routes[k] if k < len(routes) else []
                path_polar: list = []
                if route_pix:
                    rp = _project_pixels_world(route_pix, depth_m, fx, fy, cx, cy, **world_kw)
                    path_polar = [
                        {"angle": q["angle"], "distance": q["distance"]} for q in rp if q["valid"]
                    ]
                if not path_polar:
                    path_polar = [{"angle": p["angle"], "distance": p["distance"]}]
                entry = dict(p)
                entry["path"] = path_polar
                out.append(entry)
            self._self_log("projection", "world")
        else:
            # Fallback: polar approximation via view heading offset
            view_heading_deg = float(inputs.get("view_heading_deg") or 0.0)
            proj_kw = dict(
                view_heading_deg=view_heading_deg,
                heading_sign=heading_sign,
                bearing_sign=bearing_sign,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            projected = _project_pixels(pixels, depth_m, fx, fy, cx, cy, **proj_kw)
            for k, p in enumerate(projected):
                if not p["valid"]:
                    continue
                route_pix = routes[k] if k < len(routes) else []
                path_polar = []
                if route_pix:
                    rp = _project_pixels(route_pix, depth_m, fx, fy, cx, cy, **proj_kw)
                    path_polar = [
                        {"angle": q["angle"], "distance": q["distance"]} for q in rp if q["valid"]
                    ]
                if not path_polar:
                    path_polar = [{"angle": p["angle"], "distance": p["distance"]}]
                entry = dict(p)
                entry["path"] = path_polar
                out.append(entry)
            self._self_log("projection", "polar-approx")

        self._self_log("n_valid", len(out))
        self._self_log("n_input", len(pixels))
        self._self_log("path_lens", [len(e["path"]) for e in out])
        return {"candidates": json.dumps(out), "count": len(out)}


# ══════════════════════════════════════════════════════════════════════
# View glue — pull one panorama view; pack a per-view bundle for aggregate
# ══════════════════════════════════════════════════════════════════════


class ExtractViewNode(BaseCanvasNode):
    """Pull one observe_panorama view's base64 RGB / raw-mm depth / heading by index.

    Drives one VAP lane: observe_panorama(n_views=N).views -> extract_view(index=k).
    """

    node_type: ClassVar[str] = "aoplanner__extract_view"
    display_name: ClassVar[str] = "AO-Planner: Extract Panorama View"
    description: ClassVar[str] = (
        "Pull one panorama view's rgb_base64 / depth_raw_base64 / heading_deg by index"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ImageDown"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField("index", "text", "View index (0-based; 0 = current heading)", default="0"),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "observe_panorama views list"),
    ]
    output_ports = [
        PortDef("rgb_b64", "TEXT", "This view's rgb_base64"),
        PortDef("depth_b64", "TEXT", "This view's depth_raw_base64 (16-bit mm)"),
        PortDef("heading_deg", "ANY", "This view's heading offset from current heading (deg)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if isinstance(views, str):
            try:
                views = json.loads(views)
            except Exception:
                views = []
        idx = int(float((getattr(self, "config", None) or {}).get("index", 0) or 0))
        if not isinstance(views, list) or not (0 <= idx < len(views)):
            return {"rgb_b64": "", "depth_b64": "", "heading_deg": 0.0}
        v = views[idx] or {}
        return {
            "rgb_b64": v.get("rgb_base64", "") or "",
            "depth_b64": v.get("depth_raw_base64", "") or "",
            "heading_deg": float(v.get("heading_deg", 0.0) or 0.0),
        }


class MakeBundleNode(BaseCanvasNode):
    """Pack one VAP lane's rgb + projected candidates into a bundle for aggregate.

    The aggregate node's ``view_bundles`` is LIST[ANY]: wire each lane's
    ``bundle`` into it; the executor concatenates in edge-declaration order.
    """

    node_type: ClassVar[str] = "aoplanner__make_bundle"
    display_name: ClassVar[str] = "AO-Planner: Make View Bundle"
    description: ClassVar[str] = (
        "Pack one view's rgb_b64 + projected candidates into an aggregate bundle"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Package"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField(
                "view_dir", "text", "Direction label (front/backward/left/right)", default="front"
            ),
        ],
    )
    input_ports = [
        PortDef("rgb_b64", "TEXT", "This view's rgb_base64 (annotated in aggregate)"),
        PortDef(
            "candidates",
            "TEXT",
            "JSON list of projected {u,v,angle,distance} from project_waypoints",
        ),
        PortDef(
            "path_pixels",
            "TEXT",
            "JSON list of pixel routes from parse_proposal.selected_paths (1:1 with candidates by idx); used by aggregate to draw path polylines (upstream vis_ghost_nodes)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef(
            "bundle",
            "ANY",
            "{view_dir, rgb_b64, candidates, path_pixels} bundle for aggregate.view_bundles",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            cands = json.loads(inputs.get("candidates") or "[]")
        except Exception:
            cands = []
        try:
            path_pix = json.loads(inputs.get("path_pixels") or "[]")
        except Exception:
            path_pix = []
        view_dir = (getattr(self, "config", None) or {}).get("view_dir", "front")
        return {
            "bundle": {
                "view_dir": view_dir,
                "rgb_b64": inputs.get("rgb_b64", "") or "",
                "candidates": cands,
                "path_pixels": path_pix,
            }
        }


# ══════════════════════════════════════════════════════════════════════
# Proposer glue (VLM#1) — GroundingDINO box pick, proposer system prompt,
# and {Waypoints} parse. The VLM#1 call itself is a built-in vision llmCall
# wired in the graph (gpt-5-mini, temp 1.0 / max_tokens 2000).
# ══════════════════════════════════════════════════════════════════════


class PickGroundBoxNode(BaseCanvasNode):
    """Pick one GroundingDINO 'ground' box to feed model_sam__segment_box.

    AO-Planner unions SAM masks over ALL ground boxes; our static graph picks
    the single best box (top score / largest) — a recorded bucket-D
    simplification (one segment_box call per view instead of a variable fan).
    """

    node_type: ClassVar[str] = "aoplanner__pick_ground_box"
    display_name: ClassVar[str] = "AO-Planner: Pick Ground Box"
    description: ClassVar[str] = (
        "Pick the top GroundingDINO 'ground' box (xyxy) for SAM segment_box"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "BoxSelect"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="emerald",
        config_fields=[
            ConfigField(
                "strategy",
                "select",
                "Which box to pick",
                options=[
                    {"value": "top_score", "label": "Highest score"},
                    {"value": "largest", "label": "Largest area"},
                ],
                default="top_score",
            ),
        ],
    )
    input_ports = [
        PortDef("detections", "TEXT", "model_grounding_dino__detect result JSON"),
    ]
    output_ports = [
        PortDef("box", "TEXT", "JSON [x1,y1,x2,y2] for model_sam__segment_box (empty if none)"),
        PortDef("found", "BOOL", "True iff a ground box was found"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            data = json.loads(inputs.get("detections") or "{}")
        except Exception:
            data = {}
        boxes = data.get("boxes") or []
        if not boxes:
            self._self_log("found", False)
            return {"box": "", "found": False}
        strategy = (getattr(self, "config", None) or {}).get("strategy", "top_score")
        if strategy == "largest":

            def _area(b: dict) -> float:
                xy = b.get("xyxy", [0, 0, 0, 0])
                return max(0, xy[2] - xy[0]) * max(0, xy[3] - xy[1])

            best = max(boxes, key=_area)
        else:
            best = max(boxes, key=lambda b: float(b.get("score", 0.0)))
        self._self_log("picked_box", best.get("xyxy"))
        return {"box": json.dumps(best.get("xyxy", [])), "found": True}


class GroundBoxesNode(BaseCanvasNode):
    """ALL GroundingDINO 'ground' boxes → model_sam__segment_box batch input.

    The faithful AO-Planner path: SAM masks are unioned over EVERY ground
    box (the former ``model_grounding_dino__ground_mask`` did this inside
    one node; since the 2026-07-05 composition eviction the graph wires
    detect → this extractor → ``model_sam__segment_box`` →
    ``sample_waypoints.sam_result`` instead). ``PickGroundBoxNode`` remains
    the single-box simplification variant.
    """

    node_type: ClassVar[str] = "aoplanner__ground_boxes"
    display_name: ClassVar[str] = "AO-Planner: Ground Boxes"
    description: ClassVar[str] = (
        "Extract ALL detect-result boxes as [[x1,y1,x2,y2],…] for model_sam__segment_box"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "BoxSelect"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports = [
        PortDef("detections", "TEXT", "model_grounding_dino__detect result JSON"),
    ]
    output_ports = [
        PortDef("boxes", "TEXT", "JSON [[x1,y1,x2,y2],…] ('[]' if none)"),
        PortDef("count", "ANY", "Number of boxes"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            data = json.loads(inputs.get("detections") or "{}")
        except Exception:
            data = {}
        boxes = [b.get("xyxy") for b in (data.get("boxes") or []) if b.get("xyxy")]
        self._self_log("n_boxes", len(boxes))
        return {"boxes": json.dumps(boxes), "count": len(boxes)}


class ProposePrepNode(BaseCanvasNode):
    """Build the VLM#1 waypoint-proposer system prompt (verbatim) from the instruction."""

    node_type: ClassVar[str] = "aoplanner__propose_prep"
    display_name: ClassVar[str] = "AO-Planner: Proposer System Prompt"
    description: ClassVar[str] = (
        "Build the VLM#1 waypoint-proposer system prompt from the instruction (verbatim)"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Type"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef(
            "instruction", "TEXT", "Navigation instruction (embedded into the proposer prompt)"
        ),
    ]
    output_ports = [
        PortDef("system_prompt", "TEXT", "VLM#1 proposer system prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", "") or "")
        sp = _prompts.build_proposer_system(instruction)
        self._self_log("system_prompt_len", len(sp))
        return {"system_prompt": sp}


class ParseProposalNode(BaseCanvasNode):
    """Map VLM#1 {Waypoints, Paths} IDs back to candidate pixel coords.

    Both the destination Waypoints and their multi-point Paths are mapped (D-2,
    2026-06-17: Paths restored — was a single polar hop). IDs index into
    sample_waypoints' candidate_pixels (0-based, 0 = foot point). For each kept
    waypoint we emit its destination pixel plus the route (the path's grid ids ->
    pixels, made to end at the destination); project_waypoints turns each route
    into the polar hop sequence that env_habitat__step_path walks.

    `exclude_foot_point` (default True) keeps waypoint id 0 out of the
    *destination* set (a destination at the agent's own feet is a no-op), but the
    foot point is still usable as a path *start* inside a route — its near-zero
    depth simply contributes no movement (D-5).
    """

    node_type: ClassVar[str] = "aoplanner__parse_proposal"
    display_name: ClassVar[str] = "AO-Planner: Parse Waypoint Proposal"
    description: ClassVar[str] = (
        "Map VLM#1 {Waypoints, Paths} IDs → destination pixels + route pixels"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Parentheses"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "exclude_foot_point",
                "boolean",
                "Keep id 0 (foot / path-start) out of the destination set (still usable inside a route)",
                default=True,
            ),
        ],
    )
    input_ports = [
        PortDef("response", "TEXT", "VLM#1 JSON response"),
        PortDef(
            "candidate_pixels",
            "TEXT",
            "JSON list of [u,v] (from sample_waypoints) the IDs index into",
        ),
    ]
    output_ports = [
        PortDef(
            "selected_pixels", "TEXT", "JSON list of [u,v] for the chosen destination waypoints"
        ),
        PortDef(
            "selected_paths",
            "TEXT",
            "JSON list of routes (one per waypoint), each a list of [u,v] (D-2)",
        ),
        PortDef("count", "ANY", "Number of selected waypoints"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            pixels = json.loads(inputs.get("candidate_pixels") or "[]")
        except Exception:
            pixels = []
        exclude_foot = bool((getattr(self, "config", None) or {}).get("exclude_foot_point", True))
        parsed = _prompts.parse_proposal(str(inputs.get("response", "") or ""))
        waypoints = parsed["waypoints"]
        paths = parsed["paths"]
        sel: list = []
        sel_paths: list = []
        seen: set = set()
        for k, wid in enumerate(waypoints):
            if exclude_foot and wid == 0:
                continue  # foot point is a path START, never a destination (D-5)
            if not (0 <= wid < len(pixels)) or wid in seen:
                continue
            seen.add(wid)
            dest = pixels[wid]
            sel.append(dest)
            # D-2: the route to this waypoint (paths[k], paired by index). Map ids
            # -> pixels, drop out-of-range ids, and make the route end at the
            # destination (upstream routes lead TO the waypoint).
            # Skip id 0 (foot) from route intermediates — matches upstream
            # llm_waypoint_predictor_single_view:53-56 (`if point_id == 0: continue`).
            route_ids = paths[k] if k < len(paths) else []
            route = [pixels[r] for r in route_ids if r != 0 and 0 <= r < len(pixels)]
            if not route or route[-1] != dest:
                route.append(dest)
            sel_paths.append(route)
        self._self_log("waypoint_ids", waypoints)
        self._self_log("n_selected", len(sel))
        self._self_log("path_lens", [len(p) for p in sel_paths])
        return {
            "selected_pixels": json.dumps(sel),
            "selected_paths": json.dumps(sel_paths),
            "count": len(sel),
        }


# ══════════════════════════════════════════════════════════════════════
# Decider (VLM#2 PathAgent) — adapts the SmartWay 7-node structure to
# AO-Planner's ghost-ID / 4-view prompt; SmartWay's backtrack/return/tags
# machinery is intentionally dropped (pure-FM framing).
# ══════════════════════════════════════════════════════════════════════


class AggregateCandidatesNode(BaseCanvasNode):
    """Merge per-view selected waypoints into one ghost-ID candidate set and
    annotate each view with its IDs.

    Folds AO-Planner's ghost-node aggregation (zero_shot_agent vis_ghost_nodes
    + ZeroShotGraphMap.update_graph) + the PathAgent option-list build.

    D-3: Ghost IDs are episode-monotonic (never reset to 0 mid-episode), matching
    upstream ZeroShotGraphMap.update_graph which always increments ghost_cnt without
    merging.  No spatial dedup — upstream eval never merges candidates across steps;
    merge_ghost lives only in the supervised GraphMap which is dead-loaded at eval.
    The monotonic counter (next_gid) is persisted per episode in graph_state.ghost_map.
    """

    node_type: ClassVar[str] = "aoplanner__aggregate"
    display_name: ClassVar[str] = "AO-Planner: Aggregate Candidates"
    description: ClassVar[str] = (
        "Merge per-view waypoints into one ghost-ID set + annotate the 4 views"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Network"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="amber",
        config_fields=[
            ConfigField("radius", "text", "Ghost-marker radius (px)", default="12"),
            ConfigField("font_size", "text", "Ghost-label font size (px)", default="22"),
        ],
    )
    input_ports = [
        PortDef(
            "view_bundles",
            "LIST[ANY]",
            "Per-view {view_dir, rgb_b64, candidates:[{u,v,angle,distance}]} bundles (fan-in from make_bundle)",
        ),
    ]
    output_ports = [
        PortDef(
            "candidates_dict", "TEXT", "JSON {gid: {angle, distance}} manifest for resolve_action"
        ),
        PortDef(
            "action_space_text", "TEXT", "Comma-joined ghost IDs for the PathAgent Options line"
        ),
        PortDef(
            "view_images_b64", "TEXT", "JSON list of annotated per-view PNGs (1:1 with view_labels)"
        ),
        PortDef(
            "view_labels", "TEXT", "JSON list of '({dir}) Locations {ids} in Image {i}' captions"
        ),
        PortDef("count", "ANY", "Number of ghost candidates"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        from PIL import Image

        bundles = inputs.get("view_bundles") or []
        if isinstance(bundles, str):
            try:
                bundles = json.loads(bundles)
            except Exception:
                bundles = []
        cfg = getattr(self, "config", None) or {}
        radius = max(1, int(float(cfg.get("radius", 12) or 12)))
        font_size = max(6, int(float(cfg.get("font_size", 22) or 22)))

        # D-3: load monotonic ghost ID counter from graph_state (episode-scoped).
        # Upstream ZeroShotGraphMap.update_graph always increments ghost_cnt without
        # merging — we replicate that: fresh ID per candidate per step, counter never
        # resets within an episode.
        gs = getattr(ctx, "graph_state", None) if ctx else None
        raw_gm = _read(gs, "ghost_map", {"next_gid": 0})
        next_gid: int = int(raw_gm.get("next_gid", 0))

        manifest: dict[int, dict] = {}
        view_images: list[str] = []
        view_labels: list[str] = []
        for i, vb in enumerate(bundles):
            if not isinstance(vb, dict):
                continue
            view_dir = str(
                vb.get("view_dir")
                or (_prompts.DIRECTIONS[i] if i < len(_prompts.DIRECTIONS) else f"view{i}")
            )
            rgb_b64 = vb.get("rgb_b64") or ""
            cands = vb.get("candidates") or []
            pts: list = []
            ids_here: list[int] = []
            for c in cands:
                try:
                    angle = float(c.get("angle"))
                    distance = float(c.get("distance"))
                except (TypeError, ValueError, AttributeError):
                    continue
                # D-2: carry the route as a polar hop sequence for step_path;
                # fall back to a single hop to the waypoint if absent.
                raw_path = c.get("path") or [{"angle": angle, "distance": distance}]
                path = [
                    [float(h.get("angle", 0.0)), float(h.get("distance", 0.0))] for h in raw_path
                ]
                # Assign fresh monotonic ghost ID (matches upstream ghost_cnt += 1).
                gid = next_gid
                next_gid += 1
                manifest[gid] = {
                    "angle": angle,
                    "distance": distance,
                    "path": path,
                    "view_dir": view_dir,
                }
                pts.append([c.get("u", 0), c.get("v", 0)])
                ids_here.append(gid)

            # Upstream only packs views that yielded ghosts: a view with no
            # llm_result contributes neither an image nor an Options entry, and
            # "Image {i}" counts CONTRIBUTING views (zero_shot_agent.py:727-731,
            # prompt_manager.py:54-59). Skip empty views entirely.
            if not (rgb_b64 and pts):
                continue
            rgb_arr = _decode_rgb(rgb_b64)
            # Draw path polylines before ghost dots (upstream vis_ghost_nodes:
            # per-route random colour, a bottom-anchor segment from the image
            # bottom row to the route start (multi_start=True), then cv2.line
            # per segment, then vis_points for the ghost label).
            path_pixels = vb.get("path_pixels") or []
            if path_pixels:
                from PIL import ImageDraw

                img_pil = Image.fromarray(rgb_arr)
                draw = ImageDraw.Draw(img_pil)
                img_h = rgb_arr.shape[0]
                for cand in cands:
                    idx = cand.get("idx")
                    if idx is None or idx >= len(path_pixels):
                        continue
                    route = path_pixels[idx]
                    if not isinstance(route, list) or not route:
                        continue
                    # route is [[u0,v0],[u1,v1],...] ending at the waypoint
                    pts_line = [(int(p[0]), int(p[1])) for p in route if len(p) >= 2]
                    if not pts_line:
                        continue
                    color = tuple(int(c) for c in np.random.randint(0, 256, size=3))
                    # bottom-anchor: upstream draws (x0, 511) -> route start so
                    # bottom-row waypoints still show a path (llm/utils.py:164-186)
                    draw.line([(pts_line[0][0], img_h - 1), pts_line[0]], fill=color, width=2)
                    if len(pts_line) >= 2:
                        draw.line(pts_line, fill=color, width=2)
                rgb_arr = np.array(img_pil)
            annotated = _draw_markers(rgb_arr, pts, [str(x) for x in ids_here], radius, font_size)
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            ann_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            view_labels.append(
                _prompts.image_label(
                    view_dir, ", ".join(str(x) for x in ids_here), len(view_images)
                )
            )
            view_images.append(ann_b64)

        # Persist updated counter.
        _write(gs, "ghost_map", {"next_gid": next_gid})

        action_space_text = ", ".join(str(k) for k in manifest)
        self._self_log("n_candidates", len(manifest))
        self._self_log("action_space_text", action_space_text)
        # "_meta.total" = ghost ids ever minted this episode == upstream
        # len(waypoint_path_coord); resolve_action uses it to replicate the
        # upstream IndexError→STOP on an id beyond the accumulated set.
        manifest_json: dict[str, Any] = {str(k): v for k, v in manifest.items()}
        manifest_json["_meta"] = {"total": next_gid}
        return {
            "candidates_dict": json.dumps(manifest_json),
            "action_space_text": action_space_text,
            "view_images_b64": json.dumps(view_images),
            "view_labels": json.dumps(view_labels),
            "count": len(manifest),
        }


class AssemblePromptNode(BaseCanvasNode):
    """Render the PathAgent system + user prompts (verbatim _prompts)."""

    node_type: ClassVar[str] = "aoplanner__assemble_prompt"
    display_name: ClassVar[str] = "AO-Planner: Assemble PathAgent Prompt"
    description: ClassVar[str] = (
        "Render PathAgent task_description (system) + the per-step user prompt"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "FileText"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("instruction", "TEXT", "Navigation instruction"),
        PortDef("action_space_text", "TEXT", "Comma-joined ghost IDs from aggregate"),
    ]
    output_ports = [
        PortDef("task_description", "TEXT", "PathAgent system prompt (verbatim)"),
        PortDef("prompt", "TEXT", "PathAgent user prompt"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        instruction = str(inputs.get("instruction", ""))
        action_space_text = str(inputs.get("action_space_text", ""))
        t = int(getattr(ctx, "step", 0)) if ctx else 0
        gs = getattr(ctx, "graph_state", None) if ctx else None
        planning = _read(gs, "planning", [_prompts.DEFAULT_PLANNING])
        if not isinstance(planning, list) or not planning:
            planning = [_prompts.DEFAULT_PLANNING]
        history = str(_read(gs, "history", ""))
        prompt = _prompts.assemble_pathagent_prompt(
            instruction, str(planning[-1]), history, action_space_text, t
        )
        self._self_log("step", t)
        self._self_log("prompt_preview", prompt[-300:])
        return {"task_description": _prompts.build_task_description(), "prompt": prompt}


class BuildImagesNode(BaseCanvasNode):
    """Build LIST[IMAGE] for PathAgent: history images first, then current option views.

    Reads ``history_images`` from graph_state (written by UpdateHistoryNode in prior
    steps) and prepends them before the 4 current-step annotated direction views.
    This aligns with upstream ``make_graph_history`` + ``make_image_content`` which
    interleaves past-step images before the current-step option images (D-4 fix).
    """

    node_type: ClassVar[str] = "aoplanner__build_images"
    display_name: ClassVar[str] = "AO-Planner: Build PathAgent Images"
    description: ClassVar[str] = (
        "History images (from state) + current annotated views → LIST[IMAGE] + labels"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Image"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="amber")
    input_ports = [
        PortDef("view_images_b64", "TEXT", "JSON list of annotated per-view PNGs (from aggregate)"),
        PortDef("view_labels", "TEXT", "JSON list of per-image captions (from aggregate)"),
        PortDef(
            "action_space_text",
            "TEXT",
            "Comma-joined ghost IDs (from aggregate) — the t>0 Options line rides "
            "the first option image's label (upstream content order)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("images", "LIST[IMAGE]", "History + current per-view annotated RGB tiles"),
        PortDef("image_labels", "LIST[TEXT]", "Per-image captions, 1:1 with images"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        try:
            imgs_b64 = json.loads(inputs.get("view_images_b64") or "[]")
        except Exception:
            imgs_b64 = []
        try:
            labels = json.loads(inputs.get("view_labels") or "[]")
        except Exception:
            labels = []

        images: list = []
        out_labels: list[str] = []

        # ── prepend history images (D-4: upstream make_graph_history order) ──
        gs = getattr(ctx, "graph_state", None) if ctx else None
        hist_imgs = _read(gs, "history_images", [])
        if isinstance(hist_imgs, list):
            for entry in hist_imgs:
                if not isinstance(entry, dict):
                    continue
                b64 = entry.get("b64")
                caption = str(entry.get("caption", ""))
                if b64:
                    images.append(_decode_rgb(b64))
                    out_labels.append(caption)

        # ── current-step option views ─────────────────────────────────────────
        # At t>0 upstream emits the Options line as its own text item BETWEEN the
        # history images and the option images (prompt_manager.py:75-79); llmCall
        # packs one text item per image label, so the line rides the first option
        # image's label — adjacent text items are semantically identical.
        t = int(getattr(ctx, "step", 0)) if ctx else 0
        opts_prefix = ""
        if t > 0:
            action_space_text = str(inputs.get("action_space_text", "") or "")
            opts_prefix = _prompts.options_line(action_space_text, t)
        first_current = True
        for i, b64 in enumerate(imgs_b64):
            if not b64:
                continue
            images.append(_decode_rgb(b64))
            caption = str(labels[i]) if i < len(labels) else f"Image {i}"
            if first_current and opts_prefix:
                caption = opts_prefix + caption
                first_current = False
            out_labels.append(caption)

        self._self_log("n_history", len(hist_imgs) if isinstance(hist_imgs, list) else 0)
        self._self_log("n_images", len(images))
        return {"images": images, "image_labels": out_labels}


class ParseResponseNode(BaseCanvasNode):
    """Parse the PathAgent {Thought, New Planning, Action} JSON. Sole writer of `planning`."""

    node_type: ClassVar[str] = "aoplanner__parse_response"
    display_name: ClassVar[str] = "AO-Planner: Parse PathAgent Response"
    description: ClassVar[str] = (
        "Parse {Thought, New Planning, Action} → action_id / is_stop; sole writer of planning"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Parentheses"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("response", "TEXT", "PathAgent JSON response text"),
        PortDef(
            "n_candidates",
            "ANY",
            "aggregate candidate count; 0 ⇒ force STOP (no options)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("action_id", "TEXT", "Chosen ghost ID (or -1 for STOP)"),
        PortDef("is_stop", "BOOL", "True when Action == Stop"),
        PortDef("thought", "TEXT", "Thought field"),
        PortDef("new_planning", "TEXT", "New Planning field"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        parsed = _prompts.parse_pathagent(str(inputs.get("response", "") or ""))
        is_stop = bool(parsed["is_stop"])
        action_id = parsed["action_id"]
        # §3E #7: upstream short-circuits to STOP when no view yields candidates
        # (zero_shot_agent.py:827-830). Force it here rather than letting the
        # PathAgent pick into an empty option set.
        n_raw = inputs.get("n_candidates")
        if n_raw is not None:
            try:
                if int(float(n_raw)) == 0:
                    is_stop, action_id = True, -1
            except (TypeError, ValueError):
                pass
        gs = getattr(ctx, "graph_state", None) if ctx else None
        # §3E #6: only append planning when the reply actually parsed; upstream
        # leaves planning untouched on a total parse failure (carries the prior).
        if parsed.get("ok"):
            planning = _read(gs, "planning", [_prompts.DEFAULT_PLANNING])
            if not isinstance(planning, list):
                planning = [_prompts.DEFAULT_PLANNING]
            _write(gs, "planning", [*list(planning), parsed["new_planning"]])
        self._self_log("action_id", action_id)
        self._self_log("is_stop", is_stop)
        self._self_log("parsed_ok", parsed.get("ok"))
        self._self_log("thought", parsed["thought"][:200])
        self._self_log("new_planning", parsed["new_planning"][:200])
        return {
            "action_id": str(action_id),
            "is_stop": is_stop,
            "thought": parsed["thought"],
            "new_planning": parsed["new_planning"],
        }


class ResolveActionNode(BaseCanvasNode):
    """Map the chosen ghost ID to (angle, distance) via the manifest. STOP -> (0,0).

    `fallback` governs an action_id that is NOT a valid ghost id — PathAgent
    sometimes emits a step number or a hallucinated id, especially when the
    option set is sparse (upstream `parse_num` has the same hazard but rides a
    dense action space). Recovery options (M4b fix; previously a silent (0,0)
    no-op that froze the agent without stopping the loop):
      - 'forward' (default): the valid candidate with the smallest |angle|
        (least turning / most forward), so the agent keeps progressing and
        re-plans next step.
      - 'first': the lowest valid ghost id.
      - 'noop': legacy (0,0) no-op (kept for ablation).
    An empty manifest (no options at all) yields (0,0) under any fallback.
    """

    node_type: ClassVar[str] = "aoplanner__resolve_action"
    display_name: ClassVar[str] = "AO-Planner: Resolve Action"
    description: ClassVar[str] = (
        "action_id + manifest → (angle, distance) for step_hightolow; STOP → (0,0)"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "fallback",
                "select",
                "Recovery when action_id is not a valid ghost id",
                options=[
                    {"value": "forward", "label": "Most-forward valid candidate"},
                    {"value": "first", "label": "Lowest valid ghost id"},
                    {"value": "noop", "label": "No-op (0,0) — legacy"},
                ],
                default="forward",
            ),
        ],
    )
    input_ports = [
        PortDef("action_id", "TEXT", "From parse_response"),
        PortDef("is_stop", "BOOL", "From parse_response"),
        PortDef("candidates_dict", "TEXT", "JSON manifest from aggregate"),
    ]
    output_ports = [
        PortDef(
            "path_angles",
            "TEXT",
            "JSON list of per-hop yaw radians for step_path (D-2; [] on STOP)",
        ),
        PortDef(
            "path_distances",
            "TEXT",
            "JSON list of per-hop distances (m) for step_path ([] on STOP)",
        ),
        PortDef("angle", "TEXT", "Destination-hop yaw radians (back-compat single hop; 0 on STOP)"),
        PortDef("distance", "TEXT", "Destination-hop metres (back-compat single hop; 0 on STOP)"),
        PortDef(
            "chosen_view_dir",
            "TEXT",
            "Direction of chosen candidate (front/left/backward/right); empty on STOP",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        is_stop = bool(inputs.get("is_stop", False))
        try:
            aid = int(str(inputs.get("action_id", "-1")))
        except (TypeError, ValueError):
            aid = -1
        manifest: dict = {}
        try:
            v = json.loads(inputs.get("candidates_dict") or "{}")
            if isinstance(v, dict):
                manifest = v
        except Exception:
            pass
        fallback = (getattr(self, "config", None) or {}).get("fallback", "forward")

        meta = manifest.pop("_meta", None) or {}
        try:
            total_ghosts = int(meta.get("total"))
        except (TypeError, ValueError):
            total_ghosts = None

        chosen = None
        if not is_stop and total_ghosts is not None and aid >= total_ghosts:
            # Upstream: waypoint_path_coord[id] raises IndexError for an id
            # beyond the episode's accumulated ghosts → except → 'stop'
            # (zero_shot_agent.py:838-844). Replicate: STOP, no fallback.
            is_stop = True
            self._self_log("stop_out_of_range", aid)
        if not is_stop:
            entry = manifest.get(str(aid)) if aid >= 0 else None
            if isinstance(entry, dict):
                chosen = entry
            else:
                # action_id names a PAST ghost (< total but not in the current
                # manifest). Upstream would replay that ghost's stored world-coord
                # path from the current pose; the port keeps no cross-step path
                # store (D-3 deferral) and recovers per `fallback` instead.
                valid = []
                for k, e in manifest.items():
                    if not isinstance(e, dict):
                        continue
                    try:
                        valid.append((int(k), e))
                    except (TypeError, ValueError):
                        continue
                if valid and fallback != "noop":
                    if fallback == "first":
                        valid.sort(key=lambda kv: kv[0])
                    else:  # "forward": least turning
                        valid.sort(key=lambda kv: abs(float(kv[1].get("angle", 0.0))))
                    chosen = valid[0][1]
                    self._self_log("fallback", fallback)
                    self._self_log("fallback_for_aid", aid)
                    self._self_log("fallback_pick", valid[0][0])
                else:
                    self._self_log("lookup_miss", aid)

        # D-2: emit the chosen ghost id's full route as a polar hop sequence for
        # step_path; STOP / empty manifest -> empty path (step_path does step(0)).
        if chosen:
            path = chosen.get("path") or [
                [float(chosen.get("angle", 0.0)), float(chosen.get("distance", 0.0))]
            ]
        else:
            path = []
        angles = [float(h[0]) for h in path]
        dists = [float(h[1]) for h in path]
        angle = angles[-1] if angles else 0.0  # back-compat single hop = destination
        distance = dists[-1] if dists else 0.0
        chosen_view_dir = ""
        if chosen and not is_stop:
            chosen_view_dir = str(chosen.get("view_dir", ""))
        self._self_log("n_hops", len(angles))
        self._self_log("angle_rad", angle)
        self._self_log("distance_m", distance)
        self._self_log("chosen_view_dir", chosen_view_dir)
        return {
            "path_angles": json.dumps([round(a, 6) for a in angles]),
            "path_distances": json.dumps([round(d, 6) for d in dists]),
            "angle": f"{angle:.6f}",
            "distance": f"{distance:.6f}",
            "chosen_view_dir": chosen_view_dir,
        }


class UpdateHistoryNode(BaseCanvasNode):
    """Append a PathAgent-style history entry. Sole writer of `history` and `history_images`."""

    node_type: ClassVar[str] = "aoplanner__update_history"
    display_name: ClassVar[str] = "AO-Planner: Update History"
    description: ClassVar[str] = (
        "Append upstream-faithful history entry: text + chosen direction image "
        "(make_graph_history format). Sole writer of history/history_images state."
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "ClipboardList"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="emerald")
    input_ports = [
        PortDef("action_id", "TEXT", "From parse_response"),
        PortDef("is_stop", "BOOL", "From parse_response"),
        PortDef("chosen_view_dir", "TEXT", "Direction of chosen candidate (from resolve_action)"),
        PortDef(
            "view_images_b64", "TEXT", "JSON list of 4 annotated direction views (from aggregate)"
        ),
        PortDef("step_done", "BOOL", "env step done flag", optional=True),
    ]
    output_ports = [
        PortDef("history", "TEXT", "Updated history text (unused by PathAgent; kept for debug)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        is_stop = bool(inputs.get("is_stop", False))
        try:
            aid = int(str(inputs.get("action_id", "-1")))
        except (TypeError, ValueError):
            aid = -1
        t = int(getattr(ctx, "step", 0)) if ctx else 0
        gs = getattr(ctx, "graph_state", None) if ctx else None

        # ── text history (debug / back-compat) ──────────────────────────
        prior_text = str(_read(gs, "history", ""))
        entry_text = f"Step {t}, Stop." if is_stop else f"Step {t}, move towards location {aid}."
        new_hist = entry_text if (t == 0 or not prior_text) else prior_text + " " + entry_text
        _write(gs, "history", new_hist)

        # ── image history (D-4 alignment: make_graph_history format) ────
        if not is_stop:
            view_dir = str(inputs.get("chosen_view_dir", "") or "")
            _VDIR_IDX = {"front": 0, "left": 1, "backward": 2, "right": 3}
            view_idx = _VDIR_IDX.get(view_dir, -1)
            try:
                views = json.loads(inputs.get("view_images_b64") or "[]")
            except Exception:
                views = []
            img_b64 = views[view_idx] if 0 <= view_idx < len(views) else None
            if img_b64:
                caption = _prompts.history_image_label(view_dir, aid, t)
                prior_imgs = _read(gs, "history_images", [])
                if not isinstance(prior_imgs, list):
                    prior_imgs = []
                _write(gs, "history_images", [*prior_imgs, {"caption": caption, "b64": img_b64}])

        self._self_log("step", t)
        self._self_log("history_preview", new_hist[-200:])
        return {"history": new_hist}


class EmitStopNode(BaseCanvasNode):
    """Emit a constant <code>(0,0)</code> STOP action, post-loop.

    §3E #1: upstream forces a STOP on the last step (<code>zero_shot_agent.py:889-901</code>)
    so habitat registers success at the final pose; the port's loop just hits
    <code>step_budget</code> with no explicit STOP, and habitat scores success only on an
    explicit <code>step(0)</code>. Wired post-loop (<code>iterOut.final_stop → emit_stop →
    step_hightolow → evaluate</code>): the last in-loop move has already executed, so this
    only stamps the STOP. If the episode already ended via <code>is_stop</code>, the env's
    <code>step(0)</code> is a safe no-op (already-done guard). No upstream-node data dep, so
    it fires cleanly post-loop (avoids the final-fire-None trap).
    """

    node_type: ClassVar[str] = "aoplanner__emit_stop"
    display_name: ClassVar[str] = "AO-Planner: Emit STOP"
    description: ClassVar[str] = (
        "Post-loop constant (0,0) STOP to register a habitat STOP at the final pose"
    )
    category: ClassVar[str] = "processing"
    icon: ClassVar[str] = "Square"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("trigger", "ANY", "Post-loop pulse (iterOut.final_stop)", optional=True),
    ]
    output_ports = [
        PortDef("angle", "TEXT", "Constant 0 (STOP)"),
        PortDef("distance", "TEXT", "Constant 0 (STOP)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        return {"angle": "0.000000", "distance": "0.000000"}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class AoPlannerNodeSet(BaseNodeSet):
    """AO-Planner VLN-CE port — VAP geometry/overlay nodes (M2); decider clones land in M3."""

    name = "aoplanner"
    description = (
        "AO-Planner (AAAI 2025) affordances-oriented VLN port — VAP perception + PathAgent decider"
    )

    def get_tools(self) -> list:
        return [
            # M2 — VAP geometry/overlay (per view)
            SampleWaypointsNode(),
            AnnotateMarkersNode(),
            ProjectWaypointsNode(),
            # M4 — view glue (panorama lane entry + aggregate fan-in)
            ExtractViewNode(),
            MakeBundleNode(),
            # M3 — proposer glue (VLM#1 side)
            PickGroundBoxNode(),
            GroundBoxesNode(),
            ProposePrepNode(),
            ParseProposalNode(),
            # M3 — decider (VLM#2 PathAgent)
            AggregateCandidatesNode(),
            AssemblePromptNode(),
            BuildImagesNode(),
            ParseResponseNode(),
            ResolveActionNode(),
            UpdateHistoryNode(),
            EmitStopNode(),
        ]
