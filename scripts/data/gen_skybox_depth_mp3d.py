"""Generate MatterSim depth skyboxes from MP3D .glb meshes via habitat-sim.

Produces ``{viewpoint_id}_skybox_depth_small.png`` files (3072x512, 16-bit
grayscale, 6 cube faces in column order [Y+, Z+, X+, Z-, X-, Y-] per MP3D
frame, with depth in 0.25mm units = pixel/4000 metres) inside
``data/mp3d/v1/scans/{scan}/matterport_skybox_images/``.

Usage (in vlnce env):
    python scripts/data/gen_skybox_depth_mp3d.py \
        --scans 17DRP5sb8fy \
        --viewpoints 00ebbf3782c64d74aaf7dd39cd561175 \
        --resolution 512

Bulk:
    python scripts/data/gen_skybox_depth_mp3d.py \
        --scans 2azQ1b91cZZ 8194nk5LbLH ... zsNo4HB9uLZ \
        --resolution 512

Critical references in this repo:
- Format authority: third_party/Matterport3DSimulator/src/lib/NavGraph.cpp:44-73
- Radial-vs-perpendicular semantic: third_party/Matterport3DSimulator/src/lib/fragment.sh
- uint16<->metres: workspace/nodesets/server/matterport3d.py:1390 (1 unit = 0.25mm)
- habitat-sim wrapper template: workspace/nodesets/server/hmeqa.py:177-202
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

# scripts/data/gen_skybox_depth_mp3d.py → scripts → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = _REPO_ROOT / "data"
CONN_DIR = _REPO_ROOT / "third_party" / "Matterport3DSimulator" / "connectivity"
GLB_ROOT = DATA_ROOT / "habitat" / "scene_datasets" / "mp3d"
OUT_ROOT = DATA_ROOT / "mp3d" / "v1" / "scans"

VAL_UNSEEN_SCANS = [
    "2azQ1b91cZZ",
    "8194nk5LbLH",
    "EU6Fwq7SyZv",
    "QUCTc6BB5sX",
    "TbHJrupSAjP",
    "X7HyMhZNoso",
    "Z6MFQCViBuw",
    "oLBMNvg9in8",
    "pLe4wQe7qrG",
    "x8F5xyUWy9e",
    "zsNo4HB9uLZ",
]

# Per-face camera-frame rotations relative to the per-viewpoint reference
# (i1_5) camera. These are pulled VERBATIM from MatterSim's own
# scripts/depth_to_skybox.py:40-47 (the canonical skybox face order:
# ``[up, forward, right, back, left, down]``).
# Comment from that file: "Matterport camera is really y=up, x=right, -z=look."
SKYBOX_TRANSFORMS = [
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),  # 0: up
    np.eye(3, dtype=np.float64),  # 1: forward
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),  # 2: right
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),  # 3: back
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),  # 4: left
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),  # 5: down
]
PI = float(np.pi)


def _radial_multiplier_mask(resolution: int) -> np.ndarray:
    """Per-pixel multiplier to convert perpendicular-Z depth to radial depth.

    For pixel (u, v) of an H=W=resolution image with hfov=90,
    radial = z_perp * sqrt(1 + ((u-W/2)/(W/2))^2 + ((v-H/2)/(H/2))^2).
    """
    half = resolution / 2.0
    grid_u, grid_v = np.meshgrid(np.arange(resolution), np.arange(resolution))
    nx = (grid_u + 0.5 - half) / half
    ny = (grid_v + 0.5 - half) / half
    return np.sqrt(1.0 + nx * nx + ny * ny).astype(np.float32)


def _load_connectivity(scan_id: str) -> list[dict]:
    path = CONN_DIR / f"{scan_id}_connectivity.json"
    with open(path) as f:
        return json.load(f)


_M_MP3D_TO_HABITAT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
"""3x3 axis swap: MP3D world (Z-up) -> habitat world (Y-up). habitat_v = M @ mp3d_v."""


def _mp3d_pos_to_habitat(pose: list[float]) -> np.ndarray:
    """Extract translation from row-major 4x4 and convert MP3D->habitat frame."""
    x, y, z = pose[3], pose[7], pose[11]
    return _M_MP3D_TO_HABITAT @ np.array([x, y, z], dtype=np.float64)


def _mp3d_pose_rotation(pose: list[float]) -> np.ndarray:
    """Extract 3x3 rotation R (local->world, MP3D frame) from row-major 4x4 pose.

    The cube map's intrinsic axes are in the viewpoint's LOCAL frame (per
    MatterSim.cpp:460 ``cameraRotation`` -> ``Model = R * Scale``). So we set
    the habitat agent's world rotation to ``R_agent = M @ R`` and use the
    standard 6 cube-face sensor orientations in agent-local frame to render
    the local cube faces.
    """
    p = np.array(pose, dtype=np.float64).reshape(4, 4)
    return p[:3, :3]


def _rotation_to_quaternion(rot_3x3: np.ndarray):
    """Build a numpy quaternion (w,x,y,z) from a 3x3 rotation matrix."""
    import quaternion

    return quaternion.from_rotation_matrix(rot_3x3)


def _build_sim(scene_glb: str, resolution: int):
    """Create a habitat_sim.Simulator with one depth sensor (forward-looking).

    Per-face rendering uses 6 set_state() calls with different agent rotations
    (one per cube face), matching depth_to_skybox.py's per-face camera approach.
    """
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.gpu_device_id = 0

    spec = habitat_sim.SensorSpec()
    spec.uuid = "depth"
    spec.sensor_type = habitat_sim.SensorType.DEPTH
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, 0.0, 0.0]
    spec.orientation = np.zeros(3, dtype=np.float32)  # default look at -Z
    spec.parameters["hfov"] = "90"
    spec.parameters["near"] = "0.05"
    spec.parameters["far"] = "1000.0"

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _render_viewpoint(
    sim, position_habitat: np.ndarray, r_pose_mp3d: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Render 6 cube faces and stitch into 3072x512 uint16 (radial depth).

    The connectivity pose stores camera-to-world in OpenCV-style camera
    convention (x=right, y=DOWN, z=forward). depth_to_skybox.py's
    skybox_transforms are expressed in Matterport convention (x=right, y=UP,
    -z=forward). The conversion is ``diag(1, -1, -1)``: flip Y (down→up) and
    flip Z (forward sign) to preserve right-handedness::
        R_face_world_mp3d = R_pose @ C @ skybox_transforms[k]
        R_face_habitat   = M @ R_face_world_mp3d
    where C = diag(1, -1, -1). Habitat's default sensor (look at -Z) then
    renders the correct face.
    """
    import habitat_sim

    agent = sim.get_agent(0)
    M = _M_MP3D_TO_HABITAT
    C = np.diag([1.0, -1.0, -1.0])  # OpenCV-cam -> Matterport-cam

    faces_in_col_order: list[np.ndarray] = [None] * 6  # type: ignore
    for col, st in enumerate(SKYBOX_TRANSFORMS):
        r_face_habitat = M @ r_pose_mp3d @ C @ st
        state = habitat_sim.AgentState()
        state.position = position_habitat
        state.rotation = _rotation_to_quaternion(r_face_habitat)
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        depth_perp = obs["depth"].astype(np.float32)
        # Perpendicular Z -> radial Euclidean (matches MatterSim's stored convention).
        depth_radial = depth_perp * mask
        # metres -> uint16 (1 unit = 0.25mm = 1/4000 m).
        u16 = np.clip(np.rint(depth_radial * 4000.0), 0, 65535).astype(np.uint16)
        faces_in_col_order[col] = u16

    return np.hstack(faces_in_col_order)


def _save_png_uint16(path: Path, image_u16: np.ndarray) -> None:
    """Write 16-bit grayscale PNG. Uses cv2 if available, falls back to PIL."""
    try:
        import cv2

        ok = cv2.imwrite(str(path), image_u16, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")
    except ImportError:
        from PIL import Image

        Image.fromarray(image_u16, mode="I;16").save(path, format="PNG", compress_level=9)


def generate(
    scans: Iterable[str],
    resolution: int = 512,
    viewpoints_filter: set[str] | None = None,
    force: bool = False,
    sample_n: int | None = None,
) -> None:
    mask = _radial_multiplier_mask(resolution)

    for scan_id in scans:
        glb_path = GLB_ROOT / scan_id / f"{scan_id}.glb"
        out_dir = OUT_ROOT / scan_id / "matterport_skybox_images"
        if not glb_path.exists():
            print(f"[skip] {scan_id}: missing .glb at {glb_path}", flush=True)
            continue
        if not out_dir.exists():
            print(f"[skip] {scan_id}: missing out dir {out_dir}", flush=True)
            continue

        viewpoints = _load_connectivity(scan_id)
        viewpoints = [v for v in viewpoints if v.get("included", True)]
        if viewpoints_filter is not None:
            viewpoints = [v for v in viewpoints if v["image_id"] in viewpoints_filter]
        if sample_n is not None:
            viewpoints = viewpoints[:sample_n]

        if not viewpoints:
            print(f"[skip] {scan_id}: no viewpoints to render", flush=True)
            continue

        print(f"[scan] {scan_id}: {len(viewpoints)} viewpoints, glb={glb_path.name}", flush=True)
        t0 = time.time()
        sim = _build_sim(str(glb_path), resolution)
        try:
            for i, vp in enumerate(viewpoints):
                vp_id = vp["image_id"]
                out_path = out_dir / f"{vp_id}_skybox_depth_small.png"
                if out_path.exists() and not force:
                    print(f"  [skip-existing] {vp_id}", flush=True)
                    continue
                pos_h = _mp3d_pos_to_habitat(vp["pose"])
                r_pose = _mp3d_pose_rotation(vp["pose"])
                image = _render_viewpoint(sim, pos_h, r_pose, mask)
                _save_png_uint16(out_path, image)
                if (i + 1) % 25 == 0 or i == len(viewpoints) - 1:
                    print(f"  [{i + 1}/{len(viewpoints)}] {vp_id}", flush=True)
        finally:
            sim.close()
        print(f"[scan] {scan_id}: done in {time.time() - t0:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scans", nargs="+", default=VAL_UNSEEN_SCANS, help="Scan IDs (default: val_unseen list)"
    )
    parser.add_argument(
        "--viewpoints",
        nargs="*",
        default=None,
        help="If set, only render these viewpoint IDs across all scans",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing PNGs (default: skip)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Render only first N viewpoints per scan (for debugging)",
    )
    args = parser.parse_args(argv)

    vp_filter = set(args.viewpoints) if args.viewpoints else None
    generate(
        scans=args.scans,
        resolution=args.resolution,
        viewpoints_filter=vp_filter,
        force=args.force,
        sample_n=args.sample,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
