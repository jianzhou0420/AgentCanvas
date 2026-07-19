from __future__ import annotations

"""pySLAM backend — drives one ``Slam`` instance inside the ac-pyslam subprocess.

This module is imported **only** in the ac-pyslam server-mode subprocess
(``server_python`` resolves to the ``ac-pyslam`` conda env). It is *not*
imported at nodeset-scan time — the framework (agentcanvas) env has no
pyslam — so every ``import pyslam`` here is lazy, inside a method, mirroring
``model_sam`` / ``model_detany3d``.

Faithfulness (against upstream ``main_slam.py``, pinned 2026-07-06):
    - construction  : ``Slam(camera, feature_tracker_config, loop_detector_config,
                      sensor_type=..., headless=True)`` (slam.py:105)
    - presets       : ``FeatureTrackerConfigs.ORB2`` / ``LoopDetectorConfigs.DBOW3``,
                      selected by name via ``get_config_from_name`` (main_slam.py:168-204)
    - per-frame     : ``slam.track(img, img_right, depth, img_id, ts)`` (slam.py:306);
                      RGBD → ``img_right=None`` (main_slam.py:371)
    - pose read     : ``slam.tracking.cur_R`` / ``cur_t`` (main_slam.py:398-402)
    - state         : ``slam.tracking.state`` (SlamState; LOST sentinel)
    - reset (episode): ``reset_session()`` clears the map but keeps the loop DB
                      (upstream issue #131); ``reset()`` is the hard all-clear
    - teardown      : ``slam.quit()`` joins the local-mapping / loop-closing /
                      semantic / volumetric background threads (slam.py:215)

Validated end-to-end 2026-07-07 against the real ac-pyslam container on a
TUM fr1_xyz RGB-D sequence (58/60 frames OK, 2788 map points, get_map handle
written), including both former SCAFFOLD seams:
    (1) camera-config marshalling — ``Config.from_json`` wants a **nested**
        payload (``cam_settings`` + ``dataset_settings``), plus ``Camera.bf``
        for the RGB-D virtual baseline; see ``_camera_config_dict``.
    (2) sparse-map export — ``slam.map.num_points()`` / ``get_points()`` (a set
        of MapPoint, each ``.pt()`` → 3-D coord) / ``num_keyframes()``.
Critical: ``PinholeCamera`` is imported from the dispatch module ``pyslam.slam``
(not ``pyslam.slam.camera``) so the C++ ``Frame`` accepts it when USE_CPP_CORE
is on — see ``_build``.
"""

import concurrent.futures
import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger("agentcanvas.pyslam")


def _queue_size(q: Any) -> int:
    """multiprocessing.Queue.qsize() raises NotImplementedError on some platforms
    (not Linux) — degrade to 'assume drained' so the dense-map wait can't hang."""
    try:
        return int(q.qsize())
    except Exception:
        return 0


# ── camera intrinsics ─────────────────────────────────────────────────────


def _habitat_intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """Pinhole ``fx, fy, cx, cy`` for a Habitat camera.

    Habitat renders with a horizontal FOV and square pixels, principal point
    at the image centre. ``fx = (W/2) / tan(hfov/2)``; ``fy == fx`` for square
    pixels (Habitat's vertical FOV is derived from the aspect ratio).
    """
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def _camera_config_dict(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sensor_type: str,
    depth_factor: float = 1.0,
    fps: int = 10,
    bf: float | None = None,
) -> dict:
    """Assemble the dict ``PinholeCamera(dict)`` → ``Config.from_json`` expects.

    Verified against pyslam ``config.py`` (Config.from_json) + ``camera.py``
    2026-07-07: ``from_json`` reads a **nested** payload — intrinsics live under
    ``cam_settings`` and the sensor type under ``dataset_settings["sensor_type"]``.
    A flat ``{"Camera.width": …}`` dict is silently ignored (from_json's
    ``data.get("cam_settings")`` is None) and pyslam falls back to its repo
    default (KITTI, 1226x370 mono) — which then fails the RGBD shape assert in
    ``tracking.track``. Habitat/undistorted frames → zero distortion.
    ``DepthMapFactor`` is the divisor pyslam applies to raw depth (frame.py
    ``depth * camera.depth_factor``, depth_factor = 1/DepthMapFactor); pass depth
    already in metres with depth_factor=1.0 to leave it untouched.
    """
    # RGB-D needs a virtual stereo baseline: pyslam's Frame.compute_stereo_from_rgbd
    # turns depth into stereo disparity via camera.bf (= baseline * fx). Missing bf
    # → "unsupported operand / NoneType" at the first tracked frame. Habitat has no
    # real baseline, so a virtual ~5 cm one (bf = fx * 0.05) plays the ORB-SLAM2 RGB-D
    # role of splitting close (stereo-constrained) vs far (mono) map points.
    bf_val = float(bf) if bf is not None else float(fx) * 0.05
    return {
        "cam_settings": {
            "Camera.width": int(width),
            "Camera.height": int(height),
            "Camera.fx": float(fx),
            "Camera.fy": float(fy),
            "Camera.cx": float(cx),
            "Camera.cy": float(cy),
            "Camera.fps": int(fps),
            "Camera.k1": 0.0,
            "Camera.k2": 0.0,
            "Camera.p1": 0.0,
            "Camera.p2": 0.0,
            "Camera.k3": 0.0,
            "Camera.bf": bf_val,
            "Camera.RGB": 1,
            "ThDepth": 40.0,
            "DepthMapFactor": 1.0 / depth_factor if depth_factor else 1.0,
        },
        "dataset_settings": {"sensor_type": sensor_type},  # "mono" | "stereo" | "rgbd"
    }


# ── session ────────────────────────────────────────────────────────────────


class PySlamSession:
    """One pyslam ``Slam`` instance pinned to a single worker thread.

    Thread affinity is not optional: pyslam spawns background threads
    (local mapping / loop closing / semantic / volumetric) bound to the
    thread that constructed ``Slam``, and ``track()`` must run on that same
    thread. Every pyslam call is therefore marshalled onto a 1-worker
    executor — the same single-thread-affinity discipline the env nodesets
    use for GL/physics handles.

    Lifecycle vs. the map:
        ``start()``  builds ``Slam`` (once camera intrinsics are known).
        ``reset()``  clears the map for a new episode — never rebuilds ``Slam``.
        ``close()``  ``quit()``s ``Slam`` (joins bg threads) then stops the executor.
    """

    def __init__(
        self,
        *,
        sensor_type: str = "rgbd",
        feature_preset: str = "ORB2",
        loop_preset: str = "DBOW3",
        headless: bool = True,
        volumetric: bool = False,
        volumetric_type: str = "VOXEL_GRID",
        environment: str = "INDOOR",
    ) -> None:
        self.sensor_type = sensor_type
        self.feature_preset = feature_preset
        self.loop_preset = loop_preset
        self.headless = headless
        # Dense volumetric mapping is a session verb: when on, Slam's own
        # volumetric integrator (TSDF / voxel grid) consumes keyframes on its bg
        # thread and get_dense_map() extracts the accumulated volume. Off by
        # default — the integrator is extra threads + memory not every run wants.
        self.volumetric = volumetric
        self.volumetric_type = volumetric_type  # VOXEL_GRID | TSDF | VOXEL_SEMANTIC_GRID | ...
        self.environment = environment  # INDOOR | OUTDOOR — sets dense depth-trunc thresholds

        self._slam: Any = None
        self._camera: Any = None
        self._cam_cfg: dict | None = None

        self._traj: list[list] = []  # accumulated 4x4 world_T_cam poses
        self._states: list[str] = []
        self._img_id: int = 0

        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pyslam"
        )
        self._lock = threading.Lock()

    # -- executor marshalling --

    def _run(self, fn, *args, **kwargs):
        """Run *fn* on the single pyslam-affine thread and block for its result."""
        return self._exec.submit(fn, *args, **kwargs).result()

    @property
    def is_built(self) -> bool:
        return self._slam is not None

    # -- camera --

    def configure_camera(
        self,
        width: int,
        height: int,
        hfov_deg: float | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        depth_factor: float = 1.0,
        bf: float | None = None,
    ) -> None:
        """Pin the pinhole intrinsics. Either give ``hfov_deg`` (Habitat) or a
        full ``fx/fy/cx/cy``. ``bf`` is the RGB-D virtual stereo baseline*fx
        (defaults to fx*0.05). Must be called before :meth:`start`."""
        if fx is None:
            if hfov_deg is None:
                raise ValueError("configure_camera: need either hfov_deg or explicit fx/fy/cx/cy")
            fx, fy, cx, cy = _habitat_intrinsics(width, height, hfov_deg)
        self._cam_cfg = _camera_config_dict(
            width, height, fx, fy, cx, cy, self.sensor_type, depth_factor, bf=bf
        )
        log.info("pyslam camera pinned: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 width, height, fx, fy, cx, cy)

    # -- lifecycle --

    def start(self) -> None:
        if self._cam_cfg is None:
            raise RuntimeError("PySlamSession.start(): call configure_camera() first")
        self._run(self._build)

    def _build(self) -> None:
        # LAZY imports — resolve only inside the ac-pyslam subprocess.
        from pyslam.io.dataset_types import DatasetEnvironmentType, get_sensor_type
        from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
        from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

        # PinholeCamera MUST come from the dispatch module `pyslam.slam`, not
        # `pyslam.slam.camera`: when USE_CPP_CORE is on, `pyslam.slam` re-exports
        # `cpp_core.PinholeCamera` (the C++ camera), which the C++ `Frame`
        # requires. Importing the Python camera directly makes the first tracked
        # frame throw "Unable to cast PinholeCamera to shared_ptr<pyslam::Camera>".
        from pyslam.slam import PinholeCamera
        from pyslam.slam.slam import Slam

        # Dense volumetric mapping is toggled through pyslam's global Parameters
        # BEFORE Slam is constructed — Slam.init_volumetric_integrator() reads
        # kDoVolumetricIntegration in __init__ to decide whether to spin up the
        # integrator thread (slam.py:init_volumetric_integrator). Set it here so
        # the switch is per-session, not a process-wide default.
        if self.volumetric:
            from pyslam.config_parameters import Parameters

            Parameters.kDoVolumetricIntegration = True
            Parameters.kVolumetricIntegrationType = self.volumetric_type
            log.info("pyslam volumetric integration ON: type=%s", self.volumetric_type)

        env_type = DatasetEnvironmentType[self.environment.upper()]
        self._camera = PinholeCamera(self._cam_cfg)
        feature_cfg = FeatureTrackerConfigs.get_config_from_name(self.feature_preset)
        loop_off = (not self.loop_preset) or self.loop_preset.lower() == "off"
        loop_cfg = None if loop_off else LoopDetectorConfigs.get_config_from_name(self.loop_preset)

        self._slam = Slam(
            self._camera,
            feature_cfg,
            loop_cfg,
            sensor_type=get_sensor_type(self.sensor_type),
            environment_type=env_type,
            headless=self.headless,
        )
        log.info(
            "pyslam Slam built: sensor=%s feature=%s loop=%s env=%s volumetric=%s headless=%s",
            self.sensor_type, self.feature_preset, self.loop_preset,
            self.environment, self.volumetric, self.headless,
        )

    def reset(self) -> None:
        """Clear the map for a new episode (keeps the loop DB — issue #131)."""
        self._run(self._reset)

    def _reset(self) -> None:
        if self._slam is not None:
            self._slam.reset_session()
        self._traj.clear()
        self._states.clear()
        self._img_id = 0

    def close(self) -> None:
        try:
            if self._slam is not None:
                self._run(self._close)
        finally:
            self._exec.shutdown(wait=True)

    def _close(self) -> None:
        # quit() joins pyslam's background threads (硬点 c — no zombie mappers).
        self._slam.quit()
        # Give the daemons a beat to unwind before the process may exit.
        time.sleep(0.2)
        self._slam = None

    # -- per-frame --

    def track(self, rgb: np.ndarray, depth: np.ndarray | None = None,
              timestamp: float | None = None) -> dict:
        """Feed one frame. RGBD → depth carries geometry, right image is None."""
        if self._slam is None:
            raise RuntimeError("PySlamSession.track(): Slam not built — call start()")
        return self._run(self._track, rgb, depth, timestamp)

    def _track(self, rgb: np.ndarray, depth: np.ndarray | None,
               timestamp: float | None) -> dict:
        img_id = self._img_id
        self._img_id += 1
        ts = float(timestamp) if timestamp is not None else float(img_id)

        # RGBD path: img_right=None (main_slam.py:371); depth is the geometry.
        self._slam.track(rgb, None, depth, img_id, ts)

        tk = self._slam.tracking
        state = getattr(tk.state, "name", str(tk.state))
        self._states.append(state)

        pose = None
        if getattr(tk, "cur_R", None) is not None and getattr(tk, "cur_t", None) is not None:
            T = np.eye(4)
            T[:3, :3] = np.asarray(tk.cur_R, dtype=float)
            T[:3, 3] = np.asarray(tk.cur_t, dtype=float).reshape(3)
            pose = T
            self._traj.append(T.tolist())

        return {
            "pose": pose.tolist() if pose is not None else None,
            "state": state,
            "num_map_points": self._num_map_points(),
            "frame_id": img_id,
        }

    # -- getters --

    def get_trajectory(self) -> dict:
        """Accumulated camera trajectory (one 4x4 per successfully tracked frame)."""
        return {"poses": list(self._traj), "states": list(self._states),
                "num_frames": self._img_id}

    def get_map_arrays(self) -> dict:
        """Return the raw sparse map (points + counts) **without** touching disk.

        The container bridge uses this: the shim ships the points back to the
        framework side, which writes the host-side handle file — so the file is
        owned by the framework user, sidestepping rootless-Docker uid remapping.
        """
        return self._run(self._get_map_arrays)

    def _get_map_arrays(self) -> dict:
        m = getattr(self._slam, "map", None)
        points = self._map_points(m)
        return {
            "points": points,
            "num_points": len(points),
            "num_keyframes": int(self._map_num_keyframes(m)),
        }

    def get_map(self, out_dir: str | None = None) -> dict:
        """Export the sparse map (keyframes + 3-D landmarks) to a handle.

        SCAFFOLD (seam 2): heavy geometry never rides an inline wire — we dump
        to disk in this process and return the path. The exact ``slam.map``
        accessors are read defensively; confirm against the real map class on
        the first ac-pyslam import.
        """
        return self._run(self._get_map, out_dir)

    def _get_map(self, out_dir: str | None) -> dict:
        arrs = self._get_map_arrays()
        points = arrs["points"]

        out_dir = out_dir or os.environ.get(
            "PYSLAM_ARTIFACT_DIR", os.path.join(os.getcwd(), "outputs", "pyslam_maps")
        )
        os.makedirs(out_dir, exist_ok=True)
        handle = os.path.join(out_dir, f"map_{self._img_id:06d}.npz")
        np.savez_compressed(handle, points=points)
        log.info("pyslam map exported: %d points, %d keyframes → %s",
                 len(points), arrs["num_keyframes"], handle)
        return {"map_handle": handle, "num_points": arrs["num_points"],
                "num_keyframes": arrs["num_keyframes"]}

    # -- dense volumetric map (session verb; requires volumetric=True at build) --
    #
    # Unlike the sparse map (a set of landmark points), the dense map is Slam's own
    # volumetric integrator output: a TSDF mesh or a fused voxel point cloud. The
    # integrator runs on a bg thread consuming keyframes; on demand we flush the
    # keyframe queue, request a fresh output, and pop it. Verified against
    # slam.get_dense_map / VolumetricIntegratorBase (flush_keyframe_queue +
    # add_update_output_task + pop_output → VolumetricIntegrationOutput) 2026-07-08.

    def get_dense_map_arrays(self) -> dict:
        """Return the dense volumetric map as raw numpy arrays (no disk).

        The container bridge uses this: the shim ships points/colors/vertices/
        triangles to the framework side, which writes the host-owned handle (same
        rootless-uid sidestep as get_map_arrays). Empty arrays if volumetric
        integration is off or nothing has been integrated yet."""
        return self._run(self._get_dense_map_arrays)

    def _get_dense_map_arrays(self) -> dict:
        empty = {
            "points": np.empty((0, 3), np.float32), "colors": np.empty((0, 3), np.float32),
            "vertices": np.empty((0, 3), np.float32), "triangles": np.empty((0, 3), np.int32),
            "num_points": 0, "num_vertices": 0, "num_triangles": 0, "type": self.volumetric_type,
        }
        vi = getattr(self._slam, "volumetric_integrator", None)
        if vi is None:
            return empty
        try:
            # The integrator consumes keyframes on a separate bg process, so at
            # extraction time its input queue (q_in) may still hold un-integrated
            # keyframes — popping an output *now* would return a partial/empty map
            # (the bug that made this return 0 points). Flush the pending keyframes,
            # wait for q_in to drain (all integrated), THEN request + pop a complete
            # output. Verified: without the drain-wait q_in=7 → 0 points; with it
            # → full voxel cloud.
            vi.flush_keyframe_queue()
            deadline = time.monotonic() + 60.0
            while _queue_size(vi.q_in) > 0 and time.monotonic() < deadline:
                time.sleep(0.1)
            vi.add_update_output_task()
            out = vi.pop_output()
        except Exception:
            log.exception("pyslam dense-map extraction failed")
            return empty
        if out is None:
            return empty

        points, colors, vertices, triangles = self._dense_arrays(out)
        return {
            "points": points, "colors": colors, "vertices": vertices, "triangles": triangles,
            "num_points": len(points), "num_vertices": len(vertices),
            "num_triangles": len(triangles), "type": self.volumetric_type,
        }

    def get_dense_map(self, out_dir: str | None = None) -> dict:
        """Standalone/local export of the dense map to a handle (bridge uses
        get_dense_map_arrays instead). Empty handle if nothing to export."""
        arrs = self.get_dense_map_arrays()
        if arrs["num_points"] == 0 and arrs["num_vertices"] == 0:
            return {"dense_handle": "", "num_points": 0, "num_vertices": 0,
                    "num_triangles": 0, "type": self.volumetric_type}
        out_dir = out_dir or os.environ.get(
            "PYSLAM_ARTIFACT_DIR", os.path.join(os.getcwd(), "outputs", "pyslam_maps")
        )
        os.makedirs(out_dir, exist_ok=True)
        handle = os.path.join(out_dir, f"dense_{self._img_id:06d}.npz")
        np.savez_compressed(handle, points=arrs["points"], colors=arrs["colors"],
                            vertices=arrs["vertices"], triangles=arrs["triangles"])
        log.info("pyslam dense map exported: %d points, %d verts, %d tris → %s",
                 arrs["num_points"], arrs["num_vertices"], arrs["num_triangles"], handle)
        return {"dense_handle": handle, "num_points": arrs["num_points"],
                "num_vertices": arrs["num_vertices"], "num_triangles": arrs["num_triangles"],
                "type": self.volumetric_type}

    @staticmethod
    def _dense_arrays(out: Any) -> tuple:
        """Pull numpy point/colour/vertex/triangle arrays out of a
        VolumetricIntegrationOutput (open3d-backed point cloud and/or mesh)."""
        def _pts(pc: Any, attr: str) -> np.ndarray:
            v = getattr(pc, attr, None)
            if v is not None and len(v):
                return np.asarray(v, dtype=np.float32)
            o3d = getattr(pc, "point_cloud", None)  # the raw open3d PointCloud
            if o3d is not None and hasattr(o3d, attr):
                return np.asarray(getattr(o3d, attr), dtype=np.float32)
            return np.empty((0, 3), dtype=np.float32)

        points = colors = np.empty((0, 3), dtype=np.float32)
        vertices = np.empty((0, 3), dtype=np.float32)
        triangles = np.empty((0, 3), dtype=np.int32)
        pc = getattr(out, "point_cloud", None)
        if pc is not None:
            points = _pts(pc, "points")
            colors = _pts(pc, "colors")
        mesh_wrap = getattr(out, "mesh", None)
        o3d_mesh = getattr(mesh_wrap, "mesh", None) if mesh_wrap is not None else None
        if o3d_mesh is not None:
            vertices = np.asarray(getattr(o3d_mesh, "vertices", []), dtype=np.float32)
            triangles = np.asarray(getattr(o3d_mesh, "triangles", []), dtype=np.int32)
        return points, colors, vertices, triangles

    # -- map introspection (seam 2, verified against cpp_core.Map 2026-07-07) --
    # slam.map.num_points() -> int; get_points() -> set[MapPoint]; each
    # MapPoint.pt() -> 3-D world coord. num_keyframes() -> int.

    def _num_map_points(self) -> int:
        try:
            m = getattr(self._slam, "map", None)
            return int(m.num_points()) if m is not None else 0
        except Exception:  # count is best-effort telemetry
            return 0

    @staticmethod
    def _map_points(m: Any) -> np.ndarray:
        if m is None:
            return np.empty((0, 3), dtype=float)
        coords = []
        for p in m.get_points():  # a set of MapPoint
            try:
                coords.append(np.asarray(p.pt(), dtype=float).reshape(3))
            except Exception:
                continue
        return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)

    @staticmethod
    def _map_num_keyframes(m: Any) -> int:
        if m is None:
            return 0
        try:
            return int(m.num_keyframes())
        except Exception:
            return 0


# ── stateless perception (Tier-2) ────────────────────────────────────────────
#
# These functions are pure request→response over pyslam's *standalone* classes —
# they never touch a ``PySlamSession`` / ``Slam`` object, so they carry no episode
# state and need no thread affinity (unlike the streaming session, whose bg
# threads pin it to one thread). Design note §2 / nodes 7-12: pyslam exposes its
# feature front-end and evo trajectory eval as independently-callable classes, so
# they get stateless nodes (``model_sam`` style, no session).
#
# Backends verified present in the container 2026-07-07 (source tree, not README):
#   - local_features.feature_manager.feature_manager_factory  → detectAndCompute
#   - local_features.feature_matcher.feature_matcher_factory  → match
#   - utilities.evaluation.evaluate_evo  (uses ``evo``; import-clean)
# Extractors/matchers are cached by config so repeated calls reuse one instance.

_FEATURE_MANAGERS: dict = {}
_FEATURE_MATCHERS: dict = {}
_DEPTH_ESTIMATORS: dict = {}
_STATELESS_LOCK = threading.Lock()


def _feature_enums(detector: str, descriptor: str):
    """Resolve the (detector, descriptor) name pair to pyslam enum members.

    Names are the ``FeatureDetectorTypes`` / ``FeatureDescriptorTypes`` members
    (ORB, SIFT, AKAZE, BRISK, SUPERPOINT, XFEAT, …). ``ORB2`` is the ORB-SLAM2
    C++ interface — it needs the ``orbslam2_features`` binding, which the CPU
    image may lack; plain ``ORB`` (OpenCV) is the safe default (no weights, no
    C++ extension).
    """
    from pyslam.local_features.feature_types import (
        FeatureDescriptorTypes,
        FeatureDetectorTypes,
    )

    det = FeatureDetectorTypes[detector.upper()]
    des = FeatureDescriptorTypes[(descriptor or detector).upper()]
    return det, des


def _get_feature_manager(detector: str, descriptor: str, num_features: int):
    key = (detector.upper(), (descriptor or detector).upper(), int(num_features))
    with _STATELESS_LOCK:
        fm = _FEATURE_MANAGERS.get(key)
        if fm is None:
            from pyslam.local_features.feature_manager import feature_manager_factory

            det, des = _feature_enums(detector, descriptor)
            fm = feature_manager_factory(
                num_features=int(num_features), detector_type=det, descriptor_type=des
            )
            _FEATURE_MANAGERS[key] = fm
        return fm


def _keypoints_to_array(kps: Any) -> np.ndarray:
    """cv2.KeyPoint list → Nx6 float32 ``[x, y, size, angle, response, octave]``.

    pyslam standardises detector output to ``cv2.KeyPoint``; the 6-column form is
    what a downstream matcher / drawer needs and marshals cleanly over the bridge.
    """
    if kps is None or len(kps) == 0:
        return np.empty((0, 6), dtype=np.float32)
    out = np.empty((len(kps), 6), dtype=np.float32)
    for i, k in enumerate(kps):
        pt = getattr(k, "pt", None)
        if pt is None:  # already an array-like [x, y]
            x, y = float(k[0]), float(k[1])
            out[i] = (x, y, 0.0, -1.0, 0.0, 0.0)
        else:
            out[i] = (pt[0], pt[1], k.size, k.angle, k.response, k.octave)
    return out


def extract_features(
    rgb: np.ndarray,
    *,
    detector: str = "ORB",
    descriptor: str = "ORB",
    num_features: int = 2000,
) -> dict:
    """Detect keypoints + compute descriptors on one image (stateless).

    Backed by ``FeatureManager.detectAndCompute`` (pyslam local_features). Returns
    the keypoints as an Nx6 array and the NxD descriptor block. Classical
    detectors (ORB / SIFT / AKAZE / BRISK) need no weights; learned ones
    (SUPERPOINT / XFEAT / DISK / ALIKED) auto-download a checkpoint on first use.
    """
    fm = _get_feature_manager(detector, descriptor, num_features)
    kps, des = fm.detectAndCompute(np.ascontiguousarray(rgb))
    kp_arr = _keypoints_to_array(kps)
    des_arr = np.empty((0, 0), dtype=np.float32) if des is None else np.asarray(des)
    return {
        "keypoints": kp_arr,
        "descriptors": des_arr,
        "num_keypoints": int(kp_arr.shape[0]),
        "detector": detector.upper(),
        "descriptor": (descriptor or detector).upper(),
    }


def _get_feature_matcher(matcher_type: str, norm_type: int, ratio_test: float, cross_check: bool):
    key = (matcher_type.upper(), int(norm_type), float(ratio_test), bool(cross_check))
    with _STATELESS_LOCK:
        m = _FEATURE_MATCHERS.get(key)
        if m is None:
            from pyslam.local_features.feature_matcher import (
                FeatureMatcherTypes,
                feature_matcher_factory,
            )

            m = feature_matcher_factory(
                norm_type=int(norm_type),
                cross_check=bool(cross_check),
                ratio_test=float(ratio_test),
                matcher_type=FeatureMatcherTypes[matcher_type.upper()],
            )
            _FEATURE_MATCHERS[key] = m
        return m


def match_features(
    des_a: np.ndarray,
    des_b: np.ndarray,
    *,
    matcher_type: str = "BF",
    ratio_test: float = 0.7,
    cross_check: bool = False,
    norm_type: int | None = None,
) -> dict:
    """Match two descriptor blocks (stateless), returning matched index pairs.

    Backed by ``feature_matcher_factory().match`` (pyslam local_features).
    ``norm_type`` defaults from the descriptor dtype: binary descriptors (ORB /
    AKAZE / BRISK, uint8) → Hamming; float descriptors (SIFT) → L2. Classical
    matchers (BF / FLANN) need only the descriptors — images/keypoints are unused,
    so this works purely on the arrays produced by :func:`extract_features`.
    """
    import cv2

    des_a = np.asarray(des_a)
    des_b = np.asarray(des_b)
    if norm_type is None:
        norm_type = cv2.NORM_HAMMING if des_a.dtype == np.uint8 else cv2.NORM_L2
    matcher = _get_feature_matcher(matcher_type, int(norm_type), ratio_test, cross_check)
    # BF/FLANN ignore img1/img2 (only detector-free matchers use them) — pass None.
    res = matcher.match(None, None, des_a, des_b)
    idxs_a = np.asarray(res.idxs1 if res.idxs1 is not None else [], dtype=np.int64)
    idxs_b = np.asarray(res.idxs2 if res.idxs2 is not None else [], dtype=np.int64)
    return {"idxs_a": idxs_a, "idxs_b": idxs_b, "num_matches": int(idxs_a.shape[0])}


# ── dense depth prediction (neural / classical, GPU when available) ──────────
#
# pyslam bundles a bank of depth estimators (its mono-SLAM depth bootstrap) behind
# one factory. We expose it as a stateless node: RGB (+ right image for stereo) in,
# a dense metric depth map out. ``device=None`` lets pyslam pick CUDA — the
# cpu-fixed image ships a cu128 torch, so the neural backends run on the GPU while
# the classical SLAM core stays on CPU. Estimators are cached by config because
# construction loads weights onto the GPU (expensive; reuse across frames).
#
# Verified in-container 2026-07-08 (source, not README):
#   depth_estimation.depth_estimator_factory.depth_estimator_factory(
#       depth_estimator_type=DepthEstimatorType[NAME], device=None, camera=None,
#       min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType[ENV],
#       precision=torch.float16) → estimator; estimator.infer(image, image_right=None)
#       → (depth_map HxW, None).  Types: DEPTH_SGBM(stereo,CPU), DEPTH_ANYTHING_V2
#       / DEPTH_PRO (mono,GPU), DEPTH_RAFT_STEREO / DEPTH_CRESTEREO_PYTORCH (stereo,
#       GPU), DEPTH_MAST3R / DEPTH_MVDUST3R (mono,GPU).  Env: INDOOR | OUTDOOR.


def _get_depth_estimator(estimator: str, min_depth: float, max_depth: float, environment: str):
    key = (estimator.upper(), float(min_depth), float(max_depth), environment.upper())
    with _STATELESS_LOCK:
        de = _DEPTH_ESTIMATORS.get(key)
        if de is None:
            from pyslam.depth_estimation.depth_estimator_factory import (
                DepthEstimatorType,
                depth_estimator_factory,
            )
            from pyslam.io.dataset_types import DatasetEnvironmentType

            de = depth_estimator_factory(
                depth_estimator_type=DepthEstimatorType[estimator.upper()],
                device=None,  # None → pyslam picks CUDA when available
                camera=None,  # mono estimators need no camera; stereo derive from image pair
                min_depth=float(min_depth),
                max_depth=float(max_depth),
                dataset_env_type=DatasetEnvironmentType[environment.upper()],
            )
            _DEPTH_ESTIMATORS[key] = de
        return de


def predict_depth(
    rgb: np.ndarray,
    image_right: np.ndarray | None = None,
    *,
    estimator: str = "DEPTH_ANYTHING_V2",
    min_depth: float = 0.0,
    max_depth: float = 50.0,
    environment: str = "INDOOR",
) -> dict:
    """Predict a dense depth map from one image (stateless, GPU when available).

    Backed by pyslam ``depth_estimation.depth_estimator_factory`` — the same
    estimator bank pySLAM uses to bootstrap depth for a monocular stream, exposed
    here as a standalone tool. Mono estimators (DEPTH_ANYTHING_V2 / DEPTH_PRO /
    DEPTH_MAST3R) use ``rgb`` only; stereo estimators (DEPTH_SGBM / DEPTH_RAFT_STEREO
    / DEPTH_CRESTEREO_PYTORCH) additionally consume ``image_right``. ``infer``
    returns ``(depth_map, None)``; we hand back the HxW float32 depth in metres
    (metric scale depends on the estimator + ``max_depth``). Learned estimators
    auto-download a checkpoint on first use and run on the GPU (``device=None`` →
    CUDA when present); DEPTH_SGBM is pure OpenCV on the CPU.
    """
    de = _get_depth_estimator(estimator, min_depth, max_depth, environment)
    rgb = np.ascontiguousarray(rgb)
    img_r = None if image_right is None else np.ascontiguousarray(image_right)
    depth_map, _ = de.infer(rgb, img_r)
    depth = np.ascontiguousarray(np.asarray(depth_map, dtype=np.float32))
    return {
        "depth": depth,
        "estimator": estimator.upper(),
        "shape": list(depth.shape),
        "min": float(depth.min()) if depth.size else 0.0,
        "max": float(depth.max()) if depth.size else 0.0,
    }


# ── semantic segmentation (neural, GPU when available) ───────────────────────
#
# pyslam bundles a bank of segmentation backends behind one factory, used for its
# semantic mapping. Exposed here as a stateless node: RGB in, a per-pixel semantic
# map out (+ optional instance map). ``SemanticSegmentationOutput.semantics`` is
# (H,W) int labels for LABEL, (H,W,C) for PROBABILITY_VECTOR, (H,W,D) for
# FEATURE_VECTOR; ``.instances`` is an optional (H,W) instance map.
#
# Verified in-container 2026-07-08:
#   semantics.semantic_segmentation_factory.semantic_segmentation_factory(
#       semantic_segmentation_type=SemanticSegmentationType[NAME],
#       semantic_feature_type=SemanticFeatureType[FEAT],
#       semantic_dataset_type=SemanticDatasetType[DSET],
#       image_size=(512,512), device=None) → seg; seg.infer(image) → output;
#       seg.num_classes() → int.  Types: DEEPLABV3 / SEGFORMER / CLIP / EOV_SEG /
#       DETIC / ODISE / RFDETR / YOLO.  Feat: NONE|LABEL|PROBABILITY_VECTOR|
#       FEATURE_VECTOR.  Dset: CITYSCAPES|ADE20K|VOC|NYU40|FEATURE_SIMILARITY|
#       CUSTOM_SET.  (torchvision/HF weights auto-download into the container cache.)

_SEG_MODELS: dict = {}


def _get_segmenter(model: str, feature_type: str, dataset: str, image_size: tuple):
    key = (model.upper(), feature_type.upper(), dataset.upper(), tuple(image_size))
    with _STATELESS_LOCK:
        seg = _SEG_MODELS.get(key)
        if seg is None:
            import pyslam.semantics.semantic_segmentation_factory as F

            seg = F.semantic_segmentation_factory(
                semantic_segmentation_type=F.SemanticSegmentationType[model.upper()],
                semantic_feature_type=F.SemanticFeatureType[feature_type.upper()],
                semantic_dataset_type=F.SemanticDatasetType[dataset.upper()],
                image_size=tuple(image_size),
                device=None,  # None → pyslam picks CUDA when available
            )
            _SEG_MODELS[key] = seg
        return seg


def segment_semantic(
    rgb: np.ndarray,
    *,
    model: str = "DEEPLABV3",
    feature_type: str = "LABEL",
    dataset: str = "CITYSCAPES",
    image_size: tuple = (512, 512),
) -> dict:
    """Semantic-segment one image (stateless, GPU when available).

    Backed by pyslam ``semantics.semantic_segmentation_factory`` — the same
    segmentation bank pySLAM uses for semantic mapping, exposed as a standalone
    tool. Returns the per-pixel ``semantics`` map (int labels for LABEL, float
    (H,W,C)/(H,W,D) for PROBABILITY_VECTOR/FEATURE_VECTOR) and an optional
    ``instances`` map. Learned models auto-download a checkpoint on first use and
    run on the GPU (``device=None`` → CUDA when present).
    """
    seg = _get_segmenter(model, feature_type, dataset, image_size)
    out = seg.infer(np.ascontiguousarray(rgb))
    semantics = None if out.semantics is None else np.ascontiguousarray(out.semantics)
    instances = None if out.instances is None else np.ascontiguousarray(out.instances)
    try:
        num_classes = int(seg.num_classes())
    except Exception:
        num_classes = 0
    return {
        "semantics": semantics,
        "instances": instances,
        "num_classes": num_classes,
        "shape": None if semantics is None else list(semantics.shape),
        "model": model.upper(),
        "feature_type": feature_type.upper(),
    }


# ── multi-view reconstruction (DUSt3R / MASt3R / VGGT / … , GPU) ─────────────
#
# pyslam bundles a bank of feed-forward multi-view reconstructors behind one
# factory: N images in → a fused 3-D scene (global point cloud + optional mesh,
# per-view camera poses, intrinsics, depth). This is pyslam's "scene from views"
# surface — the DUSt3R-family stack — exposed here as a stateless node (no Slam
# session). Reconstructors are cached by backend name: construction loads a large
# transformer onto the GPU, so reuse across calls.
#
# Verified in-container 2026-07-08 (source, not README):
#   scene_from_views.scene_from_views_factory.scene_from_views_factory(
#       scene_from_views_type=SceneFromViewsType[NAME], device=None) → reconstructor;
#   reconstructor.reconstruct(images: List[np.ndarray], as_pointcloud=True)
#       → SceneFromViewsResult{global_point_cloud(trimesh.PointCloud),
#         global_mesh(trimesh.Trimesh), camera_poses(List[4x4]),
#         processed_images, depth_predictions, point_clouds,
#         intrinsics(List[3x3]), confidences}.
#   Types: DEPTH_ANYTHING_V3 | MAST3R | MVDUST3R | VGGT | VGGT_ROBUST | DUST3R |
#   FAST3R.  Weights mount in from data/models/pyslam/ (see _client._weight_mounts).
#   KNOWN GAP: MVDUST3R errors at import — pyslam's bundled mvdust3r has a version
#   drift vs its own dust3r copy (no normalize_pointclouds in dust3r.utils.geometry).
#   Left unfixed by decision (2026-07-08): the other 6 backends cover multi-view;
#   MASt3R is the verified default. Selecting MVDUST3R returns a benign node error.

_SCENE_RECONSTRUCTORS: dict = {}


def _allow_mast3r_checkpoint_globals() -> None:
    """Let torch>=2.6 load the MASt3R/DUSt3R checkpoints.

    pyslam's ``mast3r/model.py`` (and the DUSt3R loaders) call ``torch.load()``
    without ``weights_only=False``; torch>=2.6 flipped that default to True, and
    the officially-published MASt3R/DUSt3R checkpoints pickle an
    ``argparse.Namespace`` of train args, which the safe unpickler rejects
    (``UnsupportedGlobal: argparse.Namespace``). We allowlist exactly that global —
    the torch-sanctioned fix for a *trusted* checkpoint (downloaded from naver
    labs / Meta) — so the load succeeds without disabling the safe unpickler
    wholesale. This is an upstream pyslam incompatibility with the image's torch
    2.8, not a nodeset issue; the allowlist is a no-op for HF-safetensors backends
    (VGGT / Depth-Anything-V3 / Fast3r) that never hit torch.load."""
    import argparse
    import contextlib

    import torch

    # very old torch without the API → weights_only already False, nothing to do
    with contextlib.suppress(Exception):
        torch.serialization.add_safe_globals([argparse.Namespace])


def _get_scene_reconstructor(backend: str):
    key = backend.upper()
    with _STATELESS_LOCK:
        rec = _SCENE_RECONSTRUCTORS.get(key)
        if rec is None:
            from pyslam.scene_from_views.scene_from_views_factory import (
                scene_from_views_factory,
            )
            from pyslam.scene_from_views.scene_from_views_types import SceneFromViewsType

            _allow_mast3r_checkpoint_globals()
            rec = scene_from_views_factory(
                scene_from_views_type=SceneFromViewsType[key],
                device=None,  # None → pyslam picks CUDA when available
            )
            _SCENE_RECONSTRUCTORS[key] = rec
        return rec


def _pointcloud_arrays(pc: Any) -> tuple[np.ndarray, np.ndarray]:
    """trimesh.PointCloud → (points Nx3 float32, colors Nx3 uint8).

    trimesh stores colours as RGBA uint8 (Nx4); we drop alpha. Empty arrays if
    the cloud is None or has no vertices."""
    if pc is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    pts = np.asarray(getattr(pc, "vertices", []), dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    raw = getattr(pc, "colors", None)
    empty = np.empty((0, 3), np.uint8)
    if raw is None or not len(raw):
        return pts, empty
    cols = np.asarray(raw, dtype=np.uint8)  # trimesh stores RGBA uint8 (Nx4)
    cols = cols[:, :3] if cols.ndim == 2 and cols.shape[1] >= 3 else empty
    return pts, cols


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """trimesh.Trimesh → (vertices Nx3 float32, faces Mx3 int32). Empty if None."""
    if mesh is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int32)
    if verts.size == 0:
        verts = np.empty((0, 3), np.float32)
    if faces.size == 0:
        faces = np.empty((0, 3), np.int32)
    return verts, faces


def reconstruct_multiview(
    images: list,
    *,
    backend: str = "MAST3R",
    as_pointcloud: bool = True,
) -> dict:
    """Feed-forward multi-view 3-D reconstruction from N images (stateless, GPU).

    Backed by pyslam ``scene_from_views.scene_from_views_factory`` — the DUSt3R /
    MASt3R / VGGT / MV-DUSt3R family. Give ≥2 overlapping views; returns a fused
    global point cloud (points + colours), an optional global mesh, per-view
    camera-to-world poses (4x4), and intrinsics (3x3). ``as_pointcloud=False``
    asks the backend to also build a mesh. Weights load from the mounted
    data/models/pyslam/ folder (MASt3R / MV-DUSt3R) or HuggingFace on first use
    (VGGT / VGGT-Robust / DUSt3R / Depth-Anything-V3 / Fast3r).
    """
    imgs = [np.ascontiguousarray(im) for im in images]
    if len(imgs) < 2:
        raise ValueError("reconstruct_multiview: need at least 2 views")
    rec = _get_scene_reconstructor(backend)
    result = rec.reconstruct(imgs, as_pointcloud=bool(as_pointcloud))

    points, colors = _pointcloud_arrays(getattr(result, "global_point_cloud", None))
    vertices, faces = _mesh_arrays(getattr(result, "global_mesh", None))
    poses = [np.asarray(p, dtype=np.float32) for p in (result.camera_poses or [])]
    intr = [np.asarray(k, dtype=np.float32) for k in (getattr(result, "intrinsics", None) or [])]
    return {
        "points": points,
        "colors": colors,
        "vertices": vertices,
        "faces": faces,
        "camera_poses": poses,       # list of 4x4 cam-to-world
        "intrinsics": intr,          # list of 3x3
        "num_points": len(points),
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "num_views": len(imgs),
        "backend": backend.upper(),
    }


def _coerce_poses(poses: Any) -> list:
    """Normalise a trajectory to a list of 4x4 float ndarrays (evo's ``poses_se3``)."""
    out = []
    for p in poses:
        arr = np.asarray(p, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError(f"eval_trajectory: pose must be 4x4, got {arr.shape}")
        out.append(arr)
    return out


def _clean_stats(stats: Any) -> dict:
    """evo ``get_all_statistics()`` → a plain JSON-safe float dict."""
    if not isinstance(stats, dict):
        return {}
    return {k: float(v) for k, v in stats.items()}


def _rpe_stats(est: list, gt: list, is_monocular: bool) -> dict:
    """Best-effort RPE(translation, delta=1 frame) via evo, mirroring evaluate_evo's
    alignment convention (Umeyama, scale-corrected iff monocular).

    RPE measures local frame-to-frame drift, so it survives a **degenerate**
    (near-straight-line) trajectory that Umeyama alignment can't resolve: if the
    global align raises, we fall back to computing RPE without it — common for a
    straight-corridor VLN episode where ATE alignment is ill-posed but drift still
    matters.
    """
    try:
        from evo.core import metrics
        from evo.core.metrics import PoseRelation, Unit
        from evo.core.trajectory import PosePath3D

        traj_est = PosePath3D(poses_se3=est)
        traj_ref = PosePath3D(poses_se3=gt)
        try:
            traj_est.align(traj_ref=traj_ref, correct_scale=is_monocular)
        except Exception as align_exc:  # degenerate covariance → skip global align
            log.info("eval_trajectory: RPE without alignment (%s)", align_exc)
        rpe = metrics.RPE(
            PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=False
        )
        rpe.process_data((traj_ref, traj_est))
        return _clean_stats(rpe.get_all_statistics())
    except Exception as exc:  # RPE is a bonus over the ATE evaluate_evo already gives
        log.warning("eval_trajectory: RPE computation failed: %s", exc)
        return {}


def eval_trajectory(poses_est: Any, poses_gt: Any, *, is_monocular: bool = False) -> dict:
    """ATE (+ best-effort RPE) between an estimated and a ground-truth trajectory.

    Mirrors pyslam ``utilities/evaluation.py:evaluate_evo`` — Umeyama-align est→gt
    (scale-corrected iff monocular), then APE on the translation part. Pure math
    (``evo``), no disk / no plots (``save_metrics=save_plot=False``). Trajectories
    are truncated to the shorter length so a partial run still evaluates.
    """
    from pyslam.utilities.evaluation import evaluate_evo

    est = _coerce_poses(poses_est)
    gt = _coerce_poses(poses_gt)
    n = min(len(est), len(gt))
    if n == 0:
        return {"ate": {}, "rpe": {}, "num_poses": 0, "is_monocular": bool(is_monocular)}
    est, gt = est[:n], gt[:n]
    ate_stats, _T = evaluate_evo(
        est, gt, is_monocular, None, "", save_metrics=False, save_plot=False
    )
    return {
        "ate": _clean_stats(ate_stats),
        "rpe": _rpe_stats(est, gt, is_monocular),
        "num_poses": n,
        "is_monocular": bool(is_monocular),
    }
