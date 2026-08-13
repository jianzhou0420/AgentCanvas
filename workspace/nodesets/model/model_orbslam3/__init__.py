"""ORB-SLAM3 NodeSet — classic feature-based visual SLAM as canvas nodes.

Load:  POST /api/components/nodesets/model_orbslam3/load

First Container-Launch nodeset (design-docs/operations/container-launch.html):
``server_image`` names the docker image carrying the compiled ORB-SLAM3 stack
plus the ``orbslam3`` boost-python binding (hello-binit fork pair). This module
is imported twice — by the backend during discovery (declaration only; the
``import orbslam3`` lines below are lazy so the framework env never needs it)
and inside the container by the stock auto_host (where the import resolves and
the nodes actually run).

Session model: one module-level ``_Session`` singleton per process — the
container's auto_host process is long-lived, so SLAM state persists across
node fires. ``parallelism = "replicated"`` gives each eval worker its own
container/session.

Binding surface VERIFIED against the built image (2026-08-13, in-container
TUM fr1_xyz probe): ``System(voc, settings, Sensor.RGBD)``,
``process_image_rgbd(rgb, depth, ts) -> bool``, ``get_tracking_state()``
(``TrackingState.OK == 2``, ``LOST == 4``), ``get_trajectory_points()``
returning 13-tuples (ts + row-major 3x4). 60/60 frames tracked OK.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef
from app.components.bases import ConfigField

VOCAB_PATH = "/opt/orbslam3/ORBvoc.txt"

_SESSION: _Session | None = None


# ── In-container session wrapper ──


class _Session:
    """Owns one orbslam3.System instance and its trajectory bookkeeping."""

    def __init__(self, intrinsics: dict, depth_scale: float) -> None:
        import orbslam3  # lazy: resolves only inside the container

        self._orbslam3 = orbslam3
        self.depth_scale = depth_scale
        settings = self._write_settings(intrinsics)
        self.system = orbslam3.System(VOCAB_PATH, settings, orbslam3.Sensor.RGBD)
        self.system.set_use_viewer(False)
        self.system.initialize()
        self.frames_total = 0
        self.frames_ok = 0

    @staticmethod
    def _write_settings(intr: dict) -> str:
        """ORB-SLAM3 wants a YAML settings file; render one from intrinsics."""
        path = "/tmp/orbslam3_settings.yaml"
        fps = float(intr.get("fps", 10.0))
        depth_factor = float(intr.get("depth_factor", 1.0))
        text = "\n".join(
            [
                "%YAML:1.0",
                "File.version: \"1.0\"",
                "Camera.type: \"PinHole\"",
                f"Camera1.fx: {float(intr['fx'])}",
                f"Camera1.fy: {float(intr['fy'])}",
                f"Camera1.cx: {float(intr['cx'])}",
                f"Camera1.cy: {float(intr['cy'])}",
                "Camera1.k1: 0.0",
                "Camera1.k2: 0.0",
                "Camera1.p1: 0.0",
                "Camera1.p2: 0.0",
                f"Camera.width: {int(intr['width'])}",
                f"Camera.height: {int(intr['height'])}",
                f"Camera.fps: {int(fps)}",  # loader aborts on a float fps
                "Camera.RGB: 1",
                # Virtual stereo baseline*fx for RGBD (ORB-SLAM3 convention)
                f"Stereo.b: {float(intr.get('baseline', 0.05))}",
                "Stereo.ThDepth: 40.0",
                f"RGBD.DepthMapFactor: {depth_factor}",
                "ORBextractor.nFeatures: 1250",
                "ORBextractor.scaleFactor: 1.2",
                "ORBextractor.nLevels: 8",
                "ORBextractor.iniThFAST: 20",
                "ORBextractor.minThFAST: 7",
                "Viewer.KeyFrameSize: 0.05",
                "Viewer.KeyFrameLineWidth: 1.0",
                "Viewer.GraphLineWidth: 0.9",
                "Viewer.PointSize: 2.0",
                "Viewer.CameraSize: 0.08",
                "Viewer.CameraLineWidth: 3.0",
                "Viewer.ViewpointX: 0.0",
                "Viewer.ViewpointY: -0.7",
                "Viewer.ViewpointZ: -1.8",
                "Viewer.ViewpointF: 500.0",
                "",
            ]
        )
        with open(path, "w") as f:
            f.write(text)
        return path

    def track(self, rgb: Any, depth: Any, timestamp: float) -> dict:
        import numpy as np

        depth_m = np.asarray(depth, dtype=np.float32) * self.depth_scale
        self.system.process_image_rgbd(np.asarray(rgb), depth_m, timestamp)
        self.frames_total += 1
        state = int(self.system.get_tracking_state())
        ok = state == 2  # OK (0 SYSTEM_NOT_READY, 1 NOT_INITIALIZED, 3 LOST)
        pose = None
        if ok:
            self.frames_ok += 1
            traj = self.system.get_trajectory_points()
            if traj:
                # bare 4x4 (trajectoryViewer + pyslam convention)
                pose = _trajectory_entry_to_pose(traj[-1])["matrix"]
        return {"pose": pose, "tracking_ok": ok, "state": state}

    def trajectory(self) -> list:
        return [_trajectory_entry_to_pose(e) for e in self.system.get_trajectory_points()]

    def shutdown(self) -> None:
        self.system.shutdown()


def _trajectory_entry_to_pose(entry: Any) -> dict:
    """Binding trajectory entries are 13-tuples (verified against the built
    image): (timestamp, r00 r01 r02 t0, r10 r11 r12 t1, r20 r21 r22 t2) —
    a row-major 3x4; normalize to a 4x4 matrix."""
    ts = float(entry[0])
    m = [list(entry[1:5]), list(entry[5:9]), list(entry[9:13]), [0.0, 0.0, 0.0, 1.0]]
    return {"timestamp": ts, "matrix": m}


# ── Nodes ──


class OrbSlam3Reset(BaseCanvasNode):
    """(Re)create the SLAM session from camera intrinsics. Fire once per episode."""

    node_type = "model_orbslam3__reset"
    display_name = "ORB-SLAM3 Reset"
    description = "Create/reset the SLAM session from camera intrinsics"
    category = "model"
    icon = "RefreshCw"

    input_ports: ClassVar[list] = [
        PortDef("intrinsics", "STATE", "{fx, fy, cx, cy, width, height, ...}"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("ready", "STATE", "{ok: bool}"),
    ]
    config_fields: ClassVar[list] = [
        # Multiplier turning input depth into metres (VLN-CE normalised depth → 10.0)
        ConfigField("depth_scale", "text", "Depth scale", "1.0"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        global _SESSION
        if _SESSION is not None:
            _SESSION.shutdown()
        _SESSION = _Session(
            inputs["intrinsics"],
            depth_scale=float(self.config.get("depth_scale", 1.0)),
        )
        return {"ready": {"ok": True}}


class OrbSlam3Track(BaseCanvasNode):
    """Feed one RGBD frame; get the current camera pose."""

    node_type = "model_orbslam3__track"
    display_name = "ORB-SLAM3 Track"
    description = "Track one RGBD frame, return current pose"
    category = "model"
    icon = "Crosshair"

    input_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "RGB frame"),
        PortDef("depth", "DEPTH", "Depth frame (scaled by reset.depth_scale)"),
        PortDef("timestamp", "STATE", "{t: float seconds}; frame-counter fallback", optional=True),
    ]
    output_ports: ClassVar[list] = [
        PortDef("pose", "POSE", "4x4 world-from-camera pose (list of lists), or None when lost"),
        PortDef("tracking_ok", "BOOL", "True while tracking state is OK"),
        PortDef("num_frames_ok", "STATE", "{ok, total} frame counters"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        if _SESSION is None:
            raise RuntimeError("model_orbslam3: fire reset before track")
        # timestamp arrives as {t: float}, a bare float (env_tum), or absent
        ts = inputs.get("timestamp")
        if isinstance(ts, dict):
            t = float(ts.get("t", _SESSION.frames_total * 0.1))
        elif ts is None:
            t = _SESSION.frames_total * 0.1
        else:
            t = float(ts)
        result = _SESSION.track(inputs["rgb"], inputs["depth"], t)
        return {
            "pose": result["pose"],
            "tracking_ok": result["tracking_ok"],
            "num_frames_ok": {"ok": _SESSION.frames_ok, "total": _SESSION.frames_total},
        }


class OrbSlam3GetTrajectory(BaseCanvasNode):
    """Loop-closure-corrected trajectory so far."""

    node_type = "model_orbslam3__get_trajectory"
    display_name = "ORB-SLAM3 Trajectory"
    description = "Return the corrected camera trajectory"
    category = "model"
    icon = "Route"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Fire when upstream is done"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("trajectory", "STATE", "[{timestamp, matrix}] list"),
        PortDef("num_poses", "STATE", "{n: int}"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        if _SESSION is None:
            raise RuntimeError("model_orbslam3: fire reset before get_trajectory")
        traj = _SESSION.trajectory()
        return {"trajectory": traj, "num_poses": {"n": len(traj)}}


class OrbSlam3GetStatus(BaseCanvasNode):
    """Lightweight liveness/tracking query for agent-style consumers."""

    node_type = "model_orbslam3__get_status"
    display_name = "ORB-SLAM3 Status"
    description = "Tracking state + frame counters"
    category = "model"
    icon = "Activity"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Fire to sample status"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("status", "STATE", "{initialized, state, frames_ok, frames_total}"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        if _SESSION is None:
            return {"status": {"initialized": False}}
        return {
            "status": {
                "initialized": True,
                "state": int(_SESSION.system.get_tracking_state()),
                "frames_ok": _SESSION.frames_ok,
                "frames_total": _SESSION.frames_total,
            }
        }


class OrbSlam3SaveTrajectory(BaseCanvasNode):
    """Write the trajectory as TUM-format text; returns the file handle."""

    node_type = "model_orbslam3__save_trajectory"
    display_name = "ORB-SLAM3 Save Trajectory"
    description = "Save TUM-format trajectory, return file handle"
    category = "model"
    icon = "Save"

    input_ports: ClassVar[list] = [
        PortDef("trigger", "ANY", "Fire when upstream is done"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("handle", "TEXT", "Host-side path of the saved trajectory"),
    ]
    config_fields: ClassVar[list] = [
        ConfigField("out_dir", "text", "Container out dir", "/opt/out"),
        ConfigField("host_out_dir", "text", "Host out dir", "outputs/slam_runs"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        import os

        if _SESSION is None:
            raise RuntimeError("model_orbslam3: fire reset before save_trajectory")
        out_dir = str(self.config.get("out_dir", "/opt/out"))
        os.makedirs(out_dir, exist_ok=True)
        fname = f"traj_{_SESSION.frames_total:06d}.txt"
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            for p in _SESSION.trajectory():
                m = p["matrix"]
                # TUM: t tx ty tz qx qy qz qw
                qx, qy, qz, qw = _rot_to_quat(m)
                f.write(
                    f"{p['timestamp']:.6f} {m[0][3]:.6f} {m[1][3]:.6f} {m[2][3]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )
        host_dir = str(self.config.get("host_out_dir", "outputs/slam_runs"))
        return {"handle": os.path.join(host_dir, fname)}


def _rot_to_quat(m: list) -> tuple:
    """3x3 rotation (upper-left of 4x4) → (qx, qy, qz, qw)."""
    import numpy as np

    r = np.asarray(m, dtype=float)[:3, :3]
    tr = np.trace(r)
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2
        return (
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
            0.25 * s,
        )
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = (1.0 + r[i, i] - r[j, j] - r[k, k]) ** 0.5 * 2
    q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return (q[0], q[1], q[2], q[3])


# ── NodeSet Registration ──


class OrbSlam3NodeSet(BaseNodeSet):
    """ORB-SLAM3 visual SLAM (RGBD) via Container Launch."""

    name = "model_orbslam3"
    description = "ORB-SLAM3 feature-based visual SLAM — reset/track/trajectory nodes"

    server_image = "agentcanvas/orbslam3:latest"
    server_image_gpu = False  # ORB-SLAM3 core is CPU-only
    server_mounts: ClassVar[dict[str, str]] = {
        "outputs/slam_runs": "/opt/out",  # writable handle drop (rw by default)
    }
    parallelism = "replicated"  # stateful SLAM session — one container per worker
    expected_ram_mb = 1500

    def get_tools(self) -> list:
        return [
            OrbSlam3Reset(),
            OrbSlam3Track(),
            OrbSlam3GetTrajectory(),
            OrbSlam3GetStatus(),
            OrbSlam3SaveTrajectory(),
        ]

    async def shutdown(self) -> None:
        global _SESSION
        if _SESSION is not None:
            _SESSION.shutdown()
            _SESSION = None
