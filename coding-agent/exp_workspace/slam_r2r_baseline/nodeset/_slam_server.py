"""ORB-SLAM3 JSON-lines server — runs INSIDE the agentcanvas/orbslam3 container.

Launched by _slam.OrbSlam3Client as:
    docker run -i --rm -v <this dir>:/work:ro agentcanvas/orbslam3 \
        python3 /work/_slam_server.py

Protocol (one JSON object per line on stdin; one JSON reply per line on a
dup of the original stdout — the C++ library spams the process stdout, so we
re-point fd 1 at stderr and keep a private fd for replies):

  {"cmd": "init", "intr": {fx, fy, cx, cy, width, height, fps?}}
      -> {"ok": true}
  {"cmd": "track", "rgb": <b64 u8 HxWx3>, "depth": <b64 f32 HxW>,
   "h": int, "w": int, "t": float}
      -> {"ok": true, "pose": 4x4 list | null, "state": int, "ok_frames": int}
  {"cmd": "trajectory"}
      -> {"ok": true, "trajectory": [{"t": float, "matrix": 4x4}]}
  {"cmd": "shutdown"} -> {"ok": true} then exit
"""
from __future__ import annotations

import base64
import json
import os
import sys

# Keep a private fd for protocol replies; route everything else (including
# ORB-SLAM3's C++ prints) to stderr.
_RESP = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402

VOCAB_PATH = "/opt/orbslam3/ORBvoc.txt"
SETTINGS_PATH = "/tmp/orbslam3_settings.yaml"


def write_settings(intr):
    fps = float(intr.get("fps", 10.0))
    text = "\n".join([
        "%YAML:1.0",
        'File.version: "1.0"',
        'Camera.type: "PinHole"',
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
        f"Camera.fps: {int(fps)}",
        "Camera.RGB: 1",
        "Stereo.b: 0.05",
        "Stereo.ThDepth: 40.0",
        "RGBD.DepthMapFactor: 1.0",
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
    ])
    with open(SETTINGS_PATH, "w") as f:
        f.write(text)


def entry_to_matrix(entry):
    # 13-tuple: (timestamp, row-major 3x4)
    return [list(entry[1:5]), list(entry[5:9]), list(entry[9:13]), [0.0, 0.0, 0.0, 1.0]]


def reply(obj):
    _RESP.write(json.dumps(obj) + "\n")
    _RESP.flush()


def main():
    system = None
    orbslam3 = None
    ok_frames = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg["cmd"]
            if cmd == "init":
                import orbslam3 as _o
                orbslam3 = _o
                if system is not None:
                    system.shutdown()
                write_settings(msg["intr"])
                system = orbslam3.System(VOCAB_PATH, SETTINGS_PATH, orbslam3.Sensor.RGBD)
                system.set_use_viewer(False)
                system.initialize()
                ok_frames = 0
                reply({"ok": True})
            elif cmd == "track":
                h, w = int(msg["h"]), int(msg["w"])
                rgb = np.frombuffer(base64.b64decode(msg["rgb"]), dtype=np.uint8)
                rgb = rgb.reshape(h, w, 3).copy()
                depth = np.frombuffer(base64.b64decode(msg["depth"]), dtype=np.float32)
                depth = depth.reshape(h, w).copy()
                system.process_image_rgbd(rgb, depth, float(msg["t"]))
                state = int(system.get_tracking_state())
                pose = None
                if state == 2:  # OK
                    ok_frames += 1
                    traj = system.get_trajectory_points()
                    if traj:
                        pose = entry_to_matrix(traj[-1])
                reply({"ok": True, "pose": pose, "state": state, "ok_frames": ok_frames})
            elif cmd == "trajectory":
                traj = system.get_trajectory_points() if system is not None else []
                reply({"ok": True, "trajectory": [
                    {"t": float(e[0]), "matrix": entry_to_matrix(e)} for e in traj
                ]})
            elif cmd == "shutdown":
                if system is not None:
                    system.shutdown()
                reply({"ok": True})
                return
            else:
                reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
