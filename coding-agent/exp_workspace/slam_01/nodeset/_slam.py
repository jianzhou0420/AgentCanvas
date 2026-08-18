"""Host-side client for the containerized ORB-SLAM3 server.

Ported 2026-08-17 from the slam-frontier probe worktree (slam_frontier/
slam.py; the mount now targets THIS package directory and the in-container
server module its ``_``-prefixed name).

Spawns ``docker run -i agentcanvas/orbslam3 python3 /work/_slam_server.py``
with this directory mounted read-only at /work, and speaks the JSON-lines
protocol over the container's stdin/stdout. Same image as the
``model_orbslam3`` nodeset (Container Launch, ADR-server-005); ORB-SLAM3 is
CPU-only, no GPU flags needed.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TRACKING_OK = 2
TRACKING_LOST = 4


class OrbSlam3Client:
    def __init__(self, image: str = "agentcanvas/orbslam3:latest") -> None:
        self._proc = subprocess.Popen(
            [
                "docker", "run", "-i", "--rm",
                "-v", f"{_THIS_DIR}:/work:ro",
                image, "python3", "/work/_slam_server.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _rpc(self, msg: dict) -> dict:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("slam_server closed its stdout (container died?)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"slam_server error: {resp.get('error')}")
        return resp

    def init(self, intrinsics: dict, fps: float = 10.0) -> None:
        intr = dict(intrinsics)
        intr["fps"] = fps
        self._rpc({"cmd": "init", "intr": intr})

    def track(self, rgb: np.ndarray, depth_m: np.ndarray, t: float) -> dict:
        """Returns {"pose": 4x4 ndarray | None, "state": int, "ok_frames": int}."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth.shape
        resp = self._rpc({
            "cmd": "track",
            "rgb": base64.b64encode(rgb.tobytes()).decode(),
            "depth": base64.b64encode(depth.tobytes()).decode(),
            "h": h, "w": w, "t": float(t),
        })
        pose = resp["pose"]
        return {
            "pose": None if pose is None else np.asarray(pose, dtype=np.float64),
            "state": int(resp["state"]),
            "ok_frames": int(resp["ok_frames"]),
        }

    def trajectory(self) -> list:
        return self._rpc({"cmd": "trajectory"})["trajectory"]

    def close(self) -> None:
        with contextlib.suppress(Exception):  # teardown must not raise
            self._rpc({"cmd": "shutdown"})
        with contextlib.suppress(Exception):
            self._proc.stdin.close()  # type: ignore[union-attr]
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
