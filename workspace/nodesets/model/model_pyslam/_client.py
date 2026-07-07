from __future__ import annotations

"""pySLAM container bridge — the *framework-side* half.

:class:`PySlamContainerClient` presents the exact surface the nodeset expects
of a :class:`_backend.PySlamSession` (``is_built`` / ``configure_camera`` /
``start`` / ``reset`` / ``track`` / ``get_trajectory`` / ``get_map`` /
``close``), but every call is an HTTP round-trip to a FastAPI shim
(:mod:`_server`) running inside the pyslam Docker container. This is why the
nodeset runs in **local mode** (``server_python = None``): the framework process
never imports pyslam — the GPL dependency lives entirely behind the container
boundary, same treatment as ``habitat_sim``.

Lifecycle:
    * :meth:`start_container` — ``docker run -d`` the pyslam image with this
      nodeset dir bind-mounted (so ``_server``/``_backend`` are available inside)
      and a published loopback port, then poll ``/health`` until the shim answers.
    * per-call methods forward over HTTP.
    * :meth:`close` — POST ``/close`` (quits Slam), then ``docker rm -f``.

Container access goes through **rootless** Docker (see the ``reference_rootless_docker``
memory): the image only exists in the rootless daemon's store, so ``DOCKER_HOST``
and ``XDG_RUNTIME_DIR`` are forced to the user socket regardless of the backend
process's ambient env.

Map handoff: ``/get_map`` ships the sparse points back as ``.npz`` bytes; the
client writes them to a **host-side** file (owned by the framework user, no
rootless-volume uid remap headache) and returns that path as the handle — the
graph wire still only carries the path (design §5a).
"""

import io
import json
import logging
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

log = logging.getLogger("agentcanvas.pyslam")

DEFAULT_IMAGE = os.environ.get("PYSLAM_IMAGE", "agentcanvas/pyslam:cpu-fixed")
# Absolute path to the pyslam venv interpreter inside the image (the .pth in its
# site-packages exposes cpp_core/g2o/pyslam_utils, so no activate script needed).
CONTAINER_PY = "/home/slam/.python/venvs/pyslam/bin/python"


def _rootless_docker_env() -> dict[str, str]:
    """Env that points the docker CLI at the user's rootless daemon.

    Forced (not ``setdefault``): the pyslam image lives only in the rootless
    daemon's overlay store, and the ambient backend process may have no
    ``DOCKER_HOST`` (→ system daemon, needs sudo) or a stale one.
    """
    env = dict(os.environ)
    uid = os.getuid()
    env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    env["DOCKER_HOST"] = f"unix:///run/user/{uid}/docker.sock"
    if "/usr/bin" not in env.get("PATH", "").split(":"):
        env["PATH"] = "/usr/bin:" + env.get("PATH", "")
    return env


class PySlamContainerClient:
    """Drives a pyslam ``Slam`` session living inside a Docker container."""

    def __init__(
        self,
        *,
        sensor_type: str = "rgbd",
        feature_preset: str = "ORB2",
        loop_preset: str = "DBOW3",
        headless: bool = True,
        image: str = DEFAULT_IMAGE,
        gpu: bool | None = None,
    ) -> None:
        self.sensor_type = sensor_type
        self.feature_preset = feature_preset
        self.loop_preset = loop_preset
        self.headless = headless
        self.image = image
        # GPU is exposed to the container via rootless-Docker CDI
        # (``--device nvidia.com/gpu=all``); the ``cpu-fixed`` image already ships
        # a cu128 torch, so the neural backends (learned depth / semantic seg /
        # multiview) run on the GPU while the classic SLAM core stays on CPU.
        # Default on; ``PYSLAM_GPU=0`` forces CPU-only. start_container falls back
        # to CPU if the GPU request fails (no CDI / no GPU on the host).
        if gpu is None:
            gpu = os.environ.get("PYSLAM_GPU", "1").lower() not in ("0", "false", "no")
        self.gpu = gpu

        self._built = False
        self._container: str | None = None
        self._port: int | None = None
        self._base: str | None = None
        self._map_seq = 0

        self._docker_env = _rootless_docker_env()
        self._nodeset_dir = os.path.dirname(os.path.abspath(__file__))
        self._host_artifact_dir = os.environ.get("PYSLAM_ARTIFACT_DIR") or os.path.join(
            os.getcwd(), "outputs", "pyslam_maps"
        )

    # ── docker helpers ──────────────────────────────────────────────────────

    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", *args], env=self._docker_env, check=check, text=True,
            capture_output=True,
        )

    @staticmethod
    def _free_port() -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _run_args(self, gpu: bool) -> list[str]:
        """Assemble the ``docker run`` argv. ``gpu`` inserts the CDI device flag
        (``--device nvidia.com/gpu=all``) — rootless Docker reaches the GPU via
        CDI, not the legacy ``--gpus`` cgroup path (which fails unprivileged)."""
        gpu_args = ["--device", "nvidia.com/gpu=all"] if gpu else []
        return [
            "run", "-d", "--name", self._container,
            "--label", "agentcanvas.pyslam=1",
            *gpu_args,
            "-v", f"{self._nodeset_dir}:/opt/bridge:ro",
            "-w", "/opt/bridge",
            "-e", "PYTHONPATH=/opt/bridge",
            "-e", f"PYSLAM_SENSOR={self.sensor_type}",
            "-e", f"PYSLAM_FEATURE={self.feature_preset}",
            "-e", f"PYSLAM_LOOP={self.loop_preset}",
            "-p", f"127.0.0.1:{self._port}:8000",
            self.image,
            CONTAINER_PY, "-m", "uvicorn", "_server:app",
            "--host", "0.0.0.0", "--port", "8000",
        ]

    def start_container(self) -> None:
        """``docker run`` the shim and block until ``/health`` answers."""
        self._port = self._free_port()
        self._container = f"pyslam_bridge_{os.getpid()}_{self._port}"
        # Pre-clean any stale container with this exact name (idempotent restart).
        self._docker("rm", "-f", self._container, check=False)

        proc = None
        if self.gpu:
            try:
                proc = self._docker(*self._run_args(gpu=True))
            except subprocess.CalledProcessError as exc:
                # No CDI / no GPU on this host → degrade to CPU rather than fail.
                log.warning(
                    "pyslam GPU container start failed (%s); retrying CPU-only",
                    (exc.stderr or "").strip()[:200],
                )
                self._docker("rm", "-f", self._container, check=False)
                self.gpu = False
        if proc is None:
            proc = self._docker(*self._run_args(gpu=False))

        cid = proc.stdout.strip()
        self._base = f"http://127.0.0.1:{self._port}"
        log.info("pyslam container started: %s (%s, gpu=%s) → %s",
                 self._container, cid[:12], self.gpu, self._base)
        self._wait_healthy()

    def _container_status(self) -> str:
        p = self._docker("inspect", "-f", "{{.State.Status}}", self._container, check=False)
        return p.stdout.strip() if p.returncode == 0 else "gone"

    def _wait_healthy(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self._base + "/health", timeout=3) as r:
                    if r.status == 200:
                        log.info("pyslam shim healthy at %s", self._base)
                        return
            except Exception as exc:
                last_err = exc
                status = self._container_status()
                if status in ("exited", "dead", "gone"):
                    raise RuntimeError(
                        f"pyslam container {self._container} {status} during startup:\n"
                        f"{self._logs_tail()}"
                    ) from exc
            time.sleep(0.5)
        raise RuntimeError(
            f"pyslam shim not healthy in {timeout:.0f}s ({last_err}):\n{self._logs_tail()}"
        )

    def _logs_tail(self, n: int = 60) -> str:
        p = self._docker("logs", "--tail", str(n), self._container, check=False)
        return (p.stdout or "") + (p.stderr or "")

    def close(self) -> None:
        if self._base is not None:
            try:
                self._http("/close", timeout=30)
            except Exception:
                log.warning("pyslam /close failed (container may be down)", exc_info=True)
        if self._container is not None:
            self._docker("rm", "-f", self._container, check=False)
            log.info("pyslam container removed: %s", self._container)
        self._built = False

    # ── HTTP plumbing ───────────────────────────────────────────────────────

    def _http(
        self, path: str, *, data: bytes = b"", content_type: str = "application/octet-stream",
        timeout: float = 120.0,
    ) -> tuple[bytes, dict[str, str]]:
        if self._base is None:
            raise RuntimeError("PySlamContainerClient: container not started")
        req = urllib.request.Request(self._base + path, data=data, method="POST")
        req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"pyslam shim {path} → HTTP {exc.code}: {detail}") from exc

    def _post_json(self, path: str, obj: Any = None, *, timeout: float = 120.0) -> dict:
        data = json.dumps(obj).encode() if obj is not None else b""
        ct = "application/json" if obj is not None else "application/octet-stream"
        body, _ = self._http(path, data=data, content_type=ct, timeout=timeout)
        return json.loads(body) if body else {}

    # ── session surface (mirrors PySlamSession) ─────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._built

    def configure_camera(self, **kwargs: Any) -> None:
        cfg = {k: v for k, v in kwargs.items() if v is not None}
        self._post_json("/configure_camera", cfg)

    def start(self, timeout: float = 240.0) -> None:
        r = self._post_json("/start", timeout=timeout)
        self._built = bool(r.get("built", True))

    def reset(self) -> None:
        self._post_json("/reset")

    def track(self, rgb: np.ndarray, depth: np.ndarray | None = None,
              timestamp: float | None = None) -> dict:
        buf = io.BytesIO()
        arrays: dict[str, np.ndarray] = {"rgb": np.ascontiguousarray(rgb)}
        if depth is not None:
            arrays["depth"] = np.ascontiguousarray(depth)
        if timestamp is not None:
            arrays["timestamp"] = np.asarray(float(timestamp))
        np.savez(buf, **arrays)  # uncompressed — localhost, per-frame hot path
        body, _ = self._http("/track", data=buf.getvalue())
        return json.loads(body)

    def get_trajectory(self) -> dict:
        return self._post_json("/get_trajectory")

    def get_map(self, out_dir: str | None = None) -> dict:
        body, headers = self._http("/get_map")
        npz = np.load(io.BytesIO(body))
        points = npz["points"]
        n_pts = int(headers.get("X-Num-Points", len(points)))
        n_kf = int(headers.get("X-Num-Keyframes", 0))

        out_dir = out_dir or self._host_artifact_dir
        os.makedirs(out_dir, exist_ok=True)
        self._map_seq += 1
        handle = os.path.join(out_dir, f"map_{self._map_seq:04d}_{n_pts:06d}.npz")
        np.savez_compressed(handle, points=points)
        log.info("pyslam map handle written (host): %d points, %d kf → %s", n_pts, n_kf, handle)
        return {"map_handle": handle, "num_points": n_pts, "num_keyframes": n_kf}

    # ── stateless perception surface (Tier-2, mirrors _backend module funcs) ──

    def extract_features(
        self, rgb: np.ndarray, *, detector: str = "ORB", descriptor: str = "ORB",
        num_features: int = 2000,
    ) -> dict:
        buf = io.BytesIO()
        np.savez(buf, rgb=np.ascontiguousarray(rgb))
        q = urllib.parse.urlencode(
            {"detector": detector, "descriptor": descriptor, "num_features": int(num_features)}
        )
        body, headers = self._http(f"/extract_features?{q}", data=buf.getvalue())
        npz = np.load(io.BytesIO(body))
        return {
            "keypoints": npz["keypoints"],
            "descriptors": npz["descriptors"],
            "num_keypoints": int(headers.get("X-Num-Keypoints", len(npz["keypoints"]))),
            "detector": headers.get("X-Detector", detector.upper()),
            "descriptor": headers.get("X-Descriptor", (descriptor or detector).upper()),
        }

    def match_features(
        self, des_a: np.ndarray, des_b: np.ndarray, *, matcher_type: str = "BF",
        ratio_test: float = 0.7, cross_check: bool = False, norm_type: int | None = None,
    ) -> dict:
        buf = io.BytesIO()
        np.savez(buf, des_a=np.ascontiguousarray(des_a), des_b=np.ascontiguousarray(des_b))
        params = {
            "matcher_type": matcher_type,
            "ratio_test": float(ratio_test),
            "cross_check": int(bool(cross_check)),
        }
        if norm_type is not None:
            params["norm_type"] = int(norm_type)
        q = urllib.parse.urlencode(params)
        body, headers = self._http(f"/match_features?{q}", data=buf.getvalue())
        npz = np.load(io.BytesIO(body))
        return {
            "idxs_a": npz["idxs_a"],
            "idxs_b": npz["idxs_b"],
            "num_matches": int(headers.get("X-Num-Matches", len(npz["idxs_a"]))),
        }

    def eval_trajectory(
        self, poses_est: Any, poses_gt: Any, *, is_monocular: bool = False,
    ) -> dict:
        def _to_list(poses: Any) -> list:
            return [np.asarray(p, dtype=float).tolist() for p in poses]

        return self._post_json(
            "/eval_trajectory",
            {"poses_est": _to_list(poses_est), "poses_gt": _to_list(poses_gt),
             "is_monocular": bool(is_monocular)},
        )
