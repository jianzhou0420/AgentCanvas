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

# Two images. The CUDA image is a superset — SLAM + every neural backend, including
# the compiled multi-view kernels (curope); cpu-fixed is the leaner SLAM / feature /
# depth / semantic image (its torch is already cu128, so those run on the GPU too).
# When PYSLAM_IMAGE is unset the client picks by GPU: GPU on → :cuda (the whole
# surface works out of the box, incl. reconstruct_multiview), GPU off → :cpu-fixed
# (the multi-view backends need the GPU image anyway). PYSLAM_IMAGE overrides.
CUDA_IMAGE = "agentcanvas/pyslam:cuda"
CPU_IMAGE = "agentcanvas/pyslam:cpu-fixed"


def _default_image(gpu: bool) -> str:
    """Resolve the container image when the caller didn't pin one: env override
    first, else GPU → the full CUDA image, no-GPU → the leaner CPU image."""
    return os.environ.get("PYSLAM_IMAGE") or (CUDA_IMAGE if gpu else CPU_IMAGE)
# Absolute path to the pyslam venv interpreter inside the image (the .pth in its
# site-packages exposes cpp_core/g2o/pyslam_utils, so no activate script needed).
CONTAINER_PY = "/home/slam/.python/venvs/pyslam/bin/python"


def _ci_header(headers: dict, name: str, default: str | None = None) -> str | None:
    """Case-insensitive header lookup — Starlette lowercases response header names
    on the wire, so ``headers.get("X-Num-Classes")`` misses. Values with no array
    fallback (counts, flags) must go through this."""
    low = name.lower()
    for k, v in headers.items():
        if k.lower() == low:
            return v
    return default


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
        image: str | None = None,
        gpu: bool | None = None,
        volumetric: bool = False,
        volumetric_type: str = "VOXEL_GRID",
        environment: str = "INDOOR",
    ) -> None:
        self.sensor_type = sensor_type
        self.feature_preset = feature_preset
        self.loop_preset = loop_preset
        self.headless = headless
        # Dense volumetric mapping (get_dense_map) — passed to the in-container
        # session via env; off by default (extra integrator thread + memory).
        self.volumetric = volumetric
        self.volumetric_type = volumetric_type
        self.environment = environment
        # GPU is exposed to the container via rootless-Docker CDI
        # (``--device nvidia.com/gpu=all``); both images ship a cu128 torch, so the
        # neural backends (learned depth / semantic seg / multiview) run on the GPU
        # while the classic SLAM core stays on CPU. Default on; ``PYSLAM_GPU=0``
        # forces CPU-only. start_container falls back to CPU if the GPU request
        # fails (no CDI / no GPU on the host).
        if gpu is None:
            gpu = os.environ.get("PYSLAM_GPU", "1").lower() not in ("0", "false", "no")
        self.gpu = gpu
        # Image is resolved AFTER gpu so an unpinned image follows GPU availability
        # (GPU → the full :cuda image; CPU → the leaner :cpu-fixed). An explicit
        # ``image=`` arg or ``PYSLAM_IMAGE`` still wins — and if pinned, the runtime
        # CPU-fallback in start_container won't second-guess it.
        self._image_pinned = bool(image) or bool(os.environ.get("PYSLAM_IMAGE"))
        self.image = image or _default_image(gpu)

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

    def _weights_dir(self) -> str:
        """Host path to the external multiview-weights folder (mounted into the
        container). Defaults to ``<repo>/data/models/pyslam`` — a symlink into the
        shared data tree; ``PYSLAM_WEIGHTS_DIR`` overrides. The weights live
        outside the image by design (external-weights decision 2026-07-08): the
        image carries only code + compiled curope, the .pth files mount in here."""
        env = os.environ.get("PYSLAM_WEIGHTS_DIR")
        if env:
            return env
        root = self._nodeset_dir  # <repo>/workspace/nodesets/model/model_pyslam
        for _ in range(4):  # model_pyslam → model → nodesets → workspace → <repo>
            root = os.path.dirname(root)
        return os.path.join(root, "data", "models", "pyslam")

    def _weight_mounts(self) -> list[str]:
        """``-v`` flags binding the external weights into pyslam's expected paths.

        Only existing dirs are mounted, so a missing weights folder just means the
        matching backend errors at call time (it can't find its checkpoint) rather
        than the container failing to start. Explicit-checkpoint dirs (mast3r /
        mvdust3r / dust3r) mount read-only; the HF / torch caches mount read-write
        so runtime downloads (VGGT / DAv3 / Fast3r / DUSt3R) persist across runs."""
        wd = self._weights_dir()
        thirdparty = "/home/slam/pyslam/thirdparty"
        pairs = [
            (os.path.join(wd, "mast3r", "checkpoints"), f"{thirdparty}/mast3r/checkpoints", "ro"),
            (os.path.join(wd, "mvdust3r", "checkpoints"), f"{thirdparty}/mvdust3r/checkpoints", "ro"),
            (os.path.join(wd, "dust3r", "checkpoints"), f"{thirdparty}/dust3r/checkpoints", "ro"),
            (os.path.join(wd, "hf_cache"), "/home/slam/.cache/huggingface", "rw"),
            (os.path.join(wd, "torch_cache"), "/home/slam/.cache/torch", "rw"),
        ]
        args: list[str] = []
        for src, dst, mode in pairs:
            if os.path.isdir(src):
                args += ["-v", f"{src}:{dst}:{mode}"]
        return args

    def _run_args(self, gpu: bool) -> list[str]:
        """Assemble the ``docker run`` argv. ``gpu`` inserts the CDI device flag
        (``--device nvidia.com/gpu=all``) — rootless Docker reaches the GPU via
        CDI, not the legacy ``--gpus`` cgroup path (which fails unprivileged)."""
        gpu_args = ["--device", "nvidia.com/gpu=all"] if gpu else []
        return [
            "run", "-d", "--name", self._container,
            "--label", "agentcanvas.pyslam=1",
            *gpu_args,
            *self._weight_mounts(),
            "-v", f"{self._nodeset_dir}:/opt/bridge:ro",
            "-w", "/opt/bridge",
            "-e", "PYTHONPATH=/opt/bridge",
            "-e", f"PYSLAM_SENSOR={self.sensor_type}",
            "-e", f"PYSLAM_FEATURE={self.feature_preset}",
            "-e", f"PYSLAM_LOOP={self.loop_preset}",
            "-e", f"PYSLAM_VOLUMETRIC={'1' if self.volumetric else '0'}",
            "-e", f"PYSLAM_VOLUMETRIC_TYPE={self.volumetric_type}",
            "-e", f"PYSLAM_ENV={self.environment}",
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
                # An unpinned image was chosen for GPU (:cuda) — now that we know
                # there's no GPU, drop to the leaner CPU image, honouring the
                # "no GPU → cpu image" default. A pinned image is left untouched.
                if not self._image_pinned and self.image == CUDA_IMAGE:
                    self.image = CPU_IMAGE
                    log.info("pyslam falling back to CPU image %s", self.image)
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

    def get_dense_map(self, out_dir: str | None = None) -> dict:
        body, headers = self._http("/get_dense_map", timeout=300)
        npz = np.load(io.BytesIO(body))
        n_pts = int(_ci_header(headers, "X-Num-Points", "0") or 0)
        n_verts = int(_ci_header(headers, "X-Num-Vertices", "0") or 0)
        n_tris = int(_ci_header(headers, "X-Num-Triangles", "0") or 0)
        dense_type = _ci_header(headers, "X-Dense-Type", self.volumetric_type)
        if n_pts == 0 and n_verts == 0:
            return {"dense_handle": "", "num_points": 0, "num_vertices": 0,
                    "num_triangles": 0, "type": dense_type}
        out_dir = out_dir or self._host_artifact_dir
        os.makedirs(out_dir, exist_ok=True)
        self._map_seq += 1
        handle = os.path.join(out_dir, f"dense_{self._map_seq:04d}.npz")
        np.savez_compressed(handle, points=npz["points"], colors=npz["colors"],
                            vertices=npz["vertices"], triangles=npz["triangles"])
        log.info("pyslam dense handle written (host): %d pts, %d verts, %d tris → %s",
                 n_pts, n_verts, n_tris, handle)
        return {"dense_handle": handle, "num_points": n_pts, "num_vertices": n_verts,
                "num_triangles": n_tris, "type": dense_type}

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

    def predict_depth(
        self, rgb: np.ndarray, image_right: np.ndarray | None = None, *,
        estimator: str = "DEPTH_ANYTHING_V2", min_depth: float = 0.0,
        max_depth: float = 50.0, environment: str = "INDOOR",
    ) -> dict:
        buf = io.BytesIO()
        arrays: dict[str, np.ndarray] = {"rgb": np.ascontiguousarray(rgb)}
        if image_right is not None:
            arrays["image_right"] = np.ascontiguousarray(image_right)
        np.savez(buf, **arrays)
        q = urllib.parse.urlencode({
            "estimator": estimator, "min_depth": float(min_depth),
            "max_depth": float(max_depth), "environment": environment,
        })
        # First call loads/downloads the checkpoint onto the GPU — allow a long timeout.
        body, headers = self._http(f"/predict_depth?{q}", data=buf.getvalue(), timeout=600)
        npz = np.load(io.BytesIO(body))
        depth = npz["depth"]
        # Derive the range from the array itself — the depth map is the source of
        # truth; the X-Depth-* headers are advisory and Starlette lowercases header
        # names on the wire, so a case-sensitive lookup would silently read 0.
        return {
            "depth": depth,
            "estimator": headers.get("X-Estimator", estimator.upper()),
            "shape": list(depth.shape),
            "min": float(depth.min()) if depth.size else 0.0,
            "max": float(depth.max()) if depth.size else 0.0,
        }

    def segment_semantic(
        self, rgb: np.ndarray, *, model: str = "DEEPLABV3", feature_type: str = "LABEL",
        dataset: str = "CITYSCAPES", image_size: tuple = (512, 512),
    ) -> dict:
        buf = io.BytesIO()
        np.savez(buf, rgb=np.ascontiguousarray(rgb))
        q = urllib.parse.urlencode({
            "model": model, "feature_type": feature_type, "dataset": dataset,
            "image_size": f"{int(image_size[0])},{int(image_size[1])}",
        })
        # First call may download a checkpoint onto the GPU — allow a long timeout.
        body, headers = self._http(f"/segment_semantic?{q}", data=buf.getvalue(), timeout=600)
        npz = np.load(io.BytesIO(body))
        semantics = npz["semantics"] if "semantics" in npz.files else None
        instances = npz["instances"] if "instances" in npz.files else None
        return {
            "semantics": semantics,
            "instances": instances,
            "num_classes": int(_ci_header(headers, "X-Num-Classes", "0") or 0),
            "shape": None if semantics is None else list(semantics.shape),
            "model": _ci_header(headers, "X-Model", model.upper()),
            "feature_type": _ci_header(headers, "X-Feature-Type", feature_type.upper()),
        }

    def reconstruct_multiview(
        self, images: list, *, out_dir: str | None = None,
        backend: str = "MAST3R", as_pointcloud: bool = True,
    ) -> dict:
        buf = io.BytesIO()
        arrays = {f"img_{i}": np.ascontiguousarray(im) for i, im in enumerate(images)}
        np.savez(buf, **arrays)  # uncompressed — localhost
        q = urllib.parse.urlencode({
            "backend": backend, "as_pointcloud": int(bool(as_pointcloud)),
        })
        # First call loads a large transformer onto the GPU (+ possible HF/weights
        # download) and DUSt3R-family global alignment is iterative — long timeout.
        body, headers = self._http(f"/reconstruct_multiview?{q}", data=buf.getvalue(), timeout=1200)
        npz = np.load(io.BytesIO(body))
        n_pts = int(_ci_header(headers, "X-Num-Points", "0") or 0)
        n_verts = int(_ci_header(headers, "X-Num-Vertices", "0") or 0)
        n_faces = int(_ci_header(headers, "X-Num-Faces", "0") or 0)
        n_views = int(_ci_header(headers, "X-Num-Views", str(len(images))) or len(images))
        backend_name = _ci_header(headers, "X-Backend", backend.upper())
        poses = npz["camera_poses"]  # (N,4,4)

        out_dir = out_dir or self._host_artifact_dir
        os.makedirs(out_dir, exist_ok=True)
        self._map_seq += 1
        handle = os.path.join(out_dir, f"scene_{self._map_seq:04d}_{n_pts:06d}.npz")
        # Heavy geometry (points/colours/mesh/intrinsics) rides the handle; poses
        # are small enough to also travel inline on the wire for downstream use.
        np.savez_compressed(
            handle, points=npz["points"], colors=npz["colors"],
            vertices=npz["vertices"], faces=npz["faces"],
            camera_poses=poses, intrinsics=npz["intrinsics"],
        )
        log.info("pyslam scene handle written (host): %s %d pts, %d verts, %d views → %s",
                 backend_name, n_pts, n_verts, n_views, handle)
        return {
            "scene_handle": handle,
            "camera_poses": [poses[i].tolist() for i in range(poses.shape[0])],
            "num_points": n_pts, "num_vertices": n_verts, "num_faces": n_faces,
            "num_views": n_views, "backend": backend_name,
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
