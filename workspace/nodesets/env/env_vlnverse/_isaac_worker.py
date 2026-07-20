"""Isaac 5.1 sim worker for env_vlnverse — vendored from NavHarness.

Source (NavHarness @ dev/claude-sdk-agent, vendored 2026-07-20):
  navharness/projects/internutopia_vln_extension/envs/sim_backends/sim_workers/isaac_worker.py

Runs inside ``~/isaacsim5.1/python.sh`` (Isaac's bundled python), reads
length-prefixed msgpack command frames on stdin, writes response frames on
stdout. Human-readable logs go to stderr.

This is the only file in the env_vlnverse package that imports
``isaacsim`` / ``omni.*``, and it is deliberately standalone: it must NOT
import anything from the package (frame IO is duplicated here on purpose),
because Isaac's python cannot resolve the nodeset package.

Deviations from upstream:
  - Added a ``--check`` CLI flag: verifies ``isaacsim`` is importable and
    prints its version, WITHOUT booting SimulationApp — used by
    ``install_ac_vlnverse.sh`` to front-load "is Isaac even installed"
    failures to install time.
  - ``selftest`` scene lookup also tries the AgentCanvas data layout
    (``data/vlnverse/scene/…``) before NavHarness's ``data/scene/…``.
  - Everything else is faithful, including the ``--listen`` socket mode.
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# Frame I/O (duplicated from _sim_backend.py because this script runs in the
# isaac python and cannot import the env_vlnverse package).
# ---------------------------------------------------------------------------

_FRAME_HEADER = struct.Struct('>I')

# One-time preamble on the stdio frame channel — the parent discards every
# byte before it (python.sh / shell noise emitted before this process could
# claim fd1). Must match _sim_backend.STDIO_MAGIC byte-for-byte.
STDIO_MAGIC = b'\x00\xffVLNVERSE-FRAMES\xff\x00'


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError('parent closed the pipe')
        buf.extend(chunk)
    return bytes(buf)


def write_frame(stream, payload: bytes) -> None:
    stream.write(_FRAME_HEADER.pack(len(payload)))
    stream.write(payload)


def read_frame(stream) -> bytes:
    header = _read_exact(stream, _FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    return _read_exact(stream, length)


def _coerce_rgb(raw_rgba, height: int, width: int) -> np.ndarray:
    if raw_rgba is None or raw_rgba.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if raw_rgba.ndim == 1 and raw_rgba.size == height * width * 4:
        return raw_rgba.reshape(height, width, 4)[:, :, :3]
    if raw_rgba.ndim == 3:
        return raw_rgba[:, :, :3]
    return np.zeros((height, width, 3), dtype=np.uint8)


def _coerce_depth(raw_depth, height: int, width: int) -> np.ndarray:
    if raw_depth is None or raw_depth.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    if raw_depth.ndim == 1 and raw_depth.size == height * width:
        return raw_depth.reshape(height, width)
    if raw_depth.ndim == 2:
        return raw_depth
    return np.zeros((height, width), dtype=np.float32)


def log(msg: str) -> None:
    print(f'[isaac_worker] {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Isaac side
# ---------------------------------------------------------------------------


class IsaacWorld:
    """Owns the SimulationApp, World, Camera, current scene."""

    def __init__(self) -> None:
        self.simulation_app = None
        self.world = None
        self.camera = None
        self._sim_cfg: Optional[Dict[str, Any]] = None
        self._camera_ready = False
        self.current_scene_id: Optional[str] = None

    # ----- import helpers ------------------------------------------------

    @staticmethod
    def _imports():
        """Late imports so the module is importable without booting Isaac."""
        # In Isaac 5.x, World and Camera moved to isaacsim.core.api / sensors.
        try:
            from isaacsim.core.api import World  # type: ignore
        except ImportError:
            from omni.isaac.core import World  # type: ignore
        try:
            from isaacsim.sensors.camera import Camera  # type: ignore
        except ImportError:
            from omni.isaac.sensor import Camera  # type: ignore
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
        except ImportError:
            from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore
        try:
            from isaacsim.core.utils.rotations import euler_angles_to_quat  # type: ignore
        except ImportError:
            from omni.isaac.core.utils.rotations import euler_angles_to_quat  # type: ignore
        return World, Camera, add_reference_to_stage, euler_angles_to_quat

    # ----- ops -----------------------------------------------------------

    def start(self, sim_cfg: Dict[str, Any]) -> None:
        self._sim_cfg = sim_cfg
        gpu_id = int(sim_cfg.get('gpu_id', 0))
        headless = bool(sim_cfg.get('headless', True))
        isaac_multi_gpu = bool(sim_cfg.get('isaac_multi_gpu', False))
        renderer = str(sim_cfg.get('renderer', 'RaytracedLighting'))

        os.environ['OMNI_RTX_SKIP_DRIVER_CHECK'] = '1'
        os.environ['OMNI_GPU_SKIP_VERIFICATION'] = '1'
        os.environ['CARB_SKIP_DRIVER_VERIFICATION'] = '1'

        log(f'starting SimulationApp (headless={headless}, gpu={gpu_id}, renderer={renderer})')
        from isaacsim import SimulationApp  # type: ignore

        self.simulation_app = SimulationApp({
            'headless': headless,
            'active_gpu': gpu_id,
            'physics_gpu': gpu_id,
            'multi_gpu': isaac_multi_gpu,
            'max_gpu_count': None if isaac_multi_gpu else 1,
            'renderer': renderer,
            'anti_aliasing': int(sim_cfg.get('anti_aliasing', 0)),
            'denoiser': bool(sim_cfg.get('denoiser', False)),
        })
        log('SimulationApp running')

        World, Camera, _, _ = self._imports()

        self.world = World(stage_units_in_meters=1.0)
        log('World created')

        resolution = tuple(sim_cfg.get('camera_resolution', (1024, 1024)))
        self.camera = Camera(prim_path='/World/Camera', resolution=resolution)
        log(f'Camera prim created at /World/Camera, resolution={resolution}')

    def _finish_camera_setup(self) -> None:
        if self._camera_ready or self.camera is None:
            return
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        cfg = self._sim_cfg or {}
        focal_length = float(cfg.get('focal_length', 10.0))
        aperture = float(cfg.get('aperture', 2 * focal_length))
        self.camera.set_focal_length(focal_length)
        self.camera.set_horizontal_aperture(aperture)
        self.camera.set_vertical_aperture(aperture)
        self._camera_ready = True
        log('Camera initialised')

    def load_scene(self, scene_id: str, scene_usd_path: str) -> None:
        _, _, add_reference_to_stage, _ = self._imports()
        scene_prim_path = '/World/scene'

        log(f'load_scene({scene_id}): {scene_usd_path}')
        if self.current_scene_id is not None and self.world is not None:
            stage = self.world.scene.stage
            if stage.GetPrimAtPath(scene_prim_path):
                stage.RemovePrim(scene_prim_path)
                log(f'removed previous scene prim {scene_prim_path}')

        add_reference_to_stage(usd_path=scene_usd_path, prim_path=scene_prim_path)
        self.current_scene_id = scene_id

        if not self._camera_ready:
            self._finish_camera_setup()

        assert self.world is not None
        self.world.reset()
        for _ in range(10):
            self.world.step(render=True)
        log(f'scene {scene_id} loaded & stabilised')

    def get_camera_resolution(self) -> Tuple[int, int]:
        assert self.camera is not None
        width, height = self.camera.get_resolution()
        return int(height), int(width)

    def capture_panoramic(
        self,
        position: np.ndarray,
        heading: float,
        num_views: int = 12,
        warmup_steps: int = 10,
    ) -> List[Dict[str, np.ndarray]]:
        _, _, _, euler_angles_to_quat = self._imports()
        assert self.world is not None and self.camera is not None

        height, width = self.get_camera_resolution()
        heading_deg = float(np.degrees(float(heading)))
        position = np.asarray(position, dtype=np.float64)

        views: List[Dict[str, np.ndarray]] = []
        for view_idx in range(num_views):
            angle_deg = heading_deg + view_idx * (360.0 / num_views)
            orientation_quat = euler_angles_to_quat(
                np.array([0.0, 0.0, angle_deg]), degrees=True
            )
            self.camera.set_world_pose(position=position, orientation=orientation_quat)

            raw_rgba = None
            raw_depth = None
            for _ in range(warmup_steps):
                self.world.step(render=True)
                raw_depth = self.camera.get_depth()
                raw_rgba = self.camera.get_rgba()

            views.append({
                'rgb': _coerce_rgb(raw_rgba, height, width),
                'depth': _coerce_depth(raw_depth, height, width),
            })
        return views

    def shutdown(self) -> None:
        log('shutdown requested')
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            finally:
                self.simulation_app = None


# ---------------------------------------------------------------------------
# RPC loop
# ---------------------------------------------------------------------------


def serve(stdin: io.BufferedReader, stdout: io.BufferedWriter) -> None:
    world = IsaacWorld()
    while True:
        try:
            raw = read_frame(stdin)
        except EOFError:
            log('stdin closed, exiting')
            return

        msg = msgpack.unpackb(raw, raw=False)
        req_id = msg.get('id')
        op = msg.get('op')
        args = msg.get('args', {}) or {}

        try:
            handler = getattr(world, op, None)
            if handler is None:
                raise AttributeError(f'unknown op: {op}')
            result = handler(**args)
            resp = {'id': req_id, 'ok': True, 'data': result}
        except SystemExit:
            raise
        except BaseException:
            tb = traceback.format_exc()
            log(f'op {op!r} failed:\n{tb}')
            resp = {'id': req_id, 'ok': False, 'error': tb}

        write_frame(stdout, msgpack.packb(resp, use_bin_type=True))
        stdout.flush()

        if op == 'shutdown' and resp['ok']:
            log('shutdown acknowledged, exiting RPC loop')
            return


# ---------------------------------------------------------------------------
# Install-time check (--check): is isaacsim importable at all?
# ---------------------------------------------------------------------------


def check() -> int:
    """Verify ``isaacsim`` is importable WITHOUT booting SimulationApp.

    Exit 0 with a version line on success; exit non-zero with a one-line
    readable error otherwise. Used by ``scripts/install/install_ac_vlnverse.sh``
    so a missing/broken Isaac install fails at install time, not first run.
    """
    try:
        import isaacsim  # type: ignore  # noqa: F401
    except Exception as e:  # noqa: BLE001 — any import failure is a real failure here
        log(
            f'--check FAILED: cannot import isaacsim '
            f'({e.__class__.__name__}: {e}). Is Isaac Sim 5.1 installed, and is '
            f'this running under its bundled python (e.g. ~/isaacsim5.1/python.sh)?'
        )
        return 1
    version = getattr(isaacsim, '__version__', None)
    if not version:
        try:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version('isaacsim')
        except Exception:
            version = 'unknown'
    log(f'--check OK: isaacsim importable, version={version}')
    return 0


# ---------------------------------------------------------------------------
# Self-test entry point
# ---------------------------------------------------------------------------


def selftest(scene_id: Optional[str], usd_path: Optional[str], project_root: Optional[str]) -> int:
    if usd_path is None:
        if project_root is None:
            project_root = os.environ.get('VLNVERSE_ROOT')
        if scene_id is None:
            scene_id = 'kujiale_0003'
        if project_root is None:
            log('selftest needs --project-root or VLNVERSE_ROOT')
            return 2
        # AgentCanvas layout (data/vlnverse/scene/…) first, then the
        # NavHarness layout (data/scene/…) so the flag works from either repo.
        candidates = []
        for scene_root in ('data/vlnverse/scene', 'data/scene'):
            candidates += [
                os.path.join(project_root, scene_root, scene_id, 'start_result_navigation.usd'),
                os.path.join(project_root, scene_root, scene_id, 'start_result_navigation.usda'),
                os.path.join(project_root, scene_root, scene_id, f'{scene_id}.usd'),
                os.path.join(project_root, scene_root, scene_id, f'{scene_id}.usda'),
            ]
        usd_path = next((p for p in candidates if os.path.exists(p)), None)
        if usd_path is None:
            log(f'selftest: no USD for scene {scene_id} in {candidates}')
            return 3

    world = IsaacWorld()
    world.start({'headless': True, 'gpu_id': 0, 'renderer': 'RaytracedLighting',
                 'camera_resolution': (512, 512)})
    world.load_scene(scene_id or 'selftest', usd_path)
    views = world.capture_panoramic(np.array([0.0, 0.0, 1.5]), 0.0,
                                    num_views=4, warmup_steps=5)
    for i, v in enumerate(views):
        log(f'view {i}: rgb={v["rgb"].shape}{v["rgb"].dtype}, '
            f'depth={v["depth"].shape}{v["depth"].dtype}, '
            f'rgb_mean={float(v["rgb"].mean()):.2f}')
    try:
        from PIL import Image
        out_dir = os.path.join(project_root or '.', 'debug_outputs', 'sim_worker_selftest')
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(views[0]['rgb']).save(os.path.join(out_dir, 'view0.png'))
        log(f'wrote {out_dir}/view0.png')
    except Exception as e:
        log(f'(could not save debug image: {e})')
    world.shutdown()
    log('selftest done')
    return 0


def serve_unix_socket(socket_path: str) -> None:
    """Listen on a Unix domain socket and serve one parent connection.

    The parent process (the env_vlnverse nodeset, possibly under a
    debugger) connects to this socket after we boot Isaac, so it can attach
    interactively instead of spawning us itself.
    """
    import socket as _socket

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(1)
    log(f'listening on unix socket: {socket_path}')
    try:
        conn, _ = srv.accept()
        log('parent connected, entering RPC loop')
        rfile = conn.makefile('rb')
        wfile = conn.makefile('wb')
        try:
            serve(rfile, wfile)
        finally:
            try:
                rfile.close()
                wfile.close()
                conn.close()
            except Exception:
                pass
    finally:
        try:
            srv.close()
        finally:
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--check', action='store_true',
        help='Verify isaacsim is importable (no SimulationApp boot), then exit.',
    )
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--scene-id', default=None)
    parser.add_argument('--usd-path', default=None)
    parser.add_argument('--project-root', default=None)
    parser.add_argument(
        '--listen', default=None,
        help='Unix domain socket path to listen on. When set, the worker '
             'serves RPC over this socket instead of stdio so the parent '
             'process can be started independently (eg under a debugger).',
    )
    args = parser.parse_args()

    if args.check:
        return check()

    if args.selftest:
        return selftest(args.scene_id, args.usd_path, args.project_root)

    if args.listen:
        serve_unix_socket(args.listen)
        return 0

    # Default: stdio frames, parent is whoever launched us via subprocess.Popen.
    #
    # fd1 hygiene (deviation from upstream — observed 2026-07-20): the frame
    # channel shares fd1 with everything else in this process, and two writers
    # poison it in practice: (a) Isaac's python.sh prints a "running in conda
    # env" warning to stdout BEFORE python even starts, and (b) omni/kit logs
    # stray lines to stdout after SimulationApp boots. The parent then reads
    # log text as a length prefix ("Warn" == ~1.46 GB) and blocks forever.
    # Fix, both ends: dup the real stdout for frames, repoint fd1 at stderr so
    # every later stdout write becomes a log line, and emit STDIO_MAGIC on the
    # frame channel; the parent discards bytes until the magic (handles (a),
    # which happened before we could dup2). Socket mode needs none of this —
    # its channel never shares an fd with loggers.
    real_out = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(1, 'w', buffering=1)  # keep print() working → stderr
    frame_out = os.fdopen(real_out, 'wb')
    frame_out.write(STDIO_MAGIC)
    frame_out.flush()
    serve(sys.stdin.buffer, frame_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
