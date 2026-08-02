"""Isaac 5.1 sim-backend boundary for env_vlnverse — vendored from NavHarness.

Sources (NavHarness @ dev/claude-sdk-agent, vendored 2026-07-20):
  navharness/projects/internutopia_vln_extension/envs/sim_backends/sim_backend.py
  navharness/projects/internutopia_vln_extension/envs/sim_backends/subprocess_backend.py

The nodeset process talks to the simulator only through the
:class:`SimBackend` protocol below; all method arguments and return values
are numpy arrays or plain Python types, never Isaac objects. The actual
Isaac 5.1 runtime lives in ``_isaac_worker.py``, spawned under Isaac's
bundled python (``~/isaacsim5.1/python.sh``) and driven over
length-prefixed msgpack frames. This process must NEVER import
``omni.*`` / ``isaacsim``.

Deviations from upstream:
  - Dropped ``LocalIsaacSimBackend`` — an in-process Isaac instance inside
    the nodeset (or backend) process is forbidden; only the subprocess /
    socket clients remain.
  - Dropped ``IsaacPanoramicRenderer`` — legacy shim over a pre-built
    World/Camera pair; nothing in the nodeset constructs those.
  - Dropped ``_coerce_rgb`` / ``_coerce_depth`` — they were only consumed
    by the two dropped classes; the worker keeps its own duplicates.
  - ``SocketIsaacSimBackend``'s "start the worker first" hint now names
    this package's ``_isaac_worker.py`` instead of the NavHarness path.
  - Merged both source files into one module (protocol + frame IO +
    clients) so the package carries a single seam file.
"""

from __future__ import annotations

import io
import os
import socket as _socket
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# Data types crossing the backend boundary
# ---------------------------------------------------------------------------


@dataclass
class RenderedView:
    rgb: np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, finite (see _DEPTH_FAR_CLIP_M)


# Deterministic far clip for Isaac's non-finite depth: RTX renders +inf for
# no-hit rays (sky / open windows) and habitat consumers assume finite depth.
# A fixed constant (not the per-frame finite max) keeps the same no-hit pixel
# at the same metric value across panorama views and episodes.
_DEPTH_FAR_CLIP_M = 20.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SimBackend(Protocol):
    """Minimal simulator surface used by the VLNVerse env manager."""

    def start(self, sim_cfg: Dict[str, Any]) -> None:
        """Boot the simulator (SimulationApp + World + Camera).

        ``sim_cfg`` is a serialisable dict (headless, gpu id, renderer name,
        camera resolution / focal length / aperture, agent collider sizes...).
        """

    def load_scene(self, scene_id: str, scene_usd_path: str) -> None:
        """Load or switch to a USD scene. Must clear any previously loaded one."""

    def capture_panoramic(
        self,
        position: np.ndarray,
        heading: float,
        num_views: int = 12,
        warmup_steps: int = 10,
    ) -> List[RenderedView]:
        """Render evenly spaced panoramic views from a pose."""

    def shutdown(self) -> None:
        """Stop the simulator and free resources."""


# ---------------------------------------------------------------------------
# Stdio IPC frame helpers (shared by the clients below; the worker keeps
# its own duplicated copy because it cannot import this package)
# ---------------------------------------------------------------------------


_FRAME_HEADER = struct.Struct('>I')  # 4-byte big-endian length prefix

# One-time preamble the worker writes on its stdio frame channel before the
# first response frame. Everything before it is launcher noise — Isaac's
# python.sh prints e.g. "Warning: running in conda env..." to stdout before
# python starts, and the parent would otherwise read "Warn" as a ~1.46 GB
# length prefix and block forever (observed 2026-07-20). Must match
# _isaac_worker.STDIO_MAGIC byte-for-byte. Socket mode has no shared-fd
# problem and skips this.
STDIO_MAGIC = b'\x00\xffVLNVERSE-FRAMES\xff\x00'
# Give up scanning for the magic after this many bytes — a worker that never
# sends it (e.g. crashed in python.sh) should fail loudly, not hang.
_MAGIC_SCAN_LIMIT = 1 << 20


def _sync_to_magic(stream: io.BufferedReader) -> bytes:
    """Discard launcher noise until STDIO_MAGIC; return the discarded bytes.

    Byte-at-a-time is fine: the noise is at most a few text lines, and this
    runs exactly once per worker boot.
    """
    seen = bytearray()
    while True:
        b = stream.read(1)
        if not b:
            tail = bytes(seen[-500:]).decode('utf-8', 'replace')
            raise RuntimeError(
                'sim worker closed the pipe before the frame-channel magic; '
                f'launcher output was: {tail!r}'
            )
        seen.extend(b)
        if seen.endswith(STDIO_MAGIC):
            return bytes(seen[: -len(STDIO_MAGIC)])
        if len(seen) > _MAGIC_SCAN_LIMIT:
            raise RuntimeError(
                'sim worker never sent the frame-channel magic within '
                f'{_MAGIC_SCAN_LIMIT} bytes — wrong/old worker script?'
            )


def write_frame(stream: io.BufferedWriter, payload_bytes: bytes) -> None:
    stream.write(_FRAME_HEADER.pack(len(payload_bytes)))
    stream.write(payload_bytes)


def read_frame(stream: io.BufferedReader) -> bytes:
    header = _read_exact(stream, _FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    return _read_exact(stream, length)


def _read_exact(stream: io.BufferedReader, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError('sim worker closed the pipe before sending the full frame')
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Main-process clients for the out-of-process Isaac worker
# ---------------------------------------------------------------------------


class _FramedRPCMixin:
    """Shared msgpack/length-prefix RPC plumbing for socket and subprocess clients.

    Subclasses provide ``_rfile`` / ``_wfile`` (binary file objects) and
    override ``_on_dead_pipe(op, exc)`` to attach extra diagnostics (return
    code, socket path) when the worker disappears.
    """

    def __init__(self) -> None:
        self._next_id = 0
        self._lock = threading.Lock()

    def _call(self, op: str, **args: Any) -> Any:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            payload = msgpack.packb(
                {'id': req_id, 'op': op, 'args': args},
                use_bin_type=True,
            )
            try:
                write_frame(self._wfile, payload)
                self._wfile.flush()
                raw = read_frame(self._rfile)
            except (BrokenPipeError, EOFError, ConnectionError) as e:
                raise self._on_dead_pipe(op, e)
        resp = msgpack.unpackb(raw, raw=False)
        if resp.get('id') != req_id:
            raise RuntimeError(
                f'sim worker frame out of order: expected id {req_id}, got {resp.get("id")}'
            )
        if not resp.get('ok', False):
            raise RuntimeError(
                f'sim worker raised on op {op!r}: {resp.get("error", "<no traceback>")}'
            )
        return resp.get('data')

    def _on_dead_pipe(self, op: str, exc: BaseException) -> RuntimeError:
        return RuntimeError(f'sim worker pipe died (op={op}): {exc}')

    # ----- SimBackend surface --------------------------------------------

    def start(self, sim_cfg: Dict[str, Any]) -> None:
        self._call('start', sim_cfg=sim_cfg)

    def load_scene(self, scene_id: str, scene_usd_path: str) -> None:
        self._call('load_scene', scene_id=scene_id, scene_usd_path=scene_usd_path)

    def capture_panoramic(
        self,
        position: np.ndarray,
        heading: float,
        num_views: int = 12,
        warmup_steps: int = 10,
    ) -> List[RenderedView]:
        raw = self._call(
            'capture_panoramic',
            position=np.asarray(position, dtype=np.float64),
            heading=float(heading),
            num_views=int(num_views),
            warmup_steps=int(warmup_steps),
        )
        # Depth finiteness is enforced HERE, at the capture boundary, so every
        # consumer (DEPTH ports, raw_obs, both base64 encoders) sees the same
        # finite depth — habitat's sensor never produces non-finite values, and
        # a per-consumer clamp with a data-dependent cap would encode the same
        # no-hit sky pixel differently in every panorama view.
        #
        # Writability is enforced here too: msgpack_numpy decodes via
        # np.frombuffer → WRITEABLE=False arrays. Habitat delivers fresh
        # writable arrays, and downstream torch.from_numpy / in-place ops
        # break on read-only buffers (upstream MLLMEnvironment .copy()s for
        # exactly this reason). ~7 MB/view — negligible next to the render.
        views: List[RenderedView] = []
        for item in raw:
            # NOT ascontiguousarray: frombuffer arrays are already contiguous,
            # so it would return the same read-only object. copy=True always.
            rgb = np.array(item['rgb'], copy=True)
            depth = np.asarray(item['depth'], dtype=np.float32)
            if not np.isfinite(depth).all():
                # nan_to_num already returns a fresh writable array.
                depth = np.nan_to_num(
                    depth, nan=0.0, posinf=_DEPTH_FAR_CLIP_M, neginf=0.0
                )
            else:
                depth = np.array(depth, copy=True)
            views.append(RenderedView(rgb=rgb, depth=depth))
        return views


class SubprocessIsaacSimBackend(_FramedRPCMixin):
    """Spawns the worker as a child and owns its lifecycle."""

    def __init__(
        self,
        isaac_python: str,
        worker_script: str,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        if not os.path.isfile(isaac_python):
            raise FileNotFoundError(f'Isaac python launcher not found: {isaac_python}')
        if not os.path.isfile(worker_script):
            raise FileNotFoundError(f'sim worker script not found: {worker_script}')

        env = os.environ.copy()
        # Strip PYTHONPATH leaks from the parent env (the nodeset process
        # carries backend+workspace paths) so the Isaac 5.1 launcher sets up
        # its own paths cleanly.
        env.pop('PYTHONPATH', None)
        # Strip conda context: with it set, Isaac's python.sh prints a
        # "running in conda env" warning TO STDOUT before python starts,
        # poisoning the frame channel (see STDIO_MAGIC). The magic scan below
        # would survive it anyway — this just kills the noise at the source.
        # By prefix, not an enumerated list: conda sets more vars than the
        # obvious four and python.sh's trigger may key on any of them.
        for k in [k for k in env if k.startswith('CONDA_') or k == '_CE_CONDA']:
            env.pop(k, None)
        if env_overrides:
            env.update(env_overrides)

        self.proc = subprocess.Popen(
            [isaac_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
            env=env,
        )
        self._rfile = self.proc.stdout
        self._wfile = self.proc.stdin
        # Resync: skip any launcher noise ahead of the worker's magic preamble.
        noise = _sync_to_magic(self._rfile)
        if noise:
            sys.stderr.write(
                f'[sim_backend] discarded {len(noise)} bytes of launcher '
                f'noise before frame magic: {noise[-200:]!r}\n'
            )

    def _on_dead_pipe(self, op: str, exc: BaseException) -> RuntimeError:
        rc = self.proc.poll()
        return RuntimeError(f'sim worker died (op={op}, return_code={rc}): {exc}')

    def shutdown(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._call('shutdown')
        except Exception:
            pass
        finally:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class SocketIsaacSimBackend(_FramedRPCMixin):
    """Attaches to a worker the user pre-started with ``--listen <socket>``.

    The worker owns its own lifecycle — this client just disconnects on
    shutdown rather than killing the worker. That way the user can keep
    one warmed-up Isaac worker (slow boot done once) and run the nodeset
    process multiple times against it (e.g. while debugging, or to dodge
    the minute-scale Isaac cold boot between runs).
    """

    def __init__(self, socket_path: str, connect_timeout: float = 30.0) -> None:
        super().__init__()
        if not os.path.exists(socket_path):
            raise FileNotFoundError(
                f'sim worker socket not found: {socket_path}. '
                f'Start the worker first with:\n'
                f'    ~/isaacsim5.1/python.sh '
                f'workspace/nodesets/env/env_vlnverse/'
                f'_isaac_worker.py --listen {socket_path}'
            )
        self._sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout)
        self._sock.connect(socket_path)
        self._sock.settimeout(None)
        # makefile gives us file-like objects that read_frame / write_frame
        # already know how to drive.
        self._rfile = self._sock.makefile('rb')
        self._wfile = self._sock.makefile('wb')
        self._socket_path = socket_path

    def _on_dead_pipe(self, op: str, exc: BaseException) -> RuntimeError:
        return RuntimeError(
            f'sim worker socket dropped (op={op}, path={self._socket_path}): {exc}'
        )

    def shutdown(self) -> None:
        # Don't send 'shutdown' — the worker stays alive for reconnect.
        # Just close our end.
        for closer in (self._wfile, self._rfile, self._sock):
            try:
                closer.close()
            except Exception:
                pass
