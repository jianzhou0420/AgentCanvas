"""Pool-based spawn lifecycle for the AgentCanvas backend.

Each MCP server (i.e. each Claude Code conversation) gets its own
backend. The pool has 5 slots backed by ``flock`` lockfiles so multiple
conversations can run experiments in parallel without colliding.

1. On ``start()``: walk ports 8765-8769 and try to ``flock(LOCK_EX|LOCK_NB)``
   the corresponding lockfile under ``~/.cache/agentcanvas-mcp/locks/``.
   First port whose lockfile is unlocked → that's our slot. The fd is
   held on the instance; the kernel releases the lock when the MCP
   process dies any way (clean exit, SIGTERM, SIGKILL, OOM). Then
   spawn ``conda run -n agentcanvas --no-capture-output uvicorn
   app.main:app``, health-poll up to 90s, capture PGID via ``ss -tlnp``
   + ``ps -o pgid=``. Returns ``http://127.0.0.1:{port}``.

2. On ``stop()``: walk uvicorn's descendants, isolate
   ``app.server.auto_host`` children (each lives in its own setsid'd
   group due to ``preexec_fn=os.setsid`` in ``base_server.py``).
   ``killpg(-PGID)`` the backend group; ``killpg`` each auto_host group.
   SIGTERM, sleep 2s, then SIGKILL survivors. Then close the lock fd
   to release the slot.

Cleanup is registered via ``atexit`` and SIGTERM/SIGINT handlers; the
backend itself has an idle watchdog (60s, see ``app/main.py``) and a
terminal-state self-exit (see ``api/execution/eval.py``) so even if
the MCP process dies uncleanly the backend won't outlive its purpose.
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import logging
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# mcp_server/backend_manager.py → mcp_server → agentcanvas → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = _REPO_ROOT / "agentcanvas" / "backend"
PORT_RANGE = (8765, 8766, 8767, 8768, 8769)
HEALTH_TIMEOUT_SEC = 90
HEALTH_INTERVAL_SEC = 2
PROBE_TIMEOUT_SEC = 1.5
LOG_DIR = Path.home() / ".cache" / "agentcanvas-mcp"
POOL_LOCK_DIR = LOG_DIR / "locks"


def _descendants(root_pid: int) -> list[int]:
    """Recursively walk PPID tree from ``root_pid``, return all descendant PIDs."""
    out: list[int] = []
    try:
        children = subprocess.check_output(
            ["pgrep", "-P", str(root_pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).split()
    except subprocess.CalledProcessError:
        return out
    for c in children:
        try:
            cpid = int(c)
        except ValueError:
            continue
        out.append(cpid)
        out.extend(_descendants(cpid))
    return out


def _is_auto_host(pid: int) -> bool:
    """True if /proc/{pid}/cmdline contains the auto_host marker."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    except OSError:
        return False
    return "app.server.auto_host" in cmd


def _preexec_pdeathsig_setsid() -> None:
    """preexec_fn for ``subprocess.Popen``: ``setsid`` + Linux ``PR_SET_PDEATHSIG``.

    Mirrors ``base_server._preexec_setsid_pdeathsig`` so the kernel SIGTERMs
    the spawned backend if THIS process (the MCP server) dies any way —
    including SIGKILL, OOM-kill, or segfault — without atexit firing.

    Runs in the child between fork() and exec(). PDEATHSIG is cleared by
    fork(), so ``conda run`` re-forking the python uvicorn underneath us
    drops the flag mid-chain; ``app/main.py`` re-arms it inside the python
    interpreter to close that hole (belt-and-suspenders, same pattern as
    ``auto_host.py:_arm_pdeathsig``).
    """
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1,  # PR_SET_PDEATHSIG
            signal.SIGTERM,
            0,
            0,
            0,
        )
    except OSError:
        pass


class BackendManager:
    """Spawn-or-borrow + cleanup the AgentCanvas backend uvicorn process."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.port: int | None = None
        self.owned: bool = False
        self.uvicorn_pid: int | None = None
        self.pgid: int | None = None
        self._log_path: Path | None = None
        self._lock_fd: int | None = None
        self._lock_path: Path | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def _claim_slot(self) -> None:
        """Walk PORT_RANGE, flock the first available lockfile.

        Sets ``self.port``, ``self._lock_fd``, ``self._lock_path`` on success.
        Raises RuntimeError if all 5 slots are taken.
        """
        POOL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        for port in PORT_RANGE:
            lock_path = POOL_LOCK_DIR / f"port-{port}.lock"
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            self._lock_fd = fd
            self._lock_path = lock_path
            self.port = port
            logger.info("[backend] claimed pool slot :%s (lockfile=%s)", port, lock_path)
            return
        raise RuntimeError(
            f"backend pool full ({len(PORT_RANGE)}/{len(PORT_RANGE)} slots taken); "
            "wait for another conversation's eval to finish, or stop one explicitly"
        )

    def start(self) -> str:
        """Claim a pool slot, spawn a backend on it, return its base URL."""
        self._claim_slot()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._log_path = LOG_DIR / f"backend-{ts}-port{self.port}.log"

        logger.info("[backend] spawning on :%s, log=%s", self.port, self._log_path)

        # No `setsid -f` shell wrapper here — that would detach the
        # backend from us (parent → PID 1) and BREAK PR_SET_PDEATHSIG.
        # Instead, the preexec_fn does setsid (so we can killpg cleanly
        # later) AND arms PDEATHSIG (so the kernel kills the backend if
        # we die without atexit firing, e.g. SIGKILL).
        log_fh = open(self._log_path, "w")  # noqa: SIM115 — handed to subprocess.Popen, must outlive the `with` block
        subprocess.Popen(
            [
                "conda",
                "run",
                "-n",
                "agentcanvas",
                "--no-capture-output",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=str(BACKEND_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            preexec_fn=_preexec_pdeathsig_setsid,
        )

        # 3. Health-poll loop.
        deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._probe_health(self.port):
                break
            time.sleep(HEALTH_INTERVAL_SEC)
        else:
            self._dump_preflight_failure()
            raise RuntimeError(
                f"backend on :{self.port} did not pass /health within "
                f"{HEALTH_TIMEOUT_SEC}s; see {self._log_path}"
            )

        # 4. Capture PGID (uvicorn's process group, not the conda wrapper's).
        self._capture_pgid()
        self.url = f"http://127.0.0.1:{self.port}"
        self.owned = True
        logger.info(
            "[backend] healthy on :%s, uvicorn_pid=%s, pgid=%s",
            self.port,
            self.uvicorn_pid,
            self.pgid,
        )

        # Register cleanup hooks once we have something to clean up.
        atexit.register(self.stop)
        # SIGTERM + SIGINT — ensure clean shutdown on `kill <mcp>` and Ctrl-C.
        # Suppress OSError/ValueError when registering from non-main thread.
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, self._signal_handler)

        return self.url

    def _probe_health(self, port: int) -> bool:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=PROBE_TIMEOUT_SEC)
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _capture_pgid(self) -> None:
        """Find uvicorn PID via socket → PGID via ps."""
        out = subprocess.check_output(
            f"ss -tlnp 2>/dev/null | grep '127\\.0\\.0\\.1:{self.port} ' "
            f"| grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2",
            shell=True,
            text=True,
        ).strip()
        if not out:
            raise RuntimeError(f"backend reports healthy on :{self.port} but no PID found via ss")
        self.uvicorn_pid = int(out)
        pgid_str = subprocess.check_output(
            ["ps", "-o", "pgid=", "-p", str(self.uvicorn_pid)],
            text=True,
        ).strip()
        self.pgid = int(pgid_str)

    def _dump_preflight_failure(self) -> None:
        if not self._log_path or not self._log_path.exists():
            return
        try:
            tail = subprocess.check_output(["tail", "-n", "80", str(self._log_path)], text=True)
            err_path = self._log_path.with_suffix(".preflight_error.log")
            err_path.write_text(tail)
            logger.error("[backend] preflight failed; tail saved to %s", err_path)
        except subprocess.CalledProcessError:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _signal_handler(self, signum: int, frame: object) -> None:
        logger.info("[backend] received signal %s — stopping", signum)
        self.stop()
        # os._exit (not sys.exit) — sys.exit raises SystemExit which would
        # unwind through any in-flight atexit-invoked stop() and surface as
        # "Exception ignored". atexit has already run our cleanup once via
        # stop(); we just need the process to terminate.
        os._exit(0)

    def stop(self) -> None:
        """Idempotent. Tears down the spawned backend and releases the slot."""
        if self._stopped:
            return
        self._stopped = True

        try:
            self._stop_backend()
        finally:
            # Release the pool slot last — even if backend kill fails we
            # still want to free the lockfile so a retry can spawn.
            self._release_slot()

    def _stop_backend(self) -> None:
        if not self.owned:
            return

        if self.uvicorn_pid is None or self.pgid is None:
            logger.warning("[backend] owned but no PGID recorded — skipping kill")
            return

        # 1. Collect auto_host children before tearing down the parent
        #    group. They live in their own setsid'd groups (see
        #    base_server.py:142), so killpg(backend_pgid) does NOT reach
        #    them.
        auto_hosts = [pid for pid in _descendants(self.uvicorn_pid) if _is_auto_host(pid)]
        logger.info("[backend] stop: pgid=%s, auto_hosts=%s", self.pgid, auto_hosts)

        # 2. SIGTERM the backend group.
        self._safe_killpg(self.pgid, signal.SIGTERM)

        # 3. SIGTERM each auto_host's own group (PGID == PID due to setsid).
        for ah in auto_hosts:
            self._safe_killpg(ah, signal.SIGTERM)

        time.sleep(2)

        # 4. SIGKILL survivors.
        self._safe_killpg(self.pgid, signal.SIGKILL)
        for ah in auto_hosts:
            self._safe_killpg(ah, signal.SIGKILL)

        # 5. Warn-only port sweep (never blind-kill — the user might
        #    re-bind it themselves, and pkill would scope too widely).
        try:
            leftover = subprocess.check_output(
                f"ss -tlnp 2>/dev/null | grep '127\\.0\\.0\\.1:{self.port} ' "
                f"| grep -oE 'pid=[0-9]+' | cut -d= -f2",
                shell=True,
                text=True,
            ).strip()
            if leftover:
                logger.warning(
                    "[backend] port %s still held by PID(s): %s — investigate",
                    self.port,
                    leftover.replace("\n", " "),
                )
        except subprocess.CalledProcessError:
            pass

        logger.info("[backend] stopped (port :%s freed)", self.port)

    @staticmethod
    def _safe_killpg(pgid: int, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)

    def _release_slot(self) -> None:
        """Close the lock fd; the kernel auto-releases the flock."""
        if self._lock_fd is None:
            return
        with contextlib.suppress(OSError):
            os.close(self._lock_fd)
        logger.info("[backend] released pool slot (lockfile=%s)", self._lock_path)
        self._lock_fd = None
