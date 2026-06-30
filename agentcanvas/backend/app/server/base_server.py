"""BaseServer — process launcher, monitor, and HTTP client for server-mode nodesets.

This is the **framework-side** class. AgentCanvas uses it to start, stop,
monitor, and communicate with external server processes.

Each subclass defines server-specific properties (command, port, env vars).
The framework instantiates these from YAML configs in ``workspace/servers/``.

Usage::

    class HabitatServer(BaseServer):
        name = "habitat"
        command = "conda run -n ac-vlnce python -m app.server.examples.habitat_server"
        port = 9100

    server = HabitatServer()
    server.start()       # spawns subprocess, waits for /health
    server.connected     # True
    server.pid           # 12345
    server.stop()        # kills process
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Any

from .manifest import ServerManifest

log = logging.getLogger("agentcanvas.server")


def _preexec_setsid_pdeathsig() -> None:
    """preexec_fn: ``setsid`` + Linux ``PR_SET_PDEATHSIG``.

    Two effects in the spawned child, applied between fork and exec:

    1. :func:`os.setsid` — give the child its own session/PGID so the
       parent can ``kill -- -PGID`` the whole tree atomically.
    2. ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` (Linux only) — kernel sends
       SIGTERM to this process when its parent dies any way (SIGKILL,
       OOM-kill, segfault, panic). Survives ``exec()`` (cleared only
       for setuid binaries), so the eventual python child inherits it.
       Belt-and-suspenders pair with auto_host's own prctl call —
       together they ensure no orphaned auto_host can survive uvicorn
       death, regardless of whether the shell wrapper tail-execs.
    """
    os.setsid()
    try:
        ctypes_mod = __import__("ctypes")
        libc = ctypes_mod.CDLL("libc.so.6", use_errno=True)
        ret = libc.prctl(
            1,  # PR_SET_PDEATHSIG
            signal.SIGTERM,
            0,
            0,
            0,
        )
        if ret != 0:
            # Silent pdeathsig failure → orphan-on-parent-exit. We're
            # post-fork pre-exec, so logging is unsafe; write directly.
            errno = ctypes_mod.get_errno()
            os.write(2, f"prctl(PR_SET_PDEATHSIG) failed: errno={errno}\n".encode())
    except OSError:
        pass


class BaseServer:
    """Manages an external server process and communicates with it.

    Subclass this to define server-specific properties:
    - ``name`` — server identifier
    - ``command`` — argv list (preferred — spawns with no shell, so the
      service is a direct child and ``PR_SET_PDEATHSIG`` works one-hop) or
      a shell-string command (legacy — spawns via ``/bin/sh -c``)
    - ``port`` — port the service listens on
    - ``host`` — hostname (default ``localhost``)
    - ``startup_timeout`` — hard ceiling on /health probe during startup
      (default 1800s = 30 min). Subprocess death short-circuits this, so the
      ceiling only fires on pathological hangs (alive process, never healthy).
      Default is sized for first-run weight downloads (HF cache miss can pull
      multiple GB). Override *down* on fast-load servers if you want stuck
      detection to fire sooner.
    - ``auto_restart`` — restart on crash (default False)
    - ``env`` — extra environment variables for the subprocess
    - ``working_dir`` — working directory for the subprocess
    """

    name: str = "unnamed"
    description: str = ""
    command: str | list[str] = ""
    port: int = 9000
    host: str = "localhost"
    startup_timeout: int = 1800
    auto_restart: bool = False
    env: dict[str, str] = {}
    working_dir: str = ""

    def __init__(self, **overrides: Any) -> None:
        # Apply overrides from YAML config
        self._url_override: str | None = None
        for key, val in overrides.items():
            if key == "url":
                self._url_override = val
            elif hasattr(self, key):
                setattr(self, key, val)
        # Runtime state
        self._process: subprocess.Popen | None = None
        self._pid: int | None = None
        self._connected: bool = False
        self._manifest: ServerManifest | None = None
        self._error: str | None = None
        self._status: str = "stopped"

    # ── Properties ──

    @property
    def pid(self) -> int | None:
        """PID of the server process, or None if not running."""
        return self._pid

    @property
    def connected(self) -> bool:
        """True if the server is alive and /health returned 200."""
        return self._connected

    @property
    def status(self) -> str:
        """Current status: stopped, starting, connected, unreachable, error."""
        return self._status

    @property
    def url(self) -> str:
        """Base URL of the server."""
        if self._url_override:
            return self._url_override.rstrip("/")
        return f"http://{self.host}:{self.port}"

    # ── Lifecycle ──

    def start(self) -> None:
        """Launch the server process and wait until it's healthy.

        Raises RuntimeError if the command is empty or the server fails
        to respond within ``startup_timeout`` seconds.
        """
        if self._connected:
            log.info("Server %s already connected — skipping start", self.name)
            return

        if not self.command:
            raise RuntimeError(
                f"No command configured for server '{self.name}'. "
                "Set the 'command' class attribute or YAML field."
            )

        self._status = "starting"
        log.info("Starting server %s: %s", self.name, self.command)

        # Build subprocess environment
        proc_env = os.environ.copy()
        proc_env.update(self.env)

        cwd = self.working_dir or None

        try:
            # Inherit stdout/stderr so server logs appear in the AgentCanvas terminal.
            # A list command spawns with no shell — the service is then a DIRECT
            # child, so its PR_SET_PDEATHSIG watches us one-hop (no /bin/sh wrapper
            # to break the chain). A str command keeps the legacy shell path.
            self._process = subprocess.Popen(
                self.command,
                shell=isinstance(self.command, str),
                cwd=cwd,
                env=proc_env,
                stdout=None,  # inherit parent's stdout
                stderr=None,  # inherit parent's stderr
                preexec_fn=_preexec_setsid_pdeathsig,
            )
            self._pid = self._process.pid
            log.info("Server %s started with PID %d", self.name, self._pid)
        except Exception as e:
            self._status = "error"
            self._error = str(e)
            log.exception("Failed to start server %s", self.name)
            raise RuntimeError(f"Failed to start server '{self.name}': {e}") from e

        # Wait for /health
        if not self._wait_for_health():
            self._status = "error"
            self._error = f"Server did not become healthy within {self.startup_timeout}s"
            log.error("%s: %s", self.name, self._error)
            self.stop()
            raise RuntimeError(self._error)

        self._connected = True
        self._status = "connected"
        self._error = None
        log.info("Server %s is connected at %s (PID %d)", self.name, self.url, self._pid)

    def stop(self) -> None:
        """Stop the server process."""
        if self._process is None:
            self._connected = False
            self._status = "stopped"
            return

        log.info("Stopping server %s (PID %d)", self.name, self._pid)

        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("Server %s did not exit gracefully, sending SIGKILL", self.name)
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait(timeout=3)
        except (ProcessLookupError, OSError):
            pass  # process already gone

        self._process = None
        self._pid = None
        self._connected = False
        self._status = "stopped"
        self._manifest = None
        log.info("Server %s stopped", self.name)

    def restart(self) -> None:
        """Stop and start the server."""
        self.stop()
        self.start()

    # ── Health Check ──

    def health_check(self) -> bool:
        """Send GET /health and update ``connected`` status.

        Returns True if the server responded with 200.
        """
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        try:
            resp = httpx.get(f"{self.url}/health", timeout=5.0, **loopback_httpx_kwargs())
            alive = resp.status_code == 200
        except Exception:
            alive = False

        was_connected = self._connected
        self._connected = alive
        self._status = "connected" if alive else "unreachable"

        if was_connected and not alive:
            log.warning("Server %s became unreachable", self.name)
        elif not was_connected and alive:
            log.info("Server %s is now reachable", self.name)

        # Check if subprocess died
        if self._process is not None and self._process.poll() is not None:
            exit_code = self._process.returncode
            log.warning("Server %s process exited with code %d", self.name, exit_code)
            self._process = None
            self._pid = None
            self._connected = False
            self._status = "stopped"

        return alive

    def fetch_manifest(self) -> ServerManifest | None:
        """Fetch and cache the server manifest.

        Returns None if the server is unreachable.
        """
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        try:
            resp = httpx.get(f"{self.url}/manifest", timeout=10.0, **loopback_httpx_kwargs())
            resp.raise_for_status()
            self._manifest = ServerManifest.from_dict(resp.json())
            return self._manifest
        except Exception:
            log.warning("Failed to fetch manifest from %s", self.url)
            return None

    # ── Status ──

    def get_status(self) -> dict:
        """Return current status as a dict (for API responses)."""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "status": self._status,
            "pid": self._pid,
            "connected": self._connected,
            "error": self._error,
            "auto_restart": self.auto_restart,
        }

    # ── Internal ──

    def _wait_for_health(self) -> bool:
        """Poll GET /health until 200, the subprocess dies, or the hard ceiling.

        Liveness-driven: as long as the subprocess is alive, we keep polling
        — covers slow first-time loads (HF weight downloads, GPU init) without
        timing out on healthy-but-slow startup. Subprocess death is the
        fast-fail path; ``startup_timeout`` is only the ceiling for pathological
        hangs (alive process, never healthy). A progress line is emitted every
        ~30s so long waits don't look like a stall.
        """
        import httpx

        from ._loopback_proxy import loopback_httpx_kwargs

        start = time.time()
        deadline = start + self.startup_timeout
        health_url = f"{self.url}/health"
        interval = 0.5
        last_progress_log = start

        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                log.error(
                    "Server %s process exited during startup (code %d) after %.1fs",
                    self.name,
                    self._process.returncode,
                    time.time() - start,
                )
                return False
            try:
                resp = httpx.get(health_url, timeout=2.0, **loopback_httpx_kwargs())
                if resp.status_code == 200:
                    log.info(
                        "Server %s healthy after %.1fs",
                        self.name,
                        time.time() - start,
                    )
                    return True
            except Exception:
                pass

            now = time.time()
            if now - last_progress_log >= 30.0:
                log.info(
                    "Server %s still starting (%.0fs elapsed, process alive)",
                    self.name,
                    now - start,
                )
                last_progress_log = now

            time.sleep(interval)
            interval = min(interval * 1.5, 3.0)

        return False
