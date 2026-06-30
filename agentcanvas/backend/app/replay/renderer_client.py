"""Replay renderer client — spawn + call a renderer subprocess.

Used by env replay parsers' smooth-mode hooks. Lazy-spawns a
:mod:`app.replay.renderer_host`-launched subprocess on first frame
request, holds it warm for the FastAPI process lifetime, kills it via
:meth:`stop` (called from the parser's ``shutdown()`` hook on app exit).

Renderer-side FastAPI is opaque to this client — every renderer just
needs ``GET /health`` returning 200 and one or more POST endpoints
returning either JSON or binary content. ``post_for_bytes`` returns
the raw response body, suitable for serving directly as image bytes
from a FastAPI ``Response``.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
from pathlib import Path
from typing import Any

import httpx

from ..server.base_server import BaseServer

log = logging.getLogger("agentcanvas.replay.renderer")


class ReplayRendererClient:
    """Lazy-spawn a renderer subprocess and call its endpoints over HTTP."""

    def __init__(
        self,
        *,
        renderer_file: Path,
        class_name: str,
        python: str | None = None,
        env: dict | None = None,
        startup_timeout: int = 1800,
        port_range: tuple = (9300, 9320),
        backend_dir: Path | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self._renderer_file = Path(renderer_file).resolve()
        self._class_name = class_name
        self._python = python or sys.executable
        self._env = dict(env or {})
        self._startup_timeout = startup_timeout
        self._port_range = port_range
        # Default backend dir: this file lives at app/replay/, parents[2] = backend/
        self._backend_dir = (
            Path(backend_dir).resolve()
            if backend_dir is not None
            else Path(__file__).resolve().parents[2]
        )
        self._request_timeout = request_timeout
        self._server: BaseServer | None = None
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def post_for_bytes(self, path: str, payload: dict) -> bytes:
        """POST ``path`` with JSON ``payload``; return raw response body."""
        await self._ensure_started()
        assert self._client is not None
        resp = await self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.content

    async def post_for_json(self, path: str, payload: dict) -> Any:
        await self._ensure_started()
        assert self._client is not None
        resp = await self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _ensure_started(self) -> None:
        if self._server is not None and self._server.connected:
            return
        async with self._lock:
            if self._server is not None and self._server.connected:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._start_blocking)
            self._client = httpx.AsyncClient(
                base_url=self._server.url,
                timeout=httpx.Timeout(self._request_timeout, connect=10.0),
            )

    def _start_blocking(self) -> None:
        port = self._find_free_port()
        workspace_root = self._backend_dir.parent.parent
        pythonpath = f"{self._backend_dir}:{workspace_root}"
        cmd = (
            f"PYTHONPATH={pythonpath} {self._python} -m app.replay.renderer_host "
            f"--file {self._renderer_file} --class {self._class_name} --port {port}"
        )
        server = BaseServer(
            name=f"replay-renderer:{self._class_name}",
            command=cmd,
            port=port,
            host="127.0.0.1",
            startup_timeout=self._startup_timeout,
            working_dir=str(self._backend_dir),
            env=dict(self._env),
        )
        server.start()
        self._server = server
        log.info(
            "Replay renderer subprocess up: %s on %s (pid=%s)",
            self._class_name,
            server.url,
            server.pid,
        )

    def _find_free_port(self) -> int:
        lo, hi = self._port_range
        for p in range(lo, hi):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                log.exception("Failed to close renderer http client")
            self._client = None
        if self._server is not None:
            srv = self._server
            self._server = None
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, srv.stop)
            except Exception:
                log.exception("Failed to stop renderer subprocess")
