"""Coding-agent run manager — owns one coding-agent UI run (auto_host + driver).

Backs the Coding-Agent Monitor tab. One run at a time (v1, single worker):
``start()`` spawns a dedicated ``env_habitat`` auto_host (via ``BaseServer``,
dynamic free port, PDEATHSIG) and then the unified coding-agent driver's UI
entry (``coding-agent/uirun.py``) as a process-group child; ``stop()``
tears both down (driver first). Run state beyond process liveness is derived
from the driver's own artifacts under ``outputs/beta-coding-agent/{run_name}/`` —
``summary.json`` for finished episodes, ``episode_{i}.jsonl`` presence for the
active one — so the service holds no duplicate bookkeeping that could drift.

The habitat interpreter/source are resolved from the already-discovered
``env_habitat`` nodeset in the workspace registry — this module never imports
workspace or habitat code (framework import boundary).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("agentcanvas.coding-agent")

REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER_PATH = REPO_ROOT / "coding-agent" / "uirun.py"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "beta-coding-agent"

NODESET_NAME = "env_habitat"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class CodingAgentRunner:
    """Singleton service (lifespan-owned) managing at most one coding-agent UI run."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._server: Any = None  # BaseServer for the habitat auto_host
        self._driver: subprocess.Popen | None = None
        self._driver_log: Any = None
        self._state = "idle"  # idle | starting | running | stopping | stopped | finished | error
        self._run_name: str | None = None
        self._error: str | None = None
        self._config: dict[str, Any] = {}

    # ── lifecycle ──

    def start(self, *, episodes: str, split: str, max_turns: int, model: str | None) -> str:
        """Spawn auto_host + driver. Blocking (call via asyncio.to_thread).

        Raises RuntimeError if a run is already active.
        """
        if self._state in ("starting", "running", "stopping"):
            raise RuntimeError(f"a run is already active (state={self._state})")
        if not DRIVER_PATH.exists():
            raise RuntimeError(f"driver script missing: {DRIVER_PATH}")

        self._state = "starting"
        self._error = None
        self._run_name = time.strftime("ui_%Y%m%d_%H%M%S")
        self._config = {
            "episodes": episodes,
            "split": split,
            "max_turns": max_turns,
            "model": model,
        }
        try:
            self._spawn(episodes, split, max_turns, model)
        except Exception as exc:
            self._error = str(exc)
            self._state = "error"
            self._teardown()
            raise
        self._state = "running"
        log.info("coding-agent run %s started", self._run_name)
        return self._run_name

    def _spawn(self, episodes: str, split: str, max_turns: int, model: str | None) -> None:
        from ..server.base_server import BaseServer

        nodeset = getattr(self._registry, "_discovered_nodesets", {}).get(NODESET_NAME)
        if nodeset is None:
            raise RuntimeError(f"nodeset {NODESET_NAME!r} not discovered by the registry")
        nodeset_cls = type(nodeset)
        python = getattr(nodeset_cls, "server_python", None) or sys.executable
        source_file = getattr(nodeset, "_source_file", None)
        if source_file is None:
            raise RuntimeError(f"nodeset {NODESET_NAME!r} has no _source_file")

        backend_dir = str(REPO_ROOT / "agentcanvas" / "backend")
        port = _free_port()
        self._server = BaseServer(
            name="coding_agent_env_habitat",
            command=[
                python,
                "-m",
                "app.server.auto_host",
                "--file",
                str(source_file),
                "--class",
                nodeset_cls.__name__,
                "--port",
                str(port),
            ],
            port=port,
            startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
            working_dir=backend_dir,
            env={"PYTHONPATH": f"{backend_dir}:{REPO_ROOT}"},
        )
        self._server.start()

        run_dir = OUTPUT_ROOT / str(self._run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        driver_cmd = [
            sys.executable,
            str(DRIVER_PATH),
            "--episodes",
            episodes,
            "--split",
            split,
            "--max-turns",
            str(max_turns),
            "--server-url",
            f"http://127.0.0.1:{port}",
            "--run-name",
            str(self._run_name),
        ]
        if model:
            driver_cmd += ["--model", model]
        # Own process group so stop() can SIGTERM the driver together with its
        # claude CLI + bridge children in one killpg.
        self._driver_log = (run_dir / "driver.log").open("w")
        self._driver = subprocess.Popen(
            driver_cmd,
            cwd=str(REPO_ROOT),
            stdout=self._driver_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self) -> None:
        """Tear down driver + auto_host. Blocking (call via asyncio.to_thread)."""
        if self._state not in ("starting", "running"):
            return
        self._state = "stopping"
        self._teardown()
        self._state = "stopped"
        log.info("coding-agent run %s stopped", self._run_name)

    def _teardown(self) -> None:
        if self._driver is not None and self._driver.poll() is None:
            try:
                os.killpg(os.getpgid(self._driver.pid), signal.SIGTERM)
                try:
                    self._driver.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._driver.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._driver_log is not None:
            self._driver_log.close()
            self._driver_log = None
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                log.exception("auto_host stop raised")
            self._server = None

    def shutdown(self) -> None:
        """App-exit hook (lifespan teardown)."""
        self._teardown()

    # ── state ──

    def _reap(self) -> None:
        """Fold a self-exited driver into the state machine."""
        if self._state == "running" and self._driver is not None:
            rc = self._driver.poll()
            if rc is not None:
                if rc == 0:
                    self._state = "finished"
                else:
                    self._state = "error"
                    self._error = f"driver exited with rc={rc} (see driver.log)"
                self._teardown()

    def status(self) -> dict[str, Any]:
        self._reap()
        run_dir = self.run_dir()
        summary: dict[str, Any] = {}
        if run_dir is not None and (run_dir / "summary.json").exists():
            try:
                summary = json.loads((run_dir / "summary.json").read_text())
            except (OSError, ValueError):
                pass
        done = {e.get("index") for e in summary.get("episodes", [])}
        started = (
            sorted(
                int(p.stem.split("_")[1])
                for p in run_dir.glob("episode_*.jsonl")
                if p.stem.split("_")[1].isdigit()
            )
            if run_dir is not None and run_dir.exists()
            else []
        )
        active = next((i for i in started if i not in done), None)
        return {
            "state": self._state,
            "run_name": self._run_name,
            "error": self._error,
            "config": self._config,
            "active_episode": active if self._state == "running" else None,
            "started_episodes": started,
            "aggregate": summary.get("aggregate"),
            "episodes": [
                {
                    "index": e.get("index"),
                    "success": (e.get("metrics") or {}).get("success"),
                    "spl": (e.get("metrics") or {}).get("spl"),
                    "distance_to_goal": (e.get("metrics") or {}).get("distance_to_goal"),
                    "env_steps": (e.get("agent") or {}).get("env_steps"),
                    "called_stop": (e.get("agent") or {}).get("called_stop"),
                    "error": e.get("error"),
                }
                for e in summary.get("episodes", [])
            ],
        }

    def run_dir(self, run_name: str | None = None) -> Path | None:
        """Resolve a sanitized run directory (current run when name omitted)."""
        name = run_name or self._run_name
        if name is None:
            return None
        if not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError(f"bad run name: {name!r}")
        return OUTPUT_ROOT / name
