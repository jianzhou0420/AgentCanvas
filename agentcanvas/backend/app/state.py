"""Per-process service container for AgentCanvas.

``ProcessServices`` (reached via ``get_services()``) holds the two
constructed, lifecycle-managed services of *this* process — the
``WorkspaceComponentRegistry`` and the ``JobScheduler``. It is per-process,
not global: each OS process (backend + every eval subprocess) builds its own.
Trivial cross-cutting state (the WS connection set, ``ExecutionGuard``) lives
as separate module globals below, not on the container.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any

from .components.registry import WorkspaceComponentRegistry
from .config import get_settings
from .models import WSMessage

log = logging.getLogger("agentcanvas.state")


# ── Exclusive Execution Guard ──


class ExecutionMode(str, Enum):
    idle = "idle"
    canvas = "canvas"
    eval = "eval"


class ExecutionGuard:
    """Prevents concurrent canvas + eval execution.

    Only one execution mode (canvas or eval) can be active at a time.
    Acquire before starting execution, release on completion/error/cancel.
    """

    _lock = threading.Lock()
    _mode: ExecutionMode = ExecutionMode.idle
    _holder: str | None = None  # run_id or "canvas"

    @classmethod
    def acquire(cls, mode: ExecutionMode, holder: str) -> bool:
        """Try to acquire the execution lock. Returns True on success."""
        with cls._lock:
            if cls._mode == ExecutionMode.idle:
                cls._mode = mode
                cls._holder = holder
                return True
            return False

    @classmethod
    def release(cls, holder: str) -> None:
        """Release the execution lock if held by the given holder."""
        with cls._lock:
            if cls._holder == holder:
                cls._mode = ExecutionMode.idle
                cls._holder = None

    @classmethod
    def force_release(cls) -> None:
        """Force release regardless of holder. Use for error recovery."""
        with cls._lock:
            cls._mode = ExecutionMode.idle
            cls._holder = None

    @classmethod
    def current(cls) -> dict:
        """Return current execution state."""
        with cls._lock:
            return {"mode": cls._mode.value, "holder": cls._holder}


# Grace window before an orphaned canvas guard is reaped. The frontend's
# WSManager auto-reconnects within seconds (page refresh, network blip), so a
# few seconds of zero clients is normal; only a sustained absence means the
# canvas was actually closed.
CANVAS_ORPHAN_GRACE_SEC = 30.0


def canvas_guard_orphan_decision(
    mode: str,
    ws_count: int,
    orphan_since: float | None,
    now: float,
    grace_sec: float = CANVAS_ORPHAN_GRACE_SEC,
) -> tuple[bool, float | None]:
    """Decide whether to reap an orphaned canvas ``ExecutionGuard``.

    The canvas guard is released only on ``/run/stop`` or when ``runner.run()``
    returns (the ``_run_and_release`` ``finally``). If the frontend closes while
    a run is *paused*, neither fires: ``runner.run()`` stays suspended, the
    guard leaks, and — since ``JobScheduler`` refuses admission while the canvas
    lock is held — every eval job starves forever.

    This is the WS-liveness backstop: the frontend keeps one pinging WebSocket
    open for the whole app session, so ``ws_count == 0`` means nobody is
    watching. When the canvas guard is held with no client for ``grace_sec``,
    the guard is reaped (mirrors eval's PID/``_DONE`` orphan reaping).

    Pure decision function (no side effects) so the loop stays testable.

    Returns ``(should_reap, new_orphan_since)``:
      * not orphaned (not canvas mode, or a client is connected) → ``(False, None)``
      * first tick with no client                                → ``(False, now)``
      * still orphaned, within grace                             → ``(False, orphan_since)``
      * orphaned past grace                                      → ``(True, None)``
    """
    if mode == ExecutionMode.canvas.value and ws_count == 0:
        if orphan_since is None:
            return False, now
        if now - orphan_since >= grace_sec:
            return True, None
        return False, orphan_since
    return False, None


class ProcessServices:
    """Per-process holder of the workspace component registry + job scheduler.

    Constructed lazily once per process by :func:`get_services`. Not a global:
    a separate OS process (e.g. an eval subprocess) gets its own instance.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.workspace_component_registry = WorkspaceComponentRegistry(
            settings.workspace_dir,
            active_dir=settings.active_workspace_dir or None,
        )
        # JobScheduler is created in main.py lifespan once outputs/eval_runs/
        # is resolvable + nvidia-smi has been queried for total VRAM. M1
        # leaves ExecutionGuard.eval mode in place for the legacy in-process
        # path; the subprocess path goes through this scheduler instead.
        self.job_scheduler: Any = None
        # CodingAgentRunner — created in main.py lifespan (Coding-Agent
        # Monitor tab); None in eval subprocesses and tests.
        self.coding_agent_runner: Any = None
        # HumanRunner — created in main.py lifespan (Human tab, interactive
        # human driver over env_habitat); None in eval subprocesses and tests.
        self.human_runner: Any = None


# Module-level singleton
_state: ProcessServices | None = None


def get_services() -> ProcessServices:
    global _state
    if _state is None:
        _state = ProcessServices()
    return _state


# ── WebSocket Broadcast (cross-cutting, used by engine + API) ──

_ws_connections: set[Any] = set()


def register_ws_client(ws: Any) -> None:
    """Add a WebSocket connection to the broadcast set."""
    _ws_connections.add(ws)


def unregister_ws_client(ws: Any) -> None:
    """Remove a WebSocket connection from the broadcast set."""
    _ws_connections.discard(ws)


def ws_client_count() -> int:
    """Return the number of active WebSocket connections."""
    return len(_ws_connections)


async def broadcast(msg: WSMessage, agent_id: str | None = None) -> None:
    """Send a message to all connected WebSocket clients."""
    payload = msg.model_dump(mode="json")
    if agent_id is not None:
        payload["agent_id"] = agent_id
    dead: set[Any] = set()
    for ws in list(_ws_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _ws_connections.discard(ws)
