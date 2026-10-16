"""LoopRunner — the canvas session: pause / stop / status over one GraphRun.

Receives a GraphDefinition JSON from the frontend canvas and runs it through
:class:`~app.agent_loop.graph_run.GraphRun`. Adds the lifecycle the canvas
needs (pause / resume / stop, checkpoints, status) on top of the
executor-facing :class:`~app.agent_loop.graph_run.RunSession` contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from ..graph_def import GraphDefinition, HookDef
from ..logging import ExecutionLogger
from ..state import broadcast
from .graph_executor import GraphExecutor
from .graph_run import ExecutionPrinciples, GraphRun, RunSession

__all__ = ["ExecutionPrinciples", "LoopRunner", "get_loop_runner"]

log = logging.getLogger("agentcanvas.loop-runner")


class LoopRunner(RunSession):
    """Stateful runner for a single loop execution.

    Manages pause/stop/status state and delegates execution to GraphRun.
    Domain-agnostic — all environment and policy interaction happens through
    nodeset nodes wired on the canvas.
    """

    def __init__(
        self,
        logger: Any = None,
        env_panel_overrides: dict[str, Any] | None = None,
        server_url_overrides: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._logger: Any = logger  # ExecutionLogger | None
        # Per-runner env panel overrides (ADR-028). Stored here so each
        # run's fresh executor inherits the same routing. None or {} =
        # global registry only.
        self._env_panel_overrides: dict[str, Any] = env_panel_overrides or {}
        # Per-runner server URL overrides (ADR-028 PB-1.5). Maps nodeset
        # name → tagged subprocess URL so in-graph proxy nodes
        # (env_habitat__step_native, ...) route to this worker's own env
        # subprocess instead of the URL baked into their proxy class
        # closure. Empty/None = pass-through to the baked URL.
        self._server_url_overrides: dict[str, str] = server_url_overrides or {}
        # Each run builds a fresh executor (see run()); this one answers
        # get_checkpoints / restore_step before the first run.
        self._executor = GraphExecutor(
            logger=logger,
            env_panel_overrides=self._env_panel_overrides,
            server_url_overrides=self._server_url_overrides,
        )
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # starts un-paused

    async def run(
        self,
        loop_def: GraphDefinition,
        step_delay_ms: int = 200,
        global_hooks: list[HookDef] | None = None,
        principles: ExecutionPrinciples | None = None,
        step_budget_override: int | None = None,
    ) -> None:
        """Run a graph definition once.

        Args:
            principles: Optional eval-mode configuration. When None (default),
                        no behavioral changes — canvas path is unchanged.
            step_budget_override: Optional per-episode iteration cap that
                        overrides ``loop_def.step_budget``. Passed through to
                        the executor.
        """
        self._stop_event.clear()
        self._pause_event.set()
        self._current_step = 0
        self._metrics = None
        self.principles = principles

        # In eval batch mode with no_pause, keep pause_event always set
        pause_event = self._pause_event
        if principles and principles.no_pause:
            pause_event = asyncio.Event()
            pause_event.set()  # permanently un-paused

        # Auto-create logger for canvas runs if none was injected.
        logger = self._logger
        _self_created_logger = False
        if logger is None and self._execution_id:
            repo_root = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "..",
                )
            )
            persist_dir = os.path.join(repo_root, "outputs", "runs", self._execution_id)
            logger = ExecutionLogger(
                execution_id=self._execution_id,
                source="canvas",
                persist_dir=persist_dir,
            )
            _self_created_logger = True

        # Persist graph definition alongside the log for replay — only for
        # self-created (canvas) loggers. An injected logger points at a
        # caller-owned dir whose run-level graph.json the caller writes.
        if _self_created_logger and logger._persist_dir:
            graph_path = os.path.join(logger._persist_dir, "graph.json")
            if not os.path.exists(graph_path):
                try:
                    with open(graph_path, "w") as f:
                        json.dump(loop_def.to_dict(), f)
                except Exception as e:
                    log.warning("Failed to save graph.json: %s", e)

        # Fresh executor for each run; carry the runner's env panel +
        # server-URL overrides forward (ADR-028 PA-1, PB-1.5).
        graph_run = GraphRun(
            loop_def,
            session=self,
            logger=logger,
            env_panel_overrides=self._env_panel_overrides,
            server_url_overrides=self._server_url_overrides,
        )
        self._executor = graph_run.executor
        await graph_run.run(
            step_delay_ms=step_delay_ms,
            global_hooks=global_hooks,
            step_budget_override=step_budget_override,
            stop_event=self._stop_event,
            pause_event=pause_event,
        )

    async def pause(self) -> None:
        self._pause_event.clear()
        self._status = "paused"
        await broadcast(self._ws("nav_status", {"status": "paused", "step": self._current_step}))

    async def resume(self) -> None:
        self._pause_event.set()
        self._status = "running"
        await broadcast(self._ws("nav_status", {"status": "running", "step": self._current_step}))

    async def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._status = "idle"
        self._current_step = 0
        await broadcast(self._ws("nav_status", {"status": "idle", "step": 0}))

    def get_checkpoints(self) -> list:
        """List available checkpoint step numbers."""
        return self._executor.get_checkpoints()

    def restore_step(self, step: int) -> bool:
        """Restore execution state to a previous checkpoint step."""
        return self._executor.restore_step(step)

    def get_status(self) -> dict:
        return {
            "status": self._status,
            "step": self._current_step,
            "metrics": self._metrics,
            "execution_id": self._execution_id,
        }


# Singleton
_runner: LoopRunner | None = None


def get_loop_runner() -> LoopRunner:
    global _runner
    if _runner is None:
        _runner = LoopRunner()
    return _runner
