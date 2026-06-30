"""FastMCP server wiring — registers the 5 MVP tools.

The backend URL is bound at construction time (after BackendManager has
spawned/borrowed); each tool function captures it via closure rather
than a module global so unit tests can construct multiple servers.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import eval as eval_tools
from .tools import graph as graph_tools
from .tools import runs as runs_tools


def build_server(backend_url: str) -> FastMCP:
    """Construct a FastMCP server bound to the given backend URL.

    Tool docstrings are surfaced to the LLM as the tool's ``description``;
    type hints become the ``inputSchema``.
    """
    mcp = FastMCP("agentcanvas-backend")

    @mcp.tool()
    def eval_start(
        graph_name: str,
        episode_count: int = 10,
        worker_count: int = 1,
        step_budget: int | None = None,
        split: str = "val_unseen",
        selectors: dict[str, Any] | None = None,
        start_episode_index: int = 0,
        episode_indices: list[int] | None = None,
        episode_selectors: list[dict[str, Any]] | None = None,
        per_step_budget_sec: float | None = None,
    ) -> dict[str, Any]:
        """Start a batch eval run on a graph.

        Spawns this conversation's owned backend (already running, since
        MCP claimed a pool slot at startup). Backend self-exits when the
        run finishes; results land in ``outputs/eval_runs/{run_id}/summary.json``.

        Args:
            graph_name: Name of the graph (no extension), e.g. "navgpt_mp3d".
                See ``graph_list`` for available options.
            episode_count: Number of episodes to run. Pass -1 for all
                episodes in the split. Default 10.
            worker_count: Parallel worker processes (within this backend).
                Default 1. Larger values speed up wall time but may hit
                LLM TPM limits.
            step_budget: Per-episode iteration cap. ``None`` (default)
                lets the framework's resolver pick from the env's
                per-episode value, then graph's authored value, then
                system default.
            split: Dataset split (e.g. "val_unseen"). Pass empty string
                for graphs without splits (some EQA-shaped envs).
            selectors: Generic env-controller cascade dict (insertion
                order = the order the runner pushes fields through the
                controller). Use when the env has cascade fields beyond
                ``split`` — e.g. SIMPLER manipulation: ``selectors=
                {"task_id": "google_robot_pick_horizontal_coke_can"}``
                alongside ``split="google_robot"``. Do NOT include
                ``episode_index`` — the runner pushes that itself.
            start_episode_index: Index into the split's episode list.
                Ignored if ``episode_indices`` is set.
            episode_indices: Explicit list of episode indices to run.
                Overrides ``start_episode_index`` + ``episode_count``.
                Use for random sampling.
            episode_selectors: Per-episode selector overrides for
                cross-task sweeps in a single run. Each entry merges on
                top of run-level ``selectors`` for that episode. Length
                must match the resolved index list. Order entries
                **task-contiguous** (all episodes of task A, then all of
                task B, etc.) — workers consume contiguous chunks, so
                interleaving tasks would force every worker subprocess
                to switch tasks every episode (SAPIEN/SIMPLER crashes
                after ~15 task switches). Do NOT include ``episode_index``.
            per_step_budget_sec: Wall-clock budget per step (seconds).
                None = no budget enforced.
        """
        return eval_tools.eval_start(
            backend_url,
            graph_name=graph_name,
            episode_count=episode_count,
            worker_count=worker_count,
            step_budget=step_budget,
            split=split,
            selectors=selectors,
            start_episode_index=start_episode_index,
            episode_indices=episode_indices,
            episode_selectors=episode_selectors,
            per_step_budget_sec=per_step_budget_sec,
        )

    @mcp.tool()
    def eval_status() -> dict[str, Any]:
        """Get status of the currently active eval run.

        Returns ``{"status": "none", "run": null}`` when idle, or a full
        ``{"status", "run": {...}}`` dict with ``run.completed_count``,
        ``run.total_episodes``, ``run.elapsed_sec``, ``run.aggregate_metrics``,
        ``run.error`` etc. Poll this to track progress.
        """
        return eval_tools.eval_status(backend_url)

    @mcp.tool()
    def eval_export(run_id: str) -> dict[str, Any]:
        """Export full results for a completed eval run.

        Reads ``outputs/eval_runs/{run_id}/summary.json`` directly — no
        backend required, so historical runs from any past conversation
        are queryable. For live progress on an in-flight run, use
        ``eval_status`` instead (the summary file is only written at
        completion).

        Returns the full export JSON: ``run_id``, ``config``, ``status``,
        ``episodes[]`` (per-episode metrics), ``aggregate_metrics``,
        ``created_at``, ``finished_at``, ``elapsed_sec``, ``error``.

        Args:
            run_id: 8-char run identifier returned by ``eval_start``.
        """
        return eval_tools.eval_export(run_id)

    @mcp.tool()
    def eval_stop() -> dict[str, Any]:
        """Stop the currently active eval run.

        Backend transitions the run to ``cancelled`` status; persisted
        results include partial episodes. Returns ``{run_id, status: "stopping"}``
        or ``{error}`` if no run is active.
        """
        return eval_tools.eval_stop(backend_url)

    @mcp.tool()
    def graph_list() -> dict[str, Any]:
        """List all available graphs with their experiment profiles.

        Reads ``workspace/graphs/*.json`` directly. For each graph, also
        loads the sibling ``{graph}.exp.yaml`` if present (advisory
        defaults: ``primary_metric``, ``split``, ``worker_count``,
        ``max_steps``). Use this to pick reasonable ``eval_start`` args.
        """
        return graph_tools.graph_list()

    @mcp.tool()
    def eval_runs_list(limit: int = 50) -> dict[str, Any]:
        """List recent eval runs from disk (newest first).

        Reads ``outputs/eval_runs/*/summary.json`` directly — no backend
        required. Each entry has the same fields as ``eval_export`` minus
        the per-episode detail, plus ``episode_count_saved``. Use this
        to find historical runs by graph / status / aggregate metrics
        before pulling full results with ``eval_export(run_id)``.

        Args:
            limit: Max number of runs to return. Default 50.
        """
        return runs_tools.eval_runs_list(limit)

    return mcp
