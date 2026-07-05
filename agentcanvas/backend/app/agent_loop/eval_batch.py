"""Batch evaluation runner — runs a graph repeatedly across episodes.

Wraps LoopRunner (same engine as canvas Play button) in an episode loop.
Each episode gets a fresh LoopRunner instance for clean state isolation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..graph_def import GraphDefinition
from ..logging import ExecutionLogger
from ..models import WSMessage
from ..state import ExecutionGuard, broadcast
from .env_worker_pool import EnvWorkerPool, resolve_per_step_budget
from .loop_runner import ExecutionPrinciples, LoopRunner

log = logging.getLogger("agentcanvas.eval-batch")


def _partition_contiguous(total: int, worker_count: int) -> list[list[int]]:
    """Split ``range(total)`` into ``worker_count`` contiguous chunks.

    Used to pin episode chunks to specific worker subprocesses so each
    worker sees a bounded number of task switches (SAPIEN crash mitigation
    for cross-task sweeps; see EnvWorkerPool.acquire_specific). First
    ``total % worker_count`` workers get one extra item, matching numpy's
    ``array_split`` distribution. At ``worker_count >= total`` trailing
    workers get empty chunks and exit immediately.
    """
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    chunks: list[list[int]] = []
    base, remainder = divmod(total, worker_count)
    cursor = 0
    for w in range(worker_count):
        size = base + (1 if w < remainder else 0)
        chunks.append(list(range(cursor, cursor + size)))
        cursor += size
    return chunks


# ── Data types ──


class EvalStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    cancelled = "cancelled"
    error = "error"


@dataclass
class EpisodeResult:
    episode_index: int
    episode_id: str = ""
    scene_id: str = ""
    instruction: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    step_count: int = 0
    elapsed_sec: float = 0.0
    status: str = "pending"  # "completed" | "error" | "pending"
    error: str | None = None
    # ADR-028 PB-3: which worker drove this episode. 0 at worker_count=1.
    worker_id: int = 0
    # Effective selectors actually pushed through the env panel for this
    # episode = run-level ``selectors`` merged with the per-episode
    # override (when ``EvalConfig.episode_selectors`` is set). Empty for
    # runs that didn't use the cascade path. Drives ``aggregate_by_task``
    # grouping at run end.
    selectors: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for a batch eval run."""

    graph_name: str
    env_nodeset: str = ""
    # Generic cascade — dict insertion order is the order in which the
    # eval batch pushes fields through ``BaseEnvPanel.on_field_change``
    # before each episode. Each env nodeset declares its own cascade
    # (Habitat: dataset → split → episode_index; SIMPLER: split →
    # task_id → episode_index; HM-EQA: split → episode_index). Callers
    # can populate any subset; ``episode_index`` is always pushed last
    # by the runner and must not appear here.
    selectors: dict[str, Any] = field(default_factory=dict)
    # Legacy convenience fields — kept for serialization back-compat.
    # ``start_eval_v2`` merges these into ``selectors`` at request
    # handling time, so internal eval code reads ``selectors`` only.
    # ``dataset`` is empty for envs that don't need it (Habitat / VLN-CE);
    # populated to "R2R" / "R4R" / "RxR" / "REVERIE" / "CVDN" / "NDH" for
    # the multi-dataset MP3D nodeset (ADR-030 step-2).
    dataset: str = ""
    split: str = "val_unseen"
    episode_count: int = -1  # -1 = all
    start_episode_index: int = 0  # ADR-023: start from specific episode
    # ADR-028: worker-pool batch eval. ``worker_count`` is the number of
    # concurrent env subprocesses driving episodes in parallel. 1 = today's
    # sequential single-tenant behaviour. ``per_step_budget_sec`` bounds the
    # wall-clock budget for a single step; the per-episode timeout is
    # ``resolved_step_budget * per_step_budget_sec`` (resolved per episode
    # via the env hook + graph + API resolver chain). None falls back to
    # the env nodeset's ``default_per_step_budget_sec`` ClassVar (BaseNodeSet
    # default 5.0; Habitat 2.0; MapGPT 30.0).
    worker_count: int = 1
    # Per-episode iteration cap. When None, the framework's resolver chain
    # picks: env-supplied dynamic value (per episode) → graph's authored
    # ``step_budget`` → ``DEFAULT_STEP_BUDGET``. When set, this is the
    # explicit user override and wins over everything except the loop's
    # own stop signal (the iterOut ``stop`` input).
    step_budget: int | None = None
    per_step_budget_sec: float | None = None
    # Optional explicit list of episode indices to evaluate. When set,
    # overrides ``start_episode_index`` + ``episode_count`` and dispatches
    # workers across exactly these indices (preserves order in the queue).
    episode_indices: list[int] | None = None
    # Optional per-episode selector overrides, parallel to the resolved
    # index list. When set, ``episode_selectors[i]`` is merged on top of
    # run-level ``selectors`` before cascade push for episode ``i``. Use
    # this for cross-task sweeps (e.g. SIMPLER 25 tasks x N episodes in a
    # single run). When None, every episode shares run-level selectors —
    # pre-existing behaviour, bit-identical. Length must match the
    # resolved index list (validated at start_eval_v2).
    episode_selectors: list[dict[str, Any]] | None = None


@dataclass
class EvalRun:
    """State of a batch evaluation run."""

    run_id: str
    config: EvalConfig
    status: EvalStatus = EvalStatus.pending
    episodes: list[EpisodeResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    # Per-task breakdown of metric means, grouped by the episode's
    # effective selectors. Key precedence: ``task_id`` when present in
    # selectors, else canonical JSON of the merged dict, else
    # ``"_default"`` for legacy runs that didn't push a cascade. Empty
    # for runs with no completed episodes.
    aggregate_by_task: dict[str, dict[str, float]] = field(default_factory=dict)
    total_episodes: int = 0
    created_at: str = ""
    finished_at: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None
    # Runtime state (not persisted)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def to_summary(self) -> dict:
        """Serialize for WS broadcast / REST response.

        ``completed_count`` counts every terminal episode — both clean
        finishes (status="completed") and infrastructure failures
        (status="error"). It drives the UI progress bar so the bar reaches
        100% exactly when the batch is done. ``error_count`` is the
        subset that errored out (Python exception, worker crash, env
        timeout) so the UI can show how many of the "completed" episodes
        actually broke vs ran cleanly.

        **Neither is a task success rate.** Real per-task success (e.g.
        SIMPLER ``info["success"]``, Habitat SPL/SR) lives in
        :attr:`EpisodeResult.metrics` and must be aggregated by the
        caller. The earlier ``success_count`` alias (status=="completed",
        i.e. ``completed_count - error_count``) was removed because the
        name was misleading in manipulation/EQA contexts where
        "completed" ≠ "succeeded".
        """
        completed_count = sum(1 for ep in self.episodes if ep.status in ("completed", "error"))
        error_count = sum(1 for ep in self.episodes if ep.status == "error")
        return {
            "run_id": self.run_id,
            "graph_name": self.config.graph_name,
            "env_nodeset": self.config.env_nodeset,
            "status": self.status.value,
            "selectors": dict(self.config.selectors),
            "dataset": self.config.dataset,
            "split": self.config.split,
            "episode_count": self.config.episode_count,
            "total_episodes": self.total_episodes,
            "completed_count": completed_count,
            "error_count": error_count,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "aggregate_metrics": self.aggregate_metrics,
            "aggregate_by_task": dict(self.aggregate_by_task),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    def to_episode_summary(self, ep: EpisodeResult) -> dict:
        """Serialize an episode for WS broadcast."""
        return {
            "run_id": self.run_id,
            "episode_index": ep.episode_index,
            "episode_id": ep.episode_id,
            "scene_id": ep.scene_id,
            "instruction": ep.instruction,
            "metrics": ep.metrics,
            "step_count": ep.step_count,
            "elapsed_sec": round(ep.elapsed_sec, 1),
            "status": ep.status,
            "error": ep.error,
            "worker_id": ep.worker_id,
            "selectors": dict(ep.selectors),
        }


# ── Batch Runner ──


class BatchEvalRunner:
    """Runs a graph repeatedly across episodes using LoopRunner.

    Each episode gets a fresh LoopRunner instance (same engine as canvas).
    The runner manages the episode loop, metrics collection, persistence,
    and WebSocket event broadcasting.
    """

    def __init__(self, run: EvalRun, graph: GraphDefinition) -> None:
        self._run = run
        self._graph = graph
        self._start_time = 0.0
        # Run-level artefact dir. Each episode gets its own ExecutionLogger
        # + subdir under episodes/ (see _run_one_episode) so per-episode
        # log.jsonl / assets/ never interleave across episodes or workers.
        # graph.json is the one run-level snapshot — episode dirs hold only
        # per-episode state.
        repo_root = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
            )
        )
        self._run_dir = os.path.join(repo_root, "outputs", "eval_runs", run.run_id)
        os.makedirs(self._run_dir, exist_ok=True)
        graph_path = os.path.join(self._run_dir, "graph.json")
        try:
            with open(graph_path, "w") as f:
                json.dump(graph.to_dict(), f)
        except OSError as e:
            log.warning("Failed to write run-level graph.json: %s", e)

    async def execute(self) -> None:
        """Run the full batch evaluation."""
        run = self._run
        run.status = EvalStatus.running
        self._start_time = time.time()

        await self._broadcast_progress()

        try:
            # Determine episode list. When ``episode_indices`` is set,
            # those exact indices are used (random sampling across the
            # dataset). Otherwise fall back to the consecutive range
            # ``[start_episode_index, start_episode_index + episode_count)``.
            if run.config.episode_indices:
                episode_indices = list(run.config.episode_indices)
                total = len(episode_indices)
            else:
                total = run.config.episode_count if run.config.episode_count > 0 else 10
                episode_indices = [run.config.start_episode_index + i for i in range(total)]
            run.total_episodes = total

            principles = ExecutionPrinciples(
                no_pause=True,
                collect_metrics=True,
                suppress_nav_events=True,
                source_tag="eval",
            )

            # ADR-028 rule (3): per-episode timeout = step_budget * per-step
            # budget. The step_budget itself is resolved per-episode in
            # ``_run_one_episode`` (env-dynamic → graph → default, with the
            # API override on top), so the wall-clock timeout is also
            # computed per episode there.
            per_step_budget = resolve_per_step_budget(
                run.config.env_nodeset or None,
                run.config.per_step_budget_sec,
            )
            log.info(
                "Eval run %s: per_step_budget=%.1fs, worker_count=%d, api_step_budget=%s",
                run.run_id,
                per_step_budget,
                run.config.worker_count,
                run.config.step_budget if run.config.step_budget is not None else "auto",
            )

            # ADR-028 PB-2: drive episodes through N concurrent worker
            # coroutines. Each worker pulls indices off a shared queue,
            # leases a pool worker (which carries its tagged env panel +
            # server-url overrides), runs one episode, and aggregates
            # under a lock. At worker_count=1 the queue + gather collapse
            # into a single coroutine doing one episode at a time —
            # behaviour-equivalent to the prior sequential loop.
            async with EnvWorkerPool(
                worker_count=run.config.worker_count,
                env_nodeset=run.config.env_nodeset or None,
            ) as pool:
                # Task-contiguous dispatch: split the index list into N
                # contiguous chunks, one per worker, and pin each chunk
                # to its worker's tagged subprocess. With cross-task
                # ``episode_selectors``, a contiguous slice of the list
                # (which the caller is expected to order by task) maps to
                # ⌈tasks/N⌉ task switches per subprocess — well under
                # SAPIEN's ~15-switch crash threshold for SIMPLER. At
                # worker_count=1 or single-task runs this is a no-op
                # change (one chunk = full list).
                chunks = _partition_contiguous(total, run.config.worker_count)
                queues: list[asyncio.Queue[int]] = []
                for slice_indices in chunks:
                    q: asyncio.Queue[int] = asyncio.Queue()
                    for i in slice_indices:
                        q.put_nowait(i)
                    queues.append(q)

                results_lock = asyncio.Lock()

                async def worker_loop(worker_id: int) -> None:
                    q = queues[worker_id]
                    while True:
                        if run.stop_event.is_set():
                            break
                        try:
                            i = q.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                        async with pool.acquire_specific(worker_id) as worker:
                            episode = await self._run_one_episode(
                                i,
                                episode_indices[i],
                                worker,
                                principles,
                                per_step_budget,
                            )

                        # Episode fully complete → drop its partition from any
                        # nodeset-owned keyed container (worker-safe: only this
                        # episode's key; concurrent siblings untouched). No-op
                        # for container-less / local-mode nodesets.
                        await self._evict_episode_state(episode.episode_id)

                        async with results_lock:
                            run.episodes.append(episode)
                            run.aggregate_metrics = self._compute_aggregate(run.episodes)
                            run.aggregate_by_task = self._compute_aggregate_by_task(run.episodes)
                            run.elapsed_sec = time.time() - self._start_time
                            await self._broadcast_episode_done(episode)
                            await self._broadcast_progress()

                await asyncio.gather(*[worker_loop(w) for w in range(run.config.worker_count)])

            # Episodes complete in arbitrary order under fan-out; sort by
            # episode_index for stable persistence + replay tooling
            # (ADR-028 PB-2 scaffolding open-question #1).
            run.episodes.sort(key=lambda ep: ep.episode_index)

            # Finalize
            run.elapsed_sec = time.time() - self._start_time
            if run.stop_event.is_set():
                run.status = EvalStatus.cancelled
            else:
                run.status = EvalStatus.completed
            run.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        except Exception as e:
            log.exception("Eval run %s failed", run.run_id)
            run.status = EvalStatus.error
            run.error = str(e)
            run.elapsed_sec = time.time() - self._start_time
            run.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        finally:
            ExecutionGuard.release(run.run_id)
            await self._broadcast_complete()

    async def _evict_episode_state(self, episode_id: str) -> None:
        """Best-effort: drop ``episode_id``'s partition from any nodeset-owned
        keyed container in the server-mode nodesets this graph uses.

        This is the worker-safe replacement for the old module-global "never
        clear" race fix: it deletes exactly one key, so concurrent sibling
        episodes under other keys are untouched (constraint A). Silently
        no-ops for local-mode / container-less nodesets (the ``/containers/evict``
        route is a no-op there) and never lets an eviction failure affect the
        eval result.
        """
        if not episode_id:
            return
        try:
            import httpx

            from ..server._loopback_proxy import loopback_httpx_kwargs
            from ..state import get_services

            registry = get_services().workspace_component_registry
        except Exception:
            return
        # Nodesets used by this graph = node-type prefix before "__".
        names = {
            node.type.split("__", 1)[0]
            for node in self._graph.nodes
            if "__" in (getattr(node, "type", "") or "")
        }
        urls = []
        for name in names:
            try:
                url = registry.get_server_url(name)
            except Exception:
                url = None
            if url:
                urls.append(url)
        if not urls:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0, **loopback_httpx_kwargs()) as client:
                for url in urls:
                    # route absent / nodeset local → ignore
                    with contextlib.suppress(Exception):
                        await client.post(f"{url}/containers/evict", json={"key": episode_id})
        except Exception:
            pass

    async def _run_one_episode(
        self,
        loop_index: int,
        actual_index: int,
        worker: Any,
        principles: ExecutionPrinciples,
        per_step_budget: float,
    ) -> EpisodeResult:
        """Run a single episode on the given worker.

        Resolves the env panel per-worker: at PB-2 multi-worker, the
        worker carries a tagged ``RemoteEnvPanelProxy`` in
        ``env_panel_overrides`` so set-episode/play hits this worker's
        own subprocess. At ``worker_count=1`` the override is empty and
        we fall back to the global registry — Phase A bit-identical.

        Per-episode step budget is resolved here via the chain:
          1. ``run.config.step_budget`` (explicit API override) — wins
             over everything when set.
          2. ``state["step_budget"]`` from the env panel's on_load
             return — env-supplied per-episode value (e.g. HM-EQA's
             scene-adaptive ``int(sqrt(scene_size) * 3)``).
          3. ``self._graph.step_budget`` — graph author's static value.
          4. ``DEFAULT_STEP_BUDGET`` — framework failsafe.
        The wall-clock timeout is then ``resolved_budget * per_step_budget``.
        """
        from ..config import DEFAULT_STEP_BUDGET as _DEFAULT_STEP_BUDGET

        run = self._run
        env_nodeset = run.config.env_nodeset or "env_habitat"
        env_panel = worker.env_panel_overrides.get(env_nodeset)
        if env_panel is None:
            from ..components.env_panel import get_env_panel as _global_get_env_panel

            env_panel = _global_get_env_panel(env_nodeset)

        # Resolve effective selectors for this episode. Per-episode
        # overrides (when provided) merge on top of run-level — same
        # insertion-order semantics: run-level fields cascade first, then
        # any episode-specific override, with later keys winning on
        # conflict. Stored on EpisodeResult so persistence + per-task
        # aggregate can read it back without re-deriving the merge.
        effective_selectors: dict[str, Any] = dict(run.config.selectors)
        if run.config.episode_selectors is not None:
            override = run.config.episode_selectors[loop_index]
            for k, v in override.items():
                effective_selectors[k] = v

        ep_start = time.time()
        episode = EpisodeResult(
            episode_index=actual_index,
            status="pending",
            worker_id=worker.worker_id,
            selectors=effective_selectors,
        )
        # Pre-bind runner so the timeout/error branches can still pull
        # partial metrics + step_count when the wait_for trips mid-run.
        runner: Any = None

        # Per-episode artefact dir — log.jsonl + assets/ + episode.json all
        # land here, never interleaved with sibling episodes. Created up
        # front so episode.json is written even when the episode errors
        # before the LoopRunner (and its logger) is constructed.
        ep_dir = os.path.join(self._run_dir, "episodes", f"ep{actual_index:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        ep_execution_id = f"{run.run_id}_ep{actual_index:04d}"

        env_step_budget: int | None = None

        try:
            # Set episode via env panel before running.
            # on_field_change updates the cached index; on_action("play")
            # commits it (calls set_episode_by_index under the hood).
            if env_panel is not None:
                # Push every cascade field the caller specified, in the
                # order resolved on ``effective_selectors`` (run-level
                # first, per-episode override applied on top — see
                # earlier merge). Each env nodeset's env panel declares
                # its own cascade shape (dataset → split → episode_index
                # for Habitat/MP3D; split → task_id → episode_index for
                # SIMPLER/LIBERO; split → episode_index for HM-EQA/
                # OpenEQA). Unknown fields are no-ops on env panels that
                # don't declare them, so this stays safe across envs.
                for cascade_field, cascade_value in effective_selectors.items():
                    await env_panel.on_field_change(cascade_field, cascade_value)
                await env_panel.on_field_change("episode_index", actual_index)
                play_result = await env_panel.on_action("play", {})
                if not play_result.get("ok"):
                    raise RuntimeError(play_result.get("error", "Failed to set episode"))
                state = await env_panel.on_load()
                ep_info = state.get("current_episode", {}) if isinstance(state, dict) else {}
                if "error" not in ep_info:
                    # MP3D names these instr_id / scan; Habitat uses the
                    # canonical episode_id / scene_id. Accept both.
                    episode.episode_id = ep_info.get("episode_id", "") or ep_info.get(
                        "instr_id", ""
                    )
                    episode.scene_id = ep_info.get("scene_id", "") or ep_info.get("scan", "")
                    episode.instruction = ep_info.get("instruction", "")
                # Env may publish a per-episode step budget via on_load
                # (e.g. HM-EQA returns int(sqrt(scene_size) * 3)). Accept
                # both the new ``step_budget`` field and the legacy
                # ``max_steps_default`` for unmigrated env nodesets.
                if isinstance(state, dict):
                    raw = state.get("step_budget", state.get("max_steps_default"))
                    if isinstance(raw, int) and raw > 0:
                        env_step_budget = raw

            # Resolve the per-episode budget (precedence in docstring).
            if run.config.step_budget is not None:
                resolved_budget = run.config.step_budget
            elif env_step_budget is not None:
                resolved_budget = env_step_budget
            elif self._graph.step_budget is not None:
                resolved_budget = self._graph.step_budget
            else:
                resolved_budget = _DEFAULT_STEP_BUDGET
            # A zero/negative budget makes ``range(resolved_budget)`` run
            # zero iterations and the episode silently completes with
            # status=completed + empty metrics. That happens when an env
            # signals failure via on_load (e.g. HM-EQA returns
            # int(sqrt(scene_size)*3)=0 on a degenerate navmesh). Surface
            # it as an error so the eval result reflects reality.
            if resolved_budget < 1:
                raise RuntimeError(
                    f"step_budget resolved to {resolved_budget} "
                    f"(env={env_step_budget}, graph={self._graph.step_budget}, "
                    f"api={run.config.step_budget}); refusing to run a 0-step episode"
                )
            episode_timeout = max(1.0, resolved_budget * per_step_budget)
            log.info(
                "Episode %d (worker=%d) step_budget=%d (env=%s, graph=%s, api=%s), timeout=%.1fs",
                actual_index,
                worker.worker_id,
                resolved_budget,
                env_step_budget,
                self._graph.step_budget,
                run.config.step_budget,
                episode_timeout,
            )

            # Create fresh LoopRunner per episode (clean state). The worker's
            # env_panel_overrides bind env-panel lookups, and
            # server_url_overrides bind in-graph proxy node URLs, both to
            # this worker's tagged subprocess (ADR-028 PB-1, PB-1.5). Both
            # are empty at worker_count=1 — Phase A bit-identical.
            ep_logger = ExecutionLogger(
                execution_id=ep_execution_id,
                source="eval",
                persist_dir=ep_dir,
            )
            runner = LoopRunner(
                logger=ep_logger,
                env_panel_overrides=worker.env_panel_overrides,
                server_url_overrides=worker.server_url_overrides,
            )
            runner._execution_id = ep_execution_id

            # Run the graph once (same as canvas Play button), wrapped in
            # asyncio.wait_for so a stuck episode can never block the pool.
            # The resolved budget is passed as ``step_budget_override`` so
            # parallel workers don't race to mutate ``self._graph`` itself.
            await asyncio.wait_for(
                runner.run(
                    self._graph,
                    step_delay_ms=0,
                    principles=principles,
                    step_budget_override=resolved_budget,
                ),
                timeout=episode_timeout,
            )

            # Collect metrics from the executor's node state
            metrics = self._collect_metrics(runner)
            episode.metrics = metrics
            episode.step_count = runner._current_step
            episode.elapsed_sec = time.time() - ep_start
            # Node-error conviction (2026-07-04): the executor swallows all
            # exceptions — its error surface is ``session._status`` plus the
            # error bus — so a run that recorded node failures RETURNS
            # normally with ``_status == "error"``. Convict the episode
            # explicitly; before this, such episodes masqueraded as
            # completed (status="completed", step_count=0, metrics={} —
            # run 20260516_101057, 11/100 episodes silently dropped).
            node_errors = getattr(getattr(runner, "_executor", None), "node_errors", None) or []
            if getattr(runner, "_status", "") == "error":
                episode.status = "error"
                if node_errors:
                    head = "; ".join(
                        f"{err['node_id']}@step{err['step']}: {err['error']}"
                        for err in node_errors[:3]
                    )
                    episode.error = (
                        f"{len(node_errors)} node error(s) during run: {head}"
                        + ("; ..." if len(node_errors) > 3 else "")
                    )
                else:
                    episode.error = "graph execution ended with status=error"
            # Verdict-required: an eval graph that "completed" without any
            # graphOut snapshot produced no verdict — the loop never really
            # ran. Refuse to record that as a completed episode; it would
            # alias to success=0 and poison SR.
            elif getattr(self._graph, "eval_graph", True) and not metrics:
                episode.status = "error"
                episode.error = (
                    "no verdict: eval graph finished without a graphOut "
                    "snapshot (final side never emitted)"
                )
            else:
                episode.status = "completed"

        except asyncio.TimeoutError:
            log.warning(
                "Episode %d (worker=%d) exceeded %.1fs episode timeout",
                loop_index,
                worker.worker_id,
                episode_timeout,
            )
            episode.status = "error"
            episode.error = f"timeout after {episode_timeout:.1f}s"
            episode.elapsed_sec = time.time() - ep_start
            if runner is not None:
                episode.metrics = self._collect_metrics(runner)
                episode.step_count = getattr(runner, "_current_step", 0)
        except Exception as e:
            log.exception("Episode %d (worker=%d) failed", loop_index, worker.worker_id)
            episode.status = "error"
            episode.error = str(e)
            episode.elapsed_sec = time.time() - ep_start
            if runner is not None:
                episode.metrics = self._collect_metrics(runner)
                episode.step_count = getattr(runner, "_current_step", 0)

        # Per-episode self-describing record — lets one episode dir be
        # consumed (replay, debugging, architect tooling) without joining
        # back to run-level summary.json.
        from ..api.execution.eval_storage import episode_to_dict

        try:
            with open(os.path.join(ep_dir, "episode.json"), "w") as f:
                json.dump(episode_to_dict(episode), f, indent=2)
        except OSError as e:
            log.warning("Failed to write episode.json for ep %d: %s", actual_index, e)

        return episode

    def _collect_metrics(self, runner: LoopRunner) -> dict[str, float]:
        """Extract metrics from the graph's ``graphOut`` sinks.

        The graph declares its outputs via one or more ``graphOut`` nodes;
        each carries a ``config.portName`` that becomes the metric key. The
        executor writes ``node.state["_last_inputs"]`` after every fire
        (``graph_executor.py``); the terminal-iter fire is protected by
        the iterOut settle loop, so on a successful episode the snapshot
        reflects the terminal-step values.

        Two contributions per ``graphOut``:

        * ``portName == "metrics"`` — the value is treated as a metrics
          dict (or JSON-stringified dict, since server-mode evaluate
          nodes cross the HTTP boundary on a TEXT-typed wire) and its
          numeric fields are flattened into the returned dict.
        * Any other portName carrying a numeric/bool scalar — included
          as ``metrics[portName] = float(value)`` so eval graphs can expose
          ``success``, ``score``, etc. as first-class outputs without
          bundling them under a ``metrics`` dict.
        """
        from .scope_analysis import GRAPH_SCOPE_ID

        metrics: dict[str, float] = {}
        executor = runner._executor
        if executor is not None and hasattr(executor, "nodes"):
            for node in executor.nodes.values():
                if node.type != "graphOut":
                    continue
                # Multi-scope: only harvest graph-scope graphOuts (the
                # eval-output sinks at the outer level). Inner-scope
                # graphOut nodes preserved by `flatten_graph` (when the
                # composite has its own scope) are scope-internal latches
                # carrying per-iter cross-scope return values, NOT eval
                # metrics; skipping them keeps the harvest faithful to
                # the author's eval-output declaration.
                if hasattr(executor, "_scope_of"):
                    n_scope = executor._scope_of(node.id)
                    if n_scope and n_scope != GRAPH_SCOPE_ID:
                        continue
                snapshot = node.state.get("_last_inputs") or {}
                if not snapshot:
                    continue
                value = snapshot.get("value")
                if value is None:
                    # Permissive fallback — accept any non-None pending input
                    for v in snapshot.values():
                        if v is not None:
                            value = v
                            break
                port_name = (node.config or {}).get("portName") or ""

                if port_name == "metrics":
                    if isinstance(value, str) and value:
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            value = None
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, (int, float, bool)):
                                metrics[k] = float(v)
                elif port_name and isinstance(value, (int, float, bool)):
                    metrics[port_name] = float(value)
        if runner._metrics and isinstance(runner._metrics, dict):
            for k, v in runner._metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        return metrics

    @staticmethod
    def _compute_aggregate(episodes: list[EpisodeResult]) -> dict[str, float]:
        """Average numeric metrics across completed episodes."""
        completed = [ep for ep in episodes if ep.status == "completed" and ep.metrics]
        if not completed:
            return {}
        all_keys: set[str] = set()
        for ep in completed:
            all_keys.update(ep.metrics.keys())
        aggregated: dict[str, float] = {}
        for k in sorted(all_keys):
            values = [ep.metrics[k] for ep in completed if k in ep.metrics]
            if values:
                aggregated[k] = sum(values) / len(values)
        return aggregated

    @staticmethod
    def _episode_task_key(ep: EpisodeResult) -> str:
        """Group key for cross-task aggregation.

        Precedence: explicit ``task_id`` in selectors > canonical JSON of
        the full selectors dict > ``"_default"``. JSON fallback uses
        ``sort_keys=True`` so equivalent selectors collapse to the same
        bucket regardless of insertion order.
        """
        sel = ep.selectors or {}
        tid = sel.get("task_id")
        if isinstance(tid, str) and tid:
            return tid
        if sel:
            try:
                import json

                return json.dumps(sel, sort_keys=True, default=str)
            except Exception:
                return "_default"
        return "_default"

    @classmethod
    def _compute_aggregate_by_task(
        cls, episodes: list[EpisodeResult]
    ) -> dict[str, dict[str, float]]:
        """Per-task breakdown of metric means.

        Mirrors ``_compute_aggregate`` but partitions episodes by the
        group key from ``_episode_task_key`` first. Returns ``{}`` when
        no completed episodes carry numeric metrics.
        """
        completed = [ep for ep in episodes if ep.status == "completed" and ep.metrics]
        if not completed:
            return {}
        groups: dict[str, list[EpisodeResult]] = {}
        for ep in completed:
            groups.setdefault(cls._episode_task_key(ep), []).append(ep)
        out: dict[str, dict[str, float]] = {}
        for key, group_eps in groups.items():
            metric_keys: set[str] = set()
            for ep in group_eps:
                metric_keys.update(ep.metrics.keys())
            agg: dict[str, float] = {}
            for mk in sorted(metric_keys):
                values = [ep.metrics[mk] for ep in group_eps if mk in ep.metrics]
                if values:
                    agg[mk] = sum(values) / len(values)
            out[key] = agg
        return out

    async def _broadcast_progress(self) -> None:
        await broadcast(
            WSMessage(
                type="eval_progress",
                data=self._run.to_summary(),
                source="eval",
            )
        )

    async def _broadcast_episode_done(self, episode: EpisodeResult) -> None:
        await broadcast(
            WSMessage(
                type="eval_episode_done",
                data=self._run.to_episode_summary(episode),
                source="eval",
            )
        )

    async def _broadcast_complete(self) -> None:
        await broadcast(
            WSMessage(
                type="eval_complete",
                data=self._run.to_summary(),
                source="eval",
            )
        )
