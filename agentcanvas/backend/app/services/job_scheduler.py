"""JobScheduler — admission + queue + Popen pool for run subprocesses.

Owned as a singleton by ProcessServices. One ``tick()`` per second from the
FastAPI lifespan loop:

  - observe: attribute per-PID resource usage (VRAM + RAM) to jobs /
    shared singletons (ResourceStatsTracker)
  - reap finished Popen children, set terminal status (``succeeded`` if
    ``_DONE`` exists, else ``aborted``), sediment resource calibration
  - admit queued jobs through the measured resource gate, one inequality
    per measurable resource: calibrated estimate (declared / cold-start
    fallback) vs sampler-measured free minus standing reservations — see
    ``_admit``; canvas lock + exclusive boolean gates unchanged

Spawn is ``setsid`` so each job becomes its own process group; cancel
sends SIGTERM to the pgid (kills the run subprocess + any env worker
descendants in one shot).

Persistence: spec.json + shared_urls.json + summary.json + _DONE under
``outputs/eval_runs/{run_id}/``. The scheduler holds no in-memory truth
that disk doesn't already have. On a parent backend restart, completed
runs survive; running AND queued jobs are marked ``aborted`` by
``reconcile_aborted_runs`` (M1 — no queue durability yet).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .resource_stats import RESOURCES, ResourceStatsTracker, _source_hash
from .run_state_io import (
    atomic_write_json,
    initial_running_summary,
    is_done,
    mark_aborted,
    read_shared_urls,
    read_summary,
    write_shared_urls,
    write_spec,
)

log = logging.getLogger("agentcanvas.scheduler")

# Budget headroom reserved for shared singletons (Prismatic VLM ~14 GB etc.)
# when deriving usable VRAM from the physical total in ``create``. Override
# the whole derivation with AGENTCANVAS_USABLE_VRAM_MB.
SHARED_HEADROOM_MB = 16000


_TERMINAL_STATUSES = {"succeeded", "completed", "failed", "error", "cancelled", "aborted"}


@dataclass
class _QueuedJob:
    run_id: str
    spec: dict
    submitted_at: float
    marginal_vram_mb: int
    exclusive_gpu: bool
    priority: str = "normal"  # high / normal / low (M1 ignores; FIFO within all)
    # Shared singletons this job consumes (keys of spec's _shared_urls).
    # Used by the refcount in _shared_consumer_count to auto-unload a
    # shared nodeset once no remaining job needs it.
    shared_nodesets: tuple[str, ...] = ()


@dataclass
class _RunningJob:
    run_id: str
    proc: subprocess.Popen
    pgid: int
    marginal_vram_mb: int
    exclusive_gpu: bool
    started_at: float
    # What admission charged this job at admit time, per resource
    # (calibrated estimate, declared fallback, or empty for cold-start /
    # legacy) and which basis produced it. The job's live reservation per
    # resource = max(0, admit charge - its measured tree usage) — decays
    # to zero as the real allocation materializes in the sampler.
    admit_charges: dict[str, int] = field(default_factory=dict)
    admit_basis: str = "legacy"
    cancel_requested: bool = False
    # Open file objects for the subprocess's stdout/stderr. Held here so
    # _reap can close them on exit; otherwise we'd leak two FDs per run.
    log_files: tuple[Any, Any] | None = None
    # TODO #60: ephemeral auto_host tag (e.g. ``ephem-20260515_141201``)
    # for shared nodesets the overlay redefined. None = no ephemerals
    # spawned for this run. _reap uses this to release them on exit.
    ephem_tag: str | None = None
    # Mirrors _QueuedJob.shared_nodesets — carried across the admit
    # boundary so _reap can decrement the consumer refcount.
    shared_nodesets: tuple[str, ...] = ()


class JobScheduler:
    """Singleton scheduler. Methods are safe to call from FastAPI handlers
    (they only mutate in-memory queue + spawn Popen; no awaits except
    ``submit`` which writes a few small JSON files to disk).
    """

    def __init__(
        self,
        eval_runs_dir: Path,
        usable_vram_mb: int,
        backend_url: str = "http://127.0.0.1:8765",
        total_vram_mb: int = 0,
    ) -> None:
        self._runs_dir = Path(eval_runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._usable_vram_mb = int(usable_vram_mb)
        # Physical ceiling for submit-time feasibility rejection; 0 = unknown
        # (falls back to usable_vram_mb as the bound).
        self._total_vram_mb = int(total_vram_mb)
        self._backend_url = backend_url

        self._queue: list[_QueuedJob] = []
        self._running: dict[str, _RunningJob] = {}

        # Canvas Play exclusivity: when True, no new jobs admit. Set by
        # ExecutionGuard.canvas state — caller polls it via callback.
        self._canvas_lock_held: callable = lambda: False

        # TODO #60: WorkspaceComponentRegistry handle for ephemeral spawn on
        # overlay-modified shared nodesets. Set via
        # ``set_workspace_component_registry`` from main.py lifespan. None = no
        # overlay-aware behavior (eval submits with active_workspace_dir
        # touching shared nodesets fall back to using frozen URLs).
        self._registry: Any = None

        # Shared-singleton auto-unload bookkeeping.
        # _shared_consumer_count: refcount of jobs (queued or running)
        # consuming each shared nodeset (= keys of spec's _shared_urls).
        # _shared_loaded_by_jobs: subset that the eval/start handler
        # freshly loaded into the parent registry to back a job; only
        # these are eligible for auto-unload on refcount-zero (canvas-Play
        # -loaded singletons stay untouched even when no job references them).
        self._shared_consumer_count: dict[str, int] = {}
        self._shared_loaded_by_jobs: set[str] = set()
        # Names queued for unload — drained by _drain_pending_unloads on
        # the next tick. Lets cancel() (sync) defer the actual unload to
        # the async tick loop without awaiting in the request handler.
        self._pending_unloads: set[str] = set()

        # Resolve backend dir + workspace root for child PYTHONPATH (matches
        # the existing _load_nodeset_as_server logic in registry.py).
        # Layout: backend/app/services/job_scheduler.py
        #   parent[0]=services, [1]=app, [2]=backend, [3]=agentcanvas, [4]=workspace_root
        here = Path(__file__).resolve()
        self._backend_dir = str(here.parents[2])
        self._workspace_root = str(here.parents[4])

        # Resource attribution + calibration — also the input to the
        # measured admission gate in _admit (estimate vs measured free,
        # per resource).
        self._resource_stats = ResourceStatsTracker(
            self._runs_dir.parent / "system" / "resource_calibration.json"
        )

    @classmethod
    def create(cls, eval_runs_dir: Path, backend_url: str) -> JobScheduler:
        """Build a scheduler with the usable-VRAM budget derived here (not in
        main.py) so all admission inputs live in this module: physical total
        via nvidia-smi minus ``SHARED_HEADROOM_MB``, or the
        ``AGENTCANVAS_USABLE_VRAM_MB`` env override verbatim.
        """
        total_vram = detect_total_vram_mb()
        usable = int(
            os.environ.get(
                "AGENTCANVAS_USABLE_VRAM_MB", max(total_vram - SHARED_HEADROOM_MB, 0)
            )
        )
        return cls(
            eval_runs_dir=eval_runs_dir,
            usable_vram_mb=usable,
            backend_url=backend_url,
            total_vram_mb=total_vram,
        )

    @property
    def usable_vram_mb(self) -> int:
        """The admission budget derived in ``create`` (read-only)."""
        return self._usable_vram_mb

    @property
    def runs_dir(self) -> Path:
        """This scheduler's eval-run pool (slot-dependent; see main.py)."""
        return self._runs_dir

    # ── Configuration ──

    def set_canvas_lock_callback(self, fn) -> None:
        """Backend wires this to ``ExecutionGuard.current()['mode'] == 'canvas'``
        so canvas Play preempts new admissions.
        """
        self._canvas_lock_held = fn

    def set_workspace_component_registry(self, registry: Any) -> None:
        """Wire the WorkspaceComponentRegistry handle used for TODO #60 ephemeral
        spawn. Called from main.py lifespan after the registry scan.
        """
        self._registry = registry

    # ── Public API ──

    def _fresh_run_id(self) -> str:
        """Second-precision timestamp run_id, e.g. ``20260515_143052``.

        Collision guard: two submissions in the same wall-clock second get
        ``_2``, ``_3``, … appended. run_id is the run dir name, so it must
        be unique on disk.
        """
        base = time.strftime("%Y%m%d_%H%M%S")
        run_id = base
        n = 2
        while (self._runs_dir / run_id).exists():
            run_id = f"{base}_{n}"
            n += 1
        return run_id

    def submit(self, spec: dict) -> str:
        """Validate spec, create run dir + spec.json + shared_urls.json,
        enqueue. Returns ``run_id``. Does not spawn — admission happens
        on the next tick.

        ``spec`` shape (see plan) ::

            {
              "run_id": "..." (optional; auto-generated if absent),
              "created_at": "...",
              "eval": {EvalConfig fields},
              "scheduling": {marginal_vram_mb, exclusive_gpu, priority, ...},
              "graph": {...}
            }
        """
        # P3 feasibility check: a declaration that exceeds the physical
        # ceiling can NEVER admit — reject loudly instead of the silent
        # forever-pending it used to produce.
        sched_req = spec.get("scheduling") or {}
        declared = int(sched_req.get("marginal_vram_mb", 0) or 0)
        ceiling = max(self._usable_vram_mb, self._total_vram_mb)
        if declared > ceiling:
            raise ValueError(
                f"marginal_vram_mb={declared} exceeds this machine's ceiling "
                f"({ceiling} MB) — the job could never be admitted"
            )

        run_id = spec.get("run_id") or self._fresh_run_id()
        spec["run_id"] = run_id
        spec.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_spec(run_dir, spec)

        # Backend computed shared URLs at submit time (caller of submit
        # populates spec["_shared_urls"]); pop it before persisting spec.
        shared_urls = spec.pop("_shared_urls", {}) or {}
        # eval/start also tags the spec with which shared singletons IT
        # freshly loaded into the parent registry (vs. ones canvas-Play
        # had already loaded). Only the freshly-loaded subset is eligible
        # for auto-unload on refcount-zero; canvas-Play singletons stay.
        shared_loaded_by_us = spec.pop("_shared_loaded_by_us", []) or []
        write_shared_urls(run_dir, shared_urls)

        # Bump consumer refcount for every shared singleton this job
        # references, and record which subset we're responsible for
        # tearing down once nobody else uses them.
        for name in shared_urls:
            self._shared_consumer_count[name] = self._shared_consumer_count.get(name, 0) + 1
        for name in shared_loaded_by_us:
            self._shared_loaded_by_jobs.add(name)

        # Initial summary so /runs/{id} immediately returns something
        # sensible even before admission.
        atomic_write_json(
            run_dir / "summary.json",
            {
                **initial_running_summary(run_id, spec.get("eval", {}), spec["created_at"]),
                "status": "pending",
            },
        )

        sched = spec.get("scheduling") or {}
        self._queue.append(
            _QueuedJob(
                run_id=run_id,
                spec=spec,
                submitted_at=time.time(),
                marginal_vram_mb=int(sched.get("marginal_vram_mb", 0) or 0),
                exclusive_gpu=bool(sched.get("exclusive_gpu", False)),
                priority=sched.get("priority", "normal"),
                shared_nodesets=tuple(shared_urls.keys()),
            )
        )
        log.info(
            "submit: run_id=%s marginal_vram=%d MB exclusive=%s queue=%d",
            run_id,
            int(sched.get("marginal_vram_mb", 0) or 0),
            bool(sched.get("exclusive_gpu", False)),
            len(self._queue),
        )
        return run_id

    def cancel(self, run_id: str) -> str:
        """Returns new status string. ``queued`` → ``cancelled`` (immediate);
        ``running`` → ``cancelling`` (SIGTERM sent; reap loop finalizes).
        """
        for q in self._queue:
            if q.run_id == run_id:
                self._queue.remove(q)
                run_dir = self._runs_dir / run_id
                summary = read_summary(run_dir) or {}
                summary["status"] = "cancelled"
                summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                atomic_write_json(run_dir / "summary.json", summary)
                (run_dir / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
                # Release shared-singleton refcount; running-job cancels
                # flow through _reap which handles the same decrement.
                # _release_shared_refs is fire-and-forget here because
                # cancel() is sync and unload may be needed off the
                # caller's stack — the actual unload happens on the next
                # tick via _maybe_unload_zero_refs.
                self._release_shared_refs(q.shared_nodesets)
                log.info("cancel: %s was queued, marked cancelled", run_id)
                return "cancelled"

        running = self._running.get(run_id)
        if running is not None:
            running.cancel_requested = True
            try:
                os.killpg(running.pgid, signal.SIGTERM)
                log.info("cancel: %s SIGTERM sent to pgid=%d", run_id, running.pgid)
            except ProcessLookupError:
                pass
            return "cancelling"

        return "unknown"

    def status(self, run_id: str) -> dict | None:
        run_dir = self._runs_dir / run_id
        summary = read_summary(run_dir)
        if summary is None:
            return None
        # Augment with scheduler view if active.
        running = self._running.get(run_id)
        in_queue = next((q for q in self._queue if q.run_id == run_id), None)
        scheduler_state = (
            "running" if running else ("queued" if in_queue else summary.get("status", "unknown"))
        )
        summary["scheduler_state"] = scheduler_state
        return summary

    def list_active(self) -> dict:
        return {
            "queued": [
                {
                    "run_id": q.run_id,
                    "marginal_vram_mb": q.marginal_vram_mb,
                    "exclusive_gpu": q.exclusive_gpu,
                    "priority": q.priority,
                    "submitted_at": q.submitted_at,
                }
                for q in self._queue
            ],
            "running": [
                {
                    "run_id": r.run_id,
                    "pid": r.proc.pid,
                    "marginal_vram_mb": r.marginal_vram_mb,
                    "admit_charges": dict(r.admit_charges),
                    "admit_basis": r.admit_basis,
                    "exclusive_gpu": r.exclusive_gpu,
                    "started_at": r.started_at,
                    "cancel_requested": r.cancel_requested,
                }
                for r in self._running.values()
            ],
            "usable_vram_mb": self._usable_vram_mb,
            "reserved_vram_mb": self._reserved_mb(),
            # Measured-gate view: what admission actually decides on,
            # per resource (None = unmeasurable on this machine).
            "measured_free_mb": {
                res: self._resource_stats.measured_free_mb(res) for res in RESOURCES
            },
            "pending_reservation_mb": self._pending_reservations(),
            # Latest measured attribution (observability; additive field).
            "attribution": self._resource_stats.snapshot(),
        }

    def estimate_run(
        self, graph_name: str, node_types: list[str], worker_count: int
    ) -> dict:
        """Advisory per-resource estimate for running ``graph_name`` at
        ``worker_count``.

        Resolves the graph's shared singletons (parallelism + loaded state
        + source hash) from the registry, then defers to the calibration-
        backed estimator. Read-only: loads nothing, reserves nothing.
        """
        needed = sorted({t.split("__")[0] for t in node_types if "__" in t})
        shared_infos: list[dict] = []
        for ns in needed:
            if self._registry is None:
                break
            try:
                if self._registry._get_parallelism(ns) != "shared":
                    continue
                ns_obj = getattr(self._registry, "_discovered_nodesets", {}).get(ns)
                shared_infos.append(
                    {
                        "name": ns,
                        "loaded": self._registry.is_nodeset_loaded(ns),
                        "source_hash": _source_hash(self._registry, ns),
                        # Author presets (BaseNodeSet.expected_vram_mb /
                        # expected_ram_mb) — estimator fallback when
                        # measurement is absent.
                        "hints": {
                            "vram": getattr(ns_obj, "expected_vram_mb", None),
                            "ram": getattr(ns_obj, "expected_ram_mb", None),
                        },
                    }
                )
            except Exception:
                # Unknown/broken nodeset → surface as uncalibrated, not a 500.
                shared_infos.append({"name": ns, "loaded": False, "source_hash": None})
        return self._resource_stats.estimate(graph_name, worker_count, shared_infos)

    # ── Tick loop (called every ~1s from lifespan) ──

    async def tick(self) -> None:
        # Observe before reaping so a finishing job's last sample still lands
        # in its window. Reads the sampler's in-memory ring — non-blocking.
        self._resource_stats.observe(
            jobs={rid: r.proc.pid for rid, r in self._running.items()},
            registry=self._registry,
        )
        await self._reap()
        await self._admit()
        await self._drain_pending_unloads()

    async def shutdown(self) -> None:
        """SIGTERM every running child; reap quickly. Best-effort — backend
        is going down so PR_SET_PDEATHSIG would catch stragglers anyway.
        """
        for r in list(self._running.values()):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(r.pgid, signal.SIGTERM)
        # brief grace period
        await asyncio.sleep(0.5)
        await self._reap()
        await self._drain_pending_unloads()

    # ── Internals ──

    def _reserved_mb(self) -> int:
        return sum(r.marginal_vram_mb for r in self._running.values())

    def _exclusive_gpu_held(self) -> bool:
        return any(r.exclusive_gpu for r in self._running.values())

    def _pending_reservations(self) -> dict[str, int]:
        """Per resource: sum over running jobs of ``max(0, admit charge -
        measured tree)``.

        A just-admitted job hasn't allocated yet, so the measured-free gate
        alone would over-admit during its load window; its reservation
        covers the gap and decays to zero as the sampler sees the real
        allocation (jobs_mb is ≤5s stale — the decay lags, which only errs
        conservative)."""
        per_res = self._resource_stats.snapshot().get("resources") or {}
        out: dict[str, int] = {}
        for res in RESOURCES:
            jobs_mb = (per_res.get(res) or {}).get("jobs_mb") or {}
            out[res] = sum(
                max(0, r.admit_charges.get(res, 0) - int(jobs_mb.get(rid, 0)))
                for rid, r in self._running.items()
            )
        return out

    def _resolve_estimate(self, q: _QueuedJob) -> tuple[dict[str, int] | None, str]:
        """Admission charges for a queued job, per resource, in preference
        order: calibrated estimate (full coverage on EVERY resource, margin
        included, current loaded state) → declared marginal_vram_mb (VRAM
        only; RAM was never declared, so it stays ungated on this rung) →
        None (nothing to go on)."""
        eval_cfg = q.spec.get("eval") or {}
        node_types = [
            n.get("type", "") for n in (q.spec.get("graph") or {}).get("nodes") or []
        ]
        try:
            result = self.estimate_run(
                eval_cfg.get("graph_name") or "",
                node_types,
                eval_cfg.get("worker_count") or 1,
            )
            resources = result.get("resources") or {}
            if resources and all(
                r["estimate_mb"] is not None for r in resources.values()
            ):
                # "hint" when any component priced by an author preset
                # (expected_vram_mb / expected_ram_mb), not a measurement.
                basis = (
                    "hint"
                    if any(r.get("used_hint") for r in resources.values())
                    else "calibrated"
                )
                return {res: int(r["estimate_mb"]) for res, r in resources.items()}, basis
        except Exception:
            log.exception("estimate failed for %s — falling back to declaration", q.run_id)
        if q.marginal_vram_mb > 0:
            return {"vram": q.marginal_vram_mb}, "declared"
        return None, "unknown"

    async def _admit(self) -> None:
        if self._canvas_lock_held():
            return  # canvas Play has the GPU

        # Measured gate: per resource, physically free (sampler) minus
        # standing reservations. The calibrated estimate already carries
        # its own safety margin (SAFETY_MARGIN_MB), so no extra hysteresis
        # here. A resource whose free is None is unmeasurable on this
        # machine (no GPU → vram) and its dimension is skipped. EVERY
        # resource unmeasurable (no sampler — e.g. unit tests) → the
        # legacy declared ledger.
        free = {res: self._resource_stats.measured_free_mb(res) for res in RESOURCES}
        measured_mode = any(v is not None for v in free.values())
        pending = self._pending_reservations() if measured_mode else {}

        # FIFO; M1 ignores priority. Walk a snapshot so we can pop admitted.
        admitted: list[str] = []
        for q in list(self._queue):
            if q.exclusive_gpu and self._running:
                continue
            if self._running and any(r.exclusive_gpu for r in self._running.values()):
                continue
            if measured_mode:
                charges, basis = self._resolve_estimate(q)
                if charges is None:
                    # No calibration and no declaration: cold-start
                    # discipline — admit only on an otherwise-idle
                    # scheduler; the first run doubles as calibration.
                    if self._running:
                        continue
                    charges, basis = {}, "cold-start"
                elif any(
                    free[res] is not None and mb > free[res] - pending.get(res, 0)
                    for res, mb in charges.items()
                ):
                    continue
            else:
                legacy_free = self._usable_vram_mb - self._reserved_mb()
                if q.marginal_vram_mb > legacy_free:
                    continue
                charges, basis = {"vram": q.marginal_vram_mb}, "legacy"
            ephem_tag: str | None = None
            try:
                # TODO #60: when an active_workspace overlay redefines a
                # shared nodeset's source, spawn a tagged ephemeral
                # auto_host child before spawning the eval subprocess so
                # the new child's URL ends up in shared_urls.json. The
                # frozen singleton stays alive for other sessions.
                ephem_tag = await self._prepare_ephemerals(q)
                self._spawn(q, ephem_tag=ephem_tag, admit_charges=charges, admit_basis=basis)
                # Newly admitted jobs hold their full reservation until the
                # sampler sees their allocation.
                for res, mb in charges.items():
                    pending[res] = pending.get(res, 0) + mb
                admitted.append(q.run_id)
            except Exception:
                log.exception("admit: spawn failed for %s", q.run_id)
                # If we partially spawned ephemerals, tear them down so
                # they don't squat on VRAM.
                if ephem_tag and self._registry is not None:
                    try:
                        self._registry.unload_nodeset_ephemeral(ephem_tag)
                    except Exception:
                        log.exception(
                            "admit: failed to clean up ephemerals for %s",
                            q.run_id,
                        )
                run_dir = self._runs_dir / q.run_id
                summary = read_summary(run_dir) or {}
                summary["status"] = "error"
                summary["error"] = "spawn failed; see backend log"
                summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                atomic_write_json(run_dir / "summary.json", summary)
                (run_dir / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
                # Release shared-singleton refcount: the job went queue
                # → error without ever entering _running, so _reap won't
                # see it; deferred unload happens on the next tick.
                self._release_shared_refs(q.shared_nodesets)
                admitted.append(q.run_id)  # remove from queue
        if admitted:
            self._queue = [q for q in self._queue if q.run_id not in admitted]

    async def _prepare_ephemerals(self, q: _QueuedJob) -> str | None:
        """TODO #60. For each shared nodeset in ``shared_urls.json`` whose
        overlay source content differs from frozen, spawn a tagged
        ephemeral auto_host child and rewrite the URL on disk before the
        eval subprocess starts.

        Returns the ephemeral tag (e.g. ``"ephem-20260515_141201"``) if
        any ephemerals were spawned, else ``None``. The tag is stable per
        run — all of a run's ephemerals share the same tag so
        ``unload_nodeset_ephemeral(tag)`` cleans them up in one call.
        """
        active_ws = q.spec.get("active_workspace_dir")
        if not active_ws:
            return None
        if self._registry is None:
            log.warning(
                "admit: active_workspace_dir set but no registry wired; "
                "overlay edits to shared nodesets will NOT take effect"
            )
            return None

        from ..components.content_hash import (
            hash_nodeset_tree,
            resolve_overlay_source,
        )

        run_dir = self._runs_dir / q.run_id
        shared_urls = read_shared_urls(run_dir)
        if not shared_urls:
            return None

        # resolve_overlay_source computes frozen_source.relative_to(ws_root)
        # and joins it onto active_workspace_dir. An iter's active_workspace/
        # mirrors the *contents* of the frozen workspace/ dir (it is rooted
        # at active_workspace/{graphs,nodesets}/), so ws_root must be the
        # frozen workspace/ dir itself — NOT self._workspace_root, which is
        # the repo root (parents[4], used for the eval subprocess PYTHONPATH).
        # Passing the repo root yielded a "workspace/" segment in the
        # relative path, so the overlay candidate never existed and TODO #60
        # silently no-op'd for every shared nodeset.
        ws_root = self._registry._frozen_dir
        rewritten: dict[str, str] = {}
        tag = f"ephem-{q.run_id}"
        for ns_name, _frozen_url in shared_urls.items():
            ns = self._registry._discovered_nodesets.get(ns_name)
            if ns is None:
                continue
            frozen_source = getattr(ns, "_source_file", None)
            if not frozen_source:
                continue
            try:
                overlay_source = resolve_overlay_source(frozen_source, ws_root, active_ws)
            except Exception:
                log.exception("admit: resolve_overlay_source failed for %s", ns_name)
                continue
            if overlay_source is None:
                continue
            try:
                frozen_hash = hash_nodeset_tree(frozen_source)
                overlay_hash = hash_nodeset_tree(overlay_source)
            except FileNotFoundError:
                log.exception("admit: source missing while hashing %s", ns_name)
                continue
            if overlay_hash == frozen_hash:
                continue
            log.info(
                "admit: %s overlay diff (frozen=%s overlay=%s) — spawning ephemeral",
                ns_name,
                frozen_hash[:8],
                overlay_hash[:8],
            )
            ephem_url = await self._registry.load_nodeset_ephemeral(
                ns_name, overlay_source, tag=tag
            )
            rewritten[ns_name] = ephem_url

        if rewritten:
            shared_urls.update(rewritten)
            write_shared_urls(run_dir, shared_urls)
            return tag
        return None

    def _spawn(
        self,
        q: _QueuedJob,
        ephem_tag: str | None = None,
        admit_charges: dict[str, int] | None = None,
        admit_basis: str = "legacy",
    ) -> None:
        run_dir = self._runs_dir / q.run_id
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{self._backend_dir}:{self._workspace_root}:{env.get('PYTHONPATH', '')}"
        )
        env["AGENTCANVAS_BACKEND_URL"] = self._backend_url
        # Quota teeth: the subprocess (and any child that honors the var)
        # caps its own torch allocator at the VRAM admission charge, so a
        # job that blows past its estimate OOMs itself instead of its
        # neighbors. Best-effort — EGL contexts and non-torch allocations
        # are uncapped. RAM has no equivalent (RLIMIT_AS counts virtual
        # address space and would kill mmap-heavy simulators; cgroups need
        # privileges) — the RAM dimension is gate-only.
        vram_charge = (admit_charges or {}).get("vram", 0)
        if vram_charge > 0:
            env["AGENTCANVAS_VRAM_CAP_MB"] = str(vram_charge)

        # Per-run active-workspace overlay: propagate to subprocess env so
        # its fresh Settings() picks up active_workspace_dir at construction,
        # which flows into WorkspaceComponentRegistry via state.py. None = no overlay.
        active_ws = q.spec.get("active_workspace_dir")
        if active_ws:
            env["ACTIVE_WORKSPACE_DIR"] = active_ws

        # Files outlive this function — they're held open by Popen for
        # the subprocess's lifetime. Closed in _reap when the child exits.
        cmd_log = open(run_dir / "stdout.log", "ab", buffering=0)  # noqa: SIM115
        err_log = open(run_dir / "stderr.log", "ab", buffering=0)  # noqa: SIM115
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.eval_subprocess_main",
                "--run-dir",
                str(run_dir),
                "--backend-url",
                self._backend_url,
            ],
            cwd=self._backend_dir,
            env=env,
            stdout=cmd_log,
            stderr=err_log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid → own pgid for clean cancel
        )
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            # Race: process already exited (most likely import error).
            pgid = proc.pid
        self._running[q.run_id] = _RunningJob(
            run_id=q.run_id,
            proc=proc,
            pgid=pgid,
            marginal_vram_mb=q.marginal_vram_mb,
            exclusive_gpu=q.exclusive_gpu,
            started_at=time.time(),
            admit_charges=dict(admit_charges or {}),
            admit_basis=admit_basis,
            log_files=(cmd_log, err_log),
            ephem_tag=ephem_tag,
            shared_nodesets=q.shared_nodesets,
        )
        eval_cfg = q.spec.get("eval") or {}
        self._resource_stats.note_job_started(
            q.run_id,
            graph_name=eval_cfg.get("graph_name") or "",
            worker_count=eval_cfg.get("worker_count") or 1,
            # Union of the server-backed singletons (shared_urls keys) and
            # the graph-derived shared set — the latter also catches LOCAL
            # in-process shared nodesets (e.g. env_adapter), which have no
            # server URL and would otherwise never enter the observation
            # window (and so never calibrate).
            shared_nodesets=tuple(
                sorted(set(q.shared_nodesets) | self._graph_shared_nodesets(q.spec))
            ),
        )
        log.info(
            "admit→spawn: run_id=%s pid=%d pgid=%d charges=%s MB (%s, declared=%d)",
            q.run_id,
            proc.pid,
            pgid,
            dict(admit_charges or {}),
            admit_basis,
            q.marginal_vram_mb,
        )

    def _graph_shared_nodesets(self, spec: dict) -> set[str]:
        """Shared nodesets needed by the spec's graph — same derivation as
        ``estimate_run`` (``__``-prefix scan + parallelism filter)."""
        nodes = (spec.get("graph") or {}).get("nodes") or []
        needed = {n.get("type", "").split("__")[0] for n in nodes if "__" in n.get("type", "")}
        out: set[str] = set()
        for ns in sorted(needed):
            if self._registry is None:
                break
            try:
                if self._registry._get_parallelism(ns) == "shared":
                    out.add(ns)
            except Exception:
                continue
        return out

    async def _reap(self) -> None:
        finished: list[str] = []
        for run_id, r in self._running.items():
            rc = r.proc.poll()
            if rc is None:
                continue
            # Close the stdout/stderr file handles we opened in _spawn.
            if r.log_files is not None:
                for fh in r.log_files:
                    with contextlib.suppress(OSError):
                        fh.close()
            run_dir = self._runs_dir / run_id
            if is_done(run_dir):
                summary = read_summary(run_dir) or {}
                final_status = summary.get("status", "completed")
                log.info("reap: %s rc=%d final=%s", run_id, rc, final_status)
            else:
                # PID gone but no _DONE — aborted (Q1).
                mark_aborted(run_dir, reason=f"subprocess exited rc={rc} without _DONE")
                (run_dir / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
                log.warning("reap: %s rc=%d aborted (no _DONE)", run_id, rc)
            # TODO #60: release the ephemeral auto_host children spawned
            # for this run. Cancel (queued) goes through cancel() and
            # never reaches _reap; cancel (running) sends SIGTERM →
            # subprocess exits → _reap fires → ephemerals released here.
            if r.ephem_tag and self._registry is not None:
                try:
                    n = self._registry.unload_nodeset_ephemeral(r.ephem_tag)
                    if n:
                        log.info(
                            "reap: %s released %d ephemeral subprocess(es) (tag=%s)",
                            run_id,
                            n,
                            r.ephem_tag,
                        )
                except Exception:
                    log.exception(
                        "reap: failed to release ephemerals for %s (tag=%s)",
                        run_id,
                        r.ephem_tag,
                    )
            # Release shared-singleton refcount; nodesets that hit zero
            # land in _pending_unloads and get torn down by
            # _drain_pending_unloads on this same tick.
            self._release_shared_refs(r.shared_nodesets)
            # Sediment this job's resource observation window into calibration.
            self._resource_stats.note_job_finished(run_id, self._registry)
            finished.append(run_id)
        for run_id in finished:
            del self._running[run_id]

    def _release_shared_refs(self, names: tuple[str, ...] | list[str]) -> None:
        """Decrement the consumer refcount for each shared singleton this
        job referenced; queue zero-count names for unload if we loaded
        them on the job's behalf.
        """
        for name in names:
            count = self._shared_consumer_count.get(name, 0) - 1
            if count <= 0:
                self._shared_consumer_count.pop(name, None)
                if name in self._shared_loaded_by_jobs:
                    self._pending_unloads.add(name)
            else:
                self._shared_consumer_count[name] = count

    async def _drain_pending_unloads(self) -> None:
        """Unload every shared singleton in ``_pending_unloads`` that is
        still unreferenced. A nodeset can re-enter the consumer count
        between release and drain (a new job submitted in the same
        second), so we re-check the count under the drain.
        """
        if self._registry is None:
            self._pending_unloads.clear()
            return
        if not self._pending_unloads:
            return
        for name in list(self._pending_unloads):
            self._pending_unloads.discard(name)
            if self._shared_consumer_count.get(name, 0) > 0:
                continue  # a fresh job grabbed it before we got here
            if name not in self._shared_loaded_by_jobs:
                continue  # canvas-Play took ownership in between, leave it
            try:
                await self._registry.unload_nodeset(name)
                self._shared_loaded_by_jobs.discard(name)
                log.info(
                    "auto-unloaded shared nodeset %s (no remaining job references it)",
                    name,
                )
            except Exception:
                log.exception("auto-unload failed for shared nodeset %s", name)


# ── Helpers used by main.py at startup ──


def detect_total_vram_mb() -> int:
    """Sum of memory.total across all visible NVIDIA GPUs, in MB.

    Returns 0 if nvidia-smi is unavailable (CPU-only host) — admission
    then refuses any job declaring marginal_vram_mb > 0.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0
    total = 0
    for line in out.strip().splitlines():
        try:
            total += int(line.strip())
        except ValueError:
            continue
    return total


def reconcile_aborted_runs(eval_runs_dir: Path) -> int:
    """At backend startup, mark any persisted run with ``status='running'``
    but no ``_DONE`` as ``aborted`` (per Q1). Returns count fixed.

    Queue rows (status='pending') are NOT touched — those stay queued so
    they resume on next admission tick, except their PID is gone, so the
    user must re-submit. M1: just mark as aborted too. M3 might add real
    queue durability.
    """
    if not eval_runs_dir.exists():
        return 0
    fixed = 0
    for sub in eval_runs_dir.iterdir():
        if not sub.is_dir():
            continue
        if (sub / "_DONE").exists():
            continue
        summary = read_summary(sub)
        if summary is None:
            continue
        if summary.get("status") in {"running", "pending"}:
            mark_aborted(sub, reason="parent backend restart")
            (sub / "_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            fixed += 1
    return fixed
