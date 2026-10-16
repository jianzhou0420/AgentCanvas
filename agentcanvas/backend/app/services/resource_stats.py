"""ResourceStatsTracker — per-owner resource attribution + calibration store.

Generalization of the VRAM-only P1-P3 pipeline to multiple resources
(currently VRAM + system RAM): observe real per-PID usage during normal
eval runs, attribute it to owners, and sediment per-nodeset / per-graph
peak statistics into a calibration file that the estimator and the
admission gate consume. One observation window per job covers every
resource at once, so calibration coverage converges together.

Attribution (per JobScheduler tick; VRAM from ResourceSampler's latest
in-memory sample, RAM from one /proc walk — no extra nvidia-smi calls):

- **shared nodeset server**: PID (or an ancestor) matches a ``BaseServer``
  in the parent registry (``server_pids()``).
- **running job**: PID's PPID-chain reaches a running job's subprocess
  PID. PPID walk, not pgid match — replicated auto_host children setsid
  into their own process groups (base_server.py) but keep their parent
  link, so the ancestor chain still finds the owning run.
- **external**: everything else (Xorg, user's bare processes, the OS).
  Not charged to any job, but part of measured usage — the gate sees it
  through measured free.

RAM caveat: per-process RSS double-counts copy-on-write pages shared
across a forked worker tree, so job-tree RAM peaks over-estimate. That
errs conservative for admission; PSS would be exact but costs a
/proc/<pid>/smaps_rollup read per process per tick.

Sedimentation (on job reap): per shared nodeset and per graph tree, one
peak per resource, keyed with the nodeset's source tree hash so code
edits invalidate old stats (falls back to cold-start).

Store: ``outputs/system/resource_calibration.json`` (EWMA + observed max
per resource, atomic writes; gitignored alongside the sampler stream).
Legacy v1 files (``vram_calibration.json``, VRAM-only) migrate in place;
their RAM side starts empty and refills on the next run per graph.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from ..components.facts import NodesetFacts, RegistryFacts
from ..graph_def import nodeset_names_from_types
from .run_state_io import atomic_write_json

log = logging.getLogger("agentcanvas.resource-stats")

# The gated dimensions. Adding one = a measured-free source + a per-PID
# usage source below; everything downstream (windows, calibration,
# estimator, gate) is keyed by these names.
RESOURCES = ("vram", "ram")

_EWMA_ALPHA = 0.3
_MAX_WORKER_OBS = 32
_MAX_ANCESTOR_DEPTH = 24
_STORE_VERSION = 2
_LEGACY_STORE_NAME = "vram_calibration.json"

# Estimator: advisory margin on top of calibrated peaks (fragmentation,
# transient allocation spikes between 5s attribution samples). RAM's is
# smaller because the hard floor below already buffers it.
SAFETY_MARGIN_MB = {"vram": 1500, "ram": 1024}
# RAM free is MemAvailable — itself an estimate (reclaimable page cache,
# overcommit). Keep this floor out of the admission budget so the OS
# never gets squeezed into the swap death spiral.
RAM_FLOOR_MB = 4096
_MAX_WORKERS_PROBE = 64


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _ewma(prev: float | None, value: float) -> float:
    if prev is None:
        return float(value)
    return (1 - _EWMA_ALPHA) * float(prev) + _EWMA_ALPHA * float(value)


@dataclass
class _JobWindow:
    """Rolling per-job observation window (spawn → reap)."""

    graph_name: str
    worker_count: int
    shared_nodesets: tuple[str, ...]
    started_at: float
    # resource → peak MB of the job's whole subprocess tree.
    tree_peaks: dict[str, int] = field(default_factory=dict)
    # attribution samples that actually covered this window — 0 means we
    # never saw the machine during the run (sampler down / sub-5s job), in
    # which case sedimenting would write false zeros.
    samples_seen: int = 0
    # resource → {nodeset: peak MB} on each referenced shared nodeset's
    # server during this window. A 0 entry is a real measurement ("server
    # resident, none of this resource") — recorded only when the server's
    # PID was visible in the registry.
    shared_peaks: dict[str, dict[str, int]] = field(default_factory=dict)
    # max concurrent worker pressure on each shared nodeset during this window
    shared_workers: dict[str, int] = field(default_factory=dict)
    # content key of the graph (node types + nodeset source hashes) this
    # window observed; the graphs calibration is gated on it.
    graph_hash: str | None = None


@dataclass
class Charge:
    """What admission charges one job, per resource, and on what basis.

    ``per_resource`` is empty for ``"cold-start"`` (nothing to go on: the
    first run doubles as calibration and admits only on an idle
    scheduler). ``estimate`` is the full estimator result when one was
    computed, for observability.
    """

    per_resource: dict[str, int]
    basis: str  # "calibrated" | "hint" | "declared" | "cold-start" | "legacy"
    estimate: dict[str, Any] | None = None

    @property
    def cold_start(self) -> bool:
        return self.basis == "cold-start"


class CalibrationStore:
    """EWMA + max statistics per nodeset and per graph, per resource,
    persisted as JSON.

    Load-once, mutate in memory, atomic-write on save. Corrupt or missing
    files degrade to an empty store — calibration is advisory data, never
    load-bearing for correctness.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = self._load()

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            import json

            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _load(self) -> dict[str, Any]:
        data = self._read(self._path)
        if data is None:
            # First run after the rename: adopt the VRAM-only v1 file.
            data = self._read(self._path.parent / _LEGACY_STORE_NAME)
        if data is not None:
            if data.get("version") == _STORE_VERSION:
                return data
            if data.get("version") == 1:
                return _migrate_v1(data)
        return {"version": _STORE_VERSION, "nodesets": {}, "graphs": {}}

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def update_nodeset(
        self,
        name: str,
        peaks: dict[str, int],
        workers: int,
        source_hash: str | None,
    ) -> None:
        entry = self._data["nodesets"].get(name)
        # Source changed (or first sighting) → discard stale statistics
        # for every resource at once.
        if entry is None or (source_hash and entry.get("source_hash") != source_hash):
            entry = {"source_hash": source_hash, "n_obs": 0}
        entry["n_obs"] = int(entry.get("n_obs", 0)) + 1
        for res, peak in peaks.items():
            sub = entry.get(res) or {"worker_obs": []}
            sub["peak_mb_ewma"] = round(_ewma(sub.get("peak_mb_ewma"), peak), 1)
            sub["peak_mb_max"] = max(int(sub.get("peak_mb_max", 0)), int(peak))
            obs = sub.get("worker_obs", [])
            obs.append([int(workers), int(peak)])
            sub["worker_obs"] = obs[-_MAX_WORKER_OBS:]
            entry[res] = sub
        entry["updated_at"] = _now()
        self._data["nodesets"][name] = entry

    def update_graph(
        self,
        graph_name: str,
        worker_count: int,
        tree_peaks: dict[str, int],
        graph_hash: str | None = None,
    ) -> None:
        entry = self._data["graphs"].get(graph_name)
        # Graph content changed (node types or a nodeset's source) → the
        # old tree peaks describe a different graph; discard them. A
        # window that carried no hash leaves whatever is there alone.
        if entry is None or (graph_hash and entry.get("graph_hash") not in (None, graph_hash)):
            entry = {"by_worker_count": {}}
            self._data["graphs"][graph_name] = entry
        if graph_hash:
            entry["graph_hash"] = graph_hash
        wc_key = str(int(worker_count))
        wc_entry = entry["by_worker_count"].get(wc_key, {"n_obs": 0})
        wc_entry["n_obs"] = int(wc_entry.get("n_obs", 0)) + 1
        for res, peak in tree_peaks.items():
            sub = wc_entry.get(res) or {}
            sub["tree_peak_mb_ewma"] = round(_ewma(sub.get("tree_peak_mb_ewma"), peak), 1)
            sub["tree_peak_mb_max"] = max(int(sub.get("tree_peak_mb_max", 0)), int(peak))
            wc_entry[res] = sub
        entry["by_worker_count"][wc_key] = wc_entry
        entry["updated_at"] = _now()

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._path, self._data)
        except OSError as e:
            log.warning("calibration store: cannot write %s: %s", self._path, e)


def _migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    """v1 (flat, VRAM-only) → v2 (per-resource sub-objects). The RAM side
    starts empty — estimates stay null until one run per graph refills it."""
    out: dict[str, Any] = {"version": _STORE_VERSION, "nodesets": {}, "graphs": {}}
    for name, e in (data.get("nodesets") or {}).items():
        sub = {k: e[k] for k in ("peak_mb_ewma", "peak_mb_max", "worker_obs") if k in e}
        out["nodesets"][name] = {
            "source_hash": e.get("source_hash"),
            "n_obs": e.get("n_obs", 0),
            "updated_at": e.get("updated_at"),
            "vram": sub,
        }
    for graph, e in (data.get("graphs") or {}).items():
        by_wc: dict[str, Any] = {}
        for wc_key, we in (e.get("by_worker_count") or {}).items():
            sub = {k: we[k] for k in ("tree_peak_mb_ewma", "tree_peak_mb_max") if k in we}
            by_wc[wc_key] = {"n_obs": we.get("n_obs", 0), "vram": sub}
        out["graphs"][graph] = {"by_worker_count": by_wc, "updated_at": e.get("updated_at")}
    return out


class ResourceStatsTracker:
    """Owned by JobScheduler; all hooks are cheap and never raise.

    Owns the whole "what does (graph, worker_count) cost" ladder —
    :meth:`charge` — and the attribution + calibration it rests on. The
    registry facts it needs (parallelism, loaded state, source hash,
    author presets, server PIDs) come through the read-only
    :class:`RegistryFacts` seam set via :meth:`set_facts`.

    Call order per scheduler lifecycle:

    - ``note_job_started(run_id, ...)``   — from ``_spawn``
    - ``observe(jobs)``                   — from ``tick()``, every ~1s
    - ``note_job_finished(run_id)``       — from ``_reap``
    """

    def __init__(self, calibration_path: Path, facts: RegistryFacts | None = None) -> None:
        self._store = CalibrationStore(calibration_path)
        self._facts: RegistryFacts | None = facts
        self._windows: dict[str, _JobWindow] = {}
        # Latest attribution snapshot (observability; served via /queue).
        self._last: dict[str, Any] = {}

    @property
    def facts(self) -> RegistryFacts | None:
        return self._facts

    def set_facts(self, facts: RegistryFacts | None) -> None:
        """Wire the registry view (from main.py lifespan, after the scan)."""
        self._facts = facts

    # ── graph-derived nodeset sets (one derivation for every caller) ──

    def nodesets_of(self, node_types: list[str]) -> list[str]:
        """Nodesets a graph's node types belong to, sorted."""
        return sorted(nodeset_names_from_types(node_types))

    def shared_nodesets_of(self, node_types: list[str]) -> set[str]:
        """The ``parallelism="shared"`` subset — singletons the parent
        backend hosts for the job. Empty without registry facts."""
        if self._facts is None:
            return set()
        out: set[str] = set()
        for name in self.nodesets_of(node_types):
            try:
                if self._facts.nodeset_facts(name).parallelism == "shared":
                    out.add(name)
            except Exception:
                continue
        return out

    def _infos_for(self, node_types: list[str]) -> list[dict[str, Any]]:
        """One ``{name, parallelism, loaded, source_hash, hints}`` per
        nodeset in the graph. Without facts there is nothing to price."""
        if self._facts is None:
            return []
        infos: list[dict[str, Any]] = []
        for name in self.nodesets_of(node_types):
            try:
                f = self._facts.nodeset_facts(name)
                infos.append(
                    {
                        "name": name,
                        "parallelism": f.parallelism,
                        "loaded": f.loaded,
                        "source_hash": f.source_hash,
                        "hints": f.hints,
                    }
                )
            except Exception:
                # Unknown/broken nodeset → surface as uncalibrated, never raise.
                infos.append({"name": name, "loaded": False, "source_hash": None})
        return infos

    def _graph_hash(self, node_types: list[str] | None) -> str | None:
        """Content key of a graph for calibration gating: its node types
        plus the source hash of every nodeset it uses. None without node
        types (a legacy caller that only knows the graph name)."""
        if node_types is None:
            return None
        h = hashlib.sha256()
        for t in sorted(set(node_types)):
            h.update(t.encode())
            h.update(b"\0")
        for name in self.nodesets_of(node_types):
            src = None
            if self._facts is not None:
                try:
                    src = self._facts.nodeset_facts(name).source_hash
                except Exception:
                    src = None
            h.update(f"{name}={src or '?'}\n".encode())
        return h.hexdigest()

    def _facts_of(self, name: str) -> NodesetFacts:
        if self._facts is None:
            return NodesetFacts(name=name, parallelism="shared")
        try:
            return self._facts.nodeset_facts(name)
        except Exception:
            return NodesetFacts(name=name, parallelism="shared")

    # ── lifecycle hooks (called by JobScheduler) ──

    def note_job_started(
        self,
        run_id: str,
        graph_name: str,
        worker_count: int,
        shared_nodesets: tuple[str, ...],
        node_types: list[str] | None = None,
    ) -> None:
        self._windows[run_id] = _JobWindow(
            graph_name=graph_name or "unknown",
            worker_count=max(1, int(worker_count or 1)),
            shared_nodesets=tuple(shared_nodesets),
            started_at=time.time(),
            graph_hash=self._graph_hash(node_types),
        )

    def observe(self, jobs: dict[str, int]) -> None:
        """Attribute the latest sample. ``jobs`` maps run_id → root PID."""
        try:
            self._observe(jobs)
        except Exception:  # observability must never break the scheduler
            log.exception("resource observe failed")

    def note_job_finished(self, run_id: str) -> None:
        """Sediment the job's window into the calibration store."""
        window = self._windows.pop(run_id, None)
        if window is None:
            return
        if window.samples_seen == 0:
            log.info("resource window for %s had no samples — nothing to sediment", run_id)
            return
        # Shared nodesets loaded LOCAL (in-process, no auto_host server —
        # e.g. env_adapter) have no PID of their own: their footprint rides
        # the parent backend process. Calibrate them at 0 marginal (every
        # resource) so they never block estimate coverage. (A local nodeset
        # that secretly loads a model into the backend would be
        # under-estimated — that's an anti-pattern the attribution can't
        # see; documented limitation.)
        for res in RESOURCES:
            peaks = window.shared_peaks.setdefault(res, {})
            for name in window.shared_nodesets:
                if name not in peaks and self._facts_of(name).loaded_local:
                    peaks[name] = 0
        try:
            # 0 is a legitimate measurement (CPU-only / API-only graph);
            # samples_seen above guards against writing blind zeros.
            self._store.update_graph(
                window.graph_name,
                window.worker_count,
                {res: window.tree_peaks.get(res, 0) for res in RESOURCES},
                graph_hash=window.graph_hash,
            )
            names: set[str] = set()
            for res in RESOURCES:
                names |= set(window.shared_peaks.get(res, {}))
            for name in sorted(names):
                peaks = {
                    res: window.shared_peaks[res][name]
                    for res in RESOURCES
                    if name in window.shared_peaks.get(res, {})
                }
                self._store.update_nodeset(
                    name,
                    peaks=peaks,
                    workers=window.shared_workers.get(name, window.worker_count),
                    source_hash=self._facts_of(name).source_hash,
                )
            self._store.save()
        except Exception:
            log.exception("resource sediment failed for %s", run_id)

    def snapshot(self) -> dict[str, Any]:
        """Latest attribution split, for /queue observability."""
        return dict(self._last)

    # ── the charge ladder ──

    def estimate(
        self,
        graph_name: str,
        worker_count: int,
        shared_infos: list[dict[str, Any]] | None = None,
        *,
        node_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Advisory per-resource estimate from calibration. Read-only.

        Pass ``node_types`` and the nodeset infos are resolved through
        the registry facts — every nodeset the graph uses, shared and
        replicated. ``shared_infos`` (``{name, loaded, source_hash,
        hints}`` per singleton) is the pre-resolved form for callers
        without a registry. The job's own subprocess tree (replicated
        servers + in-process models) comes from the ``graphs``
        calibration, or — when that is missing — from the replicated
        nodesets' author presets at ``worker_count`` copies each.

        Honesty rule, per resource: ``estimate_mb`` is null unless EVERY
        component is calibrated — a missing singleton could be 10 GB, so a
        partial sum would lowball. Known parts are still reported via
        ``known_mb`` + ``breakdown``, gaps via ``uncalibrated``.

        ``max_workers`` (top-level) is the min across resources that are
        measurable on this machine; per-resource values sit inside
        ``resources[r]``.
        """
        wc = max(1, int(worker_count))
        if node_types is not None:
            infos = self._infos_for(node_types)
            graph_hash = self._graph_hash(node_types)
        else:
            infos = list(shared_infos or [])
            graph_hash = None
        data = self._store.data
        resources: dict[str, Any] = {}
        for res in RESOURCES:
            result = _estimate_once(data, graph_name, wc, infos, res, graph_hash)
            free = self.measured_free_mb(res)
            max_workers: int | None = None
            if result["estimate_mb"] is not None and free is not None:
                max_workers = 0
                for probe in range(1, _MAX_WORKERS_PROBE + 1):
                    est = _estimate_once(data, graph_name, probe, infos, res, graph_hash)[
                        "estimate_mb"
                    ]
                    if est is None or est > free:
                        break
                    max_workers = probe
            resources[res] = {
                **result,
                "safety_margin_mb": SAFETY_MARGIN_MB[res],
                "measured_free_mb": free,
                "max_workers": max_workers,
            }

        gated = [
            r["max_workers"] for r in resources.values() if r["measured_free_mb"] is not None
        ]
        combined = min(gated) if gated and all(m is not None for m in gated) else None
        return {
            "graph_name": graph_name,
            "worker_count": wc,
            "resources": resources,
            "max_workers": combined,
        }

    def charge(
        self,
        graph_name: str,
        node_types: list[str],
        worker_count: int,
        declared_vram_mb: int = 0,
        *,
        measured: bool = True,
    ) -> Charge:
        """Admission charges for one job, per resource — the one ladder.

        ``measured`` is whether the gate runs on sampler-measured free
        (any resource measurable). Rungs, in preference order:

        1. calibrated estimate — full coverage on EVERY resource, margin
           included, current loaded state (``"calibrated"``, or
           ``"hint"`` when an author preset priced any component);
        2. the declared ``marginal_vram_mb`` (VRAM only; RAM was never
           declared, so it stays ungated on this rung) — ``"declared"``;
        3. nothing to go on — ``"cold-start"`` (admit only alone).

        Without a sampler the gate is the legacy declared ledger:
        ``"legacy"`` with the declaration verbatim.
        """
        declared = int(declared_vram_mb or 0)
        if not measured:
            return Charge({"vram": declared}, "legacy")
        try:
            result = self.estimate(graph_name, worker_count, node_types=node_types)
        except Exception:
            log.exception("estimate failed for %s — falling back to declaration", graph_name)
            result = None
        if result is not None:
            resources = result.get("resources") or {}
            if resources and all(r["estimate_mb"] is not None for r in resources.values()):
                basis = (
                    "hint"
                    if any(r.get("used_hint") for r in resources.values())
                    else "calibrated"
                )
                return Charge(
                    {res: int(r["estimate_mb"]) for res, r in resources.items()}, basis, result
                )
        if declared > 0:
            return Charge({"vram": declared}, "declared", result)
        return Charge({}, "cold-start", result)

    def measured_free_mb(self, resource: str = "vram") -> int | None:
        """Physically free MB of ``resource`` right now (sampler ring, ≤1s
        stale). None when unmeasurable on this machine (no sampler; or no
        GPU for vram) — the gate then skips that dimension.

        RAM free is MemAvailable minus ``RAM_FLOOR_MB``: the floor stays
        out of the admission budget permanently."""
        sample = _latest_sample()
        if not sample:
            return None
        if resource == "vram":
            gpus = sample.get("gpus") or []
            if not gpus:
                return None
            total = sum(g.get("mem_total_mb", 0) for g in gpus)
            used = sum(g.get("mem_used_mb", 0) for g in gpus)
            return max(0, total - used)
        if resource == "ram":
            total = int(sample.get("mem_total_mb") or 0)
            if total <= 0:
                return None
            used = int(sample.get("mem_used_mb") or 0)
            return max(0, total - used - RAM_FLOOR_MB)
        return None

    # ── internals ──

    def _observe(self, jobs: dict[str, int]) -> None:
        sample = _latest_sample()
        if not sample:
            return
        procs_table = _proc_table()
        parents = {pid: pp for pid, (pp, _) in procs_table.items()}
        server_pids = _shared_server_pids(self._facts)

        proc_lists: dict[str, list[tuple[int, int]]] = {
            "vram": [
                (p["pid"], int(p.get("mem_mb", 0)))
                for p in sample.get("gpu_procs") or []
                if p.get("pid") is not None
            ],
            "ram": [(pid, rss) for pid, (_, rss) in procs_table.items() if rss > 0],
        }
        used_totals = {
            "vram": sum(g.get("mem_used_mb", 0) for g in sample.get("gpus") or []),
            "ram": int(sample.get("mem_used_mb") or 0),
        }

        per_res: dict[str, Any] = {}
        for res in RESOURCES:
            shared_mb, job_mb, attributed = _attribute(
                proc_lists[res], server_pids, jobs, parents
            )
            per_res[res] = {
                "used_total_mb": used_totals[res],
                "attributed_mb": attributed,
                "external_mb": max(0, used_totals[res] - attributed),
                "shared_mb": shared_mb,
                "jobs_mb": job_mb,
            }

        # Max-update the rolling windows.
        servers_present = set(server_pids.values())
        for window in self._windows.values():
            window.samples_seen += 1
            pressure = sum(
                w.worker_count
                for w in self._windows.values()
                if w is window or set(w.shared_nodesets) & set(window.shared_nodesets)
            )
            for name in window.shared_nodesets:
                if pressure > window.shared_workers.get(name, 0):
                    window.shared_workers[name] = pressure
        for run_id, window in self._windows.items():
            for res in RESOURCES:
                window.tree_peaks[res] = max(
                    window.tree_peaks.get(res, 0), per_res[res]["jobs_mb"].get(run_id, 0)
                )
                peaks = window.shared_peaks.setdefault(res, {})
                for name in window.shared_nodesets:
                    mem = per_res[res]["shared_mb"].get(name, 0)
                    # Record only when the server is actually resident in
                    # the registry — then even 0 is a real "none of this
                    # resource" measurement.
                    if name in servers_present or mem > 0:
                        peaks[name] = max(peaks.get(name, 0), mem)

        self._last = {"sampled_at": sample.get("ts"), "resources": per_res}


# ── estimator (pure functions over calibration data) ──


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares ``y = a + b*x``; requires ≥2 distinct x values."""
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    denom = sum((p[0] - mean_x) ** 2 for p in points)
    b = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denom
    a = mean_y - b * mean_x
    return a, b


def _nodeset_view(entry: dict[str, Any] | None, resource: str) -> dict[str, Any] | None:
    """Flatten one resource's slice of a v2 nodeset entry into the shape
    ``estimate_shared_mb`` consumes ({source_hash, peak_mb_max, worker_obs}).
    None when this resource was never calibrated (e.g. migrated v1 RAM)."""
    if not entry:
        return None
    sub = entry.get(resource)
    if sub is None:
        return None
    return {"source_hash": entry.get("source_hash"), **sub}


def _graph_view(
    entry: dict[str, Any] | None, resource: str, graph_hash: str | None = None
) -> dict[str, Any] | None:
    """Same flattening for a graphs entry (by_worker_count sub-objects).
    None when the entry was calibrated for a different graph content
    (``graph_hash`` mismatch) — stale peaks must not price a new graph."""
    if not entry:
        return None
    if graph_hash and entry.get("graph_hash") not in (None, graph_hash):
        return None
    out: dict[str, Any] = {}
    for wc_key, wc_entry in (entry.get("by_worker_count") or {}).items():
        sub = wc_entry.get(resource)
        if sub is not None:
            out[wc_key] = sub
    return {"by_worker_count": out} if out else None


def estimate_tree_mb(graph_entry: dict[str, Any] | None, wc: int) -> tuple[int, str] | None:
    """The job's subprocess tree cost (replicated servers + in-process
    models) at ``wc`` workers, from one resource's ``graphs`` view.

    Basis: ``measured`` (exact worker_count seen before) → ``fitted``
    (linear fit over ≥2 distinct worker counts) → ``scaled`` (single
    point, proportional — the tree is dominated by per-worker sims, so
    through-origin is the sane one-point extrapolation) → None.
    """
    if not graph_entry:
        return None
    by_wc = graph_entry.get("by_worker_count") or {}
    points: list[tuple[float, float]] = []
    for wc_key, entry in by_wc.items():
        try:
            peak = int(entry.get("tree_peak_mb_max", 0))
            points.append((float(int(wc_key)), float(peak)))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    exact = dict(points).get(float(wc))
    if exact is not None:
        return int(exact), "measured"
    if len({p[0] for p in points}) >= 2:
        a, b = _fit_line(points)
        return max(0, round(a + max(0.0, b) * wc)), "fitted"
    wc0, peak0 = points[0]
    return round(peak0 / wc0 * wc), "scaled"


def estimate_shared_mb(
    ns_entry: dict[str, Any] | None,
    wc: int,
    loaded: bool,
    current_hash: str | None,
    hint_mb: int | None = None,
) -> tuple[int, str] | None:
    """A shared singleton's marginal cost at ``wc`` concurrent workers,
    for one resource (``ns_entry`` is that resource's flattened view).

    Peak model: observed max, plus a worker-pressure slope when the
    calibration has ≥2 distinct worker contexts. Already-loaded
    singletons only cost their predicted growth beyond the known base
    (their residency is already inside measured usage). Stale source
    hash → measured stats retire, but the author preset (``hint_mb``,
    from ``BaseNodeSet.expected_vram_mb`` / ``expected_ram_mb``) still
    applies — presets are long-term assertions, not hash-gated. Nothing
    at all → None.
    """
    if not ns_entry or (
        current_hash and ns_entry.get("source_hash") not in (None, current_hash)
    ):
        if hint_mb is None:
            return None
        # Loaded residency is already inside measured usage — preset only
        # prices the not-yet-loaded case (no slope knowledge in a preset).
        return (0, "hint, loaded") if loaded else (int(hint_mb), "hint")
    base = int(ns_entry.get("peak_mb_max", 0))
    predicted = float(base)
    basis = "measured"
    points = [
        (float(w), float(peak))
        for w, peak in (ns_entry.get("worker_obs") or [])
        if isinstance(w, (int, float)) and isinstance(peak, (int, float))
    ]
    if len({p[0] for p in points}) >= 2:
        a, b = _fit_line(points)
        predicted = max(float(base), a + max(0.0, b) * wc)
        basis = "fitted"
    if loaded:
        return max(0, round(predicted - base)), f"{basis}, loaded"
    return round(predicted), basis


def _estimate_once(
    data: dict[str, Any],
    graph_name: str,
    wc: int,
    infos: list[dict[str, Any]],
    resource: str,
    graph_hash: str | None = None,
) -> dict[str, Any]:
    """One resource's estimate at a fixed worker count. See
    ``ResourceStatsTracker.estimate``.

    ``infos`` carry an optional ``parallelism``; entries without one are
    shared singletons (the pre-resolved caller form). Replicated nodesets
    are part of the job's own tree: priced by the graph calibration when
    there is one, else by their author presets x ``wc`` copies."""
    breakdown: dict[str, Any] = {"shared": {}, "tree": None}
    uncalibrated: list[str] = []
    known = 0
    used_hint = False

    shared_infos = [i for i in infos if i.get("parallelism", "shared") == "shared"]
    replicated_infos = [i for i in infos if i.get("parallelism") == "replicated"]

    tree = estimate_tree_mb(
        _graph_view((data.get("graphs") or {}).get(graph_name), resource, graph_hash), wc
    )
    if tree is None and replicated_infos:
        hints = [(i.get("hints") or {}).get(resource) for i in replicated_infos]
        if all(h is not None for h in hints):
            # Every replicated copy carries its preset; N workers = N copies.
            tree = (int(sum(hints)) * wc, "hint")
            used_hint = True
    if tree is None:
        uncalibrated.append("graph-tree")
    else:
        known += tree[0]
        breakdown["tree"] = {"mb": tree[0], "basis": tree[1]}

    for info in shared_infos:
        name = info["name"]
        est = estimate_shared_mb(
            _nodeset_view((data.get("nodesets") or {}).get(name), resource),
            wc,
            loaded=bool(info.get("loaded")),
            current_hash=info.get("source_hash"),
            hint_mb=(info.get("hints") or {}).get(resource),
        )
        if est is None:
            uncalibrated.append(name)
        else:
            known += est[0]
            used_hint = used_hint or est[1].startswith("hint")
            breakdown["shared"][name] = {"mb": est[0], "basis": est[1]}

    return {
        "estimate_mb": known + SAFETY_MARGIN_MB[resource] if not uncalibrated else None,
        "known_mb": known,
        "breakdown": breakdown,
        "uncalibrated": uncalibrated,
        "used_hint": used_hint,
    }


# ── module helpers (pure functions; injectable in tests) ──


def _latest_sample() -> dict[str, Any] | None:
    """Non-blocking read of ResourceSampler's in-memory ring."""
    from .resource_sampler import get_sampler

    sampler = get_sampler()
    return sampler.latest() if sampler is not None else None


def _shared_server_pids(facts: RegistryFacts | None) -> dict[int, str]:
    """PID → nodeset name for auto_host servers owned by THIS process's
    registry (tagged copies collapse to the plain nodeset name)."""
    if facts is None:
        return {}
    return dict(facts.server_pids())


def _proc_table() -> dict[int, tuple[int, int]]:
    """One /proc walk: PID → (PPID, RSS MB) for all live processes. Serves
    both the ancestry walk and the RAM per-PID usage source."""
    table: dict[int, tuple[int, int]] = {}
    for proc in psutil.process_iter(["pid", "ppid", "memory_info"]):
        info = proc.info
        mem = info.get("memory_info")
        rss_mb = int(mem.rss / (1024 * 1024)) if mem is not None else 0
        table[info["pid"]] = (info["ppid"] or 0, rss_mb)
    return table


def _attribute_pid(
    pid: int,
    roots: dict[int, str],
    server_pids: dict[int, str],
    parents: dict[int, int],
) -> tuple[str, str] | None:
    """Walk PID's ancestor chain; return ``("shared", nodeset)`` when it
    reaches a registered server PID, ``("job", run_id)`` when it reaches a
    running job's root, None otherwise (external)."""
    cur = pid
    for _ in range(_MAX_ANCESTOR_DEPTH):
        name = server_pids.get(cur)
        if name is not None:
            return ("shared", name)
        run_id = roots.get(cur)
        if run_id is not None:
            return ("job", run_id)
        nxt = parents.get(cur)
        if not nxt or nxt == cur:
            return None
        cur = nxt
    return None


def _attribute(
    procs: list[tuple[int, int]],
    server_pids: dict[int, str],
    jobs: dict[str, int],
    parents: dict[int, int],
) -> tuple[dict[str, int], dict[str, int], int]:
    """Attribute one resource's per-PID usage list to owners. Returns
    (shared nodeset → MB, run_id → MB, total attributed MB)."""
    roots = {root: run_id for run_id, root in jobs.items()}
    shared_mb: dict[str, int] = {}
    job_mb: dict[str, int] = dict.fromkeys(jobs, 0)
    attributed = 0
    for pid, mem in procs:
        if mem <= 0:
            continue
        owner = _attribute_pid(pid, roots, server_pids, parents)
        if owner is None:
            continue
        kind, key = owner
        if kind == "shared":
            shared_mb[key] = shared_mb.get(key, 0) + mem
        else:
            job_mb[key] = job_mb.get(key, 0) + mem
        attributed += mem
    return shared_mb, job_mb, attributed
