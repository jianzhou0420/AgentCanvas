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
  in the parent registry's ``_auto_servers``.
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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

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
        self, graph_name: str, worker_count: int, tree_peaks: dict[str, int]
    ) -> None:
        entry = self._data["graphs"].setdefault(graph_name, {"by_worker_count": {}})
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

    Call order per scheduler lifecycle:

    - ``note_job_started(run_id, ...)``   — from ``_spawn``
    - ``observe(jobs, registry)``         — from ``tick()``, every ~1s
    - ``note_job_finished(run_id, registry)`` — from ``_reap``
    """

    def __init__(self, calibration_path: Path) -> None:
        self._store = CalibrationStore(calibration_path)
        self._windows: dict[str, _JobWindow] = {}
        # Latest attribution snapshot (observability; served via /queue).
        self._last: dict[str, Any] = {}

    # ── lifecycle hooks (called by JobScheduler) ──

    def note_job_started(
        self,
        run_id: str,
        graph_name: str,
        worker_count: int,
        shared_nodesets: tuple[str, ...],
    ) -> None:
        self._windows[run_id] = _JobWindow(
            graph_name=graph_name or "unknown",
            worker_count=max(1, int(worker_count or 1)),
            shared_nodesets=tuple(shared_nodesets),
            started_at=time.time(),
        )

    def observe(self, jobs: dict[str, int], registry: Any | None) -> None:
        """Attribute the latest sample. ``jobs`` maps run_id → root PID."""
        try:
            self._observe(jobs, registry)
        except Exception:  # observability must never break the scheduler
            log.exception("resource observe failed")

    def note_job_finished(self, run_id: str, registry: Any | None) -> None:
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
                if name not in peaks and _is_loaded_local(registry, name):
                    peaks[name] = 0
        try:
            # 0 is a legitimate measurement (CPU-only / API-only graph);
            # samples_seen above guards against writing blind zeros.
            self._store.update_graph(
                window.graph_name,
                window.worker_count,
                {res: window.tree_peaks.get(res, 0) for res in RESOURCES},
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
                    source_hash=_source_hash(registry, name),
                )
            self._store.save()
        except Exception:
            log.exception("resource sediment failed for %s", run_id)

    def snapshot(self) -> dict[str, Any]:
        """Latest attribution split, for /queue observability."""
        return dict(self._last)

    def estimate(
        self,
        graph_name: str,
        worker_count: int,
        shared_infos: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Advisory per-resource estimate from calibration. Read-only.

        ``shared_infos``: one ``{name, loaded, source_hash, hints}`` per
        shared nodeset the graph needs (resolved by the scheduler from its
        registry); ``hints`` maps resource → author preset MB. The job's
        own subprocess tree (replicated servers + in-process models) comes
        from the ``graphs`` calibration.

        Honesty rule, per resource: ``estimate_mb`` is null unless EVERY
        component is calibrated — a missing singleton could be 10 GB, so a
        partial sum would lowball. Known parts are still reported via
        ``known_mb`` + ``breakdown``, gaps via ``uncalibrated``.

        ``max_workers`` (top-level) is the min across resources that are
        measurable on this machine; per-resource values sit inside
        ``resources[r]``.
        """
        wc = max(1, int(worker_count))
        data = self._store.data
        resources: dict[str, Any] = {}
        for res in RESOURCES:
            result = _estimate_once(data, graph_name, wc, shared_infos, res)
            free = self.measured_free_mb(res)
            max_workers: int | None = None
            if result["estimate_mb"] is not None and free is not None:
                max_workers = 0
                for probe in range(1, _MAX_WORKERS_PROBE + 1):
                    est = _estimate_once(data, graph_name, probe, shared_infos, res)[
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

    def _observe(self, jobs: dict[str, int], registry: Any | None) -> None:
        sample = _latest_sample()
        if not sample:
            return
        procs_table = _proc_table()
        parents = {pid: pp for pid, (pp, _) in procs_table.items()}
        server_pids = _shared_server_pids(registry)

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


def _graph_view(entry: dict[str, Any] | None, resource: str) -> dict[str, Any] | None:
    """Same flattening for a graphs entry (by_worker_count sub-objects)."""
    if not entry:
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
    shared_infos: list[dict[str, Any]],
    resource: str,
) -> dict[str, Any]:
    """One resource's estimate at a fixed worker count. See
    ``ResourceStatsTracker.estimate``."""
    breakdown: dict[str, Any] = {"shared": {}, "tree": None}
    uncalibrated: list[str] = []
    known = 0

    tree = estimate_tree_mb(
        _graph_view((data.get("graphs") or {}).get(graph_name), resource), wc
    )
    if tree is None:
        uncalibrated.append("graph-tree")
    else:
        known += tree[0]
        breakdown["tree"] = {"mb": tree[0], "basis": tree[1]}

    used_hint = False
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


def _shared_server_pids(registry: Any | None) -> dict[int, str]:
    """PID → nodeset name for auto_host servers owned by THIS process's
    registry. Tagged keys (``name#k``) collapse to the plain nodeset name."""
    if registry is None:
        return {}
    out: dict[int, str] = {}
    for store_key, server in getattr(registry, "_auto_servers", {}).items():
        pid = getattr(server, "_pid", None)
        if pid:
            out[int(pid)] = store_key.split("#")[0]
    return out


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


def _is_loaded_local(registry: Any | None, nodeset_name: str) -> bool:
    """True when the nodeset is loaded in-process (no auto_host server) —
    the case where per-PID attribution structurally cannot see it."""
    if registry is None:
        return False
    try:
        return bool(registry.is_nodeset_loaded(nodeset_name)) and not registry.get_server_url(
            nodeset_name
        )
    except Exception:
        return False


# Source-tree hashing walks the nodeset's files — fine on demand, too heavy
# for the scheduler's 1s admission tick. 30s staleness is harmless: the hash
# only changes on code edits, and sediment-time writes recompute anyway.
_HASH_CACHE_TTL_SEC = 30.0
_hash_cache: dict[str, tuple[float, str | None]] = {}


def _source_hash(registry: Any | None, nodeset_name: str) -> str | None:
    """Content hash of the nodeset's source tree (calibration invalidation
    key), TTL-cached. Best-effort: None when the registry or source is
    unavailable."""
    if registry is None:
        return None
    cached = _hash_cache.get(nodeset_name)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _HASH_CACHE_TTL_SEC:
        return cached[1]
    value: str | None
    try:
        from ..components.content_hash import hash_nodeset_tree

        ns = getattr(registry, "_discovered_nodesets", {}).get(nodeset_name)
        source = getattr(ns, "_source_file", None) if ns is not None else None
        value = hash_nodeset_tree(source) if source else None
    except Exception:
        value = None
    _hash_cache[nodeset_name] = (now, value)
    return value
