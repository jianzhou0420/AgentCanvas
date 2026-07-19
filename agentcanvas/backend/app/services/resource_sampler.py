"""ResourceSampler — always-on machine resource time-series for the System Log.

Backend-process singleton sampled at ~1 Hz. Writes one JSON line per sample to
``outputs/system/system-YYYYMMDD.jsonl`` and keeps a short in-memory ring so the
``/api/system/usage`` (latest) and ``/api/system/history`` endpoints answer
without re-reading disk. Distinct from ``ExecutionLogger`` (per-run node I/O):
this is machine-global and run-independent.

Why a sampler at all: ``/api/system/usage`` used to sample inline per request,
so nothing was ever persisted and the Monitor page's history was lost on every
refresh. The sampler closes that gap and adds two backend-only signals the
inline path can't see — event-loop lag and the live WebSocket client count.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger("agentcanvas.resource-sampler")

_NVIDIA_SMI = shutil.which("nvidia-smi")

# Global resource-log retention (days). 0 disables pruning. The on-disk
# footprint stays bounded by three things together: idle-gating (written only
# while a run is active) + daily files + this retention sweep — never unbounded.
RETENTION_DAYS = int(os.environ.get("AGENTCANVAS_SYSTEM_LOG_RETENTION_DAYS", "7"))


def _read_gpus() -> list[dict[str, Any]]:
    """Per-GPU util + memory via ``nvidia-smi``; ``[]`` if unavailable (never raises)."""
    if _NVIDIA_SMI is None:
        return []
    try:
        out = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return []

    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[2])
            used = int(parts[3])
            total = int(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "util_pct": util,
                "mem_used_mb": used,
                "mem_total_mb": total,
                "mem_pct": (used / total * 100.0) if total else 0.0,
            }
        )
    return gpus


def _owner_label(pid: int) -> str:
    """Best-effort owner for a GPU-using PID: 'backend', a nodeset server
    (auto_host --class X), an eval subprocess (eval:{run_id}), or the proc name."""
    try:
        import os

        if pid == os.getpid():
            return "backend"
        cmd = psutil.Process(pid).cmdline()
        joined = " ".join(cmd)
        if "auto_host" in joined:
            if "--class" in cmd:
                i = cmd.index("--class")
                if i + 1 < len(cmd):
                    return cmd[i + 1]
            return "nodeset-server"
        if "eval_subprocess_main" in joined:
            if "--run-dir" in cmd:
                i = cmd.index("--run-dir")
                if i + 1 < len(cmd):
                    return f"eval:{Path(cmd[i + 1]).name}"
            return "eval"
        return psutil.Process(pid).name()
    except Exception:
        return f"pid {pid}"


def _parse_pmon(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi pmon -c 1 -s m`` output into per-PID rows.

    Columns: gpu_idx, pid, type (C/G), fb MB, ccpm MB, command. Idle GPUs
    emit ``-`` placeholder rows; multi-GPU PIDs are summed. Pure function
    so the format assumption is unit-testable without a GPU.
    """
    mem_by_pid: dict[int, int] = {}
    type_by_pid: dict[int, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
            fb = int(parts[3])
        except ValueError:
            continue  # '-' placeholder row on idle GPUs
        mem_by_pid[pid] = mem_by_pid.get(pid, 0) + fb
        type_by_pid[pid] = parts[2]
    return [
        {"pid": pid, "mem_mb": mem, "gpu_ctx": type_by_pid.get(pid, "?")}
        for pid, mem in mem_by_pid.items()
    ]


def _gpu_processes() -> list[dict[str, Any]]:
    """Per-process GPU memory (best-effort), so the Monitor page and the
    VRAM attribution (resource_stats) can answer 'who is holding VRAM'.

    Uses ``nvidia-smi pmon`` because it lists graphics-type contexts too —
    habitat / MatterSim render via EGL and appear as type ``G``, which
    ``--query-compute-apps`` misses entirely (confirmed live 2026-07-04:
    a habitat env held 295 MB invisible to the compute-apps query).
    Falls back to the compute-only query where pmon is unsupported.
    ``[]`` when nvidia-smi is unavailable."""
    if _NVIDIA_SMI is None:
        return []
    procs: list[dict[str, Any]] | None = None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "pmon", "-c", "1", "-s", "m"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        ).stdout
        procs = _parse_pmon(out)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        procs = None
    if procs is None:
        procs = _compute_apps_processes()
    for p in procs:
        p["owner"] = _owner_label(p["pid"])
    procs.sort(key=lambda p: p["mem_mb"], reverse=True)
    return procs


def _compute_apps_processes() -> list[dict[str, Any]]:
    """Fallback: compute contexts only (misses EGL renderers — see above)."""
    try:
        out = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=True,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return []
    procs: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[1])
        except ValueError:
            continue
        procs.append({"pid": pid, "mem_mb": mem, "gpu_ctx": "C"})
    return procs


def sample_system(
    ws_clients: int = 0,
    event_loop_lag_ms: float = 0.0,
    gpu_procs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One machine snapshot. Cheap; safe to call without the sampler running
    (used as the ``/usage`` fallback). ``cpu_pct`` is the delta since the
    previous call in this process (psutil semantics). ``gpu_procs`` may be
    passed in (the sampler refreshes it on a slower cadence to avoid a 2nd
    nvidia-smi every tick); when None it is computed here."""
    vm = psutil.virtual_memory()
    return {
        "ts": datetime.utcnow().isoformat(),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True) or 0,
        "mem_used_mb": int((vm.total - vm.available) / (1024 * 1024)),
        "mem_total_mb": int(vm.total / (1024 * 1024)),
        "mem_pct": vm.percent,
        "gpus": _read_gpus(),
        "gpu_procs": _gpu_processes() if gpu_procs is None else gpu_procs,
        "event_loop_lag_ms": round(event_loop_lag_ms, 2),
        "ws_clients": ws_clients,
    }


class ResourceSampler:
    """Samples machine resources every ``period_sec`` and appends to a daily
    JSONL file, keeping the last ``history_len`` samples in memory."""

    def __init__(
        self,
        out_dir: Path,
        period_sec: float = 1.0,
        history_len: int = 600,
        ws_clients_fn: Callable[[], int] | None = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._outputs_dir = self._out_dir.parent  # .../outputs (root for per-run tee)
        self._period = period_sec
        self._history: deque[dict[str, Any]] = deque(maxlen=history_len)
        self._ws_clients_fn = ws_clients_fn
        self._task: asyncio.Task | None = None
        self._cur_day = ""  # for retention day-roll detection
        # gpu_procs (2nd nvidia-smi) refreshes on a slower cadence (~5s) and is
        # reused between — VRAM-by-process changes slowly, and this keeps the
        # sampler at ~1 nvidia-smi/s (the pre-existing baseline), not 2.
        self._gpu_procs_cache: list[dict[str, Any]] = []
        self._procs_every = max(1, round(5.0 / period_sec))
        self._procs_n = 0

    # ── queries (served by /api/system) ──
    def latest(self) -> dict[str, Any] | None:
        return self._history[-1] if self._history else None

    def history(self, n: int = 600) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        return list(self._history)[-n:]

    # ── persistence ──
    def _append_file(self, sample: dict[str, Any]) -> None:
        # Plain append (not atomic-rename): this is an append-only log, like
        # ExecutionLogger.flush(). Daily file keeps any single file bounded.
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            day = sample["ts"][:10].replace("-", "")  # YYYY-MM-DD → YYYYMMDD
            path = self._out_dir / f"system-{day}.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(sample) + "\n")
        except OSError as e:
            log.warning("resource sampler: cannot write under %s: %s", self._out_dir, e)

    def _sweep_retention(self) -> None:
        """Delete global ``system-*.jsonl`` files older than the retention window
        so the on-disk footprint stays bounded (no unbounded append-forever).
        Tune with ``AGENTCANVAS_SYSTEM_LOG_RETENTION_DAYS`` (0 disables)."""
        if RETENTION_DAYS <= 0:
            return
        cutoff = time.time() - RETENTION_DAYS * 86400
        try:
            for f in self._out_dir.glob("system-*.jsonl"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    # ── per-run tee (Q2) ──
    def _active_run_dirs(self) -> list[Path]:
        """Run dirs that should receive a copy of this machine sample — every
        currently-active run (eval jobs + the live canvas run). Semantics: the
        *machine* state during the run, not the run's exclusive consumption
        (concurrent runs each get the same machine sample)."""
        dirs: list[Path] = []
        # eval: the backend-owned JobScheduler knows the running run_ids.
        try:
            from ..state import get_services

            sched = get_services().job_scheduler
            if sched is not None:
                for rj in sched.list_active().get("running", []):
                    rid = rj.get("run_id")
                    if rid:
                        # Scheduler's own pool, not a hardcoded eval_runs/ —
                        # slot backends (/host) run against eval_runs_<suffix>.
                        d = sched.runs_dir / rid
                        if d.is_dir():
                            dirs.append(d)
        except Exception:
            pass
        # canvas: the in-process LoopRunner's current execution.
        try:
            from ..agent_loop.loop_runner import get_loop_runner

            st = get_loop_runner().get_status()
            if st.get("status") in ("running", "paused") and st.get("execution_id"):
                d = self._outputs_dir / "runs" / st["execution_id"]
                if d.is_dir():
                    dirs.append(d)
        except Exception:
            pass
        return dirs

    # ── loop ──
    def _build_sample(self, ws: int, lag_ms: float) -> dict[str, Any]:
        """Runs in a worker thread (see run_forever). Refreshes gpu_procs only
        every ~5s; reuses the cache otherwise to avoid a 2nd nvidia-smi/tick."""
        if self._procs_n % self._procs_every == 0:
            self._gpu_procs_cache = _gpu_processes()
        self._procs_n += 1
        return sample_system(
            ws_clients=ws, event_loop_lag_ms=lag_ms, gpu_procs=self._gpu_procs_cache
        )

    async def run_forever(self) -> None:
        # Prime cpu_percent so the first real sample isn't a meaningless 0.0.
        psutil.cpu_percent(interval=None)
        self._sweep_retention()  # prune stale daily files on startup
        lag_ms = 0.0
        while True:
            try:
                ws = self._ws_clients_fn() if self._ws_clients_fn else 0
                # Offload the blocking parts (nvidia-smi subprocess + psutil) to a
                # thread so the sampler never blocks the event loop it measures.
                sample = await asyncio.to_thread(self._build_sample, ws, lag_ms)
                day = sample["ts"][:10].replace("-", "")
                if day != self._cur_day:
                    self._cur_day = day
                    self._sweep_retention()  # day-roll prune (runs even when idle)
                # Always keep the in-memory ring fresh so the Live view +
                # /history work even when idle.
                self._history.append(sample)
                # Disk write ONLY while a run is active — idle = in-memory only,
                # so we never grow files 24/7. Active runs also get a per-run
                # tee (Q2): the same machine sample copied into the run dir.
                active_dirs = self._active_run_dirs()
                if active_dirs:
                    self._append_file(sample)
                    _line = json.dumps(sample) + "\n"
                    for _d in active_dirs:
                        try:
                            with open(_d / "system.jsonl", "a") as _f:
                                _f.write(_line)
                        except OSError:
                            pass
            except Exception:
                log.exception("resource sampler tick failed")
            t_sleep = time.perf_counter()
            await asyncio.sleep(self._period)
            # Event-loop lag = how much LONGER the sleep took than requested.
            # Timed around the sleep ONLY (the sample runs off-loop via
            # to_thread), so it reflects loop congestion — not the sampler's own
            # nvidia-smi/psutil work. (Timing the whole iter wrongly counted the
            # sample's thread wall-time, pegging lag at hundreds of ms.)
            lag_ms = max(0.0, (time.perf_counter() - t_sleep - self._period) * 1000)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


# ── module singleton (this backend process) ──
_sampler: ResourceSampler | None = None


def get_sampler() -> ResourceSampler | None:
    return _sampler


def set_sampler(sampler: ResourceSampler | None) -> None:
    global _sampler
    _sampler = sampler
