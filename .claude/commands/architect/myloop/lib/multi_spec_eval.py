#!/usr/bin/env python3
"""multi_spec_eval.py — myloop multi-spec measured-eval wrapper.

The Python side of myloop's per-iter EXPERIMENT phase. Reads an
ExperimentSpec envelope (K ≥ 1 specs, each with passes ≥ 1),
runs each spec's measured eval against the JobScheduler backend,
aggregates per-pass results, and writes per-spec eval_metadata files
that DISTILL can consume.

Invocation contract (called from `loop.md § 3c-(b)`):

    python multi_spec_eval.py run \
        --graph        <g> \
        --version      vN \
        --iter         <n> \
        --spec-list    <path-to-spec.json envelope> \
        --eval-spec-ids "<comma-joined spec_ids to eval>" \
        --staging      <path-to-.staging/iter_<n>/> \
        --frozen-root  <repo root> \
        --admission    <profile name in .claude/commands/experiment/profiles.yaml>

Outputs:
  - <staging>/eval_metadata_<spec_id>.json  (one per spec in --eval-spec-ids)
  - <staging>/multi_spec_eval_log.md         (always)
  - mutates outputs/design_runs/myloop/<g>/v<N>/experiment_design.yaml:
    locks `baseline:` on profiles used at their passes_required ≥ 3
    floor for the first time.

Mix policy — ONE parallel wave per iter:
  All (spec, pass_idx) submissions for this iter go into a single
  parallel wave. There are no internal "batches" — the wrapper computes
  the full submission list, POSTs every /api/eval/v2/start in rapid
  succession, then waits for every `_DONE` file. The JobScheduler
  arbitrates admission based on VRAM; submissions that don't fit
  immediately are queued and run as capacity frees up. This is what
  "合集的实验" means: the iter's whole measured eval is one logical
  mix experiment, reassembled into K per-spec results afterwards.

  Example shapes:
    - K=2, passes=[1, 1]    → 2 submissions in parallel
    - K=1, passes=[3]       → 3 submissions in parallel (same spec, 3 draws)
    - K=2, passes=[1, 3]    → 4 submissions in parallel
    - K=3, passes=[3, 3, 3] → 9 submissions in parallel

  No sequential batches. No artificial wait between waves.

Worker-count allocation (binary-search makespan, ep-weighted):
  Each graph has a per-method max worker cap = `perf_<graph>.worker_count`
  (the perf profile's worker_count, the GPU/CPU-fit upper bound for
  this method). The wrapper minimizes wave wall clock by binary-searching
  the smallest target wave count W such that

      Σ_i min(profile_wc_i, ⌈ep_i / W⌉) ≤ perf_cap

  Then per-submission `wc_i = min(profile_wc_i, ⌈ep_i / W⌉)`, and the
  actual wave count for sub_i = `⌈ep_i / wc_i⌉` (≤ W when not clamped
  by profile_wc, equal to wave-count floor when clamped).

  Why ep-weighted: under constant per-episode time across the wave
  (true within one graph since step_budget x per_step_budget_sec is
  uniform across the smoke/perf/custom profiles), wall clock = W x t_per_ep
  is minimized when all subs hit the same wave count. This is the
  optimal makespan allocation; asymmetric mixes (e.g., a 216-ep perf
  spec + three 30-ep custom-subset passes) get the long sub more workers so
  it doesn't hold up the wave.

  Examples (mapgpt: perf_cap=40, perf profile (216 ep, wc=40), a
  30-ep custom-subset profile (wc=30)):
    K=1 perf solo (N=1)              → wc=[40], W=1, waves=6, wall=30min
    K=1 custom 3-pass (N=3)          → wc=[10]x3, W=3, waves=3, wall=15min
    K=2 all-perf (N=2)               → wc=[20]x2, W=11, waves=11, wall=55min
    K=2 perf+custom [1,3] (N=4)      → wc=[27, 4, 4, 4], W=8, waves=8, wall=40min
    K=3 all-custom 3-pass (N=9)      → wc=[4]x9, W=8, waves=8, wall=40min

  `profile_wc` is a hard upper bound (a 30-ep-profile sub never gets
  more than 30 workers even if cap would allow). If `perf_<graph>` is
  absent (custom-only run), no cap is applied — each submission keeps
  its profile worker_count. Allocation REPLACES whatever worker_count
  the spec's own profile or eval_profile.overrides declared.

The wave issues HTTP POST /api/eval/v2/start in succession for every
submission, then polls /api/eval/v2/runs/<run_id> until every run_id
reaches a terminal state (completed / failed / cancelled). JobScheduler
admission control serializes/parallelizes the wave as VRAM allows.

Exit codes:
    0   every spec in --eval-spec-ids has a usable eval_metadata file
        (per-spec outcome_class may still be "crash" — that's data)
    1   infra failure: backend unreachable, /start refused, etc.
        Partial eval_metadata files may have been written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # PyYAML, available in agentcanvas conda env

# Sibling import path: bin/_common.py lives at .claude/commands/experiment/bin/_common.py
# This file lives at .claude/commands/architect/myloop/lib/multi_spec_eval.py
# parents:[lib, myloop, architect, commands] → parents[3] is .claude/commands/
_HERE = Path(__file__).resolve()
_BIN = _HERE.parents[3] / "experiment" / "bin"
sys.path.insert(0, str(_BIN))
from _common import DEFAULT_BACKEND, die_unreachable  # noqa: E402

ERROR_PATTERNS = (
    "ERROR",
    "Traceback",
    "OutOfMemoryError",
    "CUDA out of memory",
    "Killed",
    "RuntimeError",
)


# ----------------------------------------------------------------------
# Tiny HTTP helpers (mirrors submit.py — same stdlib style, no requests dep)
# ----------------------------------------------------------------------


def _http(method: str, url: str, payload: dict | None = None, timeout: float = 10.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


# ----------------------------------------------------------------------
# Profile + experiment_design.yaml handling
# ----------------------------------------------------------------------


def load_experiment_design(path: Path) -> dict:
    """Return the experiment_design.yaml as a dict (top-level keys are
    profile names; each value is a dict of fields)."""
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"experiment_design.yaml at {path} is not a mapping")
    return data


def write_experiment_design(path: Path, data: dict) -> None:
    """Write the experiment_design.yaml. Preserves dict ordering (Python
    3.7+ dict + PyYAML default flow style off). Comments are NOT preserved
    — PyYAML can't round-trip them. The first multi-spec landing accepts
    this; a future ruamel.yaml swap can preserve comments if needed."""
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def read_admission_profile(profile_name: str) -> dict:
    """Read .claude/commands/experiment/profiles.yaml for marginal_vram_mb +
    exclusive_gpu. We reuse submit.py's parser via subprocess? No — we
    just re-implement the tiny YAML walk here, but using PyYAML since
    profiles.yaml is also valid YAML."""
    profiles_path = _BIN.parent / "profiles.yaml"
    with profiles_path.open() as f:
        all_yaml = yaml.safe_load(f) or {}
    experiments = all_yaml.get("experiments") or {}
    defaults = all_yaml.get("defaults") or {}
    entry = experiments.get(profile_name) or defaults
    return {
        "marginal_vram_mb": int(entry.get("vram_mb", 22000) or 0),
        "exclusive_gpu": bool(entry.get("exclusive_gpu", True)),
        "priority": "normal",
    }


# ----------------------------------------------------------------------
# Submission planning — flatten K specs x their passes into ONE parallel
# wave. The whole iter's eval is one logical "mix experiment".
# ----------------------------------------------------------------------


def plan_submissions(specs_to_eval: list[dict]) -> list[dict]:
    """Return flat list of (spec_id, pass_idx, spec) submissions.

    All entries are intended to submit IN PARALLEL — the wrapper POSTs
    every /api/eval/v2/start in succession, then waits for every
    `_DONE` file. The JobScheduler arbitrates admission based on VRAM;
    submissions that don't fit immediately are queued.

    Examples:
      K=2 passes=[1, 1]    → 2 submissions (A pass0, B pass0)
      K=1 passes=[3]       → 3 submissions (A pass0, A pass1, A pass2)
      K=2 passes=[1, 3]    → 4 submissions
      K=3 passes=[3, 3, 3] → 9 submissions
    """
    subs: list[dict] = []
    for s in specs_to_eval:
        passes = int(s.get("passes", 1))
        for p in range(passes):
            subs.append(
                {
                    "spec_id": s["spec_id"],
                    "pass_idx": p,
                    "spec": s,
                }
            )
    return subs


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------


def resolve_method_max_workers(design: dict, graph: str, log: list[str]) -> int | None:
    """Look up the `perf_<graph>` profile's `worker_count` — this is the
    per-method max worker cap that the iter's parallel wave must
    collectively respect.

    Returns None if `perf_<graph>` is absent (e.g. custom-only experiment_design).
    Callers fall back to per-spec profile worker_count in that case.
    """
    perf_key = f"perf_{graph}"
    profile = design.get(perf_key)
    if not isinstance(profile, dict):
        log.append(
            f"  [worker-cap] perf profile {perf_key!r} missing — fall back to per-spec profile worker_count"
        )
        return None
    wc = profile.get("worker_count")
    if not isinstance(wc, int) or wc < 1:
        log.append(
            f"  [worker-cap] perf profile {perf_key!r} has invalid worker_count={wc!r} — fall back"
        )
        return None
    log.append(f"  [worker-cap] method_max_workers = {wc} (from {perf_key}.worker_count)")
    return wc


def allocate_workers(*, method_max: int | None, submissions: list[dict]) -> list[int]:
    """Compute worker_count per submission for the iter's parallel wave,
    minimizing wave wall clock (makespan = max actual wave count across
    submissions). Returns a list of int wc, same order as ``submissions``.

    Each entry in ``submissions`` is a dict carrying at least:
        - "ep_count":   int, episode count this submission will run
        - "profile_wc": int, hard upper bound on workers for this submission
                        (= spec.profile.worker_count, e.g. custom=30, perf=40)

    Algorithm — binary-search smallest target wave count W such that
        Σ min(profile_wc_i, ⌈ep_i / W⌉) ≤ method_max
    then wc_i = min(profile_wc_i, ⌈ep_i / W⌉).

    Under constant per-episode time across the wave this is the optimal
    makespan allocation. profile_wc clamps the per-sub worker ceiling so
    a sub never gets more than its profile declares.

    Edge cases:
      - method_max is None (no perf_<graph> in design) → each sub gets
        its profile_wc (no cross-sub cap to enforce).
      - submissions is empty → return [].
      - submissions with ep_count=0 → defensive: caller filters but the
        algorithm still returns 1 worker per such sub.
      - even N (one worker per sub) exceeds cap → return [1]*N; the
        JobScheduler queue serializes whichever can't fit.
    """
    if not submissions:
        return []
    if method_max is None:
        return [max(1, int(s["profile_wc"])) for s in submissions]

    def need(W: int) -> int:
        return sum(
            min(int(s["profile_wc"]), math.ceil(int(s["ep_count"]) / W)) for s in submissions
        )

    # Floor case: N (one worker per sub) doesn't fit → degrade gracefully.
    min_need = sum(min(int(s["profile_wc"]), 1) for s in submissions)
    if min_need > method_max:
        return [1] * len(submissions)

    max_ep = max(int(s["ep_count"]) for s in submissions)
    if max_ep <= 0:
        return [1] * len(submissions)

    lo, hi = 1, max_ep
    while lo < hi:
        mid = (lo + hi) // 2
        if need(mid) <= method_max:
            hi = mid
        else:
            lo = mid + 1
    W = lo
    return [
        max(1, min(int(s["profile_wc"]), math.ceil(int(s["ep_count"]) / W))) for s in submissions
    ]


def submit_one(
    *,
    backend: str,
    graph: str,
    spec: dict,
    overlay_dir: Path | None,
    eval_profile_yaml: dict,
    admission: dict,
    worker_count_override: int | None,
    log: list[str],
) -> dict:
    """POST /api/eval/v2/start for one (spec, one pass). Returns
    {"run_id": ...} on success or {"error": ...} on submit failure.

    Note: this is ONE pass of the spec. Caller orchestrates the wave
    across all (spec, pass_idx) combinations. `worker_count_override`
    reflects the wave's per-submission allocation (see
    `allocate_workers`); if None, the profile's own worker_count is
    used (fallback when no perf_<graph> profile is present).
    """
    eval_block: dict[str, Any] = {"graph_name": graph}
    # Fields the backend understands (per submit.py's eval block schema).
    # `episode_selectors` is list-of-dicts (per-episode {suite, task_id}) —
    # required by multi-suite envs like LIBERO where the harness can ONLY
    # advance task_id via episode_selectors. `dataset` is the dataset key
    # some envs need. Both are list/dict-shaped but the JSON path handles
    # them fine; only the for-loop gate needs to admit them.
    for key in (
        "episode_count",
        "worker_count",
        "step_budget",
        "per_step_budget_sec",
        "split",
        "episode_indices",
        "episode_selectors",
        "dataset",
        "start_episode_index",
    ):
        if key in eval_profile_yaml:
            eval_block[key] = eval_profile_yaml[key]

    # eval_profile.overrides from the spec — late overrides (still pre worker-count override)
    for k, v in (spec.get("eval_profile", {}).get("overrides", {}) or {}).items():
        eval_block[k] = v

    # Apply wave-level worker allocation (highest priority — overrides both
    # profile worker_count and any eval_profile.overrides.worker_count).
    if worker_count_override is not None:
        eval_block["worker_count"] = worker_count_override

    payload = {
        **eval_block,
        "via_subprocess": True,
        "marginal_vram_mb": admission["marginal_vram_mb"],
        "exclusive_gpu": admission["exclusive_gpu"],
        "priority": admission["priority"],
    }
    if overlay_dir is not None:
        payload["active_workspace_dir"] = str(overlay_dir.resolve())

    try:
        _status, body = _http(
            "POST",
            f"{backend}/api/eval/v2/start",
            payload,
            timeout=180.0,
        )
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace") if hasattr(e, "read") else str(e)
        log.append(f"  [submit fail] HTTP {e.code}: {body_txt[:300]}")
        return {"error": f"HTTP {e.code}: {body_txt[:300]}"}
    except (urllib.error.URLError, OSError) as e:
        log.append(f"  [submit fail] {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}

    run_id = body.get("run_id")
    if not run_id:
        log.append(f"  [submit fail] no run_id in response: {body}")
        return {"error": f"no run_id: {body}"}
    log.append(
        f"  [submit OK]  spec={spec['spec_id']} run_id={run_id} initial={body.get('status')}"
    )
    return {"run_id": run_id, "initial_status": body.get("status")}


def wait_for_run_dir_done(repo_root: Path, run_id: str, max_wait_sec: float = 86400.0) -> bool:
    """Poll for `_DONE` file in the run dir. Returns True on terminal
    completion, False on timeout."""
    run_dir = repo_root / "outputs" / "eval_runs" / run_id
    started = time.time()
    while True:
        if (run_dir / "_DONE").exists():
            return True
        if time.time() - started > max_wait_sec:
            return False
        time.sleep(2.0)


def wait_for_wave(repo_root: Path, run_ids: list[str], log: list[str]) -> None:
    """Block until every run_id in the wave has its `_DONE` file."""
    pending = set(run_ids)
    log.append(f"  [wait] {len(pending)} run(s) in this wave: {sorted(pending)}")
    started = time.time()
    last_print = 0.0
    while pending:
        done_now = set()
        for rid in pending:
            run_dir = repo_root / "outputs" / "eval_runs" / rid
            if (run_dir / "_DONE").exists():
                done_now.add(rid)
        if done_now:
            for rid in done_now:
                log.append(f"  [done] {rid} ({time.time() - started:.0f}s into wave)")
            pending -= done_now
            continue
        now = time.time()
        if now - last_print > 60.0:
            log.append(f"  [wait] {len(pending)} still running after {now - started:.0f}s")
            last_print = now
        time.sleep(2.0)


# ----------------------------------------------------------------------
# Result extraction + per-spec aggregation
# ----------------------------------------------------------------------


def read_run_summary(repo_root: Path, run_id: str) -> dict:
    """Load outputs/eval_runs/<run_id>/summary.json. Returns dict with
    'aggregate_metrics' and 'episodes' (list of per-ep records)."""
    sj = repo_root / "outputs" / "eval_runs" / run_id / "summary.json"
    if not sj.exists():
        return {"aggregate_metrics": {}, "episodes": [], "_missing": True}
    return json.loads(sj.read_text())


def extract_per_ep_success(repo_root: Path, run_id: str) -> list[int]:
    """Return [ep_position -> 0/1 success] for this run, preserving the
    eval's emission order. Prefers summary.json's `episodes` list (the
    authoritative roll-up: one entry per dispatched episode, in order)
    because per-ep dirs are keyed by `episode_index` and collide for
    multi-suite evals (e.g. LIBERO's 40 (suite, task_id) pairs all
    re-using episode_indices [0..4] → only 5 dirs survive, last-task
    wins). Per-ep dirs are used only as a fallback when summary.json
    has no `episodes` block."""
    summ = read_run_summary(repo_root, run_id)
    out: list[int] = []
    for e in summ.get("episodes") or []:
        m = e.get("metrics") or {}
        s = m.get("success") if m.get("success") is not None else m.get("Success")
        if s is None:
            continue
        out.append(1 if float(s) > 0.5 else 0)
    if out:
        return out
    # Fallback: per-ep dirs (works for single-suite evals where ep dirs
    # don't collide).
    base = repo_root / "outputs" / "eval_runs" / run_id / "episodes"
    eps: list[tuple[int, int]] = []
    if base.exists():
        for ep_dir in sorted(base.glob("ep*")):
            ep_json = ep_dir / "episode.json"
            if not ep_json.exists():
                continue
            data = json.loads(ep_json.read_text())
            metrics = data.get("metrics") or {}
            succ = metrics.get("success")
            if succ is None:
                succ = metrics.get("Success")
            if succ is None:
                continue
            try:
                ep_idx = int(ep_dir.name[2:])
            except ValueError:
                ep_idx = len(eps)
            eps.append((ep_idx, 1 if float(succ) > 0.5 else 0))
    eps.sort()
    return [s for _, s in eps]


def aggregate_passes(
    *,
    spec_id: str,
    spec: dict,
    pass_results: list[dict],  # list of {pass_idx, run_id, per_ep_success, summary}
    repo_root: Path,
) -> dict:
    """Aggregate the K passes for one spec into the per-spec
    `eval_metadata_<spec_id>.json` payload shape."""
    pass_results = sorted(pass_results, key=lambda r: r["pass_idx"])
    run_ids = [r["run_id"] for r in pass_results]
    artifacts_dirs = [f"outputs/eval_runs/{rid}/" for rid in run_ids]

    # Per-pass per-ep success matrix (outer = passes, inner = eps).
    per_ep_matrix: list[list[int]] = [r["per_ep_success"] for r in pass_results]

    # Per-pass mean success.
    per_pass_sr: list[float] = []
    for matrix_row in per_ep_matrix:
        if not matrix_row:
            per_pass_sr.append(float("nan"))
        else:
            per_pass_sr.append(sum(matrix_row) / len(matrix_row))

    passes_n = len(per_pass_sr)
    mean_sr: float | None = (
        statistics.fmean(per_pass_sr)
        if per_pass_sr and not any(math.isnan(x) for x in per_pass_sr)
        else None
    )
    sd_sr: float | None = (
        statistics.stdev(per_pass_sr) if passes_n >= 2 and mean_sr is not None else None
    )

    # robust_sr — per-ep majority vote across passes (ceil(passes/2) success threshold)
    robust_sr: float | None = None
    if (
        passes_n >= 2
        and per_ep_matrix
        and all(len(row) == len(per_ep_matrix[0]) for row in per_ep_matrix)
    ):
        threshold = math.ceil(passes_n / 2)
        n_eps = len(per_ep_matrix[0])
        wins = 0
        for ep_i in range(n_eps):
            successes = sum(per_ep_matrix[p][ep_i] for p in range(passes_n))
            if successes >= threshold:
                wins += 1
        robust_sr = wins / n_eps if n_eps else None

    # Pull other aggregate metrics from the first pass's summary.json
    # (assume they're consistent across passes; could be averaged but
    # not part of the noise-control story).
    extra_metrics: dict[str, Any] = {}
    if pass_results:
        first_summary = pass_results[0].get("summary") or {}
        agg = first_summary.get("aggregate_metrics") or {}
        for k, v in agg.items():
            if k in ("success", "Success"):
                continue  # absorbed into mean_sr
            extra_metrics[k] = v

    metrics_digest = {
        "mean_sr": mean_sr,
        "sd_sr": sd_sr,
        "robust_sr": robust_sr,
        "score": mean_sr,  # alias for trace.md compatibility
        **extra_metrics,
    }

    # outcome_class: "ok" if every pass produced a non-empty per_ep_success.
    crashed = any(not r["per_ep_success"] for r in pass_results)
    outcome_class = "crash" if crashed else "ok"

    payload = {
        "spec_kind": spec.get("kind"),
        "patch_applied": spec.get("patch") is not None,
        "implementer_status": "OK" if spec.get("patch") is not None else "N/A",
        "implementer_attempts": 1,  # caller may overwrite if it tracked apply-step retries
        "passes": passes_n,
        "run_ids": run_ids,
        "artifacts_dirs": artifacts_dirs,
        "metrics_digest": metrics_digest,
        "per_ep_success": per_ep_matrix,
        "outcome_class": outcome_class,
        "baseline_cache_hit": False,
    }
    return payload


# ----------------------------------------------------------------------
# Baseline locking — append `baseline:` block to a profile in
# experiment_design.yaml when (a) profile has passes_required ≥ 3 and
# (b) no baseline is yet locked and (c) at least one ok run of this iter
# satisfied passes >= passes_required.
# ----------------------------------------------------------------------


def maybe_lock_baseline(
    *,
    design_yaml_path: Path,
    profile_name: str,
    eval_metadata: dict,
    iter_n: int,
    log: list[str],
) -> None:
    design = load_experiment_design(design_yaml_path)
    profile = design.get(profile_name)
    if not isinstance(profile, dict):
        log.append(
            f"  [baseline] profile {profile_name!r} not in experiment_design.yaml — skip lock"
        )
        return
    required = int(profile.get("passes_required", 0))
    if required < 3:
        return  # No baseline needed for 1-pass profiles
    if profile.get("baseline"):
        return  # Already locked, immutable
    if eval_metadata.get("outcome_class") != "ok":
        log.append(f"  [baseline] {profile_name}: outcome not ok, deferring lock")
        return
    if int(eval_metadata.get("passes", 0)) < required:
        log.append(f"  [baseline] {profile_name}: passes < required, deferring lock")
        return

    md = eval_metadata.get("metrics_digest") or {}
    profile["baseline"] = {
        "mean_sr": md.get("mean_sr"),
        "sd_sr": md.get("sd_sr"),
        "robust_sr": md.get("robust_sr"),
        "passes": required,
        "run_ids": list(eval_metadata.get("run_ids") or [])[:required],
        "locked_at": f"iter_{iter_n}",
        "locked_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    design[profile_name] = profile
    write_experiment_design(design_yaml_path, design)
    log.append(
        f"  [baseline] locked profile {profile_name}: mean={md.get('mean_sr')} "
        f"sd={md.get('sd_sr')} robust={md.get('robust_sr')} at iter_{iter_n}"
    )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    backend = os.environ.get("AGENTCANVAS_BACKEND_URL", DEFAULT_BACKEND)
    repo_root = Path(args.frozen_root).resolve()
    staging = Path(args.staging).resolve()
    spec_list = Path(args.spec_list).resolve()
    if not spec_list.is_file():
        print(f"[multi-spec] envelope missing: {spec_list}", file=sys.stderr)
        return 1
    envelope = json.loads(spec_list.read_text())
    iter_n = int(envelope.get("iter", 0))
    all_specs = envelope.get("specs") or []
    eval_ids = [s.strip() for s in (args.eval_spec_ids or "").split(",") if s.strip()]
    specs_to_eval = [s for s in all_specs if s["spec_id"] in eval_ids]
    if not specs_to_eval:
        print(
            f"[multi-spec] no specs to eval (envelope K={len(all_specs)}, "
            f"--eval-spec-ids={eval_ids})",
            file=sys.stderr,
        )
        return 1

    # experiment_design.yaml lives at outputs/design_runs/myloop/<graph>/<vN>/
    vN_dir = repo_root / "outputs" / "design_runs" / "myloop" / args.graph / args.version
    design_yaml = vN_dir / "experiment_design.yaml"
    if not design_yaml.exists():
        print(f"[multi-spec] experiment_design.yaml missing at {design_yaml}", file=sys.stderr)
        return 1
    design = load_experiment_design(design_yaml)

    admission = read_admission_profile(args.admission)

    # Probe backend
    try:
        st, _ = _http("GET", f"{backend}/api/eval/v2/queue", timeout=5)
        if st != 200:
            raise RuntimeError(f"queue probe returned {st}")
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
        return die_unreachable(backend, e, prog="multi-spec-eval")

    log: list[str] = []
    log.append(f"# multi_spec_eval — iter_{iter_n} on {args.graph} v{args.version}")
    log.append(f"backend = {backend}")
    log.append(f"admission_profile = {args.admission} ({admission})")
    log.append(f"K specs to eval = {len(specs_to_eval)}: {[s['spec_id'] for s in specs_to_eval]}")

    # Resolve method-wide worker cap from `perf_<graph>` profile.
    # The iter's parallel wave distributes this cap across all submissions
    # (sum of passes across specs) so the total worker count never exceeds
    # what perf_<graph> declares.
    method_max_workers = resolve_method_max_workers(design, args.graph, log)

    # Plan ONE parallel wave: every (spec, pass_idx) combination.
    submissions = plan_submissions(specs_to_eval)
    total_subs = len(submissions)
    log.append(f"plan: {total_subs} submission(s) — single parallel wave")
    for s in submissions:
        log.append(f"  {s['spec_id']} pass_idx={s['pass_idx']}")

    # Stage 1: resolve per-submission profile + ep_count + profile_wc.
    # Submissions that can't be resolved (unknown profile) get marked as
    # errored here and skipped from worker allocation.
    sub_records: list[dict] = []
    for entry in submissions:
        spec = entry["spec"]
        spec_id = entry["spec_id"]
        profile_name = (spec.get("eval_profile") or {}).get("name")
        profile_yaml = design.get(profile_name)
        if not isinstance(profile_yaml, dict):
            sub_records.append(
                {
                    "entry": entry,
                    "error": f"unknown profile {profile_name!r}",
                }
            )
            continue
        overrides = spec.get("eval_profile", {}).get("overrides") or {}
        ep_count = int(overrides.get("episode_count", profile_yaml.get("episode_count", 0)))
        profile_wc = int(profile_yaml.get("worker_count", 1))
        sub_records.append(
            {
                "entry": entry,
                "profile_name": profile_name,
                "profile_yaml": profile_yaml,
                "ep_count": ep_count,
                "profile_wc": profile_wc,
            }
        )

    # Stage 2: allocate workers across the WHOLE wave (binary-search makespan).
    alloc_input = [
        {"ep_count": r["ep_count"], "profile_wc": r["profile_wc"]}
        for r in sub_records
        if "error" not in r
    ]
    allocated_list = allocate_workers(
        method_max=method_max_workers,
        submissions=alloc_input,
    )

    # Stage 3: log the wave allocation table.
    log.append("\n## Worker allocation (binary-search makespan)")
    log.append(f"method_max (perf_<graph>.worker_count) = {method_max_workers}")
    valid_records = [r for r in sub_records if "error" not in r]
    if not valid_records:
        log.append("  no resolvable submissions — no allocation table")
    else:
        actual_waves = [
            math.ceil(r["ep_count"] / max(1, allocated_list[i]))
            for i, r in enumerate(valid_records)
        ]
        target_W = max(actual_waves) if actual_waves else 0
        total_alloc = sum(allocated_list)
        spare = method_max_workers - total_alloc if method_max_workers is not None else None
        log.append(
            f"  N = {len(valid_records)} submissions, "
            f"target wave count W = {target_W}, "
            f"total allocated = {total_alloc}, "
            f"spare = {spare}"
        )
        log.append(
            f"    {'spec_id':<24} {'pass':>4} {'profile':<24} "
            f"{'ep':>5} {'pcap':>5} {'wc':>4} {'waves':>6}"
        )
        for i, r in enumerate(valid_records):
            entry = r["entry"]
            log.append(
                f"    {entry['spec_id']:<24} {entry['pass_idx']:>4} "
                f"{r['profile_name']:<24} {r['ep_count']:>5} "
                f"{r['profile_wc']:>5} {allocated_list[i]:>4} "
                f"{actual_waves[i]:>6}"
            )

    # Stage 4: submit every (resolved) submission in a single parallel wave.
    log.append(f"\n## Submitting wave ({len(valid_records)} submissions)")
    submitted: list[tuple[dict, dict]] = []  # (submission_entry, submission_result)
    infra_failed = False
    valid_iter_idx = 0
    for r in sub_records:
        if "error" in r:
            log.append(f"  [submit fail] spec={r['entry']['spec_id']} {r['error']}")
            submitted.append((r["entry"], {"error": r["error"]}))
            continue
        entry = r["entry"]
        spec = entry["spec"]
        spec_id = entry["spec_id"]
        wc = allocated_list[valid_iter_idx]
        valid_iter_idx += 1
        overlay = (staging / f"active_workspace_{spec_id}") if spec.get("patch") else None
        res = submit_one(
            backend=backend,
            graph=args.graph,
            spec=spec,
            overlay_dir=overlay,
            eval_profile_yaml=r["profile_yaml"],
            admission=admission,
            worker_count_override=wc,
            log=log,
        )
        submitted.append((entry, res))

    # Wait for the whole wave
    run_ids = [r["run_id"] for _, r in submitted if "run_id" in r]
    if not run_ids:
        log.append("  [wave] no successful submissions; marking infra failure")
        infra_failed = True
    else:
        wait_for_wave(repo_root, run_ids, log)

    # Record results by spec
    results_by_spec: dict[str, list[dict]] = {s["spec_id"]: [] for s in specs_to_eval}
    for entry, res in submitted:
        spec_id = entry["spec_id"]
        pass_idx = entry["pass_idx"]
        if "run_id" not in res:
            results_by_spec[spec_id].append(
                {
                    "pass_idx": pass_idx,
                    "run_id": None,
                    "per_ep_success": [],
                    "summary": {"aggregate_metrics": {}},
                    "submit_error": res.get("error"),
                }
            )
            continue
        rid = res["run_id"]
        summary = read_run_summary(repo_root, rid)
        per_ep = extract_per_ep_success(repo_root, rid)
        results_by_spec[spec_id].append(
            {
                "pass_idx": pass_idx,
                "run_id": rid,
                "per_ep_success": per_ep,
                "summary": summary,
            }
        )

    # Aggregate per spec + write eval_metadata files
    for spec in specs_to_eval:
        spec_id = spec["spec_id"]
        passes_results = results_by_spec.get(spec_id, [])
        eval_metadata = aggregate_passes(
            spec_id=spec_id,
            spec=spec,
            pass_results=passes_results,
            repo_root=repo_root,
        )
        out_path = staging / f"eval_metadata_{spec_id}.json"
        out_path.write_text(json.dumps(eval_metadata, indent=2) + "\n")
        log.append(
            f"\n[written] {out_path.name}: outcome={eval_metadata['outcome_class']} "
            f"passes={eval_metadata['passes']} mean_sr={eval_metadata['metrics_digest']['mean_sr']} "
            f"sd_sr={eval_metadata['metrics_digest']['sd_sr']} robust_sr={eval_metadata['metrics_digest']['robust_sr']}"
        )

        # Maybe lock baseline for this profile (idempotent — only first 3-pass run locks)
        profile_name = (spec.get("eval_profile") or {}).get("name")
        if profile_name:
            maybe_lock_baseline(
                design_yaml_path=design_yaml,
                profile_name=profile_name,
                eval_metadata=eval_metadata,
                iter_n=iter_n,
                log=log,
            )

    # Write the log
    (staging / "multi_spec_eval_log.md").write_text("\n".join(log) + "\n")

    return 1 if infra_failed else 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="multi_spec_eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="execute multi-spec measured eval for one iter")
    run.add_argument("--graph", required=True)
    run.add_argument("--version", required=True, help="vN (e.g. v0, v1)")
    run.add_argument("--iter", dest="iter_n", type=int, required=True)
    run.add_argument("--spec-list", required=True, help="path to spec.json envelope")
    run.add_argument(
        "--eval-spec-ids", required=True, help="comma-joined spec_ids that should be evaluated"
    )
    run.add_argument("--staging", required=True, help="path to .staging/iter_<n>/")
    run.add_argument("--frozen-root", required=True, help="repo root")
    run.add_argument(
        "--admission",
        required=True,
        help="profile name in .claude/commands/experiment/profiles.yaml",
    )

    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
