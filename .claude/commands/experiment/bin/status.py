#!/usr/bin/env python3
"""Query JobScheduler status for one or all runs.

Companion to ``submit.py``. ``submit.py`` is a foreground submitter that
polls inline and Ctrl-C => cancel. ``status.py`` is the read/observe
side: detach from a previously-submitted ``run_id`` and come back later,
across conversations, without killing it.

Usage::

    status.py                     # list queued + running (and recent on stderr)
    status.py <run_id>            # one-shot snapshot of one run
    status.py <run_id> --watch    # follow status until _DONE; Ctrl-C exits cleanly (no cancel)
    status.py <run_id> --cancel   # POST cancel and exit

Backend defaults to $AGENTCANVAS_BACKEND_URL or http://127.0.0.1:5173
(Vite proxy that forwards /api + /ws to the real backend at :8000).

Exit codes::

    0   query succeeded (or watched run reached terminal status `done`)
    1   watched run reached terminal status `error / aborted / cancelled`
    2   backend unreachable / run_id not found
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Sibling module — bin/ is auto-added to sys.path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_BACKEND, die_unreachable

TERMINAL = {"done", "error", "aborted", "cancelled", "completed"}


def _http(method: str, url: str, payload: dict | None = None, timeout: float = 5.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode() or "{}"
        return resp.status, json.loads(body)


def _short(rid: str) -> str:
    return rid[:8] if len(rid) > 12 else rid


def _print_run_one_line(prefix: str, info: dict) -> None:
    state = info.get("scheduler_state") or info.get("status", "?")
    eps = info.get("episodes") or []
    cfg = info.get("config") or {}
    total = cfg.get("episode_count") or len(eps) or "?"
    graph = cfg.get("graph_name") or info.get("graph_name") or "—"
    print(f"{prefix} state={state:10s} eps={len(eps)}/{total} graph={graph}", flush=True)


def _cmd_list(backend: str) -> int:
    try:
        _, q = _http("GET", f"{backend}/api/eval/v2/queue")
    except (urllib.error.URLError, RuntimeError) as e:
        return die_unreachable(backend, e, prog="status")

    print(
        f"=== JobScheduler @ {backend} === "
        f"vram {q.get('reserved_vram_mb', 0)}/{q.get('usable_vram_mb', 0)} MB",
    )
    running = q.get("running", [])
    queued = q.get("queued", [])
    print(f"\n-- running ({len(running)}) --")
    if not running:
        print("  (none)")
    for r in running:
        line = (
            f"  {_short(r['run_id'])}  pid={r['pid']:<7} "
            f"vram={r['marginal_vram_mb']:>5}MB  "
            f"started={r.get('started_at', '—')}"
        )
        if r.get("cancel_requested"):
            line += "  CANCELLING"
        print(line)
        # Augment with episode progress if available.
        try:
            _, info = _http("GET", f"{backend}/api/eval/v2/runs/{r['run_id']}", timeout=3)
            _print_run_one_line(f"    └─ {r['run_id']}", info)
        except Exception:
            pass

    print(f"\n-- queued ({len(queued)}) --")
    if not queued:
        print("  (none)")
    for q_ in queued:
        excl = " excl-gpu" if q_.get("exclusive_gpu") else ""
        print(
            f"  {_short(q_['run_id'])}  prio={q_.get('priority', '?'):<6} "
            f"vram={q_['marginal_vram_mb']:>5}MB{excl}  "
            f"submitted={q_.get('submitted_at', '—')}"
        )
    return 0


def _cmd_one(backend: str, run_id: str) -> int:
    try:
        _, info = _http("GET", f"{backend}/api/eval/v2/runs/{run_id}", timeout=5)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[status] run_id '{run_id}' not found at {backend}", file=sys.stderr)
            return 2
        print(f"[status] /runs/{run_id} returned {e.code}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, RuntimeError) as e:
        return die_unreachable(backend, e, prog="status")

    state = info.get("scheduler_state") or info.get("status", "?")
    eps = info.get("episodes") or []
    cfg = info.get("config") or {}
    total = cfg.get("episode_count") or len(eps) or "?"
    print(f"run_id        : {run_id}")
    print(f"status        : {state}")
    if info.get("scheduler_state") and info.get("scheduler_state") != info.get("status"):
        print(f"persisted     : {info.get('status', '?')}")
    print(f"graph         : {cfg.get('graph_name') or info.get('graph_name') or '—'}")
    print(f"episodes      : {len(eps)} / {total}")
    if "started_at" in info:
        print(f"started_at    : {info.get('started_at')}")
    if "finished_at" in info:
        print(f"finished_at   : {info.get('finished_at')}")
    if info.get("error"):
        print(f"error         : {info['error']}")
    # Aggregate metrics if any episode has them.
    metrics_keys: list[str] = []
    for ep in eps:
        for k in ep.get("metrics") or {}:
            if k not in metrics_keys:
                metrics_keys.append(k)
    if metrics_keys:
        print("aggregate     :")
        for k in metrics_keys:
            vals = [
                ep["metrics"][k]
                for ep in eps
                if isinstance(ep.get("metrics"), dict) and k in ep["metrics"]
            ]
            if vals:
                print(f"  {k:14s}: mean={sum(vals) / len(vals):.3f}  n={len(vals)}")
    return 0


def _cmd_watch(backend: str, run_id: str, interval: float) -> int:
    interrupted = {"v": False}

    def _on_sigint(signum, frame):
        interrupted["v"] = True
        print("\n[watch] Ctrl-C — exiting watch (run NOT cancelled)", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    last_state, last_eps = None, -1
    while not interrupted["v"]:
        try:
            _, info = _http("GET", f"{backend}/api/eval/v2/runs/{run_id}", timeout=5)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[watch] run_id '{run_id}' not found", file=sys.stderr)
                return 2
            print(f"[watch] /runs/{run_id} returned {e.code}", file=sys.stderr)
            time.sleep(interval)
            continue
        except Exception as e:
            print(f"[watch] poll error: {e}", file=sys.stderr)
            time.sleep(interval)
            continue
        state = info.get("scheduler_state") or info.get("status", "?")
        eps = info.get("episodes") or []
        if state != last_state or len(eps) != last_eps:
            _print_run_one_line(f"[{time.strftime('%H:%M:%S')}]", info)
            last_state, last_eps = state, len(eps)
        if state in TERMINAL:
            return 0 if state in {"done", "completed"} else 1
        time.sleep(interval)
    # interrupted — best-effort one final snapshot
    return 0


def _cmd_cancel(backend: str, run_id: str) -> int:
    try:
        _status, body = _http("POST", f"{backend}/api/eval/v2/runs/{run_id}/cancel", {}, timeout=5)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[cancel] run_id '{run_id}' not found", file=sys.stderr)
            return 2
        print(f"[cancel] returned {e.code}", file=sys.stderr)
        return 2
    except Exception as e:
        return die_unreachable(backend, e, prog="cancel")
    new_state = body.get("status") or body.get("scheduler_state") or "?"
    print(f"[cancel] run_id={run_id} new_state={new_state}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="status.py", description=__doc__)
    p.add_argument("run_id", nargs="?", help="run id; omit to list queue + running")
    p.add_argument(
        "--watch",
        action="store_true",
        help="follow until terminal state (Ctrl-C exits without cancel)",
    )
    p.add_argument("--cancel", action="store_true", help="POST cancel and exit")
    p.add_argument("--backend", default=DEFAULT_BACKEND, help="backend URL")
    p.add_argument("--interval", type=float, default=2.0, help="watch poll interval (s)")
    args = p.parse_args(argv)

    if args.cancel and args.watch:
        print("[status] --cancel and --watch are mutually exclusive", file=sys.stderr)
        return 2
    if (args.cancel or args.watch) and not args.run_id:
        print("[status] --cancel/--watch require a run_id", file=sys.stderr)
        return 2

    if not args.run_id:
        return _cmd_list(args.backend)
    if args.cancel:
        return _cmd_cancel(args.backend, args.run_id)
    if args.watch:
        return _cmd_watch(args.backend, args.run_id, args.interval)
    return _cmd_one(args.backend, args.run_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
