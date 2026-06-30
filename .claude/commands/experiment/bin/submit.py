#!/usr/bin/env python3
"""Submit a graph eval to the running agentcanvas backend's JobScheduler.

Replaces the per-run uvicorn spawn + admit.py dance from the legacy
/experiment:run skill. The backend at $AGENTCANVAS_BACKEND_URL (default
http://127.0.0.1:5173, the Vite dev-server proxy that forwards /api and
/ws to the real backend at :8000) is assumed to be already running and
to host the JobScheduler we built in M1. We're a thin HTTP client.

Usage::

    submit.py <profile> <graph_name> [key=value ...] [--eval-overrides FILE]

Profile resolves to ``marginal_vram_mb`` + ``exclusive_gpu`` from
``.claude/commands/experiment/profiles.yaml``. ``key=value`` pairs land in the
eval block (e.g. ``episode_count=10 worker_count=2 step_budget=15``).
``key=value`` parsing (``_coerce``) only handles scalars; list/dict
eval params (``episode_indices``, ``episode_selectors``) must be passed
via ``--eval-overrides FILE`` — a JSON file merged into the eval block
*before* the ``key=value`` pairs, so an explicit ``key=value`` still
wins over the file.

Lifecycle:
    POST /api/eval/v2/start (via_subprocess=true)
    poll /api/eval/v2/runs/{id} until ``_DONE`` exists
    on Ctrl-C: POST /api/eval/v2/runs/{id}/cancel
    on non-success: tail stderr.log error markers

Exit codes:
    0   completed
    1   error / aborted / cancelled
    2   submit / health failure
    130 interrupted by user (after best-effort cancel)
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

EXP_DIR = Path(__file__).resolve().parents[1]
PROFILES_YAML = EXP_DIR / "profiles.yaml"
ERROR_PATTERNS = (
    "ERROR",
    "Traceback",
    "OutOfMemoryError",
    "CUDA out of memory",
    "Killed",
    "RuntimeError",
)


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


def _read_profile(name: str) -> dict:
    """Pure-stdlib YAML-light parser for the simple profile shape we have.
    We only need ``vram_mb`` (int) + ``exclusive_gpu`` (bool) + ``notes`` per
    entry, and a ``defaults:`` block. Avoid taking a yaml dependency.
    """
    text = PROFILES_YAML.read_text()
    # Tiny parser: walk indent-2 nested blocks. Sufficient for our flat schema.
    profiles: dict[str, dict] = {}
    defaults: dict = {}
    current: dict | None = None
    section: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            current = None
            continue
        if section == "defaults" and indent == 2 and ":" in stripped:
            k, _, v = stripped.partition(":")
            defaults[k.strip()] = _coerce(v.strip())
            continue
        if section == "experiments":
            if indent == 2 and stripped.endswith(":"):
                ent = stripped[:-1]
                profiles[ent] = {}
                current = profiles[ent]
                continue
            if indent == 4 and current is not None and ":" in stripped:
                k, _, v = stripped.partition(":")
                current[k.strip()] = _coerce(v.strip())
                continue
    entry = profiles.get(name) or defaults
    if name not in profiles:
        sys.stderr.write(
            f"[submit] profile '{name}' not in profiles.yaml; using defaults "
            f"(vram_mb={entry.get('vram_mb', 22000)}, exclusive_gpu={entry.get('exclusive_gpu', True)})\n"
        )
    return {
        "marginal_vram_mb": int(entry.get("vram_mb", 22000) or 0),
        "exclusive_gpu": bool(entry.get("exclusive_gpu", True)),
        "priority": "normal",
    }


def _coerce(s: str):
    s = s.strip()
    # Strip inline YAML comment from an unquoted scalar ("8000  # note" -> "8000").
    # Quoted values are left intact so a literal '#' inside quotes survives.
    if s[:1] not in ('"', "'"):
        hpos = s.find(" #")
        if hpos != -1:
            s = s[:hpos]
    s = s.strip().strip('"').strip("'")
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _build_eval_block(graph_name: str, kvs: list[str], overrides_path: str | None = None) -> dict:
    out: dict = {"graph_name": graph_name}
    # JSON overrides land first; scalar key=value pairs override them so an
    # explicit CLI arg still wins over the file.
    if overrides_path:
        with open(overrides_path) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"--eval-overrides {overrides_path}: expected a JSON object, "
                f"got {type(loaded).__name__}"
            )
        loaded.pop("graph_name", None)  # never let the file override the graph
        out.update(loaded)
    for arg in kvs:
        if "=" not in arg:
            sys.stderr.write(f"[submit] ignoring non-kv arg: {arg!r}\n")
            continue
        k, _, v = arg.partition("=")
        out[k.strip()] = _coerce(v)
    return out


def _print_state_change(label: str, state: str, ep_count: int, last: tuple) -> tuple:
    if (state, ep_count) != last:
        print(f"[run]   {state:<11}  episodes={ep_count}", flush=True)
        return (state, ep_count)
    return last


def _tail_stderr(run_dir: Path) -> None:
    log = run_dir / "stderr.log"
    if not log.exists():
        print(f"[run] no stderr.log at {log}", file=sys.stderr)
        return
    print("[run] subprocess stderr error markers (tail 30):", file=sys.stderr)
    matches: list[str] = []
    for line in log.read_text().splitlines():
        if any(p in line for p in ERROR_PATTERNS):
            matches.append(line)
    for line in matches[-30:]:
        print(line, file=sys.stderr)
    print(f"[run] full subprocess stderr: {log}", file=sys.stderr)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="submit.py", description=__doc__)
    p.add_argument("profile", help="profile name in profiles.yaml")
    p.add_argument("graph_name", help="graph name in workspace/graphs/")
    p.add_argument("kvs", nargs="*", help="eval block overrides as key=value")
    p.add_argument("--backend", default=DEFAULT_BACKEND, help="backend URL")
    p.add_argument(
        "--workspace",
        default=None,
        help=(
            "Absolute path to an active workspace overlay (e.g. an architect "
            "iter's active_workspace/). When set, the backend loads this dir "
            "on top of the frozen workspace — same-named nodesets/graphs/etc "
            "override frozen by name. Default = no overlay, run against "
            "frozen workspace only."
        ),
    )
    p.add_argument(
        "--eval-overrides",
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON object merged into the eval block before the "
            "key=value pairs. The only way to pass list/dict eval params "
            "(episode_indices, episode_selectors) — _coerce handles scalars "
            "only. 'graph_name' in the file is ignored."
        ),
    )
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument(
        "--max-wait-sec",
        type=float,
        default=86400.0,
        help="hard ceiling on polling time; 0 = forever",
    )
    args = p.parse_args(argv)

    # Health — probe a JSON endpoint under /api/ so this works whether the
    # caller hits the backend directly (:8000) or via the Vite frontend
    # proxy (:5173, which only forwards /api and /ws).
    try:
        status, body = _http("GET", f"{args.backend}/api/eval/v2/queue", timeout=5)
        if status != 200:
            raise RuntimeError(f"queue probe returned {status}")
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
        return die_unreachable(args.backend, e, prog="submit")

    sched = _read_profile(args.profile)
    try:
        eval_block = _build_eval_block(args.graph_name, args.kvs, args.eval_overrides)
    except (OSError, json.JSONDecodeError, RuntimeError) as e:
        sys.stderr.write(f"[submit] --eval-overrides load failed: {e}\n")
        return 2
    # Validate --workspace path up front for a cleaner error than the
    # backend's 400 response.
    workspace_arg: str | None = None
    if args.workspace:
        ws_path = Path(args.workspace).resolve()
        if not ws_path.is_dir():
            sys.stderr.write(
                f"[submit] --workspace path does not exist or is not a dir: {ws_path}\n"
            )
            return 2
        workspace_arg = str(ws_path)
    payload = {
        **eval_block,
        "via_subprocess": True,
        "marginal_vram_mb": sched["marginal_vram_mb"],
        "exclusive_gpu": sched["exclusive_gpu"],
        "priority": sched["priority"],
    }
    if workspace_arg:
        payload["active_workspace_dir"] = workspace_arg
    print(
        f"[submit] profile={args.profile} graph={args.graph_name} "
        f"vram={sched['marginal_vram_mb']}MB exclusive={sched['exclusive_gpu']}"
        + (f" active_workspace={workspace_arg}" if workspace_arg else "")
    )

    try:
        # /start synchronously spawns env-worker subprocesses inside
        # ensure_nodesets_for_graph(worker_count=N) before returning, so
        # large worker_count + heavy envs (habitat-sim) push this far past
        # the default 10s. 180s is comfortable for 10x hmeqa cold-start.
        status, body = _http("POST", f"{args.backend}/api/eval/v2/start", payload, timeout=180.0)
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace")
        print(f"[submit] /start returned {e.code}: {body_txt[:500]}", file=sys.stderr)
        return 2
    run_id = body.get("run_id")
    if not run_id:
        print(f"[submit] /start did not return run_id: {body}", file=sys.stderr)
        return 2
    print(f"[submit] run_id={run_id} initial_state={body.get('status')}")

    # parents[4]: bin/ → experiment/ → commands/ → .claude/ → repo root
    repo_root = Path(__file__).resolve().parents[4]
    run_dir = repo_root / "outputs" / "eval_runs" / run_id
    backend = args.backend

    interrupted = {"v": False}

    def _on_interrupt(signum, frame):
        if interrupted["v"]:
            return
        interrupted["v"] = True
        print(f"[run] interrupted — POST /runs/{run_id}/cancel", file=sys.stderr)
        try:
            _http("POST", f"{backend}/api/eval/v2/runs/{run_id}/cancel", {}, timeout=5)
        except Exception as e:
            print(f"[run] cancel failed: {e}", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    started = time.time()
    last = ("", -1)
    while True:
        if (run_dir / "_DONE").exists():
            break
        if args.max_wait_sec and time.time() - started > args.max_wait_sec:
            print(f"[run] poll timeout after {args.max_wait_sec}s", file=sys.stderr)
            break
        try:
            _, info = _http("GET", f"{backend}/api/eval/v2/runs/{run_id}", timeout=5)
            state = info.get("status", "?")
            eps = len(info.get("episodes") or [])
            last = _print_state_change(run_id, state, eps, last)
        except Exception as e:
            print(f"[run]   (poll error: {e})", file=sys.stderr)
        time.sleep(args.poll_interval)

    try:
        _, final = _http("GET", f"{backend}/api/eval/v2/runs/{run_id}", timeout=10)
    except Exception as e:
        print(f"[run] could not fetch final status: {e}", file=sys.stderr)
        return 1
    final_status = final.get("status", "?")
    print(f"[run] final status={final_status} elapsed={final.get('elapsed_sec')}s")
    aggregate = final.get("aggregate_metrics") or {}
    if aggregate:
        print(f"[run] aggregate: {aggregate}")

    if interrupted["v"]:
        return 130
    if final_status == "completed":
        return 0
    _tail_stderr(run_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
