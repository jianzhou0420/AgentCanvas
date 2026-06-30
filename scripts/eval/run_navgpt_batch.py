"""Run NavGPT-MP3D graph against N R2R val_unseen episodes via the backend API.

Each run is named `{EXEC_PREFIX}navgpt_ep{NN}_{instr_id}` so the log dir is
self-identifying.

Usage:
    python scripts/eval/run_navgpt_batch.py              # 10 episodes, stride=3, starting at 0
    python scripts/eval/run_navgpt_batch.py 20 0 3       # 20 episodes, starting at 0, stride=3
    python scripts/eval/run_navgpt_batch.py 10 0 1       # 10 consecutive episodes (no stride)

Env vars (used by /experiment/navgpt-mp3d skill):
    EXEC_PREFIX   prefix for execution_id → isolates log dirs per experiment
    SUMMARY_OUT   override summary CSV path (default outputs/navgpt_batch_summary.csv)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import request

BACKEND = "http://localhost:8000"
# scripts/eval/run_navgpt_batch.py → scripts → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = _REPO_ROOT / "workspace" / "graphs" / "vln" / "unverified" / "navgpt_mp3d.json"
SUMMARY_PATH = Path(
    os.environ.get("SUMMARY_OUT", str(_REPO_ROOT / "outputs" / "navgpt_batch_summary.csv"))
)
EXEC_PREFIX = os.environ.get("EXEC_PREFIX", "")

N_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# R2R has 3 instruction variants per path (instr_id = "{path}_{0|1|2}").
# Stride=3 skips sibling variants → each run covers a distinct path.
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
POLL_SEC = 3
TIMEOUT_SEC = 900  # per episode


def _req(method: str, path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        f"{BACKEND}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _slugify(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "ep"


def set_episode(idx: int) -> dict:
    _req("POST", "/api/env-panels/env_mp3d/field/episode_index", {"value": idx})
    return _req("POST", "/api/env-panels/env_mp3d/action/reset", {"params": {}})


def get_state() -> dict:
    return _req("GET", "/api/env-panels/env_mp3d/state")


def run_graph(graph: dict, execution_id: str) -> None:
    _req(
        "POST",
        "/api/navigate/run",
        {"loop_definition": graph, "execution_id": execution_id, "step_delay_ms": 200},
    )


def wait_for_done(execution_id: str) -> dict:
    t0 = time.time()
    while True:
        s = _req("GET", "/api/navigate/run/status")
        if s.get("execution_id") == execution_id and s.get("status") == "done":
            return s
        if time.time() - t0 > TIMEOUT_SEC:
            with contextlib.suppress(Exception):
                _req("POST", "/api/navigate/run/stop")
            return {"status": "timeout", "step": s.get("step"), "metrics": None}
        time.sleep(POLL_SEC)


def fetch_metrics(execution_id: str) -> dict:
    """Scan log.jsonl for the last Evaluate node firing + NavGPT stop signal."""
    p = _REPO_ROOT / "outputs" / "runs" / execution_id / "log.jsonl"
    if not p.exists():
        return {}
    last_metrics = {}
    stop_step = None
    try:
        with p.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nt = e.get("node_type", "")
                if nt == "env_mp3d__evaluate":
                    metrics_str = (e.get("outputs") or {}).get("metrics")
                    if isinstance(metrics_str, str) and metrics_str:
                        with contextlib.suppress(json.JSONDecodeError):
                            last_metrics = json.loads(metrics_str)
                if nt == "navgpt_mp3d_tools__parse_action":
                    inner = e.get("inner_log") or []
                    logs = {
                        item.get("key"): item.get("value")
                        for item in inner
                        if isinstance(item, dict)
                    }
                    if logs.get("is_stop") and stop_step is None:
                        stop_step = e.get("step")
    except OSError:
        pass
    return {"metrics": last_metrics, "stop_step": stop_step}


def main() -> int:
    graph = json.loads(GRAPH_PATH.read_text())
    print(f"[batch] running {N_EPS} episodes starting at index {START}")
    print(f"[batch] graph: {GRAPH_PATH}  nodes={len(graph['nodes'])}  edges={len(graph['edges'])}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_PATH.exists():
        SUMMARY_PATH.write_text(
            "episode_index,instr_id,path_len,execution_id,status,steps,stop_step,"
            "success,oracle_success,nav_error,spl,trajectory_length\n"
        )

    ok_count = 0
    succ_count = 0
    osucc_count = 0
    ne_sum = 0.0
    ne_n = 0
    for i in range(START, START + N_EPS * STRIDE, STRIDE):
        print(f"\n[batch] ── episode {i} ──")
        try:
            set_episode(i)
        except Exception as exc:
            print(f"[batch] set_episode({i}) failed: {exc}")
            continue

        st = get_state()
        ep = st.get("current_episode", {})
        instr_id = ep.get("instr_id", f"ep{i}")
        path_len = ep.get("path_len", 0)
        instr_preview = (ep.get("instruction", "") or "")[:80]
        print(f"[batch] instr_id={instr_id}  path_len={path_len}  instr={instr_preview!r}")

        exec_id = f"{EXEC_PREFIX}navgpt_ep{i:02d}_{_slugify(instr_id)}"
        print(f"[batch] execution_id={exec_id}")

        try:
            run_graph(graph, exec_id)
        except Exception as exc:
            print(f"[batch] run_graph failed: {exc}")
            continue

        final = wait_for_done(exec_id)
        status = final.get("status", "?")
        steps = final.get("step", -1)
        print(f"[batch] finished: status={status} steps={steps}")

        meta = fetch_metrics(exec_id)
        m = meta.get("metrics") or {}
        success = m.get("success", m.get("SR", ""))
        osucc = m.get("oracle_success", m.get("OSR", ""))
        ne = m.get("nav_error", m.get("NE", ""))
        spl = m.get("SPL", m.get("spl", ""))
        trajlen = m.get("trajectory_length", m.get("TL", ""))
        stop_step = meta.get("stop_step", "")

        if success is True or success == 1 or success == "1":
            succ_count += 1
        if osucc is True or osucc == 1 or osucc == "1":
            osucc_count += 1
        try:
            ne_sum += float(ne)
            ne_n += 1
        except (TypeError, ValueError):
            pass
        ok_count += 1 if status == "done" else 0

        print(
            f"[batch]   metrics: SR={success} OSR={osucc} NE={ne} SPL={spl} "
            f"TL={trajlen}  stop_step={stop_step}"
        )

        with SUMMARY_PATH.open("a") as f:
            f.write(
                f"{i},{instr_id},{path_len},{exec_id},{status},{steps},{stop_step},"
                f"{success},{osucc},{ne},{spl},{trajlen}\n"
            )

    total = N_EPS
    ne_mean = ne_sum / ne_n if ne_n else float("nan")
    print("\n[batch] ═══════════════════════════════════════════")
    print(f"[batch] finished {ok_count}/{total} cleanly")
    print(f"[batch] SR = {succ_count}/{total} = {100.0 * succ_count / total:.1f}%")
    print(f"[batch] OSR = {osucc_count}/{total} = {100.0 * osucc_count / total:.1f}%")
    print(f"[batch] mean NE = {ne_mean:.2f} m  (n={ne_n})")
    print(f"[batch] summary → {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
