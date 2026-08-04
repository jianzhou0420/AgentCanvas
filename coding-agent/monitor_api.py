"""Monitor API — the read-side contract over coding-agent run artifacts.

coding-agent is standalone: every run — std cells via stdrun.py and UI runs
via ac_support/uirun.py alike — writes ONE artifact format, and this module
is the single place that format and its scoring semantics are defined for
consumers. Any monitor (the AgentCanvas Coding-Agent Monitor today, anything
else tomorrow) reads runs through this surface and never reaches into driver
internals.

Run-dir layout — ``OUTPUT_ROOTS[harness] / {run_name}/``:

    summary.json          incrementally updated by the driver; the
                          ``episodes`` list holds one record per episode
                          (fields used by scoring below; authoritative
                          producer: driver.run_cell / EventSink)
    episode_{i}.jsonl     per-event stream, EventSink vocabulary
                          (single writer by construction)
    raw/episode_{i}.jsonl harness-native message dump
    live_{i}/             per-episode frames the agent actually saw
    driver.log            driver stdout (UI runs)

Scoring rule (the honest-SR denominator, frozen 2026-08-01): an episode is
SCORED iff its ``metrics.success`` exists and it is not a limit-exceeded
casualty; turn-exhaustion counts as failure, a rate-limited episode that
never navigated is excluded.

Spawn contract for UI runs — ``ac_support/uirun.py`` (run names ``ui_*``,
off-board by construction)::

    uirun.py --episodes 0-9 --split rand100 --max-turns 80
             --server-url http://127.0.0.1:9200 --run-name ui_...
             [--model <sdk model id>]   # blank -> the CLI's default model

This directory is hyphenated (not importable by module name); load by path::

    spec = importlib.util.spec_from_file_location(
        "coding_agent_monitor_api", REPO_ROOT / "coding-agent" / "monitor_api.py")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from cells import OUTPUT_ROOTS  # single source of the per-harness output roots

REPO_ROOT = _HERE.parent
UIRUN_PATH = _HERE / "ac_support" / "uirun.py"
RUN_STATS_PATH = _HERE / "ac_support" / "run_stats.py"


def is_limit_exceeded(ep: dict[str, Any]) -> bool:
    """A rate-limit / 'limit exceeded' casualty: the run errored WITHOUT taking a
    single navigation step (error set + env_steps 0). It never really attempted
    the task, so it counts as neither a pass nor a fail — excluded from SR and
    shown as ⚠, not ✅/❌. A genuine turn-exhausted run DID navigate (env_steps
    > 0) and is a real failure that stays in the denominator."""
    return bool(ep.get("error")) and not ((ep.get("agent") or {}).get("env_steps") or 0)


def display_aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute SR/SPL/stop from raw episodes for display — always honest
    regardless of what the driver wrote to summary.json (a live run's stored
    aggregate can lag or predate the current scoring rule). Turn-exhausted =
    failure (counts); limit-exceeded = excluded."""
    scored = [
        e for e in episodes
        if (e.get("metrics") or {}).get("success") is not None and not is_limit_exceeded(e)
    ]
    agg: dict[str, Any] = {"episode_count": len(scored)}
    if not scored:
        return agg

    def _mean_metric(key: str) -> float | None:
        vals = [
            float((e.get("metrics") or {})[key])
            for e in scored
            if isinstance((e.get("metrics") or {}).get(key), (int, float))
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    for key in ("success", "spl", "ndtw", "oracle_success", "distance_to_goal"):
        val = _mean_metric(key)
        if val is not None:
            agg[key] = val
    agg["stop_rate"] = round(
        sum(1 for e in scored if (e.get("agent") or {}).get("called_stop")) / len(scored), 4
    )
    return agg
