"""Runs tools — list past eval runs by reading the filesystem.

Mirrors ``agentcanvas/backend/app/api/execution/eval_storage.py:list_runs()``
schema, but is independent of any backend (and so survives backend death).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# tools/runs.py → tools → mcp_server → agentcanvas → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_RUNS_DIR = _REPO_ROOT / "outputs" / "eval_runs"


def eval_runs_list(limit: int = 50) -> dict[str, Any]:
    """List recent eval runs (newest first), excluding per-episode detail.

    Each entry has the same fields as ``eval_export`` minus ``episodes``,
    plus ``episode_count_saved`` for quick at-a-glance progress checks.
    """
    if not EVAL_RUNS_DIR.exists():
        return {"runs": [], "runs_dir": str(EVAL_RUNS_DIR)}

    subdirs = [d for d in EVAL_RUNS_DIR.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    results: list[dict[str, Any]] = []
    for sub in subdirs[:limit]:
        path = sub / "summary.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            summary = {k: v for k, v in data.items() if k != "episodes"}
            summary["episode_count_saved"] = len(data.get("episodes", []))
            results.append(summary)
        except (OSError, ValueError) as exc:
            results.append(
                {
                    "run_id": sub.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"runs": results, "runs_dir": str(EVAL_RUNS_DIR)}
