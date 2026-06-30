"""Graph discovery — reads workspace/graphs/ directly (no backend call).

Per-graph experiment profiles (``{graph}.exp.yaml``) are surfaced
alongside as advisory metadata so the LLM can pick a sensible
``worker_count`` / ``primary_metric`` without round-tripping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# tools/graph.py → tools → mcp_server → agentcanvas → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_DIR = _REPO_ROOT / "workspace" / "graphs"
EXP_PROFILES_DIR = _REPO_ROOT / "workspace" / "architect" / "exp_profiles"


def graph_list() -> dict[str, Any]:
    """List all graphs in workspace/graphs/, with optional exp profile from workspace/architect/exp_profiles/."""
    graphs: list[dict[str, Any]] = []
    if not GRAPHS_DIR.exists():
        return {"graphs": [], "graphs_dir": str(GRAPHS_DIR), "warning": "graphs dir not found"}

    for json_path in sorted(GRAPHS_DIR.glob("*.json")):
        name = json_path.stem
        entry: dict[str, Any] = {"name": name, "path": str(json_path)}
        profile_path = EXP_PROFILES_DIR / f"{name}.yaml"
        if profile_path.exists():
            try:
                profile = yaml.safe_load(profile_path.read_text()) or {}
                # Surface only the operationally-useful keys; skip rationale
                # prose so the LLM context stays small.
                entry["profile"] = {
                    k: profile.get(k)
                    for k in (
                        "split",
                        "worker_count",
                        "step_budget",
                        "per_step_budget_sec",
                        "primary_metric",
                        "secondary_metrics",
                    )
                    if k in profile
                }
            except yaml.YAMLError as e:
                entry["profile_error"] = str(e)
        graphs.append(entry)

    return {"graphs": graphs, "graphs_dir": str(GRAPHS_DIR)}
