"""Agent loop execution — GraphExecutor behind graph_run, plus the runners.

``graph_run`` is the one place a graph is prepared against the registry and
run once (``prepare_graph`` / ``GraphRun``); ``loop_runner`` (canvas) and
``eval_batch`` (batch eval) are adapters over it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyEntry:
    """Metadata for a loadable neural policy checkpoint."""

    id: str
    name: str
    checkpoint: str
    config: str
