"""Agent loop execution — LoopRunner + GraphExecutor + GraphExecutor.

The frontend sends a LoopDefinition (nodes + edges + config) and this module
executes it via GraphExecutor.
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
