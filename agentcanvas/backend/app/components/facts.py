"""Registry facts — the read-only view that admission and attribution read.

The scheduler's resource accounting needs a handful of facts about a
nodeset (its parallelism contract, whether it is loaded and where, its
source tree hash for calibration invalidation, its author presets) and
about the registry (the PIDs of the servers it owns, the frozen
workspace dir). This module is that seam: the registry implements it,
the resource tracker consumes it, tests substitute it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodesetFacts:
    """What admission needs to know about one nodeset."""

    name: str
    parallelism: str  # "shared" | "replicated" (ADR-server-003)
    loaded: bool = False  # loaded in this registry: local, hosted, or remote-attached
    hosted: bool = False  # has a server (auto_host / container / remote); False = in-process
    source_file: str | None = None
    source_hash: str | None = None  # content hash of the source tree; None = unknown
    expected_vram_mb: int | None = None  # author presets (BaseNodeSet.expected_*_mb)
    expected_ram_mb: int | None = None

    @property
    def loaded_local(self) -> bool:
        """Loaded in-process — the case per-PID attribution cannot see."""
        return self.loaded and not self.hosted

    @property
    def hints(self) -> dict[str, int | None]:
        """Author presets per resource, the estimator's fallback pricing."""
        return {"vram": self.expected_vram_mb, "ram": self.expected_ram_mb}


class RegistryFacts(ABC):
    """Read-only registry view for resource admission and attribution."""

    @abstractmethod
    def nodeset_facts(self, name: str) -> NodesetFacts:
        """Facts for ``name``; an undiscovered nodeset still gets its
        parallelism fallback and ``loaded=False``."""

    @abstractmethod
    def server_pids(self) -> dict[int, str]:
        """PID → nodeset name for the auto_host servers this registry owns."""

    @property
    @abstractmethod
    def frozen_dir(self) -> Path:
        """The frozen ``workspace/`` dir overlays are resolved against."""
