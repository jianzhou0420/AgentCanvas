"""Abstract VlnEnvAdaptor — stage 1 + stage 5 of the 5-stage adapter pipeline.

Mirrors ``env_adapter/robots/base_robot.py`` but for VLN navigation
envs (R2R-CE, RxR-CE, …). Stage 1 packs raw_obs into canonical;
stage 5 picks an env action index from canonical_action.

Subclasses live in this folder as one ``<env>.py`` file each (filename is
the dropdown option). Each module must export:
  - one VlnEnvAdaptor subclass
  - module-level ``DEFAULT_KWARGS: dict``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .canonical import (
    CanonicalDict,
    CanonicalNavInfo,
)


class VlnEnvAdaptor(ABC):
    """Abstract base for env-family adapters."""

    env_family: str = ""  # "r2rce" | "rxrce" — concrete subclass overrides

    @abstractmethod
    def get_canonical_info(self) -> CanonicalNavInfo:
        """Static metadata about this env family."""

    @abstractmethod
    def env_to_canonical(
        self, raw_obs: dict[str, Any], instruction: Any = None
    ) -> CanonicalDict:
        """Wrap a raw Habitat observation into a CanonicalDict.

        ``instruction`` is the tokenized instruction from the dedicated
        instruction port ({"tokens", "text"}); ``None`` falls back to whatever
        the env adapter finds bundled in ``raw_obs``."""

    @abstractmethod
    def canonical_to_env(self, canonical_action: CanonicalDict) -> int:
        """Pick an env action index from a CanonicalDict[action]."""
