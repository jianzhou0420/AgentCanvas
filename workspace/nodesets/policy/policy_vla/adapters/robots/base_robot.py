"""Abstract base class for robot adaptors.

Vendored from vlaworkspace/adaptors/robots/base_robot.py — no upstream import.
"""

from __future__ import annotations

import abc

from ..canonical import CanonicalDict, CanonicalInfo


class RobotAdaptor(abc.ABC):
    """Abstract base class for robot adaptors.

    Data flow:
        Dataset -> dataset_to_canonical() -> canonical obs
        Env     -> env_to_canonical()     -> canonical obs (no actions)
        canonical action -> canonical_to_env() -> env action
    """

    @abc.abstractmethod
    def get_canonical_info(self) -> CanonicalInfo: ...

    @abc.abstractmethod
    def dataset_to_canonical(self, data: dict) -> CanonicalDict: ...

    @abc.abstractmethod
    def env_to_canonical(self, data: dict) -> CanonicalDict: ...

    @abc.abstractmethod
    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict: ...

    @abc.abstractmethod
    def get_state_dim(self) -> int: ...

    @abc.abstractmethod
    def get_action_dim(self) -> int: ...

    @abc.abstractmethod
    def get_norm_stats_keys(self) -> tuple[str, ...]: ...

    @abc.abstractmethod
    def env_obs(self) -> dict: ...

    @abc.abstractmethod
    def env_action(self) -> dict: ...

    @abc.abstractmethod
    def datasets(self) -> dict: ...
