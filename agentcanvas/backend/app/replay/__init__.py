from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.replay.interface import (
    BaseReplayParser,
    GenericReplayParser,
    ReplayEpisode,
    ReplayStep,
)

if TYPE_CHECKING:
    from app.replay.renderer_client import ReplayRendererClient

__all__ = [
    "BaseReplayParser",
    "GenericReplayParser",
    "ReplayEpisode",
    "ReplayRendererClient",
    "ReplayStep",
]


def __getattr__(name: str) -> Any:
    # Lazy — renderer_client needs httpx, which a pure-local Graph SDK run
    # (registry → interface) must not require.
    if name == "ReplayRendererClient":
        from app.replay.renderer_client import ReplayRendererClient

        return ReplayRendererClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
