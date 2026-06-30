"""LLM/VLM subsystem — multi-provider client, profiles, and CLI.

Public API::

    from app.llm import llm_complete, llm_complete_n, get_llm_config, LLMConfig
    from app.llm import vlm_complete, vlm_complete_n
    from app.llm import get_profile_store, LLMProfile, ProfileStore
    from app.llm import PROVIDER_REGISTRY, resolve_provider_config
"""

from __future__ import annotations

from .call import (
    API_TYPE_TO_LITELLM_PREFIX,
    LLMConfig,
    get_llm_config,
    llm_complete,
    llm_complete_n,
    vlm_complete,
    vlm_complete_n,
)
from .profiles import LLMProfile, ProfileStore, get_profile_store
from .providers import PROVIDER_REGISTRY, ProviderDef, resolve_provider_config

__all__ = [
    "API_TYPE_TO_LITELLM_PREFIX",
    "PROVIDER_REGISTRY",
    "LLMConfig",
    "LLMProfile",
    "ProfileStore",
    "ProviderDef",
    "get_llm_config",
    "get_profile_store",
    "llm_complete",
    "llm_complete_n",
    "resolve_provider_config",
    "vlm_complete",
    "vlm_complete_n",
]
