"""LLM/VLM subsystem — multi-provider client, profiles, keystore, and CLI.

Public API::

    from app.llm import complete, complete_sync, Request, Completion
    from app.llm import llm_complete, llm_complete_n, get_llm_config, LLMConfig
    from app.llm import vlm_complete, vlm_complete_n, get_llm_config_direct
    from app.llm import get_profile_store, LLMProfile, ProfileStore
    from app.llm import PROVIDER_REGISTRY, resolve_provider_config
    from app.llm import get_key_store, get_provider_key_source
    from app.llm import finalize_params, get_capabilities
"""

from __future__ import annotations

from .call import (
    get_llm_config,
    get_llm_config_direct,
    llm_complete,
    llm_complete_n,
    vlm_complete,
    vlm_complete_n,
)
from .complete import Completion, Request, complete, complete_sync
from .keystore import KeyStore, get_key_store
from .profiles import LLMProfile, ProfileStore, get_profile_store
from .providers import (
    API_TYPE_TO_LITELLM_PREFIX,
    PROVIDER_REGISTRY,
    LLMConfig,
    ProviderDef,
    get_provider_key_source,
    resolve_provider_config,
)
from .rulebook import finalize_params, get_capabilities

__all__ = [
    "API_TYPE_TO_LITELLM_PREFIX",
    "PROVIDER_REGISTRY",
    "Completion",
    "KeyStore",
    "LLMConfig",
    "LLMProfile",
    "ProfileStore",
    "ProviderDef",
    "Request",
    "complete",
    "complete_sync",
    "finalize_params",
    "get_capabilities",
    "get_key_store",
    "get_llm_config",
    "get_llm_config_direct",
    "get_profile_store",
    "get_provider_key_source",
    "llm_complete",
    "llm_complete_n",
    "resolve_provider_config",
    "vlm_complete",
    "vlm_complete_n",
]
