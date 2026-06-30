"""Provider registry — maps provider IDs to API defaults.

API keys are NEVER stored on profiles. Each provider has a standard
environment-variable name (the de-facto convention each vendor's SDK
uses). The runtime reads the key from that env var at call time, so
``profiles.json`` can be checked into git or shared without leaking
secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderDef:
    label: str
    base_url: str
    api_type: str  # "openai" | "anthropic" | "google" | "ollama"
    default_model: str
    litellm_prefix: str  # litellm model prefix (e.g. "openai", "anthropic", "gemini")
    env_var: str  # standard env var name for the API key (e.g. "OPENAI_API_KEY")


PROVIDER_REGISTRY: dict[str, ProviderDef] = {
    # --- OpenAI protocol ---
    "openai": ProviderDef(
        "OpenAI", "https://api.openai.com/v1", "openai", "gpt-4o", "openai", "OPENAI_API_KEY"
    ),
    "openrouter": ProviderDef(
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "openai",
        "openai/gpt-4o",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "together": ProviderDef(
        "Together AI",
        "https://api.together.xyz/v1",
        "openai",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together_ai",
        "TOGETHERAI_API_KEY",
    ),
    "moonshot": ProviderDef(
        "Moonshot AI",
        "https://api.moonshot.cn/v1",
        "openai",
        "kimi-latest-8k",
        "openai",
        "MOONSHOT_API_KEY",
    ),
    "mistral": ProviderDef(
        "Mistral AI",
        "https://api.mistral.ai/v1",
        "openai",
        "mistral-large-latest",
        "mistral",
        "MISTRAL_API_KEY",
    ),
    "xai": ProviderDef(
        "xAI (Grok)", "https://api.x.ai/v1", "openai", "grok-2-latest", "xai", "XAI_API_KEY"
    ),
    "nvidia": ProviderDef(
        "NVIDIA NIM",
        "https://integrate.api.nvidia.com/v1",
        "openai",
        "meta/llama-3.3-70b-instruct",
        "nvidia_nim",
        "NVIDIA_NIM_API_KEY",
    ),
    "huggingface": ProviderDef(
        "Hugging Face",
        "https://router.huggingface.co/v1",
        "openai",
        "meta-llama/Llama-3.3-70B-Instruct",
        "huggingface",
        "HF_TOKEN",
    ),
    "deepseek": ProviderDef(
        "DeepSeek",
        "https://api.deepseek.com/v1",
        "openai",
        "deepseek-chat",
        "deepseek",
        "DEEPSEEK_API_KEY",
    ),
    "qianfan": ProviderDef(
        "Baidu Qianfan",
        "https://qianfan.baidubce.com/v2",
        "openai",
        "ernie-4.0-8k",
        "openai",
        "QIANFAN_API_KEY",
    ),
    "modelstudio": ProviderDef(
        "Alibaba ModelStudio",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai",
        "qwen-max",
        "openai",
        "DASHSCOPE_API_KEY",
    ),
    "volcengine": ProviderDef(
        "Volcengine (ByteDance)",
        "https://ark.cn-beijing.volces.com/api/v3",
        "openai",
        "doubao-pro-32k",
        "openai",
        "ARK_API_KEY",
    ),
    "byteplus": ProviderDef(
        "BytePlus ModelArts",
        "https://ark.ap-southeast.bytepluses.com/api/v3",
        "openai",
        "doubao-pro-32k",
        "openai",
        "BYTEPLUS_API_KEY",
    ),
    "venice": ProviderDef(
        "Venice AI",
        "https://api.venice.ai/api/v1",
        "openai",
        "qwen3-235b",
        "openai",
        "VENICE_API_KEY",
    ),
    "vllm": ProviderDef(
        "vLLM", "http://localhost:8000/v1", "openai", "default", "openai", "VLLM_API_KEY"
    ),
    "sglang": ProviderDef(
        "SGLang", "http://localhost:30000/v1", "openai", "default", "openai", "SGLANG_API_KEY"
    ),
    # --- Anthropic protocol ---
    "anthropic": ProviderDef(
        "Anthropic",
        "https://api.anthropic.com",
        "anthropic",
        "claude-sonnet-4-6",
        "anthropic",
        "ANTHROPIC_API_KEY",
    ),
    "minimax": ProviderDef(
        "MiniMax",
        "https://api.minimaxi.com/v1",
        "openai",
        "MiniMax-M2.5",
        "openai",
        "MINIMAX_API_KEY",
    ),
    "kimi-coding": ProviderDef(
        "Kimi Coding",
        "https://api.moonshot.cn/anthropic",
        "anthropic",
        "kimi-k2.5",
        "anthropic",
        "MOONSHOT_API_KEY",
    ),
    "xiaomi": ProviderDef(
        "Xiaomi MiMo",
        "https://api.xiaomimimo.com/anthropic",
        "anthropic",
        "mimo-v2-flash",
        "anthropic",
        "XIAOMI_API_KEY",
    ),
    "synthetic": ProviderDef(
        "Synthetic",
        "https://api.synthetic.new/anthropic",
        "anthropic",
        "synthetic-latest",
        "anthropic",
        "SYNTHETIC_API_KEY",
    ),
    # --- Google protocol ---
    "google": ProviderDef(
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        "google",
        "gemini-2.5-flash",
        "gemini",
        "GEMINI_API_KEY",
    ),
    # --- Ollama protocol ---
    "ollama": ProviderDef(
        "Ollama (local)",
        "http://localhost:11434",
        "ollama",
        "llama3.2",
        "ollama_chat",
        "",  # local server — no key
    ),
    # --- Custom ---
    "custom": ProviderDef("Custom", "", "openai", "", "openai", "AGENTCANVAS_API_KEY"),
}


def get_provider_api_key(provider_id: str) -> str:
    """Read the API key for ``provider_id`` from its standard env var.

    Returns ``""`` if no env var is set or the provider is unknown. Ollama
    always returns ``""`` (no key needed).
    """
    reg = PROVIDER_REGISTRY.get(provider_id)
    if reg is None or not reg.env_var:
        return ""
    return os.environ.get(reg.env_var, "")


def resolve_provider_config(profile) -> dict:
    """Merge a profile with its registry defaults, returning a dict with
    keys: api_key, base_url, model, api_type, litellm_prefix.

    ``api_key`` is read from the standard env var for the profile's
    provider (e.g. ``OPENAI_API_KEY``). Profiles never carry keys.
    """
    reg = PROVIDER_REGISTRY.get(profile.provider)
    base_url = profile.base_url or (reg.base_url if reg else "")
    api_type = profile.api_type or (reg.api_type if reg else "openai")
    model = profile.model or (reg.default_model if reg else "")
    litellm_prefix = reg.litellm_prefix if reg else "openai"
    return {
        "api_key": get_provider_api_key(profile.provider),
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_type": api_type,
        "litellm_prefix": litellm_prefix,
    }
