"""Profile → :class:`LLMConfig` resolution, plus the four historical call
helpers as thin wrappers over :func:`app.llm.complete.complete`.

Public API:
    get_llm_config()         — build LLMConfig from the active / a named profile
    get_llm_config_direct()  — build LLMConfig from an inline (provider, model)
    llm_complete()           — text-only completion → str | None
    llm_complete_n()         — text-only multi-sample → list[str]
    vlm_complete()           — text + images → str | None
    vlm_complete_n()         — text + images multi-sample → list[str]

The wrappers only reshape arguments into a :class:`Request`; every rule
(messages, rulebook, native ``n`` + fallback, usage, error policy) lives
in :mod:`app.llm.complete`.
"""

from __future__ import annotations

import logging

from .complete import Request, complete
from .profiles import LLMProfile, get_profile_store
from .providers import LLMConfig, resolve_provider_config

log = logging.getLogger("agentcanvas.llm")


def get_llm_config(profile_name: str = "") -> LLMConfig | None:
    """Build LLMConfig from a named profile, or the default (active) profile.

    The API key is read from the provider's standard env var (e.g.
    ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) — see
    :func:`app.llm.providers.get_provider_api_key`. Profiles never carry keys.

    Resolution:
      1. Explicit ``profile_name`` → load that profile
      2. Active profile in profiles.json → load active profile
      3. No usable profile → return ``None`` (mock mode)

    A profile is "usable" when its provider's env var is set — or it's
    Ollama, which doesn't need a key.
    """
    store = get_profile_store()
    name = profile_name or store.get_active()

    if not name:
        log.debug("LLM config: no active profile")
        return None

    profile = store.get(name)
    if profile is None:
        log.debug("LLM config: profile '%s' not found", name)
        return None

    cfg = resolve_provider_config(profile)
    if not cfg["api_key"] and cfg["api_type"] != "ollama":
        log.debug(
            "LLM config: profile '%s' has no API key in env (provider=%s)",
            name,
            profile.provider,
        )
        return None

    log.debug("LLM config: resolved from profile '%s' (model=%s)", name, cfg["model"])
    return LLMConfig(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_type=cfg["api_type"],
        litellm_prefix=cfg["litellm_prefix"],
        provider=profile.provider,
    )


def get_llm_config_direct(provider: str, model: str) -> LLMConfig | None:
    """Build LLMConfig from an inline ``(provider, model)`` reference —
    the node-pinned Direct mode. Same usability rule as profiles: returns
    ``None`` (mock mode) when the provider's key is missing (Ollama
    excepted)."""
    if not provider or not model:
        return None
    cfg = resolve_provider_config(LLMProfile(provider=provider, model=model))
    if not cfg["api_key"] and cfg["api_type"] != "ollama":
        log.debug("LLM config: direct (%s, %s) has no API key", provider, model)
        return None
    return LLMConfig(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_type=cfg["api_type"],
        litellm_prefix=cfg["litellm_prefix"],
        provider=provider,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Thin wrappers — the historical call shapes
# ══════════════════════════════════════════════════════════════════════════════


async def llm_complete(
    config: LLMConfig,
    messages: list[dict],
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> str | None:
    """Text-only completion over a ready message list (the user turn is
    its last entry). Returns the assistant text, or ``None`` on failure
    (raises under ``AGENTCANVAS_STRICT_ERRORS``)."""
    completion = await complete(
        config,
        Request(
            history=list(messages),
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        ),
    )
    return completion.text


async def llm_complete_n(
    config: LLMConfig,
    messages: list[dict],
    n: int,
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> list[str]:
    """Multi-sample text completion: up to ``n`` texts, empty on failure."""
    completion = await complete(
        config,
        Request(
            history=list(messages),
            system_prompt=system_prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        ),
    )
    return completion.texts


async def vlm_complete(
    config: LLMConfig,
    prompt: str,
    images: list[str],
    *,
    image_labels: list[str] | None = None,
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    detail: str = "low",
    prior_messages: list[dict] | None = None,
    stop: list[str] | None = None,
    mime: str = "image/png",
) -> str | None:
    """Completion over a text prompt plus base64 images (``data:`` prefix
    excluded). ``prior_messages`` is text-only history inserted before the
    current turn; ``detail`` / ``mime`` are the image hint and data-URL
    type. Returns the assistant text, or ``None`` on failure."""
    completion = await complete(
        config,
        Request(
            prompt=prompt,
            images=list(images),
            image_labels=image_labels,
            system_prompt=system_prompt,
            history=list(prior_messages or []),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            image_detail=detail,
            image_mime=mime,
        ),
    )
    return completion.text


async def vlm_complete_n(
    config: LLMConfig,
    prompt: str,
    images: list[str],
    n: int,
    *,
    image_labels: list[str] | None = None,
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    detail: str = "low",
    stop: list[str] | None = None,
    mime: str = "image/png",
) -> list[str]:
    """Multi-sample vision completion: up to ``n`` texts, empty on failure."""
    completion = await complete(
        config,
        Request(
            prompt=prompt,
            images=list(images),
            image_labels=image_labels,
            system_prompt=system_prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            image_detail=detail,
            image_mime=mime,
        ),
    )
    return completion.texts
