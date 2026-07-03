"""Async LLM/VLM client — litellm-backed, multi-provider (100+ providers).

Public API:
    llm_complete()    — text-only chat completion
    llm_complete_n()  — text-only multi-sample completion (OpenAI ``n``)
    vlm_complete()    — multimodal completion (text + images)
    vlm_complete_n()  — multimodal multi-sample completion (OpenAI ``n``)
    get_llm_config()  — build LLMConfig from active profile
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from dataclasses import dataclass

import litellm

from .profiles import LLMProfile, get_profile_store
from .providers import resolve_provider_config
from .rulebook import finalize_params

log = logging.getLogger("agentcanvas.llm")


def _strict_errors_enabled() -> bool:
    # Read fresh on every call so env-worker subprocesses can inherit the
    # backend's launch-time setting without caching. Set
    # AGENTCANVAS_STRICT_ERRORS=1 at backend launch to make LLM/VLM API
    # failures raise instead of returning a fallback (None / []).
    return os.environ.get("AGENTCANVAS_STRICT_ERRORS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ── Per-node usage hook ────────────────────────────────────────────────
# The graph executor sets this contextvar to a fresh accumulator dict
# before invoking ``BaseCanvasNode.execute()``. Every ``llm_complete``
# / ``llm_complete_n`` / ``vlm_complete`` call reached during that
# ``execute()`` writes its usage stats into the bucket. The executor
# reads + emits the bucket as one log entry per node — nodesets never
# need to plumb usage themselves.
_current_node_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_current_node_usage", default=None
)


def _accumulate_usage(response: object) -> None:
    """Add this response's usage into the active per-node bucket, if any."""
    bucket = _current_node_usage.get()
    if bucket is None:
        return
    u = _extract_usage(response)
    bucket["calls"] = int(bucket.get("calls", 0)) + 1
    bucket["prompt_tokens"] = int(bucket.get("prompt_tokens", 0)) + u["prompt_tokens"]
    bucket["completion_tokens"] = int(bucket.get("completion_tokens", 0)) + u["completion_tokens"]
    bucket["total_tokens"] = int(bucket.get("total_tokens", 0)) + u["total_tokens"]
    bucket["cached_tokens"] = int(bucket.get("cached_tokens", 0)) + u["cached_tokens"]
    bucket["usd_cost"] = float(bucket.get("usd_cost", 0.0)) + u["usd_cost"]
    if u["model"] and not bucket.get("model"):
        bucket["model"] = u["model"]


# Suppress litellm's verbose internal logging and telemetry
litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Fallback mapping when no profile is available (env-only mode).
# Exported: also used by key_validator.py.
API_TYPE_TO_LITELLM_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "ollama": "ollama_chat",
}


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    api_type: str  # "openai" | "anthropic" | "google" | "ollama"
    litellm_prefix: str  # litellm model prefix (e.g. "openai", "gemini")
    provider: str = ""  # PROVIDER_REGISTRY key — keys the parameter rulebook


def _to_litellm_model(config: LLMConfig) -> str:
    """Map LLMConfig to a litellm model string (e.g. ``'openai/gpt-4o'``)."""
    return f"{config.litellm_prefix}/{config.model}"


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


def _finalized_sampling_params(
    config: LLMConfig,
    temperature: float | None,
    max_tokens: int | None,
) -> dict:
    """Two-state params → finalize. Only explicitly-set values enter the
    request; the rulebook then fixes what it knows (locked / required /
    range), recording every adjustment in the log."""
    params: dict = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = int(max_tokens)
    params, adjustments = finalize_params(config.provider, config.model, params)
    for adj in adjustments:
        log.info("finalize [%s/%s]: %s", config.provider or "?", config.model, adj)
    return params


# ══════════════════════════════════════════════════════════════════════════════
# LLM completion — text-only
# ══════════════════════════════════════════════════════════════════════════════


def _extract_usage(response: object) -> dict:
    """Pull token-usage info off a litellm response into a flat dict.

    Returns ``{model, prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, usd_cost}``. Missing fields default to 0 / "". Safe on
    any response shape — best-effort, never raises.

    ``usd_cost`` uses ``litellm.completion_cost`` which carries an internal
    model→price table; returns 0.0 if litellm doesn't recognise the model
    (e.g. local Ollama, custom vLLM endpoints).
    """
    out = {
        "model": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "usd_cost": 0.0,
    }
    try:
        out["model"] = getattr(response, "model", "") or ""
        u = getattr(response, "usage", None)
        if u is None:
            return out
        out["prompt_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
        out["completion_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
        out["total_tokens"] = int(getattr(u, "total_tokens", 0) or 0)
        ptd = getattr(u, "prompt_tokens_details", None)
        if ptd is not None:
            cached = getattr(ptd, "cached_tokens", None)
            if cached is None and isinstance(ptd, dict):
                cached = ptd.get("cached_tokens", 0)
            out["cached_tokens"] = int(cached or 0)
    except Exception:
        pass
    try:
        cost = litellm.completion_cost(completion_response=response)
        out["usd_cost"] = float(cost or 0.0)
    except Exception:
        pass
    return out


async def llm_complete(
    config: LLMConfig,
    messages: list[dict],
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> str | None:
    """Call LLM API. Returns assistant text or None on failure.

    ``temperature`` / ``max_tokens`` are two-state: ``None`` (default)
    means the parameter is omitted from the request entirely, so the
    vendor's own default applies. Explicit values pass through the
    rulebook (``finalize_params``) before the wire.

    Token usage is accumulated into the per-node bucket set by the graph
    executor (via the ``_current_node_usage`` ContextVar). Nodesets do
    not plumb usage manually — the executor emits one ``usage`` log
    entry per node firing.
    """
    try:
        all_messages: list[dict] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        timeout = 120.0 if config.api_type == "ollama" else 60.0

        extra: dict = _finalized_sampling_params(config, temperature, max_tokens)
        if stop:
            extra["stop"] = stop

        response = await litellm.acompletion(
            model=_to_litellm_model(config),
            messages=all_messages,
            api_key=config.api_key or None,
            api_base=config.base_url or None,
            timeout=timeout,
            num_retries=0,
            **extra,
        )
        _accumulate_usage(response)
        return response.choices[0].message.content
    except Exception as exc:
        log.error("LLM API call failed: %s: %s", type(exc).__name__, exc)
        if _strict_errors_enabled():
            raise
        return None


async def llm_complete_n(
    config: LLMConfig,
    messages: list[dict],
    n: int,
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> list[str]:
    """Multi-sample LLM call (OpenAI ``n`` parameter).

    Returns a list of length up to ``n``. Each entry is the text content
    of one ``choices[i]``; failed / empty choices are dropped. Empty list
    on API error.

    Usage is accumulated into the per-node bucket via ``_current_node_usage``
    (same hook as :func:`llm_complete`). On a multi-sample call,
    ``completion_tokens`` is the total across all ``n`` samples;
    ``prompt_tokens`` is paid once.
    """
    if n <= 1:
        text = await llm_complete(
            config,
            messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return [text] if text else []

    try:
        all_messages: list[dict] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        timeout = 120.0 if config.api_type == "ollama" else 60.0

        extra: dict = _finalized_sampling_params(config, temperature, max_tokens)
        extra["n"] = n
        if stop:
            extra["stop"] = stop

        response = await litellm.acompletion(
            model=_to_litellm_model(config),
            messages=all_messages,
            api_key=config.api_key or None,
            api_base=config.base_url or None,
            timeout=timeout,
            num_retries=0,
            **extra,
        )
        _accumulate_usage(response)
        out: list[str] = []
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content:
                out.append(content)
        return out
    except Exception as exc:
        log.error("LLM API call (n=%d) failed: %s: %s", n, type(exc).__name__, exc)
        if _strict_errors_enabled():
            raise
        return []


# ══════════════════════════════════════════════════════════════════════════════
# VLM (Vision-Language) completion — text + images
# ══════════════════════════════════════════════════════════════════════════════


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
    """Call VLM API with text prompt and base64-encoded images.

    Args:
        config: LLM provider configuration.
        prompt: Text prompt to send alongside images.
        images: List of base64-encoded PNG/JPEG strings (no ``data:`` prefix).
        image_labels: Optional parallel list of per-image captions; when
            position ``i`` has a non-empty string, ``{"type":"text","text":
            image_labels[i]}`` is emitted immediately before ``images[i]``.
            Short/missing/empty entries emit the image unlabeled — this
            matches MapGPT's ``gpt_infer`` interleave shape.
        system_prompt: Optional system prompt.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        detail: OpenAI image ``detail`` hint (``"low"`` / ``"high"`` / ``"auto"``).
            Defaults to ``"low"`` (token-cheap) to preserve the historical
            behaviour of every existing caller; ports that need high-resolution
            visual grounding (e.g. Three-Step Nav, whose upstream uses ``high``)
            pass ``detail="high"`` explicitly.
        prior_messages: Optional conversation history inserted between the
            system message and the current user turn (llmCall conversation
            mode). History entries are plain text turns; images ride only
            the current turn.
        stop: Optional stop sequences, forwarded verbatim.
        mime: data-URL MIME type for the images (``"image/png"`` default;
            pass ``"image/jpeg"`` when the base64 payloads are JPEG — e.g.
            Open-Nav-family ports whose upstream sends JPEG re-encodes).

    Returns:
        Assistant response text, or ``None`` on failure.
    """
    if not images:
        return await llm_complete(
            config,
            [*(prior_messages or []), {"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    try:
        # Build OpenAI-format multimodal message — litellm translates
        # to provider-native formats (Anthropic, Google, Ollama) automatically.
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images):
            label = image_labels[i] if image_labels and i < len(image_labels) else ""
            if label:
                content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img}", "detail": detail},
                }
            )

        all_messages: list[dict] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(prior_messages or [])
        all_messages.append({"role": "user", "content": content})

        timeout = 120.0 if config.api_type == "ollama" else 90.0

        extra: dict = _finalized_sampling_params(config, temperature, max_tokens)
        if stop:
            extra["stop"] = stop

        response = await litellm.acompletion(
            model=_to_litellm_model(config),
            messages=all_messages,
            api_key=config.api_key or None,
            api_base=config.base_url or None,
            timeout=timeout,
            num_retries=0,
            **extra,
        )
        _accumulate_usage(response)
        return response.choices[0].message.content
    except Exception as exc:
        log.error("VLM API call failed: %s: %s", type(exc).__name__, exc)
        if _strict_errors_enabled():
            raise
        return None


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
    """Multi-sample VLM call — vision counterpart to :func:`llm_complete_n`.

    Returns a list of length up to ``n``. Each entry is the text content of
    one candidate; failed / empty candidates are dropped. Empty list on a
    total failure.

    Provider-native ``n`` is attempted first: the (large) image prompt is
    then billed once for all ``n`` samples. Providers that ignore ``n`` for
    vision requests — notably Anthropic — collapse the response to a single
    choice; this is detected and the shortfall is filled with concurrent
    single-sample :func:`vlm_complete` calls, so the caller always gets up
    to ``n`` candidates regardless of provider support. A genuine API error
    (auth, timeout, …) still surfaces through the fallback calls, which
    re-raise under ``AGENTCANVAS_STRICT_ERRORS``.

    Usage is accumulated into the per-node bucket via ``_current_node_usage``
    (same hook as :func:`vlm_complete`).
    """
    if n <= 1:
        text = await vlm_complete(
            config,
            prompt,
            images,
            image_labels=image_labels,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            detail=detail,
            stop=stop,
            mime=mime,
        )
        return [text] if text else []

    # No images → this is really a text-only multi-sample call.
    if not images:
        return await llm_complete_n(
            config,
            [{"role": "user", "content": prompt}],
            n=n,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    # Build the multimodal message once (same interleave shape as
    # ``vlm_complete``) — reused for the native-``n`` attempt.
    content: list[dict] = []
    for i, img in enumerate(images):
        label = image_labels[i] if image_labels and i < len(image_labels) else ""
        if label:
            content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img}", "detail": detail},
            }
        )
    content.append({"type": "text", "text": prompt})

    all_messages: list[dict] = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.append({"role": "user", "content": content})

    timeout = 120.0 if config.api_type == "ollama" else 90.0

    out: list[str] = []
    try:
        response = await litellm.acompletion(
            model=_to_litellm_model(config),
            messages=all_messages,
            api_key=config.api_key or None,
            api_base=config.base_url or None,
            timeout=timeout,
            num_retries=0,
            n=n,
            **({"stop": stop} if stop else {}),
            **_finalized_sampling_params(config, temperature, max_tokens),
        )
        _accumulate_usage(response)
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            text = getattr(msg, "content", None) if msg else None
            if text:
                out.append(text)
    except Exception as exc:
        # Native ``n`` rejected by the provider — not fatal: fall through
        # to the concurrent single-sample path below. A real API error
        # will re-surface there.
        log.warning(
            "VLM multi-sample (n=%d) native call failed (%s: %s); "
            "falling back to concurrent single-sample calls",
            n,
            type(exc).__name__,
            exc,
        )

    if len(out) >= n:
        return out[:n]

    # Native ``n`` unsupported or partial — fill the shortfall with
    # concurrent single-sample VLM calls.
    shortfall = n - len(out)
    extra = await asyncio.gather(
        *[
            vlm_complete(
                config,
                prompt,
                images,
                image_labels=image_labels,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                detail=detail,
                stop=stop,
                mime=mime,
            )
            for _ in range(shortfall)
        ]
    )
    out.extend(text for text in extra if text)
    return out
