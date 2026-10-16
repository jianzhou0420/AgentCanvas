"""One completion module: a :class:`Request` in, assistant texts out.

Everything that used to be copied across the four ``*_complete`` helpers
lives here once — the message builder, the parameter rulebook applied to
*every* parameter, provider-native ``n`` with a concurrent single-sample
fallback on every path, usage accounting, timeout policy and the error
policy (``AGENTCANVAS_STRICT_ERRORS``).

The one thing that varies is the wire: :class:`Transport` is the seam,
:class:`LiteLLMTransport` the production adapter, and tests bring a fake.

Public interface::

    completion = await complete(config, Request(prompt="hi", n=3))
    completion.texts        # up to n non-empty assistant texts
    completion.text         # the first, or None
    completion.adjustments  # every rulebook deviation from the caller's intent

    complete_sync(config, request)   # blocking twin for vendored sync code
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import litellm

from ..standard.telemetry import _current_node_usage
from .providers import LLMConfig
from .rulebook import finalize_params

log = logging.getLogger("agentcanvas.llm")

# Suppress litellm's verbose internal logging and telemetry
litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def strict_errors_enabled() -> bool:
    """Read fresh on every call so env-worker subprocesses inherit the
    backend's launch-time setting without caching. Set
    ``AGENTCANVAS_STRICT_ERRORS=1`` at backend launch to make API failures
    raise instead of returning a fallback (``None`` / ``[]``)."""
    return os.environ.get("AGENTCANVAS_STRICT_ERRORS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Interface
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Request:
    """One completion: a conversation plus the sampling knobs.

    The current user turn is ``prompt`` (text) and/or ``images`` (base64
    payloads, no ``data:`` prefix). ``history`` holds the prior turns, in
    OpenAI message form; a caller that already has the user turn in its
    message list leaves ``prompt`` and ``images`` empty. ``system_prompt``
    goes first when set.

    Images are interleaved after the prompt text, each preceded by its
    ``image_labels[i]`` caption when one is given (MapGPT's ``gpt_infer``
    shape). ``image_detail`` is the OpenAI ``detail`` hint and
    ``image_mime`` the data-URL type (``"image/jpeg"`` for JPEG payloads).

    ``temperature`` / ``max_tokens`` are two-state: ``None`` is omitted from
    the request so the vendor default applies. ``n`` asks for that many
    samples; ``stop`` sequences pass verbatim; ``retries`` are the
    transport's own retries (off by default).
    """

    prompt: str = ""
    images: list[str] = field(default_factory=list)
    image_labels: list[str] | None = None
    system_prompt: str = ""
    history: list[dict] = field(default_factory=list)
    n: int = 1
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    image_detail: str = "low"
    image_mime: str = "image/png"
    retries: int = 0


@dataclass
class Completion:
    """What a :class:`Request` came back as."""

    texts: list[str]
    adjustments: list[str] = field(default_factory=list)

    @property
    def text(self) -> str | None:
        """The first sample, or ``None`` when nothing came back."""
        return self.texts[0] if self.texts else None


@dataclass
class Reply:
    """One transport round trip: the non-empty assistant texts (one per
    returned choice) and the usage the provider reported for it."""

    texts: list[str]
    usage: dict | None = None


class Transport(ABC):
    """The wire. ``messages`` is the finished OpenAI-form conversation and
    ``params`` the finalized request parameters (``temperature``,
    ``max_tokens``, ``n``, ``stop`` — whatever survived the rulebook).
    Raise on any failure; the module owns the error policy."""

    @abstractmethod
    async def complete(
        self,
        config: LLMConfig,
        messages: list[dict],
        params: dict,
        *,
        timeout: float,
        retries: int,
    ) -> Reply: ...


# ══════════════════════════════════════════════════════════════════════════════
# The module
# ══════════════════════════════════════════════════════════════════════════════


async def complete(
    config: LLMConfig,
    request: Request,
    *,
    transport: Transport | None = None,
) -> Completion:
    """Run one :class:`Request` against ``config`` and return its texts.

    Up to ``request.n`` non-empty texts come back. Provider-native ``n`` is
    attempted first when the rulebook allows it (the prompt is then billed
    once); any shortfall — a provider that ignores ``n``, returns fewer
    choices, or rejects the native call — is filled with concurrent
    single-sample calls, so the caller sees the same contract on every
    provider.

    Failures follow the strict-errors policy: with
    ``AGENTCANVAS_STRICT_ERRORS`` set the exception propagates, otherwise
    it is logged and the failed sample is simply missing (``texts`` may be
    empty). Token usage of every round trip lands in the per-node bucket
    the executor set (``_current_node_usage``).
    """
    wire = transport or LITELLM
    n = max(1, int(request.n or 1))
    timeout = _timeout_for(config, request)
    try:
        messages = build_messages(request)
    except ValueError as exc:
        log.error("LLM request invalid: %s", exc)
        if strict_errors_enabled():
            raise
        return Completion(texts=[])
    params, adjustments = _finalize(config, request, n)
    for adj in adjustments:
        log.info("finalize [%s/%s]: %s", config.provider or "?", config.model, adj)
    native_n = params.pop("n", None)
    detail = params.pop("image_detail", None)
    if request.images:
        messages = _apply_detail(messages, detail)

    async def one() -> str | None:
        try:
            reply = await wire.complete(
                config, messages, params, timeout=timeout, retries=request.retries
            )
        except Exception as exc:
            log.error("LLM API call failed: %s: %s", type(exc).__name__, exc)
            if strict_errors_enabled():
                raise
            return None
        _accumulate_usage(reply.usage)
        return reply.texts[0] if reply.texts else None

    texts: list[str] = []
    if native_n is not None:
        try:
            reply = await wire.complete(
                config,
                messages,
                {**params, "n": native_n},
                timeout=timeout,
                retries=request.retries,
            )
        except Exception as exc:
            # Native ``n`` rejected — not fatal: the single-sample fill
            # below re-surfaces a genuine API error.
            log.warning(
                "LLM multi-sample (n=%d) native call failed (%s: %s); "
                "falling back to concurrent single-sample calls",
                native_n,
                type(exc).__name__,
                exc,
            )
        else:
            _accumulate_usage(reply.usage)
            texts = [t for t in reply.texts if t][:n]

    shortfall = n - len(texts)
    if shortfall > 0:
        filled = await asyncio.gather(*[one() for _ in range(shortfall)])
        texts.extend(t for t in filled if t)
    return Completion(texts=texts, adjustments=adjustments)


def complete_sync(
    config: LLMConfig,
    request: Request,
    *,
    transport: Transport | None = None,
) -> Completion:
    """Blocking twin of :func:`complete` for vendored synchronous code.

    Vendored method code (VoxPoser's LMP) is called from inside a node's
    ``forward`` while the executor's event loop is running on this thread,
    so the call runs on a worker thread with its own loop. The caller's
    ContextVars travel with it, so usage still lands in the node's bucket.
    """
    coro = complete(config, request, transport=transport)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


# ── Message builder ────────────────────────────────────────────────────


def build_messages(request: Request) -> list[dict]:
    """The OpenAI-form conversation for ``request``: system prompt, prior
    turns, then the current user turn when ``prompt`` or ``images`` is
    set. Raises ``ValueError`` when there is no user turn at all."""
    messages: list[dict] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.extend(request.history)
    if request.images:
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        labels = request.image_labels or []
        for i, img in enumerate(request.images):
            label = labels[i] if i < len(labels) else ""
            if label:
                content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{request.image_mime};base64,{img}"},
                }
            )
        messages.append({"role": "user", "content": content})
    elif request.prompt:
        messages.append({"role": "user", "content": request.prompt})
    elif not request.history:
        raise ValueError("Request has no user turn (no prompt, no images, no history)")
    return messages


def _apply_detail(messages: list[dict], detail: str | None) -> list[dict]:
    """Stamp the OpenAI ``detail`` hint onto the current turn's images —
    only when the rulebook let it through (``None`` = dropped)."""
    if not detail:
        return messages
    turn = messages[-1]
    if not isinstance(turn.get("content"), list):
        return messages
    for part in turn["content"]:
        if part.get("type") == "image_url":
            part["image_url"]["detail"] = detail
    return messages


def _finalize(config: LLMConfig, request: Request, n: int) -> tuple[dict, list[str]]:
    """Only explicitly-set values enter the request; the rulebook then
    fixes what it knows (locked / required / range / unsupported) and
    records every adjustment."""
    params: dict = {}
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.max_tokens is not None:
        params["max_tokens"] = int(request.max_tokens)
    if request.stop:
        params["stop"] = list(request.stop)
    if n > 1:
        params["n"] = n
    if request.images:
        params["image_detail"] = request.image_detail
    return finalize_params(config.provider, config.model, params)


def _timeout_for(config: LLMConfig, request: Request) -> float:
    if config.api_type == "ollama":
        return 120.0
    return 90.0 if request.images else 60.0


# ── Per-node usage hook ────────────────────────────────────────────────
# The graph executor sets ``_current_node_usage`` to a fresh accumulator
# before invoking ``BaseCanvasNode.execute()``. Every round trip reached
# during that ``execute()`` adds its usage here; the executor emits the
# bucket as one log entry per node — nodesets never plumb usage themselves.


def _accumulate_usage(u: dict | None) -> None:
    bucket = _current_node_usage.get()
    if bucket is None:
        return
    u = u or {}
    bucket["calls"] = int(bucket.get("calls", 0)) + 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
        bucket[key] = int(bucket.get(key, 0)) + int(u.get(key, 0) or 0)
    bucket["usd_cost"] = float(bucket.get("usd_cost", 0.0)) + float(u.get("usd_cost", 0.0) or 0.0)
    if u.get("model") and not bucket.get("model"):
        bucket["model"] = u["model"]


# ══════════════════════════════════════════════════════════════════════════════
# litellm adapter
# ══════════════════════════════════════════════════════════════════════════════


class LiteLLMTransport(Transport):
    """The production wire: ``litellm.acompletion`` against the provider's
    endpoint; litellm translates the OpenAI-form conversation (images
    included) to each provider's native format."""

    async def complete(
        self,
        config: LLMConfig,
        messages: list[dict],
        params: dict,
        *,
        timeout: float,
        retries: int,
    ) -> Reply:
        response = await litellm.acompletion(
            model=f"{config.litellm_prefix}/{config.model}",
            messages=messages,
            api_key=config.api_key or None,
            api_base=config.base_url or None,
            timeout=timeout,
            num_retries=retries,
            **params,
        )
        texts: list[str] = []
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content:
                texts.append(content)
        return Reply(texts=texts, usage=_extract_usage(response))


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


LITELLM = LiteLLMTransport()
