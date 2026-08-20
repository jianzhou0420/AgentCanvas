"""API provider registry — one place that knows every platform we call.

The harnesses used to hard-code both the key variables and the name
heuristics (mini_swe._check_auth matched substrings; the DashScope key had
to be loaded INTO ``OPENAI_API_KEY`` because litellm's openai route reads
that variable regardless of vendor). This module gives each vendor a
canonical key variable plus explicit legacy fallbacks, so:

- ``DASHSCOPE_API_KEY`` finally exists — the OpenAI column and the qwen API
  column can run in the same shell;
- the old recipe (DashScope key in ``OPENAI_API_KEY``) keeps working
  unchanged via the fallback chain, so recorded runs stay reproducible.

``vendor_of()`` mirrors the resolution order the harnesses already used —
local prefixes first, then explicit api_base, then name substrings. It is
deliberately conservative: an unrecognized model returns None and the
caller keeps its historical behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Vendor:
    name: str
    key_env: str = ""                       # canonical key variable ("" = keyless)
    key_fallbacks: tuple[str, ...] = ()     # legacy variables honored if canonical unset
    wire: str = "openai-chat"               # native wire format (documentation-grade)


VENDORS: dict[str, Vendor] = {
    "anthropic": Vendor("anthropic", "ANTHROPIC_API_KEY", wire="anthropic-messages"),
    "openai": Vendor("openai", "OPENAI_API_KEY", wire="openai-chat"),
    # DashScope compatible-mode (qwen API columns). Canonical variable is new
    # (2026-08-20); the OPENAI_API_KEY fallback keeps the historical
    # "load the DashScope key into OPENAI_API_KEY for the run shell" recipe
    # working byte-identically when the canonical one is unset.
    "dashscope": Vendor("dashscope", "DASHSCOPE_API_KEY", ("OPENAI_API_KEY",)),
    # Locally served — no key involved; lifecycle owned by the mini adapter.
    "ollama": Vendor("ollama", wire="ollama"),
    "hosted_vllm": Vendor("hosted_vllm"),
}


def vendor_of(model_id: str, extra: dict | None = None) -> Vendor | None:
    """Resolve a harness-facing model id (+cell extra) to a Vendor.

    Order mirrors the pre-registry heuristics exactly: local prefixes win,
    an explicit DashScope api_base wins over the openai/ prefix, name
    substrings decide the rest. None = caller keeps historical behavior.
    """
    model = (model_id or "").lower()
    api_base = str((extra or {}).get("api_base", "")).lower()
    if model.startswith("ollama"):
        return VENDORS["ollama"]
    if model.startswith("hosted_vllm/"):
        return VENDORS["hosted_vllm"]
    if "dashscope" in api_base:
        return VENDORS["dashscope"]
    if any(s in model for s in ("anthropic", "claude")):
        return VENDORS["anthropic"]
    if model.startswith(("gpt", "openai/")):
        return VENDORS["openai"]
    return None


def resolve_key(vendor: Vendor) -> tuple[str | None, str | None]:
    """(key value, variable it came from) — canonical first, then fallbacks.

    Returns (None, None) for keyless vendors and when nothing is set; the
    caller decides whether that is fatal (mini_swe raises before episode 0).
    """
    for var in (vendor.key_env, *vendor.key_fallbacks):
        if var and os.environ.get(var):
            return os.environ[var], var
    return None, None
