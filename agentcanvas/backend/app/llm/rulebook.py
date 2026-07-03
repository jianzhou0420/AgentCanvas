"""Parameter rulebook — verified per-(provider, model-family) facts.

The one licensed source of "this model constrains that parameter"
knowledge. Two consumers, same data:

* :func:`finalize_params` — call-time normalization in ``call.py``,
  guaranteeing no request that violates a locally-known rule leaves the
  process (fix what is fixable, record every adjustment).
* :func:`get_capabilities` — the JSON the frontend renders at edit time
  (locked sliders, clamped ranges, unsupported notes), served by
  ``GET /api/providers/{id}/capabilities``.

Contents are VERIFIED FACTS ONLY — a model with no matching entry passes
through untouched. litellm already performs some transport translation
natively (e.g. reasoning-model token-param renames); entries here are
the traps it does not cover.

Verdict kinds:

* ``locked``      — the parameter accepts exactly one value; force it.
* ``required``    — the API rejects requests without it; inject the
                    fallback when unset.
* ``range``       — clamp to ``(min, max)``.
* ``unsupported`` — not honored by the provider; dropped from the
                    request (``note`` says what compensates, if anything).
* ``min_hint``    — advisory floor; never adjusted, surfaced in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any


@dataclass(frozen=True)
class ParamRule:
    param: str
    kind: str  # "locked" | "required" | "range" | "unsupported" | "min_hint"
    value: Any = None
    note: str = ""


# OpenAI o-series reasoning models share the gpt-5 family's constraints:
# default-only temperature, and a reasoning budget that eats small
# max_tokens values (tokens are billed as completion tokens).
_OPENAI_REASONING_RULES = [
    ParamRule(
        "temperature",
        "locked",
        1.0,
        "o-series reasoning models accept only the default temperature",
    ),
    ParamRule(
        "max_tokens",
        "min_hint",
        2000,
        "reasoning tokens count against the cap; small caps yield empty output",
    ),
]

# provider_id -> model pattern (fnmatch) -> rules. Later patterns override
# earlier ones per (param, kind) when both match.
RULEBOOK: dict[str, dict[str, list[ParamRule]]] = {
    "openai": {
        "o1*": _OPENAI_REASONING_RULES,
        "o3*": _OPENAI_REASONING_RULES,
        "o4*": _OPENAI_REASONING_RULES,
        "gpt-5*": [
            ParamRule(
                "temperature",
                "locked",
                1.0,
                "gpt-5 family accepts only the default; other values silently return empty",
            ),
            ParamRule(
                "max_tokens",
                "min_hint",
                2000,
                "below ~2000 the reasoning budget swallows the output (silent empty)",
            ),
        ],
    },
    "anthropic": {
        "*": [
            ParamRule("max_tokens", "required", 4096, "Anthropic rejects requests without it"),
            ParamRule("temperature", "range", (0.0, 1.0), "Anthropic temperature range is 0–1"),
            ParamRule(
                "n",
                "unsupported",
                None,
                "no native multi-sample; shortfall is filled by concurrent single-sample calls",
            ),
            ParamRule(
                "image_detail",
                "unsupported",
                None,
                "OpenAI-only hint; ignored by Anthropic",
            ),
        ],
    },
    "google": {
        "*": [
            ParamRule(
                "temperature",
                "range",
                (0.0, 2.0),
                "Gemini temperature range is 0–2",
            ),
            ParamRule(
                "image_detail",
                "unsupported",
                None,
                "OpenAI-only hint; ignored by Gemini",
            ),
        ],
    },
    "deepseek": {
        "deepseek-chat*": [
            ParamRule(
                "temperature",
                "range",
                (0.0, 2.0),
                "DeepSeek chat temperature range is 0–2",
            ),
        ],
        "deepseek-reasoner*": [
            ParamRule(
                "temperature",
                "unsupported",
                None,
                "accepted but ignored by deepseek-reasoner",
            ),
        ],
    },
    "ollama": {
        "*": [
            ParamRule(
                "n",
                "unsupported",
                None,
                "no native multi-sample; shortfall is filled by concurrent single-sample calls",
            ),
            ParamRule(
                "image_detail",
                "unsupported",
                None,
                "OpenAI-only hint; ignored by Ollama",
            ),
        ],
    },
}


def rules_for(provider_id: str, model: str) -> list[ParamRule]:
    """All rules whose model pattern matches, later patterns overriding
    earlier ones per (param, kind)."""
    provider_rules = RULEBOOK.get(provider_id, {})
    merged: dict[tuple[str, str], ParamRule] = {}
    for pattern, rules in provider_rules.items():
        if fnmatch(model, pattern):
            for rule in rules:
                merged[(rule.param, rule.kind)] = rule
    return list(merged.values())


def get_capabilities(provider_id: str, model: str) -> dict:
    """JSON-ready verdicts for one (provider, model) — ``{param: [verdicts]}``."""
    out: dict[str, list[dict]] = {}
    for rule in rules_for(provider_id, model):
        out.setdefault(rule.param, []).append(
            {"kind": rule.kind, "value": rule.value, "note": rule.note}
        )
    return out


def finalize_params(provider_id: str, model: str, params: dict) -> tuple[dict, list[str]]:
    """Normalize ``params`` (explicitly-set values only; unset keys absent)
    against the rulebook.

    Returns ``(finalized_params, adjustments)`` where every entry in
    ``adjustments`` is a human-readable record of one deviation from the
    caller's intent. ``min_hint`` rules never adjust — they are advisory.
    """
    finalized = dict(params)
    adjustments: list[str] = []
    for rule in rules_for(provider_id, model):
        current = finalized.get(rule.param)
        if rule.kind == "locked":
            if rule.param in finalized and current != rule.value:
                finalized[rule.param] = rule.value
                adjustments.append(
                    f"{rule.param} {current} → {rule.value} (locked: {rule.note})"
                )
        elif rule.kind == "required":
            if rule.param not in finalized or finalized[rule.param] is None:
                finalized[rule.param] = rule.value
                adjustments.append(
                    f"{rule.param} unset → {rule.value} (required: {rule.note})"
                )
        elif rule.kind == "range":
            if current is not None:
                lo, hi = rule.value
                clamped = min(max(current, lo), hi)
                if clamped != current:
                    finalized[rule.param] = clamped
                    adjustments.append(
                        f"{rule.param} {current} → {clamped} (range {lo}–{hi})"
                    )
        elif rule.kind == "unsupported":
            if rule.param in finalized:
                dropped = finalized.pop(rule.param)
                adjustments.append(
                    f"{rule.param}={dropped} dropped (unsupported: {rule.note})"
                )
    return finalized, adjustments
