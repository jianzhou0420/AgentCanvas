"""Rulebook finalize semantics — the outbound-invariant contract."""

from __future__ import annotations

from .rulebook import finalize_params, get_capabilities


def test_gpt5_temperature_locked():
    params, adj = finalize_params("openai", "gpt-5-nano", {"temperature": 0.7})
    assert params["temperature"] == 1.0
    assert len(adj) == 1 and "temperature" in adj[0]


def test_gpt5_unset_temperature_stays_unset():
    # Two-state: unset is never sent, so the lock has nothing to force.
    params, adj = finalize_params("openai", "gpt-5-nano", {})
    assert "temperature" not in params
    assert adj == []


def test_gpt5_min_hint_is_advisory_only():
    params, adj = finalize_params("openai", "gpt-5", {"max_tokens": 500})
    assert params["max_tokens"] == 500
    assert adj == []


def test_anthropic_required_max_tokens_injected():
    params, adj = finalize_params("anthropic", "claude-sonnet-4-6", {})
    assert params["max_tokens"] == 4096
    assert any("max_tokens" in a for a in adj)


def test_anthropic_explicit_max_tokens_untouched():
    params, _ = finalize_params("anthropic", "claude-sonnet-4-6", {"max_tokens": 800})
    assert params["max_tokens"] == 800


def test_anthropic_temperature_clamped():
    params, adj = finalize_params("anthropic", "claude-sonnet-4-6", {"temperature": 1.5})
    assert params["temperature"] == 1.0
    assert any("range" in a for a in adj)


def test_anthropic_unsupported_dropped():
    params, adj = finalize_params("anthropic", "claude-sonnet-4-6", {"n": 5})
    assert "n" not in params
    assert any("unsupported" in a for a in adj)


def test_unknown_model_passes_through():
    params, adj = finalize_params("openai", "gpt-4o", {"temperature": 0.2, "max_tokens": 64})
    assert params == {"temperature": 0.2, "max_tokens": 64}
    assert adj == []


def test_unknown_provider_passes_through():
    params, adj = finalize_params("", "whatever", {"temperature": 0.2})
    assert params == {"temperature": 0.2}
    assert adj == []


def test_capabilities_shape():
    caps = get_capabilities("openai", "gpt-5-mini")
    assert caps["temperature"][0]["kind"] == "locked"
    assert caps["temperature"][0]["value"] == 1.0
    assert caps["max_tokens"][0]["kind"] == "min_hint"
    assert get_capabilities("openai", "gpt-4o") == {}
