"""Strict-mode error refactor — smoke tests.

Verifies AGENTCANVAS_STRICT_ERRORS=1 makes LLM/VLM API failures surface
(raise) instead of being swallowed (return None / []). Targets the
focus_llm class of silent-failure bug: a misconfigured llmCall node hit
litellm.UnsupportedParamsError on every call for two full architect
iterations because the exception sink returned an empty string and
downstream consumers had `if x else ""` fallbacks.

Run:
    python -m pytest agentcanvas/backend/app/test_strict_errors.py -v
or:
    cd agentcanvas/backend && python -m app.test_strict_errors
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from .llm.call import (
    LLMConfig,
    _strict_errors_enabled,
    llm_complete,
    llm_complete_n,
    vlm_complete,
    vlm_complete_n,
)

_FAKE_CFG = LLMConfig(
    api_key="sk-test",
    base_url="",
    model="gpt-5-mini",
    api_type="openai",
    litellm_prefix="openai",
)


class _FocusLLMBug(Exception):
    """Stand-in for litellm.UnsupportedParamsError from the focus_llm bug."""


def _clear_strict() -> None:
    os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)


def _set_strict() -> None:
    os.environ["AGENTCANVAS_STRICT_ERRORS"] = "1"


# ══════════════════════════════════════════════════════════════════════
# Helper: env flag truthy/falsy parsing
# ══════════════════════════════════════════════════════════════════════


def test_strict_flag_truthy_values() -> None:
    for v in ("1", "true", "TRUE", "yes", "on"):
        os.environ["AGENTCANVAS_STRICT_ERRORS"] = v
        assert _strict_errors_enabled() is True, f"failed for {v!r}"
    for v in ("0", "false", "", "no", "off", "random"):
        os.environ["AGENTCANVAS_STRICT_ERRORS"] = v
        assert _strict_errors_enabled() is False, f"failed for {v!r}"
    _clear_strict()
    assert _strict_errors_enabled() is False


# ══════════════════════════════════════════════════════════════════════
# llm_complete (single-sample text)
# ══════════════════════════════════════════════════════════════════════


def test_llm_complete_swallows_when_strict_off() -> None:
    _clear_strict()
    with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
        result = asyncio.run(llm_complete(_FAKE_CFG, [{"role": "user", "content": "x"}]))
    assert result is None, f"expected None on swallowed error, got {result!r}"


def test_llm_complete_raises_when_strict_on() -> None:
    _set_strict()
    try:
        with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
            raised = False
            try:
                asyncio.run(llm_complete(_FAKE_CFG, [{"role": "user", "content": "x"}]))
            except _FocusLLMBug:
                raised = True
            assert raised, "expected llm_complete to raise in strict mode"
    finally:
        _clear_strict()


# ══════════════════════════════════════════════════════════════════════
# llm_complete_n (multi-sample text)
# ══════════════════════════════════════════════════════════════════════


def test_llm_complete_n_swallows_when_strict_off() -> None:
    _clear_strict()
    with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
        result = asyncio.run(llm_complete_n(_FAKE_CFG, [{"role": "user", "content": "x"}], n=3))
    assert result == [], f"expected [] on swallowed error, got {result!r}"


def test_llm_complete_n_raises_when_strict_on() -> None:
    _set_strict()
    try:
        with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
            raised = False
            try:
                asyncio.run(llm_complete_n(_FAKE_CFG, [{"role": "user", "content": "x"}], n=3))
            except _FocusLLMBug:
                raised = True
            assert raised, "expected llm_complete_n to raise in strict mode"
    finally:
        _clear_strict()


# ══════════════════════════════════════════════════════════════════════
# vlm_complete (text + images) — uses non-empty images list to hit the
# VLM path; empty images delegates to llm_complete, already covered.
# ══════════════════════════════════════════════════════════════════════


def test_vlm_complete_swallows_when_strict_off() -> None:
    _clear_strict()
    with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
        result = asyncio.run(vlm_complete(_FAKE_CFG, "prompt", images=["fake-b64-image-payload"]))
    assert result is None, f"expected None on swallowed error, got {result!r}"


def test_vlm_complete_raises_when_strict_on() -> None:
    _set_strict()
    try:
        with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
            raised = False
            try:
                asyncio.run(vlm_complete(_FAKE_CFG, "prompt", images=["fake-b64-image-payload"]))
            except _FocusLLMBug:
                raised = True
            assert raised, "expected vlm_complete to raise in strict mode"
    finally:
        _clear_strict()


# ══════════════════════════════════════════════════════════════════════
# vlm_complete_n (multi-sample text + images) — provider-native ``n`` with
# a concurrent single-sample fallback for providers that ignore it.
# ══════════════════════════════════════════════════════════════════════


def _make_resp(text: str) -> object:
    """Minimal litellm-shaped response with a single choice."""

    class _Msg:
        content = text

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


def test_vlm_complete_n_swallows_when_strict_off() -> None:
    # Native ``n`` raises AND every fallback single-sample call raises →
    # vlm_complete_n must still return [] rather than propagating.
    _clear_strict()
    with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
        result = asyncio.run(
            vlm_complete_n(_FAKE_CFG, "prompt", ["fake-b64-image-payload"], n=3)
        )
    assert result == [], f"expected [] on swallowed error, got {result!r}"


def test_vlm_complete_n_raises_when_strict_on() -> None:
    # In strict mode the genuine API error surfaces through the fallback
    # single-sample calls.
    _set_strict()
    try:
        with patch("litellm.acompletion", side_effect=_FocusLLMBug("boom")):
            raised = False
            try:
                asyncio.run(
                    vlm_complete_n(
                        _FAKE_CFG, "prompt", ["fake-b64-image-payload"], n=3
                    )
                )
            except _FocusLLMBug:
                raised = True
            assert raised, "expected vlm_complete_n to raise in strict mode"
    finally:
        _clear_strict()


def test_vlm_complete_n_fallback_fills_to_n() -> None:
    # Provider rejects native ``n`` for vision but answers single-sample
    # calls fine → vlm_complete_n must fill all n candidates via fallback.
    _clear_strict()
    counter = {"n": 0}

    def _side_effect(*_args, **kwargs):
        if int(kwargs.get("n", 1) or 1) > 1:
            raise _FocusLLMBug("n unsupported for vision request")
        counter["n"] += 1
        return _make_resp(f"candidate-{counter['n']}")

    with patch("litellm.acompletion", side_effect=_side_effect):
        result = asyncio.run(
            vlm_complete_n(_FAKE_CFG, "prompt", ["fake-b64-image-payload"], n=3)
        )
    assert len(result) == 3, f"expected 3 candidates via fallback, got {result!r}"
    assert len(set(result)) == 3, f"expected distinct candidates, got {result!r}"


# ══════════════════════════════════════════════════════════════════════
# Standalone driver — runs all tests without pytest
# ══════════════════════════════════════════════════════════════════════


def _main() -> int:
    tests = [
        test_strict_flag_truthy_values,
        test_llm_complete_swallows_when_strict_off,
        test_llm_complete_raises_when_strict_on,
        test_llm_complete_n_swallows_when_strict_off,
        test_llm_complete_n_raises_when_strict_on,
        test_vlm_complete_swallows_when_strict_off,
        test_vlm_complete_raises_when_strict_on,
        test_vlm_complete_n_swallows_when_strict_off,
        test_vlm_complete_n_raises_when_strict_on,
        test_vlm_complete_n_fallback_fills_to_n,
    ]
    failed: list[str] = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__} — {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ERROR {t.__name__} — {type(e).__name__}: {e}")
            failed.append(t.__name__)
    print()
    print(f"Result: {len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
