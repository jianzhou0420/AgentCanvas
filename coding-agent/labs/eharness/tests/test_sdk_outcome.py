"""§14.13 — ClaudeSdkAdapter outcome classification must reach SessionOutcome.

The adapter computes a careful error taxonomy (rate-limit retry, scored
truncations, broken sessions) — the regression was constructing
SessionOutcome from raw is_error instead, which re-flagged every clean
success and broke the retry whitelist. These tests pin classify_outcome()
AND that run()'s SessionOutcome uses it verbatim (source-level check, so
the test runs without the SDK installed).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harnesses.claude_sdk import ClaudeSdkAdapter, classify_outcome


def R(subtype, is_error=True, result=""):
    return SimpleNamespace(subtype=subtype, is_error=is_error, result=result)


def test_success_with_is_error_is_scored():
    # SDK sets is_error=True even on clean success (fable ep40) — must score.
    assert classify_outcome(R("success")) is None


def test_max_turns_is_scored_truncation():
    assert classify_outcome(R("error_max_turns")) is None


def test_max_budget_is_scored_truncation():
    # outside the whitelist the driver would retry the priciest episodes
    assert classify_outcome(R("error_max_budget_usd")) is None


def test_rate_limited_is_retryable():
    out = classify_outcome(R("success", result="temporarily limiting requests"))
    assert out == "rate_limited"


def test_execution_error_is_broken_session():
    assert classify_outcome(R("error_during_execution")) == "sdk result error_during_execution"
    assert classify_outcome(R(None)) == "sdk result is_error"


def test_clean_no_error_flag():
    assert classify_outcome(R("success", is_error=False)) is None


def test_run_uses_classifier_not_raw_is_error():
    src = inspect.getsource(ClaudeSdkAdapter.run)
    assert "classify_outcome(result_msg)" in src
    assert 'error=("sdk result is_error"' not in src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
