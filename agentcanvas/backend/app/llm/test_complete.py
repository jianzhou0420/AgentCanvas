"""The completion module through its interface, on a fake transport.

Covers what the four historical helpers each used to re-implement:
message shape, the rulebook applied to every parameter, native ``n`` with
the single-sample fallback, usage accounting, timeout policy and the
strict-errors policy. The strict-mode cases guard the focus_llm class of
silent failure — a misconfigured llmCall node hit
``litellm.UnsupportedParamsError`` on every call for two architect
iterations because the exception sink returned an empty string.

Run: ``cd agentcanvas/backend && python -m pytest app/llm/test_complete.py -v``
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ..standard.telemetry import _current_node_usage, new_usage_bucket
from .call import llm_complete, llm_complete_n, vlm_complete, vlm_complete_n
from .complete import (
    Completion,
    Reply,
    Request,
    Transport,
    build_messages,
    complete,
    complete_sync,
    strict_errors_enabled,
)
from .providers import LLMConfig

# ══════════════════════════════════════════════════════════════════════
# Doubles
# ══════════════════════════════════════════════════════════════════════

OPENAI = LLMConfig("sk-test", "", "gpt-4o", "openai", "openai", provider="openai")
GPT5 = LLMConfig("sk-test", "", "gpt-5-mini", "openai", "openai", provider="openai")
ANTHROPIC = LLMConfig(
    "sk-ant", "", "claude-sonnet-4-6", "anthropic", "anthropic", provider="anthropic"
)
OLLAMA = LLMConfig("", "http://localhost:11434", "llama3.2", "ollama", "ollama_chat", "ollama")


class _Boom(Exception):
    """Stand-in for litellm.UnsupportedParamsError from the focus_llm bug."""


class FakeTransport(Transport):
    """Answers every call with ``choices`` texts (capped by the request's
    ``n``); ``fail`` raises on every call, ``fail_native_n`` only on
    calls that carry ``n``. Records each call for assertions."""

    def __init__(
        self,
        text: str = "reply",
        *,
        choices: int | None = None,
        fail: bool = False,
        fail_native_n: bool = False,
        usage: dict | None = None,
    ):
        self.text = text
        self.choices = choices
        self.fail = fail
        self.fail_native_n = fail_native_n
        self.usage = usage
        self.calls: list[dict] = []

    async def complete(self, config, messages, params, *, timeout, retries):
        self.calls.append(
            {
                "model": config.model,
                "messages": messages,
                "params": dict(params),
                "timeout": timeout,
                "retries": retries,
            }
        )
        if self.fail or (self.fail_native_n and "n" in params):
            raise _Boom("boom")
        n = int(params.get("n", 1))
        count = n if self.choices is None else min(n, self.choices)
        k = len(self.calls)
        return Reply(
            texts=[f"{self.text}-{k}-{i}" for i in range(count)],
            usage=dict(self.usage) if self.usage else None,
        )


@pytest.fixture
def no_strict():
    os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)
    yield
    os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)


@pytest.fixture
def strict():
    os.environ["AGENTCANVAS_STRICT_ERRORS"] = "1"
    yield
    os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)


def _run(coro):
    return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════
# Message shape
# ══════════════════════════════════════════════════════════════════════


def test_messages_system_history_then_user_turn():
    req = Request(
        prompt="now",
        system_prompt="sys",
        history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    )
    assert build_messages(req) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "now"},
    ]


def test_messages_history_may_carry_the_user_turn():
    req = Request(history=[{"role": "user", "content": "a"}])
    assert build_messages(req) == [{"role": "user", "content": "a"}]


def test_messages_images_interleaved_after_prompt_with_labels():
    req = Request(
        prompt="look",
        images=["AAA", "BBB"],
        image_labels=["first:"],
        image_mime="image/jpeg",
    )
    content = build_messages(req)[-1]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1] == {"type": "text", "text": "first:"}
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert content[3]["image_url"]["url"] == "data:image/jpeg;base64,BBB"
    assert len(content) == 4  # second image unlabeled


def test_messages_empty_request_is_a_failure(no_strict):
    wire = FakeTransport()
    assert _run(complete(OPENAI, Request(), transport=wire)).texts == []
    assert wire.calls == []


def test_messages_empty_request_raises_under_strict(strict):
    with pytest.raises(ValueError):
        _run(complete(OPENAI, Request(), transport=FakeTransport()))


# ══════════════════════════════════════════════════════════════════════
# Rulebook over every parameter
# ══════════════════════════════════════════════════════════════════════


def test_only_explicit_params_reach_the_wire():
    wire = FakeTransport()
    _run(complete(OPENAI, Request(prompt="x"), transport=wire))
    assert wire.calls[0]["params"] == {}


def test_locked_temperature_is_adjusted_and_reported():
    wire = FakeTransport()
    out = _run(complete(GPT5, Request(prompt="x", temperature=0.2), transport=wire))
    assert wire.calls[0]["params"]["temperature"] == 1.0
    assert any("temperature" in a and "locked" in a for a in out.adjustments)


def test_required_max_tokens_is_injected():
    wire = FakeTransport()
    out = _run(complete(ANTHROPIC, Request(prompt="x"), transport=wire))
    assert wire.calls[0]["params"]["max_tokens"] == 4096
    assert any("required" in a for a in out.adjustments)


def test_stop_passes_through():
    wire = FakeTransport()
    _run(complete(OPENAI, Request(prompt="x", stop=["END"]), transport=wire))
    assert wire.calls[0]["params"]["stop"] == ["END"]


def test_image_detail_kept_when_the_provider_honours_it():
    wire = FakeTransport()
    _run(complete(OPENAI, Request(prompt="x", images=["AAA"], image_detail="high"), transport=wire))
    part = wire.calls[0]["messages"][-1]["content"][1]
    assert part["image_url"]["detail"] == "high"
    assert "image_detail" not in wire.calls[0]["params"]


def test_image_detail_dropped_where_unsupported():
    wire = FakeTransport()
    out = _run(
        complete(
            ANTHROPIC, Request(prompt="x", images=["AAA"], image_detail="high"), transport=wire
        )
    )
    part = wire.calls[0]["messages"][-1]["content"][1]
    assert "detail" not in part["image_url"]
    assert any("image_detail" in a for a in out.adjustments)


# ══════════════════════════════════════════════════════════════════════
# Native n + fallback, on every path
# ══════════════════════════════════════════════════════════════════════


def test_native_n_is_one_call():
    wire = FakeTransport()
    out = _run(complete(OPENAI, Request(prompt="x", n=3), transport=wire))
    assert len(out.texts) == 3
    assert len(wire.calls) == 1 and wire.calls[0]["params"]["n"] == 3


def test_native_n_shortfall_is_filled_with_single_samples():
    wire = FakeTransport(choices=1)
    out = _run(complete(OPENAI, Request(prompt="x", n=3), transport=wire))
    assert len(out.texts) == 3 and len(set(out.texts)) == 3
    assert [c["params"].get("n") for c in wire.calls] == [3, None, None]


def test_native_n_rejected_falls_back_to_single_samples(no_strict):
    wire = FakeTransport(fail_native_n=True)
    out = _run(complete(OPENAI, Request(prompt="x", images=["AAA"], n=3), transport=wire))
    assert len(out.texts) == 3
    assert [c["params"].get("n") for c in wire.calls] == [3, None, None, None]


def test_n_unsupported_by_rulebook_skips_the_native_attempt():
    wire = FakeTransport()
    out = _run(complete(ANTHROPIC, Request(prompt="x", n=3), transport=wire))
    assert len(out.texts) == 3
    assert len(wire.calls) == 3 and all("n" not in c["params"] for c in wire.calls)
    assert any("n=3 dropped" in a for a in out.adjustments)


def test_n_one_is_a_single_call_without_n():
    wire = FakeTransport()
    out = _run(complete(OPENAI, Request(prompt="x", n=1), transport=wire))
    assert out.text == "reply-1-0"
    assert len(wire.calls) == 1 and "n" not in wire.calls[0]["params"]


# ══════════════════════════════════════════════════════════════════════
# Error policy
# ══════════════════════════════════════════════════════════════════════


def test_strict_flag_truthy_values():
    for v in ("1", "true", "TRUE", "yes", "on"):
        os.environ["AGENTCANVAS_STRICT_ERRORS"] = v
        assert strict_errors_enabled() is True, f"failed for {v!r}"
    for v in ("0", "false", "", "no", "off", "random"):
        os.environ["AGENTCANVAS_STRICT_ERRORS"] = v
        assert strict_errors_enabled() is False, f"failed for {v!r}"
    os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)
    assert strict_errors_enabled() is False


def test_failure_is_swallowed_when_strict_off(no_strict):
    wire = FakeTransport(fail=True)
    assert _run(complete(OPENAI, Request(prompt="x"), transport=wire)).text is None
    assert _run(complete(OPENAI, Request(prompt="x", n=3), transport=wire)).texts == []
    assert (
        _run(complete(OPENAI, Request(prompt="x", images=["AAA"], n=3), transport=wire)).texts == []
    )


def test_failure_raises_when_strict_on(strict):
    wire = FakeTransport(fail=True)
    with pytest.raises(_Boom):
        _run(complete(OPENAI, Request(prompt="x"), transport=wire))
    with pytest.raises(_Boom):
        _run(complete(OPENAI, Request(prompt="x", n=3), transport=wire))
    with pytest.raises(_Boom):
        _run(complete(OPENAI, Request(prompt="x", images=["AAA"], n=3), transport=wire))


# ══════════════════════════════════════════════════════════════════════
# Usage, timeout, retries
# ══════════════════════════════════════════════════════════════════════


def test_usage_accumulates_into_the_node_bucket():
    wire = FakeTransport(
        choices=1, usage={"model": "m", "prompt_tokens": 10, "completion_tokens": 5}
    )

    async def run():
        bucket = new_usage_bucket()
        token = _current_node_usage.set(bucket)
        try:
            await complete(OPENAI, Request(prompt="x", n=2), transport=wire)
        finally:
            _current_node_usage.reset(token)
        return bucket

    bucket = _run(run())
    assert bucket["calls"] == 2  # native attempt + one fill
    assert bucket["prompt_tokens"] == 20 and bucket["completion_tokens"] == 10
    assert bucket["model"] == "m"


def test_timeout_policy():
    wire = FakeTransport()
    _run(complete(OPENAI, Request(prompt="x"), transport=wire))
    _run(complete(OPENAI, Request(prompt="x", images=["AAA"]), transport=wire))
    _run(complete(OLLAMA, Request(prompt="x"), transport=wire))
    assert [c["timeout"] for c in wire.calls] == [60.0, 90.0, 120.0]


def test_retries_reach_the_transport():
    wire = FakeTransport()
    _run(complete(OPENAI, Request(prompt="x", retries=2), transport=wire))
    assert wire.calls[0]["retries"] == 2


def test_complete_sync_inside_a_running_loop_keeps_the_bucket():
    wire = FakeTransport(usage={"prompt_tokens": 3})

    async def forward():
        bucket = new_usage_bucket()
        token = _current_node_usage.set(bucket)
        try:
            out = complete_sync(OPENAI, Request(prompt="x"), transport=wire)
        finally:
            _current_node_usage.reset(token)
        return out, bucket

    out, bucket = _run(forward())
    assert isinstance(out, Completion) and out.text == "reply-1-0"
    assert bucket["calls"] == 1 and bucket["prompt_tokens"] == 3


def test_complete_sync_without_a_loop():
    wire = FakeTransport()
    assert complete_sync(OPENAI, Request(prompt="x"), transport=wire).text == "reply-1-0"


# ══════════════════════════════════════════════════════════════════════
# The four historical wrappers reshape, nothing more
# ══════════════════════════════════════════════════════════════════════


def test_wrappers_reshape_onto_one_request(monkeypatch):
    seen: list[Request] = []

    async def fake_complete(config, request, *, transport=None):
        seen.append(request)
        return Completion(texts=["t1", "t2"])

    monkeypatch.setattr("app.llm.call.complete", fake_complete)
    msgs = [{"role": "user", "content": "q"}]
    assert _run(llm_complete(OPENAI, msgs, system_prompt="s", stop=["x"])) == "t1"
    assert _run(llm_complete_n(OPENAI, msgs, n=2, temperature=0.5)) == ["t1", "t2"]
    assert _run(vlm_complete(OPENAI, "p", ["AAA"], detail="high", prior_messages=msgs)) == "t1"
    assert _run(vlm_complete_n(OPENAI, "p", ["AAA"], n=2, mime="image/jpeg")) == ["t1", "t2"]

    text, text_n, vis, vis_n = seen
    assert text.history == msgs and text.prompt == "" and text.stop == ["x"]
    assert text.system_prompt == "s" and text.n == 1
    assert text_n.n == 2 and text_n.temperature == 0.5
    assert vis.prompt == "p" and vis.images == ["AAA"] and vis.history == msgs
    assert vis.image_detail == "high" and vis.n == 1
    assert vis_n.n == 2 and vis_n.image_mime == "image/jpeg" and vis_n.history == []
