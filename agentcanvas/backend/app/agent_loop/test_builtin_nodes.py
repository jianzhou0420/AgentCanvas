"""Direct forward() tests for the processing-tier builtin nodes.

``textParse`` sits between every demo-graph llmCall and the env step —
its parsing rules ARE the action interface, so each mode is pinned
here: choice (keyword list, scan strategy, default fallback), regex
(first capture / whole match), json_field (dot path + ``{...}`` block
rescue). ``historyLog`` is the no-container history mechanism promised
by its docstring; ``nullSource`` is the sanctioned typed-None seed
(see ``feedback_initialize_semantics`` — init slots are NOT auto-None,
this node is the explicit alternative).

Loop/scope builtins (iterIn/iterOut) are covered by the executor-level
suites (``test_executor_scopes``, ``test_iterin_port_configs``,
``test_post_loop``); llmCall's provider plumbing needs an LLM-client
fake and stays a known gap.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .builtin_nodes import (
    HistoryLogNode,
    NullSourceNode,
    TextParseNode,
    _quat_to_heading_deg,
)


class _FakeCtx:
    """Attribute-proxy ctx mirroring ``_NodeStateProxy`` semantics:
    reads of unset keys return None, writes persist."""

    def __init__(self, step: int = 0) -> None:
        object.__setattr__(self, "_store", {"step": step})

    def __getattr__(self, key: str) -> Any:
        return object.__getattribute__(self, "_store").get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        object.__getattribute__(self, "_store")[key] = value


def _fire(node_cls: type, config: dict, inputs: dict, ctx: Any = None) -> dict:
    node = node_cls()
    node.config = config
    return asyncio.run(node.forward(inputs, ctx if ctx is not None else _FakeCtx()))


# ── textParse · choice mode ─────────────────────────────────────────────

_CHOICES = {"mode": "choice", "choices": "STOP,FORWARD,LEFT,RIGHT"}


def test_choice_matches_keyword_and_reports_action_index() -> None:
    out = _fire(TextParseNode, _CHOICES, {"text": "I will go LEFT now"})
    assert out["value"] == "LEFT"
    assert out["index"] == 2  # position in the list == discrete action id
    # Matched span removed verbatim; interior whitespace is NOT collapsed.
    assert out["rest"] == "I will go  now"


def test_choice_is_case_insensitive() -> None:
    out = _fire(TextParseNode, _CHOICES, {"text": "answer: forward"})
    assert out["value"] == "FORWARD"
    assert out["index"] == 1


def test_choice_last_line_wins_over_full_text() -> None:
    # Reasoning mentions FORWARD, but the final answer line says LEFT —
    # last_line_first must pick the conclusion, not the deliberation.
    text = "Maybe FORWARD is good...\nFinal answer: LEFT"
    out = _fire(TextParseNode, _CHOICES, {"text": text})
    assert out["value"] == "LEFT"


def test_choice_full_text_scan_uses_choice_list_priority() -> None:
    # full_text mode scans the whole text in CHOICE-LIST order — the
    # earlier list entry wins regardless of position in the text.
    text = "Maybe FORWARD is good...\nFinal answer: LEFT"
    out = _fire(TextParseNode, {**_CHOICES, "scan": "full_text"}, {"text": text})
    assert out["value"] == "FORWARD"


def test_choice_no_match_falls_back_to_default_with_index() -> None:
    out = _fire(TextParseNode, {**_CHOICES, "default": "STOP"}, {"text": "no keywords here"})
    assert out["value"] == "STOP"
    assert out["index"] == 0  # default is a list member → its index
    out = _fire(TextParseNode, {**_CHOICES, "default": "PANIC"}, {"text": "nothing"})
    assert out["value"] == "PANIC"
    assert out["index"] == -1  # non-member default → no action id


def test_choice_no_match_no_default_yields_empty() -> None:
    out = _fire(TextParseNode, _CHOICES, {"text": "nothing relevant"})
    assert out == {"value": "", "index": -1, "rest": "nothing relevant"}


# ── textParse · regex mode ──────────────────────────────────────────────


def test_regex_first_capture_group_and_rest() -> None:
    out = _fire(
        TextParseNode,
        {"mode": "regex", "pattern": r"mark:\s*(\d)"},
        {"text": "Your mark: 7 (out of 9)"},
    )
    assert out["value"] == "7"
    # Whole match span removed; interior whitespace not collapsed.
    assert out["rest"] == "Your  (out of 9)"


def test_regex_without_groups_returns_whole_match() -> None:
    out = _fire(TextParseNode, {"mode": "regex", "pattern": r"\d+"}, {"text": "abc 42 def"})
    assert out["value"] == "42"


def test_regex_no_match_returns_default() -> None:
    out = _fire(
        TextParseNode,
        {"mode": "regex", "pattern": r"\d+", "default": "0"},
        {"text": "no digits"},
    )
    assert out["value"] == "0"
    assert out["rest"] == "no digits"


# ── textParse · json_field mode ─────────────────────────────────────────


def test_json_field_dot_path() -> None:
    out = _fire(
        TextParseNode,
        {"mode": "json_field", "json_key": "plan.next_step"},
        {"text": '{"plan": {"next_step": "turn_left"}}'},
    )
    assert out["value"] == "turn_left"


def test_json_field_rescues_embedded_block_from_prose() -> None:
    text = 'Sure! Here is the plan:\n{"action": "STOP"}\nGood luck.'
    out = _fire(TextParseNode, {"mode": "json_field", "json_key": "action"}, {"text": text})
    assert out["value"] == "STOP"


def test_json_field_missing_key_returns_default() -> None:
    out = _fire(
        TextParseNode,
        {"mode": "json_field", "json_key": "missing", "default": "n/a"},
        {"text": '{"action": "STOP"}'},
    )
    assert out["value"] == "n/a"


def test_json_field_unparseable_text_returns_default() -> None:
    out = _fire(
        TextParseNode,
        {"mode": "json_field", "json_key": "a", "default": "d"},
        {"text": "not json at all"},
    )
    assert out["value"] == "d"


# ── historyLog ──────────────────────────────────────────────────────────


def test_history_accumulates_across_firings() -> None:
    node = HistoryLogNode()
    node.config = {}
    ctx = _FakeCtx(step=0)
    asyncio.run(node.forward({"entry": "went left"}, ctx))
    ctx.step = 1
    out = asyncio.run(node.forward({"entry": "went right"}, ctx))
    assert out["history"] == "Step 0: went left\nStep 1: went right"


def test_history_empty_placeholder_and_blank_entries_ignored() -> None:
    ctx = _FakeCtx()
    out = _fire(HistoryLogNode, {}, {"entry": "   "}, ctx)
    assert out["history"] == "(no history yet)"


def test_history_trims_to_max_entries_with_omission_header() -> None:
    node = HistoryLogNode()
    node.config = {"max_entries": 5}
    ctx = _FakeCtx()
    for i in range(8):
        ctx.step = i
        out = asyncio.run(node.forward({"entry": f"e{i}"}, ctx))
    lines = out["history"].splitlines()
    assert lines[0] == "(...3 earlier steps omitted)"
    assert lines[1] == "Step 3: e3"
    assert lines[-1] == "Step 7: e7"


def test_history_bad_template_falls_back_per_line() -> None:
    node = HistoryLogNode()
    node.config = {"template": "Step {stepp}: {entry}"}  # typo'd placeholder
    ctx = _FakeCtx(step=2)
    out = asyncio.run(node.forward({"entry": "x"}, ctx))
    assert out["history"] == "Step 2: x"


# ── nullSource ──────────────────────────────────────────────────────────


def test_null_source_emits_none_and_resolves_authored_wire_type() -> None:
    out = _fire(NullSourceNode, {"wire_type": "IMAGE"}, {})
    assert out == {"value": None}
    _ins, outs = NullSourceNode._resolve_ports({"wire_type": "IMAGE"})
    assert outs[0].wire_type == "IMAGE"
    _ins, outs = NullSourceNode._resolve_ports({})
    assert outs[0].wire_type == "ANY"  # default preserved


# ── pose helper ─────────────────────────────────────────────────────────


def test_quat_to_heading_identity_is_zero() -> None:
    assert _quat_to_heading_deg([0.0, 0.0, 0.0, 1.0]) == 0.0


def test_quat_to_heading_quarter_turn() -> None:
    # 90° yaw about Y: q = [0, sin(45°), 0, cos(45°)]
    assert _quat_to_heading_deg([0.0, 0.7071068, 0.0, 0.7071068]) == 90.0


def test_quat_to_heading_short_or_empty_defaults_zero() -> None:
    assert _quat_to_heading_deg([]) == 0.0
    assert _quat_to_heading_deg([0.1, 0.2]) == 0.0
