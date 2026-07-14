"""Harness adapters — one per closed/open agent stack.

Each adapter implements driver.HarnessAdapter: run ONE clean session for one
episode, emitting through the shared EventSink. Everything else (placement,
prompts, evaluation, artifacts, summary) lives in driver.py.
"""

from __future__ import annotations


def get_adapter(harness: str):
    if harness == "sdk":
        from harnesses.claude_sdk import ClaudeSdkAdapter
        return ClaudeSdkAdapter()
    if harness == "mini":
        from harnesses.mini_swe import MiniSweAdapter
        return MiniSweAdapter()
    if harness == "codex":
        from harnesses.codex_cli import CodexCliAdapter
        return CodexCliAdapter()
    raise KeyError(f"unknown harness {harness!r}")
