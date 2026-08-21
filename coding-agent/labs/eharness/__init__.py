"""eharness — the embodied harness shell.

A universal shell wrapped around any executor loop (mini / claude-sdk / codex /
future VLA). Organs live at exactly two interposition points and never inside a
loop:

  ① the tool boundary — ``wrapper.HarnessedToolset`` wraps any NodesetToolSet:
     V0 guards, decision-coupled state writes, keyframe promotion, heartbeat,
     the pre-stop verification gate;
  ② the session boundary — ``planner.Planner`` (the O(1) commander loop) and
     ``executor.run_subgoal`` (a budget-bounded sub-session that returns a
     structured Receipt).

Design references (blackboard :8100, NavHarness tab): agentic-loop-spec.html
(the spec this implements) and implementation-plan.html v2. Hard rules carried
from there: no coordinates anywhere (frame hashes + landmarks only), memory
lives on disk (live_dir) — any loop's context holds renderings, never the
source of truth; baseline cells are untouched (the wrapper is additive; nothing
in mini/toolset.py changes).
"""

# Lazy attribute access instead of eager imports: wrapper pulls in judge which
# pulls in litellm, and that chain used to make even the pure-geometry modules
# (depthmap, landmarks) unimportable in an env without LLM deps — geometry
# tests could not run where habitat lives. `import eharness.depthmap` must
# never require litellm.
_LAZY = {
    "StateBlock": "eharness.state_block",
    "HarnessedToolset": "eharness.wrapper",
    "Planner": "eharness.planner",
    "Receipt": "eharness.receipts",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
