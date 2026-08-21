"""NavToolsModel + rollout-only image pruning.

mini's qwen ollama cells run ``image_window = 0`` — every panorama ever seen
stays in the payload (cells.py:249-251). We keep that exactly, because the
`base` arm has to remain comparable to the existing std_mini_qwen3.5-4b_wp
board, and because the panorama history IS the agent's memory of the route.

What cannot stay is the rollouts: ~19 looks x 5 sheets = 95 large images. So a
second, narrower window applies to those alone — the newest look keeps its
sheets, every earlier look drops them to a stub.

The two are distinguished structurally, not by a marker: in a tool message
built by imagine_toolset, image 0 is the panorama and images 1.. are rollout
sheets. Nothing else in the pipeline emits multi-image tool results.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_AC = os.environ.get("AGENTCANVAS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # coding-agent/imaginevln -> repo root
_MINI = os.path.join(_AC, "coding-agent", "harnesses", "mini")
if _MINI not in sys.path:
    sys.path.insert(0, _MINI)

from model import NavToolsModel  # noqa: E402

ROLLOUT_STUB = "[rollout sheets from an earlier look elided — only the newest look keeps them]"


def _image_idxs(content: Any) -> list[int]:
    if not isinstance(content, list):
        return []
    return [i for i, p in enumerate(content)
            if isinstance(p, dict) and p.get("type") == "image_url"]


REASONING_DROPPED = "[earlier reasoning dropped]"


class ImagineNavToolsModel(NavToolsModel):
    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return super()._prepare_messages_for_api(
            _drop_stale_reasoning(_prune_imagine(_prune_rollouts(messages))))


IMAGINE_MARK = "[imagine]"      # imagine_toolset.IMAGINE_MARK — first text of imagine() results
IMAGINE_STUB = ("[predicted walk-throughs from an earlier imagine() elided — you have "
                "moved since / a newer imagine() exists; only the current round's "
                "predictions are ever shown]")
_MOVE_KINDS = ("goto", "step")


def _tool_kind(msg: dict) -> str | None:
    """The toolset's own `info.kind` (stored under extra), falling back to the
    [imagine] text marker — the SDK line prunes by session rebuild, the mini
    line by this."""
    kind = ((msg.get("extra") or {}).get("info") or {}).get("kind")
    if kind:
        return kind
    c = msg.get("content")
    if isinstance(c, list) and c and isinstance(c[0], dict) \
            and str(c[0].get("text", "")).startswith(IMAGINE_MARK):
        return "imagine"
    return None


def _prune_imagine(messages: list[dict]) -> list[dict]:
    """phase-2 (hybrid_imagine): imagine() results carry ONLY rollout sheets
    (no panorama), so the structural rule above must not touch them. Keep the
    sheets of the newest imagine() only, and only while the agent has NOT
    moved since (a goto/step after it makes every prediction stale — the SDK
    line rebuilds the session at that point; here the sheets become a stub)."""
    last_imagine = last_move = None
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        k = _tool_kind(msg)
        if k == "imagine" and _image_idxs(msg.get("content")):
            last_imagine = i
        elif k in _MOVE_KINDS:
            last_move = i
    if last_imagine is None:
        return messages
    keep = last_imagine if (last_move is None or last_move < last_imagine) else None
    out: list[dict] = []
    for i, msg in enumerate(messages):
        if (msg.get("role") == "tool" and i != keep and _tool_kind(msg) == "imagine"
                and _image_idxs(msg.get("content"))):
            parts = [p for p in msg["content"]
                     if not (isinstance(p, dict) and p.get("type") == "image_url")]
            parts.append({"type": "text", "text": IMAGINE_STUB})
            out.append({**msg, "content": parts})
        else:
            out.append(msg)
    return out


def _drop_stale_reasoning(messages: list[dict]) -> list[dict]:
    """Keep only the newest turn's chain of thought.

    Qwen3.5 thinks for 500-2000 tokens per decision; carrying every past turn's
    reasoning forward crowds the window with stale deliberation about positions
    the agent has already left. The position history and the panoramas are the
    memory that should persist — not the thinking that produced them."""
    last = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("reasoning_content"):
            last = i
    if last is None:
        return messages
    return [msg if (i == last or msg.get("role") != "assistant"
                    or not msg.get("reasoning_content"))
            else {k: v for k, v in msg.items() if k != "reasoning_content"}
            for i, msg in enumerate(messages)]


def _prune_rollouts(messages: list[dict]) -> list[dict]:
    """Keep image 0 of every tool message; keep images 1.. only in the last
    tool message that has any."""
    newest = None
    for i, msg in enumerate(messages):
        if (msg.get("role") == "tool" and len(_image_idxs(msg.get("content"))) > 1
                and _tool_kind(msg) != "imagine"):      # phase-2 results: _prune_imagine
            newest = i
    if newest is None:
        return messages

    out: list[dict] = []
    for i, msg in enumerate(messages):
        idxs = _image_idxs(msg.get("content")) if msg.get("role") == "tool" else []
        if i == newest or len(idxs) <= 1 or _tool_kind(msg) == "imagine":
            out.append(msg)
            continue
        drop = set(idxs[1:])
        parts = [p for j, p in enumerate(msg["content"]) if j not in drop]
        parts.append({"type": "text", "text": ROLLOUT_STUB})
        out.append({**msg, "content": parts})
    return out
