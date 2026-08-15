"""NavAgent — DefaultAgent plus curated event stream and blob-elided trajectories.

The loop itself is mini's, untouched. The two additions serve the run
artifacts: ``event_hook`` mirrors every appended message into the claude-SDK
path's curated event vocabulary (assistant_text / thinking / tool_use /
tool_result / exit) so ``episode_{i}.jsonl`` keeps the same shape the backend
monitor reads, and ``serialize`` elides base64 image payloads so the per-step
trajectory dump stays readable (frames live in ``live_{i}/`` anyway).
"""

from __future__ import annotations

from typing import Any, Callable

from minisweagent.agents.default import DefaultAgent


def elide_blobs(obj: Any) -> Any:
    """Recursively replace base64-sized strings with a marker (agent20 parity)."""
    if isinstance(obj, dict):
        return {k: elide_blobs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide_blobs(x) for x in obj]
    if isinstance(obj, str) and len(obj) > 4000 and " " not in obj[:200]:
        return f"<blob {len(obj)} chars elided>"
    return obj


class NavAgent(DefaultAgent):
    def __init__(
        self, *args: Any, event_hook: Callable[[str, dict], None] | None = None,
        opening_parts: list | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._event_hook = event_hook
        # Content to graft onto the agent's FIRST user message. run() clears
        # self.messages and rebuilds system+instance, so this is the only place
        # a first look can be handed over — and without one the opening move is
        # taken blind (EP0 turned 45° into a wall before it had ever seen the
        # room, because `observe` is not on the model's menu any more).
        self._opening_parts = list(opening_parts or [])

    def add_messages(self, *messages: dict) -> list[dict]:
        if self._opening_parts:
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                msg["content"] = list(content) + self._opening_parts
                self._opening_parts = []
                break
        if self._event_hook is not None:
            for msg in messages:
                try:
                    self._emit(msg)
                except Exception:  # noqa: BLE001 — logging must never break a run
                    pass
        return super().add_messages(*messages)

    def _emit(self, msg: dict) -> None:
        emit = self._event_hook
        role = msg.get("role")
        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning:
                emit("thinking", {"chars": len(reasoning), "text": reasoning})
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                emit("assistant_text", {"text": content})
            for action in msg.get("extra", {}).get("actions", []):
                emit("tool_use", {
                    "id": action.get("tool_call_id"),
                    "name": action.get("tool"),
                    "input": action.get("args"),
                })
        elif role == "tool":
            texts = []
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        texts.append("<image elided>")
            emit("tool_result", {"tool_use_id": msg.get("tool_call_id"), "texts": texts})
        elif role == "exit":
            emit("exit", {
                "exit_status": msg.get("extra", {}).get("exit_status"),
                "content": elide_blobs(msg.get("content")),
            })
        elif role == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                emit("user_text", {"text": content})

    def serialize(self, *extra_dicts: dict) -> dict:
        return elide_blobs(super().serialize(*extra_dicts))
