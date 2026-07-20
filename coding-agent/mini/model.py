"""Nav-tools model — mini-swe-agent's litellm model speaking our toolset.

Replaces the single hardcoded BASH_TOOL with the toolset's declared schemas,
parses tool calls into ``{"tool", "args", "tool_call_id"}`` actions for the
environment, and renders observations as multimodal tool messages (image
content parts pass through litellm to each provider's format — the same
mechanism mini's own multimodal mode uses).

``image_window`` is the one declared deviation from mini's "no context
management": with N camera frames per episode the raw linear history overflows
the context window, so at API-call time only the newest K images are kept and
older ones collapse to a text stub. The stored trajectory keeps everything;
0 disables windowing.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks

IMAGE_ELIDED_STUB = "[earlier camera frame elided to save context]"

FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and (finish_reason == "length" or (finish_reason == "tool_calls" and not has_tool_calls)) -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a tool call, so it was cut off. Respond more concisely and finish with exactly one tool call. If you need to think more, do so briefly.
{%- else -%}
Tool call error:

<error>
{{ error }}
</error>

Every response MUST include at least one tool call to one of your navigation tools ({{ tool_names }}). Call step with arguments like {"actions": [1, 1, 2]} using integers 0-3.
{%- endif %}"""


class NavToolsModelConfig(LitellmModelConfig):
    tools: list[dict[str, Any]] = []
    """Neutral tool schemas ({name, description, input_schema}) from the toolset."""
    image_window: int = 0
    """Keep only the newest K images in the API payload (0 = keep all)."""
    format_error_template: str = FORMAT_ERROR_TEMPLATE


class NavToolsModel(LitellmModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(config_class=NavToolsModelConfig, **kwargs)
        self._tool_names = [t["name"] for t in self.config.tools]
        self._litellm_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self.config.tools
        ]

    def _query(self, messages: list[dict[str, str]], **kwargs: Any):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=self._litellm_tools,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " Set the provider API key (e.g. ANTHROPIC_API_KEY) in the environment."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        # mini's set_cache_control asserts single-part content, which breaks on
        # our [image, text] tool results (upstream mini-swe-agent 2.4.5 bug) —
        # reimplement its default_end semantics multipart-safe instead of
        # calling super().
        prepared = [
            {k: v for k, v in msg.items() if k != "extra"}
            for msg in self._apply_image_window(messages)
        ]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        if self.config.set_cache_control == "default_end":
            prepared = self._set_cache_control_multipart(prepared)
        return prepared

    @staticmethod
    def _set_cache_control_multipart(messages: list[dict]) -> list[dict]:
        """default_end cache breakpoint on the last message, any content shape."""
        messages = copy.deepcopy(messages)
        for entry in messages:
            entry.pop("cache_control", None)
            if isinstance(entry.get("content"), list):
                for part in entry["content"]:
                    if isinstance(part, dict):
                        part.pop("cache_control", None)
        if not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") == "tool" or not content:
            # message-level marker (mini's own workaround for tool messages)
            last["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            last["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["cache_control"] = {"type": "ephemeral"}
                    break
            else:
                last["cache_control"] = {"type": "ephemeral"}
        return messages

    def _apply_image_window(self, messages: list[dict]) -> list[dict]:
        k = self.config.image_window
        if k <= 0:
            return messages
        kept = 0
        out: list[dict] = []
        for msg in reversed(messages):
            content = msg.get("content")
            if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in content
            ):
                new_parts: list[Any] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        kept += 1
                        if kept > k:
                            new_parts.append({"type": "text", "text": IMAGE_ELIDED_STUB})
                            continue
                    new_parts.append(part)
                msg = {**msg, "content": new_parts}
            out.append(msg)
        return list(reversed(out))

    def _format_error(self, error: str, *, has_tool_calls: bool, finish_reason: Any) -> FormatError:
        content = Template(self.config.format_error_template, undefined=StrictUndefined).render(
            error=error,
            actions=[],
            has_tool_calls=has_tool_calls,
            finish_reason=finish_reason,
            tool_names=", ".join(self._tool_names),
        )
        return FormatError(
            {"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}}
        )

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls into env actions. Raises FormatError on bad calls."""
        finish_reason = response.choices[0].finish_reason
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            raise self._format_error(
                "No tool calls found in the response. Every response MUST include "
                "at least one tool call.",
                has_tool_calls=False,
                finish_reason=finish_reason,
            )
        actions = []
        for tool_call in tool_calls:
            error_msg = ""
            args: Any = {}
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except Exception as e:
                error_msg = f"Error parsing tool call arguments: {e}."
            if tool_call.function.name not in self._tool_names:
                error_msg += (
                    f"Unknown tool '{tool_call.function.name}'; "
                    f"available: {', '.join(self._tool_names)}."
                )
            if not isinstance(args, dict):
                error_msg += "Tool call arguments must be a JSON object."
            if error_msg:
                raise self._format_error(
                    error_msg.strip(), has_tool_calls=True, finish_reason=finish_reason
                )
            actions.append(
                {"tool": tool_call.function.name, "args": args, "tool_call_id": tool_call.id}
            )
        return actions

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """One role:"tool" message per action, multimodal content straight from
        the toolset (no Jinja observation template — images can't ride text)."""
        actions = message.get("extra", {}).get("actions", [])
        not_executed = {
            "content": [{"type": "text", "text": json.dumps({"error": "action was not executed"})}],
            "info": {"error": "action was not executed"},
        }
        padded = outputs + [not_executed] * (len(actions) - len(outputs))
        results = []
        for action, output in zip(actions, padded):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": action["tool_call_id"],
                    "content": output["content"],
                    "extra": {
                        "info": output.get("info", {}),
                        "timestamp": time.time(),
                    },
                }
            )
        return results
