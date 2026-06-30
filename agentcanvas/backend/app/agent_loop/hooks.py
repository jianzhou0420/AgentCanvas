"""Hook system for graph execution events.

Hooks are shell commands that fire at graph lifecycle boundaries
(GraphStart, GraphComplete, GraphError) or around node execution
(PreNodeExecute, PostNodeExecute).

Communication protocol:
  stdin  → JSON payload (event, node_type, node_id, data)
  stdout ← JSON response {"action": "continue"|"block"|"modify", "modified_data": {...}}

Fail-open: any error (timeout, crash, malformed JSON) returns action="continue".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from dataclasses import dataclass
from typing import Any

from ..graph_def import HookDef, NodeDef

log = logging.getLogger("agentcanvas.hooks")


# ---------------------------------------------------------------------------
# safe_serialize — convert arbitrary objects to JSON-safe representations
# ---------------------------------------------------------------------------


def safe_serialize(obj: Any, max_depth: int = 3) -> Any:
    """Convert arbitrary Python objects to JSON-safe representations.

    Rules:
    - dict: recurse (up to max_depth)
    - list/tuple: recurse (up to max_depth)
    - str, int, float, bool, None: pass through
    - numpy ndarray: {"__type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    - PIL.Image: {"__type": "Image", "size": [w, h], "mode": obj.mode}
    - bytes: {"__type": "bytes", "length": len(obj)}
    - All other types: str(obj) truncated to 200 chars
    - At max_depth: str(obj) truncated to 200 chars
    """
    if max_depth <= 0:
        return str(obj)[:200]

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return {k: safe_serialize(v, max_depth - 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_serialize(v, max_depth - 1) for v in obj]

    # numpy ndarray — check by type name to avoid hard import dependency
    type_name = type(obj).__name__
    module_name = getattr(type(obj), "__module__", "") or ""

    if "numpy" in module_name and type_name == "ndarray":
        return {
            "__type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    # PIL Image
    if module_name.startswith("PIL") and hasattr(obj, "size") and hasattr(obj, "mode"):
        w, h = obj.size
        return {"__type": "Image", "size": [w, h], "mode": obj.mode}

    if isinstance(obj, bytes):
        return {"__type": "bytes", "length": len(obj)}

    return str(obj)[:200]


# ---------------------------------------------------------------------------
# HookResult
# ---------------------------------------------------------------------------


@dataclass
class HookResult:
    """Result returned from running a set of hooks for one event."""

    action: str = "continue"  # "continue" | "block" | "modify"
    modified_data: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# HookRunner
# ---------------------------------------------------------------------------


class HookRunner:
    """Indexes and runs hooks for a graph's lifecycle events.

    Built once per graph execution from the graph's ``HookDef`` list.
    Zero overhead when no hooks are configured — ``has_hooks()`` returns
    False and ``run_hooks()`` returns immediately without any async work.
    """

    def __init__(self, hooks: list[HookDef]) -> None:
        # Index enabled hooks by event for O(1) lookup
        self._by_event: dict[str, list[HookDef]] = {}
        for h in hooks:
            if h.enabled:
                self._by_event.setdefault(h.event, []).append(h)
        self._empty = not bool(self._by_event)

    def has_hooks(self) -> bool:
        """Return True if any hooks are configured (fast-path guard)."""
        return not self._empty

    def _matches(
        self,
        hook: HookDef,
        node_type: str | None,
        node_id: str | None = None,
    ) -> bool:
        """Check if a hook matches the given node_type (and optionally node_id)."""
        # Node-level hooks: match by exact node ID
        if hook.match_node_id is not None:
            return node_id is not None and hook.match_node_id == node_id
        # Type-based matching
        pattern = hook.match_node_type
        if pattern == "*":
            return True
        if node_type is None:
            return False
        if pattern.endswith("__*"):
            return node_type.startswith(pattern[:-1])
        return pattern == node_type

    async def _run_single(
        self,
        hook: HookDef,
        payload: dict,
    ) -> HookResult:
        """Run a single hook subprocess with timeout. Fail-open on any error."""
        try:
            args = shlex.split(hook.command)
        except ValueError as e:
            log.warning("Hook command parse error (%s): %s", hook.command, e)
            return HookResult(error=str(e))

        stdin_bytes = json.dumps(payload).encode()
        timeout_s = hook.timeout_ms / 1000.0

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("Hook timed out after %dms: %s", hook.timeout_ms, hook.command)
            with contextlib.suppress(Exception):
                proc.kill()
            return HookResult(error="timeout")
        except Exception as e:
            log.warning("Hook subprocess error (%s): %s", hook.command, e)
            return HookResult(error=str(e))

        if stderr_bytes:
            log.debug("Hook stderr (%s): %s", hook.command, stderr_bytes.decode(errors="replace"))

        if not stdout_bytes.strip():
            return HookResult()

        try:
            response = json.loads(stdout_bytes)
        except json.JSONDecodeError as e:
            log.warning("Hook stdout not valid JSON (%s): %s", hook.command, e)
            return HookResult(error=f"invalid JSON: {e}")

        action = response.get("action", "continue")
        if action not in ("continue", "block", "modify"):
            log.warning("Unknown hook action '%s', treating as continue", action)
            action = "continue"

        return HookResult(
            action=action,
            modified_data=response.get("modified_data"),
        )

    async def run_hooks(
        self,
        event: str,
        node_type: str | None = None,
        node_id: str | None = None,
        payload: dict | None = None,
    ) -> HookResult:
        """Run all matching hooks for an event sequentially.

        First ``block`` result wins and stops further hook execution.
        ``modify`` results accumulate (later hook can re-modify).
        Fail-open: errors return ``continue`` and do not stop execution.
        """
        if self._empty:
            return HookResult()

        hooks_for_event = self._by_event.get(event, [])
        if not hooks_for_event:
            return HookResult()

        full_payload = {"event": event, "node_type": node_type}
        if payload:
            full_payload.update(payload)

        merged_action = "continue"
        merged_data: dict[str, Any] | None = None

        for hook in hooks_for_event:
            if not self._matches(hook, node_type, node_id):
                continue

            result = await self._run_single(hook, full_payload)

            if result.action == "block":
                return HookResult(action="block")

            if result.action == "modify" and result.modified_data is not None:
                merged_action = "modify"
                if merged_data is None:
                    merged_data = {}
                merged_data.update(result.modified_data)

        return HookResult(action=merged_action, modified_data=merged_data)


# ---------------------------------------------------------------------------
# Hook merge utilities — combine hooks from multiple sources
# ---------------------------------------------------------------------------


def extract_node_hooks(nodes: list[NodeDef]) -> list[HookDef]:
    """Extract per-node hooks from ``NodeDef.config``, auto-setting match fields.

    Each node's ``config["hooks"]`` (if present) is a list of hook dicts.
    ``match_node_type`` is auto-set to the node's type and ``match_node_id``
    to the node's id, ensuring the hook fires only for that specific node.
    """
    result: list[HookDef] = []
    for node in nodes:
        raw_hooks = node.config.get("hooks")
        if not raw_hooks:
            continue
        for h_raw in raw_hooks:
            if isinstance(h_raw, dict):
                h = HookDef.from_dict(h_raw)
            elif isinstance(h_raw, HookDef):
                h = h_raw
            else:
                continue
            h.match_node_type = node.type
            h.match_node_id = node.id
            result.append(h)
    return result


def merge_hooks(
    global_hooks: list[HookDef],
    graph_hooks: list[HookDef],
    node_hooks: list[HookDef],
) -> list[HookDef]:
    """Merge hooks from 3 sources in precedence order.

    Order: global -> graph -> node (least specific first).

    - ``block``: first block wins (global policy can block early)
    - ``modify``: last modify wins (node-level has final say)
    """
    merged: list[HookDef] = []
    merged.extend(global_hooks)
    merged.extend(graph_hooks)
    merged.extend(node_hooks)
    return merged


def load_hooks_file(path: str) -> list[HookDef]:
    """Load hook definitions from a JSON file.

    Returns an empty list if the file doesn't exist or is malformed.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, list):
            log.warning("hooks file %s is not a JSON array — ignoring", path)
            return []
        return [HookDef.from_dict(h) for h in data]
    except Exception as e:
        log.warning("Failed to load hooks file %s: %s", path, e)
        return []
