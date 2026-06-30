"""Tests for WorkspaceComponentRegistry.ensure_shared_nodesets_for_graph.

Verifies the subprocess-eval path's load filter: parent backend only
loads ``parallelism="shared"`` nodesets, leaving replicated / env
nodesets for the eval subprocess to own. Catches regressions in the
filter logic without spawning real auto_host subprocesses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ..graph_def import GraphDefinition
from .registry import WorkspaceComponentRegistry


def _stub_ns(name: str, parallelism: str) -> Any:
    """Minimal stub matching what ensure_shared_nodesets_for_graph reads
    via _get_parallelism + is_nodeset_loaded (just ``type(ns).parallelism``
    and dict membership — no BaseNodeSet interface needed).
    """

    class _Stub:
        pass

    _Stub.parallelism = parallelism  # type: ignore[attr-defined]
    inst = _Stub()
    inst.name = name
    inst.description = f"stub {name}"
    return inst


def _graph_using(*node_types: str) -> GraphDefinition:
    return GraphDefinition.from_dict(
        {
            "nodes": [{"id": f"n{i}", "type": nt} for i, nt in enumerate(node_types)],
            "edges": [],
        }
    )


@pytest.fixture
def registry(tmp_path: Path) -> WorkspaceComponentRegistry:
    r = WorkspaceComponentRegistry(scan_dir=tmp_path)
    # Two discovered nodesets: one shared singleton (VLM), one replicated env.
    vlm = _stub_ns("vlm", parallelism="shared")
    env = _stub_ns("env_habitat", parallelism="replicated")
    r._discovered_nodesets["vlm"] = vlm
    r._discovered_nodesets["env_habitat"] = env
    r._discovered_tool_names["vlm"] = ["vlm__embed"]
    r._discovered_tool_names["env_habitat"] = ["env_habitat__step"]
    return r


def test_loads_only_shared_singleton(registry: WorkspaceComponentRegistry) -> None:
    """env_habitat (replicated) must not be loaded; vlm (shared) must."""
    load_calls: list[str] = []

    async def fake_load(name: str, mode: str = "local", worker_count: int = 1) -> dict:
        load_calls.append(name)
        registry._live_nodesets[name] = registry._discovered_nodesets[name]
        return {"name": name, "tools": [], "mode": mode}

    registry.load_nodeset = fake_load  # type: ignore[assignment]

    graph = _graph_using("vlm__embed", "env_habitat__step")
    result = asyncio.run(registry.ensure_shared_nodesets_for_graph(graph))

    assert load_calls == ["vlm"], "replicated env must not be loaded by parent"
    assert result["loaded"] == ["vlm"]
    assert "env_habitat" not in result["loaded"]
    assert "env_habitat" not in result["already_loaded"]
    assert "env_habitat" not in result["failed"]


def test_skips_already_loaded_shared(registry: WorkspaceComponentRegistry) -> None:
    """Pre-loaded shared singletons go into already_loaded, not loaded."""
    load_calls: list[str] = []

    async def fake_load(name: str, **kw: Any) -> dict:
        load_calls.append(name)
        return {"name": name}

    registry.load_nodeset = fake_load  # type: ignore[assignment]
    # Mark vlm as live (canvas-Play preloaded it).
    registry._live_nodesets["vlm"] = registry._discovered_nodesets["vlm"]

    graph = _graph_using("vlm__embed")
    result = asyncio.run(registry.ensure_shared_nodesets_for_graph(graph))

    assert load_calls == []
    assert result["already_loaded"] == ["vlm"]
    assert result["loaded"] == []


def test_unknown_nodeset_reported(registry: WorkspaceComponentRegistry) -> None:
    """Graph references a nodeset prefix we never discovered → unknown bucket."""
    graph = _graph_using("ghost__act")
    result = asyncio.run(registry.ensure_shared_nodesets_for_graph(graph))
    # _get_parallelism falls back to "shared" for unknown names not in the
    # known-replicated set, so the unknown path triggers.
    assert result["unknown"] == ["ghost"]
    assert result["loaded"] == []


def test_load_failure_recorded(registry: WorkspaceComponentRegistry) -> None:
    """load_nodeset raising lands in 'failed', does not propagate."""

    async def boom(name: str, **kw: Any) -> dict:
        raise RuntimeError("simulated load failure")

    registry.load_nodeset = boom  # type: ignore[assignment]

    graph = _graph_using("vlm__embed")
    result = asyncio.run(registry.ensure_shared_nodesets_for_graph(graph))
    assert result["failed"] == ["vlm"]
    assert result["loaded"] == []
