"""Tests for ``Fleet`` — per-worker routing over every replicated nodeset
a graph needs, against a fake registry (its interface: ``parallelism``,
``get_server_url``, ``unload_nodeset``) and the real env-panel registry.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.components.env_panel import register_env_panel, unregister_env_panel
from app.components.fleet import Fleet
from app.graph_def import GraphDefinition


def _graph(*node_types: str) -> GraphDefinition:
    return GraphDefinition.from_dict(
        {"nodes": [{"id": f"n{i}", "type": nt} for i, nt in enumerate(node_types)], "edges": []}
    )


class _FakeRegistry:
    def __init__(self, parallelism: dict[str, str], tagged: dict[str, int]) -> None:
        self._parallelism = parallelism
        self._tagged = tagged  # name → worker_count spawned
        self.unloaded: list[str] = []

    def parallelism(self, name: str) -> str:
        return self._parallelism.get(name, "shared")

    def get_server_url(self, name: str, tag: int | None = None) -> str | None:
        n = self._tagged.get(name)
        if n is None or tag is None or tag >= n:
            return None
        return f"http://127.0.0.1:{9000 + tag}/{name}"

    async def unload_nodeset(self, name: str) -> dict:
        self.unloaded.append(name)
        return {"name": name}


class _Panel:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def tagged_panels():
    names = [f"env_sim#{k}" for k in range(2)]
    for n in names:
        register_env_panel(_Panel(n))  # type: ignore[arg-type]
    yield
    for n in names:
        unregister_env_panel(n)


def test_single_worker_is_an_empty_route_and_touches_no_registry() -> None:
    async def run() -> None:
        async with Fleet(_graph("env_sim__step", "model_slam__track"), 1) as fleet:
            w = fleet.worker(0)
            assert w.worker_id == 0
            assert w.env_panel_overrides == {} and w.server_url_overrides == {}
            assert fleet.routed == []

    asyncio.run(run())  # registry=None and never resolved


def test_routes_every_replicated_nodeset_per_worker(tagged_panels: Any) -> None:
    reg = _FakeRegistry(
        parallelism={"env_sim": "replicated", "model_slam": "replicated", "vlm": "shared"},
        tagged={"env_sim": 2, "model_slam": 2},
    )
    graph = _graph("env_sim__step", "model_slam__track", "vlm__describe")

    async def run() -> None:
        async with Fleet(graph, 2, registry=reg) as fleet:
            assert fleet.routed == ["env_sim", "model_slam"]
            for k in range(2):
                w = fleet.worker(k)
                # every replicated nodeset (env or not) routes to its own copy
                assert w.server_url_overrides == {
                    "env_sim": f"http://127.0.0.1:{9000 + k}/env_sim",
                    "model_slam": f"http://127.0.0.1:{9000 + k}/model_slam",
                }
                # only the nodeset with an env panel gets a panel route
                assert list(w.env_panel_overrides) == ["env_sim"]
                assert w.env_panel_overrides["env_sim"].name == f"env_sim#{k}"
                assert "vlm" not in w.server_url_overrides  # shared: baked URL
        assert reg.unloaded == ["env_sim", "model_slam"]  # teardown of all routed

    asyncio.run(run())


def test_missing_tagged_copy_is_a_clear_error() -> None:
    reg = _FakeRegistry(parallelism={"env_sim": "replicated"}, tagged={"env_sim": 1})

    async def run() -> None:
        with pytest.raises(RuntimeError, match="env_sim#1 is missing"):
            async with Fleet(_graph("env_sim__step"), 2, registry=reg):
                pass

    asyncio.run(run())


def test_worker_lookup_guards() -> None:
    fleet = Fleet(_graph(), 1)
    with pytest.raises(RuntimeError, match="outside async with"):
        fleet.worker(0)

    async def run() -> None:
        async with fleet:
            with pytest.raises(ValueError, match="out of range"):
                fleet.worker(1)

    asyncio.run(run())
    with pytest.raises(ValueError):
        Fleet(_graph(), 0)
