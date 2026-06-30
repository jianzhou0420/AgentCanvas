"""Tests for WorkspaceComponentRegistry.load_nodeset_ephemeral + unload_nodeset_ephemeral.

TODO #60 — mocks BaseServer so the test runs without spawning real
subprocesses. Verifies:
  * ephemeral spawn registers under tagged key, leaves untagged untouched
  * unload by tag is idempotent and stops the right child(ren)
  * package-mode resolves to --module argv
  * single-file mode resolves to --file argv
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from .bases import BaseNodeSet
from .registry import WorkspaceComponentRegistry


class _DummyServer:
    """Stands in for BaseServer — captures argv, exposes a fake URL."""

    instances: list[_DummyServer] = []
    fail_on_start: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs["name"]
        self.command = kwargs["command"]
        self.port = kwargs["port"]
        self.env = kwargs.get("env", {})
        self.url = f"http://localhost:{self.port}"
        self.stopped = False
        type(self).instances.append(self)

    def start(self) -> None:
        if type(self).fail_on_start:
            raise RuntimeError("forced start failure")

    def fetch_manifest(self) -> dict[str, Any]:
        return {"node_types": []}

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def patched_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkspaceComponentRegistry:
    """Registry pointed at a fake workspace, with BaseServer monkeypatched."""
    workspace_root = tmp_path / "workspace"
    (workspace_root / "nodesets").mkdir(parents=True)
    reg = WorkspaceComponentRegistry(scan_dir=workspace_root)

    import app.server.base_server as bs_mod

    monkeypatch.setattr(bs_mod, "BaseServer", _DummyServer)
    _DummyServer.instances = []
    _DummyServer.fail_on_start = False

    class FakeNS(BaseNodeSet):
        name = "fake_vlm"
        parallelism = "shared"

        def get_tools(self):
            return []

    inst = FakeNS()
    inst._source_file = str(workspace_root / "nodesets" / "fake_vlm.py")
    reg._discovered_nodesets["fake_vlm"] = inst
    return reg


def test_ephemeral_single_file(patched_registry: WorkspaceComponentRegistry, tmp_path: Path) -> None:
    async def run() -> None:
        overlay_src = tmp_path / "overlay" / "nodesets" / "fake_vlm.py"
        overlay_src.parent.mkdir(parents=True)
        overlay_src.write_text("class FakeNS: pass\n")
        url = await patched_registry.load_nodeset_ephemeral(
            "fake_vlm", overlay_src, tag="ephem-abc"
        )
        assert url == _DummyServer.instances[0].url
        assert "fake_vlm#ephem-abc" in patched_registry._auto_servers
        assert "fake_vlm" not in patched_registry._auto_servers
        argv = _DummyServer.instances[0].command
        assert "--file" in argv and str(overlay_src) in argv
        assert "--module" not in argv

    asyncio.run(run())


def test_ephemeral_package_mode(patched_registry: WorkspaceComponentRegistry, tmp_path: Path) -> None:
    async def run() -> None:
        pkg = tmp_path / "overlay" / "nodesets" / "fake_vlm"
        pkg.mkdir(parents=True)
        init = pkg / "__init__.py"
        init.write_text("class FakeNS: pass\n")
        url = await patched_registry.load_nodeset_ephemeral("fake_vlm", init, tag="ephem-xyz")
        assert url == _DummyServer.instances[0].url
        argv = _DummyServer.instances[0].command
        assert "--module" in argv
        mod_idx = argv.index("--module")
        assert argv[mod_idx + 1] == "nodesets.fake_vlm"
        pythonpath = _DummyServer.instances[0].env["PYTHONPATH"]
        assert str((tmp_path / "overlay").resolve()) in pythonpath

    asyncio.run(run())


def test_ephemeral_unload(patched_registry: WorkspaceComponentRegistry, tmp_path: Path) -> None:
    async def run() -> None:
        overlay_src = tmp_path / "overlay" / "nodesets" / "fake_vlm.py"
        overlay_src.parent.mkdir(parents=True)
        overlay_src.write_text("class FakeNS: pass\n")
        await patched_registry.load_nodeset_ephemeral("fake_vlm", overlay_src, tag="ephem-1")
        assert "fake_vlm#ephem-1" in patched_registry._auto_servers
        n_stopped = patched_registry.unload_nodeset_ephemeral("ephem-1")
        assert n_stopped == 1
        assert _DummyServer.instances[0].stopped is True
        assert "fake_vlm#ephem-1" not in patched_registry._auto_servers

    asyncio.run(run())


def test_ephemeral_unload_idempotent(patched_registry: WorkspaceComponentRegistry) -> None:
    assert patched_registry.unload_nodeset_ephemeral("never-existed") == 0


def test_ephemeral_idempotent_same_tag(patched_registry: WorkspaceComponentRegistry, tmp_path: Path) -> None:
    async def run() -> None:
        overlay_src = tmp_path / "overlay" / "nodesets" / "fake_vlm.py"
        overlay_src.parent.mkdir(parents=True)
        overlay_src.write_text("class FakeNS: pass\n")
        url1 = await patched_registry.load_nodeset_ephemeral(
            "fake_vlm", overlay_src, tag="ephem-same"
        )
        url2 = await patched_registry.load_nodeset_ephemeral(
            "fake_vlm", overlay_src, tag="ephem-same"
        )
        assert url1 == url2
        assert len(_DummyServer.instances) == 1

    asyncio.run(run())


def test_ephemeral_unknown_nodeset_raises(
    patched_registry: WorkspaceComponentRegistry, tmp_path: Path
) -> None:
    async def run() -> None:
        overlay_src = tmp_path / "overlay" / "nodesets" / "missing.py"
        overlay_src.parent.mkdir(parents=True)
        overlay_src.write_text("# nothing\n")
        with pytest.raises(ValueError, match="Unknown nodeset"):
            await patched_registry.load_nodeset_ephemeral(
                "unknown_ns_xyz", overlay_src, tag="ephem-bad"
            )

    asyncio.run(run())


def test_ephemeral_does_not_touch_frozen(
    patched_registry: WorkspaceComponentRegistry, tmp_path: Path
) -> None:
    async def run() -> None:
        frozen = _DummyServer(name="frozen", command=["frozen"], port=9999)
        patched_registry._auto_servers["fake_vlm"] = frozen
        overlay_src = tmp_path / "overlay" / "nodesets" / "fake_vlm.py"
        overlay_src.parent.mkdir(parents=True)
        overlay_src.write_text("class FakeNS: pass\n")
        await patched_registry.load_nodeset_ephemeral("fake_vlm", overlay_src, tag="ephem-iter1")
        assert "fake_vlm" in patched_registry._auto_servers
        assert "fake_vlm#ephem-iter1" in patched_registry._auto_servers
        assert frozen.stopped is False
        patched_registry.unload_nodeset_ephemeral("ephem-iter1")
        assert "fake_vlm" in patched_registry._auto_servers
        assert frozen.stopped is False

    asyncio.run(run())
