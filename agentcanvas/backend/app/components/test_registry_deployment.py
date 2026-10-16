"""Tests for the registry's deployment views — the public surface the
scheduler, the snapshot endpoint, the resource sampler, and the eval API
read instead of ``_auto_servers`` / ``_live_nodesets``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import WorkspaceComponentRegistry


class _Srv:
    def __init__(self, url: str, pid: int | None = None) -> None:
        self.url = url
        self._pid = pid


def _registry(tmp_path: Path) -> WorkspaceComponentRegistry:
    return WorkspaceComponentRegistry(scan_dir=tmp_path / "workspace")


def test_server_urls_collapse_tags_untagged_wins(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg._auto_servers["env_x#1"] = _Srv("http://h:1")
    reg._auto_servers["env_x#0"] = _Srv("http://h:0")
    reg._auto_servers["vlm"] = _Srv("http://h:7")
    reg._auto_servers["vlm#ephem-a"] = _Srv("http://h:8")
    assert reg.server_urls() == {"env_x": "http://h:1", "vlm": "http://h:7"}


def test_server_pids_collapse_tags(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg._auto_servers["env_x#0"] = _Srv("u", pid=11)
    reg._auto_servers["env_x#1"] = _Srv("u", pid=12)
    reg._auto_servers["vlm"] = _Srv("u", pid=13)
    reg._auto_servers["remote"] = _Srv("u", pid=None)  # attached, not owned
    assert reg.server_pids() == {11: "env_x", 12: "env_x", 13: "vlm"}


def test_loaded_nodeset_names_spans_local_and_hosted(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg._live_nodesets["local_ns"] = object()  # type: ignore[assignment]
    reg._auto_servers["env_x#0"] = _Srv("u")
    reg._auto_servers["env_x#1"] = _Srv("u")
    assert reg.loaded_nodeset_names() == ["env_x", "local_ns"]


def test_parallelism_reads_class_then_falls_back_to_env_prefix(tmp_path: Path) -> None:
    reg = _registry(tmp_path)

    class _Shared:
        parallelism = "shared"

    class _Rep:
        parallelism = "replicated"

    reg._discovered_nodesets["env_declared_shared"] = _Shared()  # type: ignore[assignment]
    reg._discovered_nodesets["model_rep"] = _Rep()  # type: ignore[assignment]
    assert reg.parallelism("env_declared_shared") == "shared"  # class wins
    assert reg.parallelism("model_rep") == "replicated"
    # import-failed on this env: env_ prefix and the known set are replicated
    assert reg.parallelism("env_never_imported") == "replicated"
    assert reg.parallelism("env_habitat") == "replicated"
    assert reg.parallelism("model_never_imported") == "shared"


def test_nodeset_instance_prefers_live(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    disc: Any = object()
    live: Any = object()
    reg._discovered_nodesets["x"] = disc
    assert reg.nodeset_instance("x") is disc
    reg._live_nodesets["x"] = live
    assert reg.nodeset_instance("x") is live
    assert reg.nodeset_instance("nope") is None
