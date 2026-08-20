"""Unit tests for the registry's container-mode server assembly
(``_build_container_server``): host→mount path translation, server_mounts
resolution, and the loopback executor-URL rewrite. No docker involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.components.registry import WorkspaceComponentRegistry
from app.server.container_server import WORKSPACE_MOUNT


class _ContainerNodeSetCls:
    server_image = "agentcanvas/testimg:latest"
    server_image_gpu = False
    server_container_python = "python3"
    server_mounts: dict[str, str] = {}
    startup_timeout = 60


def _build(tmp_path: Path, nodeset_cls: type, source_arg: list[str]):
    root = tmp_path
    (root / "workspace").mkdir(exist_ok=True)
    reg = WorkspaceComponentRegistry(scan_dir=root / "workspace")
    return reg._build_container_server(
        nodeset_cls=nodeset_cls,
        server_label="auto_testns",
        image=nodeset_cls.server_image,
        port=9400,
        source_arg=source_arg,
        class_name="TestNS",
        pythonpath=f"{root}/agentcanvas/backend:{root}",
        workspace_root=str(root),
        executor_url="http://127.0.0.1:8000",
        ns_server_env={"FOO": "bar"},
    )


def test_paths_translate_under_workspace_mount(tmp_path: Path) -> None:
    srv = _build(
        tmp_path,
        _ContainerNodeSetCls,
        ["--file", f"{tmp_path}/workspace/nodesets/common/example.py"],
    )
    assert srv.inner_command[0] == "python3"
    assert (
        f"{WORKSPACE_MOUNT}/workspace/nodesets/common/example.py" in srv.inner_command
    )
    assert srv.container_env["PYTHONPATH"] == (
        f"{WORKSPACE_MOUNT}/agentcanvas/backend:{WORKSPACE_MOUNT}"
    )
    assert srv.mounts[0] == f"{tmp_path}:{WORKSPACE_MOUNT}:ro"


def test_module_source_arg_passes_through(tmp_path: Path) -> None:
    srv = _build(tmp_path, _ContainerNodeSetCls, ["--module", "nodesets.x.y"])
    assert "--module" in srv.inner_command
    assert "nodesets.x.y" in srv.inner_command


def test_source_outside_repo_root_is_a_hard_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="outside the repo root"):
        _build(tmp_path, _ContainerNodeSetCls, ["--file", "/somewhere/else/ns.py"])


def test_server_mounts_resolve_and_skip_missing(tmp_path: Path) -> None:
    (tmp_path / "data" / "weights").mkdir(parents=True)

    class _WithMounts(_ContainerNodeSetCls):
        server_mounts = {
            "data/weights": "/opt/weights:ro",
            "data/absent": "/opt/absent:ro",
        }

    srv = _build(
        tmp_path,
        _WithMounts,
        ["--file", f"{tmp_path}/workspace/ns.py"],
    )
    specs = srv.mounts
    assert f"{tmp_path}/data/weights:/opt/weights:ro" in specs
    assert not any("absent" in s for s in specs)


def test_missing_rw_mount_dir_is_created_not_skipped(tmp_path: Path) -> None:
    class _WithRwDrop(_ContainerNodeSetCls):
        server_mounts = {"outputs/drop": "/opt/out"}

    srv = _build(
        tmp_path,
        _WithRwDrop,
        ["--file", f"{tmp_path}/workspace/ns.py"],
    )
    assert (tmp_path / "outputs" / "drop").is_dir()
    assert f"{tmp_path}/outputs/drop:/opt/out" in srv.mounts


def test_container_user_passes_through(tmp_path: Path) -> None:
    class _WithUser(_ContainerNodeSetCls):
        server_container_user = "0:0"

    srv = _build(
        tmp_path,
        _WithUser,
        ["--file", f"{tmp_path}/workspace/ns.py"],
    )
    assert srv.user == "0:0"
    # the base class default keeps the image's USER
    srv_default = _build(
        tmp_path,
        _ContainerNodeSetCls,
        ["--file", f"{tmp_path}/workspace/ns.py"],
    )
    assert srv_default.user is None


def test_executor_url_rewritten_to_host_gateway(tmp_path: Path) -> None:
    srv = _build(
        tmp_path,
        _ContainerNodeSetCls,
        ["--file", f"{tmp_path}/workspace/ns.py"],
    )
    assert (
        srv.container_env["AGENTCANVAS_EXECUTOR_URL"]
        == "http://host.docker.internal:8000"
    )
    assert srv.container_env["FOO"] == "bar"
    assert srv.gpu is False
