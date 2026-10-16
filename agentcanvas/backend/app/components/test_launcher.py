"""Tests for ``launcher`` — the vehicle rule, store keys, source argv, and
the server object a hosted nodeset gets. Nothing is started: BaseServer /
ContainerServer constructors only record what ``start()`` would run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.components.launcher import (
    Launcher,
    auto_host_command,
    requires_server,
    server_key,
    source_args,
    split_server_key,
    vehicle_for,
)
from app.server.container_server import WORKSPACE_MOUNT


class _Local:
    server_python = None
    server_image = None


class _OtherPython:
    server_python = "/opt/envs/other/bin/python"
    server_image = None


class _SamePython:
    server_python = sys.executable
    server_image = None


class _Image:
    server_python = "/opt/envs/other/bin/python"  # image wins over python
    server_image = "agentcanvas/testimg:latest"
    server_image_gpu = True
    server_container_python = "/venv/bin/python"
    server_mounts: dict[str, str] = {}
    startup_timeout = 60


# ── vehicle ──


def test_vehicle_rule_precedence() -> None:
    assert vehicle_for(_Local) == "local"
    assert vehicle_for(_SamePython) == "local"  # this interpreter = in-process
    assert vehicle_for(_OtherPython) == "native"
    assert vehicle_for(_Image) == "container"
    # explicit server mode or a fan-out forces a native subprocess for a
    # nodeset that could run in-process; never a container without an image
    assert vehicle_for(_Local, mode="server") == "native"
    assert vehicle_for(_Local, worker_count=2) == "native"
    assert vehicle_for(_Image, worker_count=2) == "container"


def test_requires_server_is_not_just_python_set() -> None:
    assert not requires_server(_Local)
    assert not requires_server(_SamePython)
    assert requires_server(_OtherPython)
    assert requires_server(_Image)


# ── keys ──


def test_server_key_round_trip() -> None:
    assert server_key("env_x") == "env_x"
    assert server_key("env_x", 0) == "env_x#0"
    assert server_key("env_x", "ephem-1") == "env_x#ephem-1"
    assert split_server_key("env_x") == ("env_x", None)
    assert split_server_key("env_x#3") == ("env_x", "3")
    assert split_server_key("env_x#ephem-1") == ("env_x", "ephem-1")


# ── source argv ──


def test_source_args_single_file(tmp_path: Path) -> None:
    src = tmp_path / "workspace" / "nodesets" / "model" / "thing.py"
    argv, root = source_args(src)
    assert argv == ["--file", str(src)]
    assert root is None


def test_source_args_package_resolves_dotted_module_under_nodesets_root(tmp_path: Path) -> None:
    src = tmp_path / "overlay" / "nodesets" / "env" / "env_x" / "__init__.py"
    argv, root = source_args(src)
    assert argv == ["--module", "nodesets.env.env_x"]
    assert root == tmp_path / "overlay"


def test_source_args_package_outside_nodesets_or_bad_identifier_falls_back(tmp_path: Path) -> None:
    src = tmp_path / "elsewhere" / "pkg" / "__init__.py"
    assert source_args(src)[0] == ["--file", str(src)]
    bad = tmp_path / "overlay" / "nodesets" / "adas-subagent" / "__init__.py"
    assert source_args(bad)[0] == ["--file", str(bad)]


def test_auto_host_command_shape() -> None:
    assert auto_host_command("py", ["--file", "x.py"], "NS", 9000) == [
        "py",
        "-m",
        "app.server.auto_host",
        "--file",
        "x.py",
        "--class",
        "NS",
        "--port",
        "9000",
    ]


# ── Launcher.build ──


def test_build_native_uses_nodeset_python_and_backend_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTCANVAS_EXECUTOR_URL", "http://127.0.0.1:8123")
    launcher = Launcher(tmp_path)
    src = tmp_path / "workspace" / "nodesets" / "model" / "thing.py"
    srv = launcher.build(_OtherPython, source_file=src, label="auto_thing", port=9001)
    assert srv.name == "auto_thing"
    assert srv.port == 9001
    assert srv.command[0] == _OtherPython.server_python
    assert srv.command[1:3] == ["-m", "app.server.auto_host"]
    assert "--class" in srv.command and "_OtherPython" in srv.command
    assert srv.working_dir == str(tmp_path / "agentcanvas" / "backend")
    assert srv.env["PYTHONPATH"] == f"{tmp_path}/agentcanvas/backend:{tmp_path}"
    assert srv.env["AGENTCANVAS_EXECUTOR_URL"] == "http://127.0.0.1:8123"


def test_build_native_package_prepends_import_root_and_carries_server_env(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENTCANVAS_EXECUTOR_URL", "http://127.0.0.1:8123")

    class _WithEnv(_OtherPython):
        server_env = {"LD_PRELOAD": "/x.so"}

    launcher = Launcher(tmp_path)
    src = tmp_path / "overlay" / "nodesets" / "env" / "env_x" / "__init__.py"
    srv = launcher.build(_WithEnv, source_file=src, label="auto_env_x#1", port=9002)
    assert "--module" in srv.command and "nodesets.env.env_x" in srv.command
    assert srv.env["PYTHONPATH"].startswith(f"{tmp_path}/overlay:")
    assert srv.env["LD_PRELOAD"] == "/x.so"


def test_build_container_translates_paths_and_rewrites_executor_url(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENTCANVAS_EXECUTOR_URL", "http://127.0.0.1:8123")
    launcher = Launcher(tmp_path)
    src = tmp_path / "workspace" / "nodesets" / "model" / "thing.py"
    srv = launcher.build(_Image, source_file=src, label="auto_thing", port=9003)
    assert srv.image == _Image.server_image
    assert srv.gpu is True
    assert srv.inner_command[0] == "/venv/bin/python"
    assert f"{WORKSPACE_MOUNT}/workspace/nodesets/model/thing.py" in srv.inner_command
    assert srv.container_env["AGENTCANVAS_EXECUTOR_URL"] == "http://host.docker.internal:8123"
    assert srv.mounts[0] == f"{tmp_path}:{WORKSPACE_MOUNT}:ro"
