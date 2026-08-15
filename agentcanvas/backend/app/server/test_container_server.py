"""Unit tests for container server mode — command assembly and path
translation only. No docker daemon required; the full-stack container
round-trip is exercised by the integration smoke (mock image + trivial
nodeset), not here.
"""

from __future__ import annotations

from app.server.container_server import (
    WORKSPACE_MOUNT,
    ContainerServer,
    container_name,
)


def _server(**kw) -> ContainerServer:
    defaults = dict(
        image="agentcanvas/testimg:latest",
        inner_command=["python3", "-m", "app.server.auto_host", "--port", "9300"],
        port=9300,
        name="auto_testns",
    )
    defaults.update(kw)
    return ContainerServer(**defaults)


def test_container_name_flattens_worker_tag() -> None:
    assert container_name("auto_ns#1", 9301) == "ac_auto_ns_1_9301"


def test_command_publishes_loopback_port_and_host_gateway() -> None:
    cmd = _server()._docker_command(gpu=False)
    assert cmd[:4] == ["docker", "run", "--rm", "--init"]
    assert "127.0.0.1:9300:9300" in cmd
    assert "host.docker.internal:host-gateway" in cmd
    # image comes before the inner command, nothing after it but the argv
    idx = cmd.index("agentcanvas/testimg:latest")
    assert cmd[idx + 1 :] == ["python3", "-m", "app.server.auto_host", "--port", "9300"]


def test_gpu_flag_adds_cdi_device_only_when_requested() -> None:
    srv = _server(gpu=True)
    assert "nvidia.com/gpu=all" in srv._docker_command(gpu=True)
    assert "nvidia.com/gpu=all" not in srv._docker_command(gpu=False)


def test_mounts_and_container_env_ride_the_command() -> None:
    srv = _server(
        mounts=[f"/repo:{WORKSPACE_MOUNT}:ro", "/weights:/opt/w:ro"],
        container_env={"PYTHONPATH": f"{WORKSPACE_MOUNT}/agentcanvas/backend"},
    )
    cmd = srv._docker_command(gpu=False)
    assert f"/repo:{WORKSPACE_MOUNT}:ro" in cmd
    assert "/weights:/opt/w:ro" in cmd
    i = cmd.index("-e")
    assert cmd[i + 1] == f"PYTHONPATH={WORKSPACE_MOUNT}/agentcanvas/backend"


def test_client_env_targets_rootless_daemon() -> None:
    srv = _server()
    assert srv.env["DOCKER_HOST"].startswith("unix:///run/user/")
    assert srv.env["XDG_RUNTIME_DIR"].startswith("/run/user/")
