"""ContainerServer — launch a server-mode nodeset inside a Docker container.

Server mode has two launch vehicles (ADR-020 lineage); this is the second:

- local mode          → in-process (no launch at all)
- Native Launch       → host subprocess: ``server_python -m app.server.auto_host``
- Container Launch    → ``docker run <server_image> … -m app.server.auto_host``

The container serves the SAME protocol (``GET /manifest``, ``POST /call/{fn}``,
``GET /health``, optional ``/mcp``) on a port published to ``127.0.0.1``, so
everything downstream of :class:`BaseServer` (health polling, manifest fetch,
proxy-node generation, env-panel bridge, multi-worker fan-out) is unchanged.

Design invariants (generalized from the model_pyslam container bridge, the
validated prototype this replaces at the framework level):

- **No framework code baked into images.** The repo root is bind-mounted
  read-only at :data:`WORKSPACE_MOUNT`; the container runs the stock
  ``app.server.auto_host`` from that mount. Images only need the model's own
  deps plus the small ``requirements-serve.txt`` set (fastapi/uvicorn/…).
- **Rootless daemon reached explicitly.** ``DOCKER_HOST`` / ``XDG_RUNTIME_DIR``
  are forced to the user socket so a shell-inherited system-daemon override
  can't point at a store that lacks the image.
- **GPU via CDI.** ``--device nvidia.com/gpu=all`` (legacy ``--gpus`` fails on
  rootless + cgroup v1). When the GPU start fails and the nodeset didn't
  hard-require GPU, the start is retried once without the device flag.
- **Foreground ``docker run`` as a direct child.** BaseServer's liveness
  polling, pdeathsig chain, and PGID kill apply to the docker client, which
  proxies SIGTERM to PID 1 in the container (``--init`` reaps). ``--rm`` plus
  a deterministic ``--name`` (pre-reaped on start) covers the one orphan case
  the conda path doesn't have: the client dying by SIGKILL while the container
  lives on.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

from .base_server import BaseServer

log = logging.getLogger("agentcanvas.server")

WORKSPACE_MOUNT = "/opt/agentcanvas-workspace"
GPU_DEVICE_ARGS = ["--device", "nvidia.com/gpu=all"]


def rootless_docker_env() -> dict[str, str]:
    """Env pointing the docker CLI at the current user's rootless daemon."""
    uid = os.getuid()
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    env["DOCKER_HOST"] = f"unix:///run/user/{uid}/docker.sock"
    return env


def container_name(server_label: str, port: int) -> str:
    """Deterministic docker name for a server label (``#`` is not a legal
    docker-name char, so worker tags are flattened)."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", server_label)
    return f"ac_{safe}_{port}"


def image_exists(image: str) -> bool:
    """True when ``image`` is present in the rootless daemon's local store."""
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        env=rootless_docker_env(),
        capture_output=True,
    )
    return proc.returncode == 0


class ContainerServer(BaseServer):
    """BaseServer whose process is a foreground ``docker run``.

    Constructor keywords (beyond BaseServer's):
    - ``image`` — docker image tag (required).
    - ``inner_command`` — argv executed inside the container (the stock
      auto_host invocation with container-side paths).
    - ``gpu`` — request the GPU via CDI; degrades to CPU on start failure.
    - ``mounts`` — extra ``-v`` specs (host:container[:ro]), already resolved.
    - ``container_env`` — env vars set inside the container (``-e``); the
      inherited ``env`` dict still applies to the docker *client* process.
    """

    def __init__(
        self,
        *,
        image: str,
        inner_command: list[str],
        gpu: bool = False,
        mounts: list[str] | None = None,
        container_env: dict[str, str] | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__(**overrides)
        self.image = image
        self.inner_command = list(inner_command)
        self.gpu = gpu
        self.mounts = list(mounts or [])
        self.container_env = dict(container_env or {})
        # The docker client must talk to the rootless daemon regardless of
        # what the backend inherited from its shell.
        self.env = {**self.env, **{
            k: v
            for k, v in rootless_docker_env().items()
            if k in ("DOCKER_HOST", "XDG_RUNTIME_DIR")
        }}

    # ── command assembly ──

    def _docker_command(self, *, gpu: bool) -> list[str]:
        name = container_name(self.name, self.port)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{self.port}:{self.port}",
            # Lets in-container code reach the executor's /api/internal
            # (container-access face B, reverse log push) via
            # host.docker.internal — the registry rewrites loopback executor
            # URLs to that name.
            "--add-host",
            "host.docker.internal:host-gateway",
        ]
        if gpu:
            cmd += GPU_DEVICE_ARGS
        for spec in self.mounts:
            cmd += ["-v", spec]
        for key, val in self.container_env.items():
            cmd += ["-e", f"{key}={val}"]
        cmd += [self.image, *self.inner_command]
        return cmd

    def _reap_stale_container(self) -> None:
        """Remove a leftover container from a previous SIGKILL'd client."""
        name = container_name(self.name, self.port)
        proc = subprocess.run(
            ["docker", "rm", "-f", name],
            env=rootless_docker_env(),
            capture_output=True,
            text=True,
        )
        # docker >= 23 exits 0 even when nothing matched; it echoes the
        # name only when a container was actually removed.
        if proc.returncode == 0 and proc.stdout.strip():
            log.warning("Reaped stale container %s before start", name)

    # ── lifecycle ──

    def start(self) -> None:
        if not image_exists(self.image):
            raise RuntimeError(
                f"Server image '{self.image}' not found in the rootless docker "
                f"store. Build it (see the nodeset's docker/ dir) or pull it, "
                f"then reload the nodeset."
            )
        self._reap_stale_container()
        self.command = self._docker_command(gpu=self.gpu)
        try:
            super().start()
        except RuntimeError:
            if not self.gpu:
                raise
            # CDI absent / GPU busy — one degrade attempt without the device
            # flag (matches the pyslam-bridge behaviour this generalizes).
            log.warning(
                "GPU container start failed for %s; retrying without GPU",
                self.name,
            )
            self._reap_stale_container()
            self.command = self._docker_command(gpu=False)
            super().start()

    def stop(self) -> None:
        # `docker stop` first: it delivers SIGTERM to the containerized
        # service and waits, which `--rm`s the container. Killing only the
        # client (super().stop()) also works when attached, but going through
        # the daemon is the authoritative path and covers a detached client.
        name = container_name(self.name, self.port)
        subprocess.run(
            ["docker", "stop", "-t", "10", name],
            env=rootless_docker_env(),
            capture_output=True,
        )
        super().stop()
