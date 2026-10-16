"""Launcher — how a nodeset gets a process: the vehicle decision and the command.

Server mode has two launch vehicles (ADR-020 lineage, ADR-server-005):

- ``"local"``      → in-process, no launch at all
- ``"native"``     → host subprocess: ``server_python -m app.server.auto_host``
- ``"container"``  → ``docker run <server_image> … -m app.server.auto_host``

This module is the one place that decides which vehicle a nodeset class
rides (:func:`vehicle_for`) and assembles the server object for it
(:meth:`Launcher.build`). Every caller that used to re-derive either —
the regular load, the ephemeral overlay spawn, ``list_nodesets``'
``requires_server``, the SDK catalog — asks here instead. The tagged
store-key shape (``name#k``) also lives here so the registry, the fleet,
and the resource sampler agree on it.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..server.base_server import BaseServer

log = logging.getLogger("agentcanvas.components.launcher")

Vehicle = Literal["local", "native", "container"]

AUTO_HOST_MODULE = "app.server.auto_host"


# ── Vehicle ──


def vehicle_for(nodeset_cls: type, *, mode: str = "local", worker_count: int = 1) -> Vehicle:
    """The launch vehicle for ``nodeset_cls``.

    One rule, in precedence order: a ``server_image`` always means the
    container (image wins over python); a ``server_python`` other than this
    interpreter means a native subprocess; an explicit ``mode="server"`` or
    a fan-out (``worker_count > 1``) forces a native subprocess for a
    nodeset that could otherwise run in-process; everything else is local.
    """
    if getattr(nodeset_cls, "server_image", None):
        return "container"
    sp = getattr(nodeset_cls, "server_python", None)
    if sp and sp != sys.executable:
        return "native"
    if mode == "server" or worker_count > 1:
        return "native"
    return "local"


def requires_server(nodeset_cls: type) -> bool:
    """True when the nodeset cannot run in this process (image or other python)."""
    return vehicle_for(nodeset_cls) != "local"


# ── Store keys ──


def server_key(name: str, tag: int | str | None = None) -> str:
    """The auto-server store key: ``name`` untagged, ``name#tag`` tagged."""
    return f"{name}#{tag}" if tag is not None else name


def split_server_key(key: str) -> tuple[str, str | None]:
    """Inverse of :func:`server_key`: ``(name, tag-or-None)``."""
    name, _, tag = key.partition("#")
    return name, (tag or None)


# ── Command assembly ──


def find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def source_args(source_file: str | Path) -> tuple[list[str], Path | None]:
    """auto_host's source argv for ``source_file``, plus an import root.

    Single-file nodesets ride ``--file <path>``. Package nodesets (a folder
    with ``__init__.py``) must be re-imported by dotted name in the
    subprocess so relative / absolute imports inside the package resolve:
    they ride ``--module <dotted>`` and the returned import root (the parent
    of the ``nodesets/`` dir containing the source) must be prepended to
    PYTHONPATH so the dotted name resolves to THIS source, not a
    frozen-workspace twin. Works uniformly for ``workspace/nodesets/…`` and
    any overlay (``active_workspace/nodesets/…``); a path whose components
    are not legal identifiers falls back to ``--file``.
    """
    src = Path(source_file)
    if src.name != "__init__.py":
        return ["--file", str(src)], None
    import_root: Path | None = None
    for parent in src.parents:
        if parent.name == "nodesets":
            import_root = parent.parent
            break
    if import_root is None:
        return ["--file", str(src)], None
    try:
        rel = src.parent.relative_to(import_root)
    except ValueError:
        return ["--file", str(src)], None
    dotted = rel.as_posix().replace("/", ".")
    if not all(comp.isidentifier() for comp in dotted.split(".")):
        return ["--file", str(src)], None
    return ["--module", dotted], import_root


def auto_host_command(python: str, source_arg: list[str], class_name: str, port: int) -> list[str]:
    """argv for one auto_host subprocess (no shell: a direct child, so
    PR_SET_PDEATHSIG watches us one-hop — see BaseServer.start())."""
    return [python, "-m", AUTO_HOST_MODULE, *source_arg, "--class", class_name, "--port", str(port)]


@dataclass
class Launcher:
    """Builds the server object that hosts one nodeset subprocess.

    ``workspace_root`` is the repo root (the parent of ``workspace/``); the
    backend dir and the base PYTHONPATH derive from it.
    """

    workspace_root: Path

    @property
    def backend_dir(self) -> Path:
        return self.workspace_root / "agentcanvas" / "backend"

    def pythonpath(self, import_root: Path | None = None) -> str:
        base = f"{self.backend_dir}:{self.workspace_root}"
        return f"{import_root}:{base}" if import_root is not None else base

    def build(
        self,
        nodeset_cls: type,
        *,
        source_file: str | Path,
        label: str,
        port: int,
        executor_url: str | None = None,
    ) -> BaseServer:
        """The server for ``nodeset_cls`` loaded from ``source_file``.

        Picks the vehicle with :func:`vehicle_for` (never ``"local"`` here:
        a caller that reaches the launcher has already decided to host).
        ``executor_url`` is the base URL the subprocess uses to call back
        to the executor's ``/api/internal``; an explicit
        ``AGENTCANVAS_EXECUTOR_URL`` in the environment wins, otherwise it
        resolves from Settings. The nodeset's ``server_env`` rides along.
        Nothing is started: the caller calls ``start()``.
        """
        from ..config import resolve_executor_url

        source_arg, import_root = source_args(source_file)
        pythonpath = self.pythonpath(import_root)
        class_name = nodeset_cls.__name__
        ns_server_env = getattr(nodeset_cls, "server_env", None) or {}
        executor_url = (
            executor_url or os.environ.get("AGENTCANVAS_EXECUTOR_URL") or resolve_executor_url()
        )
        if vehicle_for(nodeset_cls, mode="server") == "container":
            return build_container_server(
                nodeset_cls=nodeset_cls,
                server_label=label,
                image=nodeset_cls.server_image,
                port=port,
                source_arg=source_arg,
                class_name=class_name,
                pythonpath=pythonpath,
                workspace_root=str(self.workspace_root),
                executor_url=executor_url,
                ns_server_env=ns_server_env,
            )
        # Lazy: tests substitute BaseServer at its home module.
        from ..server.base_server import BaseServer

        python = getattr(nodeset_cls, "server_python", None) or sys.executable
        return BaseServer(
            name=label,
            command=auto_host_command(python, source_arg, class_name, port),
            port=port,
            startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
            working_dir=str(self.backend_dir),
            env={
                "PYTHONPATH": pythonpath,
                "AGENTCANVAS_EXECUTOR_URL": executor_url,
                **ns_server_env,
            },
        )


def build_container_server(
    *,
    nodeset_cls: type,
    server_label: str,
    image: str,
    port: int,
    source_arg: list[str],
    class_name: str,
    pythonpath: str,
    workspace_root: str,
    executor_url: str,
    ns_server_env: dict[str, str],
) -> Any:
    """Assemble a ContainerServer for a ``server_image`` nodeset.

    The container runs the stock auto_host from the read-only repo mount,
    so every host path riding the command line or PYTHONPATH is rewritten
    under WORKSPACE_MOUNT. Sources outside the repo root can't be
    translated — that's a hard error, not a silent wrong path.
    """
    from ..server.container_server import WORKSPACE_MOUNT, ContainerServer

    root = str(Path(workspace_root).resolve())

    def _translate(path: str) -> str:
        resolved = str(Path(path).resolve()) if path.startswith("/") else path
        if resolved.startswith(root):
            return WORKSPACE_MOUNT + resolved[len(root) :]
        raise RuntimeError(
            f"Container-mode nodeset {server_label}: source path {path} "
            f"is outside the repo root {root} and cannot be mounted."
        )

    inner_source_arg = [_translate(a) if a.startswith("/") else a for a in source_arg]
    inner_pythonpath = ":".join(
        _translate(p) if p.startswith("/") else p for p in pythonpath.split(":")
    )
    inner_command = auto_host_command(
        getattr(nodeset_cls, "server_container_python", "python3"),
        inner_source_arg,
        class_name,
        port,
    )

    mounts = [f"{root}:{WORKSPACE_MOUNT}:ro"]
    for host_path, container_spec in (getattr(nodeset_cls, "server_mounts", None) or {}).items():
        hp = Path(os.path.expandvars(os.path.expanduser(host_path)))
        if not hp.is_absolute():
            hp = Path(root) / hp
        if hp.exists():
            mounts.append(f"{hp.resolve()}:{container_spec}")
        elif not container_spec.endswith(":ro"):
            # A read-write mount is an output drop — create it rather
            # than skip it (a skipped rw mount silently strands the
            # container's writes inside the container filesystem).
            hp.mkdir(parents=True, exist_ok=True)
            mounts.append(f"{hp.resolve()}:{container_spec}")
            log.info("%s: created missing rw server_mounts dir %s", server_label, hp)
        else:
            log.warning(
                "%s: server_mounts host path %s missing — skipped (the "
                "capability needing it will error at call time)",
                server_label,
                hp,
            )

    # Loopback executor URLs are unreachable from inside the container;
    # host.docker.internal is mapped via --add-host=host-gateway.
    container_executor_url = executor_url.replace("127.0.0.1", "host.docker.internal").replace(
        "localhost", "host.docker.internal"
    )

    return ContainerServer(
        name=server_label,
        image=image,
        inner_command=inner_command,
        gpu=getattr(nodeset_cls, "server_image_gpu", False),
        user=getattr(nodeset_cls, "server_container_user", None),
        mounts=mounts,
        container_env={
            "PYTHONPATH": inner_pythonpath,
            "AGENTCANVAS_EXECUTOR_URL": container_executor_url,
            **ns_server_env,
        },
        port=port,
        startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
    )
