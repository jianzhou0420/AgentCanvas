"""RemoteContainerProxy — subprocess-side handle to an executor-home state
container (face B of the cross-nodeset container-access prototype —
docs/.../tmp/state-container-cross-nodeset-access.html §2.3).

A server-mode node that holds an access-grant to an executor home container is
handed one of these under ``ctx.containers["<id>"]`` instead of a real
``StateContainer``. It exposes the SAME call surface (``read`` / ``write`` /
``evict``) so node code (``ctx.containers["x"].write(...)``) is unchanged —
that is the "position transparency" the design targets. Each call is forwarded
**synchronously** over HTTP to the executor's
``/api/internal/containers/{execution_id}/{container_id}/{op}`` endpoint; the
executor holds the single source of truth, nothing is cached here.

Caveats:
- Synchronous blocking ``httpx.Client`` inside the subprocess's async event
  loop — intentional: it matches the local ``StateContainer`` sync call surface
  byte-for-byte (node code calls ``ctx.containers["x"].read(...)`` synchronously).
  Making this non-blocking would require an async container API, a cross-nodeset
  ripple out of scope here.
- Transport is msgpack (``pack_body`` / ``unpack_body``): ndarray/torch/PIL ride
  raw-byte blobs and are restored on unpack, so no ``__ndarray__`` marker step.

3.8-safe: ``from __future__`` annotations + all imports lazy (this module is
imported inside the auto-hosted subprocess, which may run a pinned 3.8 interp).
"""

from __future__ import annotations

from typing import Any


class RemoteContainerProxy:
    """Forwards read/write/evict to an executor-home StateContainer over HTTP."""

    def __init__(self, base_url: str, execution_id: str, container_id: str) -> None:
        self._base = base_url.rstrip("/")
        self._execution_id = execution_id
        self._container_id = container_id

    def _post(self, op: str, payload: dict) -> dict:
        import httpx

        from . import serialization
        from ._loopback_proxy import loopback_httpx_kwargs

        url = "{}/api/internal/containers/{}/{}/{}".format(
            self._base, self._execution_id, self._container_id, op
        )
        with httpx.Client(timeout=30.0, **loopback_httpx_kwargs()) as client:
            resp = client.post(
                url,
                content=serialization.pack_body(payload),
                headers={"Content-Type": serialization.MSGPACK_CONTENT_TYPE},
            )
            resp.raise_for_status()
            return serialization.unpack_body(resp.content)

    def read(self, name: str, key: str | None = None) -> Any:
        # msgpack restores ndarray/torch/PIL on unpack — no marker step needed.
        data = self._post("read", {"name": name, "key": key, "execution_id": self._execution_id})
        return data.get("value")

    def write(self, name: str, data: Any, key: str | None = None) -> None:
        self._post(
            "write",
            {
                "name": name,
                "data": data,
                "key": key,
                "execution_id": self._execution_id,
            },
        )

    def evict(self, key: str) -> None:
        self._post("evict", {"key": key, "execution_id": self._execution_id})
