"""AutoHostClient — the one client of an auto_host subprocess.

Every caller that talks to a hosted nodeset (the proxy node's ``forward``,
the remote env panel, the Human tab driver, the registry's manifest and
env-panel fetches, the eval loop's per-episode eviction, the executor's
container broker) goes through this module: one place for the ``/call``
request/response envelope, msgpack-vs-JSON negotiation, transport
accounting, the env-panel and container routes, health, and one error
policy — :class:`ServerUnreachable` when the process cannot be reached,
:class:`CallError` when it answered with an error.

The transport is a seam: ``transport=`` takes any ``httpx`` transport, so
tests drive the client against an in-process ASGI app (or a
``MockTransport``) with no subprocess.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..components.bases import FireList, firelist_from_dict
from . import serialization
from ._loopback_proxy import loopback_httpx_kwargs
from .manifest import ServerManifest
from .serialization import deserialize_value, serialize_value

log = logging.getLogger("agentcanvas.server-client")

HEALTH_TIMEOUT_S = 5.0


class CallError(RuntimeError):
    """The server answered with an error (non-2xx), or its reply was unreadable."""

    def __init__(self, message: str, *, status: int | None = None, url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class ServerUnreachable(CallError):
    """The server process could not be reached (connect / read failure)."""


@dataclass
class CallResult:
    """What one ``/call`` produced, decoded."""

    outputs: dict[str, Any] = field(default_factory=dict)
    # A hosted spawner's FireList (outputs is then {}).
    firelist: FireList | None = None
    # Read-only previews of the nodeset's owned containers (display only).
    containers: dict = field(default_factory=dict)
    # The hosted node's own ``_self_log`` entries for this call.
    log: list[dict] = field(default_factory=list)
    # LLM usage the hosted node accumulated during this call, if any.
    usage: dict | None = None
    # Round-trip accounting: rtt_ms, req_bytes, resp_bytes, serialize_ms, deserialize_ms.
    transport: dict = field(default_factory=dict)


class AutoHostClient:
    """A client bound to one auto_host base URL.

    ``json_only`` keeps the legacy JSON regime (per-port wire-type
    serialization, base64 images) — the migration window a few callers
    still rely on; the default is msgpack when available.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        json_only: bool = False,
        transport: Any = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._timeout = timeout
        self._msgpack = serialization.MSGPACK_OK and not json_only
        self._transport = transport

    @property
    def url(self) -> str:
        return self._url

    def _client_kwargs(self, timeout: float | None = None) -> dict:
        kwargs: dict = {
            "base_url": self._url,
            "timeout": timeout if timeout is not None else self._timeout,
            **loopback_httpx_kwargs(),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return kwargs

    # ── /call ──

    def _encode_call(
        self,
        inputs: dict,
        config: dict | None,
        exec_ctx: dict | None,
        input_wire_types: dict[str, str] | None,
    ) -> tuple[dict, int, float]:
        """The request kwargs for ``post``, the body size, and the serialize time."""
        payload: dict[str, Any] = {}
        if input_wire_types is None:
            payload = {k: v for k, v in inputs.items() if v is not None}
        else:
            # Native values ride msgpack (the codec is type-driven); only the
            # JSON regime needs per-port wire-type serialization.
            for port_name, wire_type in input_wire_types.items():
                val = inputs.get(port_name)
                if val is None:
                    continue
                payload[port_name] = val if self._msgpack else serialize_value(val, wire_type)
        body: dict[str, Any] = {"inputs": payload, "config": config or {}}
        if exec_ctx:
            body["_exec"] = exec_ctx
        if not self._msgpack:
            return {"json": body}, 0, 0.0
        t = time.perf_counter()
        packed = serialization.pack_body(body)
        return (
            {
                "content": packed,
                "headers": {
                    "Content-Type": serialization.MSGPACK_CONTENT_TYPE,
                    "Accept": serialization.MSGPACK_CONTENT_TYPE,
                },
            },
            len(packed),
            (time.perf_counter() - t) * 1000,
        )

    def _decode_call(
        self,
        resp: httpx.Response,
        output_wire_types: dict[str, str] | None,
        *,
        rtt_ms: float,
        req_bytes: int,
        serialize_ms: float,
    ) -> CallResult:
        self._raise_for_status(resp)
        is_msgpack = resp.headers.get("content-type", "").startswith(
            serialization.MSGPACK_CONTENT_TYPE
        )
        t = time.perf_counter()
        try:
            data = serialization.unpack_body(resp.content) if is_msgpack else resp.json()
        except Exception as e:
            raise CallError(f"unreadable reply: {e}", url=str(resp.url)) from e
        deserialize_ms = (time.perf_counter() - t) * 1000
        transport = {
            "rtt_ms": rtt_ms,
            "req_bytes": req_bytes,
            "resp_bytes": len(resp.content),
            "serialize_ms": serialize_ms,
            "deserialize_ms": deserialize_ms,
        }
        serialization.accumulate_transport(**transport)
        if not isinstance(data, dict):
            raise CallError(f"unexpected reply shape: {type(data).__name__}", url=str(resp.url))

        result = CallResult(
            containers=data.get("containers") or {},
            log=list(data.get("log") or []),
            usage=data.get("usage") or None,
            transport=transport,
        )
        if "__firelist__" in data:
            result.firelist = firelist_from_dict(data)
            return result
        raw_outputs = data.get("outputs", data)
        if output_wire_types is None:
            result.outputs = dict(raw_outputs)
            return result
        for port_name, wire_type in output_wire_types.items():
            val = raw_outputs.get(port_name)
            if val is None:
                continue
            result.outputs[port_name] = val if is_msgpack else deserialize_value(val, wire_type)
        return result

    async def call(
        self,
        function_name: str,
        inputs: dict,
        config: dict | None = None,
        *,
        exec_ctx: dict | None = None,
        input_wire_types: dict[str, str] | None = None,
        output_wire_types: dict[str, str] | None = None,
    ) -> CallResult:
        """``POST /call/{function_name}``. Wire types (port → wire_type)
        drive the JSON regime's per-port codec; without them values pass
        through as-is. ``exec_ctx`` is the cross-nodeset container grant."""
        kwargs, req_bytes, ser_ms = self._encode_call(inputs, config, exec_ctx, input_wire_types)
        path = f"/call/{function_name}"
        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                t = time.perf_counter()
                resp = await client.post(path, **kwargs)
                rtt_ms = (time.perf_counter() - t) * 1000
        except httpx.TransportError as e:
            raise ServerUnreachable(
                f"Server unreachable: {self._url} ({e.__class__.__name__})", url=self._url + path
            ) from e
        return self._decode_call(
            resp, output_wire_types, rtt_ms=rtt_ms, req_bytes=req_bytes, serialize_ms=ser_ms
        )

    def call_sync(
        self,
        function_name: str,
        inputs: dict,
        config: dict | None = None,
        *,
        exec_ctx: dict | None = None,
        input_wire_types: dict[str, str] | None = None,
        output_wire_types: dict[str, str] | None = None,
    ) -> CallResult:
        """Blocking twin of :meth:`call` for synchronous drivers."""
        kwargs, req_bytes, ser_ms = self._encode_call(inputs, config, exec_ctx, input_wire_types)
        path = f"/call/{function_name}"
        try:
            with httpx.Client(**self._client_kwargs()) as client:
                t = time.perf_counter()
                resp = client.post(path, **kwargs)
                rtt_ms = (time.perf_counter() - t) * 1000
        except httpx.TransportError as e:
            raise ServerUnreachable(
                f"Server unreachable: {self._url} ({e.__class__.__name__})", url=self._url + path
            ) from e
        return self._decode_call(
            resp, output_wire_types, rtt_ms=rtt_ms, req_bytes=req_bytes, serialize_ms=ser_ms
        )

    # ── plain JSON routes (env panel, manifest, health, containers) ──

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.is_success:
            return
        detail: Any = None
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text[:500]
        raise CallError(
            f"{resp.status_code} from {resp.request.method} {resp.url}: {detail}",
            status=resp.status_code,
            url=str(resp.url),
        )

    async def _json(self, method: str, path: str, body: dict | None = None) -> Any:
        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                resp = await client.request(method, path, json=body)
        except httpx.TransportError as e:
            raise ServerUnreachable(
                f"Server unreachable: {self._url} ({e.__class__.__name__})", url=self._url + path
            ) from e
        self._raise_for_status(resp)
        return resp.json()

    def _json_sync(
        self, method: str, path: str, body: dict | None = None, timeout: float | None = None
    ) -> Any:
        try:
            with httpx.Client(**self._client_kwargs(timeout)) as client:
                resp = client.request(method, path, json=body)
        except httpx.TransportError as e:
            raise ServerUnreachable(
                f"Server unreachable: {self._url} ({e.__class__.__name__})", url=self._url + path
            ) from e
        self._raise_for_status(resp)
        return resp.json()

    async def panel_state(self) -> dict:
        return await self._json("GET", "/env-panel/state")

    async def panel_options(self, field_name: str) -> list:
        return await self._json("GET", f"/env-panel/options/{field_name}")

    async def panel_field(self, name: str, value: Any) -> dict:
        return await self._json("POST", f"/env-panel/field/{name}", {"value": value})

    async def panel_action(self, name: str, params: dict | None = None) -> dict:
        return await self._json("POST", f"/env-panel/action/{name}", {"params": params or {}})

    def panel_info_sync(self) -> dict:
        return self._json_sync("GET", "/env-panel/info")

    def panel_field_sync(self, name: str, value: Any) -> dict:
        return self._json_sync("POST", f"/env-panel/field/{name}", {"value": value})

    def panel_action_sync(self, name: str, params: dict | None = None) -> dict:
        return self._json_sync("POST", f"/env-panel/action/{name}", {"params": params or {}})

    def manifest_sync(self) -> ServerManifest:
        return ServerManifest.from_dict(self._json_sync("GET", "/manifest"))

    async def health(self, timeout: float = HEALTH_TIMEOUT_S) -> bool:
        """True iff ``GET /health`` answers 200. Never raises."""
        try:
            async with httpx.AsyncClient(**self._client_kwargs(timeout)) as client:
                return (await client.get("/health")).status_code == 200
        except Exception:
            return False

    def health_sync(self, timeout: float = HEALTH_TIMEOUT_S) -> bool:
        """Blocking twin of :meth:`health`. Never raises."""
        try:
            with httpx.Client(**self._client_kwargs(timeout)) as client:
                return client.get("/health").status_code == 200
        except Exception:
            return False

    # ── containers (msgpack, same transport as the executor broker) ──

    async def evict(self, key: str) -> dict:
        """Nodeset-wide per-key eviction (``POST /containers/evict``)."""
        return await self._json("POST", "/containers/evict", {"key": key})

    async def container_op(self, container_id: str, op: str, payload: dict) -> dict:
        """``read`` / ``write`` / ``evict`` on one owned container."""
        path = f"/containers/{container_id}/{op}"
        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                resp = await client.post(
                    path,
                    content=serialization.pack_body(payload),
                    headers={"Content-Type": serialization.MSGPACK_CONTENT_TYPE},
                )
        except httpx.TransportError as e:
            raise ServerUnreachable(
                f"Server unreachable: {self._url} ({e.__class__.__name__})", url=self._url + path
            ) from e
        self._raise_for_status(resp)
        return serialization.unpack_body(resp.content)
