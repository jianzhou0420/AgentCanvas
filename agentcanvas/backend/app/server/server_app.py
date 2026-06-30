"""ServerApp — base class for building server-mode processes.

This is the **server-side** base class. External systems subclass this
to build a FastAPI service that speaks the AgentCanvas manifest protocol.

The resulting service exposes:
- ``GET /manifest``  — function schemas with typed ports
- ``POST /call/{fn}`` — invoke a function with auto-serialized I/O
- ``GET /health``    — liveness check

Usage::

    class MySLAMApp(ServerApp):
        name = "slam"
        port = 9001

        def get_functions(self):
            return [ServerFunction(name="localize", ...)]

    if __name__ == "__main__":
        MySLAMApp().serve()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from . import event_push, serialization
from .manifest import FunctionSchema, PortSchema, ServerManifest
from .serialization import deserialize_value, serialize_value

log = logging.getLogger(__name__)


@dataclass
class ServerFunction:
    """A single callable function exposed by the server."""

    name: str
    description: str = ""
    input_ports: list[PortSchema] = field(default_factory=list)
    output_ports: list[PortSchema] = field(default_factory=list)
    config_schema: dict = field(default_factory=dict)
    ui_config: dict = field(default_factory=dict)
    handler: Callable[..., Any] = field(default=None)  # type: ignore[assignment]


class ServerApp(ABC):
    """Base class for server-mode processes.

    Subclass this and implement :meth:`get_functions` to build a service.
    Handlers receive native Python types — serialization is automatic.
    """

    name: str = "unnamed"
    description: str = ""
    version: str = "1.0"
    port: int = 9000

    def __init__(self) -> None:
        self._functions: dict[str, ServerFunction] = {}
        self._port_maps: dict[str, dict] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=self.name,
        )

    @abstractmethod
    def get_functions(self) -> list[ServerFunction]:
        """Declare all functions this server exposes."""
        ...

    async def on_startup(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Called when the server starts. Override for initialization."""

    async def on_shutdown(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Called when the server stops. Override for cleanup."""

    def get_env_panel_instance(self) -> Optional[Any]:
        """Return a BaseEnvPanel instance hosted by this server, or None.

        AutoServerApp overrides this to expose its nodeset's env panel (if
        any) so the canvas-side ``RemoteEnvPanelProxy`` can forward calls
        across the subprocess boundary.
        """
        return None

    def get_container_previews(self) -> dict:
        """Return read-only previews of nodeset-owned containers, or ``{}``.

        AutoServerApp overrides this to expose its subprocess-local containers'
        ``get_preview()`` so the ``/call`` response can piggyback them back to
        the canvas for display. Default: none.
        """
        return {}

    def get_owned_container(self, container_id: str) -> Any:
        """Return a nodeset-owned ``StateContainer`` by id, or ``None``.

        AutoServerApp overrides this so the cross-nodeset container-access
        endpoints (``/containers/{id}/read|write``) can serve a container this
        subprocess homes — the executor broker forwards faces A/C here. Default:
        owns nothing.
        """
        return None

    def evict_container_key(self, key: str) -> int:
        """Evict one key's partition from every nodeset-owned container.

        AutoServerApp overrides this to call ``container.evict(key)`` on each
        subprocess-local owned container — the worker-safe per-episode cleanup
        invoked from the eval worker loop at episode end. Default: no-op
        (this server owns no containers). Returns the number of containers
        touched.
        """
        return 0

    async def run_blocking(self, fn: Callable, *args: Any) -> Any:
        """Run a blocking function in the thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def get_manifest(self) -> ServerManifest:
        """Build the server manifest from registered functions."""
        self._register_functions()
        functions = []
        for fn in self._functions.values():
            functions.append(
                FunctionSchema(
                    name=fn.name,
                    description=fn.description,
                    input_ports=list(fn.input_ports),
                    output_ports=list(fn.output_ports),
                    config_schema=fn.config_schema,
                    ui_config=fn.ui_config,
                )
            )
        return ServerManifest(
            name=self.name,
            version=self.version,
            description=self.description,
            functions=functions,
        )

    def _register_functions(self) -> None:
        if self._functions:
            return
        for fn in self.get_functions():
            self._functions[fn.name] = fn
            self._port_maps[fn.name] = {
                "inputs": {p.name: p.wire_type for p in fn.input_ports},
                "outputs": {p.name: p.wire_type for p in fn.output_ports},
            }

    def _build_app(self) -> Any:
        """Create the FastAPI application with manifest + call + health routes."""
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware

        self._register_functions()

        app = FastAPI(title=f"AgentCanvas Server: {self.name}")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self

        @app.on_event("startup")
        async def _startup() -> None:
            await server.on_startup()

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            await server.on_shutdown()
            server._executor.shutdown(wait=False)

        @app.get("/manifest")
        async def manifest() -> dict:
            return server.get_manifest().to_dict()

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "name": server.name, "version": server.version}

        @app.post("/call/{function_name}")
        # NOTE: takes a raw ``Request`` so the route accepts BOTH msgpack
        # (Move 1, default) and JSON (migration window), sniffed by
        # Content-Type. ``Request``/``Response`` are imported at module top so
        # FastAPI's runtime forwardref evaluation resolves them even under the
        # pinned Py3.8 auto-host interpreter (see ruff per-file-ignore). No
        # ``-> dict`` annotation: the route returns a ``Response`` directly.
        async def call_function(function_name: str, request: Request):
            import json

            raw = await request.body()
            use_msgpack = serialization.MSGPACK_CONTENT_TYPE in request.headers.get(
                "content-type", ""
            )
            if not raw:
                body: dict = {}
            elif use_msgpack:
                body = serialization.unpack_body(raw)
            else:
                body = json.loads(raw)

            def _respond(obj: dict):
                # Drain events buffered during this call (Move 3) so server-node
                # logs reach the canvas at call granularity.
                event_push.flush()
                # msgpack request → msgpack response; JSON request → JSON
                # response. The client knows which it sent and decodes to match.
                if use_msgpack:
                    return Response(
                        content=serialization.pack_body(obj),
                        media_type=serialization.MSGPACK_CONTENT_TYPE,
                    )
                return JSONResponse(obj)

            fn = server._functions.get(function_name)
            if fn is None:
                raise HTTPException(404, f"Unknown function: {function_name}")
            if fn.handler is None:
                raise HTTPException(501, f"No handler for: {function_name}")

            raw_inputs = body.get("inputs", body)
            port_map = server._port_maps[function_name]
            inputs: dict[str, Any] = {}
            for port_name, wire_type in port_map["inputs"].items():
                if port_name in raw_inputs:
                    # msgpack inputs are already native (decoded on unpack);
                    # only JSON needs per-port wire-type deserialization.
                    val = raw_inputs[port_name]
                    inputs[port_name] = val if use_msgpack else deserialize_value(val, wire_type)

            config = body.get("config", {})

            # Cross-nodeset container-access prototype (face B): ``_exec`` carries
            # granted executor-home container ids + execution_id. Pass it only to
            # handlers that accept a third arg (auto-generated handlers do);
            # legacy 2-arg handlers (manual ServerFunction) are called unchanged.
            # Arity is inspected (not try/except) so an internal TypeError in the
            # handler is never mistaken for a signature mismatch + double-called.
            exec_ctx = body.get("_exec")
            import inspect

            try:
                _nparams = len(inspect.signature(fn.handler).parameters)
            except (ValueError, TypeError):
                _nparams = 2

            try:
                if _nparams >= 3:
                    result = await fn.handler(inputs, config, exec_ctx)
                else:
                    result = await fn.handler(inputs, config)
            except Exception as e:
                log.exception("Handler %s failed", function_name)
                # First-class the error (Move 3, #54): push it to the executor so
                # it surfaces on the canvas instead of being demoted to a
                # swallowed {"error": ...} value by the proxy.
                import traceback as _tb

                _exid = exec_ctx.get("execution_id") if isinstance(exec_ctx, dict) else None
                event_push.emit_event(
                    "error",
                    str(e),
                    code="SUBPROC_NODE_FAIL",
                    node_id=function_name,
                    nodeset=getattr(server, "name", None),
                    execution_id=_exid,
                    details={"function": function_name, "traceback": _tb.format_exc()},
                )
                event_push.flush()
                raise HTTPException(500, str(e)) from e

            # C.5 Dynamic Fire-List handover: server-side DynamicFireListNode
            # subclasses return a ``FireList`` object instead of a port-output
            # dict. Serialize to a JSON-safe envelope; the framework-side
            # proxy reconstructs a FireList from this shape and the engine
            # then dispatches each child via ``_fire_dynamic_children``.
            # FireSpec fields are all primitive / dict-of-primitive so a
            # shallow dataclass-to-dict conversion suffices.
            from ..components.bases import FireList as _FireList

            if isinstance(result, _FireList):
                return _respond(
                    {
                        "__firelist__": {
                            "specs": [
                                {
                                    "node_type": s.node_type,
                                    "inputs": s.inputs,
                                    "config": s.config,
                                    "label": s.label,
                                    "capture_outputs": s.capture_outputs,
                                }
                                for s in result.specs
                            ],
                            "spawner_outputs": result.spawner_outputs,
                            "aggregator": result.aggregator,
                        }
                    }
                )

            outputs: dict[str, Any] = {}
            for port_name, wire_type in port_map["outputs"].items():
                if port_name in result:
                    # msgpack outputs stay native (the codec encodes binary
                    # types on pack); only JSON needs per-port serialization.
                    val = result[port_name]
                    outputs[port_name] = val if use_msgpack else serialize_value(val, wire_type)

            response: dict[str, Any] = {"outputs": outputs}
            # Piggyback read-only owned-container previews for display (no new
            # channel). Empty for nodesets that own no containers.
            previews = server.get_container_previews()
            if previews:
                response["containers"] = previews
            return _respond(response)

        @app.post("/containers/evict")
        # ``Optional[dict]`` (not ``dict | None``) — see the /call note above:
        # FastAPI evaluates signatures at runtime under a pinned Py3.8 host.
        async def evict_container(body: Optional[dict] = None) -> dict:
            """Worker-safe per-key eviction of a nodeset-owned container.

            The only mutation channel besides ``/call``. Called from the eval
            worker loop at episode end with the same key (``episode_id``) the
            nodes used; drops just that key's partition, never siblings."""
            body = body or {}
            key = body.get("key")
            if key is None:
                raise HTTPException(400, "evict requires 'key'")
            evicted = server.evict_container_key(key)
            return {"evicted": evicted, "key": key}

        # Cross-nodeset container-access prototype (faces A/C): read/write a
        # container THIS subprocess homes. The executor broker forwards here
        # when a node in another process holds a grant to it. Same JSON shape
        # as the executor's /api/internal/containers endpoint.
        # Same msgpack transport as the executor broker (/api/internal/...).
        async def _container_body(request: Request) -> dict:
            raw = await request.body()
            return serialization.unpack_body(raw) if raw else {}

        def _container_respond(obj: dict) -> Response:
            return Response(
                content=serialization.pack_body(obj),
                media_type=serialization.MSGPACK_CONTENT_TYPE,
            )

        @app.post("/containers/{container_id}/read")
        async def container_read(container_id: str, request: Request):
            body = await _container_body(request)
            c = server.get_owned_container(container_id)
            if c is None:
                raise HTTPException(404, f"unknown owned container {container_id!r}")
            try:
                val = c.read(body.get("name"), key=body.get("key"))
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            return _container_respond({"value": val})

        @app.post("/containers/{container_id}/write")
        async def container_write(container_id: str, request: Request):
            body = await _container_body(request)
            c = server.get_owned_container(container_id)
            if c is None:
                raise HTTPException(404, f"unknown owned container {container_id!r}")
            try:
                c.write(body.get("name"), body.get("data"), key=body.get("key"))
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            return _container_respond({"ok": True})

        @app.post("/containers/{container_id}/evict")
        async def container_evict_one(container_id: str, request: Request):
            body = await _container_body(request)
            key = body.get("key")
            if key is None:
                raise HTTPException(400, "evict requires 'key'")
            c = server.get_owned_container(container_id)
            if c is None:
                raise HTTPException(404, f"unknown owned container {container_id!r}")
            c.evict(key)
            return _container_respond({"ok": True})

        # Optional env panel bridge — if this server hosts a BaseEnvPanel,
        # expose schema + state + field/action endpoints so the agentcanvas
        # main process can register a RemoteEnvPanelProxy.
        panel = server.get_env_panel_instance()
        if panel is not None:
            from dataclasses import asdict

            @app.get("/env-panel/info")
            async def env_panel_info() -> dict:
                return {
                    "name": getattr(panel, "name", ""),
                    "display_name": getattr(panel, "display_name", ""),
                    "fields": [asdict(f) for f in getattr(panel, "fields", [])],
                    "actions": [asdict(a) for a in getattr(panel, "actions", [])],
                }

            @app.get("/env-panel/state")
            async def env_panel_state() -> dict:
                return await panel.on_load()

            @app.get("/env-panel/options/{field}")
            async def env_panel_options(field: str) -> list:
                return await panel.get_options(field)

            @app.post("/env-panel/field/{field}")
            async def env_panel_field(field: str, body: Optional[dict] = None) -> dict:
                body = body or {}
                return await panel.on_field_change(field, body.get("value"))

            @app.post("/env-panel/action/{action}")
            async def env_panel_action(action: str, body: Optional[dict] = None) -> dict:
                body = body or {}
                params = body.get("params") or {}
                return await panel.on_action(action, params)

        return app

    def serve(self, host: str = "0.0.0.0", **kwargs: Any) -> None:
        """Start the server with uvicorn. Blocks."""
        import uvicorn

        app = self._build_app()
        log.info("Starting %s server on %s:%d", self.name, host, self.port)
        uvicorn.run(app, host=host, port=self.port, **kwargs)
