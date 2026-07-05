"""Scan ``workspace/`` for class-based components and bridge to existing registries."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

from ..agent_loop.hooks import load_hooks_file
from ..graph_def import GraphDefinition, HookDef
from ..replay.interface import BaseReplayParser, GenericReplayParser
from .bases import BaseCanvasNode, BaseNodeSet

log = logging.getLogger("agentcanvas.components")

SOURCE_TAG = "workspace"


class _RemoteAutoServerShim:
    """Placeholder for an auto_host subprocess owned by another process.

    Slots into ``_auto_servers[name]`` so ``get_server_url`` and
    ``is_nodeset_loaded`` work for run subprocesses that attach to
    parent-owned shared singletons. ``stop()`` is intentionally a no-op
    because we do NOT own the lifetime.
    """

    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.connected = True

    def stop(self) -> None:  # pragma: no cover — trivial no-op
        pass

    def fetch_manifest(self):  # pragma: no cover — not used after register
        return None


class WorkspaceComponentRegistry:
    """Discovers class-based components under a scan directory.

    Directory layout::

        scan_dir/
        ├── nodesets/ *.py   (BaseNodeSet subclasses — atomic node groups)
        └── policies/ *.py   (BaseCanvasNode subclasses with policy metadata)

    Call :meth:`scan_all` at startup or on reload.  Discovered components
    """

    def __init__(
        self,
        scan_dir: str | Path,
        active_dir: str | Path | None = None,
    ) -> None:
        # frozen workspace = read-only faithful baseline
        # active workspace = optional overlay loaded after frozen; same-name
        # entries override frozen via register_node()'s last-write-wins on
        # NODE_HANDLERS and last-key-wins on _discovered_nodesets.
        self._frozen_dir = Path(scan_dir)
        self._active_dir = Path(active_dir) if active_dir else None
        # `_scan_dir` is an alias toggled during scan_all() — kept so
        # existing internal callers (_scan_subdir, _load_global_hooks,
        # _scan_and_register_servers) stay unchanged.
        self._scan_dir = self._frozen_dir
        self._scan_cycle = 0
        # Local bookkeeping
        self._policies: dict[str, Any] = {}  # policy_id → PolicyEntry
        # Discovered nodeset instances (persistent across load/unload)
        self._discovered_nodesets: dict[str, BaseNodeSet] = {}  # name → instance
        self._discovered_tool_names: dict[
            str, list[str]
        ] = {}  # name → node_types (from get_tools at discovery)
        # Live instances for lifecycle management
        self._live_nodesets: dict[str, BaseNodeSet] = {}  # name → instance (loaded)
        # Track which node names each component registered
        self._nodeset_node_names: dict[str, list[str]] = {}
        # Standalone node types registered from workspace/nodes/ + workspace/policies/
        # (always-on, registered at scan time). Tracked so unregister_all() can
        # evict them from NODE_HANDLERS — otherwise deleted standalone nodes
        # linger in the live registry until a full backend restart.
        self._standalone_node_types: set[str] = set()
        # Server mode
        self._servers: dict[str, dict] = {}  # server_name → status dict
        self._server_instances: dict[str, Any] = {}  # server_name → BaseServer instance
        self._server_node_types: dict[str, list[str]] = {}  # server_name → [node_type, ...]
        # Auto-hosted servers (nodesets deployed in server mode via AutoServerApp)
        self._auto_servers: dict[str, Any] = {}  # nodeset_name → BaseServer
        # Cross-nodeset container-access prototype: owned container id → owner
        # nodeset name, so the executor broker can forward a cross-process
        # read/write to the subprocess that homes the container (faces A / C).
        self._home_containers: dict[str, str] = {}
        self._auto_server_node_types: dict[str, list[str]] = {}  # nodeset_name → [node_type, ...]
        self._auto_server_containers: dict[
            str, list[dict]
        ] = {}  # nodeset_name → [ContainerDef dict]
        # Replay parsers — env nodesets that declare ``replay_parser`` register
        # a BaseReplayParser instance here keyed by nodeset name.
        self._replay_parsers: dict[str, BaseReplayParser] = {}
        # Global hooks (loaded from workspace/hooks.json)
        self._global_hooks: list[HookDef] = []

    # ── Public API ──

    def scan_all(self) -> dict[str, int]:
        """Scan all subdirectories.  Returns ``{"nodesets": N, ...}``.

        Two passes: frozen workspace first, then optional active overlay.
        Active scan registrations override frozen for same-named entries
        because register_node() and _discovered_nodesets[name] both
        last-write-wins. hooks.json uses explicit active-first override
        (see _load_global_hooks).
        """
        self._scan_cycle += 1
        self.unregister_all()

        # Pass 1 — frozen workspace
        self._scan_dir = self._frozen_dir
        n_nodesets = self._scan_and_register_nodesets()
        n_policies = self._scan_and_register_policies()
        n_nodes = self._scan_and_register_nodes()
        n_servers = self._scan_and_register_servers()

        # Pass 2 — active workspace overlay (if configured + exists)
        active_counts = {"nodesets": 0, "policies": 0, "nodes": 0, "servers": 0}
        if self._active_dir is not None and self._active_dir.is_dir():
            log.info(
                "Active workspace overlay: %s — running second scan pass",
                self._active_dir,
            )
            self._scan_dir = self._active_dir
            active_counts["nodesets"] = self._scan_and_register_nodesets()
            active_counts["policies"] = self._scan_and_register_policies()
            active_counts["nodes"] = self._scan_and_register_nodes()
            active_counts["servers"] = self._scan_and_register_servers()
            # Reset alias to frozen so non-scan code paths read frozen by default.
            self._scan_dir = self._frozen_dir

        # Hooks: full-file override (active wins if present, else frozen)
        self._load_global_hooks()

        counts = {
            "nodesets": n_nodesets + active_counts["nodesets"],
            "policies": n_policies + active_counts["policies"],
            "nodes": n_nodes + active_counts["nodes"],
            "servers": n_servers + active_counts["servers"],
        }
        if self._active_dir is not None and self._active_dir.is_dir():
            counts["active_overlay"] = active_counts  # type: ignore[assignment]
        log.info("Component scan #%d complete: %s", self._scan_cycle, counts)
        return counts

    def unregister_all(self) -> None:
        """Remove all previously-registered ``workspace`` components.

        Note: does NOT call shutdown — call :meth:`shutdown_all` first
        if you need to release resources before re-scanning.
        """
        self._policies.clear()
        self._discovered_nodesets.clear()
        self._discovered_tool_names.clear()
        self._live_nodesets.clear()
        self._nodeset_node_names.clear()
        # Stop managed servers before clearing
        for server in self._server_instances.values():
            if server.connected:
                try:
                    server.stop()
                except Exception:
                    log.exception("Failed to stop server %s", server.name)
        self._servers.clear()
        self._server_instances.clear()
        self._server_node_types.clear()
        # Stop auto-hosted servers
        for server in self._auto_servers.values():
            try:
                server.stop()
            except Exception:
                log.exception("Failed to stop auto-server %s", server.name)
        self._auto_servers.clear()
        self._auto_server_node_types.clear()
        self._auto_server_containers.clear()
        self._replay_parsers.clear()
        self._global_hooks.clear()
        # Evict standalone node/policy types from the global handler registry.
        # These are registered at scan time (not load time), so without this a
        # node deleted from disk lingers in NODE_HANDLERS until backend restart.
        if self._standalone_node_types:
            from ..agent_loop.builtin_nodes import NODE_HANDLERS
            from ..standard.node_io import invalidate_cache

            for nt in self._standalone_node_types:
                NODE_HANDLERS.pop(nt, None)
            self._standalone_node_types.clear()
            invalidate_cache()

    async def initialize_all(self) -> None:
        """Call ``initialize()`` on all live nodesets.

        Called after :meth:`scan_all` as a separate step so scanning stays
        fast (just imports) and initialization (potentially slow — GPU,
        network) is explicit.
        """
        for ns in self._live_nodesets.values():
            try:
                await ns.initialize()
                log.info("Initialized nodeset: %s", ns.name)
            except Exception:
                log.exception("Failed to initialize nodeset: %s", ns.name)

    async def shutdown_all(self) -> None:
        """Call ``shutdown()`` on all live nodesets and auto-servers."""
        for ns in self._live_nodesets.values():
            try:
                await ns.shutdown()
                log.info("Shut down nodeset: %s", ns.name)
            except Exception:
                log.exception("Failed to shut down nodeset: %s", ns.name)
        for server in self._auto_servers.values():
            try:
                server.stop()
                log.info("Stopped auto-server: %s", server.name)
            except Exception:
                log.exception("Failed to stop auto-server: %s", server.name)
        # Replay parsers may hold long-lived renderer subprocesses
        # (v1 smooth-mode). Stop them so backend shutdown reaps the tree.
        for name, parser in self._replay_parsers.items():
            try:
                await parser.shutdown()
                log.info("Shut down replay parser: %s", name)
            except Exception:
                log.exception("Failed to shut down replay parser: %s", name)

    # ── Targeted hot-reload (nodeset-source watcher) ──

    def _resolve_nodeset_reimport(self, changed: Path):
        """Map a changed ``.py`` file under ``nodesets/`` to a fresh-import thunk.

        Mirrors :meth:`_scan_subdir`'s layout rules (role buckets, single
        files, package nodesets). Returns ``(nested_subdir, thunk)`` or
        ``None`` when the file can't be attributed to a scannable nodeset
        entry — i.e. an underscore-prefixed helper module (imported *by* a
        nodeset but never scanned itself, e.g. ``method/_explore_eqa_tsdf.py``)
        or a path outside the scanned tree.
        """
        nsroot = (self._scan_dir / "nodesets").resolve()
        try:
            rel = changed.resolve().relative_to(nsroot)
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) < 2:
            return None
        role, entry = parts[0], parts[1]
        # An underscore on the role or entry segment means a private/helper
        # path the scanner skips — not attributable to one nodeset.
        if role.startswith("_") or entry.startswith("_"):
            return None
        nested = f"nodesets/{role}"
        role_dir = self._scan_dir / "nodesets" / role
        # Single .py file directly under the role bucket: <role>/X.py
        if len(parts) == 2 and entry.endswith(".py"):
            pf = role_dir / entry
            return (nested, lambda: self._import_module(nested, pf))
        # Directory entry under the role: a package nodeset <role>/X/__init__.py
        pkg_dir = role_dir / entry
        if (pkg_dir / "__init__.py").exists():
            return (nested, lambda: self._import_package(nested, pkg_dir))
        return None

    async def hot_reload_nodeset_sources(self, changed_files) -> dict:
        """Re-import + hot-reload only the nodesets whose source files changed.

        Conservative, targeted reload used by the nodeset-source watcher:

        * **Local** nodesets currently loaded in-process are reloaded in place
          (unload old instance → swap fresh → load).
        * **Server** (auto-hosted) nodesets — often GPU-backed and possibly
          serving a live eval over HTTP — are **not** torn down; they're only
          re-discovered (so the next explicit load is fresh) and flagged
          ``stale_servers`` for a manual ``/api/components/reload``.
        * Not-currently-loaded nodesets are re-discovered (``discovered``).
        * Files that don't map to a scan entry (underscore helper modules,
          deletions) are returned in ``unresolved`` — the caller may recommend
          a full reload; eval subprocesses pick them up automatically on their
          next launch since they re-scan from disk.

        Unaffected nodesets (including loaded GPU servers) are never touched.
        """
        reloaded: list[str] = []
        stale_servers: list[str] = []
        discovered: list[str] = []
        unresolved: list[str] = []

        thunks: list = []
        seen_nested: set[str] = set()
        for f in changed_files:
            resolved = self._resolve_nodeset_reimport(Path(f))
            if resolved is None:
                unresolved.append(str(f))
                continue
            nested, thunk = resolved
            # Dedupe package re-imports when several files in one package change.
            key = f"{nested}::{Path(f).parent}"
            if key in seen_nested:
                continue
            seen_nested.add(key)
            thunks.append((str(f), thunk))

        if not thunks:
            return {
                "reloaded": reloaded,
                "stale_servers": stale_servers,
                "discovered": discovered,
                "unresolved": unresolved,
            }

        # Fresh import cycle so synthetic module names differ from the last
        # scan → the source is genuinely re-executed from disk.
        self._scan_cycle += 1
        fresh: dict[str, Any] = {}
        for f, thunk in thunks:
            try:
                mod = thunk()
            except Exception:
                log.exception("hot-reload: re-import failed for %s", f)
                unresolved.append(f)
                continue
            if mod is None:
                continue
            for cls in self._find_subclasses(mod, BaseNodeSet):
                inst = cls()
                inst._source_file = getattr(mod, "__file__", None)
                fresh[inst.name] = inst

        for name, inst in fresh.items():
            is_server = bool(
                name in self._auto_server_node_types or self._tagged_server_keys_for(name)
            )
            is_local = name in self._nodeset_node_names and not is_server
            # Refresh discovery bookkeeping regardless of load state.
            self._discovered_nodesets[name] = inst
            try:
                self._discovered_tool_names[name] = [
                    getattr(t, "node_type", getattr(t, "name", "")) for t in inst.get_tools()
                ]
            except Exception:
                self._discovered_tool_names[name] = []
            self._maybe_register_replay_parser(inst)

            if is_server:
                stale_servers.append(name)
                continue
            if is_local:
                try:
                    await self.unload_nodeset(name)
                    await self.load_nodeset(name, mode="local")
                    reloaded.append(name)
                except Exception:
                    log.exception("hot-reload: in-place reload failed for %s", name)
                continue
            discovered.append(name)

        return {
            "reloaded": reloaded,
            "stale_servers": stale_servers,
            "discovered": discovered,
            "unresolved": unresolved,
        }

    # ── Per-component load / unload ──

    async def load_nodeset(self, name: str, mode: str = "local", worker_count: int = 1) -> dict:
        """Load (or reload) a discovered nodeset by name.

        Args:
            name: NodeSet identifier.
            mode: ``"local"`` (in-process) or ``"server"`` (auto-hosted
                  in a subprocess via AutoServerApp).
            worker_count: ADR-028 PB-1. When > 1, spawn ``worker_count``
                  tagged server-mode subprocesses for parallel eval.
                  Forces ``mode="server"`` regardless of the request,
                  since fan-out only makes sense for isolated subprocesses.
                  Single-worker / canvas-Play uses the default of 1.
        """
        ns = self._live_nodesets.get(name) or self._discovered_nodesets.get(name)
        if ns is None:
            # Not discovered — try to find it from a fresh scan of just nodesets/
            instances = self._scan_subdir("nodesets", BaseNodeSet)
            for inst in instances:
                if inst.name == name:
                    ns = inst
                    self._discovered_nodesets[name] = inst
                    break
        if ns is None:
            raise ValueError(f"Unknown nodeset: {name}")

        # If already loaded, unload first (covers both untagged singletons
        # and PB-1 tagged multi-worker copies).
        if (
            name in self._nodeset_node_names
            or name in self._auto_server_node_types
            or self._tagged_server_keys_for(name)
        ):
            await self.unload_nodeset(name)

        # Auto-route to server mode if nodeset needs a different Python (ADR-020).
        # Note: server-mode proxy nodes get srv_ prefix (TODO #33 to unify).
        sp = getattr(type(ns), "server_python", None)
        if mode == "local" and sp and sp != sys.executable:
            log.info("Auto-routing %s to server mode (server_python=%s)", name, sp)
            mode = "server"
        if worker_count > 1 and mode != "server":
            log.info("Auto-routing %s to server mode (worker_count=%d)", name, worker_count)
            mode = "server"

        if mode == "server":
            return await self._load_nodeset_as_server(name, ns, worker_count=worker_count)

        # Local mode (default): initialize first, then register tools.
        # worker_count > 1 was force-routed to server mode above, so local
        # mode is always single-instance.
        tools = ns.get_tools()
        self._live_nodesets[name] = ns
        await ns.initialize()
        self._register_tool_instances(tools)
        self._register_env_panel_for(ns)
        # Track names: BaseCanvasNode uses .node_type
        names = [
            getattr(t, "name", None) or getattr(t, "node_type", type(t).__name__) for t in tools
        ]
        self._nodeset_node_names[name] = names
        log.info("Loaded nodeset: %s (%d nodes, local)", name, len(tools))
        return {"name": name, "tools": names, "mode": "local"}

    async def unload_nodeset(self, name: str) -> dict:
        """Unload a nodeset — deregister its nodes and shut it down.

        Handles local nodesets, single-instance auto-hosted server nodesets,
        and ADR-028 PB-1 multi-worker tagged copies (walks every
        ``f"{name}#k"`` key plus the bare ``name``).
        """
        # Auto-hosted server mode (covers both untagged single-instance
        # and PB-1 multi-worker tagged copies)
        if self._tagged_server_keys_for(name):
            return self._unload_auto_server_nodeset(name)

        # Local mode
        ns = self._live_nodesets.get(name)
        if ns is None:
            raise ValueError(f"NodeSet not loaded: {name}")

        self._unregister_env_panel_for(ns)
        await ns.shutdown()
        node_names = self._nodeset_node_names.pop(name, [])
        # Deregister from canvas node registry (BaseCanvasNode entries)
        from ..agent_loop.builtin_nodes import NODE_HANDLERS
        from ..standard.node_io import invalidate_cache

        for nt in node_names:
            NODE_HANDLERS.pop(nt, None)
        # Also remove by node_type for canvas nodes
        for tool in ns.get_tools():
            nt = getattr(tool, "node_type", None)
            if nt:
                NODE_HANDLERS.pop(nt, None)
        invalidate_cache()
        del self._live_nodesets[name]
        log.info("Unloaded nodeset: %s (%d nodes removed)", name, len(node_names))
        return {"name": name, "tools_removed": node_names}

    # ── Auto-hosted server mode (ADR-009) ──

    async def _load_nodeset_as_server(
        self, name: str, ns: BaseNodeSet, worker_count: int = 1
    ) -> dict:
        """Launch a nodeset in one or more subprocesses via AutoServerApp.

        Each subprocess serves the nodeset in server mode.  Proxy
        canvas nodes are generated from the manifest and registered in
        NODE_HANDLERS — identical to YAML-based servers.

        At ``worker_count > 1`` (ADR-028 PB-1), spawn N independent
        subprocesses on N free ports and register N tagged
        ``RemoteEnvPanelProxy`` instances under ``f"{name}#{k}"`` for
        ``k in [0, N)``. Proxy node classes are generated once (one
        ``node_type`` → one class) with worker 0's URL baked into the
        closure; per-worker URL routing for in-graph proxy calls is
        added in PB-1.5 via ``ctx._executor.get_server_url(...)``.
        """
        from ..agent_loop.builtin_nodes import NODE_HANDLERS, register_node
        from ..server.base_server import BaseServer
        from ..server.proxy import generate_proxy_nodes
        from ..standard.node_io import invalidate_cache
        from .env_panel import unregister_env_panel

        nodeset_cls = type(ns)
        # Use _source_file stored during discovery (ADR-020). Dynamically-loaded
        # modules aren't in sys.modules, so inspect.getfile() fails on their classes.
        source_file = getattr(ns, "_source_file", None)
        if source_file is None:
            mod = sys.modules.get(nodeset_cls.__module__)
            source_file = getattr(mod, "__file__", None) or inspect.getfile(nodeset_cls)
        class_name = nodeset_cls.__name__
        python = getattr(nodeset_cls, "server_python", None) or sys.executable

        # Build PYTHONPATH: backend dir + workspace root
        workspace_root = str(self._scan_dir.parent.resolve())
        backend_dir = str(Path(workspace_root) / "agentcanvas" / "backend")
        pythonpath = f"{backend_dir}:{workspace_root}"

        # Allocate all ports up front to minimise the bind-race window
        # (socket.bind(('', 0)) between concurrent eval starts could collide).
        ports = [self._find_free_port() for _ in range(worker_count)]
        has_env_panel = getattr(nodeset_cls, "env_panel", None) is not None
        node_types: list[str] = []
        # Bookkeeping for rollback if a later worker's startup or env panel
        # registration fails. Each tuple: (tag-or-None, store_key, server).
        started: list[tuple[int | None, str, BaseServer]] = []

        # Package-mode nodesets (folder with __init__.py) need to be re-imported
        # via dotted module name in the subprocess so relative/absolute imports
        # inside the package resolve. Single-file nodesets keep using --file.
        #
        # The import root is the parent of the `nodesets/` dir containing the
        # source — works uniformly for frozen `workspace/nodesets/...` and for
        # any overlay (active_workspace/nodesets/...). Computing `rel` against
        # the repo root instead breaks on overlays whose path contains
        # hyphenated method dirs (e.g. `adas-subagent`) or hidden dirs (e.g.
        # `.staging`), producing invalid dotted module names.
        source_arg = ["--file", str(source_file)]
        if Path(source_file).name == "__init__.py":
            src_path = Path(source_file)
            import_root: Path | None = None
            for parent in src_path.parents:
                if parent.name == "nodesets":
                    import_root = parent.parent
                    break
            if import_root is not None:
                try:
                    rel = src_path.parent.relative_to(import_root)
                    dotted = rel.as_posix().replace("/", ".")
                    # Validate every component is a legal Python identifier;
                    # else fall back to --file.
                    if all(comp.isidentifier() for comp in dotted.split(".")):
                        source_arg = ["--module", dotted]
                        # Prepend import_root to PYTHONPATH so the dotted name
                        # resolves to THIS source (not a frozen-workspace twin
                        # of the same dotted name).
                        pythonpath = f"{import_root}:{pythonpath}"
                except ValueError:
                    # source_file is not under any `nodesets/` — keep --file.
                    pass

        try:
            for k in range(worker_count):
                tag = k if worker_count > 1 else None
                port = ports[k]
                # argv list (no shell): keeps auto_host a DIRECT child of
                # this process, so its PR_SET_PDEATHSIG watches us one-hop.
                # A /bin/sh wrapper here would break that chain — see
                # BaseServer.start().
                command = [
                    python,
                    "-m",
                    "app.server.auto_host",
                    *source_arg,
                    "--class",
                    class_name,
                    "--port",
                    str(port),
                ]
                server_label = f"auto_{name}#{k}" if tag is not None else f"auto_{name}"
                # Pull optional server-side env vars from the nodeset class
                # (e.g. LD_PRELOAD for the hmeqa NVIDIA-driver-570 shim).
                # PYTHONPATH used to be a shell-command prefix; it now rides
                # the env dict so the spawn needs no shell.
                ns_server_env = getattr(nodeset_cls, "server_env", None) or {}
                # Cross-nodeset container-access (face B) + reverse log/error
                # push: the executor callback base URL a subprocess uses to reach
                # /api/internal. An explicit env override wins; otherwise resolve
                # from Settings (host/port) rather than the old hardcoded
                # localhost:8000 dev default.
                from ..config import resolve_executor_url

                executor_url = os.environ.get("AGENTCANVAS_EXECUTOR_URL") or resolve_executor_url()
                server = BaseServer(
                    name=server_label,
                    command=command,
                    port=port,
                    startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
                    working_dir=backend_dir,
                    env={
                        "PYTHONPATH": pythonpath,
                        "AGENTCANVAS_EXECUTOR_URL": executor_url,
                        **ns_server_env,
                    },
                )
                server.start()

                manifest = server.fetch_manifest()
                if manifest is None:
                    server.stop()
                    raise RuntimeError(
                        f"Auto-server for nodeset {name} (tag={tag}) "
                        f"started but failed to return manifest"
                    )

                if k == 0:
                    # All N subprocesses serve the same manifest; generate
                    # proxy classes once. Each proxy bakes worker 0's URL
                    # into its closure — PB-1.5 swaps per-worker URLs at
                    # call time via ctx._executor.get_server_url(...).
                    proxy_classes = generate_proxy_nodes(server.url, manifest)
                    for cls in proxy_classes:
                        register_node(cls)
                        node_types.append(cls.node_type)
                        log.info("    → auto-server node: %s", cls.node_type)
                    if proxy_classes:
                        invalidate_cache()

                store_key = f"{name}#{k}" if tag is not None else name
                self._auto_servers[store_key] = server
                # Record which owned containers this nodeset homes (faces A/C):
                # cid → base nodeset name, resolved to a URL via get_server_url.
                for _cdef in manifest.containers or []:
                    _cid = _cdef.get("id") if isinstance(_cdef, dict) else None
                    if _cid:
                        self._home_containers[_cid] = name
                # Container-ownership guardrails — checked once (manifest is
                # identical across workers).
                if k == 0 and manifest.containers:
                    _cids = [c.get("id") for c in manifest.containers if isinstance(c, dict)]
                    self._check_container_ownership(name, _cids, worker_count)
                started.append((tag, store_key, server))
                # Server-mode env panel bridge: fetch the env panel schema
                # from the spawned subprocess and register a
                # RemoteEnvPanelProxy locally. The proxy forwards every
                # BaseEnvPanel call over HTTP, so the canvas panel works
                # identically in local and server modes. Tagged registration
                # raises on failure so the outer rollback can stop the
                # already-started servers.
                if has_env_panel:
                    self._register_remote_env_panel(name, server.url, tag=tag)
        except Exception:
            log.exception(
                "Failed to bring up worker_count=%d server(s) for %s; rolling back",
                worker_count,
                name,
            )
            for tag, store_key, server in started:
                try:
                    server.stop()
                except Exception:
                    log.exception("Rollback: failed to stop %s", store_key)
                self._auto_servers.pop(store_key, None)
                if tag is not None:
                    try:
                        unregister_env_panel(f"{name}#{tag}")
                    except Exception:
                        log.exception(
                            "Rollback: failed to unregister env panel %s#%d",
                            name,
                            tag,
                        )
            for nt in node_types:
                NODE_HANDLERS.pop(nt, None)
            raise

        self._auto_server_node_types[name] = node_types
        # Nodeset-owned (nodeset-level) container schemas from the manifest, for
        # at-rest display in the canvas State panel.
        self._auto_server_containers[name] = list(getattr(manifest, "containers", []))
        self._live_nodesets[name] = ns

        log.info(
            "Loaded nodeset: %s (%d nodes, server mode, worker_count=%d, ports=%s)",
            name,
            len(node_types),
            worker_count,
            ports,
        )
        return {
            "name": name,
            "tools": node_types,
            "mode": "server",
            "worker_count": worker_count,
            "ports": ports,
        }

    def _unload_auto_server_nodeset(self, name: str) -> dict:
        """Stop all auto-hosted subprocesses for a nodeset (single-instance
        or PB-1 multi-worker tagged copies) and deregister proxy nodes /
        env panels."""
        from ..agent_loop.builtin_nodes import NODE_HANDLERS
        from ..standard.node_io import invalidate_cache
        from .env_panel import unregister_env_panel

        # Stop the untagged single-worker env panel (if loaded that way)
        # via the existing class-name-based path. Tagged env panels are
        # unregistered explicitly below.
        ns = self._live_nodesets.get(name)
        if ns is not None:
            self._unregister_env_panel_for(ns)

        keys = self._tagged_server_keys_for(name)
        for key in keys:
            server = self._auto_servers.pop(key, None)
            if server is not None:
                try:
                    server.stop()
                except Exception:
                    log.exception("Failed to stop auto-server for %s", key)
            if "#" in key:
                # PB-1 tagged env panel — proxy was registered under the
                # tagged name in _register_remote_env_panel.
                try:
                    unregister_env_panel(key)
                except Exception:
                    log.exception("Failed to unregister tagged env panel %s", key)

        node_types = self._auto_server_node_types.pop(name, [])
        self._auto_server_containers.pop(name, None)
        for nt in node_types:
            NODE_HANDLERS.pop(nt, None)
        if node_types:
            invalidate_cache()

        self._live_nodesets.pop(name, None)

        log.info(
            "Unloaded nodeset: %s (%d server nodes removed, %d subprocess(es))",
            name,
            len(node_types),
            len(keys),
        )
        return {"name": name, "tools_removed": node_types}

    # ── TODO #60: overlay-aware ephemeral spawn ──

    async def load_nodeset_ephemeral(
        self,
        name: str,
        source_path: str | Path,
        tag: str,
    ) -> str:
        """Spawn a tagged auto_host child loaded from an explicit source.

        TODO #60 entry point for content-hashed overlay routing: when an
        eval's ``active_workspace_dir`` redefines a ``shared`` nodeset's
        source, this routes the eval to a fresh subprocess loaded from
        the overlay path while leaving the frozen singleton (registered
        under the untagged ``name`` key) untouched.

        The child is stored under ``f"{name}#{tag}"`` in
        ``_auto_servers``. **No proxy node classes are registered** — the
        eval subprocess wires its own proxy via ``register_remote_nodeset``
        in ``eval_subprocess_main`` using the URL returned here.

        Caller owns teardown via :meth:`unload_nodeset_ephemeral`.

        Args:
            name: nodeset identifier (matches the frozen ``_discovered_nodesets`` key).
            source_path: absolute path to the overlay's ``.py`` or ``__init__.py``.
            tag: ephemeral tag, e.g. ``"ephem-20260515_141201"`` — must
                 not collide with worker-tag ints used by PB-1.

        Returns:
            The new child's HTTP URL.
        """
        from ..server.base_server import BaseServer

        ns = self._discovered_nodesets.get(name)
        if ns is None:
            # Discovered map may be cold if no graph has referenced this nodeset
            # yet — scan and try again. Mirrors the bootstrap in load_nodeset().
            instances = self._scan_subdir("nodesets", BaseNodeSet)
            for inst in instances:
                if inst.name == name:
                    ns = inst
                    self._discovered_nodesets[name] = inst
                    break
        if ns is None:
            raise ValueError(f"Unknown nodeset: {name}")

        nodeset_cls = type(ns)
        class_name = nodeset_cls.__name__
        python = getattr(nodeset_cls, "server_python", None) or sys.executable

        workspace_root = str(self._scan_dir.parent.resolve())
        backend_dir = str(Path(workspace_root) / "agentcanvas" / "backend")
        pythonpath = f"{backend_dir}:{workspace_root}"

        src = Path(source_path)
        source_arg = ["--file", str(src)]
        if src.name == "__init__.py":
            # Package mode: pass --module so relative/absolute intra-package
            # imports inside the overlay resolve. The overlay's package root
            # must be on PYTHONPATH; the overlay's `nodesets/` parent gets
            # added below so dotted-module resolution lands in the overlay,
            # not the frozen copy.
            overlay_workspace_root: Path | None = None
            for parent in src.parents:
                if parent.name == "nodesets":
                    overlay_workspace_root = parent.parent
                    break
            if overlay_workspace_root is not None:
                try:
                    rel = src.parent.relative_to(overlay_workspace_root)
                    dotted = rel.as_posix().replace("/", ".")
                    source_arg = ["--module", dotted]
                    # Prepend overlay root so the overlay's package wins over
                    # any frozen-workspace package of the same dotted name.
                    pythonpath = f"{overlay_workspace_root}:{pythonpath}"
                except ValueError:
                    pass  # Keep --file fallback

        store_key = f"{name}#{tag}"
        if store_key in self._auto_servers:
            # Idempotent: same tag already spawned — return existing URL.
            return self._auto_servers[store_key].url

        port = self._find_free_port()
        command = [
            python,
            "-m",
            "app.server.auto_host",
            *source_arg,
            "--class",
            class_name,
            "--port",
            str(port),
        ]
        server_label = f"auto_{name}#{tag}"
        ns_server_env = getattr(nodeset_cls, "server_env", None) or {}
        server = BaseServer(
            name=server_label,
            command=command,
            port=port,
            startup_timeout=getattr(nodeset_cls, "startup_timeout", 1800),
            working_dir=backend_dir,
            env={"PYTHONPATH": pythonpath, **ns_server_env},
        )
        server.start()
        manifest = server.fetch_manifest()
        if manifest is None:
            server.stop()
            raise RuntimeError(
                f"Ephemeral auto-server for nodeset {name} (tag={tag}) "
                f"started but failed to return manifest"
            )

        self._auto_servers[store_key] = server
        log.info(
            "Loaded ephemeral nodeset: %s tag=%s port=%d (source=%s)",
            name,
            tag,
            port,
            src,
        )
        return server.url

    def unload_nodeset_ephemeral(self, tag: str) -> int:
        """Stop and pop every ``_auto_servers`` entry with this ephemeral tag.

        Returns the number of subprocesses stopped. Idempotent — calling
        with an unknown tag is a no-op. Does NOT touch ``_live_nodesets``
        or ``_auto_server_node_types`` because ephemerals never registered
        proxy nodes or marked themselves "live" in the parent registry.
        """
        suffix = f"#{tag}"
        keys = [k for k in self._auto_servers if k.endswith(suffix)]
        for key in keys:
            server = self._auto_servers.pop(key, None)
            if server is None:
                continue
            try:
                server.stop()
            except Exception:
                log.exception("Failed to stop ephemeral auto-server %s", key)
        if keys:
            log.info("Unloaded %d ephemeral subprocess(es) for tag=%s", len(keys), tag)
        return len(keys)

    # ── Internals ──

    @staticmethod
    def _find_free_port() -> int:
        """Find an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _tagged_server_keys_for(self, name: str) -> list[str]:
        """Return all ``_auto_servers`` keys for a nodeset (tagged or untagged).

        At single-worker / canvas-Play: returns ``[name]`` if loaded.
        At ADR-028 PB-1 multi-worker: returns ``[f"{name}#0", ...,
        f"{name}#N-1"]``. The two key shapes are mutually exclusive for a
        given nodeset (a load always unloads any prior copies first).
        """
        prefix = f"{name}#"
        return [k for k in self._auto_servers if k == name or k.startswith(prefix)]

    def get_server_url(self, name: str, tag: int | None = None) -> str | None:
        """Return the server URL for an auto-hosted nodeset (optionally tagged).

        Used by ADR-028 PB-2 to populate ``WorkerHandle.server_url_overrides``
        so each worker's ``LoopRunner`` can route in-graph proxy node calls
        to its own env subprocess (PB-1.5).
        """
        key = f"{name}#{tag}" if tag is not None else name
        server = self._auto_servers.get(key)
        return server.url if server is not None else None

    def get_container_home_url(self, container_id: str) -> str | None:
        """URL of the server subprocess that homes ``container_id`` (cross-nodeset
        container-access prototype, faces A/C), or None if it is not a known
        sub-homed container (then it's executor-home or unknown)."""
        owner = self._home_containers.get(container_id)
        if not owner:
            return None
        return self.get_server_url(owner)

    # ── Eval introspection ──

    def detect_env_nodesets_for_graph(self, graph: GraphDefinition) -> list[str]:
        """Find env nodesets used by a graph via registry-backed lookup.

        Checks all three registries (local-mode, server-mode, discovered/unloaded)
        to reverse-map graph node types to their owning nodeset.

        Falls back to node-type prefix extraction (e.g. "env_habitat__step"
        → "env_habitat") when the nodeset wasn't discovered (import failed
        on a different Python env).

        Works for both local and server mode node types.
        """
        graph_node_types = {node.type for node in graph.nodes}
        candidate_nodesets: set[str] = set()

        # Primary: registry-backed lookup
        for registry_dict in (
            self._nodeset_node_names,  # loaded local-mode
            self._auto_server_node_types,  # loaded server-mode
            self._discovered_tool_names,  # unloaded (from scan)
        ):
            for ns_name, node_types in registry_dict.items():
                if graph_node_types & set(node_types):
                    candidate_nodesets.add(ns_name)

        # Fallback: extract nodeset names from node type prefixes.
        # Convention: "nodeset_name__node_name" (double underscore separator).
        # This handles nodesets that failed to import during scan (e.g.
        # env_habitat on agentcanvas env without habitat-sim).
        if not candidate_nodesets:
            for nt in graph_node_types:
                if "__" in nt:
                    prefix = nt.split("__")[0]
                    candidate_nodesets.add(prefix)

        return list(candidate_nodesets)

    async def get_eval_metadata_for_nodeset(self, name: str) -> dict:
        """Get eval metadata for a nodeset. Returns empty dict if not an env.

        Falls back to static metadata for known env nodesets that couldn't
        be imported during scan (e.g. env_habitat on agentcanvas env).
        """
        ns = self._live_nodesets.get(name) or self._discovered_nodesets.get(name)
        if ns is not None:
            return await ns.get_eval_metadata()

        # Static fallback for known env nodesets that failed to import
        _KNOWN_ENV_METADATA: dict[str, dict] = {
            "env_habitat": {
                "env_name": "habitat_vlnce",
                "splits": ["val_unseen", "val_seen", "train"],
                "episode_counts": {},
                "metrics": ["spl", "success", "ndtw", "sdtw", "path_length", "distance_to_goal"],
                "supports_set_episode": False,
                "step_budget": 500,
            },
        }
        return _KNOWN_ENV_METADATA.get(name, {})

    def is_nodeset_loaded(self, name: str) -> bool:
        """Check if a nodeset is currently loaded (local, single-instance
        server, or any PB-1 tagged multi-worker copy)."""
        return name in self._live_nodesets or bool(self._tagged_server_keys_for(name))

    def _get_parallelism(self, ns_name: str) -> str:
        """Return ``"replicated"`` or ``"shared"`` for a nodeset by name.

        Reads ``BaseNodeSet.parallelism`` ClassVar on the discovered class.
        Falls back to a static set for env nodesets that fail to import on
        the current Python env (e.g. ``env_habitat`` on the ``agentcanvas``
        env without habitat-sim) — those still need ``"replicated"`` so
        eval fan-out works.
        """
        ns = self._live_nodesets.get(ns_name) or self._discovered_nodesets.get(ns_name)
        if ns is not None:
            return getattr(type(ns), "parallelism", "shared")
        _KNOWN_REPLICATED = {"env_habitat", "env_mp3d", "env_hmeqa"}
        return "replicated" if ns_name in _KNOWN_REPLICATED else "shared"

    def _check_container_ownership(
        self, name: str, container_ids: list, worker_count: int
    ) -> str | None:
        """Load-time guardrails for a server nodeset that owns state containers.

        Returns a short status code (also used by tests) and logs a warning:

        - ``"shared-stateful"`` (#68) — a ``shared`` server hosts ONE subprocess
          for all workers, so any mutable owned container is shared and WILL
          race under ``worker_count>1``. The nodeset should be ``replicated``
          (per-worker private copy) or stateless. Warn now; escalate to a hard
          reject once no legitimate shared+stateful nodeset remains.
        - ``"replicated-fanout-unroutable"`` (#17 residual) — cross-nodeset
          access to a ``replicated`` nodeset's containers is not routable under
          fan-out: the home registry maps cid→base name and
          ``get_container_home_url`` resolves an untagged URL that does not exist
          (subprocesses are stored as ``{name}#{k}``). Owner-local
          (same-subprocess) access is unaffected.
        - ``None`` — no concern.
        """
        if not container_ids:
            return None
        para = self._get_parallelism(name)
        if para == "shared":
            log.warning(
                "[#68] nodeset %r is parallelism='shared' but owns %d state "
                "container(s) %s — a shared server shares ONE subprocess across "
                "workers, so this state races under worker_count>1. Mark the "
                "nodeset parallelism='replicated' (per-worker private copy) or "
                "make it stateless.",
                name,
                len(container_ids),
                container_ids,
            )
            return "shared-stateful"
        if para == "replicated" and worker_count > 1:
            log.warning(
                "[#17] replicated nodeset %r homes %d container(s) %s under "
                "worker_count=%d — cross-nodeset access to them from another "
                "nodeset is not routable yet (broker resolves an untagged home "
                "URL). Owner-local access is fine.",
                name,
                len(container_ids),
                container_ids,
                worker_count,
            )
            return "replicated-fanout-unroutable"
        return None

    async def ensure_shared_nodesets_for_graph(self, graph: GraphDefinition) -> dict:
        """Load only the ``parallelism="shared"`` nodesets a graph needs.

        Counterpart to :meth:`ensure_nodesets_for_graph` for the subprocess
        eval path: parent backend only owns shared singletons (long-lived,
        cross-job reusable — e.g. Prismatic VLM); the eval subprocess
        owns its own ``replicated`` / env-class nodesets via its own
        ``ensure_nodesets_for_graph`` call.

        This avoids the old "load everything then unload non-shared" dance
        in /api/eval/v2/start that wasted minutes spawning env_habitat in
        the parent only to kill it before subprocess spawn.

        Workers are always singleton here — fan-out (``worker_count > 1``)
        is a property of replicated nodesets and is decided by the eval
        subprocess, not the parent.

        Returns the same shape as :meth:`ensure_nodesets_for_graph`:
        ``{"loaded": [...], "already_loaded": [...], "failed": [...], "unknown": [...]}``.
        """
        needed: set[str] = set()
        for node in graph.nodes:
            if "__" in node.type:
                needed.add(node.type.split("__")[0])

        result: dict[str, list[str]] = {
            "loaded": [],
            "already_loaded": [],
            "failed": [],
            "unknown": [],
        }
        for ns_name in sorted(needed):
            if self._get_parallelism(ns_name) != "shared":
                continue  # subprocess eval owns these
            if self.is_nodeset_loaded(ns_name):
                result["already_loaded"].append(ns_name)
            elif ns_name in self._discovered_nodesets:
                try:
                    await self.load_nodeset(ns_name)
                    result["loaded"].append(ns_name)
                except Exception as e:
                    log.warning("Failed to auto-load shared nodeset %s: %s", ns_name, e)
                    result["failed"].append(ns_name)
            else:
                result["unknown"].append(ns_name)
        return result

    async def ensure_nodesets_for_graph(
        self, graph: GraphDefinition, worker_count: int = 1
    ) -> dict:
        """Auto-load all nodesets required by a graph's node types.

        Convention: node types with '__' (e.g. "env_habitat__step") belong
        to a nodeset named by the prefix (e.g. "env_habitat").

        Shared by both canvas execution and eval batch execution.

        Args:
            graph: Graph to inspect.
            worker_count: ADR-028 PB-1. When > 1, ``parallelism="replicated"``
                nodesets are spawned as ``worker_count`` tagged subprocesses
                for parallel eval; any existing single-instance load is
                unloaded first. ``parallelism="shared"`` nodesets stay
                singleton — K callers rendezvous through
                ``BatchedInferenceServer`` at runtime instead.

        Returns: {"loaded": [...], "already_loaded": [...], "failed": [...], "unknown": [...]}
        """
        needed: set[str] = set()
        for node in graph.nodes:
            if "__" in node.type:
                needed.add(node.type.split("__")[0])

        result: dict[str, list[str]] = {
            "loaded": [],
            "already_loaded": [],
            "failed": [],
            "unknown": [],
        }
        for ns_name in sorted(needed):
            # Dispatch on the nodeset's declared parallelism contract
            # (ADR-server-003): only "replicated" nodesets fan out into
            # N tagged subprocesses.
            parallelism = self._get_parallelism(ns_name)

            if worker_count > 1 and parallelism == "replicated":
                # Multi-worker env: always (re)spawn N tagged copies. If a
                # prior load exists (untagged from canvas Play, or a different
                # worker_count from a prior eval run), unload it first so the
                # new fan-out is clean.
                if self.is_nodeset_loaded(ns_name):
                    try:
                        await self.unload_nodeset(ns_name)
                    except Exception as e:
                        log.warning("Failed to unload %s before re-spawn: %s", ns_name, e)
                        result["failed"].append(ns_name)
                        continue
                if ns_name in self._discovered_nodesets:
                    try:
                        await self.load_nodeset(ns_name, worker_count=worker_count)
                        result["loaded"].append(ns_name)
                    except Exception as e:
                        log.warning(
                            "Failed to spawn %s with worker_count=%d: %s",
                            ns_name,
                            worker_count,
                            e,
                        )
                        result["failed"].append(ns_name)
                else:
                    result["unknown"].append(ns_name)
                continue

            # Singleton path (existing behaviour, untouched at worker_count=1).
            if self.is_nodeset_loaded(ns_name):
                result["already_loaded"].append(ns_name)
            elif ns_name in self._discovered_nodesets:
                try:
                    await self.load_nodeset(ns_name)
                    result["loaded"].append(ns_name)
                except Exception as e:
                    log.warning("Failed to auto-load nodeset %s: %s", ns_name, e)
                    result["failed"].append(ns_name)
            else:
                result["unknown"].append(ns_name)
        return result

    def list_nodesets(self) -> list[dict]:
        """List all discovered nodesets with loaded status and available nodes."""
        result = []
        for name, ns in self._discovered_nodesets.items():
            tagged = bool(self._tagged_server_keys_for(name))
            loaded = name in self._live_nodesets or tagged
            mode = "server" if tagged else "local"
            # For loaded nodesets: show registered node names
            # For unloaded: show discovered node names (probed at scan time)
            if loaded:
                tools = (
                    self._auto_server_node_types.get(name, [])
                    if mode == "server"
                    else self._nodeset_node_names.get(name, [])
                )
            else:
                tools = self._discovered_tool_names.get(name, [])
            # Check if this nodeset requires server mode (different Python env)
            sp = getattr(type(ns), "server_python", None)
            requires_server = bool(sp and sp != sys.executable)
            result.append(
                {
                    "name": name,
                    "description": ns.description,
                    "loaded": loaded,
                    "mode": mode,
                    "requires_server": requires_server,
                    "tools": tools,
                    "containers": self._auto_server_containers.get(name, []),
                }
            )
        return result

    def nodeset_mode(self, name: str) -> str:
        """``"server"`` if the nodeset currently runs auto-hosted, else ``"local"``."""
        return "server" if self._tagged_server_keys_for(name) else "local"

    def nodeset_source_info(self, name: str) -> dict | None:
        """Source-file info for a discovered nodeset (loaded or not).

        Backs the canvas source editor (``/nodesets/{name}/source``).
        Returns ``None`` when the nodeset is unknown or recorded no
        ``_source_file`` at discovery.
        """
        ns = self._discovered_nodesets.get(name)
        if ns is None:
            return None
        source_file = getattr(ns, "_source_file", None)
        if not source_file:
            return None
        sp = getattr(type(ns), "server_python", None)
        return {
            "name": name,
            "source_file": Path(source_file),
            "mode": self.nodeset_mode(name),
            "requires_server": bool(sp and sp != sys.executable),
            "loaded": self.is_nodeset_loaded(name),
        }

    def get_policies(self) -> dict[str, Any]:
        """Return all registered policies as ``{policy_id: PolicyEntry}``."""
        return dict(self._policies)

    def list_policies(self) -> list[dict[str, str]]:
        return [
            {"id": p.id, "name": p.name, "checkpoint": p.checkpoint, "config": p.config}
            for p in self._policies.values()
        ]

    def get_global_hooks(self) -> list[HookDef]:
        """Return global hooks loaded from ``workspace/hooks.json``."""
        return list(self._global_hooks)

    # ── Non-Python resource lookup (graph JSON / exp.yaml) ──
    # Single source of truth for callers that need to resolve a graph or
    # profile file across frozen + active workspaces. Active wins if
    # present; otherwise fall through to frozen.

    def resolve_graph_path(self, name: str) -> Path | None:
        """Locate ``graphs/{name}.json`` — active overlay first, then frozen.

        ``name`` may be a relative-path id (``vln/verified/mapgpt_mp3d``) or a
        bare stem (``mapgpt_mp3d``). The flat ``graphs/{name}.json`` lookup
        already handles path-ids; for bare stems that no longer sit at the
        graphs root (post hierarchy reorg), fall back to a recursive search
        for ``graphs/**/{name}.json``. Returns ``None`` if nothing matches.
        """
        for base in (self._active_dir, self._frozen_dir):
            if base is None:
                continue
            flat = base / "graphs" / f"{name}.json"
            if flat.exists():
                return flat
            # Bare-stem fallback: a single recursive match under graphs/.
            if "/" not in name and "\\" not in name:
                graphs_root = base / "graphs"
                if graphs_root.is_dir():
                    matches = sorted(graphs_root.rglob(f"{name}.json"))
                    if matches:
                        if len(matches) > 1:
                            log.warning(
                                "resolve_graph_path: %r is ambiguous (%d matches under %s); "
                                "using %s",
                                name,
                                len(matches),
                                graphs_root,
                                matches[0],
                            )
                        return matches[0]
        return None

    def resolve_exp_yaml_path(self, name: str) -> Path | None:
        """Locate ``graphs/{name}.exp.yaml`` — active overlay first, then frozen."""
        if self._active_dir is not None:
            active_path = self._active_dir / "graphs" / f"{name}.exp.yaml"
            if active_path.exists():
                return active_path
        frozen_path = self._frozen_dir / "graphs" / f"{name}.exp.yaml"
        if frozen_path.exists():
            return frozen_path
        return None

    def _load_global_hooks(self) -> None:
        """Load global hook definitions from ``hooks.json``.

        Overlay policy: full-file override. If active workspace has a
        ``hooks.json``, use it; otherwise fall back to frozen workspace.
        Hook arrays are not merged — partial-merge ordering across two
        files is too fragile.
        """
        hooks_path = self._frozen_dir / "hooks.json"
        if self._active_dir is not None:
            active_hooks = self._active_dir / "hooks.json"
            if active_hooks.exists():
                hooks_path = active_hooks
        self._global_hooks = load_hooks_file(str(hooks_path))
        if self._global_hooks:
            log.info("Loaded %d global hook(s) from %s", len(self._global_hooks), hooks_path)

    # ── Internal scanning ──

    def _import_module(self, subdir: str, py_file: Path) -> Any:
        """Import a single .py file with a unique module name per scan cycle."""
        safe_subdir = subdir.replace("/", "_").replace("\\", "_")
        mod_name = f"_ws_{safe_subdir}_{py_file.stem}_{self._scan_cycle}"
        # Clean stale entry
        sys.modules.pop(mod_name, None)
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _import_package(self, subdir: str, pkg_dir: Path) -> Any:
        """Import a subdirectory package via its ``__init__.py``.

        Temporarily adds the package's parent to ``sys.path`` so that
        relative imports within the package work (e.g. ``from .localize
        import LocalizeTool``).
        """
        init_file = pkg_dir / "__init__.py"
        mod_name = f"_ws_{subdir}_{pkg_dir.name}_{self._scan_cycle}"
        sys.modules.pop(mod_name, None)

        # Add parent dir to sys.path so intra-package imports resolve
        parent = str(pkg_dir.parent)
        added = parent not in sys.path
        if added:
            sys.path.insert(0, parent)
        try:
            spec = importlib.util.spec_from_file_location(
                mod_name,
                init_file,
                submodule_search_locations=[str(pkg_dir)],
            )
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod  # needed for relative imports
            spec.loader.exec_module(mod)
            return mod
        finally:
            if added:
                sys.path.remove(parent)

    def _find_subclasses(self, mod: Any, base_cls: type) -> list[type]:
        """Find all concrete subclasses of *base_cls* in a module."""
        found: list[type] = []
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, base_cls)
                and attr is not base_cls
                # Skip intermediate base classes imported into the module
                and not getattr(attr, "__abstractmethods__", None)
            ):
                found.append(attr)
        return found

    def _scan_subdir(self, subdir: str, base_cls: type) -> list[Any]:
        """Scan ``scan_dir/subdir/`` for *base_cls* subclasses.

        Supports three layouts:
        - **Single files**: ``subdir/foo.py`` — imported directly
        - **Packages**: ``subdir/foo/__init__.py`` — imported as a package
          (internal relative imports like ``from .bar import Baz`` work)
        - **Buckets**: ``subdir/foo/*.py`` (no ``__init__.py``) — each .py
          file imported as a standalone module, preserving its real source
          path for server-mode auto-host.  Used for grouping files that
          share a non-lint concern (e.g. ``nodesets/server/`` for files
          that run under a different Python interpreter).
        """
        target_dir = self._scan_dir / subdir
        if not target_dir.is_dir():
            return []

        instances: list[Any] = []

        # 1. Scan single .py files
        for py_file in sorted(target_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_from(
                subdir,
                py_file.name,
                lambda pf=py_file: self._import_module(subdir, pf),
                base_cls,
                instances,
            )

        # 2. Scan subdirectories — packages (__init__.py) or buckets
        for child in sorted(target_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            if (child / "__init__.py").exists():
                # Package mode
                self._load_from(
                    subdir,
                    child.name + "/",
                    lambda c=child: self._import_package(subdir, c),
                    base_cls,
                    instances,
                )
            else:
                # Bucket mode — scan nested .py files as standalone modules,
                # plus any package subdirs inside the bucket (so a folder-style
                # nodeset like ``policy/policy_adapter_vla/__init__.py`` is discovered
                # under the ``server/`` bucket).
                nested = f"{subdir}/{child.name}"
                for py_file in sorted(child.glob("*.py")):
                    if py_file.name.startswith("_"):
                        continue
                    self._load_from(
                        nested,
                        py_file.name,
                        lambda pf=py_file, nsd=nested: self._import_module(nsd, pf),
                        base_cls,
                        instances,
                    )
                for pkg_dir in sorted(child.iterdir()):
                    if not pkg_dir.is_dir() or pkg_dir.name.startswith("_"):
                        continue
                    if not (pkg_dir / "__init__.py").exists():
                        continue
                    self._load_from(
                        nested,
                        pkg_dir.name + "/",
                        lambda p=pkg_dir, nsd=nested: self._import_package(nsd, p),
                        base_cls,
                        instances,
                    )

        return instances

    def _load_from(
        self, subdir: str, label: str, importer, base_cls: type, instances: list
    ) -> None:
        """Import a module/package and collect base_cls subclass instances."""
        try:
            mod = importer()
            if mod is None:
                return
            for cls in self._find_subclasses(mod, base_cls):
                inst = cls()
                # Store source file for server mode auto-host (ADR-020).
                # Dynamically-loaded modules aren't in sys.modules, so
                # inspect.getfile() fails.  Capture mod.__file__ here.
                inst._source_file = getattr(mod, "__file__", None)
                instances.append(inst)
                log.info(
                    "  [%s] %s from %s",
                    subdir,
                    getattr(cls, "name", None)
                    or getattr(cls, "policy_id", None)
                    or getattr(cls, "agent_id", None)
                    or getattr(cls, "node_type", None)
                    or cls.__name__,
                    label,
                )
        except Exception:
            log.exception("Failed to load component from %s/%s", subdir, label)

    # ── Helper: register env panel (BaseEnvPanel) instance for a nodeset ──

    def _register_env_panel_for(
        self,
        ns: BaseNodeSet,
        *,
        mode: str = "local",
        server_url: str | None = None,
    ) -> None:
        """Instantiate and register a nodeset's env panel, if any.

        ``mode`` and ``server_url`` are stamped on the env panel so it can
        return an accurate status when the env is reachable (local) or
        running in a separate subprocess (server).
        """
        panel_cls = getattr(type(ns), "env_panel", None)
        if panel_cls is None:
            return
        try:
            from .env_panel import register_env_panel

            panel = panel_cls()
            panel._context = {
                "mode": mode,
                "server_url": server_url,
                "nodeset_name": ns.name,
            }
            register_env_panel(panel)
        except Exception:
            log.exception("Failed to register env panel for nodeset %s", ns.name)

    async def register_remote_nodeset(self, name: str, url: str) -> None:
        """Register a nodeset whose auto_host subprocess is owned by ANOTHER process.

        Used by ``eval_subprocess_main`` (run subprocess) to attach to shared
        singletons (Prismatic VLM, etc.) that the parent backend has
        already loaded — no second copy is spawned, no second 14 GB
        load. The subprocess gets proxy node classes + a remote
        env panel pointing at ``url``; ``is_nodeset_loaded(name)``
        returns True after this so subsequent
        ``ensure_nodesets_for_graph`` calls treat it as already-loaded.

        Lifecycle: the shim placed in ``_auto_servers[name]`` has a
        no-op ``stop()`` because we do NOT own the underlying process
        (killing it would break sibling run subprocesses sharing the
        same singleton).
        """
        import httpx

        from ..agent_loop.builtin_nodes import register_node
        from ..server._loopback_proxy import loopback_httpx_kwargs
        from ..server.manifest import ServerManifest
        from ..server.proxy import generate_proxy_nodes
        from ..standard.node_io import invalidate_cache

        with httpx.Client(timeout=15.0, **loopback_httpx_kwargs()) as client:
            resp = client.get(f"{url.rstrip('/')}/manifest")
            resp.raise_for_status()
            manifest = ServerManifest.from_dict(resp.json())

        proxy_classes = generate_proxy_nodes(url, manifest)
        node_types: list[str] = []
        for cls in proxy_classes:
            register_node(cls)
            node_types.append(cls.node_type)
        if proxy_classes:
            invalidate_cache()

        self._auto_servers[name] = _RemoteAutoServerShim(name=name, url=url)
        self._auto_server_node_types[name] = node_types
        self._auto_server_containers[name] = list(getattr(manifest, "containers", []))

        # Register env panel (best-effort at single-worker shape).
        self._register_remote_env_panel(name, url, tag=None)

        log.info(
            "Registered remote nodeset %s @ %s (%d proxy nodes)",
            name,
            url,
            len(node_types),
        )

    def _register_remote_env_panel(
        self, nodeset_name: str, server_url: str, tag: int | None = None
    ) -> None:
        """Fetch /env-panel/info from a spawned auto-server and register a
        RemoteEnvPanelProxy that forwards all calls over HTTP.

        At ``tag is None`` (single-worker / canvas-Play): registration is
        best-effort and failures are logged + swallowed — the panel just
        won't show this nodeset's env panel.

        At ``tag is not None`` (ADR-028 PB-1 multi-worker): the proxy's
        ``name`` is overridden to ``f"{nodeset_name}#{tag}"`` so each
        worker can address its own subprocess via
        ``get_env_panel(f"{nodeset_name}#{k}")``. Failures raise — the
        multi-worker invariants depend on every tagged env panel being
        live, so the caller can roll back the spawn.
        """
        try:
            import httpx

            from ..server._loopback_proxy import loopback_httpx_kwargs
            from .env_panel import RemoteEnvPanelProxy, register_env_panel

            with httpx.Client(timeout=10.0, **loopback_httpx_kwargs()) as client:
                resp = client.get("{}/env-panel/info".format(server_url.rstrip("/")))
                resp.raise_for_status()
                info = resp.json()
            proxy = RemoteEnvPanelProxy(server_url, info)
            if tag is not None:
                proxy.name = f"{nodeset_name}#{tag}"
            register_env_panel(proxy)
            log.info(
                "Registered RemoteEnvPanelProxy for %s -> %s",
                proxy.name,
                server_url,
            )
        except Exception:
            log.exception(
                "Failed to register remote env panel proxy for %s (server_url=%s, tag=%s)",
                nodeset_name,
                server_url,
                tag,
            )
            if tag is not None:
                raise

    def _unregister_env_panel_for(self, ns: BaseNodeSet) -> None:
        """Unregister a nodeset's env panel, if any."""
        panel_cls = getattr(type(ns), "env_panel", None)
        if panel_cls is None:
            return
        try:
            from .env_panel import unregister_env_panel

            panel_name = getattr(panel_cls, "name", None)
            if panel_name:
                unregister_env_panel(panel_name)
        except Exception:
            log.exception("Failed to unregister env panel for nodeset %s", ns.name)

    # ── Helper: register tool/node instances from a nodeset ──

    def _register_tool_instances(self, tools: list) -> int:
        """Register instances as canvas nodes for graph execution."""
        from ..agent_loop.builtin_nodes import register_node
        from ..standard.node_io import invalidate_cache

        for inst in tools:
            if hasattr(type(inst), "node_type"):
                register_node(type(inst))

        if tools:
            invalidate_cache()
        return len(tools)

    # ── Category-specific bridge logic ──

    def _scan_and_register_nodesets(self) -> int:
        """Scan nodesets/ for BaseNodeSet subclasses.  Discovery only — does NOT
        auto-load.  Use ``load_nodeset(name)`` to activate a nodeset and
        register its nodes.  Probes ``get_tools()`` to know what each
        nodeset would provide (for dependency detection)."""
        instances = self._scan_subdir("nodesets", BaseNodeSet)
        for inst in instances:
            self._discovered_nodesets[inst.name] = inst
            # Probe node types for dependency detection (without registering)
            try:
                tools = inst.get_tools()
                self._discovered_tool_names[inst.name] = [
                    getattr(t, "node_type", getattr(t, "name", "")) for t in tools
                ]
            except Exception:
                self._discovered_tool_names[inst.name] = []
            self._maybe_register_replay_parser(inst)
            log.info(
                "    → discovered nodeset: %s (%d nodes)",
                inst.name,
                len(self._discovered_tool_names[inst.name]),
            )
        return len(instances)

    def _maybe_register_replay_parser(self, ns: BaseNodeSet) -> None:
        """Register a replay parser for this nodeset.

        Resolution order:
            1. Class-declared ``replay_parser`` file (relative to nodeset
               source) — load it and register the contained
               :class:`BaseReplayParser` subclass.
            2. Otherwise, if the nodeset name starts with ``env_``,
               auto-register a :class:`GenericReplayParser` bound to that
               name. The generic parser uses the convention
               ``{nodeset_name}__reset`` for episode boundaries — works
               for any env that follows the standard node naming.

        Failures fall through to the generic fallback when applicable.
        """
        rel_path = getattr(type(ns), "replay_parser", None)
        if not rel_path:
            self._maybe_register_generic_replay_parser(ns)
            return
        source_file = getattr(ns, "_source_file", None)
        if source_file is None:
            log.warning(
                "Nodeset %s declares replay_parser=%s but has no "
                "_source_file; falling back to generic",
                ns.name,
                rel_path,
            )
            self._maybe_register_generic_replay_parser(ns)
            return
        parser_path = (Path(source_file).parent / rel_path).resolve()
        if not parser_path.exists():
            log.warning(
                "Replay parser file %s not found (declared by %s); falling back to generic",
                parser_path,
                ns.name,
            )
            self._maybe_register_generic_replay_parser(ns)
            return
        mod_name = f"_replay_parser_{ns.name}_{self._scan_cycle}"
        sys.modules.pop(mod_name, None)
        try:
            spec = importlib.util.spec_from_file_location(mod_name, parser_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load spec for {parser_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        except Exception:
            log.exception(
                "Failed to import replay parser %s for nodeset %s",
                parser_path,
                ns.name,
            )
            return
        # Filter to parsers defined IN the loaded file — not ones merely
        # imported (e.g. ``from app.replay.interface import
        # GenericReplayParser`` brings the base in as a module attribute,
        # which we don't want to instantiate as the env's parser).
        parsers = [
            c
            for c in self._find_subclasses(mod, BaseReplayParser)
            if getattr(c, "__module__", "") == mod_name
        ]
        if not parsers:
            log.warning(
                "Replay parser file %s declared by %s contains no "
                "locally-defined BaseReplayParser subclass",
                parser_path,
                ns.name,
            )
            return
        try:
            parser_inst = parsers[0]()
        except Exception:
            log.exception(
                "Failed to instantiate replay parser %s for nodeset %s",
                parsers[0].__name__,
                ns.name,
            )
            return
        self._replay_parsers[ns.name] = parser_inst
        log.info(
            "    → replay parser registered: %s -> %s",
            ns.name,
            parsers[0].__name__,
        )

    def _maybe_register_generic_replay_parser(self, ns: BaseNodeSet) -> None:
        """Auto-register a :class:`GenericReplayParser` for env_* nodesets.

        Convention: any nodeset whose name starts with ``env_`` is assumed
        to log episode boundaries as ``{name}__reset`` events. If that
        assumption breaks for a particular env, declare a custom
        ``replay_parser`` file on the nodeset class to override.
        """
        if not ns.name.startswith("env_"):
            return
        if ns.name in self._replay_parsers:
            return
        self._replay_parsers[ns.name] = GenericReplayParser(ns.name)
        log.info(
            "    → replay parser auto-registered (generic): %s",
            ns.name,
        )

    def get_replay_parser(self, name: str) -> BaseReplayParser | None:
        """Return the replay parser registered for a nodeset, or ``None``."""
        return self._replay_parsers.get(name)

    def _scan_and_register_policies(self) -> int:
        """Scan policies/ for BaseCanvasNode subclasses with policy metadata.

        Policy files are now direct BaseCanvasNode subclasses with
        ``policy_id``, ``name``, ``checkpoint``, ``config_path`` attributes.
        """
        from ..agent_loop import PolicyEntry
        from ..agent_loop.builtin_nodes import register_node
        from ..standard.node_io import invalidate_cache

        instances = self._scan_subdir("policies", BaseCanvasNode)
        for inst in instances:
            pid = getattr(inst, "policy_id", None)
            if pid is None:
                continue
            entry = PolicyEntry(
                id=pid,
                name=getattr(inst, "name", pid),
                checkpoint=getattr(inst, "checkpoint", ""),
                config=getattr(inst, "config_path", ""),
            )
            self._policies[entry.id] = entry
            if hasattr(type(inst), "node_type"):
                register_node(type(inst))
                self._standalone_node_types.add(inst.node_type)
        if instances:
            invalidate_cache()
        return len(instances)

    def _scan_and_register_nodes(self) -> int:
        """Scan nodes/ for BaseCanvasNode subclasses.  Inject into NODE_HANDLERS."""
        from ..agent_loop.builtin_nodes import register_node
        from ..standard.node_io import invalidate_cache

        instances = self._scan_subdir("nodes", BaseCanvasNode)
        for inst in instances:
            register_node(type(inst))
            self._standalone_node_types.add(inst.node_type)
            log.info("    → registered node type: %s", inst.node_type)
        # Invalidate cached schemas so new nodes appear
        if instances:
            invalidate_cache()
        return len(instances)

    # ── Server mode ──

    def _scan_and_register_servers(self) -> int:
        """Scan servers/ for YAML configs, optionally start managed servers,
        fetch manifests, and generate proxy nodes."""
        from ..agent_loop.builtin_nodes import register_node
        from ..standard.node_io import invalidate_cache

        servers_dir = self._scan_dir / "servers"
        if not servers_dir.is_dir():
            return 0

        total_nodes = 0
        for yaml_file in sorted(servers_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue
            try:
                total_nodes += self._load_server(yaml_file, register_node)
            except Exception:
                log.exception("Failed to load server config: %s", yaml_file.name)

        if total_nodes > 0:
            invalidate_cache()
        return total_nodes

    def _load_server(self, yaml_file: Path, register_fn: Any) -> int:
        """Load a single server YAML, optionally start it, fetch manifest,
        register proxy nodes."""
        import yaml

        with open(yaml_file) as f:
            config = yaml.safe_load(f) or {}

        url = config.get("url", "")
        enabled = config.get("enabled", True)
        managed = config.get("managed", False)

        if not url:
            log.warning("Server config %s has no 'url' — skipping", yaml_file.name)
            return 0

        # Always create BaseServer instance (even if disabled) so
        # start_server() can find it later when the user clicks Start.
        from ..server.base_server import BaseServer

        server = BaseServer(
            name=yaml_file.stem,
            url=url,
            command=config.get("command", ""),
            port=config.get("port", 9000),
            host=config.get("host", "localhost"),
            description=config.get("description", ""),
            startup_timeout=config.get("startup_timeout", 30),
            auto_restart=config.get("auto_restart", False),
            working_dir=config.get("working_dir", ""),
        )
        self._server_instances[yaml_file.stem] = server

        if not enabled:
            log.info("Server %s disabled — available for manual start", yaml_file.stem)
            self._servers[yaml_file.stem] = server.get_status()
            self._servers[yaml_file.stem]["nodes"] = []
            return 0

        # Start managed server
        if managed and server.command:
            try:
                server.start()
            except RuntimeError as e:
                log.warning("Failed to start managed server %s: %s", yaml_file.stem, e)
                self._servers[yaml_file.stem] = server.get_status()
                self._servers[yaml_file.stem]["nodes"] = []
                return 0

        # Fetch manifest (server must be running at this point)
        manifest = server.fetch_manifest()
        if manifest is None:
            log.warning("Server unreachable: %s (%s) — skipping", yaml_file.stem, url)
            self._servers[yaml_file.stem] = server.get_status()
            self._servers[yaml_file.stem]["nodes"] = []
            return 0

        # Generate and register proxy nodes
        from ..server.proxy import generate_proxy_nodes

        proxy_classes = generate_proxy_nodes(url, manifest)
        node_types = []
        for cls in proxy_classes:
            register_fn(cls)
            node_types.append(cls.node_type)
            log.info("    → server node: %s", cls.node_type)

        status = server.get_status()
        status["nodes"] = node_types
        status["version"] = manifest.version
        self._servers[yaml_file.stem] = status
        self._server_node_types[yaml_file.stem] = node_types
        log.info("Loaded server '%s' from %s: %d functions", manifest.name, url, len(proxy_classes))
        return len(proxy_classes)

    def start_server(self, name: str) -> dict:
        """Start a managed server by name. Returns status dict."""
        server = self._server_instances.get(name)
        if server is None:
            return {"error": f"Unknown server: {name}"}
        try:
            server.start()
        except RuntimeError as e:
            return {"error": str(e)}

        # Re-fetch manifest and register nodes if newly connected
        if server.connected:
            from ..agent_loop.builtin_nodes import register_node
            from ..standard.node_io import invalidate_cache

            manifest = server.fetch_manifest()
            if manifest:
                from ..server.proxy import generate_proxy_nodes

                proxy_classes = generate_proxy_nodes(server.url, manifest)
                node_types = []
                for cls in proxy_classes:
                    register_node(cls)
                    node_types.append(cls.node_type)
                invalidate_cache()
                self._server_node_types[name] = node_types
                status = server.get_status()
                status["nodes"] = node_types
                status["version"] = manifest.version
                self._servers[name] = status
                return status

        return server.get_status()

    def stop_server(self, name: str) -> dict:
        """Stop a managed server by name. Returns status dict."""
        server = self._server_instances.get(name)
        if server is None:
            return {"error": f"Unknown server: {name}"}
        server.stop()
        status = server.get_status()
        status["nodes"] = []
        self._servers[name] = status
        return status

    def list_servers(self) -> list[dict]:
        """List all registered server-mode nodesets with live status."""
        result = []
        for name, server in self._server_instances.items():
            status = server.get_status()
            status["nodes"] = self._server_node_types.get(name, [])
            result.append(status)
        return result
