"""REST endpoints for class-based component management (workspace/)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ...components import nodeset_source
from ...services.nodeset_watcher import roots_for
from ...state import ExecutionGuard, ExecutionMode, get_services

router = APIRouter()


# ── Node schema helpers (moved from navigate_api.py) ──


def _serialize_ui_config(cls: type) -> dict:
    """Serialize a node class's ui_config to JSON-safe dict."""
    ui = getattr(cls, "ui_config", None)
    if ui is None:
        return {
            "color": "",
            "layout": "block",
            "width": "",
            "min_height": "",
            "rounding": "",
            "min_width": "",
            "max_width": "",
            "config_fields": [],
            "display_fields": [],
        }
    return {
        "color": ui.color,
        "layout": ui.layout,
        "width": ui.width,
        "min_height": ui.min_height,
        "rounding": ui.rounding,
        "min_width": ui.min_width,
        "max_width": ui.max_width,
        "config_fields": [asdict(f) for f in ui.config_fields],
        "display_fields": [asdict(f) for f in ui.display_fields],
    }


def _build_profile_options() -> list[dict]:
    """Build dynamic select options from configured LLM profiles."""
    from ...llm import PROVIDER_REGISTRY, get_profile_store

    store = get_profile_store()
    profiles = store.list_profiles()
    active = store.get_active()

    # Default option — shows which profile is the fallback
    if active and active in profiles:
        reg = PROVIDER_REGISTRY.get(profiles[active].provider)
        default_label = f"Default ({reg.label if reg else profiles[active].provider} / {profiles[active].model})"
    else:
        default_label = "Default (none)"

    options = [{"value": "", "label": default_label}]
    for name, prof in profiles.items():
        reg = PROVIDER_REGISTRY.get(prof.provider)
        label = f"{reg.label if reg else prof.provider} / {prof.model}"
        options.append({"value": name, "label": label})
    return options


_DYNAMIC_PROFILES_SENTINEL = "__DYNAMIC_PROFILES__"


def _inject_dynamic_options(result: list[dict]) -> None:
    """Replace __DYNAMIC_PROFILES__ sentinel in config_fields with live profiles."""
    profile_options = _build_profile_options()
    for node_schema in result:
        ui = node_schema.get("ui_config")
        if not ui:
            continue
        for field in ui.get("config_fields", []):
            opts = field.get("options")
            if not opts:
                continue
            if any(o.get("value") == _DYNAMIC_PROFILES_SENTINEL for o in opts):
                field["options"] = profile_options


@router.get("/")
async def list_components():
    """List all registered workspace components by category."""
    state = get_services()
    return {
        "nodesets": list(state.workspace_component_registry._discovered_nodesets.keys()),
        "nodes": list(state.workspace_component_registry._nodeset_node_names.keys()),
    }


@router.post("/reload")
async def reload_components():
    """Rescan workspace/ and reload all components."""
    state = get_services()

    # Shutdown existing nodesets/envs before re-scanning
    await state.workspace_component_registry.shutdown_all()

    counts = state.workspace_component_registry.scan_all()

    # Initialize newly scanned nodesets
    await state.workspace_component_registry.initialize_all()

    return {"ok": True, "components": counts}


# ── Server mode ──


@router.get("/servers")
async def list_servers():
    """List all registered server-mode nodesets with live status."""
    return get_services().workspace_component_registry.list_servers()


@router.post("/servers/{name}/start")
async def start_server(name: str):
    """Start a managed server-mode nodeset by name."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        get_services().workspace_component_registry.start_server,
        name,
    )


@router.post("/servers/{name}/stop")
async def stop_server(name: str):
    """Stop a managed server-mode nodeset by name."""
    return get_services().workspace_component_registry.stop_server(name)


@router.post("/servers/{name}/restart")
async def restart_server(name: str):
    """Restart a managed server-mode nodeset by name."""
    import asyncio

    registry = get_services().workspace_component_registry
    registry.stop_server(name)
    return await asyncio.get_running_loop().run_in_executor(
        None,
        registry.start_server,
        name,
    )


# ── NodeSet management ──


@router.get("/nodesets")
async def list_nodesets():
    """List all discovered nodesets with loaded status."""
    return get_services().workspace_component_registry.list_nodesets()


@router.post("/nodesets/{name}/load")
async def load_nodeset(name: str, mode: str = "local"):
    """Load (or reload) a nodeset by name.

    Query params:
        mode: ``"local"`` (in-process) or ``"server"`` (auto-hosted subprocess).
    """
    try:
        result = await get_services().workspace_component_registry.load_nodeset(name, mode=mode)
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/nodesets/{name}/unload")
async def unload_nodeset(name: str):
    """Unload a nodeset — deregister its nodes and shut down."""
    try:
        result = await get_services().workspace_component_registry.unload_nodeset(name)
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/nodesets/ensure")
async def ensure_nodesets(req: dict):
    """Auto-load all nodesets required by a graph's node types.

    Body: { "node_types": ["env_habitat__step", "policy_adapter_vlnce__predict", ...] }
    Returns: { "loaded": [...], "already_loaded": [...], "failed": [...], "unknown": [...] }

    Shared logic used by both canvas and eval.
    """
    from ...graph_def import GraphDefinition, NodeDef

    node_types = req.get("node_types", [])
    # Build a minimal graph with just these node types for the registry method
    nodes = [
        NodeDef(id=f"_{i}", type=nt, label="", position=[0, 0]) for i, nt in enumerate(node_types)
    ]
    fake_graph = GraphDefinition(name="_ensure", nodes=nodes)
    registry = get_services().workspace_component_registry
    result = await registry.ensure_nodesets_for_graph(fake_graph)
    return result


@router.get("/nodesets/{name}/eval-metadata")
async def get_nodeset_eval_metadata(name: str):
    """Get eval metadata for a nodeset (splits, episodes, metrics)."""
    registry = get_services().workspace_component_registry
    metadata = await registry.get_eval_metadata_for_nodeset(name)
    loaded = registry.is_nodeset_loaded(name)
    return {"name": name, "loaded": loaded, "metadata": metadata}


# ── NodeSet source (canvas source editor) ──


def _resolve_source_target(name: str, rel_file: str | None) -> tuple[dict, str, Path]:
    """Shared GET/PUT resolution: nodeset info + guarded target path.

    Returns ``(info, rel_file, target)``; raises HTTPException on any miss.
    """
    registry = get_services().workspace_component_registry
    info = registry.nodeset_source_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown nodeset (or no source file): {name}")
    anchor, package_dir = nodeset_source.resolve_source(info["source_file"])
    if rel_file is None:
        rel_file = "__init__.py" if package_dir is not None else anchor.name
    try:
        target = nodeset_source.resolve_target(
            info["source_file"], rel_file, roots_for(registry)
        )
    except nodeset_source.SourcePathError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"No such source file: {rel_file}") from e
    return info, rel_file, target


@router.get("/nodesets/{name}/source")
async def get_nodeset_source(name: str, file: str | None = None, node_type: str | None = None):
    """Read a nodeset source file for the canvas source editor.

    Query params:
        file:      Relative path within a package nodeset (default: the
                   entry file — ``__init__.py`` for packages).
        node_type: Optional graph node type; when resolvable, the response
                   includes ``class_name`` so the editor can scroll to it.
    """
    info, rel_file, target = _resolve_source_target(name, file)
    _, package_dir = nodeset_source.resolve_source(info["source_file"])
    is_package = package_dir is not None
    files = (
        nodeset_source.list_package_files(package_dir)
        if is_package
        else [info["source_file"].name]
    )

    class_name = None
    if node_type and "__" in node_type:
        from ...agent_loop.builtin_nodes import NODE_HANDLERS

        cls = NODE_HANDLERS.get(node_type)
        class_name = cls.__name__ if cls is not None else None

    return {
        "name": name,
        "mode": info["mode"],
        "requires_server": info["requires_server"],
        "loaded": info["loaded"],
        "is_package": is_package,
        "files": files,
        "file": rel_file,
        "content": target.read_text(encoding="utf-8"),
        "mtime_ns": target.stat().st_mtime_ns,
        "class_name": class_name,
    }


def _scoped_candidates(info: dict) -> list[str]:
    """Files to search for a node's class: entry file first, then siblings."""
    anchor, package_dir = nodeset_source.resolve_source(info["source_file"])
    if package_dir is None:
        return [anchor.name]
    entry = "__init__.py"
    rest = [f for f in nodeset_source.list_package_files(package_dir) if f != entry]
    return [entry, *rest]


@router.get("/nodesets/{name}/source/scoped")
async def get_nodeset_source_scoped(name: str, node_type: str):
    """A node's editable slice of its nodeset source (Source tab).

    Locates the class assigning ``node_type = "<node_type>"`` across the
    nodeset's files and returns its segments: module-level globals, the
    functions the class transitively references, and the class itself —
    each with the 1-based line range used to splice edits back.
    """
    registry = get_services().workspace_component_registry
    info = registry.nodeset_source_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown nodeset (or no source file): {name}")
    roots = roots_for(registry)
    for rel_file in _scoped_candidates(info):
        try:
            target = nodeset_source.resolve_target(info["source_file"], rel_file, roots)
        except (nodeset_source.SourcePathError, FileNotFoundError):
            continue
        try:
            segments = nodeset_source.extract_scoped_view(
                target.read_text(encoding="utf-8"), node_type
            )
        except SyntaxError:
            continue  # broken sibling file — the class may live elsewhere
        if segments is not None:
            return {
                "name": name,
                "node_type": node_type,
                "mode": info["mode"],
                "requires_server": info["requires_server"],
                "loaded": info["loaded"],
                "file": rel_file,
                "mtime_ns": target.stat().st_mtime_ns,
                "segments": segments,
            }
    raise HTTPException(
        status_code=404,
        detail=f"No class with node_type={node_type!r} found in nodeset {name!r}",
    )


@router.put("/nodesets/{name}/source/scoped")
async def put_nodeset_source_scoped(name: str, req: dict):
    """Splice edited scoped segments back into the nodeset source file.

    Body: ``{ "file": str, "node_type": str, "base_mtime_ns": int,
    "segments": [{start_line, end_line, text}, ...] }``. Same reload
    semantics as the whole-file PUT (watcher picks local nodesets up;
    server-mode stays stale until an explicit restart).
    """
    rel_file = req.get("file")
    node_type = req.get("node_type")
    base_mtime_ns = req.get("base_mtime_ns")
    segments = req.get("segments")
    if not isinstance(rel_file, str) or not isinstance(segments, list) or not segments:
        raise HTTPException(
            status_code=400,
            detail="body must include 'file' (str) and non-empty 'segments' (list)",
        )

    info, rel_file, target = _resolve_source_target(name, rel_file)

    if base_mtime_ns is not None and target.stat().st_mtime_ns != base_mtime_ns:
        raise HTTPException(
            status_code=409, detail="file changed on disk since it was loaded — re-open to refresh"
        )

    try:
        merged = nodeset_source.splice_segments(
            target.read_text(encoding="utf-8"), segments
        )
    except (nodeset_source.ScopedEditError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid segments: {e}") from e

    try:
        nodeset_source.syntax_check(merged, filename=rel_file)
    except nodeset_source.SourceSyntaxError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "syntax", "msg": e.msg, "line": e.lineno, "offset": e.offset},
        ) from e

    target.write_text(merged, encoding="utf-8")
    # Fresh ranges (line numbers may have shifted) so the client can keep editing.
    fresh = (
        nodeset_source.extract_scoped_view(merged, node_type)
        if isinstance(node_type, str)
        else None
    )
    run_active = ExecutionGuard.current().get("mode") != ExecutionMode.idle.value
    return {
        "ok": True,
        "mtime_ns": target.stat().st_mtime_ns,
        "mode": info["mode"],
        "stale": info["mode"] == "server",
        "run_active": run_active,
        "segments": fresh,
    }


@router.put("/nodesets/{name}/source")
async def put_nodeset_source(name: str, req: dict):
    """Write a nodeset source file (canvas source editor save).

    Body: ``{ "file": str, "content": str, "base_mtime_ns": int | None }``.

    The write itself does not reload anything: local nodesets are picked
    up by the nodeset watcher (deferred while a run holds the
    ExecutionGuard — surfaced as ``run_active``); server-mode nodesets are
    never auto-reloaded (``stale: true`` — restart the server or POST
    ``/api/components/reload`` to apply).
    """
    rel_file = req.get("file")
    content = req.get("content")
    base_mtime_ns = req.get("base_mtime_ns")
    if not isinstance(rel_file, str) or not isinstance(content, str):
        raise HTTPException(
            status_code=400, detail="body must include 'file' (str) and 'content' (str)"
        )

    info, rel_file, target = _resolve_source_target(name, rel_file)

    if base_mtime_ns is not None and target.stat().st_mtime_ns != base_mtime_ns:
        raise HTTPException(
            status_code=409, detail="file changed on disk since it was loaded — re-open to refresh"
        )

    try:
        nodeset_source.syntax_check(content, filename=rel_file)
    except nodeset_source.SourceSyntaxError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "syntax", "msg": e.msg, "line": e.lineno, "offset": e.offset},
        ) from e

    target.write_text(content, encoding="utf-8")
    run_active = ExecutionGuard.current().get("mode") != ExecutionMode.idle.value
    return {
        "ok": True,
        "mtime_ns": target.stat().st_mtime_ns,
        "mode": info["mode"],
        "stale": info["mode"] == "server",
        "run_active": run_active,
    }


# ── Node Schemas ──


@router.get("/node-schemas")
async def list_node_schemas():
    """List all available node types (built-in + nodeset) with port metadata and ui_config."""
    from ...agent_loop.builtin_nodes import NODE_HANDLERS

    seen: set[str] = set()
    result = []
    for _node_type, cls in NODE_HANDLERS.items():
        canon = cls.node_type
        if canon in seen:
            continue
        seen.add(canon)
        result.append(
            {
                "type": canon,
                "display_name": cls.display_name,
                "description": getattr(cls, "description", ""),
                "category": cls.category,
                "icon": getattr(cls, "icon", ""),
                "kind": getattr(cls, "kind", "block"),
                "ports_mode": getattr(cls, "ports_mode", "mirror"),
                "config_schema": getattr(cls, "config_schema", {}),
                "default_config": getattr(cls, "default_config", {}),
                "ui_config": _serialize_ui_config(cls),
                "input_ports": [
                    {
                        "name": p.name,
                        "wire_type": p.wire_type,
                        "description": p.description,
                        "optional": p.optional,
                    }
                    for p in cls.input_ports
                ],
                "output_ports": [
                    {
                        "name": p.name,
                        "wire_type": p.wire_type,
                        "description": p.description,
                        "optional": p.optional,
                    }
                    for p in cls.output_ports
                ],
            }
        )

    _inject_dynamic_options(result)
    return result
