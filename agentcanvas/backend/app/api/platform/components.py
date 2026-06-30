"""REST endpoints for class-based component management (workspace/)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from ...state import get_services

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

    Body: { "node_types": ["env_habitat__step", "policy_cma__forward", ...] }
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
