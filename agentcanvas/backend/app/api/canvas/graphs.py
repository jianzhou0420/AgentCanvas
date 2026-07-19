"""CRUD API for saved graph definitions.

Two storage roots, each a real on-disk folder tree (subdirectories allowed):
- ``workspace/graphs/``       — editable graph templates (kind="graph")
- ``workspace/graph_nodes/``  — archived composite nodes  (kind="node")

A graph's ``_id`` is its POSIX path **relative to its kind root, without the
``.json`` suffix** — e.g. ``experiments/foo`` for
``workspace/graphs/experiments/foo.json``. Top-level files keep a bare-stem id
(``foo``), so pre-hierarchy graphs are unaffected.

Endpoints
---------
GET    /api/graphs                 List all saved graphs + graph nodes
GET    /api/graphs/folders         List folder paths (incl. empty) for a kind
POST   /api/graphs/folders         Create a folder (mkdir -p)
POST   /api/graphs/folders/rename  Rename / move a folder
DELETE /api/graphs/folders         Delete a folder (recursive opt-in)
POST   /api/graphs/layout          Auto-layout a graph (positions only)
POST   /api/graphs/{id}/move       Move / rename a graph file
GET    /api/graphs/{id}            Load a specific graph/node
POST   /api/graphs                 Save a new graph or node
PUT    /api/graphs/{id}            Overwrite an existing graph/node
DELETE /api/graphs/{id}            Delete a saved graph/node

Path safety: every client-supplied relative path goes through ``_safe_join``,
which resolves it under the kind root and rejects anything that escapes (``..``,
absolute paths, symlink traversal) with HTTP 400.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, model_validator

from ...graph_def import GraphDefinition
from ...layout import build_height_map, layout_graph

log = logging.getLogger("agentcanvas.graphs")

router = APIRouter()

# canvas/graphs.py -> canvas/ -> api/ -> app/ -> backend/ -> agentcanvas/ -> vlnworkspace/
_WORKSPACE = Path(__file__).resolve().parents[5] / "workspace"
GRAPHS_DIR = _WORKSPACE / "graphs"
GRAPH_NODES_DIR = _WORKSPACE / "graph_nodes"


# ── Request models ──


class GraphSaveRequest(BaseModel):
    """Payload for saving a graph.  Validated then converted to GraphDefinition."""

    # Reject unknown fields with HTTP 422 instead of silently dropping them.
    # The previous default ("ignore") had silently lost `containers` /
    # `access_grants` for every save until this schema declared them.
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    nodes: list = []
    edges: list = []
    containers: list = []
    access_grants: list = []
    step_budget: int | None = 500
    eval_graph: bool = True
    kind: str = "graph"  # "graph" or "node"
    group: str = ""  # legacy flat group for graph nodes — superseded by `folder`
    folder: str = ""  # target subdirectory under the kind root ("" = root)
    presetId: str | None = None
    hooks: list = []

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_max_iterations(cls, data: Any) -> Any:
        # Backward compat: pre-refactor saved graphs and (briefly) the
        # frontend send ``maxIterations``. Map it onto ``step_budget`` so
        # extra="forbid" doesn't reject the request.
        if isinstance(data, dict) and "maxIterations" in data and "step_budget" not in data:
            data = dict(data)
            data["step_budget"] = data.pop("maxIterations")
        elif isinstance(data, dict) and "maxIterations" in data:
            data = dict(data)
            data.pop("maxIterations", None)
        return data


class FolderRequest(BaseModel):
    kind: str = "graph"
    path: str


class FolderRenameRequest(BaseModel):
    kind: str = "graph"
    path: str
    new_path: str


class GraphMoveRequest(BaseModel):
    dest_folder: str = ""
    new_name: str | None = None


# ── Helpers ──


def _ensure_dirs() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_NODES_DIR.mkdir(parents=True, exist_ok=True)


def _base_for_kind(kind: str) -> Path:
    return GRAPH_NODES_DIR if kind == "node" else GRAPHS_DIR


def _base_of(path: Path) -> Path:
    """Return which kind root a resolved file/dir lives under."""
    resolved = path.resolve()
    for base in (GRAPHS_DIR, GRAPH_NODES_DIR):
        b = base.resolve()
        if resolved == b or b in resolved.parents:
            return base
    raise HTTPException(status_code=400, detail=f"Path outside graph roots: {path}")


def _safe_join(base: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``base``; reject traversal / absolute / escape.

    ``rel`` is treated as a relative POSIX-ish path; leading/trailing slashes
    are stripped. The resolved result must equal ``base`` or be contained in it.
    """
    rel = (rel or "").strip().replace("\\", "/").strip("/")
    candidate = (base / rel).resolve()
    base_resolved = base.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"Path escapes graph root: {rel!r}")
    return candidate


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _rel_id(path: Path, base: Path) -> str:
    """POSIX relative path of a graph file under its root, sans ``.json``."""
    return path.resolve().relative_to(base.resolve()).with_suffix("").as_posix()


def _folder_of(path: Path, base: Path) -> str:
    """POSIX relative parent dir of a graph file under its root ("" = root)."""
    parent = path.resolve().parent.relative_to(base.resolve()).as_posix()
    return "" if parent == "." else parent


def _load_graph(path: Path, base: Path) -> dict[str, Any]:
    """Load and validate a graph JSON file, returning its dict representation."""
    raw = json.loads(path.read_text())
    gd = GraphDefinition.from_dict(raw)
    result = gd.to_dict()
    result["_id"] = _rel_id(path, base)
    result["_path"] = str(path)
    result["folder"] = _folder_of(path, base)
    return result


def _save_graph(path: Path, gd: GraphDefinition) -> None:
    """Write a GraphDefinition to disk as formatted JSON (parent dirs ensured)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gd.to_dict(), indent=2))


def _find_graph(graph_id: str) -> Path | None:
    """Find a graph file by relative-path id across both roots.

    Traversal attempts raise HTTP 400 via ``_safe_join``; a clean miss is None.
    """
    for base in (GRAPHS_DIR, GRAPH_NODES_DIR):
        p = _safe_join(base, f"{graph_id}.json")
        if p.is_file():
            return p
    return None


# ── Folder endpoints (declared before the {graph_id:path} catch-alls) ──


@router.get("/folders")
async def list_folders(kind: str = Query("graph")) -> list[str]:
    """List every subdirectory (including empty ones) under a kind root."""
    _ensure_dirs()
    base = _base_for_kind(kind)
    base_resolved = base.resolve()
    folders = [
        d.resolve().relative_to(base_resolved).as_posix() for d in base.rglob("*") if d.is_dir()
    ]
    return sorted(folders)


@router.post("/folders")
async def create_folder(req: FolderRequest) -> dict[str, str]:
    """Create a folder (mkdir -p) under a kind root."""
    _ensure_dirs()
    base = _base_for_kind(req.kind)
    target = _safe_join(base, req.path)
    if target == base.resolve():
        raise HTTPException(status_code=400, detail="Folder path is required")
    target.mkdir(parents=True, exist_ok=True)
    rel = target.relative_to(base.resolve()).as_posix()
    log.info("Created %s folder %s", req.kind, rel)
    return {"created": rel}


@router.post("/folders/rename")
async def rename_folder(req: FolderRenameRequest) -> dict[str, str]:
    """Rename / move a folder within a kind root."""
    _ensure_dirs()
    base = _base_for_kind(req.kind)
    src = _safe_join(base, req.path)
    dst = _safe_join(base, req.new_path)
    if not src.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {req.path}")
    if dst.exists():
        raise HTTPException(status_code=409, detail=f"Target already exists: {req.new_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    rel = dst.relative_to(base.resolve()).as_posix()
    log.info("Renamed %s folder %s -> %s", req.kind, req.path, rel)
    return {"path": rel}


@router.delete("/folders")
async def delete_folder(
    kind: str = Query("graph"),
    path: str = Query(...),
    recursive: bool = Query(False),
) -> dict[str, str]:
    """Delete a folder. Non-empty folders require ``recursive=true``."""
    _ensure_dirs()
    base = _base_for_kind(kind)
    target = _safe_join(base, path)
    if target == base.resolve():
        raise HTTPException(status_code=400, detail="Cannot delete the root")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {path}")
    has_contents = any(target.iterdir())
    if has_contents and not recursive:
        raise HTTPException(status_code=409, detail="Folder not empty (pass recursive=true)")
    if recursive:
        shutil.rmtree(target)
    else:
        target.rmdir()
    log.info("Deleted %s folder %s (recursive=%s)", kind, path, recursive)
    return {"deleted": path}


# ── Graph endpoints ──


@router.get("")
async def list_graphs() -> list[dict[str, Any]]:
    """List all saved graph definitions from both roots (recursive)."""
    _ensure_dirs()
    results: list = []
    for base in (GRAPHS_DIR, GRAPH_NODES_DIR):
        for p in sorted(base.rglob("*.json")):
            try:
                results.append(_load_graph(p, base))
            except Exception as exc:
                log.warning("Failed to read graph %s: %s", p, exc)
    return results


@router.post("")
async def save_graph(req: GraphSaveRequest) -> dict[str, str]:
    """Save a new graph definition (optionally inside ``folder``)."""
    _ensure_dirs()
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Graph name is required")
    base = _base_for_kind(req.kind)
    rel = (
        f"{req.folder}/{_slug(req.name)}.json" if req.folder.strip() else f"{_slug(req.name)}.json"
    )
    path = _safe_join(base, rel)
    gd = GraphDefinition.from_dict(req.dict())
    _save_graph(path, gd)
    graph_id = _rel_id(path, base)
    log.info("Saved %s %s to %s", req.kind, graph_id, path)
    return {"id": graph_id, "path": str(path)}


@router.post("/layout")
async def layout_graph_api(request: Request) -> dict[str, Any]:
    """Apply auto-layout to a graph definition (updates positions only).

    Accepts an optional ``dimensions`` field ({node_id: {width, height}}) with
    the canvas-measured node sizes, so columns are spaced by real width and
    rows by real height.  Absent → fixed-pitch fallback.
    """
    body = await request.json()
    dims = body.pop("dimensions", None) if isinstance(body, dict) else None
    return layout_graph(body, node_heights=build_height_map(), node_dims=dims)


@router.post("/{graph_id:path}/move")
async def move_graph(graph_id: str, req: GraphMoveRequest) -> dict[str, str]:
    """Move / rename a graph file within its kind root. Returns the new id."""
    _ensure_dirs()
    src = _find_graph(graph_id)
    if src is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    base = _base_of(src)
    new_stem = _slug(req.new_name) if req.new_name else src.stem
    rel = f"{req.dest_folder}/{new_stem}.json" if req.dest_folder.strip() else f"{new_stem}.json"
    dst = _safe_join(base, rel)
    if dst == src.resolve():
        return {"id": _rel_id(dst, base), "path": str(dst)}
    if dst.exists():
        raise HTTPException(status_code=409, detail=f"Target already exists: {rel}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    new_id = _rel_id(dst, base)
    log.info("Moved graph %s -> %s", graph_id, new_id)
    return {"id": new_id, "path": str(dst)}


@router.get("/{graph_id:path}")
async def get_graph(graph_id: str) -> dict[str, Any]:
    """Load a specific graph definition by relative-path id."""
    _ensure_dirs()
    path = _find_graph(graph_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    return _load_graph(path, _base_of(path))


@router.put("/{graph_id:path}")
async def update_graph(graph_id: str, req: GraphSaveRequest) -> dict[str, str]:
    """Overwrite an existing graph definition in place."""
    _ensure_dirs()
    path = _find_graph(graph_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    base = _base_of(path)
    gd = GraphDefinition.from_dict(req.dict())
    _save_graph(path, gd)
    log.info("Updated graph %s at %s", graph_id, path)
    return {"id": _rel_id(path, base), "path": str(path)}


@router.post("/validate")
async def validate_graph_endpoint(req: Request, ensure: bool = Query(False)) -> dict[str, Any]:
    """Dev-time whole-graph wire-type check (non-raising).

    Body: a graph definition (same shape as ``POST /api/graphs``). Resolves
    each edge's source/target port types and reports shape mismatches. Coverage
    of nodeset (env/method) edges depends on those nodesets being loaded — pass
    ``?ensure=true`` to load the graph's nodesets first for full coverage;
    otherwise edges touching not-yet-loaded nodesets are reported as ``skipped``.

    Returns ``wire_errors`` + a ``coverage`` summary. Does NOT raise on
    mismatches — the editor renders them; the hard 400 lives on the run/eval
    path (ADR-027 staged enforcement).
    """
    from ...graph_def import wire_type_report
    from ...state import get_services

    body = await req.json()
    graph = GraphDefinition.from_dict(body.get("graph", body))

    if ensure:
        try:
            await get_services().workspace_component_registry.ensure_nodesets_for_graph(graph)
        except Exception as e:
            log.warning("validate: ensure_nodesets failed: %s", e)

    report = wire_type_report(graph)
    return {
        "wire_errors": report["errors"],
        "coverage": {
            "checked": report["checked"],
            "skipped": report["skipped"],
            "total_edges": report["total_edges"],
            "unresolved_node_types": report["unresolved_node_types"],
        },
    }


@router.delete("/{graph_id:path}")
async def delete_graph(graph_id: str) -> dict[str, str]:
    """Delete a saved graph definition."""
    _ensure_dirs()
    path = _find_graph(graph_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    path.unlink()
    log.info("Deleted graph %s", graph_id)
    return {"deleted": graph_id}
