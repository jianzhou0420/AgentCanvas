"""SSG — Spatial Scene Graph nodeset (SpatialNav port, M1).

Wraps the three user-facing operations exposed by the ported
``tmp/_reference/spatialnav/spatial`` module:

- ``ssg__render_map``    — agent-centric top-down map (PIL.Image) from
                           MP3D ground-truth ``.house`` layout +
                           ``_semantic.ply`` point cloud.
- ``ssg__reset_episode`` — per-episode reset for ``prev_level_id`` and
                           per-scan ``room_cache`` (footgun mitigation
                           from the NOTICE → Known Divergences list).
- ``ssg__query_objects`` — instruction-aware object retrieval via
                           ``SceneObjectGraph.get_local_objects`` +
                           ``get_interest_objects``.

Lifecycle:
    - ``SceneGraphMapper`` is a process-wide singleton, lazy-loaded on
      first ``ssg__render_map`` call. Per-scan point-cloud + layout
      caches live on the mapper; they are retained across episodes for
      performance but cleared on ``shutdown()``.
    - ``SceneObjectGraph`` is per-``data_name`` singleton (``R2R`` vs
      ``REVERIE`` have different ``interested_objects`` jsonl). Loads a
      ``SentenceTransformer("all-MiniLM-L6-v2")`` on first use.
    - The upstream ``room_cache`` is ``defaultdict(dict)`` and grows
      unbounded — we swap in an LRU ``OrderedDict`` with configurable
      ``max_cache_size`` on first construction.

CWD-relative path footguns in the upstream source (``MP3D_SCENE_DIR``
at ``mapper.py:33``, ``data/models/all-MiniLM-L6-v2`` at
``agraph.py:37``, ``data/tasks/{R2R,REVERIE}_category.jsonl`` at
``agraph.py:48``) are neutralised here by routing all paths through
``ConfigField`` values and bypassing ``SceneObjectGraph.__init__``
with a thin subclass.

Load:
    POST /api/components/nodesets/ssg/load

Reference (for M2):
    workspace/graphs/spatialnav_mp3d.json will fork navgpt_mp3d.json
    and add ``ssg__render_map`` + ``ssg__reset_episode`` +
    ``ssg__query_objects`` in parallel with the existing
    BLIP-2 / R-CNN perception chain.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Path bootstrap — make ``tmp._reference.spatialnav.*`` importable from
# inside the agentcanvas backend (which runs with cwd = backend dir).
# ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton manager (thread-safe, lazy init)
# ──────────────────────────────────────────────────────────────────────
_manager_lock = threading.Lock()
_manager: _SSGManager | None = None


def _get_manager(config: dict) -> _SSGManager:
    """Return (or initialise) the process-wide SSG manager, seeding it
    from the calling node's ``config`` dict on first access."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = _SSGManager.from_config(config)
        return _manager


class _LRUDict(OrderedDict):
    """Bounded LRU OrderedDict; used to swap upstream's unbounded
    ``room_cache``. Move-on-access + evict-oldest-on-insert."""

    def __init__(self, *args: Any, max_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_size = max_size

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._max_size:
            self.popitem(last=False)

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


class _SSGManager:
    """Owns the singleton ``SceneGraphMapper`` and per-``data_name``
    ``SceneObjectGraph`` instances. All attribute access is
    threadsafe via instance locks."""

    def __init__(
        self,
        mp3d_data_dir: str,
        category_file: str,
        st_model_path: str,
        category_jsonl_dir: str,
        max_cache_size: int,
    ) -> None:
        self.mp3d_data_dir = mp3d_data_dir
        self.category_file = category_file
        self.st_model_path = st_model_path
        self.category_jsonl_dir = category_jsonl_dir
        self.max_cache_size = max_cache_size
        self._mapper: Any = None
        self._agraphs: dict[str, Any] = {}
        self._mapper_lock = threading.Lock()
        self._agraph_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: dict) -> _SSGManager:
        return cls(
            mp3d_data_dir=config.get("mp3d_data_dir", "data/scene_datasets/mp3d/v1/tasks/mp3d"),
            category_file=config.get(
                "category_file",
                "data/scene_datasets/mp3d/v1/tasks/mp3d/category_spatiallm_mapping.csv",
            ),
            st_model_path=config.get("st_model_path", "sentence-transformers/all-MiniLM-L6-v2"),
            category_jsonl_dir=config.get("category_jsonl_dir", "data/tasks"),
            max_cache_size=int(config.get("max_cache_size", 500)),
        )

    def get_mapper(self, scene_config_overrides: dict[str, Any]) -> Any:
        """Return the process-wide ``SceneGraphMapper``, constructing it
        on first call. ``scene_config_overrides`` are only honoured on
        first construction (the singleton is pinned after that)."""
        with self._mapper_lock:
            if self._mapper is None:
                from tmp._reference.spatialnav.spatial.mapper import (  # type: ignore
                    SceneGraphConfig,
                    SceneGraphMapper,
                )

                kwargs = dict(
                    point_cloud_dir=self.mp3d_data_dir,
                    layout_dir=self.mp3d_data_dir,
                    category_file_path=self.category_file,
                )
                for k, v in scene_config_overrides.items():
                    if v is not None:
                        kwargs[k] = v
                cfg = SceneGraphConfig(**kwargs)
                self._mapper = SceneGraphMapper(config=cfg, scans=[], env_type="mp3d")
                log.info("SSG: SceneGraphMapper initialised (layout_dir=%s)", self.mp3d_data_dir)
            return self._mapper

    def get_agraph(self, data_name: str, visibility_radius: float) -> Any:
        """Return the per-``data_name`` ``SceneObjectGraph``. First call
        triggers SentenceTransformer model load (~80 MB)."""
        key = data_name.upper()
        with self._agraph_lock:
            if key not in self._agraphs:
                self._agraphs[key] = self._build_agraph(key, visibility_radius)
            return self._agraphs[key]

    def reset_episode(self, scan_id: str, evict_scan_from_room_cache: bool = False) -> None:
        """Clear ``prev_level_id`` on the mapper and all loaded
        ``SceneObjectGraph``s. Optionally evict the given scan from the
        room cache (forces fresh object retrieval next episode)."""
        if self._mapper is not None:
            self._mapper.prev_level_id = None
        for graph in self._agraphs.values():
            graph.prev_level_id = None
            if evict_scan_from_room_cache and scan_id in graph.room_cache:
                graph.room_cache.pop(scan_id, None)

    def _build_agraph(self, data_name: str, visibility_radius: float) -> Any:
        """Construct a ``SceneObjectGraph`` bypassing its CWD-relative
        ``__init__`` path hardcoding. Loads SentenceTransformer from
        ``st_model_path``; loads category jsonl from
        ``{category_jsonl_dir}/{data_name}_category.jsonl`` if present,
        else leaves ``interested_objects`` empty (warning logged)."""
        import torch
        from sentence_transformers import SentenceTransformer
        from tmp._reference.spatialnav.spatial.agraph import (  # type: ignore
            SceneObjectConfig,
            SceneObjectGraph,
        )
        from tmp._reference.spatialnav.spatial.reader import read_category  # type: ignore

        cfg = SceneObjectConfig(
            layout_dir=self.mp3d_data_dir,
            category_file_path=self.category_file,
            visibility_radius=visibility_radius,
        )

        graph = SceneObjectGraph.__new__(SceneObjectGraph)
        graph.config = cfg
        graph.scans = []
        graph.env_type = "mp3d"
        graph.data_name = data_name

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        graph.model = SentenceTransformer(self.st_model_path, device=device)
        log.info("SSG: SentenceTransformer loaded from %s on %s", self.st_model_path, device)

        graph.houses = {}
        graph.category_mapping = read_category(cfg.category_file_path, cfg.category_col)

        jsonl_path = Path(self.category_jsonl_dir) / f"{data_name}_category.jsonl"
        graph.interested_objects = self._load_interested(jsonl_path, data_name)
        graph.room_cache = _LRUDict(max_size=self.max_cache_size)
        graph.prev_level_id = None
        return graph

    @staticmethod
    def _load_interested(jsonl_path: Path, data_name: str) -> dict:
        if not jsonl_path.is_file():
            log.warning(
                "SSG: category jsonl %s not found; interest_objects disabled for %s",
                jsonl_path,
                data_name,
            )
            return {}
        with jsonl_path.open() as f:
            lines = [json.loads(ln) for ln in f]
        if data_name == "R2R":
            return {
                line["path_id"]: list(itertools.chain.from_iterable(line["category"]))
                for line in lines
            }
        if data_name == "REVERIE":
            return {
                line["id"]: list(itertools.chain.from_iterable(line["category"])) for line in lines
            }
        log.warning("SSG: unknown data_name=%r; interest_objects empty", data_name)
        return {}


# ──────────────────────────────────────────────────────────────────────
# Node 1 — ssg__render_map
# ──────────────────────────────────────────────────────────────────────


class SSGRenderMapNode(BaseCanvasNode):
    """Render the agent-centric top-down SSG map.

    Wraps ``SceneGraphMapper.get_visual_map(scan_id, position,
    orientation, navigable_viewpoints, history_viewpoints)``. Lazy-
    loads the layout + point cloud for ``scan_id`` on first call.
    Returns a ``PIL.Image`` with agent red dot + orientation arrow,
    blue candidate dots (optionally numbered), orange history
    trajectory, and grey room-boundary rectangles with room-type text.
    """

    node_type = "ssg__render_map"
    display_name = "SSG: Render Map"
    description = (
        "Render the SpatialNav top-down map (agent-centric, room-labelled) "
        "from MP3D ground-truth .house + _semantic.ply"
    )
    category = "perception"
    icon = "Map"

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "mp3d_data_dir",
                "text",
                "MP3D data dir (contains scan subfolders with .house / _semantic.ply)",
                default="data/scene_datasets/mp3d/v1/tasks/mp3d",
            ),
            ConfigField(
                "category_file",
                "text",
                "MP3D category mapping CSV",
                default="data/scene_datasets/mp3d/v1/tasks/mp3d/category_spatiallm_mapping.csv",
            ),
            ConfigField(
                "grid_size",
                "slider",
                "Map grid size (meters/pixel)",
                default=0.010,
                min=0.005,
                max=0.050,
                step=0.001,
            ),
            ConfigField(
                "agent_front_up",
                "toggle",
                "Rotate map so agent always faces up",
                default=False,
            ),
            ConfigField(
                "rotation_strategy",
                "select",
                "Rotation strategy (only if agent_front_up)",
                options=[
                    {"value": "absolute", "label": "Absolute (exact yaw)"},
                    {"value": "relative", "label": "Relative (snap to 90°)"},
                ],
                default="absolute",
            ),
            ConfigField(
                "crop_map",
                "toggle",
                "Crop map around agent + candidates",
                default=False,
            ),
            ConfigField(
                "draw_history",
                "toggle",
                "Draw past-trajectory dots on map",
                default=True,
            ),
            ConfigField(
                "draw_room_bounds",
                "toggle",
                "Draw grey rectangles around rooms",
                default=False,
            ),
            ConfigField(
                "draw_room_labels",
                "toggle",
                "Draw room-type text overlays",
                default=True,
            ),
            ConfigField(
                "draw_navigable_index",
                "toggle",
                "Label candidate dots with their index",
                default=False,
            ),
        ],
    )

    input_ports: ClassVar[list[PortDef]] = [
        PortDef("scan_id", "TEXT", "MP3D scan ID (e.g. '2t7WUuJeko7')"),
        PortDef(
            "position_json",
            "TEXT",
            "Agent 3D position as JSON [x, y, z] (MP3D sim coords, metres)",
        ),
        PortDef(
            "heading_deg",
            "TEXT",
            "Agent heading in degrees (float; 0 = north, positive = east)",
        ),
        PortDef(
            "elevation_deg",
            "TEXT",
            "Agent elevation in degrees (default '0')",
            optional=True,
        ),
        PortDef(
            "navigable_json",
            "TEXT",
            "Candidate viewpoints as JSON dict {vpid: {position, global_order?}}",
            optional=True,
        ),
        PortDef(
            "history_json",
            "TEXT",
            "History viewpoints as JSON dict {vpid: {position, history_order, within_same_level}}",
            optional=True,
        ),
    ]
    output_ports: ClassVar[list[PortDef]] = [
        PortDef("map_image", "IMAGE", "Top-down SSG map (PIL-compatible ndarray)"),
        PortDef("level_id", "TEXT", "Which floor/level the agent is currently on"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        config = getattr(self, "config", None) or {}
        scan_id = (inputs.get("scan_id") or "").strip()
        position_raw = inputs.get("position_json") or "[0,0,0]"
        heading_raw = inputs.get("heading_deg") or "0"
        elevation_raw = inputs.get("elevation_deg") or "0"
        navigable_raw = inputs.get("navigable_json") or "{}"
        history_raw = inputs.get("history_json") or "{}"

        if not scan_id:
            self._self_log("error", "scan_id is empty")
            return {"map_image": None, "level_id": ""}

        position = _parse_json(position_raw, default=[0.0, 0.0, 0.0])
        heading = float(heading_raw) if heading_raw else 0.0
        elevation = float(elevation_raw) if elevation_raw else 0.0
        navigable = _parse_json(navigable_raw, default={})
        history = _parse_json(history_raw, default={})

        manager = _get_manager(config)
        overrides = dict(
            grid_size=float(config.get("grid_size", 0.010)),
            agent_front_up=bool(config.get("agent_front_up", False)),
            rotation_strategy=str(config.get("rotation_strategy", "absolute")),
            crop_map=bool(config.get("crop_map", False)),
            draw_history=bool(config.get("draw_history", True)),
            draw_room_bounds=bool(config.get("draw_room_bounds", False)),
            draw_room_labels=bool(config.get("draw_room_labels", True)),
            draw_navigable_index=bool(config.get("draw_navigable_index", False)),
        )

        self._self_log("scan_id", scan_id)
        self._self_log("navigable_count", len(navigable))
        self._self_log("history_count", len(history))

        loop = asyncio.get_event_loop()

        def _render() -> tuple[Any, int]:
            # Mapper construction is heavy (reads category CSV, may assert on
            # missing files) — run inside the executor so its exceptions
            # surface through the same try/except as the rendering path.
            mapper = manager.get_mapper(overrides)
            pil_img = mapper.get_visual_map(
                scan_id,
                position,
                (heading, elevation),
                navigable_viewpoints=navigable or None,
                history_viewpoints=history or None,
            )
            level_id = mapper.prev_level_id if mapper.prev_level_id is not None else -1
            return pil_img, level_id

        try:
            pil_img, level_id = await loop.run_in_executor(None, _render)
        except Exception as exc:  # pragma: no cover — logs + empty return
            log.exception("SSG render_map failed for scan=%s: %s", scan_id, exc)
            self._self_log("error", f"{type(exc).__name__}: {exc}")
            return {"map_image": None, "level_id": ""}

        self._self_log("level_id", level_id)
        return {"map_image": pil_img, "level_id": str(level_id)}


# ──────────────────────────────────────────────────────────────────────
# Node 2 — ssg__reset_episode
# ──────────────────────────────────────────────────────────────────────


class SSGResetEpisodeNode(BaseCanvasNode):
    """Clear ``prev_level_id`` + optionally evict scan from object cache.

    Wire this in the Initialize path of the graph, fed from
    ``env_mp3d__reset.scan_id``. Mitigates the upstream footgun where
    ``SceneGraphMapper.prev_level_id`` persists across episodes and
    produces stale level IDs for staircase scenes.
    """

    node_type = "ssg__reset_episode"
    display_name = "SSG: Reset Episode"
    description = (
        "Per-episode reset for SceneGraphMapper.prev_level_id and "
        "SceneObjectGraph state (footgun mitigation)"
    )
    category = "processing"
    icon = "RotateCcw"

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="slate",
        config_fields=[
            ConfigField(
                "evict_scan_from_room_cache",
                "toggle",
                "Also evict this scan's room_cache entries (fresher, slower)",
                default=False,
            ),
        ],
    )

    input_ports: ClassVar[list[PortDef]] = [
        PortDef("scan_id", "TEXT", "Current episode's scan ID"),
        PortDef(
            "trigger",
            "ANY",
            "Optional upstream signal to sequence after (e.g. reset.done)",
            optional=True,
        ),
    ]
    output_ports: ClassVar[list[PortDef]] = [
        PortDef("done", "BOOL", "True after reset applied"),
        PortDef("scan_id_out", "TEXT", "Pass-through scan_id"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        config = getattr(self, "config", None) or {}
        scan_id = (inputs.get("scan_id") or "").strip()
        evict = bool(config.get("evict_scan_from_room_cache", False))

        global _manager
        if _manager is not None:
            _manager.reset_episode(scan_id, evict_scan_from_room_cache=evict)
            self._self_log("reset", "applied")
        else:
            # Manager uninitialised — no state to clear yet; still emit done=True
            # so the Initialize path doesn't block.
            self._self_log("reset", "manager not initialised (no-op)")
        return {"done": True, "scan_id_out": scan_id}


# ──────────────────────────────────────────────────────────────────────
# Node 3 — ssg__query_objects
# ──────────────────────────────────────────────────────────────────────


class SSGQueryObjectsNode(BaseCanvasNode):
    """Return SSG-local objects + instruction-aligned interest objects.

    Wraps ``SceneObjectGraph.get_local_objects`` (for objects within
    ``visibility_radius`` of the agent's viewpoint) and
    ``get_interest_objects`` (filtered by the natural-language
    instruction via category matching + SentenceTransformer cosine
    similarity).

    First call per ``data_name`` loads SentenceTransformer +
    ``{data_name}_category.jsonl`` (optional; if jsonl missing,
    interest_objects returns empty).
    """

    node_type = "ssg__query_objects"
    display_name = "SSG: Query Objects"
    description = (
        "Nearby objects + instruction-aligned interest objects from the SpatialNav SceneObjectGraph"
    )
    category = "perception"
    icon = "Search"

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="teal",
        config_fields=[
            ConfigField(
                "data_name",
                "select",
                "Dataset (determines which category jsonl is loaded)",
                options=[
                    {"value": "R2R", "label": "R2R"},
                    {"value": "REVERIE", "label": "REVERIE"},
                ],
                default="R2R",
            ),
            ConfigField(
                "visibility_radius",
                "slider",
                "Local visibility radius (metres)",
                default=3.0,
                min=0.5,
                max=10.0,
                step=0.5,
            ),
            ConfigField(
                "mp3d_data_dir",
                "text",
                "MP3D data dir (shared with render_map; pinned on first use)",
                default="data/scene_datasets/mp3d/v1/tasks/mp3d",
            ),
            ConfigField(
                "category_file",
                "text",
                "MP3D category mapping CSV",
                default="data/scene_datasets/mp3d/v1/tasks/mp3d/category_spatiallm_mapping.csv",
            ),
            ConfigField(
                "st_model_path",
                "text",
                "SentenceTransformer model path or HF identifier",
                default="sentence-transformers/all-MiniLM-L6-v2",
            ),
            ConfigField(
                "category_jsonl_dir",
                "text",
                "Directory containing {R2R,REVERIE}_category.jsonl",
                default="data/tasks",
            ),
            ConfigField(
                "max_cache_size",
                "slider",
                "LRU bound on per-viewpoint object cache",
                default=500,
                min=50,
                max=5000,
                step=50,
            ),
        ],
    )

    input_ports: ClassVar[list[PortDef]] = [
        PortDef("scan_id", "TEXT", "MP3D scan ID"),
        PortDef("viewpoint_id", "TEXT", "Current viewpoint ID (cache key)"),
        PortDef("position_json", "TEXT", "Agent 3D position as JSON [x, y, z]"),
        PortDef(
            "path_id",
            "TEXT",
            "Episode path_id (for interest-object lookup); empty disables interest query",
            optional=True,
        ),
        PortDef(
            "instruction",
            "TEXT",
            "Instruction text (for interest-object filtering)",
            optional=True,
        ),
    ]
    output_ports: ClassVar[list[PortDef]] = [
        PortDef(
            "local_objects_json",
            "TEXT",
            "JSON list of {id, category, obb_center, distance} within visibility_radius",
        ),
        PortDef(
            "interest_objects_json",
            "TEXT",
            "JSON list of instruction-relevant category names (may be empty)",
        ),
        PortDef("room_type", "TEXT", "Room type at agent's current position (e.g. 'bedroom')"),
        PortDef("level_id", "TEXT", "Current level/floor ID as string"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        config = getattr(self, "config", None) or {}
        scan_id = (inputs.get("scan_id") or "").strip()
        viewpoint_id = (inputs.get("viewpoint_id") or "").strip()
        position = _parse_json(inputs.get("position_json") or "[0,0,0]", default=[0.0, 0.0, 0.0])
        path_id = (inputs.get("path_id") or "").strip()
        instruction = (inputs.get("instruction") or "").lower()

        data_name = str(config.get("data_name", "R2R")).upper()
        visibility_radius = float(config.get("visibility_radius", 3.0))

        if not scan_id or not viewpoint_id:
            self._self_log("error", "scan_id or viewpoint_id missing")
            return {
                "local_objects_json": "[]",
                "interest_objects_json": "[]",
                "room_type": "",
                "level_id": "",
            }

        manager = _get_manager(config)
        loop = asyncio.get_event_loop()

        def _query() -> dict:
            graph = manager.get_agraph(data_name, visibility_radius)
            local_info = graph.get_local_objects(
                scan_id=scan_id,
                viewpoint_id=viewpoint_id,
                viewpoint_pos=position,
                visibility_radius=visibility_radius,
            )
            interest: list[str] = []
            if path_id and instruction:
                try:
                    interest = graph.get_interest_objects(path_id, instruction) or []
                except Exception as exc:
                    log.warning("SSG get_interest_objects failed: %s", exc)
            local_objects = [
                {
                    "id": getattr(obj, "id", None),
                    "category": getattr(obj, "category", "unknown"),
                }
                for obj in local_info.get("local_objects", [])
            ]
            return {
                "local_objects_json": json.dumps(local_objects, ensure_ascii=False),
                "interest_objects_json": json.dumps(interest, ensure_ascii=False),
                "room_type": local_info.get("room_type") or "",
                "level_id": str(
                    local_info.get("level_id") if local_info.get("level_id") is not None else ""
                ),
            }

        try:
            out = await loop.run_in_executor(None, _query)
        except Exception as exc:  # pragma: no cover — logs + empty return
            log.exception(
                "SSG query_objects failed for scan=%s vpid=%s: %s", scan_id, viewpoint_id, exc
            )
            self._self_log("error", f"{type(exc).__name__}: {exc}")
            return {
                "local_objects_json": "[]",
                "interest_objects_json": "[]",
                "room_type": "",
                "level_id": "",
            }

        self._self_log("local_count", len(json.loads(out["local_objects_json"])))
        self._self_log("interest_count", len(json.loads(out["interest_objects_json"])))
        return out


# ──────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────


def _parse_json(raw: Any, *, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


# ──────────────────────────────────────────────────────────────────────
# NodeSet
# ──────────────────────────────────────────────────────────────────────


class SSGNodeSet(BaseNodeSet):
    """Spatial Scene Graph — SpatialNav port (M1).

    Three nodes for agent-centric top-down map rendering + nearby /
    interest object queries over MP3D ground-truth layout + semantic
    point cloud. All nodes share a process-wide singleton manager with
    per-scan lazy loading and per-episode reset.

    In-process (Python 3.10) — no server mode required. Depends on
    pyntcloud, sentence-transformers, matplotlib, opencv-python,
    shapely (added to the agentcanvas conda env during M1 port).

    Load:
        POST /api/components/nodesets/ssg/load

    Data expectations (all via ConfigField, absolute paths recommended):
        - {mp3d_data_dir}/{scan}/{scan}.house            (MP3D layout)
        - {mp3d_data_dir}/{scan}/{scan}_semantic.ply     (semantic pc)
        - {category_file}                                 (CSV)
        - {st_model_path} or HF identifier                (SentenceTransformer)
        - {category_jsonl_dir}/{data_name}_category.jsonl (optional)
    """

    name = "ssg"
    display_name = "Spatial Scene Graph"
    description = (
        "SpatialNav SSG: agent-centric top-down map + object retrieval "
        "(MP3D ground-truth layout + semantic point cloud)"
    )

    def get_tools(self) -> list:
        return [
            SSGRenderMapNode(),
            SSGResetEpisodeNode(),
            SSGQueryObjectsNode(),
        ]

    async def initialize(self) -> None:
        log.info("SSG nodeset initialised (SceneGraphMapper + SceneObjectGraph load on first use)")

    async def shutdown(self) -> None:
        global _manager
        if _manager is not None:
            log.info("SSG nodeset shutdown — dropping mapper + agraph singletons")
            _manager = None
