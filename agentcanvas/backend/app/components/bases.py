"""Base classes for class-based agent component definitions.

Authors place subclasses in ``workspace/`` subdirectories.
The :class:`WorkspaceComponentRegistry` discovers them at startup (or on reload)
and bridges each instance into the runtime registries.

Import convention for ``workspace`` files::

    from app.components import BaseCanvasNode, BaseNodeSet, PortDef
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..graph_def import ContainerDef, GraphDefinition

# ── Port Definition ──


@dataclass
class PortDef:
    """Declares an input or output port on a canvas node.

    The ``name`` becomes the React Flow Handle ``id`` on the frontend
    and the key in the ``inputs`` / return dict on the backend.
    """

    name: str  # port name (= Handle id)
    wire_type: str  # "IMAGE", "ACTION", "TEXT", etc.
    description: str = ""
    optional: bool = False  # if False, node won't fire until this port has data


# ── Node UI Config ──


@dataclass
class ConfigField:
    """Declares an inline config control rendered on the node UI.

    The ``field_type`` determines the widget:
    - ``"label"``:    Read-only ``label: value`` text.
    - ``"slider"``:   Range input (uses ``min``, ``max``, ``step``).
    - ``"text"``:     Single-line text input.
    - ``"select"``:   Dropdown (uses ``options``).
    - ``"toggle"``:   Checkbox / switch.
    - ``"textarea"``: Multi-line text input.
    """

    name: str  # config key (e.g. "temperature")
    field_type: str  # "slider" | "text" | "select" | "toggle" | "textarea" | "label"
    label: str = ""  # display label
    default: Any = None  # default value
    min: float | None = None  # slider min
    max: float | None = None  # slider max
    step: float | None = None  # slider step
    options: list[dict] | None = None  # select: [{"value": "x", "label": "X"}]
    placeholder: str = ""  # text / textarea placeholder
    show_persist_toggle: bool = (
        False  # port_list only: render a per-port persist checkbox (iterIn ports)
    )


@dataclass
class DisplayField:
    """Read-only runtime data display widget rendered on the node UI.

    Unlike ``ConfigField`` (user-editable controls), display fields show
    live data from WebSocket ``nodeOutputs``.

    The ``display_type`` determines the widget:
    - ``"image_viewer"``:  Renders base64-encoded image as ``<img>`` tag.
    - ``"log_list"``:      Scrollable list of log entries with step numbers.
    - ``"metric_table"``:  Key-value table for metrics (SPL, SR, etc.).
    """

    name: str  # unique field ID
    display_type: str  # "image_viewer" | "log_list" | "metric_table"
    label: str = ""  # display label
    data_key: str = ""  # viewer_data.fields sub-field (matches port name)
    max_visible: int = 10  # max entries for log_list
    accumulate: bool = False  # True = append to array; False = replace


@dataclass
class NodeUIConfig:
    """Visual configuration for GenericBlockRenderer.

    Every ``BaseCanvasNode`` has a ``ui_config`` ClassVar with sensible
    defaults.  Override to customise how the node appears on the canvas
    without writing a custom ``.tsx`` component.

    Attributes:
        color:         Tailwind colour key (e.g. ``"amber"``).  Empty
                       string means auto-derive from ``category``.
        layout:        ``"block"`` (standard rectangle) or ``"strip"``
                       (narrow vertical gate like IterIn).
        width:         Explicit width for strip nodes (e.g. ``"44px"``).
        min_height:    Explicit min-height (e.g. ``"140px"``).
        rounding:      CSS rounding class (e.g. ``"rounded-r-lg"``).
        config_fields: Inline config controls rendered on the node UI.
        display_fields: Read-only runtime data display widgets.
    """

    color: str = ""
    layout: str = "block"
    width: str = ""
    min_height: str = ""
    rounding: str = ""
    min_width: str = ""
    max_width: str = ""
    config_fields: list[ConfigField] = field(default_factory=list)
    display_fields: list[DisplayField] = field(default_factory=list)


# ── Canvas Nodes (BASE — must come first, other classes inherit) ──


class BaseCanvasNode(ABC):
    """Universal base class for everything on the canvas (ADR-007).

    All canvas elements — blocks (atomic functions), composites (nested
    graphs), and control nodes (iteration/boundary) — inherit from this
    class.  Tools, skills, policies, and agents also inherit from it,
    making them first-class canvas citizens.

    The GraphExecutor treats all subclasses identically at runtime.

    Subclass attributes — Identity:
        node_type:    Unique type key used in graph JSON (e.g. ``"envStep"``).
        display_name: Human label for the canvas sidebar.
        description:  What this node does (tooltip, docs, search).
        category:     Grouping in the sidebar (e.g. ``"environment"``).
        icon:         Lucide icon name for frontend rendering.

    Subclass attributes — Kind:
        kind:         ``"block"`` (atomic function), ``"composite"`` (has
                      children graph), or ``"control"`` (structural mechanics).

    Subclass attributes — Ports:
        input_ports:  List of :class:`PortDef` describing accepted inputs.
        output_ports: List of :class:`PortDef` describing produced outputs.

    Subclass attributes — Children:
        children:     ``GraphDefinition | None`` — subgraph for composites.
                      ``None`` for blocks and controls.

    Subclass attributes — Config:
        config_schema:  JSON Schema dict for the node's config panel.
        default_config: Default config values when dropped onto canvas.

    Implement :meth:`forward` with the node's logic.
    """

    # ── Identity ──
    node_type: ClassVar[str]
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = "custom"
    icon: ClassVar[str] = ""

    # ── Kind ──
    kind: ClassVar[str] = "block"  # "block" | "composite" | "control"

    # ── Ports ──
    input_ports: ClassVar[list[PortDef]] = []
    output_ports: ClassVar[list[PortDef]] = []

    # ── Children (nested graph for composites) ──
    children: ClassVar[GraphDefinition | None] = None

    # ── Config ──
    config_schema: ClassVar[dict] = {}
    default_config: ClassVar[dict] = {}

    # ── UI (read by GenericBlockRenderer — no .tsx needed) ──
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig()

    # ── Batched inference (ADR-028 PC-1) ──
    # Opt-in: when True, the registry replaces this node's in-process handler
    # with a ``BatchedClient`` keyed by ``(node.id, config_hash)``. K worker
    # callers rendezvous in ``BatchedInferenceServer``; the underlying
    # ``forward()`` is invoked once per flush with the per-sample inputs
    # stacked along ``batch_dim``. Server is pure-functional — any per-call
    # state (e.g. RNN hidden states) must live on the wire as explicit
    # input/output ports.
    batched: ClassVar[bool] = False
    batch_dim: ClassVar[str] = ""  # name of the input port carrying the batch axis

    # After-loop nodes — nodes that run once AFTER the dataflow loop
    # (run-end verdict, or a genuine post-loop reasoning stage) — are not
    # declared by any flag: they are simply the nodes wired downstream of
    # the loop iterOut's ``final_*`` handles, fed exactly once at scope
    # termination. See ``GraphExecutor._emit_final_side`` /
    # ``_after_loop_pass``. (History: a ``final_fire`` ClassVar was retired
    # 2026-05-21 for a per-node ``config.post_loop`` flag, itself retired
    # 2026-06-11 when the final side made after-loop membership derivable.)

    def __init__(self) -> None:
        # Instance attributes set by the executor at graph-build time
        self.config: dict = {}  # per-node config from graph JSON
        self.node_id: str = ""  # node ID from graph JSON
        self._log_buffer: list[dict] = []  # voluntary inner log (per-firing)

    def _self_log(self, key: str, value: Any) -> None:
        """Record an internal detail during ``forward()``.

        Call this inside your ``forward()`` to log domain-specific data that
        the executor cannot see from the outside (e.g. assembled LLM prompts,
        API responses, token counts).  The executor collects these entries via
        :meth:`log` after ``forward()`` returns.
        """
        self._log_buffer.append({"key": key, "value": value})

    def log(self) -> list[dict]:
        """Return voluntary log entries from the last ``forward()`` call.

        Called by the executor after ``forward()`` completes.  Override to
        filter, transform, or add computed summaries.  The default returns
        the raw ``_self_log()`` buffer.
        """
        return self._log_buffer

    @abstractmethod
    async def forward(self, inputs: dict, ctx: Any) -> dict:
        """Execute the node.

        Args:
            inputs:  ``{port_name: value}`` for each filled input port.
            ctx:     Per-node state proxy (reads/writes persistent state).

        Returns:
            ``{port_name: value}`` for each output port.
        """
        ...

    # ── Lifecycle ──

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        """Return (input_ports, output_ports) for a node given its instance config.

        Default: returns class-level ports. Subclasses that support dynamic ports
        (e.g. IterIn/IterOut with config.ports) override this.
        """
        return (cls.input_ports, cls.output_ports)

    async def initialize(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Called once before first execution. Override for setup (GPU, models)."""

    async def shutdown(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Called on teardown. Override for cleanup."""


# ── Tools ──


## BaseTool removed — all nodeset nodes now subclass BaseCanvasNode directly
## with explicit node_type, input_ports, output_ports.


# ── Dynamic Fire-List (C.5) ──
#
# A spawner node's ``forward()`` returns a ``FireList`` instead of the usual
# ``{port_name: value}`` dict. The GraphExecutor recognises the sentinel
# type, suppresses normal edge-propagation, and fires each ``FireSpec`` in
# order against the live ``NODE_HANDLERS`` registry. After children complete,
# the spawner's :meth:`DynamicFireListNode.aggregate` collapses their results
# into the spawner's declared output ports, which then flow on normal wires.
#
# Children are ephemeral — they never enter ``self.nodes`` / ``adjacency`` /
# ``scope_forest`` / ``ready_queue``. They cannot be loop pivots, cannot sit
# on the final side, cannot have their own scope. These constraints are what makes
# C.5 small (~300 LOC engine) while still covering linear dynamic call
# sequences like VoxPoser's composer → LMPs → execute chain. See
# ``.claude/plans/foamy-kindling-puppy.md`` for the full design.


@dataclass
class FireSpec:
    """One dynamic child firing requested by a spawner node's ``forward()``."""

    node_type: str  # resolved via NODE_HANDLERS at fire time
    inputs: dict[str, Any] = field(default_factory=dict)  # → child's pending_inputs
    config: dict[str, Any] = field(default_factory=dict)
    label: str = ""  # human label for log / trace
    capture_outputs: list[str] | None = None
    # ↑ if set, only these output port keys survive into the FireList result
    #   (whitelist); default = all keys returned by the child's forward().


@dataclass
class FireList:
    """Sentinel returned by ``DynamicFireListNode.forward()`` instead of a dict.

    The executor recognises ``isinstance(result, FireList)`` and dispatches
    each spec sequentially. After all children complete, the spawner's
    :meth:`DynamicFireListNode.aggregate` is called with the ordered list of
    child output dicts; its return value is merged with :attr:`spawner_outputs`
    and propagated through the spawner's normal output wires.

    For server-mode spawners (where ``forward()`` runs in a subprocess and
    the framework holds only a proxy class without the original
    ``aggregate()`` method), :attr:`aggregator` carries a declarative recipe
    the engine applies instead. See ``_apply_declarative_aggregator``.
    """

    specs: list[FireSpec] = field(default_factory=list)
    spawner_outputs: dict[str, Any] = field(default_factory=dict)
    # ↑ values the spawner emits ALONGSIDE the children (e.g. raw composer
    #   text, captured plan dump, metrics). These flow on the spawner's
    #   declared output ports just like a normal node's return dict, MERGED
    #   with the result of ``aggregate(child_results)``. On key collision,
    #   ``aggregate`` wins (the children-derived values).
    aggregator: dict[str, Any] = field(default_factory=dict)
    # ↑ Declarative aggregation recipe — engine uses this when set, falling
    #   back to ``instance.aggregate(child_results)`` only when empty.
    #   Supported shapes:
    #     {"kind": "passthrough_last"}                  — output = last child's dict
    #     {"kind": "passthrough_index", "index": N}     — output = child[N]'s dict
    #     {"kind": "merge_all"}                         — output = merge of all children
    #     {"kind": "rename", "from": "k1", "to": "k2"}  — wrap a passthrough kind
    #                                                     under nested ``inner: {...}``
    #   {} (default) = no declarative recipe; engine calls instance.aggregate.
    #   Designed so server-mode proxies (no aggregate() method) can still
    #   collapse children's results without a second HTTP round-trip to the
    #   spawner. Local DynamicFireListNode subclasses can keep their
    #   ``aggregate()`` method and leave this empty.


class DynamicFireListNode(BaseCanvasNode):
    """Spawner node that returns a :class:`FireList` from ``forward()``.

    The engine fires each ``FireSpec`` sequentially against the live node
    registry (same lookup as the main dataflow loop). Children are ephemeral
    — they appear in the log with ``parent_node_id`` set to this node's id,
    but they are not in the static graph topology and never enter the
    dataflow ready_queue.

    Subclass contract:
        * Override :meth:`forward` to return a ``FireList``. The engine
          asserts on the return type; returning a plain ``dict`` from a
          ``DynamicFireListNode`` is a programmer error.
        * Override :meth:`aggregate` to collapse the children's output dicts
          (in firing order) into this node's declared output ports.

    Children fire SEQUENTIALLY. A child's outputs are visible to subsequent
    children only via this spawner's runtime / container state, NOT via
    auto-wiring (the engine does not propagate child→child).

    Nested ``FireList`` (a child itself returning ``FireList``) is rejected
    by the engine in C.5; the dispatcher raises ``NotImplementedError``.
    Upgrading to full nested-subgraph support (option "C" in the design
    discussion) is a separate, later workstream.
    """

    # ``kind = "block"`` because the spawner is, from the topology / scope
    # analyzer's POV, a regular atomic block — it doesn't open a new scope
    # or manipulate iter machinery. The dynamism is purely a per-firing
    # detail and never reaches the static scope_forest.
    kind: ClassVar[str] = "block"

    @abstractmethod
    async def forward(self, inputs: dict, ctx: Any) -> FireList:  # type: ignore[override]
        """Return a :class:`FireList` describing the dynamic child firings.

        The engine will then sequentially fire each spec, collect their
        output dicts in order, and pass them to :meth:`aggregate` — UNLESS
        the returned FireList carries a declarative ``aggregator`` recipe,
        in which case the engine applies that recipe instead of calling
        ``aggregate``.
        """
        ...

    def aggregate(self, child_results: list[dict]) -> dict:
        """Collapse children's output dicts into this node's output ports.

        Called once after all FireSpecs complete IFF the FireList does not
        carry a declarative ``aggregator`` recipe. ``child_results[i]`` is
        the dict returned by the i-th child's ``forward()`` (filtered by
        that spec's ``capture_outputs`` if set). For an empty FireList,
        ``child_results == []``.

        Return a ``{port_name: value}`` dict matching this node's declared
        ``output_ports``. The engine merges this on top of ``spawner_outputs``
        from the returned ``FireList``, then propagates the merged dict
        through normal output wires.

        Default implementation raises ``NotImplementedError`` — subclasses
        must either override this OR set ``FireList.aggregator`` to a
        declarative recipe in their ``forward()`` return value. The latter
        is the only path that works for server-mode spawners, where the
        framework-side proxy class does NOT inherit this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.aggregate() not overridden and the "
            f"returned FireList carries no declarative aggregator. Either "
            f"implement aggregate(self, child_results) → dict, or set "
            f"FireList.aggregator (e.g. {{'kind': 'merge_all'}}) in forward()."
        )


# ── NodeSets ──


def conda_env_python(env_name: str, env_var: str) -> str | None:
    """Resolve a named conda env's interpreter for ``server_python``.

    Resolution order:
      1. ``$<env_var>`` when set — explicit override, returned as-is.
      2. ``<conda root>/envs/<env_name>/bin/python`` derived from the
         running interpreter, so the same nodeset file resolves on any
         machine regardless of where conda lives.

    Returns ``None`` when neither resolves; the nodeset then loads in
    local mode (``sys.executable``). Nodesets that hard-require a
    dedicated env should name their ``scripts/install/install_*.sh``
    in the class docstring so the eventual import error is actionable.
    """
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit
    prefix = Path(sys.prefix)
    # Running inside <root>/envs/<name> → root is two levels up; running
    # inside the base env → envs/ is a direct child.
    root = prefix.parent.parent if prefix.parent.name == "envs" else prefix
    candidate = root / "envs" / env_name / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return None


class BaseNodeSet(ABC):
    """A loadable group of domain-specific canvas nodes (tools, environments, skills) that share initialization and shutdown lifecycle. Not to be confused with arbitrary node collections — a NodeSet is a deployment unit.

    Subclass attributes:
        name:        Unique nodeset identifier (e.g. ``"slam"``).
        description: One-line summary.

    Implement :meth:`get_tools` to return all node instances in this set.
    Override :meth:`initialize` / :meth:`shutdown` for systems that need
    heavyweight setup (GPU memory, model loading, ROS connections, etc.).
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    server_python: ClassVar[str | None] = (
        None  # Python interpreter for server mode; None = sys.executable
    )
    # Parallelism contract (ADR-server-003). At eval worker_count > 1:
    #   "shared"     → 1 instance, K callers may rendezvous through
    #                  BatchedInferenceServer (stateless tools).
    #   "replicated" → N independent tagged subprocesses, one per worker
    #                  (stateful — env scene state, simulator handles).
    # Default "shared" keeps non-env nodesets singleton; env nodesets must
    # opt in to "replicated".
    parallelism: ClassVar[Literal["shared", "replicated"]] = "shared"
    # Optional control-plane panel (BaseEnvPanel subclass). When set, the
    # WorkspaceComponentRegistry instantiates and registers it on load. Imported lazily
    # to avoid a circular import with app.components.env_panel.
    env_panel: ClassVar[type | None] = None
    # Per-nodeset default for the eval per-episode timeout (ADR-028). The
    # batch runner clamps each episode at ``max_steps * per_step_budget_sec``.
    # Override on env/policy nodesets whose step latency diverges from the
    # framework default. Set generously (30s) so per-step overhead from
    # shared-singleton VLM contention under high worker_count doesn't burn
    # the wall-clock before num_step is reached.
    default_per_step_budget_sec: ClassVar[float] = 30.0
    # Optional path to a sibling .py file declaring a
    # :class:`BaseReplayParser` subclass — relative to this nodeset's
    # source file. Set on env nodesets that support log replay. The
    # parser module must be importable from the main FastAPI process
    # (i.e. no simulator imports — pure log-walking code).
    replay_parser: ClassVar[str | None] = None

    @abstractmethod
    def get_tools(self) -> list:
        """Return all node instances provided by this nodeset."""
        ...

    def get_containers(self) -> list[ContainerDef]:
        """Return state containers **owned** by this nodeset (nodeset-level).

        Unlike graph-level (home) containers — authored per-graph and built in
        the executor process — a nodeset-owned container lives in **this
        nodeset's process** (its own subprocess in server mode) and is shared
        **by reference** by the nodeset's own nodes via ``ctx.containers[id]``.
        It is never serialized across the process boundary; only a read-only
        preview is surfaced for display. Override to declare owned state;
        default: none.

        See ``app/agent_loop/state_containers.py`` for the runtime container
        and ``ContainerDef``/``StateDef`` (``app/graph_def.py``) for the shape.
        """
        return []

    async def initialize(self, **kwargs: Any) -> None:  # noqa: B027 — optional lifecycle hook
        """Initialize the system before tools are used.

        Called by the WorkspaceComponentRegistry after scanning. Override for systems
        that need heavyweight setup (e.g. loading a neural network, connecting
        to a ROS node, allocating GPU memory).

        Default: no-op.
        """

    async def shutdown(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Shutdown the system and release resources.

        Called during component unregister or app shutdown. Override to clean
        up GPU memory, close connections, etc.

        Default: no-op.
        """

    async def get_eval_metadata(self) -> dict:
        """Return eval-relevant metadata for batch evaluation.

        Env nodesets should override this to expose their capabilities.
        Non-env nodesets return an empty dict (default).

        Returns a dict with keys:
            env_name:             Human-readable env name (e.g. "habitat_vlnce")
            splits:               Available dataset splits (e.g. ["val_unseen", "val_seen"])
            episode_counts:       Episodes per split (only after init; empty dict before)
            metrics:              Metric names produced (e.g. ["spl", "success", "ndtw"])
            supports_set_episode: Whether set_episode_by_index() is available
            step_budget:          Per-episode iteration cap (the framework's
                                 resolver chain in eval_batch reads this
                                 after each episode reset to override the
                                 graph-level default).
        """
        return {}


# ── Skills ──


## BaseSkill removed — skills are now BaseCanvasNode subclasses in
## vln_skills nodeset (workspace/nodesets/vln_skills.py).


## BaseAgent removed — agent definitions were purely declarative config.
## Agent manifests can be defined as JSON/YAML in workspace/agent/ if needed.


## BasePolicy removed — policy definitions are now direct BaseCanvasNode
## subclasses with policy_id, name, checkpoint, config_path attributes in workspace/nodesets/.
