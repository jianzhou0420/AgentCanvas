"""Tests for C.5 Dynamic Fire-List dispatch.

Validates that a ``DynamicFireListNode`` spawner can return a ``FireList``
from ``execute()`` and the engine sequentially fires each ``FireSpec``,
aggregates the results, merges with ``spawner_outputs``, and propagates the
final dict through normal output wires.

Covers:
  * sequential order preserved
  * empty FireList → aggregate([]) → spawner_outputs propagate
  * child error stops sequence, surfaces as spawner error
  * spawner_outputs + aggregate co-exist, aggregate wins on collision
  * nested FireList (child returns one) rejected as NotImplementedError
  * forbidden child types (control / boundary) rejected
  * spawner inside iterIn/iterOut loop fires fresh children every outer iter
  * capture_outputs whitelist filters child output dicts
  * access-grant inheritance: child sees the spawner's containers

These tests bypass the env layer entirely — test nodes are tiny
``BaseCanvasNode`` / ``DynamicFireListNode`` subclasses registered into the
global handler registry at import time.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from typing import Any

from ..components.bases import (
    BaseCanvasNode,
    DynamicFireListNode,
    FireList,
    FireSpec,
    PortDef,
)
from ..graph_def import (
    AccessGrantDef,
    ContainerDef,
    EdgeDef,
    GraphDefinition,
    NodeDef,
)
from .builtin_nodes import register_node
from .graph_executor import GraphExecutor

# ── test nodes ────────────────────────────────────────────────────────────


class _SeedNode(BaseCanvasNode):
    """Fires once at run start to kick off downstream wiring."""

    node_type = "_test_dyn_seed"
    input_ports = []
    output_ports = [PortDef("value", "ANY")]

    async def forward(self, inputs, ctx):
        return {"value": 1}


class _RecorderChild(BaseCanvasNode):
    """Child node — records its fire on a shared list pulled from config."""

    node_type = "_test_dyn_recorder_child"
    input_ports = [PortDef("payload", "ANY", optional=True)]
    output_ports = [PortDef("echo", "ANY"), PortDef("extra", "ANY")]

    # Shared cross-test record bucket — keyed by spawner id so parallel
    # tests don't bleed. Test cleans its key before exercising.
    BUCKETS: dict[str, list] = {}

    async def forward(self, inputs, ctx):
        bucket_key = self.config.get("bucket_key", "_default")
        bucket = _RecorderChild.BUCKETS.setdefault(bucket_key, [])
        record = {
            "config": dict(self.config),
            "inputs": dict(inputs),
            "node_id": self.node_id,
        }
        bucket.append(record)
        return {
            "echo": inputs.get("payload"),
            "extra": self.config.get("extra_value", None),
        }


class _RaisingChild(BaseCanvasNode):
    """Child node that always raises — used to test sequence stops on error."""

    node_type = "_test_dyn_raising_child"
    input_ports = []
    output_ports = [PortDef("ok", "ANY")]

    async def forward(self, inputs, ctx):
        raise RuntimeError("intentional child failure for test")


class _NestedFireListChild(BaseCanvasNode):
    """Child that itself returns a FireList — should be rejected by engine."""

    node_type = "_test_dyn_nested_firelist_child"
    input_ports = []
    output_ports = [PortDef("ok", "ANY")]

    async def forward(self, inputs, ctx):
        return FireList(
            specs=[
                FireSpec(node_type="_test_dyn_recorder_child", inputs={}),
            ]
        )


class _ContainerProbeChild(BaseCanvasNode):
    """Child that asserts on its container access (proves grant inheritance)."""

    node_type = "_test_dyn_container_probe_child"
    input_ports = []
    output_ports = [PortDef("ok", "BOOL"), PortDef("container_keys", "ANY")]

    async def forward(self, inputs, ctx):
        containers = getattr(ctx, "_containers", None) or {}
        return {"ok": bool(containers), "container_keys": sorted(containers.keys())}


class _SpawnerNode(DynamicFireListNode):
    """Test spawner — emits a FireList shaped by its config.

    Config keys:
      ``specs``: list of dicts ``{node_type, inputs, config?, label?,
                  capture_outputs?}``
      ``spawner_outputs``: dict of direct spawner output values
      ``aggregate_override``: dict to return verbatim from aggregate()
                              (else aggregate returns ``{"agg": [...]}``)
      ``raise_in_aggregate``: bool — raise inside aggregate() to test
                              error path through the aggregate stage
    """

    node_type = "_test_dyn_spawner"
    input_ports = [PortDef("trigger", "ANY", optional=True)]
    output_ports = [
        PortDef("agg", "ANY"),
        PortDef("subtask_text", "TEXT"),
        PortDef("done", "BOOL"),
    ]

    async def forward(self, inputs, ctx) -> FireList:
        specs_cfg = self.config.get("specs", [])
        specs = [
            FireSpec(
                node_type=s["node_type"],
                inputs=dict(s.get("inputs", {})),
                config=dict(s.get("config", {})),
                label=s.get("label", ""),
                capture_outputs=s.get("capture_outputs"),
            )
            for s in specs_cfg
        ]
        return FireList(
            specs=specs,
            spawner_outputs=dict(self.config.get("spawner_outputs", {})),
        )

    def aggregate(self, child_results: list[dict]) -> dict:
        if self.config.get("raise_in_aggregate"):
            raise RuntimeError("intentional aggregate failure for test")
        override = self.config.get("aggregate_override")
        if override is not None:
            return dict(override)
        return {"agg": list(child_results)}


class _SinkNode(BaseCanvasNode):
    """Captures a value into state for assertion."""

    node_type = "_test_dyn_sink"
    input_ports = [PortDef("value", "ANY")]
    output_ports = [PortDef("done", "BOOL")]

    async def forward(self, inputs, ctx):
        ctx.captured = inputs.get("value")
        ctx.captured_history = (ctx.captured_history or []) + [inputs.get("value")]
        return {"done": True}


# Register all test nodes once at import time
for _cls in (
    _SeedNode,
    _RecorderChild,
    _RaisingChild,
    _NestedFireListChild,
    _ContainerProbeChild,
    _SpawnerNode,
    _SinkNode,
):
    register_node(_cls)


# ── session + runner helpers (mirrors test_executor_scopes patterns) ──────


class _StubSession:
    _status = "idle"
    _current_step = 0
    _metrics: dict | None = None
    _execution_id = "test"
    principles = None

    def _ws(self, msg_type: str, data: Any = None):
        from ..models import WSMessage

        return WSMessage(type=msg_type, data=data, execution_id=self._execution_id)


def _node(id: str, type: str, label: str = "", **config) -> NodeDef:
    return NodeDef(id=id, type=type, label=label or id, config=dict(config))


def _edge(id: str, source: str, target: str, sh: str = "value", th: str = "trigger") -> EdgeDef:
    return EdgeDef(id=id, source=source, target=target, sourceHandle=sh, targetHandle=th)


@contextmanager
def _strict_errors():
    """Re-raise node failures up through the main loop (else the executor
    quietly converts them to ``{"error": ...}`` and continues). Required by
    the error-path tests below to verify that a spawner / child failure is
    surfaced rather than silently swallowed.
    """
    prev = os.environ.get("AGENTCANVAS_STRICT_ERRORS")
    os.environ["AGENTCANVAS_STRICT_ERRORS"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AGENTCANVAS_STRICT_ERRORS", None)
        else:
            os.environ["AGENTCANVAS_STRICT_ERRORS"] = prev


def _run(graph: GraphDefinition, step_budget: int = 50) -> tuple[GraphExecutor, _StubSession]:
    """Drive ``graph`` to completion or error; return ``(executor, session)``.

    ``GraphExecutor.run`` catches all node-firing exceptions internally
    and flips ``session._status`` to ``"error"``, so tests that exercise
    error paths must inspect the session afterwards rather than
    ``pytest.raises``-ing on ``_run``.
    """
    from ..graph_def import _synthesize_iterin_ports

    _synthesize_iterin_ports(graph)
    if graph.step_budget is None:
        graph.step_budget = step_budget
    exe = GraphExecutor()
    sess = _StubSession()
    asyncio.run(exe.run(graph, sess, step_delay_ms=0))
    return exe, sess


def _minimal_spawn_to_sink_graph(spawner_config: dict, *, bucket_key: str) -> GraphDefinition:
    """seed → spawner → sink, with spawner emitting one of its declared
    output ports onto the sink. The sink captures the value so the test
    can read it back from state.

    The bucket_key isn't used by the spawner itself; tests reset
    ``_RecorderChild.BUCKETS[bucket_key]`` and pass it in via per-spec
    config so children record into the right bucket.
    """
    return GraphDefinition(
        name="dyn_firelist_minimal",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node("spawner", "_test_dyn_spawner", **spawner_config),
            _node("sink", "_test_dyn_sink"),
        ],
        edges=[
            _edge("e_seed_to_spawner", "seed", "spawner", sh="value", th="trigger"),
            _edge("e_spawner_to_sink", "spawner", "sink", sh="agg", th="value"),
        ],
    )


# ── 1. Sequential order preserved ─────────────────────────────────────────


def test_returns_firelist_dispatches_children_in_order() -> None:
    """3 specs → 3 child fires in declared order, aggregate sees that order."""
    bk = "test_order"
    _RecorderChild.BUCKETS[bk] = []
    spawner_cfg = {
        "specs": [
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "first"},
                "config": {"bucket_key": bk, "tag": "A"},
            },
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "second"},
                "config": {"bucket_key": bk, "tag": "B"},
            },
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "third"},
                "config": {"bucket_key": bk, "tag": "C"},
            },
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key=bk)
    exe, _sess = _run(g)
    bucket = _RecorderChild.BUCKETS[bk]
    assert [r["config"]["tag"] for r in bucket] == ["A", "B", "C"]
    assert [r["inputs"]["payload"] for r in bucket] == ["first", "second", "third"]
    # Ephemeral child ids must follow the spawner::dynN convention
    assert [r["node_id"] for r in bucket] == [
        "spawner::dyn0",
        "spawner::dyn1",
        "spawner::dyn2",
    ]
    # Sink received the aggregated list (3 entries, each a child output dict)
    sink_state = exe.nodes["sink"].state
    assert isinstance(sink_state.get("captured"), list)
    assert len(sink_state["captured"]) == 3
    assert [c.get("echo") for c in sink_state["captured"]] == ["first", "second", "third"]


# ── 2. Empty FireList ─────────────────────────────────────────────────────


def test_empty_firelist_calls_aggregate_with_empty_list() -> None:
    """Empty specs → aggregate([]) called → spawner_outputs propagate."""
    spawner_cfg = {
        "specs": [],
        "spawner_outputs": {"agg": "from_spawner_outputs"},
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key="test_empty")
    exe, _sess = _run(g)
    # aggregate({}) returns {"agg": []}, spawner_outputs has agg="from_spawner_outputs"
    # aggregated wins on collision → sink sees []
    assert exe.nodes["sink"].state["captured"] == []


def test_spawner_outputs_propagate_when_aggregate_returns_empty() -> None:
    """When aggregate returns dict missing a port, spawner_outputs fills it."""
    spawner_cfg = {
        "specs": [],
        "spawner_outputs": {"subtask_text": "captured_text"},
        "aggregate_override": {},  # aggregate returns {}; spawner_outputs supplies subtask_text
    }
    # Need a graph that wires the subtask_text port to a sink, not agg
    g = GraphDefinition(
        name="dyn_spawner_outputs_test",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node("spawner", "_test_dyn_spawner", **spawner_cfg),
            _node("sink", "_test_dyn_sink"),
        ],
        edges=[
            _edge("e_seed", "seed", "spawner", sh="value", th="trigger"),
            _edge("e_to_sink", "spawner", "sink", sh="subtask_text", th="value"),
        ],
    )
    exe, _sess = _run(g)
    assert exe.nodes["sink"].state["captured"] == "captured_text"


# ── 3. Child error stops sequence ─────────────────────────────────────────


def test_child_error_bubbles_and_stops_sequence() -> None:
    """spec[1] raises → spec[2] never fires; run terminates with error.

    The executor's outer try/except wraps the dataflow loop and flips
    ``session._status`` to ``"error"`` instead of re-raising, so we
    detect failure via the session + the partial side-effect (only the
    first child's bucket entry survived; the third child never fired).
    """
    bk = "test_child_error"
    _RecorderChild.BUCKETS[bk] = []
    spawner_cfg = {
        "specs": [
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "ok_1"},
                "config": {"bucket_key": bk, "tag": "ok_1"},
            },
            {
                "node_type": "_test_dyn_raising_child",
                "inputs": {},
            },
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "never_fires"},
                "config": {"bucket_key": bk, "tag": "never"},
            },
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key=bk)
    with _strict_errors():
        _exe, sess = _run(g)
    assert sess._status == "error"
    bucket = _RecorderChild.BUCKETS[bk]
    # First child fired; second raised; third NEVER fired.
    assert [r["config"]["tag"] for r in bucket] == ["ok_1"]


# ── 4. spawner_outputs + aggregate co-exist (aggregate wins on collision) ──


def test_spawner_outputs_and_aggregate_merge_aggregate_wins() -> None:
    """spawner_outputs has agg='from_outputs'; aggregate also emits agg —
    aggregate's value wins on key collision."""
    bk = "test_merge"
    _RecorderChild.BUCKETS[bk] = []
    spawner_cfg = {
        "specs": [
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "p"},
                "config": {"bucket_key": bk},
            },
        ],
        "spawner_outputs": {"agg": "from_outputs", "subtask_text": "kept"},
        "aggregate_override": {"agg": "from_aggregate"},
    }
    # Multi-edge graph: capture both agg and subtask_text through a probe
    g = GraphDefinition(
        name="dyn_merge",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node("spawner", "_test_dyn_spawner", **spawner_cfg),
            _node("sink_agg", "_test_dyn_sink"),
            _node("sink_text", "_test_dyn_sink"),
        ],
        edges=[
            _edge("e_seed", "seed", "spawner", sh="value", th="trigger"),
            _edge("e_agg", "spawner", "sink_agg", sh="agg", th="value"),
            _edge("e_text", "spawner", "sink_text", sh="subtask_text", th="value"),
        ],
    )
    exe, _sess = _run(g)
    # aggregate.agg ("from_aggregate") wins over spawner_outputs.agg ("from_outputs")
    assert exe.nodes["sink_agg"].state["captured"] == "from_aggregate"
    # subtask_text only in spawner_outputs → passes through
    assert exe.nodes["sink_text"].state["captured"] == "kept"


# ── 5. Nested FireList rejected ───────────────────────────────────────────


def test_nested_firelist_rejected() -> None:
    """A child whose execute() returns FireList terminates the run with error.

    The dispatcher raises ``NotImplementedError`` inside
    ``_fire_dynamic_children`` when a child's result is a ``FireList``; that
    propagates up through ``_fire_node`` (the spawner) and is caught by the
    main loop's outer except, flipping ``session._status`` to ``"error"``.
    """
    spawner_cfg = {
        "specs": [
            {"node_type": "_test_dyn_nested_firelist_child", "inputs": {}},
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key="test_nested")
    with _strict_errors():
        _exe, sess = _run(g)
    assert sess._status == "error"


# ── 6. Forbidden child types ──────────────────────────────────────────────


def test_forbidden_child_type_rejected() -> None:
    """Spawning a structural / control type as a dynamic child errors the run.

    The guard inside ``_fire_dynamic_children`` raises ``ValueError`` before
    the child is even instantiated. The main loop's outer except catches it
    and flips ``session._status`` to ``"error"``.
    """
    spawner_cfg = {
        "specs": [
            {"node_type": "iterOut", "inputs": {"stop": True}},
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key="test_forbidden")
    with _strict_errors():
        _exe, sess = _run(g)
    assert sess._status == "error"


# ── 7. capture_outputs whitelist ──────────────────────────────────────────


def test_capture_outputs_whitelist_filters_child_outputs() -> None:
    """capture_outputs=['echo'] → only 'echo' survives; 'extra' is dropped."""
    bk = "test_capture"
    _RecorderChild.BUCKETS[bk] = []
    spawner_cfg = {
        "specs": [
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "hello"},
                "config": {"bucket_key": bk, "extra_value": 42},
                "capture_outputs": ["echo"],
            },
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key=bk)
    exe, _sess = _run(g)
    agg = exe.nodes["sink"].state["captured"]
    assert agg == [{"echo": "hello"}]  # 'extra' filtered out


# ── 8. Spawner inside iterIn/iterOut loop ─────────────────────────────────


def test_spawner_inside_iter_loop_fires_fresh_children_per_iter() -> None:
    """Spawner inside an outer iter scope fires children EACH outer iter.

    Outer loop runs 3 iters (budget-capped). Spawner
    emits 2 children per iter → total 6 children fire. The child ids must
    reset per outer iter (always ``spawner::dyn0`` / ``::dyn1``) — the
    sequence number is per-firing, not run-global.
    """
    bk = "test_loop"
    _RecorderChild.BUCKETS[bk] = []
    g = GraphDefinition(
        name="dyn_in_loop",
        eval_graph=False,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node(
                "iter_in",
                "iterIn",
                pairedWith="iter_out",
                initPorts=[{"name": "x", "wire_type": "ANY", "persist": True}],
            ),
            _node(
                "spawner",
                "_test_dyn_spawner",
                specs=[
                    {
                        "node_type": "_test_dyn_recorder_child",
                        "inputs": {"payload": "A"},
                        "config": {"bucket_key": bk, "tag": "A"},
                    },
                    {
                        "node_type": "_test_dyn_recorder_child",
                        "inputs": {"payload": "B"},
                        "config": {"bucket_key": bk, "tag": "B"},
                    },
                ],
            ),
            _node("outer_body_counter", "_test_dyn_sink"),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=[{"name": "v", "wire_type": "ANY"}],
            ),
        ],
        edges=[
            _edge("e_seed", "seed", "iter_in", sh="value", th="init_x"),
            _edge("e_in_to_spawner", "iter_in", "spawner", sh="init_x", th="trigger"),
            _edge("e_spawner_to_body", "spawner", "outer_body_counter", sh="agg", th="value"),
            # The step_budget caps the outer loop at 3 iters; no stop wire
            # needed for this test.
            _edge("e_body_to_iter_out", "outer_body_counter", "iter_out", sh="done", th="v"),
        ],
        step_budget=3,  # cap outer at 3 iters
    )
    _exe, _sess = _run(g)
    bucket = _RecorderChild.BUCKETS[bk]
    # 3 outer iters x 2 children = 6 fires
    assert len(bucket) == 6
    # Tags interleave A,B,A,B,A,B
    assert [r["config"]["tag"] for r in bucket] == ["A", "B", "A", "B", "A", "B"]
    # Child ids reset per outer iter — every iter produces dyn0 + dyn1
    assert [r["node_id"] for r in bucket] == ["spawner::dyn0", "spawner::dyn1"] * 3


# ── 9. Access-grant inheritance (child sees spawner's containers) ─────────


def test_child_inherits_spawner_access_grants() -> None:
    """A dynamic child should see the containers granted to its spawner.

    Without the ``_dynamic_parent_id`` fallback in ``_fire_node``'s
    access-grant lookup, child id ``spawner::dyn0`` would never appear in
    ``_access_grant_index`` and ``_ContainerProbeChild`` would see no
    containers — which would break any child that needs the per-episode
    runtime container shared with its spawner (the VoxPoser use case).
    """
    bk = "test_grants"
    _RecorderChild.BUCKETS[bk] = []
    g = GraphDefinition(
        name="dyn_grants",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node(
                "spawner",
                "_test_dyn_spawner",
                specs=[
                    {
                        "node_type": "_test_dyn_container_probe_child",
                        "inputs": {},
                    },
                ],
            ),
            _node("sink", "_test_dyn_sink"),
        ],
        edges=[
            _edge("e_seed", "seed", "spawner", sh="value", th="trigger"),
            _edge("e_to_sink", "spawner", "sink", sh="agg", th="value"),
        ],
        containers=[
            ContainerDef(id="shared_runtime", label="shared"),
        ],
        access_grants=[
            AccessGrantDef(id="ag_spawner", node_id="spawner", container_id="shared_runtime"),
        ],
    )
    exe, _sess = _run(g)
    agg = exe.nodes["sink"].state["captured"]
    # The single child output should report ok=True and container_keys=['shared_runtime']
    assert len(agg) == 1
    assert agg[0]["ok"] is True
    assert agg[0]["container_keys"] == ["shared_runtime"]


# ── Declarative aggregators (server-mode path) ────────────────────────────


class _DeclarativeAggregatorSpawner(BaseCanvasNode):
    """Spawner that returns FireList with a declarative ``aggregator`` recipe
    and NO ``aggregate()`` method (mimics server-mode proxy class).
    """

    node_type = "_test_dyn_decl_aggregator_spawner"
    input_ports = [PortDef("trigger", "ANY", optional=True)]
    output_ports = [
        PortDef("echo", "ANY"),
        PortDef("extra", "ANY"),
        PortDef("renamed", "ANY"),
    ]

    async def forward(self, inputs, ctx):
        recipe = self.config.get("aggregator_recipe", {})
        bk = self.config.get("bucket_key", "_default")
        return FireList(
            specs=[
                FireSpec(
                    node_type="_test_dyn_recorder_child",
                    inputs={"payload": "first"},
                    config={"bucket_key": bk, "extra_value": "ev_first"},
                ),
                FireSpec(
                    node_type="_test_dyn_recorder_child",
                    inputs={"payload": "second"},
                    config={"bucket_key": bk, "extra_value": "ev_second"},
                ),
            ],
            spawner_outputs={"echo": "spawner_default"},
            aggregator=recipe,
        )


register_node(_DeclarativeAggregatorSpawner)


def _decl_aggregator_graph(recipe: dict, bucket_key: str) -> GraphDefinition:
    return GraphDefinition(
        name="decl_agg",
        eval_graph=False,
        step_budget=10,
        nodes=[
            _node("seed", "_test_dyn_seed"),
            _node(
                "spawner",
                "_test_dyn_decl_aggregator_spawner",
                aggregator_recipe=recipe,
                bucket_key=bucket_key,
            ),
            _node("sink_echo", "_test_dyn_sink"),
            _node("sink_extra", "_test_dyn_sink"),
        ],
        edges=[
            _edge("e_seed", "seed", "spawner", sh="value", th="trigger"),
            _edge("e_echo", "spawner", "sink_echo", sh="echo", th="value"),
            _edge("e_extra", "spawner", "sink_extra", sh="extra", th="value"),
        ],
    )


def test_aggregator_passthrough_last() -> None:
    """``{"kind": "passthrough_last"}`` → output dict = last child's result.

    Last child's ``echo='second'`` overrides spawner_outputs.echo='spawner_default',
    and last child's ``extra='ev_second'`` populates the extra port.
    """
    bk = "test_decl_passlast"
    _RecorderChild.BUCKETS[bk] = []
    g = _decl_aggregator_graph({"kind": "passthrough_last"}, bk)
    exe, _sess = _run(g)
    assert exe.nodes["sink_echo"].state["captured"] == "second"
    assert exe.nodes["sink_extra"].state["captured"] == "ev_second"


def test_aggregator_passthrough_index() -> None:
    """``{"kind": "passthrough_index", "index": 0}`` → output = first child."""
    bk = "test_decl_passidx"
    _RecorderChild.BUCKETS[bk] = []
    g = _decl_aggregator_graph({"kind": "passthrough_index", "index": 0}, bk)
    exe, _sess = _run(g)
    assert exe.nodes["sink_echo"].state["captured"] == "first"
    assert exe.nodes["sink_extra"].state["captured"] == "ev_first"


def test_aggregator_merge_all() -> None:
    """``{"kind": "merge_all"}`` → output = dict-merge of all children
    (later children overwrite earlier on key collision)."""
    bk = "test_decl_merge"
    _RecorderChild.BUCKETS[bk] = []
    g = _decl_aggregator_graph({"kind": "merge_all"}, bk)
    exe, _sess = _run(g)
    # Second child's echo='second' overwrites first's echo='first'
    assert exe.nodes["sink_echo"].state["captured"] == "second"
    assert exe.nodes["sink_extra"].state["captured"] == "ev_second"


def test_aggregator_unknown_kind_errors() -> None:
    """Unknown aggregator kind raises ValueError → run terminates with error."""
    bk = "test_decl_unknown"
    _RecorderChild.BUCKETS[bk] = []
    g = _decl_aggregator_graph({"kind": "_no_such_kind"}, bk)
    with _strict_errors():
        _exe, sess = _run(g)
    assert sess._status == "error"


# ── 10. Log entries carry parent_node_id / dynamic_index ──────────────────


def test_dynamic_child_log_carries_parent_id_and_index() -> None:
    """Phase 2 logging extension: each child's log entry has ``parent_node_id``
    pointing to the spawner and ``dynamic_index`` matching its position in
    the FireList. Spawner's own log entry has both fields ``None`` (it's a
    static-topology node, not a child of anything).
    """
    from ..logging.logger import ExecutionLogger

    bk = "test_log_provenance"
    _RecorderChild.BUCKETS[bk] = []
    spawner_cfg = {
        "specs": [
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "x"},
                "config": {"bucket_key": bk, "tag": "x"},
            },
            {
                "node_type": "_test_dyn_recorder_child",
                "inputs": {"payload": "y"},
                "config": {"bucket_key": bk, "tag": "y"},
            },
        ],
    }
    g = _minimal_spawn_to_sink_graph(spawner_cfg, bucket_key=bk)
    # Wire a fresh ExecutionLogger so we can inspect entries post-run.
    from ..graph_def import _synthesize_iterin_ports

    _synthesize_iterin_ports(g)
    if g.step_budget is None:
        g.step_budget = 10
    exe = GraphExecutor()
    sess = _StubSession()
    logger = ExecutionLogger(execution_id="test_log_provenance", source="canvas")
    exe._logger = logger
    asyncio.run(exe.run(g, sess, step_delay_ms=0))

    entries = list(logger._buffer)
    by_node = {(e.node_id): e for e in entries}

    # Spawner entry — no provenance fields.
    spawner_entry = by_node.get("spawner")
    assert spawner_entry is not None
    assert spawner_entry.parent_node_id is None
    assert spawner_entry.dynamic_index is None

    # Child entries — provenance fields populated.
    child0 = by_node.get("spawner::dyn0")
    child1 = by_node.get("spawner::dyn1")
    assert child0 is not None and child1 is not None
    assert child0.parent_node_id == "spawner"
    assert child0.dynamic_index == 0
    assert child1.parent_node_id == "spawner"
    assert child1.dynamic_index == 1
