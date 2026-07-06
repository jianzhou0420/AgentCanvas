"""Contract tests for the six iterIn port configurations.

``IterInNode``'s docstring enumerates the legal configs per port, indexed
by ``(persist, init-writer present, iterOut-writer present)``:

  C1 run-constant       persist=T, init=T, loop=F
  C2 step-0 one-shot    persist=F, init=T, loop=F
  C3 loop-carried       persist=F, init=F, loop=T
  P1 seeded + refreshed persist=T, init=T, loop=T
  P5 pure feedback      persist=T, init=F, loop=T

Synthesis is always-prefix (``graph_def._synthesize_iterin_ports``):
``initPorts.x`` → handle ``init_x``, paired ``iterOut.ports.x`` →
handle ``iterout_x`` — separate slots, no cross-writer merge. P1 is
therefore realised by wiring BOTH handles into one consumer port; the
routing loop walks edges in declaration order and later writes win, so
the refresh edge must come after the seed edge. The reversed order
freezes the consumer on the seed — the modern residue of the 2026-05-05
CMA "100% LEFT spin" pathology — pinned here as a documented hazard.

Shared harness (``_node`` / ``_edge`` / ``_run`` + ``_FireCounter``)
comes from ``test_executor_scopes``; importing it registers the node.

Timeline used by every test (``loop_iters=3``):

* ``body`` fires once per iteration, emitting ``value`` = 1, 2, 3; its
  third fire raises ``done`` → ``iterOut.stop`` ends the run after
  iteration 2 (three iterations: 0, 1, 2).
* The boundary handoff writes ``iterout_*`` slots after iterations 0
  and 1 only — the terminal boundary stops instead of handing off.
"""

from __future__ import annotations

from ..graph_def import EdgeDef, GraphDefinition
from .test_executor_scopes import _edge, _node, _run

_LOOP_ITERS = 3


def _cfg_graph(
    x_init: dict | None,
    x_on_iterout: bool,
    consumer_edges: list[EdgeDef],
    x_iterout_persist: bool | None = None,
) -> GraphDefinition:
    """One-scope loop plus a ``consumer`` fed from iterIn's ``x`` slots.

    ``driver`` (C1-style run-constant) keeps ``body`` firing every
    iteration regardless of the ``x`` config under test.
    """
    init_ports = [{"name": "driver", "wire_type": "ANY", "persist": True}]
    if x_init is not None:
        init_ports.append({"name": "x", "wire_type": "ANY", **x_init})

    iterout_ports = [{"name": "v", "wire_type": "ANY"}]
    if x_on_iterout:
        port: dict = {"name": "x", "wire_type": "ANY"}
        if x_iterout_persist is not None:
            port["persist"] = x_iterout_persist
        iterout_ports.append(port)

    edges = [
        _edge("e_seed_driver", "seed", "iter_in", sh="value", th="init_driver"),
        _edge("e_driver_body", "iter_in", "body", sh="init_driver", th="trigger"),
        _edge("e_body_stop", "body", "iter_out", sh="done", th="stop"),
        _edge("e_body_v", "body", "iter_out", sh="value", th="v"),
    ]
    if x_init is not None:
        edges.append(_edge("e_seed_x", "seed", "iter_in", sh="value", th="init_x"))
    if x_on_iterout:
        edges.append(_edge("e_body_x", "body", "iter_out", sh="value", th="x"))
    edges.extend(consumer_edges)

    return GraphDefinition(
        name="iterin_cfg",
        eval_graph=False,
        step_budget=_LOOP_ITERS * 4,
        nodes=[
            _node("seed", "_test_fire_counter"),
            _node("iter_in", "iterIn", pairedWith="iter_out", initPorts=init_ports),
            _node("body", "_test_fire_counter", done_after=_LOOP_ITERS),
            _node(
                "iter_out",
                "iterOut",
                pairedWith="iter_in",
                ports=iterout_ports,
            ),
            _node("consumer", "_test_fire_counter"),
        ],
        edges=edges,
    )


def _consumer(exe) -> tuple[int, object]:
    state = exe.nodes["consumer"].state
    return state.get("total_fires") or 0, state.get("last_trigger")


# ── C1 · run-constant — persist=T, init writer only ─────────────────────


def test_c1_run_constant_feeds_every_iteration() -> None:
    exe = _run(
        _cfg_graph(
            x_init={"persist": True},
            x_on_iterout=False,
            consumer_edges=[_edge("e_c", "iter_in", "consumer", sh="init_x", th="trigger")],
        )
    )
    fires, last = _consumer(exe)
    assert fires == _LOOP_ITERS  # every iteration
    assert last == 1  # always the run-start seed (seed fires once, value=1)


# ── C2 · step-0 one-shot — persist=F, init writer only ──────────────────


def test_c2_one_shot_feeds_iteration_zero_only() -> None:
    exe = _run(
        _cfg_graph(
            x_init={"persist": False},
            x_on_iterout=False,
            consumer_edges=[_edge("e_c", "iter_in", "consumer", sh="init_x", th="trigger")],
        )
    )
    fires, last = _consumer(exe)
    assert fires == 1  # slot cleared after the iter-0 fire; consumer starves after
    assert last == 1


# ── C3 · loop-carried — persist=F, iterOut writer only ──────────────────


def test_c3_loop_carried_feeds_from_iteration_one() -> None:
    exe = _run(
        _cfg_graph(
            x_init=None,
            x_on_iterout=True,
            x_iterout_persist=False,
            consumer_edges=[_edge("e_c", "iter_in", "consumer", sh="iterout_x", th="trigger")],
        )
    )
    fires, last = _consumer(exe)
    # No init writer: nothing at iter 0. Handoffs after iters 0 and 1
    # feed iters 1 and 2; the terminal boundary stops without handing off.
    assert fires == _LOOP_ITERS - 1
    assert last == 2  # body's iter-1 value


# ── P5 · pure feedback — persist=T, iterOut writer only ─────────────────


def test_p5_pure_feedback_matches_c3_within_one_run() -> None:
    exe = _run(
        _cfg_graph(
            x_init=None,
            x_on_iterout=True,
            x_iterout_persist=True,
            consumer_edges=[_edge("e_c", "iter_in", "consumer", sh="iterout_x", th="trigger")],
        )
    )
    fires, last = _consumer(exe)
    # persist=T only changes slot retention after fire; every boundary
    # rewrites the slot anyway, so a simple loop observes C3 behavior.
    assert fires == _LOOP_ITERS - 1
    assert last == 2


# ── P1 · seeded + refreshed — both writers, refresh edge declared last ──


def test_p1_seed_then_refresh_when_refresh_edge_is_later() -> None:
    exe = _run(
        _cfg_graph(
            x_init={"persist": True},
            x_on_iterout=True,
            consumer_edges=[
                _edge("e_ci", "iter_in", "consumer", sh="init_x", th="trigger"),
                _edge("e_cl", "iter_in", "consumer", sh="iterout_x", th="trigger"),
            ],
        )
    )
    fires, last = _consumer(exe)
    assert fires == _LOOP_ITERS  # seeded at iter 0, refreshed after
    assert last == 2  # iter 2 sees body's iter-1 value, not the frozen seed


# ── Hazard · P1 with reversed edge order freezes on the seed ────────────


def test_p1_reversed_edge_order_freezes_consumer_on_seed() -> None:
    """Pins the dual-wire hazard: with ``init_x`` routed AFTER
    ``iterout_x`` (and init persist=T so the seed slot never clears),
    the later edge wins every iteration and the consumer is frozen on
    step-0 data. Historical incident: 2026-05-05 CMA "100% LEFT spin".
    If this assertion ever flips, the engine started merging writers or
    reordering fan-in — update [[feedback_iterin_dual_wire_obs_freeze]].
    """
    exe = _run(
        _cfg_graph(
            x_init={"persist": True},
            x_on_iterout=True,
            consumer_edges=[
                _edge("e_cl", "iter_in", "consumer", sh="iterout_x", th="trigger"),
                _edge("e_ci", "iter_in", "consumer", sh="init_x", th="trigger"),
            ],
        )
    )
    fires, last = _consumer(exe)
    assert fires == _LOOP_ITERS
    assert last == 1  # frozen on the seed — the documented hazard


# ── Capture semantics · declared-but-unwired init port stays absent ─────


def test_unwired_init_port_is_not_autofilled_with_none() -> None:
    """The init side is a captured-bundle surface, not a defaults
    provider (see [[feedback_initialize_semantics]]): a declared but
    unwired ``initPorts`` entry never materialises, so a consumer wired
    to it starves rather than receiving an auto-filled ``None``.
    """
    exe = _run(_cfg_graph_unwired())
    fires, _ = _consumer(exe)
    assert fires == 0  # never fed — no auto-None


def _cfg_graph_unwired() -> GraphDefinition:
    """Same shape as ``_cfg_graph`` C1 but WITHOUT the seed→init_x edge."""
    g = _cfg_graph(
        x_init={"persist": True},
        x_on_iterout=False,
        consumer_edges=[_edge("e_c", "iter_in", "consumer", sh="init_x", th="trigger")],
    )
    g.edges = [e for e in g.edges if e.id != "e_seed_x"]
    return g
