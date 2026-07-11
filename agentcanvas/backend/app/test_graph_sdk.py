"""Tests for the Graph SDK Graph surface (:mod:`app.graph_sdk`).

Covers the whole authoring→running→interop→codegen arc, all hermetic (no env
servers, no GPU, no network):

* pure-Python in-process ``g.run()`` and its ``RunResult``
* code ⇄ JSON round-trip
* the ``g.loop()`` iterIn/iterOut sugar (persist semantics + feed fan-out)
* ``g.hook()`` and ``g.composite()`` builder sugar
* nodeset auto-detection for ``Graph.run(load_nodesets="auto")``
* the inverse ``graph_to_code`` codegen, round-tripped against the verified
  MapGPT-MP3D graph

The one path that genuinely needs an env server + a conda env + model weights
(a real ``env_mp3d`` episode) is out of scope for a unit test — that is an
experiment and runs behind ``/experiment:run``.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from .graph_def import GraphDefinition
from .graph_sdk import Graph

# ── running: pure-Python nodes in-process ────────────────────────────────


def test_pure_python_run():
    from .graph_sdk_demo import build as build_demo

    result = build_demo().run(validate=True)
    assert result["total"] == 36
    assert isinstance(result.metrics, dict)


def test_json_round_trip():
    from .graph_sdk_demo import build as build_demo

    d1 = build_demo().to_dict()
    d2 = GraphDefinition.from_dict(d1).to_dict()
    assert d1 == d2


# ── authoring sugar: loop / hooks / composite ────────────────────────────


def test_loop_persist_and_feed():
    g = Graph(name="loop")
    loop = g.loop(init=[("instr", "TEXT")], carry=[("x", "TEXT")])

    ii = next(nd for nd in g.definition.nodes if nd.type == "iterIn")
    io = next(nd for nd in g.definition.nodes if nd.type == "iterOut")

    init_ports = {p["name"]: p for p in ii.config["initPorts"]}
    # init-only ports persist across steps; carried ports do NOT persist on the
    # init side (the iterOut carry, persist=True, supersedes after step 0).
    assert init_ports["instr"]["persist"] is True
    assert init_ports["x"]["persist"] is False
    out_ports = {p["name"]: p for p in io.config["ports"]}
    assert set(out_ports) == {"x"}
    assert out_ports["x"]["persist"] is True

    cons = g.add("noop", id="c")
    # feed() of a carried port wires BOTH init and iterout sides (2 edges);
    # an init-only port wires only the init side (1 edge).
    assert len(loop.feed("x", cons.in_("p"))) == 2
    assert len(loop.feed("instr", cons.in_("q"))) == 1

    with pytest.raises(ValueError):
        loop.carry("instr", cons.out("z"))  # init-only cannot carry back
    with pytest.raises(KeyError):
        loop.seed("nope", cons.out("z"))  # undeclared port


def test_hooks_round_trip():
    g = Graph(name="h")
    g.add("graphIn", id="a", portName="x")
    g.hook("GraphStart", "echo hi")
    g.hook("PostNodeExecute", "log.sh", match_node_type="llmCall", timeout_ms=5000)

    assert len(g.definition.hooks) == 2
    gd2 = GraphDefinition.from_dict(g.to_dict())
    assert len(gd2.hooks) == 2
    assert gd2.hooks[0].event == "GraphStart"
    assert gd2.hooks[0].command == "echo hi"
    assert gd2.hooks[1].match_node_type == "llmCall"
    assert gd2.hooks[1].timeout_ms == 5000


def test_composite_round_trip():
    sub = Graph(name="inner")
    si = sub.graph_in("v")
    so = sub.graph_out("w")
    sub.connect(si.out("v"), so.in_("value"))

    g = Graph(name="outer")
    comp = g.composite("box", sub, label="Box")
    assert comp.type == "compositeNode"

    nd = next(n for n in g.definition.nodes if n.id == "box")
    assert nd.subgraph is not None
    assert len(nd.subgraph.nodes) == 2

    gd2 = GraphDefinition.from_dict(g.to_dict())
    box2 = next(n for n in gd2.nodes if n.id == "box")
    assert box2.subgraph is not None
    assert len(box2.subgraph.nodes) == 2


# ── running: nodeset auto-detection (no actual load) ─────────────────────


def test_wants_nodesets_detection():
    pure = Graph(name="p")
    pure.add("iterIn", id="i")
    pure.add("llmCall", id="l")
    assert pure._wants_nodesets("auto") is False
    assert pure._wants_nodesets(True) is True

    ns = Graph(name="n")
    ns.add("mapgpt__observe", id="o")
    assert ns._wants_nodesets("auto") is True
    assert ns._wants_nodesets(False) is False


# ── interop: inverse codegen (graph → standalone builder) ────────────────


def test_to_code_of_demo_is_valid_python():
    from .graph_sdk_demo import build as build_demo

    src = build_demo().to_code()
    compile(src, "<gen>", "exec")  # emits syntactically valid Python
    assert "def build()" in src
    assert ".connect(" in src


def _verified_mapgpt_path():
    from .mapgpt_mp3d_sdk import VERIFIED_JSON

    return VERIFIED_JSON


@pytest.mark.skipif(
    not _verified_mapgpt_path().exists(),
    reason="verified MapGPT-MP3D graph JSON not present",
)
def test_codegen_round_trip_mapgpt():
    from .graph_sdk_codegen import graph_to_code
    from .mapgpt_mp3d_sdk import _diff, signature

    gd = GraphDefinition.from_dict(json.loads(_verified_mapgpt_path().read_text()))
    src = graph_to_code(gd)
    compile(src, "<gen>", "exec")

    ns: dict = {"__name__": "_gen_mapgpt_test"}  # __name__ ≠ __main__ so guard is inert
    exec(src, ns)
    built = ns["build"]().to_dict()

    diffs = _diff(signature(built), signature(gd.to_dict()))
    assert not diffs, f"codegen round-trip mismatch: {diffs}"


# ── default-on post-load validation (check=True) ─────────────────────────


def _demo_registered():
    # importing the demo module registers demo_const / demo_add / demo_scale
    import app.graph_sdk_demo  # noqa: F401


def test_check_passes_on_clean_graph():
    from .graph_sdk import GraphValidationError
    from .graph_sdk_demo import build as build_demo

    build_demo()._check_resolved()  # must not raise

    with pytest.raises(GraphValidationError):
        raise GraphValidationError("smoke")  # is a ValueError subclass
    assert issubclass(GraphValidationError, ValueError)


def test_check_catches_unknown_node_type():
    from .graph_sdk import Graph, GraphValidationError

    _demo_registered()
    g = Graph(name="t")
    g.add("demo_addd")  # typo of demo_add
    with pytest.raises(GraphValidationError) as ei:
        g._check_resolved()
    msg = str(ei.value)
    assert "unknown node type" in msg
    assert "demo_add" in msg  # did-you-mean suggestion


def test_check_catches_bad_port():
    from .graph_sdk import Graph, GraphValidationError

    _demo_registered()
    g = Graph(name="t")
    c = g.add("demo_const", id="c")
    a = g.add("demo_add", id="a")
    out = g.graph_out("total")
    g.connect(c.out("valuee"), a.in_("a"))  # typo: valuee (real port is 'value')
    g.connect(c.out("value"), a.in_("b"))
    g.connect(a.out("result"), out.in_("value"))
    with pytest.raises(GraphValidationError) as ei:
        g._check_resolved()
    msg = str(ei.value)
    assert "not an output port" in msg
    assert "'value'" in msg  # did-you-mean suggestion


def test_check_catches_required_unwired():
    from .graph_sdk import Graph, GraphValidationError

    _demo_registered()
    g = Graph(name="t")
    g.add("demo_add", id="a")  # inputs a, b required and unwired
    with pytest.raises(GraphValidationError, match="required input"):
        g._check_resolved()


def test_check_skips_generated_loop_ports():
    from .graph_sdk import Graph, GraphValidationError

    _demo_registered()
    g = Graph(name="loop")
    loop = g.loop(init=[("instr", "TEXT")], carry=[("x", "TEXT")])
    body = g.add("demo_const", id="b")
    sink = g.add("demo_scale", id="s")
    loop.feed("x", sink.in_("x"))
    loop.carry("x", body.out("value"))
    # iterIn/iterOut ports are synthesised by loop(), not user-typed — the
    # check must never flag them, or it would false-positive on every loop.
    try:
        g._check_resolved()
    except GraphValidationError as e:
        assert "iterIn" not in str(e) and "iterOut" not in str(e), str(e)


def test_check_can_be_disabled():
    from .graph_sdk_demo import build as build_demo

    # check=False must not raise even though (here) the graph is clean anyway;
    # exercised to lock the opt-out param in the public signature.
    build_demo().run(validate=False, check=False)


# ── introspection: catalog / describe / nodesets ─────────────────────────


def test_catalog_lists_builtins():
    from .graph_sdk import catalog

    cat = catalog(refresh=True)
    assert isinstance(cat, list) and cat == sorted(cat)
    assert "graphOut" in cat and "graphIn" in cat and "iterIn" in cat
    # server=False filter still includes builtins
    assert "graphOut" in catalog(server=False)


def test_describe_builtin_shape():
    from .graph_sdk import describe

    d = describe("graphOut")
    assert d["node_type"] == "graphOut"
    assert d["nodeset"] is None
    assert d["server"] is False and d["env"] is None
    assert any(p["name"] == "value" for p in d["inputs"])
    assert isinstance(d["outputs"], list)


def test_describe_unknown_raises():
    from .graph_sdk import describe

    with pytest.raises(KeyError):
        describe("definitely__not__a__node")


def test_describe_local_node():
    import app.graph_sdk_demo  # noqa: F401 — registers demo_*

    from .graph_sdk import describe

    d = describe("demo_add", refresh=True)
    assert d["server"] is False
    assert {p["name"] for p in d["inputs"]} == {"a", "b"}
    assert [p["name"] for p in d["outputs"]] == ["result"]


def test_nodesets_env_metadata_without_spawn():
    from .graph_sdk import describe, nodesets

    nss = {n["name"]: n for n in nodesets(refresh=True)}
    # env_habitat is a server nodeset — its metadata must be readable WITHOUT
    # spawning a subprocess. Guarded so the test is robust if the workspace
    # nodeset is absent on this checkout.
    envh = nss.get("env_habitat")
    if envh is not None:
        assert envh["server"] is True
        assert envh["env"] == "ac-vlnce"
        assert "env_habitat__reset" in envh["node_types"]
        d = describe("env_habitat__reset")
        assert d["server"] is True and d["nodeset"] == "env_habitat"


# ── typed authoring: g.add(NodeClass) + generated stubs ──────────────────


def test_add_accepts_real_node_class():
    import app.graph_sdk_demo  # noqa: F401 — registers demo_*
    from app.agent_loop.builtin_nodes import NODE_HANDLERS

    from .graph_sdk import Graph, NodeProxy

    cls = NODE_HANDLERS["demo_add"]
    g = Graph(name="t")
    h = g.add(cls)  # pass the class, not the "demo_add" string
    assert h.type == "demo_add"
    assert not isinstance(h, NodeProxy)  # real class → plain NodeHandle


def test_add_node_proxy_returns_typed_handle():
    from .graph_sdk import Graph, NodeProxy

    class Foo(NodeProxy):
        node_type = "graphOut"

    g = Graph(name="t")
    h = g.add(Foo)
    assert isinstance(h, Foo)  # typed handle instance, not a bare NodeHandle
    assert h.type == "graphOut"


def test_add_class_without_node_type_raises():
    from .graph_sdk import Graph

    class Bare:
        pass

    with pytest.raises(ValueError, match="node_type"):
        Graph(name="t").add(Bare)


def test_emit_nodeset_module_compiles():
    from .graph_sdk import _emit_nodeset_module

    nodes = [
        {
            "node_type": "model_sam__segment_auto",
            "server": True,
            "env": "ac-fm",
            "description": "Segment everything in the image.",
            "inputs": [{"name": "image_b64"}],
            "outputs": [{"name": "masks"}],
        }
    ]
    src = _emit_nodeset_module("model_sam", nodes)
    compile(src, "<gen>", "exec")  # must be valid Python
    assert "class SegmentAuto(NodeProxy):" in src
    assert "node_type = 'model_sam__segment_auto'" in src
    assert "Literal['image_b64']" in src
    assert "Literal['masks']" in src


def test_generate_node_stubs_writes_valid_python(tmp_path):
    import pathlib

    from .graph_sdk import generate_node_stubs

    written = generate_node_stubs(dst=tmp_path, refresh=True)
    assert any(p.endswith("__init__.py") for p in written)
    for p in written:  # every emitted file must be valid Python
        compile(pathlib.Path(p).read_text(), p, "exec")

    envh = tmp_path / "env_habitat.py"  # guarded: only if the nodeset is present
    if envh.exists():
        src = envh.read_text()
        assert "class Reset(NodeProxy):" in src
        assert "node_type = 'env_habitat__reset'" in src


# ── run observability: on_event stream (RunEvent) ────────────────────────


def test_on_event_streams_lifecycle():
    from .graph_sdk import RunEvent
    from .graph_sdk_demo import build as build_demo

    events: list[RunEvent] = []
    result = build_demo().run(on_event=events.append)

    assert result["total"] == 36  # run still returns its RunResult normally
    assert all(isinstance(e, RunEvent) for e in events)
    kinds = [e.kind for e in events]
    assert kinds[0] == "graph_start"
    assert kinds[-1] == "graph_complete"
    # every node that fired produced a node_start then a node_finish (5 nodes:
    # two consts, add, scale, graphOut — a pure DAG, each fires exactly once).
    assert kinds.count("node_start") == kinds.count("node_finish") == 5
    # the demo_add fire carries its summed output live on the event
    add_fin = next(
        e for e in events if e.kind == "node_finish" and e.node_type == "demo_add"
    )
    assert add_fin.outputs["result"] == 12
    assert add_fin.duration_ms is not None


def test_on_event_reports_node_error():
    import app.graph_sdk_demo  # noqa: F401 — registers demo_const
    from app.agent_loop.builtin_nodes import register_node
    from app.components.bases import BaseCanvasNode, PortDef

    from .graph_sdk import Graph, RunEvent

    class BoomNode(BaseCanvasNode):
        node_type = "demo_boom"
        input_ports = [PortDef("x", "ANY")]
        output_ports = [PortDef("y", "ANY")]

        async def forward(self, inputs: dict, ctx) -> dict:
            raise ValueError("boom")

    register_node(BoomNode)

    g = Graph(name="err")
    c = g.add("demo_const", value=1)
    b = g.add("demo_boom")
    c.out("value") >> b.in_("x")

    events: list[RunEvent] = []
    # the run may re-raise (node-error conviction) — the event fires either way
    with contextlib.suppress(Exception):
        g.run(on_event=events.append)

    errs = [e for e in events if e.kind == "node_error"]
    assert errs and errs[0].node_type == "demo_boom"
    assert "boom" in (errs[0].error or "")
    assert errs[0].duration_ms is not None


def test_on_event_accepts_async_callback():
    from .graph_sdk import RunEvent
    from .graph_sdk_demo import build as build_demo

    events: list[RunEvent] = []

    async def sink(ev: RunEvent) -> None:  # async callbacks are awaited
        events.append(ev)

    build_demo().run(on_event=sink)
    assert any(e.kind == "node_finish" for e in events)


def test_run_event_str_is_compact():
    from .graph_sdk import RunEvent

    s = str(
        RunEvent(
            kind="node_finish",
            node_type="demo_add",
            node_id="demo_add_0",
            step=2,
            duration_ms=1.5,
            outputs={"result": 12},
        )
    )
    assert "node_finish" in s and "demo_add" in s and "out=[result]" in s


# ── server-node record / replay (hermetic — no real server) ──────────────


def _register_fake_server_node():
    """A pretend server node: node_type carries the '__' nodeset convention but
    it's a plain local class, so record/replay can be exercised with no server."""
    import app.graph_sdk_demo  # noqa: F401 — registers demo_const
    from app.agent_loop.builtin_nodes import register_node
    from app.components.bases import BaseCanvasNode, PortDef

    class FakeEcho(BaseCanvasNode):
        node_type = "fake__echo"
        input_ports = [PortDef("x", "ANY")]
        output_ports = [PortDef("y", "ANY")]

        async def forward(self, inputs: dict, ctx) -> dict:
            return {"y": inputs["x"]}

    register_node(FakeEcho)


def _echo_graph():
    from .graph_sdk import Graph

    g = Graph(name="rr")
    c = g.add("demo_const", value=7)
    e = g.add("fake__echo")
    out = g.graph_out("v")
    c.out("value") >> e.in_("x")
    e.out("y") >> out.in_("value")
    return g


def test_record_replay_round_trip(tmp_path):
    from pathlib import Path

    from app.agent_loop.builtin_nodes import NODE_HANDLERS

    _register_fake_server_node()
    cassette = tmp_path / "echo.mpk"

    # RECORD — fake node runs for real; load_nodesets=False (it's already
    # registered, there is no real "fake" nodeset to load).
    rec = _echo_graph().run(record=cassette, load_nodesets=False)
    assert rec["v"] == 7
    assert Path(cassette).exists() and Path(cassette).stat().st_size > 0

    # Simulate "the server is gone": drop the node type entirely. Replay must
    # rebuild the proxy class from the cassette's manifest and serve offline.
    NODE_HANDLERS.pop("fake__echo", None)
    assert "fake__echo" not in NODE_HANDLERS

    rep = _echo_graph().run(replay=cassette)
    assert rep["v"] == 7  # recorded output flowed through, no server touched
    # Player unregisters the class it rebuilt → global registry left clean.
    assert "fake__echo" not in NODE_HANDLERS


def test_replay_exhausted_on_extra_call(tmp_path):
    from .graph_sdk import Graph, RunEvent

    _register_fake_server_node()
    cassette = tmp_path / "echo1.mpk"
    _echo_graph().run(record=cassette, load_nodesets=False)  # exactly ONE fake__echo call

    # A replay graph that fires fake__echo TWICE exhausts the 1-call cassette.
    g = Graph(name="rr2")
    c = g.add("demo_const", value=7)
    a = g.add("fake__echo", id="a")
    b = g.add("fake__echo", id="b")
    oa = g.graph_out("va")
    ob = g.graph_out("vb")
    c.out("value") >> a.in_("x")
    c.out("value") >> b.in_("x")
    a.out("y") >> oa.in_("value")
    b.out("y") >> ob.in_("value")

    # The extra call surfaces as a node_error (the run convicts it internally
    # rather than always re-raising) — assert the drift is loud, not silent.
    events: list[RunEvent] = []
    with contextlib.suppress(Exception):
        g.run(replay=cassette, on_event=events.append)

    errs = [e for e in events if e.kind == "node_error" and e.node_type == "fake__echo"]
    assert errs, "second fake__echo fire should exhaust the 1-call cassette"
    assert "recorded result" in (errs[0].error or "")


def test_record_replay_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _echo_graph().run(record=tmp_path / "a.mpk", replay=tmp_path / "b.mpk")


def _cassette_path():
    from pathlib import Path

    return (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "cassettes"
        / "habitat_sam_first_frame.mpk"
    )


def _load_example07():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "examples" / "07_habitat_sam.py"
    spec = importlib.util.spec_from_file_location("example07_replay", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # registers the local glue nodes; imports no habitat
    return mod


@pytest.mark.skipif(not _cassette_path().exists(), reason="habitat-sam cassette not present")
def test_habitat_sam_replay():
    """The real payoff: example 07's Habitat→SAM graph — normally a GPU + two
    foreign conda envs — runs as a hermetic unit test off a recorded cassette.
    Replays env_habitat__* + model_sam__* from the cassette while the local glue
    nodes (RgbToPng / OverlayMasks) run for real, so it fails loudly if a port
    or interface drifts from the recording. No server, no GPU, no conda env."""
    import numpy as np

    g = _load_example07().build()
    r = g.run(replay=str(_cassette_path()))
    assert r["num_masks"] == 35  # recorded SAM output flows through the real overlay
    frame = np.asarray(r.outputs["annotated"])
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
