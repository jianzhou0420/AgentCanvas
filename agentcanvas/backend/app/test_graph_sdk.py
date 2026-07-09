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
