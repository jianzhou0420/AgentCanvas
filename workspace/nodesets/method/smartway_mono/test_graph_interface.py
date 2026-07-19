"""Graph-level interface tests: ``smartway_mono_ce.json`` vs current port contracts.

Pins the two facts that justified keeping the mono graph promotable after the
2026-06 gym-interface unification — with no simulator, no GPU, and no LLM call
(method-node *logic* equivalence lives separately in
``workspace/nodesets/method/smartway/test_equivalence.py``):

1. **Wire-type validity** — every edge in ``smartway_mono_ce.json`` resolves
   against the port declarations the nodesets export *today* (static
   introspection via ``app.tools.validate_graph``): 0 errors, 0 skipped edges,
   0 unresolved node types. The decomposed ``smartway_ce.json`` is checked
   too, as the known-good control.
2. **Env-wiring parity** — with method nodes collapsed to an opaque marker,
   the env-side edge set (``env_habitat__*`` / ``smartway_waypoint__*`` /
   ``smartway_perception__*`` handles) is identical to ``smartway_ce.json``,
   the decomposed variant that runs post-refactor in production. The two
   graphs differ only in how the method segment is cut, never in how they
   touch the environment.

Both graphs are located by filename glob under ``workspace/graphs/vln/`` so
the tests survive promotion between ``unverified/`` and ``verified/``.

Run from the backend (so ``app.*`` is importable):

    cd agentcanvas/backend && \
      PYTHONPATH=../..:. python -m pytest \
        ../../workspace/nodesets/method/smartway_mono/test_graph_interface.py -v
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

# Ensure workspace + backend are importable.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agentcanvas" / "backend"))

from app.tools.validate_graph import _check_one, introspect_nodesets, make_resolver

_VLN_GRAPHS = REPO_ROOT / "workspace" / "graphs" / "vln"
_ENV_PREFIXES = ("env_habitat__", "smartway_waypoint__", "smartway_perception__")


def _graph_path(filename: str) -> Path:
    hits = glob.glob(str(_VLN_GRAPHS / "**" / filename), recursive=True)
    assert len(hits) == 1, f"expected exactly one {filename} under {_VLN_GRAPHS}, got {hits}"
    return Path(hits[0])


@pytest.fixture(scope="module")
def resolver():
    schema, _ok, failed = introspect_nodesets()
    assert not failed, f"nodeset modules failed to import: {failed}"
    return make_resolver(schema)


# ── 1. Wire-type validity against today's port declarations ────────────────


@pytest.mark.parametrize("filename", ["smartway_mono_ce.json", "smartway_ce.json"])
def test_wire_types_fully_resolved(filename: str, resolver) -> None:
    rep = _check_one(_graph_path(filename), resolver)
    assert rep["errors"] == [], f"wire-type mismatches in {filename}: {rep['errors']}"
    assert rep["skipped"] == 0, f"{rep['skipped']} unresolved edges in {filename}"
    assert rep["unresolved_node_types"] == []
    assert rep["checked"] == rep["total_edges"]


# ── 2. Env-side wiring parity: mono == decomposed modulo the method cut ────


def _env_edge_set(path: Path) -> set[tuple]:
    g = json.loads(path.read_text())
    types = {n["id"]: (n.get("data", {}).get("nodeType") or n.get("type")) for n in g["nodes"]}
    edges: set[tuple] = set()
    for e in g["edges"]:
        st, tt = types[e["source"]], types[e["target"]]
        src_env, tgt_env = st.startswith(_ENV_PREFIXES), tt.startswith(_ENV_PREFIXES)
        if not (src_env or tgt_env):
            continue
        # Collapse the method side (node type AND handle) so the two
        # decompositions compare purely on their env-facing surface.
        edges.add(
            (
                st if src_env else "METHOD",
                e.get("sourceHandle") if src_env else None,
                tt if tgt_env else "METHOD",
                e.get("targetHandle") if tgt_env else None,
            )
        )
    return edges


def test_env_wiring_parity_with_decomposed() -> None:
    mono = _env_edge_set(_graph_path("smartway_mono_ce.json"))
    decomp = _env_edge_set(_graph_path("smartway_ce.json"))
    assert mono == decomp, (
        f"env-side wiring diverged:\n"
        f"  only in mono:   {sorted(mono - decomp)}\n"
        f"  only in decomp: {sorted(decomp - mono)}"
    )
