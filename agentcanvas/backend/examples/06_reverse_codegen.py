"""Example 06 — reverse: compile an existing graph JSON into a builder script.

    PYTHONPATH=. python examples/06_reverse_codegen.py [graph.json]

The inverse of examples 01-03: take any graph (canvas-authored, loaded from
JSON, or code-built) and emit a self-contained Python script that rebuilds it
via the Graph SDK API — roadmap F4. Round-trips: the emitted script's graph is
semantically identical to the input. Defaults to the verified MapGPT-MP3D graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentcanvas import graph_to_code
from app.graph_def import GraphDefinition
from app.mapgpt_mp3d_sdk import VERIFIED_JSON, _diff, signature

if __name__ == "__main__":
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else VERIFIED_JSON
    gd = GraphDefinition.from_dict(json.loads(src_path.read_text()))

    code = graph_to_code(gd)              # or: Graph.from_definition(gd).to_code()
    print(code[:900] + "\n…\n")
    print(f"generated {len(code.splitlines())} lines of builder source from {src_path.name}")

    # prove the round-trip: exec the emitted script, rebuild, compare signatures
    ns: dict = {"__name__": "_gen"}
    exec(compile(code, "<gen>", "exec"), ns)
    rebuilt = ns["build"]().to_dict()
    diffs = _diff(signature(rebuilt), signature(gd.to_dict()))
    assert not diffs, diffs
    print("ROUND-TRIP MATCH — generated script rebuilds the graph exactly.")
