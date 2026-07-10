"""Example 03 — rebuild the verified MapGPT-MP3D graph in code, prove faithful.

    PYTHONPATH=. python examples/03_mapgpt_from_code.py

Builds the real MapGPT-MP3D navigation graph (22 nodes / 49 wires, an
iterIn/iterOut episode loop, a state container + access grants) purely through
the Graph SDK API, then checks its semantic signature against the
hand-authored, verified canvas JSON. No env / GPU — build + compare only.
"""

from __future__ import annotations

import json

from app.graph_def import GraphDefinition
from app.mapgpt_mp3d_sdk import VERIFIED_JSON, _diff, build, signature

if __name__ == "__main__":
    g = build()
    built = g.to_dict()
    orig = GraphDefinition.from_dict(json.loads(VERIFIED_JSON.read_text())).to_dict()

    print(f"code-built: {len(built['nodes'])} nodes, {len(built['edges'])} edges")
    diffs = _diff(signature(built), signature(orig))
    if diffs:
        print("MISMATCH:", *diffs, sep="\n  - ")
        raise SystemExit(1)
    print("MATCH — code-built graph is semantically identical to the verified JSON.")
    print("Save it with: g.save('mapgpt_from_code.json')  # opens unchanged in the canvas")
