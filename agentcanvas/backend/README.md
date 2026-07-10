# AgentCanvas — Graph SDK

Build and run [AgentCanvas](https://github.com/jianzhou0420/AgentCanvas)
VLN-agent graphs **in Python code**, LangGraph-style — no graph JSON, no canvas
GUI required. The same `GraphDefinition` a code-built graph produces also opens
unchanged in the canvas, so code ⇄ JSON ⇄ canvas is fully reversible.

## What's New

**2026-07-09 · v0.1 — first Graph SDK release.** The code-first surface grew from
a build-only PoC into a full SDK:

- **Run nodeset graphs in-process** — `g.run()` auto-loads the graph's nodesets
  (spawning env-server subprocesses, tearing them down after), so real graphs
  like MapGPT-MP3D run straight from Python, not only via the backend.
- **Batch eval from code** — `g.eval(episodes=N, dataset=…, split=…)` drives the
  same `BatchEvalRunner` the backend uses; metrics averaged over completed episodes.
- **Reverse codegen** — `graph_to_code()` / `Graph.to_code()` compiles any graph
  back into a standalone builder script (round-trips exactly).
- **Authoring sugar** — `g.loop()` / `g.hook()` / `g.composite()`.
- **Pip-installable** — `pip install -e .` → `from agentcanvas import Graph`, no `PYTHONPATH`.

Verified end-to-end on MapGPT-MP3D (1-ep `g.run` + 2-ep `g.eval`).

## Install

The SDK surface is stdlib-only and importable as soon as the package is on the
path. The heavy runtime dependencies a real *nodeset* run needs (FastAPI,
torch, litellm, simulators) are provisioned by the `agentcanvas` conda env via
`requirements.txt` — kept out of `pyproject.toml` so the SDK stays light.

```bash
pip install -e .          # exposes `agentcanvas` and `app`
```

## Build & run

```python
from agentcanvas import Graph

g = Graph(name="demo")
src = g.add("const_source", value=7)
inc = g.add("increment")
out = g.graph_out("result")
g.connect(src.out("value"), inc.in_("x"))
g.connect(inc.out("y"),     out.in_("value"))

result = g.run()           # in-process, no backend
print(result["result"])    # 8
```

Pure-Python / builtin graphs run with a bare `g.run()`. **Nodeset** graphs
(`mapgpt__*`, `env_mp3d__*`, …) run too — `g.run()` defaults to
`load_nodesets="auto"`, scanning the workspace registry and auto-loading (and
tearing down) every nodeset the graph needs, spawning server-mode subprocesses
for env nodesets exactly as the backend does. A real multi-episode env run is
still an *experiment* and belongs behind `/experiment:run`.

## Ergonomics

| Helper | What it does |
|--------|--------------|
| `g.loop(init=…, carry=…)` | iterIn/iterOut episode loop with correct `persist`; wire via `loop.seed / feed / carry / stop`. |
| `g.hook(event, command)` | Lifecycle shell hooks (`GraphStart`, `PreNodeExecute`, …). |
| `g.composite(id, subgraph)` | Nested-subgraph composite node (flattened before execution). |
| `g.to_dict()` / `g.save(path)` | Serialise to canvas graph JSON. |
| `Graph.from_definition(gd)` | Wrap a loaded `GraphDefinition`. |
| `g.to_code()` | **Inverse** — compile any graph back into a standalone builder script. |

See `app/graph_sdk.py` and `app/graph_sdk_codegen.py` for the full API, and
`app/mapgpt_mp3d_sdk.py` for a real graph rebuilt in code.
