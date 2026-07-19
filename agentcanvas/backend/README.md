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
- **Pip-installable** — `pip install -e .` at the repo root → `from agentcanvas import Graph`, no `PYTHONPATH`.

Verified end-to-end on MapGPT-MP3D (1-ep `g.run` + 2-ep `g.eval`).

## Install

`pyproject.toml` lives at the **repo root** (it packages this directory), so a
fresh clone installs with a plain editable install from there:

```bash
pip install -e .                  # from the repo root — exposes `agentcanvas` and `app`
pip install -e ".[server]"        # + server-node proxy & cassette record/replay
pip install -e ".[llm]"           # + the llmCall builtin node
pip install -e ".[backend]"       # + the full FastAPI canvas backend
```

The SDK surface is stdlib-only; the core install needs only `pydantic-settings`
(workspace discovery). Everything heavier is an extra, and full nodeset runs
(torch, simulators) still ride the dedicated conda envs via server mode —
`requirements.txt` provisions the `agentcanvas` env as before. An editable
install resolves `workspace/` from the source tree automatically, so graphs,
nodesets, and cassettes work with zero extra configuration.

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
