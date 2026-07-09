# Graph SDK examples

Runnable examples for the Graph SDK (`agentcanvas` / `app.graph_sdk`).
Run each from the backend dir with the package on the path:

```bash
cd agentcanvas/backend
PYTHONPATH=. python examples/01_pure_python.py       # or, after `pip install -e .`, drop PYTHONPATH
```

| # | File | Needs | What it shows |
|---|------|-------|---------------|
| 01 | `01_pure_python.py` | — | Build 3 tiny nodes, run `(7+5)*3=36` in-process. The "hello world". |
| 02 | `02_authoring_sugar.py` | — | `g.loop()` / `g.hook()` / `g.composite()` sugar; serialise to canvas JSON. |
| 03 | `03_mapgpt_from_code.py` | verified JSON | Rebuild the real MapGPT-MP3D graph in code; assert it matches the verified JSON. |
| 04 | `04_run_one_episode.py` | **env + GPU + LLM** | Run **one** MapGPT-MP3D episode in-process via `g.run(load_nodesets=True)`. |
| 05 | `05_batch_eval.py` | **env + GPU + LLM** | Batch-eval N episodes via `g.eval(...)`; print aggregate SR/SPL. |
| 06 | `06_reverse_codegen.py` | verified JSON | Compile a graph JSON back into a builder script (`g.to_code()`); round-trip. |

## ⚠ Examples 04 & 05 are experiments

They spawn the `env_mp3d` simulator (server-mode, in the `ac-mp3d` conda env),
use the GPU (MatterSim CUDA-GL renderer) and the LLM (`gpt-5-mini`). They need
the `ac-mp3d` env, MP3D data on disk, and a `gpt-5-mini` key. A single episode
or a tiny batch is fine to run directly; a real multi-episode sweep is an
**experiment** and belongs behind `/experiment:run` (GPU admission, port
regulation) — not a bare invocation.

The code path is identical either way; the only difference is admission control.
