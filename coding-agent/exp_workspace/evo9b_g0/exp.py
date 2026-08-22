"""evo9b_g0 — eharness-evo generation-0 student arm (exp_workspace contract).

One folder = one METHOD ARM. This is the evolution line's parent/baseline:
Qwen3.5-9B on the mini (open ReAct) harness with the byte-identical bare
briefing (prompts.py copied verbatim from exp_workspace/bare/), and the
TEACHER turn cap (max_turns=200, user ruling 08-22 "和老师们设置一样,200turn")
instead of the std local cap of 100. Sampling is pinned in the -std Modelfile
(seed 0, factory temp 1.0/top_p .95/top_k 20); the mini adapter reads it back
from /api/show and refuses to run unpinned.

Everything else = STD_FROZEN (rand100 0-99, 512 px, 500 actions, 2400 s).
Never edit after boards run — evolution forks this folder (one atomic delta
per child arm; see labs/evolution/m1_candidates/).

Serve the env exactly like the bare arm (ac-vlnce python, port 9200):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
    --module exp_workspace.bare.nodeset --class EnvHabitatNodeSet --port 9200

Run (agentcanvas python):
  python coding-agent/stdrun.py run evo9b_g0
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_expp_evo9b_g0", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    base = cell("mini", "qwen3.5-9b", "bare")
    spec = replace(base, name="evo9b_g0", max_turns=200, exp_dir=str(EXP_DIR))
    CELLS[spec.name] = spec
