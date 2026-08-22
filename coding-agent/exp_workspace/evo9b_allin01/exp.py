"""evo9b_allin01 — eharness-evo M1 all-in probe arm (exp_workspace contract).

Fork of exp_workspace/evo9b_g0/ with EXACTLY ONE delta: the briefing appends
SKILLS_BLOCK (cand_01..06 verbatim; cand_07 held out — see prompts.py).
Everything else byte-inherits the parent: mini + qwen3.5-9b, max_turns=200,
STD_FROZEN knobs, -std pinned sampling. Gate: labs/evolution/paired_gate.py
vs evo9b_g0 on the same rand100 episodes. Never edit after boards run.

Run (agentcanvas python; env 9200 up like the bare arm):
  python coding-agent/stdrun.py run evo9b_allin01
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_expp_evo9b_allin01", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    base = cell("mini", "qwen3.5-9b", "bare")
    spec = replace(base, name="evo9b_allin01", max_turns=200, exp_dir=str(EXP_DIR))
    CELLS[spec.name] = spec
