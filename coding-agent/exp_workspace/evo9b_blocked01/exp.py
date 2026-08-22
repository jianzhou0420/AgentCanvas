"""evo9b_blocked01 — eharness-evo first T1 (tool-face) candidate arm.

Fork of exp_workspace/evo9b_g0/ with EXACTLY ONE delta: the student's step()
result surfaces realized motion — moved_m, forward_blocked{requested,blocked},
collided_last (+ a factual note when forwards were blocked) — and the step
tool description says so. The env already measured these (bare nodeset
step_discrete -> info.actual_translation_m / collided); the mini toolset
dropped them at render. Motivation: gen-0 ep0 rerun = 100 calls x [1] into a
wall, all-in ep0 = 100 x [1,1,1,1,1] into the same wall — the 9B cannot infer
"blocked" from pixels, and text telling it to compare views did nothing.
Information supply, no advice (harness supplies, model judges). Briefing is
byte-identical to evo9b_g0. Never edit after boards run.

Run (agentcanvas python; env 9200 up):  python coding-agent/stdrun.py run evo9b_blocked01
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_expp_evo9b_blocked01", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    base = cell("mini", "qwen3.5-9b", "bare")
    spec = replace(base, name="evo9b_blocked01", max_turns=200, exp_dir=str(EXP_DIR),
                   extra=base.extra + (("blocked_signal", 1),))
    CELLS[spec.name] = spec
