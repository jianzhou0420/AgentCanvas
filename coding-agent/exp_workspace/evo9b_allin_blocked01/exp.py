"""evo9b_allin_blocked01 — eharness-evo combined probe arm (user ruling 08-22).

= evo9b_allin01's briefing (bare + SKILLS_BLOCK, cand_01..06 verbatim)
+ evo9b_blocked01's tool-face delta (step result reports moved_m /
forward_blocked / collided_last; blocked_signal=1). TWO declared delta units
in one arm — a deliberate speed-over-attribution probe: "can text habits +
a supplied blocked signal together lift the 9B at all". Attribution (which
of the two carried it) is deferred to leave-one-out arms if this moves.
max_turns=200, STD_FROZEN otherwise, -std pinned sampling. Never edit after
boards run.
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_expp_evo9b_allin_blocked01", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    base = cell("mini", "qwen3.5-9b", "bare")
    spec = replace(base, name="evo9b_allin_blocked01", max_turns=200, exp_dir=str(EXP_DIR),
                   extra=base.extra + (("blocked_signal", 1),))
    CELLS[spec.name] = spec
