"""evo9b_g2 — eharness-evo generation 2 (engineer round after the P7 diagnosis).

Parent: evo9b_allin_blocked01. Delta set (combined, speed-over-attribution):
  T1  turn macros  — step accepts 4/5/6 (L90/R90/turn-around) + reports turned_deg
                     (EP3: "turn around" executed as 8 lefts = 120°, looped 300 steps)
  T1+T4 remember() — the student saves one-line facts; notes echoed in every
                     step/observe result (EP3: saw the bed through the right
                     doorway at step 185, lost it one call later)
  T1  revisit hint — ahash of observe frames; "this view matches one you saw
                     N steps ago" (EP2 ±90° ping-pong, EP3 dead-end loop)
  T2  skill 5 rewritten for macros, skill 7 added (notes/revisit usage)
Inherits blocked_signal. All organs are information supply or model-authored
facts — no harness judgment. max_turns=200, STD_FROZEN otherwise.
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_expp_evo9b_g2", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    base = cell("mini", "qwen3.5-9b", "bare")
    spec = replace(base, name="evo9b_g2", max_turns=200, exp_dir=str(EXP_DIR),
                   extra=base.extra + (("blocked_signal", 1), ("turn_macros", 1),
                                       ("memo", 1), ("revisit", 1)))
    CELLS[spec.name] = spec
