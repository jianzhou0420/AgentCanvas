"""slam_r2r_baseline — experiment manifest (exp_workspace contract).

One folder = one experiment: this file declares the frozen knobs and cells;
bridge.py / prompts.py / nodeset/ beside it are this experiment's OWN copies
(execution code is duplicated per experiment, only orchestration is shared).
Never edit this folder after its board runs — fork a new folder instead.

Serve the env with:
  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend \
  <ac-habitat033 python> -m app.server.auto_host \
    --module exp_workspace.slam_r2r_baseline.nodeset \
    --class EnvSlamVlnceNodeSet --port 92xx
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

_FROZEN = {
    "benchmark": "slamr2r",
    "dataset": None,       # one corpus (R2R_VLNCE); split selector only
    "split": "rand100",    # official-rotation rand100 (rebuilt 2026-08-17)
    "episodes": "0-99",
    "max_turns": 200,
    "rgb_resolution": 512,  # recorded, not a knob — SlamEnv renders 512²
    "step_budget": 500,
    "episode_timeout": 2400,
    "instruments": 0,       # arm carried per-cell via extra (this exp: 0/bare)
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    prev = BENCHMARK_FROZEN.get("slamr2r")
    assert prev is None or prev == _FROZEN, "slamr2r frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["slamr2r"] = dict(_FROZEN)
    for _h, _m in sdk_models:
        base = cell(_h, _m, "bare", "default")
        spec = replace(base, name=f"{EXP_NAME}_{_h}_{_m}",
                       condition="slamr2r", benchmark="slamr2r",
                       exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
    BATCHES["SL"] = [f"{EXP_NAME}_sdk_{m}" for m in claude_models]
