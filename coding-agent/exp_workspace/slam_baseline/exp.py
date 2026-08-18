"""slam_baseline — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM (this one: the bare MIP toolface, no SLAM
instruments). The data口径 (benchmark / corpus / split) is declared
per-profile below; cell name = run dir = lineage name holds per profile.

Serve the env with:
  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend \
  <ac-habitat033 python> -m app.server.auto_host \
    --module exp_workspace.slam_baseline.nodeset \
    --class EnvSlamVlnceNodeSet --port 92xx
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

_METHOD = {
    "episodes": "0-99",
    "max_turns": 200,
    "rgb_resolution": 512,  # recorded, not a knob — SlamEnv renders 512²
    "step_budget": 500,
    "episode_timeout": 2400,
    "instruments": 0,       # arm carried per-cell via extra (this exp: 0/bare)
}

PROFILES = {
    "slamr2r": {
        "cell_prefix": "slam_r2r_baseline",
        "condition": "slamr2r",
        "batch": "SL",
        "frozen": {"benchmark": "slamr2r", "dataset": None,
                   "split": "rand100", **_METHOD},
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    for bench, prof in PROFILES.items():
        prev = BENCHMARK_FROZEN.get(bench)
        assert prev is None or prev == prof["frozen"], \
            f"{bench} frozen knobs diverged across exp folders"
        BENCHMARK_FROZEN[bench] = dict(prof["frozen"])
        for _h, _m in sdk_models:
            base = cell(_h, _m, "bare", "default")
            spec = replace(base, name=f"{prof['cell_prefix']}_{_h}_{_m}",
                           condition=prof["condition"], benchmark=bench,
                           exp_dir=str(EXP_DIR))
            CELLS[spec.name] = spec
        BATCHES[prof["batch"]] = [f"{prof['cell_prefix']}_sdk_{m}"
                                  for m in claude_models]
