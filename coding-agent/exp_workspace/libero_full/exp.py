"""libero_full — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the LIBERO SENSOR rung: observe adds the wrist view + proprio
readout, step reports measured EE movement, auto-observe carries the
post-move views. Sensors and feedback only — no skills, no planner.
Migrated from bridges/libero_bridge.py + prompts.py libero branch
2026-08-18 with byte-parity; libero_sdk_*_full cell names unchanged. The
folder bridge has this arm's four identity flags BAKED (grep "BAKED" in
bridge.py); all four libero folders assert the SAME shared "libero" frozen.

Serve (ac-libero python, MUJOCO_GL=egl):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent MUJOCO_GL=egl \
  ~/miniforge3/envs/ac-libero/bin/python -m app.server.auto_host \
    --module exp_workspace.libero_full.nodeset --class EnvLiberoNodeSet --port 9250
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"


def _libero_frozen() -> dict:
    return {
        "benchmark": "libero",
        "dataset": None,           # suite rides the split selector (env panel
        "split": "libero_object",  # maps a suite-named split to the suite)
        # Flat episode index over the suite: task_id = k % tasks_per_suite,
        # init state = k // tasks_per_suite — episodes 0-9 run every task once
        # at init state 0; 0-99 is 10 rollouts per task (init states 0-9), the
        # same n=100 shape as the R2R board.
        "episodes": "0-99",
        "tasks_per_suite": 10,  # libero_spatial / object / goal / 10 all have 10
        # Same turn/fuse posture as the hmeqa line (long tool-call episodes,
        # full-history resend cost tail): 150 turns + $18/episode USD fuse.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        # The env's own per-episode cap (_SUITE_MAX_STEPS = 2500 control
        # ticks); the bridge mirrors it to report remaining budget.
        "step_budget": 2500,
        "episode_timeout": 2400,
        "rgb_resolution": 256,  # env_libero server render size (recorded, not a knob)
    }

PROFILES = {
    "libero": {"cell_prefix": "libero", "condition": "libero_full",
               "frozen": _libero_frozen()},
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    prof = PROFILES["libero"]
    prev = BENCHMARK_FROZEN.get("libero")
    assert prev is None or prev == prof["frozen"], \
        "libero frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["libero"] = dict(prof["frozen"])
    for _h, _m in (("sdk", m) for m in claude_models):
        _base = cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"libero_{_h}_{_m}_full",
                       condition="libero_full", benchmark="libero", bare=False,
                       extra=_base.extra + (("auto_observe", 1),),
                       exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
