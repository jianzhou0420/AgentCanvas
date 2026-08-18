"""slam_01 — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM (this one: ORB-SLAM3 instruments v1 —
get_map/frontier/trajectory over the bare toolface). The data口径
(benchmark / corpus / split) is NOT part of the folder identity any more:
each entry in PROFILES declares one benchmark with its own frozen knobs and
cell-name prefix, all sharing this folder's bridge/prompts/nodeset code.
Cell name = run dir = lineage name still holds — per profile.

Serve the env with:
  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend \
  <ac-habitat033 python> -m app.server.auto_host \
    --module exp_workspace.slam_01.nodeset \
    --class EnvSlamVlnceNodeSet --port 92xx
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# method knobs shared by every profile of this arm
_METHOD = {
    "episodes": "0-99",
    "max_turns": 200,
    "rgb_resolution": 512,  # recorded, not a knob — SlamEnv renders 512²
    "step_budget": 500,
    "episode_timeout": 2400,
    "instruments": 0,       # arm carried per-cell via extra (this exp: 1)
}

PROFILES = {
    # R2R-CE rand100 (official rotations, rebuilt 2026-08-17). dataset=None
    # keeps the pre-profile behavior byte-for-byte: the driver pushes no
    # dataset field and the nodeset defaults to the r2r corpus.
    "slamr2r": {
        "cell_prefix": "slam_r2r_01",
        "condition": "slamr2r_instr",
        "batch": "SLI",
        "frozen": {"benchmark": "slamr2r", "dataset": None,
                   "split": "rand100", **_METHOD},
    },
    # RxR-CE rand100 (rand100_en canon, rebuilt from run records 2026-08-18):
    # dataset="rxr" reaches the env panel and switches the nodeset's corpus
    # table (RxR_VLNCE_v0, {split}_guide.json.gz naming).
    "slamrxr": {
        "cell_prefix": "slam_rxr_01",
        "condition": "slamrxr_instr",
        "batch": "SXI",
        "frozen": {"benchmark": "slamrxr", "dataset": "rxr",
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
                           extra=base.extra + (("instruments", 1),),
                           exp_dir=str(EXP_DIR))
            CELLS[spec.name] = spec
        BATCHES[prof["batch"]] = [f"{prof['cell_prefix']}_sdk_{m}"
                                  for m in claude_models]
