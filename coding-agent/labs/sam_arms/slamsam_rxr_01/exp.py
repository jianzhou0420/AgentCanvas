"""slamsam_rxr_01 — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM (the slamsam lineage, rung 01, map v1): jian's
SLAM-map shape — step primitives + a pulled get_map — with TWO additions of
ours: the async SAM 3 landmark layer painted on the map (a label the body
walks up to and does not see decays), and recall() over the frames the model
already saw. Map = jian's map v1 (slam_r2r_01: OccupancyMap + the
numbered-frontier renderer, ids stable), built by the side-car from the
env's measured motion (NOT ORB-SLAM3 — this line runs on the std VLN-CE
env, habitat-sim 0.1.7, so its numbers pool with the std R2R/RxR boards and
NEVER with jian's slam_* hs0.3.3 boards). bridge.py / prompts.py /
toolset.py / slam_sidecar.py / slam/ / nodeset/ beside this file are this
experiment's OWN copies. Never edit after boards run — fork a new folder.

Serve the env with (ac-vlnce python, cwd agentcanvas/backend):
  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend \
  <ac-vlnce python> -m app.server.auto_host \
    --module exp_workspace.slamsam_rxr_01.nodeset \
    --class EnvHabitatNodeSet --port 92xx
SAM 3: model_sam auto_host on 9220 (LEAN_SAM_URL overrides the address).
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# The frozen protocol of this arm (profile form): its OWN benchmark key so
# the std R2R/RxR boards' cfg is never touched; the verbs fall through to
# env_habitat (driver.env_verb_prefix default).
_METHOD = {
    "episodes": "0-99",
    "max_turns": 200,
    "rgb_resolution": 512,
    "step_budget": 500,
    "episode_timeout": 2400,
}
PROFILES = {
    "slamsamrxr": {
        "cell_prefix": "slamsam_rxr_01",
        "condition": "slamsam_rxr_01",
        "batch": "SSX",
        "frozen": {"benchmark": "slamsamrxr", "dataset": "RxR-CE",
                   "split": "rand100_en", **_METHOD},
    },
}
FROZEN = PROFILES["slamsamrxr"]["frozen"]

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

FIRST_PROMPT = _prompts.FIRST_PROMPT   # auto-observe: no observe() to call


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
