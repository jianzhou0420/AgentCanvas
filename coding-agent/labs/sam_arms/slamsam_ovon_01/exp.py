"""slamsam_ovon_01 — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM: HM3D-OVON ObjectNav on jian's SLAM-instrument
surface (auto-observe step + get_pose / get_map / get_trajectory) with our
SAM 3 landmark layer keyed to the ONE goal word and the TARGET PUSH (a
confirmed detection cuts the running step short and hands the model direction
+ distance + the map; every later result carries the target's dir/dist from
the current pose so it can close in under 1 m — user 2026-08-18: "只要 SAM
检测出我要找的这个物体,立刻反馈给 Agent … 让 Agent 无限去贴近这个物体").
Map v1 (jian slam_r2r_01), built by the side-car from the env's measured
motion (NOT ORB-SLAM3). bridge.py / prompts.py / toolset.py /
slam_sidecar.py / slam/ / nodeset/ beside this file are this experiment's OWN
copies. Never edit this folder after its board runs — fork a new folder.

Serve the env with (ac-objnav python, cwd agentcanvas/backend):
  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend OVON_SPLIT=mip100_unseen \
  <ac-objnav python> -m app.server.auto_host \
    --module exp_workspace.slamsam_ovon_01.nodeset \
    --class EnvOvonNodeSet --port 92xx
SAM 3: model_sam auto_host on 9220 (LEAN_SAM_URL overrides the address).
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# The data protocol is jian's ovon-unseen line VERBATIM (cells.OBJNAV_FROZEN:
# mip100_unseen — the scene-stratified seed-42 hundred of val_unseen — 150
# turns + $18 fuse, 500 steps); the register() assert below guarantees this
# folder never drifts from that block.
PROFILES = {
    "ovon-unseen": {
        "cell_prefix": "slamsam_ovon_01",
        "condition": "slamsam_ovon_01",
        "batch": "SSO",
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


FIRST_PROMPT = _prompts.FIRST_PROMPT   # auto-observe: no observe() to call


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    for bench, prof in PROFILES.items():
        assert bench in BENCHMARK_FROZEN, \
            f"{EXP_NAME}: benchmark {bench!r} must be a registered board line"
        for _h, _m in sdk_models:
            base = cell(_h, _m, "bare", "default")
            spec = replace(base, name=f"{prof['cell_prefix']}_{_h}_{_m}",
                           condition=prof["condition"], benchmark=bench,
                           exp_dir=str(EXP_DIR))
            CELLS[spec.name] = spec
        BATCHES[prof["batch"]] = [f"{prof['cell_prefix']}_sdk_{m}"
                                  for m in claude_models]
