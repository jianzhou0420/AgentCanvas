"""slamsam_objnav_02 — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM: ObjectNav on jian's SLAM-instrument surface
(auto-observe step + get_pose / get_map / get_trajectory) with SAM 3 as a
TOOL — detect_target() runs the detector for the exact goal word on the view
the model is looking at and returns the overlay + direction / distance / score
per instance (score ≥ 0.85 only); only those high-score matches are stamped
on the map; nothing watches the frames while walking; on STOP the harness
runs a FINAL PUSH toward the detected target (~0.5 m) before stopping (user
2026-08-18: "把 SAM 变成一个工具 … 纯 ReAct 的模式 … pillow is not cushion, 严格
地去检测, 就只有这一个词"; "置信度只有在 0.8 或者 0.85 以上才可以"; "只有在它的
score 很高的情况下再标"; "人为地把这个往前推, 离目标足够近"). Successor of
slamsam_ovon_01 (the streaming push arm, OVON only), widened to the three
ObjectNav boards — one folder, three profiles, the SAME code:
  hm3d         objectnav_hm3d_v1  mip100          env_objnav (nodeset_objnav/)
  mp3d         objectnav_mp3d_v1  mip100          env_objnav (nodeset_objnav/)
  ovon-unseen  HM3D-OVON          mip100_unseen   env_ovon   (nodeset_ovon/)
Map v1 (jian slam_r2r_01), built by the side-car from the env's measured
motion (NOT ORB-SLAM3). bridge.py / prompts.py / toolset.py / slam_sidecar.py /
slam/ / nodeset_ovon/ / nodeset_objnav/ beside this file are this
experiment's OWN copies. Never edit this folder after its boards run — fork a
new folder.

Serve the env with (ac-objnav python, cwd agentcanvas/backend):
  hm3d:  PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend \
         OBJNAV_DATASET=hm3d_v1 OBJNAV_SPLIT=mip100 <ac-objnav python> -m app.server.auto_host \
           --module exp_workspace.slamsam_objnav_02.nodeset_objnav \
           --class EnvObjnavNodeSet --port 92xx
  mp3d:  same with OBJNAV_DATASET=mp3d_v1
  ovon:  PYTHONPATH=… OVON_SPLIT=mip100_unseen <ac-objnav python> -m app.server.auto_host \
           --module exp_workspace.slamsam_objnav_02.nodeset_ovon \
           --class EnvOvonNodeSet --port 92xx
SAM 3: model_sam auto_host on 9220 (LEAN_SAM_URL overrides the address).
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULES = {
    "hm3d": f"exp_workspace.{EXP_NAME}.nodeset_objnav",
    "mp3d": f"exp_workspace.{EXP_NAME}.nodeset_objnav",
    "ovon-unseen": f"exp_workspace.{EXP_NAME}.nodeset_ovon",
}

# The data protocol of every profile is the board line VERBATIM
# (cells.OBJNAV_FROZEN: mip100 / mip100_unseen — the scene-stratified seed-42
# hundred of val — 150 turns + $18 fuse, 500 steps); the register() assert
# below guarantees this folder never drifts from those blocks.
PROFILES = {
    "hm3d": {
        "cell_prefix": "slamsam_hm3d_02",
        "condition": "slamsam_objnav_02",
        "batch": "SSH2",
    },
    "mp3d": {
        "cell_prefix": "slamsam_mp3d_02",
        "condition": "slamsam_objnav_02",
        "batch": "SSM2",
    },
    "ovon-unseen": {
        "cell_prefix": "slamsam_ovon_02",
        "condition": "slamsam_objnav_02",
        "batch": "SSO2",
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
