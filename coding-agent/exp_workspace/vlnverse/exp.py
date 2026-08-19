"""vlnverse — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the bare two-tool surface on
VLNVerse (Isaac Sim 5.1, 262 kujiale scans behind the nodeset's msgpack RPC
seam) — the MIP paper's appendix-B beyond-R2R-CE line. Same bridge and
briefing text as the bare arm (mcp_bridge copy; the verb namespace comes
from the driver's HABITAT_VERB_PREFIX=env_vlnverse, keyed off the
benchmark), so the surface stays byte-identical to habitat bare.

Serve the env with (ac-vlnverse python — lean env, Isaac spawns its own
bundled python behind the RPC seam; cold boot ~25 s):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-vlnverse/bin/python -m app.server.auto_host \
    --module exp_workspace.vlnverse.nodeset --class EnvVLNVerseNodeSet \
    --port 9260
(env built by scripts/install/install_ac_vlnverse.sh — absent on this
machine 2026-08-18; rebuild before the next vlnverse board.)
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

PROFILES = {
    # PROVISIONAL 口径 (unchanged from the pre-migration frozen block): the
    # first 100 episodes of fine/val_unseen IN SPLIT-FILE ORDER —
    # deterministic but scene-clustered, NOT comparable to the R2R std
    # numbers. Freeze a mip100-style sample (sample_episodes.py needs a
    # vlnverse backend) before this line carries a new paper claim.
    "vlnverse": {
        "cell_prefix": "vlnverse",
        "condition": "vlnverse",
        "frozen": {
            "benchmark": "vlnverse",
            "dataset": "fine",         # fine-grained instructions (the R2R analogue)
            "split": "val_unseen",     # 825 episodes
            "episodes": "0-99",        # split-file order — see PROVISIONAL note
            # Same posture as the R2R std line, not the objnav one: a
            # point-to-point instruction follow takes std-v2's 200 turns.
            "max_turns": 200,
            "rgb_resolution": 512,     # accepted-and-ignored: Isaac renders 1024²
            "step_budget": 500,
            "episode_timeout": 2400,
        },
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    prof = PROFILES["vlnverse"]
    prev = BENCHMARK_FROZEN.get("vlnverse")
    assert prev is None or prev == prof["frozen"], \
        "vlnverse frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["vlnverse"] = dict(prof["frozen"])
    for _h, _m in (("sdk", m) for m in claude_models):
        base = cell(_h, _m, "bare", "default")
        spec = replace(base, name=f"vlnverse_{_h}_{_m}", condition="vlnverse",
                       benchmark="vlnverse", exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
