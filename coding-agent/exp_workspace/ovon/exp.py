"""ovon — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. The same ObjectNav single-tool bare surface as
exp_workspace/objnav (identical bridge/briefing copies), on the HM3D-OVON
corpus — three val-split profiles, each its own board: ovon-seen /
ovon-syn (val_seen_synonyms) / ovon-unseen. Post-paper line. Migrated from
bridges/objnav_bridge_singlestep.py 2026-08-18 with byte-parity; ovon-*
cell names unchanged. No dataset selector on the panel (one dataset — the
split IS the axis) → dataset None.

Serve (ac-objnav python):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-objnav/bin/python -m app.server.auto_host \
    --module exp_workspace.ovon.nodeset --class EnvOvonNodeSet --port 9235
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"
REPO_ROOT = EXP_DIR.parents[2]
SPLITS_DIR = REPO_ROOT / "coding-agent" / "splits"


def _ovon_frozen(benchmark: str, split: str, manifest_stem: str) -> dict:
    manifest = SPLITS_DIR / f"{manifest_stem}.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"{benchmark}: missing split manifest {manifest} — run "
            "coding-agent/scripts/sample_episodes.py --materialize")
    return {
        "benchmark": benchmark,
        "dataset": None,
        "split": split,       # mip100*: derived dataset-layer split
        "episodes": "0-99",   # the whole MIP100 split, manifest order
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same objnav posture: 150 turns + $18/episode USD fuse.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,  # = habitat's ObjectNav max_episode_steps
        "episode_timeout": 2400,
    }


PROFILES = {
    "ovon-seen": {"cell_prefix": "ovon-seen", "condition": "ovon-seen",
                  "batch": "OVS",
                  "frozen": _ovon_frozen("ovon-seen", "mip100_seen",
                                         "ovon_val_seen_n100_seed42")},
    "ovon-syn": {"cell_prefix": "ovon-syn", "condition": "ovon-syn",
                 "batch": "OVY",
                 "frozen": _ovon_frozen("ovon-syn", "mip100_seen_synonyms",
                                        "ovon_val_seen_synonyms_n100_seed42")},
    "ovon-unseen": {"cell_prefix": "ovon-unseen", "condition": "ovon-unseen",
                    "batch": "OVU",
                    "frozen": _ovon_frozen("ovon-unseen", "mip100_unseen",
                                           "ovon_val_unseen_n100_seed42")},
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
            _base = cell(_h, _m, "bare", "default")
            spec = replace(_base, name=f"{bench}_{_h}_{_m}", condition=bench,
                           benchmark=bench, exp_dir=str(EXP_DIR))
            CELLS[spec.name] = spec
        BATCHES[prof["batch"]] = [f"{bench}_sdk_{m}" for m in claude_models]
