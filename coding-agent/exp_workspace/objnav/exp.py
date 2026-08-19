"""objnav — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the ObjectNav single-tool bare
surface (objnav_bridge_singlestep: ONE step(actions) that executes and
returns the resulting forward view, step([]) = look; user decision
2026-08-16, promoted from the 2026-07-22 mechanism experiment) on the two
habitat ObjectNav corpora — profiles hm3d (objectnav_hm3d_v1) and mp3d
(objectnav_mp3d_v1). Post-paper line (never entered the MIP paper).
Migrated from bridges/objnav_bridge_singlestep.py + prompts.py objnav
branch 2026-08-18 with byte-parity; hm3d_*/mp3d_* cell names unchanged.
The pre-park partial runs (hm3d_sdk_fable-5, 38 eps) used the TWO-tool
bridge — a different protocol; never pool them with these boards.

Serve (ac-objnav python):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-objnav/bin/python -m app.server.auto_host \
    --module exp_workspace.objnav.nodeset --class EnvObjnavNodeSet --port 9230
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"
REPO_ROOT = EXP_DIR.parents[2]
SPLITS_DIR = REPO_ROOT / "coding-agent" / "splits"


def _objnav_frozen(benchmark: str, dataset: str, split: str,
                   manifest_stem: str) -> dict:
    manifest = SPLITS_DIR / f"{manifest_stem}.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"{benchmark}: missing split manifest {manifest} — run "
            "coding-agent/scripts/sample_episodes.py --materialize")
    return {
        "benchmark": benchmark,
        "dataset": dataset,
        "split": split,       # mip100: derived dataset-layer split
        "episodes": "0-99",   # the whole MIP100 split, manifest order
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Objnav deviates from std-v2's max_turns=200 (user decision
        # 2026-07-22): 150 turns + an $18/episode USD fuse (cap-burning
        # episodes cost 2-4x a success under full-history resend).
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,  # = habitat's ObjectNav max_episode_steps
        "episode_timeout": 2400,
    }


PROFILES = {
    # hm3d: objectnav_hm3d_v1 (val: 2000 eps / 20 scenes / 6 categories)
    "hm3d": {"cell_prefix": "hm3d", "condition": "hm3d", "batch": "OH",
             "frozen": _objnav_frozen("hm3d", "hm3d_v1", "mip100",
                                      "hm3d_val_n100_seed42")},
    # mp3d: objectnav_mp3d_v1 (val: 2195 eps / 11 scenes / 21 categories)
    "mp3d": {"cell_prefix": "mp3d", "condition": "mp3d", "batch": "OM",
             "frozen": _objnav_frozen("mp3d", "mp3d_v1", "mip100",
                                      "mp3d_val_n100_seed42")},
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

    # the 2026-08-16 opus-5 baseline sweep across all five objnav-family
    # boards (the ovon trio registers in exp_workspace/ovon — batch entries
    # resolve lazily, so this list may span both folders)
    BATCHES["O5N"] = [f"{b}_sdk_opus-5"
                      for b in ("hm3d", "mp3d",
                                "ovon-seen", "ovon-syn", "ovon-unseen")]
