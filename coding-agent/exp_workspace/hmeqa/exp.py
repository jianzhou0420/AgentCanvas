"""hmeqa — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the HM-EQA bare surface
(hmeqa_bridge: observe / step_discrete / answer("A".."D") terminal) with
THREE data profiles riding the one nodeset/bridge/briefing — exactly the
split-agnostic PROFILES shape:
  hmeqa    — explore-eqa mip100 (the board line, MIP appendix B)
  mthm3d   — MemoryEQA's MT-HM3D corpus, mip100 (label-stratified)
  hmeqa500 — the FULL 500-question val set (the paper's appendix-B 76.2
             number ran this 口径 as the named run hmeqa_full500_fable-5;
             this profile gives it a board seat — driver.HMEQA_BENCHMARKS
             carries the family membership)
Migrated from bridges/hmeqa_bridge.py + prompts.py hmeqa branch 2026-08-18
with byte-parity; hmeqa_*/mthm3d_* cell names unchanged.

口径 caveats when reporting: mthm3d answers skew "A" (66/100 in mip100,
67.4% full) — always report the constant-A control beside SR; published
MemoryEQA numbers are on the full 1,587.

Serve (ac-hmeqa python):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-hmeqa/bin/python -m app.server.auto_host \
    --module exp_workspace.hmeqa.nodeset --class EnvHmeqaNodeSet --port 9240
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"
REPO_ROOT = EXP_DIR.parents[2]
SPLITS_DIR = REPO_ROOT / "coding-agent" / "splits"


def _hmeqa_frozen() -> dict:
    manifest = SPLITS_DIR / "hmeqa_val_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"hmeqa: missing split manifest {manifest} — restore it from git "
            "(tracked; sampler coding-agent/scripts/sample_episodes.py)")
    derived = REPO_ROOT / "data" / "hm3d" / "hmeqa" / "questions_mip100.csv"
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"hmeqa: missing materialized split {derived} — regenerate with "
            "sample_episodes.py --benchmark hmeqa --materialize")
    return {
        "benchmark": "hmeqa",
        "dataset": None,       # single-CSV benchmark: no dataset selector
        "split": "mip100",   # dataset-layer derived split (mip semantics)
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same turn/fuse posture as the ObjectNav lines (exploration task,
        # full-history resend cost tail): 150 turns + $18/episode USD fuse.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
        # Camera-tilt actions 4/5 on the toolface (native −30° pitch makes
        # near-overhead fixtures unobservable). Frozen ON for this arm.
        "tilt_actions": True,
    }


def _mthm3d_frozen() -> dict:
    manifest = SPLITS_DIR / "mt_hm3d_val_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"mthm3d: missing split manifest {manifest} — run "
            "coding-agent/scripts/sample_episodes.py --benchmark mt_hm3d --materialize")
    derived = REPO_ROOT / "data" / "hm3d" / "mt_hm3d" / "questions_mip100.csv"
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"mthm3d: missing materialized split {derived} — regenerate with "
            "sample_episodes.py --benchmark mt_hm3d --materialize")
    return {
        "benchmark": "mthm3d",
        "dataset": "mt_hm3d",  # env panel dataset field (second corpus)
        "split": "mip100",
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
        "tilt_actions": True,
    }


def _hmeqa500_frozen() -> dict:
    # The full-500 val 口径 of the archive run hmeqa_full500_fable-5
    # (2026-08: SR 76.2 on n=500) — same knobs, split=val, all rows.
    return {
        "benchmark": "hmeqa500",
        "dataset": None,
        "split": "val",
        "episodes": "0-499",
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
        "tilt_actions": True,
    }


PROFILES = {
    "hmeqa": {"cell_prefix": "hmeqa", "condition": "hmeqa", "batch": "EQ",
              "frozen": _hmeqa_frozen()},
    "mthm3d": {"cell_prefix": "mthm3d", "condition": "mthm3d",
               "frozen": _mthm3d_frozen()},
    "hmeqa500": {"cell_prefix": "hmeqa500", "condition": "hmeqa500",
                 "frozen": _hmeqa500_frozen()},
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

    # hmeqa board: sdk claude trio (names unchanged: hmeqa_sdk_<m>)
    for _h, _m in (("sdk", m) for m in claude_models):
        _base = cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"hmeqa_{_h}_{_m}", condition="hmeqa",
                       benchmark="hmeqa", exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
    BATCHES["EQ"] = [f"hmeqa_sdk_{m}" for m in claude_models]

    # mthm3d board: the objnav sdk column (trio + opus-5), names unchanged
    for _h, _m in sdk_models:
        _base = cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"mthm3d_{_h}_{_m}", condition="mthm3d",
                       benchmark="mthm3d", exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec

    # hmeqa500: the one seat the paper number occupies (fable only)
    _base = cell("sdk", "fable-5", "bare", "default")
    spec = replace(_base, name="hmeqa500_sdk_fable-5", condition="hmeqa500",
                   benchmark="hmeqa500", exp_dir=str(EXP_DIR))
    CELLS[spec.name] = spec
