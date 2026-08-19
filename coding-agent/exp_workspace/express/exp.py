"""express — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the EXPRESS-Bench free-form-EQA
surface (express_bridge: observe / step_discrete / answer(text); scored
driver-side by the benchmark's gpt-4o-mini judge over answer + final frame
→ C / C* / E_path / d_T). Post-paper line (not in the MIP paper). Migrated
from bridges/express_bridge.py + prompts.py express branch 2026-08-18 with
byte-parity; express_* cell names unchanged.

The judge needs OPENAI_API_KEY in the DRIVER's environment. Serve
(ac-hmeqa python, EXPRESS_PYTHON override honoured by the nodeset):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-hmeqa/bin/python -m app.server.auto_host \
    --module exp_workspace.express.nodeset --class EnvExpressNodeSet --port 9245
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"
REPO_ROOT = EXP_DIR.parents[2]
SPLITS_DIR = REPO_ROOT / "coding-agent" / "splits"


def _express_frozen() -> dict:
    manifest = SPLITS_DIR / "express_all_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"express: missing split manifest {manifest} — run "
            "coding-agent/scripts/sample_episodes.py --benchmark express --materialize")
    derived = (REPO_ROOT / "data" / "hm3d" / "express_bench"
               / "express-bench_mip100.json")
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"express: missing materialized split {derived} — regenerate "
            "with sample_episodes.py --benchmark express --materialize")
    return {
        "benchmark": "express",
        "dataset": None,       # single-JSON benchmark: no dataset selector
        "split": "mip100",
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same posture as the hmeqa/mthm3d lines: 150 turns + $18 fuse,
        # std 500-step budget enforced bridge-side.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
    }


PROFILES = {
    "express": {"cell_prefix": "express", "condition": "express",
                "frozen": _express_frozen()},
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    prof = PROFILES["express"]
    prev = BENCHMARK_FROZEN.get("express")
    assert prev is None or prev == prof["frozen"], \
        "express frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["express"] = dict(prof["frozen"])
    for _h, _m in sdk_models:
        _base = cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"express_{_h}_{_m}", condition="express",
                       benchmark="express", exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
