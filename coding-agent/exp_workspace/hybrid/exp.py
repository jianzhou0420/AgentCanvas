"""hybrid — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the agent-selected hybrid surface
(hybrid_bridge: primitive step 0-3 AND waypoint goto in ONE toolface, the
look-then-move gate making the lens choice the interface choice) — the MIP
paper's §4.5 tab:hybrid arm. Migrated from the shared
bridges/hybrid_bridge.py + prompts.py hybrid branch on 2026-08-18 with
byte-parity; cells keep their historical std_*_hybrid names. mini cells
keep their in-process port (toolset.HybridToolSet, byte gate against the
shared bridge — folder copy and shared file must not drift).

Servers: env auto_host (this folder's nodeset, ac-vlnce python) + the
waypoint predictor (--wp-server), same pair as the wp arm.
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# ── the arm's boards (moved verbatim from cells.py, 2026-08-18) ──
HYBRID_BOARD = (
    ("sdk", "fable-5"), ("sdk", "sonnet-5"),
)
LOCAL_HYBRID_BOARD = (
    ("mini", "qwen3.5-4b"), ("mini", "qwen3.5-9b"),
)

PROFILES = {
    # R2R-CE rand100 (STD_FROZEN fallback, like bare/wp). Hybrid always runs
    # separate (non-auto-observe); WP_MAX_TURNS 100 via the cell.
    "r2r": {
        "cell_prefix": "std",
        "condition": "hybrid",
        "frozen": None,
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    for _h, _m in HYBRID_BOARD:
        spec = replace(cell(_h, _m, "hybrid"), exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
    for _h, _m in LOCAL_HYBRID_BOARD:
        spec = replace(cell(_h, _m, "hybrid"), exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
