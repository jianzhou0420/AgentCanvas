"""wp — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the waypoint-selection surface
(wp_bridge: depth-predicted candidates numbered on a 4-view panorama, the
agent picks goto/stop) — the MIP paper's §4.4 interface ablation column and
the §4.6 long-horizon RxR-wp column. Migrated from the shared
bridges/wp_bridge.py + prompts.py wp branch on 2026-08-18 with byte-parity;
the r2r profile's cells keep their historical std_*_wp names. mini cells
keep their in-process port (harnesses/mini/toolset.py WaypointToolSet, byte
gate check_equivalence.py against the shared bridge — folder copy and
shared file must not drift).

Needs TWO servers: the env auto_host (this folder's nodeset, ac-vlnce
python — same launch as exp_workspace/bare) and the waypoint predictor
(--wp-server, smartway_waypoint auto_host on :9210 in the ac-wp env over
this folder's ac_wp_predictor_shim/ tree — owned by the wp arm since
2026-08-19; the hybrid/imagine arms launch the same tree; see
coding-agent/README.md).
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# ── the arm's boards (moved verbatim from cells.py, 2026-08-18) ──
# sdk + codex reach the env through the stdio bridge; the mini column rides
# toolset.WaypointToolSet (in-process port, gated by check_equivalence.py).
WP_BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "opus-5"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
)
LOCAL_WP_BOARD = (
    ("mini", "qwen3.5-4b", "wp"),
    ("mini", "qwen3.5-9b", "wp"),
)
API_WP_BOARD = (
    ("mini", "qwen3.7-plus", "wp"),
    ("mini", "qwen3.6-plus", "wp"),
    ("mini", "qwen3.5-plus", "wp"),
)
# RxR profile column: mirrors the bare arm's rxr seats.
RXR_SDK_MODELS = ("sonnet-5", "opus-4.8", "fable-5", "opus-5")

PROFILES = {
    # R2R-CE rand100 (STD_FROZEN fallback — benchmark "r2r" is NOT
    # re-registered, exactly like exp_workspace/bare). wp_max_moves 30 /
    # WP_MAX_TURNS 100 come from the orchestration layer as before.
    "r2r": {
        "cell_prefix": "std",
        "condition": "wp",
        "frozen": None,
    },
    # RxR-CE rand100_en — the paper's §4.6 RxR-wp 口径, frozen from the
    # tab:long-horizon run (rxr_wp_fable_rand100): max_turns 300,
    # wp_max_moves 45, classic observe/move alternation (auto_observe baked
    # OFF per cell). The benchmark-level "rxr" frozen dict is the BARE arm's
    # (loader asserts identity across folders); this arm's own caps ride the
    # cell (max_turns) and the cell extra (wp_max_moves), which override it.
    # The opus-5/sonnet RxR-wp archive runs used OTHER caps (mt150/moves50)
    # — different protocol, they stay off-board.
    "rxr": {
        "cell_prefix": "rxr",
        "condition": "wp",
        "batch": "RXW",
        "frozen": {"benchmark": "rxr", "dataset": "RxR-CE",
                   "split": "rand100_en", "max_turns": 70, "episodes": "0-99",
                   "rgb_resolution": 512, "step_budget": 500,
                   "episode_timeout": 2400},
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int, *,
                   wp_max_moves: int = 30) -> str:
    return _prompts.build_briefing(instruction, step_budget,
                                   wp_max_moves=wp_max_moves)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    def _own(spec):
        return replace(spec, exp_dir=str(EXP_DIR))

    # r2r profile — the std wp column, names unchanged, exp_dir now set.
    for _h, _m in WP_BOARD:
        spec = _own(cell(_h, _m, "wp"))
        CELLS[spec.name] = spec
    for _h, _m, _c in LOCAL_WP_BOARD:
        spec = _own(cell(_h, _m, _c))
        CELLS[spec.name] = spec
    for _h, _m, _c in API_WP_BOARD:
        spec = _own(cell(_h, _m, _c))
        CELLS[spec.name] = spec

    # rxr profile — new board seats for the paper's long-horizon wp 口径.
    prof = PROFILES["rxr"]
    prev = BENCHMARK_FROZEN.get("rxr")
    assert prev is None or prev == prof["frozen"], \
        "rxr frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["rxr"] = dict(prof["frozen"])
    for _m in RXR_SDK_MODELS:
        base = cell("sdk", _m, "wp")
        spec = replace(base, name=f"rxr_sdk_{_m}_wp",
                       benchmark="rxr", max_turns=300,
                       extra=base.extra + (("auto_observe", "0"),
                                           ("wp_max_moves", "45")),
                       exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
    BATCHES[prof["batch"]] = [f"rxr_sdk_{m}_wp" for m in RXR_SDK_MODELS]

    # the wp batches (moved verbatim from cells.py, 2026-08-18)
    BATCHES.update({
        # waypoint pilots (needs --wp-server; see coding-agent/README.md)
        "W": ["std_sdk_sonnet-5_wp", "std_sdk_fable-5_wp"],
        # open-weight waypoint pilots, local GPU $0 (needs --wp-server)
        "WQ": ["std_mini_qwen3.5-4b_wp", "std_mini_qwen3.5-9b_wp"],
    })
