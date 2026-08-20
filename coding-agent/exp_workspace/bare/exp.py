"""bare — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the minimal two-tool bare surface
(observe + step over the stdio MCP bridge) — the MIP paper's main arm: the
§4.2 main board (tab:main-results / tab:bare-board), the §4.3 effort ablation
(the *_max tier of the same cells), and the §4.6 long-horizon RxR-bare column.
Migrated from the shared bridges/mcp_bridge.py + prompts.py bare branch on
2026-08-18 with byte-parity (cell specs, briefing, tool schemas); the r2r
profile's cells keep their historical std_* names — only CellSpec.exp_dir is
new. The shared bridges/ dir was dissolved 2026-08-20 (this folder's
bridge.py was cmp-verified identical first); vlnverse carries its own copy.

Serve the env with (ac-vlnce python — py3.8, habitat 0.1.7):
  cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
    --module exp_workspace.bare.nodeset --class EnvHabitatNodeSet --port 9200
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
NODESET_MODULE = f"exp_workspace.{EXP_NAME}.nodeset"

# ── the arm's boards (moved verbatim from cells.py, 2026-08-18) ──
# main board: bare-only, effort-tiered. Each closed harness vs the open mini
# harness on the SAME models — claude side sdk↔mini, openai side codex↔mini.
BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
    ("mini", "sonnet-5"), ("mini", "opus-4.8"), ("mini", "fable-5"),
    ("mini", "gpt-5.5"), ("mini", "gpt-5.6"),
)
EFFORT_TIERS = ("default", "max")
# qwen API flagships — same mini harness as the local qwen column; untier-ed.
QWEN_API_BOARD = (
    ("mini", "qwen3.7-plus"),
    ("mini", "qwen3.6-plus"),
    ("mini", "qwen3.8-plus"),
    ("mini", "qwen3.5-plus"),
)
# Opus 5 probe: three targeted cells, NOT a CLAUDE_MODELS fan-out.
OPUS5_CELLS = (
    ("mini", "opus-5", "bare", "default"),
    ("sdk",  "opus-5", "bare", "default"),
    ("sdk",  "opus-5", "bare", "max"),
)
# open-weight column: locally served, untier-ed, LOCAL_MAX_TURNS cap.
LOCAL_BOARD = (
    ("mini", "qwen3.5-4b", "bare"),
    ("mini", "qwen3.5-9b", "bare"),
    ("mini", "qwen3.8-27b", "bare"),
)
# cross-API seats (2026-08-20): a harness driven through the litellm gateway
# to the OTHER vendor's model — the harness×model decoupling probes. The
# route is the mini column's litellm slug for the same model key; the
# gateway (coding-agent/api/proxy.py) serves it under the harness's native
# wire format. xapi_ prefix, NOT std_: off-board until the seats are vetted.
XAPI_BOARD = (
    ("sdk", "gpt-5.6", "openai/responses/gpt-5.6"),
    ("codex", "opus-4.8", "claude-opus-4-8"),
)
# RxR profile column: the SDK Claude rows the RxR archive runs cover
# (rxr_bare_{sonnet,opus*,fable,opus5} named runs — pre-profile, off-board).
RXR_SDK_MODELS = ("sonnet-5", "opus-4.8", "fable-5", "opus-5")

# method knobs shared by every profile of this arm (r2r values = STD_FROZEN;
# the r2r profile is NOT re-registered into BENCHMARK_FROZEN — benchmark
# "r2r" keeps resolving to cells.STD_FROZEN so the frozen std board stays
# byte-identical, wp/hybrid r2r cells included. This dict documents
# the arm's shared posture and feeds the rxr profile.)
_METHOD = {
    "episodes": "0-99",
    "rgb_resolution": 512,
    "step_budget": 500,
    "episode_timeout": 2400,
}

PROFILES = {
    # R2R-CE rand100, std-v2 protocol (max_turns 200; local qwen cells carry
    # their own 100 cap via CellSpec.max_turns). frozen lives in
    # cells.STD_FROZEN — see _METHOD note above.
    "r2r": {
        "cell_prefix": "std",
        "condition": "bare",
        "frozen": None,  # STD_FROZEN fallback (benchmark "r2r")
    },
    # RxR-CE rand100_en — the paper's §4.6 long-horizon 口径, frozen from the
    # archive runs (rxr_bare_fable_rand100 et al.): max_turns 70, classic
    # observe/step alternation (auto_observe baked OFF per cell — the driver
    # would otherwise default it ON for dataset RxR-CE, a post-paper
    # convenience this frozen arm rejects).
    "rxr": {
        "cell_prefix": "rxr",
        "condition": "bare",
        "batch": "RX",
        "frozen": {"benchmark": "rxr", "dataset": "RxR-CE",
                   "split": "rand100_en", "max_turns": 70, **_METHOD},
    },
}

_spec = importlib.util.spec_from_file_location(f"_expp_{EXP_NAME}", EXP_DIR / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)


def build_briefing(instruction: str, step_budget: int) -> str:
    return _prompts.build_briefing(instruction, step_budget)


def register(*, CELLS, BATCHES, BENCHMARK_FROZEN, cell, replace,
             sdk_models, claude_models) -> None:
    def _own(spec):
        return replace(spec, exp_dir=str(EXP_DIR))

    # r2r profile — the std board, names unchanged, exp_dir now set.
    for _h, _m in BOARD:
        for _t in EFFORT_TIERS:
            spec = _own(cell(_h, _m, "bare", _t))
            CELLS[spec.name] = spec
    for _h, _m in QWEN_API_BOARD:
        spec = _own(cell(_h, _m, "bare"))
        CELLS[spec.name] = spec
    for _h, _m, _c, _t in OPUS5_CELLS:
        spec = _own(cell(_h, _m, _c, _t))
        CELLS[spec.name] = spec
    for _h, _m, _c in LOCAL_BOARD:
        spec = _own(cell(_h, _m, _c))
        CELLS[spec.name] = spec
    for _h, _m, _route in XAPI_BOARD:
        base = cell(_h, _m, "bare", "default")
        spec = replace(base, name=f"xapi_{_h}_{_m}_bare",
                       extra=base.extra + (("api_gateway", "litellm"),
                                           ("gateway_route", _route)),
                       exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec

    # rxr profile — new board seats for the paper's long-horizon 口径.
    prof = PROFILES["rxr"]
    prev = BENCHMARK_FROZEN.get("rxr")
    assert prev is None or prev == prof["frozen"], \
        "rxr frozen knobs diverged across exp folders"
    BENCHMARK_FROZEN["rxr"] = dict(prof["frozen"])
    for _m in RXR_SDK_MODELS:
        base = cell("sdk", _m, "bare", "default")
        spec = replace(base, name=f"rxr_sdk_{_m}_bare_default",
                       benchmark="rxr",
                       extra=base.extra + (("auto_observe", "0"),),
                       exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
    BATCHES[prof["batch"]] = [f"rxr_sdk_{m}_bare_default"
                              for m in RXR_SDK_MODELS]

    # the bare batches (moved verbatim from cells.py, 2026-08-18)
    BATCHES.update({
        # new-model probes: Opus 5 trio and the Qwen3.8 sweep row.
        "O5": ["std_mini_opus-5_bare_default", "std_sdk_opus-5_bare_default",
               "std_sdk_opus-5_bare_max"],
        "Q8": ["std_mini_qwen3.8-plus_bare"],
        # default-effort main experiment (paper main table)
        "Ad": [f"std_sdk_{m}_bare_default" for m in claude_models],
        "Bd": ["std_mini_sonnet-5_bare_default", "std_mini_opus-4.8_bare_default"],  # anthropic key
        "Gd": ["std_mini_gpt-5.5_bare_default", "std_mini_gpt-5.6_bare_default"],    # openai key
        "Xd": ["std_codex_gpt-5.5_bare_default", "std_codex_gpt-5.6_bare_default"],
        # max-effort ablation (already run)
        "A": [f"std_sdk_{m}_bare_max" for m in claude_models],
        "B": ["std_mini_sonnet-5_bare_max", "std_mini_opus-4.8_bare_max"],
        "G": ["std_mini_gpt-5.5_bare_max", "std_mini_gpt-5.6_bare_max"],
        "X": ["std_codex_gpt-5.5_bare_max", "std_codex_gpt-5.6_bare_max"],
        # local GPU, $0. The mini adapter brings ollama up with the context
        # pinned; it refuses to run if it can't.
        "Q": ["std_mini_qwen3.5-4b_bare", "std_mini_qwen3.5-9b_bare"],
    })
