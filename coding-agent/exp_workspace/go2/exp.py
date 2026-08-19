"""go2 — experiment manifest (exp_workspace contract, profile form).

One folder = one METHOD ARM. This one is the bare surface re-embodied on
the real Unitree Go2 (go2_bridge + go2_host) — the MIP paper's §4.8 /
appendix-D deployment line. NOT part of the std board: the instruction is
operator-supplied (--set instruction=...), the driver skips evaluate (no
ground truth — success is judged by a human from the recording), and the
server is go2_host.py on the robot's own machine (folder copy; CycloneDDS
is layer-2, so the SDK cannot run here). No nodeset/ — the host IS the env.

Run:
  # on the robot's machine (see go2_host.py header for flags):
  python exp_workspace/go2/go2_host.py --port 9300
  # here:
  stdrun.py run go2_sdk_fable-5 --servers http://<robot-host>:9300 \
    --set instruction="..."
"""
import importlib.util
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name

# sdk trio, bare surface, default effort (moved verbatim from cells.py).
GO2_BOARD = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))

PROFILES = {
    # No dataset/split — the operator supplies the instruction; the frozen
    # posture (500 steps / 2400 s / std turn cap) rides STD_FROZEN via the
    # cell's benchmark "r2r", exactly as before the migration.
    "go2": {
        "cell_prefix": "go2",
        "condition": "go2",
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
    for _h, _m in GO2_BOARD:
        _base = cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"go2_{_h}_{_m}", condition="go2",
                       go2=True, exp_dir=str(EXP_DIR))
        CELLS[spec.name] = spec
