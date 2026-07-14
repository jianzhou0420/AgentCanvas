"""std-v1 cell registry — the standard board as code.

A cell is one run: harness × model × condition, with every frozen knob
pinned (see docs/pages/developer-guide/tmp/coding-agent/standard-experiments.html).
The runner takes cell names, not free-form flags; deviating from the freeze
requires --nonstd, which renames the run so it can never sit on the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── frozen configuration (std-v1; change anything → that's std-v2) ──
STD_FROZEN: dict = {
    "dataset": "R2R-CE",
    "split": "rand100",
    "episodes": "0-49",
    "max_turns": 80,
    "rgb_resolution": 224,
    "step_budget": 500,
    "episode_timeout": 2400,
}

# model key (board column) → model id passed to the harness
MODELS = {
    "sonnet-5": "claude-sonnet-5",
    "opus-4.8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "gpt-5.5": "gpt-5.5",  # codex appendix column
    # local appendix column: served by a user-space ollama (>=0.32) on a
    # dedicated port so the machine's own ollama service stays untouched
    "qwen3-vl-8b": "ollama_chat/qwen3-vl:8b",
}

# per-model default knobs, recorded into the run config. image_window is the
# local column's registered deviation: 80-turn full-history multimodal
# overflows a local model's context, so only the newest K frames ride the
# payload (cloud cells run image_window=0 — never compare across this).
MODEL_EXTRA: dict[str, dict] = {
    "qwen3-vl-8b": {"api_base": "http://127.0.0.1:11435", "image_window": 4},
}

CONDITIONS = {
    "bare": {"bare": True, "skill": None},
    "nav": {"bare": False, "skill": "ledger-nav"},
}

# harness key → output root (the Monitor's SOURCE_ROOTS, unchanged)
OUTPUT_ROOTS = {
    "sdk": REPO_ROOT / "outputs" / "beta-coding-agent",
    "mini": REPO_ROOT / "outputs" / "beta-react-harness",
    "codex": REPO_ROOT / "outputs" / "beta-codex-agent",
}


@dataclass(frozen=True)
class CellSpec:
    name: str          # std_sdk_opus-4.8_bare
    harness: str       # sdk | mini | codex
    model_key: str     # board column
    model_id: str      # harness-facing model string
    condition: str     # bare | nav
    bare: bool
    skill: str | None
    extra: tuple = ()  # model-default knobs as (key, value) pairs (hashable)

    @property
    def extra_dict(self) -> dict:
        return dict(self.extra)

    @property
    def output_root(self) -> Path:
        return OUTPUT_ROOTS[self.harness]

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.name


def _cell(harness: str, model_key: str, condition: str) -> CellSpec:
    cond = CONDITIONS[condition]
    return CellSpec(
        name=f"std_{harness}_{model_key}_{condition}",
        harness=harness,
        model_key=model_key,
        model_id=MODELS[model_key],
        condition=condition,
        bare=cond["bare"],
        skill=cond["skill"],
        extra=tuple(sorted(MODEL_EXTRA.get(model_key, {}).items())),
    )


CLAUDE_MODELS = ("sonnet-5", "opus-4.8", "fable-5")

CELLS: dict[str, CellSpec] = {}
for _h in ("sdk", "mini"):
    for _m in CLAUDE_MODELS:
        for _c in ("bare", "nav"):
            spec = _cell(_h, _m, _c)
            CELLS[spec.name] = spec
# appendix columns (outside the 12-cell board; same freeze applies):
# codex = OpenAI's closed harness; qwen3-vl-8b = locally served via mini
for _c in ("bare", "nav"):
    for _h, _m in (("codex", "gpt-5.5"), ("mini", "qwen3-vl-8b")):
        spec = _cell(_h, _m, _c)
        CELLS[spec.name] = spec

BATCHES = {
    "A": [f"std_sdk_{m}_{c}" for m in CLAUDE_MODELS for c in ("bare", "nav")],
    "B": [f"std_mini_{m}_{c}" for m in ("sonnet-5", "opus-4.8") for c in ("bare", "nav")],
    "C": [f"std_mini_fable-5_{c}" for c in ("bare", "nav")],
    "X": [f"std_codex_gpt-5.5_{c}" for c in ("bare", "nav")],       # appendix: codex
    "L": [f"std_mini_qwen3-vl-8b_{c}" for c in ("bare", "nav")],    # appendix: local
}


def get_cell(name: str) -> CellSpec:
    if name not in CELLS:
        known = "\n  ".join(sorted(CELLS))
        raise KeyError(f"unknown cell {name!r}; known cells:\n  {known}")
    return CELLS[name]
