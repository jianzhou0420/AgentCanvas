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

# model key (board column) → model id passed to the harness.
# gpt slugs are identical on the codex CLI and litellm's openai route;
# gpt-5.6 on codex needs CLI > 0.142 (upgrade + re-probe before batch X).
MODELS = {
    "sonnet-5": "claude-sonnet-5",
    "opus-4.8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.6": "gpt-5.6",
}

# per-model default knobs, recorded into the run config (e.g. a local model's
# api_base + image_window). Empty on the current board.
MODEL_EXTRA: dict[str, dict] = {}

CONDITIONS = {
    "bare": {"bare": True, "skill": None},
    "nav": {"bare": False, "skill": "ledger-nav"},
    # ablation: bare tool surface + BARE briefing, but the stock Claude Code
    # persona is KEPT (preset system prompt, briefing appended instead of
    # replacing). sdk-only — mini has no persona; codex can't drop its own.
    "persona": {"bare": True, "skill": None, "persona": True},
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
    condition: str     # bare | nav | persona
    bare: bool
    skill: str | None
    persona: bool = False  # keep the harness's stock persona (ablation)
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
        persona=cond.get("persona", False),
        extra=tuple(sorted(MODEL_EXTRA.get(model_key, {}).items())),
    )


CLAUDE_MODELS = ("sonnet-5", "opus-4.8", "fable-5")

# current board: bare-only (nav / persona conditions stay defined above but
# unregistered for now). Design: each closed harness vs the open mini harness
# on the SAME models — claude side sdk↔mini (sonnet/opus; fable is sdk-only),
# openai side codex↔mini (gpt-5.5 / gpt-5.6).
BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
    ("mini", "sonnet-5"), ("mini", "opus-4.8"),
    ("mini", "gpt-5.5"), ("mini", "gpt-5.6"),
)

CELLS: dict[str, CellSpec] = {}
for _h, _m in BOARD:
    spec = _cell(_h, _m, "bare")
    CELLS[spec.name] = spec

BATCHES = {
    "A": [f"std_sdk_{m}_bare" for m in CLAUDE_MODELS],
    "B": ["std_mini_sonnet-5_bare", "std_mini_opus-4.8_bare"],      # API key: anthropic
    "G": ["std_mini_gpt-5.5_bare", "std_mini_gpt-5.6_bare"],        # API key: openai
    "X": ["std_codex_gpt-5.5_bare", "std_codex_gpt-5.6_bare"],      # 5.6 needs CLI upgrade
}


def get_cell(name: str) -> CellSpec:
    if name not in CELLS:
        known = "\n  ".join(sorted(CELLS))
        raise KeyError(f"unknown cell {name!r}; known cells:\n  {known}")
    return CELLS[name]
