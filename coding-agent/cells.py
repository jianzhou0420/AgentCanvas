"""std-v2 cell registry — the standard board as code.

A cell is one run: harness × model × condition, with every frozen knob
pinned (see docs/pages/developer-guide/tmp/coding-agent/standard-experiments.html).
The runner takes cell names, not free-form flags; deviating from the freeze
requires --nonstd, which renames the run so it can never sit on the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── frozen configuration (std-v2; change anything → that's std-v3) ──
# std-v2 (user decision 2026-07-15): rgb 224→512, max_turns 80→200.
# std-v1 ep0 smokes (224/80) are archived under <output_root>/archive/.
STD_FROZEN: dict = {
    "dataset": "R2R-CE",
    "split": "rand100",
    "episodes": "0-99",  # full SmartWay sample (user decision 2026-07-16); was 0-49
    "max_turns": 200,
    "rgb_resolution": 512,
    "step_budget": 500,
    "episode_timeout": 2400,
}

# model key (board column) → model id passed to the harness.
# gpt slugs are USUALLY identical on the codex CLI and litellm's openai route —
# but NOT for gpt-5.6. Probed 2026-07-17 (codex 0.144.5): plain "gpt-5.6" 400s on
# a ChatGPT account ("The 'gpt-5.6' model is not supported when using Codex with a
# ChatGPT account" — an ACCOUNT-ENTITLEMENT gate, not a CLI-version issue; the old
# "needs CLI > 0.142" guess is disproven). The account-specific variant
# "gpt-5.6-sol" DOES run on codex. mini/litellm (OPENAI_API_KEY) uses plain
# "gpt-5.6". So the concrete slug resolves per (harness, model_key), see
# MODEL_ID_OVERRIDE. Caveat: whether "gpt-5.6-sol" is bit-identical to "gpt-5.6"
# or a codex-delivery variant is unverified — the E5(codex)↔E9(mini) comparison
# carries that slug asymmetry; the run config records the real slug for audit.
MODELS = {
    "sonnet-5": "claude-sonnet-5",
    "opus-4.8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.6": "gpt-5.6",
}

# concrete slug differs by access path even for the "same" board model: codex
# reaches gpt-5.6 only as the ChatGPT-account variant "gpt-5.6-sol".
MODEL_ID_OVERRIDE: dict[tuple[str, str], str] = {
    ("codex", "gpt-5.6"): "gpt-5.6-sol",
}


def _model_id(harness: str, model_key: str) -> str:
    return MODEL_ID_OVERRIDE.get((harness, model_key), MODELS[model_key])

# reasoning-effort tiers — the board runs each cell at two tiers, carried in
# the cell name (…_default / …_max) so both sit on disk without colliding.
# Thinking policy (user decisions 2026-07-14 / 2026-07-17): thinking is ON for
# Claude in BOTH tiers (adaptive); only the effort param moves.
#   max     — elevated / ablation: Claude effort="max" (API-accepted on all
#             three board models, raw-probed; litellm's client gate is stale,
#             see mini_swe._unlock_claude_effort_max), GPT "xhigh" (server-
#             enumerated top; verified on codex's ChatGPT account too).
#   default — the effort a normal user gets: Claude sends NO effort param (the
#             API picks the model default), GPT = "medium" (codex/openai
#             default) EXCEPT codex+gpt-5.6, whose default is "low" (user
#             decision 2026-07-17 — see _tier_extra). Claude keeps adaptive
#             thinking; the effort knob is the only thing dropped.
# Cross-vendor labels are NOT commensurable — actual thinking spend is in the
# per-call usage logs; report those alongside any comparison.
EFFORT_TIERS = ("default", "max")


def _tier_extra(harness: str, model_key: str, tier: str) -> dict:
    """Per-(harness, model, tier) knobs, recorded into the run config."""
    is_gpt = model_key.startswith("gpt")
    if tier == "max":
        if harness == "sdk":
            return {"effort": "max"}
        if harness == "codex":
            return {"effort": "xhigh"}
        if harness == "mini":
            return ({"reasoning_effort": "xhigh"} if is_gpt
                    else {"thinking": "adaptive", "effort": "max"})
    else:  # default
        if harness == "sdk":
            return {}                                # no effort; thinking adaptive (harness default)
        if harness == "codex":
            # gpt-5.6 (as the ChatGPT-account "gpt-5.6-sol" variant) defaults to
            # "low" on the codex CLI, not "medium" (user decision 2026-07-17);
            # gpt-5.5 keeps the medium GPT vendor default.
            return {"effort": "low" if model_key == "gpt-5.6" else "medium"}
        if harness == "mini":
            return ({"reasoning_effort": "medium"} if is_gpt
                    else {"thinking": "adaptive"})   # keep thinking, drop effort
    return {}

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
    effort_tier: str = "max"  # default | max — reasoning-effort tier
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


def _cell(harness: str, model_key: str, condition: str,
          tier: str = "max") -> CellSpec:
    cond = CONDITIONS[condition]
    return CellSpec(
        name=f"std_{harness}_{model_key}_{condition}_{tier}",
        harness=harness,
        model_key=model_key,
        model_id=_model_id(harness, model_key),
        condition=condition,
        bare=cond["bare"],
        skill=cond["skill"],
        persona=cond.get("persona", False),
        effort_tier=tier,
        extra=tuple(sorted(_tier_extra(harness, model_key, tier).items())),
    )


CLAUDE_MODELS = ("sonnet-5", "opus-4.8", "fable-5")

# current board: bare-only (nav / persona conditions stay defined above but
# unregistered for now). Design: each closed harness vs the open mini harness
# on the SAME models — claude side sdk↔mini (sonnet/opus/fable), openai side
# codex↔mini (gpt-5.5 / gpt-5.6). (mini·fable-5 added 2026-07-17 on user
# request — completes the claude sdk↔mini pairing; runs via litellm→Anthropic
# at the same default=high regime as mini·sonnet/opus.)
BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
    ("mini", "sonnet-5"), ("mini", "opus-4.8"), ("mini", "fable-5"),
    ("mini", "gpt-5.5"), ("mini", "gpt-5.6"),
)

CELLS: dict[str, CellSpec] = {}
for _h, _m in BOARD:
    for _t in EFFORT_TIERS:
        spec = _cell(_h, _m, "bare", _t)
        CELLS[spec.name] = spec

# persona ablation (paper 4.2, E14/E15): stock Claude Code persona kept, bare
# briefing appended — sdk-only, sonnet/opus, at default effort so it pairs with
# the bare main cells (E1/E2).
for _h, _m in (("sdk", "sonnet-5"), ("sdk", "opus-4.8")):
    spec = _cell(_h, _m, "persona", "default")
    CELLS[spec.name] = spec

# batches carry the effort tier: *_default = the vendor-default main experiment
# (paper main table); *_max = the elevated ablation (already run).
BATCHES = {
    # default-effort main experiment (2026-07-17: to run)
    "Ad": [f"std_sdk_{m}_bare_default" for m in CLAUDE_MODELS],
    "Bd": ["std_mini_sonnet-5_bare_default", "std_mini_opus-4.8_bare_default"],  # anthropic key
    "Gd": ["std_mini_gpt-5.5_bare_default", "std_mini_gpt-5.6_bare_default"],    # openai key
    "Xd": ["std_codex_gpt-5.5_bare_default", "std_codex_gpt-5.6_bare_default"],
    # max-effort ablation (already run)
    "A": [f"std_sdk_{m}_bare_max" for m in CLAUDE_MODELS],
    "B": ["std_mini_sonnet-5_bare_max", "std_mini_opus-4.8_bare_max"],
    "G": ["std_mini_gpt-5.5_bare_max", "std_mini_gpt-5.6_bare_max"],
    "X": ["std_codex_gpt-5.5_bare_max", "std_codex_gpt-5.6_bare_max"],
}


# ── experiment registry (paper §4, E-numbered) ────────────────────────────
# Explicit map from the plan's experiment numbers to board cells: request a run
# by number ("run E7") and eyeball the exact knobs here. `section`/`label` are
# the paper's grouping (not derivable); `cell` is the single source of truth for
# every frozen knob and reasoning-effort tier (resolve via get_cell / the
# `experiments` command). In scope (user decision 2026-07-17): 4.1 main (default
# tier), 4.2 persona, 4.3 effort (max tier). OUT of scope and intentionally
# unregistered: E10-E13 (mini · qwen*), E21-E24 (Waypoint), E25-E28 (VLNVerse).
EXPERIMENTS: dict[str, dict] = {
    # 4.1 Main — R2R-CE, bare tools, vendor-DEFAULT effort (paper main table)
    "E1": {"section": "4.1 main", "label": "SDK · sonnet-5",   "cell": "std_sdk_sonnet-5_bare_default"},
    "E2": {"section": "4.1 main", "label": "SDK · opus-4.8",   "cell": "std_sdk_opus-4.8_bare_default"},
    "E3": {"section": "4.1 main", "label": "SDK · fable-5",    "cell": "std_sdk_fable-5_bare_default"},
    "E4": {"section": "4.1 main", "label": "Codex · gpt-5.5",  "cell": "std_codex_gpt-5.5_bare_default"},
    "E5": {"section": "4.1 main", "label": "Codex · gpt-5.6",  "cell": "std_codex_gpt-5.6_bare_default"},
    "E6": {"section": "4.1 main", "label": "mini · sonnet-5",  "cell": "std_mini_sonnet-5_bare_default"},
    "E7": {"section": "4.1 main", "label": "mini · opus-4.8",  "cell": "std_mini_opus-4.8_bare_default"},
    "E8": {"section": "4.1 main", "label": "mini · gpt-5.5",   "cell": "std_mini_gpt-5.5_bare_default"},
    "E9": {"section": "4.1 main", "label": "mini · gpt-5.6",   "cell": "std_mini_gpt-5.6_bare_default"},
    # 4.2 Persona — R2R-CE, stock Claude Code persona kept + bare briefing
    # appended, default effort (sdk-only; mini has no persona, codex can't drop its own)
    "E14": {"section": "4.2 persona", "label": "SDK · +persona · sonnet-5", "cell": "std_sdk_sonnet-5_persona_default"},
    "E15": {"section": "4.2 persona", "label": "SDK · +persona · opus-4.8", "cell": "std_sdk_opus-4.8_persona_default"},
    # 4.3 Effort — R2R-CE, bare, elevated effort (max=Claude effort=max /
    # codex xhigh / mini-gpt reasoning_effort=xhigh); the max-tier ablation
    "E16": {"section": "4.3 effort", "label": "SDK · effort=max · sonnet-5",  "cell": "std_sdk_sonnet-5_bare_max"},
    "E17": {"section": "4.3 effort", "label": "SDK · effort=max · opus-4.8",  "cell": "std_sdk_opus-4.8_bare_max"},
    "E18": {"section": "4.3 effort", "label": "SDK · effort=max · fable-5",   "cell": "std_sdk_fable-5_bare_max"},
    "E19": {"section": "4.3 effort", "label": "Codex · effort=xhigh · gpt-5.5", "cell": "std_codex_gpt-5.5_bare_max"},
    "E20": {"section": "4.3 effort", "label": "mini · effort=xhigh · gpt-5.5",  "cell": "std_mini_gpt-5.5_bare_max"},
}


def get_cell(name: str) -> CellSpec:
    if name not in CELLS:
        known = "\n  ".join(sorted(CELLS))
        raise KeyError(f"unknown cell {name!r}; known cells:\n  {known}")
    return CELLS[name]


def get_experiment(num: str) -> CellSpec:
    """Resolve a paper experiment number (E1..E20) to its board cell."""
    entry = EXPERIMENTS.get(num.upper())
    if entry is None:
        raise KeyError(f"unknown experiment {num!r}; known: {', '.join(EXPERIMENTS)}")
    return get_cell(entry["cell"])


def resolve_cell(token: str) -> CellSpec:
    """Accept either a cell name or an E-number and return the CellSpec."""
    return get_experiment(token) if token.upper() in EXPERIMENTS else get_cell(token)
