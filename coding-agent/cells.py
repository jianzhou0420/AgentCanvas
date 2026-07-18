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

# ── frozen configuration (std-v2; change anything → that's std-v3) ──
#
# 2026-07-14: the turn cap is set by WHO OWNS THE COMPUTE, and it is the one knob
# that differs across the board. Rented compute (API billing or a subscription's
# rate limit) runs 100; compute we own runs 200. Both numbers are recorded in every
# summary.json and printed by `stdrun.py board` — the cell NAME does not carry the
# cap, so read the board, never the name, when comparing across columns.
STD_FROZEN: dict = {
    "dataset": "R2R-CE",
    "split": "rand100",
    "episodes": "0-49",
    "max_turns": 100,
    "rgb_resolution": 512,     # 2026-07-14: 512 is the default. At 224 px the
                               # landmark an R2R instruction names ("stop at the
                               # corner of the bar") is a smudge — see the ep0
                               # frame at 224 vs 1024. Cells run before this are
                               # preserved as `*__t150_rgb224`.
    "step_budget": 500,
    "episode_timeout": 2400,
}
# A 512 px / 100 turn probe is parked at `*__t100_rgb512` (6 eps, not a result).
# Note it would NOT have been a clean resolution ablation against the 224 px cells:
# it moved rgb 224→512 AND turns 150→100 together. The one inference that survives
# that confound is directional — fewer turns can only hurt, so a 512 cell BEATING
# the 224 one would prove resolution matters (conservatively). A clean ablation
# needs 512 px at the SAME turn cap.

# Compute we own (local GPU, no API bill or rate limit) can take a bigger cap;
# the knob is kept so the rented and owned columns can diverge. It is currently
# set equal to the rented cap, so this override is a no-op today.
#
# The rationale is a batching effect: mini's ReAct loop allows one tool call per
# LLM turn, so a turn becomes environment actions only as fast as the model
# batches them into a single step([...]) call. A model that emits one action per
# turn is turn-limited long before the step budget binds, so a turn cap tuned on
# models that batch is a wall for one that doesn't — which is why a bare cell at a
# low cap measures a turn-limited agent, not a navigation ceiling. The per-model
# batching-rate and success-rate measurements behind these numbers live in the
# private research repo, not here.
LOCAL_MAX_TURNS = 100

# wp condition only: the decision-step cap (one goto = one step), VLN-MME's
# ``max_step``. Enforced by wp_bridge.py (truncates the episode) and stated in
# the wp briefing. Not a low-level MOVE_FORWARD count — those stay on the 500
# step_budget above.
WP_MAX_MOVES = 30
# wp cells force visible reasoning: a thinking budget (so thinking blocks are
# substantive, not adaptive one-liners) on top of the prompt's ReAct rule.
WP_THINK_BUDGET = 4000
# The SDK turn cap must sit ABOVE the move budget so the move cap (enforced
# in wp_bridge) is what actually ends an episode, not the harness. Measured
# ~3 SDK turns per move (observe + reason + goto), so 30 moves ~= 90 turns;
# 150 leaves ample margin. (bare/nav stay on STD_FROZEN's 100.)
WP_MAX_TURNS = 150

# model key (board column) → model id passed to the harness.
# gpt slugs are identical on the codex CLI and litellm's openai route;
# gpt-5.6 on codex needs CLI > 0.142 (upgrade + re-probe before batch X).
MODELS = {
    "sonnet-5": "claude-sonnet-5",
    "opus-4.8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.6": "gpt-5.6",
    # open-weight column, served locally by ollama (litellm's ollama_chat route).
    # The mini adapter's _is_local() keys off the "ollama" prefix: no provider
    # key, cost tracking relaxed, no anthropic cache_control.
    #
    # bf16 = full precision, so the 4b→9b scaling contrast carries no quantization
    # confound. `-std` = a Modelfile carrying the std-v2 serving config:
    #
    #   temperature 1.0 / top_p .95 / top_k 20 — Qwen's FACTORY values, untouched
    #   seed 0                                 — the whole reason a run reproduces
    #   presence_penalty 1.5 → 0.5             — Qwen's default, lowered (a call,
    #                                            not a measured fix: a 6-frame sweep
    #                                            found NO robust effect on batching)
    #   repeat_penalty   1.1 → 1.0             — OLLAMA's default, never specified
    #                                            by Qwen; a straight correction
    #
    # Stock ollama passes NO seed, so before this every episode was an
    # irreproducible lottery ticket; temp=1.0 + a fixed seed reproduces byte-for-
    # byte (determinism comes from the seed, NOT from a zero temperature). The
    # sampling lives in the Modelfile because litellm's ollama route DROPS
    # presence_penalty silently (drop_params=True) — pinning it client-side is a
    # no-op. The adapter reads the sampling back from /api/show and refuses to run
    # a cell whose sampling is not pinned.
    "qwen3.5-4b": "ollama_chat/qwen3.5:4b-bf16-std",
    "qwen3.5-9b": "ollama_chat/qwen3.5:9b-bf16-std",
    # qwen API column, served by Alibaba DashScope's OpenAI-compatible endpoint.
    # litellm's `openai/` route + the api_base in MODEL_EXTRA below; the key
    # rides OPENAI_API_KEY (set it to the DashScope key for the run shell —
    # litellm's openai route reads that var regardless of who the vendor is).
    # The mini adapter's _is_local_model() treats explicit-api_base models like
    # local ones: no anthropic/openai key assertion, cost tracking relaxed
    # (litellm has no price entry for dashscope slugs).
    "qwen3.7-plus": "openai/qwen3.7-plus",
    "qwen3.6-plus": "openai/qwen3.6-plus",
}

# per-model default knobs, recorded into the run config (e.g. a local model's
# api_base + image_window). The qwen cells take none: litellm reaches ollama
# at its default base, and image_window stays 0 (all frames) so the open
# column sees the same visual history as every other mini cell — at 224 px
# that peaks far below the 128k serve context.
MODEL_EXTRA: dict[str, dict] = {
    # DashScope compatible-mode base. INTL endpoint — the key in use was issued
    # by the international Model Studio (the mainland endpoint rejects it).
    "qwen3.7-plus": {
        "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "qwen3.6-plus": {
        "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
}

CONDITIONS = {
    "bare": {"bare": True, "skill": None},
    "nav": {"bare": False, "skill": "ledger-nav"},
    # ablation: bare tool surface + BARE briefing, but the stock Claude Code
    # persona is KEPT (preset system prompt, briefing appended instead of
    # replacing). sdk-only — mini has no persona; codex can't drop its own.
    "persona": {"bare": True, "skill": None, "persona": True},
    # waypoint action space (wp_bridge.py): depth-predicted candidate
    # waypoints drawn numbered on a 4-view panorama; the agent picks one
    # (goto) or stops. bare=True keeps the mcp_bridge mechanisms
    # (clearance / look_around / STOP gate) out of the comparison — wp is
    # its own tool surface, not bare + extras. Needs a second auto_host
    # (waypoint predictor, --wp-server).
    "wp": {"bare": True, "skill": None, "wp": True},
    # wp + the anti-circling waypoint ledger skill. bare stays True: for wp the
    # flag only gates the mcp_bridge mechanisms (which wp never uses), so the
    # skill appends through build_briefing's wp branch, not the bare gate.
    "wp-nav": {"bare": True, "skill": "wp-ledger-nav", "wp": True},
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
    wp: bool = False   # waypoint-selection action space (wp_bridge.py)
    extra: tuple = ()  # model-default knobs as (key, value) pairs (hashable)
    max_turns: int | None = None  # None → STD_FROZEN (rented compute: 100)

    @property
    def extra_dict(self) -> dict:
        return dict(self.extra)

    @property
    def is_local(self) -> bool:
        """Served on our own GPU — no meter, no rate limit."""
        return self.model_id.startswith(("ollama", "hosted_vllm/"))

    @property
    def output_root(self) -> Path:
        return OUTPUT_ROOTS[self.harness]

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.name


def _cell(harness: str, model_key: str, condition: str) -> CellSpec:
    cond = CONDITIONS[condition]
    model_id = MODELS[model_key]
    return CellSpec(
        name=f"std_{harness}_{model_key}_{condition}",
        harness=harness,
        model_key=model_key,
        model_id=model_id,
        condition=condition,
        bare=cond["bare"],
        skill=cond["skill"],
        persona=cond.get("persona", False),
        wp=cond.get("wp", False),
        extra=tuple(sorted(MODEL_EXTRA.get(model_key, {}).items())),
        # the turn cap follows the compute, not the model — see LOCAL_MAX_TURNS
        # (wp needs headroom above its move budget — see WP_MAX_TURNS)
        max_turns=(
            WP_MAX_TURNS if cond.get("wp")
            else LOCAL_MAX_TURNS if model_id.startswith(("ollama", "hosted_vllm/"))
            else None
        ),
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
    # qwen API flagship — same mini harness as the local qwen column, so the
    # 4b → 9b → API-flagship scaling read stays within one stack. Rented
    # compute → STD_FROZEN's 100-turn cap, same as everyone (local is also
    # 100 right now, so the whole column is directly comparable).
    ("mini", "qwen3.7-plus"),
    ("mini", "qwen3.6-plus"),
)

# open-weight column: the same mini harness, locally served, and the only
# cells that carry BOTH conditions — the closed board answers "which stack",
# this one answers "does the nav scaffolding (mechanisms + ledger-nav) buy a
# small model anything the frontier models don't need".
LOCAL_BOARD = (
    ("mini", "qwen3.5-4b", "bare"), ("mini", "qwen3.5-4b", "nav"),
    ("mini", "qwen3.5-9b", "bare"), ("mini", "qwen3.5-9b", "nav"),
)

# waypoint-action-space pilots. sdk + codex only: both reach the env
# through the stdio bridge, so wp_bridge.py covers them for free; the mini
# column waits for the toolset.py port (checked by check_equivalence.py).
WP_BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"),
)

# open-weight waypoint pilots. The mini harness now reaches wp through
# toolset.WaypointToolSet (in-process port of wp_bridge.py, gated by
# check_equivalence.py) — so qwen runs wp with no MCP subprocess, same path as
# bare/nav. wp is the action space that structurally removes the two failure
# modes the step()-space 2×2 found in small models: batching starvation (one
# goto = one real move the predictor executes, so a single-action model is not
# penalized) and the stopping wall (a discrete "pick a number / stop" choice).
# `wp` = the action space alone; `wp-nav` = plus the anti-circling ledger skill.
LOCAL_WP_BOARD = (
    ("mini", "qwen3.5-4b", "wp"), ("mini", "qwen3.5-4b", "wp-nav"),
    ("mini", "qwen3.5-9b", "wp"), ("mini", "qwen3.5-9b", "wp-nav"),
)

CELLS: dict[str, CellSpec] = {}
for _h, _m in BOARD:
    spec = _cell(_h, _m, "bare")
    CELLS[spec.name] = spec
for _h, _m, _c in LOCAL_BOARD:
    spec = _cell(_h, _m, _c)
    CELLS[spec.name] = spec
for _h, _m in WP_BOARD:
    # both the action space alone (wp) and + the anti-circling skill (wp-nav),
    # symmetric with LOCAL_WP_BOARD so the skill's effect is a paired contrast
    for _c in ("wp", "wp-nav"):
        spec = _cell(_h, _m, _c)
        CELLS[spec.name] = spec
for _h, _m, _c in LOCAL_WP_BOARD:
    spec = _cell(_h, _m, _c)
    CELLS[spec.name] = spec

BATCHES = {
    "A": [f"std_sdk_{m}_bare" for m in CLAUDE_MODELS],
    "B": ["std_mini_sonnet-5_bare", "std_mini_opus-4.8_bare"],      # API key: anthropic
    "G": ["std_mini_gpt-5.5_bare", "std_mini_gpt-5.6_bare"],        # API key: openai
    "X": ["std_codex_gpt-5.5_bare", "std_codex_gpt-5.6_bare"],      # 5.6 needs CLI upgrade
    # local GPU, $0. nav (full mechanisms + ledger-nav) runs BEFORE bare: it is
    # the condition that might actually work, and a batch this long can always be
    # cut short — better to lose the control than the treatment. The mini adapter
    # brings ollama up with the context pinned; it refuses to run if it can't.
    "Q": ["std_mini_qwen3.5-4b_nav", "std_mini_qwen3.5-9b_nav",
          "std_mini_qwen3.5-4b_bare", "std_mini_qwen3.5-9b_bare"],
    # waypoint pilots (needs --wp-server; see coding-agent/README.md)
    "W": ["std_sdk_sonnet-5_wp", "std_sdk_fable-5_wp"],
    # open-weight waypoint pilots, local GPU $0 (needs --wp-server). wp-nav
    # (skill) before wp (control): the treatment might actually work, and a long
    # batch can be cut short — better to lose the control than the treatment.
    "WQ": ["std_mini_qwen3.5-4b_wp-nav", "std_mini_qwen3.5-4b_wp",
           "std_mini_qwen3.5-9b_wp-nav", "std_mini_qwen3.5-9b_wp"],
}


def get_cell(name: str) -> CellSpec:
    if name not in CELLS:
        known = "\n  ".join(sorted(CELLS))
        raise KeyError(f"unknown cell {name!r}; known cells:\n  {known}")
    return CELLS[name]
