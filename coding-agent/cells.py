"""std-v2 cell registry — the standard board as code.

A cell is one run: harness × model × condition, with every frozen knob
pinned (see docs/pages/developer-guide/tmp/coding-agent/standard-experiments.html).
The runner takes cell names, not free-form flags; deviating from the freeze
requires --nonstd, which renames the run so it can never sit on the board.

Two experiment lines share this registry (merged 2026-07-19):
- the MAIN BOARD (closed/frontier models, effort-tiered `…_default` / `…_max`
  cell names) — std-v2 freeze below applies verbatim;
- the WP / LOCAL line (waypoint action space, open-weight qwen cells; untier-ed
  names like `std_sdk_sonnet-5_wp`) — carries its own per-cell turn caps via
  ``max_turns`` (WP_MAX_TURNS / LOCAL_MAX_TURNS). Runs from before this merge
  may have used other caps; every run's summary.json records the cap that
  actually applied — read the board, never the name, when comparing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

# Compute we own (local GPU, no API bill or rate limit) can take its own cap;
# the knob is kept so the rented and owned columns can diverge.
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
# 150 leaves ample margin. (bare/nav stay on STD_FROZEN's cap.)
WP_MAX_TURNS = 150

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

# concrete slug differs by access path even for the "same" board model: codex
# reaches gpt-5.6 only as the ChatGPT-account variant "gpt-5.6-sol".
MODEL_ID_OVERRIDE: dict[tuple[str, str], str] = {
    ("codex", "gpt-5.6"): "gpt-5.6-sol",
}


def _model_id(harness: str, model_key: str) -> str:
    return MODEL_ID_OVERRIDE.get((harness, model_key), MODELS[model_key])

# per-model default knobs, recorded into the run config (e.g. a local model's
# api_base + image_window). The qwen ollama cells take none: litellm reaches
# ollama at its default base, and image_window stays 0 (all frames) so the open
# column sees the same visual history as every other mini cell.
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

# reasoning-effort tiers — the main board runs each cell at two tiers, carried
# in the cell name (…_default / …_max) so both sit on disk without colliding.
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
# The wp / local qwen cells are untier-ed (tier=None): no tier suffix in the
# name, no effort knob injected — their knobs come from MODEL_EXTRA alone.
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
    name: str          # std_sdk_opus-4.8_bare_default | std_sdk_sonnet-5_wp
    harness: str       # sdk | mini | codex
    model_key: str     # board column
    model_id: str      # harness-facing model string
    condition: str     # bare | nav | persona | wp | wp-nav
    bare: bool
    skill: str | None
    persona: bool = False  # keep the harness's stock persona (ablation)
    wp: bool = False   # waypoint-selection action space (wp_bridge.py)
    go2: bool = False  # real Unitree Go2 embodiment (go2_bridge.py)
    effort_tier: str | None = None  # default | max | None (untier-ed wp/local cells)
    extra: tuple = ()  # model/tier knobs as (key, value) pairs (hashable)
    max_turns: int | None = None  # None → STD_FROZEN's cap (std-v2: 200)

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


def _cell(harness: str, model_key: str, condition: str,
          tier: str | None = None) -> CellSpec:
    cond = CONDITIONS[condition]
    model_id = _model_id(harness, model_key)
    extra = dict(MODEL_EXTRA.get(model_key, {}))
    if tier is not None:
        extra.update(_tier_extra(harness, model_key, tier))
    return CellSpec(
        name=(f"std_{harness}_{model_key}_{condition}"
              + (f"_{tier}" if tier is not None else "")),
        harness=harness,
        model_key=model_key,
        model_id=model_id,
        condition=condition,
        bare=cond["bare"],
        skill=cond["skill"],
        persona=cond.get("persona", False),
        wp=cond.get("wp", False),
        effort_tier=tier,
        extra=tuple(sorted(extra.items())),
        # the turn cap follows the cell line: wp needs headroom above its move
        # budget (WP_MAX_TURNS); local GPU carries its own cap (LOCAL_MAX_TURNS);
        # everything else takes STD_FROZEN's 200.
        max_turns=(
            WP_MAX_TURNS if cond.get("wp")
            else LOCAL_MAX_TURNS if model_id.startswith(("ollama", "hosted_vllm/"))
            else None
        ),
    )


CLAUDE_MODELS = ("sonnet-5", "opus-4.8", "fable-5")

# main board: bare-only, effort-tiered. Design: each closed harness vs the open
# mini harness on the SAME models — claude side sdk↔mini (sonnet/opus/fable),
# openai side codex↔mini (gpt-5.5 / gpt-5.6). (mini·fable-5 added 2026-07-17 on
# user request — completes the claude sdk↔mini pairing; runs via litellm→
# Anthropic at the same default=high regime as mini·sonnet/opus.)
BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
    ("mini", "sonnet-5"), ("mini", "opus-4.8"), ("mini", "fable-5"),
    ("mini", "gpt-5.5"), ("mini", "gpt-5.6"),
)

# qwen API flagship — same mini harness as the local qwen column, so the
# 4b → 9b → API-flagship scaling read stays within one stack. Untier-ed
# (vendor-default reasoning), bare only.
QWEN_API_BOARD = (
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
    for _t in EFFORT_TIERS:
        spec = _cell(_h, _m, "bare", _t)
        CELLS[spec.name] = spec

# persona ablation (paper 4.2, E14/E15): stock Claude Code persona kept, bare
# briefing appended — sdk-only, sonnet/opus, at default effort so it pairs with
# the bare main cells (E1/E2).
for _h, _m in (("sdk", "sonnet-5"), ("sdk", "opus-4.8")):
    spec = _cell(_h, _m, "persona", "default")
    CELLS[spec.name] = spec

# wp / local / qwen-API line — untier-ed names (match the existing on-disk runs)
for _h, _m in QWEN_API_BOARD:
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

# real-robot pilots (2026-07-20): the sdk cells re-embodied on the Unitree Go2
# via go2_bridge.py. BARE surface — observe + step only, no look_around, no
# skill (user decision 2026-07-20: match the habitat main board's condition),
# default effort. NOT part of the std board: instruction is operator-supplied
# (--set instruction=...), the driver skips evaluate (no ground truth — success
# is judged by a human from the recording), and the server is the go2 host on
# the robot's machine, not an agentcanvas backend.
GO2_BOARD = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))
for _h, _m in GO2_BOARD:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"go2_{_h}_{_m}", condition="go2", go2=True)
    CELLS[spec.name] = spec

# batches: the tiered main board carries the effort tier in the cell name
# (*_default = vendor-default main experiment, *_max = elevated ablation);
# Q/W/WQ are the untier-ed wp/local line.
BATCHES = {
    # default-effort main experiment (paper main table)
    "Ad": [f"std_sdk_{m}_bare_default" for m in CLAUDE_MODELS],
    "Bd": ["std_mini_sonnet-5_bare_default", "std_mini_opus-4.8_bare_default"],  # anthropic key
    "Gd": ["std_mini_gpt-5.5_bare_default", "std_mini_gpt-5.6_bare_default"],    # openai key
    "Xd": ["std_codex_gpt-5.5_bare_default", "std_codex_gpt-5.6_bare_default"],
    # max-effort ablation (already run)
    "A": [f"std_sdk_{m}_bare_max" for m in CLAUDE_MODELS],
    "B": ["std_mini_sonnet-5_bare_max", "std_mini_opus-4.8_bare_max"],
    "G": ["std_mini_gpt-5.5_bare_max", "std_mini_gpt-5.6_bare_max"],
    "X": ["std_codex_gpt-5.5_bare_max", "std_codex_gpt-5.6_bare_max"],
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


# ── experiment registry (paper §4, E-numbered) ────────────────────────────
# Explicit map from the plan's experiment numbers to board cells: request a run
# by number ("run E7") and eyeball the exact knobs here. `section`/`label` are
# the paper's grouping (not derivable); `cell` is the single source of truth for
# every frozen knob and reasoning-effort tier (resolve via get_cell / the
# `experiments` command). In scope (user decision 2026-07-17): 4.1 main (default
# tier), 4.2 persona, 4.3 effort (max tier). OUT of scope and intentionally
# unregistered: E10-E13 (mini · qwen*), E21-E24 (Waypoint), E25-E28 (VLNVerse) —
# the qwen/wp CELLS above cover that line without E-numbers.
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
