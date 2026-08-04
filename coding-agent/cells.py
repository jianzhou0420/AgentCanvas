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

# Derived mip{N} evaluation splits (user decisions 2026-07-22): seed-42
# scene-stratified proportional samples of an official val, MATERIALIZED at
# the dataset layer (same form as R2R-CE's rand100) — a derived split file
# the env panel selects (split="mip100"), eval running episodes 0-99 of it.
# Audit manifests: coding-agent/bridges/splits/*_n100_seed42.json (committed
# — data/ is gitignored, so the manifest is the ONE tracked record of what
# each mip split contains). Generator: sample_episodes.py, in git history at
# cecd19c — manifest + generator regenerate the materialized CSV
# byte-identically. The ObjectNav family (hm3d / mp3d / ovon×3) that shared
# this mechanism never entered the MIP paper and was removed 2026-08-03
# (cells + helper last at a942483; bridges/splits at cecd19c).

SPLITS_DIR = REPO_ROOT / "coding-agent" / "bridges" / "splits"


# ── HM-EQA benchmark line (2026-07-29): explore-eqa multiple-choice EQA ──
# A benchmark line at the same level as the habitat-r2r std line. Success is
# answer correctness (env_hmeqa__evaluate compares the agent's letter to GT);
# the episode ends by answer("A".."D") via hmeqa_bridge.py, not STOP.
# Deviations from the benchmark-native protocol (all documented in the
# bridge docstring): discrete 0.25 m / 30° actions via env_hmeqa__
# step_discrete (native is TSDF-planned free-pose teleports; explore-eqa's
# scene-size-dependent num_step budget is a TELEPORT budget and does not
# apply), step budget enforced bridge-side at the std 500. The camera stays
# benchmark-native (640×480, hfov 120, starting tilted 30° down).
# Episodes: split="mip100" — the MATERIALIZED dataset-layer split
# (data/hm3d/hmeqa/questions_mip100.csv, row k = k-th sampled original
# row) selected via the env panel, episodes 0-99 into it. Same semantics as
# the objnav mip splits (aligned 2026-07-29 on user request; before that
# the cell carried raw val indices — the trial20/smoke run dirs are indexed
# in THAT original-CSV space).
def _hmeqa_frozen() -> dict:
    manifest = SPLITS_DIR / "hmeqa_val_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"hmeqa: missing split manifest {manifest} — restore it from git "
            "(tracked; sampler sample_episodes.py lives at cecd19c)")
    derived = REPO_ROOT / "data" / "hm3d" / "hmeqa" / "questions_mip100.csv"
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"hmeqa: missing materialized split {derived} — regenerate with "
            "sample_episodes.py --benchmark hmeqa --materialize (git, cecd19c)")
    return {
        "benchmark": "hmeqa",
        "dataset": None,       # single-CSV benchmark: no dataset selector
        "split": "mip100",   # dataset-layer derived split (mip semantics)
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same turn/fuse posture as the ObjectNav lines (exploration task,
        # full-history resend cost tail): 150 turns + $18/episode USD fuse.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
        # Camera-tilt actions 4/5 on the toolface (user decision 2026-07-29,
        # option B: the native −30° pitch made near-overhead ceiling
        # fixtures structurally unobservable — ep4 smoke analysis). Mask
        # with `--nonstd --set tilt_actions=0`: bridge validation, tool
        # description, and briefing all follow the one flag.
        "tilt_actions": True,
    }


BENCHMARK_FROZEN: dict[str, dict] = {"hmeqa": _hmeqa_frozen()}


# ── VLNVerse line (2026-08-02): instruction-following VLN in Isaac Sim 5.1 ──
# Seventh benchmark line, but NOT an ObjectNav sibling: vlnverse is the same
# task shape as habitat-r2r (a language instruction, STOP within 3 m of the
# goal), just a different simulator + scene corpus (262 kujiale scans, Isaac
# 5.1 renders at 1024²/90° behind an msgpack RPC seam). So it reuses the R2R
# branch end to end — mcp_bridge with HABITAT_VERB_PREFIX=env_vlnverse, the
# std briefing verbatim (identical action space: 0=STOP, 1=FWD 0.25 m,
# 2/3=turn 15°), driver reset/evaluate through env_verb_prefix().
#
# PROVISIONAL — this is deliberately NOT a frozen protocol yet. The other six
# lines run a MATERIALIZED, scene-stratified, seed-42 sample of 100 (rand100 /
# mip100) with a committed audit manifest; no such sample exists for
# vlnverse because sample_episodes.py has no vlnverse backend. Until it does,
# this line runs the first 100 episodes of fine/val_unseen IN SPLIT-FILE ORDER
# — deterministic and reproducible, but scene-clustered (the split is grouped
# by scan, so eps 0-99 cover only a handful of the 262 scenes) and therefore
# NOT comparable to the R2R std numbers. Use it for smoke + development;
# freeze a mip100-style sample before it carries a paper claim.
BENCHMARK_FROZEN["vlnverse"] = {
    "benchmark": "vlnverse",
    "dataset": "fine",         # fine-grained instructions (the R2R analogue)
    "split": "val_unseen",     # 825 episodes
    "episodes": "0-99",        # split-file order — see PROVISIONAL note above
    # Same posture as the R2R std line, not the objnav one: this is a
    # point-to-point instruction follow, not open-ended exploration, so it
    # takes std-v2's 200 turns rather than objnav's 150 + USD fuse.
    "max_turns": 200,
    "rgb_resolution": 512,     # accepted-and-ignored: Isaac renders 1024²
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
# Turn cap for the goto-based surfaces (wp and hybrid). Re-measured 2026-07-31
# at ~2.6 SDK turns per move (observe + reason + goto), so a 30-move budget is
# ~80 turns and a wp episode only turn-exhausts after ~38 moves — well past the
# move cap that is supposed to end it. Lowered 150 -> 100 so the whole wp column
# sits at the SAME cap as bare/nav (the R2R std, rented compute) and is directly
# comparable; the move cap in wp_bridge still binds first.
# NOTE (two-machine merge 2026-08-01): runs recorded before this date used 150.
# They are a different protocol — do not pool them with post-merge wp numbers.
# RxR needs longer trajectories and lifts BOTH caps off-board:
#   --nonstd --set wp_max_moves=N max_turns=M
WP_MAX_TURNS = 100

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
    # Opus 5 (released 2026-07; added 2026-07-25). Probed on the R2R board via
    # three targeted cells (OPUS5_CELLS below), NOT by joining CLAUDE_MODELS —
    # that would fan the model out across the whole sdk/mini/wp/objnav matrix.
    # _tier_extra keys off harness + gpt-prefix only, so opus-5 (sdk/mini, non-gpt)
    # resolves its effort tiers correctly without any board membership.
    "opus-5": "claude-opus-5",
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
    # Qwen3.8 flagship (released 2026-07; added 2026-07-25). Slug assumed from the
    # 3.6/3.7-plus pattern — VERIFY the exact DashScope model string against Model
    # Studio before launching the run.
    "qwen3.8-plus": "openai/qwen3.8-plus",
    # Qwen3.5-plus — the API sibling of the local 4b/9b column, so the
    # small-open -> API-flagship scaling read covers 3.5 as well.
    "qwen3.5-plus": "openai/qwen3.5-plus",
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
    "qwen3.8-plus": {
        "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "qwen3.5-plus": {
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

# The nav / wp-nav (ledger skill) and persona ablation conditions were
# retired 2026-08-03 with the skills/ dir — none entered the MIP paper.
# Last present at a942483; skills/ files at cecd19c.
CONDITIONS = {
    "bare": {"bare": True},
    # waypoint action space (wp_bridge.py): depth-predicted candidate
    # waypoints drawn numbered on a 4-view panorama; the agent picks one
    # (goto) or stops. bare=True keeps the mcp_bridge mechanisms
    # (clearance / look_around / STOP gate) out of the comparison — wp is
    # its own tool surface, not bare + extras. Needs a second auto_host
    # (waypoint predictor, --wp-server).
    "wp": {"bare": True, "wp": True},
    # Agent-selected Hybrid Interface: primitive actions (step 0-3) AND the
    # waypoint tool (goto) in ONE surface, plus two matching lenses to look
    # through; the model decides per move which to use, and may switch freely.
    # The tools enforce the pairing (look chooses the interface, you cannot look
    # twice in a row), so the choice of lens IS the choice of interface.
    # bare=True keeps it a minimal, workflow-free surface like bare/wp. Needs the
    # waypoint predictor (--wp-server), same as wp.
    #   sdk  -> coding-agent/bridges/hybrid_bridge.py
    #   mini -> toolset.HybridToolSet (in-process port; see check_equivalence.py)
    "hybrid": {"bare": True, "hybrid": True},
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
    condition: str     # bare | wp | hybrid (std); go2 / hmeqa / ui via replace()
    bare: bool
    wp: bool = False   # waypoint-selection action space (wp_bridge.py)
    go2: bool = False  # real Unitree Go2 embodiment (go2_bridge.py)
    hybrid: bool = False  # primitive + waypoint in one surface (hybrid_bridge.py)
    benchmark: str = "r2r"  # r2r (habitat-r2r std line) | hm3d | mp3d | ovon*
                            # | hmeqa | vlnverse
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
        wp=cond.get("wp", False),
        hybrid=cond.get("hybrid", False),
        effort_tier=tier,
        extra=tuple(sorted(extra.items())),
        # the turn cap follows the cell line: wp and hybrid both move by goto
        # and share WP_MAX_TURNS; local GPU carries its own cap
        # (LOCAL_MAX_TURNS); everything else takes STD_FROZEN's 200.
        max_turns=(
            WP_MAX_TURNS if cond.get("wp") or cond.get("hybrid")
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
    ("mini", "qwen3.8-plus"),   # newest Qwen flagship (2026-07-25) — sweep row
    ("mini", "qwen3.5-plus"),   # API sibling of the local 4b/9b column
)

# open-weight column: the same mini harness, locally served — the descend-to-
# small read of the bare surface. (The nav pairing that once rode here was
# retired with the skill condition, 2026-08-03.)
LOCAL_BOARD = (
    ("mini", "qwen3.5-4b", "bare"),
    ("mini", "qwen3.5-9b", "bare"),
)

# waypoint-action-space pilots. sdk + codex only: both reach the env
# through the stdio bridge, so wp_bridge.py covers them for free; the mini
# column waits for the toolset.py port (checked by check_equivalence.py).
WP_BOARD = (
    ("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "opus-5"), ("sdk", "fable-5"),
    ("codex", "gpt-5.5"), ("codex", "gpt-5.6"),
)

# open-weight waypoint pilots. The mini harness now reaches wp through
# toolset.WaypointToolSet (in-process port of wp_bridge.py, gated by
# check_equivalence.py) — so qwen runs wp with no MCP subprocess, same path as
# bare. wp is the action space that structurally removes the two failure
# modes the step()-space 2×2 found in small models: batching starvation (one
# goto = one real move the predictor executes, so a single-action model is not
# penalized) and the stopping wall (a discrete "pick a number / stop" choice).
LOCAL_WP_BOARD = (
    ("mini", "qwen3.5-4b", "wp"),
    ("mini", "qwen3.5-9b", "wp"),
)

CELLS: dict[str, CellSpec] = {}
for _h, _m in BOARD:
    for _t in EFFORT_TIERS:
        spec = _cell(_h, _m, "bare", _t)
        CELLS[spec.name] = spec

# wp / local / qwen-API line — untier-ed names (match the existing on-disk runs)
for _h, _m in QWEN_API_BOARD:
    spec = _cell(_h, _m, "bare")
    CELLS[spec.name] = spec

# Opus 5 probe (2026-07-25): three targeted R2R cells, bare surface, so the newest
# frontier Opus lands in the sweep (mini row → Table 2 / harness Table 4) and the
# effort ablation (sdk default+max → Table 3) without expanding CLAUDE_MODELS.
OPUS5_CELLS = (
    ("mini", "opus-5", "bare", "default"),
    ("sdk",  "opus-5", "bare", "default"),
    ("sdk",  "opus-5", "bare", "max"),
)
for _h, _m, _c, _t in OPUS5_CELLS:
    spec = _cell(_h, _m, _c, _t)
    CELLS[spec.name] = spec
for _h, _m, _c in LOCAL_BOARD:
    spec = _cell(_h, _m, _c)
    CELLS[spec.name] = spec
for _h, _m in WP_BOARD:
    spec = _cell(_h, _m, "wp")
    CELLS[spec.name] = spec
for _h, _m, _c in LOCAL_WP_BOARD:
    spec = _cell(_h, _m, _c)
    CELLS[spec.name] = spec

# API-qwen waypoint pilots. Same mini + toolset.WaypointToolSet path as the
# local qwen wp cells, so the wp column now spans local 4b/9b AND the API
# flagships. Untier-ed; WP_MAX_TURNS puts every wp cell at the same 100.
API_WP_BOARD = (
    ("mini", "qwen3.7-plus", "wp"),
    ("mini", "qwen3.6-plus", "wp"),
    ("mini", "qwen3.5-plus", "wp"),
)
for _h, _m, _c in API_WP_BOARD:
    spec = _cell(_h, _m, _c)
    CELLS[spec.name] = spec

# Agent-selected Hybrid Interface — primitive actions AND the waypoint tool in
# one surface, the model choosing when to use/combine/switch. Same models as the
# wp column so the hybrid vs primitive-only (bare) vs waypoint-only (wp)
# contrast stays paired. Needs --wp-server, same as wp.
HYBRID_BOARD = (
    ("sdk", "fable-5"), ("sdk", "sonnet-5"),
)
for _h, _m in HYBRID_BOARD:
    spec = _cell(_h, _m, "hybrid")
    CELLS[spec.name] = spec

# open-weight hybrid pilots. The mini harness reaches the hybrid surface through
# toolset.HybridToolSet (in-process port of hybrid_bridge.py) — same path as the
# WaypointToolSet wp cells, no MCP subprocess. This is the descend-to-small read
# for the interface-choice condition: bare (SR ~0) and wp (the rescue, SR ~0.3)
# already pin the two standalone action spaces for qwen-4b; hybrid asks whether a
# small model can PICK between them turn by turn, or whether the added choice —
# a freedom the frontier models exploit — is one more thing it cannot execute.
LOCAL_HYBRID_BOARD = (
    ("mini", "qwen3.5-4b"), ("mini", "qwen3.5-9b"),
)
for _h, _m in LOCAL_HYBRID_BOARD:
    spec = _cell(_h, _m, "hybrid")
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

# HM-EQA board (2026-07-29): sdk trio, bare surface, default effort — cell
# names carry the benchmark prefix (hmeqa_sdk_fable-5), parallel to std_* and
# go2_*. Servers: env_hmeqa auto_host (ac-hmeqa env); the driver refuses a
# mismatched server name.
SDK_TRIO = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"hmeqa_{_h}_{_m}", condition="hmeqa",
                   benchmark="hmeqa")
    CELLS[spec.name] = spec

# VLNVerse board (2026-08-02): same sdk trio, bare surface, default effort.
# Server: env_vlnverse auto_host (ac-vlnverse env, Isaac 5.1 worker behind it);
# the driver refuses a mismatched server name. wp/hybrid cells are NOT built
# for this line — the driver refuses that combination outright (CCW pano
# indexing + a habitat-trained waypoint predictor); see driver.py.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"vlnverse_{_h}_{_m}", condition="vlnverse",
                   benchmark="vlnverse")
    CELLS[spec.name] = spec

# batches: the tiered main board carries the effort tier in the cell name
# (*_default = vendor-default main experiment, *_max = elevated ablation);
# Q/W/WQ are the untier-ed wp/local line.
BATCHES = {
    # new-model probes (2026-07-25): Opus 5 trio (mini sweep + sdk default/max
    # effort) and the Qwen3.8 sweep row. All R2R-CE rand100, reusing STD_FROZEN.
    "O5": ["std_mini_opus-5_bare_default", "std_sdk_opus-5_bare_default",
           "std_sdk_opus-5_bare_max"],
    "Q8": ["std_mini_qwen3.8-plus_bare"],
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
    # local GPU, $0. The mini adapter brings ollama up with the context
    # pinned; it refuses to run if it can't.
    "Q": ["std_mini_qwen3.5-4b_bare", "std_mini_qwen3.5-9b_bare"],
    # waypoint pilots (needs --wp-server; see coding-agent/README.md)
    "W": ["std_sdk_sonnet-5_wp", "std_sdk_fable-5_wp"],
    # open-weight waypoint pilots, local GPU $0 (needs --wp-server)
    "WQ": ["std_mini_qwen3.5-4b_wp", "std_mini_qwen3.5-9b_wp"],
    # HM-EQA board — servers: env_hmeqa auto_host (ac-hmeqa env)
    "EQ": [f"hmeqa_sdk_{m}" for m in CLAUDE_MODELS],
}


# ── experiment registry (paper §4, E-numbered) ────────────────────────────
# Explicit map from the plan's experiment numbers to board cells: request a run
# by number ("run E7") and eyeball the exact knobs here. `section`/`label` are
# the paper's grouping (not derivable); `cell` is the single source of truth for
# every frozen knob and reasoning-effort tier (resolve via get_cell / the
# `experiments` command). In scope (user decision 2026-07-17): 4.1 main (default
# tier), 4.3 effort (max tier). OUT of scope and intentionally unregistered:
# E10-E13 (mini · qwen*), E21-E24 (Waypoint), E25-E28 (VLNVerse) — the qwen/wp
# CELLS above cover that line without E-numbers. The persona pair (E14/E15)
# was retired 2026-08-03 with the persona condition — never entered the paper.
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
    # 4.3 Effort — R2R-CE, bare, elevated effort (max=Claude effort=max /
    # codex xhigh / mini-gpt reasoning_effort=xhigh); the max-tier ablation
    "E16": {"section": "4.3 effort", "label": "SDK · effort=max · sonnet-5",  "cell": "std_sdk_sonnet-5_bare_max"},
    "E17": {"section": "4.3 effort", "label": "SDK · effort=max · opus-4.8",  "cell": "std_sdk_opus-4.8_bare_max"},
    "E18": {"section": "4.3 effort", "label": "SDK · effort=max · fable-5",   "cell": "std_sdk_fable-5_bare_max"},
    "E19": {"section": "4.3 effort", "label": "Codex · effort=xhigh · gpt-5.5", "cell": "std_codex_gpt-5.5_bare_max"},
    "E20": {"section": "4.3 effort", "label": "mini · effort=xhigh · gpt-5.5",  "cell": "std_mini_gpt-5.5_bare_max"},
    # new-model probes (2026-07-25). Opus 5 slots into the same tables as the
    # other Claude models; qwen3.8 stays unregistered (mini·qwen line, like E10-E13).
    "E29": {"section": "4.1 main",   "label": "mini · opus-5",                 "cell": "std_mini_opus-5_bare_default"},
    "E30": {"section": "4.1 main",   "label": "SDK · opus-5",                  "cell": "std_sdk_opus-5_bare_default"},
    "E31": {"section": "4.3 effort", "label": "SDK · effort=max · opus-5",     "cell": "std_sdk_opus-5_bare_max"},
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
