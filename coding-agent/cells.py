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

Since 2026-08-18 this module holds ORCHESTRATION only: the model zoo,
condition machinery, frozen std config and the batch/experiment registries.
Every experiment line's cells (and its bridge/prompts/nodeset execution
code) register from coding-agent/exp_workspace/<folder>/exp.py via
``_load_exp_workspace()`` at the bottom; only the research lines (imagine ·
eharness · sdkeh) still register inline below. Cell names are unchanged —
the migration is invisible on the board.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── frozen configuration (std-v2; change anything → that's std-v3) ──
# std-v2: rgb 512, max_turns 200 (std-v1 ran 224 / 80).
# std-v1 ep0 smokes (224/80) are archived under <output_root>/archive/.
STD_FROZEN: dict = {
    "dataset": "R2R-CE",
    "split": "rand100",
    "episodes": "0-99",  # the full 100-episode SmartWay sample
    "max_turns": 200,
    "rgb_resolution": 512,
    "step_budget": 500,
    "episode_timeout": 2400,
}

# Derived mip{N} evaluation splits: seed-42
# scene-stratified proportional samples of an official val, MATERIALIZED at
# the dataset layer (same form as R2R-CE's rand100) — a derived split file
# the env panel selects (split="mip100"), eval running episodes 0-99 of it.
# Audit manifests: coding-agent/splits/*_n100_seed42.json (committed
# — data/ is gitignored, so the manifest is the ONE tracked record of what
# each mip split contains). Generator: sample_episodes.py, in git history at
# cecd19c — manifest + generator regenerate the materialized CSV
# byte-identically. The ObjectNav family (hm3d / mp3d / ovon×3) that shared
# this mechanism never entered the MIP paper and was removed
# (cells + helper last at a942483; splits then lived at bridges/splits).
#
# coding-agent/splits/ is the single home for ALL split data (2026-08-18):
# flat *_seed42.json manifests here, plus r2r/ and rxr/ subdirs holding the
# habitat-format split dirs (rand100, rand100_smartway, heldout100, RxR
# rand100). The dataset tree under data/habitat/datasets/ keeps symlinks at
# the old locations so the VLN-CE loader path template — and the frozen
# exp_workspace folders — resolve unchanged.

SPLITS_DIR = REPO_ROOT / "coding-agent" / "splits"


# Benchmark frozen configs live with their exp_workspace folders
# (2026-08-18): hmeqa/mthm3d/hmeqa500 → exp_workspace/hmeqa · libero →
# exp_workspace/libero_* · vlnverse → exp_workspace/vlnverse · hm3d/mp3d →
# exp_workspace/objnav · ovon-* → exp_workspace/ovon · express →
# exp_workspace/express · rxr → exp_workspace/{bare,wp} · slam* →
# exp_workspace/slam_*. Each folder's exp.py registers (and cross-asserts)
# its entries via the loader at the bottom of this module.
BENCHMARK_FROZEN: dict[str, dict] = {}


# ── ObjectNav-family benchmarks# ObjectNav-family membership — driver branches import this tuple; the
# frozen configs + cells live in exp_workspace/objnav and exp_workspace/ovon.
OBJNAV_BENCHMARKS = ("hm3d", "mp3d", "ovon-seen", "ovon-syn", "ovon-unseen")


# ── slamr2r benchmark line (2026-08-17): SLAM-instrumented R2R-CE ──
# The promoted SLAM-instrument probe (formerly slamrun.py + the slam-frontier
# worktree): R2R-CE episodes on habitat-sim 0.3.3 via env_slam_vlnce — every
# coarse action decomposes into 5 micro-steps auto-fed to ORB-SLAM3 + an
# occupancy grid. TWO conditions per model: bare (slam_r2r_baseline_sdk_<m>)
# and instrumented (slam_r2r_01_sdk_<m>, extra instruments=1) — the arm delta
# is exactly three read-only query tools (get_pose/get_map/get_trajectory)
# plus the briefing addendum (prompts.SLAM_INSTRUMENT_ADDENDUM); this is the
# "egocentric self-localization wall" A/B. 口径 caveats: habitat 0.3.3
# dynamics, NOT 0.1.7-comparable — never pool with std_sdk_* R2R numbers;
# metrics come from env_slam_vlnce__evaluate (full NE/SR/SPL/OSR/nDTW suite,
# upgraded from the probe's stop-gated-geodesic-only rule; probe_slam_* run
# dirs predate the promotion and stay off-board).
# slamr2r frozen knobs are registered by the exp_workspace slam folders
# (each folder's exp.py carries an identical copy; the loader asserts they
# have not diverged).

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
# Turn cap for the goto-based surfaces (wp and hybrid). Measured
# at ~2.6 SDK turns per move (observe + reason + goto), so a 30-move budget is
# ~80 turns and a wp episode only turn-exhausts after ~38 moves — well past the
# move cap that is supposed to end it. Lowered 150 -> 100 so the whole wp column
# sits at the SAME cap as bare/nav (the R2R std, rented compute) and is directly
# comparable; the move cap in wp_bridge still binds first.
# NOTE: early recorded wp runs used a 150 cap — a different protocol; do
# not pool them with current wp numbers.
# RxR needs longer trajectories and lifts BOTH caps off-board:
#   --nonstd --set wp_max_moves=N max_turns=M
WP_MAX_TURNS = 100

# model key (board column) → model id passed to the harness.
# gpt slugs are USUALLY identical on the codex CLI and litellm's openai route —
# but NOT for gpt-5.6. Probed on codex 0.144.5: plain "gpt-5.6" 400s on
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
    # Opus 5 (released 2026-07). Probed on the R2R board via
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
    # local capacity probe: same shell, same pinned sampling, 3x the
    # weights — separates "the harness is wrong" from "9B cannot do the
    # cross-modal step" without needing an API key or the SDK path
    "qwen3.6-27b": "ollama_chat/qwen3.6-27b-udq4-std:latest",
    # qwen API column, served by Alibaba DashScope's OpenAI-compatible endpoint.
    # litellm's `openai/` route + the api_base in MODEL_EXTRA below; the key
    # rides OPENAI_API_KEY (set it to the DashScope key for the run shell —
    # litellm's openai route reads that var regardless of who the vendor is).
    # The mini adapter's _is_local_model() treats explicit-api_base models like
    # local ones: no anthropic/openai key assertion, cost tracking relaxed
    # (litellm has no price entry for dashscope slugs).
    "qwen3.7-plus": "openai/qwen3.7-plus",
    "qwen3.6-plus": "openai/qwen3.6-plus",
    # Qwen3.8 flagship (released 2026-07). Slug assumed from the
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
    # OpenAI now rejects function tools + reasoning_effort on gpt-5.6 in
    # /v1/chat/completions ("use /v1/responses or set
    # reasoning_effort to 'none'"). litellm's responses bridge keeps tools AND
    # reasoning, so mini reaches gpt-5.6 through it; gpt-5.5 chat stays fine.
    # NB the "openai/" prefix also flips mini's _is_local_model() heuristics
    # (no OPENAI_API_KEY assert, cost tracking relaxed) — acceptable.
    ("mini", "gpt-5.6"): "openai/responses/gpt-5.6",
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
# Thinking policy: thinking is ON for
# Claude in BOTH tiers (adaptive); only the effort param moves.
#   max     — elevated / ablation: Claude effort="max" (API-accepted on all
#             three board models, raw-probed; litellm's client gate is stale,
#             see mini_swe._unlock_claude_effort_max), GPT "xhigh" (server-
#             enumerated top; verified on codex's ChatGPT account too).
#   default — the effort a normal user gets: Claude sends NO effort param (the
#             API picks the model default), GPT = "medium" (codex/openai
#             default) EXCEPT codex+gpt-5.6, whose default is "low" (see
#             _tier_extra). Claude keeps adaptive
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
            # "low" on the codex CLI, not "medium";
            # gpt-5.5 keeps the medium GPT vendor default.
            return {"effort": "low" if model_key == "gpt-5.6" else "medium"}
        if harness == "mini":
            return ({"reasoning_effort": "medium"} if is_gpt
                    else {"thinking": "adaptive"})   # keep thinking, drop effort
    return {}

# The nav / wp-nav (ledger skill) and persona ablation conditions were
# retired with the skills/ dir — none entered the MIP paper.
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
    # ImagineVLN: the wp surface + goto-auto-observe. `imagine` adds a
    # world-model rollout sheet per candidate on every look; `imagine0` is the
    # matched control with the rollouts off — identical tools, identical
    # auto-observe, one fewer image set. Both write to OUTPUT_ROOTS["imagine"].
    "imagine": {"bare": True, "imagine": True, "imagine_rollouts": True},
    "imagine0": {"bare": True, "imagine": True, "imagine_rollouts": False},
}

# harness key → output root (the Monitor's SOURCE_ROOTS, unchanged)
OUTPUT_ROOTS = {
    "sdk": REPO_ROOT / "outputs" / "beta-coding-agent",
    "mini": REPO_ROOT / "outputs" / "beta-react-harness",
    "codex": REPO_ROOT / "outputs" / "beta-codex-agent",
    # embodied-harness shell (eharness/): two-tier planner/sub-agent over the
    # mini executor. Own root so its runs never mix with the mini baselines.
    "eharness": REPO_ROOT / "outputs" / "beta-eharness",
    # ImagineVLN runner (ImagineVLN/agent/run_mapgpt.py) writes the same
    # summary.json + episode_{i}.jsonl + live_{i}/ layout into its own root.
    "imagine": REPO_ROOT / "outputs" / "beta-imaginevln",
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
    imagine: bool = False           # ImagineVLN: wp surface + goto-auto-observe
    imagine_rollouts: bool = True   # ...with world-model rollout sheets (False = control)
    benchmark: str = "r2r"  # r2r (habitat-r2r std line) | hm3d | mp3d | ovon*
                            # | hmeqa | vlnverse | libero
    effort_tier: str | None = None  # default | max | None (untier-ed wp/local cells)
    extra: tuple = ()  # model/tier knobs as (key, value) pairs (hashable)
    max_turns: int | None = None  # None → STD_FROZEN's cap (std-v2: 200)
    exp_dir: str = ""  # exp_workspace folder owning this cell's execution
                       # code (bridge.py / prompts.py / nodeset); "" = the
                       # classic shared-code cells

    @property
    def extra_dict(self) -> dict:
        return dict(self.extra)

    @property
    def is_local(self) -> bool:
        """Served on our own GPU — no meter, no rate limit."""
        return self.model_id.startswith(("ollama", "hosted_vllm/"))

    @property
    def output_root(self) -> Path:
        # anything with the eharness organs lives under the eharness root —
        # 5173's embodied-harness source lists that root, and its 🧠 panel
        # reads the live_0/state.json the organ bridge writes (用户 2026-08-07:
        # 只要是 eh 就放到 embodied harness 5173 那里)
        if dict(self.extra).get("eh_bridge"):
            return OUTPUT_ROOTS["eharness"]
        if self.imagine:
            return OUTPUT_ROOTS["imagine"]
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
        imagine=cond.get("imagine", False),
        imagine_rollouts=cond.get("imagine_rollouts", True),
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

CELLS: dict[str, CellSpec] = {}
# (The bare / wp / hybrid / go2 / benchmark-line boards all register from
# their exp_workspace folders via the loader at the bottom of this module;
# only the research lines below — imagine, eharness, sdkeh — register here.)

# ── ImagineVLN line (2026-08-14): the wp surface + world-model rollouts ──
# imagine  = rollout sheets on every look;  imagine0 = the matched control.
# Both add goto-auto-observe, both write to outputs/beta-imaginevln, and both
# need the world-model service (ImagineVLN/service/mw_service.py, :9270) on top
# of the habitat + waypoint-predictor auto_hosts the wp cells already need.
IMAGINE_BOARD = (("mini", "qwen3.5-4b"), ("mini", "qwen3.5-9b"))
for _h, _m in IMAGINE_BOARD:
    for _c in ("imagine", "imagine0"):
        spec = _cell(_h, _m, _c)
        CELLS[spec.name] = spec

# ── eharness line (2026-08-04): the embodied-harness shell over the wp surface ──
# Two-tier planner/sub-agent, SAME model for both tiers and the V1 judge (the
# tiers separate context and permissions, not weights). Rides the wp toolset —
# the strongest small-model baseline (9b wp = 44) is the number to beat.
# Knobs are additive: the wp baseline cells above are untouched.
EH_EXTRA = {
    "paradigm": "solo",       # solo (default; solo-harness.html) | two_tier (S4 arm)
    "auto_view": 1,           # imageless moves carry their outcome view (transient)
    "verify_moves": 1,        # expect-vs-outcome reconciliation per move
    "compact_at": 12000,      # L2 trigger, estimated request tokens
    "image_window": 6,        # bounded visual context per sub-session (B′ arm)
    "subgoal_budget": 60,     # default env steps per delegate (planner may override)
    "subgoal_turn_cap": 30,   # turn cap per sub-session
    "planner_max_turns": 24,  # commander decisions per episode
    "verify": 1,              # V1 judge on (verify=0 → ablation arm)
}
EHARNESS_BOARD = (
    ("eharness", "qwen3.5-9b", "wp"),   # primary target (user 2026-08-04)
    ("eharness", "qwen3.5-4b", "wp"),
    ("eharness", "sonnet-5", "wp"),     # frontier reference on the same shell
    # bare (primitives) under the same shell — the wrapper is surface-agnostic
    # by construction, so this measures organ value on the WEAKEST interface
    # (9b bare baseline = 7: batching starvation + stopping wall; the organs
    # address the second, not the first — expect help, not rescue).
    ("eharness", "qwen3.5-9b", "bare"),
    ("eharness", "qwen3.5-4b", "bare"),
    ("eharness", "sonnet-5", "bare"),   # frontier reference, weakest interface
)
for _h, _m, _c in EHARNESS_BOARD:
    spec = _cell(_h, _m, _c)
    spec = replace(spec, extra=tuple(sorted({**dict(spec.extra), **EH_EXTRA}.items())))
    CELLS[spec.name] = spec

# Depth-waypoint arm (2026-08-05): the SAME organs and the same numbered-
# candidate surface as wp, but the candidates come from ONE 90° depth frame
# through eharness/depthmap.py instead of from smartway_waypoint__predict.
# It is a strictly better-informed proposer on three counts verified in code:
# the learned heatmap cannot express a stride past 3.00 m (12 bins × 0.25 m);
# it is fed per-frame min-max-normalised 8-bit depth, so absolute scale is not
# in its input at all; and it offers candidates with no reachability guarantee
# while step_hightolow blind-walks and slides. Geometry measures the free space
# it is proposing into. SAM 3 (sam_url) grounds the route's landmark nouns as
# advisory garnish; with the detector off the arm is pure geometry, which is
# the ablation that isolates the proposer.
_dwp_base = CELLS["std_eharness_qwen3.5-9b_bare"]
_dwp = replace(
    _dwp_base,
    name="std_eharness_qwen3.5-9b_dwp",
    extra=tuple(sorted({**dict(_dwp_base.extra), "dwp": 1, "judge_think": "1",
                        "sam_url": "http://127.0.0.1:9220",
                        # landmarks every look: with detector-friendly phrases from
                        # the splitter this is the strongest signal the model
                        # gets, and ~2.5 s/phrase is affordable next to a 9B turn
                        "landmark_every": 1, "dwp_max_moves": 60,
                        # block a move every 6 and demand a yes/no on
                        # "segment done?" / "stop condition met?"
                        "reflect_every": 6}.items())),
)
CELLS[_dwp.name] = _dwp

# Capacity probe on the SAME shell: 3x the weights, identical organs, identical
# pinned sampling. If the 27B follows the route where the 9B wanders, the
# bottleneck is the model's cross-modal step (tying "the detector sees the bar
# counter ahead" to "candidate 2 points there"), not the action space — and no
# further harness tuning will fix it. Local, so no API key and no SDK port.
# The other end of the scale column, same organs, same pinned sampling. The
# question is not "does the 4b succeed" — it is whether the harness's help
# CHANGES SIGN with model size. An in-context demo already did exactly that in
# this repo (helped the 4b, hurt the 27b), so a second instance would stop that
# being an anecdote. Local, so no key and no SDK port.
_dwp4 = replace(_dwp, name="std_eharness_qwen3.5-4b_dwp",
                model_key="qwen3.5-4b",
                model_id=MODELS["qwen3.5-4b"])
CELLS[_dwp4.name] = _dwp4

_dwp27 = replace(_dwp, name="std_eharness_qwen3.6-27b_dwp",
                 model_key="qwen3.6-27b",
                 model_id=MODELS["qwen3.6-27b"])
CELLS[_dwp27.name] = _dwp27



# S4 executor port: SDK loop (subscription auth) + eharness organs living in
# the bridge process (bridges/eharness_bridge.py); judge = local qwen — the
# perception-diversity arm the reception-desk failure motivated (executor and
# judge no longer share one pair of eyes). Derived via _cell (not a CELLS
# lookup): the bare std cells register later, from exp_workspace/bare/, and
# this line must NOT inherit that folder's exp_dir anyway.
_sdkeh_base = _cell("sdk", "sonnet-5", "bare", "default")
_sdkeh = replace(
    _sdkeh_base,
    name="std_sdkeh_sonnet-5_bare",
    extra=tuple(sorted({**dict(_sdkeh_base.extra),
                        "eh_bridge": 1, "judge_think": "1"}.items())),
)
CELLS[_sdkeh.name] = _sdkeh

# …and the same executor on the geometry line. This is the arm that asks the
# question the whole ablation ladder is built around: does the harness's help
# change SIGN with capability? An in-context demo already flipped between 4b and
# 27b in this repo; the menu, the pacing cap and the place memory are all
# candidates to flip too. Subscription auth (no key), local qwen judge.
_sdkeh_dwp = replace(
    _sdkeh,
    name="std_sdkeh_sonnet-5_dwp",
    extra=tuple(sorted({**dict(_sdkeh.extra), "dwp": 1,
                        "sam_url": "http://127.0.0.1:9220",
                        "landmark_every": 1}.items())),
)
CELLS[_sdkeh_dwp.name] = _sdkeh_dwp

# Opus 5 on the same sdk-executor + eharness-organs + geometry surface — the
# strong-model-first probe of the freshly landed §10/§12 payload (double image
# + telemetry). Subscription auth like every sdk cell; judge stays local qwen.
_sdkeh_dwp_opus = replace(_sdkeh_dwp, name="std_sdkeh_opus-5_dwp",
                          model_key="opus-5", model_id=MODELS["opus-5"])
CELLS[_sdkeh_dwp_opus.name] = _sdkeh_dwp_opus

# The go2 / hmeqa(+mthm3d/hmeqa500) / vlnverse / libero(×4 rungs) /
# objnav-family / express boards register from their exp_workspace folders.
# These two tuples remain the shared sdk columns the loader hands to every
# folder (register(..., sdk_models=OBJNAV_SDK_MODELS, claude_models=...)):
SDK_TRIO = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))
OBJNAV_SDK_MODELS = SDK_TRIO + (("sdk", "opus-5"),)

# slamr2r cells (the SLAM lineage) now live in exp_workspace/ — one folder
# per experiment, self-registering via each folder's exp.py (loaded at the
# bottom of this module). Only orchestration is shared.

# batches: the tiered main board carries the effort tier in the cell name
# (*_default = vendor-default main experiment, *_max = elevated ablation);
# Q/W/WQ are the untier-ed wp/local line.
BATCHES = {
    # ALL batches now register from the exp_workspace folders (bare: O5 Q8
    # Ad Bd Gd Xd A B G X Q RX · wp: W WQ RXW · hmeqa: EQ · libero_bare: LB
    # · objnav: OH OM O5N · ovon: OVS OVY OVU · slam folders: SL SLI SL2
    # SXI).
}


# ── experiment registry (paper §4, E-numbered) ────────────────────────────
# Explicit map from the plan's experiment numbers to board cells: request a run
# by number ("run E7") and eyeball the exact knobs here. `section`/`label` are
# the paper's grouping (not derivable); `cell` is the single source of truth for
# every frozen knob and reasoning-effort tier (resolve via get_cell / the
# `experiments` command). In scope: 4.1 main (default
# tier), 4.3 effort (max tier). OUT of scope and intentionally unregistered:
# E10-E13 (mini · qwen*), E21-E24 (Waypoint), E25-E28 (VLNVerse) — the qwen/wp
# CELLS above cover that line without E-numbers. The persona pair (E14/E15)
# was retired with the persona condition — never entered the paper.
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
    # new-model probes. Opus 5 slots into the same tables as the
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


# ── exp_workspace loader (2026-08-18): one folder = one experiment ──
# Each exp_workspace/<name>/exp.py self-registers its cells/batches/frozen
# knobs and carries its OWN bridge/prompts/nodeset copies. Loaded LAST so
# folders can rely on the full registry machinery above. A broken folder
# fails loudly — silently dropping cells would corrupt board runs.

def _load_exp_workspace() -> None:
    import importlib.util as _ilu
    root = Path(__file__).resolve().parent / "exp_workspace"
    if not root.is_dir():
        return
    for exp_py in sorted(root.glob("*/exp.py")):
        spec = _ilu.spec_from_file_location(f"_exp_{exp_py.parent.name}", exp_py)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.register(CELLS=CELLS, BATCHES=BATCHES,
                     BENCHMARK_FROZEN=BENCHMARK_FROZEN,
                     cell=_cell, replace=replace,
                     sdk_models=OBJNAV_SDK_MODELS, claude_models=CLAUDE_MODELS)


_load_exp_workspace()
