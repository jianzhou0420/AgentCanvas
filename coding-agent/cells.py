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
# Audit manifests: coding-agent/bridges/splits/*_n100_seed42.json (committed
# — data/ is gitignored, so the manifest is the ONE tracked record of what
# each mip split contains). Generator: sample_episodes.py, in git history at
# cecd19c — manifest + generator regenerate the materialized CSV
# byte-identically. The ObjectNav family (hm3d / mp3d / ovon×3) that shared
# this mechanism never entered the MIP paper and was removed
# (cells + helper last at a942483; bridges/splits at cecd19c).

SPLITS_DIR = REPO_ROOT / "coding-agent" / "bridges" / "splits"


# ── HM-EQA benchmark line: explore-eqa multiple-choice EQA ──
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
# the objnav mip splits. (Earlier runs carried raw val indices — the
# trial20/smoke run dirs are indexed in THAT original-CSV space.)
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
        # Camera-tilt actions 4/5 on the toolface: the native −30° pitch
        # makes near-overhead ceiling fixtures structurally unobservable.
        # Mask
        # with `--nonstd --set tilt_actions=0`: bridge validation, tool
        # description, and briefing all follow the one flag.
        "tilt_actions": True,
    }


# ── LIBERO manipulation line: coding-agent drives a robot arm ──
# The minimal two-tool interface (observe + step) re-embodied on a Franka
# Panda in LIBERO (Liu et al. 2023). step() takes the env's NATIVE 7-D
# continuous action space (one action = one OSC control tick) — no bridge-side
# discretization, unlike the nav lines' four primitives; see libero_bridge.py.
# No terminal action: LIBERO detects success from scene state, so an episode
# ends on task success or budget exhaustion, never by agent declaration.
def _libero_frozen() -> dict:
    return {
        "benchmark": "libero",
        "dataset": None,           # suite rides the split selector (env panel
        "split": "libero_object",  # maps a suite-named split to the suite)
        # Flat episode index over the suite: task_id = k % tasks_per_suite,
        # init state = k // tasks_per_suite — episodes 0-9 run every task once
        # at init state 0; 0-99 is 10 rollouts per task (init states 0-9), the
        # same n=100 shape as the R2R board.
        "episodes": "0-99",
        "tasks_per_suite": 10,  # libero_spatial / object / goal / 10 all have 10
        # Same turn/fuse posture as the hmeqa line (long tool-call episodes,
        # full-history resend cost tail): 150 turns + $18/episode USD fuse.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        # The env's own per-episode cap (_SUITE_MAX_STEPS = 2500 control
        # ticks); the bridge mirrors it to report remaining budget.
        "step_budget": 2500,
        "episode_timeout": 2400,
        "rgb_resolution": 256,  # env_libero server render size (recorded, not a knob)
    }


BENCHMARK_FROZEN: dict[str, dict] = {
    "hmeqa": _hmeqa_frozen(),
    "libero": _libero_frozen(),
}


# ── VLNVerse line: instruction-following VLN in Isaac Sim 5.1 ──
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


# ── ObjectNav-family benchmarks (re-armed 2026-08-16 from cecd19c): five
# lines — hm3d (objectnav_hm3d_v1), mp3d (objectnav_mp3d_v1) and the three
# OVON val splits — each its own board with its own frozen config, exactly
# as parked on 2026-08-03. Two changes against the parked state:
# (a) split naming follows the repo-wide TEAPS->MIP rename (a942483): the
#     materialized dataset-layer splits on disk are mip100/, the env panels
#     select split="mip100...", manifests live in bridges/splits/;
# (b) the bare surface is the SINGLE-TOOL step(actions) interface
#     (objnav_bridge_singlestep.py — user decision 2026-08-16, promoted
#     from the 2026-07-22 mechanism experiment where it beat the two-tool
#     observe/step alternation on the sonnet smoke, SR 0.2 vs 0.0). The
#     pre-park partial runs (hm3d_sdk_fable-5, 38 eps) used the TWO-tool
#     bridge — a different protocol; never pool them with these boards.
def _objnav_frozen(benchmark: str, dataset: str | None, split: str,
                   manifest_stem: str) -> dict:
    manifest = SPLITS_DIR / f"{manifest_stem}.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"{benchmark}: missing split manifest {manifest} — run "
            "coding-agent/sample_episodes.py --materialize")
    return {
        "benchmark": benchmark,
        "dataset": dataset,
        "split": split,       # mip100*: derived dataset-layer split
        "episodes": "0-99",   # the whole MIP100 split, manifest order
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Objnav deviates from std-v2's max_turns=200 (user decision 2026-07-22;
        # documented in the paper appendix): 150 turns + an $18/episode USD
        # fuse. The fuse is the CLI's own --max-budget-usd — session ends with
        # subtype error_max_budget_usd, scored as clean truncation (like
        # error_max_turns), never retried. Motivation: hm3d smokes showed
        # cap-burning episodes cost 2-4x a success (worst $38.9) because
        # full-history resend makes cost grow ~quadratically in turns.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,  # = habitat's ObjectNav max_episode_steps
        "episode_timeout": 2400,
    }


# hm3d: objectnav_hm3d_v1 (val: 2000 eps / 20 scenes / 6 categories)
# mp3d: objectnav_mp3d_v1 (val: 2195 eps / 11 scenes / 21 categories)
# ovon-*: HM3D-OVON (3000 eps / 36 scenes per val split); no dataset
# selector on its panel (one dataset — the split IS the axis) -> None.
OBJNAV_FROZEN: dict[str, dict] = {
    "hm3d": _objnav_frozen("hm3d", "hm3d_v1", "mip100",
                           "hm3d_val_n100_seed42"),
    "mp3d": _objnav_frozen("mp3d", "mp3d_v1", "mip100",
                           "mp3d_val_n100_seed42"),
    "ovon-seen": _objnav_frozen("ovon-seen", None, "mip100_seen",
                                "ovon_val_seen_n100_seed42"),
    "ovon-syn": _objnav_frozen("ovon-syn", None, "mip100_seen_synonyms",
                               "ovon_val_seen_synonyms_n100_seed42"),
    "ovon-unseen": _objnav_frozen("ovon-unseen", None, "mip100_unseen",
                                  "ovon_val_unseen_n100_seed42"),
}
OBJNAV_BENCHMARKS = tuple(OBJNAV_FROZEN)  # the five ObjectNav-family lines
BENCHMARK_FROZEN.update(OBJNAV_FROZEN)


# ── MT-HM3D benchmark line (2026-08-17): env_hmeqa's second corpus ──
# MemoryEQA's benchmark (Zhai et al., arXiv 2505.13948): 1,587 multi-choice
# questions over 828 HM3D scenes, ridden through the SAME nodeset, bridge,
# briefing and frozen knobs as the hmeqa line — the ONLY deltas are the
# panel dataset field (mt_hm3d) and the split corpus. Split mip100 is
# LABEL-stratified, not scene-stratified (user decision 2026-08-16: scene
# strata degenerate on 828 scenes/100 slots and skewed Counting to 63%);
# manifest records the method. 口径 caveats when reporting: answers skew
# "A" (66/100 in mip100, 67.4% in the full corpus — ALWAYS report the
# constant-A control beside SR), and published MemoryEQA numbers are on
# the full 1,587.
def _mthm3d_frozen() -> dict:
    manifest = SPLITS_DIR / "mt_hm3d_val_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"mthm3d: missing split manifest {manifest} — run "
            "coding-agent/sample_episodes.py --benchmark mt_hm3d --materialize")
    derived = REPO_ROOT / "data" / "hm3d" / "mt_hm3d" / "questions_mip100.csv"
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"mthm3d: missing materialized split {derived} — regenerate with "
            "sample_episodes.py --benchmark mt_hm3d --materialize")
    return {
        "benchmark": "mthm3d",
        "dataset": "mt_hm3d",  # env_hmeqa _DATASET_SPECS key (panel field)
        "split": "mip100",
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same posture as the hmeqa line: 150 turns + $18 fuse, std 500-step
        # budget enforced bridge-side, tilt actions on.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
        "tilt_actions": True,
    }


BENCHMARK_FROZEN["mthm3d"] = _mthm3d_frozen()


# ── EXPRESS-Bench benchmark line (2026-08-17): free-form EQA on HM3D ──
# Jiang et al. ICCV 2025: 2,044 open-vocab QA over 174 scenes, env_express +
# express_bridge.py (answer(text)), scored by the benchmark's gpt-4o-mini
# judge over answer + final frame → C / C* / E_path / d_T (driver-side).
# Split mip100 = scene-stratified seed-42 sample of the FULL 2,044 ("all":
# the benchmark's train scenes are eval too; published numbers are on the
# full set — state the 口径). Deviation from the benchmark-native protocol
# (documented in the bridge docstring): discrete 0.25 m / 30° actions via
# env_express__step_discrete replace the native free-pose teleports, and
# the std 500-step budget replaces the scene-sized teleport budget.
def _express_frozen() -> dict:
    manifest = SPLITS_DIR / "express_all_n100_seed42.json"
    if not manifest.exists():  # provenance must exist — never run unaudited
        raise FileNotFoundError(
            f"express: missing split manifest {manifest} — run "
            "coding-agent/sample_episodes.py --benchmark express --materialize")
    derived = (REPO_ROOT / "data" / "hm3d" / "express_bench"
               / "express-bench_mip100.json")
    if not derived.exists():  # the split the env panel selects must exist too
        raise FileNotFoundError(
            f"express: missing materialized split {derived} — regenerate "
            "with sample_episodes.py --benchmark express --materialize")
    return {
        "benchmark": "express",
        "dataset": None,       # single-JSON benchmark: no dataset selector
        "split": "mip100",
        "episodes": "0-99",
        "episodes_manifest": str(manifest.relative_to(REPO_ROOT)),
        # Same posture as the hmeqa/mthm3d lines: 150 turns + $18 fuse,
        # std 500-step budget enforced bridge-side.
        "max_turns": 150,
        "max_budget_usd": 18.0,
        "step_budget": 500,
        "episode_timeout": 2400,
    }


BENCHMARK_FROZEN["express"] = _express_frozen()


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

# main board: bare-only, effort-tiered. Design: each closed harness vs the open
# mini harness on the SAME models — claude side sdk↔mini (sonnet/opus/fable),
# openai side codex↔mini (gpt-5.5 / gpt-5.6). (mini·fable-5 completes the
# claude sdk↔mini pairing; runs via litellm→Anthropic at the same
# default=high regime as mini·sonnet/opus.)
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
    ("mini", "qwen3.8-plus"),   # newest Qwen flagship — sweep row
    ("mini", "qwen3.5-plus"),   # API sibling of the local 4b/9b column
)

# open-weight column: the same mini harness, locally served — the descend-to-
# small read of the bare surface. (The nav pairing that once rode here was
# retired with the skill condition.)
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

# Opus 5 probe: three targeted R2R cells, bare surface, so the newest
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
# judge no longer share one pair of eyes).
_sdkeh_base = CELLS["std_sdk_sonnet-5_bare_default"]
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

# real-robot pilots: the sdk cells re-embodied on the Unitree Go2
# via go2_bridge.py. BARE surface — observe + step only, no look_around, no
# skill (matches the habitat main board's condition),
# default effort. NOT part of the std board: instruction is operator-supplied
# (--set instruction=...), the driver skips evaluate (no ground truth — success
# is judged by a human from the recording), and the server is the go2 host on
# the robot's machine, not an agentcanvas backend.
GO2_BOARD = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))
for _h, _m in GO2_BOARD:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"go2_{_h}_{_m}", condition="go2", go2=True)
    CELLS[spec.name] = spec

# HM-EQA board: sdk trio, bare surface, default effort — cell
# names carry the benchmark prefix (hmeqa_sdk_fable-5), parallel to std_* and
# go2_*. Servers: env_hmeqa auto_host (ac-hmeqa env); the driver refuses a
# mismatched server name.
SDK_TRIO = (("sdk", "sonnet-5"), ("sdk", "opus-4.8"), ("sdk", "fable-5"))
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"hmeqa_{_h}_{_m}", condition="hmeqa",
                   benchmark="hmeqa")
    CELLS[spec.name] = spec

# VLNVerse board: same sdk trio, bare surface, default effort.
# Server: env_vlnverse auto_host (ac-vlnverse env, Isaac 5.1 worker behind it);
# the driver refuses a mismatched server name. wp/hybrid cells are NOT built
# for this line — the driver refuses that combination outright (CCW pano
# indexing + a habitat-trained waypoint predictor); see driver.py.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"vlnverse_{_h}_{_m}", condition="vlnverse",
                   benchmark="vlnverse")
    CELLS[spec.name] = spec

# LIBERO board: sdk trio, bare surface (observe + 7-D step only),
# default effort — cell names carry the benchmark prefix (libero_sdk_fable-5),
# parallel to std_* / go2_* / hmeqa_*. Servers: env_libero auto_host
# (ac-libero env, MUJOCO_GL=egl); the driver refuses a mismatched server name.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"libero_{_h}_{_m}", condition="libero",
                   benchmark="libero")
    CELLS[spec.name] = spec

# LIBERO full condition: the SENSOR rung of the interface ladder,
# after the fable ep0 anatomy showed the bare wall is the depth/height DoF.
# Same two tools; observe adds the wrist view + proprio readout, step reports
# measured EE movement, and auto-observe carries the post-move views (halves
# the look-move turn cost, nav-line precedent). Sensors and feedback only —
# no skills, no planner, no task logic; see libero_bridge.py.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"libero_{_h}_{_m}_full",
                   condition="libero_full", benchmark="libero", bare=False,
                   extra=_base.extra + (("auto_observe", 1),))
    CELLS[spec.name] = spec

# LIBERO toolbox condition: max out the tool surface first, attribute
# downward later. Atomic tools —
# observe_third_person / observe_wrist / get_state / get_objects (the
# simulator's GT scene readout) / move_to servo / gripper macro — plus the
# native step() escape hatch, all over the env's frozen VoxPoser-era nodes.
# Deliberately PRIVILEGED (GT positions): this rung asks "does the loaded
# surface complete tasks at all", not "is the interface minimal"; the
# minimal-interface story keeps bare/full, and ablation walks down from
# here one tool at a time. See libero_bridge.py TOOLBOX.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"libero_{_h}_{_m}_tb",
                   condition="libero_toolbox", benchmark="libero", bare=False,
                   extra=_base.extra + (("toolbox", 1),))
    CELLS[spec.name] = spec

# LIBERO toolbox-vision condition: the toolbox with its ONE privileged
# tool swapped out — get_objects
# (sim GT) → pixel_to_3d (depth backprojection over the new
# env_libero__pixel_to_3d node; camera geometry + depth buffer only).
# Everything else identical, so _tb vs _tbv prices exactly the perception
# privilege. sonnet _tb baseline: 9/10.
for _h, _m in SDK_TRIO:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"libero_{_h}_{_m}_tbv",
                   condition="libero_toolbox_vision", benchmark="libero",
                   bare=False,
                   extra=_base.extra + (("toolbox", 1), ("toolbox_gt", 0)))
    CELLS[spec.name] = spec

# ObjectNav-family boards (re-armed 2026-08-16): bare surface — which for
# this family means the single-tool step() interface, see OBJNAV_FROZEN
# note — default effort, sdk column. The go2/hmeqa trio plus opus-5: the
# 2026-08-16 baseline sweep runs the opus-5 row first.
OBJNAV_SDK_MODELS = SDK_TRIO + (("sdk", "opus-5"),)
for _bench in OBJNAV_BENCHMARKS:
    for _h, _m in OBJNAV_SDK_MODELS:
        _base = _cell(_h, _m, "bare", "default")
        spec = replace(_base, name=f"{_bench}_{_h}_{_m}", condition=_bench,
                       benchmark=_bench)
        CELLS[spec.name] = spec

# MT-HM3D board (2026-08-17): same sdk column as the objnav re-arm —
# servers: env_hmeqa auto_host (ac-hmeqa env); the driver refuses a
# mismatched server name.
for _h, _m in OBJNAV_SDK_MODELS:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"mthm3d_{_h}_{_m}", condition="mthm3d",
                   benchmark="mthm3d")
    CELLS[spec.name] = spec

# EXPRESS-Bench board (2026-08-17): same sdk column — servers: env_express
# auto_host (ac-hmeqa env, EXPRESS_PYTHON override); the driver refuses a
# mismatched server name. The judge additionally needs OPENAI_API_KEY in
# the driver's environment (gpt-4o-mini, driver-side).
for _h, _m in OBJNAV_SDK_MODELS:
    _base = _cell(_h, _m, "bare", "default")
    spec = replace(_base, name=f"express_{_h}_{_m}", condition="express",
                   benchmark="express")
    CELLS[spec.name] = spec

# slamr2r cells (the SLAM lineage) now live in exp_workspace/ — one folder
# per experiment, self-registering via each folder's exp.py (loaded at the
# bottom of this module). Only orchestration is shared.

# batches: the tiered main board carries the effort tier in the cell name
# (*_default = vendor-default main experiment, *_max = elevated ablation);
# Q/W/WQ are the untier-ed wp/local line.
BATCHES = {
    # new-model probes: Opus 5 trio (mini sweep + sdk default/max
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
    # LIBERO board — servers: env_libero auto_host (ac-libero env)
    "LB": [f"libero_sdk_{m}" for m in CLAUDE_MODELS],
    # ObjectNav-family boards — one batch per benchmark line (peer lines, not
    # one merged "objnav" batch). Servers: env_objnav auto_host for OH/OM,
    # env_ovon auto_host for the OV* trio (the driver refuses a mismatch).
    "OH": [f"hm3d_sdk_{m}" for m in CLAUDE_MODELS],
    "OM": [f"mp3d_sdk_{m}" for m in CLAUDE_MODELS],
    "OVS": [f"ovon-seen_sdk_{m}" for m in CLAUDE_MODELS],
    "OVY": [f"ovon-syn_sdk_{m}" for m in CLAUDE_MODELS],
    "OVU": [f"ovon-unseen_sdk_{m}" for m in CLAUDE_MODELS],
    # the 2026-08-16 opus-5 baseline sweep across all five objnav boards
    "O5N": [f"{b}_sdk_opus-5" for b in OBJNAV_BENCHMARKS],
    # slamr2r A/B — servers: env_slam_vlnce auto_host (ac-habitat033 env)
    # SL / SLI / SL2 (slam lineage) are registered by exp_workspace folders
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
