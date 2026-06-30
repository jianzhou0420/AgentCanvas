# aflow — AFlow-style search on AgentCanvas

> **Framing.** AFlow-style port of Zhang et al. 2024, *"AFlow:
> Automating Agentic Workflow Generation"* (ICLR 2025 Oral). The paper
> sells AFlow as "MCTS over workflows"; the **implementation** in
> upstream `optimizer.py` is narrower: *score-softmax + uniform-mix
> sampling over a flat round list, with per-parent anti-replay
> filtering on modification descriptions*. There is no tree traversal,
> no UCB, no visit count, no backprop. The "tree" exists only as a
> `father_node` field that's written but never read as a tree. Treat
> the MCTS framing as rhetorical; this README documents what AFlow
> actually runs.
>
> **Not a 1:1 reproduction — "AFlow-style".** Upstream AFlow searches
> by composing a *given, curated operator set* (Custom, ScEnsemble,
> Programmer, …) — the operator set is a fixed run input. This port
> has no equivalent operator library; it searches free-form
> changes over the live node catalog instead — each iter's
> implementer sub-agent edits the graph + nodesets natively. What is
> ported faithfully is AFlow's *search policy* (softmax-mix parent
> selection + per-parent anti-replay + experience); its *search
> space* (operator composition) is NOT reproduced. Read "port"
> throughout this README as "AFlow-style", not "1:1 reproduction".

Sibling of `.claude/commands/architect/adas-subagent/`. Reuses the 5-skill
layout, archive contract, and `vN/iter_M` run-dir scheme. Replaces
ADAS's 3-call Reflexion with AFlow's single-call + anti-replay; adds
parent selection at the head of each iter; adds per-parent experience
aggregation. Evaluation is **two-tier** (smoke / perf) — same shape as
adas-subagent. (The three-tier `smoke / search / perf` design from
2026-05-20 was reverted on 2026-05-25; see §3 for the v0 mapgpt failure
that motivated this.)

> **Cross-variant contract**:
> `.claude/commands/architect/_common/files-contract.md` defines the
> shared iteration files (run-dir layout, resolve protocol,
> `{graph}.yaml` schema, edit whitelist, backend API). aflow
> reuses it; this README only documents the **deltas** vs adas-subagent.
>
> This variant's concrete file set is declared in
> `config.yaml § manifest` — every file it writes, classified into a
> global file-type, with purpose / schema / access (per
> `_common/files-contract.md § 4`). That manifest is the single source
> of truth for file *identity*; this README narrates only rationale.

## Three structural contracts

### 1. One iter = one Claude conversation (anti-replay retry chain)

The `loop` skill owns the **main Claude conversation** for an entire
iter. That conversation invokes `proposer`, `implementer`, and
`evaluator` as same-conversation phases.

**Inside `proposer`**, instead of adas-subagent's 3-call Reflexion, the
conversation thread is the **anti-replay retry chain** from upstream
`optimizer.py:132–183`:

```
while True:
    parent     = select_round(get_top_rounds(K))      # softmax-mix sample
    experience = load_parent_experience(parent)        # success + failure list
    prompt     = build_optimize_prompt(parent, experience, log_x3)
    response   = LLM.call_with_format(GraphOptimize)   # one call
    if response.parse_failed:
        continue                                        # FormatError fallback
    if check_modification(response.modification, parent.experience):
        continue                                        # duplicate — resample
    break
```

Bounded by a soft cap `K_retry` (default 5) to avoid spinning forever
on near-converged archives (Q5 in `docs/pages/aas/reference/aflow/algorithm.html`).

**Inside `implementer`**, the retry loop (`retry_max=3` from config) is
inherited from the **method-free `_common/implementer.md`** as an
AgentCanvas-substrate add-on — upstream AFlow has no debug retry
because workflows are stateless Python eval on cheap MMLU-style
benchmarks. Each retry is one independent
`Agent({subagent_type: "general-purpose"})` sub-agent spawn; retry
is triggered only by runtime failure (crash / incomplete / step=0 /
malformed metric), not by low/zero metric values.

Maximum sub-agent invocations per iter:
- proposer: 1 (the single anti-replay loop runs as a Python helper
  + N LLM calls; whether each LLM call is a sub-agent spawn or a
  Claude thinking-pass is decided per Q below)
- implementer: up to 3 (debug retry sub-agents)
- evaluator: 0 (pure infrastructure)

### 2. archive.jsonl is independent of iter dirs + carries 2 new fields

Located at `outputs/design_runs/aflow/{graph}/v{N}/archive.jsonl`. Same
per-vN, JSONL, append-only contract as adas-subagent. **Two extra fields**
relative to adas-subagent:

- `parent_iter_id` — the iter chosen by `select_round` for this
  child. Lets the experience loader group entries by parent for the
  "Absolutely prohibit X" injection.
- `modification` — verbatim natural-language description of the
  change (from the LLM's `<modification>` field). Lets
  `check_modification` detect duplicates against parent's prior
  attempts.

A bare numeric `score` is also stored alongside the human-readable
`fitness` string, because `select_round`'s softmax needs a float,
not `"95% Bootstrap CI: (lo%, hi%), Median: x%"`. Convention:
`score = bootstrap_median` — the bootstrap median is already a
fraction in [0, 1] (as returned by `bootstrap_confidence_interval`),
stored verbatim with no rescaling. `select_round` itself does the
`score * 100` scaling internally, matching upstream `data_utils.py`.

Entry schema:

```json
{
  "generation": int | "initial",
  "iter_id": "iter_N",
  "parent_iter_id": "iter_M" | null,
  "name": "<from LLM>",
  "thought": "<verbatim from proposer's LLM output>",
  "modification": "<verbatim from <modification> field>",
  "graph_summary": { "nodes": [...], "wires": [...], "loop": {...} },
  "diff_narrative": "<plain-text What-changed>",
  "fitness": "95% Bootstrap CI: (lo%, hi%), Median: x%",
  "score": 0.715
}
```

Pre-append cleanup: none required. Neither `reflection` (aflow has
no Reflexion chain) nor `debug_thought` (no longer in the schema —
implementer's retry sub-agents edit the overlay natively and return
`{edit_summary, extra_targets}`) appears in proposer / implementer
output.

### 3. Two-tier evaluation (smoke + perf, with F3 deviation)

Two eval tiers, two `{graph}.yaml` blocks:

- **Smoke** (`smoke_<graph>`, ~5 eps) — runs inside `implementer`'s
  debug retry loop. Cheap gate: "does the patch even run?".
- **Performance** (`perf_<graph>`, full paper-comparable set) — the
  **per-iter ranking eval**, also the source of headline numbers.
  `evaluator` runs this every iter (`config.yaml
  evaluator.profile_key = perf_<graph>`) and writes neutral metrics;
  `loop.md`'s Atomic Writer turns `acc_list` into `fitness_str` + the
  bare `score` that `select_round`'s softmax consumes. **Every
  `archive.jsonl` score is a `perf_` single-pass value.** Step 8 adds
  one verification rerun on top-1, top-2, and baseline to calibrate
  against LLM run-to-run stochasticity (the F3 caveat below); reruns
  live in `final_report.md`, not `archive.jsonl`.

**Why two tiers (revert from three-tier 2026-05-20 → 2026-05-25):**
The 2026-05-20 design inserted a `search_<graph>` middle tier — a
~30-ep stratified subset — to avoid the per-iter perf cost. The
v0 mapgpt_mp3d run exposed why this didn't work:

- Subset size N=30 → binomial SE ≈ 9pp at p=0.5; with
  `validation_rounds=1` on top, the noise floor was ~9pp.
- Genuine gains in this regime are ~5–10pp.
- Result: the loop ranked iters on a signal smaller than its noise.
  iter_6 won the search at +13pp over baseline; on perf it was
  −4.2pp. The whole archive was a noise gradient.

Running `perf_<graph>` every iter at N=216 drops the binomial SE to
~3.4pp, restoring SNR > 1 against the ~10pp gains we're trying to
detect. Cost goes up ~5× per iter ($ wise), but wall is similar
(perf was paid 2× post-loop in the old design, plus 20 search runs
that turned out to be ~the same wall as perf runs at the right
worker_count). See `outputs/design_runs/aflow/mapgpt_mp3d/v0/
final_report.md` for the full v0 postmortem.

**F3 deviation from upstream**: upstream averages
`validation_rounds=5` separate passes per round; we still run 1 pass.
The two-tier revert fixes *episode-sampling* noise (small-N → full-N)
but not *LLM run-to-run* stochasticity (temperature=1 nondeterminism
across passes). The bootstrap CI in `fitness_str` resamples *within*
the one pass — it captures the first kind, not the second. This is
the one forced algorithmic adaptation from upstream that remains.

**F3 is not localized to evaluation cost — it propagates into the
search policy.** `select_round` softmax-ranks parents by `score`, and
`check_convergence` watches top-3 `score` stability; both *assume*
`score` is comparable across iters. Under a single pass, the unmodeled
run-to-run variance rides on every `score`: a genuinely-mediocre iter
that drew a lucky pass can be mis-ranked into the top-K and attract
children, and the convergence predicate becomes close to meaningless
(it is one reason `z=0` convergence is left advisory — see
§ "Termination"). Mitigations in this port:

1. **N=full perf** removes the dominant noise component (episode
   sampling) — see two-tier rationale above.
2. **Verification reruns** in step 8 (top-1, top-2, baseline each get
   one additional pass) calibrate the residual LLM stochasticity and
   flag when the in-loop top iter was a lucky pass.
3. Raise `aflow.lambda_uniform` if `select_round` collapses onto one
   parent (loop.md's parent-distribution diagnostic flags this).
4. Future option (not enabled): empirical noise floor — run baseline
   3× before launching a new run to set a "minimum credible Δ"
   threshold; iters within that threshold get treated as ties.

## 5 skills

| Skill | Role | Invoked by |
|------|------|-----------|
| `/architect:aflow:understand` | P0 — load context (files-contract + concept/contract docs + Section 1 invariants + graph state). Read-only. | manual (or `loop` P0) |
| `/architect:aflow:loop` | Orchestrator. Owns the per-iter Claude conversation. Drives proposer→workspace-checkout→implementer→evaluator, Atomic Writer, archive append, convergence check, termination. | user (entry point) |
| `/architect:aflow:proposer` | Single LLM call + anti-replay retry chain. Parent selection (softmax-mix) runs INSIDE this skill. Output: staging `proposal.md` with `parent_iter_id` in frontmatter. | `loop` (each iter) |
| `/architect:aflow:implementer` | apply patch → Smoke → classify → debug retry ≤3. SKIP on exhaust. **Reused from adas-subagent** with one preamble: skips the bootstrap step if `.staging/iter_n/active_workspace/` is already populated (loop's Workspace Checkout did it). | `loop` (each iter, after proposer + Workspace Checkout) |
| `/architect:aflow:evaluator` | Method-free eval runner — `experiment:run perf_<graph>` (full set, per iter) → neutral staging `metrics.json` (`acc_list` + `primary_metric_value` + `secondary_metrics`). bootstrap_CI / `fitness_str` + bare numeric `score` are added later by loop's Atomic Writer. | `loop` (each iter, after implementer; also iter_0 baseline; also step-8 verification reruns) |

## Run-dir layout (delta vs files-contract)

```
outputs/design_runs/aflow/{graph}/v{N}/
├── archive.jsonl                # per-vN, append-only (entries carry parent_iter_id + modification + score)
├── trace.md
├── lineage.md
├── iter_0/                      # baseline only (NO 7-seed palette in aflow)
│   ├── graph.json
│   ├── active_workspace/        # empty/absent for iter_0
│   ├── metrics.json             # with fitness_str + score
│   └── (no proposal.md, no debug_log.md)
├── iter_1/                      # first evolved gen
│   ├── parent.txt               # = parent_iter_id from proposal.md (NEW: not always parent=iter_{n-1})
│   ├── proposal.md              # frontmatter includes parent_iter_id + modification
│   ├── graph.json               # convenience copy of active_workspace/graphs/{graph}.json
│   ├── active_workspace/        # bootstrapped from parent's active_workspace (parent = softmax-sampled, NOT archive head)
│   ├── debug_log.md
│   ├── metrics.json             # with fitness_str + score
│   ├── summary.csv, export.json
│   └── (no analysis.md, no revision.md, no report.md)
├── .staging/
│   └── iter_n/                  # populated by proposer/implementer/evaluator + loop's Workspace Checkout; mv to iter_n/ on success
└── .loop_state/
```

## Termination

- `--max-iters` reached (default 20, mirrors upstream `max_rounds`)
- User-touched STOP file: `{RUN_DIR}/.loop_state/STOP`
- Consecutive SKIPs ≥ K (`--max-consecutive-skips`, default 3)
- Convergence check fires: top-3 mean stability over 5 consecutive
  rounds within `z·σ` — verbatim upstream, advisory only. Under F3
  (`validation_rounds=1`) every per-round σ is 0, so `z·σ` collapses
  to 0 and the check fires only on *exact* top-3-mean equality
  **regardless of `z`** — the `aflow.convergence_z` tunable is inert
  unless multi-pass eval is reintroduced (see loop.md step 6). Note:
  upstream `optimizer.py` defaults
  `check_convergence=False` — the convergence branch is OFF unless
  explicitly enabled, so upstream by default runs the full
  `max_rounds`. This port runs the check as an always-on advisory;
  that is a mild divergence from upstream's default.

## What this variant does NOT do

- **No 3-call Reflexion** — replaced by single-call + anti-replay
  (D1).
- **No 7-seed reference palette** — verbatim upstream, only `iter_0`
  baseline is pre-seeded.
- **No archive-head parent assumption** — every iter, parent is
  freshly sampled via `select_round` (D2).
- **No tree traversal / UCB / visit counts** — the "MCTS" framing is
  rhetorical; actual mechanism is flat softmax sampling.

## What's preserved verbatim from upstream

| aflow element | Upstream anchor | Verbatim? |
|---|---|---|
| `get_top_rounds(K)` + round_1 unconditional inclusion | `data_utils.py:40–59` | Yes (logic) |
| `select_round` softmax-mix sampling, `α=0.2`, `λ=0.3` | `data_utils.py:61–109` | Formula: yes. Defaults: match upstream CODE (`DEFAULT_ALPHA`/`DEFAULT_LAMBDA`) — but paper Eq.3 states `α=0.4`, `λ=0.2`; upstream paper/code disagree, this port follows the code |
| `check_modification` anti-replay | `experience_utils.py:69–80` | Yes (semantics — but see Q3 below for normalization) |
| `experience.json = {father, modification, before, after, succeed}` aggregated as `{parent: {success, failure}}` | `experience_utils.py:12–53, 91–95` | Yes (logic) — implemented by reading `archive.jsonl` instead of per-iter `experience.json` files |
| `WORKFLOW_INPUT` template (`experience`, `score`, `graph`, `prompt`, `operator_description`, `log×3`) | `optimize_prompt.py:16–35` | Yes (slot list) — adapted text (see deltas below) |
| FormatError regex-extract fallback | `optimizer.py:164–174` | Yes (semantics) |
| `validation_rounds=5` averaging | `optimizer.py:194` | **F3 deviation** — reduced to 1 pass + bootstrap_CI |
| `check_convergence(top_k=3, z=0, consecutive_rounds=5)` | `convergence_utils.py:68–113` | Yes (algorithm) — `z` exposed as `exp.yaml` tunable |
| Failed iter not in archive | `optimizer.py:96, 105` (`score=None` → skip child write) | Yes (semantics) |
| `bootstrap_confidence_interval(acc_list, 100000, 0.95)` | ADAS `utils.py:31–76` (not upstream AFlow — AFlow has no bootstrap) | aflow inherits from adas-subagent to handle the F3 deviation; not an AFlow-upstream element |
| `fitness_str = "95% Bootstrap CI: ..."` | ADAS `utils.py:76` (same) | Same — inherited |

## What's intentionally adapted (not verbatim)

- **Workflow representation**: upstream mutates `class Workflow`
  Python source as a free-form string in `<graph>` XML tag + a
  separate prompt file in `<prompt>` tag, applied by a wholesale
  rewrite. We instead use a **change spec** (`{intent, targets}`):
  the proposer describes the change in prose and lists the
  `{graph}.json` + nodeset `.py` files it touches, and the
  implementer sub-agent edits those files natively (see
  `_common/implementer.md`). Upstream's `<graph>` and `<prompt>`
  slots collapse into one `intent`. (The earlier typed `graph_edits`
  op DSL was retired 2026-05-20 — an op enum for a deterministic
  replayer was redundant once the implementer is an agent.)
- **Per-parent experience storage**: upstream writes
  `processed_experience.json` per parent round dir. We aggregate from
  `archive.jsonl` at read time, keyed by `parent_iter_id`. Same
  semantics; one fewer file.
- **`modification` normalization** (Q3 in algorithm.html): upstream
  does exact string equality. We default to whitespace-normalized
  lowercase comparison (cheap deviation against trivial paraphrase
  drift); tunable via `aflow.replay_norm` in `exp.yaml`. Even with
  normalization `check_modification` rarely fires — LLM free-form
  `modification` text almost never collides — so the real anti-replay
  pressure is the experience block in the prompt, not the string
  check. The check + bounded `replay_max_retries` port upstream's
  (equally weak) guard faithfully; `replay_norm: embed` is the only
  setting that makes it actually bite.
- **Anti-replay soft cap** (Q5): upstream's `while True` is unbounded.
  We cap at `aflow.replay_max_retries: 5` (configurable); exceeding
  → SKIP iter.
- **Two-tier evaluation**: same shape as upstream (which scores every
  round on the full validation set). We add a `smoke_` gate inside
  `implementer` for AgentCanvas-substrate runtime sanity, then run
  `perf_<graph>` every iter as the ranking signal. The pre-2026-05-25
  three-tier `smoke / search / perf` design — which used a small frozen
  `search_` subset for per-iter ranking and ran `perf_` once post-loop
  — was reverted because the subset's noise floor (~9pp at N=30) was
  larger than the true between-iter gains (~5–10pp); see §3.
- **Iter dir + staging dir**: upstream is largely stateless
  (`write_graph_files` overwrites). We need the staging-then-atomic
  pattern for crash safety.

## Key tunables in `{graph}.yaml`

```yaml
# Standard files-contract profiles (two tiers — see §3)
smoke_<graph>: { episode_count: 5,   worker_count: 1, ... }    # implementer debug gate
perf_<graph>:  { episode_count: 216, worker_count: 40, ... }   # every iter as ranking signal + step-8 verification reruns

# aflow-specific
aflow:
  sample: 4                    # top-K parents for select_round (upstream default)
  alpha: 0.2                   # softmax temperature (upstream default)
  lambda_uniform: 0.3          # uniform-mix weight (upstream default)
  replay_max_retries: 5        # soft cap on anti-replay loop (NEW vs upstream)
  replay_norm: lower_ws        # "verbatim" | "lower_ws" | "embed" (Q3)
  convergence_z: 0.0           # top-3 stability tolerance (upstream default)
  convergence_top_k: 3
  convergence_consecutive: 5
```

## Quick reference

- Files contract: `.claude/commands/architect/_common/files-contract.md`
- Site page (paper analysis + Section 2 port design):
  `docs/pages/aas/reference/aflow/algorithm.html`
- Upstream code: `third_party/AFlow/scripts/{optimizer,optimizer_utils/data_utils,optimizer_utils/experience_utils,prompts/optimize_prompt}.py`
- adas-subagent skills (reused for implementer / files-contract):
  `.claude/commands/architect/adas-subagent/`
