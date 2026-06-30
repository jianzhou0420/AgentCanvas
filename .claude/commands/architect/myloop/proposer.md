# THINK phase — produce the next ExperimentSpec

> **Required reading before invoking**:
> - `myloop/README.md` — the myloop mental model (knowledge-distillation orchestrator)
> - `myloop/schemas.md` — every file's schema, especially `ExperimentSpec`
> - `_common/files-contract.md` § "Edit whitelist" — what the EXPERIMENT apply-step is later allowed to patch

This skill IS the THINK phase of one myloop iter. It is one half of
the hard-paired `THINK → EXPERIMENT` rhythm. The other half is
EXPERIMENT (`loop.md § 3c`, inline in the loop — myloop has no
implementer skill): the apply step (if `spec.patch`) + a
`/experiment:run` call (always).

## Contract

**Inputs** (read at start of skill):
- `outputs/design_runs/myloop/{graph}/v{N}/goal.md`                    (read-only)
- `outputs/design_runs/myloop/{graph}/v{N}/constraints.md`             (read-only; OPTIONAL — hard rules, skip if absent)
- `outputs/design_runs/myloop/{graph}/v{N}/knowledge.md`               (append-only)
- `outputs/design_runs/myloop/{graph}/v{N}/search_space.md`            (read-only here; REFLECT writes it — read the latest `## reflection_N` for the current Frontier + the `## Axes` list)
- `outputs/design_runs/myloop/{graph}/v{N}/experience.jsonl`           (read-only; DISTILL writes this)
- `outputs/design_runs/myloop/{graph}/v{N}/hypotheses.jsonl`           (mutable: append; DISTILL also mutates)
- `outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_*/record.json` (read-only; what each prior iter tested and learned — already DISTILL'd)
- `outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_*/critique_*.json` (read-only; what CRITIC predicted on prior iters, per spec — DISTILL judges accuracy)
- `outputs/design_runs/myloop/{graph}/v{N}/.staging/iter_{n}/critique_<spec_id>_round_1.json` (read-only; ONLY on `--respond-to-critique --spec-id <X>` rebuttal pass — the in-iter round-1 critique for that one spec)
- `outputs/design_runs/myloop/{graph}/v{N}/experiment_design.yaml`     (mutable: append profiles)
- `outputs/design_runs/myloop/{graph}/v{N}/tools/*.py`                 (mutable: add files)
- `workspace/graphs/{graph}.json` + `workspace/nodesets/`              (read-only; current graph state)
- `outputs/eval_runs/*/`                                               (read-only; selectively — ONLY for episode-subset construction, see Step 3)

THINK MAY read raw eval artifacts (`outputs/eval_runs/*/`) — but for
ONE purpose: constructing a targeted episode subset for the next
experiment (Step 3, "Experiment design"). For *lessons* — what prior
iters concluded — THINK reads the distilled `hypotheses.jsonl` /
`experience.jsonl` / `knowledge.md`, NOT the raw logs; re-deriving
DISTILL's conclusions from logs is wasted work. Raw logs are MB-scale:
grep / sample / use a `tools/` filter, never cat them whole.

**Mandatory output** (one file):
- `.staging/iter_{n}/spec.json` — a complete `ExperimentSpec` envelope `{iter, specs:[...]}` (schema in `schemas.md` § 8). THINK MAY emit `K ≥ 1` specs in one turn (`max_specs_per_iter` in config.yaml caps K; default 3). Each spec is an independent experiment with its own patch / eval / risk_vectors / passes; CRITIC + EXPERIMENT + DISTILL all operate per-spec downstream.

**Optional side-effects** (all under vN dir, eager-write during the think turn):
- `hypotheses.jsonl`           — append / delete-line
- `knowledge.md`               — append bullets (NEVER overwrite existing)
- `experiment_design.yaml`     — append new profile (e.g. a probe entry)
- `tools/*.py`                 — add a new tool file
- `.staging/iter_{n}/milestone.md` — current sub-goal text (orchestrator's near-term plan)

**SKIP semantics**: if THINK cannot produce a useful envelope (i.e.
even one viable spec is out of reach), return `status=SKIP_THINK_EMPTY`
instead of writing spec.json. This means only "no experiment in the
**current frontier**" — it does NOT terminate the run. The loop escalates to REFLECT
(`reflect.md`); REFLECT re-surveys the whole intervention space and
either hands back a new frontier (THINK is re-run once) or returns
`SPACE_EXHAUSTED` (only then does the run end). So THINK should
return `SKIP_THINK_EMPTY` when *the axes it can see* are dry — and
NOT pre-emptively declare global saturation; that judgement is
REFLECT's. Manufacturing a weak experiment to avoid the SKIP is
equally wrong: an honest SKIP that triggers REFLECT is the designed
path off a local optimum.

## Arguments

```
/architect:myloop:proposer [<graph> [<version> [<iter>]]]
                           [--graph <name>] [--version <N>] [--iter <M>]
                           [--respond-to-critique --spec-id <spec_id>]   # rebuttal pass after CRITIC REVISE/BLOCK on one specific spec
```

Iter resolution: loop passes the next iter index (`iter_n`); manual
invocation defaults to `max(committed iters) + 1`.

`--respond-to-critique --spec-id <spec_id>` is set by the loop when
CRITIC returned `REVISE` or `BLOCK` on round 1 for **one specific
spec** in the envelope. THINK is re-spawned within the same iter
with the **other specs left untouched**; it must read
`.staging/iter_{n}/critique_<spec_id>_round_1.json` (the loop
preserves the round-1 critique at that path before calling
proposer) and either:
(a) rewrite ONLY that spec's entry in the envelope to address the
    concerns (the rewritten spec will be vetted by CRITIC round 2
    in isolation), OR
(b) keep the spec largely intact but add a `block_override` field
    on that spec documenting why the critique is wrong for this
    case (the override is recorded; the spec proceeds to EXPERIMENT
    regardless of CRITIC round 2's verdict).

THINK MUST NOT touch sibling specs in the envelope during a rebuttal
pass — they have their own CRITIC verdicts (possibly OK, possibly
pending their own round 2). Without the flag, any prior
`.staging/iter_{n}/spec.json` is overwritten cleanly with a fresh
envelope.

## Pre-conditions

- `goal.md` exists (hard contract; loop enforces, but this skill
  re-checks and errs `MISSING_GOAL` if absent).
- `knowledge.md` exists (loop bootstraps it from seed if vN is
  fresh — see `loop.md § Bootstrap`).
- `.staging/iter_{n}/` does NOT exist yet (or is empty). This skill
  creates it.

## Steps

### 1. Resolve + announce

```
[myloop:proposer] iter=iter_{n}  graph={graph}  v{N}
                  goal.md       = <ok>
                  knowledge.md  = {bytes} bytes / {N_sections} sections
                  experience    = {K} entries
                  hypotheses    = {H} open
                  prior iters   = {M} committed
```

### 2. Read prior iter's record (if any)

If `iteration/iter_{n-1}/record.json` exists, read it for **context
only** — to see what spec the previous iter tested and what its
outcome was. Do NOT re-derive *lessons* from raw eval logs; that work
was already done by iter_{n-1}'s DISTILL phase
(`/architect:myloop:distill`) and the conclusions live in
`hypotheses.jsonl` / `experience.jsonl` / `knowledge.md`. (Reading
raw logs to *select episodes* for the next experiment is a different
task — that is allowed; see Step 3.)

If no prior record (first iter ever), skip this step.

### 3. THINK (free-form reasoning by a sub-agent, bounded by spec contract)

Invoke a single sub-agent — the orchestrator's reasoning lives here.
This is NOT a 3-call Reflexion chain; it is one tool-augmented
reasoning pass with a strict output contract.

```python
resp = Agent({
  "subagent_type": "general-purpose",
  "description":   f"myloop THINK iter_{n} on {graph}",
  "prompt": render_think_prompt(
      graph=graph, vN=N, iter_n=n,
      goal_md           = read("goal.md"),
      constraints_md    = read_if_exists("constraints.md"),   # None if absent
      knowledge_md      = read("knowledge.md"),
      search_space_md   = read("search_space.md"),            # § Axes + latest reflection's Frontier
      experience_jsonl  = read("experience.jsonl"),
      hypotheses_jsonl  = read("hypotheses.jsonl"),
      experiment_design = read("experiment_design.yaml"),
      tool_index        = ls_with_docstrings("tools/"),
      prior_iters_brief = render_recent_iters(last=3),
      schema_doc        = read(".claude/commands/architect/myloop/schemas.md"),
  ),
})
```

`render_think_prompt` composes:

```
You are the myloop orchestrator. Your job is knowledge distillation
toward the goal — NOT direct fitness optimization. You operate on a
hard contract: this think turn MUST end by producing a single JSON
envelope of the form `{"iter": N, "specs": [...]}` containing K ≥ 1
ExperimentSpec entries — or signaling SKIP_THINK_EMPTY if there is
genuinely nothing to test.

=== WHAT GOES IN THE ENVELOPE ===

The envelope's `specs` list is how you say "this iter's investigation
breaks down into these pieces". K = how many genuinely independent
pieces the iter's question already has, NOT a knob you tune separately
from the reasoning.

If your reasoning naturally converges on one focused question — write
one spec. If, while reasoning, you find you're actually holding two
distinct conjectures whose outcomes don't depend on each other — write
two. If you want to measure a candidate's effect AND its noise floor on
the same ep set in the same iter — that's two specs. The envelope
shape just reflects how your investigation is actually structured.

Don't force K=1 out of conservatism — serializing independent questions
across iters wastes wall-clock and obscures cross-spec interactions
that DISTILL can otherwise catch in one pass. Don't force K>1 out of
performative breadth either — empty specs cost CRITIC + eval budget
and dilute signal.

Cost shape: the Python multi-spec wrapper submits every spec×pass
combination in ONE parallel wave (one admission window, JobScheduler
arbitrates VRAM), so K>1 with independent overlays is NOT 2× wall-clock
— it's 2× eval API calls and 2× CRITIC spawns. Be honest about whether
additional specs carry independent information; don't be artificially
expensive.

Cap: K ≤ {max_specs_per_iter} (config). Each spec needs a unique
`spec_id` of the form `spec_iter_{n}_{LETTER}` (LETTER = A, B, C, ...
in order). Each spec is an INDEPENDENT experiment with its own patch /
eval_profile / risk_vectors / passes — CRITIC vets each separately,
EXPERIMENT applies each on its own `active_workspace_<id>/` overlay,
DISTILL produces a per-spec verdict. There is no shared overlay across
specs; patches do not stack within a single iter.

Each spec carries `passes: int` (≥ 1). passes > 1 makes the Python
eval wrapper run that spec's eval `passes` times against the same
ep set, producing mean_sr / sd_sr / robust_sr (per-ep majority-vote
SR).

**`passes_required` floor — non-negotiable**: every profile in
experiment_design.yaml carries a `passes_required` field, derived
from `episode_count`:
  - episode_count < 119  → passes_required = 3
  - episode_count ≥ 119  → passes_required = 1
  - smoke profiles       → passes_required = 0 (smoke is a
                           correctness gate, never a measurement)

If `profile.baseline` is locked (i.e. some prior iter has run this
profile under multi-pass policy), `baseline.passes` is also a floor.

Your spec's `passes` MUST be ≥ both floors. If you set it lower,
proposer's validator auto-promotes you to the required value
(non-fatal warning logged). You MAY set passes HIGHER than the
floor — e.g. passes=5 on a small subset where you really want
tight CI — but expensive. Default to the profile's
`passes_required`.

The Python wrapper submits every (spec, pass_idx) combination in
ONE parallel wave (the iter's "合集 / combined experiment"); the
JobScheduler arbitrates admission by VRAM. Worker count per
submission is auto-allocated so the total across the wave stays
within `perf_<graph>.worker_count`. This is opaque to you — just
set `passes` and trust the wrapper.

You may freely use Read / Grep / Bash / Edit / Write during this
turn to:
  - append bullets to knowledge.md (pure facts you know a priori
    that aren't yet captured — most knowledge accrual happens in
    DISTILL, but you may add bullets here too)
  - append lines to hypotheses.jsonl (new conjectures worth testing)
  - register a new eval profile in experiment_design.yaml
  - author a new tool in tools/<name>.py

=== Experiment design — choosing this iter's episode set ===
The smoke / perf entries in experiment_design.yaml are just a starting
menu. The episode set you test is yours to design:

  - smoke_<graph> (≈3 ep) is the EXPERIMENT apply-step's correctness
    gate ("does the change run"). It is NEVER this spec's eval_profile
    — 3 ep cannot measure anything.
  - perf_<graph> (full, paper-comparable) is the escalation tier —
    invoke only when goal.md § Escalation / constraints authorize it.
  - A THINK-composed targeted subset (kind=custom) is the DEFAULT
    measurement tier. If a hypothesis is about one failure mode, test
    exactly the episodes that exhibit it. To build one:
      1. Read the relevant prior run's raw logs — usually the
         iter_0 baseline at outputs/eval_runs/<baseline_run_id>/.
         grep / sample / write a tools/*.py filter; do NOT cat
         MB-scale log.jsonl files whole.
      2. Collect the episode indices that exhibit the failure mode.
      3. Append a profile to experiment_design.yaml with explicit
         episode_indices + a derived_from block recording the
         failure mode and source run (schemas.md § 6).
      4. Point spec.eval_profile.name at it. The profile persists —
         later iters testing the same failure mode just reuse it.

Reading raw logs IS allowed, for episode selection. For *lessons*
(what prior iters concluded) read the distilled hypotheses.jsonl /
experience.jsonl / knowledge.md instead — don't re-derive them.

You MUST NOT modify workspace/* — patches go through
ExperimentSpec.patch; the EXPERIMENT apply-step (loop.md § 3c)
realizes them and enforces the edit whitelist.

=== Search space — which intervention axis (READ THIS) ===
search_space.md maps the INTERVENTION SPACE this run searches. Its
`## Axes` section is the taxonomy (prompt-content / topology /
control-flow / action-space / observation-pipeline / state-memory /
model-component-config — and any REFLECT has added). Its latest
`## reflection_N` section carries a per-axis Coverage table and a
ranked **Frontier** — the axes the REFLECT phase judged still worth
searching, with a mechanism sketch for each.

EACH SPEC in the envelope MUST carry an `intervention_axis` field
naming exactly one axis from `## Axes`. The set of axes across your
K specs is what determines whether THIS ITER counts as concentrated
or jumped — see the axis-jump rule below.

THE AXIS-JUMP RULE. myloop's known failure is THINK refining one
axis to death (e.g. five prompt-content variants, all refuted) and
never trying a structurally different intervention. So:
  - If a Frontier exists in the latest reflection, PREFER an axis on
    it — the Frontier is REFLECT's considered judgement of where the
    unsearched headroom is.
  - If the last few committed iters all share one `intervention_axis`
    (in the Recent-iters block, look at the union of axes per iter),
    the DEFAULT is to ensure THIS iter touches at least one DIFFERENT
    axis on at least one of its specs. Staying entirely on the
    concentrated axis (i.e. every spec in this iter's envelope uses
    the same concentrated axis) is allowed ONLY if you write an
    explicit rebuttal in your reasoning trace — a concrete argument
    that an untested sub-lever on that axis still has a mechanism-
    grounded path the Frontier missed.
  - Multi-spec breadth IS an axis-jump tool: if you genuinely want
    to continue refining one axis but feel the rule's pressure,
    emit K=2 with one spec on the refined axis + one spec on a
    Frontier axis. Both get tested; the iter is no longer "concentrated".
  - Picking a fresh axis does not mean picking a bad experiment: the
    intervention still needs a real mechanism and must fit budget +
    constraints. If no viable experiment exists in any axis, return
    SKIP_THINK_EMPTY (the loop escalates to REFLECT; that is correct,
    not a failure).

=== HARD CONSTRAINTS (non-negotiable) ===
{constraints_block_or_"(none — no constraints.md file)"}

These rules are absolute. Your spec.patch (if any) and any
hypothesis / knowledge you append MUST respect them. If you cannot
propose a useful experiment under these constraints, return
SKIP_THINK_EMPTY rather than violating.

=== RISK VECTORS — REQUIRED structural declaration (per spec, when patch is non-null) ===

For EACH spec in your envelope where `patch` is non-null, you MUST
emit a `risk_vectors` block (schemas.md § 8). This is not optional
and not optional-via-empty: the act of enumerating risk vectors per
spec is itself the cross-check that catches the structural pathologies
past iters have recurred on. Specifically:

  - state_io.{reads, writes, grants_required}: list every
    `gs.read(...)` / `gs.write(...)` your proposed code touches; for
    every new node whose code touches graph_state, include an
    `access_grants` entry in your graph JSON edit AND list its id
    in `grants_required`. If you forget the access_grants entry,
    your mechanism will silently no-op (cf. iter_3, iter_7
    pathology — see experience.jsonl).
  - llm_calls: one entry per llm_complete / vlm_complete / llmCall
    introduced or modified. For reasoning-aware models (gpt-5-*,
    o1-*, o3-*), max_tokens must be ≥ 2000 (reasoning_tokens
    consume the budget before visible content). Set
    `reasoning_aware: true` so CRITIC verifies the budget.
  - fallback_branches: trace each error/empty/parse-error path. If
    a fallback produces an output indistinguishable from a real
    decision, the fallback IS the mechanism (cf. iter_2 fail-OPEN
    rubber-stamp, iter_7 fail-CLOSED reroute) — restructure so
    fallbacks are `fallback_is_inert: true` (preserve baseline /
    return BYPASS), not load-bearing.
  - globally_firing_nodes_touched: list any node whose `execute`
    fires on every step of every ep (build_options, observe,
    render_prompt, parse_action, iter_in, iter_out, step). If
    non-empty, your `expected_signal` SHOULD include a counter-check
    on the (S)-bucket (baseline-success eps) — patches at these
    nodes can churn the (S) bucket while lifting (c)/(d), and that
    cost is invisible at subset-eval scale (cf. iter_12).
  - mechanism_fire_predicate: state how you will OBSERVE the
    mechanism firing. The proposed code MUST `_self_log(...)` a
    counter or per-step value DISTILL can grep, independent of
    aggregate metrics (cf. iter_3 silent no-op caught only by lack
    of a fire log; iter_9 gate-too-strict caught only by 0/27 fire
    count).

CRITIC will validate each spec's `risk_vectors` independently in the
next phase. The validator re-runs the same checks structurally;
CRITIC pattern-matches each spec against past refuted experiences.
Fill it honestly per spec — under-declaring a risk vector is the
kind of mistake that gets THAT spec blocked at round 2 (sibling
specs in the envelope still proceed).

=== RECENT PRIOR CRITIQUES (context — DISTILL judges their accuracy, not you) ===
{recent_critiques_or_"(no prior critiques yet)"}

Past CRITIC predictions live here for context. The fact that
CRITIC flagged X on iter_K does not mean iter_K's outcome confirmed
the prediction — DISTILL evaluates that against the actual eval.
Read these to see what kinds of pathologies CRITIC has been
catching; do not "respond" to each one — that is DISTILL's job.

{if --respond-to-critique:}
=== IN-ITER CRITIQUE — REWRITE OR REBUT (ONE SPEC ONLY) ===

CRITIC fired on spec `{rebuttal_spec_id}` in this iter's envelope
and returned:
  verdict: {round_1_verdict}
  predicted_failure_modes:
{round_1_critique}

You are rewriting THIS ONE SPEC (`{rebuttal_spec_id}`) only. The
other specs in the envelope have their own CRITIC verdicts —
DO NOT touch them. Your job is to emit a new envelope where:
  - the entry for spec_id=`{rebuttal_spec_id}` is rewritten (or
    annotated with `block_override`),
  - every other spec is COPIED VERBATIM from the prior envelope.

You have two options for the targeted spec:

  (a) ACCEPT the critique. Rewrite that spec's entry to fix the
      flagged issue. The rewritten spec will be vetted by CRITIC
      round 2 in isolation; if round 2 also flags issues, that
      spec is dropped (commits as `outcome_class="critic_block"`),
      but sibling specs still proceed.

  (b) REBUT the critique. Add a `block_override` field to that
      spec:
        "block_override": {
          "critique_id": "<copy from above>",
          "rebuttal":    "<argument — name the SPECIFIC structural
                          difference between this spec and the
                          reference experience CRITIC matched
                          against>"
        }
      The spec proceeds to EXPERIMENT regardless. DISTILL then
      judges whether your rebuttal was right (predicted_outcome
      didn't materialize) or wrong (it did) and records the
      verdict.

Pick (a) unless you have a concrete argument why CRITIC is matching
on surface similarity. "I think this is different" is not a rebuttal —
"the access_grant IS added at intent §A.4 line ... so the structural
feature CRITIC matched does not hold" is.

=== Heuristic — when iter_n == 0 (no prior committed iter) ===
You have no per-ep evidence yet. iter_0 MUST be a no-patch baseline
run (spec.patch = null) on a real measurement tier — perf_<graph>
when the goal's success criterion is defined there, otherwise a
wide-enough custom subset (≥ ~30 ep). NOT a 3-ep probe: iter_0's
per-episode logs are the corpus every later iter mines to build
failure-mode subsets (see "Experiment design" above), so the
baseline must be wide enough to expose those failure modes. A
patch-applying iter_0 has no comparison point — any apparent
"lift" or "regression" is noise. Use iter_0 to establish ground
truth + a rich log corpus; let iter_1 onward do the patches.

=== goal.md ===
{goal_md}

=== knowledge.md ===
{knowledge_md}

=== search_space.md (§ Axes + all reflections; the LATEST reflection's Frontier is your guide) ===
{search_space_md}

=== experience.jsonl (last 20 entries) ===
{experience_tail}

=== hypotheses.jsonl (all open) ===
{hypotheses_full}

=== experiment_design.yaml ===
{experiment_design}

=== tools/ index ===
{tool_index}

=== Recent iters (last 3) ===
{prior_iters_brief}

=== Schemas (single source of truth) ===
{schema_doc_excerpt}

Now reason. When done, emit a SINGLE fenced ```json block as your
final visible chunk. The JSON must conform to the ExperimentSpec
envelope shape (schemas.md § 8):

```json
{
  "iter": {n},
  "specs": [
    {
      "spec_id": "spec_iter_{n}_A",
      "kind": "...",
      "intervention_axis": "...",
      "passes": 1,
      ...all other ExperimentSpec fields per schemas.md § 8...
    }
    // additional specs if K>1; each MUST have unique spec_id, intervention_axis,
    // and (if patch != null) risk_vectors
  ]
}
```

If the axes you can see are dry across the whole envelope, instead
emit `{"status": "SKIP_THINK_EMPTY", "reason": "..."}` — the loop
will escalate to REFLECT, which is the designed way out, not a
failure. Do NOT emit an envelope with `specs: []` — empty is treated
as malformed.
```

### 4. Parse + validate ExperimentSpec envelope

After the sub-agent returns:

- Extract the LAST fenced ```json block via `parse_final_json`.
- If `{"status": "SKIP_THINK_EMPTY", ...}`: do not write spec.json;
  return `SKIP_THINK_EMPTY` to loop (loop translates to termination
  candidate).
- **Envelope-level validation**:
  - Top-level keys: `iter` (int, must equal the loop's `n`), `specs`
    (non-empty list).
  - `len(specs) ≤ max_specs_per_iter` (from config.yaml; default 3).
    Over-cap → `SKIP_INVALID_SPEC` with `reason="K > max_specs_per_iter"`.
  - All `spec_id` values are unique within `specs[]` and follow the
    `spec_iter_{n}_{LETTER}` pattern (regex
    `spec_iter_\d+_[A-Z]+`). Duplicates / malformed → `SKIP_INVALID_SPEC`.
  - On `--respond-to-critique --spec-id <S>` rebuttal pass: the new
    envelope must contain a spec with `spec_id=S` AND every other
    spec from the prior `spec.json` must appear UNCHANGED (loop
    diff-checks them). If a sibling spec is altered → `SKIP_INVALID_SPEC`
    with `reason="rebuttal touched sibling spec"`.
- **For each entry in `specs[]`** (apply the per-spec rules below;
  ALL specs must pass or the envelope is rejected):
  - Required fields present (per `schemas.md § 8`).
  - `kind` ∈ {probe, perf, custom}.
  - `passes` is an integer ≥ 1.
  - **`passes ≥ profile.passes_required`** (read from
    `experiment_design.yaml` for the referenced `eval_profile.name`).
    If `spec.passes < profile.passes_required`: **auto-promote** in
    the validated envelope (set `spec.passes := profile.passes_required`)
    + emit a WARNING. Auto-promotion is a recoverable correction,
    not a SKIP — the spec proceeds at the required pass count. The
    rationale: `passes_required` derives from `episode_count` and is
    not negotiable noise-floor protection; THINK is allowed to ask
    for MORE passes than required (multi-pass on an N≥119 profile is
    legitimate) but not fewer.
  - **`passes ≥ profile.baseline.passes`** if `profile.baseline` is
    locked AND `passes < baseline.passes`: same auto-promote rule.
    (Once a baseline locks at 3-pass, future specs on the profile
    must run at least 3 passes for the comparison to be valid.)
  - `intervention_axis` is present and names an axis listed in
    `search_space.md § Axes` (or is `"none"` for a pure
    data-collection no-patch probe). Unknown axis → `SKIP_INVALID_SPEC`.
  - `eval_profile.name` exists as a key in `experiment_design.yaml`
    (re-read the file in case THINK appended one this turn).
  - If `patch` is non-null, it is `{intent, targets}` (per
    `schemas.md § 8`): `intent` is non-empty prose and every entry of
    `targets` is under `workspace/` — `workspace/graphs/<name>.json`,
    `workspace/nodesets/<X>.py`, or
    `workspace/nodesets/server/<X>.py` / `workspace/nodesets/server/<X>/...`
    (TODO #60: server-mode is editable; the whitelist is enforced
    ALSO by the EXPERIMENT apply-step, we early-check here to fail
    fast). Out-of-workspace paths (`agentcanvas/backend/app/**`,
    `third_party/**`) → fail fast.
  - **`risk_vectors` REQUIRED when patch is non-null** (per
    `schemas.md § 8`). All five sub-blocks must be present and
    typed correctly:
    - `state_io.{reads, writes, grants_required}` are lists of
      strings (may be empty lists if the patch makes no graph_state
      access).
    - `llm_calls` is a list of objects; each entry must carry
      `purpose` (str), `model_profile` (str), `max_tokens` (int),
      `reasoning_aware` (bool), `fallback_on_empty` (str),
      `fallback_on_parse_error` (str), `fallback_is_inert` (bool).
      May be empty list if patch introduces no new LLM call.
    - `globally_firing_nodes_touched` is a list of strings (may be
      empty).
    - `mechanism_fire_predicate` is a non-empty string.
    A missing `risk_vectors` block, or any required sub-field
    missing, → `SKIP_INVALID_SPEC` with the specific spec_id +
    field gap named.
  - **Programmatic lint pass** (deterministic, no LLM):
    - For every `node_id` listed in `risk_vectors.state_io.grants_required`,
      verify the patch.intent text contains the substring `access_grants` AND
      the substring `<that node_id>` (case-sensitive). This catches
      the iter_3/iter_7 access_grant-missing pathology at THINK-time
      without waiting for CRITIC.
    - For every entry in `risk_vectors.llm_calls` with
      `reasoning_aware: true` AND `max_tokens < 2000`: report a
      WARNING (do NOT fail the spec) — CRITIC will likely BLOCK this
      anyway. THINK is allowed to proceed if it knows what it's
      doing (e.g. running an explicit ablation).
    - Failures here → `SKIP_INVALID_SPEC` with the specific spec_id
      + gap; the envelope is not written and loop increments
      `consecutive_skips`. (One bad spec rejects the whole envelope —
      THINK should re-emit all specs together.)
  - **`block_override` schema** (only when present, see `schemas.md § 8`):
    object with non-empty string fields `critique_id` and `rebuttal`.
    Loop verifies `critique_id` matches the `critique_id` of the
    round-1 critique on disk for THIS spec_id; mismatch →
    `SKIP_INVALID_SPEC`.
  - **Constraint guard** (only if `constraints.md` exists and `patch`
    is non-null): coarse string-match — scan `patch.intent` for any
    `node_id.config.<key>` fragment a `constraints.md` bullet forbids
    (a bullet containing BOTH `MUST` / `MUST NOT` AND that literal
    `node_id.config.<key>` or `node_id.*.<key>` substring). If found →
    flag a candidate violation and read that bullet to the user as the
    reason. This catches the common cases (model profile / temperature
    / max_tokens forbids) without parsing markdown semantically.
- On validation failure: signal `SKIP_INVALID_SPEC` with the specific
  error — loop logs, increments consecutive_skips, continues.

### 5. Write `.staging/iter_{n}/spec.json` (envelope)

```bash
mkdir -p outputs/design_runs/myloop/{graph}/v{N}/.staging/iter_{n}/

# Rebuttal pass: preserve the round-1 envelope for forensics before overwriting.
# The whole envelope is preserved (not just the rebutted spec) — easier to
# diff the before/after, and the sibling specs being identical is part of
# the validated invariant.
if [ "$RESPOND_TO_CRITIQUE" = "true" ] && [ -f .staging/iter_{n}/spec.json ]; then
    cp .staging/iter_{n}/spec.json .staging/iter_{n}/spec_round_1.json
fi

echo "$VALIDATED_SPEC_ENVELOPE_JSON" > outputs/design_runs/myloop/{graph}/v{N}/.staging/iter_{n}/spec.json
```

Also stage a copy of the THINK sub-agent's full text return at
`.staging/iter_{n}/think_trace.md` for forensics — on the rebuttal
pass, append to (don't overwrite) the existing trace, so both
rounds' reasoning is preserved.

### 6. Return to loop

```
[myloop:proposer] OK
                  spec.json   = .staging/iter_{n}/spec.json   (envelope, K={len(specs)})
                  specs       = [{spec_id} (axis={ax}, kind={k}, patch={p}, passes={pp}), ...]
                  side-effects= {N file writes during think}
                  → loop will run CRITIC then EXPERIMENT next
```

Return `status=OK` to loop.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/.staging/iter_{n}/spec.json` | this skill | ExperimentSpec envelope (schema § 8) — `{iter, specs:[...]}`; the loop's CRITIC + EXPERIMENT phases consume it |
| `v{N}/.staging/iter_{n}/spec_round_1.json` | this skill (rebuttal only) | Round-1 envelope preserved before rebuttal-pass overwrite |
| `v{N}/.staging/iter_{n}/think_trace.md` | this skill | Full sub-agent return text — forensics; appended (not overwritten) on rebuttal pass |
| `v{N}/.staging/iter_{n}/milestone.md` | this skill (optional) | Current sub-goal narrative |
| `v{N}/hypotheses.jsonl` | this skill (eager during think) | New / resolved entries |
| `v{N}/knowledge.md` | this skill (eager during think) | New bullets appended |
| `v{N}/experiment_design.yaml` | this skill (eager during think) | New probe profile (if defined) |
| `v{N}/tools/*.py` | this skill (eager during think) | New tool (if authored) |
_Note: `experience.jsonl` is **not** written by this skill — that's DISTILL's domain (`/architect:myloop:distill`). THINK only reads it._

## Notes

- **No 3-call Reflexion.** myloop deliberately drops the ADAS R0/R1/R2
  scaffold. The forcing function is now "must produce ExperimentSpec",
  which gives THINK enough structure. If we find single-pass thinking
  is too shallow in practice, we can wrap this skill with a critique
  pass — but not by default.
- **THINK is one sub-agent spawn.** Independent sample, full tool
  access. The orchestrator-as-agent pattern in literal form.
- **Eager writes vs staged writes**: hypothesis / knowledge / tool /
  design changes happen during the think turn directly to vN files
  (not staged), because if the think turn produces a probe targeting
  a brand-new hypothesis, that hypothesis MUST be visible to
  EXPERIMENT and DISTILL. The atomic boundary in myloop is the iter
  dir promotion, not these document writes.
  - Recovery: if THINK crashes mid-write, vN files may be partially
    updated. Loop's `.loop_state/` carries the last-known-good iter
    index; on resume, the orchestrator reads files as-is and
    continues — partial writes are recoverable by reasoning, not
    rollback.
- **edit whitelist for patches**: enforced by the EXPERIMENT
  apply-step (`loop.md § 3c`, via `_common/lib/overlay.py`) per
  `_common/files-contract.md`. This skill only fast-fails ill-formed
  targets; final authority is the apply-step.
- **per-episode trace as the ground for episode selection**:
  `outputs/eval_runs/{run_id}/episodes/ep0003/log.jsonl` records every
  node firing, llmCall, and observation for that episode. THINK reads
  these — selectively — to construct targeted episode subsets (Step 3,
  "Experiment design"). It does NOT read them to re-derive lessons:
  DISTILL has already promoted that signal into
  hypotheses/experience/knowledge by the time the next iter's THINK
  starts.
