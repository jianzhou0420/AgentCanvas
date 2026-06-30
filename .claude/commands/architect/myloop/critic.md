# CRITIC phase — pre-EXPERIMENT vetting against accumulated experience

> **Required reading before invoking**:
> - `myloop/README.md` — the myloop mental model (CRITIC is a phase between THINK and EXPERIMENT, one fire per patched spec)
> - `myloop/schemas.md` — esp. `ExperimentSpec` envelope (§ 8), `risk_vectors` sub-block,
>   per-spec `critique_<spec_id>.json` (§ 15), and the per-spec `critic` block of `IterRecord.specs[*]` (§ 5)
> - `_common/files-contract.md` § "Edit whitelist" — what `risk_vectors.state_io.grants_required` is supposed to cover

This skill IS the **CRITIC phase** of one myloop iter, invoked
**once per spec** in the iter's envelope. It runs after THINK
(`proposer.md`) and before EXPERIMENT (`loop.md § 3c`) — a new
phase in the per-iter rhythm, inserted because single-pass THINK
under spec-production pressure does not reliably cross-check its
own design against `experience.jsonl`. The iter_3 → iter_7 →
iter_9 recurrence of the same `access_grant`-missing pathology is
the canonical motivating case: each iter had `experience.jsonl`
(and later `knowledge.md`) entries describing the pathology in its
prompt, yet THINK still produced a spec that triggered it.

CRITIC reads ONE spec (identified by `spec_id`) from the iter's
envelope, the run's `experience.jsonl` (especially refuted entries),
and the live graph, and predicts which past pathology this spec
is most likely to recur. The cost is one (or at most two) extra
sub-agent spawns **per spec**; the saving is a full EXPERIMENT
(one full eval, ~15–25 min, ~\$0.5) per blocked recurrence. K-spec
iters incur K× CRITIC spawns — sibling specs have independent
verdicts (one BLOCK does not stop another spec's eval).

CRITIC carries no "produce a spec" pressure. Like REFLECT, it is a
**separate sub-agent spawn** with a single cognitive task — predict
failure modes — and the same independent-sampling argument applies:
an agent asked to self-audit *while* under pressure to ship a spec
rationalises rather than names its risks; a fresh sub-agent does not
have that pressure.

## When the loop invokes CRITIC

The loop (`loop.md § 3b.5`) invokes CRITIC after THINK (`3b`) and
before EXPERIMENT (`3c`), once per spec in the envelope where
`patch != null`. No-patch probes / baselines have nothing to
pattern-match against past pathologies — CRITIC is skipped for
those specs and that spec's `critic` block is omitted from the
record entirely.

For each patched spec, CRITIC may fire **up to twice** (round 1 +
at most one re-spawn if THINK rewrote that spec in response to a
`REVISE` / `BLOCK` verdict). The second-round verdict on a spec is
final regardless of what it says — the loop does not third-guess.

Per-spec independence: in a K-spec iter, spec A may receive
verdict OK on round 1 while spec B receives REVISE and goes to
round 2. The loop dispatches them independently; sibling outcomes
do not affect each other.

## Contract

**Inputs** (read at start of skill — bound to the `--spec-id` argument):

- This iter's transient envelope (under `.staging/iter_{n}/`):
  - `spec.json` — the full envelope (REQUIRED). CRITIC extracts the
    entry where `spec_id == <--spec-id>`. Round 1 consumes the
    THINK output; round 2 consumes the THINK rewrite of that one
    spec.
  - `critique_<spec_id>_round_1.json` — the round-1 critique for
    THIS spec (only present if this is round 2; the loop preserves
    it at this path before round 2 fires; CRITIC reads it for
    self-consistency)
- Persistent working memory:
  - `outputs/design_runs/myloop/{graph}/v{N}/goal.md`              (read-only)
  - `outputs/design_runs/myloop/{graph}/v{N}/constraints.md`       (read-only; OPTIONAL)
  - `outputs/design_runs/myloop/{graph}/v{N}/experience.jsonl`     (read-only — refuted entries are the pattern library)
  - `outputs/design_runs/myloop/{graph}/v{N}/knowledge.md`         (read-only — facts grounding the predictions)
  - `outputs/design_runs/myloop/{graph}/v{N}/search_space.md`      (read-only — axis context)
  - `outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_*/record.json`
    (read-only — needed to read prior `critique_*.json` entries when
    a spec has a `block_override` referencing a past critique)
- Live graph / nodesets:
  - `.staging/iter_{n}/active_workspace_<spec_id>/`                (read-only — the per-spec overlay THINK proposes; not yet applied)
  - `workspace/graphs/{graph}.json` + `workspace/nodesets/`        (read-only — the frozen reference state, for diff reasoning)

CRITIC reads the **distilled** state for pattern matching, not raw
eval logs. (If a specific check hinges on one trajectory detail, a
targeted grep is allowed — but rare; raw-log mining is DISTILL's
job.)

**Mandatory output** (one file, single visible chunk from the
sub-agent must be a fenced ```json block conforming to the
`critique_<spec_id>.json` schema):

- `.staging/iter_{n}/critique_<spec_id>.json` — `Critique` per
  `schemas.md § 15`. The output's `spec_id` field MUST equal the
  `--spec-id` argument.

**Staged forensic** (promoted into the iter dir on commit):

- `.staging/iter_{n}/critique_trace_<spec_id>.md` — the CRITIC
  sub-agent's full text return for this spec.

**Round semantics**:

- **Round 1** is always the first CRITIC fire on an iter's first
  spec. `critic_round` in the output = `1`.
- **Round 2** fires only if round 1 returned `REVISE` or `BLOCK`
  *and* THINK produced a rewritten spec in response. CRITIC re-spawns
  on the new spec; `critic_round = 2`. Round 2's verdict is binding
  regardless (no round 3).

**Return status** (to loop):

- `OK` — proceed to EXPERIMENT, no concerns above noise floor.
- `WARN` — proceed to EXPERIMENT; concerns logged but no specific
  past pathology matched at BLOCK-threshold strength.
- `REVISE` — concerns are concrete enough that THINK should be given
  one rewrite chance; loop kicks THINK back. Only valid on round 1.
- `BLOCK` — high-confidence prediction of recurrence of a specific
  past pathology; iter skipped (no EXPERIMENT) unless THINK rebuts
  via `spec.block_override`. The verdict requires all three hard
  conditions listed below; without them, max severity is `REVISE`.
- `SKIP_INVALID_CRITIQUE` — sub-agent returned malformed output; loop
  treats as `WARN` (proceeds to EXPERIMENT) and increments
  `consecutive_skips`.

## BLOCK requires three hard conditions

To prevent CRITIC from suppressing genuine novelty by pattern-matching
on surface similarity, `verdict = "BLOCK"` is only valid when **every
entry in `predicted_failure_modes` carries**:

1. **`reference_experience_ids`** — at least one `exp_id` from
   `experience.jsonl` with `verdict = "refuted"` and a
   tag-or-narrative-level structural match (not "looks similar"; the
   critique must name the shared structural feature).

2. **`specific_check`** — a field-level or line-level concrete
   pointer to the part of `spec.json` (or the proposed
   `active_workspace_<spec_id>/` overlay) that fails the check. Examples:
   - "this spec's patch.targets includes `workspace/graphs/{g}.json`; the
     proposed graph edit (intent §A.4) adds node `<X>` but does not
     add an `access_grants` entry for it; node code (intent §B.2)
     calls `ctx.graph_state.read('history')`."
   - "spec.patch.intent §B specifies `max_tokens=200` on a
     gpt-5-mini VLM call; gpt-5 family is reasoning-aware and 200
     is dominated by reasoning_tokens — visible content will be
     empty (see exp_iter7_001)."

3. **`predicted_outcome`** — a verifiable post-hoc prediction the
   eval will either confirm or refute. Examples:
   - "verifier_continue_count will be 0/N (i.e. the gate
     mechanism-fire metric will be 0)."
   - "verifier _self_log shows VLM response empty rate ≥ 95%."

If any predicted_failure_mode is missing any of these three, CRITIC
must **downgrade** the verdict from `BLOCK` to `REVISE` (or `WARN`
if no `specific_check` either). The sub-agent prompt enforces this;
the loop's validator (Step 4) re-enforces it.

## Arguments

```
/architect:myloop:critic [<graph> [<version> [<iter>]]]
                         [--graph <name>] [--version <N>] [--iter <M>]
                         --spec-id <spec_id>           # REQUIRED — which spec to vet
                         [--round 1|2]                 # default 1
```

`--iter` is the iter the CRITIC is vetting (the loop passes it so
the trace stages into the right `.staging/iter_{n}/`). `--spec-id`
identifies one entry in the iter's `spec.json` envelope; CRITIC
operates on that entry only. `--round` is recorded in
`critique_<spec_id>.json` and lets round 2 read round 1's critique
for self-consistency.

## Pre-conditions

- `.staging/iter_{n}/spec.json` exists and parses as a valid envelope
  per `schemas.md § 8`.
- An entry with `spec_id == <--spec-id>` exists in the envelope's
  `specs[]`.
- That entry has non-null `patch` (else the loop skips CRITIC for
  this spec; this skill is not invoked on no-patch specs).
- That entry has a `risk_vectors` block (REQUIRED per
  `schemas.md § 8` from iter_1+; THINK is contracted to produce it).
  Round 1 may receive a spec without `risk_vectors` if a transitional
  THINK build is in use — in that case CRITIC still runs but flags
  the absence as a `WARN`.

## Steps

### 1. Resolve + announce

```
[myloop:critic] iter=iter_{n}  graph={graph}  v{N}  round={1|2}
                spec_id    = {spec_id}
                envelope   = .staging/iter_{n}/spec.json  (K = {len(specs)})
                patch?     = {true (else this skill not invoked)}
                experience = {K} entries ({K_refuted} refuted)
                prior_iters_with_critic = {M}
```

### 2. Read spec + prior context

Read `.staging/iter_{n}/spec.json` and extract the entry whose
`spec_id` equals `--spec-id`. Bind:

- `intervention_axis`, `target`, `patch.intent`, `patch.targets`,
  `risk_vectors` — the surface CRITIC will validate for this spec.
- If round 2: also read
  `.staging/iter_{n}/critique_<spec_id>_round_1.json` (round 1's
  critique for this spec, preserved by the loop before round 2
  fires) for self-consistency.
- If this spec's `block_override` is present (THINK rebutting a prior
  CRITIC BLOCK on this spec): note it; CRITIC re-evaluates whether
  the rebuttal is sound (i.e. whether the override's claimed
  differences from the reference experience hold up against the
  spec's actual content).
- Sibling specs in the envelope are visible (the file holds them all)
  but CRITIC does NOT vet them — each gets its own CRITIC spawn.
  Sibling specs may be inspected for context (e.g. "is the sibling
  spec testing a confounding mechanism?") but no verdict is rendered
  on them here.

### 3. Read experience.jsonl + filter to refuted

`experience.jsonl` is the pattern library. CRITIC focuses on
`verdict = "refuted"` entries by default — those are the closed-case
pathologies — but may also read `"inconclusive"` and even
`"confirmed"` entries when the spec's `risk_vectors` indicate it is
attempting a structurally similar mechanism to a known confirmed
one (in which case CRITIC may flag a likely-redundant retest).

For each refuted entry, build an internal table of
`{exp_id, tags, structural_features, original_pathology}`. This is
the lookup CRITIC pattern-matches against in Step 4.

### 4. CRITIC sub-agent

Invoke a single sub-agent:

```python
resp = Agent({
  "subagent_type": "general-purpose",
  "description":   f"myloop CRITIC iter_{n} {spec_id} round_{round} on {graph}",
  "prompt": render_critic_prompt(
      graph=graph, vN=N, iter_n=n, round=round, spec_id=spec_id,
      goal_md           = read("goal.md"),
      constraints_md    = read_if_exists("constraints.md"),
      knowledge_md      = read("knowledge.md"),
      experience_jsonl  = read("experience.jsonl"),       # full
      search_space_md   = read("search_space.md"),         # axes + latest reflection only
      envelope_json     = read(".staging/iter_{n}/spec.json"),                     # full envelope (sibling specs visible for context)
      this_spec_json    = extract_spec(envelope, spec_id),                          # the spec being vetted
      prior_critique    = read_if_round_2(f".staging/iter_{n}/critique_{spec_id}_round_1.json"),
      active_workspace_overlay = ls_or_diff(f".staging/iter_{n}/active_workspace_{spec_id}/"),
      schema_doc        = read(".claude/commands/architect/myloop/schemas.md"),
  ),
})
```

`render_critic_prompt` composes:

```
You are the myloop CRITIC. Your job is to predict whether the spec
that THINK just produced will recur a past pathology recorded in
experience.jsonl, BEFORE the EXPERIMENT phase pays for the eval.

You produce one critique. You may use Read / Grep / Bash freely to
ground predictions — for example, to verify whether the proposed
graph JSON adds an access_grants entry that the proposed nodeset
code requires, or whether a referenced helper function exists in
the current nodeset.

You DO NOT modify any file. CRITIC is read-only — your contract is
to return a single fenced ```json block conforming to schemas.md
§ 15 (Critique).

=== HARD CONSTRAINTS ON YOUR VERDICT ===

You may return one of four verdicts:
  - OK       : proceed to EXPERIMENT, no concerns above noise floor.
  - WARN     : proceed, but concerns are logged for next-iter context.
  - REVISE   : concrete enough that THINK should rewrite once.
               Use for: structurally-plausible recurrence with weak
               reference evidence; or schema/risk-vector gaps THINK
               can clearly fix.
  - BLOCK    : same-pathology recurrence is high-confidence. Iter
               WILL be skipped (no EXPERIMENT cost) unless THINK
               rebuts via spec.block_override.

BLOCK requires THREE hard conditions per predicted failure mode:
  1. reference_experience_ids: at least one refuted exp_id with a
     stated structural-feature match (not "looks similar" — name
     the shared structural feature).
  2. specific_check: field-level or line-level pointer to the part
     of THIS spec that fails the check. Be concrete: cite intent
     section / paragraph, file path, line range when possible.
  3. predicted_outcome: a verifiable post-hoc prediction the eval
     will either confirm or refute (e.g. "verifier_continue_count
     will be 0/N").

If any predicted_failure_mode is missing any of these three pieces,
your verdict CANNOT be BLOCK. Downgrade to REVISE (specific_check
present) or WARN (no specific_check).

=== WHAT TO LOOK FOR ===

Cross-check the spec's risk_vectors block against the reality of
the patch.intent text and the active_workspace overlay. THINK was
asked to declare risk vectors as a structural forcing function;
typical mismatches:

  - state_io.reads names "history" but graph JSON edit has no
    access_grants for the new node → silent no-op pathology
    (see exp_iter3_001, exp_iter7_002 for templates)
  - llm_calls.max_tokens is 200 on a reasoning-aware model
    (gpt-5-*, o1-*, o3-*) → VLM-empty response pathology
    (see exp_iter7_*)
  - fallback_branches that produce a non-trivial decision (not
    "preserve baseline") → fallback-becomes-mechanism pathology
    (see exp_iter2_*, exp_iter7_*)
  - globally_firing_nodes_touched is non-empty but expected_signal
    has no (S)-bucket counter-check → cross-bucket churn risk
  - mechanism_fire_predicate references a counter the proposed
    nodeset code does not actually log → silent inert pathology
    (see exp_iter3_001, exp_iter9_*)
  - state_io.writes includes a field that is later regex-parsed by
    the same or another node → state-pollution self-match pathology
    (see exp_iter1_001)

These are the recurring patterns from this run. Match against the
actual experience.jsonl entries — do not invent pathologies.

=== POWER / NOISE-FLOOR CHECKS (pre-flight) ===

In addition to pathology-recurrence prediction, you are responsible
for two **non-pathology** checks that catch experiments doomed to
be statistically inconclusive — both result in `WARN` or `REVISE`,
never `BLOCK`:

  - **passes_below_required**: if `this_spec.passes <
    profile.passes_required` (look up in experiment_design.yaml).
    Tag: `passes_below_required`. Verdict: `REVISE` (THINK can
    trivially bump passes). proposer's Step-4 validator auto-promotes,
    so you typically see this only if the validator was bypassed —
    treat as a sanity backstop. predicted_outcome: "without
    multi-pass, mean_sr ± sd_sr cannot be computed — single-draw
    score is indistinguishable from sampling noise at N_eps={n}".

  - **predicted_delta_within_noise_floor**: if `profile.baseline`
    is locked AND THINK's `expected_signal` describes a numeric
    target (e.g. "long-instr success ≥ 0.20" vs baseline mean_sr
    0.10), compute predicted_delta = |target − baseline.mean_sr|.
    If `predicted_delta < 2 · baseline.sd_sr` → tag:
    `predicted_delta_within_noise_floor`. Verdict: `WARN` (the
    experiment may still surface qualitative evidence even if the
    quantitative lift is below detection floor — let THINK proceed).
    predicted_outcome: "delta of {predicted_delta} is below 2×sd
    ({2·sd_sr}); even with passes={spec.passes}, this lift will
    likely be reported as inconclusive by DISTILL".

These checks ARE separate from the BLOCK three-condition gate: they
do not require `reference_experience_ids` (the references are the
profile's own baseline + the 119-threshold derivation, not a prior
refuted exp_id). They are recorded under the same
`predicted_failure_modes` array in the critique.json with their
respective `pathology_tag` strings.

=== ROUND 2 ===
{if round == 2:}
This is the SECOND CRITIC fire on iter_{n}. Round 1's critique is
shown below. THINK has rewritten the spec in response. Your job is
to judge whether the rewrite addresses the round-1 concerns. If it
does → verdict OK or WARN. If it doesn't → BLOCK (round 2's verdict
is binding; no round 3). Even on BLOCK at round 2, THINK still gets
to rebut via spec.block_override → the spec proceeds to EXPERIMENT
with the override on record.

=== goal.md (direction) ===
{goal_md}

=== constraints.md ===
{constraints_or_"(no constraints.md)"}

=== knowledge.md ===
{knowledge_md}

=== experience.jsonl (full; verdict-tagged) ===
{experience_jsonl}

=== search_space.md (axes + latest reflection's Frontier) ===
{search_space_summary}

=== spec being vetted (id: {spec_id}) ===
{this_spec_json}

=== sibling specs in the envelope (context only — DO NOT render verdict on these) ===
{sibling_specs_json_or_"(K=1; no siblings)"}

=== active_workspace_{spec_id} overlay (paths only; you may Read individual files via tools) ===
{overlay_listing}

=== Round-1 critique for this spec (round 2 only) ===
{prior_critique_or_"(round 1)"}

=== Schema reference ===
{critique_schema_excerpt — schemas.md § 15}

Now reason. When done, emit a SINGLE fenced ```json block as your
final visible chunk, conforming to Critique (schemas.md § 15). All
fields REQUIRED unless marked OPTIONAL in the schema.
```

### 5. Parse + validate `critique_<spec_id>.json`

Extract the LAST fenced ```json block via `parse_final_json`.

Validate:
- Required top-level keys present (per `schemas.md § 15`).
- `spec_id` equals the `--spec-id` argument. Mismatch → return
  `SKIP_INVALID_CRITIQUE`.
- `verdict` ∈ {`OK`, `WARN`, `REVISE`, `BLOCK`, `SKIP_INVALID_CRITIQUE`}.
- `critic_round` ∈ {`1`, `2`} and matches the `--round` arg.
- If `verdict == "BLOCK"`: **every** entry in `predicted_failure_modes`
  has non-empty `reference_experience_ids`, non-empty
  `specific_check`, non-empty `predicted_outcome`. If any entry is
  missing any of these → **auto-downgrade** to `REVISE` (or `WARN`
  if no `specific_check`) and append a `validator_note:
  "BLOCK downgraded to REVISE — predicted_failure_mode N missing
  <field>"` to the critique. Log the downgrade.
- If `verdict == "REVISE"` and `critic_round == 2`: **auto-downgrade**
  to `WARN` (round 2 cannot REVISE — at most BLOCK or pass through).

On validation failure (malformed JSON, missing required top-level
keys, unknown verdict, spec_id mismatch): return
`status = SKIP_INVALID_CRITIQUE` to loop without writing
`critique_<spec_id>.json`. Loop logs, increments
`consecutive_skips`, proceeds with this spec to EXPERIMENT (treat
as `WARN`).

### 6. Write `.staging/iter_{n}/critique_<spec_id>.json`

```bash
echo "$VALIDATED_CRITIQUE_JSON" > .staging/iter_{n}/critique_${spec_id}.json
```

Also stage `.staging/iter_{n}/critique_trace_<spec_id>.md` — the
sub-agent's full text return for this spec (forensic; useful when the
critique disagrees with later DISTILL on whether the prediction was
correct).

### 7. Return to loop

```
[myloop:critic] {OK | WARN | REVISE | BLOCK}
                spec_id     = {spec_id}
                round       = {1 | 2}
                predicted   = {N} failure mode(s)
                {if BLOCK:}
                references  = {comma-joined exp_ids}
                {if REVISE/BLOCK:}
                top concern = "{first predicted_failure_mode.pathology_tag}"
                critique    = .staging/iter_{n}/critique_<spec_id>.json
                → loop will {proceed this spec to EXPERIMENT | re-spawn THINK on this spec | skip eval for this spec}
```

Return one of `{OK, WARN, REVISE, BLOCK}` to loop. Loop applies the
verdict semantics per `loop.md § 3b.5`. Sibling specs in the
envelope are unaffected.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/.staging/iter_{n}/critique_<spec_id>.json` | this skill | Critique (schema § 15); the loop's verdict-dispatch consumes it |
| `v{N}/.staging/iter_{n}/critique_trace_<spec_id>.md` | this skill | Full sub-agent return text — forensics |

_No working-memory writes._ CRITIC does not eager-append to
`knowledge.md`, `experience.jsonl`, or `hypotheses.jsonl`. Its
output is one critique per fire; if it predicts a pathology, the
prediction is recorded in `critique_<spec_id>.json` and either acted on
in-iter (REVISE/BLOCK) or carried to next iter's THINK context.
DISTILL is the one that promotes lessons (including the post-hoc
verdict on whether THIS critique was right) into working memory.

## Notes

- **CRITIC does not block forever.** Maximum two CRITIC rounds per
  spec. After round 2's verdict on a spec (binding), that spec
  proceeds — either to EXPERIMENT (OK/WARN), to a one-shot rewrite
  then EXPERIMENT (round-2 BLOCK with rebuttal), or to a skip-without-
  eval (round-2 BLOCK without rebuttal). Sibling specs are unaffected
  and may be at different rounds simultaneously. Per-iter cost
  increment from CRITIC is bounded at ~2K sub-agent spawns where K
  is the number of patched specs.

- **CRITIC is read-only on working memory.** No appends to
  `experience.jsonl` / `knowledge.md` / `hypotheses.jsonl`. The
  reason: CRITIC's predictions are *not yet* validated until the
  EXPERIMENT runs (or is skipped). DISTILL, which sees both the
  per-spec critique and that spec's eval outcome, is the right
  phase to promote critic accuracy as experience.

- **CRITIC's own accuracy becomes experience.** When DISTILL runs
  (next phase), it reads each spec's `critique_<spec_id>.json` and
  compares each `predicted_outcome` to that spec's actual eval
  result. A confirmed prediction may become an `experience.jsonl`
  entry tagged `critic_TP` ("critic right"); a refuted prediction
  becomes `critic_FP` ("critic wrong"). Future CRITIC reads these
  and calibrates: pathology tags with a history of `critic_FP` get
  weighted down. This is the double-layer learning loop — THINK
  learns about mechanisms; CRITIC learns about itself.

- **CRITIC is one sub-agent spawn.** Same independence argument as
  REFLECT/DISTILL: a fresh Claude, not the same instance that just
  wrote the spec. This is what makes the "active cross-check" land
  — the iter_3 → iter_7 recurrence happened despite identical
  context because THINK was busy producing, not auditing.

- **Skipping CRITIC for no-patch specs.** When a spec's `patch ==
  null` (baselines, no-patch probes), no mechanism is being
  introduced; CRITIC has nothing structural to vet and is skipped
  by the loop for that spec. That spec's `critic` block is omitted
  from the IterRecord entirely. Other specs in the same envelope
  still get CRITIC fires as normal.

- **Schema discipline as forcing function.** The `risk_vectors`
  block in `spec.json` (schemas.md § 8) is the structural surface
  CRITIC validates against. THINK *must* fill these fields (it
  can't gloss). Even before CRITIC pattern-matches, the act of
  having to write `state_io.grants_required: [...]` forces THINK
  to enumerate access_grants — the pure-prompt-engineering
  alternative would have been a passive "remember to add
  access_grants" bullet that, as the iter_3 → iter_7 record shows,
  THINK does not reliably honor.

- **No cross-run state.** CRITIC reads only this vN's
  `experience.jsonl`. A fresh vN bootstraps with empty experience
  → CRITIC's first round-1 fires on iter_1 will return mostly OK /
  WARN (no patterns to match). This is by design — paper-fairness
  for fresh-run baselines requires the algorithm to bootstrap from
  empty.
