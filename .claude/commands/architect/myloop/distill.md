# DISTILL phase — sharpen working memory while the iter is still warm

> **Required reading before invoking**:
> - `myloop/README.md` — v1 mental model
> - `myloop/schemas.md` — every file's schema, especially `experience.jsonl` (§3),
>   `hypotheses.jsonl` (§4), `knowledge.md` (§2), the per-spec `distill`
>   block under `IterRecord.specs[*]` (§5), and `iter_summary` (§5)

This skill is the **DISTILL phase** of one myloop iter. It runs after
EXPERIMENT inside the same iter — NOT folded into the next iter's
THINK. **One sub-agent spawn per iter** even when K > 1 specs ran —
the single spawn sees all K specs side-by-side so it can produce
both per-spec verdicts AND cross-spec lessons in one pass. This
separation between DISTILL and THINK exists because:

- THINK plans the *next* experiment — it may read raw logs, but for
  one purpose: targeting the next experiment's episode set.
- DISTILL digests *this* iter's results with **the full warm context
  of the iter just run** (this iter's spec envelope, `think_trace.md`,
  `expected_signal` per spec, the freshly-written per-spec eval
  artifacts) and promotes the lessons into working memory.

The split is by timing and cognitive task — "design the next test"
vs "digest the K tests just run" — not by log access. Separate
sub-agent spawns keep each focused.

## Contract

**Inputs** (read at start of skill):

- Persistent working memory (current state):
  - `outputs/design_runs/myloop/{graph}/v{N}/goal.md`              (read-only)
  - `outputs/design_runs/myloop/{graph}/v{N}/constraints.md`       (read-only; OPTIONAL — skip if absent)
  - `outputs/design_runs/myloop/{graph}/v{N}/knowledge.md`         (mutable: append)
  - `outputs/design_runs/myloop/{graph}/v{N}/experience.jsonl`     (mutable: append)
  - `outputs/design_runs/myloop/{graph}/v{N}/hypotheses.jsonl`     (mutable: append + line-delete)
- This iter's transient evidence (under `.staging/iter_{n}/`):
  - `spec.json`                                    — envelope `{iter, specs:[K specs]}`
  - `think_trace.md`                               — why THINK picked these K specs
  - `eval_metadata_<spec_id>.json`                 — ONE per spec; aggregated metrics (mean_sr/sd_sr/robust_sr) + run_ids[] + per_ep_success[][]
  - `critique_<spec_id>.json` (per patched spec)   — final-round CRITIC verdict for that spec
  - `critique_<spec_id>_round_1.json` (per spec with round 2) — round-1 critique preserved
  - `debug_log_<spec_id>.md` (per patched spec)    — that spec's apply-step smoke retry history
  - `multi_spec_eval_log.md`                       — Python wrapper's wave plan + per-submission worker allocation + timings
- This iter's raw eval artifacts (across all specs):
  - `outputs/eval_runs/<run_id>/summary.json` — for each `run_id` in
    every spec's `eval_metadata_<id>.run_ids[]` (passes>1 specs have
    multiple run_ids)
  - `outputs/eval_runs/<run_id>/episodes/ep*/episode.json`
  - `outputs/eval_runs/<run_id>/episodes/ep*/log.jsonl`   (sample carefully; full read can be MB-scale)

**Mandatory output** (K files + one iter-level summary):

- `.staging/iter_{n}/distill_<spec_id>.json` — ONE per spec in
  `spec.json` envelope. Each conforms to the per-spec `distill` block
  of `IterRecord.specs[*]` (schemas.md § 5):
  ```json
  {
    "spec_id":                "spec_iter_{n}_A",
    "verdict":                "confirmed | refuted | inconclusive",
    "promoted_to_experience": ["exp_iter{n}_001", ...],
    "resolved_hypotheses":    ["hyp_4", ...],
    "new_hypotheses":         [],
    "knowledge_diffs":        [{"section": "...", "bullet_id": "..."}, ...]
  }
  ```
  All four list fields MAY be empty if nothing was distilled for
  that spec (e.g., experiment crashed before producing usable
  signal, or this spec genuinely confirmed nothing).

- `.staging/iter_{n}/distill_summary.json` — iter-level integrated
  summary:
  ```json
  {
    "milestone_after":   "<one sentence — cross-spec near-term focus>",
    "cross_spec_lesson": "<one or two sentences — if any spec's result informs another's interpretation, name it; else empty>"
  }
  ```
  `milestone_after` MAY be empty string; commit's `iter_summary.milestone_after` reads from this file.

**Side-effects** (eager writes during the distill turn):

- `experience.jsonl` — append entries for any lesson learned (whether
  confirmed, refuted, or inconclusive). One entry per resolved
  hypothesis at minimum.
- `hypotheses.jsonl` — line-delete any resolved hypothesis (matched by
  `hyp_id`); append new conjectures surfaced this iter.
- `knowledge.md` — append bullets to the appropriate section for any
  new pure fact uncovered (every new bullet ends with `(added: iter_{n})`).

**SKIP semantics**: if DISTILL genuinely has nothing to write (rare —
even experiment crashes can be distilled into a refuted-patch experience),
return `status=SKIP_DISTILL_EMPTY`. Loop logs and continues — the iter
still commits, but its `record.json` omits the `distill` block.

**Failure semantics**: if the sub-agent returns malformed output (no
distill files written, top-level shape wrong, or invalid JSON),
return `status=SKIP_INVALID_DISTILL`. Loop logs, sets
`consecutive_skips++`, and commits the iter with every spec's
`distill` block omitted from `record.specs[*]`. Recovery is implicit
— the next iter's THINK will still see this iter's spec envelope +
per-spec experiment blocks in `record.json` and can re-reason about
it manually if it cares.

## Arguments

```
/architect:myloop:distill [<graph> [<version> [<iter>]]]
                          [--graph <name>] [--version <N>] [--iter <M>]
```

Iter resolution: loop passes the current iter index (`iter_n`); manual
invocation defaults to `max(committed iters)` (DISTILL targets the
**most-recently-staged** iter, not yet committed).

## Pre-conditions

- `.staging/iter_{n}/spec.json` envelope exists and parses.
- For each spec in `specs[]`, an
  `.staging/iter_{n}/eval_metadata_<spec_id>.json` exists (loop
  synthesizes synthetic metadata files for `critic_block` /
  `implementer_skip` specs so DISTILL has uniform per-spec input).
- For each `run_id` in any spec's `eval_metadata_<id>.run_ids[]`,
  `outputs/eval_runs/<run_id>/_DONE` exists (the run finalized).

If a spec's `eval_metadata_<id>.outcome_class = "crash"` /
`"implementer_skip"` / `"critic_block"`, DISTILL still produces a
per-spec verdict for it — the lesson is "patch X failed because Y"
or "CRITIC blocked X because Z", which IS information worth
promoting to `experience.jsonl`.

## Steps

### 1. Resolve + announce

```
[myloop:distill] iter=iter_{n}  graph={graph}  v{N}
                 K specs        = {len(specs)} ({comma-joined spec_ids})
                 per-spec       = A: kind=custom outcome=ok mean_sr=0.31 (passes=3)
                                  B: kind=custom outcome=critic_block
                 (etc.)
```

### 2. DISTILL (one sub-agent spawn for the whole iter)

Invoke a single sub-agent with all K specs' evidence in one prompt.
The single spawn is intentional — it lets the agent reason about
cross-spec patterns (e.g. "A's confirmed mechanism makes B's
refutation harder to interpret because they share node X").

```python
resp = Agent({
  "subagent_type": "general-purpose",
  "description":   f"myloop DISTILL iter_{n} on {graph} (K={len(specs)})",
  "prompt": render_distill_prompt(
      graph=graph, vN=N, iter_n=n,
      goal_md             = read("goal.md"),
      constraints_md      = read_if_exists("constraints.md"),
      knowledge_md        = read("knowledge.md"),
      experience_jsonl    = read("experience.jsonl"),
      hypotheses_jsonl    = read("hypotheses.jsonl"),
      envelope_json       = read(".staging/iter_{n}/spec.json"),         # full envelope
      think_trace_md      = read(".staging/iter_{n}/think_trace.md"),
      per_spec_metadata   = {                                            # K entries
          sid: read(f".staging/iter_{n}/eval_metadata_{sid}.json")
          for sid in [s.spec_id for s in envelope.specs]
      },
      per_spec_critique   = {                                            # only patched specs
          sid: read(f".staging/iter_{n}/critique_{sid}.json")
          for sid in patched_spec_ids
      },
      raw_eval_roots      = {                                            # nested: spec_id → list of run dirs (length=passes)
          sid: [f"outputs/eval_runs/{rid}/" for rid in metadata[sid].run_ids]
          for sid in [s.spec_id for s in envelope.specs]
      },
      schema_doc          = read(".claude/commands/architect/myloop/schemas.md"),
  ),
})
```

`render_distill_prompt` composes:

```
You are the myloop orchestrator running the DISTILL phase of iter_{n}.
Your job is to sharpen working memory based on what THIS ITER's K
specs just observed — NOT to plan the next experiment (that is
THINK's job in iter_{n+1}).

This iter ran K = {K} specs in parallel (different patches against
the same baseline). Your output is K per-spec distill blocks PLUS
one iter-level summary. The single-spawn design lets you reason
cross-spec: if A's confirmed mechanism makes B's result reinterpret-
able, say so in the cross_spec_lesson field.

You may freely use Read / Grep / Bash / Edit / Write during this turn
to:
  - read raw eval artifacts under outputs/eval_runs/{run_id}/episodes/ep*/
    (per-episode log.jsonl is the highest-signal file; sample
    selectively, do NOT cat all of them)
  - append entries to experience.jsonl
  - delete-line + append to hypotheses.jsonl
  - append bullets to knowledge.md (under existing or new sections,
    each ending with `(added: iter_{n})`)

You MUST NOT:
  - modify workspace/* (patches are EXPERIMENT's domain)
  - touch goal.md (user-owned)
  - rewrite or delete existing knowledge.md bullets (append-only in v1)
  - modify spec.json envelope, think_trace.md, or any
    eval_metadata_<spec_id>.json (they are this iter's frozen
    evidence)

=== HARD CONSTRAINTS (non-negotiable) ===
{constraints_block_or_"(none — no constraints.md file)"}

These rules are absolute. Any new hypothesis you append or any
knowledge bullet you write MUST respect them. New hypotheses that
implicitly require violating a constraint (e.g. "test whether
gpt-4o lifts SR" when constraints fix gpt-5-mini) are forbidden —
don't write them.

=== goal.md ===
{goal_md}

=== knowledge.md (current state) ===
{knowledge_md}

=== experience.jsonl (current state, last 20 entries) ===
{experience_tail}

=== hypotheses.jsonl (currently open) ===
{hypotheses_full}

=== This iter's envelope (K = {K} specs) ===
{envelope_json}

=== This iter's think_trace (why THINK picked these specs) ===
{think_trace_md}

=== Per-spec eval metadata ===
{for spec_id, metadata in per_spec_metadata.items():}
--- {spec_id} ---
{metadata}

=== Per-spec CRITIC critique (only for patched specs) ===
{for spec_id, critique in per_spec_critique.items():}
--- {spec_id} ---
{critique}

=== Where to find raw eval data (per spec, per pass) ===
{for spec_id, run_dirs in raw_eval_roots.items():}
--- {spec_id} (passes={len(run_dirs)}) ---
{for run_dir in run_dirs:}
  {run_dir}/
  ├── summary.json
  ├── episodes/ep0000/  ep0001/  ...
  │   ├── episode.json
  │   └── log.jsonl

=== Schemas (single source of truth) ===
{schema_doc_excerpt}

Now reason about ALL K specs together. Then for each spec_id in
[{comma-joined spec_ids}]:

1. Decide that spec's verdict (confirmed | refuted | inconclusive)
   based on its eval evidence + CRITIC's prediction (if any).
   passes>1 specs: weigh by mean_sr ± sd_sr AND robust_sr; a
   spec whose mean_sr lifts the score but whose robust_sr stays at
   baseline is more inconclusive than confirmed.

2. For each open hypothesis in `hypotheses.jsonl` resolved by this
   spec's evidence:
     - Append an `experience.jsonl` entry per §3 schema, including
       `tags` carrying both the pathology context AND the spec_id
       (so future CRITIC can disambiguate sibling-spec evidence).
     - Delete that hypothesis's line from `hypotheses.jsonl`.

3. If this spec's evidence surfaces a new pattern worth a future
   test, append a new `hyp_*` line to `hypotheses.jsonl`.

4. If this spec's evidence reveals a new pure fact (about the
   system, graph, env, dataset, or LLM behavior), append a bullet
   to `knowledge.md`, ending with `(added: iter_{n}, spec_id)`.

5. If this spec had a `critique_<spec_id>.json` (CRITIC fired):
   evaluate each predicted_failure_mode against actual eval evidence.
   Tag the resulting experience entry with `critic_TP` or
   `critic_FP` per loop.md § 4. For critic_block specs (no eval),
   tag `critic_unverified`.

Then synthesize the iter-level summary:

6. `cross_spec_lesson`: if A's result informs B's interpretation
   (or vice versa), state it in one or two sentences. Common
   cases: "A confirmed mechanism X, which means B's negative result
   on X-adjacent mechanism Y is more decisive"; "A and B both
   touched node Z, but only A's `_self_log` shows the mechanism
   firing — B's outcome is contaminated". If no cross-spec
   coupling, leave empty.

7. `milestone_after`: one sentence on the next iter's focus.
   Considers ALL K specs' results, not just the best.

When done, emit a SINGLE fenced ```json block as your final visible
chunk with this exact shape:

```json
{
  "per_spec": {
    "spec_iter_{n}_A": {
      "verdict": "confirmed | refuted | inconclusive",
      "promoted_to_experience": ["exp_iter{n}_001", ...],
      "resolved_hypotheses": [],
      "new_hypotheses": [],
      "knowledge_diffs": []
    },
    "spec_iter_{n}_B": { ...same shape... }
  },
  "iter_summary": {
    "milestone_after": "...",
    "cross_spec_lesson": "..."
  }
}
```

Also Write per-spec files:
  `.staging/iter_{n}/distill_<spec_id>.json` (one per spec_id in per_spec)
  `.staging/iter_{n}/distill_summary.json` (the iter_summary block)

If you genuinely have nothing to distill for the WHOLE iter, emit
`{"status": "SKIP_DISTILL_EMPTY", "reason": "..."}` instead and do
not write the distill files. Partial-skip (some specs have lessons,
others don't) is normal — just emit empty lists for the lessonless
specs, do NOT use SKIP_DISTILL_EMPTY.
```

### 3. Parse + validate the output

After the sub-agent returns:

- Extract the LAST fenced ```json block.
- If `{"status": "SKIP_DISTILL_EMPTY", ...}`: do not write any
  distill files; return `SKIP_DISTILL_EMPTY` to loop.
- Otherwise validate top-level shape:
  - Required keys: `per_spec` (dict, keyed by spec_id), `iter_summary`
    (dict with `milestone_after` + `cross_spec_lesson` strings).
  - `per_spec` keys must cover every spec_id in the envelope's
    `specs[]` (no missing per-spec verdicts). Extra keys → fail.
- For each entry in `per_spec`:
  - Required keys: `verdict` ∈ {confirmed, refuted, inconclusive},
    `promoted_to_experience`, `resolved_hypotheses`, `new_hypotheses`,
    `knowledge_diffs`.
  - Each `hyp_id` in `resolved_hypotheses` no longer appears in
    `hypotheses.jsonl` (the agent should have deleted them).
  - Each `hyp_id` in `new_hypotheses` does appear in `hypotheses.jsonl`.
  - Each `exp_id` in `promoted_to_experience` appears in
    `experience.jsonl`.
- On validation failure: signal `SKIP_INVALID_DISTILL` with the
  specific error. Loop logs, increments `consecutive_skips`, and
  commits the iter with every spec's `distill` block omitted.

### 4. Stage per-spec distill files + summary + trace

```bash
# One distill_<spec_id>.json per spec
for spec_id in <all spec_ids in per_spec>; do
    jq ".per_spec.\"${spec_id}\" + {spec_id: \"${spec_id}\"}" <<< "$DISTILL_JSON" \
      > .staging/iter_{n}/distill_${spec_id}.json
done

# Iter-level summary (one file, simple object)
jq '.iter_summary' <<< "$DISTILL_JSON" \
  > .staging/iter_{n}/distill_summary.json

# Full sub-agent text return for forensics (shared across specs)
echo "$SUBAGENT_RAW_OUTPUT" > .staging/iter_{n}/distill_trace.md
```

### 5. Return to loop

```
[myloop:distill] OK
                 K specs       = {len(per_spec)}
                 distill files = .staging/iter_{n}/distill_{spec_ids}.json
                                 + .staging/iter_{n}/distill_summary.json
                 per-spec verdict = A:confirmed B:refuted (etc.)
                 resolved      = {N} hypotheses (across all specs)
                 new           = {M} hypotheses
                 experience    = {K} entries appended
                 knowledge     = {L} bullets appended
                 → loop will commit iter next
```

Return `status=OK` to loop. Loop merges each `distill_<spec_id>.json`
into the corresponding `record.specs[*].distill` block at atomic
commit, and reads `distill_summary.json` for `iter_summary.milestone_after`.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/.staging/iter_{n}/distill_<spec_id>.json` | this skill | Per-spec distill block (one per spec; merged into record.specs[*].distill at commit) |
| `v{N}/.staging/iter_{n}/distill_summary.json` | this skill | Iter-level summary (cross_spec_lesson + milestone_after for iter_summary) |
| `v{N}/.staging/iter_{n}/distill_trace.md` | this skill | Full sub-agent return text — forensics (one file, covers all K specs) |
| `v{N}/experience.jsonl` | this skill (eager) | Appended lessons learned (per-spec entries; tagged with spec_id in `tags`) |
| `v{N}/hypotheses.jsonl` | this skill (eager) | Resolved entries deleted, new entries appended |
| `v{N}/knowledge.md` | this skill (eager) | Appended bullets tagged `(added: iter_{n}, spec_id)` |

## Notes

- **DISTILL is one sub-agent spawn.** Same architecture as THINK,
  separate cognitive task. Each spawn has focused inputs.
- **Warm context is the whole point.** This skill runs IMMEDIATELY
  after EXPERIMENT in the same iter, so `spec.json` + `think_trace.md`
  + `expected_signal` + freshly-written `log.jsonl` files are all
  cognitively adjacent. For *lessons*, the next iter's THINK reads the
  distilled hypotheses/experience/knowledge rather than re-deriving
  them from logs — though it may still read raw logs to target its
  next experiment's episode set.
- **Per-ep `log.jsonl` is the signal source.** episode.json gives
  aggregate-per-ep metrics; log.jsonl gives node-by-node trace
  (every llmCall input/output, every observation, every action).
  When testing a hypothesis like "agent overshoots goal viewpoint",
  the answer is in log.jsonl, not episode.json. Sample 2–3 eps deeply
  rather than skim all eps shallowly.
- **No revert chain.** Eager writes during DISTILL stay on disk even
  if DISTILL is later marked SKIP_INVALID. Recovery is by reasoning,
  not rollback: the next THINK sees the (possibly inconsistent) state
  and reasons about it.
- **Failure modes**: see the failure-semantics section above. There
  are no DISTILL-specific failure modes worth designing fail-safes
  for — only generic sub-agent infra crash + contract violation,
  which `SKIP_INVALID_DISTILL` covers.
