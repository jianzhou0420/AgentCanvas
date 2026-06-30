# REFLECT phase — map the search space, redirect THINK off a local optimum

> **Required reading before invoking**:
> - `myloop/README.md` — the myloop mental model (REFLECT is the meta phase)
> - `myloop/schemas.md` — esp. `search_space.md` (§ 12) and
>   `ExperimentSpec.intervention_axis` (§ 8)
> - `_common/files-contract.md` § "Edit whitelist"

This skill IS the REFLECT phase of myloop — a **meta** phase that
does NOT run every iter. THINK proposes the next experiment; DISTILL
digests the last one; **REFLECT audits the search space itself** —
which *kinds* of intervention the run has tried, which it has not, and
whether THINK is stuck refining one local subspace.

REFLECT exists because single-pass THINK, under the hard "must produce
an ExperimentSpec" contract and a self-reinforcing hypothesis chain,
reliably ruts: it keeps refining whichever `intervention_axis` its
first iters happened to pick and never steps back to ask "am I even
searching the right space?". REFLECT is the diversification operator
that breaks that local optimum.

REFLECT is a **separate sub-agent spawn** — deliberately. It carries
no "produce a spec" pressure; its sole cognitive task is mapping the
space. The same reasoning that justifies DISTILL being separate from
THINK (README) justifies REFLECT being separate: an agent asked to
self-audit *while* under pressure to ship an experiment rationalises
the rut instead of naming it.

## When the loop invokes REFLECT

The loop (`loop.md § 3`) fires REFLECT on any of three triggers — it
is never on the per-iter critical path otherwise:

1. **heartbeat** — every `config.orchestrator.reflect.heartbeat_iters`
   committed iters (default 3).
2. **axis concentration** — the last
   `config.orchestrator.reflect.axis_concentration_k` committed specs
   (default 3) all share one `intervention_axis`.
3. **SKIP escalation** — THINK returned `SKIP_THINK_EMPTY`. That does
   not terminate the run directly; the loop escalates to REFLECT.
   Only a REFLECT that returns `SPACE_EXHAUSTED` terminates.

At most one REFLECT runs per iter (a heartbeat + a skip-escalation in
the same iter collapse to one spawn).

## Contract

**Inputs** (read at start of skill):
- `outputs/design_runs/myloop/{graph}/v{N}/goal.md`                    (read-only)
- `outputs/design_runs/myloop/{graph}/v{N}/constraints.md`             (read-only; OPTIONAL — hard rules, skip if absent)
- `outputs/design_runs/myloop/{graph}/v{N}/search_space.md`            (working-memory; REFLECT appends a `## reflection_N` section)
- `outputs/design_runs/myloop/{graph}/v{N}/knowledge.md`               (read-only here — facts the run has distilled)
- `outputs/design_runs/myloop/{graph}/v{N}/experience.jsonl`           (read-only — closed lessons; their verdicts feed coverage)
- `outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_*/record.json` (read-only — every `experiment.intervention_axis` + `outcome_class` + metrics)
- `workspace/graphs/{graph}.json` + `workspace/nodesets/`              (read-only — to reason about what interventions the graph actually admits)

REFLECT reads the **distilled** state, not raw eval logs. Coverage is
an axis-level audit; it does not need per-episode traces. (If a
coverage call genuinely hinges on one trajectory detail, a targeted
grep is allowed — but that is rare; raw-log mining is THINK's job.)

**Mandatory output**:
- a new `## reflection_N` section appended to `search_space.md`
  (schema in `schemas.md § 12`) — coverage table + ranked **Frontier**
  + any taxonomy extensions.

**Staged output** (promoted into the iter dir on commit):
- `.staging/iter_{n}/reflection_trace.md` — the REFLECT sub-agent's
  full text return, forensics.

**Return status** (to loop):
- `FRONTIER_OPEN` — one or more axes are untouched or under-explored
  and at least one has a mechanism-plausible path to the goal. The
  appended section's Frontier subsection ranks them. The loop's next
  THINK reads `search_space.md` and is bound by the axis-jump rule
  (`proposer.md`).
- `SPACE_EXHAUSTED` — every axis the graph admits has been explored
  and refuted/saturated, OR every untouched axis is forbidden by
  `constraints.md` / has no mechanism-grounded path to the goal.
  This is the *only* legitimate path to a `SATURATED` termination.
  REFLECT MUST justify it per-axis in the appended section.
- `SKIP_INVALID_REFLECT` — the sub-agent failed to produce a
  well-formed section. Loop logs, increments `consecutive_skips`,
  continues (the prior `search_space.md` state still stands).

## Arguments

```
/architect:myloop:reflect [<graph> [<version> [<iter>]]]
                          [--graph <name>] [--version <N>] [--iter <M>]
                          [--trigger heartbeat|concentration|skip]
```

`--iter` is the iter the REFLECT precedes (the loop passes it so the
trace stages into the right `.staging/iter_{n}/`). `--trigger` is
recorded in the section header for forensics.

## Pre-conditions

- `search_space.md` exists (loop bootstraps it from
  `data/seed_search_space.md` at vN bootstrap — see `loop.md §
  Bootstrap`).
- `goal.md` exists.
- At least `iter_0` is committed (REFLECT has nothing to audit on a
  fresh vN — the loop never fires it before iter_0's commit).

## Steps

### 1. Resolve + announce

```
[myloop:reflect] iter=iter_{n}  graph={graph}  v{N}  trigger={trigger}
                 search_space.md = {bytes} bytes / {N_axes} axes / {N_refl} prior reflections
                 iters audited   = {M} committed
                 axis histogram  = prompt-content:{a} topology:{b} control-flow:{c} ...
```

The axis histogram is computed by reading `experiment.intervention_axis`
from every committed `iteration/iter_*/record.json` — every iter
carries the field (it is a required `ExperimentSpec` field).

### 2. Read prior coverage

If `search_space.md` already has `## reflection_*` sections, read the
most recent one — it is the previous frontier and coverage verdict.
REFLECT's job this turn is to *update* it against the iters committed
since, not to re-derive from scratch.

### 3. REFLECT (free-form reasoning by a sub-agent, bounded by the section contract)

Invoke a single sub-agent. Its task is **search-space cartography**,
NOT experiment proposal.

```python
resp = Agent({
  "subagent_type": "general-purpose",
  "description":   f"myloop REFLECT before iter_{n} on {graph}",
  "prompt": render_reflect_prompt(
      graph=graph, vN=N, iter_n=n, trigger=trigger,
      goal_md            = read("goal.md"),
      constraints_md     = read_if_exists("constraints.md"),
      search_space_md    = read("search_space.md"),
      knowledge_md       = read("knowledge.md"),
      experience_jsonl   = read("experience.jsonl"),
      iter_records       = read_all("iteration/iter_*/record.json"),
      graph_json         = read(f"workspace/graphs/{graph}.json"),
      nodeset_index      = ls_with_signatures("workspace/nodesets/"),
      schema_doc         = read(".claude/commands/architect/myloop/schemas.md"),
  ),
})
```

`render_reflect_prompt` composes:

```
You are the myloop orchestrator's REFLECT phase. You are NOT proposing
an experiment. Your sole job: produce an honest map of the
INTERVENTION SEARCH SPACE this run is exploring, and judge whether
THINK is stuck in a local subspace.

myloop's failure mode you exist to catch: THINK is a single-pass
agent under a hard "produce an ExperimentSpec" contract, fed a
self-reinforcing hypothesis chain by DISTILL. It reliably keeps
proposing experiments on whichever `intervention_axis` its first iters
picked, refines that one lever class until every variant is refuted,
then returns SKIP_THINK_EMPTY and claims global saturation — when in
fact only ONE axis was ever searched.

=== The intervention taxonomy ===
search_space.md § Axes lists the axes (prompt-content / topology /
control-flow / action-space / observation-pipeline / state-memory /
model-component-config). These are a starting menu, NOT a closed set.
Read workspace/graphs/{graph}.json and the nodesets: if the graph
admits an intervention kind no listed axis captures, ADD an axis
(append to § Axes with a definition + "why distinct" line). Do not
force a real intervention into an ill-fitting axis.

=== Your three tasks ===
1. CLASSIFY. Read `experiment.intervention_axis` from every committed
   iter's record.json and produce the axis histogram. (If a record
   is somehow missing the field, classify that iter yourself from its
   spec.json / patch intent.)
2. ASSESS COVERAGE. For each axis, assign a status:
     - exhausted  — tested with ≥1 measurement-tier run AND the
       distilled verdict is that the axis cannot net-lift toward the
       goal (cite the experience.jsonl exp_ids / iters).
     - partial    — touched but not refuted; an untested sub-lever
       with a mechanism-grounded path remains.
     - untouched  — no committed iter has an intervention on it.
   Be honest in BOTH directions: do not mark an axis exhausted on one
   probe, and do not mark it open if every realisation it admits is
   forbidden by constraints.md or has no plausible mechanism.
3. RANK A FRONTIER. Order the untouched + partial axes by how
   plausibly an intervention there could move the goal metric, given
   the distilled causal map in knowledge.md and the graph's actual
   structure. For each frontier axis give: a one-line mechanism
   sketch (why it *could* work), the known ceiling/risk, and whether
   any constraints.md rule limits it. The frontier is advisory to
   THINK, not a command — but THINK is accountable to it (the
   axis-jump rule).

=== HARD CONSTRAINTS (non-negotiable) ===
{constraints_block_or_"(none — no constraints.md file)"}
An axis whose every realisation a constraint forbids is NOT a
frontier axis — record it as constraint-closed, not open.

=== When to return SPACE_EXHAUSTED ===
Only if EVERY axis is either `exhausted` or constraint-closed or has
no mechanism-grounded path to the goal — and you have said so
per-axis. SPACE_EXHAUSTED is the only route to terminating the run;
do not return it to be tidy. If even one axis is genuinely open,
return FRONTIER_OPEN. Equally: do not invent a frontier axis with no
real mechanism just to keep the run alive — an honest SPACE_EXHAUSTED
beats a manufactured frontier.

=== goal.md ===
{goal_md}

=== search_space.md (current) ===
{search_space_md}

=== knowledge.md ===
{knowledge_md}

=== experience.jsonl ===
{experience_jsonl}

=== Committed iters (record.json — axis, outcome, metrics each) ===
{iter_records_brief}

=== Graph + nodesets ===
{graph_json}
{nodeset_index}

=== Schemas (single source of truth — search_space.md is § 12) ===
{schema_doc_excerpt}

Now reason. When done, emit a SINGLE fenced ```markdown block: the
`## reflection_N` section to append to search_space.md, conforming to
schemas.md § 12 (header with trigger + iter range; per-axis coverage
table; ranked Frontier subsection; any § Axes extensions called out).
Then, as the final line, emit a SINGLE fenced ```json block:
{"status": "FRONTIER_OPEN" | "SPACE_EXHAUSTED" | "SKIP_INVALID_REFLECT",
 "frontier_axes": ["<ranked>", ...],   // [] if SPACE_EXHAUSTED
 "reason": "..."}
```

### 4. Parse + validate

- Extract the ```markdown section and the final ```json status block.
- Validate the section has: a `## reflection_N` header (N = prior
  reflection count + 1), a per-axis coverage table covering every
  axis in `search_space.md § Axes` (plus any the agent added), and a
  `### Frontier` subsection.
- Validate `status` ∈ {FRONTIER_OPEN, SPACE_EXHAUSTED,
  SKIP_INVALID_REFLECT}. If `FRONTIER_OPEN`, `frontier_axes` MUST be
  non-empty and every entry MUST be an axis named in the section.
- Cross-check: if the agent extended `§ Axes`, the new axis text must
  be present in the section's table.
- On malformed output → return `SKIP_INVALID_REFLECT` with the error.

### 5. Eager-write + stage

- **Eager-append** the validated `## reflection_N` section to
  `search_space.md`. If the agent extended `§ Axes`, splice those
  new axis bullets into `§ Axes` first (still append-only — new
  bullets, never rewriting existing ones; each ends `(added: iter_n)`).
- Write the sub-agent's full return text to
  `.staging/iter_{n}/reflection_trace.md`.

### 6. Return to loop

```
[myloop:reflect] {status}
                 reflection_{N} appended to search_space.md
                 frontier    = {frontier_axes joined}  (or "— (space exhausted)")
                 axes        = {n_exhausted} exhausted / {n_partial} partial / {n_untouched} untouched / {n_closed} constraint-closed
                 → loop will {re-run THINK with this frontier | terminate SATURATED}
```

Return `status` to loop.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/search_space.md` | this skill (eager) | new `## reflection_N` section; possible `§ Axes` extensions |
| `v{N}/.staging/iter_{n}/reflection_trace.md` | this skill | full REFLECT sub-agent return — forensics; promoted to `iteration/iter_{n}/` on commit |

REFLECT writes **no** spec, hypothesis, or knowledge bullet — those
are THINK's / DISTILL's domain. REFLECT only maps the space and
redirects. The iter's `record.json` carries a small `reflect` block
(schemas.md § 5) when REFLECT fired for that iter; the loop writes it.

## Notes

- **REFLECT is not every iter.** Heartbeat + concentration + skip
  escalation only. Most iters are a plain THINK → EXPERIMENT →
  DISTILL triple; REFLECT is the periodic / triggered meta check.
- **REFLECT cannot itself be the rut.** It is a fresh sub-agent each
  spawn, with no spec-shipping pressure and an explicit instruction
  to extend the taxonomy from the graph. It can still miss an axis
  (unknown unknowns are unkillable) — mitigation is the extensible
  taxonomy and the per-spawn re-audit, not a completeness claim.
- **REFLECT does not edit `workspace/*`** and proposes no patch. It
  is pure cartography.
- **The frontier is advisory, enforced softly.** REFLECT ranks; the
  loop's axis-jump rule (`proposer.md`) makes THINK *accountable* to
  the frontier (default: jump off a concentrated axis; staying needs
  a written rebuttal the next REFLECT reviews) — but THINK still
  reasons freely within that accountability.
- **SPACE_EXHAUSTED is load-bearing.** It is the *only* honest way
  the loop reaches `SATURATED`. A `SKIP_THINK_EMPTY` from THINK
  is downgraded to "no experiment in the current frontier" and
  escalates here; the run ends only when REFLECT — the phase that
  actually looked at the whole space — says so.
