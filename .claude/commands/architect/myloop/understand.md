# Build the myloop mental model

> **NOT a `_common/understand.md` stub.** myloop's mental model
> diverges materially from ADAS-shape variants (no archive, eleven
> structured files, goal-driven, a triggered REFLECT meta-phase).
> This file is standalone.

## Purpose

Run this skill at the start of a session before invoking
`/architect:myloop:loop` (or any other myloop skill). It loads the
myloop mental model into your conversation so subsequent skills don't
have to.

If `/architect:myloop:loop` is invoked with `--skip-understand`,
this skill is bypassed. Re-running it inside an already-warmed
conversation is cheap but redundant.

## Arguments

```
/architect:myloop:understand [<graph> [<version>]]
                             [--graph <name>] [--version <N>]
                             [--for loop | adhoc]   default: adhoc
```

## Steps — 4-layer model load

### Layer 0: Variant identity

- Read `myloop/README.md` (full).
- Read `myloop/schemas.md` (full — single source of truth).

If `--for loop` (i.e. about to start a loop): also read
`myloop/proposer.md`, `myloop/critic.md`, `myloop/distill.md`,
`myloop/reflect.md`, and `myloop/loop.md` so the orchestrator's
THINK + CRITIC + DISTILL + REFLECT contracts are all loaded
(EXPERIMENT is inline in `loop.md § 3c`).

### Layer 1: Shared framework contracts

- Skim `_common/files-contract.md` § "Edit whitelist",
  § "Resolve protocol". (myloop reuses these.)
- Note the emphasis: **THINK → EXPERIMENT → DISTILL triple + a
  triggered REFLECT meta-phase; eleven-file working memory (incl.
  `search_space.md` + the `trace.md`/`lineage.md` rollups);
  goal-driven termination; `SATURATED` only on a
  REFLECT `SPACE_EXHAUSTED` verdict; no archive.jsonl.**

### Layer 2: Working-memory state for this {graph}/v{N}

If `<graph>` + `<version>` were given (or inferable from cwd
context):

```
RUN_DIR=outputs/design_runs/myloop/{graph}/v{N}/
```

Read in this order (fall back gracefully if missing):

| File | Read mode | Why |
|---|---|---|
| `goal.md` | full | the north star; everything downstream is relative to this |
| `knowledge.md` | full | pure facts about the system + graph (no surprises later) |
| `search_space.md` | full | intervention-axis taxonomy + the latest REFLECT coverage / frontier |
| `experience.jsonl` | last 20 entries | recent closed-case findings |
| `hypotheses.jsonl` | full (all open) | what's currently being investigated |
| `experiment_design.yaml` | full | available eval profiles incl. orchestrator-authored probes |
| `tools/*.py` | docstring of each | tool index — what helpers are available |
| `trace.md` (if exists) | full | one-row-per-iter history table — the whole run at a glance |
| `lineage.md` (if exists) | last 3 sections | recent per-iter narrative (think / change / metrics / distill) |
| `iteration/iter_*.json` | last 3 | recent activity, what was just tried |
| `SUMMARY.md` (if exists) | full | a previous run's wrap-up, if this vN was already terminated once |

Print a one-screen summary:

```
[myloop:understand] Layer 2 — vN working memory
  goal.md         = <ok>  ultimate: "<first sentence>"
  knowledge.md    = {N_sections} sections, {N_bullets} bullets
  search_space    = {N_axes} axes, {N_refl} reflections (frontier: <top axis>)
  experience      = {K} entries (last verdict: <recent>)
  hypotheses      = {H} open (highest priority: <hyp_id> "<conjecture>")
  experiment_design = {D} profiles ({D_user} authored during prior iters)
  tools           = {T} (most-recent: <filename> — "<docstring summary>")
  iters           = {M} committed (last: iter_{M}, K={K}, outcome={iter_summary.outcome_class}, best={best_score:.3f} on {best_spec_id})
```

(For v0 records or any iter using the pre-2026-05-24 single-spec
schema — `experiment` block at top level rather than `specs[]` — the
print falls back to `outcome={experiment.outcome_class},
metric={experiment.metrics_digest.success:.3f}` and omits the K /
best_spec_id fields.)

### Layer 3: Workspace + recent eval artifacts

- `workspace/graphs/{graph}.json` — current graph topology (count
  nodes / wires; list custom nodeset prefixes).
- `workspace/architect/exp_profiles/{graph}.yaml` — graph's eval profiles
  (smoke, perf, full as baseline).
- If recent iters reference eval `run_id`s, spot-check
  `outputs/eval_runs/{run_id}/summary.json` for one of them so the
  per-ep artifact layout is fresh in memory.

## What you should be able to answer after this

After running understand, you (the assistant / orchestrator's host
session) should be able to answer without re-reading files:

- What is this run trying to achieve? (from goal.md § Ultimate)
- When does it stop? (from goal.md § Termination)
- What facts do we already know? (knowledge.md sections)
- Which intervention axes are explored vs still open? (search_space.md
  § Axes + the latest reflection's frontier)
- What's currently being investigated? (open hypotheses)
- What tools are at the orchestrator's disposal? (platform + tools/*.py)
- What was just tried and what happened? (last 3 iters)
- What eval profiles can the next iter use? (experiment_design.yaml)

If any of these are unanswerable, re-read the missing file before
proceeding.

## Outputs

This skill **does not write to disk**. It is read-only mental-model
loading. The next skill (typically loop or proposer) consumes the
warmed conversation context.

## Notes

- **Layer 2 is the largest read budget**. If you are continuing a
  long-running vN with 50 iters, do NOT read all 50 `iter_*.json`
  — last 3 is enough; older iters' content has already been
  distilled into knowledge.md / experience.jsonl.
- **`experience.jsonl`'s tail is the right slice**, not the head.
  Most-recent lessons dominate near-term reasoning.
- **No archive.jsonl.** If you see one, it's a leftover from a prior
  adas-subagent run in the same RUN_DIR — ignore it (myloop doesn't
  read it). Don't conflate with `iteration/`.
