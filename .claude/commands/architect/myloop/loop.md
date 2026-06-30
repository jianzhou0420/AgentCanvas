# Orchestrate myloop cycles

> **Required reading before invoking**:
> - `myloop/README.md` — the myloop mental model (knowledge-distillation orchestrator)
> - `myloop/schemas.md` — every file's schema, especially `ExperimentSpec`, `IterRecord`, `search_space.md` (§ 12)
> - `myloop/reflect.md` — the REFLECT meta-phase contract + trigger semantics
> - `_common/files-contract.md` — edit whitelist, resolve protocol

This skill is the **outer orchestrator** of myloop. It owns:
- bootstrapping a fresh `vN/` dir
- adopting a cached frozen baseline for `iter_0` from the shared
  `baseline_registry.yaml` instead of re-running it (§ 3b₀)
- enforcing the hard `THINK → EXPERIMENT → DISTILL` triple every iter
- the EXPERIMENT phase itself (sandwiched between proposer.md / THINK
  and distill.md / DISTILL — both are separate sub-agent spawns)
- firing the **REFLECT** meta-phase on its three triggers (heartbeat /
  axis-concentration / SKIP-escalation — see § 3a′) and acting on its
  verdict
- atomic commit of the iter dir
- termination polling against `goal.md § Termination`

myloop has **no archive.jsonl** — working memory is the eleven
vN-scoped files described in `schemas.md` (including the
`trace.md` / `lineage.md` rollups).

**Why REFLECT exists.** A single-pass THINK, left alone, ruts: it
keeps refining one `intervention_axis` and, on running out of
variants there, returns `SKIP_THINK_EMPTY` claiming *global*
saturation when only one axis was ever searched. So `SKIP_THINK_EMPTY`
does NOT terminate the run directly — it escalates to the REFLECT
meta-phase, and only a REFLECT verdict of `SPACE_EXHAUSTED` yields a
`SATURATED` termination.

## Arguments

```
/architect:myloop:loop [<graph> [<version>]]
                      [--graph <name>] [--version <N>]
                      [--goal "<one-line ultimate>"]    bootstrap goal.md if missing; ignored if goal.md exists
                      [--cons-file <path>]              constraints layer 3: append the markdown at <path>
                      [--constraints "<text>"]          constraints layer 4: append this inline text
                      [--max-iters N]            override config.orchestrator.caps.max_iters
                      [--max-consecutive-skips K] override config.orchestrator.caps.max_consecutive_skips
                      [--from-iter M]            default auto-resolve from iteration/
                      [--allow-old-version]
```

## Pre-conditions

- `workspace/graphs/{graph}.json` exists.
- `workspace/architect/exp_profiles/{graph}.yaml` exists with at least `smoke_<graph>`
  (the EXPERIMENT apply-step's smoke retry needs it).
- Goal must be specifiable in one of two forms (hard contract — both
  missing = abort):
  - **detailed**: `outputs/design_runs/myloop/{graph}/v{N}/goal.md`
    pre-exists with all three sections (Ultimate / Escalation /
    Termination) per `myloop/schemas.md § 1`.
  - **one-liner**: `--goal "<text>"` passed on the command line; loop
    bootstraps a goal.md with the text as Ultimate and built-in
    defaults for Escalation / Termination (see §2a below).

## Steps

### 1. Resolve graph + version + entry iter

Apply the **Resolve protocol** (`_common/files-contract.md`). Print:

```
RUN_DIR=outputs/design_runs/myloop/{graph}/v{N}
  graph         = {graph}    (input: <auto> | <exact> | "<raw>" → <resolved>)
  version       = {N}        (input: <auto> | <N>)
  pipeline      = myloop (knowledge-distillation orchestrator)
  entry iter    = iter_{M+1}  (resume) | iter_0  (fresh)
  goal.md       = <ok>  |  <MISSING — abort>
  knowledge.md  = {bytes} bytes
  search_space  = {N_axes} axes / {N_refl} reflections
  experience    = {K} entries
  hypotheses    = {H} open
  iters         = {M} committed
  cap           = max-iters={max_iters}, max-consecutive-skips={K}
```

### 2. Bootstrap (only if vN is fresh)

If `outputs/design_runs/myloop/{graph}/v{N}/` does not exist (or exists but
is empty of `goal.md`):

**2a. Goal — hard contract (with `--goal` fallback)**

Resolution priority:

1. If `goal.md` exists AND `--goal` was passed → **prefer the file**;
   print `[myloop:loop] notice: goal.md exists, ignoring --goal`.
2. If `goal.md` exists → validate it has all three sections (Ultimate,
   Escalation, Termination). If a section is missing, abort with the
   section name.
3. If `goal.md` is missing AND `--goal "<text>"` was passed → bootstrap
   `goal.md` using `<text>` as the Ultimate paragraph and the built-in
   defaults below for Escalation + Termination. Print
   `[myloop:loop] bootstrapped goal.md from --goal (Ultimate only;
   defaults applied for Escalation + Termination)`.
4. If both missing → abort:
   ```
   [myloop:loop] ABORT — goal is required to start a myloop run.

   Two ways to provide it:
     a) Author outputs/design_runs/myloop/{graph}/v{N}/goal.md with
        Ultimate / Escalation / Termination sections (see
        myloop/schemas.md § 1).
     b) Re-invoke with: --goal "<one-line direction>"
        e.g. --goal "push MapGPT72 success past 0.50"
   ```
   Exit non-zero.

**Built-in defaults (for `--goal` bootstrap)**

```markdown
## Escalation
- probe / custom targeted subset (1–30 ep, THINK-composed) — run
  freely; build failure-mode subsets from past logs as needed. A
  custom subset large enough to measure (≥ ~30 ep) is the default
  per-iter measurement tier
- perf_{graph} (full paper-comparable subset) — run only when a
  custom ≥30-ep run shows non-degenerate lift (≥ 0.05 success over
  the prior recorded best), to amortize the cost
- smoke_{graph} is NOT an escalation tier — it is the EXPERIMENT
  apply-step's correctness gate (≈3 ep, "does the change run"), never
  this iter's measurement

## Termination
- saturated: 5 consecutive iters with distill.knowledge_diffs == [] AND
  distill.resolved_hypotheses == [] AND distill.new_hypotheses == []
- stuck:     consecutive_skips ≥ 3
- budget:    cumulative cost.gpu_min ≥ 600 OR cumulative
             cost.llm_tokens ≥ 5_000_000
- cap:       50 committed iters (override via --max-iters)
```

The defaults intentionally do NOT include a "goal-achieved" predicate
(e.g. "success ≥ 0.50") because that threshold is graph- and
target-specific. Users who want one can edit `goal.md` after bootstrap
to add it.

**2b. Seed knowledge + search space**

- Copy `.claude/commands/architect/myloop/data/seed_knowledge.md`
  → `knowledge.md` if the latter does not exist.
- Copy `.claude/commands/architect/myloop/data/seed_search_space.md`
  → `search_space.md` if the latter does not exist.

**2c. Empty working-memory files**

- Create empty `experience.jsonl`, `hypotheses.jsonl`.
- Create `iteration/` directory.
- Create `tools/` directory.
- If `experiment_design.yaml` does not exist, copy
  `workspace/architect/exp_profiles/{graph}.yaml` (the smoke/perf baselines)
  → `experiment_design.yaml`. (Probe entries the orchestrator
  authors will be appended there during runs.)
- Create `.staging/` and `.loop_state/` directories.

**2d. constraints.md (bootstrap-merged from up to 4 layers)**

If `vN/constraints.md` already exists (e.g. user pre-authored it, or
this is a resume run), leave it untouched. Otherwise, merge layers
in order — each layer appends if present, separated by a `## --- from
<source> ---` audit-trail header. Final concatenation is written to
`vN/constraints.md`.

| Layer | Source | Notes |
|---|---|---|
| 1 | `.claude/commands/architect/myloop/data/constraints/common.md` | pipeline-wide myloop discipline; always present in the repo |
| 2 | `.claude/commands/architect/myloop/data/constraints/{graph}.md` | per-graph hard rules; optional (only graphs we've ever worked on) |
| 3 | `--cons-file <path>` | append the markdown at that path; if path is relative, relative to repo root |
| 4 | `--constraints "<text>"` | append the inline text verbatim (wrapped under `## --- from --constraints inline ---`) |

If layers 1–4 are all empty (no common.md, no graph.md, no flags) →
do NOT create `vN/constraints.md`. The absent-file path means
"no hard rules"; downstream phases handle that case.

Soft validate after merge: at least one `MUST` or `MUST NOT` bullet
present. If validation fails (e.g. user-provided file has no rules),
print a notice and proceed (treat as "no constraints" but leave the
file in place for visibility).

Example merge:

```
# Constraints

## --- from data/constraints/common.md ---
<contents of common.md>

## --- from data/constraints/mapgpt_mp3d.md ---
<contents of graph-specific file>

## --- from --cons-file /tmp/my_extra.md ---
<contents of /tmp/my_extra.md>

## --- from --constraints inline ---
- MUST NOT change build_options.stop_after this run; only test prompt edits.
```

### 3. Per-iter state machine

For `n in range(start_iter, start_iter + max_iters)`:

```
============ myloop iter_{n} ============
```

**3a. Termination poll**

Read `goal.md § Termination`. For each predicate:

- "metric ≥ X on profile P": find the most-recent committed iter
  containing a spec whose `spec_kind == P` and check its
  `experiment.metrics_digest.mean_sr` (or `score`).
- "K consecutive iters with no new knowledge AND no resolved hypothesis":
  scan last K `iteration/iter_*/record.json` — an iter is "unproductive"
  iff EVERY entry in `specs[]` has an empty/absent `distill` block
  (`knowledge_diffs == []`, `resolved_hypotheses == []`,
  `new_hypotheses == []`, OR no `distill` block at all). One spec
  with non-empty distill counts as a productive iter.
- "cumulative GPU-minutes ≥ X" / "cumulative tokens ≥ X": sum `cost` over
  all committed iters (iter-level, not per-spec).
- Other predicates: orchestrator-extensible — write a helper in
  `tools/check_*.py` and reference its function name.

If any predicate hits → terminate with that status (`GOAL_MET` /
`SATURATED` / `BUDGET`). Go to step 6 (termination summary) — no
iter is committed.

> **Note on the `saturated` predicate.** The `goal.md § Termination`
> `saturated` predicate (5 consecutive no-diff iters) is a coarse
> heartbeat poll and still stands. But the *primary* saturation
> mechanism is the REFLECT verdict `SPACE_EXHAUSTED` (3a′ / 3b) —
> that is the one that has actually surveyed the whole intervention
> space. The poll predicate is a backstop.

**3a′. REFLECT trigger check**

Before THINK, decide whether the REFLECT meta-phase
(`/architect:myloop:reflect`) fires this iter. REFLECT is NOT on the
per-iter critical path — it fires only on a trigger. Read
`config.orchestrator.reflect` for thresholds (`heartbeat_iters`,
`axis_concentration_k`; both default 3). Evaluate:

- **heartbeat** — `n − last_reflection_iter ≥ heartbeat_iters`
  (where `last_reflection_iter` is read from `.loop_state/`; −∞ if
  REFLECT has never run). Never fires for `iter_0` — REFLECT has
  nothing to audit before the first commit.
- **axis concentration** — the last `axis_concentration_k` committed
  iters all have `iter_summary.axes_touched` equal to one common
  singleton (i.e. every spec across the last K iters used the SAME
  one axis). A K-spec iter that touched 2+ axes breaks the
  concentration streak by definition — multi-spec breadth is itself
  an axis-jump.

If either holds → fire REFLECT now:

```
Invoke /architect:myloop:reflect {graph} v{N} --iter {n} --trigger {heartbeat|concentration}
  - audits every committed iter's intervention_axis + outcome
  - eager-appends a `## reflection_N` section to search_space.md
  - stages .staging/iter_{n}/reflection_trace.md
  - returns: status = FRONTIER_OPEN | SPACE_EXHAUSTED | SKIP_INVALID_REFLECT
```

- `FRONTIER_OPEN` → continue to 3b; THINK reads the fresh frontier.
- `SPACE_EXHAUSTED` → terminate with status `SATURATED`. Go to step 6.
- `SKIP_INVALID_REFLECT` → log, increment `consecutive_skips`,
  continue to 3b anyway (the prior `search_space.md` still stands; a
  malformed REFLECT must not strand the loop).

Set `.loop_state/last_reflection_iter = n` and remember the REFLECT
result — **at most one REFLECT spawn per iter**: if it fired here, the
SKIP-escalation path in 3b reuses this result instead of re-spawning.

**3b₀. iter_0 baseline-cache fast-path** (evaluated only when `n == 0`;
for `n ≥ 1` skip straight to 3b)

`iter_0` is always a no-patch eval of the **frozen** workspace — the
generic baseline. The frozen graph does not change between vN
bootstraps or across methods, so re-running it every time wastes a full
baseline eval (tens of minutes / dollars). Before THINK, consult the
shared registry `outputs/design_runs/baseline_registry.yaml` (schema +
reuse contract in that file's own header). Compute the current
authored-graph fingerprint and look for an adoptable entry — substitute
the absolute repo root and `{graph}`:

```bash
python3 - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
REPO  = Path("<repo-root>")           # absolute repo root
GRAPH = "<graph>"                      # this run's graph name
try:
    import yaml
except ImportError:
    print("MISS reason=no-yaml"); sys.exit(0)

def fingerprint(graph):
    # sha256(16hex) over authored graph JSON + referenced workspace
    # nodeset .py files. Each file -> `<repo-rel-posix>\0<bytes>`.
    # Framework builtins (no file under workspace/nodesets/) are skipped.
    gj = REPO / "workspace" / "graphs" / f"{graph}.json"
    g  = json.loads(gj.read_bytes())
    pref = {n["type"].split("__", 1)[0]
            for n in g.get("nodes", []) if "__" in n.get("type", "")}
    files, nsr = [gj], REPO / "workspace" / "nodesets"
    for p in sorted(pref):
        flat, pkg = nsr / f"{p}.py", nsr / p
        if   flat.is_file(): files.append(flat)
        elif pkg.is_dir():   files += sorted(f for f in pkg.rglob("*.py") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(f.resolve().relative_to(REPO).as_posix().encode())
        h.update(b"\0"); h.update(f.read_bytes())
    return h.hexdigest()[:16]

reg = REPO / "outputs" / "design_runs" / "baseline_registry.yaml"
entries = (((yaml.safe_load(reg.read_text()) if reg.is_file() else {}) or {})
           .get("baselines", {}).get(GRAPH) or [])
fp = fingerprint(GRAPH)
ok = [e for e in entries
      if e.get("graph_fingerprint") == fp
      and (REPO / "outputs" / "archive_runs" / str(e["run_id"]) / "_DONE").is_file()]
ok.sort(key=lambda e: (e.get("eval") or {}).get("episode_count", 0), reverse=True)
print("HIT " + json.dumps(ok[0]) if ok
      else f"MISS reason=fingerprint={fp},entries={len(entries)}")
PYEOF
```

- **HIT** — adopt the cached baseline as `iter_0`. SKIP 3b (THINK) and
  3c (EXPERIMENT) entirely; jump to step 4 (DISTILL):
  1. **Symlink.** Ensure `outputs/eval_runs/{run_id}` is a relative
     symlink to `../archive_runs/{run_id}`
     (`ln -sfn ../archive_runs/{run_id} outputs/eval_runs/{run_id}`).
  2. **Synthesize `.staging/iter_0/spec.json`** as a K=1 envelope:
     ```json
     {
       "iter": 0,
       "specs": [
         {
           "spec_id": "spec_iter_0_A",
           "kind": "<entry's profile tier>",
           "intervention_axis": "none",
           "passes": 1,
           "patch": null,
           "eval_profile": {"name": "<entry profile>", "overrides": {}},
           "target": {
             "hypothesis_id": null,
             "design_intent": "iter_0 ground-truth baseline, adopted from baseline_registry.yaml run <run_id>"
           },
           "expected_signal": [/* entry's recorded metrics */]
         }
       ]
     }
     ```
  3. **Synthesize `.staging/iter_0/think_trace.md`** — short paragraph
     recording the cache adoption (run_id + fingerprint). Step 5's
     commit extracts the `think` block `rationale` from this; its
     `file_updates` is `[]`.
  4. **Synthesize `.staging/iter_0/eval_metadata_spec_iter_0_A.json`**
     from `outputs/archive_runs/{run_id}/summary.json`:
     `patch_applied: false`, `implementer_status: "N/A"`,
     `implementer_attempts: 0`, `passes: 1`, `run_ids: ["{run_id}"]`,
     `artifacts_dirs: ["outputs/eval_runs/{run_id}/"]`,
     `metrics_digest` (= `aggregate_metrics`, plus
     `mean_sr/sd_sr/robust_sr/score` populated from `success`),
     `per_ep_success` (wrap the single-pass per-ep list as
     `[[...]]` for shape uniformity), `outcome_class: "ok"`,
     `baseline_cache_hit: true`.
  5. Proceed to **step 4 (DISTILL)**. DISTILL digests the cached corpus
     into this vN's fresh working memory — the cache skips the eval,
     never the distillation. At commit (step 5) the `iter_0` record's
     `specs[0].experiment` block carries `baseline_cache_hit: true`
     and the `think` block is loop-synthesized. The hard `THINK →
     EXPERIMENT → DISTILL` contract still holds: a cache-adopted
     `iter_0` has a real (cached) experiment, not an absent one.

- **MISS** — fall through to normal **3b** (THINK) → **3c**
  (EXPERIMENT). The `MISS reason=...` line says why (no entry /
  fingerprint stale / run-dir gone). After a successful frozen `iter_0`
  eval, 3c's registration step (**3c-(b′)**) writes the fresh run into
  the cache so the next vN / method on this frozen graph hits.

**3b. THINK** — invoke `/architect:myloop:proposer`

```
Invoke /architect:myloop:proposer {graph} v{N} --iter {n}
  - reads goal/knowledge/experience/hypotheses + recent iters' records
    (distilled state for lessons; MAY read raw eval logs to construct
    a targeted episode subset for the next experiment)
  - reasons (one sub-agent spawn, full tool access)
  - may eager-write hypotheses/knowledge/experiment_design/tools
  - writes .staging/iter_{n}/spec.json (ExperimentSpec)
  - returns: status = OK | SKIP_THINK_EMPTY | SKIP_INVALID_SPEC | MISSING_GOAL
```

Outcome handling:

- `OK`: continue to 3c.
- `SKIP_THINK_EMPTY`: **does NOT terminate the run.** It means only
  "THINK found no experiment in the *current frontier*" — not that
  the search space is exhausted. Escalate to REFLECT:
  - **If REFLECT already ran this iter (3a′ heartbeat/concentration)**
    — regardless of its result — THINK already had the freshest
    available frontier and still could not act. Do NOT re-spawn
    REFLECT (at most one per iter). Count it as a skip:
    `rm -rf .staging/iter_{n}/`, increment `consecutive_skips`; if
    `≥ max_consecutive_skips` terminate `STUCK`, else continue to
    `n+1`. (The `## reflection_N` section REFLECT appended to
    `search_space.md` stays — only the staged `reflection_trace.md`
    is lost with the dir; the durable record is the section.)
  - **Otherwise (no REFLECT yet this iter)** — fire REFLECT now with
    `--trigger skip`:
    ```
    Invoke /architect:myloop:reflect {graph} v{N} --iter {n} --trigger skip
    ```
    - `FRONTIER_OPEN` → re-run THINK (3b) **once** with the new
      frontier. If that THINK returns `OK` → continue to 3c. If it
      returns `SKIP_THINK_EMPTY` again → count as a skip (rm staging,
      increment `consecutive_skips`, `STUCK` check, continue `n+1`).
    - `SPACE_EXHAUSTED` → terminate with status `SATURATED`. Go to
      step 6. **This is the only legitimate `SATURATED`-by-skip
      path** — the run ends because the phase that surveyed the whole
      space said so, not because one THINK pass gave up.
    - `SKIP_INVALID_REFLECT` → log, increment `consecutive_skips`;
      `STUCK` check; continue `n+1`.
- `SKIP_INVALID_SPEC`: `rm -rf .staging/iter_{n}/`, increment
  `consecutive_skips`, log; if `consecutive_skips ≥ max_consecutive_skips`
  terminate with status `STUCK`; else continue to `n+1`.
- `MISSING_GOAL`: should not happen post-bootstrap; abort with hard error.

**3b.5. CRITIC** — pre-EXPERIMENT pathology vetting, per spec

Read `.staging/iter_{n}/spec.json` (the envelope). Partition specs:

- **`patched_specs`** — entries with `patch != null`. These need CRITIC.
- **`no_patch_specs`** — entries with `patch == null`. CRITIC is
  skipped for these; no critique file is written; their `critic`
  block is omitted from the IterRecord. (Baselines / data-collection
  probes.)

If `patched_specs` is empty, skip the whole CRITIC section and jump
to 3c.

**Round 1 — fire CRITIC for each patched spec.**

For each `spec_id` in `patched_specs` (sequential by default; the
loop MAY fan out in parallel — round-1 fires share no state):

```
Invoke /architect:myloop:critic {graph} v{N} --iter {n} \
       --spec-id {spec_id} --round 1
  - reads the envelope, extracts this spec, reads experience.jsonl +
    knowledge.md + active_workspace_{spec_id} overlay
  - predicts whether this spec will recur a past refuted pathology
  - writes .staging/iter_{n}/critique_{spec_id}.json
  - returns: status = OK | WARN | REVISE | BLOCK | SKIP_INVALID_CRITIQUE
```

Record the per-spec round-1 verdict in an in-memory map.

**Rebuttal — re-spawn THINK for specs whose round 1 was REVISE/BLOCK.**

For each `spec_id` with round-1 verdict `REVISE` or `BLOCK`
(processed sequentially — each rebuttal mutates `spec.json`):

1. Preserve the round-1 critique for forensics:
   ```bash
   mv .staging/iter_{n}/critique_{spec_id}.json \
      .staging/iter_{n}/critique_{spec_id}_round_1.json
   ```

2. Re-invoke proposer with `--spec-id` targeting this spec:
   ```
   Invoke /architect:myloop:proposer {graph} v{N} --iter {n} \
          --respond-to-critique --spec-id {spec_id}
     - reads .staging/iter_{n}/critique_{spec_id}_round_1.json
     - rewrites ONLY this spec entry in spec.json (siblings byte-identical)
       OR adds spec.block_override to this entry
     - writes new .staging/iter_{n}/spec.json (envelope; old preserved as spec_round_1.json)
     - returns: status = OK | SKIP_THINK_EMPTY | SKIP_INVALID_SPEC
   ```

3. Outcome dispatch:
   - **`SKIP_THINK_EMPTY` / `SKIP_INVALID_SPEC`**: THINK gave up on
     this spec. Mark it `critic_block`: it is excluded from the
     EXPERIMENT phase; its `experiment.outcome_class` will be
     `"critic_block"` in the record. Sibling specs continue.
     **Do not** increment `consecutive_skips` for this — the iter
     overall may still produce a valid experiment via other specs.
   - **`OK`**: fire CRITIC round 2 on this spec:
     ```
     Invoke /architect:myloop:critic {graph} v{N} --iter {n} \
            --spec-id {spec_id} --round 2
     ```
     Round-2 dispatch for this spec:
     - **`OK` / `WARN` / `SKIP_INVALID_CRITIQUE`**: this spec
       proceeds to EXPERIMENT. Record `critic_round = 2`,
       `block_override = (this spec has block_override field)`.
     - **`BLOCK`**:
       - If this spec's `block_override` IS present (THINK rebutted):
         proceed to EXPERIMENT. Record `critic_round = 2`,
         `block_override = true`. DISTILL judges whether the override
         was justified.
       - If `block_override` is NOT present (THINK rewrote but CRITIC
         still blocks): mark this spec `critic_block`. Eval skipped
         for this spec; sibling specs continue.

**Post-CRITIC partition** — at the end of 3b.5, classify each
envelope entry into one of:

- `to_eval`: `patched_specs` survivors (final-round verdict OK / WARN
  / SKIP_INVALID_CRITIQUE, OR round-2 BLOCK + block_override) PLUS
  all `no_patch_specs`.
- `to_block`: `patched_specs` marked `critic_block` (round-2 BLOCK
  without override, OR proposer SKIP_* on rebuttal).

**Empty `to_eval`**: if every spec in the envelope is `to_block`
(K-spec iter where every patched spec was blocked AND there are no
no-patch specs), then there is nothing to evaluate this iter.
Increment `consecutive_skips`. Skip 3c. Run DISTILL anyway (it
records the blocks as experience). Commit at 5 with
`iter_summary.outcome_class = "critic_block"`. Continue to `n+1`.

Otherwise, proceed to 3c with `to_eval` as the spec set that gets
applied + evaluated.

**3c. EXPERIMENT** — per-spec apply + Python-wrapped multi-spec eval

Read `.staging/iter_{n}/spec.json` (the envelope). The `to_eval`
spec list from 3b.5 is the input to this phase. Split it into:

- **`patched_to_eval`** — `to_eval` specs with `patch != null`.
  Each needs the apply step (a) before the wrapper-driven eval.
- **`no_patch_to_eval`** — `to_eval` specs with `patch == null`.
  These skip (a) entirely; the wrapper evaluates them against the
  frozen workspace.

**(a) Per-spec apply step (sequential)**

myloop has **no separate implementer skill** — the apply step is
inline here. For each `spec_id` in `patched_to_eval`, run an
independent apply-step loop that turns `spec.patch` (a prose
`intent` + a `targets` file list) into real edits under
`.staging/iter_{n}/active_workspace_<spec_id>/`, smoke-tests them,
and retries with a fresh editing sub-agent if the change does not
run. Each spec gets its OWN overlay — patches do NOT stack across
specs in one iter (independent interventions). The editing work
is isolated in a sub-agent spawn so its tool I/O never enters the
loop's own context. Knobs: `retry_max` (default 3) and
`smoke_profile_key` from `config.yaml § experiment`. Apply order
across specs is sequential; sibling apply failures do not block
each other (a SKIP_RUNTIME_FAIL on spec A still lets spec B's
apply + eval proceed).

For each `spec_id` in `patched_to_eval`:

  **a-1. Seed `active_workspace_<spec_id>/` from the parent iter.**
  Frozen `workspace/` is NEVER modified.

  ```bash
  AW=.staging/iter_{n}/active_workspace_${spec_id}
  mkdir -p $AW/{graphs,nodesets}
  PARENT_AW=outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_{n-1}/active_workspace_${PARENT_BEST_SPEC_ID}
  # PARENT_BEST_SPEC_ID = parent's iter_summary.best_spec_id (if exists)
  # — i.e. seed from the parent's WINNING overlay so this spec builds on it.
  # Fallback: any present active_workspace_*/ in parent's iter dir; else frozen workspace.
  if [ -z "$(ls -A $AW 2>/dev/null)" ] && [ -d "$PARENT_AW" ]; then
      cp -r "$PARENT_AW"/. $AW/
  fi
  ```

  `$PARENT_AW` is the implicit revert anchor for this spec — a
  failed attempt resets by re-copying from it. (`iter_0` never
  patches, so `n ≥ 1` here.)

  **a-2. Edit-and-smoke retry loop** — for `attempt in range(retry_max)`:

  1. **Seed targets into the overlay**:

     ```bash
     python .claude/commands/architect/_common/lib/overlay.py prepare \
       --active-ws $AW \
       --frozen-root . --graph {graph} \
       <each path in spec.patch.targets>
     ```

     Non-zero exit = § 7 hard wall hit
     (`agentcanvas/backend/app/**` / `third_party/**`): set this
     spec's `outcome = "edit_error"`, skip to the failure path.

  2. **Spawn one editing sub-agent** — independent `Agent`
     (`subagent_type: general-purpose`, full tool access) that
     realizes `spec.patch.intent` by editing the seeded overlay
     files natively. Prompt mirrors the previous single-spec version
     but all paths reference `active_workspace_<spec_id>/`. On retry,
     the prior attempt's failures are passed in.

  3. **Post-edit validation** — `.json` must `json.load`, `.py`
     must `ast.parse`. Failure → `outcome = "edit_error"`.

  4. **Pin the LLM profile**:

     ```bash
     python .claude/commands/architect/_common/lib/pin_llm_profile.py pin \
       --active-ws $AW
     ```

  5. **Smoke eval** (per spec, in isolation — NOT mixed; smoke is
     a per-spec correctness gate, not a measurement):

     ```bash
     /experiment:run <smoke_profile_key> {graph} \
       --workspace=$ABS/.staging/iter_{n}/active_workspace_${spec_id} \
       <profile params>
     ```

  6. **Classify — runtime correctness only.** Same rules as the
     prior single-spec version: episode count match, step_count > 0,
     metrics valid. `outcome = "ok" | "crash" | "incomplete" |
     "step=0" | "malformed_metric"`.

  7. **`outcome == "ok"`** → break, this spec is ready for measured
     eval. **Failure** → reset overlay, append to this spec's
     `debug_attempts[]`, next attempt with fresh sub-agent.

  **a-3. This spec's exhaustion → SKIP_RUNTIME_FAIL.** If all
  `retry_max` attempts fail:
  - record this spec's `experiment.outcome_class =
    "implementer_skip"`, `implementer_status = "SKIP_RUNTIME_FAIL"`,
    `implementer_attempts = retry_max`.
  - REMOVE this spec from the wrapper's input list (no measured
    eval for it).
  - write `.staging/iter_{n}/debug_log_<spec_id>.md` (this spec's
    per-attempt retry history).
  - sibling specs continue.

  **a-4. This spec's success.** On `outcome == "ok"`:
  - set this spec's `implementer_status = "OK"`,
    `implementer_attempts = len(debug_attempts) + 1`.
  - write `.staging/iter_{n}/debug_log_<spec_id>.md`.
  - `cp $AW/graphs/{graph}.json
       .staging/iter_{n}/graph_<spec_id>.json` — convenience copy.
  - spec is added to `ready_for_eval`.

**After per-spec apply**: build `ready_for_eval` = (all surviving
`patched_to_eval` specs that hit a-4) + (all `no_patch_to_eval`
specs). If `ready_for_eval` is empty (every patched spec failed
apply and there are no no-patch specs), skip (b); DISTILL still
runs to record the apply-step failures as experience.

**(b) Multi-spec measured eval — Python wrapper**

For all `ready_for_eval` specs, invoke the multi-spec eval wrapper.
The wrapper (Python script at
`.claude/commands/architect/myloop/lib/multi_spec_eval.py`) decides
how to submit every (spec, pass_idx) combination as ONE parallel
wave to the backend's JobScheduler, then reassembles per-spec
results from the K eval runs.

```bash
python .claude/commands/architect/myloop/lib/multi_spec_eval.py run \
  --graph        {graph} \
  --version      v{N} \
  --iter         {n} \
  --spec-list    .staging/iter_{n}/spec.json \
  --eval-spec-ids "<comma-joined spec_ids from ready_for_eval>" \
  --staging      .staging/iter_{n}/ \
  --frozen-root  . \
  --admission    <inferred from graph name; e.g. mapgpt_mp3d → mapgpt-mp3d>
```

**Wrapper contract** (user-implemented; this skill specifies the
interface):

| Aspect | Contract |
|---|---|
| Input | the envelope at `--spec-list` + the per-spec overlays at `.staging/iter_n/active_workspace_<id>/` (already prepped by apply step) |
| Mix policy | **ONE parallel wave per iter** — all `(spec, pass_idx)` combinations across `ready_for_eval` submit together in a single wave (`total_submissions = sum over specs of spec.passes`); the JobScheduler arbitrates admission based on VRAM and queues what doesn't fit immediately. There are no internal "batches" — this is the literal sense of "合集的实验" (one combined experiment per iter, reassembled into K per-spec results afterward). |
| Worker cap | wrapper reads `perf_<graph>.worker_count` from experiment_design.yaml as the **method-wide max worker cap**. Per-submission `worker_count` is computed by binary-searching the smallest target wave count W such that `Σᵢ min(profile_wc_i, ⌈ep_i / W⌉) ≤ perf_cap`, then `wc_i = min(profile_wc_i, ⌈ep_i / W⌉)`. This minimizes wave wall clock (makespan-min, ep-weighted) under constant per-episode time — long-ep specs in a mixed wave get more workers so they don't hold up the wave. Example for `perf_<graph>.worker_count=P` with one perf-tier sub (ep=Ep, pcap=P) plus three custom-subset subs (ep=Es, pcap=Sp): all four hit the same target wave count W, with workers allocated proportional to ep counts. If `perf_<graph>` is absent (custom-only run), no cap applied — each submission keeps its profile worker_count. The override REPLACES whatever the spec's own profile or eval_profile.overrides set. See `lib/multi_spec_eval.py::allocate_workers` for the implementation + `lib/test_allocate_workers.py` for worked scenarios. |
| Per-spec output | `.staging/iter_n/eval_metadata_<spec_id>.json` — one file per spec, matching the `experiment` block of `IterRecord` (`schemas.md § 5`), with `run_ids[]` (length = passes), `metrics_digest` aggregated to `mean_sr / sd_sr / robust_sr / score / ...`, `per_ep_success[][]` (outer = passes, inner = eps), `outcome_class ∈ {ok, crash}`. The wrapper computes the aggregates; loop.md just reads the file. |
| Forensic log | `.staging/iter_n/multi_spec_eval_log.md` — wrapper's wave plan, worker allocation per submission, submission ids, per-pass timing. Always written, even on crash. |
| Exit codes | `0` = all `--eval-spec-ids` got a usable `eval_metadata_<id>.json` (one or more may have `outcome_class="crash"` internally — that's data, not infra failure); `1` = infra failure (e.g. backend unreachable, JobScheduler refused admission, wrapper bug). On exit 1, partial `eval_metadata_*` files MAY be present (whatever the wrapper managed to write). |

**On wrapper exit 1** (infra failure): treat as crash for every spec
in `ready_for_eval` that did NOT get an `eval_metadata_<id>.json`
written (synthesize a `crash` metadata file with
`stderr_tail` from the wrapper log). Continue to step 4 — DISTILL
still records the crash as evidence.

**On wrapper exit 0**: each `ready_for_eval` spec has its
`eval_metadata_<id>.json` on disk, ready for DISTILL.

**Baseline-lock side effect**: the wrapper also locks per-profile
baseline numbers when a profile is used at its `passes_required ≥ 3`
floor for the first time. For each profile encountered in
`ready_for_eval`:

  - if `profile.passes_required ≥ 3` AND `profile.baseline` is
    absent (or `null`) in `experiment_design.yaml` AND at least one
    spec on that profile this iter has `outcome_class == "ok"` and
    `passes ≥ passes_required`:

    The wrapper appends a `baseline:` block to that profile entry
    in `experiment_design.yaml`:
    ```yaml
    <profile_name>:
      ... existing fields ...
      baseline:
        mean_sr:   <from eval_metadata_<spec_id>.metrics_digest.mean_sr>
        sd_sr:     <from eval_metadata_<spec_id>.metrics_digest.sd_sr>
        robust_sr: <from eval_metadata_<spec_id>.metrics_digest.robust_sr>
        passes:    <profile.passes_required>
        run_ids:   <from eval_metadata_<spec_id>.run_ids[:passes_required]>
        locked_at: "iter_<n>"
        locked_ts: "<ISO timestamp>"
    ```

    If multiple specs in the same iter use the same profile (rare
    but possible for K-spec no-patch probes), the wrapper picks the
    FIRST spec (by spec_id order) as the baseline source. Once
    written, the baseline is immutable — future iters using the
    same profile must compare against THESE numbers.

  - if `profile.baseline` is already locked: the wrapper does NOT
    overwrite it. Subsequent specs on the same profile compare
    against the locked baseline in DISTILL (and CRITIC for
    pre-flight power check).

  - if `profile.passes_required < 3` (e.g. perf_<graph> with N=216):
    no baseline locking. Single-pass profiles are not subject to
    sd_sr tracking; comparisons use the parent iter's run.

This is an append-only mutation to `experiment_design.yaml`; the
wrapper preserves all other profile fields and comment headers.

**(b′) Register the iter_0 frozen baseline** — runs only when
`n == 0`, the envelope is K=1, the single spec's `patch == null`,
and its `eval_metadata_<spec_id>.outcome_class == "ok"`. Logic
unchanged from prior single-spec version (mv to archive_runs,
symlink, append to `baseline_registry.yaml`) — just adjust file
paths to use the spec_id-suffixed `eval_metadata_<spec_id>.json`
and use the spec's `eval_profile.name` from the envelope.

**(c) Per-spec experiment-block snapshot** — already done by the
wrapper. loop.md performs no additional file writes here; the
wrapper's `eval_metadata_<spec_id>.json` IS the snapshot DISTILL
reads at startup.

For any spec marked `critic_block` in 3b.5 OR `implementer_skip` in
a-3 (no `eval_metadata_<id>.json` from the wrapper), loop.md
synthesizes a minimal `eval_metadata_<id>.json` so DISTILL has a
uniform per-spec input:

```json
{
  "spec_kind":            "<from envelope.specs[spec_id].kind>",
  "patch_applied":        false,
  "implementer_status":   "N/A | SKIP_RUNTIME_FAIL",
  "implementer_attempts": 0 | retry_max,
  "passes":               0,
  "run_ids":              [],
  "artifacts_dirs":       [],
  "metrics_digest":       {},
  "per_ep_success":       [],
  "outcome_class":        "critic_block | implementer_skip",
  "baseline_cache_hit":   false
}
```

### 4. DISTILL — invoke `/architect:myloop:distill`

```
Invoke /architect:myloop:distill {graph} v{N} --iter {n}
  - reads .staging/iter_{n}/{spec.json (envelope), think_trace.md}
  - reads .staging/iter_{n}/eval_metadata_<spec_id>.json — ONE per spec in the envelope (synthesized for critic_block / implementer_skip specs)
  - reads .staging/iter_{n}/critique_<spec_id>.json + critique_<spec_id>_round_1.json (per patched spec; absent for no-patch specs)
  - reads outputs/eval_runs/<run_id>/episodes/ep*/{episode.json, log.jsonl} for every run_id across all specs (selectively)
  - reasons (ONE sub-agent spawn — prompt enumerates all K specs side-by-side)
  - eager-writes experience.jsonl, hypotheses.jsonl, knowledge.md
  - writes .staging/iter_{n}/distill_<spec_id>.json — ONE per spec (per-spec verdict + lessons)
  - returns: status = OK | SKIP_DISTILL_EMPTY | SKIP_INVALID_DISTILL
```

The single DISTILL spawn handles all K specs in one cognitive pass.
This is intentional — it lets DISTILL produce **cross-spec lessons**
("A confirmed mechanism X; B refuted mechanism Y on the same eps —
they don't interact"), not just per-spec verdicts. The output is K
per-spec `distill_<spec_id>.json` files (each conforming to the
`distill` block of an entry in `record.specs[]`), plus the
iter-level `iter_summary.milestone_after` baked into the sub-agent's
final summary.

DISTILL has one extra responsibility per spec when its
`critique_<spec_id>.json` is present: judge whether CRITIC's
`predicted_outcome` for each `predicted_failure_mode` materialized
in that spec's eval. For each prediction:

- If the predicted outcome MATERIALIZED: append an `experience.jsonl`
  entry tagged `critic_TP` (with the spec_id in the entry's metadata).
- If the predicted outcome did NOT materialize: append `critic_FP`.
- If this spec's `outcome_class = "critic_block"` (round-2 BLOCK
  without override, OR proposer SKIP on rebuttal — no eval ran for
  this spec): append `critic_unverified` — the block is recorded
  but accuracy is unknowable. Sibling specs that DID run may still
  produce critic_TP/critic_FP entries normally.

When this spec's `block_override` is present (THINK rebutted a BLOCK
and the spec went to EXPERIMENT), DISTILL evaluates whether the
override was justified per-spec — see distill.md § "critic
accuracy".

These critic-accuracy entries are what gives myloop its double-layer
learning: THINK learns about graph mechanisms; CRITIC learns about
itself.

Outcome handling:

- `OK`: continue to step 5 (commit) — each `distill_<spec_id>.json`
  will be merged into the corresponding `specs[*].distill` block in
  `record.json`.
- `SKIP_DISTILL_EMPTY`: continue to step 5 — every spec's `distill`
  block in `record.json` is OMITTED (genuinely nothing to distill,
  e.g. a no-op probe iter). Do not increment `consecutive_skips`.
- `SKIP_INVALID_DISTILL`: continue to step 5 — every spec's
  `distill` block is OMITTED. Increment `consecutive_skips` and
  log. Eager writes the DISTILL agent already made (if any) stay
  on disk; the next iter's THINK will reason about partial state.

### 5. Atomic commit (the iter writer)

Build `IterRecord` (per schemas.md § 5):

```python
iter_record = {
  "iter": n,
  "ts_start": THINK_STARTED_AT,
  "ts_end":   datetime.utcnow().isoformat(),
  "think": {
    "rationale":   extracted from .staging/iter_{n}/think_trace.md,
    "file_updates": diff-detect changes proposer made to
                    hypotheses/knowledge/experiment_design/tools this turn,
    "spec_ref":    "iteration/iter_{n}/spec.json"   # post-mv path; the envelope
  },
  # "reflect" block — present ONLY if REFLECT fired this iter (3a′
  # or 3b SKIP-escalation). Iter-level, shared across all specs.
  "reflect": <{trigger, status, frontier_axes, reflection_id}>  # OR omit

  # "specs" list — one entry per spec in the envelope. Each entry
  # carries that spec's per-spec critic / experiment / distill blocks.
  "specs": [
      {
        "spec_id":           envelope.specs[i].spec_id,
        "spec_kind":         envelope.specs[i].kind,
        "intervention_axis": envelope.specs[i].intervention_axis,
        "passes":            envelope.specs[i].passes,

        # "critic" block — present ONLY if CRITIC fired for THIS spec
        # (i.e. this spec had patch != null). Built from the FINAL
        # round's critique_<spec_id>.json:
        #   verdict, critic_round, predicted_failure_modes_count,
        #   reference_experience_ids (flattened across this spec's
        #   failure modes, deduplicated),
        #   block_override (= bool: this spec has block_override),
        #   critique_ref ("iteration/iter_{n}/critique_<spec_id>.json").
        "critic": <{...}>  # OR omit if this spec had patch=null

        "experiment": {
          "patch_applied":        bool(spec.patch) and this_spec_apply_OK,
          "implementer_status":   "OK | SKIP_RUNTIME_FAIL | N/A",
          "implementer_attempts": <from this spec's debug_log_<spec_id>.md>,
          "run_ids":              <from eval_metadata_<spec_id>.run_ids>,   # length = passes
          "artifacts_dirs":       <from eval_metadata_<spec_id>.artifacts_dirs>,
          "metrics_digest":       <from eval_metadata_<spec_id>.metrics_digest>,  # mean_sr / sd_sr / robust_sr / score / ...
          "per_ep_success":       <from eval_metadata_<spec_id>.per_ep_success>,   # [[pass1], [pass2], ...]
          "outcome_class":        "ok | crash | implementer_skip | critic_block",
          "baseline_cache_hit":   <bool — true iff iter_0 adopted a cached baseline via § 3b₀>
        },

        # "distill" block — merge from .staging/iter_{n}/distill_<spec_id>.json
        # if DISTILL returned OK. Omit on SKIP_DISTILL_EMPTY or
        # SKIP_INVALID_DISTILL.
        "distill": <contents of .staging/iter_{n}/distill_<spec_id>.json>  # OR omit
      }
      # ... one entry per spec in the envelope
  ],

  # iter_summary — the trace.md / lineage.md rollup. Computed at
  # commit time from specs[*] per schemas.md § 5 outcome_class rule.
  "iter_summary": {
    "outcome_class":   "confirmed | mixed | refuted | inert | critic_block | crash",
    "best_score":      max(specs[*].experiment.metrics_digest.mean_sr where outcome_class=ok),
    "best_spec_id":    <spec_id achieving best_score>,
    "axes_touched":    <unique set of specs[*].intervention_axis>,
    "milestone_after": <from DISTILL's iter-level summary or last spec's distill>,
    "n_specs":         len(specs),
    "n_confirmed":     <count of specs[*].distill.verdict == "confirmed">,
    "n_refuted":       <count of specs[*].distill.verdict == "refuted">,
    "n_critic_block":  <count of specs[*].experiment.outcome_class == "critic_block">,
    "n_crash":         <count of specs[*].experiment.outcome_class == "crash">,
  },

  "cost": {
    "gpu_min":    estimated_from_wall_time,
    "llm_tokens": null,
    "wall_sec":   ts_end - ts_start
  }
}
```

Atomic-commit transaction:

1. `mv .staging/iter_{n}/ → outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_{n}/`
   (this includes spec.json envelope, think_trace.md,
   reflection_trace.md if REFLECT fired, all per-spec
   `critique_<id>.json` / `critique_<id>_round_1.json` /
   `critique_trace_<id>.md`, all per-spec `eval_metadata_<id>.json`,
   all per-spec `distill_<id>.json` / `distill_trace.md`, all
   per-spec `active_workspace_<id>/` / `debug_log_<id>.md`, and the
   wrapper's `multi_spec_eval_log.md`).
2. Write `outputs/design_runs/myloop/{graph}/v{N}/iteration/iter_{n}/record.json`
   (the IterRecord above — each spec's `distill` block is the
   contents of its `distill_<spec_id>.json` if present, omitted
   otherwise).
3. **Regenerate the derived rollups.** Scan every
   `iteration/iter_*/record.json` (including the one just written) and
   rewrite `v{N}/trace.md` and `v{N}/lineage.md` wholesale from them
   (`schemas.md § 13 / § 14`). These are pure projections of the
   IterRecords — `record.json` is the per-iter source of truth, the
   rollups are regenerable from it at any time. This step is therefore
   **NOT part of the atomic-rollback set**: if it fails, log a warning
   and proceed — the next commit (or step 6) regenerates them.
4. (eager-writes already done by THINK + DISTILL — knowledge.md,
   hypotheses.jsonl, experience.jsonl, experiment_design.yaml,
   tools/*.py. No additional commit needed.)
5. Update `.loop_state/last_committed_iter = n`.

If any of (1)/(2)/(5) fails: rollback (rm -rf the iter dir if mv
succeeded; do not write `record.json`). Increment
`consecutive_skips`. The eager-writes during THINK stay — they are
already in vN and represent partial work that DISTILL of a future iter
can reconcile (orchestrator reasons about consistency). Step (3) is
derived and never triggers a rollback.

```
[myloop:loop] iter_{n} committed
              K specs     = {len(specs)} ({comma-joined spec_ids})
              axes        = {comma-joined unique axes}
              best        = {best_spec_id} → {primary_metric}={best_score:.3f}
              outcome     = {iter_summary.outcome_class}
              per-spec    = A:ok B:critic_block C:crash  (etc.)
              consecutive_skips reset.
```

### 6. Termination summary

```
=== /architect:myloop:loop summary ===
graph           = {graph}
version         = v{N}
pipeline        = myloop
iters committed = {M_end}
status          = GOAL_MET | SATURATED | STUCK | BUDGET | CAPPED | USER_STOP
                  (SATURATED = a REFLECT verdict of SPACE_EXHAUSTED,
                   or the goal.md saturated poll)

key file counts:
  knowledge.md  = {N_sections} sections, {N_bullets} bullets
  search_space  = {N_axes} axes, {N_refl} reflections; final frontier: {frontier or "— exhausted"}
  experience    = {K_exp} entries
  hypotheses    = {H_open} open (started with {H_start})
  tools         = {T} authored
  experiment_design = {D} profiles ({D_user_authored} added during run)

cumulative cost:
  gpu_min       = {sum}
  wall_sec      = {sum}
  llm_tokens    = {sum_or_unknown}

primary_metric trajectory (per iter): see trace.md (one row per iter)
```

Write this summary to `outputs/design_runs/myloop/{graph}/v{N}/SUMMARY.md`
so resuming sessions / external readers can grep it. Then regenerate
`trace.md` + `lineage.md` one final time (`schemas.md § 13 / § 14`) —
this covers a run that resumed and terminated without committing a
fresh iter, whose rollups would otherwise be one iter stale.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/goal.md` | **user** (pre-flight) | Mission spec — direction, escalation, termination |
| `v{N}/knowledge.md` | proposer + distill (eager) | Pure facts, append-only |
| `v{N}/search_space.md` | bootstrap + reflect (eager) | Intervention-space map + per-REFLECT coverage, append-only |
| `v{N}/experience.jsonl` | distill (eager) | Lessons learned |
| `v{N}/hypotheses.jsonl` | proposer + distill (eager) | Open conjectures, mutable |
| `v{N}/experiment_design.yaml` | bootstrap + proposer | Eval-profile registry |
| `v{N}/tools/*.py` | proposer (eager) | Orchestrator-authored utilities |
| `v{N}/iteration/iter_{n}/` | loop (atomic mv) | Committed iter evidence (spec.json envelope + think_trace + reflection_trace + per-spec critique_<id> / critique_trace_<id> / eval_metadata_<id> / distill_<id> / active_workspace_<id>/ / debug_log_<id> + distill_trace + multi_spec_eval_log) |
| `v{N}/iteration/iter_{n}/record.json` | loop | Dense IterRecord with `specs[]` per-spec blocks + `iter_summary` rollup (incl. optional `reflect` block) |
| `v{N}/iteration/iter_{n}/critique_<spec_id>.json` | critic | One per spec with `patch != null` (final round); round-1 preserved as `critique_<spec_id>_round_1.json` if round 2 ran for that spec |
| `v{N}/SUMMARY.md` | loop (at termination) | Run summary |
| `v{N}/trace.md` | loop (each commit + termination) | One-row-per-iter axis/metric/outcome history table — regenerated from record.json (schemas.md § 13) |
| `v{N}/lineage.md` | loop (each commit + termination) | One-section-per-iter narrative — regenerated from record.json (schemas.md § 14) |
| `v{N}/.staging/iter_{n}/` | proposer / reflect / loop (EXPERIMENT) / distill | Transient — promoted on success |
| `v{N}/.loop_state/` | loop | Bookkeeping (last_committed_iter, last_reflection_iter, consecutive_skips) |

## Notes

- **No archive.jsonl.** myloop has none — working memory is the
  eleven vN-scoped files. Anyone wanting `archive`-style flat history
  can read `iteration/iter_*/record.json` in order (each has spec +
  metrics + distill, and a `reflect` block on iters where REFLECT
  fired), or scan the `trace.md` / `lineage.md` rollups for a
  human-readable digest of the same.
- **`trace.md` / `lineage.md` are derived rollups.** Regenerated from
  `iteration/iter_*/record.json` at every atomic commit (§5 step 3)
  and at termination (§6). `record.json` is the per-iter source of
  truth; the rollups are a human-scannable projection of it and can be
  rebuilt at any time — they never gate a commit.
- **REFLECT is a meta-phase, not a per-iter phase.** It fires on
  heartbeat / axis-concentration / SKIP-escalation (§ 3a′, 3b), at
  most once per iter. Its job is search-space cartography — it
  maintains `search_space.md` and hands THINK a ranked frontier so
  THINK does not rut in one `intervention_axis`. A `SATURATED`
  termination means REFLECT returned `SPACE_EXHAUSTED` (it surveyed
  the whole space) — `SKIP_THINK_EMPTY` alone never terminates.
- **CRITIC is a per-spec phase (only when that spec's patch != null).**
  Distinct from REFLECT: CRITIC vets ONE candidate spec at a time
  against accumulated refuted experience; REFLECT audits the whole
  intervention space. CRITIC fires at most twice per spec (round 1 +
  at most one round 2 on REVISE/BLOCK), bounding the per-iter cost
  overhead at ~2K sub-agent spawns where K is the number of patched
  specs. A `critic_block` outcome on a spec (CRITIC round-2 BLOCK
  without `spec.block_override`, or proposer SKIP on rebuttal) means
  that spec did not run measured eval; sibling specs in the same
  iter still proceed. The iter still has critique files for blocked
  specs and a DISTILL pass that records each block as experience.
  This is the "save a wasted eval" case per spec — distinct from
  `crash` (eval ran and failed) or `implementer_skip` (apply step
  failed). The double-layer learning (THINK learns mechanisms,
  CRITIC learns itself via `critic_TP` / `critic_FP` tags in
  experience.jsonl) is what makes the cost worthwhile.
- **DISTILL is its own phase inside the same iter** (distill.md).
  Running DISTILL right after EXPERIMENT — not folded into the next
  iter's THINK — keeps each sub-agent's cognitive task focused:
  THINK plans the next experiment (and may read raw logs to target
  its episode set); DISTILL digests the warm fresh outcome of the
  iter just run (spec + expected signal + per-ep traces all current).
  Base cost: two sub-agent spawns per iter (THINK + DISTILL); CRITIC
  adds 1–2 more on patched iters (§ 3b.5), so a patched iter with no
  rebuttal is 3 spawns, a rebuttal iter 4. REFLECT, when triggered,
  adds one more (§ 3a′).
- **Hard contract**: every committed iter has a `think` block AND
  at least one entry in `specs[]` (K ≥ 1). A spec with
  `experiment.outcome_class = "implementer_skip"` or `"critic_block"`
  is still a valid (failed) experiment for contract purposes — the
  iter committed, the apply step or CRITIC paid for the lesson.
- **No revert chain.** SKIPs from proposer or the EXPERIMENT
  apply-step don't rollback prior file edits — they leave them in place. The
  orchestrator's next THINK will see them and reason about whether
  to retract (e.g. delete a stillborn hypothesis it just appended).
- **Compactable at any phase boundary** — re-invoke
  `/architect:myloop:loop` with the same args to resume; resolves
  current iter from `iteration/` count and `.loop_state/`.
- **No vN bump from inside the loop.** Major pivots are manual.
  Per `_common/files-contract.md` write-skill version protection.
- **iter_0 baseline cache (§ 3b₀ / 3c-(b′)).** `iter_0` is always a
  K=1 no-patch frozen-workspace baseline, identical across vN
  bootstraps and across methods. The shared
  `outputs/design_runs/baseline_registry.yaml` records completed
  frozen runs keyed by an authored-graph content fingerprint. A HIT
  lets `iter_0` adopt a cached run — real bytes in
  `outputs/archive_runs/{run_id}/`, a relative symlink left in
  `outputs/eval_runs/{run_id}/` — and skip THINK + the eval; DISTILL
  still runs to digest the corpus into this vN. A MISS runs `iter_0`
  normally (still K=1) and registers the result. Multi-spec breadth
  (K>1) is a feature of iter_1+; iter_0 stays K=1 by construction.
  The cache covers ONLY the `iter_0` eval.

- **Python multi-spec eval wrapper.** Lives at
  `.claude/commands/architect/myloop/lib/multi_spec_eval.py`. Its
  contract is documented in § 3c (b). It owns single-wave submission
  (all K specs × passes go in ONE parallel wave) and per-submission
  worker-count allocation under the `perf_<graph>.worker_count` cap
  so loop.md stays agnostic — the loop hands it a list of
  `--eval-spec-ids` and reads back per-spec `eval_metadata_<id>.json`.
  The wave plan and worker allocation are not visible to THINK /
  CRITIC / DISTILL — they always see aggregated `metrics_digest` +
  the full `per_ep_success[][]` matrix.
