# Orchestrate adas-subagent cycles

> **Required reading before invoking**:
> - `.claude/commands/architect/adas-subagent/README.md` — adas-subagent's 3 core
>   contracts (one-iter-one-conversation, archive.jsonl independent,
>   two-tier evaluation)
> - `.claude/commands/architect/_common/files-contract.md` — shared
>   run-dir layout, resolve protocol, edit whitelist

This skill is the **orchestrator**. It owns the per-iter Claude
conversation and drives 3 worker skills in sequence: `proposer` →
`implementer` → `evaluator`. The proposer's `msg_list` persists across
its own 3 Reflexion calls (this is what makes adas-subagent
ADAS-faithful at the propose layer); implementer's retries spawn
*fresh* sub-agents (no msg_list continuation — pure coding-agent
debug); evaluator is pure infrastructure.

> **Library + data** (see README § "Library + data"):
> - Pre-seed step 3b appends `data/reference_seeds.json` (7 ADAS
>   reference patterns) to a fresh `archive.jsonl`.
> - Atomic Writer (step 5) calls `lib/helpers.py:atomic_commit`,
>   `render_graph_summary`, `update_trace_md`, `append_lineage_md`.
> - Pre-seed iter_0 baseline appends via `append_reference_seed`.
> Import: `sys.path.insert(0, ".claude/commands/architect/adas-subagent/lib")` then
> `from helpers import ...`.

## Arguments

```
/architect:adas-subagent:loop [<graph> [<version>]]
                       [--graph <name>] [--version <N>]
                       [--max-iters N]            default 5
                       [--max-consecutive-skips K] default 3
                       [--from-iter M]            default auto-resolve
                       [--skip-understand]        skip P0
                       [--skip-preseed]           skip iter_0 baseline + reference seed injection
                       [--allow-old-version]
```

Resolve protocol: see files-contract § "Resolve protocol". Graph fuzzy
source: `outputs/design_runs/*/`.

## Pre-conditions

- `workspace/graphs/{graph}.json` exists.
- `workspace/architect/exp_profiles/{graph}.yaml` exists with **two profiles** (or
  auto-bootstrap will create them with conservative defaults):
  - `smoke_<graph>`: `episode_count: 5`, `worker_count: 1`,
    `episode_indices: [0, 25, 50, 75, 99]` (deterministic),
    `per_step_budget_sec` lowered if desired
  - `perf_<graph>`: `episode_count: 100`, `worker_count: 4–8`
- If resuming: `outputs/design_runs/adas-subagent/{graph}/v{N}/archive.jsonl` exists
  and is valid JSONL (one entry per line, last line parseable).

## Steps

### 1. Resolve graph + version + entry iter

Apply the **Resolve protocol** per files-contract. Print:

```
RUN_DIR=outputs/design_runs/adas-subagent/{graph}/v{N}
  graph         = {graph}
  version       = {N}
  pipeline      = adas-subagent (coding-agent-era ADAS port + sub-agent Reflexion)
  entry iter    = iter_{M}  (resume) | iter_0  (fresh)
  archive       = {RUN_DIR}/archive.jsonl  ({K} entries)
  cap           = max-iters={max_iters}, max-consecutive-skips={K}
```

### 2. P0 — Auto-understand

Unless `--skip-understand`, invoke
`/architect:adas-subagent:understand <graph> <vN> --for loop` once. Reuses
the adas-subagent understand skill. (Skip when context is already loaded in
the current Claude session — re-running is cheap but redundant.)

### 3. Pre-seed (only if archive.jsonl is empty)

If `archive.jsonl` does not exist or is empty:

**3a. Baseline iter_0**

If `iter_0/` does not exist:
- Snapshot current `workspace/{graphs,nodesets}/*` into staging.
- Invoke `/architect:adas-subagent:evaluator` on the baseline (no proposer/
  implementer involvement).
- Commit `iter_0/` via Atomic Writer (step 5 below).
- Append archive entry:
  ```json
  {
    "generation": "initial",
    "iter_id": "iter_0",
    "name": "<user-provided baseline name or graph filename>",
    "thought": "Baseline graph as provided by the user — design starting point.",
    "graph_summary": <rendered from workspace/graphs/{graph}.json>,
    "diff_narrative": "Baseline (no parent).",
    "fitness": "<from evaluator>"
  }
  ```

**3b. Reference seed injection**

Append 7 reference pattern entries to `archive.jsonl` (text-only,
`fitness: null`, `iter_id: null`, `generation: "reference"`). The 7
patterns are the ADAS-direct port:
1. Chain-of-Thought (COT)
2. Self-Consistency with CoT (COT_SC)
3. Self-Refine (Reflexion)
4. LLM Debate
5. Step-back Abstraction
6. Quality-Diversity
7. Dynamic Assignment of Roles

Content for each: `name`, `thought` (verbatim from
`third_party/ADAS/_mmlu/mmlu_prompt.py` seed dicts,
adapted from "code" descriptions), `graph_summary` =
`"(reference pattern, no graph implementation provided — meta-LLM
should adapt the pattern to AgentCanvas)"`, `diff_narrative` = 1-line
plain-text description of the pattern.

(Q4 in HTML open questions — whether to keep ADAS-direct, swap to
VLN-specific, or hybrid — is still open. Default: ADAS-direct, mirrors
the paradigm-independent structure of upstream's initial archive.)

If `--skip-preseed`: skip step 3 entirely. Caller is responsible for
having a valid `archive.jsonl` already.

### 4. Per-iter state machine

For `n in range(start_iter, start_iter + max_iters)`:

```
Print: ============ adas-subagent generation n ============

# Sub-step 4a: proposer (same Claude conversation continues)
Invoke /architect:adas-subagent:proposer
  - Builds analyze view (Python helper inside proposer skill)
  - 3 LLM calls (propose / Reflexion_1 / Reflexion_2)
  - Writes .staging/iter_n/proposal.md
  - On any LLM exception → signals SKIP, no staging written
  - Returns: status = OK | SKIP_LLM_EXCEPTION

If SKIP_LLM_EXCEPTION:
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  if consecutive_skips >= max_consecutive_skips: terminate (STUCK)
  continue to n+1

# Sub-step 4b: implementer (method-free — pure coding-agent debug)
Invoke /architect:adas-subagent:implementer
  - Reads .staging/iter_n/proposal.md (`# Change` spec: {intent, targets})
  - For attempt in range(retry_max=config):
      seed targets + spawn editing sub-agent (native Edit/Write) →
      Smoke eval (5 ep) → classify by RUNTIME CORRECTNESS ONLY:
        PASS = exit=0 AND all eps completed AND step_count>0 AND valid numeric metric
        FAIL = crash | incomplete | step=0 | malformed_metric | edit_error
      if PASS: break (success, overlay holds the edited state)
      else: reset active_workspace to parent, spawn FRESH editing
            sub-agent with proposal + failure trace as context
  - Writes .staging/iter_n/debug_log.md
  - Low/zero metric values do NOT trigger retry — that's archive data
  - On retry_max-exhausted → reverts workspace, signals SKIP
  - Returns: status = OK | SKIP_RUNTIME_FAIL

If SKIP_RUNTIME_FAIL:
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  if consecutive_skips >= max_consecutive_skips: terminate (STUCK)
  continue to n+1

# Sub-step 4c: evaluator (same conversation)
Invoke /architect:adas-subagent:evaluator
  - Runs 100-ep eval via experiment:run on perf_<graph>
  - Writes .staging/iter_n/{metrics.json (neutral schema: run_id,
                            episode_count, acc_list, primary_metric,
                            primary_metric_value, secondary_metrics),
                            summary.csv, export.json}
  - Does NOT compute fitness_str — bootstrap_CI lives in step 5 below.
  - No retry on low value distribution.
  - Returns: status = OK | EVAL_INFRA_FAILURE

If EVAL_INFRA_FAILURE (eval API errored, NOT low score):
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  continue to n+1
# Low fitness is NOT a failure here — we still commit, archive learns.

# Sub-step 4d: Atomic Writer (loop's own responsibility)
See step 5.

consecutive_skips = 0   # reset on any successful commit
```

### 5. Atomic Writer (commit step)

On worker success, perform these 5 actions **as one transaction**:

1. **Enrich `metrics.json` with `fitness_str`** (this is where the
   variant's method knowledge lives — the evaluator is method-free):
   ```python
   import sys, json
   sys.path.insert(0, ".claude/commands/architect/adas-subagent/lib")
   from helpers import bootstrap_confidence_interval

   staging_metrics = ".staging/iter_{n}/metrics.json"
   metrics = json.load(open(staging_metrics))
   fitness_str, median = bootstrap_confidence_interval(
       metrics["acc_list"], num_bootstrap_samples=100000,
       confidence_level=0.95
   )
   metrics["fitness_str"] = fitness_str
   json.dump(metrics, open(staging_metrics, "w"))
   ```
   `bootstrap_confidence_interval` is verbatim upstream ADAS
   (`utils.py:31–76`). Do not modify the `fitness_str` literal format —
   `[ARCHIVE]` injection and the meta-LLM prompt match the pattern
   verbatim.
2. `mv .staging/iter_{n}/ → outputs/design_runs/adas-subagent/{graph}/v{N}/iteration/iter_{n}/`
3. Render `graph_summary` from the iter's effective graph at
   `iter_n/active_workspace/graphs/{graph}.json` (the implementer
   wrote the patched state here; frozen workspace was never touched).
4. Append one JSONL line to
   `outputs/design_runs/adas-subagent/{graph}/v{N}/archive.jsonl`:
   ```json
   {
     "generation": n,
     "iter_id": "iter_n",
     "name": "<from proposal.md frontmatter>",
     "thought": "<from proposal.md, verbatim>",
     "graph_summary": <rendered>,
     "diff_narrative": "<from proposal.md What-changed section>",
     "fitness": "<from staging/metrics.json.fitness_str>"
   }
   ```
   **Strip** `reflection` before writing — it's R1/R2 scaffolding and
   must not pollute archive (verbatim upstream `search.py:232–235`).
   `debug_thought` is no longer in the schema (implementer's retry
   sub-agent edits the overlay natively and returns
   `{edit_summary, extra_targets}`, not a Reflexion-style debug field).
5. Update `trace.md` (new row) and append a section to `lineage.md`.

If any step fails: abort, do NOT leave the iter dir partially written
(rm -rf the mv'd dir if mv succeeded; do not append archive line).
This is the only place archive.jsonl is written; failure = no entry =
SKIP semantics.

### 6. Termination

Terminate the outer loop when any holds:
- `n >= start_iter + max_iters` — CAPPED
- File `{RUN_DIR}/.loop_state/STOP` exists — USER_STOP
- `consecutive_skips >= max_consecutive_skips` — STUCK

Print summary:

```
=== /architect:adas-subagent:loop summary ===
graph          = {graph}
version        = v{N}
pipeline       = adas-subagent
iters run      = {M_end} - {M_start}
archive size   = {K} entries  ({K_initial} initial + {K_reference} reference + {K_evolved} evolved)
consecutive    = {consecutive_skips}
final metric   = {primary_metric} from archive head = {value}
                 (vs baseline {baseline_value}, Δ {delta})
status         = COMPLETED | STUCK | CAPPED | USER_STOP
```

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/archive.jsonl` | loop (this skill, step 5) | One JSONL line per committed iter; meta-LLM's working memory |
| `v{N}/iteration/iter_{n}/` | loop (Atomic Writer step) | Committed evidence: graph, snapshot, proposal.md, debug_log.md, metrics.json, summary.csv, export.json |
| `v{N}/trace.md` | loop | One row per committed iter |
| `v{N}/lineage.md` | loop | Narrative section per committed iter |
| `v{N}/.staging/iter_{n}/` | proposer / implementer / evaluator | **Transient** — pre-commit working dir; mv to iter_{n}/ on success, rm -rf on SKIP |
| `v{N}/.loop_state/` | loop | Bookkeeping (consecutive_skips counter, etc.) |

## Notes

- **Never bumps `vN`.** Major pivots are manual via
  `--new-version` flag on a fresh experiment invocation. (Per
  files-contract write-skill version protection.)
- **No revert chain.** SKIPs ARE the rollback. Workspace reverts
  inside implementer's retry loop; on exhaustion the implementer
  leaves workspace at last-known-good archive head state. loop
  bookkeeping only tracks `consecutive_skips` for stuck detection.
- **archive.jsonl is per-vN.** A new vN starts a fresh archive — meta-LLM
  doesn't see prior-version failures (clean slate for major pivot).
- **Compactable at any phase boundary** — re-invoke
  `/architect:adas-subagent:loop` with the same args to resume; resolves
  current iter from `wc -l archive.jsonl`.
- **Cleanup on partial failure**: any time a worker signals SKIP or
  an exception bubbles up, `rm -rf .staging/iter_{n}/` to ensure
  consistent state. The Atomic Writer is the ONLY commit point.
