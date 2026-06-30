# Orchestrate aflow cycles

> **Required reading before invoking**:
> - `.claude/commands/architect/aflow/README.md` — aflow's 3 core
>   contracts (one-iter-one-conversation with anti-replay,
>   archive.jsonl with `parent_iter_id` + `modification` + `score`,
>   two-tier evaluation: `smoke_` in implementer + `perf_` every iter
>   as the ranking signal, with F3 deviation)
> - `.claude/commands/architect/_common/files-contract.md` — shared
>   run-dir layout, resolve protocol, edit whitelist
> - Upstream reference:
>   - `third_party/AFlow/scripts/optimizer.py:80–198`
>     (`optimize` outer loop + `_optimize_graph` inner)

This skill is the **orchestrator**. It owns the per-iter Claude
conversation and drives proposer → **Workspace Checkout** →
implementer → evaluator → Atomic Writer → convergence check. The
`msg_list` (Claude's context) persists across all phases.

## Arguments

```
/architect:aflow:loop [<graph> [<version>]]
                      [--graph <name>] [--version <N>]
                      [--max-iters N]              default 20 (mirrors upstream max_rounds)
                      [--max-consecutive-skips K]  default 3
                      [--from-iter M]              default auto-resolve
                      [--skip-understand]          skip P0
                      [--skip-preseed]             skip iter_0 baseline
                      [--allow-old-version]
```

Resolve protocol: see files-contract § "Resolve protocol". Graph
fuzzy source: `outputs/design_runs/*/`.

## Pre-conditions

- `workspace/graphs/{graph}.json` exists.
- `workspace/architect/exp_profiles/{graph}.yaml` exists with **three blocks**:
  - `smoke_<graph>` (small profile for implementer's debug retry)
  - `perf_<graph>` (full paper-comparable set — runs EVERY iter as the
    ranking signal that `select_round`'s softmax consumes; also reused
    in step 8 for top-1/2 verification reruns)
  - `aflow:` block (K_sample, α, λ, replay_cap, replay_norm,
    convergence_z, top_k, consecutive)
  Auto-bootstrap fills missing entries with conservative defaults.
  (The pre-2026-05-25 three-tier design with a frozen `search_<graph>`
  subset has been retired — see README § "Two-tier evaluation" for the
  v0 mapgpt_mp3d failure that motivated this.)
- If resuming: `outputs/design_runs/aflow/{graph}/v{N}/archive.jsonl`
  exists and is valid JSONL.

## Steps

### 1. Resolve graph + version + entry iter

Apply the **Resolve protocol** per files-contract. Print:

```
RUN_DIR=outputs/design_runs/aflow/{graph}/v{N}
  graph         = {graph}
  version       = {N}
  pipeline      = aflow (AFlow port — softmax parent + per-parent anti-replay)
  entry iter    = iter_{M}  (resume) | iter_0  (fresh)
  archive       = {RUN_DIR}/archive.jsonl  ({K} entries)
  aflow cfg     = sample={K}, α={α}, λ={λ}, replay_cap={cap}, replay_norm={norm}, conv_z={z}
  cap           = max-iters={max_iters}, max-consecutive-skips={K}
```

### 2. P0 — Auto-understand

Unless `--skip-understand`, invoke
`/architect:aflow:understand <graph> <vN> --for loop` once.

### 3. Pre-seed (only if archive.jsonl is empty)

**No 7-seed palette in aflow** — verbatim upstream, only the
baseline is pre-seeded.

If `archive.jsonl` does not exist or is empty:

**3a. Baseline iter_0**

If `iter_0/` does not exist:
- Snapshot current `workspace/{graphs,nodesets}/*` into staging.
- Invoke `/architect:aflow:evaluator --mode baseline` on the
  baseline (no proposer/implementer involvement).
- Commit `iter_0/` via Atomic Writer (step 5).
- Append archive entry:
  ```json
  {
    "generation": "initial",
    "iter_id": "iter_0",
    "parent_iter_id": null,
    "name": "<user-provided baseline name or graph filename>",
    "thought": "Baseline graph as provided by the user — design starting point.",
    "modification": "(baseline)",
    "graph_summary": <rendered from workspace/graphs/{graph}.json>,
    "diff_narrative": "Baseline (no parent).",
    "fitness": "<from evaluator>",
    "score": <bootstrap_median in [0,1] as float>
  }
  ```

`modification` is set to the sentinel string `"(baseline)"` — never
matched by `check_modification`.

If `--skip-preseed`: skip step 3 entirely. Caller is responsible for
having a valid `archive.jsonl` already (must include at least one
entry with non-null `score`).

### 4. Per-iter state machine

For `n in range(start_iter, start_iter + max_iters)`:

```
Print: ============ aflow generation n ============

# Sub-step 4a: proposer (same Claude conversation continues)
Invoke /architect:aflow:proposer
  - Reads archive.jsonl
  - Runs anti-replay retry loop (bounded by aflow.replay_max_retries):
      * select_round → softmax-sample parent from top-K
      * load parent's experience (success + failure modifications)
      * build optimize_prompt (parent's graph + prompt + log×3 + experience)
      * single LLM call (or sub-agent spawn — see proposer.md § Sub-agent contract)
      * check_modification(response.modification, parent.experience)
        - if duplicate: continue (resample parent)
        - if FormatError after retry: continue
      * break on accepted response
  - Writes .staging/iter_n/proposal.md with frontmatter:
      parent_iter_id, modification, name, thought, patch
  - On replay_cap exceeded → signals SKIP
  - On any sub-agent exception → signals SKIP
  - Returns: status = OK | SKIP_REPLAY_EXHAUSTED | SKIP_LLM_EXCEPTION

If SKIP_*:
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  if consecutive_skips >= max_consecutive_skips: terminate (STUCK)
  continue to n+1

# Sub-step 4b: Workspace Checkout (LOOP's own responsibility — NEW vs adas-subagent)
parent_iter_id = read proposal.md frontmatter
PARENT_AW="outputs/design_runs/aflow/{graph}/v{N}/iteration/iter_{parent_iter_id}/active_workspace"

mkdir -p .staging/iter_{n}/active_workspace
if [ -d "$PARENT_AW" ]; then
    cp -r "$PARENT_AW"/. .staging/iter_{n}/active_workspace/
fi
# If parent is iter_0 (no active_workspace because baseline didn't mutate
# anything): start empty, implementer will seed-from-frozen on first touch.

Print: [aflow:loop] Workspace Checkout — parent=iter_{parent_iter_id} → .staging/iter_{n}/active_workspace/

# Sub-step 4c: implementer (method-free — pure coding-agent debug)
Invoke /architect:aflow:implementer
  - Detects pre-populated .staging/iter_{n}/active_workspace/ (loop
    just did the parent checkout) → skips its own bootstrap step
  - Reads .staging/iter_n/proposal.md (`# Change` spec: {intent, targets})
  - For attempt in range(retry_max=config):
      seed targets + spawn editing sub-agent (native edit) → Smoke eval
      (5 ep) → classify by RUNTIME CORRECTNESS ONLY:
        PASS = exit=0 AND all eps completed AND step_count>0 AND valid numeric metric
        FAIL = crash | incomplete | step=0 | malformed_metric | edit_error
      if PASS: break
      else: reset .staging/iter_n/active_workspace/ to parent's checkout,
            spawn FRESH editing sub-agent with proposal + failure trace
            (native Edit/Write)
  - Writes .staging/iter_n/debug_log.md
  - Low/zero metric values do NOT trigger retry — that's archive data
  - On retry_max-exhausted → SKIP
  - Returns: status = OK | SKIP_RUNTIME_FAIL

If SKIP_RUNTIME_FAIL:
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  if consecutive_skips >= max_consecutive_skips: terminate (STUCK)
  continue to n+1

# Sub-step 4d: evaluator (same conversation)
Invoke /architect:aflow:evaluator --mode iter
  - Runs the perf_<graph> eval (full paper-comparable set, e.g. 216 ep
    for mapgpt_mp3d) via experiment:run — profile_key=perf_<graph>
    from config.yaml. Every iter pays a perf eval; this is the cost
    aflow now wears to keep the per-iter ranking signal sound (post
    2026-05-25, see README § "Two-tier evaluation").
  - Writes .staging/iter_n/{metrics.json (neutral schema: run_id,
                            episode_count, acc_list, primary_metric,
                            primary_metric_value, secondary_metrics),
                            summary.csv, export.json}
  - Does NOT compute fitness_str / score — bootstrap_CI lives in
    step 5 below (method knowledge belongs to loop, not evaluator).
  - No retry on low value distribution.
  - Returns: status = OK | EVAL_INFRA_FAILURE

If EVAL_INFRA_FAILURE:
  rm -rf .staging/iter_n/
  consecutive_skips += 1
  continue to n+1
# Low fitness is NOT a failure — we still commit; archive learns.

# Sub-step 4e: Atomic Writer (loop's own responsibility) — step 5

# Sub-step 4f: Convergence check (loop's own responsibility) — step 6

consecutive_skips = 0   # reset on any successful commit
```

### 5. Atomic Writer (commit step)

On worker success, perform these 6 actions **as one transaction**:

1. **Enrich `metrics.json` with `fitness_str` + bare numeric `score`**
   (this is where the variant's method knowledge lives — the evaluator
   is method-free):
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
   metrics["score"]       = float(median)   # bare bootstrap median in [0,1] for select_round softmax
   json.dump(metrics, open(staging_metrics, "w"))
   ```
   `bootstrap_confidence_interval` is verbatim upstream ADAS
   (`utils.py:31–76`); aflow inherits it here because of the F3
   deviation (validation_rounds=1 + resample-within-set instead of
   upstream's 5-pass averaging). Do not modify the `fitness_str`
   literal format.
2. `mv .staging/iter_{n}/ → outputs/design_runs/aflow/{graph}/v{N}/iteration/iter_{n}/`
3. Render `graph_summary` from the iter's effective graph at
   `iter_n/active_workspace/graphs/{graph}.json`.
4. Read `proposal.md` frontmatter for `parent_iter_id`, `modification`,
   `name`, `thought`, and the patch's `diff_narrative` section.
5. Read `metrics.json` for `fitness_str` and `score` (just enriched
   in step 1).
6. Append one JSONL line to
   `outputs/design_runs/aflow/{graph}/v{N}/archive.jsonl`:
   ```json
   {
     "generation": n,
     "iter_id": "iter_n",
     "parent_iter_id": "iter_{parent}",
     "name": "<from proposal.md frontmatter>",
     "thought": "<from proposal.md, verbatim>",
     "modification": "<from proposal.md frontmatter, verbatim>",
     "graph_summary": <rendered>,
     "diff_narrative": "<from proposal.md What-changed section>",
     "fitness": "<from staging/metrics.json.fitness_str>",
     "score": <from staging/metrics.json.score>
   }
   ```
   No `debug_thought` / `reflection` stripping needed — neither field
   exists in the new contract. aflow's proposer never emitted
   `reflection` (no Reflexion chain), and implementer's retry sub-agent
   edits the overlay natively and returns `{edit_summary, extra_targets}`
   rather than a Reflexion-style `debug_thought` field.
7. Update `trace.md` (new row, with `parent_iter_id` column) and
   append a section to `lineage.md`.

If any step fails: abort, rm -rf the mv'd dir if mv succeeded; do
not append archive line.

### 6. Convergence check (after Atomic Writer)

Run `check_convergence(top_k=conv_top_k, z=conv_z, consecutive_rounds=conv_consecutive)`
against the updated `archive.jsonl`. Verbatim from upstream
`convergence_utils.py:68–113`:

```python
def check_convergence(archive, top_k, z, consecutive_rounds):
    """
    Faithful port of upstream convergence_utils.py:68-113.
    Each archive entry contributes exactly ONE `score` (aflow runs
    validation_rounds=1, so a "round" has a single score and its
    per-round std is 0).
    Returns (converged: bool, conv_start: int|None, conv_end: int|None).
    """
    import numpy as np
    avg_scores = [e["score"] for e in archive]   # one score per iter
    stds       = [0.0] * len(avg_scores)         # validation_rounds=1 -> std 0
    if len(avg_scores) < top_k + 1:
        return False, None, None
    convergence_count = 0
    previous_y = sigma_y_previous = None
    for i in range(len(avg_scores)):
        idx = np.argsort(avg_scores[:i + 1])[::-1][:top_k]   # top-k so far
        top_k_scores = [avg_scores[j] for j in idx]
        top_k_stds   = [stds[j] for j in idx]
        y_current = float(np.mean(top_k_scores))
        sigma_y_current = float(np.sqrt(sum(s**2 for s in top_k_stds) / (top_k**2)))
        if previous_y is not None:
            delta_y = y_current - previous_y
            sigma_delta_y = float(np.sqrt(sigma_y_current**2 + sigma_y_previous**2))
            if abs(delta_y) <= z * sigma_delta_y:
                convergence_count += 1
                if convergence_count >= consecutive_rounds:
                    return True, i - consecutive_rounds + 1, i
            else:
                convergence_count = 0
        previous_y, sigma_y_previous = y_current, sigma_y_current
    return False, None, None
```

**The `z` tunable is inert under F3.** Upstream's per-round σ is the
std-dev across `validation_rounds=5` passes; aflow runs one pass
(see README § "Three structural contracts" / F3), so every per-round
std is 0, `sigma_delta_y` is always 0, and the predicate
`abs(delta_y) <= z·sigma_delta_y` reduces to `delta_y == 0` **for any
value of `z`**. Convergence therefore fires only on *exact* equality
of the top-k mean across `consecutive_rounds` consecutive rounds — a
near-no-op on continuous metrics. This is faithful to running
upstream's algorithm at `validation_rounds=1`; it also means
`aflow.convergence_z` in `{graph}.yaml` has no effect unless
multi-pass eval is reintroduced. Keep convergence as an advisory
terminator — `max_iters` / `consecutive_skips` / the STOP file are the
real stopping conditions.

If converged: terminate the outer loop with status `CONVERGED`.
Print the convergence trace (which `top_k` means matched across
which rounds).

### 7. Termination

Terminate the outer loop when any holds:
- `n >= start_iter + max_iters` — CAPPED
- File `{RUN_DIR}/.loop_state/STOP` exists — USER_STOP
- `consecutive_skips >= max_consecutive_skips` — STUCK
- `check_convergence` returned True — CONVERGED

Print the loop summary (every `score` in the archive is already a
`perf_<graph>` value — step 8 only adds verification reruns on top):

```
=== /architect:aflow:loop summary ===
graph          = {graph}
version        = v{N}
pipeline       = aflow
iters run      = {M_end} - {M_start}
archive size   = {K} entries  ({K_initial} initial + {K_evolved} evolved)
consecutive    = {consecutive_skips}
best perf      = {primary_metric} from archive top = {value}  (iter_{best})
                 (vs baseline {baseline_value}, Δ {delta})  [perf_<graph>, in-loop single pass]
parent dist    = iter_X: {n_children}, iter_Y: {n_children}, ...
                 (how select_round distributed children across top-K parents)
status         = COMPLETED | STUCK | CAPPED | USER_STOP | CONVERGED
```

### 8. Verification reruns (top-1 + top-2 each get one extra perf_)

In-loop `perf_<graph>` scores already use the full paper-comparable set,
so the headline number does NOT need a separate post-loop perf eval.
But every in-loop score is a SINGLE pass under `validation_rounds=1`,
so LLM run-to-run stochasticity is unmodeled (see README § F3 deviation).
The top iter could be a lucky pass. Verification reruns calibrate this.

1. Sort archive by `score` descending. Take the top **two** evolved
   iters (skip `iter_0`; if archive has <2 evolved iters, take whatever
   exists).
2. For each top iter, run **one** additional `perf_<graph>` eval:
   - `cp -r .../iteration/iter_{best}/active_workspace/. .staging/final_report_best_rerun/active_workspace/`
     (if the iter is `iter_0`, omit `--workspace` — pure frozen.)
   - `/architect:aflow:evaluator --mode iter --iter final_report_best_rerun --profile-key perf_<graph>`
   - Same for top-2 → `final_report_top2_rerun`.
3. Run one verification rerun on `iter_0` baseline as well
   (`final_report_base_rerun`), so the Δ calibration uses paired reruns.
4. Enrich each rerun `metrics.json` with `fitness_str` + bootstrap median
   (same helper as Atomic Writer step 1).
5. Write `v{N}/final_report.md` with both passes for each iter:
   ```
   iter_{best}:
     in-loop perf   = {score_in_loop}  fitness {fitness_in_loop}
     rerun perf     = {score_rerun}    fitness {fitness_rerun}
     mean (2 pass)  = {mean}           Δ-vs-baseline-mean = {delta}
   ```
   Flag any iter whose two passes differ by > 5pp as "high LLM
   stochasticity — single-pass ranking unreliable here".
6. Print:
   ```
   === aflow final report ===
   graph         = {graph}  v{N}
   best iter     = iter_{best}
     in-loop SR  = {primary_metric} {s_loop}    fitness {f_loop}
     rerun SR    = {primary_metric} {s_rerun}   fitness {f_rerun}
     mean SR     = {s_mean}
   top-2 iter    = iter_{top2}  (same 3 lines)
   baseline      = iter_0   (in-loop + rerun + mean — only mean if baseline ran twice)
   Δ (mean−base) = {delta}
   report        = v{N}/final_report.md
   ```

The verification reruns are NOT appended to `archive.jsonl`. The
archive captures the search trajectory under one consistent statistical
regime (one perf pass per iter); mixing in second passes would distort
`select_round`'s softmax ranking if the loop were resumed. The reruns
live in `final_report.md` only.

Total step-8 cost: **3 extra perf runs** (top-1, top-2, baseline) — vs
the old design's 1–2 perf runs. ~15% more wall, but the rerun
discipline is what the v0 mapgpt failure showed we need.

## Outputs

| Path | Writer | What |
|------|--------|------|
| `v{N}/archive.jsonl` | loop (step 5) | One JSONL line per committed iter; meta-LLM's working memory; carries `parent_iter_id` + `modification` + `score` |
| `v{N}/iteration/iter_{n}/` | loop (Atomic Writer) | Committed evidence: graph, snapshot, proposal.md (with parent_iter_id frontmatter), debug_log.md, metrics.json (with score), summary.csv, export.json |
| `v{N}/trace.md` | loop | One row per committed iter (includes `parent_iter_id` column) |
| `v{N}/lineage.md` | loop | Narrative section per committed iter |
| `v{N}/final_report.md` | loop (step 8) | Verification reruns of top-1 + top-2 + baseline (one extra `perf_<graph>` pass each) — calibrates in-loop single-pass scores against LLM run-to-run stochasticity. Kept out of archive.jsonl. |
| `v{N}/.staging/iter_{n}/` | proposer / implementer / evaluator + loop's Workspace Checkout step | **Transient** — pre-commit working dir |
| `v{N}/.staging/iter_{n}/active_workspace/` | **loop's Workspace Checkout step (NEW)** + implementer's patch application | Bootstrapped from softmax-sampled parent's `iter_{parent}/active_workspace/`, NOT from archive head |
| `v{N}/.loop_state/` | loop | Bookkeeping |

## Notes

- **Workspace Checkout is loop's responsibility, NOT implementer's**
  — this is the cleanest contract: parent selection happens inside
  proposer (it's part of the anti-replay loop), proposer writes
  `parent_iter_id` to proposal.md frontmatter, loop reads it and
  does the `cp`, then invokes implementer. Implementer's existing
  bootstrap step (adas-subagent's `PARENT_AW=.../iter_{parent}`)
  becomes a no-op when it detects `.staging/iter_n/active_workspace/`
  is already populated. **No fork of adas-subagent's implementer needed**
  beyond a 5-line conditional at the top of step 2.
- **No archive-head assumption.** Every iter, parent is freshly
  softmax-sampled inside proposer. Live `workspace/` stays at the
  user's pre-loop state (advisory) — backend reads from
  `.staging/iter_n/active_workspace/` via the `--workspace` flag
  during eval (see implementer step 3b for the flag).
- **Never bumps `vN`.** Major pivots are manual.
- **archive.jsonl is per-vN.** A new vN starts a fresh archive —
  meta-LLM doesn't see prior-version failures, AND prior `parent_iter_id`
  references don't dangle.
- **Resume.** Re-invoke `/architect:aflow:loop` with the same args;
  resolves current iter from `wc -l archive.jsonl`. Anti-replay
  works correctly on resume because all parent experience is in
  archive.jsonl (no per-iter `experience.json` files to reconcile).
- **Cleanup on partial failure**: any time a worker signals SKIP,
  `rm -rf .staging/iter_{n}/` ensures consistent state. The Atomic
  Writer is the ONLY commit point.
- **Parent distribution diagnostic** in the summary helps catch a
  failure mode where softmax collapses onto one parent (typically
  iter_0 if its score is much higher than evolved iters' early
  attempts). If you see one parent dominating, consider raising
  `aflow.lambda_uniform` toward 0.5+ for more exploration.
