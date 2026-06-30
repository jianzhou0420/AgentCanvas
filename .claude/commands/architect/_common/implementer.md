# Realize the proposed change + smoke-test ({VARIANT})

> **Common implementer skill.** Invoked indirectly by the
> `adas-subagent` / `aflow` `implementer.md` stub (e.g.
> `/architect:adas-subagent:implementer`). The stub binds `VARIANT`,
> `VARIANT_DIR`, and points to this file. (`myloop` does NOT use this
> skill — its apply step is inline in `myloop/loop.md § 3c`.)
> All variant-specific values (smoke profile key, retry cap,
> side-effect artifacts) are read from `<VARIANT_DIR>/config.yaml`
> § `implementer:`.
>
> **Scope**: assumes sub-agent-spawn retry executor (each attempt =
> one independent `Agent({subagent_type: "general-purpose"})`).
> Variants with a different retry executor keep their own
> variant-specific `implementer.md`.
>
> **Required reading**:
> - `<VARIANT_DIR>/README.md`
> - `_common/files-contract.md` § 7 "Edit whitelist", § 3 / § 8
>   (`active_workspace` overlay)

This skill turns a proposer's **intended change** into real edits under
`.staging/iter_n/active_workspace/`, runs a smoke eval, and retries with
a fresh sub-agent if the change doesn't execute cleanly.

**Native editing — no patch DSL.** A proposer hands over a *change
spec* (a prose `intent` + a `targets` file list), not a typed op list.
The implementer spawns a tool-augmented sub-agent that edits the
seeded overlay files directly with native Edit/Write. The old typed
`graph_edits` op-applier was retired 2026-05-20: an agentic implementer
edits files itself, so serialising intent into an op enum for a
deterministic replayer was an ADAS-era vestige (see
`files-contract.md` § 7). The only deterministic helper kept is
`_common/lib/overlay.py` — it seeds frozen files into the overlay and
enforces the § 7 whitelist; it does not edit.

**It is method-free.** PASS means "the change runs end-to-end on the
smoke profile without crashing, stalling, or producing malformed
outputs." The smoke metric *values* are deliberately not consulted —
low or zero scores are not retry triggers. Low scores are real data
for archive / proposer consumption, not failure signals here.

`retry_max = 3` by default; override via `implementer.retry_max` in
config.

## Arguments

```
/architect:{VARIANT}:implementer [<graph> [<version> [<iter>]]]
                                 [--graph <name>] [--version <N>] [--iter <M>]
                                 [--retry-max N]              default from config.retry_max
```

## Pre-conditions

- The proposer's change spec exists for `iter_n` — `proposal.md`
  (`adas-subagent`, `aflow`), in `.staging/iter_n/`.
- `workspace/{graphs,nodesets}/*` is at last-known-good state
  (parent iter's head — frozen is never mutated regardless).
- Smoke profile (`config.smoke_profile_key`, typically
  `smoke_<graph>`) exists in `workspace/architect/exp_profiles/{graph}.yaml`.

## Steps

### 1. Resolve + read the change spec

```
[{VARIANT}:implementer] iter=iter_{n}  reading .staging/iter_{n}/<change-spec>
```

Read the proposer's deliverable for `iter_n`:

- `adas-subagent` / `aflow`: `.staging/iter_n/proposal.md` — the
  `# Change` section + frontmatter.

Extract two things:

- **`intent`** — the prose description of what to change and why.
- **`targets`** — a list of workspace-prefixed paths the change is
  expected to touch (e.g. `workspace/graphs/{graph}.json`,
  `workspace/nodesets/navgpt.py`). May be empty.

If the change spec carries **no change** (a proposal with an empty
`# Change` / no targets), this is a **no-patch probe**: skip Step 3, leave `active_workspace/` as the
parent copy (Step 2's bootstrap still runs to seed it), and jump to
Step 4 with `outcome = "ok"`, `n_attempts = 0`.

### 2. Bootstrap iter_{n}/active_workspace/ from parent

Frozen `<repo>/workspace/` is NEVER modified. The iter's mutation set
lives at `.staging/iter_{n}/active_workspace/` (promoted to
`iter_{n}/active_workspace/` on success by the loop's Atomic Writer).

```bash
mkdir -p .staging/iter_{n}/active_workspace/{graphs,nodesets}

# If the loop already populated active_workspace (e.g. aflow's
# softmax-sampled parent checkout), skip the copy. Otherwise copy
# from the parent iter's active_workspace.
if [ -z "$(ls -A .staging/iter_{n}/active_workspace 2>/dev/null)" ]; then
    PARENT_AW=outputs/design_runs/{method}/{graph}/v{N}/iteration/iter_{parent}/active_workspace
    if [ -d "$PARENT_AW" ]; then
        cp -r "$PARENT_AW"/. .staging/iter_{n}/active_workspace/
    fi
fi
```

The parent iter's `active_workspace/` is the implicit revert anchor —
on a failed attempt we reset `.staging/iter_{n}/active_workspace/` by
re-copying from `$PARENT_AW`.

### 3. Edit-and-smoke retry loop (up to `retry_max` attempts)

Every attempt is one independent editing sub-agent — attempt 0 and the
retries are the same mechanism (this is the native-editing
simplification: no separate deterministic apply step).

For `attempt in range(retry_max)`:

#### 3a. Seed targets + spawn the editing sub-agent

**Seed** every declared target into the overlay (deterministic; also
runs the § 7 whitelist check):

```bash
python .claude/commands/architect/_common/lib/overlay.py prepare \
  --active-ws .staging/iter_{n}/active_workspace \
  --frozen-root . --graph {graph} \
  <each target from the change spec>
```

If `overlay.py` exits non-zero, a target hit the § 7 **hard wall**
(`agentcanvas/backend/app/**` / `third_party/**` — no overlay exists
there). Classify `outcome = "edit_error"`, skip 3b/3c, jump to 3f.
`[off-scope WARN]` lines are non-blocking — surface them, continue.

**Spawn** the editing sub-agent — an independent Claude sample with
full tool access that edits the seeded overlay files directly:

```python
prompt = f"""\
You are the implementer in an architecture-search iteration. A
proposer designed a change to an agent graph; your job is to make that
change real by editing files, then return a short summary.

## Change intent

{intent}

## Files to edit (already seeded into the overlay)

{for t in targets:}  - .staging/iter_{n}/active_workspace/{t without "workspace/"}
{end for}

## Rules (files-contract.md § 7)

- Edit ONLY under `.staging/iter_{n}/active_workspace/{{graphs,nodesets}}/`
  with native Edit/Write. NEVER touch frozen `workspace/`,
  `agentcanvas/backend/app/**`, or `third_party/**`.
- If realizing the intent needs a file not in the seeded list (e.g. a
  transitively-imported nodeset), seed it FIRST with:
    python .claude/commands/architect/_common/lib/overlay.py prepare \\
      --active-ws .staging/iter_{n}/active_workspace --frozen-root . \\
      --graph {graph} workspace/nodesets/<that-nodeset>...
  then edit the seeded copy.
- A graph JSON is config: keep node ids unique, keep edges consistent
  with the nodes they reference, generate stable edge ids. After
  editing a `.json` it MUST still `json.load`; after editing a `.py`
  it MUST still `ast.parse`.
{if attempt > 0:}
## This is retry {attempt} of {retry_max - 1}

A prior attempt's edits failed the {episode_count}-episode smoke eval.
Do NOT change the strategic intent — only fix what stops it running.

### Failed attempts so far
{for prev in debug_attempts:}
- attempt {prev.attempt + 1}: outcome={prev.outcome} — {prev.err_summary}
  edits: {prev.edit_summary}
{end for}

### Most recent failure
- outcome: {outcome}
- err_summary: {err_summary}
- evidence: per-episode logs `outputs/eval_runs/{smoke_run_id}/episodes/`,
  backend admit log `outputs/eval_runs/{smoke_run_id}/admit.log`
- the overlay has been reset to the parent state — re-apply your edits.
{end if}

Use Read / Grep / Bash freely. Return a SINGLE fenced ```json block
with keys:
  - `edit_summary`: 1-3 sentences — which files you changed and how.
  - `extra_targets`: list of any workspace-prefixed paths you seeded
    beyond the declared targets (or []).
"""

resp = Agent({
    "subagent_type": "general-purpose",
    "description":   f"{VARIANT} implement iter_{n} attempt={attempt+1}",
    "prompt":        prompt,
})
edit_result = parse_final_json(resp)   # {edit_summary, extra_targets}
```

**Post-edit validation** — after the sub-agent returns, validate every
edited overlay file (the implementer does this, not the sub-agent):

```python
import ast, json
for f in changed_files_under(".staging/iter_{n}/active_workspace"):
    if f.endswith(".json"):
        try: json.load(open(f))
        except Exception as e: outcome = "edit_error"; err_summary = f"{f}: {e}"
    elif f.endswith(".py"):
        try: ast.parse(open(f).read())
        except SyntaxError as e: outcome = "edit_error"; err_summary = f"{f}: {e}"
```

On `edit_error`: skip 3b/3c, jump to 3f.

#### 3a-pin. Pin every `llmCall` node to the enforced profile

Architecture search lets the editing sub-agent (driven by the
proposer's `intent`) add `llmCall` nodes with any `profile` /
`temperature`. The architect enforces a **single LLM** — after
post-edit validation, deterministically rewrite every `llmCall` node in
the iter's overlay graphs:

```bash
python .claude/commands/architect/_common/lib/pin_llm_profile.py pin \
  --active-ws .staging/iter_{n}/active_workspace
```

This sets `config.profile = gpt-5-mini`, `config.temperature = 1`, and
drops the dead `config.model` field on every `llmCall` node, regardless
of what the proposer asked for. It is non-blocking (never an
`edit_error`) and idempotent. It runs **before Smoke** so the smoke
eval exercises the pinned configuration. The proposer's `intent` may
still *name* a different model — the pin silently overrides it;
`gpt-5-mini` is multimodal and the gpt-5 family requires
`temperature=1` (see the script docstring for the rationale and the
coupled-constants note).

#### 3b. Run artifact hook: `after_edit`

```python
run_artifact_hook("after_edit", VARIANT_DIR, state={
    "edit_summary": edit_result["edit_summary"],
    "edit_diff":    render_diff(active_ws_before, active_ws_after),
}, staging_dir=".staging/iter_{n}")
```

#### 3c. Smoke eval

Every eval parameter comes from the `<smoke_profile_key>` block in
`{graph}.yaml` — the implementer transcribes the profile's OWN values
onto the CLI and hardcodes NO count. (A hardcoded count that disagrees
with the profile makes Step 3d's `profile.episode_count` check
misfire.)

```bash
/experiment:run <smoke_profile_key> {graph} \
  --workspace={absolute_path}/.staging/iter_{n}/active_workspace \
  episode_count=<profile.episode_count> worker_count=<profile.worker_count> \
  step_budget=<profile.step_budget> per_step_budget_sec=<profile.per_step_budget_sec> \
  split=<profile.split>
```

If the `<smoke_profile_key>` block pins `episode_indices` instead of
an `episode_count`, forward `episode_indices=...` instead. Backend
overlays `active_workspace` on frozen at eval time via the
`--workspace` flag.

Capture: per-episode metric values (`acc_list`), per-episode
`step_count`, the `experiment:run` exit code, and any backend error
in the admit log. The metric *values* are recorded forensically only —
they do NOT influence the PASS/FAIL classification below.

#### 3d. Classify — runtime correctness ONLY

PASS iff ALL of the following hold:

1. `experiment:run` exit code == 0 (no backend / harness error).
2. Episode count returned by `/api/eval/v2/export` ==
   `profile.episode_count` (no episode crashed or timed out).
3. For each episode: `step_count > 0` (agent took at least one
   action — covers the "wire/port mismatch never fires" failure).
4. For each episode: `metrics[primary_metric]` is a valid number
   (not `None`, not `NaN`, not a string).

```python
import math

if exit_code != 0 or backend_error_in_admit_log:
    outcome = "crash"
    err_summary = backend_error_tail or stderr_tail
elif len(export["episodes"]) < profile["episode_count"]:
    outcome = "incomplete"
    err_summary = (
        f"Only {len(export['episodes'])} of {profile['episode_count']} "
        f"episodes completed — likely a mid-run crash or timeout"
    )
elif any(ep["step_count"] == 0 for ep in export["episodes"]):
    outcome = "step=0"
    err_summary = (
        "Agent emitted zero steps in at least one episode — "
        "likely wire/port mismatch, the controller never fires"
    )
elif any(
    ep["metrics"].get(primary_metric) is None
    or not isinstance(ep["metrics"][primary_metric], (int, float))
    or math.isnan(ep["metrics"][primary_metric])
    for ep in export["episodes"]
):
    outcome = "malformed_metric"
    err_summary = (
        f"At least one episode returned non-numeric / missing "
        f"`{primary_metric}` — env or scorer is broken"
    )
else:
    outcome = "ok"   # success — exit retry loop
    break
```

**Low or zero metric values do NOT trigger retry.** `mean(acc_list) = 0.0`
on a clean run is data, not failure — it enters archive as a valid
signal that the change didn't work, and the next proposer can learn
from that. The implementer's only judgement is whether the change
*runs*.

#### 3e. Run artifact hook: `after_each_smoke_attempt`

```python
run_artifact_hook("after_each_smoke_attempt", VARIANT_DIR, state={
    "attempt_i":         attempt,
    "outcome":           outcome,      # ok, crash, incomplete, step=0, malformed_metric, edit_error
    "smoke_acc_list":    acc_list,     # forensic only
    "smoke_run_id":      smoke_run_id, # null when outcome == "edit_error"
    "subagent_return":   edit_result,
    "err_summary":       err_summary if outcome != "ok" else None,
}, staging_dir=".staging/iter_{n}")
```

(Used by verbose variants to dump per-attempt sub-agent returns.)

#### 3f. Failure path → reset overlay + next attempt

If `outcome != "ok"`:

1. **Reset active_workspace to parent state** so the next retry starts
   from a known-good baseline, not the broken intermediate:

   ```bash
   rm -rf .staging/iter_{n}/active_workspace
   mkdir -p .staging/iter_{n}/active_workspace
   if [ -d "$PARENT_AW" ]; then
       cp -r "$PARENT_AW"/. .staging/iter_{n}/active_workspace/
   fi
   ```

2. **Append a per-attempt entry to in-memory `debug_attempts[]`**
   (serialized to `debug_log.md` at the end):

   ```python
   debug_attempts.append({
       "attempt":        attempt,
       "outcome":        outcome,
       "err_summary":    err_summary,
       "smoke_acc_list": acc_list,
       "edit_summary":   edit_result["edit_summary"] if edit_result else None,
   })
   ```

3. Continue to the next attempt — 3a re-seeds the targets and spawns a
   fresh editing sub-agent. The retry sub-agent is an independent
   Claude sample; it inherits the same `intent` plus the failure
   context (no `msg_list` continuation, no shared KV cache). It is a
   pure debugging agent — fix what prevents execution, do not redesign.

#### 3g. Exhaustion → SKIP_RUNTIME_FAIL

If the loop completes all `retry_max` attempts without
`outcome == "ok"`:

```
print(f"[{VARIANT}:implementer] SKIP_RUNTIME_FAIL — change didn't "
      f"run cleanly after {retry_max} attempts")
return status=SKIP_RUNTIME_FAIL
```

Loop's response is variant-defined; typically `rm -rf .staging/iter_n/`
and `consecutive_skips += 1`. Frozen workspace was never touched.
Any meta-reasoning the loop wants to do about the failure (e.g.
surfacing the failure trace to the next iter's proposer) is the loop's
own decision, not implementer's.

### 4. Success path — finalize staging

On `outcome == "ok"`:

1. `.staging/iter_{n}/active_workspace/` is the changed state — keep
   as-is. Frozen workspace is untouched.

2. Write `debug_log.md` to staging:

   ```markdown
   ---
   iter_id: iter_{n}
   attempts: {len(debug_attempts) + 1}
   final_outcome: ok
   smoke_acc_list: {final acc_list}
   final_mean: {mean}
   ---

   # Smoke result (final attempt)
   acc_list = {acc_list}
   mean = {mean}
   per-step counts = {counts}

   # Change
   {edit_result.edit_summary}

   # Retry history
   {for each entry: ## Attempt {n+1}: {outcome}, err_summary, edit_summary}
   ```

3. Convenience top-level copies for analyze/report tooling:

   ```bash
   cp .staging/iter_{n}/active_workspace/graphs/{graph}.json .staging/iter_{n}/graph.json 2>/dev/null || true
   cp workspace/architect/exp_profiles/{graph}.yaml          .staging/iter_{n}/{graph}.yaml
   ```

   (The `graph.json` copy is skipped for a no-patch probe that left the
   graph frozen — analyze/report fall back to frozen per
   `files-contract.md` § 1.)

4. Run artifact hook `after_success`:

   ```python
   run_artifact_hook("after_success", VARIANT_DIR, state={
       "final_acc_list": acc_list,
       "debug_attempts": debug_attempts,
       "n_attempts":     len(debug_attempts) + 1,
   }, staging_dir=".staging/iter_{n}")
   ```

### 5. Return to loop

```
[{VARIANT}:implementer] OK — change applied + smoke passed
                  attempts        = {n_attempts}
                  final smoke mean = {mean}    # forensic only
                  staging         = .staging/iter_{n}/
                  → evaluator next
```

Return `status=OK` to loop.

## Outputs

| Path | What |
|------|------|
| `v{N}/.staging/iter_{n}/debug_log.md` | Smoke result + per-attempt retry history |
| `v{N}/.staging/iter_{n}/graph.json` | Convenience copy of `active_workspace/graphs/{graph}.json` (if the graph was edited) |
| `v{N}/.staging/iter_{n}/{graph}.yaml` | Convenience copy of frozen `workspace/architect/exp_profiles/{graph}.yaml` |
| `v{N}/.staging/iter_{n}/active_workspace/` | The iter's complete mutation set vs frozen |
| Plus any `implementer.artifacts` declared by variant config | (e.g. per-attempt sub-agent returns) |
| `workspace/{graphs,nodesets}/*` | **Frozen — untouched.** Evaluator overlays active_workspace on top |

## Notes

- **Native editing.** The implementer sub-agent edits overlay files
  with Edit/Write. `_common/lib/overlay.py` only seeds files into the
  overlay and runs the § 7 whitelist — it never edits. There is no
  patch DSL.
- **LLM profile pin (step 3a-pin).** After editing and before Smoke,
  `_common/lib/pin_llm_profile.py` rewrites every `llmCall` node in the
  overlay graphs to a single enforced profile (`gpt-5-mini`,
  `temperature=1`). This is a hard, deterministic pin and the sole
  authority on the profile — the proposer prompt no longer names one.
  Architect-wide (all variants share this skill).
- **Method-free PASS criterion.** The implementer only judges runtime
  correctness (exit-clean, all eps completed, step>0, valid numeric
  metric). Score values are not consulted.
- **Each attempt = independent Claude sample** with full tool access.
  No msg_list continuation, no shared KV cache. Retry sub-agents are
  pure debugging agents inheriting the proposal `intent` as read-only
  context.
- **Overlay reset on every failed attempt** to the parent's
  `active_workspace/` — avoids cross-attempt contamination from a
  broken intermediate state.
- **`workspace/*` frozen.** All writes land under
  `.staging/iter_{n}/active_workspace/`.
- **Edit whitelist (`files-contract.md` § 7).** Hard wall: edits
  outside `workspace/` (`agentcanvas/backend/app/**`, `third_party/**`)
  are blocked by `overlay.py` → `edit_error` → retry. Soft scope: a
  graph / nodeset outside the iter graph's used set only prints an
  `[off-scope WARN]`, never blocks (transitive deps are legitimately
  off-prefix).
- **Server-mode nodesets** (`nodesets/server/**`) are editable as of
  TODO #60 (2026-05-15). `parallelism="shared"` server nodesets get
  auto-spawned as ephemeral auto_host children at eval admit time when
  overlay content hashes differently from frozen. First iter touching
  one pays ~30–60 s spawn cost; VRAM doubles for the eval's duration.
- **SKIP_RUNTIME_FAIL signals**: "this change can't be made to run in
  `retry_max` attempts." Loop decides what to do (typically drop the
  iter, no archive append, consecutive_skips++). Any meta-reasoning
  about *why* it failed is the loop's job — implementer reports facts,
  not strategy.

## Variant config schema (this skill)

```yaml
# <variant>/config.yaml
implementer:
  smoke_profile_key: smoke_<graph>
  retry_executor:    subagent_spawn               # only option in common
  retry_max:         3                            # config knob; default 3
  artifacts: []                                   # file-level side-effects
```

No `patch_applier` field — patch application was retired 2026-05-20.
The seed helper is the fixed shared path
`.claude/commands/architect/_common/lib/overlay.py`.
