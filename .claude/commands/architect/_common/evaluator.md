# Eval runner ({VARIANT})

> **Common evaluator skill.** Invoked indirectly by each variant's
> `evaluator.md` stub (e.g. `/architect:adas-subagent:evaluator`). The
> stub binds `VARIANT`, `VARIANT_DIR`, and points to this file. All
> variant-specific values (profile key, side-effect artifacts) are read
> from `<VARIANT_DIR>/config.yaml § evaluator:`.
>
> **Required reading**:
> - `_common/files-contract.md` § "Per-graph experiment profile"

The evaluator runs one eval against the iter's current workspace state,
collects per-episode `primary_metric` values, and writes a neutral
`metrics.json` to staging. It has **no method knowledge** — no
retry policy, no fitness-string format, no archive shape, no scoring
heuristic. Anything method-specific (bootstrap CI, fitness strings,
softmax scores, archive enrichment) is the variant's loop /
Atomic Writer / lib, never this skill.

## Arguments

```
/architect:{VARIANT}:evaluator [<graph> [<version> [<iter>]]]
                               [--graph <name>] [--version <N>] [--iter <M>]
                               [--mode iter | baseline]      default: iter (loop infers)
                               [--profile-key <key>]         default: config.evaluator.profile_key
```

`--mode baseline` is for the pre-seed call (iter_0): runs the eval
against the current workspace state without expecting prior
proposer/implementer work. `--mode iter` is the normal per-iter call.

`--profile-key` overrides which `{graph}.yaml` block the eval uses.
Default is `config.evaluator.profile_key`.

## Pre-conditions

- `workspace/{graphs,nodesets}/*` is at the state to be evaluated:
  - `--mode iter`: implementer just succeeded; workspace is patched
  - `--mode baseline`: workspace is the user's baseline graph
- The profile (key from `config.yaml § evaluator.profile_key`) exists
  in `workspace/architect/exp_profiles/{graph}.yaml` (or auto-bootstrap
  creates with conservative defaults).
- Eval API reachable at `BACKEND_URL` from `/experiment:run` admission.

## Steps

### 1. Resolve

`<profile_key>` = `--profile-key` if passed, else `config.evaluator.profile_key`.

```
[{VARIANT}:evaluator] iter=iter_{n}  mode={iter|baseline}  profile=<profile_key>
```

For `--mode iter`: ensure `.staging/iter_{n}/` exists (implementer
just created it). For `--mode baseline`: create `.staging/iter_0/`
if needed.

### 2. Run eval

`/experiment:run` is graph-only post 2026-05-07 — the legacy
`<profile> -- <command>` form is gone, and the wrapper does NOT read
`{graph}.yaml`. The eval-side profile block is passed via
`--eval-overrides` — a JSON file merged into the eval block. This is
mandatory (not an optimisation): `submit.py`'s `key=value` parser
handles scalars only, so list params (`episode_indices`,
`episode_selectors`) can ONLY reach the backend through this file.

Write the resolved profile block to a temp JSON, then submit:

```python
import json, yaml
prof  = yaml.safe_load(open(f"workspace/architect/exp_profiles/{graph}.yaml"))
block = prof["<profile_key>"]          # <profile_key> resolved in step 1
                                       # e.g. perf_mapgpt_mp3d
overrides_path = f".staging/iter_{n}/eval_overrides.json"
json.dump(block, open(overrides_path, "w"))
```

```bash
/experiment:run <admission_profile> {graph} \
    --eval-overrides=<absolute_path>/.staging/iter_{n}/eval_overrides.json
```

The profile block carries the full eval-block payload — `episode_count`,
`worker_count`, `step_budget`, `per_step_budget_sec`, `split`, and (when
present) `episode_indices` / `episode_selectors`. No `key=value` pairs
are needed; pass them only to override a single field ad hoc.

`<admission_profile>` is the entry in `.claude/commands/experiment/profiles.yaml`
(e.g. `mapgpt-mp3d`), distinct from the eval-side `<profile_key>` in
`workspace/architect/exp_profiles/{graph}.yaml` (e.g. `perf_mapgpt_mp3d`).

`primary_metric` is read separately (step 3) — it is a flat field of
`{graph}.yaml`, not part of the per-tier eval block.

For `--mode iter` add `--workspace=<absolute_path>/.staging/iter_{n}/active_workspace`
so the eval subprocess overlays the iter's mutations on frozen. For
`--mode baseline` (iter_0) omit `--workspace=` so the run uses pure
frozen workspace.

Capture the `run_id` from `/experiment:run` output (printed to
stdout by `submit.py`). The submit script polls
`/api/eval/v2/runs/{run_id}` until terminal and exits non-zero on
failure.

### 3. Collect per-episode values

When the run completes (or fails — see step 6):

- Query `{BACKEND_URL}/api/eval/v2/export?run_id={run_id}` → `export.json`
- Build `acc_list`:
  ```python
  primary_metric = profile["primary_metric"]
  acc_list = [ep["metrics"][primary_metric] for ep in export["episodes"]]
  assert len(acc_list) == profile["episode_count"]
  ```

  `acc_list` is the upstream-convention call-site name preserved
  across the architect codebase. Values are per-episode `primary_metric`
  reads, not necessarily 0/1 — continuous metrics in [0, 1] (SPL,
  nDTW, …) flow through unchanged.

### 4. Write staging metrics

```python
mean_acc = float(np.mean(acc_list))

metrics = {
    "run_id":                run_id,
    "episode_count":         len(acc_list),
    "acc_list":              acc_list,
    "primary_metric":        primary_metric,
    "primary_metric_value":  mean_acc,
    "secondary_metrics":     <aggregated profile.secondary_metrics dict>,
}

write_json(".staging/iter_{n}/metrics.json", metrics)

# Standard files-contract artifacts
cp /tmp/summary.csv  .staging/iter_{n}/summary.csv
cp /tmp/export.json  .staging/iter_{n}/export.json
echo "$run_id" > .staging/iter_{n}/eval_run_id.txt
```

The schema above is the entire evaluator output contract. Anything
beyond these fields (fitness strings, bare scores, archive keys, CI
intervals) is the loop's responsibility to add at Atomic-Writer time.

### 5. Variant artifact hooks (file-level extras)

After step 4, run the `after_staging` artifact hook for this skill:

```python
state = {
    "backend_export_full": export,       # full /api/eval/v2/export payload
    "acc_list":            acc_list,
    "mean_acc":            mean_acc,
    "run_id":              run_id,
    "staging_dir":         Path(".staging/iter_{n}"),
}
run_artifact_hook("after_staging", VARIANT_DIR, state, ".staging/iter_{n}")
```

`run_artifact_hook` reads `config.yaml § evaluator.artifacts` (a list
of `{hook, write, from}` entries), pulls the named object from `state`
at that hook, and writes a file under `.staging/iter_{n}/<write>`.
Variants with no `artifacts` declared = no-op.

(Hook implementation: see `_common/lib/hooks.py` once it exists;
until then, the variant config's `artifacts:` list is a forward-looking
declaration — actual writes happen when an implementation ships. The
contract is documented; the variant config declares intent.)

### 6. Infra failure handling

If `experiment:run` returns non-zero exit, eval API errors, or the
run never reaches `episode_count` completion (e.g., backend crash):

```
print("[{VARIANT}:evaluator] EVAL_INFRA_FAILURE — backend or harness error")
print("  experiment:run exit  = $exit_code")
print("  episodes completed   = $n_completed / $n_expected")
print("  last error           = $last_error_from_backend_log")
return status=EVAL_INFRA_FAILURE
```

Loop decides the SKIP bookkeeping; evaluator only reports the signal.

### 7. Return to loop

```
[{VARIANT}:evaluator] OK
                episodes  = {len(acc_list)}
                mean      = {mean_acc:.4f}
                staging   = .staging/iter_{n}/metrics.json
                → loop's Atomic Writer next
```

Return `status=OK` to loop on any completed run, regardless of value
distribution. Low / all-zero means is data, not a failure — the loop
is free to interpret it however its method requires.

## Outputs

| Path | What |
|------|------|
| `v{N}/.staging/iter_{n}/metrics.json` | Neutral schema (run_id, episode_count, acc_list, primary_metric, primary_metric_value, secondary_metrics) |
| `v{N}/.staging/iter_{n}/summary.csv` | Per-episode rows (standard) |
| `v{N}/.staging/iter_{n}/export.json` | Full `/api/eval/v2/export` payload (standard) |
| `v{N}/.staging/iter_{n}/eval_run_id.txt` | Backend's `run_id` |
| Plus any `evaluator.artifacts` declared by variant config | (e.g. raw export dump for verbose variants) |

## Notes

- **No method knowledge here.** Bootstrap CI, fitness strings, archive
  scoring, retry-on-low-fitness — all of these are the variant loop /
  Atomic Writer / lib, never this skill.
- **No retry tier here.** Low values are data; infra failure is a SKIP
  signal, with the bookkeeping owned by the loop.
- **All commit logic lives in loop's Atomic Writer.** Evaluator only
  writes to staging.
- **Profile drift across iters**: if profile's `primary_metric` or
  `episode_count` changes mid-run, downstream consumers (archive,
  trace.md) become non-comparable. Per files-contract, this requires
  `--new-version`.
- **The `experiment:run` wrapper is mandatory** — bare
  `python tools/eval_perf.py` bypasses admission control.

## Variant config schema (this skill)

```yaml
# <variant>/config.yaml
evaluator:
  profile_key:  perf_<graph>           # which {graph}.yaml profile to use (no default fallback)
  artifacts:    []                      # [{hook, write, from}] — file-level side-effects
```
