# AAS Experiment Instructions

This doc has two parts.

- **Part 1 — Concepts and discipline.** Read by Claude Code at session start and by humans who need to know how the experiment system is shaped. Covers what counts as an experiment, the backend / profile / artifact model, the pollution and leakage threat model, the discipline rules, and the incidents that motivated them.
- **Part 2 — User instructions.** Read by a human about to type commands. Two quick starts (a single graph eval and a first architect cycle) and an after-run cleanup checklist.

If you are reading this because you are about to run something, jump to Part 2. If you are reading this to update Claude's mental model of the experiment system, stay in Part 1.

---

# Part 1 — Concepts and discipline

## 1.1 What counts as an experiment

Three flows produce experiments here. Pick by what you are doing, not by which command is shortest.

| Flow | When to use | Skill / Entry |
|---|---|---|
| **Architect cycle** | Multi-iter architecture search on one graph (`iter_0` → `iter_M`); reports a search trajectory, not a single number | `/architect:<variant>:loop` (`adas-subagent`, `aflow`, `myloop`) |
| **Single graph eval** | One number on one graph + split (reproduction, baseline, smoke); no architect history | `/experiment:run <profile> <graph_name> [k=v ...]` |
| **Ad-hoc Python smoke** | Driving a nodeset / runtime / pure script directly; debugging an env or model without a graph | bash (`python <eval-script>.py`, manual driver scripts) |

The wrapper (`/experiment:run`) is **graph-only** since 2026-05-07 — it submits a graph eval to the backend's `JobScheduler`. Ad-hoc Python smoke runs in bash directly (the old `<profile> -- <cmd>` form is gone). See `feedback_experiment_run_graph_only`.

Architect cycles internally fan out to single graph evals, so they hit the same admission control.

This doc applies to all three flows. Pure graph-correctness smokes (`vram_mb: 0`, `doc-smoke` profile) and unit tests that don't touch a GPU are out of scope.

## 1.2 The user's backend

Two ports are involved, but one belongs to the backend and the other is a proxy. After this section, the rest of the doc just says "the user's backend".

- **The backend (`uvicorn` / FastAPI) listens on `:8000`.** This is the long-lived process the user starts manually (`cd agentcanvas/backend && uvicorn app.main:app --reload --port 8000`). It owns the `JobScheduler`, the cross-session admission ledger, the run subprocess lifecycle.
- **The Vite frontend dev server listens on `:5173`** and proxies `/api` + `/ws` to `:8000`. The user typically has this open in the browser too.
- **Clients (skills, `submit.py`, `status.py`) default to `:5173`** via `AGENTCANVAS_BACKEND_URL`. They talk through the proxy. The skill code does not connect to `:8000` directly under normal use.

There is **one** backend per host. `/experiment:run` is the single sanctioned entry; submissions land in one queue shared across all Claude sessions on the host. See `.claude/commands/experiment/README.md` for the scheduler internals.

Rules from this:
- Never start a sibling uvicorn. Never `pkill uvicorn` / `pkill auto_host` / blanket-regex kills. Backend lifetime belongs to `/experiment:teardown` (explicit user action) and to `PR_SET_PDEATHSIG` cleanup at the kernel level (see `project_pdeathsig_cleanup`).
- If the backend is down, ask the user to start it. Do not auto-spawn one.

## 1.3 Profile system

Two yaml files, easy to confuse. They are read by different callers and contain different things.

| File | Read by | Contains | Notes |
|---|---|---|---|
| `.claude/commands/experiment/profiles.yaml` | `bin/submit.py` (the `/experiment:run` wrapper) | Resource catalog per named profile: `vram_mb`, `exclusive_gpu`, `priority` | Unknown profile names silently fall back to `exclusive_gpu: true` + full VRAM, which serializes every sibling session. **Add the entry before first run.** |
| `workspace/architect/exp_profiles/<graph>.yaml` | `/architect:experiment` skill, humans | Per-graph eval defaults: `split`, `worker_count`, `step_budget`, `per_step_budget_sec`, `episode_count` | **`/experiment:run` does NOT read this file** (see `feedback_experiment_run_ignores_expyaml`). Treat it as human-readable documentation of what the eval *should* be — pass values explicitly on the CLI to actually run them. |

**The CLI is the source of truth at run time.** If you edited the profile yaml and ran `/experiment:run smartway-ce smartway2_ce`, the run uses the wrapper's defaults (`split=val_unseen`, `episode_count=-1`, `worker_count=1`), not your edits. Always pass the values you care about explicitly.

## 1.4 Run artifacts layout

Where output lands depends on which flow produced it.

**Single graph eval (`/experiment:run`):**

```
outputs/eval_runs/{run_id}/
├── spec.json          # what was actually submitted (eval block, profile)
├── summary.json       # aggregate metrics, terminal state
├── summary.csv        # per-episode rows
├── shared_urls.json   # nodeset endpoints used
├── stderr.log
├── stdout.log
├── graph.json         # frozen snapshot of the graph that ran
├── _DONE              # terminal sentinel
└── episodes/
    └── ep{NNNN}/
        ├── log.jsonl        # per-step node events
        ├── episode.json     # per-episode metadata
        └── assets/          # captured frames, intermediate state
```

Cross-session, persistent. `run_id` is timestamp-based.

**Architect cycle (`/architect:<variant>:loop`):**

```
outputs/design_runs/{method}/{graph}/v{N}/
├── config.md          # target + Created: timestamp
├── trace.md           # one row per iter (metric table)
├── lineage.md         # narrative log across iters
├── <variant vN-level files>   # e.g. myloop: goal.md, knowledge.md, hypotheses.jsonl
└── iteration/
    ├── iter_0/        # baseline (no parent, no active mutation)
    └── iter_M/
        ├── graph.json
        ├── <graph>.yaml             # convenience copy of the profile
        ├── active_workspace/        # files this iter overrides vs frozen workspace
        ├── eval_run_id.txt          # points back to outputs/eval_runs/{run_id}/
        ├── metrics.json
        ├── summary.csv
        ├── export.json
        └── <variant sentinel files>
```

`{method}` is the variant slug. `v{N}` bumps manually with `--new-version` (major pivots only). For the full contract see `.claude/commands/architect/_common/files-contract.md`.

Both trees are gitignored by default. Promote a run to `outputs/archive/{slug}-YYYY-MM-DD/` only when you can also commit the exact graph + profile that produced it.

## 1.5 Threat model: pollution and leakage

### 1.5.1 Pollution — bad context entering iter_M

| Mode | Where it sneaks in | Why it hurts |
|---|---|---|
| **Stale memory facts** | `.claude/memory/architect/*.md` and `platform/*.md` cached at session start | Iter proposer cites a rule that was already superseded; design moves backward |
| **Cross-iter bleed** | Proposer reads iter_{M−1}'s narrative + metrics together → anchored to one local minimum | Hill-climb tunnel-vision; loop tightens onto local fixes instead of stepping back to meta-hypothesis sweeps |
| **Cross-variant bleed** | Same graph touched by `adas-subagent` then `aflow` without `--new-version`; v{N} mixes two methods' history | Lineage becomes uninterpretable; rollback semantics break |
| **Profile drift** | `workspace/architect/exp_profiles/<graph>.yaml` mutated mid-run | iter_M and iter_{M+1} eval'd on different settings, metric delta is meaningless |
| **Number drift** | Treating "paper-reported number" as current baseline without re-verifying on the same split (see `feedback_verify_model_from_log`) | Fitness compared against fictional ceiling |

### 1.5.2 Leakage — sensitive context escaping iter_M

| Mode | Where it leaks | Why it hurts |
|---|---|---|
| **Val/test answers into prompt** | Proposer / evaluator prompts include per-episode `instruction_id`, oracle waypoint list, or `final_action` | Search overfits to leaked episodes, val SR no longer generalizes |
| **Oracle path / GT semantics into design** | Graph reads `episode.oracle_*` fields meant for scoring | Design exploits scorer rather than the navigation task |
| **Secrets in logs** | `OPENAI_API_KEY` / HF tokens echoed in `stdout.log`, `backend.log`, then archived to `outputs/eval_runs/` | Push the run-dir to git, secret is now in history |
| **Future-iter hindsight** | Iter M's evaluator reads anything under `iter_{M+1}/` or `iter_{M+2}/` | Loop becomes acausal; reported convergence is fake |
| **Cross-graph contamination** | Memory entries written from graph A appear as "facts" while working on graph B | Generalization claim collapses |

## 1.6 Discipline rules

### 1.6.1 Pre-flight invariants

A run that violates these will produce a number you cannot trust. Verify before starting iter_0 (or a single eval).

1. **Profile is frozen.** `workspace/architect/exp_profiles/<graph>.yaml` matches what you will report as the run's settings. Mutate before iter_0 or after `--new-version`, never mid-v{N}.
2. **Baseline is re-derived, not cited.** Read it from `iter_0/metrics.json` after a real eval, not from a paper or prior chat. Cross-check `iter_0/log.jsonl`'s `planner_llm.inner_log[model]` matches the model you think you're using.
3. **Resource profile exists in `.claude/commands/experiment/profiles.yaml`.** Unknown profile names fall back to full-VRAM exclusive, serializing every sibling.
4. **Memory store is clean.** Run `/memory:hygiene` to catch auto-writes from the previous session, then skim `.claude/memory/architect/` and `.claude/memory/platform/` for entries that contradict what you're about to assume.
5. **Backend invariants hold.** The user's backend is up; no sibling uvicorn or auto_host hanging around; no `pkill` candidates in flight.
6. **Working tree state is recorded.** `git rev-parse HEAD` + `git status` clean (or known-dirty with reason logged in `v{N}/config.md`).
7. **No silent resume.** If `iter_M/` already exists, point the loop at the highest existing iter explicitly; do not let it overwrite.

### 1.6.2 In-flight rules

While a cycle is running:

1. **One backend, one queue.** Never start a parallel uvicorn or load a model directly to dodge admission.
2. **`$BACKEND_URL`, not a port literal.** Inside any wrapped command read the URL from env. Hardcoding `:5173` or `:8000` breaks if the proxy ever moves.
3. **Proposer cannot read future iters.** Skill code must enumerate `iter_M` with `M < current`. If globbing `iter_*/`, sort + truncate explicitly.
4. **Proposer cannot read evaluator's per-episode rows.** It sees aggregate `metrics.json` and the failure summary distilled by the variant — never raw `summary.csv` rows or per-step `log.jsonl`. Per-episode detail is for the human reviewer.
5. **Never edit `iter_M/` after `_DONE`.** Sealed iters are part of lineage. Revise by writing `iter_{M+1}/` or bumping `v{N}`.
6. **Never edit the profile yaml mid-v{N}.** `--new-version` first if you need a setting change.
7. **No `pkill`, no blanket regex kills.** Backend lifetime belongs to `/experiment:teardown` only. The `/experiment:run` cleanup trap kills its own PGID — that is the authorized blast radius.
8. **Status checks go through `/experiment:status`.** Don't hand-roll `curl` + `jq` on `summary.json` (`feedback_use_experiment_status`).
9. **On non-zero exit, read the backend tail dump first.** `/experiment:run` auto-dumps the last 30 ERROR / OOM / Traceback / Killed lines. Read those before grepping raw logs (`feedback_experiment_run_silent_failures`).

### 1.6.3 Post-run invariants

Before declaring an iter or eval done:

1. **`status="completed"` is not enough.** Also check `step_count > 0` and `metrics` non-empty. GraphExecutor silently marks an episode complete when a node returns `{"error": ...}` instead of the declared port (`project_silent_episode_completion_on_node_error`).
2. **Separate the report from the design log.** `lineage.md` is what the next iter inherits — keep it factual (what changed, what the metric did). Speculation, framing, TODOs go in `v{N}/notes.md` or `iter_M/report.md`.
3. **Redact secrets from any tail dump you paste back.** If a backend log tail contains an `Authorization:` header or token, strip it before chat or commit.
4. **Memory writes are surprise-only.** Save a new memory entry only if the loop uncovered a fact that is (a) non-obvious from code or git history and (b) likely to bite a future iter. Single-graph wins go to `lineage.md`, not to `.claude/memory/architect/`.
5. **Cross-graph claims need cross-graph evidence.** Verify "method X helps" on ≥2 graphs before promoting to memory.

## 1.7 Real incidents (do not repeat)

These are not hypothetical — they happened in this repo. Read them as "do not repeat".

- **Model identity drift.** Profile said `gpt-4o` but logs were `gpt-4-1106-preview` (since-retired). Fix: always cross-check `log.jsonl` `inner_log[model]` (`feedback_verify_model_from_log`, `feedback_openeqa_reasoner_model`).
- **Paper-number ghost baseline.** MapGPT comparison used 0.535 from a different subset than the 216-ep MapGPT_72 we actually evaluate; real paper number on our subset is 0.477. Always re-derive the baseline from the same split you're scoring on.
- **Silent episode loss as fake completion.** GraphExecutor counted a node-error as a finished episode (`step_count=0`), inflating reported completion rate while truncating the run (`project_silent_episode_completion_on_node_error`).
- **Iter init persistence false → input freeze.** Initialize port with `persist=false` looked correct at iter 0, starved consumers from iter 1 onward — every subsequent iter scored against frozen step-0 observations (`feedback_initialize_persist_default`, `feedback_iterin_dual_wire_obs_freeze`).
- **Bare `python <eval-script>.py`** that loaded a model on GPU without admission → OOM'd a sibling session's run mid-iter. Fix: always submit through `/experiment:run` for graph evals; bash for ad-hoc only when you know it's safe.
- **Profile mid-run mutation.** `<graph>.yaml` edited between iter_3 and iter_4 to "try a different worker_count" — the metric delta was no longer attributable to the design change. Fix: `--new-version` for any setting change.
- **R2R `val_unseen` is stratified.** Eps 0–49 SR is ~0.68; eps 150–199 is ~0.28 on the same graph. Quoting a 50-ep prefix overstates the headline. Use MapGPT72 (216-ep random sample) for paper-comparable numbers (`project_r2r_val_unseen_stratification`).

---

# Part 2 — User instructions

Read this section when you are about to type commands. One quick start, one after-run wrap-up.

## 2.1 Quick start

Two steps. Everything between them is fully automated.

1. **Launch the user's backend.** One terminal, foreground:

   ```bash
   bash agentcanvas/run_dev.sh
   ```

   The script activates the `agentcanvas` conda env, starts the uvicorn backend at `:8000`, and brings up the Vite frontend at `:5173`. Leave it running.

2. **Open a fresh Claude Code session and invoke the variant loop:**

   ```
   /architect:<variant>:loop <graph>
   ```

   `<variant>` ∈ `{adas-subagent, aflow, myloop}`. The loop runs end-to-end without further input: iters land at `outputs/design_runs/<variant>/<graph>/v{N}/iteration/iter_M/`. Walk away. Come back when it finishes.

A fresh session is preferred over reusing an existing one — the architect loop's context budget is best protected from unrelated prior conversation.

## 2.2 After a run finishes

Invoke:

```
/architect:cleanup
```

One skill, two actions:

1. **Archive `outputs/design_runs/`** — dry-run plan shown first, you approve, then live execution. Moves cycle output to `outputs/design_runs/<method>/<graph>/_archive/` and rsyncs to a sibling root outside the repo. See `.claude/commands/architect/data/archive_outputs.py`.
2. **Scrub auto-memory writes** — invokes `/memory:hygiene` to classify and prune any new entries that the auto-memory system wrote during the run.

See `.claude/commands/architect/cleanup.md` for the full step list. If you skip this and start the next run, the previous run's auto-memory writes will still be in `.claude/memory/`, polluting the new session's proposer context.

---

# Tooling reference

Commands you'll reach for in either part.

| Need | Command |
|---|---|
| Run a graph eval with admission control | `/experiment:run <profile> <graph_name> [k=v ...]` |
| Run an ad-hoc Python smoke that loads a model | bash directly (wrapper is graph-only post 2026-05-07) |
| Check status of a run | `/experiment:status [run_id]` |
| Stop the user's backend (only when asked) | `/experiment:teardown` |
| Drive an architect cycle | `/architect:<variant>:loop` |
| Variant-specific run-state + JIT framework load | `/architect:<variant>:understand` |
| Architect-overview mental model | `/architect-overview:understand` |
| Wrap up after an AAS run (archive + memory scrub) | `/architect:cleanup` |
| Memory scrub only | `/memory:hygiene` |
| Raw memory diff (detection helper) | `python3 .claude/commands/memory/data/memory_diff.py [--scope X] [--since REF]` |
| Archive outputs/design_runs/ (helper) | `python3 .claude/commands/architect/data/archive_outputs.py [--dry-run]` |
| Per-run artefacts | `outputs/eval_runs/<run_id>/` (eval) + `outputs/design_runs/<method>/<graph>/v{N}/iteration/iter_M/` (architect) |
| Shared run-dir contract | `.claude/commands/architect/_common/files-contract.md` |
| Experiment infra notes | `.claude/commands/experiment/README.md` |

# When in doubt

- If a discipline rule seems to block what you want to do, **stop and ask the human**. The cost of pausing is small; the cost of a contaminated multi-iter run is a re-run from `iter_0`.
- If you discover a new pollution / leakage mode while running, add a one-line entry to §1.7 **and** save a memory under `.claude/memory/architect/` so the next session inherits the lesson.
- If something in this doc contradicts the current state of the code or skills, the code wins — fix this doc first, then continue.
