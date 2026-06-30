# myloop — knowledge-distillation orchestrator

> **Framing.** Maximizing agent performance is myloop's **goal**.
> Knowledge distillation is the *implementation mechanism* chosen to
> reach it — not a separate end. Mechanically myloop is a
> knowledge-distillation loop: an agent-shaped orchestrator that
> persistently sharpens its understanding of a problem, stored across
> eleven structured files. It sharpens understanding because that is how
> it climbs the performance goal — the distillation serves the metric,
> not the other way round.
>
> A single-pass THINK, left alone, ruts: it refines one
> `intervention_axis` (e.g. prompt edits) and, on exhausting that one
> lever class, declares *global* saturation when only one axis was
> ever searched. So myloop runs a **REFLECT** meta-phase that
> periodically maps the whole intervention space (`search_space.md`)
> and redirects THINK off the local optimum.
>
> This is the deliberate departure from ADAS-shape variants
> (`adas-subagent`, `aflow`) in the same family. Those optimize
> `fitness` against an `archive`. myloop optimizes `understanding`
> against a `goal`.

---

## One-paragraph contract

A user specifies the run's goal — either by authoring a full `goal.md`
(Ultimate / Escalation / Termination sections) or by passing
`--goal "<one-liner>"` on the command line (the loop bootstraps a
`goal.md` with the one-liner as Ultimate plus built-in defaults for
Escalation + Termination). myloop then bootstraps the rest of the
`vN/` working directory and runs a series of **iterations**. Each
iteration is a hard-paired `THINK → CRITIC → EXPERIMENT → DISTILL`,
with the latter three running **per spec** when THINK emits K > 1
specs in one envelope:

- **THINK** (`/architect:myloop:proposer`) — one tool-augmented
  sub-agent spawn per iter. It reads the persistent working memory,
  reasons, optionally writes to hypotheses/knowledge/tools/
  experiment_design eagerly, and **must** produce an
  `ExperimentSpec` envelope containing K ≥ 1 specs (cap K ≤ 3 via
  `config.yaml § orchestrator.caps.max_specs_per_iter`). Multi-spec
  breadth is for testing two genuinely-different mechanisms in
  parallel within one iter — see proposer.md "MULTI-SPEC PER ITER".
- **CRITIC** (`/architect:myloop:critic`) — pre-EXPERIMENT pathology
  vetting, **one sub-agent spawn per patched spec** (no-patch specs
  skip CRITIC). Each spawn predicts whether that spec will recur a
  past refuted pathology; verdicts are independent across specs (one
  BLOCK does not stop another spec's eval).
- **EXPERIMENT** — owned inline by `loop.md § 3c`. Per spec: an
  isolated editing sub-agent realizes the patch into
  `active_workspace_<spec_id>/`, smoke-eval gates correctness, then
  the **Python multi-spec eval wrapper**
  (`.claude/commands/architect/myloop/lib/multi_spec_eval.py`)
  handles the measured eval — submitting ALL `(spec, pass_idx)`
  combinations across `ready_for_eval` as **one parallel wave** (the
  iter's "合集的实验"). Per-submission `worker_count` is allocated by
  a binary-search makespan minimizer over the wave (ep-weighted under
  `perf_<graph>.worker_count` as method-wide cap, clamped by each
  spec's `profile.worker_count`); the JobScheduler arbitrates admission
  by VRAM. The wrapper writes
  per-spec `eval_metadata_<spec_id>.json` files. Even probe-sized
  experiments (1 ep) count; the rhythm cannot be broken.
- **DISTILL** (`/architect:myloop:distill`) — **one sub-agent spawn
  per iter** that sees all K specs side-by-side. Produces K per-spec
  verdicts (`distill_<spec_id>.json`) plus an iter-level summary
  (`distill_summary.json` with `cross_spec_lesson` +
  `milestone_after`).

On top of the per-iter triple sits **REFLECT**
(`/architect:myloop:reflect`) — a **meta-phase** that does NOT run
every iter. It fires on three triggers (heartbeat every K iters /
the last K iters sharing one `intervention_axis` / a THINK
`SKIP_THINK_EMPTY` escalation). It is a separate sub-agent spawn with
no "produce a spec" pressure; its only job is **search-space
cartography**: it audits which `intervention_axis` each committed
iter used, maintains `search_space.md` (the taxonomy + per-REFLECT
coverage map), and hands THINK a ranked **frontier** of unexplored
axes. REFLECT is the diversification operator that keeps THINK from
ruting in one lever class.

Each iter commits an `IterRecord` to `iteration/iter_n/record.json`
(carrying a `reflect` block on iters where REFLECT fired).

Termination is goal-driven, not iter-capped. Loop polls
`goal.md § Termination` at the start of every iter. A
`SKIP_THINK_EMPTY` from THINK never terminates the run — it escalates
to REFLECT, and only a REFLECT verdict of `SPACE_EXHAUSTED` (the
whole space surveyed and closed) yields a `SATURATED` stop.

---

## The eleven files (per `outputs/design_runs/myloop/{graph}/v{N}/`)

| File | Content | Time scale | Writer | Mutation pattern |
|---|---|---|---|---|
| `goal.md` | mission spec: ultimate goal + escalation + termination | long, user-owned | **user only** | static (per vN) |
| `constraints.md` (optional) | hard rules every phase must respect (MUST / MUST NOT bullets) | long, mixed-ownership | bootstrap-merged (common.md + {graph}.md + flags); user-editable thereafter | static between iters; edit to relax / tighten |
| `knowledge.md` | pure facts about the system / graph / dataset | long, cross-vN | proposer + distill (eager) | append-only |
| `search_space.md` | intervention-axis taxonomy + per-REFLECT coverage / frontier | long | bootstrap-seeded; reflect (eager) | append-only |
| `experience.jsonl` | lessons learned — both confirmed and refuted hypotheses | long | distill (eager) | append-only |
| `hypotheses.jsonl` | open conjectures awaiting test | long but consumable | proposer + distill (eager) | append + line-delete on resolve |
| `iteration/iter_n/record.json` | one dense `IterRecord` per committed iter | per-iter | loop | written once at commit |
| `experiment_design.yaml` | eval-profile registry (smoke + full baselines + orchestrator-authored probes) | long | bootstrap + proposer (eager) | append-only |
| `tools/*.py` | orchestrator-authored utility functions for analyzing runs | long | proposer (eager) | add-only |
| `trace.md` | one-row-per-iter axis/metric/outcome history table — the at-a-glance progress view | per-iter | loop | regenerated wholesale from `record.json` each commit |
| `lineage.md` | one-section-per-iter narrative (think / change / metrics / distill / reflect) | per-iter | loop | regenerated wholesale from `record.json` each commit |

**No `archive.jsonl`.** That was the ADAS-shape concept,
deliberately not adopted by myloop.

Full schemas for each file live in `schemas.md`; the machine-readable
manifest — every file's type / purpose / access — is in
`config.yaml § manifest` (per `_common/files-contract.md § 4`).

---

## Loop diagram

```
┌────────────────────────────────────────────────┐
│  goal.md                                       │
│   ├─ Ultimate     (what direction)             │
│   ├─ Escalation   (probe/custom → perf upgrades)│
│   └─ Termination  (when to stop)               │
└──────────────────────┬─────────────────────────┘
                       │
              ┌────────▼────────┐
              │   for each iter │
              └────────┬────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ Termination poll vs goal § Termination   │  ← if hit, exit
   └───────────────────┬──────────────────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ REFLECT trigger? (meta-phase)            │
   │  ─ fires on: heartbeat / axis-           │
   │     concentration / THINK-SKIP escalate  │
   │  ─ /architect:myloop:reflect (1 sub-     │
   │     agent, no spec pressure)             │
   │  ─ maps search_space.md; ranks a frontier│
   │  ─ SPACE_EXHAUSTED → exit SATURATED      │
   └───────────────────┬──────────────────────┘
                       │ (frontier handed to THINK)
   ┌───────────────────▼──────────────────────┐
   │ THINK   (/architect:myloop:proposer)     │
   │  ─ reason (1 sub-agent, full tools)      │
   │  ─ reads goal/knowledge/search_space/    │
   │     experience/hypotheses/prior records  │
   │  ─ obeys the axis-jump rule (frontier)   │
   │  ─ may eager-write hypotheses / knowledge│
   │  ─ MUST produce spec.json envelope with  │
   │     K ≥ 1 specs (cap K ≤ 3);             │
   │     SKIP_THINK_EMPTY → REFLECT           │
   └───────────────────┬──────────────────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ CRITIC (/architect:myloop:critic)        │  ← per patched spec
   │  ─ K sub-agent spawns (one per spec      │
   │     with patch != null)                  │
   │  ─ each predicts pathology recurrence    │
   │  ─ OK / WARN / REVISE / BLOCK per spec   │
   │  ─ REVISE/BLOCK → kick THINK back on     │
   │     THAT spec only (sibling specs        │
   │     proceed independently)               │
   │  ─ writes critique_<spec_id>.json each   │
   └───────────────────┬──────────────────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ EXPERIMENT  (inline in loop.md § 3c)     │
   │  ─ per spec: apply step → smoke retry    │
   │      writes active_workspace_<spec_id>/  │
   │  ─ Python multi_spec_eval wrapper runs   │
   │      measured eval for all surviving     │
   │      specs (ONE parallel wave: all K     │
   │      specs × passes submit at once;      │
   │      JobScheduler arbitrates VRAM)       │
   │  ─ writes eval_metadata_<spec_id>.json   │
   │      per spec                            │
   └───────────────────┬──────────────────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ DISTILL (/architect:myloop:distill)      │
   │  ─ ONE sub-agent for the whole iter      │
   │  ─ sees all K specs side-by-side         │
   │  ─ produces per-spec verdict +           │
   │     cross-spec lesson + iter milestone   │
   │  ─ writes distill_<spec_id>.json (×K) +  │
   │     distill_summary.json                 │
   │  ─ experience.append, hypotheses.del/add,│
   │     knowledge.append (eager)             │
   └───────────────────┬──────────────────────┘
                       │
   ┌───────────────────▼──────────────────────┐
   │ Atomic commit (loop)                     │
   │  ─ mv .staging/iter_n → iteration/iter_n/│
   │  ─ write iteration/iter_n/record.json    │
   │     (IterRecord with specs[] per-spec    │
   │      blocks + iter_summary rollup)       │
   │  ─ knowledge/experience/hypotheses/      │
   │     experiment_design/tools already on   │
   │     disk from eager writes               │
   └──────────────────────────────────────────┘
```

---

## Hard contracts

1. **Goal required pre-flight.** Loop refuses to start without one,
   provided in either form: a fully-authored `goal.md` (three
   sections) OR a `--goal "<text>"` flag (loop bootstraps `goal.md`
   from the one-liner with built-in defaults for Escalation +
   Termination). `constraints.md` is separate, OPTIONAL, and at
   bootstrap merges from up to four layers (in order):
   `data/constraints/common.md`, `data/constraints/{graph}.md`,
   `--cons-file <path>`, `--constraints "<text>"`. After bootstrap
   it is user-editable.
2. **Every committed iter has THINK + EXPERIMENT + DISTILL.** No "I
   only thought, no experiment" iters. Probe of 1 ep is the floor.
   Multi-spec iters: every spec in the envelope (K ≥ 1) has an
   experiment block in `record.specs[]` (even if the block records
   `critic_block` or `implementer_skip`). DISTILL is also non-
   skippable structurally — `SKIP_DISTILL_EMPTY` only means the
   agent returned empty distill blocks; the phase still ran.
3. **Apply-step SKIP / CRITIC BLOCK counts as an experiment.** If a
   spec's patch can't pass smoke retry in 3 attempts, that spec
   commits with `experiment.outcome_class="implementer_skip"`; a
   round-2 BLOCK without rebuttal commits as `"critic_block"`.
   Either is a refuted-patch data point. Sibling specs in the same
   iter proceed independently.
4. **`knowledge.md` is append-only.** Mistakes get corrected by
   appending a superseding bullet, not by editing the old one. Diff
   visibility matters; experience may revisit later.
5. **`workspace/*` is never modified mid-iter.** Patches live under
   per-spec `.staging/iter_n/active_workspace_<spec_id>/` overlays
   and ride to commit. Frozen workspace is the rollback anchor.
   Patches do NOT stack across specs in one iter — each spec gets
   its own independent overlay.
6. **`SKIP_THINK_EMPTY` never terminates the run.** It escalates to
   REFLECT. Only a REFLECT verdict of `SPACE_EXHAUSTED` — the
   whole intervention space surveyed and every axis exhausted or
   constraint-closed — produces a `SATURATED` stop. Every
   `ExperimentSpec` carries an `intervention_axis`; when the last K
   committed iters share one axis, THINK must jump axes or write a
   rebuttal (the axis-jump rule).

---

## How myloop relates to ADAS-family variants

| Dimension | adas-subagent / aflow | myloop |
|---|---|---|
| Loop shape | `propose → 3-retry-eval → archive` fixed | `THINK → CRITIC (×K patched specs) → EXPERIMENT (Python wrapper mixes K specs) → DISTILL (single spawn, K specs)` per iter + triggered REFLECT meta-phase |
| Experiments per iter | 1 | K ≥ 1 (cap 3) — multi-spec breadth for parallel mechanism probes |
| Noise control | none | per-spec `passes` field — multi-pass replay with paired-ep aggregation (mean_sr / sd_sr / robust_sr) |
| Working memory | one flat `archive.jsonl` | eleven structured files (incl. `search_space.md` + the `trace.md` / `lineage.md` rollups) |
| THINK structure | 3-call Reflexion (R0/R1/R2) | one tool-augmented sub-agent, free-form, frontier-guided |
| Optimization target | `fitness` (bootstrap CI median) | `goal` (any direction, optimization OR investigation) |
| Search-space awareness | implicit (archive replay-guard only) | explicit — `search_space.md` axes + REFLECT coverage map |
| Failure handling | `n -= 1`, generation NOT archived | committed as evidence (`outcome_class` records it) |
| Meta-LLM tools | none upstream; sub-agent has Read/Bash in adas-subagent | sub-agent + orchestrator-authored `tools/*.py` |
| Termination | iter cap or consecutive_skips | goal.md § Termination predicates + REFLECT `SPACE_EXHAUSTED` |

---

## Layout

```
.claude/commands/architect/myloop/
├── README.md            ← (this file) the myloop mental model
├── schemas.md           ← single source of truth for every file shape
├── understand.md        ← reading list to load mental model at session start
├── config.yaml          ← paths, caps (incl. max_specs_per_iter), EXPERIMENT knobs + files manifest
├── proposer.md          ← THINK phase — emits K-spec envelope
├── critic.md            ← CRITIC phase — per-spec pre-EXPERIMENT pathology vetting (one fire per patched spec)
├── reflect.md           ← REFLECT meta-phase (search-space cartography)
├── distill.md           ← DISTILL phase — one spawn for the whole iter (handles K specs side-by-side)
├── loop.md              ← orchestrator outer loop + per-spec EXPERIMENT phase (apply + smoke inline; measured eval delegated to multi_spec_eval wrapper)
├── data/
│   ├── seed_knowledge.md     ← bootstrap content for knowledge.md
│   └── seed_search_space.md  ← bootstrap content for search_space.md
└── lib/
    ├── helpers.py             ← misc utilities
    └── multi_spec_eval.py     ← Python wrapper for K-spec measured eval; owns single-wave submission + worker-count allocation under perf cap (loop.md § 3c contract)
```

**No `implementer.md` / `evaluator.md`.** myloop is self-contained:
the EXPERIMENT phase (apply patch + smoke-test + run eval) is inline
in `loop.md § 3c`, not delegated to `_common/` skills. The only
`_common/` file myloop still shares is `files-contract.md` (the
run-dir layout / resolve protocol / edit whitelist contract). The
ADAS-family variants (`adas-subagent`, `aflow`) keep using
`_common/implementer.md` + `_common/evaluator.md`.

Overlay seeding + the § 7 edit whitelist are handled by the shared
`_common/lib/overlay.py`, and `llmCall`-profile pinning by
`_common/lib/pin_llm_profile.py` — both deterministic helpers that
honor `files-contract.md` (the typed-patch applier was retired
2026-05-20 — the editing sub-agent edits the overlay natively).

---

## What's deliberately deferred

- `knowledge.md` allowed to mutate (supersede or retract).
- `hypotheses.jsonl` priority ordering as a first-class search policy.
- Multi-objective `goal` (compose direction + investigation + budget
  trade-off).
- `tools/` runtime metadata (usage count, last-used iter, deprecation).
- A critique pass on the *individual* `ExperimentSpec` (REFLECT audits
  the search space, not the quality of one proposal).
- Structured `constraints.yaml` with jsonpath validators (still
  concat-only markdown today — see schemas.md § 11).
