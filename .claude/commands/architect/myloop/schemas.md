# myloop — schemas (single source of truth)

This file defines the shape of every artifact the orchestrator reads
and writes. Every other myloop skill points back here rather than
inlining the schemas.

---

## 1. `goal.md` (markdown, user-authored, read-only to orchestrator)

Required structure (one `# Goal` header + three `## ...` sections):

```markdown
# Goal

## Ultimate
<one paragraph stating what this run is for. Direction-setting.
 Examples:
   - "Maximize success on MapGPT72 (216-ep R2R val_unseen subset)."
   - "Understand the causes of oracle_success-but-not-success on
      instructions longer than 35 tokens.">

## Escalation
<plain-English rules for when to upgrade probe / custom → perf.
 Example:
   - "Run probe / custom targeted subsets (1–30 ep) freely.
    - A custom ≥30-ep subset is the default per-iter lift
      measurement.
    - Run perf (216 ep) only when a custom ≥30-ep run beats the
      best-recorded perf mean by ≥ 0.05.">

## Termination
<list of predicates; loop terminates on first match. Each is a
 short rule the orchestrator can evaluate from file state:
   - "full eval success ≥ 0.50"   (goal-achieved)
   - "5 consecutive iters with no new knowledge entry and no
      hypothesis resolved"        (saturated)
   - "cumulative GPU-minutes ≥ 600 OR cumulative LLM-tokens ≥ 5M"
                                  (budget)>
```

**Hard contract** (one of two ways must be satisfied):
- `goal.md` exists at this path with **all three** sections (Ultimate,
  Escalation, Termination) present and non-empty, OR
- Loop was invoked with `--goal "<text>"` and no `goal.md` exists yet
  — in which case the loop bootstraps `goal.md` using `<text>` as the
  Ultimate paragraph and applies built-in defaults for Escalation +
  Termination (see `loop.md § 2a` for the default ladder + predicates).

If both forms are missing the loop aborts; partial `goal.md` (e.g.
only the Ultimate section) is treated as malformed and also aborts —
the `--goal` fast-path is for `goal.md`-NOT-present, not for
goal.md-fill-in-the-blanks. Users who started with `--goal` may
freely edit the resulting `goal.md` afterwards to tighten the
defaults.

---

## 2. `knowledge.md` (markdown, append-only, single document)

Pure facts about the system, graph, env, dataset. **Monotonically
grows.** The orchestrator may add new sections / bullets but does
not delete or rewrite existing entries.

Structure (free-form within `## Section` headers; each bullet ends
with `(added: iter_N)` for diff visibility):

```markdown
# Knowledge

## AgentCanvas system
- Graph = JSON topology + nodeset Python. (added: bootstrap)
- llmCall config keys actually read by runtime: profile, temperature,
  max_tokens, system_prompt, template, mode, n, stop. The `model` field
  is NOT read. (added: bootstrap)
- ...

## Graph: mapgpt_mp3d
- Step-budget is capped at 15 by the eval-side `step_budget`, even if
  the graph's internal step_budget=50. (added: iter_3)
- ...

## Dataset / env
- R2R val_unseen has 11k+ episodes; MapGPT72 is the 216-ep subset
  used as paper-comparable headline. (added: bootstrap)
- ...
```

---

## 3. `experience.jsonl` (JSONL, append-only, machine + human readable)

One record per "lesson learned" — closed-case empirical results. Both
confirmed and refuted hypotheses end up here. Append only.

```jsonl
{
  "exp_id": "exp_iter3_001",
  "iter_id": "iter_3",
  "ts": "2026-05-15T14:21:33",
  "lesson": "On long-instruction (>35 token) episodes, switching planner_llm temperature from 1.0 to 0.3 reduced early-stop rate from 0.42 to 0.18.",
  "evidence": {
    "run_ids":  ["20260515_143112"],
    "metrics_before": {"success_long": 0.10},
    "metrics_after":  {"success_long": 0.24}
  },
  "verdict": "confirmed | refuted | inconclusive",
  "resolved_hypothesis": "hyp_4 | null",
  "tags": ["long-instruction", "planner_llm", "temperature"]
}
```

---

## 4. `hypotheses.jsonl` (JSONL, mutable: append + per-line delete)

Open conjectures awaiting test. Entries are removed when verified
(result transcribed to `experience.jsonl`).

```jsonl
{
  "hyp_id": "hyp_4",
  "created_iter": "iter_2",
  "conjecture": "Long-instruction early-stop is caused by planner_llm's confidence saturation at step ~8 due to high temperature.",
  "rationale": "iter_1 + iter_2 both show 4/5 long-instr eps stop at step 8 with confidence > 0.95; planner uses temperature=1.0.",
  "test_design": "Run smoke with planner_llm temperature lowered to 0.3; expect early-stop rate to drop on long-instruction subset.",
  "priority": "high | medium | low"
}
```

Removal semantics: when an iter resolves a hypothesis, the iter's
DISTILL step removes the line matching `hyp_id` and appends a new
`exp_*` entry to `experience.jsonl`. Atomic — both edits live in the
same iter's atomic commit.

---

## 5. `iteration/iter_{n}/record.json` (one file per iter, JSON object)

The dense record. Written at iter commit time. Sits inside the iter's
own subdirectory (alongside the envelope `spec.json`, `think_trace.md`,
per-spec `eval_metadata_<spec_id>.json`, per-spec
`distill_<spec_id>.json` (when applicable), `distill_trace.md`, and
per-spec `active_workspace_<spec_id>/` / `debug_log_<spec_id>.md`
from EXPERIMENT).

Format = `IterRecord`. **Shape note**: an iter has ONE `think` block
and ONE `reflect` block (shared across all specs, because THINK and
REFLECT are per-iter phases) but per-spec `critic` / `experiment` /
`distill` blocks nested under `specs[]` (because each spec is an
independent intervention with its own vetting / eval / lesson). The
iter-level `iter_summary` rollup is the trace.md row.

```json
{
  "iter": 3,
  "ts_start": "2026-05-15T14:18:02",
  "ts_end":   "2026-05-15T14:35:11",
  "think": {
    "rationale": "Hypothesis hyp_4 (long-instr early-stop ↔ planner saturation) ready to test; this iter probes two angles in parallel — A (prompt-content: lower temperature) and B (topology: add a verifier node).",
    "file_updates": [
      {"file": "milestone",  "op": "set",    "content": "Resolve hyp_4 this iter."},
      {"file": "knowledge",  "op": "append", "section": "Graph: mapgpt_mp3d", "bullet": "planner_llm default temperature is 1.0 (added: iter_3)"}
    ],
    "spec_ref": "iteration/iter_3/spec.json"   // envelope path
  },
  "reflect": {                                  // OPTIONAL — present only when REFLECT fired this iter
    "trigger":        "heartbeat | concentration | skip",
    "status":         "FRONTIER_OPEN | SPACE_EXHAUSTED",
    "frontier_axes":  ["topology", "control-flow"],
    "reflection_id":  "reflection_2"
  },
  "specs": [
    {
      "spec_id": "spec_iter_3_A",
      "spec_kind": "custom",
      "intervention_axis": "prompt-content",
      "passes": 1,
      "critic": {                               // OPTIONAL — omitted when this spec had patch=null; see § 15
        "verdict":                       "OK | WARN | REVISE | BLOCK",
        "critic_round":                  1,
        "predicted_failure_modes_count": 2,
        "reference_experience_ids":      ["exp_iter3_001"],
        "block_override":                false,
        "critique_ref":                  "iteration/iter_3/critique_spec_iter_3_A.json"
      },
      "experiment": {
        "patch_applied":        true,
        "implementer_status":   "OK | SKIP_RUNTIME_FAIL | N/A",
        "implementer_attempts": 1,
        "run_ids":              ["20260515_143112"],                  // length = passes
        "artifacts_dirs":       ["outputs/eval_runs/20260515_143112/"], // length = passes
        "metrics_digest": {
          "mean_sr":   0.20,                    // mean SR across passes (= score for passes=1)
          "sd_sr":     null,                    // sd across passes; null when passes=1
          "robust_sr": null,                    // per-ep majority-vote SR; null when passes=1
          "score":     0.20,                    // alias = mean_sr (kept for trace.md compatibility)
          "spl":       0.09,
          "nav_error": 9.3
        },
        "per_ep_success":       [[1, 0, 0, 0, 0]],   // always nested: outer = passes (length = passes), inner = eps. passes=1 → [[ep_list]] (length-1 outer); empty (no eval ran for this spec, e.g. critic_block) → []
        "outcome_class":        "ok | crash | implementer_skip | critic_block"
      },
      "distill": {                              // OPTIONAL — omit on SKIP_DISTILL_EMPTY / SKIP_INVALID_DISTILL OR if this spec had no eval
        "verdict":               "confirmed | refuted | inconclusive",
        "promoted_to_experience": ["exp_iter3_001"],
        "resolved_hypotheses":    ["hyp_4"],
        "new_hypotheses":         [],
        "knowledge_diffs":        [{"section": "Graph: mapgpt_mp3d", "bullet_id": "kb_iter3_001"}]
      }
    },
    {
      "spec_id": "spec_iter_3_B",
      "spec_kind": "custom",
      "intervention_axis": "topology",
      "passes": 3,
      "critic":      { /* same shape, refs critique_spec_iter_3_B.json */ },
      "experiment":  {
        "patch_applied":  true,
        "run_ids":        ["20260515_143510", "20260515_145001", "20260515_150702"],  // 3 passes
        "metrics_digest": {"mean_sr": 0.31, "sd_sr": 0.018, "robust_sr": 0.28, "score": 0.31, ...},
        "per_ep_success": [[1,0,...], [1,0,...], [1,1,...]],   // 3 × 30
        "outcome_class":  "ok"
      },
      "distill":     { /* per-spec distill */ }
    }
  ],
  "iter_summary": {                             // REQUIRED — the trace.md / lineage.md rollup
    "outcome_class":   "confirmed | mixed | refuted | inert | critic_block | crash",
    "best_score":      0.31,                    // max of specs[*].experiment.metrics_digest.mean_sr (over specs with outcome_class=ok)
    "best_spec_id":    "spec_iter_3_B",
    "axes_touched":    ["prompt-content", "topology"],   // unique set across specs
    "milestone_after": "Verifier looks promising; iter_4 should test composition with the temperature drop.",
    "n_specs":         2,
    "n_confirmed":     1,
    "n_refuted":       1,
    "n_critic_block":  0,
    "n_crash":         0
  },
  "cost": {
    "gpu_min":     4.2,
    "llm_tokens":  68000,
    "wall_sec":    1029                         // iter-level: max(spec wall_sec) if specs ran in parallel, sum if sequential — wrapper reports
  }
}
```

**iter_summary.outcome_class rollup rule** (computed at commit time from `specs[*].experiment.outcome_class` + `specs[*].distill.verdict`):

| Spec verdicts | iter_summary.outcome_class |
|---|---|
| ≥ 1 spec `distill.verdict = "confirmed"` AND ≥ 1 spec `"refuted"` | `mixed` |
| ≥ 1 spec `confirmed`, no `refuted` | `confirmed` |
| all specs `refuted` or `inconclusive` | `refuted` |
| all specs `experiment.outcome_class = "implementer_skip"` or no patch | `inert` |
| all specs `experiment.outcome_class = "critic_block"` | `critic_block` |
| any spec `experiment.outcome_class = "crash"` AND others not all `confirmed` | `crash` |

The `implementer_status` / `implementer_attempts` field names and the
`outcome_class` value `"implementer_skip"` are **historical**: myloop
has no implementer skill — the apply step is inline in the EXPERIMENT
phase (`loop.md § 3c`). The names are kept stable so the IterRecord
schema does not churn; read them as "apply-step status / attempts".

**K=1 path**: a single-spec iter still uses `specs[]` (a one-element
list) and per-spec file suffixes — the schema is uniform regardless
of breadth, so trace/lineage regen and rollup logic does not branch.

**Schema version**: records under `v{N}/iteration/iter_*/` from
2026-05-24 onward use this shape. v0 records use the prior single-spec
shape (`experiment` / `critic` / `distill` blocks at the top level);
trace.md / lineage.md regen for vN ≥ 1 reads only new-shape records.

---

## 6. `experiment_design.yaml` (YAML, mutable — orchestrator can extend)

Registry of available eval configurations. Initial baselines + room
to add probes. Each entry mirrors the graph's `*.exp.yaml` profile
shape so `/experiment:run` consumes them directly.

```yaml
# Platform-baseline profiles (always present)
smoke_mapgpt_mp3d:
  episode_count: 3
  worker_count: 3
  episode_indices: [0, 35, 70]
  split: MapGPT72_first
  step_budget: 15
  per_step_budget_sec: 120
  description: "EXPERIMENT apply-step correctness gate — 'does the change run'. NOT a measurement tier."
  passes_required: 0                     # smoke is never multi-pass; correctness gate only

perf_mapgpt_mp3d:
  episode_count: 216
  worker_count: 40
  split: MapGPT72
  step_budget: 15
  per_step_budget_sec: 120
  description: "Full paper-comparable headline; expensive."
  passes_required: 1                     # N_eps=216 ≥ 119 → 1-pass acceptable (Bernoulli SE already <0.05 SR)

# THINK-composed targeted subsets (appended over time). A subset is
# built by reading a prior run's raw logs, collecting the episode
# indices that exhibit ONE failure mode, and recording the source in
# `derived_from`. The profile persists — later iters reuse it.
fm_premature_stop_iter2:
  episode_count: 18
  worker_count: 18
  episode_indices: [3, 7, 12, 19, 22, 28, 31, 40, 47, 55, 61, 68, 70, 77, 84, 91, 96, 103]
  split: MapGPT72
  step_budget: 15
  per_step_budget_sec: 120
  description: "The 18 MapGPT72 eps where the agent issued STOP > 3m from the goal viewpoint."
  passes_required: 3                     # auto-computed: 18 < 119 → 3
  derived_from:
    iter: "iter_2"
    source_run: "20260515_130940"        # the iter_0 baseline run whose logs were scanned
    failure_mode: "premature STOP — agent stops before reaching goal"
    filter_tool: "tools/failing_eps.py"
  baseline:                              # LOCKED on first 3-pass use of this profile; immutable thereafter
    mean_sr:   0.523
    sd_sr:     0.021
    robust_sr: 0.480                     # per-ep ≥⌈passes/2⌉ success rate
    passes:    3
    run_ids:   ["20260524_140112", "20260524_140530", "20260524_140950"]
    locked_at: "iter_4"                  # the iter that first ran the profile under 3-pass policy
    locked_ts: "2026-05-24T14:18:02"
```

**`passes_required` field**:

- `passes_required` is a per-profile field, auto-computed at profile
  creation time:
  - `0` for smoke profiles (correctness gate, not measurement)
  - `3` for any measurement profile with `episode_count < 119`
  - `1` for any measurement profile with `episode_count ≥ 119`
- THINK MAY override per-spec via `spec.passes` — but only UPWARD
  (i.e. `spec.passes ≥ profile.passes_required`). Specs with
  `passes < passes_required` are rejected by proposer's Step-4 lint
  with `SKIP_INVALID_SPEC`. CRITIC also flags this as a `REVISE`
  pathology.
- The 119 threshold comes from the noise-floor math: SR ≈ 0.45 →
  single-ep Bernoulli sd ≈ 0.497 → N ≥ 122 to get SE ≤ 0.045
  (≈ ±0.09 SR CI95). Below 119, single-pass noise dominates plausible
  effect sizes; multi-pass with sd reporting is required.

**`baseline` field** (LOCKED on first 3-pass use of the profile, immutable thereafter):

- Present iff at least one iter has run this profile under
  `passes_required ≥ 3`. Locked by the multi-spec wrapper after the
  iter's eval completes (`loop.md § 3c-(b)` post-wrapper step).
- `baseline.passes` MUST equal the profile's `passes_required`. If
  THINK later runs a spec with `passes > baseline.passes` against
  this profile, the comparison uses the first `baseline.passes` of
  the new run's results for fairness; the extra passes are recorded
  but not part of the lift computation against baseline.
- `locked_at` is the iter that first ran the profile under the
  required-passes policy; this is the iter whose eval populated the
  baseline numbers.
- Subsequent specs using the profile read `baseline.mean_sr` and
  `baseline.sd_sr` for delta + power judgment in CRITIC and DISTILL.
- A profile's baseline is never edited or unlocked — if the
  underlying eval env changes such that the baseline is no longer
  meaningful, create a new profile (e.g. `perf_mapgpt_mp3d_v2`).

---

## 7. `tools/*.py` (Python files, orchestrator-authored)

One function per file (convention; multi-function allowed but the
first public function is the canonical entry). Function docstring
is its registry metadata.

```python
# tools/by_instr_len.py
"""
Filter episodes by instruction token length.

Usage:
    from tools.by_instr_len import filter_eps
    eps = filter_eps(run_id="20260515_130940", min_tokens=35)

Returns: list[int] of episode indices satisfying the filter.

Created: iter_2 (to support hyp_4 test design).
"""
import json
from pathlib import Path

def filter_eps(run_id: str, min_tokens: int = 35) -> list[int]:
    base = Path("outputs/eval_runs") / run_id / "episodes"
    out = []
    for ep_dir in sorted(base.glob("ep*")):
        ep_meta = json.loads((ep_dir / "episode.json").read_text())
        if len(ep_meta["instruction"].split()) >= min_tokens:
            out.append(int(ep_dir.name[2:]))
    return out
```

The orchestrator discovers tools by `ls tools/*.py` + reading each
file's module docstring. No central registry file — the directory IS
the registry.

---

## 8. `ExperimentSpec` envelope (transient, in `.staging/iter_{n}/spec.json`)

Produced by THINK, consumed by CRITIC + EXPERIMENT. Atomic-promoted
into the iter dir on commit. **The envelope holds a list of K ≥ 1
specs** — myloop supports multiple independent experiments per iter
(`max_specs_per_iter` cap in config.yaml; default 3). Each entry in
`specs[]` is one self-contained experiment with its own patch /
eval_profile / risk_vectors; CRITIC + EXPERIMENT + DISTILL all operate
per-spec.

```json
{
  "iter": 3,
  "specs": [
    {
      "spec_id": "spec_iter_3_A",
      "kind": "probe | perf | custom",
      "intervention_axis": "prompt-content",
      "target": {
        "hypothesis_id": "hyp_4",
        "design_intent": "Test whether lowering planner temperature to 0.3 reduces long-instr early-stop."
      },
      "patch": {
        "intent": "Lower the planner llmCall temperature to 0.3 — testing whether it reduces long-instruction early-stop.",
        "targets": ["workspace/graphs/mapgpt_mp3d.json"]
      },
      "eval_profile": {
        "name": "fm_premature_stop_iter2",
        "overrides": {}
      },
      "passes": 1,                              // REQUIRED; ≥ 1. The multi-spec wrapper runs `passes` eval draws on the same ep set; results are aggregated into mean_sr / sd_sr / robust_sr in record.json. passes>1 is a separate axis from K-spec: K controls cross-spec breadth, passes controls within-spec noise reduction. See loop.md § 3c (multi_spec_eval wrapper).
      "expected_signal": [
        "If hyp_4 holds: long-instr success ≥ 0.20 (was 0.10 in the iter_0 baseline).",
        "If hyp_4 refuted: long-instr success stays ≤ 0.15."
      ],
      "risk_vectors": {                         // REQUIRED iff patch is non-null; see § 15 + critic.md
        "state_io": {
          "reads":              ["history"],
          "writes":             ["planning"],
          "grants_required":    ["ag_replan_gate"]
        },
        "llm_calls": [
          {
            "purpose":                "stop verifier",
            "model_profile":          "gpt-5-mini",
            "max_tokens":             2000,
            "reasoning_aware":        true,
            "fallback_on_empty":      "preserve baseline (verifier returns BYPASS)",
            "fallback_on_parse_error":"preserve baseline",
            "fallback_is_inert":      true
          }
        ],
        "globally_firing_nodes_touched": ["build_options"],
        "mechanism_fire_predicate":      "self._self_log('replan_fired', True) called on every replan_gate.forward() entry; expect replan_fired=True on ≥ 6/27 (c)-bucket eps"
      },
      "block_override": {                       // OPTIONAL — present iff THINK is rebutting a prior CRITIC BLOCK
        "critique_id":  "critique_iter_3_spec_iter_3_A_round_1",
        "rebuttal":     "This spec differs from exp_iter3_001 because <X>; the access_grant is added at intent §A.4 (line ...). The structural feature CRITIC matched does not hold here."
      },
      "budget_hint": { "ep_count": 30, "gpu_min": 8 }
    }
    // additional specs (e.g. spec_iter_3_B, spec_iter_3_C) follow the same shape;
    // each is independent — its own patch, overlay, eval, critique, distill.
  ]
}
```

**Per-spec field semantics** (each entry in `specs[]`):

Field `spec_id` is unique within the iter. Format
`spec_iter_{n}_{LETTER}` where `LETTER` is `A`, `B`, `C`, ... in the
order THINK emitted them. CRITIC, EXPERIMENT, DISTILL, and every
per-spec staged artifact (`critique_<spec_id>.json`,
`eval_metadata_<spec_id>.json`, `active_workspace_<spec_id>/`,
`debug_log_<spec_id>.md`) refer to a spec by this id. **K = 1 still
uses the suffix** — schema is uniform regardless of breadth.

Field `passes` is REQUIRED, ≥ 1. The Python multi-spec wrapper
(`loop.md § 3c`) reads it to decide how many eval draws to issue
against the same ep set; per-pass results are aggregated by the
wrapper into `metrics_digest.mean_sr / sd_sr / robust_sr` (and the
raw `per_ep_success` matrix is kept for forensics). `passes = 1` is
the default; THINK chooses higher when the expected signal is small
relative to the eval's noise floor.

Field `risk_vectors` is **REQUIRED whenever `patch` is non-null**.
It is the structural surface CRITIC validates (`critic.md`); THINK
must enumerate, not gloss, the parts of the design most prone to
recurrence of past pathologies. Each sub-block has fixed semantics:

- `state_io.{reads,writes}` — every `gs.read(...)` / `gs.write(...)`
  field name the proposed code touches; `grants_required` is the
  list of `access_grants` entries the proposed graph JSON edit
  MUST add (one per new node whose code touches `ctx.graph_state`).
- `llm_calls` — one entry per `llm_complete` / `vlm_complete` /
  `llmCall` introduced or modified. `reasoning_aware` is `true` iff
  the model_profile is in the reasoning-tokens family (gpt-5-*,
  o1-*, o3-*); for those, `max_tokens` must be sized for reasoning
  + visible content (typically ≥ 2000) or the call risks empty
  visible output. `fallback_is_inert` is `true` iff every error /
  empty / parse-error path returns to baseline behavior (no decision
  made); `false` iff a fallback path produces an output
  indistinguishable from a real decision (the iter_2/iter_7
  fallback-becomes-mechanism pathology).
- `globally_firing_nodes_touched` — paths to nodes whose `forward`
  fires on every step of every ep (e.g. `build_options`, `observe`,
  `render_prompt`, `parse_action`). When non-empty, `expected_signal`
  SHOULD include a counter-check against the (S)-bucket of
  baseline-success eps (cross-bucket churn risk from iter_12).
- `mechanism_fire_predicate` — a sentence specifying *how* the
  mechanism will be observable in `inner_log` / `_self_log` /
  metrics. Required to make CRITIC's mechanism-fire-gate check
  concrete and to prevent silent-no-op pathologies (iter_3,
  iter_9).

Field `block_override` is OPTIONAL and is the THINK rebuttal channel
against a CRITIC `BLOCK` verdict on **this specific spec**. When
present, the spec proceeds to EXPERIMENT even though CRITIC blocked
it; both the original critique and the override are recorded in the
iter dir, and DISTILL evaluates whether the override was justified
against the eval outcome (`critic.md` § Notes). Without
`block_override`, a CRITIC `BLOCK` on round 2 causes **just this
spec** to commit with `outcome_class="critic_block"` (eval skipped);
sibling specs in the same iter proceed independently — one spec's
BLOCK does not stop another's eval.

Field `patch` is a **change spec**, not a typed op list: `intent` is
prose (what to change and why), `targets` lists the workspace-prefixed
files the change touches. The EXPERIMENT phase (`loop.md § 3c`,
inline — myloop has no implementer skill) spawns an editing sub-agent
that reads `intent` and edits the seeded `targets` natively — the
typed `graph_edits` op DSL was retired 2026-05-20.

Field `patch` MAY be `null` — for no-patch probes (re-running an
existing design on a new ep subset, gathering data for a
distillation step, etc.).

Field `kind` is a coarse size/intent label: `probe` (≤ ~10-ep
ad-hoc), `perf` (full paper-comparable), `custom` (a THINK-composed
targeted subset — e.g. a failure-mode subset, see § 6 — and the
default per-iter measurement tier when sized ≥ ~30 ep). It does not
select the eval — `eval_profile.name` does; `kind` is for the iter
record and escalation accounting. `smoke` is intentionally NOT a
`kind`: a 3-ep smoke is the EXPERIMENT apply-step's correctness gate,
never an iter's measured experiment.

Field `eval_profile.name` MUST exist as a key in
`experiment_design.yaml`. If THINK wants a not-yet-defined profile,
it must first write that entry into `experiment_design.yaml` in the
same think turn — including a THINK-composed targeted subset built
from raw-log inspection.

Field `intervention_axis` (REQUIRED, per-spec) is the single
intervention-taxonomy axis this spec's `patch` targets — one of the
axis names in `search_space.md § Axes` (`prompt-content` / `topology`
/ `control-flow` / `action-space` / `observation-pipeline` /
`state-memory` / `model-component-config`, plus any REFLECT has
added). For a no-patch probe (`patch == null`) set it to the axis of
the design the probe re-runs, or `"none"` if the probe is pure data
collection on the frozen baseline. THINK self-labels each spec; the
loop copies it into `record.json`'s `specs[*].intervention_axis`;
REFLECT reads the per-spec labels to measure search-space coverage.

It is the field that makes "THINK ruts in one subspace" detectable —
see `search_space.md` (§ 12) and `reflect.md`. The axis-jump rule
(`proposer.md`) operates at iter granularity: when the last K
committed iters all touched only one `intervention_axis` (across all
their specs combined), the next iter must leave that axis on at
least one spec, or carry a written rebuttal. Multi-spec breadth
within a single iter is itself an axis-jump tool — THINK can probe
two axes in one iter rather than serially across iters.

---

## 9. Directory layout for one vN

```
outputs/design_runs/myloop/{graph}/v{N}/
├── goal.md                     # USER-AUTHORED, read-only to orchestrator
├── constraints.md              # USER-AUTHORED, OPTIONAL — hard rules every phase must respect (§ 11)
├── knowledge.md                # pure fact, append-only
├── search_space.md             # intervention-space map + per-REFLECT coverage, append-only (§ 12)
├── experience.jsonl            # lessons learned, append-only
├── hypotheses.jsonl            # open conjectures, mutable (line-delete on resolve)
├── experiment_design.yaml      # eval configs, orchestrator-extensible
├── tools/                      # orchestrator-authored *.py utilities
│   └── *.py
├── trace.md                    # rollup — one-row-per-iter history table (§ 13)
├── lineage.md                  # rollup — one-section-per-iter narrative (§ 14)
├── SUMMARY.md                  # rollup — run summary, written at termination
├── iteration/                  # one subdir per committed iter (iter_0 = baseline)
│   ├── iter_0/
│   │   ├── record.json                                 # dense IterRecord (§ 5)
│   │   ├── spec.json                                   # ExperimentSpec envelope (§ 8) — specs[] list
│   │   ├── think_trace.md                              # THINK sub-agent forensic trace (one per iter — THINK is per-iter)
│   │   ├── reflection_trace.md                         # REFLECT sub-agent forensic trace (only iters where REFLECT fired)
│   │   ├── critique_<spec_id>.json                     # CRITIC output (one per spec with patch != null; § 15)
│   │   ├── critique_<spec_id>_round_1.json             # round-1 critique preserved if round 2 ran (per spec)
│   │   ├── critique_trace_<spec_id>.md                 # CRITIC sub-agent forensic trace (per spec)
│   │   ├── eval_metadata_<spec_id>.json                # one per spec; run_ids[] (len=passes) + aggregate metrics + per-ep
│   │   ├── distill_<spec_id>.json                      # per-spec DISTILL verdict (merged into record.json's specs[*].distill); omit on SKIP_*
│   │   ├── distill_trace.md                            # DISTILL sub-agent forensic trace (single spawn covers all K specs)
│   │   ├── active_workspace_<spec_id>/                 # one overlay per spec with patch != null (independent overlays — no shared-overlay multi-patch)
│   │   ├── debug_log_<spec_id>.md                      # one apply-step retry history per spec with patch != null
│   │   └── multi_spec_eval_log.md                      # Python wrapper's wave plan + worker allocation + timings (loop.md § 3c)
│   ├── iter_1/
│   └── ...
├── .staging/iter_{n}/          # transient, mv'd to iteration/iter_{n}/ on commit
│   ├── spec.json               # ExperimentSpec envelope from THINK
│   ├── think_trace.md
│   ├── reflection_trace.md
│   ├── critique_<spec_id>.json (one per spec with patch != null)
│   ├── critique_<spec_id>_round_1.json (if round 2 ran on that spec)
│   ├── critique_trace_<spec_id>.md
│   ├── eval_metadata_<spec_id>.json (one per spec)
│   ├── distill_<spec_id>.json (one per spec)
│   ├── distill_trace.md
│   ├── debug_log_<spec_id>.md (one per spec with patch != null)
│   ├── active_workspace_<spec_id>/ (one per spec with patch != null)
│   ├── multi_spec_eval_log.md
│   └── (other artifacts the orchestrator chose to stage)
└── .loop_state/                # bookkeeping for resume / termination
    └── ...
```

**Suffix convention**: every per-spec artifact uses the literal
`spec_id` string (e.g. `spec_iter_5_A`) as suffix. The suffix is
applied even at K=1 — schema uniformity is preferred over
filename-length savings.

No `archive.jsonl` — that is the ADAS-shape concept, deliberately not
part of myloop (working memory is the **eleven** vN-scoped files above:
goal / constraints / knowledge / search_space / experience /
hypotheses / experiment_design / tools / SUMMARY / trace / lineage —
not a flat list).
`constraints.md` is optional; when absent there are simply no hard
rules and every phase reasons unrestricted.

---

## 10. What gets atomic-committed each iter

The iter has four phases: THINK, CRITIC (per spec), EXPERIMENT (one
wrapper call covering K specs), DISTILL (one spawn covering K specs),
then commit. Eager writes to working-memory files happen during
THINK and DISTILL, NOT at commit time. The atomic commit is just the
staging promotion + record write:

1. `mv .staging/iter_{n}/` → `iteration/iter_{n}/`
2. Build `IterRecord`:
   - top-level `think` / `reflect` / `cost` from the iter's shared
     phase artifacts.
   - `specs[]`: for each spec_id in `spec.json` envelope,
     - merge `critic` block from `critique_<spec_id>.json` if present
       (else omit — patch was null for that spec),
     - merge `experiment` block from
       `eval_metadata_<spec_id>.json`,
     - merge `distill` block from `distill_<spec_id>.json` if present
       (else omit).
   - `iter_summary`: rollup computed from `specs[]` per § 5 table.
3. Write `iteration/iter_{n}/record.json`.
4. Update `.loop_state/last_committed_iter = n`.

Eager writes already on disk by commit time (from THINK and DISTILL):

- `knowledge.md` — appended bullets (by either phase)
- `hypotheses.jsonl` — appended (by either) + line-deleted (by DISTILL)
- `experience.jsonl` — appended (DISTILL only)
- `experiment_design.yaml` — appended (THINK only)
- `tools/*.py` — added (THINK only)

All 4 commit-time steps must succeed atomically; failure → abort, leave
last-known-good state, increment `consecutive_skips` in `.loop_state/`.
Eager-writes by THINK / DISTILL stay on disk regardless.

---

## 11. `constraints.md` (markdown, user-authored, OPTIONAL, read-only to orchestrator)

Hard rules that every phase (THINK / DISTILL) must respect. Distinct
from `goal.md` (direction) and `knowledge.md` (facts) — these are
non-negotiable invariants. When the file is absent there are no hard
rules and every phase reasons unrestricted.

Format: free-form markdown bullets, each starting with `MUST` or
`MUST NOT`. Group by section.

```markdown
# Constraints

## Model choice
- MUST keep `planner_llm.config.profile = "gpt-5-mini"` across all iters
  (gpt-4o over-confidence-stops on ReAct prompts; gpt-5-nano under-performs).
- MUST keep `planner_llm.config.temperature = 1.0` (gpt-5 family
  requires temperature == 1.0; litellm raises UnsupportedParamsError
  otherwise).

## Topology
- MUST keep exploded topology (observe / update_map / build_options /
  render_prompt). MUST NOT collapse back into a monolithic `plan_step`.
```

**Enforcement**:

- **THINK / DISTILL prompt**: when `constraints.md` exists the loop
  injects it into each sub-agent's prompt with a hard-rule preamble:
  "the following constraints are NOT negotiable; if you cannot
  propose a useful experiment / lesson under them, return
  `SKIP_THINK_EMPTY` / `SKIP_DISTILL_EMPTY` rather than violating".
- **Proposer §4 validate**: simple string-match guard — if
  `spec.patch.intent` describes a change a constraint forbids (e.g.
  constraint mentions `planner_llm.config.profile` and the `intent`
  talks about swapping the planner profile), report
  `SKIP_INVALID_SPEC` with the violated rule. Coarse-grained; only
  catches obvious cases.
- **EXPERIMENT apply-step**: NOT involved (it owns the edit whitelist
  for filesystem boundaries, not semantic invariants).

**Deferred**: structured `constraints.yaml` with explicit jsonpath
validators that loop can enforce hard. Out of scope for now.

**Lifecycle**: never edited by orchestrator. User edits it between
iters if they want to add / loosen rules. Changes take effect on the
next THINK / DISTILL spawn.

**Bootstrap merge** (when `vN/constraints.md` does not exist at loop
invoke): loop merges up to four layers in order, separated by
`## --- from <source> ---` audit-trail headers, into the final
`vN/constraints.md`. Layers:

| Layer | Source | Always present? |
|---|---|---|
| 1 | `.claude/commands/architect/myloop/data/constraints/common.md` | yes (shipped with skill) |
| 2 | `.claude/commands/architect/myloop/data/constraints/{graph}.md` | only if a per-graph file exists |
| 3 | `--cons-file <path>` flag at invoke | only if flag passed |
| 4 | `--constraints "<text>"` flag at invoke | only if flag passed |

If all four layers are empty (no common.md present, no graph file, no
flags), `vN/constraints.md` is NOT created — equivalent to "no hard
rules". Pre-authored `vN/constraints.md` (user edited it before loop
invoke) is left untouched; merge only runs on first bootstrap.

The merge is by string concatenation; there is no rule de-duplication,
override semantics, or precedence resolution. If layer 4 contradicts
layer 1, sub-agent reasoning (soft enforcement) decides — typically
the layer-4 author intended an exception and phrased it that way
(e.g. "EXCEPTION FOR THIS RUN: layer-1 model-fix is relaxed; gpt-4o
permitted for hyp_X test"). Structured override semantics may be
added later; the merge is concat-only today.

---

## 12. `search_space.md` (markdown, working-memory, append-only)

The explicit map of the **intervention space** the run is searching.
It exists to prevent a structural failure mode: single-pass THINK,
under the "must produce an ExperimentSpec" contract and a
self-reinforcing hypothesis chain, ruts in whichever
`intervention_axis` its first iters picked, refines that one lever
class to exhaustion, then returns `SKIP_THINK_EMPTY` claiming *global*
saturation — when only one axis was ever searched. `search_space.md`
gives THINK a representation of the whole space; the REFLECT phase
(`reflect.md`) maintains it.

Bootstrap-seeded from `data/seed_search_space.md`. Two parts:

**`## Axes`** — the intervention taxonomy. Each axis: a name, a
one-line definition, a "why distinct" note, example interventions.
The seed ships seven (`prompt-content`, `topology`, `control-flow`,
`action-space`, `observation-pipeline`, `state-memory`,
`model-component-config`). REFLECT MAY append a new axis when the
graph / codebase exposes an intervention kind none captures; existing
axis bullets are never rewritten (append-only, `(added: iter_n)`).
`ExperimentSpec.intervention_axis` (§ 8) MUST name an axis present
here.

**`## Coverage`** — one `## reflection_N` section per REFLECT spawn,
append-only. Schema of a section:

```markdown
## reflection_2  (trigger: concentration · after iter_7 · audited iter_0..iter_7)

| axis | status | iters | verdict |
|---|---|---|---|
| prompt-content        | exhausted | iter_1..7 | 5 sub-levers, all net-neg-to-neutral on the per-iter custom tier; exp_iter1_001..exp_iter7_001 |
| topology              | untouched | —         | no committed intervention |
| control-flow          | untouched | —         | — |
| action-space          | untouched | —         | (note: iter_1/2/7 edited build_options TEXT — that is prompt-content, not action-space) |
| observation-pipeline  | partial   | iter_5/6  | one realisation (Map-text enrichment) tested as prompt-content; the perceive-side pipeline itself untouched |
| state-memory          | untouched | —         | — |
| model-component-config| closed    | —         | sampling-param realisations forbidden by constraints.md (fixed model/temp) |

### Frontier (ranked)
1. **topology** — mechanism: an ensemble / verifier node changes behaviour while leaving every prompt byte-identical, so it escapes the prompt-content fragility. Ceiling: only recovers variance-type failures. Constraints: none.
2. **control-flow** — mechanism: ... Ceiling: ... Constraints: ...
3. ...

### Axes extensions this reflection
(none)  |  - <new-axis>: <definition> — <why distinct> (added: iter_N)
```

`status` ∈ `{untouched | partial | exhausted | closed}` — `closed`
means every realisation the axis admits is forbidden by
`constraints.md`. The `### Frontier` is the ranked advisory the next
THINK consumes; `proposer.md`'s axis-jump rule makes THINK
accountable to it. REFLECT returns `SPACE_EXHAUSTED` (→ loop
`SATURATED`) only when every axis is `exhausted` or `closed`.

**Writer**: REFLECT only (eager-append, like `knowledge.md`).
**Readers**: REFLECT (prior coverage), proposer / THINK (current
frontier + axis list). **Lifecycle**: grows one section per REFLECT;
never rewritten.

---

## 13. `trace.md` (markdown, rollup, regenerated each commit)

A one-row-per-committed-iter scannable history table — the answer to
"what has this run done so far, at a glance". It is a **pure
projection** of `iteration/iter_*/record.json`: it carries no data not
already in the IterRecords. The loop regenerates it wholesale at every
atomic commit (`loop.md § 5` step 3) and once more at termination
(`§ 6`). Because it is derived, `record.json` is always the source of
truth, `trace.md` is never part of the atomic-commit rollback set, and
a stale or missing `trace.md` is healed by the next regeneration.

```markdown
# Trace — myloop {graph} v{N}

| iter | K | axes               | kinds       | patches | best metric   | outcome   | hyp Δ | REFLECT           | milestone |
|------|---|--------------------|-------------|---------|---------------|-----------|-------|-------------------|-----------|
| 0    | 1 | none               | perf        | —       | success 0.42  | ok        | +0/-0 | —                 | baseline established |
| 1    | 1 | prompt-content     | custom      | yes     | success 0.41  | refuted   | +1/-1 | —                 | prompt tweak net-neutral |
| 3    | 2 | prompt-content, topology | custom, custom | 2/2 | success 0.49  | mixed     | +1/-2 | r2: FRONTIER_OPEN | A refuted; B confirmed (verifier) |
```

Columns, each rendered from one IterRecord (§ 5):

| Column | Source |
|---|---|
| `iter` | `iter` |
| `K` | `iter_summary.n_specs` (the number of specs in this iter; `1` for the K=1 case) |
| `axes` | comma-joined `specs[*].intervention_axis` (deduplicated; preserves THINK's emission order across specs) |
| `kinds` | comma-joined `specs[*].spec_kind` (deduplicated) |
| `patches` | for K=1: `yes` / `—`. For K>1: `n/K` where n = count of specs with `experiment.patch_applied=true` |
| `best metric` | primary metric of `specs[best_spec_id].experiment.metrics_digest.mean_sr` (`—` if all specs crashed / had no run) |
| `outcome` | `iter_summary.outcome_class` (`confirmed` / `mixed` / `refuted` / `inert` / `critic_block` / `crash`) |
| `hyp Δ` | `+sum(specs[*].distill.new_hypotheses)/-sum(specs[*].distill.resolved_hypotheses)` (`+0/-0` if no spec had a distill block) |
| `REFLECT` | `{reflect.reflection_id}: {reflect.status}` if a top-level `reflect` block is present, else `—` |
| `milestone` | `iter_summary.milestone_after`, one line, truncated |

---

## 14. `lineage.md` (markdown, rollup, regenerated each commit)

A one-section-per-committed-iter narrative — the human-readable
"what happened and why" companion to `trace.md`'s table. Same
provenance contract: a pure projection of the IterRecords, regenerated
wholesale each commit + at termination, never the source of truth.

```markdown
# Lineage — myloop {graph} v{N}

## iter_3 — K=2 — mixed
parent: iter_2

THINK: hyp_4 (long-instr early-stop) probed on two axes in parallel —
  A (prompt-content: lower temperature) and B (topology: verifier node).

### spec_iter_3_A — prompt-content — refuted
CRITIC (round 1): OK · no refuted reference matched
EXPERIMENT: kind=custom · patch applied · passes=1 · run 20260515_143112 ·
  mean_sr 0.20 · outcome ok
DISTILL: verdict refuted — temperature change net-neutral on long-instr subset

### spec_iter_3_B — topology — confirmed
CRITIC (round 1): OK
EXPERIMENT: kind=custom · patch applied · passes=3 · run_ids
  [20260515_143510, 20260515_145001, 20260515_150702] ·
  mean_sr 0.31, sd 0.018, robust 0.28 · outcome ok
DISTILL: verdict confirmed — verifier node lifts SR (resolved hyp_5; new hyp_8)

ITER SUMMARY: best=spec_iter_3_B (0.31) ·
  knowledge +1 bullet (Graph: {graph}) ·
  milestone → iter_4 should test composition of B with the temperature drop
REFLECT (concentration): reflection_2 → FRONTIER_OPEN ·
  frontier: control-flow, state-memory
```

Each section is rendered from one IterRecord:

- **heading** — `## iter_{n} — K={iter_summary.n_specs} — {iter_summary.outcome_class}`
- **parent** — implicit-linear lineage: `iter_{n-1}` (`—` for `iter_0`)
- **THINK** line — `think.rationale`
- For each `spec` in `record.specs[]`, one `### {spec_id} — {intervention_axis} — {distill.verdict|"no-distill"}` sub-section with:
  - **CRITIC** line — final-round verdict + reference summary; omitted if `critic` block absent for the spec
  - **EXPERIMENT** line — `spec_kind` / `patch_applied` / `passes` /
    `run_ids` / digest of `metrics_digest` (mean_sr, sd_sr, robust_sr when passes>1; just score when passes=1) / `outcome_class`
  - **DISTILL** line — verdict + new/resolved hypotheses one-liner.
    Collapses to `DISTILL: (skipped — SKIP_DISTILL_EMPTY)` when the
    spec's `distill` block is absent.
- **ITER SUMMARY** — `iter_summary.best_spec_id` + `best_score` +
  cross-spec knowledge digest + `milestone_after`.
- **REFLECT** line — `reflect.trigger` / `reflect.status` /
  `reflect.frontier_axes`. Omitted entirely when the record has no
  top-level `reflect` block.

For K=1 iters, the per-spec sub-section header may be elided and the
spec lines folded into the iter section to keep narrative concise —
renderer decision, schema does not require the sub-header.

---

## 15. `iteration/iter_{n}/critique_<spec_id>.json` (one per CRITIC fire, JSON object)

Produced by CRITIC (`critic.md`), consumed by the loop's verdict
dispatch (`loop.md § 3b.5`) and by DISTILL (which evaluates whether
the prediction was right). Atomic-promoted into the iter dir on
commit. **One file per spec on which CRITIC fired** (i.e. for each
spec with `patch != null`). Filename includes the `spec_id` so multi-
spec iters keep critique provenance unambiguous; K=1 iters still use
the suffixed filename for schema uniformity. Format = `Critique`:

```json
{
  "critique_id":   "critique_iter_5_spec_iter_5_A_round_1",
  "iter":          5,
  "spec_id":       "spec_iter_5_A",                       // the spec this critique vets
  "critic_round":  1,                                     // 1 or 2; round 2 fires only on round-1 REVISE/BLOCK
  "ts":            "2026-05-23T18:42:11",
  "verdict":       "OK | WARN | REVISE | BLOCK",
  "verdict_summary": "<one sentence — why this verdict>",
  "predicted_failure_modes": [
    {
      "pathology_tag":             "access_grant_missing",      // short structural label (see common tags below)
      "reference_experience_ids":  ["exp_iter3_001", "exp_iter7_002"],
      "specific_check":            "specs[A].patch.targets includes workspace/graphs/mapgpt_mp3d.json. The proposed graph edit (intent §A.4) adds node 'replan_gate' which patch.intent §B.2 says reads gs.read('history') — but the graph JSON edit does NOT add an `access_grants` entry for replan_gate. This is structurally identical to exp_iter3_001.",
      "predicted_outcome":         "replan_gate's `_self_log('fired', True)` will be 0/27 across the eval (the gate's gs.read returns None due to defensive `if gs:` short-circuit; the gate body never executes).",
      "severity":                  "info | minor | major | critical",
      "confidence":                0.92
    }
  ],
  "recommended_action": "<one line — what THINK should do (e.g. add ag_replan_gate to the graph JSON, then re-submit)>",
  "validator_notes":    []
}
```

**Field semantics**:

- `critique_id` is unique within this run. Format
  `critique_iter_{n}_{spec_id}_round_{r}`. Stable across mv.

- `spec_id` is the id of the spec being vetted; equals the
  `spec_id` field of the corresponding entry in
  `iter_{n}/spec.json`'s `specs[]` envelope.

- `verdict` ∈ {`OK`, `WARN`, `REVISE`, `BLOCK`} — see `critic.md`
  for full semantics. Round 2 verdicts CANNOT be `REVISE` (the
  validator auto-downgrades to `WARN`).

- `predicted_failure_modes` may be empty when `verdict = "OK"`. For
  `WARN` / `REVISE`, at least one entry. For `BLOCK`, at least one
  entry **and every entry** carries non-empty
  `reference_experience_ids`, `specific_check`, and
  `predicted_outcome` — these are the three hard conditions for
  BLOCK (`critic.md`). The validator (`critic.md` Step 5)
  auto-downgrades the verdict if any are missing.

- `confidence` is the sub-agent's self-reported probability that
  the predicted_outcome will hold. Not statistically calibrated;
  used as a tie-breaker by DISTILL when computing critic accuracy
  stats.

- `validator_notes` is appended by the loop's CRITIC validator (not
  by the sub-agent) when it auto-downgrades a verdict. Format:
  `"BLOCK downgraded to REVISE — predicted_failure_mode 0 missing reference_experience_ids"`.

**Common `pathology_tag` values** (the recurring failure modes
observed in `experience.jsonl`; expand as new ones surface):

| Tag | Pathology | Reference template entries |
|---|---|---|
| `access_grant_missing` | new node reads `ctx.graph_state` but graph JSON has no `access_grants` entry → silent no-op | iter_3, iter_7 |
| `reasoning_model_max_tokens` | reasoning-aware model with low `max_tokens` → empty visible content | iter_7 |
| `fallback_becomes_mechanism` | error / empty / parse-error fallback path produces a non-trivial decision indistinguishable from the intended mechanism | iter_2, iter_7 |
| `silent_inert_mechanism` | gate condition too strict / lookup window too narrow → `_self_log('fired', True)` never triggers | iter_9 |
| `cross_bucket_churn_unguarded` | `globally_firing_nodes_touched` is non-empty but `expected_signal` has no (S)-bucket counter-check | iter_12 |
| `state_pollution_selfmatch` | node writes a field that is later regex-parsed by the same or another node, and the write format collides with the read regex | iter_1 |
| `passes_below_required` | `spec.passes < profile.passes_required` — eval cannot compute `sd_sr`, single-draw score is indistinguishable from noise. Verdict: `REVISE`. References: the profile's `passes_required` field (derived from N_eps < 119) | (derivation; no exp_id) |
| `predicted_delta_within_noise_floor` | `profile.baseline` is locked AND `|expected_target − baseline.mean_sr| < 2·baseline.sd_sr` — even with multi-pass, the lift will likely be reported inconclusive. Verdict: `WARN`. References: the profile's `baseline.sd_sr`. | (derivation; no exp_id) |

THINK is encouraged to use these tags in `experience.jsonl`
entries as it closes hypotheses, so CRITIC's pattern matching can
key on them rather than free-text narrative.

**Lifecycle**: written once per fire (round 1 + at most one round 2).
Never edited after write. The iter dir holds the round-1 critique
under `critique_<spec_id>.json`; if round 2 fired on that spec, the
round-2 critique overwrites `critique_<spec_id>.json` and the round-1
critique is preserved at `critique_<spec_id>_round_1.json` for
forensics. Per-spec independence: round 2 may fire on spec A and not
on spec B in the same iter.
