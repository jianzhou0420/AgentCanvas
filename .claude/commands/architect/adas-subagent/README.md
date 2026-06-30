# adas-subagent — coding-agent-era ADAS port to AgentCanvas

> **Framing note.** Upstream ADAS (Hu et al. 2024) lives in the
> `reasoning module = stateless LLM call` paradigm — the only option in
> 2024. adas-subagent proposes `reasoning module = tool-augmented Claude
> conversation with context management` as the coding-agent-era
> replacement. This is the primary contribution; "faithful 1:1
> reproduction" is **not** the goal. Several paradigm-independent
> elements (3-call Reflexion structure, `Reflexion_prompt_1 / 2` text,
> `bootstrap_CI` algorithm, archive contract) are preserved **verbatim**
> — the "verbatim" labels in tables below are accurate at that
> per-element level. Phrases like "preserves upstream's X" should be
> read as "structurally accurate translation of paradigm-independent
> element X," not as a 1:1 fidelity claim about the overall pipeline.

Port of Hu et al. 2024, **"Automated Design of Agentic Systems"**, to the
AgentCanvas / coding-agent setting. The meta-LLM iteratively proposes
graph + nodeset edits against a target graph; each generation's design
intent, structural snapshot, and bootstrap-CI fitness are appended to an
archive that drives the next generation's prompt.

> **Structural delta vs upstream**: each of upstream `search.py`'s
> `get_json_response_from_gpt_reflect` calls (the 3 Reflexion rounds in
> proposer and the retry-on-smoke-fail call in implementer) becomes one
> `Agent({subagent_type: "general-purpose", ...})` spawn — a fully
> tool-augmented independent Claude conversation. This preserves ADAS's
> per-call sampling diversity (3 independent samples per propose, not
> one autoregressive trace) while gaining the "Claude is the meta-LLM"
> convenience (no external API key, full tool access on every sample).
> See § 1 ("One iter = one Claude conversation + N independent
> sub-agent samples") for the contract, and the "What's intentionally
> adapted" table for the two adaptations sub-agent invocation forces
> (no pinned stateless-API params; `role: assistant` history rendered
> as text in sub-agent prompts).

> **Cross-variant contract**:
> `.claude/commands/architect/_common/files-contract.md` defines the
> shared iteration files (run-dir layout, resolve protocol, `{graph}.yaml`
> schema, edit whitelist, backend API). adas-subagent reuses it for
> evidence storage; this README only documents the **deltas** specific
> to adas-subagent.
>
> This variant's concrete file set is declared in
> `config.yaml § manifest` — every file it writes, classified into a
> global file-type, with purpose / schema / access (per
> `_common/files-contract.md § 4`). That manifest is the single source
> of truth for file *identity*; this README narrates only rationale.

## Three structural contracts (deltas vs adas v1)

### 1. One iter = one Claude conversation + N independent sub-agent samples

The `loop` skill owns the **main Claude conversation** for an entire
iter; that main conversation invokes `proposer`, `implementer`, and
`evaluator` as same-conversation phases (no nested skill conversations).
The shared `msg_list` survives across all three phases.

**Inside `proposer` and `implementer`**, each of upstream
`search.py`'s `get_json_response_from_gpt_reflect` calls becomes one
`Agent({subagent_type: "general-purpose", ...})` invocation — a
**fully tool-augmented Claude sub-agent**, not a stateless API call
and not a Claude thinking-pass. Specifically:

- **proposer**: 3 sub-agent spawns (propose / Reflexion_1 /
  Reflexion_2), each seeing the cumulative `msg_list` rendered as
  text. Three spawns = three independent Claude samples — this is
  the property that preserves upstream's per-call resampling structure
  within the modernized paradigm.
- **implementer**: up to `retry_max=3` (config) fresh debugging
  sub-agents on Smoke runtime failure. Each retry is an independent
  `Agent(general-purpose)` spawn that sees the **proposal** + **last
  failure trace** as read-only text (no `msg_list` continuation,
  no `debug_thought` Reflexion field — pure coding-agent debug).
  Triggered only by runtime failure (crash / incomplete / step=0 /
  malformed metric); low or zero metric values are archive data,
  not retry triggers.
- **evaluator**: no sub-agent — it's pure infrastructure (run 100ep,
  compute bootstrap CI).

Maximum sub-agent invocations per iter = 3 propose + 3 debug retry
= 6. Minimum = 3 (propose with Smoke passing first try).

**Why sub-agents and not Claude's own thinking-pass?** A single
Claude pass is one autoregressive sample no matter how many "rounds"
it narrates internally; KV cache is shared across the rounds, so
"diversity from independent sampling" is lost. Upstream ADAS's
3-call Reflexion gets its quality-diversity behavior from 3
**independent samples** at temperature 0.8 — `Agent(...)` spawns
restore this property because each spawn is a fresh Claude. Tool
access on each sub-agent is the second half of the contract: this
upgrades "stateless gpt-4o call" to "tool-augmented Claude
conversation" (the modernization framing), which only pays off if
sub-agents actually use their tools (Read, Grep, Bash) to ground
proposals in log.jsonl / archive / source.

This matches upstream `search.py`'s single-`msg_list` semantics
(`search.py:183–223`) at the textual-content level; the
`role: assistant` tagging upstream gets is approximated by rendering
prior outputs as text inside the sub-agent's user prompt (Agent tool
takes only a single prompt string).

Adas v1 split proposer/implementer/evaluator across separate Claude
conversations AND collapsed each Reflexion round into a single
thinking-pass — losing both the reflection chain and the sampling
diversity. Adasv3 fixes both: one conversation envelope (for
msg_list continuity) AND independent sub-agent spawns (for
diversity).

### 2. archive.jsonl is independent of iter dirs

Located at `outputs/design_runs/adas-subagent/{graph}/v{N}/archive.jsonl`. **Not
derived from iter dirs.** Each entry is self-contained
(`graph_summary` is rendered structurally at append time). The iter
dir holds evidence (graph snapshot, metrics, exports) for
reproducibility; archive.jsonl is the meta-LLM's curated working
memory.

Entry schema (see HTML "Archive contract" §):

```json
{
  "generation": int | "initial" | "reference",
  "iter_id": "iter_N" | null,
  "name": "<meta-LLM-given>",
  "thought": "<verbatim from proposer's final LLM output>",
  "graph_summary": { "nodes": [...], "wires": [...], "loop": {...} },
  "diff_narrative": "<plain-text What-changed>",
  "fitness": "95% Bootstrap CI: (lo%, hi%), Median: x%" | null
}
```

Pre-append cleanup: strip `reflection` field (verbatim upstream
`search.py:232–235`; `debug_thought` is no longer in the schema —
implementer's retry sub-agents edit the overlay natively and return
`{edit_summary, extra_targets}`).

### 3. Two-tier evaluation

- **Smoke** (5 eps, ~10 min) — runs inside `implementer`'s debug retry
  loop. Cheap gate that answers "does the change even run?". Retry is
  triggered by runtime-correctness failures only — crash / incomplete
  / step=0 / malformed_metric / edit_error. Low or zero metric values
  do NOT trigger retry (that is archive data, not a failure — see
  `_common/implementer.md` § 3d).
- **Performance** (100 eps, ~1–2 h) — `evaluator` runs the eval and
  writes neutral metrics (`acc_list`, `primary_metric_value`,
  `secondary_metrics`) to staging. **`bootstrap_CI` → `fitness_str`
  lives in `loop.md`'s Atomic Writer**, not in evaluator — evaluator
  is method-free. **No retry on low fitness** — too expensive
  (3×~2h = ~6h/gen infeasible). Low-fitness entries enter archive
  as-is; meta-LLM learns next generation.

Smoke and Performance each need their own `{graph}.yaml`
profile entry:
- `smoke_<graph>` — `episode_count=5`, `worker_count=1`, fixed
  episode_indices (deterministic for debug reproducibility)
- `perf_<graph>` — `episode_count=100`, `worker_count=4–8`

## 5 skills

| Skill | Role | Invoked by |
|------|------|-----------|
| `/architect:adas-subagent:understand` | P0 — load context (files-contract + concept/contract docs + graph state). Read-only. | manual (or `loop` P0) |
| `/architect:adas-subagent:loop` | Orchestrator. Owns the per-iter Claude conversation. Drives the 3 worker skills, Atomic Writer, archive append, termination. | user (entry point) |
| `/architect:adas-subagent:proposer` | 3-call Reflexion (analyze helper → propose → R1 → R2). Outputs staging `proposal.md`. | `loop` (each iter) |
| `/architect:adas-subagent:implementer` | native edit (sub-agent realizes the `# Change` spec) → Smoke → classify → debug retry ≤3. SKIP on exhaust. | `loop` (each iter, after proposer) |
| `/architect:adas-subagent:evaluator` | Method-free eval runner — `experiment:run` perf eval → neutral staging `metrics.json` (`acc_list` + `primary_metric_value` + `secondary_metrics`). bootstrap_CI / `fitness_str` are added later by loop's Atomic Writer (via `lib/helpers.py:bootstrap_confidence_interval`). | `loop` (each iter, after implementer; also iter_0 baseline) |

## Library + data

Load-bearing Python helpers and seed data live alongside the skill
markdown — skills import these instead of inlining the logic.

| Path | Purpose |
|------|---------|
| `lib/helpers.py` | `bootstrap_confidence_interval` (fitness_str), `render_graph_summary` (archive entries), `build_acc_list_from_export`, `write_evaluator_staging`, `atomic_commit` (Atomic Writer 3-step transaction), `append_reference_seed`, `update_trace_md`, `append_lineage_md`, `render_backend_llm_cheat_sheet` (proposer step 3.5) |
| `data/reference_seeds.json` | The 7 ADAS reference patterns (COT, COT_SC, Self-Refine, LLM Debate, Step-back, Quality-Diversity, Dynamic Role Assignment) injected into a fresh `archive.jsonl` at pre-seed time. |

Overlay seeding + the §7 edit whitelist are handled by the shared
`_common/lib/overlay.py` — the typed `graph_edits` patch applier was
retired 2026-05-20 (the implementer sub-agent edits the overlay
natively; serialising intent into an op enum for a deterministic
replayer was an ADAS-era vestige).

Skills import these by adding `lib/` to `sys.path` (no package
init — these are scripts, not a Python package):

```python
import sys
sys.path.insert(0, ".claude/commands/architect/adas-subagent/lib")
from helpers import bootstrap_confidence_interval, atomic_commit
```

These files are tracked in git so a fresh Claude session resuming a
compacted conversation can locate them directly from `README.md` and
the per-skill steps.

## Run-dir layout (delta vs files-contract)

Standard iter dir per files-contract, **plus** these adas-subagent-specific
files:

```
outputs/design_runs/adas-subagent/{graph}/v{N}/
├── archive.jsonl                # NEW — per-vN, append-only
├── trace.md                     # standard
├── lineage.md                   # standard
├── iter_0/                      # baseline (no proposer/implementer)
│   ├── graph.json
│   ├── active_workspace/        # complete mutation set vs frozen (empty/absent for iter_0)
│   ├── metrics.json             # standard + new "fitness_str" top-level
│   └── (no proposal.md, no debug_log.md — iter_0 has no design)
├── iter_1/                      # first evolved gen
│   ├── parent.txt
│   ├── proposal.md              # NEW — replaces adas v1's analysis.md + revision.md
│   ├── graph.json               # convenience copy of active_workspace/graphs/{graph}.json
│   ├── active_workspace/        # bootstrapped from parent's active_workspace, layered with this iter's edits
│   ├── debug_log.md             # NEW name (was adas v1's debug_log.md too)
│   ├── metrics.json             # with fitness_str
│   ├── summary.csv, export.json
│   └── (no analysis.md, no revision.md, no report.md — folded into proposer/loop)
├── .staging/                    # NEW — transient pre-commit dir
│   └── iter_n/                  # populated by proposer/implementer/evaluator; mv to iter_n/ on success, rm -rf on SKIP
└── .loop_state/                 # standard
```

## Termination

- `--max-iters` reached
- User-touched STOP file: `{RUN_DIR}/.loop_state/STOP`
- Consecutive SKIPs ≥ K (`--max-consecutive-skips`, default 3)

## What this variant does NOT do

- **No revert chain** — adas v1's revert-trigger / rollback bookkeeping
  is gone. SKIPs ARE the rollback (workspace reverts on each retry; on
  retries-exhausted, workspace stays at last-known-good archive head).
- **No sub-stage targets** (R1/R2/R3) — adas v1 / AAS's per-stage
  proposal-targeting is absent. There is one proposer output per iter.
- **No tournament / parallel candidates** — single-population
  hill-climb with archive injection, same as upstream ADAS.

## Paradigm-independent structural anchors preserved verbatim

| adas-subagent element | Upstream anchor | Verbatim? |
|---|---|---|
| 3-call Reflexion (propose / R1 / R2) | `search.py:188, 194, 198` | Yes (call sequence) — implemented as 3 independent `Agent(...)` sub-agent spawns |
| `Reflexion_prompt_1`, `Reflexion_prompt_2` | `mmlu_prompt.py:497, 525` | Verbatim text (only `"code"` → `"patch"` field rename) |
| Failed iter not in archive | `search.py:202, 225` (`n -= 1; continue`) | Verbatim semantics |
| `bootstrap_confidence_interval(acc_list, 100000, 0.95)` | `utils.py:31` | Verbatim algorithm |
| `fitness_str = "95% Bootstrap CI: (lo%, hi%), Median: x%"` | `utils.py:76` | Verbatim format |
| `del next_solution['reflection']` before archive append | `search.py:232–235` | Verbatim semantics for the `reflection` field; `debug_thought` is no longer in the schema |
| `get_prompt(archive)` injects archive verbatim into `[ARCHIVE]` | `mmlu_prompt.py:535–541` | Same shape (graph_summary in place of code) |
| `get_reflexion_prompt(archive[-1])` separately echoes most recent | `mmlu_prompt.py:544–547` | Verbatim semantics |
| Sampling diversity (3 independent samples per propose; up to 3 more per debug retry) | `search.py` call sites at temperature=0.8 | **Equivalent semantics** via `Agent(...)` spawns (each spawn is one independent Claude sample). Upstream's specific `temperature=0.8 / response_format=json_object / max_tokens=4096 / model=gpt-4o-2024-05-13` no longer applies — those are stateless-API knobs; sub-agents are not stateless API calls. |

## What's intentionally adapted (not verbatim)

- **Debug retry mechanism**: upstream `debug_max=3` with `mean(acc)<0.01`
  triggering Reflexion-style `debug_thought` push-back on a shared
  `msg_list` → `retry_max=3` (config knob, default 3) with **runtime
  correctness only** (crash / incomplete / step=0 / malformed metric)
  triggering a **fresh debugging sub-agent** spawn per retry. The
  proposal is read-only context; the retry sub-agent's job is "make it
  run", not "Reflexion-improve". Low/zero metric values are now real
  data for archive, not retry triggers. The `debug_thought` JSON field
  is removed from all schemas. Justification: modern coding agents
  (Claude sub-agents with tool access) are strong enough debuggers
  that Reflexion-style chained reasoning offers little marginal value
  over fresh-spawn-with-failure-trace; the cost (msg_list state +
  retry-on-bad-score conflating "broken patch" with "uninteresting
  patch") was buying ADAS-fidelity, not quality.
- **Meta-LLM mechanism**: stateless `get_json_response_from_gpt_reflect`
  API call → `Agent({subagent_type: "general-purpose"})` sub-agent
  spawn. The sub-agent has full tool access (Read, Grep, Bash,
  WebSearch, nested Agent), so each "LLM call" can ground itself in
  the codebase, eval logs, archive entries, etc. — the
  "coding-agent modernization" of stateless API calls. Structural
  property preserved: each spawn is one independent Claude sample,
  so N spawns = N independent samples (matching upstream's
  per-call independence at `temperature=0.8`).
- **`role: assistant` history → rendered text**: Agent tool accepts
  one user-string prompt, so prior assistant outputs are serialized
  as text inside subsequent sub-agents' prompts (rendered with
  `[role]` block markers). Textual content is identical; semantic
  role-tagging is approximated.
- **Artifact**: graph + nodeset patch, not Python `code` string
- **archive entry "code" slot**: `graph_summary` (rendered structural snapshot) + `diff_narrative`, not full graph.json (context budget) and not patch (not self-contained)
- **Pre-seed**: 7 reference patterns as text-only descriptors (fitness:null), NOT all evaluated upfront (upstream costs ~seconds per seed; ours would be 7×~2h = 14h infeasible)
- **Two-tier evaluation**: Smoke + Performance split (upstream conflated due to MMLU cheapness)
- **Iter dir**: standard files-contract layout (upstream has no iter dirs, only `archive.json`)
- **Staging dir** for atomic commit: upstream is stateless (`exec(code_str)`); we need workspace state management

## Quick reference

- Files contract: `.claude/commands/architect/_common/files-contract.md`
- Upstream code: `third_party/ADAS/_mmlu/{search,mmlu_prompt,utils}.py`
