# Build the {VARIANT} mental model

> **Common understand skill.** Invoked indirectly by each variant's
> `understand.md` stub (e.g. `/architect:adas-subagent:understand`).
> The stub binds `VARIANT`, `VARIANT_DIR`, and points to this file. All
> variant-specific values (archive path, required exp.yaml blocks,
> archive-preview format, Layer-1 emphasis text) are read from
> `<VARIANT_DIR>/config.yaml` § `understand:`.

Loads the minimum context needed to operate the **{VARIANT}** pipeline
on one specific graph. Run once per Claude session (or after
`/compact`), then invoke `/architect:{VARIANT}:loop` or any individual
worker skill.

Does **not** execute pipeline steps. Reads files into context only.

## Arguments

```
/architect:{VARIANT}:understand [<graph> [<version>]]
                                [--graph <name>] [--version <N>]
                                [--for <skill>]              default loop
                                [--full]
```

- `<graph>`, `<version>` — fuzzy/positional per files-contract.
- `--for <skill>` — declare next variant skill to invoke (`loop` |
  `proposer` | `implementer` | `evaluator`). Drives Tier 2 doc
  selection. Default `loop` (union).
- `--full` — load all Tier 2 docs. Use on first contact or after
  framework changes.

## What this loads

Three layers in order.

### Layer 1 — variant mechanics (~25–30 K)

- `_common/files-contract.md` — shared run-dir layout, resolve
  protocol, edit whitelist, `{graph}.yaml` schema, eval API.
- `<VARIANT_DIR>/README.md` — variant's structural contracts +
  skill map.
- `<VARIANT_DIR>/{for}.md` — the worker skill the user intends to
  invoke (or `loop.md` by default).

If `config.yaml § understand.layer1_emphasis` is set (string or file
path), print that text after reading Layer 1 to flag non-obvious
invariants the user must respect.

### Layer 2 — AgentCanvas concept + contract (Tier 1 + JIT Tier 2)

Tier 1 + Tier 2 selection (variant-independent — the architect
pipeline's required docs do not depend on which variant you're
running). Quick map:

**Tier 1** (always, ~57 K):
- `docs/pages/developer-guide/core/glossary.html`
- `docs/pages/developer-guide/capabilities/customizable-node-system.html`
- `docs/pages/developer-guide/capabilities/graph-execution-engine.html`
- `docs/pages/developer-guide/design-docs/graph-system.html`
- `docs/pages/developer-guide/design-docs/graph/batch-eval.html`
- `docs/pages/developer-guide/design-docs/execution-logs.html`

**Tier 2** (JIT by `--for`) — files live under
`docs/pages/developer-guide/design-docs/<name>.html`:

| `--for` | Tier 2 files | ~tok |
|---|---|---:|
| `loop` (default) | wire-types, llm-config-system, plugin-servers, loop-control-system | ~39 K |
| `proposer` | wire-types, llm-config-system | ~22 K |
| `implementer` | wire-types, plugin-servers | ~22 K |
| `evaluator` | plugin-servers | ~12 K |
| `--full` | union | ~39 K |

### Layer 3 — Graph-specific (~30–40 K)

For the resolved `(graph, vN)`:

- `workspace/graphs/{graph}.json` — live topology.
- `workspace/architect/exp_profiles/{graph}.yaml` — validate each entry in
  `config.yaml § understand.exp_yaml_required` exists; warn if
  missing (auto-bootstrap will create with defaults). If
  `understand.exp_yaml_extra_blocks` is non-empty, also validate
  those top-level blocks exist (e.g. `aflow:`).
- For each `<nodeset>__<node>` prefix in `{graph}.json`: read
  `workspace/nodesets/{prefix}.py` if it exists; SKIP `nodesets/server/**`.
- `config.yaml § understand.archive_path` if set — load all entries
  into context; this is what the meta-LLM sees at propose time.
- `outputs/design_runs/{method}/{graph}/v{N}/trace.md` — full metric history.
- Last 3 sections of `outputs/design_runs/{method}/{graph}/v{N}/lineage.md`.
- Latest iter: max-M `iter_M/`. Read whichever of `proposal.md`,
  `debug_log.md`, `metrics.json`, `summary.csv`, `export.json` exist.

## Steps

### 1. Resolve graph + version

Apply files-contract resolve protocol. Validate vN exists; if not,
ERR — most variants require an existing run-dir (use the bootstrap
hint from `config.yaml § understand.bootstrap_hint`, or create the
run-dir manually with an iter_0 baseline (no edits)).

Print:

```
RUN_DIR=outputs/design_runs/{method}/{graph}/v{N}
  graph    = {graph}    (input: <auto> | <exact> | "<raw>" → <resolved>)
  version  = {N}        (input: <auto> | <N>)
  for      = {skill}    (input: --for | default loop)
  mode     = standard | full
  variant  = {VARIANT}
  archive  = {archive_path or "n/a"}   ({K} entries: counts per type)
```

### 2. Layer 1 — variant mechanics

Read in order:
1. `_common/files-contract.md`
2. `<VARIANT_DIR>/README.md`
3. `<VARIANT_DIR>/{for}.md` (default `loop.md`)
4. If `understand.layer1_emphasis` is set: print/echo the emphasis
   text to flag non-obvious invariants.

### 3. Layer 2 — Concept + contract

Read Tier 1 (6 files). Then Tier 2 selected by `--for` (or `--full`).

### 4. Layer 3 — Graph-specific

Per files-contract § "Resolve protocol":

a. `workspace/graphs/{graph}.json` — extract nodeset prefixes.
b. For each prefix `P`:
   - `workspace/nodesets/{P}.py` → read if top-level
   - `workspace/nodesets/server/{P}.py` → SKIP, print note
c. `workspace/architect/exp_profiles/{graph}.yaml`. **Validate** each entry in
   `understand.exp_yaml_required` exists. Validate top-level blocks
   in `understand.exp_yaml_extra_blocks` exist. Warn (don't error) if
   missing.
d. If `understand.archive_path` set: load all archive entries into
   context. Render preview per `understand.archive_preview`:
   - `simple` — count entries by `generation` type (e.g.
     `initial / reference / evolved`).
   - `extended` — additionally print top-K by `score`
     (next select_round candidates) + per-parent experience preview
     (for variants using `parent_iter_id` like aflow).
e. `outputs/design_runs/{method}/{graph}/v{N}/trace.md`.
f. `outputs/design_runs/{method}/{graph}/v{N}/lineage.md` last 3 sections.
g. Latest iter: max-M `iter_M/`. Read whichever of `proposal.md`,
   `debug_log.md`, `metrics.json`, `summary.csv`, `export.json` exist.

### 5. Print loaded summary

```
=== /architect:{VARIANT}:understand summary ===
graph     = {graph}
version   = {N}
for       = {skill}
mode      = standard | full

Layer 1 (variant mechanics):  {N1} files, ~{T1}K tokens
Layer 2 (contracts):           {N2} files, ~{T2}K tokens   (Tier 1: {t1}, Tier 2: {t2})
Layer 3 (graph-specific):      {N3} files, ~{T3}K tokens

archive       = {archive_path or "n/a"}  ({K} entries)
  <per-type counts as defined in understand.archive_entry_types>
  <if understand.archive_preview == extended: top-K + per-parent>
latest iter   = iter_{M}
iter trail    = iter_0 → ... → iter_{M}

Skipped:
  - server-mode nodesets: {list, if any}
  - other vN: {list, if any}

Ready. Next: invoke /architect:{VARIANT}:{for} (or any {VARIANT} worker).
```

## Guardrails

- **Read-only.** No edits, no spawned backends.
- **No subagent delegation** — reads populate THIS context for the
  upcoming loop/worker invocation.
- **Resolve once, no iter resolution.** vN-level read; iter
  resolution is loop/worker's job.
- **Token estimates advisory.** Real cost includes formatting +
  tool-call overhead.
- **Idempotent.** Re-running this skill is harmless.
- **Auto-invoked by `/architect:{VARIANT}:loop` P0** unless
  `--skip-understand`. Manual invocation is only needed when:
  (a) invoking individual workers without going through loop, or
  (b) wanting a tighter Tier 2 selection via `--for`.
- **Read before edit still applies.** Workers (`proposer`,
  `implementer`, `evaluator`) MUST still Read each file they edit
  in their own turn, even though understand has pre-loaded them.

## Variant config schema (this skill)

```yaml
# <variant>/config.yaml
understand:
  archive_path:        outputs/design_runs/{method}/{graph}/v{N}/archive.jsonl   # null if no archive
  archive_entry_types: [initial, reference, evolved]                   # for per-type counts
  archive_preview:     simple | extended                                # simple = counts only
  exp_yaml_required:   [smoke_<graph>, perf_<graph>]                    # mandatory profile keys
  exp_yaml_extra_blocks: []                                             # extra top-level blocks (e.g. ["aflow"])
  layer1_emphasis:     null                                             # inline text or null
  bootstrap_hint:      /architect:<variant>:loop --skip-understand      # shown in resolve ERR
```
