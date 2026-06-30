# Architect files contract

> **Layered contract.** This file is the **global** layer — folder
> skeleton, file-type taxonomy, the one universal file, manifest
> schema. Each variant's concrete file set lives in its **local**
> layer, `<variant>/config.yaml § manifest`. See "Scope — global vs
> local" below.
>
> Rewritten 2026-05-20 from the earlier ADAS-shaped contract. The old
> "universal" layer hard-named ADAS-family files (`metrics.json`,
> `parent.txt`, `trace.md`, `config.md`, `lineage.md`, `summary.csv`,
> `export.json`) that myloop never produced — a latent contract
> violation. (`parent.txt` in fact was produced by *no* variant.) The
> universal layer now holds exactly **one** concrete file
> (`active_workspace/`, §3); every other file is classified by *type*
> (§2) and declared per variant in a manifest (§4). The sentinel-file
> convention is folded into that manifest.

## Scope — global vs local

| Layer | Lives in | Defines |
|---|---|---|
| **global** (this file) | `_common/files-contract.md` | folder skeleton; the file-type taxonomy (placement / mutability / lifecycle per type); the one universal file; the manifest schema; resolve protocol; versioning; edit whitelist; backend bridge; frozen-workspace rule |
| **local** | `<variant>/config.yaml § manifest` | a manifest instance — every file this variant writes, each classified into a global type, with a one-line purpose, a schema pointer, and an access matrix; plus the lineage model and phase-sentinel table |

The principle: **global defines the type system and the manifest
schema; local fills in a manifest instance.** A file inherits its
mechanical rules (which folder, can it be mutated, when is it written)
from its *type*; the local manifest only adds *identity* (purpose,
shape, who touches it). Adding a new variant file never edits this
file — you classify it.

---

## 1. Folder layout

```
workspace/                                     # FROZEN inputs — architect NEVER writes
└── architect/exp_profiles/{graph}.yaml         #   per-graph eval profile (an `input` file)

outputs/design_runs/
└── {method}/                                   # variant slug: adas-subagent | aflow | myloop | ...
    └── {graph}/                                # one dir per graph this method has worked on
        ├── _archive/                           # method+graph-scoped historical snapshots (frozen layout)
        └── v{N}/                               # major version (manual --new-version)
            ├── <vN-scoped files>               # type ∈ {input, working-memory, rollup}
            ├── iteration/
            │   ├── iter_0/                     # baseline (no parent, no mutation)
            │   │   └── <per-iter files>        # type ∈ {mutation, eval-output, phase-artifact, iter-record}
            │   ├── iter_1/
            │   └── ...
            ├── .staging/iter_M/                # type bookkeeping — transient pre-commit
            └── .loop_state/                    # type bookkeeping — resume / termination state
```

**Placement is determined by file *type*, not by filename** (see §2).
A skill that wants to know "where does file X go" reads X's type from
the variant manifest, then applies the type's placement rule.

**Method-scoped roots** (2026-05-15): the run-dir root is
`outputs/design_runs/{method}/{graph}/v{N}/`. Each method owns its own
tree; nothing is shared across methods (including `_archive/`).

**Iteration container** (2026-05-15): per-iter dirs live under
`v{N}/iteration/iter_{M}/`, NOT directly under `v{N}/`.

**Backward-compat read rule**: pre-migration iters have a flat
`workspace_snapshot/` instead of `active_workspace/{graphs,nodesets}/`.
Read-only skills (`understand`, `analyze`, `report`) should try
`{ITER}/active_workspace/...` first, then fall back to
`{ITER}/workspace_snapshot/...`.

---

## 2. File-type taxonomy

Every file an architect run writes is exactly one of **eight types**.
A type fixes three mechanical rules — **placement** (which folder),
**mutability** (can it change after first write), **lifecycle** (when
it is written). The variant manifest classifies each of its files into
one type; it does not restate these rules.

| Type | Definition | Placement | Mutability | Lifecycle |
|---|---|---|---|---|
| `input` | user- or bootstrap-authored configuration the run consumes | `vN/` root, or frozen `workspace/` | architect read-only | set before / at bootstrap |
| `mutation` | the workspace overlay for one iter — see §3, §8 | `iter_M/` | overlay (last-write-wins vs frozen) | written during the iter |
| `eval-output` | products of a backend eval run | `iter_M/` | immutable once written | written when eval finalizes |
| `phase-artifact` | a product of one variant pipeline phase (proposal, spec, trace, debug log, lineage pointer, graph snapshot…) | `iter_M/` | immutable once committed | written by the phase that owns it |
| `iter-record` | the one canonical record/index of an iter | `iter_M/` | written once | written at iter commit |
| `working-memory` | knowledge that persists across iters (archive, distilled facts, self-authored tools) | `vN/` root | variant-declared (`append-only` \| `mutable`) | grows across the run |
| `rollup` | cross-iter human-readable digest | `vN/` root | `append-only` or write-once-at-end | per iter, or at termination |
| `bookkeeping` | resume / termination / staging state | `vN/` root (hidden dirs) | transient | engine-managed |

**Boundary notes** (resolved for this redesign):

- **Lineage pointers** (`parent.txt`, a per-iter `graph.json` copy,
  `eval_run_id.txt`) are **not** a separate type — they are
  `phase-artifact`s written by whichever phase records them. The
  lineage *model* is declared once in the manifest (§4), not per file.
- **`tools/*.py`** (a variant's self-authored analysis code) is
  `working-memory` — placement / mutability / lifecycle all match. Its
  manifest `purpose` line carries the "self-authored utility"
  distinction; the taxonomy does not need a ninth type for it.

---

## 3. The universal file

Exactly **one** concrete file is mandatory for every variant,
regardless of search algorithm:

> Every `iter_M/` has an `active_workspace/` (type `mutation`) — the
> complete mutation set of that iter relative to the frozen
> `workspace/`. It MAY be empty or absent for an iter with no
> mutations (e.g. a baseline or a no-patch probe); the eval then falls
> through to frozen for every file.

This is universal because it is coupled to the framework, not to the
search algorithm: the backend's eval API consumes it as
`active_workspace_dir` (§8).

Two further per-iter properties are **roles bound in the manifest**,
not universal files:

- **metrics payload** — the file holding this iter's eval result. The
  variant declares which file plays this role (ADAS-family:
  `metrics.json`, an `eval-output`; myloop: `record.json`, an
  `iter-record`).
- **lineage** — how `iter_M` finds its parent. The variant declares a
  lineage *model* (§4): `implicit_linear` (parent = `iter_{M-1}`) or
  `parent_pointer` (the parent iter id is recorded in a
  variant-declared location — a standalone pointer file, or a field in
  a phase-artifact such as `proposal.md` frontmatter).

---

## 4. Variant manifest schema

Every variant **MUST** ship a `manifest:` block in
`<variant>/config.yaml`. It is the single source of truth for file
**identity**; the variant's `schemas.md` (or equivalent) remains the
single source of truth for file **shape**, linked by each entry's
`schema:` field.

```yaml
manifest:

  # --- how iter_M finds its parent ---
  lineage:
    model: implicit_linear            # implicit_linear | parent_pointer
    pointer: <location>               # REQUIRED iff parent_pointer — a file path,
                                      # or "<file>:<field>" for a frontmatter field

  # --- which declared file plays the "metrics payload" role (§3) ---
  metrics_payload: record.json

  # --- every file/dir this variant writes, one entry each ---
  files:
    <filename-or-dir>:
      type:       <one of the 8 taxonomy types>      # REQUIRED
      purpose:    "<one line — what this file is for>"  # REQUIRED
      schema:     <pointer, e.g. schemas.md#4>        # optional; omit for trivial files
      mutation:   <append-only | mutable | immutable> # optional; refines the type default
      written_by: [<skill>, ...]                      # access matrix
      read_by:    [<skill>, ...]

  # --- phase-sentinel table (drives Form 1 default-iter resolution, §5) ---
  phase_sentinels:
    <skill>:
      needs:  <filename produced by the previous phase, or null>
      writes: <filename this phase produces>
```

**Global rules attached to the manifest:**

1. A variant with no `manifest:` block is non-conformant — `understand`
   and validators ERR.
2. **The manifest is the authoritative source of file identity.** A
   variant's own skills (`<variant>/*.md`) MAY name their own files
   directly — they are variant-specific by construction, so this is
   not a violation. Shared `_common/` skills and any cross-variant
   consumer (validators, a fresh session reconstructing run state)
   **SHOULD** resolve file identity through the manifest rather than
   hard-code variant file names, so they stay correct as variants
   diverge. (`_common/understand.md` today hard-codes an ADAS-shaped
   file list in its Layer-3 step — migrating that to manifest-driven
   lookup is a known follow-up, not a conformance gate.)
3. The manifest classifies files into §2 types; it does **not** restate
   placement / mutability / lifecycle (those are inherited from type).
4. `experiment` / `evaluator`-style entry-point skills have no `needs`
   in `phase_sentinels` — they may open a fresh `iter_{max(M)+1}`.
5. **`bookkeeping`-type dirs are manifest-exempt.** Engine-managed
   transient state (`.staging/`, `.loop_state/`) is fully fixed by §1
   / §2 and identical across variants — a variant need NOT list it in
   `files:`. Every *non-`bookkeeping`* file/dir the variant writes
   MUST still have an entry.

The concrete manifest instance for each variant lives in
`<variant>/config.yaml`. See `myloop/config.yaml` for the reference
example; `adas-subagent/config.yaml` and `aflow/config.yaml` classify
their existing ADAS-family file set with no behaviour change.

---

## 5. Resolve protocol (graph / version / iter)

All `/architect:*` skills locate the target `(graph, vN, iter_M)` the
same way. Three input forms, mixable.

**Form 1 — default (no args)**
```
/architect:<skill>
```
- `graph` = most-recent-mtime entry under the skill's graph source (table below).
- `version` = max N of `v<N>/`; if none exists, `v0`.
- `iter` = driven by the variant's `phase_sentinels` (§4): latest iter
  where `needs` exists and `writes` does not. Entry-point skills
  default to `iter_{max(M)+1}`.

**Form 2 — named flags**
```
/architect:<skill> --graph <name> --version <N> --iter <M>
```
Any subset; unset slots fall back to Form 1 defaults.

**Form 3 — positional triple**
```
/architect:<skill> <graph> [<version> [<iter>]]
```
Fixed order, trailing slots optional, no skipping.

**Fuzzy matching** — `graph` only. Resolution order: (1) exact, (2)
case-insensitive exact with `-`/space normalised to `_`, (3)
case-insensitive substring. 1 hit → use it, print
`graph: "<input>" → <resolved>`. 0 hits → ERR + list all graphs. ≥2
hits → ERR + list matches, require re-run (never auto-pick — hides
errors when pipelines run in parallel). `version` / `iter` accept bare
or prefixed integers (`0`/`v0`/`iter_3`), exact match only — no fuzzy.

**Graph source of truth per skill**

| Skill class | Graph fuzzy source | On missing run-dir |
|---|---|---|
| `experiment` / entry points | `{ITER}/active_workspace/graphs/*.json` else `workspace/graphs/*.json` | OK — auto-creates `v0/iteration/iter_0/` |
| `analyze` / `revise` / `debug` / `report` / variant analogues | `outputs/design_runs/{method}/*/` | ERR — run-dir must already exist |

**Required print after resolve**
```
RUN_DIR=outputs/design_runs/{method}/{graph}/v{N}/iteration/iter_{M}
  graph   = {graph}    (input: <auto> | <exact> | "<raw>" → <resolved>)
  version = {N}        (input: <auto> | <N>)
  iter    = iter_{M}   (input: <auto> | <M>)
```

Mandatory for every **single-iter entry-point** skill (`experiment`,
`analyze`, `revise`, `debug`, `report` and variant analogues) — one
that resolves a single `(graph, vN, iter)` and acts on it.

**Orchestrator exception.** A variant's `loop` / outer-orchestrator
skill resolves once but then runs *many* iters per invocation, so it
cannot name one `iter_{M}`. It prints a **vN-scoped** block instead —
`RUN_DIR` ending at `v{N}`, no `iter` line, but still the `graph` /
`version` lines with their `(input: …)` provenance — plus its own
per-iter banner as each iter opens. Phase skills it drives internally
with an explicit `--iter` (`proposer` / `implementer` / `distill`)
inherit that iter and MAY print a compact phase banner.

---

## 6. Versioning (vN) and iteration (iter_M)

- **`iter_M`** = one cycle within a vN; auto-increments. What a cycle
  *is* is variant-defined. The only universal per-iter invariant is §3
  (`active_workspace/`).
- **Bootstrap rule**: a write-skill creating `iter_{M+1}` first copies
  `iter_M/active_workspace/` into `iter_{M+1}/active_workspace/`, then
  layers this iter's edits. `iter_0` starts empty.
- **`vN`** = major pivot; bumped **manually only** via
  `experiment --new-version`. Each vN starts from current
  `workspace/` — no cross-version copy.
- **current = max(N)**; derived from the filesystem, no pointer file.
- **Iter rerun is overwrite-in-place** — for transient retry, not
  variance estimation.
- **Write-skill version protection**: any skill whose `phase_sentinels`
  `writes` is non-empty, plus entry points, refuses to write a
  non-latest `vN` without `--allow-old-version`. Read-only skills
  (`analyze`, `report`, `understand`) need no flag.

---

## 7. Edit whitelist

Write skills are bounded by **two** edit boundaries with different
strength. They differ because the `active_workspace/` overlay (§3, §8)
only covers `workspace/{graphs,nodesets}/` — everything outside it has
no sandboxed copy.

**Hard wall (always BLOCK).** `agentcanvas/backend/app/**` (framework)
and `third_party/**` (vendored) are **never** editable, regardless of
graph. These paths have no overlay: a write there is a real, global,
cross-session mutation that also hits the user's `:8000` backend.
A patch touching them → BLOCKED escalation, surface to user. The
overlay does not replace this wall — it does not reach here.

**Soft scope (warn, do not block).** Everything under
`{ITER}/active_workspace/{graphs,nodesets}/` is writable — the overlay
sandboxes it, so a bad edit is discarded with the iter and never
touches frozen `workspace/`. The expected scope of one iter is the
iter's graph plus the nodesets it uses (distinct `<nodeset>` prefixes
of each node's `type` field, `<nodeset>__<node>`); a nodeset may live
as a flat `<nodeset>.py`, a package `<nodeset>/**`, or a server-mode
equivalent under `server/<nodeset>(.py | /**)`. If a patch touches a
graph or nodeset **outside** that expected scope, the implementer
**warns** (it is usually a proposer path mistake or hallucination) but
does **not** block — transitive nodeset dependencies (a nodeset
imported by another, never named in any node `type`) are legitimately
off-prefix and must remain editable.

Server nodesets are in-scope as of TODO #60 (2026-05-15): an overlay
edit to a `parallelism="shared"` server nodeset triggers an ephemeral
`auto_host` spawn at eval-admit time.

---

## 8. Backend API + active_workspace bridge

All skills use `/api/eval/v2/{start,status,export,stop}` against
`{BACKEND_URL}`. Health probe: `{BACKEND_URL}/health` (NOT
`/api/health`). No stride — episodes run consecutively from
`start_episode_index`. How `{BACKEND_URL}` is established is
variant-specific (see the variant README).

**Active-workspace overlay** — every `POST /api/eval/v2/start` MUST
include `active_workspace_dir={ITER}/active_workspace` (absolute path)
so the eval subprocess overlays the iter's mutation set on frozen
(frozen → active, last-write-wins). Without it the run uses pure frozen
workspace — correct only for an iter with no mutations.
`POST /api/eval/v2/introspect` accepts the same field.

This is the bridge that makes the `mutation`-type file (§3) load-bearing
at runtime — it is why `active_workspace/` is the one universal file.

---

## 9. Frozen workspace contract

`<repo>/workspace/` is the **frozen** faithful baseline. Architect
skills NEVER write to it. All mutations land in
`{ITER}/active_workspace/` and the backend overlays them at run time.

Legal writers to frozen `workspace/`:
- Canvas UI (`/api/canvas/graphs.py`) — the human editor.
- One-shot user setup scripts (data preprocessing).
- `experiment` auto-creating `workspace/architect/exp_profiles/{g}.yaml`
  on first run (an `input` file, not method code).

---

## Source of truth

| Question | Where the answer lives |
|---|---|
| Which folder does file X go in? | X's `type` in the variant manifest → §2 placement rule |
| Can file X be mutated after writing? | X's `type` (+ optional `mutation` override) in the manifest |
| What is file X *for*? | X's `purpose` in the variant manifest |
| What does file X's content look like? | X's `schema` pointer in the manifest → variant `schemas.md` |
| Who writes / reads file X? | X's `written_by` / `read_by` in the manifest |
| Which iter does a default-form skill target? | the variant `phase_sentinels` table → §5 |
| What file names does variant V use? | `V/config.yaml § manifest` — and **only** there |
