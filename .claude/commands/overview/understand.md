Read the project overview by directly reading the canonical doc-site directories (`developer-guide/core/`, `developer-guide/capabilities/`). Design-docs are deferred to on-demand reads.

---

## Steps

1. **Read `.claude/PROJECT_OVERVIEW.md`** — non-doc-site context: Quick Reference (stack, entry points, envs), Top-Level Repo Map, Don't-Touch / Don't-Search Zones, `workspace/` Subdirectories (ADR-canvas-003 + ADR-server-001 semantics), Data Layout (ADR-platform-005), NodeSet Catalog.

2. **Read doc-site canonical content directly, in parallel.** Do **not** Glob `developer-guide/core/*.md` — some files in that directory are on-demand references, not overview material. Read exactly this explicit file list and nothing else:

   **`developer-guide/core/` — 5 files (overview essentials only)**:
   - `docs/core/blueprint.md` — problem, key idea, 7 capabilities, design principles, core components
   - `docs/core/architecture.md` — system diagram, data flow, execution engines, component system, tech stack
   - `docs/core/codebase-map.md` — per-package file walkthrough
   - `docs/core/glossary.md` — load-bearing terms
   - `docs/core/roadmap.md` — **open work only** (TODO / Feature / Env / Method / Planned / Deferred)

   **`developer-guide/core/decisions/` — field-scoped ADR index files** (replaces the retired `decisions-short.md` single file; the per-field `index.md` files together give one-line summaries of every active ADR):
   - `docs/core/decisions/index.md` — field overview + counts
   - `docs/core/decisions/canvas/index.md`
   - `docs/core/decisions/components/index.md`
   - `docs/core/decisions/dataflow/index.md`
   - `docs/core/decisions/executor/index.md`
   - `docs/core/decisions/server/index.md`
   - `docs/core/decisions/eval/index.md`
   - `docs/core/decisions/observability/index.md`
   - `docs/core/decisions/platform/index.md`
   - Skip `decisions/_legacy/` — superseded ADRs, not overview material.

   **`developer-guide/capabilities/` — Glob is fine here, the directory is uniform**:
   - Glob `docs/capabilities/*.md` and Read every file **except** `index.md` (nav stub).

   **Files explicitly NOT read at overview time** (read on-demand when a task needs them):
   - `developer-guide/core/decisions/<field>/adr-*-*.md` — full ADR bodies (Context / Alternatives / Rationale). Consult one specific file when a task references it; the field index files are enough for the overview summary.
   - `developer-guide/core/decisions/_legacy/*.md` — superseded decisions kept for forensic reading.
   - `developer-guide/core/roadmap-done.md` — historical archive of shipped features + completed TODOs. Consult only when you need to confirm what has already been delivered.
   - `developer-guide/core/major-versions.md` — version-cut history; consult on demand.
   - `developer-guide/core/index.md` — nav landing stub.
   - `developer-guide/design-docs/*.md` — deep-dive references. Read individually when a task requires the schema / internals (e.g. writing a new node type → `canvas-system.md`; debugging cycles → `loop-control-system.md`; designing state → `state-containers.md`; authoring a graph JSON → `graph-system.md`).

   Don't summarize through an intermediate digest — read the real files. Budget ~30-40K tokens for the loaded content.

3. **Present a combined summary** synthesised from what you just read:
   - **Project Overview**: key architecture and entry points (from `developer-guide/core/blueprint.md` + `developer-guide/core/architecture.md`)
   - **Recent Decisions**: 5 most-recent ADRs across fields (id + title + status) — pick the latest entries from each field's `decisions/<field>/index.md` (most field index files are date-ordered).
   - **Roadmap**: in-progress + planned from `developer-guide/core/roadmap.md`
   - **Key Terms**: load-bearing terms from `developer-guide/core/glossary.md`

4. **Run `/todo/fetch`** to print the current TODO list (must be the last thing printed).
