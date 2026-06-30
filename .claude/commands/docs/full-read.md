# Understand the Doc-Site (Full Read)

Read all documentation from the doc-site. Glob-driven — survives doc-site restructures.

For session-start context, use `/overview:understand` — it reads `developer-guide/core/` + `developer-guide/capabilities/` + the field-scoped ADR index files directly plus `.claude/PROJECT_OVERVIEW.md` for non-doc-site context.

This command is the **superset read**: everything `/overview:understand` loads, plus the Diátaxis-split impl docs (`ds-*`), nodeset docs, tutorials, getting-started, and the full ADR bodies (every `decisions/<field>/adr-*-*.md` rather than just the field index files). Use when you need implementation details that the core + capabilities + design-docs trio doesn't cover.

## Steps

### Step 1 — Read core docs in parallel

- Glob `docs-site/docs/developer-guide/core/*.md` (blueprint, architecture, glossary, codebase-map, roadmap, roadmap-done, major-versions, index)
- `docs-site/docs/developer-guide/process/my-procedures.md` (workflow rules)
- `docs-site/docs/index.md`

### Step 2 — Read capability narratives in parallel

- Glob `docs-site/docs/developer-guide/capabilities/*.md` — one file per capability (the "why" layer)

### Step 3 — Read feature topic details in parallel

- Glob `docs-site/docs/developer-guide/design-docs/*.md` — one file per unsplit topic (the "how" layer)

### Step 4 — Read Diátaxis-split implementation docs in parallel

- Glob `docs-site/docs/developer-guide/ds-concepts/*.md`
- Glob `docs-site/docs/developer-guide/ds-tutorial/*.md`
- Glob `docs-site/docs/developer-guide/ds-recipes/*.md`
- Glob `docs-site/docs/developer-guide/ds-api-reference/*.md`

### Step 5 — Read ADR bodies in parallel (field-scoped)

The consolidated `core/decisions.md` was retired. ADRs now live as one file per decision under `developer-guide/core/decisions/<field>/adr-{field}-{NNN}-{slug}.md`. Read everything **except** `_legacy/` (superseded forensic-only):

- `docs-site/docs/developer-guide/core/decisions/index.md` — field overview + counts
- Glob `docs-site/docs/developer-guide/core/decisions/canvas/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/components/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/dataflow/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/executor/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/server/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/eval/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/observability/*.md`
- Glob `docs-site/docs/developer-guide/core/decisions/platform/*.md`
- Skip `developer-guide/core/decisions/_legacy/*.md` unless the user asks for legacy lookup.

### Step 6 — Read catalog and onboarding in parallel

- Glob `docs-site/docs/developer-guide/nodesets/*.md`
- Glob `docs-site/docs/developer-guide/tutorials/*.md`
- Glob `docs-site/docs/developer-guide/getting-started/*.md`

### Step 7 — Present a summary with this structure

- **Blueprint**: 2–3 sentence summary of problem + solution + principles
- **Architecture**: key layers and data flow
- **Capabilities**: list of N capability files with one-line summaries
- **Features**: list of N feature topics with one-line summaries
- **Diátaxis-split topics**: list of topics that have all 4 quadrant files present
- **Recent Decisions**: 5 most-recent ADRs across fields (id + title + status) — pick the latest entries from each field's `decisions/<field>/index.md` table.
- **Roadmap**: in-progress and planned next
- **Key Terms**: flag load-bearing glossary entries
- **Procedures**: brief reminder of the active workflow rules (from `my-procedures.md`)

### Step 8 — Note any docs that are empty or stub-only

Print a list of files under 20 lines or containing `<!-- TODO` markers — these need content.

### Step 9 — Remind the user of related commands

- `/overview:understand` — session-start context; reads `developer-guide/core/` + `developer-guide/capabilities/` + `decisions/<field>/index.md` directly plus `.claude/PROJECT_OVERVIEW.md`
- `/docs/adr` — record a new field-scoped Architecture Decision Record
- `/docs/update-docs` — surgical updates for non-architectural changes
- `/docs/diataxis-split` — split a topic into the 4 Diátaxis quadrants
- `/commit` — log changes and commit
- To serve the doc-site locally: `bash docs-site/run_dev.sh`
