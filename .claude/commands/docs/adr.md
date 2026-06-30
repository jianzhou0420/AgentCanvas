# Record an Architecture Decision

You are recording a new Architecture Decision Record (ADR) and propagating it across the doc-site.

ADRs are **field-scoped**. The retired single `core/decisions.md` file has been replaced by one file per decision under `developer-guide/core/decisions/<field>/adr-{field}-{NNN}-{slug}.md`, with field-local numbering (`adr-canvas-001`, `adr-canvas-002`, …). Legacy global IDs (`ADR-NNN`) survive only in each file's frontmatter `old_id:` for forensic lookup.

The active fields are: `canvas`, `components`, `dataflow`, `executor`, `server`, `eval`, `observability`, `platform`. (Plus `_legacy` — superseded ADRs only, never write new entries there.)

## Input

The user will describe a decision. If they didn't provide details, ask for:

- What was the decision?
- What was the context / what prompted it?
- What alternatives were considered?
- Why this choice?
- **Which field does it belong to?** Pick exactly one of `canvas`, `components`, `dataflow`, `executor`, `server`, `eval`, `observability`, `platform`. If unclear, read `docs-site/docs/developer-guide/core/decisions/index.md` — the field table there describes each field's scope. When two fields are plausible, pick the one whose subsystem **owns the change**, not the one that consumes it (e.g. a new wire-coercion rule belongs to `dataflow`, even if the executor implements the firing).

## Steps

1. **Get the current timestamp** — run `date "+%Y-%m-%d %H:%M"` to get the precise timestamp. Use this exact value for ALL `last updated:` headers and changelog entries throughout this ADR. Do NOT guess or hardcode the time.

2. **Read current state**

   - Read `docs-site/docs/developer-guide/core/decisions/index.md` — confirm the field exists in the field table.
   - Read `docs-site/docs/developer-guide/core/decisions/<field>/index.md` — the field-scoped table lists every existing ADR in this field. The next ADR number is `max(existing NNN) + 1`, zero-padded to 3 digits.
   - Read `docs-site/docs/developer-guide/process/my-procedures.md` to confirm the change-tracking rules.
   - Optionally `Glob docs-site/docs/developer-guide/core/decisions/<field>/adr-*.md` to double-check the highest existing number (the index table is canonical, but a `Glob` cross-check catches drift).

   Derive the new ADR's:
   - **Slug** — short kebab-case description (e.g. `worker-pool-and-batched-inference`, `data-layout`). Keep it ≤ 6 words.
   - **Filename** — `adr-{field}-{NNN}-{slug}.md`.
   - **Canonical ID** — `ADR-{field}-{NNN}` (used in `id:` frontmatter, doc references, glossary entries).

3. **Write the ADR file** at `docs-site/docs/developer-guide/core/decisions/<field>/adr-{field}-{NNN}-{slug}.md`. Use this exact frontmatter + body shape — every existing ADR follows it:

   ```markdown
   ---
   id: ADR-{field}-{NNN}
   date: {YYYY-MM-DD from timestamp}
   status: accepted
   field: {field}
   ---

   # ADR-{field}-{NNN}: {Title}

   **Date**: {YYYY-MM-DD HH:MM from timestamp}
   **Status**: accepted
   **Context**: {what prompted this decision}
   **Decision**: {what was chosen}
   **Alternatives**: {what else was considered}
   **Rationale**: {why this choice}
   **Affected docs**: {populated in step 5 — concrete file list}
   ```

   Add `old_id: ADR-NNN` to the frontmatter **only** if this ADR is being created to replace an entry that previously lived in the retired global numbering — leave it out for genuinely new ADRs.

4. **Update the field index table** — open `docs-site/docs/developer-guide/core/decisions/<field>/index.md` and append a row to the ADR table:

   ```markdown
   | [adr-{field}-{NNN}](adr-{field}-{NNN}-{slug}.md) | {Title} | {YYYY-MM-DD} | accepted |
   ```

   Match the column shape of the existing rows in that field's table — some include a `Supersedes` column. Keep date order (most recent at the bottom unless the field uses reverse order; respect the local convention).

5. **Update the root decisions index** — open `docs-site/docs/developer-guide/core/decisions/index.md` and bump the **Count** column for `{field}` by 1, plus the running total in the prose underneath the field table (e.g. `**Total: 33 active + 2 legacy = 35 ADRs**`). Update its `last updated:` timestamp from step 1.

6. **Identify affected docs** — walk the full `docs-site/docs/developer-guide/` tree and list every file the decision changes. Do not limit to `core/`: most ADRs touch narrative, reference, and implementation docs too. Use `Glob` on each subdir if unsure what exists.

   **Core (`developer-guide/core/`)** — always review:
   - `blueprint.md` — goals, principles, components, features
   - `architecture.md` — system structure, data flow, execution engines, tech stack
   - `glossary.md` — any new or renamed term
   - `roadmap.md` — move items between Done / TODO / Deferred; add changelog line
   - `decisions/<field>/index.md` + `decisions/index.md` — handled in steps 4–5

   **Capabilities (`developer-guide/capabilities/`)** — narrative "why it exists" docs:
   - Any capability whose mechanism or interface the ADR redefines (e.g. `customizable-node-system.md`, `graph-execution-engine.md`, `real-time-observability.md`)

   **Features (`developer-guide/design-docs/`)** — topic-level reference:
   - Any feature whose data model, wire shape, or algorithm the ADR changes (e.g. `canvas-system.md`, `state-containers.md`, `wire-types.md`, `graph-system.md`, `graph-executor.md`)

   **Diátaxis-split implementation (`developer-guide/ds-{concepts,tutorial,recipes,api-reference}/`)** — per-concept Diátaxis split:
   - `ds-concepts/<slug>.md` — design rationale per affected base class
   - `ds-tutorial/<slug>.md` — walkthroughs that now demonstrate the new API
   - `ds-recipes/<slug>.md` — task snippets that used the old shape
   - `ds-api-reference/<slug>.md` — port tables, lifecycle signatures, ClassVars
   - `developer-guide/core/codebase-map.md` — new files, renamed modules, deleted files

   **Tutorials (`developer-guide/tutorials/`)** — end-to-end author guides:
   - Any tutorial whose step-by-step instructions are now out of date (e.g. `component-cookbook.md`, `habitat-nodeset.md`, `execution-logs.md`)

   **NodeSet docs (`developer-guide/nodesets/`)** — per-nodeset reference:
   - Any nodeset whose ports, env panel, or server-mode behaviour the ADR alters

   **`.claude/PROJECT_OVERVIEW.md`** — flag it for refresh via `/overview:update` on the next rebuild (don't hand-edit; the command regenerates from canonical sources).

   **Output of this step**: a concrete file list that populates the `**Affected docs**:` field of the new ADR file. If the list has only `core/` files, double-check — most real ADRs touch 6–10 files across 3+ subdirs.

7. **Update each affected doc**:
   - Modify the relevant content section to reflect the decision — not just append a pointer. If the doc described the old behaviour, rewrite that passage.
   - Add a changelog entry at the bottom of the file: `- [YYYY-MM-DD HH:MM from step 1]: [what changed] (see ADR-{field}-{NNN})`
   - Update the `last updated:` timestamp at the top to the value from step 1.
   - Edit files in parallel where content is independent; sequential only when one edit depends on another's result.

8. **Summary** — print a summary of all files changed and the new ADR id (`ADR-{field}-{NNN}` plus the path to its file).
