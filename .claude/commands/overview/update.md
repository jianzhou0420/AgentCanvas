Rebuild `.claude/PROJECT_OVERVIEW.md` — the small non-doc-site context file loaded at session start.

PROJECT_OVERVIEW.md is deliberately **not** a doc-site digest. Doc-site content (`core/`, `capabilities/`, `design-docs/`) is read directly by `/overview:understand`. This file holds only the orthogonal context that isn't available from doc-site reads or `CLAUDE.md`:

- **Quick Reference** — stack versions + entry points + conda envs
- **Top-Level Repo Map** — one line per top-level directory
- **Don't-Touch / Don't-Search Zones** — Grep/Glob exclusions
- **workspace/ Subdirectories** — semantics of the code surface (ADR-canvas-003 + ADR-server-001)
- **Data Layout** — ADR-platform-005 hybrid scheme (`data/habitat|mp3d|opennav|outputs`)
- **NodeSet Catalog** — one row per top-level file in `workspace/nodesets/`

**Target size**: ≤ 100 lines / ~3K tokens. Every line is paid by every future session.

---

## Section 1 — Read canonical sources

Read all of these in parallel (independent):

- `CLAUDE.md` — env block (ADR-platform-004) + any Standardization changes
- `agentcanvas/backend/requirements.txt` — backend deps → versions
- `agentcanvas/frontend/package.json` — frontend stack → versions
- `docs/core/decisions/platform/adr-platform-005-data-layout.md` — data-layout invariants for the **Data Layout** table
- `docs/core/decisions/canvas/adr-canvas-003-graph-node-system.md` — `kind` field semantics for the **workspace/ Subdirectories** table
- `docs/core/decisions/server/adr-server-001-auto-server-app.md` — server-mode invariants for the same table
  *(The single consolidated `core/decisions.md` was retired in favour of one file per ADR under `decisions/<field>/`. Read each ADR file directly when its invariants drive a section below.)*
- Glob `workspace/nodesets/**/*.py` — nodeset file catalog
- Glob `docs/nodesets/*.md` — per-nodeset docs (for the "What it wraps" column)
- Shell: `ls /path/to/vlnworkspace` and `ls /path/to/vlnworkspace/workspace` and `ls /path/to/vlnworkspace/data` — verify top-level + workspace + data dirs actually exist as described (drift check)

---

## Section 2 — Regenerate each section

Preserve the file header exactly:

```markdown
# AgentCanvas — Project Overview

> Non-doc-site context. Doc-site content (core / capabilities / design-docs) is read directly by `/overview:understand`.
```

Then rebuild each section below, in order. Do not add doc-site summary sections (blueprint, architecture, glossary, capabilities, ADR table, roadmap, KB) — those were intentionally removed.

### Quick Reference
- Stack table: Backend / ML-Simulator / Frontend / Eval harness / Docs — versions from requirements.txt + package.json; env names from CLAUDE.md "Environment" section.
- Entry points line: `run_dev.sh`, `python -m vlnworkspace.eval`, `bash docs/run_dev.sh` with the canonical ports (8000 / 5173 / 8001).
- Conda envs (ADR-platform-004) — 1 line.
- `workspace/` one-liner.

### Top-Level Repo Map
Table of visible top-level directories. One-line role each. Verify against `ls` output; remove any row whose directory no longer exists, add a row for any new directory that has user-authored content (skip gitignored derivatives like `site/`, `node_modules/`, `__pycache__/`).

### Don't-Touch / Don't-Search Zones
Bullet list of search-exclusion paths. Rarely changes — update only when a new vendored/output dir appears or an existing one is removed. Keep entries grouped: vendored · scratch · build outputs · standard artefacts · eval outputs · tool state.

### workspace/ Subdirectories
Table: Subdir | Kind | Role. One row per top-level entry in `workspace/` (dirs and `hooks.json`). For graph dirs, mark the `kind` field (`kind="graph"` editable vs `kind="node"` frozen, ADR-canvas-003). For nodesets, split `nodesets/` (local mode) and `nodesets/server/` (AutoServerApp, ADR-server-001) since the runtime isolation boundary matters.

### Data Layout (ADR-platform-005)
Table sourced from the **Decision** paragraph of `developer-guide/core/decisions/platform/adr-platform-005-data-layout.md`. Four rows: `data/habitat/`, `data/mp3d/v1/scans/<scan>/`, `data/opennav/`, `data/outputs/`. Include the env-var overrides for opennav (`OPENNAV_*`). If ADR-platform-005 is ever superseded, read the successor ADR (linked from `decisions/platform/index.md`) and regenerate this table from the new decision paragraph; do not mix old + new shapes.

### NodeSet Catalog
Table: NodeSet | File | What it wraps.
- One row per **top-level** file in `workspace/nodesets/` (a file like `mapgpt.py`) plus one row per **top-level** file in `workspace/nodesets/server/` (a file like `server/habitat.py`).
- "What it wraps" = first non-empty sentence of the matching `docs/nodesets/<name>.md` file, trimmed to ≤ 120 chars. If no matching doc, summarise from the Python file's module-level docstring or class docstring.
- If the matching doc is missing, include the row but mark "What it wraps" with `— (no doc yet)` so drift is visible.

---

## Section 3 — Write and report

1. Use `Write` (not `Edit`) — full regeneration. No "Last updated" footer — `git log` is the changelog.
2. Count lines; fail loud if the file exceeds **120 lines**.
3. Print a short diff report:

```
[overview] Rebuilt .claude/PROJECT_OVERVIEW.md
  Total:     86 / 120 lines  (was: 84)
  Content deltas since last rebuild:
    Top-level dirs: +N / -N
    Workspace subdirs: +N / -N
    Nodesets:    +N / -N   (now listing K rows)
    Data layout: (unchanged / updated per ADR-NNN)
    Stack versions: <list of changed pins, or "no change">
```

Do not print the full file content.
