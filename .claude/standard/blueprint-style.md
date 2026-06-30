# Blueprint Style — Format and Content

This file fixes both how a `blueprint.md` *looks on the page* (Part I, §§1–8) and *what it must say* (Part II, §§9–13). Applies to every file named `blueprint.md` under `docs/` (developer-guide, research papers, future siblings). Builds on top of [markdown-style.md](markdown-style.md) — only blueprint-specific deltas live here.

The cross-doc boundary rules (§12 — what belongs in blueprint vs `decisions.md` / `discussions.md` / `roadmap.md` / `references/`) are written here as a minimal source of truth. The canonical edition will eventually move to a doc-system overview / README that ties all sibling-doc standards together; this section is provisional until that exists.

## Table of Contents

**Part I — Format**

1. [Header Block](#1-header-block)
2. [Section Numbering](#2-section-numbering)
3. [Table of Contents](#3-table-of-contents)
4. [Status Section Format](#4-status-section-format)
5. [Changelog Format](#5-changelog-format)
6. [Cross-Links](#6-cross-links)
7. [Diagrams](#7-diagrams)
8. [Naming Conventions](#8-naming-conventions)

**Part II — Content**

9. [Required Sections (Spine)](#9-required-sections-spine)
10. [Body Variants — Platform vs Paper](#10-body-variants--platform-vs-paper)
11. [Section Content Rules](#11-section-content-rules)
12. [Boundary With Sibling Files](#12-boundary-with-sibling-files)
13. [Recurring Content Patterns](#13-recurring-content-patterns)

---

## 1. Header Block

Every blueprint starts with the same four lines, in this exact order:

```markdown
# Blueprint

last updated: YYYY-MM-DD HH:MM

---
```

- **Title is exactly `# Blueprint`.** No project / paper name in the H1 — the doc-site nav already supplies that context. Avoid `# AgentCanvas Blueprint`, `# Blueprint — Embodied AAS`, etc.
- **`last updated:` is mandatory and includes the time** (precision: minute). Get it via `date '+%Y-%m-%d %H:%M'`. Date-only is not accepted.
- The horizontal rule after `last updated:` is mandatory.

A **lede paragraph** (1–3 sentences orienting the reader and pointing to sibling files) MAY appear between the rule and the ToC. Its presence and content are a structural choice — see the structure standard (TBD).

## 2. Section Numbering

- Top-level body sections are numbered: `## 1. Section Name`, `## 2. ...`, etc.
- Subsections continue the number: `### 2.1 Subsection`, `#### 2.1.3 Sub-sub`.
- `## 0. Section Name` is reserved — only use it for **load-bearing axioms** (claims that, if violated, force a rewrite of the whole doc, not an edit). Do not use `§0` for ordinary "design principles".
- The trailing `## Changelog` is **not** numbered (it is metadata, not body).
- Section numbers are part of the doc's addressing scheme — once published, do not renumber a section without bumping every cross-reference. Insert as `§N.M.5` (sub-numbering) before renumbering.

## 3. Table of Contents

- A manual ToC is **required**, placed immediately before the first body section.
- The ToC heading is `## Table of Contents` (not `## Contents`, `## Index`, `## TOC`).
- Entries are numbered, mirror H2 / H3 numbering, and use anchor links: `1. [Goal](#1-goal)`.
- Nest at most one level (H3). Do not list H4 or deeper.
- A horizontal rule (`---`) follows the ToC.

## 4. Status Section Format

If the blueprint has a Status section (whether to include one is a structural choice — TBD), it MUST follow this shape:

```markdown
## N. Status

**Maturity:** `<placeholder | drafting | locked | shipped>`

<one to two sentences naming the current bottleneck and the next concrete step>
```

No multi-paragraph essays; no embedded changelogs; no decision rationale. Detail belongs in `discussions.md` or commit messages.

## 5. Changelog Format

- The `## Changelog` heading lives **after** a final horizontal rule, at the bottom of the file.
- Entries are reverse chronological (newest first).
- Each entry is **one line**:

  ```markdown
  - YYYY-MM-DD HH:MM: <≤140-char summary>. Rationale → [discussions.md §X](discussions.md#x)
  ```

- If an entry needs more than 140 characters of summary, the rationale belongs in `discussions.md` (or `decisions.md` if it records a closed decision) and the changelog line links there.
- No annotations like `(latest)`, `(later)`, `(major)` — chronological order is given by the timestamp.
- Date-only timestamps are accepted on legacy entries but new entries must include time.

## 6. Cross-Links

- Use **relative paths** for sibling files: `[roadmap](roadmap.md)`, `[decisions §3](decisions.md#3-x)`, `[references](references/index.md)`.
- Do not use site-absolute paths (`/research/...`) or `../..` chains that traverse above the doc-site root.
- When linking to a numbered section in another file, include the section number: `[discussions.md §3.6](discussions.md#36-ceiling-reframe)`.

## 7. Diagrams

- ASCII art is acceptable for diagrams up to ~5 boxes / arrows.
- Beyond that, use a Mermaid block (`​```mermaid`) — doc-site already loads the Mermaid plugin.
- Tables are preferred over diagrams whenever the relationship is "compare items along shared attributes".

## 8. Naming Conventions

These names are reserved when used as section / file titles, to avoid collisions across siblings:

| Section name in blueprint | Means | Sibling file with same role |
|---|---|---|
| `Open Questions` | unresolved design questions still under debate | — |
| `Status` | current maturity + bottleneck | — |
| (do **not** use) `Open Design Decisions` | ambiguous with `decisions.md` | use `Open Questions` instead |
| (do **not** use) `Decisions` | clashes with sibling `decisions.md` | put resolved decisions in `decisions.md` |
| (do **not** use) `Discussion` | clashes with sibling `discussions.md` | put rationale in `discussions.md` |

---

# Part II — Content

Part I governs how a blueprint *looks*. Part II governs what it must *say*: which sections are required, what each section must contain (and must **not** contain), and where content goes when it does not belong in blueprint.

These rules are **strict**: a blueprint that omits a required section or violates a content rule is non-compliant, not "stylistically light".

## 9. Required Sections (Spine)

Every blueprint MUST contain these elements, in this order:

1. **Lede paragraph** — between the header rule and the ToC.
2. **§1 Goal** (or **§1 Problem Statement**, synonyms — pick whichever fits the subject).
3. **Variant body** — see [§10](#10-body-variants--platform-vs-paper).
4. **Differentiation** — required when the doc claims novelty / improvement over existing work.
5. **§N−1 Open Questions** — required; an explicit `_None at this time._` is acceptable when everything is settled.
6. **§N Status** — required.

The body sections (item 3) carry their own numbering between Goal and Open Questions; their count and titles depend on which variant applies.

## 10. Body Variants — Platform vs Paper

A blueprint is one of two kinds. Pick the variant that matches the subject; do not mix.

### 10.1 Platform Blueprint

Describes a *built (or under-construction) system*. Used by `developer-guide/core/blueprint.md` and any future sibling describing a subsystem.

Required body sections (between §1 Goal and Differentiation), in this order:

| Section | Purpose |
|---|---|
| **Capabilities** | The user-visible feature claims. Each capability = a one-line claim + a "How" block giving the implementation in one paragraph. |
| **Design Principles** | Load-bearing rules the platform commits to (e.g., "framework has zero domain knowledge"). Each principle = the rule + a one-line consequence. |
| **Core Components** | Table: `Component` / `Location` / `Role`. One row per component the system can't function without. |
| **Core Features** *(optional)* | Categorized feature inventory for users navigating the platform. Skip if Capabilities already covers this exhaustively. |

### 10.2 Paper Blueprint

Describes a *paper that gets shipped*. Used by `research/**/blueprint.md`.

Required body sections (between §1 Goal and Differentiation), in this order:

| Section | Purpose |
|---|---|
| **Literature Review** | At minimum a synthesis paragraph plus one axis-table (rows = nearby papers, columns = differentiating axes). Detail goes in `references/`; this section is the synthesis only. |
| **Methodology / What Gets Built** | Components, operators, experimental harness. Include cost estimates (LOC, days, GPU-days) when build cost is non-trivial. |
| **Experimental Matrix** | Table: `Benchmark` × `Method` × `Seeds` × `Hero metric`. One row per benchmark. |
| **Faithfulness / Mandatory Defenses** *(when applicable)* | Operational rules, ablations, or writing positions that pre-empt likely reviewer attacks. Include only if the paper makes claims that summon known reviewer reflexes. |

## 11. Section Content Rules

For each spine section, what it must / must not contain.

### 11.1 Lede Paragraph

- 1–3 sentences.
- Must include: the subject in one phrase; a scope boundary ("does not cover X — see Y"); pointers to siblings (`roadmap.md`, `decisions.md`, `discussions.md`, `references/`).
- Must NOT: motivate / sell / pitch the subject. The lede orients, it does not advocate. Advocacy goes in §1 Goal.

### 11.2 Goal / Problem Statement

- States the load-bearing claim or problem in prose that quotes standalone (a reader citing one paragraph elsewhere should not lose meaning).
- Must include: a list of what counts as "done" / "demonstrated" / "shipped" — concrete, checkable items.
- Must NOT: design discussion, alternatives considered, decision rationale. Those go to `discussions.md`.

### 11.3 Differentiation

- Use a comparison table whenever comparing ≥2 attributes across ≥2 alternatives. Prose-only differentiation is non-compliant under this standard (per [feedback memory: tables for compare-on-shared-attributes](../memory/feedback_doc_section_subsection_correlation.md)).
- Each row = one nearest neighbor (paper, project, framework). The differentiator must appear in an explicit column, not implied by surrounding prose.
- Cite alternatives with paper / project links (relative within the doc-site; external links use plain URLs).

### 11.4 Open Questions

- Each item is a **question** or a **forced choice between named options**. Vague TODOs are non-compliant.
  - Bad: `consider per-episode budgeting`
  - Good: `Per-episode budget — fixed cost cap (X tokens) vs adaptive (target SR)?`
- When a question closes, move it to `decisions.md` as a decision entry; remove it from this list.
- An empty section is acceptable but must read explicitly: `_None at this time._`

### 11.5 Status

- Format covered in [§4](#4-status-section-format).
- Content: maturity tag + **one specific** bottleneck + the next concrete step.
  - Bad bottleneck: `drafting`, `in progress`.
  - Good bottleneck: `blocked on F8 implementation; next: write BaseSearchOperator base class`.

## 12. Boundary With Sibling Files

Blueprint hosts the **current commitments**: what we're building / claiming, and what's still open. The following content does NOT belong in blueprint:

| Content | Goes in |
|---|---|
| Resolved design decisions with rationale | `decisions.md` (one entry per decision) |
| Exploratory thinking, abandoned approaches, multi-paragraph rationale | `discussions.md` |
| Phasing, milestones, sequenced TODOs, dependency ordering | `roadmap.md` |
| Prior-art annotated bibliography, per-paper deep dives | `references/` |
| Step-by-step how-to / API specifics / tutorials | `developer-guide/tutorials/`, `ds-recipes/`, `ds-api-reference/` |

**Heuristic:** if a section runs >2 paragraphs explaining *why we chose X*, that is a `discussions.md` entry. Link from blueprint, don't inline.

**Heuristic:** if a changelog entry needs >140 chars to summarise the change, the rationale lives in `discussions.md`; the changelog line links there (per [§5](#5-changelog-format)).

## 13. Recurring Content Patterns

- **Differentiation tables.** Required for any "we're new / different vs X" claim — see [§11.3](#113-differentiation).
- **"What this is NOT claiming"** subsection. For any strong claim, append a subsection listing what the claim does *not* cover. Pre-empts reviewer over-reading. Example: `paper2 §8.3` (substrate-as-necessary-condition non-claims).
- **Quantitative anchors.** When claiming "complex / cheap / many / fast", give a number. `~200 LOC`, `~30 candidates × 100 episodes`, `<1s p99` — not `small / large / many`.
- **Sourceable claims.** Every claim that depends on a sibling fact (e.g. "scoring uses paper-reported $X$ — see [decisions.md §3](decisions.md#3-x)") must link the source. A claim without a linkable anchor is a candidate for moving to `discussions.md` until it has one.

---

## Changelog

- 2026-05-07 15:40: Added Part II — Content (§§9–13): required-sections spine, platform-vs-paper variant body schemas, per-section content rules, sibling-file boundary table, recurring patterns. Updated header to drop "format only" scope. Section numbering strict (decision: fork-2 → strict). Schema strategy: one core spine + two variant bodies (decision: fork-1 → C). Boundary rules placed here provisionally; canonical edition will move to a doc-system overview README later (decision: fork-3).
- 2026-05-07 15:13: Initial draft — format rules only (header block, numbering, ToC, Status / Changelog format, cross-links, diagrams, reserved names). Structural / content rules deferred.
