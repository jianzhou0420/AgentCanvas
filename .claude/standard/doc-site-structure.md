# Doc-site folder structure

The doc-site under `docs/` is hand-authored HTML — no markdown sources, no build step. Filesystem layout is `docs/{index.html, assets/, pages/{developer-guide,research}/...}` with build infra in `docs/_lib/`. Sidebar navigation comes from `docs/_lib/_nav.py`, not from a filesystem walk alone. **Read before** moving folders, renaming sections, or adding a new top-level section.

> Page-level content rules (chrome, headings, cross-links) live in [html-authoring.md](html-authoring.md) — read that for content, this for layout.

---

## 1. Two tabs, fixed list of sections

`docs/_lib/_nav.py` declares two top-level tabs (Developer Guide / Research) and the ordered list of sections under each. Each entry is a dict:

```python
{"label": "Capabilities", "dir": "capabilities", "key": "capability"}
```

- `label` — sidebar group label (whatever you want)
- `dir` — docs-relative directory (e.g. `pages/developer-guide/capabilities`); pages are `*.html` inside, recursively
- `key` — CSS accent class (`guide`, `capability`, `designdoc`, `research`, `decisions`, `blueprint`, …)
- `files_only: True` — only include top-level `.html`, no nested subdirs (used by Research's "Overview" section to surface flat files at `pages/research/`)
- `depth_limit: 2` — include top-level files + each subdir's `index.html` as a flat entry (rare; use when a section shouldn't expand its full tree)

Within a section, pages are discovered alphabetically by filename stem with `index.html` pinned first. Subdirectories become nested collapsible `<details>` groups in the sidebar; the subdir's `index.html` provides the group's title and href.

## 2. Adding a new section

1. Create the directory under `docs/pages/{tab}/`, drop your `*.html` files in (use `index.html` if you want a landing page).
2. Add a new entry to the matching tab's `sections` list in `docs/_lib/_nav.py`.
3. Pick an existing `key` (CSS accent) or add a new `.page-header.<key>` rule in `assets/style.css`.
4. Run `python3 docs/_lib/_wrap_handwritten.py` to re-render chrome on every page. The script walks `docs/index.html` + `docs/pages/**/*.html`, extracts each `<main>` block, and re-renders via the new nav.

## 3. Adding a sub-folder inside an existing section

Just create the subdirectory and drop pages in. The sidebar auto-renders it as a nested `<details>` group — no nav config needed. If the subdir has `index.html`, its `<title>` becomes the group label. Run the wrap script afterwards to propagate the new sidebar to every page.

## 4. Section ordering inside `_nav.py`

Order in the Python list is the order in the sidebar. Current grouping:

- **Developer Guide**: Getting Started · Core (with nested Decisions) · Capabilities · Design Docs · DS-* (data-science guides) · Tutorials · Nodesets · Demos · Community · Process
- **Research**: Overview (flat files at `docs/pages/research/`) · Two Papers · Agent Lit Review · Embodied AI Lit Review · Misc · Presentations

Within a section, an optional `_order.json` (JSON list of subdir names) in the section dir pins child-group order; unlisted subdirs follow alphabetically.

## 4b. Sub-grouping discipline (membership tests)

When a section's flat page list grows unwieldy, group pages into subdirs **by contract owner** — which layer owns what the page specifies — never by topic similarity or reader journey. Rules (ratified 2026-06-11 with the design-docs regrouping):

1. **One membership test per group**, stated as a single line in the group's `index.html` lede. A page joins a group only if it passes the test. Example (design-docs/components/): "does this page define a contract you implement when authoring a component?"
2. **Pages failing every test stay flat at section root.** Flat is the legal default, not a failure — don't force-fit.
3. **One nesting level only.** Sidebar `<details>` beyond one level is unusable; deeper structure goes into page sections.
4. **Minimum group size 2.** A single-page group means the axis is wrong; return the page to root.
5. **New pages default to flat** until a membership test clearly claims them.

Current design-docs groups: `graph/` (graph representation + execution; ADR fields dataflow+executor) · `components/` (authoring contracts; canvas+components) · `operations/` (run + observe; server+eval+observability) · `surfaces/` (programmatic APIs for external programs; platform). Each group index declares its ADR-field correspondence — the two taxonomies cross-reference but are not forced isomorphic.

Support-status registries (VLN/VLA/EQA) live at `nodesets/status/` — they are registries that change with nodeset coverage, not design docs.

## 5. Per-page metadata overrides

`HANDWRITTEN` in `docs/_lib/_wrap_handwritten.py` is a small dict of page-specific overrides for auto-derived metadata — only needed when a page wants a non-default `section_class` (CSS accent) or a custom `has_right_toc` flag. For most pages the wrap script auto-derives everything (tab, breadcrumbs, default section_class) from the file path.

Override targets currently:
- `index.html` (site root) — `section_class=blueprint`, `has_right_toc=False`
- `pages/developer-guide/core/{index, blueprint, architecture, codebase-map, glossary, roadmap, roadmap-done, major-versions}.html` — `section_class` for accent colour
