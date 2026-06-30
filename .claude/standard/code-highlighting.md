# Code Highlighting (doc-site)

**Scope: every code block pasted into a `docs/` page.** Inline `<code>` fragments are exempt (monospace styling is enough).

## The rule — two-stage static pipeline

**Stage 1 — tokenize offline, paste static spans.** All highlighting is done at authoring/sync time by Pygments, producing plain `<span class="…">` markup that goes into the page verbatim. The token classes are Pygments' short names (`k`/`s2`/`c1`/`nf`/`bp`/…); their colours live **once** in `docs/assets/style.css` under the `.hl` scope, GitHub-Primer palette, light + dark via `[data-theme]`.

**Stage 2 — decorate on top (optional).** Line-number gutters, band headers, cross-links, sync stamps are added *around* the stage-1 output. Stage 1 guarantees spans never cross line boundaries, so decorators can interleave per line freely.

## How to produce a block

- **One-off paste** (a snippet in any page):

  ```bash
  python docs/_lib/_hl.py path/to/file.py [--lang python] [--gutter]
  ```

  prints a paste-ready `<pre class="hl">…</pre>` block. `--lang` accepts any Pygments lexer name (`python`, `typescript`, `json`, `bash`, …); stdin works too.

- **Synced listing** (a block that must track a source file): follow the `docs/_lib/_sync_run_listing.py` pattern — `<!-- xxx:begin/end -->` markers in the page, a script that extracts the source, calls `_hl.highlight_lines()`, decorates, and writes between the markers with a **commit-pinned sync stamp**. Band split keys off the source file's own banner comments; doc-side §-cross-links live in the script's mapping table, never in the source.

## Forbidden

- **Client-side JS highlighting** (highlight.js et al.) — render flash, JS dependency, and it re-tokenizes whole blocks so it destroys stage-2 markup (band headers, gutters, anchors). The 2026-06-08 hljs experiment (`006de0e2`) was dismantled for these reasons; do not re-wire it.
- **Hand-rolled tokenizers** — Pygments is the single tokenizer. (The interim regex tokenizer in `_sync_run_listing.py` was replaced 2026-06-11.)
- **Editor-exported HTML** (VSCode copy-with-formatting etc.) — VSCode's semantic layer is Pylance-bound and not reproducible outside the editor (the abandoned `docs/_tools/vscode-highlight` attempt); pasted editor HTML can never be regenerated.
- **Page-local palettes** — token colours are defined once in `style.css`; pages must not carry their own copies. (Watch for class collisions when a page has legacy span CSS: e.g. `.p` is Pygments' punctuation class.)

## Class contract

The displaying `<pre>` must carry `hl` (alone, or alongside page-local layout classes, e.g. `class="rxcode hl"`). Environment note: use the `agentcanvas` env's python (Pygments 2.20); the system python3 ships 2.3.1.

## Comments-as-annotations (synced listings)

For listings synced from our own source files, the annotation layer **is the source comments** — explanations belong in the file (file-level self-coherent, no §-numbers in code), and the doc side adds only structure (bands, §-links). If a listing needs a better explanation, improve the source comment, not the HTML.
