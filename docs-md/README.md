# docs-md

Markdown documents meant to be read **directly on GitHub** (issues references, design notes, RFCs, walkthroughs, etc.).

Not part of the HTML doc-site at `docs/` — that one is HTML-first and rendered by `docs/_lib/_wrap_handwritten.py`. Markdown placed here is rendered by GitHub's web UI, nothing else.

## When to put a file here

- It targets readers browsing the repo on github.com.
- It doesn't need the doc-site's sidebar / cross-tab nav.
- It isn't a top-level convention file (README, CONTRIBUTING, INSTALL, LICENSE, VERSIONING — those stay at the repo root).

## When NOT to use this folder

- User-facing product/feature docs → `docs/pages/developer-guide/` (write as HTML).
- AAS / architect site → `docs/pages/aas/`.
- GitHub-specific config (workflows, issue templates, CODEOWNERS) → `.github/`.
- Personal research/scratch → stays gitignored under `docs/pages/research/` or outside the repo.
