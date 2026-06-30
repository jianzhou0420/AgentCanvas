# aas — Agent Architecture Search site

Plain static HTML site for the AgentCanvas architect pipelines.
Two parts: **reference/** (hand-written, stable) and **mirror/**
(auto-generated from skill markdown).

Lives **inside `docs/`** as the `AAS` top tab; served by the same
`docs/run_dev.sh` dev server. Its internal layout is intentionally
its own (1080px max-width, own CSS) — a thin cross-site nav strip
is injected at the top of each page by `docs/_lib/_aas_nav.py`
so users can hop back to Developer Guide / Research.

**reference/** is a *collection of methods* — one high-level,
hand-written understanding page per architect method (helps both
development and a reader new to the method). **mirror/** is the
literal execution view — what each skill actually looks like,
auto-rendered from the skill markdown.

## Layout

```
docs/pages/aas/
├── index.html                Landing — two cards into reference/ vs mirror/
├── build_mirror.py           Python script: regenerates mirror/ from skill .md
├── README.md
│
├── reference/                Part 1 — hand-written, stable; the method collection
│   ├── index.html            Per-method algorithm cards + side-by-side table
│   ├── files-contract.html   Reader's guide to .claude/commands/architect/_common/files-contract.md
│   ├── adas/                 ADAS port — algorithm + paper + v1 per-skill HTML
│   ├── aflow/                AFlow — MCTS over code-represented workflows
│   ├── as/                   AgentSquare module-search loop
│   └── myloop/               AAS-extended hill-climb
│
└── mirror/                   Part 2 — auto-generated (do not hand-edit)
    ├── index.html            Tree view, grouped by variant
    ├── _common/              1:1 HTML mirror of .claude/commands/architect/_common/*.md
    ├── adas-subagent/, aflow/, myloop/
    └── ...
```

## View

```bash
# Recommended: served as part of the docs/ dev server (with cross-site nav)
bash docs/run_dev.sh
# → http://127.0.0.1:8092/aas/index.html

# Or open directly (cross-site links still work, just no live reload)
xdg-open docs/pages/aas/index.html
```

## Rebuilding `mirror/`

```bash
# from repo root
python docs/pages/aas/build_mirror.py
# The script auto-runs docs/_lib/_aas_nav.py afterward, so the
# regenerated pages keep the AAS↔docs cross-nav strip.
```

What it does:

- Walks `.claude/commands/architect/**/*.md` (50 files across 8
  variant dirs).
- For each `.md`, renders one HTML page under `mirror/` preserving
  directory structure (e.g. `adas/loop.md` → `mirror/adas/loop.html`).
- Adds breadcrumbs, an auto-generated TOC, and a footer pointing
  back to the source markdown path.
- Rewrites intra-mirror `.md` cross-references (in `<a href>`) to `.html`.
- Clears `mirror/` and rebuilds from scratch on every run (idempotent).

Requires the `markdown` PyPI package:

```bash
pip install markdown
```

The script is otherwise dependency-light (stdlib only).

## Source of truth

| Page | Source of truth |
|------|-----------------|
| `reference/<method>/algorithm.html` | Copy of `.claude/commands/architect/<variant>/algorithm.html` — skill folder wins on drift |
| `reference/files-contract.html` | Hand-written reader's guide; the contract itself is `.claude/commands/architect/_common/files-contract.md` |
| `reference/adas/` | ADAS understanding page + paper. Current port is `adas-subagent` (see `mirror/`) |
| `mirror/**/*.html` | Auto-generated from `.claude/commands/architect/**/*.md` — never hand-edit; rerun `build_mirror.py` |

To sync `reference/<variant>/algorithm.html` after editing the
skill-folder copy:

```bash
cp .claude/commands/architect/myloop/algorithm.html docs/pages/aas/reference/myloop/
# (also aflow/ if it exists in skill folder)
```
