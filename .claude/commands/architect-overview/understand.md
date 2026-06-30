Read the architect-pipeline ecosystem at overview level: the hand-written method-level pages under `docs/pages/aas/reference/` plus the skill tree under `.claude/commands/architect/` (each variant's `README.md` + the shared `_common/files-contract.md`). Skill bodies (`loop.md` / `proposer.md` / `implementer.md` / `evaluator.md` for the ADAS-family variants; `loop.md` / `proposer.md` / `reflect.md` / `distill.md` for `myloop`, which has no implementer/evaluator skill — its EXPERIMENT phase is inline in `loop.md`), the auto-generated `docs/pages/aas/mirror/` tree, and the raw upstream paper / v1 HTML under `reference/adas/{paper,skills}/` are skipped by default and loaded only via scope flags.

For developer-side context use `/overview:understand`. For research-side context use `/research:understand`. This command is the **architect-side analogue** — same idea (read one canonical tree and present a structured digest), different directories.

## Args

`/architect-overview:understand [scope]` — `scope` selects what to read:

- (empty, default) — **method reference + variant skill-tree READMEs**. `algorithm.html` files (one per method) + landing pages + `_common/files-contract.md` + each variant's `README.md` (3 variants). ~9 files, light.
- `<variant>` — drill into one variant's full skill set. For the ADAS-family variants reads `<variant>/{README,loop,proposer,implementer,evaluator,understand}.md`; for `myloop` reads `myloop/{README,loop,proposer,reflect,distill,schemas,understand}.md` (myloop has **no** `implementer.md` / `evaluator.md` — its EXPERIMENT apply+smoke+eval is inline in `loop.md § 3c`). Plus `_common/files-contract.md` and (if present) `config.yaml`. Variant names: `adas-subagent`, `aflow`, `myloop`.
- `mirror` — the auto-generated `docs/pages/aas/mirror/` tree. Normally skip — it is a 1:1 HTML render of the same skill `.md` already read in `<variant>` scope. Use only when investigating the `build_mirror.py` rendering itself.
- `full` — default scope **plus** every variant's full skill set **plus** the upstream paper HTML under `reference/adas/paper/index.html`. Large — only invoke when about to write a cross-variant comparison or land a structural change that touches the shared contract.

## Steps

### Step 1 — Read landings + shared contract (always)

Read in parallel:

- `docs/pages/aas/README.md` — site purpose, reference-vs-mirror split, rebuild command
- `docs/pages/aas/index.html` — landing card layout (two cards into reference/ vs mirror/)
- `docs/pages/aas/reference/index.html` — per-method algorithm cards + the side-by-side comparison table
- `docs/pages/aas/reference/files-contract.html` — reader's guide to the shared contract
- `.claude/commands/architect/_common/files-contract.md` — the canonical shared contract (run-dir layout, resolve protocol, edit whitelist, `{graph}.yaml` profile schema, eval API)
- `.claude/commands/architect/_common/understand.md` — the shared variant-understand skill template (so you know what `<variant>:understand` actually loads)

### Step 2 — Read method-level reference (default + `full`)

Read in parallel — one `algorithm.html` per method:

- `docs/pages/aas/reference/adas/algorithm.html` — ADAS port (3-call Reflexion, archive-driven)
- `docs/pages/aas/reference/aflow/algorithm.html` — AFlow (MCTS over code-represented workflows)
- `docs/pages/aas/reference/fastsmartway/algorithm.html` — Fast SmartWay (IROS 2025) port
- `docs/pages/aas/reference/myloop/algorithm.html` — AAS-extended hill-climb

Do **not** glob `docs/pages/aas/reference/adas/paper/` at default scope — the paper PDF render is forensic / on-demand.

### Step 3 — Read variant skill-tree READMEs (default + `full`)

Read each variant's `README.md` in parallel. These document the per-variant structural deltas + skill map:

- `.claude/commands/architect/adas-subagent/README.md`
- `.claude/commands/architect/aflow/README.md`
- `.claude/commands/architect/myloop/README.md`

### Step 4 — Variant drill-down (`<variant>` scope only)

When `scope` matches a variant name, read in parallel:

- `.claude/commands/architect/<variant>/README.md`
- `.claude/commands/architect/<variant>/loop.md` — orchestrator (for `myloop`, also owns the inline EXPERIMENT apply+smoke+eval — § 3c)
- `.claude/commands/architect/<variant>/proposer.md` — design generation
- `.claude/commands/architect/<variant>/implementer.md` — applying proposed edits **(adas-subagent / aflow only — `myloop` has no implementer skill; skip for myloop)**
- `.claude/commands/architect/<variant>/evaluator.md` — fitness measurement **(adas-subagent / aflow only — `myloop` has no evaluator skill; skip for myloop)**
- `.claude/commands/architect/<variant>/understand.md` — the variant's own understand skill (a thin stub binding VARIANT + delegating to `_common/understand.md` for the ADAS family; for `myloop` a standalone file)
- `.claude/commands/architect/<variant>/config.yaml` — variant-specific knobs (read by `_common/understand.md` for the ADAS family; for `myloop` it holds paths/caps/EXPERIMENT-knobs + the files manifest)
- `.claude/commands/architect/_common/files-contract.md` — shared contract (re-read in case skipped Step 1)

Variant-specific extras to also load when present:

- `myloop/distill.md`, `myloop/schemas.md` — myloop's extended phases
- `myloop/lib/`, `adas-subagent/lib/` — list the directory (do not read every file); call out which files exist so the user knows what helper code each variant ships with

Skip `<variant>/data/` payloads (large captured run artefacts) unless the user asks.

### Step 5 — Mirror / full extras

**`mirror` scope**: glob `docs/pages/aas/mirror/_common/*.html` + one variant's `docs/pages/aas/mirror/<variant>/*.html` — but only after confirming the user actually wants the rendered HTML, since each file is a derivative of an `.md` already covered in `<variant>` scope.

**`full` scope only** — read every variant's full skill set (ADAS family: loop / proposer / implementer / evaluator / understand / config.yaml; `myloop`: loop / proposer / reflect / distill / schemas / understand / config.yaml) **plus** the upstream paper HTML:

- `docs/pages/aas/reference/adas/paper/index.html` — paper landing
- Skip the `.pdf` itself unless asked (`adas-arxiv-2408.08435.pdf`)

This bumps the read budget from ~30 K tokens to several hundred K — only invoke when writing a cross-variant comparison, editing the shared contract, or porting upstream changes.

### Step 6 — Present a summary with this structure

Synthesise from what was just read. Do **not** summarise through an intermediate digest — pull facts straight from the files.

1. **Method snapshot table** — one row per method (default scope: 3 methods from `reference/*/algorithm.html`; variant scope: just the in-scope variant):

   | Method | Core algorithm (one line) | Search space | Reasoning module paradigm | Status |
   |---|---|---|---|---|

   "Reasoning module paradigm" = stateless LLM call vs tool-augmented Claude conversation (this is the framing axis used in `adas-subagent/README.md`).

2. **Variant map** — for the architect skill tree, list which `.claude/commands/architect/<variant>/` corresponds to which method in §1. Flag variants whose `algorithm.html` is missing from `reference/`.

3. **Shared contract surface** — pull from `_common/files-contract.md`:
   - run-dir layout (`workspace/design_runs/{graph}/vN/iter_M/`)
   - resolve protocol (how `<graph> [<version>]` args fuzzy-match)
   - edit whitelist (which files a variant may touch)
   - `{graph}.yaml` profile schema sections every variant must populate
   - eval API surface

4. **Per-variant deltas** — at default scope, two lines per variant pulled from each `README.md`'s opening framing:
   - **What is preserved verbatim** from upstream (if applicable)
   - **The structural delta** vs siblings (e.g. adas-subagent's "tool-augmented sub-agent per Reflexion round" vs aflow's "MCTS over code-represented workflows")

5. **`reference/` vs `mirror/` distinction** — restate the contract from `docs/pages/aas/README.md`: `reference/` is hand-written conceptual; `mirror/` is auto-generated from skill `.md`. Remind the user that `mirror/` rebuilds idempotently via `python docs/pages/aas/build_mirror.py` and should never be hand-edited.

6. **Pointer map for on-demand reads** — list which file class to consult for which question. Examples:
   - "How does ADAS's archive prompt evolve?" → `reference/adas/algorithm.html` + `adas-subagent/proposer.md`
   - "Can a variant write to `workspace/graphs/`?" → `_common/files-contract.md` § edit whitelist
   - "What does myloop's distill phase do?" → `myloop/distill.md` + `myloop/schemas.md`
   - "Render the rendered HTML for skill X" → `docs/pages/aas/mirror/<variant>/<skill>.html`

### Step 7 — Remind the user of related commands

- `/overview:understand` — developer-side analogue (reads `developer-guide/core/` + `capabilities/` + ADR field indices)
- `/architect-overview:understand <variant>` — drill into one variant's full skill set
- `/architect-overview:understand full` — superset (every variant's full skill set + upstream paper HTML)
- `/architect:<variant>:understand` — the variant's **operational** understand skill (loads run-state + JIT framework docs; different from this overview-level read)
- `python docs/pages/aas/build_mirror.py` — regenerate `docs/pages/aas/mirror/` after editing any skill `.md`
