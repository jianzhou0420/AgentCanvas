# /gaps:ground — write Ground + As-is

The **first** skill of the GAPS flow (Ground · As-is · Problem · Solution). Given a
target — a feature, subsystem, nodeset, or any code about to change, large or small —
create the GAPS document's `§0 Ground` and `Part 1 — As-is`, **grounded in the
target's mental model and its actual code**. Leave the Problem as a stub for
`/gaps:problem`. Scale the depth to the change — a small target needs only a short
Ground + a one-facet As-is.

> Read `.claude/commands/gaps/README.md` (the GAPS mental model) before invoking if
> you don't already have it. This skill writes **only** the Ground and As-is bands —
> it invents no problems and proposes no solutions.

## Usage

```
/gaps:ground <target>          e.g. /gaps:ground server-mode transport
```

`<target>` names the feature/subsystem. If absent, ask the user one line for it.

## Why ground first

A proposal is only as good as the current-state model it rests on. If As-is is
hand-waved, the Problem will be a symptom and the Solution will patch the wrong
layer. As-is must be **grounded**: every claim about today's behaviour points at a
`file:line` and, ideally, a real call site — so the later bands argue against the
code, not a memory of it.

## Steps

### 1. Scope the target + build the mental model
- Restate the target and its boundary in one line; confirm with the user if fuzzy.
- **The mental-model source depends on what kind of object the change targets** —
  read it from the doc-site / code, not from memory. The target is *not* always a
  framework feature with an ADR; match the row(s) that fit:

  | Target kind | Where its mental model lives |
  |-------------|------------------------------|
  | Framework feature / subsystem | `core/{blueprint,architecture,glossary}.html` + the `core/decisions/<field>/` ADR(s) that introduced it + any `core/../design-docs/` page |
  | A capability | `core/../capabilities/<name>.html` — the capability page *is* the spec |
  | A design-doc-level mechanism | the relevant `design-docs/` page (e.g. `graph/state-containers.html`, `loop-control-system.html`) |
  | A nodeset | `nodesets/<name>.html` doc (if present) + the nodeset code under `workspace/nodesets/<role>/<name>` + `.claude/standard/nodeset-layout.md` + any topic memory |
  | A graph / composite | the graph JSON in `workspace/graphs/` + `.claude/tutorials/skill-graph-json.md` |

  Most targets pull from more than one row. These give the *why* and the
  load-bearing terms — the raw material for Ground.
- **The user may point you directly** — at the target ("ground `env_libero`") or at
  its mental-model source ("the model lives in `design-docs/graph/state-containers.html`").
  Honor that pointer first, then auto-discover the rest from the rows above.
- **If no mental-model source exists** (no doc page, no ADR, code-only or scattered
  across files), **do not fabricate one.** Report the gap to the user — say plainly
  what you looked for and didn't find — and **discuss to settle a working mental
  model together** before grounding: agree what the target *is for* and where its
  boundary sits, then capture that agreement as the basis of `§0 Ground` (note in the
  `Status` callout that the model was co-defined here, not pre-existing). An absent
  mental model is a finding, not a blocker.
- Read the **actual code** the target lives in (framework package, nodeset module,
  or graph JSON). Note the files + key functions you will cite. A grounded As-is
  needs `file:line`, not paraphrase.

### 2. Create the docs-tmp page
- Path: `docs/pages/developer-guide/tmp/<slug>.html` (kebab-case slug). Reuse an
  existing tmp page (`transport-signal-discussion.html`) as a structural template —
  copy its `<head>` + layout chrome, then replace `<main>`.
- Add a `Status:` callout right after the lede: `scratch / discussion draft`, what As-is is
  grounded to (`file:line` list), and a "candidate landing on finalize:
  `design-docs/` + a domain ADR" line.
- Follow `.claude/standard/html-authoring.md` (numbered `<h2>/<h3>` with stable
  kebab ids, page-relative links) and `.claude/standard/code-highlighting.md` for
  any code block.

### 3. Write `§0 Ground` — what the feature is *for*
- The **purpose**: why this target exists, what forces *made* it necessary. Quote
  the doc or ADR that introduced it where possible — an ADR for a framework
  decision (the example quotes ADR-server-001: *"deployment topology becomes a
  config decision, not a code decision"*), or the capability / nodeset / design-doc
  page for those target kinds.
- If the feature serves more than one purpose, separate them and say which is the
  **live forcing function** today vs a reserved-but-unused capability — this framing
  feeds the constraints the Solution must respect.
- End Ground by naming the **tension** that the rest of the doc will examine (the
  example: "single-process gives three things for free; the process boundary breaks
  all three"). This is the bridge into As-is — but do **not** yet write it as a
  problem.

### 4. Write `Part 1 — As-is` — how it works today
- One **sub-section per facet** of the current implementation. Lead each with a
  one-line conclusion, then the grounded detail.
- **Ground everything**: `file:line` citations, a real call site / data path, ASCII
  diagrams for control/data flow, tables for "what goes which way". Mirror the
  example's `§1.1` two-paths table, `§1.2` round-trip diagram, `§1.4` folk-workaround
  walkthrough.
- Where today's behaviour rests on an **implicit assumption** (co-location, single
  writer, a `DO-NOT clear` comment), surface it explicitly — these become the seams
  the Problem and Solution hinge on.
- Stay descriptive. Note *that* something is fragile, but do not yet declare it the
  problem or sketch a fix.

### 5. Stub the Problem
- Add the final As-is sub-section `§1.n Problem` as a **stub**: a single
  `<div class="callout note">` reading *"filled by `/gaps:problem` — grill the user
  to the root, then fold in adjacent Claude-found problems."*

### 6. Re-wrap + hand off
- Run `python3 docs/_lib/_wrap_handwritten.py` so the new page enters nav + search.
- Tell the user: Ground + As-is are grounded and on the dev server (give the
  `http://…:8092/pages/developer-guide/tmp/<slug>.html` URL); the next step is
  `/gaps:problem` to locate what we are actually solving.

## Guardrails
- **Do not** write a problem statement or any Part 2 / solution content. That is the
  job of the next two skills.
- **Do not** ground As-is from memory or from the ADR text alone — open the code.
- Keep Ground tight; the depth belongs in As-is.
