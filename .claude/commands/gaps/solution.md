# /gaps:solution — draft the proposal, then open the discussion

The **third** skill of the GAPS flow. The doc now has `§0 Ground`, `Part 1 —
As-is`, and a root-level `Problem` (from `/gaps:ground` + `/gaps:problem`). This
skill drafts `Part 2 — Solution` — the proposal — and then **hands the doc back to
the user for refinement discussion**. It deliberately stops before producing an
execution plan; the finished GAPS doc goes to a **Plan agent** downstream.

> A Solution is a *stance*, not a patch list. Resist "fix facet 1, fix facet 2, …" —
> if the Problem is one root with N faces, the Solution should be one move that
> catches all N (the example: "promote to an execution station" with two capabilities catches transport
> + topology + state + logging at once).

## Usage

```
/gaps:solution
```

No arguments. Operates on the GAPS doc finished by `/gaps:problem`.

## Steps

### 1. Re-read Ground + As-is + Problem
Load the whole doc. Pull out the Problem's facets and the positive goal — the
Solution must answer **each facet** and meet **each goal**.

### 2. Draft `Part 2 — Solution`
- **Lead with a top-level stance.** State the single reframe that dissolves the root,
  before any mechanism. (The example opens Part 2 by naming the root — "one stateless,
  one-way interface" — then proposes upgrading the subprocess to a first-class
  "execution station".)
- **Restate the load-bearing constraints** from Ground as design constraints (the
  example's constraint A / B). The Solution must not violate them; calling them out early
  prevents tempting-but-broken options.
- **Traceability table**: rows = the Problem's facets (+ goal), columns = where As-is
  blocks it / which part of the proposal catches it. This is the proof the stance
  covers the whole root, not part of it (example: the §2 intro table + §2.7).
- **End-state panorama + change surface**: an ASCII picture of the proposed end
  state, and a table of `# / file / what changes / which part it belongs to` (the
  example's §2.7). Concrete enough to scope, not so concrete it's a plan.
- Mark Part 2 `draft / for discussion` in its intro and in the `Status` callout.

### 3. Seed `§2.n Open questions / decision points`
- List the real forks the proposal leaves open — carrier choices, ownership rules,
  v1-vs-deferred scope, consistency/aliasing traps. Mark any already-decided ones
  struck-through with the resolution (example: ~~msgpack vs Arrow~~ → decided msgpack).
- Put a `← decide this first` marker on the question that unblocks the others.

### 4. Re-wrap + open the discussion
- Run `python3 docs/_lib/_wrap_handwritten.py`.
- Tell the user, in plain language: **the GAPS draft is complete (Ground + As-is +
  Problem + Solution); now refine it through discussion** — walk the open questions,
  push back on the stance, iterate Part 2 in place. The doc stays in docs-tmp and
  keeps refining until the open questions converge.
- State the **hand-off**: once converged, the finished GAPS doc is handed to a **Plan
  agent** to design the execution plan (ordered steps, edits, verification). This
  skill does **not** write that plan.

## Guardrails
- **Do not** write an execution plan, step ordering, or per-file diff sequence —
  that is the downstream Plan agent's job. The change-surface table is a *scope*,
  not a sequence.
- **Do not** silently close open questions to look finished. An honest open-questions
  list is the point of the discussion stage.
- One move for one root. If the Solution reads as a per-facet patchwork, you haven't
  found the stance yet — say so rather than shipping the patchwork.
