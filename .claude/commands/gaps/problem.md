# /gaps:problem — grill to a root-level Problem

The **second** skill of the GAPS flow. The doc already has `§0 Ground` + `Part 1 —
As-is` (from `/gaps:ground`) with a stubbed Problem section. This skill
**interrogates the user** until the real problem is located, writes it as the
`Problem` that closes the As-is band, then **proposes adjacent problems Claude
found** while grounding As-is — folding in the ones the user agrees to solve together.

> The skill's core discipline: **do not accept a vague problem.** A symptom
> ("server mode crashes sometimes") is not a problem. Keep grilling until it is
> grounded to a mechanism and an As-is sub-section.

## Usage

```
/gaps:problem
```

No arguments. Operates on the GAPS doc created by `/gaps:ground` (ask which page if
more than one tmp doc is open).

## Steps

### 1. Re-read Ground + As-is
Read the whole doc so the grilling is grounded in the As-is facets, not generic.
List to yourself the seams `/gaps:ground` surfaced (implicit assumptions, fragile
spots) — these are your candidate roots.

### 2. Ask, then grill
- Open with one question: **"What problem did you hit — what went wrong, or what do
  you want to change?"**
- Then **grill**. Each round, drive from symptom toward root:
  - *symptom* → "what did you actually observe?" (error, wrong number, fragility)
  - *mechanism* → "which step in As-is produces it?" — pin it to a `§1.x` sub-section
    and a `file:line`.
  - *root* → "is this the cause or another symptom of something deeper?" Push until
    the user (or the grounded code) agrees you've hit bottom.
- Use `AskUserQuestion` for genuine forks (which of two roots, scope in/out), and
  one-line questions otherwise. **Do not stop at the first plausible answer** — a
  GAPS problem is worth one or two more "why"s than feels comfortable.
- If the user's framing contradicts the grounded As-is, say so and reconcile before
  writing.

### 3. Write the Problem
- Write `§1.n Problem` stating the **root**, not the symptom. Tie it back to the
  As-is facets: each part of the problem links to the `§1.x` that exhibits it.
- If the root is one cause with several faces, use the example's shape: one
  root-line, then a `callout warn` per facet, each facet naming the `§1.x` it comes
  from and the symptom it produces (the example: *transport / topology / state* — three faces, one root).
- State the **goal** positively too (the example: *correct / efficient / location-transparent*) — the bar
  the Solution will be measured against.

### 4. Propose adjacent problems Claude found
- From grounding As-is, you likely noticed **related problems sharing the same root**
  that the user didn't raise. Present them as a short list: each = one line + the
  `§1.x` evidence + why it's same-root.
- Ask the user, per item, **"solve together in this doc, or out of scope?"** Respect
  the answer — `less but right`, don't pad the problem with things the user wants to defer.
- Fold accepted items into the Problem (add a facet or a bullet), keeping the "one
  root" framing intact. Note rejected ones in one muted line as explicitly out of
  scope, so they aren't re-litigated.

### 5. Re-wrap + hand off
- Run `python3 docs/_lib/_wrap_handwritten.py`.
- Tell the user the Problem is set, recap the root in one plain sentence (per memory
  `feedback_bug_reports_plain_language`), and point to `/gaps:solution` to draft the
  proposal.

## Guardrails
- **Do not** write any solution / Part 2 content — even if the fix feels obvious.
  Naming the fix here pre-empts the Solution discussion.
- **Do not** let the problem stay at symptom level. If you cannot tie it to an As-is
  sub-section, you haven't grilled enough.
- Surface, don't bury, any framework/code surprise found while grilling (memory
  `feedback_report_framework_holes`).
