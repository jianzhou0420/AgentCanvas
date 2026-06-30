# /gaps:result — present the outcome (Part 3), SVG as an aid

The **fourth** skill of the GAPS flow. After the `Part 2 — Solution` has converged
in discussion (and, optionally, after the change has landed), produce `Part 3 —
Result`: a fast, visual presentation of the outcome. It does **not** introduce new
design decisions — it presents the converged Solution so a reader (or a supervisor)
can grasp the end state quickly.

> **SVG is an aid, not the whole section.** A Before → After diagram is the single
> most powerful way to make a change legible — but the *substance* of the result
> (what changed, whether it worked, the numbers, the caveats) lives in a short
> outcome callout and prose/tables too. Don't reduce Part 3 to a caption-only
> diagram gallery. Match the depth to the change: a small one may need just a
> before/after table and two lines; a big one earns 2–3 diagrams.

## Usage

```
/gaps:result
```

No arguments. Operates on the GAPS doc whose Solution has converged. If you run it
before convergence, you are presenting a moving target — kick back to discussion.

## Steps

### 1. Re-read the whole doc
Especially the Solution's **stance**, **end-state**, and **change-surface table**,
plus the **As-is diagrams** in Part 1 — you reuse the As-is picture as the "before".

### 2. Lead with an outcome callout (the substance)
Open Part 3 with a `callout` stating **landed vs proposed** and the concrete facts:
files shipped, tests/smoke run + result, metric (e.g. SR), and any unplanned
deviations found during implementation. This is text, and it carries the real
result — the diagrams below visualize it, they don't replace it. (See the worked
example's Part 3 status callout: "landed… smoke 10w×10ep, SR 0.30, 0 error".)

### 3. Add the visual aids that actually help
- **Before → After** is the natural headline — As-is topology vs Solution
  end-state, side by side. Reuse / restyle the As-is ASCII diagram as the "before"
  so the two are comparable.
- Add a **flow** or **change-surface** diagram *only if it earns its place* (the
  example uses three: before→after, keyed lifecycle, change-surface-by-layer). Two
  to three diagrams is plenty; a small change may need none — a before/after table
  is a legitimate substitute.
- Each SVG gets a **substantive one-line caption** stating the takeaway + landed
  status, not a decorative label.

### 4. Author SVG in the doc-site idiom (when you use it)
Lift the `<style id="…-diagrams">` `.rxsvg` block and the `<marker>` def from
`core/architecture.html` or `core/codebase-map.html` and adapt:
- `<svg class="rxsvg" viewBox="0 0 W H" role="img" aria-label="…full prose description…">`
  — the `aria-label` must convey the whole picture to a screen reader.
- **Theme-aware only**: fill/stroke via CSS vars (`var(--fg)`, `var(--muted)`,
  `var(--bg-alt)`, `var(--fg-dim)`, an accent var). **No hardcoded hex** — must work
  light and dark.
- Reuse the helper classes: `.box` / `.accent` (emphasis) / `.ghost` (dashed group) /
  `.edge` (arrow, `marker-end`) / `.dim` / `.mono`.

Tables/prose/ASCII `<pre>` are all fair game in Part 3 — use whichever shows the
outcome most clearly. Code blocks follow `.claude/standard/code-highlighting.md`;
everything else follows `.claude/standard/html-authoring.md`.

### 5. Re-wrap
Run `python3 docs/_lib/_wrap_handwritten.py` so the new section enters the right-TOC.
Update the top `Status` callout: result presented; landed vs proposed.

## Guardrails
- **SVG aids, it doesn't replace.** The outcome's substance (what changed, did it
  work, the numbers, the caveats) must be readable in text/tables even with the
  images stripped. A diagram with no informative caption is decoration — cut or
  caption it.
- **Theme-aware + accessible.** Every SVG uses CSS vars (never hex) and carries an
  `aria-label` that conveys the whole picture.
- **Present, don't redesign.** Part 3 reflects the converged Solution. If drawing it
  exposes a gap, that's a finding — kick back to `/gaps:solution` / discussion, don't
  invent a fix in the picture.
- **Scale to the change.** Not every Result needs a diagram; a before/after table +
  a couple of lines is the right size for a small change.
