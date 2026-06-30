# Design-Doc Style — what a good design doc must do

Rules for every page under `docs/pages/developer-guide/design-docs/**` (the
`graph/`, `components/`, `operations/`, `surfaces/` groups). This standard fixes
**what a design doc must say and in what order**; it builds on three siblings and
only adds design-doc-specific deltas:

- **page mechanics** (anatomy, numbered `<h2>/<h3>` ids, lede, last-updated,
  re-wrap) → [html-authoring.md](html-authoring.md)
- **placement + sub-grouping** (which group a page joins, membership tests) →
  [doc-site-structure.md](doc-site-structure.md) §4b
- **code blocks** (static Pygments spans, no client highlighter) →
  [code-highlighting.md](code-highlighting.md)

Read those for the *how-to-type-it*; read this for *what makes the page good*.
Before authoring, open a canonical exemplar (see end) and match its shape.

---

## The one rule (inverted pyramid)

> **A reader must know what this thing does and how it does it from the first
> screen — before any detail section. Later sections only add depth.**

Everything below is this rule made concrete. A design doc that buries "what it
is" under three sections of mechanism has failed, no matter how correct the
mechanism is. Lead with the conclusion; earn the detail.

The rule recurses: it governs the **page** (the opening band, §1) and it governs
**each detail section** (every section leads with its own one-line conclusion,
§2).

---

## The other rule — faithful to the code (except the mental model)

> **The mental model is the only licensed simplification. Everything else on the
> page MUST be literally true of the shipped code.**

A design doc describes what the code *does* — never what it *should* do, what you
*wish* it did, or what it *will* do someday. The single, deliberate exception is the
mental model (§1's "what it does", and the opening diagram that draws it): a model is
*allowed* to idealize — omit, round off, smooth over — so the reader gets a clean
picture to think with. That licence stops at the model's edge.

- **Mental model** — may simplify, and is labeled as the model, not the mechanism.
  It is a teaching abstraction, not a literal trace of the code. **It is the
  author's to define, never Claude's to invent.** If the author has not supplied a
  mental model, **leave the slot empty** — do not manufacture one. A fabricated
  model is a guess dressed up as the one licensed simplification, which breaks the
  faithfulness rule at the exact spot it is meant to hold. (Here "author" = the
  human directing the doc; an agent editing on their behalf carries the model the
  human gave, or leaves it blank for them to fill.)
- **Everything else** — detail sections, tables, the smallest X, code blocks, every
  citation, and **the diagrams** — is held to literal code-truth. If the code does
  something ugly, the doc says the ugly thing. No aspirational present tense.
- **The implementation diagram depicts the real code, not the mental model.** The
  mandatory opening SVG draws the *real* code — the actual processes, components, and
  call / data flow, labelable with `file·function` — not an idealized picture. It is
  verified against the source like any other claim: if a box or arrow can't be
  pointed at in the code, it doesn't belong in the figure.
- **Optional two-view: a mental-model diagram may accompany it.** When the clean model
  and the real implementation differ enough that the gap itself is worth seeing, pair
  the implementation diagram with a **mental-model diagram** — the author's simplified
  picture (the project's existing "two-view" pattern). Then **label both unmistakably**
  (e.g. "Mental model" vs "How the code does it"), put the model first (intuition) and
  the implementation second (reality). The mental-model diagram follows the
  mental-model rules — author-defined, may simplify, blank if the author gave none;
  the implementation diagram follows code-truth. The divergence between the two is
  itself a §4 deviation worth naming.
- **The gap between the clean model and the literal code is not hidden — it is
  exactly what §4 (the deviation section) exists to surface.** Model may simplify →
  §4 names every place that simplification departs from reality → all remaining
  prose is faithful. The three rules interlock; none licenses fiction on its own.
- **Future / proposed / planned** behaviour is not "current behaviour you haven't
  shipped yet" — mark it explicitly as not-yet-built and link the roadmap/ADR item.
  Never write a proposal in the present tense as if it already runs.
- **Verify before you write** (§3): open the code, don't paraphrase memory. A
  sentence you cannot point at in the source does not go on the page.

---

## 1. The mandatory opening band (checklist)

Before the first detail `<h2>`, every design doc MUST carry these five, in order.
This is a checklist, not a suggestion — a page missing any one is incomplete.

```
☐ tagline       H1 subtitle, one line — the nouns this page owns
☐ lede          one <p class="lede"> — what the page is for + its hard
                promise + where to go for adjacent material (tutorial / ADR)
☐ opening SVG   one <svg class="rxsvg"> at the very start — immediately after
                the lede, before §1. MANDATORY, the single most important element.
☐ §1 What it does   plain-language mental model, no jargon-first
☐ smallest X    the minimal concrete instance that makes it real
─────────────────────────────────────────────────────────────────────
   then: detail sections, freely arranged per the page's own logic
```

> **The one diagram (重中之重).** Every design doc MUST open with an
> `<svg class="rxsvg">` placed at the very start — immediately after the lede and
> before §1, never tucked inside a later section or below the detail. The two-view
> pair, when used, sits there too (model then implementation). The **north star is a
> single diagram a reader can grok
> the whole page from** — the real architecture, the actual process / component
> layout, the true call or data flow. It **depicts the code's implementation, not
> the mental model**: every box is a real component or process, every arrow a real
> call or data path, each labelable with `file·function` and verifiable against the
> source (the mental model stays in §1 prose — or, optionally, in its own clearly
> labelled companion diagram; see the two-view note under "the other rule"). Spend
> real effort here; this is the highest-leverage part of the page. If one
> self-contained diagram is genuinely
> unachievable (the subject is too multi-faceted for one picture), you still MUST
> ship **at least one diagram that aids understanding** — a partial view is
> mandatory, zero diagrams is not an option. A table may accompany the diagram but
> never substitutes for it.

**tagline** — the `<p class="tagline">` under the `<h1>`. Name the load-bearing
nouns (`BatchEvalRunner, EnvWorkerPool, per-worker routing, parallelism modes`),
not a sentence. It is the page's index entry.

**lede** — one `<p class="lede">`. Three jobs in one paragraph: (a) what the page
is for; (b) the one **hard promise** the subject makes (batch-eval's: "`worker_count`
is a *speed* dial, never a *result* dial — 1 and 8 workers score bit-identically");
(c) the off-ramps — a tutorial for how-to-use, the ADR(s) for why. The reader who
stops here still knows whether to keep reading.

**§1 What it does** — the conceptual model in plain words, mental-model-first, not
API-first. State what problem it solves and the shape of the solution before any
type name. A newcomer should finish §1 able to explain the thing to a colleague.
The idealized framing here is the **author's mental model** — see "the other rule":
carry the one the author gave, and if they gave none, leave that framing blank and
describe what the code does plainly and faithfully rather than inventing an
abstraction to fill the slot.

**smallest X** — the minimal concrete instance: the smallest graph, the smallest
`POST`, the smallest config, the smallest call. It turns the abstract model into
something runnable in the reader's head (batch-eval: "the smallest run is a single
POST — a graph name and an episode count; everything else defaults", then the
code block). **Rule:** if the subject has a natural minimal instance, you MUST
show it. A pure-concept page (e.g. `wire-types`) with no runnable instance may
substitute one minimal diagram or table — but **"I can't find a smallest example"
is a smell**: stop and ask why the subject resists a minimal instance before
writing around it.

**opening SVG** — covered by "the one diagram" callout above; it is the single most
important element of the page, so it gets its own rule rather than a line here.

---

## 2. The detail band — every section leads with its conclusion

After the opening band, sections are arranged by the page's own logic (there is no
fixed §-list — a `graph/wire-types` page and an `operations/plugin-servers` page differ
legitimately). Two rules hold across all of them:

- **BLUF per section.** Each `<h2>/<h3>` opens with a one-line conclusion, *then*
  the supporting detail. The reader skimming only the first sentence of each
  section should get the spine of the whole page. This is the inverted pyramid
  applied recursively.
- **Pick the right container for the relationship** (per the doc-section
  correlation rule): when several items share the same attributes and you're
  comparing them, use a **table**; when one item needs elaboration, use a
  **subsection**. Don't flatten a comparison into prose or inflate one concept
  into a table.

---

## 3. Ground every "how it works today" claim

A design doc describes real shipped code — every current-behaviour claim points at
the code that backs it, so future readers argue with the code, not a memory of it.

- **Cite `file·function` as the primary form** — `eval_batch.py · BatchEvalRunner.execute`,
  `registry.py · _resolve_nodeset_reimport`. Symbol references survive edits above
  them; line numbers do not.
- **Line numbers are optional precision**, added only where they genuinely help
  pinpoint (`registry.py:1221`). Treat them as perishable: any insert above shifts
  them, so a doc that pins lines everywhere signs up for recurring re-verification.
  (This standard exists partly because a line citation drifted `1084 → 1221`
  mid-session after an unrelated insert.)
- Ground claims about *current* behaviour. Don't cite code for a proposal or a
  future state — those are explicitly marked as not-yet-built (§4).

---

## 4. The deviation section — mandatory when the code deviates from the model

The opening band tells a clean mental model. Real implementations have warts: the
shipped code sometimes deviates from the model the reader was just handed. **When
it does, you MUST surface it in a dedicated section** — do not let the clean model
stand as if it were the whole truth.

- One section (e.g. "Where it deviates from the mental model"), **one
  `<div class="callout warn">` per deviation**, each naming the model claim it
  breaks and the real behaviour (see `reload-and-code-freshness.html` §4: shared
  servers drift, watchers auto-reload, targeted watcher is partial).
- A deviation that is a *known gap / TODO* says so and links the tracking item;
  a deviation that is *load-bearing-on-purpose* explains why the model is the
  simplification and the code is the reality.
- Omit the section ONLY when the implementation genuinely matches the stated model
  with no caveats. "I didn't find deviations" usually means you didn't look — the
  honest seam is the most valuable part of the page.

---

## 5. Visual language

- **Diagrams are SVG, not ASCII.** Author `<svg class="rxsvg">` themed off CSS vars
  (`var(--fg)`, `var(--muted)`, `var(--bg-alt)`, an accent var) so they track
  light/dark; carry an `aria-label` that conveys the whole picture. No box-drawing
  ASCII art, no hardcoded hex. Lift the `.rxsvg` style block + `<marker>` def from
  `core/architecture.html`. The **opening diagram (§1) is mandatory**; further
  diagrams are added only where they earn their place (two-to-three per page is
  plenty — don't pad into a gallery).
- **Code blocks are `<pre class="hl">`** with static Pygments spans — never a
  client-side highlighter, never `<div class="highlight">`. Full rules:
  [code-highlighting.md](code-highlighting.md).
- Tables/`<ol>`/`<ul>` for everything else; style is inherited.

---

## 6. Placement & mechanics (defer, don't re-derive)

- **Which group the page joins** (and whether it stays flat at design-docs root):
  the per-group membership test in [doc-site-structure.md](doc-site-structure.md)
  §4b — group by *contract owner*, one nesting level, min group size 2.
- **Page anatomy, numbered-id slugs, lede markup, last-updated, re-wrap after
  edits**: [html-authoring.md](html-authoring.md). After any edit run
  `python3 docs/_lib/_wrap_handwritten.py` (or let the dev server auto-wrap).

---

## Self-check — the membership test for the page itself

Before calling a design doc done, a reader who sees only the first screen
(tagline + lede + opening SVG + §1 + smallest X) should be able to answer:

1. **Could I understand the page from the diagram alone?** — the north-star test.
   If yes, the opening SVG has done its job; if not, is it at least aiding the model?
2. **What is this?** — in one sentence, without scrolling.
3. **How does it work, roughly?** — the shape of the mechanism, from §1.
4. **Show me the smallest real instance.** — the smallest X is on screen.
5. **What's the catch?** — the hard promise (lede) and, if any, the deviations (§4).

If any answer requires reading a detail section, the opening band has failed its
job — fix the band, not the detail. Question 1 is the one to obsess over.

Then, scanning the whole page, one more pass: **is every statement outside the
mental model literally true of the code right now?** Any "should / will / ideally"
in a present-tense description, any claim you can't point at in the source, any
proposal not marked not-yet-built — fix it or move it to §4 / a roadmap link.

---

## Canonical exemplars

Match these before authoring:

- `design-docs/graph/batch-eval.html` — the cleanest opening band
  (tagline → lede-with-hard-promise → §1 what-it-does + smallest POST → two-process path).
- `design-docs/components/` (the group) — the contract-owner shape; each page
  defines a contract you implement when authoring a component.
- `design-docs/operations/reload-and-code-freshness.html` — the deviation section
  done right (§4, one `callout warn` per drift from the mental model).
