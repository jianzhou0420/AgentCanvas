# GAPS — Ground · As-is · Problem · Solution

A fixed paradigm for **thinking a change through before you build it** — at any
scale, from a small fix to a new subsystem: before you touch code or write a plan,
you crystallize the change into one **four-part document** — **Ground → As-is →
Problem → Solution** — and refine it until it is sharp. The name is the point: the
doc exists to expose the **gap** between how things work today and what's needed,
and to propose how to close it. **Scale the depth to the change** — a small problem
is a short single-page doc; a big one is multi-facet.

> **Active.** The skill set is registered under `.claude/commands/gaps/` —
> `/gaps:ground`, `/gaps:problem`, `/gaps:solution`. The tutorial lives at
> `.claude/tutorials/skill-gaps.md`. This file is the mental model (co-located,
> also invocable as `/gaps:README` — a README.md under `.claude/commands/` registers
> as a skill, per memory `reference_command_md_only_registration`).

---

## 1. Why a paradigm (the mental model)

Changes go wrong in a predictable way, big and small: people jump to the
**Solution** ("let's use gRPC / a new node type / msgpack") before they have
grounded the **As-is** and *named* the **Problem**. The proposal then solves a
symptom, not the root, and the review re-derives context from scratch every time.

GAPS forces the order. The document is a **single living artifact** that four
skills + human discussion progressively sharpen — three that *reason* (ground /
problem / solution) and one that *shows* (result). Each skill writes its band and
stops; the human stays in the loop between bands.

```
  /gaps:ground       /gaps:problem        /gaps:solution      (discuss)      /gaps:result      Plan agent
  ──────────►        ──────────►          ──────────►         ──────────►    ──────────►       ──────────►
  Ground + As-is     Problem              Solution draft      converge       Part 3 Result     execution plan
  why it exists +    grill user → root,   one stance for one  the open       SVG before→after  (downstream;
  how it works,      then fold adjacent   root; end-state +   questions      + new flow +       NOT written by
  grounded to code   Claude-found probs   change surface + Qs                change-surface     these skills)
```

**Refinement is the point.** GAPS is never written in one pass. `/gaps:ground`
produces a grounded Ground+As-is; `/gaps:problem` does not accept a vague problem;
`/gaps:solution` opens — not closes — the design discussion. The doc is "done"
only when the human says the open questions have converged.

## 2. The artifact

One **doc-site HTML page** under `docs/pages/developer-guide/tmp/` (the docs-tmp
scratch area — same folder as the worked example). It follows
`.claude/standard/html-authoring.md` (HTML-first, page-relative links, numbered
`<h2>/<h3>` with stable ids) and `.claude/standard/code-highlighting.md` (Pygments
`<pre class="hl">`, never a client-side highlighter).

Canonical skeleton (ids are stable, kebab-case):

| Letter | Section | Written by | Status while in flight |
|--------|---------|-----------|------------------------|
| **G** | `§0 Ground` — what the feature is *for*; the forces that created it; grounded to the ADR(s) that introduced it | `/gaps:ground` | done early, rarely revised |
| **A** | `Part 1 — As-is` (`§1.1…§1.n`) — how it works *today*, grounded to `file:line` + a real call site; one sub-section per facet, with ASCII diagrams / tables | `/gaps:ground` | done early |
| **P** | `§1.last Problem` — what we are actually solving; the root, not a symptom; tied back to the As-is facets | `/gaps:problem` | the hinge — sharpened by grilling |
| **S** | `Part 2 — Solution` (`§2.1…§2.n`) — the proposal: a *top-level stance*, end-state panorama, change-surface table; **not** a per-facet patch list | `/gaps:solution` | drafted, then refined |
| **S** | `§2.last Open questions / decision points` | `/gaps:solution` + discussion | the live edge until convergence |
| *(Result)* | `Part 3 — Result` (`§3.1…§3.n`) — visual presentation of the outcome: an outcome callout (landed vs proposed + the numbers) plus SVG aids (Before → After the headline); SVG aids, it doesn't replace the substance | `/gaps:result` | optional, after Solution converges |

> Naming note: **Ground** = orientation/why-it-exists (grounded in the ADRs);
> **As-is** = the current implementation (grounded in the code). Both are
> "grounded", in different sources — keep them as two separate parts.

A `Status:` callout near the top always states: scratch / discussion draft, what As-is is
grounded to, and where it lands when finalized.

## 3. The four skills

| Skill | Role | Stops at |
|-------|------|----------|
| `/gaps:ground <target>` | Read the target's **mental model** (whichever fits: blueprint/architecture/glossary, a capability page, a design-doc, an ADR, or a nodeset doc) and its **actual code**; write `§0 Ground` + `Part 1 As-is`. Leave a Problem **stub**. Invent no problems, propose no solutions. | a grounded Ground+As-is with a stubbed problem |
| `/gaps:problem` | **Grill the user** until the problem is located (symptom → mechanism → root, tied to an As-is facet); write the `Problem`. Then **propose adjacent problems Claude found** while grounding As-is; on user OK, fold them in. | a sharp, root-level problem |
| `/gaps:solution` | From Ground / As-is / Problem, draft `Part 2 — Solution`: top-level stance + end-state + change surface + open questions. Mark as draft. Then **prompt the user to refine through discussion**, and note the finished GAPS doc goes to a Plan agent. | a Solution draft + an open discussion |
| `/gaps:result` | After the Solution converges, write `Part 3 — Result`: an outcome callout (landed vs proposed + the numbers) plus **SVG aids** — Before → After the headline, in the doc-site `.rxsvg` idiom (theme-aware). SVG aids understanding, it doesn't replace the substance. Presents the converged Solution; introduces no new design. | a visual Part 3 |

Each skill **reads the whole doc first** and only writes its own band — the doc is
the shared state between invocations. Skill 1 writes two parts (Ground + As-is);
skills 2–4 write one each. `/gaps:result` is optional — reach for it when the
outcome is worth *showing* (a presentation, a supervisor update, a landed change).

## 4. What GAPS is *not*

- **Not an ADR.** An ADR records a *decided* structural change after it lands. GAPS
  is the upstream exploration that *precedes* the decision. A finished GAPS doc
  often becomes the raw material for one ADR (see the example's §2.7 "slot #67's three
  options into place").
- **Not a plan.** GAPS says *what* and *why*, with a proposed *shape*. The **execution
  plan** (ordered steps, file edits, verification) is produced **downstream by a
  Plan agent** that consumes the finished GAPS doc. `/gaps:solution` deliberately
  stops before step-by-step sequencing.
- **Not committed docs.** It lives in docs-tmp. On finalize, the `Status` callout's
  "candidate landing" line says where the durable artifacts go (`design-docs/` +
  a domain ADR).

## 5. Worked example

`docs/pages/developer-guide/tmp/transport-signal-discussion.html` — *Server-mode and
local interaction: one process boundary, three faces*. It is mid-refinement (in the post-`/gaps:solution`
discussion stage; the Solution is not yet final), which is exactly what a live GAPS
doc looks like.

Section → skill map:

| Section in the example | Band | Produced by |
|------------------------|------|-------------|
| `§0 Background: why server mode exists` | G (Ground) | `/gaps:ground` |
| `Part 1 §1.1–1.5` (two paths / round-trip / serialization / explore-eqa workaround / star topology) | A (As-is) | `/gaps:ground` |
| `§1.6 Problem statement: what we're solving` (three faces, one root) | P (Problem) | `/gaps:problem` |
| `Part 2 §2.1–2.7` (promote to an execution station: two capabilities + msgpack foundation + location transparency + change surface) | S (Solution) | `/gaps:solution` |
| `§2.8 open for discussion / decision points` | S.Open | `/gaps:solution` + discussion |

Note how the Problem (`§1.6`) is **one root expressed as three facets**, each facet
links back to an As-is sub-section, and the Solution (`Part 2`) answers each facet
in a traceability table (`§2` intro + `§2.7`). That facet→facet traceability is the
quality bar `/gaps:problem` and `/gaps:solution` aim for.

## 6. Inherited conventions

- `.claude/standard/html-authoring.md` — page anatomy, links, numbered sections.
- `.claude/standard/code-highlighting.md` — static Pygments code blocks.
- Memory: `feedback_docsite_new_page_needs_devserver_restart` (nav auto-refreshes per
  wrap pass), `reference_command_md_only_registration` (only `.md` registers).
