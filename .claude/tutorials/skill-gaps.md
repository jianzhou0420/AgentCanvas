# Skill: GAPS — Ground · As-is · Problem · Solution

## When to Use

You are about to make a change — at **any scale**, from a small fix to a new
subsystem — and you want to *think it through and converge with the user* before
any code or plan. GAPS is the front-end of that: it produces a sharp four-part
document whose whole job is to expose the **gap** between As-is and what's needed.
For a big change it then feeds a Plan agent; for a small one the Solution may go
straight to implementation.

**Scale the depth to the change**, don't gate on it: a small problem is a short
single-page doc (a few lines of Ground, a one-facet As-is, a tight Problem, a
direct Solution); a big one is multi-facet with traceability tables. The only thing
that doesn't need GAPS is a **truly mechanical edit** (a typo, a rename) where there
is no problem to frame — that goes straight to a commit.

## The shape (four-part essay)

```
  §0 Ground    — what the feature is FOR; the forces that made it (ADR-grounded)
  Part 1 As-is — how it works TODAY, grounded to file:line (one sub-section/facet)
  §1.last Problem — what we're actually solving: the ROOT, tied back to As-is facets
  Part 2 Solution — the proposal: one stance for one root; end-state + change surface
  §2.last Open Qs — the live discussion edge until it converges
  Part 3 Result — (optional) show the outcome: an outcome callout + SVG aids (before→after)
```

`Ground` = why it exists (grounded in ADRs); `As-is` = how it works (grounded in
code) — two separate parts, two sources. The document is **one HTML page** in
`docs/pages/developer-guide/tmp/` (docs-tmp), authored per
`.claude/standard/html-authoring.md` and `.claude/standard/code-highlighting.md`.
It is a **living artifact** — four skills (three that reason, one that shows) plus
human discussion sharpen it; it is never one-shot.

## Driving a GAPS doc from zero to plan-ready

| Step | You run | What happens | You end with |
|------|---------|--------------|--------------|
| 1 | `/gaps:ground <target>` | Claude reads the target's mental model (whichever fits: blueprint/architecture/glossary, a capability page, a design-doc, an ADR, or a nodeset doc) and its real code, writes `§0 Ground` + `Part 1 As-is`, stubs the Problem | grounded Ground + As-is |
| 2 | `/gaps:problem` | Claude **grills you** to the root problem, writes the `Problem`, then proposes adjacent same-root problems for you to include or defer | a sharp Problem |
| 3 | `/gaps:solution` | Claude drafts `Part 2 Solution` (one stance, traceability table, end-state + change surface, open questions) and opens the refinement discussion | a Solution draft + open Qs |
| 4 | *(discuss)* | You and Claude iterate Part 2 in place — answer the open questions, push on the stance | a converged GAPS doc |
| 5 | `/gaps:result` *(optional)* | Claude writes `Part 3 Result` — an outcome callout (landed vs proposed + the numbers) plus SVG aids (before→after the headline, theme-aware `.rxsvg`); SVG aids, doesn't replace the substance | a visual Part 3 to show / share |
| 6 | *(hand off)* | The finished GAPS doc goes to a **Plan agent** to design the execution plan | an execution plan (downstream) |

Each skill reads the whole doc and writes only its own band — the doc is the shared
state. Skill 1 writes two parts (Ground + As-is); skills 2–4 write one each.
`/gaps:result` is optional — reach for it when the outcome is worth *showing*. Stay
in the loop between steps; the human gate between bands is intentional.

## Worked example (mid-refinement)

`docs/pages/developer-guide/tmp/transport-signal-discussion.html` — *Server-mode 与
local 的交互:一道进程边界,三个面*. It is currently in **step 4** (post-`/gaps:solution`
discussion; the Solution is not yet final), which is what a live GAPS doc looks like
in practice.

Read it alongside this table to see each band:

| Section | Band | Skill | What to notice |
|---------|------|-------|----------------|
| `§0 背景:为什么有 server mode` | Ground | ground | purpose split into "live forcing function" (env isolation) vs "reserved capability" (distributed); derives the two physical constraints the Solution must respect |
| `§1.1–1.5` | As-is | ground | every claim grounded — `proxy.py:60-64`, `server_app.py:183-185`, the explore-eqa `_TSDF_PLANNERS` folk workaround; implicit assumptions (co-location, single writer) surfaced |
| `§1.6 Problem statement` | Problem | problem | **one root, three facets** (传输/拓扑/状态), each facet linked to its `§1.x`; positive goal stated (正确/高效/位置透明) |
| `Part 2 §2.1–2.7` | Solution | solution | **one stance for one root** ("升格为执行站", two capabilities), constraints A/B restated, traceability table (§2 intro), end-state + change-surface table (§2.7) |
| `§2.8 待讨论 / 决策点` | Solution.Open | solution + discuss | honest open questions, `← 先拍这个` marker, struck-through decided ones (~~msgpack vs Arrow~~) |

The quality bar to imitate: **facet→facet traceability**. The Problem names N facets
of one root; the Solution answers all N in a table; the change surface maps each part
of the Solution to a file. If your GAPS doc can't draw those lines, a band isn't sharp
enough yet.

## What GAPS is not

- **Not an ADR** — an ADR records a *decided* change after it lands; GAPS *precedes*
  the decision and often becomes one ADR's raw material.
- **Not a plan** — `/gaps:solution` stops at *scope* (change-surface table), not
  *sequence*. Ordered steps / edits / verification are the **Plan agent's** job.
- **Not durable docs** — it lives in docs-tmp. On finalize, the `Status` callout's
  "candidate landing" line says where it goes (`design-docs/` + a domain ADR).

## Conventions inherited

- `.claude/standard/html-authoring.md` — page anatomy, page-relative links, numbered
  `<h2>/<h3>` with stable kebab ids; re-wrap with `python3 docs/_lib/_wrap_handwritten.py`.
- `.claude/standard/code-highlighting.md` — static Pygments `<pre class="hl">`.
