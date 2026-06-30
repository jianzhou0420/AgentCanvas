# Method NodeSet Doc — authoring spec (paper-analysis + port style)

> Normative reference for the `/docs:method-nodeset-doc` skill. The skill is the
> *process* (gather → audit → draft → diagrams → verify → publish); this file is the
> *spec* it fills, and §6 is the copy-paste `<main>` skeleton to clone into a new page.
> Read §1–§5 to know what each section must contain.

A method nodeset is a **port** of a published method onto the canvas, so its page must
answer three questions an API reference cannot: ① how the upstream method actually works,
② what our port keeps, what it changes, and *why*, ③ whether the numbers reproduce. The
style is borrowed from the AAS reference pages (`docs/pages/aas/reference/` — adas / aflow
/ myloop): upstream analysis with source-line anchors, then the port, then a five-bucket
delta table.

**Scope.** This spec standardises *documentation*. The env-side counterpart,
`docs/pages/developer-guide/nodesets/env/template.html`, standardises *code* (the env
interface contract). This spec does **not** apply to env nodesets, nor to
common/foundation-model nodesets (no upstream paper to analyse — use judgement; the delta
buckets still often apply). First page written against it:
`docs/pages/developer-guide/nodesets/method/mapgpt.html`. Pages that predate it (navgpt,
explore-eqa, policy-cma, policy-vla) should be migrated when next touched.

---

## 1. Principles

- **Double duty.** One page serves two readers: someone *developing or auditing the port*
  (needs anchors, deltas, evidence) and someone *new to the method* (needs the logic-level
  §1 without reading upstream code).
- **Foreground our port, don't recap the paper.** §1 explains the upstream mechanism only
  as far as needed to make §2–§3 checkable. The page's centre of gravity is the delta
  table — what is verbatim, what is forced, what is chosen.
- **Every claim has an anchor.** Upstream behaviour cites `file.py:lines` in the pinned
  upstream commit; our numbers cite a `run_id` (and disclose the episode slice); status
  claims cite the graph path (`verified/` vs `unverified/`).
- **Audit, not advocacy.** The page documents the port *as it is* — its job is to expose a
  bad port, not to defend one. A bucket-D rationale must cite *recorded* intent (docstring,
  code comment, commit message, memory, constraints file); a plausible rationale invented
  at writing time is laundering, not documentation. A difference with no recorded
  justification files under **bucket E**, and a defective port gets a defective verdict.
- **Boundary compliance is part of the doc.** A method nodeset is reasoning-only: it
  consumes env nodes (`step_*` / `observe_*` families, see the env contract) and
  foundation-model nodes over wires. §2.4 states this explicitly — it is the per-page
  enforcement of the method-vs-foundation-model boundary principle (roadmap TODO #56).

## 2. The section skeleton

Title pattern: `<Method> on AgentCanvas — paper analysis + <nodeset> port` — for the `<h1>`
and `<title>` only. Add `<meta name="nav-title" content="<Method>">` in `<head>` (before
`<!-- site-layout -->`) so the sidebar and breadcrumb show just the method name (no
"NodeSet", no "on AgentCanvas …"). Then:

| § | Section | Required? | What goes in |
|---|---|---|---|
| — | **Lede + at-a-glance** | required | One-paragraph lede (what method, what benchmark, what the port is). An at-a-glance table: upstream paper · pinned commit · env spaces consumed · FM nodes consumed · node count · graph(s) · verified status + headline number vs paper · **fidelity verdict** (from the §3 audit, with date). |
| 1 | **Upstream method analysis** | required | Logic-level walkthrough of the upstream core loop, with `file.py:lines` anchors into the pinned commit, **plus an upstream-flow SVG** (same drawing rules as §2.1; mirror the port diagram's layout so the two compare box-by-box). Close with *"Upstream invariants worth marking"* — the load-bearing pieces the port must preserve. For neural-policy nodesets see §4 variant note. |
| 2 | **The AgentCanvas port** | required | 2.1 per-step flow — an **inline SVG diagram, not ASCII art** (see §2.1 rule); 2.2 node inventory table with ports as `handle:TYPE` and each node's role in the loop; 2.3 state & memory (graph_state entries / containers: reducer, value type, lifetime, writer); 2.4 boundary contract (which env `step_*`/`observe_*` nodes, which FM nodes, compliance statement); 2.5 prompt assets (where they live, what is verbatim). |
| 3 | **Delta — five buckets** | required | The signature section; see §3. Every row also carries an orthogonal **equivalence icon** (🟢/🟡/🟠). |
| 4 | **Evaluation** | required | Paper-reported number(s) · our number(s) with `run_id`, date, model, and episode-slice disclosure · current verification status of the graph(s) · known gaps. Never quote a partial-split number without naming the slice. |
| 5 | **Usage** | required, short | Load command, graph file path(s), an indicative `/experiment:run` line. Keep this thin — wiring detail lives in the graph JSON, not prose. |
| 6 | **What this nodeset is NOT** | optional | Boundary declarations that pre-empt confusion: sibling variants it is distinct from, env logic it deliberately does not contain, model lock-ins it deliberately avoids. |
| 7 | **Source files + changelog** | required | Table: nodeset `.py` · graph JSON(s) · upstream location (`third_party/` submodule or `workspace/nodesets/_upstream/<name>/` fetch script) · prompt/data sidecars. Then a dated changelog list. |

### 2.1 Diagram rule — inline SVG, not ASCII

Both required diagrams — the upstream flow in §1 and the port's per-step flow in §2.1 —
and any other diagram on the page must be **inline SVG**, following the convention
established by the AAS reference pages and `design-docs/graph/graph-executor.html`. Draw
the two with **mirrored layouts** (upstream function ↔ corresponding canvas node in the
same grid position) so they compare box-by-box. Rules:

- **Theme-aware**: text and neutral strokes use CSS variables (`var(--fg)`, `var(--muted)`,
  `var(--bg-alt)`); literal accent colors get `[data-theme="dark"]` overrides. Styles live
  in a page-scoped `<style id="...">` block in `<head>` (before the `<!-- site-layout -->`
  marker — the wrap script preserves it).
- **Color-coded by node family**, with a legend row: env nodes (cyan), method nodes
  (violet), LLM calls (amber), loop control / guards (rose), state containers (dashed
  violet — matching the canvas's dashed-violet state convention). Data wires are solid
  arrows (`marker-end`); state access is dashed with no arrowhead.
- **The loop must be visible.** These methods are iterative — draw the loop-back edge (port
  side: `iterOut → iterIn`; upstream side: the rollout `for t in range(...)`) in the
  loop-control accent color, so the figure reads as a cycle, not a one-shot pipeline.
- **Accessible**: `role="img"` + an `aria-label` that narrates the flow in one sentence.
- **Responsive**: `viewBox` + `width:100%; height:auto; max-width:~880px` — no fixed pixel
  dimensions.

Reference implementation: MapGPT §2.1 (`mapgpt.html#21-flow`, class `mgsvg`, marker defs in
a hidden `<svg>`). Pick fresh, page-unique marker ids per page.

## 3. The five-bucket delta table

This is what makes the page worth reading. Every difference between upstream and the port
is filed into exactly one bucket, as a table row with an upstream anchor and a one-line
reason. Buckets:

| Bucket | Meaning | Row shape |
|---|---|---|
| **A. Preserved verbatim** | Survives the port unchanged — byte-for-byte text (prompts, constants, format strings) or unchanged semantics (algorithms, fallback tricks). | element · upstream anchor · how preserved |
| **B. Forced by substrate** | Changed because the artifact is now a typed graph + nodeset, not a Python script: monolith split into nodes, object attributes → state containers, in-process calls → wires, offline preprocessing → online nodes. | upstream · port · why forced |
| **C. Forced by environment / cost** | Changed because our env, data, or budget differs: model availability, eval cost, batch limits, dataset slices. | upstream · port · cost/env reason |
| **D. Intentional divergences** | Judgement calls — not forced by anything. Each row carries a *recorded* rationale (cite where it was recorded); this is where reviewers should push back. | upstream · port · rationale (with source) |
| **E. Unexplained / defects** | Audit output — differences that are neither verbatim, forced, nor covered by recorded intent: missed behaviours, silent drift, candidate bugs. An empty E bucket must be *earned by the fidelity audit* (state what was probed), never assumed. | element · upstream anchor · observed in port · severity · tracking (TODO / issue) |

**Filing discipline:** if a row could go in two buckets, pick the *earliest* applicable
bucket (A > B > C > D > E) — "verbatim but relocated" is A with a note, not D. **D requires
recorded evidence of intent; without it the row falls to E.** An empty bucket is itself
information: say "none" (and for E, say what the audit probed). The audit's bottom line is
a **fidelity verdict** in the at-a-glance Status: `faithful` ·
`faithful with justified deviations` · `divergent` · `defective` — with the audit date and
scope (claim-driven vs inventory-first).

### 3.1 The equivalence icon (orthogonal axis)

The bucket letter says *why* a row differs; it does **not** say *whether the behaviour
actually changed*. Those are independent axes: a substrate-forced split (B) normally
preserves behaviour exactly, while a cost-forced model swap (C) can change it outright. So
every bucket-table row **leads with an `Equiv.` column** carrying one of three marks (wrap
each in a `<span title="…">` so the tooltip names it):

| Mark | Name | Meaning |
|---|---|---|
| 🟢 | equivalent | identical or semantically identical behaviour (output matches up to float noise) |
| 🟡 | near-equivalent | same intent; a mechanism, library, or edge case differs — output usually matches |
| 🟠 | divergent | behaviour can differ — by design or, rarely, as a defect |

- **Orthogonal to the letter.** An A or B row is almost always 🟢; a C row may be 🟢
  (e.g. batch=1 — behaviour unchanged) or 🟠 (e.g. `gpt-4` → `gpt-5-mini`); a D row is
  typically 🟡/🟠. The mark is the *leading* column of every bucket table, before the
  columns named in the *Row shape* column above.
- **Never ❌ or 🔴.** A divergence is a design fact, not an error — a failure cross would
  mis-read an intentional or substrate-forced choice as a bug. The "this is wrong" signal
  is carried by **bucket E** (a genuine defect is a 🟠 row in E with a severity and a TODO),
  never by the icon.

Reference implementation: `method/discussnav.html` §3 — the legend paragraph directly under
the §3 heading, plus the `Equiv.` first column on all five bucket tables.

## 4. Variant note — neural-policy nodesets

Policy wrappers (`policy_cma`, `policy_vla`, `policy_octo`, `policy_vlnce`) have no
reasoning loop to analyse. The skeleton still applies with §1 re-aimed:

- **§1 becomes the inference-contract analysis**: input tensors and preprocessing,
  normalisation statistics, action space and postprocessing, checkpoint provenance, and the
  upstream eval harness it was validated in.
- **§3 is where the value is.** Conversion deltas (axis conventions, gripper-bit
  interpretation, normalisation constants, vendored-vs-external import paths) are exactly
  the class of bug that has cost real eval runs — file every one, even "obvious" ones.
- **§2.3 state** usually degenerates to "stateless forward pass" — say so explicitly.

## 5. Verification checklist — the review pass is part of writing

Drafting from the nodeset source alone is **not enough**: the first MapGPT draft, written
from `mapgpt.py` and its docstring, contained three substantive wiring errors that only
checking the artifacts caught. Before a page is done, run all six checks:

1. **Graph truth.** Dump the graph JSON programmatically (nodes, every edge as
   `source.handle → target.handle`, node configs) and verify the §2.1 diagram edge-by-edge.
   Watch for: termination's real inputs, evaluate's real trigger, declared-but-unwired
   ports, and *config-borne assets* — a node can exist in the nodeset yet the graph may
   carry its payload as a config field instead (MapGPT: the system prompt rides
   `llmCall.config`, the `system_prompt` node is not instantiated). Read LLM profile /
   temperature / max_tokens from the graph config, never from memory.
2. **Upstream truth.** Locate the pinned commit (fetch script → `upstream/`, or an existing
   clone) and confirm `git rev-parse HEAD` matches the pin. Verify every `file.py:lines`
   anchor with `sed`. For every "verbatim" claim, do a *programmatic byte-diff* — rebuild
   the upstream string and compare against the nodeset constant **and** any graph-config
   duplicate.
3. **Fidelity audit (port ↔ upstream, inventory-first).** The anchor checks above start
   from the *port's own claims* — circular, and a bad port's docstring only claims what it
   got right. Walk the *upstream* core loop and enumerate every behaviour-bearing element
   (constants, thresholds, format strings, orderings, fallback paths, state transitions)
   independently, then locate each in the port. Check *call-site reality*, not signatures —
   SmartWay's `fuse_close_node=True` default is dead code (rollout passes `False`; body
   never reads it). Audit the port's *extras* too: behaviour upstream doesn't have needs
   recorded intent or it is bucket E. Run the equivalence/unit tests if they exist (e.g.
   `test_equivalence.py`) — citing a test you didn't run is advocacy. Output: every
   difference adjudicated into A–E + the fidelity verdict.
4. **Eval truth.** Trace each number to a `run_id` + archive dir; disclose model, episode
   count, split/slice, and key run params. Apply slice discipline (e.g. R2R val_unseen
   stratification). Check whether engine-level known gaps apply (e.g. roadmap TODO #64
   loop-body-evaluate SR under-count) and say so in §4.
5. **Usage truth.** Check command syntax against the *current* skill doc (e.g.
   `/experiment:run` is graph-only since 2026-05-07 — no `-- <cmd>`) and profile names
   against `.claude/commands/experiment/profiles.yaml`.
6. **Publish mechanics.** Register the page in `method/index.html`; run
   `python3 docs/_lib/_wrap_handwritten.py`; XML-parse every inline SVG; assert the key
   claims/strings landed in the final file. Record the review pass as a changelog entry.

## 6. Copy-paste scaffold

The `<main>` body for a new method-nodeset page. Copy it into
`docs/pages/developer-guide/nodesets/method/<nodeset>.html`; chrome (sidebar / breadcrumb /
TOC) is re-baked by `docs/_lib/_wrap_handwritten.py`, so you author only `<main>` plus two
head extras — the `<meta name="nav-title">` (§2) and the diagrams' page-scoped
`<style id="...">`, both placed before `<!-- site-layout -->` (see
`.claude/standard/html-authoring.md`). The §1 heading carries the upstream citation
`(Author et al. Year)`, matching every shipped page (mapgpt / smartway / navgpt /
explore-eqa / voxposer). Comments mark what each section must hold; the full contract is in
§2 (skeleton), §3 (delta buckets), §4 (policy-nodeset variant).

```html
<header class="page-header guide">
  <h1><Method> on AgentCanvas — paper analysis + <nodeset> port</h1>
</header>

<p class="lede">One paragraph: upstream method + venue/year, benchmark, what the port is.</p>

<table><!-- at-a-glance: paper · pinned commit · env spaces consumed · FM nodes consumed
     · #nodes · graph(s) · verified status + headline number vs paper · FIDELITY VERDICT
     (faithful | faithful with justified deviations | divergent | defective, + audit date) --></table>

<h2 id="1-upstream"><span class="num">1.</span> Upstream method analysis (Author et al. Year)</h2>
<!-- logic-level core loop, file.py:line anchors into the pinned commit
     + upstream-flow inline SVG (mirror §2.1's layout, box-for-box).
     Policy nodesets: re-aim §1 at the inference contract (§4). -->
<h3 id="11-invariants"><span class="num">1.1</span> Upstream invariants worth marking</h3>

<h2 id="2-port"><span class="num">2.</span> The AgentCanvas port</h2>
<h3 id="21-flow"><span class="num">2.1</span> Per-step flow</h3>
<!-- inline SVG diagram (theme-aware, legend, visible loop-back edge); never ASCII art. §2.1 -->
<h3 id="22-nodes"><span class="num">2.2</span> Node inventory</h3>
<!-- table: node · ports as handle:TYPE · role in the loop -->
<h3 id="23-state"><span class="num">2.3</span> State &amp; memory</h3>
<!-- graph_state entries / containers: reducer · value type · lifetime · writer.
     Policy nodesets usually: "stateless forward pass" — say so. -->
<h3 id="24-boundary"><span class="num">2.4</span> Boundary contract</h3>
<!-- which env step_*/observe_* nodes, which FM nodes; reasoning-only compliance statement -->
<h3 id="25-prompts"><span class="num">2.5</span> Prompt assets</h3>
<!-- where prompts live, what is verbatim -->

<h2 id="3-delta"><span class="num">3.</span> Delta vs upstream — five buckets</h2>
<p>Each row carries an <strong>equivalence icon</strong> (orthogonal to the bucket letter — the letter says <em>why</em> it differs, the mark says whether <em>behaviour</em> changed): <span title="equivalent">🟢</span> equivalent (identical or semantically identical behaviour) · <span title="near-equivalent">🟡</span> near-equivalent (same intent; a mechanism, library, or edge case differs — output usually matches) · <span title="divergent">🟠</span> divergent (behaviour can differ — by design or, rarely, a defect). Never use ❌/🔴 — a divergence is a design fact, not an error; a genuine defect is a 🟠 row in bucket E with a severity.</p>
<h3 id="3a-verbatim">A. Preserved verbatim</h3>
<h3 id="3b-substrate">B. Forced by substrate</h3>
<h3 id="3c-cost">C. Forced by environment / cost</h3>
<h3 id="3d-intentional">D. Intentional divergences</h3>
<h3 id="3e-defects">E. Unexplained / defects</h3>
<!-- every bucket table LEADS with an `Equiv.` <th> (🟢/🟡/🟠, §3.1); file EVERY difference
     into exactly one bucket; A>B>C>D>E precedence. D requires CITED recorded intent or it
     falls to E. Empty buckets say "none" (E says what the audit probed). §3. -->

<h2 id="4-eval"><span class="num">4.</span> Evaluation</h2>
<!-- paper number · our number(s) with run_id + date + model + episode slice · graph status · known gaps -->
<h2 id="5-usage"><span class="num">5.</span> Usage</h2>
<!-- load command · graph path(s) · indicative /experiment:run line (graph-only) -->
<h2 id="6-not"><span class="num">6.</span> What this nodeset is NOT</h2>
<h2 id="7-sources"><span class="num">7.</span> Source files</h2>
<!-- nodeset .py · graph JSON(s) · upstream location · prompt/data sidecars -->
<h2 id="8-changelog"><span class="num">8.</span> Changelog</h2>
```

## Changelog

- **2026-06-11**: Initial template, distilled from the AAS reference pages (adas / aflow /
  myloop) + the env-template precedent. First instantiation: MapGPT.
- **2026-06-11**: Fidelity-audit upgrade after user review ("would this flow defend a bad
  port?" — yes, it would have): bucket E (unexplained/defects), fidelity verdict in
  at-a-glance, "audit, not advocacy" principle (bucket-D rationales must cite recorded
  intent), §5 gains the inventory-first fidelity-audit check.
- **2026-06-11**: Added §5 Verification checklist (graph / upstream / eval / usage / publish
  truth) from the MapGPT review-pass learnings; process captured as the
  `/docs:method-nodeset-doc` skill.
- **2026-06-13**: Split the former `method/template.html` into this normative **spec**
  (co-located with the skill) and a pure copy-paste scaffold (kept at the original
  `template.html` path). The skill now defers to this file for rules; the template page is
  scaffold-only. No normative content changed in the split.
- **2026-06-13**: Removed the doc-site scaffold page; absorbed its one unique artifact — the
  literal `<main>` copy-paste skeleton — back into §6 here (everything else on that page
  duplicated §1–§5). Baked the `(Author et al. Year)` upstream citation into the §1 heading
  to match the shipped pages. The spec is now the single source for both rules and skeleton.
- **2026-06-16**: Added the **equivalence-icon** convention as §3.1 (and into the §6
  scaffold): a leading `Equiv.` column — 🟢 equivalent / 🟡 near-equivalent / 🟠 divergent —
  *orthogonal* to the A–E bucket letter (the letter says *why* a row differs, the mark says
  *whether behaviour changed*). Explicit "never ❌/🔴" rule (a divergence is a design fact;
  defects ride bucket E). Also corrected the stale "four buckets" naming to "five" (bucket E
  has existed since 2026-06-11). First implementation: `method/discussnav.html` §3.
