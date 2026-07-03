# Method NodeSet Doc — write or migrate a method-nodeset page

You are writing (or migrating) a doc-site page for a **method nodeset** in the
paper-analysis + port style. The normative skeleton, diagram rules, four+one-bucket
delta discipline, verification checklist, and the copy-paste `<main>` skeleton (§6) live
in the **authoring spec** `.claude/commands/_data/method-nodeset-doc-spec.markdown`
(co-located with this skill) — this skill is the *process* that fills it. Reference
implementations: `method/mapgpt.html`, `method/smartway.html`.

**The page is an audit, not an advocacy.** Its job is to expose a bad port as readily as
it celebrates a good one. Two distinct audits are both mandatory and must not be
conflated: **doc ↔ artifact** (everything the page claims is true of the code) and
**port ↔ upstream** (the code actually implements the method). A page that passes the
first while skipping the second is a defense brief, not documentation.

## When to use

- The user asks to document a method nodeset (mapgpt, smartway, navgpt, spatialnav,
  opennav, ssg, voxposer, …) or a policy nodeset (policy_cma, policy_vla, policy_octo,
  policy_vlnce).
- The user asks to migrate one of the pre-template API-reference-style pages
  (navgpt, explore-eqa, policy-cma, policy-vla) to the template.

Not for env nodesets (`docs/.../nodesets/env/` follows the env interface template) or
common/foundation-model nodesets (no upstream paper to analyse — use judgement; the
delta buckets still often apply).

## Read first

1. `.claude/commands/_data/method-nodeset-doc-spec.markdown` — the authoring spec: principles
   (§1), section skeleton (§2), diagram rules (§2.1), bucket filing discipline incl.
   **bucket E** and the **fidelity verdict** (§3), policy-variant note (§4), verification
   checklist (§5), copy-paste `<main>` skeleton (§6).
2. `.claude/standard/html-authoring.md` — page anatomy, wrap workflow.

## Step 1 — Gather sources (parallel, before writing a word)

| Source | Where | What it gives |
|---|---|---|
| Nodeset `.py` | `workspace/nodesets/<name>.py` (or folder / `server/<name>/`) | Node inventory, ports, config defaults, verbatim constants. The module docstring usually carries upstream `file:line` anchors and the design rationale for the node split — quote it, but treat it as the **port author's claim set**, to be audited in Step 2, not as ground truth. |
| Graph JSON | `workspace/graphs/**/<graph>.json` (check `unverified/`) | **The authority for §2.1.** Dump programmatically: node list, every edge as `source.handle → target.handle`, every node config. Never draw the diagram from the nodeset source alone. |
| Upstream repo | `workspace/nodesets/_upstream/<name>/fetch_upstream.sh` → `upstream/` (gitignored); check for an existing clone first (`tmp/`, `third_party/`) | Pinned commit for the fidelity audit. Confirm `git rev-parse HEAD` == the pin recorded in the nodeset header. |
| Eval evidence | `docs/pages/research/.../method-checklist.html`, `.claude/memory/eval/`, `outputs/archive/`, `outputs/design_runs/` | run_id, archive dir, model, episode count/split, run params, paper-reported number. Design-run `iter_0` records are locked no-patch baselines. |
| Env nodeset | `workspace/nodesets/server/<env>.py` + `docs/.../nodesets/env/<env>.html` | Boundary contract (§2.4): which `step_*`/`observe_*` families, sentinel handling (STOP / `(0,0)`). |
| Run syntax | `.claude/commands/experiment/run.md` + `.claude/commands/experiment/profiles.yaml` | §5 Usage — current CLI form and a real profile name. |

Useful dump snippet (adapt paths; config may live at `node['config']`, not
`node['data']['config']` — check both):

```bash
python3 -c "
import json
g=json.load(open('workspace/graphs/vln/unverified/<graph>.json'))
print('nodes',len(g['nodes']),'edges',len(g['edges']))
for e in g['edges']: print(f\"{e['source']}.{e.get('sourceHandle')} -> {e['target']}.{e.get('targetHandle')}\")
for n in g['nodes']:
    c=(n.get('data') or {}).get('config') or n.get('config')
    if c: print(n['id'], json.dumps(c)[:300])
"
```

## Step 2 — Fidelity audit (port ↔ upstream, MANDATORY before drafting)

This is the step that decides whether the page reads "faithful port" or "defective port".
Run it **inventory-first**, never claims-first:

1. **Build the upstream behaviour inventory.** Walk the upstream core loop (rollout
   script + manager class) and enumerate every behaviour-bearing element — constants,
   thresholds, format strings, orderings, fallback paths, state transitions, gating
   conditions — *independent of what the port's docstring advertises*. Auditing only the
   port's own anchors is circular: a bad port's docstring only claims what it got right.
2. **Check call-site reality, not signatures.** What does the rollout *actually pass and
   call*? Worked examples: SmartWay's `fuse_close_node=True` default is dead code (both
   call sites pass `False`, the body never reads it — port's omission is correct);
   `make_history_v2` exists but the live call is v1 (`:509`, v2 commented `:510`).
3. **Locate each inventory element in the port.** Byte-diff every string/constant
   programmatically (rebuild the upstream string, compare against the nodeset constant
   AND any graph-config duplicate); compare semantics for logic; mark anything missing.
4. **Audit the port's extras.** Behaviour the port has that upstream doesn't (extra
   fallbacks, extra state, changed encodings) needs *recorded* intent or it is bucket E.
5. **Run the tests, don't cite them.** If an equivalence/unit test exists
   (e.g. `smartway/test_equivalence.py`), execute it and record the result on the page:
   `PYTHONPATH=agentcanvas/backend:. <agentcanvas-python> -m pytest <file> -q`.
6. **Adjudicate every difference into a bucket** (A verbatim > B substrate-forced >
   C env/cost-forced > D recorded-intent > E unexplained/defect). The hard rule:
   **D requires citable recorded intent** (docstring, code comment, commit, memory,
   constraints file). If you catch yourself composing a plausible justification that is
   written nowhere, stop — the row is E.
7. **Issue the fidelity verdict** for the at-a-glance Status: `faithful` ·
   `faithful with justified deviations` · `divergent` · `defective`, with audit date and
   scope. E-bucket findings are surfaced to the user as candidate TODOs and written into
   the page with severity — they are the audit's product, not an embarrassment to soften.

## Step 3 — Draft per the template skeleton

Title `<Method> on AgentCanvas — paper analysis + <nodeset> port` (h1 + <title> only;
also add `<meta name="nav-title" content="<Method>">` in `<head>` before
`<!-- site-layout -->` — sidebar + breadcrumb then show just the method name); then lede +
at-a-glance (incl. **Fidelity** row), §1 upstream analysis (+ invariants), §2 port
(flow / inventory / state / boundary / prompts), §3 buckets A–E, §4 eval, §5 usage,
§6 NOT, §7 sources, §8 changelog. Copy the `<main>` scaffold from spec §6. For policy
nodesets, re-aim §1 at the inference contract (spec §4). **Changelog placement (since
2026-07-02)**: the §8 changelog goes in the gitignored fragment
`docs/_changelogs/<page-rel>` (heading + entry list, no leading `<hr>`), NOT in the
page HTML — a section committed into a page is stripped from the published site by
the CI guard (`docs/_lib/_strip_changelogs.py`); the local dev site re-attaches
fragments via `nav.js`.

Drafting rules that earn their keep:

- **Neutral register.** Findings, not narrative. If the audit found defects, the lede
  says so; the verdict is not negotiable downward by prose.
- **Eval slice discipline**: every number carries run_id, model, episode count,
  split/slice. "Paper-reported: not recorded in-repo" is a valid, honest cell.
- Check applicable engine-level known gaps (e.g. roadmap TODO #64 loop-body-evaluate
  SR under-count; silent-episode-completion at high worker counts) and state them in §4.

## Step 4 — Diagrams (two inline SVGs, mirrored)

- §1 upstream flow + §2.1 port flow, **mirrored grid** (upstream fn ↔ corresponding node
  in the same position). Copy the `mgsvg` pattern from `mapgpt.html`/`smartway.html`:
  page-scoped `<style id="...">` in `<head>` **before** `<!-- site-layout -->` (the wrap
  script preserves it), one hidden `<defs>` svg with arrow markers before the *first*
  diagram (marker ids unique per page — pick fresh ids per page).
- Color code + legend: env cyan / method violet / LLM amber / loop-control rose /
  state dashed violet. Data wires = solid + arrowhead; state access = dashed, no arrowhead.
- **The loop must be visible** — rose loop-back edge (port: `iterOut → iterIn`; upstream:
  the rollout `for t in range(...)`).
- Theme-aware (`var(--fg)`, `var(--muted)`, `var(--bg-alt)`; literal accents get
  `[data-theme="dark"]` overrides), `role="img"` + one-sentence `aria-label`,
  `viewBox` + `width:100%; max-width:~880px`.
- Escape `>` as `&gt;` inside SVG text; never ASCII-art diagrams.

## Step 5 — Doc-truth verification (MANDATORY, after drafting)

Step 2 audited the code against upstream; this step audits the **page against the
artifacts** (spec §5 in full). The MapGPT draft, written carefully from the nodeset
source, still had three substantive errors only this pass caught (termination wired from
`step.terminated` not `is_stop`; evaluate triggered by `loop_nav` not `step.info`;
system prompt carried in `llmCall.config`, node not instantiated). Concretely:

1. **Graph truth** — re-dump edges/configs; check the diagram edge-by-edge; hunt
   declared-but-unwired ports and config-borne assets; read LLM profile/temp/max_tokens
   from the graph config.
2. **Upstream truth** — verify every `file:line` anchor the *page* cites with `sed`
   against the pinned commit.
3. **Eval truth** — trace numbers to run_id/archive; disclose slices; check
   stratification and engine known-gaps.
4. **Usage truth** — current `/experiment:run` form (graph-only since 2026-05-07, no
   `-- <cmd>`), profile exists in `profiles.yaml`.
5. Fix findings and add a review-pass changelog entry to the page's
   `docs/_changelogs/<page-rel>` fragment.

## Step 6 — Publish

1. Add a row to `docs/pages/developer-guide/nodesets/method/index.html`.
2. `python3 docs/_lib/_wrap_handwritten.py` — rebakes chrome/TOC, refreshes `nav.json` +
   search index.
3. Validate: XML-parse each inline SVG; assert key strings landed; confirm the page is in
   `nav.json`.
4. Report to the user: page URL (dev server :8092), the **fidelity verdict + E-bucket
   findings first**, then the doc-truth findings table, then side-findings (config drift,
   bad docstring anchors, duplicated prompt copies) as candidate TODOs — surface, don't
   silently file.

## Gotchas (earned the hard way)

- **Audit, not advocacy.** The strongest failure mode of this skill is rationale
  laundering: filling bucket D with justifications nobody recorded. If the intent isn't
  citable, the row is E — full stop.
- **Citing a test you didn't run is advocacy.** Execute it; paste the pass/fail count.
- **Dev-server auto-wrap mtime race**: while `docs/run_dev.sh` runs, every save triggers a
  re-wrap, which invalidates Edit-tool reads. For multi-edit batches, prefer one atomic
  `python3` replace script with `assert s.count(old) == 1` per edit.
- **Nodeset ≠ graph**: a node shipped by the nodeset may not be instantiated by the
  reference graph (and its payload may ride another node's config). Say so in §2.2.
- **Wrap preserves head extras** between `<meta>` block and `<!-- site-layout -->` — put
  the diagram `<style>` and the `nav-title` meta there, nowhere else (the wrap carries
  forward only `<style>` blocks + the nav-title meta; any other custom head tag is dropped).
- **The dev server caches `_lib` modules.** If you change `_nav.py`/`_wrap_handwritten.py`,
  the running `docs/run_dev.sh` keeps wrapping with the OLD code (it imports once at
  startup) and will silently undo your page edits — restart it before re-tagging pages.
- **Don't trust prior prose** (memories, old docs, even the nodeset docstring) for graph
  wiring, model params, upstream anchors, or CLI syntax — they drift; the artifacts are
  the authority. (SmartWay's docstring cited `base_il_trainer.py:55–57` for
  `last_distance`; the real source is `one_stage_prompt_manager.py:54–56`.)
- The graph may live under `workspace/graphs/<task>/unverified/` — that placement is
  itself a §4 status fact (gym-migration re-eval wave, TODO #64).
