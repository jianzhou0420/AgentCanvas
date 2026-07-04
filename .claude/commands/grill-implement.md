# /grill-implement — close every fixable gap between a port and its upstream

A **loop** that runs until two conditions are both satisfied:

1. **E bucket is empty** — no unexplained differences or defects remain.
2. **Every B/C/D row is genuinely irreducible** — each one has been reconsidered this
   session and confirmed stuck due to a real framework constraint, cost limit, or
   citable recorded intent.  A D row whose rationale was invented at writing time is not
   "genuinely irreducible" — it is E.

The loop terminates when there is nothing left to fix and the doc truthfully reflects that.

## Usage

```
/grill-implement <nodeset-name>
```

---

## Invariant: inventory-first, never claims-first

Walking the port's own docstring or existing doc to find differences is circular — a bad
port only claims what it got right. **Every pass MUST walk the upstream source first**,
enumerate ALL behaviour-bearing elements independently, then locate each in the port.
Starting from the port's claims is a protocol violation regardless of convenience.

---

## Bucket rules (read before every loop iteration)

| Bucket | What belongs here | What does NOT belong |
|---|---|---|
| **A** | Byte-identical text OR equivalent behaviour — even if the mechanism differs. "Verbatim but relocated" is A. An item fixed to be equivalent is A. | Anything still in B/C/D/E. |
| **B** | Mechanism **must** differ because Python monolith → typed graph (node split, attribute → state container, inline call → wire). Behaviour is **still equivalent** (🟢). If fixing it would require an architecture impossible in AgentCanvas, it stays B. | Items that were hard to implement but are now equivalent (those are A). Items that could be made equivalent but weren't (those are E or D). |
| **C** | Upstream used a model / dataset / API we genuinely cannot access at this cost tier. | Anything that is a choice, not a constraint (choices are D). |
| **D** | There is a **citable** recorded reason to differ: docstring, commit message, memory file, constraints file, code comment, or explicit user decision recorded in this session. | Rationale invented at writing time. Without a real source the row is E. |
| **E** | Audit output — differences that are neither verbatim, forced, nor covered by citable intent. Must be earned empty, never assumed empty. | — |

**Priority:** A > B > C > D > E. A row that qualifies for A is A, full stop.

**Equivalence icon** (orthogonal to bucket):
- 🟢 equivalent — output matches (most A and B rows)
- 🟡 near-equivalent — same intent, mechanism or edge case differs
- 🟠 divergent — behaviour can differ

---

## The loop

```
LOOP:
  PHASE 1 — Upstream inventory
  PHASE 2 — Port inventory
  PHASE 3 — Adjudicate & triage
  PHASE 4 — Fix E rows
  PHASE 5 — Rehabilitate misplaced B/C/D rows
  PHASE 6 — Update doc + verify
  CHECK termination → if satisfied, STOP; else LOOP
```

---

### PHASE 1 — Upstream inventory (walk upstream source cold)

Identify the upstream files for the core loop and read them in this order:
0. **The run-config chain FIRST** — the actual paper-run invocation (`run.sh` → run
   yaml → task yaml → code defaults). Enumerate every EFFECTIVE value: horizon,
   enabled abilities, sensor resolutions, obs transforms, model id, sampling params.
   Config overrides code defaults — a loop walked without its config is a different
   program. (Three-Step: abilities `[continue,stay]`, horizon 8 AND the 1024×1024
   RGB camera all lived in this layer; the camera survived three code-only passes.)
1. The rollout / eval script (main loop, step budget, termination) — **including its
   metric computation**: which measures are enabled/stripped, hand-computed vs
   official, stop-gated or not. The paper's headline number is *defined* here, not
   in the benchmark's docs. (Three-Step strips habitat's measures and hand-computes
   `success = distance ≤ 3` with no STOP gate — the same trajectories score 0.08
   official vs 0.24 upstream-ruler.)
2. The VLM call sites (prompts, parse functions, fallback paths)
3. The env interaction class (action encoding, state access)
4. Geometry / projection utilities
5. State management / graph structures
6. **The observation/pixel pipeline, per consumer** — for each model that eats
   pixels (VLM, perception models, policy nets), trace render → transforms → encode
   → final bytes: resolution, resize mode and order, crop, dtype promotions,
   normalization, image codec + quality, channel packing. (Three-Step: five separate
   defects lived in this one category — 1024 render, ResizerPerSensor(area),
   ImageNet Normalize, JPEG q75, uint8 depth packing.)

**Fork rule.** If the pinned upstream is a fork (or its README/code says "subclasses
X" / "based on X"), `git diff <parent>..<fork>` is a MANDATORY Phase-1 artifact —
every fork-side delta is an inventory item. The fork's deltas are exactly what the
port must NOT inherit from a sibling port of the parent. (Three-Step vs Open-Nav:
the diff contained the 1024 camera, the obs-transform switch, and the stop-gate
removal from the SR formula — the three most expensive misses of the campaign.)

For each file, enumerate **every** behaviour-bearing element:
- String constants and format strings (prompts, seeds, fallbacks, captions)
- Numeric constants and thresholds (grid spacing, depth scale, merge distances, step sizes)
- Sensor & preprocessing constants (resolution, resize/crop mode, normalization, image codec/quality, dtype promotions)
- The metric ruler (the success/SPL formulas actually computed — see file 1)
- Conditional branches and fallback paths (early-exits, error handlers, parse failures)
- Ordering decisions (which call precedes which)
- State accumulations (what grows across steps, what resets per episode, what resets per step)
- Dead code at eval — methods loaded but never called (mark as dead; absence in port is 🟢)

### PHASE 2 — Port inventory

For each upstream element from Phase 1:
- Where in the port? Which node, method, line?
- Is the string/constant identical, near-identical, or different?
- Is the behaviour path replicated, approximated, or absent?

Classify each as: **match** / **near-match** (note the difference) / **gap**.

**The port inventory INCLUDES every shared/env nodeset the port consumes** (env,
waypoint, perception). "Reused unchanged from <sibling port>" is a claim to verify
against THIS upstream, not an exemption. Two mandatory sub-audits on reused
components:
- **Config match** — the reused component's effective config (sensor H×W,
  transforms, preprocessing) vs this upstream's run-config chain.
- **Silent-fallback audit** — grep the reused components for try/except around
  model/checkpoint loads; a swallowed load failure must be made FATAL before any
  eval is trusted. (The shared DDPPO depth encoder had silently never loaded —
  every Open-Nav-family run to date ran on zero depth features, and no eval said so.)

### PHASE 3 — Adjudicate & triage

For every gap and near-match, answer in order:

1. **Is the behaviour equivalent despite a mechanism difference?** → A 🟢
2. **Must the mechanism differ because of the graph substrate?** → B (🟢 if equivalent, 🟡/🟠 if not)
3. **Is it forced by env/cost (no access to the upstream model/data)?** → C
4. **Is there a citable reason to differ?** → D (cite the source explicitly in the row)
5. **None of the above** → E (file with severity: HIGH/MED/LOW and a fix target)

Also re-examine every existing B/C/D row:
- **B row re-check:** Is the behaviour now equivalent? If yes → A. Can we make it equivalent? If yes → treat as E and fix it this iteration.
- **D row re-check:** Is the rationale citable? If invented → E. Is it still the right call? If not → fix it (E → A).
- **C row re-check:** Is it truly a constraint, or a choice? Choice → D (cite the decision source).

**Attribution discipline for headline-metric gaps.** Filing the *gap itself* under C
("model tier", "cost") is a HYPOTHESIS, not an adjudication — it stays marked
`hypothesis` until a control run at the paper's tier confirms it, and the doc may
not state it as the explanation before then. Statistical claims about the gap use a
computed binomial P, never eyeballing. (The "residual SR is model-tier" C-row stood
in the doc for ten days and was refuted by the paper-tier control run: 0/25,
P ≈ 3×10⁻⁵.)

### PHASE 4 — Fix E rows (code)

For each open E row, from highest severity first:

1. Read the upstream file and port implementation side by side.
2. Apply the minimal fix — only what the fidelity gap requires; no surrounding refactor.
3. If the fix changes ports or edges, update the graph JSON.
4. Syntax check: `python3 -c "import ast; ast.parse(open(path).read()); print('ok')"`
5. Verify with the strongest available oracle. For any row claiming byte-identity,
   the evidence bar is an **upstream-import harness**: import the REAL upstream
   modules (stub external clients via `sys.modules`), AST-extract-and-exec the
   un-importable methods, and byte-compare port vs upstream function-by-function on
   identical inputs. Re-typed expected strings are NOT byte evidence — three cold
   passes re-reading the same code missed what the harness caught in one run.
   Reference impl: `tmp/verify/threestepnav_upstream_equiv.py` (82 checks).
6. Mark the E row "fixed YYYY-MM-DD" in the doc; plan the row's destination bucket.

### PHASE 5 — Rehabilitate misplaced B/C/D rows

After fixing E rows, run the B/C/D re-checks from Phase 3 again on any rows touched by
the Phase 4 fixes (a fix may shift the classification of a sibling row).

Move every row to its correct bucket. When moving B → A, the row shape changes:
- B shape: `Equiv. · Upstream · Port · Why forced`
- A shape: `Equiv. · Element · Upstream anchor · How preserved`

Rewrite the row content accordingly (don't just swap it in without updating the columns).

**Emptying E means MOVING, not annotating.** "E is empty" means the §3E table has **zero
rows** — NOT "the rows left in E are all explained as non-defects". A finding that turns out
to be substrate-forced, env/cost-forced, or a deliberate-with-recorded-reason divergence must
be **relocated** to its real bucket (B / C / D); a finding that is not even a port↔upstream
delta (a shared-infra or engine limitation) moves to a §4 *Known engine gaps* note. Parking a
"documented non-defect" in E with a "not really a defect" caveat is exactly the failure this
phase prevents — it leaves §3E non-empty while the verdict claims `faithful`, so the page
contradicts itself. After this phase, §3E carries only the `Probed: …` proof, optionally plus
a one-line note of where the residuals were re-filed.

**Sweep for stale cross-bucket pointers left by a fix.** When a row moves out of E (or E→A),
other places that referenced it — §2.x prose, a sibling B/C/D row, the at-a-glance cell — may
still say "open in bucket E" / "still omitted" / "is still low". Grep the page for these and
clear every one: a fixed item that another section still calls "open in bucket E" is a stale
self-contradiction, not a live finding.

### PHASE 6 — Update doc + verify

0. **In-vivo check before the doc claims anything.** Run a small smoke and assert
   against `log.jsonl`, not just static checks: decode one image b64 and verify
   resolution + codec magic bytes (`/9j/` = JPEG); read the model id from
   `inner_log[model]`; check perception outputs non-empty; count gate/judge firings.
   Static byte-equality can be green while the runtime path is broken (NumPy ≥ 2
   raised OverflowError on an upstream-faithful expression — visible only in-vivo).
1. Apply all row moves and additions to `docs/pages/developer-guide/nodesets/method/<nodeset>.html`.
2. Update the **Fidelity** at-a-glance cell:
   - `faithful` — E empty, D rows all have cited rationale, no open B row that could be A
   - `faithful with justified deviations` — E empty, D rows present with cited rationale
   - `divergent` — open E rows present
   - `defective` — HIGH-severity E open
3. **Sweep every summary surface for consistency with §4's final numbers** — the at-a-glance
   cell is NOT just the Fidelity verdict label, and the buckets/§4 are NOT the only place a
   number lives. The trap: the faithful-config re-eval usually runs **last**, *after* the
   verdict + buckets were first drafted, so the deep sections get the new run_id/metrics while
   the headline surfaces stay frozen on the pre-eval values — and the page silently
   contradicts itself (e.g. changelog says `step_budget=8` faithful while §5 usage still shows
   `20`; verdict says "re-measuring" while §4 carries the landed number). After the eval lands,
   re-read and reconcile **all** of these against §4's run_id + metrics:
   - the **Status vs paper** at-a-glance row (headline SR / number + verified status)
   - **§4 Eval** body — the authoritative numbers + run_id (this is the source of truth)
   - **§5 Usage** — the indicative command's `step_budget` / model / abilities must match the
     faithful config the buckets settled on, not a stale earlier value
   - **§6 What this is NOT** — any "not yet reproducing — SR X vs Y" bullet
   - the **`method/index.html` row** for this nodeset
   - **every changelog line** that said "re-measuring" / "pending" — flip it to the landed result
   Then `grep` the page for the *superseded* numbers and config
   (`grep -nE "<old SR>|step_budget <old>|re-measur" <page> <index>`); **zero stale hits is the
   bar.** A changelog entry frozen mid-action is acceptable only if a later entry carries the
   landed number and the frozen one points forward to it (not a dangling "re-measuring").
4. Add a dated changelog entry (one line per fix, one line per reclassification batch).
5. Rebuild doc-site: `python3 docs/_lib/_wrap_handwritten.py`

### CHECK — termination test

The loop STOPS only when ALL of the following are true:

- [ ] **E is empty** — the §3E table has **zero rows** (a residual that proved non-defect was *moved* in PHASE 5 to its real bucket B/C/D, or to a §4 *Known engine gaps* note, never parked in E with a caveat), and the phrase "Probed: <list what was walked>" appears in the §3E section to prove it was earned. Then **grep the whole page for stale cross-bucket pointers** a now-fixed row left behind — `grep -nE "open in bucket E|still omitted|still missing|is still <code>"` — and confirm zero hits beyond the general convention sentence; a fixed item another section still calls "open in bucket E" is a self-contradiction, not a live finding.
- [ ] **Every B row** was re-examined this session and confirmed: the mechanism must differ due to the graph substrate AND making it equivalent would require an AgentCanvas architectural change that does not currently exist.
- [ ] **Every C row** was re-examined this session and confirmed: the upstream resource (model, dataset, API) is genuinely unavailable at the current cost/access tier — not just inconvenient.
- [ ] **Every D row** has a **cited source** for its rationale — a real docstring line, commit hash, memory file path, or explicit in-session decision that the user confirmed. "This seems reasonable" is not a citation.
- [ ] **The metric rulers are pinned on both sides** — the upstream's actual success/SPL computation (measures kept/stripped, stop-gated or not) and ours. Any "reproduced / not reproduced vs paper" claim states which ruler it used; if the rulers diverge, §4 dual-reports (SR* under the upstream rule alongside the official metric) and never compares numbers across rulers.
- [ ] The doc reflects the current code **and the final eval** — **no stale claim survives on any summary surface** (at-a-glance Status row · §4 · §5 usage config · §6 · `index.html` row · changelog). A `grep` of the page for the superseded numbers/config returns zero hits (per PHASE 6 step 3); the page does not contradict itself.

If any box is unchecked, do another iteration.

---

## Termination criteria in plain language

A B row is irreducible if: **the mechanism must be different AND the behaviour cannot be made equivalent without a framework change** that would take significant effort and is not currently planned. If the behaviour IS equivalent (even via a different mechanism), the row is A regardless.

A C row is irreducible if: **the upstream model or dataset is inaccessible at this cost tier and no equivalent substitute exists**. If we made a substitution choice (e.g. gpt-5-mini for Gemini), that choice plus its rationale stays in C (or D if it was intentional, not cost-forced).

A D row is irreducible if: **there is a citable recorded decision to diverge** and the user still stands by it. If the rationale is stale or wrong, fix the code.

An empty E bucket is irreducible if: **you walked the upstream core loop from cold and found nothing new**. State what you probed.

---

## Output at each iteration end

After PHASE 6, report:

```
Iteration N complete.
E: <N open / closed>. Fixed: <list>
B→A moves: <list or "none">
D re-examinations: <list or "none">
Termination check: [STOP / CONTINUE — reason]
```

---

## Reference

- Bucket rules: `.claude/commands/_data/method-nodeset-doc-spec.markdown §3`
- Page skeleton: same file §6
- Authoring standard: `.claude/standard/html-authoring.md`
- Reference pages: `method/mapgpt.html`, `method/smartway.html`
