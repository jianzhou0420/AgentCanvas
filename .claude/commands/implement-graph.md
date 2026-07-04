# /implement-graph — Port a paper + repo into an AgentCanvas graph

Port a published agent-method (paper + upstream repo) into AgentCanvas as a runnable graph + nodeset. **Monolith first, decompose later.** The validated monolith is the I/O contract — splitting afterwards becomes a mechanical refactor whose test is end-to-end metric parity.

## Usage

```
/implement-graph <paper-ref> <upstream-repo-path> [env=<env-nodeset>]
```

- `<paper-ref>` — arXiv ID, paper URL, or local PDF path.
- `<upstream-repo-path>` — local checkout under `third_party/zz_just_for_refer/<method>/` (clone there first if missing).
- `env=<env-nodeset>` — target env nodeset name (`env_mp3d` / `env_vlnce` / `simpler` / …); infer from paper if omitted.

The upstream `agent.py` (or equivalent loop driver) **plus the effective run-config chain** (`run.sh` → run yaml → task yaml → code defaults) is the source of truth: paper text describes intent, code + config describe behaviour, and behaviour is what you reproduce. Config overrides code defaults — a port that read only the Python has not read the program. Read the paper for context; reproduce against the code.

## Where things live

```
workspace/nodesets/<method>.py                method-side reasoning nodes
workspace/graphs/<method>_<env>.json          graph topology
workspace/architect/exp_profiles/<method>_<env>.yaml  eval profile (smoke / perf / split / step_budget)
.claude/commands/experiment/profiles.yaml              admission-control profile (vram, exclusive_gpu)
```

**Import boundary (hard).** Method nodesets must never import `habitat`, `habitat_sim`, `MatterSim`, `vlnce_baselines`, or any other env runtime — those live in env nodesets, accessed across the server-mode boundary. See `agentcanvas/backend/app/test_import_boundary.py`.

## Component-creation rules

These are mandatory; read each *before* touching the corresponding file type:

- `.claude/standard/component-creation.md` — universal node rules.
- `.claude/tutorials/skill-canvas-node.md` — adding a `BaseCanvasNode` subclass.
- `.claude/tutorials/skill-nodeset.md` — adding a `BaseNodeSet` subclass.
- `.claude/tutorials/skill-graph-json.md` — editing graph JSON.
- `.claude/tutorials/skill-env-nodeset.md` — only if you're also wrapping a new simulator (rare).

## Phase 1 — Monolithic prototype (gated)

**Goal**: one runnable graph end-to-end, with the right I/O at the graph and nodeset boundaries. Compactness > granularity at this phase.

### 1. Read the upstream code

Before writing anything, answer five questions from the upstream `agent.py` + its run config:

1. **Loop shape.** What's one iteration? Once per env step (NavGPT)? Once per viewpoint visit with multiple LLM calls inside (MapGPT)? Multiple passes with fan-out + aggregation (DiscussNav)?
2. **Cross-iter state.** What survives across iterations? A scratchpad string? An action list? A topo-map dict? An accumulated dialogue?
3. **Action-space bridge.** What does the env actually consume — viewpoint ID? Discrete index? Continuous waypoint? How does the LLM's text get translated?
4. **Episode-fixed values.** What's computed once at episode start (instruction, landmarks, scene metadata) and read by every iteration?
5. **Observation pipeline.** What exact bytes does each model consume? Sensor resolution and obs transforms from the *run config* (not the sim's defaults), resize mode/order, normalization, image codec + quality. If the upstream is a fork, run `git diff parent..fork` first — sensor/transform changes hide there. (Three-Step forked Open-Nav to render 1024×1024; the port inherited a sibling's 224 default and lost weeks to it.)

These decide what nodes you need, where state containers go, and what `Initialize` carries vs what `iterIn` re-emits.

**Reuse gate for `env=`.** Before binding an existing env nodeset, diff its effective sensor/obs config against THIS upstream's task yaml. On mismatch, set the per-graph knobs (e.g. `env_habitat__reset.rgb_resolution`) or extend the env nodeset — never silently inherit a sibling port's perception defaults ("correct for Open-Nav, wrong for Three-Step").

### 2. Draft the nodeset monolithically

**"Monolithic" means collapsed to the framework's floor, NOT one super-node.** Three hard boundaries forbid further collapsing the method side, so even Phase 1 has multiple method nodes:

| Boundary | Why it can't be absorbed into a method node |
|---|---|
| **LLM call must be built-in `llmCall`** | Owns OpenAI client, retries, token accounting, multimodal assembly (incl. `image_labels` interleaving), profile/cache switching. Inlining an OpenAI call into a custom node forfeits `/experiment` token stats and profile control. |
| **Env step must be `env_<sim>__*`** | Env nodesets run server-mode (replicated) via proxy. Method nodesets are forbidden from importing `habitat` / `MatterSim` / `vlnce_baselines` (see import boundary). |
| **Each LLM call splits the iter** | Work before the LLM call (prompt assembly) and after (parse + decide action) are different node bodies — they sandwich the `llmCall`, can't merge across it. |

So the floor for a single-LLM-call loop is **~3 method nodes**: one prompt-builder before the LLM call, one parser/decider after the LLM call and before env step, one post-step bookkeeping node (writes history / state container). N LLM calls per iter ⇒ ~(2N+1) method nodes. SmartWay-CE (1 LLM call/iter) → `plan_step` / `decide_action` / `update_history`. NavGPT-MP3D (1 call) → same shape. DiscussNav (multi-LLM debate) → more.

**Mirror the upstream `agent.py`'s top-level methods 1:1 within that floor.** If upstream is `plan_step()` + `parse_action()` + `update_history()`, that's the 3 method nodes — pack each method's full body into one node. Don't pre-decompose *within* a method along architectural seams you haven't validated yet — premature sub-decomposition introduces cross-node coupling (state-read races, lost env-boundary derivations) that a working monolith already proved you don't need.

**Side-benefit of being at the floor, not below it.** Each wire becomes an inspectable contract at graph level. Two of the SmartWay-CE porting bugs surfaced this way: the Place-ID bug was a one-character diff inside `plan_step` (200 LOC, scannable); the `image_labels` bug was a *missing edge* between `plan_step` and `planner_llm` — a class of bug that doesn't exist if the method is one super-node, because the contract is hidden inside Python. ([[project_smartway2_place_id_bug]], [[project_smartway2_image_labels_bug]])

**Verbatim prompts as module-level string constants** with a comment citing `<upstream-file>:<line-range>`. Don't paraphrase. Prompt text *is* the experiment; reword it and you've changed it.

**Cross-iter state goes in a `StateContainer`** with `access_grants` to readers/writers. Edges carry single-iter values. Smuggling state through `iterOut → iterIn` passthrough is the road to the dual-wire freeze (see Landmines).

**Method nodeset = pure reasoning.** No env-runtime imports. If a method-side value needs to survive across env calls, that value belongs in a `lifetime="episode"` state container or on `Initialize.config.ports`.

### 3. Build the graph

Standard loop scaffold:

```
reset → init edges → iterIn → <method nodes> → step → iterOut
                                          (done → iter_out.stop)
                                                  ↓ final_stop (once, at termination)
                                              evaluate
                                                  ↓
                                              graphOut (metrics)
```

LLM calls go through built-in `llmCall`; the method nodeset only does prompt assembly + parsing.

**`step_budget` is method-specific.** Use the upstream paper's value (usually in their config), not a uniform safety margin. NavGPT-MP3D uses 15. MapGPT-MP3D uses 50. VLN-CE methods often need 150 because the env's discrete-step contract demands it.

### 4. Wire up eval

- Drop a `workspace/architect/exp_profiles/<method>_<env>.yaml` (mirror an existing one, e.g. `mapgpt_mp3d.yaml`): `split`, `worker_count`, `step_budget`, `per_step_budget_sec`, `primary_metric`.
- Add a row in `.claude/commands/experiment/profiles.yaml` if the (vram, exclusive_gpu) signature differs from existing entries — `experiments:` block, keyed by your profile name.

### 5. Phase-1 gate

Run a smoke + a calibration eval, in that order:

```bash
# Smoke — 3 ep, single worker
/experiment:run <profile> <graph_name> episode_count=3 worker_count=1 step_budget=15

# Calibration — 20 ep × 20 worker on the paper's split
/experiment:run <profile> <graph_name> episode_count=20 worker_count=20 split=<paper-split>
```

The gate passes when **all four** hold:

1. All 20 episodes complete without backend error / timeout.
2. Per-step `planner_llm.inner_log[rendered_prompt]` (in `outputs/eval_runs/<run_id>/episodes/ep*/log.jsonl`) reads like the paper's described prompt format. Eyeball one episode's first 3 steps.
3. **Eyeball the rendered pixels too**: decode one step's image b64 from `log.jsonl`, check resolution + codec magic bytes against the upstream's config (a five-minute check that catches a 224-vs-1024 mismatch on day one, instead of after the paper-tier run refutes your SR).
4. Headline SR lands in the paper's noise band on the same split — but **first pin how the paper computes that number from the upstream eval code** (measures kept or stripped? stop-gated or distance-only?). Compare with the same ruler; if the upstream ruler differs from the benchmark's official one, record the definition in the exp profile + doc and dual-report. (Three-Step's SR has no STOP gate: the same trajectories score 0.08 official vs 0.24 upstream-ruler — an unpinned ruler reads as a permanent 20-point gap.) For the band itself, compute the binomial noise (20 ep at SR=0.2 has ~±9 SR-pt MC noise — be charitable).

If (3) misses badly, **debug the monolith — do not decompose**. Most common culprits:

- **Wrong model**: profile name in the graph drifted from what's actually being called. Always read the model from `log.jsonl planner_llm.inner_log[model]`, never trust the profile name (see [[feedback_verify_model_from_log]]).
- **Wrong stop-gating, prompt segment order, or option formatting** — diff your assembled prompt against the upstream's rendered prompt for an identical input.
- **Wrong split / start_episode_index** — the paper's 216-ep MapGPT72 subset isn't `val_unseen[:216]`.

**Phase 1 done = ship-able port.** A monolith that produces correct metrics is a complete deliverable.

## Phase 2 — Decompose (optional, ungated)

Stop after Phase 1 unless you have a concrete reason to split:

1. You want to **A/B individual stages** (swap one part, hold the rest).
2. The architect loop will **mutate one stage at a time** — needs explicit seams.
3. Downstream consumers need a **stable, fine-grained I/O contract** that the split makes explicit.

**Counter-signal — do NOT split** when the mono is already at the framework floor (~2N+1 nodes for N LLM calls) and the only fat node is fat because of a **conditional LLM call** (a static graph cannot express it as a graph-level `llmCall` in any split): the decomposition merely relocates state-writing. `threestepnav`'s Phase-2 decomp was built, proven byte-equivalent (13/13), and later deleted for exactly this reason.

**Keep the validated mono. Decomp goes in a NEW nodeset + NEW graph.** Stage the decomp as a sibling nodeset (e.g. `mapgpt` → `mapgpt2`, or `smartway2` → `smartway3` during development). Once the decomp ships and proves itself you can optionally promote it to the canonical name (`smartway` = decomp, `smartway_mono` = reference baseline), but the validated mono stays alongside as ground truth. The mono stays as ground truth — its `SmartwayMono{PlanStep,DecideAction}` classes are what the byte-equivalence test compares against. Don't in-place edit a validated mono until the decomp has shipped and proved itself.

### How to split safely

1. **Identify natural seams** along the monolith's concerns: env-feature extraction → state update → option/manifest construction → view rendering. Not every monolith needs all four cuts — let the function's body suggest where.
2. **Single-writer-per-state-key is the seam-finder.** Inspect the monolith's `gs.write(...)` calls — each state key should be claimed by exactly one node in the decomp. Multi-writer patterns usually hide a refactor: the SmartWay `backtrack` latch had two writers (plan_step's "read + clear at top", update_history's "set True if return picked") that collapsed to one when we noticed `backtrack = is_return` every step is semantically equivalent, with `initial_value=false` serving iter 0. After the latch collapse, `update_topology` became a pure reader — and that node fell out naturally as the first method node post-`iterIn`. Most split decisions surface this way: chase the writes.
3. **Snapshot state at the first method node post-`iterIn`.** That node reads `graph_state` once and emits `history_snap` / `planning_snap` / `last_*` as explicit data ports for the rest of the iter. Downstream nodes consume the snapshots, never re-read `graph_state`. This eliminates time-order coupling the monolith hid (everything inside one function reads state once, atomically — splitting without snapshotting introduces ordering bugs).
4. **Freeze env-boundary derivations in the first downstream node.** Anything that needs the env-side current values (heading deltas, panorama tile crops, pose-relative math) must be computed in the *first* node downstream of `iterIn` and ride forward as data — splitting that seam later loses access. The single most common failure of premature decomposition.
5. **Pass snapshots via explicit data ports**, not by re-reading `graph_state` mid-iter. (Corollary of #3, generalised: applies to any derived value, not just state.)

### Validation — byte-equivalence first, eval as wiring sanity

Two-tier. The first tier is the actual gate; the second only confirms nothing in the env-side plumbing broke.

**Tier 1 (mandatory gate): byte-equivalence unit test.** Feed the **same** fake/handcrafted inputs into mono and decomp, byte-compare every output port and every state write. Fast (~0.2 s for 10 scenarios), deterministic, GPU/API/network-free, and pinpoints the offending node if it fails. End-to-end eval can't do any of that. Reference impl: `workspace/nodesets/smartway/test_equivalence.py`. The recipe:

- One `FakeGraphState` (dict-backed `read` / `write`) + one `FakeCtx` (`step` + `graph_state`).
- For each scenario: build a fresh `gs` from a snapshot dict, run mono once, run decomp's composed pipeline (in graph-edge order), byte-compare outputs + state.
- **Determinism gotcha**: any node that mints `uuid.uuid4()` will produce different IDs on the two runs. Monkey-patch the *module-bound* `uuid` symbol in BOTH nodesets with a resettable counter, and call `_reset_uuid_counter()` at the top of each runner. Patch is per-module so production code is unaffected outside the test.
- Cover the per-step branches: t=0, t≥2 stop-letter, latch on/off, synth-return on/off, real-RGB decode path, every LLM-response shape (fenced / bare / prose-wrapped / malformed).

**Tier 2 (wiring sanity, not parity gate): one mini eval.** 100 ep on the paper's split. The signal is **completion rate**, not SR. If 100/100 episodes finish with non-empty metrics, server-mode wiring is intact. SR/OSR landing in mono's noise band is a confirmation, not a gate — Tier 1 already proved per-input equivalence. If completion < 100%, suspect framework cold-start races (waypoint/perception singleton 404 at session start), not your decomp ([[project_silent_episode_completion_on_node_error]]).

### Worked example — MapGPT-MP3D

- Nodeset: `workspace/nodesets/mapgpt.py` · Graph: `workspace/graphs/mapgpt_mp3d.json`
- **Phase 1**: single atomic `plan_step` node packed nine concerns (state read · current-vp fold · candidate scan + tile crop + direction-phrase · adjacency update · state write · stop gating · letter prefixing · manifest emission · 7-field prompt render · image-list assembly). Plus `system_prompt` / `parse_action` / `update_history` / `image_budget` = 5 method nodes total. 216-ep MapGPT72 with gpt-5-nano: SR band 0.185–0.231, SPL ~0.15, nDTW ~0.38. **End-to-end parity locked in — this is the I/O contract for Phase 2.**
- **Phase 2** (2026-05-15): `plan_step` split into 4 — `observe` (env→features) · `update_map` (sole state owner) · `build_options` (manifest assembler) · `render_prompt` (template). Direction phrases are computed in `observe` and frozen into `candidates_json` — a prior attempt that deferred them to a view-serializer stage lost current-heading access. `topo_snapshot` rides as an explicit ANY-typed wire from `update_map` to both consumers; never re-read from `graph_state` mid-iter. 20-ep gpt-5-nano sanity: **SR 0.200** (squarely in Phase-1 noise band). On gpt-5-mini: SR 0.550 / oracle_SR 0.85 — within paper-reported CI on the gpt-4o end.
- The point: the manifest pattern (`[{letter, vp, phrase}]` consumed by `parse_action` as a table-lookup) survived the split unchanged. That's exactly what monolith-first buys you — a stable contract on the seam you actually care about.

### Worked example — SmartWay-CE (smartway_mono → smartway)

- Mono: `workspace/nodesets/smartway_mono/__init__.py` · Decomp: `workspace/nodesets/smartway/__init__.py` · Graphs: `smartway_mono_ce.json` / `smartway_ce.json` · Equiv test: `workspace/nodesets/smartway/test_equivalence.py`
- **Phase 1 (`smartway_mono`)**: 3 method nodes — `plan_step` (175 LOC, 11 concerns) / `decide_action` (100 LOC, 5 concerns) / `update_history` (50 LOC, 3 concerns). rand100 (paper's official 100-ep subset of val_unseen) with gpt-4o-2024-08-06: SR=0.270 / OSR=0.494 / NE=7.06 at n=89 (11 silent-failed eps from framework race). Paper SR=0.29, OSR=0.51, NE=7.01 — locked in.
- **Phase 2 (`smartway`, 2026-05-16)**: 3 → 7 method nodes. `plan_step` split into `update_topology` (sole writer: `nodes_list` / `graph` / `trajectory`; emits state snapshots) + `build_action_options` (pure: per-cand "Place N" + stop-letter prefix) + `assemble_prompt` (pure template render) + `build_images` (pure RGB decode + parallel `image_labels`). `decide_action` split into `parse_response` (sole writer: `planning`) + `resolve_action` (`picked_index` → angle/distance/is_return). `update_history` unchanged but absorbed the `backtrack` latch's clear-half — now sole writer of `backtrack`, with `backtrack = is_return` every step. Validation: **byte-equivalent on 10 scenarios in 0.18s**, then **100/100 rand100 eps completed** (vs mono's 89/100) with SR=0.300 / OSR=0.630 / NE=6.20 — SR matches paper exactly, OSR borderline +13 pt over mono (mostly segment-selection because mono's denominator was missing the 11 silent-failed eps).
- The point: single-writer-per-state-key drove every seam decision. `update_topology` exists because three keys needed an owner that wasn't `plan_step`; `parse_response` exists because `planning` needed an owner that wasn't `decide_action` half-doing the write; the `backtrack` latch collapse is what made the snapshot pattern (`history_snap` / `planning_snap`) clean. The byte-equivalence test was 30 LOC of fixture + 200 LOC of scenarios and ran in 0.18s — orders of magnitude cheaper than the 15-min mini-eval, and stronger evidence.

## Phase 3 — Document the port

A ported nodeset isn't a deliverable until it has a method-nodeset doc page. Once the nodeset is validated (Phase 1 monolith shipped, and/or Phase 2 decomp passed), write its page in the **method-nodeset doc template style**:

```
/docs:method-nodeset-doc <nodeset>
```

That skill is the process; the normative *spec* it fills is `.claude/commands/_data/method-nodeset-doc-spec.markdown`. The page lands at `docs/pages/developer-guide/nodesets/method/<nodeset>.html` and must carry:

- **Lede + at-a-glance** — upstream paper · pinned commit · env spaces · FM nodes · node count · graph(s) · verified status + headline number vs paper · **fidelity verdict** (`faithful` / `faithful with justified deviations` / `divergent` / `defective`, dated).
- **§1 Upstream analysis** — logic-level core loop with `file.py:line` anchors into the pinned commit + an upstream-flow inline SVG; close with the invariants the port must preserve.
- **§2 The port** — per-step inline SVG (mirror §1's layout, visible loop-back edge); node inventory (`handle:TYPE`); state & memory (reducer/lifetime/writer); boundary contract (which `env_*` + FM nodes; reasoning-only statement); prompt assets (what's verbatim).
- **§3 Five-bucket delta** (the signature section) — every difference filed into exactly one of **A** verbatim · **B** substrate-forced · **C** env/cost-forced · **D** intentional (with *cited* recorded intent) · **E** unexplained/defects, each row leading with an equivalence icon (🟢/🟡/🟠). The byte-equivalence test (Phase 2 Tier 1) is the evidence behind the A-bucket rows; the eval run_id is the evidence behind §4.
- **§4 Eval · §5 Usage · §7 Sources + changelog** per the spec.

Then register the page in `method/index.html` and rebuild: `python3 docs/_lib/_wrap_handwritten.py`.

**To drive fidelity to convergence afterward**, run `/grill-implement <nodeset>` — the inventory-first loop that walks the upstream cold, empties bucket E, and confirms every B/C/D row is genuinely irreducible. (It needs the upstream source on disk — keep the `_upstream/<name>/` clone or `tmp/_reference/<name>/` checkout that Phase 1 read from; a gitignored reference that gets cleaned up later cannot be re-audited.)

## Landmines

- **v3 `iterIn` dual-wire freeze.** Every loop port has `init_<name>` + `iterout_<name>`. Wire both to the same downstream input *and* set `init.persist=true` ⇒ consumer reads the iter-0 value forever. Symptom: agent picks the same action every step. Fix: `persist=false` on init, or wire only the `iterout_<name>` flavour to in-loop consumers. ([[feedback_iterin_dual_wire_obs_freeze]])
- **Verify model from `log.jsonl`, not the profile name.** Profiles drift; the field `planner_llm.inner_log[model]` is authoritative. ([[feedback_verify_model_from_log]])
- **Multi-LLM fan-out is sequential.** The executor runs nodes serially per superstep until parallel execution lands. DiscussNav-style "12 directions × VLM + summarize × 12 + pred × 5 retries" works but is slow. Options: chain serially, use one `llmCall` with `batched=true` if prompts are independent, or accept the latency.
- **Fork upstream? Diff it against its parent first.** A method repo that forks a baseline carries its real deltas in `git diff parent..fork` — sensor config, obs transforms, metric formulas. A port that reuses the parent's substrate inherits exactly the wrong defaults. (Three-Step vs Open-Nav: 1024 camera, CenterCropper→ResizerPerSensor switch, stop-gate removal from the SR formula.)
- **Silent model-load fallback in shared nodesets.** A try/except around a checkpoint load that degrades to zeros poisons every consumer family-wide, and no eval will tell you. Loads must be FATAL; grep reused components for swallowed load failures before trusting any number that flows through them (the shared DDPPO depth encoder had never loaded in any Open-Nav-family run).
- **All-zero metrics ≠ your wiring is broken.** Fingerprint: `summary.json` shows N/N episodes `status="completed"` with `step_count=0` and `metrics={}`, elapsed ~20 s each. GraphExecutor silently completes the episode when an upstream node returns `{'error': '...'}` instead of its declared port dict — port routing only forwards declared keys, so the error dict is dropped, downstream stays unready, the loop drains empty and exits "clean". Usually a shared singleton (`smartway_waypoint`, `smartway_perception`, `opennav_*`) controller-registered 404 at session start because its server wasn't ready when proxy generation ran. **First check episode 0's `log.jsonl` for `{'error': ...}` outputs — find the node BEFORE the gap. If it's an env/perception node, restart the backend and resubmit; nothing wrong with your method nodes.** ([[project_silent_episode_completion_on_node_error]])

## Stop signals

Push back on the port when:

- Upstream uses a **private / closed model** (unreleased checkpoint or internal-only API) — behaviour is unreproducible.
- Upstream uses an **env we haven't wrapped** (REVERIE object grounding, GibsonEnv, ProcTHOR) — wrapping the env is a separate project.
- The "method" is a **fine-tuned single-network model** (HAMT, DUET, VLN-SIG, RecVLN-BERT) — that's a `policy_*` adapter task, not a graph port. Wrapping it as one `llmCall` discards the architectural structure the paper is about.
- **Per-episode cost is unclear** and could blow budget — multi-LLM debate methods can hit ~$30+/ep at gpt-4 prices.

## What "done" looks like

Phase 1 with the headline metric in the paper's noise band, on the paper's split, with verbatim prompts and the upstream loop shape preserved. That's a ship-able port. Tighter parity is a research project, not a porting deliverable. Phase 2 is upside, not closure. The deliverable is fully complete when the validated port also carries its method-nodeset doc page (Phase 3) in the template style — that page, not the code alone, is what a reviewer reads to trust the port.
