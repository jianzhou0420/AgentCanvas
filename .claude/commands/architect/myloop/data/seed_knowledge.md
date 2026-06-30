# Knowledge

Seed knowledge — copied into `outputs/design_runs/myloop/{graph}/v{N}/knowledge.md`
when bootstrapping a fresh vN. Orchestrator extends this with
graph-specific and dataset-specific facts discovered during the run.
Every bullet ends with `(added: bootstrap | iter_N)` for diff
visibility.

---

## AgentCanvas system

- A graph is **two things**: a JSON topology file at
  `workspace/graphs/{name}.json` (nodes + wires + config) and the
  Python implementations of any custom node prefixes at
  `workspace/nodesets/{prefix}.py`. The JSON references node prefixes
  by id; the runtime maps prefix → module. (added: bootstrap)
- A graph's edit scope (TODO #60, 2026-05-15): patches MAY target ANY
  path under `workspace/` — graphs, local nodesets, server-mode
  nodesets at `workspace/nodesets/server/**` (both single-file and
  package-mode), env wrappers, policies. The EXPERIMENT apply-step
  applies edits into the iter's `active_workspace/` overlay and
  mirrors enclosing Python packages on first touch so package-mode
  edits stay self-contained. NEVER editable: `agentcanvas/backend/app/**`
  (framework code) and `third_party/**`. Server-mode `shared`
  nodesets auto-spawn an ephemeral auto_host child at eval admit time
  when the overlay's source content hashes differently from frozen —
  first iter touching one pays ~30-60 s spawn cost and VRAM doubles
  for the eval's duration (frozen + ephemeral coexist). (added: bootstrap;
  updated 2026-05-15 for TODO #60)
- `llmCall` node config keys actually read by the runtime:
  `profile, temperature, max_tokens, system_prompt, template, mode,
  n, stop`. The `model` field is **not** read — writing
  `"model": "gpt-4o-mini"` does nothing. To route to a specific LLM
  use `profile`. (added: bootstrap)
- gpt-5 family profiles REQUIRE `temperature == 1.0`. gpt-4o / Claude
  families accept `0 ≤ temperature ≤ 2`. Wrong-family + wrong-temp
  combinations raise `litellm.UnsupportedParamsError`. (added: bootstrap)
- Backend strict-mode (`AGENTCANVAS_STRICT_ERRORS=1`) propagates
  llmCall failures; without it, failed calls silently return empty
  strings. (added: bootstrap)

## How experiments produce data

- Every `/experiment:run` call writes to
  `outputs/eval_runs/{run_id}/` where `run_id` is a second-precision
  timestamp like `20260515_130940`. (added: bootstrap)
- Per-run top-level files: `spec.json`, `summary.json`, `graph.json`,
  `stdout.log`, `stderr.log`, `_DONE`. (added: bootstrap)
- **Per-episode subdirs**: `episodes/ep{i:04d}/`, each with:
  - `episode.json` — this ep's metrics (single self-describing row of
    summary.json)
  - `log.jsonl` — per-node-firing ExecutionLogger stream for this ep
    (every node invocation: inputs, outputs, timing, sometimes inner
    LLM trace)
  - `assets/` — image artifacts (panorama frames, etc.; can be 10s of
    MB per ep). (added: bootstrap)
- `summary.json` aggregates per-ep rows plus run-level metrics.
  Aggregated metrics are means over completed eps. (added: bootstrap)
- An ep that never reached the goal but oracle could (the agent had a
  reachable path but stopped wrong) has `oracle_success=1, success=0`.
  This is the dominant failure mode worth disentangling. (added: bootstrap)

## EXPERIMENT phase + `/experiment:run`

- myloop has **no implementer / evaluator skill** — the EXPERIMENT
  phase (`loop.md § 3c`) is inline in the loop. When
  `ExperimentSpec.patch` is non-null it applies the change: an
  isolated editing sub-agent edits the seeded files under
  `.staging/iter_n/active_workspace/` natively, then a 3-ep smoke
  eval gates runtime correctness only (exit-clean, all eps complete,
  step>0, valid metric — score *values* are not consulted), retrying
  with a fresh sub-agent up to 3 times. On exhaustion the iter still
  commits with `outcome_class="implementer_skip"` (a refuted-patch
  data point). Frozen `workspace/*` is never modified; the eval reads
  the iter's mutations via the `--workspace=<active_workspace>`
  overlay. (added: bootstrap)
- `/experiment:run <admission> <graph> <key=value...>` submits a
  graph eval to the agentcanvas backend's JobScheduler. The
  admission profile lives in `.claude/commands/experiment/profiles.yaml`
  (VRAM declaration); the eval params come from
  `experiment_design.yaml` OR direct CLI overrides. Returns a
  `run_id` on completion. (added: bootstrap)

## Dataset / env (R2R-class)

- R2R val_unseen has 11k+ instructions. (added: bootstrap)
- **`MapGPT72`** — 216 paths (3 instructions × 72 scans). The
  paper-comparable headline subset for MapGPT-class methods.
  Defined at `data/mp3d/tasks/R2R/R2R_MapGPT72.json`. (added: bootstrap)
- **`MapGPT72_first`** — 72 paths (1 instruction × 72 scans —
  the *first* instruction per path). Useful as a **mid-size
  experiment**: full 72-scan coverage at ~1/3 the cost of MapGPT72.
  Good middle layer between 5-ep smoke and 216-ep full when an iter
  needs scan-level breadth without the 3× instruction repetition.
  Defined at `data/mp3d/tasks/R2R/R2R_MapGPT72_first.json`. (added: bootstrap)
- val_unseen front-loads easy scans; use MapGPT72 (or
  MapGPT72_first) for any number you'd compare against a paper,
  not arbitrary 100-ep slices of full val_unseen. (added: bootstrap)
- Per-ep metrics from env_mp3d: `success`, `spl`, `nDTW`, `SDTW`,
  `oracle_success`, `nav_error`, `oracle_error`, `trajectory_length`,
  `trajectory_steps`. (added: bootstrap)

## Failure-mode prior art

- ADAS-style failure classes seen on VLN agents: (a) `step=0` (wiring
  bug, agent never fires), (b) `all-zero acc` (semantic bug,
  agent fires but always picks wrong direction), (c) `wandered`
  (nav_error >> oracle_error), (d) `early-stop`
  (oracle_success && !success: agent stopped at wrong location).
  Most-improvable bucket is usually (d). (added: bootstrap)

## Where else to look (pointers, not contents)

- Glossary of all node/wire/port/iter-in/iter-out terms:
  `docs/core/glossary.md`. (added: bootstrap)
- Architect pipeline shared contract:
  `.claude/commands/architect/_common/files-contract.md`. (added: bootstrap)
- This variant's own README + schemas + understand:
  `.claude/commands/architect/myloop/{README,schemas,understand}.md`.
  (added: bootstrap)
