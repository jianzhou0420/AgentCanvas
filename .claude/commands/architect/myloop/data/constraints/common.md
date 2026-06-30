# Constraints — common (all myloop runs)

Pipeline-wide discipline that applies regardless of which graph is
being worked on. Bootstrap merges this file as layer 1 into every
vN/constraints.md.

## Edit scope

- Patches MAY target ANY path under `workspace/` — including
  `workspace/graphs/*.json`, top-level `workspace/nodesets/*.py`,
  server-mode nodesets at `workspace/nodesets/server/**`, env
  wrappers, policies, hooks, anything. All edits land in
  `{ITER}/active_workspace/` (isolated overlay) and never pollute
  the frozen `workspace/` tree. You may pull additional information
  out of an env nodeset, rewrite a base class, swap a server-side
  model wrapper — whatever the experiment design requires.

  **Implementation note for server-mode edits (TODO #60, 2026-05-15)**:
  server nodesets at `workspace/nodesets/server/**` with
  `parallelism="shared"` (VLM / detection / policy inference services)
  used to require a manual backend restart to test overlay edits. They
  now auto-spawn an ephemeral auto_host child at eval admit time
  whenever the overlay's source content hashes differently from the
  frozen baseline. Implications:
    - First iter that touches a `shared` nodeset pays a one-time spawn
      cost (≈ 30–60 s for VLM-class models). Subsequent iters with the
      same overlay content reuse nothing (per-eval lifecycle); plan
      `eval_profile.overrides.per_step_budget_sec` accordingly.
    - VRAM doubles for the duration of the eval (frozen singleton +
      ephemeral child both live). If the iter's overlay touches a
      ~14 GB VLM on a 24 GB GPU, raise `marginal_vram_mb` on the
      experiment profile by the ephemeral's expected footprint, or
      explicitly `unload_nodeset(name)` on the frozen one beforehand.
    - `parallelism="replicated"` server nodesets (habitat / simpler /
      libero / openeqa / etc.) already hot-reload from the overlay —
      they spawn per-eval-worker inside the eval subprocess, no special
      treatment needed.

- Outside `workspace/` is NEVER editable:
    - `agentcanvas/backend/app/**` (framework code)
    - `third_party/**` (vendored upstream)
  The EXPERIMENT apply-step enforces this filesystem-level (via
  `_common/lib/overlay.py`); declared here so THINK does not waste
  tokens proposing patches that will bounce.

- The graph's loop semantics rest on a **two-pivot mechanism**
  (ADR-dataflow-008): a two-sided `iterIn` (its `config.initPorts`
  declare run-invariant seeds, wired in as canvas edges targeting
  `init_<name>` handles; its outputs are the per-iteration loop-carry
  bundle) and `iterOut` (collects per-iteration outputs + a `stop`
  BOOL halt input; transfers back to the paired iterIn — no canvas
  wire; its `final_<name>` outputs emit once at termination and feed
  the after-loop verdict stage).
  Both are EDITABLE — port shapes, `persist` flags, wire rewiring,
  adding / removing pivot nodes — BUT any patch MUST keep the pair
  semantically coherent so the graph still runs. A standalone
  `initialize` node no longer exists; graphs carrying one are
  rejected at load.
  Known silent failure modes (each wastes an iter):
    - `initPorts` entry with `persist=false` for a value loop-body
      nodes read every iteration → consumer starves on iter 1+ (the
      one-shot slot empties after iter 0). Run-invariants need
      `persist=true`.
    - Same input port wired to BOTH an `init_X` source AND an
      `iterout_X` source with `init.persist=true` → consumer freezes
      at iter 0 (the persisted init value never gets shadowed).
    - Missing or unwired `iter_out.stop` signal → loop runs to budget
      cap every episode; agent never issues STOP cleanly.
    - Verdict (evaluate/graphOut) wired from inside the loop body →
      validator rejects it; verdict inputs must ride the pivot (be an
      iterOut port, consumed via its `final_<name>` handle, which emits
      once at termination).
  Reference: `docs/pages/developer-guide/core/glossary.html` covers
  iter-ports, init-side semantics, and stop/final-side wiring in full.
