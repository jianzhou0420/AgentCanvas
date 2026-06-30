# Apply patch and smoke-test

> **Common skill stub** — the procedure lives in
> `.claude/commands/architect/_common/implementer.md`. Execute that
> file's steps with this variant binding:
>
> - `VARIANT      = aflow`
> - `VARIANT_DIR  = .claude/commands/architect/aflow/`
> - `CONFIG       = <VARIANT_DIR>/config.yaml § implementer:`
>
> Variant-specific values (smoke profile key, patch applier path,
> retry cap, side-effect artifacts) come from the config — do not
> duplicate them here.
>
> Note: aflow's `loop` skill pre-populates
> `.staging/iter_n/active_workspace/` via softmax-sampled parent
> checkout before invoking this skill. The common implementer's
> Step 2 detects pre-populated state and skips its own bootstrap;
> no extra config needed.
