# Eval runner

> **Common skill stub** — the procedure lives in
> `.claude/commands/architect/_common/evaluator.md`. Execute that file's
> steps with this variant binding:
>
> - `VARIANT      = adas-subagent`
> - `VARIANT_DIR  = .claude/commands/architect/adas-subagent/`
> - `CONFIG       = <VARIANT_DIR>/config.yaml § evaluator:`
>
> Variant-specific values (profile key, side-effect artifacts) come
> from the config — do not duplicate them here. The evaluator is
> method-free: bootstrap CI + `fitness_str` are produced in
> `loop.md`'s Atomic Writer, not here.
