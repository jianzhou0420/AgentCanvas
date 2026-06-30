# smartway_ce — frozen-config constraints

Additional HARD constraints for any architect / adas-subagent loop running on
this graph. These are user-fixed config decisions that the meta-LLM must NOT
modify; the architect's design space is the rest of the graph + nodesets.

## planner_llm config is FROZEN

The `planner_llm` node (id `planner_llm`, type `llmCall`) has been deliberately
configured by the user. **Do NOT modify any of these fields**:

- `config.profile` — set to `"gpt-5-mini"` (gpt-5 family)
- `config.temperature` — set to `1.0` (required by gpt-5 family)
- `config.max_tokens` — set to `6000`
- `config.response_format` — set to `"json_object"`
- `config.mode` — set to `"single_turn"`
- `config.template` — set to `"{plan}"` (passthrough from `assemble_prompt.prompt`)

Allowed: modify `planner_llm.config.ports`, `system_prompt`, or the `template`
**only if** you also rewire the upstream `assemble_prompt` node to produce the
new prompt shape — and only when the design genuinely requires it. Otherwise
leave `planner_llm` exactly as it stands.

**Reason**: the user is benchmarking gpt-5-mini specifically; swapping the
planner LLM defeats the experiment.

## Wrong changes — forbidden for this graph

A `patch.intent` that does any of these to the existing `planner_llm`
node is forbidden:

1. Change `planner_llm.config.profile` (e.g. swap to `gpt-4o`).
2. Change `planner_llm.config.temperature` (e.g. set to `0.0`).
3. Change `planner_llm.config.max_tokens` (e.g. set to `600`).

## Allowed: NEW llmCall nodes

You may add new `llmCall` nodes (critic, verifier, summarizer, etc.) with any
profile / temperature combination that the cheat sheet permits — those are
under the architect's control, not frozen.
