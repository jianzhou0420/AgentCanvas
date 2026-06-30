# Constraints — graph: mapgpt_mp3d

Graph-specific hard rules for `mapgpt_mp3d`. Bootstrap merges this
file as layer 2 (after common.md) into vN/constraints.md.

## Model choice

- MUST keep `planner_llm.config.profile = "gpt-5-mini"` across all
  iters. Do NOT propose patches that change it to `gpt-4o`,
  `gpt-4o-mini`, `gpt-5-nano`, `gpt-4-vision`, Claude, or any other
  profile. Rationale: gpt-5-mini is the chosen baseline for this
  vN; comparison points against the MapGPT paper (0.477 gpt-4v /
  0.463 gpt-4o on 216-ep MapGPT72) are stable only if the model
  stays fixed across iters. Model-swap experiments belong in a
  separate vN with its own merged constraints.

- MUST keep `planner_llm.config.temperature = 1.0`. The gpt-5
  family raises `litellm.UnsupportedParamsError` for any other
  value — proposing temperature != 1.0 will crash smoke and waste
  an iter.
