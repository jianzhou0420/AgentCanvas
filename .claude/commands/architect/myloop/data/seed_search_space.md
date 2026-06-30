# Search Space

> Bootstrap seed for `search_space.md` — the explicit map of the
> **intervention space** a myloop run is searching. The orchestrator's
> THINK phase keeps proposing experiments; without this file it has no
> representation of *what kinds of change are possible*, and reliably
> ruts in whichever axis its first few iters happened to pick. The
> REFLECT phase (`myloop/reflect.md`) maintains the Coverage section
> below and hands THINK a ranked frontier.
>
> This file is `working-memory`, append-only. The Axes taxonomy is a
> starting menu — REFLECT MAY extend it when the graph / codebase
> exposes an intervention kind not listed. Coverage grows one
> `## reflection_N` section per REFLECT spawn.

## Axes (intervention taxonomy)

A patch to an agent graph changes exactly one or more of these axes.
Two patches on the *same* axis are variations of one lever; two
patches on *different* axes are structurally distinct experiments.
THINK self-labels every `ExperimentSpec` with one `intervention_axis`
(schemas.md § 8); REFLECT reads those labels to measure coverage.

- **prompt-content** — the *text* an LLM/VLM node reads: system
  prompts, prompt templates, option / action phrasing, rendered
  observation or map text. The most accessible axis and usually the
  first a run reaches for. Caution: editing the text a single
  decision LLM reads tends to reroute its whole output distribution,
  so recover-some/break-more is the characteristic failure here. A
  Python edit to a function whose *output* is prompt text still
  counts as this axis — the axis is "what the LLM reads", not "JSON
  vs code".

- **topology** — the node/wire graph *structure*: adding or removing
  nodes, rewiring, fanning a node into parallel branches, ensembles /
  self-consistency voting, sub-graphs, aggregator / verifier / critic
  nodes. Distinct from prompt-content because it can leave every
  existing prompt byte-identical and still change behaviour.

- **control-flow** — the loop and branching *logic*: loop / iteration
  structure, stop / STOP logic (the `iter_out.stop` halt input),
  retry, backtracking, conditional branching, and the two-pivot
  mechanism (two-sided iterIn / iterOut with its final side).
  Changes *when* and *how often* nodes fire, not what any one of
  them reads.

- **action-space** — *which* choices reach the decision-maker:
  pruning, filtering, reordering, or augmenting the candidate-action
  set. Distinct from prompt-content: re-*wording* an option is
  prompt-content; removing it from the set the agent can pick is
  action-space.

- **observation-pipeline** — what the agent *perceives* and how it is
  computed, upstream of any prompt: sensors, feature extraction,
  what is captured vs dropped, resolution / modality / preprocessing.
  Changes the information available before it is ever rendered.

- **state-memory** — what information *persists* across steps /
  iterations and how it is structured: working memory, history
  buffers, scratchpads, the carried-state schema, what iterOut feeds
  back. Changes what the agent can remember, not what it sees now.

- **model-component-config** — per-node configuration within the
  run's hard constraints: sampling parameters (`n`, `max_tokens`,
  `stop`), and adding / swapping *non-restricted* components (a tool,
  a non-LLM module, a deterministic helper). Excludes anything a
  `constraints.md` rule forbids (e.g. a fixed model profile /
  temperature).

REFLECT MAY add an axis if the graph or codebase exposes an
intervention kind none of the above captures — append it here with a
definition and a "why distinct" line, in the same `(added: ...)`
discipline as knowledge.md.

## Coverage

> One `## reflection_N` section per REFLECT spawn. Each records, per
> axis: status ∈ {untouched | partial | exhausted}, the iters that
> touched it, the verdict, and a ranked **Frontier** of the axes the
> next THINK should prefer. Bootstrap leaves this empty — the first
> REFLECT back-fills coverage from the iters committed so far.

_(no reflections yet — the first REFLECT will populate this section)_
