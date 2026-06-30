# Propose a new design — single call + anti-replay

> **Required reading**:
> - `aflow/README.md` § "Three structural contracts"
> - `_common/files-contract.md` § "Edit whitelist"
> - Upstream reference (read once):
>   - `third_party/AFlow/scripts/optimizer.py:132–183`
>     (`_optimize_graph` — the inner `while True` anti-replay loop)
>   - `third_party/AFlow/scripts/optimizer_utils/data_utils.py:40–109`
>     (`get_top_rounds` + `select_round` — softmax-mix parent sampling)
>   - `third_party/AFlow/scripts/optimizer_utils/experience_utils.py:12–80`
>     (`load_experience` + `check_modification`)
>   - `third_party/AFlow/scripts/prompts/optimize_prompt.py`
>     (`WORKFLOW_INPUT` template + `WORKFLOW_OPTIMIZE_PROMPT`)

This skill is invoked **inside the loop's Claude conversation** — not
as a separate sub-agent. The single LLM call + anti-replay retry loop
runs as Python orchestration inside this skill. The anti-replay
thread is NOT a Reflexion chain: each retry gets a freshly sampled
parent + fresh prompt, no cross-attempt critique.

## Arguments

```
/architect:aflow:proposer [<graph> [<version> [<iter>]]]
                          [--graph <name>] [--version <N>] [--iter <M>]
                          [--replay-max-retries K]  default from exp.yaml aflow.replay_max_retries
```

Iter resolution: `loop` passes the next iter index (`iter_n`); manual
invocation defaults to "the iter to be created" (max(M)+1).

## Pre-conditions

- `archive.jsonl` exists at `outputs/design_runs/aflow/{graph}/v{N}/` with
  at least one entry (the baseline with non-null `score`). If empty:
  this skill ERR — loop must run pre-seed first.
- `.staging/iter_n/` does **not** exist yet (or is empty). This
  skill creates it.
- `workspace/architect/exp_profiles/{graph}.yaml` has an `aflow:` block (defaults
  if missing: `sample=4, alpha=0.2, lambda_uniform=0.3,
  replay_max_retries=5, replay_norm="lower_ws"`).

## Steps

### 1. Resolve + read archive + read aflow config

Apply resolve protocol. Read full archive:

```python
archive = [json.loads(line) for line in open(archive_path)]
aflow_cfg = exp_yaml["aflow"]  # K, α, λ, replay_cap, replay_norm
```

Print:

```
[aflow:proposer] iter=iter_{n}  archive=v{N}/archive.jsonl ({K_arch} entries)
                 cfg: K_sample={K} α={α} λ={λ} replay_cap={cap} norm={norm}
```

### 2. Build per-parent experience map (in-skill Python)

Aggregate archive into:

```python
experience_map = {
    "iter_0": {"success": [], "failure": []},
    "iter_3": {"success": ["mod_str_A", ...], "failure": ["mod_str_B", ...]},
    ...
}
```

For each archive entry `e` with `e.parent_iter_id != null`:
- Find the parent's `score` (look up the entry with `iter_id == e.parent_iter_id`).
- If `e.score > parent.score`: append `e.modification` to
  `experience_map[e.parent_iter_id]["success"]`.
- Else: append to `["failure"]`.

This mirrors upstream `experience_utils.py:12–53` (`load_experience`),
implemented over `archive.jsonl` instead of per-iter
`experience.json` files. Sentinel: skip entries with
`modification == "(baseline)"` (no anti-replay against the baseline
entry itself).

### 3. Anti-replay retry loop

```python
for retry in range(aflow_cfg.replay_max_retries):
    # 3a. select_round (softmax-mix parent sampling)
    #   get_top_rounds: top-K archive entries by `score`. iter_0 is
    #     ALWAYS force-included even when it is outside the numeric
    #     top-K — verbatim upstream data_utils.py:53-55 (round_1
    #     unconditional inclusion); keeps the baseline reachable as a
    #     parent for the whole run.
    top_K = get_top_rounds(archive, K=aflow_cfg.sample)   # iter_0 always included
    #   select_round: p = λ·uniform + (1−λ)·softmax(α · score·100).
    #     The `score·100` rescale is REQUIRED and load-bearing. `score`
    #     is a bootstrap-median fraction in [0,1]; with α=0.2 a [0,1]
    #     score gap gives exp(α·Δ)≈1 — a near-flat softmax, so parent
    #     selection silently collapses to ≈uniform. Upstream rescales
    #     first (data_utils.py:67, `scores * 100`). select_round MUST
    #     apply the ·100 internally; do NOT pass pre-scaled scores.
    parent = select_round(top_K, alpha=aflow_cfg.alpha, lam=aflow_cfg.lambda_uniform)
    parent_id = parent["iter_id"]

    # 3b. Load parent's effective state
    parent_graph = read(f"outputs/.../iter_{parent_id}/active_workspace/graphs/{graph}.json")
    parent_prompt = render_nodeset_prompts(f"outputs/.../iter_{parent_id}/active_workspace/nodesets/")
    parent_log_samples = sample_log_failures(parent_id, n=3)  # 3 random failure samples from log.jsonl

    # 3c. Load parent's experience slice
    parent_exp = experience_map.get(parent_id, {"success": [], "failure": []})

    # 3d. Build optimize_prompt
    prompt = build_optimize_prompt(
        parent=parent,
        parent_graph=parent_graph,
        parent_prompt=parent_prompt,
        parent_score=parent["score"],
        parent_log_samples=parent_log_samples,
        experience=parent_exp,
        operator_descriptions=load_operator_descriptions(graph),  # see § 5 below
    )

    # 3e. Single LLM call (or sub-agent spawn — see § 4 Sub-agent contract)
    try:
        response = call_meta_llm(prompt, schema=GraphOptimizeSchema)
    except FormatError:
        # 3f. Regex-extract fallback (verbatim optimizer.py:164–174)
        try:
            response = regex_extract(raw_response, tags=["modification", "graph", "prompt"])
        except ExtractionFailed:
            continue  # → next retry, fresh parent

    # 3g. check_modification — anti-replay
    if check_modification(response.modification, parent_exp, norm=aflow_cfg.replay_norm):
        # Duplicate against this parent's history → resample
        print(f"[aflow:proposer] retry {retry+1}/{cap}: modification duplicate against parent={parent_id}")
        continue

    # Accepted
    break
else:
    # Loop completed without break — exhausted retries
    print(f"[aflow:proposer] SKIP — anti-replay exhausted after {cap} retries")
    return status=SKIP_REPLAY_EXHAUSTED
```

### 4. Sub-agent contract (the LLM call mechanism)

The single `call_meta_llm` in step 3e is **one
`Agent({subagent_type: "general-purpose", ...})` spawn**. Each spawn
is one independent Claude sample with full tool access — same
mechanism as adas-subagent's propose/R1/R2 sub-agents, but here only **one
spawn per anti-replay attempt** (the anti-replay loop's diversity
comes from re-sampling the parent, not from sub-agent reflection).

```python
resp = Agent({
    "subagent_type": "general-purpose",
    "description": f"aflow propose iter_{n} retry={retry}",
    "prompt": (
        "You are the meta-LLM in an AFlow architecture-search loop. "
        "Your task: propose ONE modification to the parent agent's "
        "graph + nodeset Python.\n\n"
        "You have full tool access (Read / Grep / Bash). Use it to "
        "inspect the parent's state, archive entries, eval logs under "
        f"outputs/eval_runs/, related literature.\n\n"
        f"=== Parent ===\nparent_iter_id: {parent_id}\nparent_score: {parent.score}\n\n"
        f"=== Parent's effective graph ===\n{parent_graph_summary}\n\n"
        f"=== Parent's prior modifications (DO NOT REPEAT these) ===\n"
        f"Successful (already explored, don't redo):\n  - {success_list}\n"
        f"Failed (also don't redo):\n  - {failure_list}\n\n"
        f"=== 3 random failure episodes from parent's log.jsonl ===\n{log_samples}\n\n"
        f"=== Operator catalog (available canvas-node types) ===\n{operator_descriptions}\n\n"
        f"=== Optimize prompt (verbatim from upstream prompts/optimize_prompt.py) ===\n{WORKFLOW_OPTIMIZE_PROMPT}\n\n"
        'Return a SINGLE fenced ```json block as your final message '
        'with keys {"modification", "thought", "name", "patch"}. '
        'The "modification" field is a one-sentence natural-language '
        "description of your change (used for anti-replay against the "
        'parent\'s prior attempts). The "patch" field is a change '
        "spec (prose intent + target file list) — see proposer.md "
        "§ 6. Do NOT edit workspace/* directly; the implementer "
        "realizes the change."
    ),
})
response = parse_final_json(resp)
```

Sub-agent contract:
- `subagent_type: "general-purpose"` — full tool access.
- Sub-agent MAY use any tool to ground its proposal.
- Sub-agent MUST emit a single fenced ```json block as the final
  visible chunk.
- Sub-agent MUST NOT edit `workspace/*` directly — patch application
  is implementer's job.
- Sub-agent MAY use `Agent(...)` recursively if grounding requires
  it.

On sub-agent exception (Agent error, no parseable JSON,
malformed JSON): `continue` the anti-replay loop (treat same as
FormatError fallback path). Verbatim upstream `optimizer.py:170–174`
semantics: extraction failures are retries, not iter-SKIPs.

After `aflow_cfg.replay_max_retries` exhaustion → SKIP iter.

### 5. Operator descriptions

Upstream injects `load_operators_description(self.operators)` —
descriptions of the workflow-internal building blocks (Custom,
ScEnsemble, Programmer, etc.). The AgentCanvas analogue is the
loaded NodeSets' canvas-node catalog.

Default behavior (Q7 in algorithm.html, defaulted to "filter to
parent's nodes + curated common list"):

```python
def load_operator_descriptions(graph):
    # Per-graph: read the graph's nodeset prefixes from {graph}.json,
    # fetch their node schemas via GET /api/components/node-schemas,
    # plus a curated short-list of commonly-proposed builtin types
    # (controllers, logic, state, history).
    used_prefixes = extract_nodeset_prefixes(graph)
    used_schemas = fetch_schemas(used_prefixes)
    common_schemas = fetch_schemas(["controller", "logic", "state", "history"])
    return render_as_markdown(used_schemas + common_schemas)
```

If prompt-budget pressure appears: switch to "parent's nodes only"
(narrower) or "full catalog" (broader). Tunable.

### 6. Patch schema (the `patch` field)

Same change-spec schema as adas-subagent (see its proposer.md § 5).
F1 from algorithm.html chose a **non-wholesale** mutation surface
(don't re-emit whole nodeset files — AgentCanvas agents are large
multi-file artifacts, unlike ADAS/AFlow's small `forward()`). Since
2026-05-20 that incremental change is expressed as a prose `intent` +
`targets` list and realized by the implementer sub-agent's native
editing — NOT a typed `graph_edits` op enum. (The op enum was the
compromise forced by routing edits through a deterministic replayer;
an agentic implementer edits files directly and needs none — which is
also what F1 originally wanted, "free-form patch, no typed action
enum".)

```json
{
  "modification": "Add a HistoryTracker node upstream of NavGPTReason; wire its summary into navgpt context; raise max_steps from 5 to 15.",
  "thought": "**Insights:** ... **Overall Idea:** ... **Implementation:** ...",
  "name": "Self-Audit Chain-of-Thought via History Tracker",
  "patch": {
    "intent": "Prose: exactly what to change and why — specific enough for the implementer to realize it by editing files. Describe the change; do NOT write op lists or jsonpaths.",
    "targets": ["workspace/graphs/{graph}.json", "workspace/nodesets/<prefix>.py"]
  }
}
```

`patch` MAY be `null` for a no-change probe. `modification` (the
one-line anti-replay key, § 7) is separate from `patch.intent` (the
full change description the implementer consumes) — keep both.

Upstream's separate `<graph>` / `<prompt>` output slots collapse into
one `intent`: a prompt-only change is just an `intent` naming the
nodeset file and the new prompt text, with that nodeset in `targets`.

### 7. `check_modification` semantics

Verbatim upstream from `experience_utils.py:69–80` with one
non-default option:

```python
def check_modification(new_mod: str, parent_exp: dict, norm: str = "lower_ws") -> bool:
    """
    Return True if new_mod is a duplicate (should be rejected and resampled).
    Return False if new_mod is novel (should be accepted).
    """
    candidates = parent_exp["success"] + parent_exp["failure"]
    if norm == "verbatim":
        return new_mod in candidates
    elif norm == "lower_ws":
        # Default — Q3 deviation from upstream; cheap, robust against trivial paraphrase
        def normalize(s):
            return " ".join(s.lower().split())  # collapse whitespace, lowercase
        return normalize(new_mod) in [normalize(c) for c in candidates]
    elif norm == "embed":
        # Cosine sim > 0.95 — requires embedding model call per check
        # Not enabled by default; opt-in via exp.yaml
        return any(cosine_sim(new_mod, c) > 0.95 for c in candidates)
```

Default `norm="lower_ws"` documented as the only soft deviation from
upstream's verbatim string equality (Q3 in algorithm.html).

### 8. Exception handling

| Failure | Action |
|---|---|
| FormatError (LLM schema validation fails) | regex-extract fallback (step 3f) |
| Regex-extract fails | `continue` anti-replay loop |
| `check_modification` returns True | `continue` anti-replay loop |
| Sub-agent exception (Agent error, no JSON) | `continue` anti-replay loop |
| Anti-replay retry cap exceeded | `return SKIP_REPLAY_EXHAUSTED` |
| Any other Python exception | `return SKIP_LLM_EXCEPTION` |

Verbatim upstream `optimizer.py:96–105` semantics: any inner-loop
exception → score=None → no child written. Loop's response:
`rm -rf .staging/iter_{n}/`, `consecutive_skips += 1`.

### 9. Write proposal.md to staging

On success:

```bash
mkdir -p outputs/design_runs/aflow/{graph}/v{N}/.staging/iter_{n}/
```

Write `.staging/iter_n/proposal.md`:

```markdown
---
generation: {n}
iter_id: iter_{n}
parent_iter_id: iter_{parent_id}            # NEW vs adas-subagent
name: <from response.name>
modification: <verbatim from response.modification>   # NEW vs adas-subagent — load-bearing for anti-replay on future iters
replay_retries: {retry+1}                   # how many anti-replay loops this iter took
---

# Thought

<response.thought verbatim>

# Modification (one-line description, for anti-replay)

<response.modification verbatim>

# What changed (diff narrative)

<proposer writes a human-readable summary of the intended change>

# Change

```json
{patch from response}
```

# Selection trace

- Parent sampled: iter_{parent_id}  (score: {parent.score:.4f})
- Top-K candidates: {top_K_summary — list of (iter_id, score)}
- Anti-replay retries: {retry+1}
- Parent's prior experience size: success={N_succ}, failure={N_fail}
```

### 10. Return to loop

```
[aflow:proposer] OK — proposed "<name>"
                 parent          = iter_{parent_id}
                 anti-replay     = {retry+1}/{cap} retries
                 targets         = {count}
                 staging         = .staging/iter_{n}/proposal.md
```

Return `status=OK` to loop.

## Outputs

| Path | What |
|------|------|
| `v{N}/.staging/iter_{n}/proposal.md` | Frontmatter (with `parent_iter_id` + `modification` + `replay_retries`) + thought + diff narrative + `# Change` spec (`{intent, targets}`) + selection trace |

Nothing else written until Atomic Writer commits.

## Notes

- **Parent selection happens INSIDE proposer**, not as a separate
  skill. Upstream's `select_round` is a 30-line numpy helper, and
  promoting it to a skill would force the anti-replay loop to cross
  skill boundaries — breaking the "one iter = one Claude
  conversation" contract. Decision: keep it as a Python helper.
- **`parent_iter_id` in proposal.md frontmatter is load-bearing**:
  loop reads it to do the Workspace Checkout step before invoking
  implementer.
- **`modification` is the verbatim LLM output** — never
  paraphrased/synthesized by us. This is what gets stored in
  `archive.jsonl` and compared on future anti-replay checks. The
  whole anti-replay mechanism collapses if we rewrite this field.
- **Why no Reflexion chain?** Upstream AFlow doesn't have one. Each
  anti-replay attempt is a fresh proposal against a fresh parent;
  the LLM is not asked to critique its own prior attempt. Cross-attempt
  learning comes from the experience injection ("DO NOT REPEAT these
  modifications"), not from chain-of-thought reflection. This is the
  fundamental algorithmic difference between AFlow and ADAS.
- **Why bound the anti-replay loop?** Upstream's `while True` is
  unbounded — for a near-converged archive where the top-K parents
  have exhausted their modification space, it spins forever (in
  practice the LLM eventually generates novel text, but we can't
  rely on that). The soft cap K (default 5) converts the hang into
  an explicit SKIP. Document in `aflow/README.md` deltas table.
- **No archive injection** — unlike ADAS, AFlow does NOT show the
  meta-LLM the full archive. The prompt only includes the SELECTED
  PARENT's slice (its graph, prompt, score, experience, log samples).
  This is much cheaper context-wise and stays faithful to upstream.
- **iter_0 baseline injection**: `modification: "(baseline)"` is a
  sentinel string. `check_modification` skips it (never matches
  against the baseline's own pseudo-modification).
- **Resume**: archive.jsonl is self-contained; on resume, the
  experience map is rebuilt from scratch from archive entries. No
  per-iter state files needed.
- **`score` must be a fraction in [0,1].** `score` is the bootstrap
  median of the iter's `perf_<graph>` primary metric. `select_round`'s
  `score·100` rescale (§ 3a) assumes that range — a primary metric not
  in [0,1] (e.g. a raw distance or step count) breaks the softmax
  temperature. VLN success-style metrics (SR / SPL / OSR) satisfy this;
  if a target graph's primary metric does not, normalize it to [0,1]
  before it is written as `score`.
- **`check_modification` rarely fires in practice.** It is exact (or
  `lower_ws`-normalized) string matching on the LLM's free-form
  `modification` text; two semantically-identical changes almost never
  produce colliding strings. The real anti-replay pressure is the
  experience block injected into the prompt ("DO NOT REPEAT these").
  The string check + bounded `replay_max_retries` faithfully port
  upstream's (equally weak) guard — they are not the load-bearing
  mechanism. `replay_norm: embed` is the only setting that makes the
  check actually bite.
