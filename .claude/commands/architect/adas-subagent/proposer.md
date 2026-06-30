# Propose a new design — 3-call Reflexion

> **Required reading**:
> - `adas-subagent/README.md` § "Three structural contracts"
> - `_common/files-contract.md` § "Edit whitelist"
> - Upstream reference (read once):
>   - `third_party/ADAS/_mmlu/search.py:180–203`
>   - `third_party/ADAS/_mmlu/mmlu_prompt.py:226–528`
>     (especially `base`, `Reflexion_prompt_1`, `Reflexion_prompt_2`,
>     `get_prompt`, `get_reflexion_prompt`)

This skill is invoked **inside the loop's Claude conversation** —
not as a separate sub-agent. The 3 LLM calls share Claude's
`msg_list` so the Reflexion chain is preserved.

## Arguments

```
/architect:adas-subagent:proposer [<graph> [<version> [<iter>]]]
                           [--graph <name>] [--version <N>] [--iter <M>]
```

Iter resolution: `loop` passes the next iter index (`iter_n`); manual
invocation defaults to "the iter to be created" (max(M)+1).

## Pre-conditions

- `archive.jsonl` exists at `outputs/design_runs/adas-subagent/{graph}/v{N}/` with
  at least one entry (the baseline). If empty: this skill ERR — loop
  must run pre-seed first.
- `workspace/graphs/{graph}.json` reflects the current archive head's
  state (loop has not let any failed implementer leave the workspace
  dirty).
- `.staging/iter_n/` does **not** exist yet (or is empty). This skill
  creates it.

## Steps

### 1. Resolve + read archive

Apply resolve protocol. Print:

```
[adas-subagent:proposer] iter=iter_{n}  archive=v{N}/archive.jsonl ({K} entries)
```

Read the full archive: `archive = [json.loads(line) for line in
open(archive_path)]`. The full list will be injected into the prompt
verbatim.

### 2. Analyze helper (Python, in-skill)

Build a compact summary of the archive head's behavior — this is the
adas v1 `analyze` step folded into proposer (decided in
`algorithm.html` open question Q7).

- Read `archive[-1]`'s `iter_id`, then read
  `outputs/.../iter_{archive[-1].iter_id}/`:
  - `metrics.json` for primary/secondary metric values
  - `summary.csv` for per-episode rows
  - `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/log.jsonl` for per-episode
    trajectory logs (downsample if > 1 MB)
- Generate a textual `analysis_summary` describing dominant failure
  modes (e.g., "82/100 fail at long-instruction episodes; instruction
  length > 35 tokens correlates with 4× higher stop-too-early rate;
  history truncation kicks in at step 12 and seems to mislead
  navigator").

This summary is **not** written to disk yet — it's inserted into the
proposer prompt body. (It becomes part of `proposal.md`'s
"What changed" / "Why" sections after the LLM uses it.)

### 3. Build the meta-LLM base prompt

Compose verbatim from upstream `mmlu_prompt.py:226 base`, with these
ADAPTED placeholders:

- `[ARCHIVE]` → `json.dumps(archive)` (the full archive list)
- `[EXAMPLE]` → one minimal example entry, see below

**Domain adaptation** (replace MMLU-specific text):

| Upstream slot | adas-subagent substitution |
|---|---|
| "MMLU benchmark" | "VLN / EQA / VLA benchmark (graph's profile yaml declares which)" |
| "An example question from MMLU: ..." | "An example episode from {graph}.yaml's split" |
| "# The utility code: ... LLMAgentBase / Info / FORMAT_INST" | "# The AgentCanvas graph model: brief glossary of node / wire / port / iter-in/out (5 lines max)" |
| "forward(self, taskInfo)" | "graph topology + nodeset Python files" |
| `[ARCHIVE]` injection | Same — `json.dumps(archive)` verbatim |
| Output instruction: `thought`, `name`, `code` | `thought`, `name`, `patch` (see § 5 patch schema) |
| WRONG implementation examples | Adapted to graph-edit failure modes: wiring mismatched ports, editing server-mode nodesets, forgetting to bump max_steps after adding loop nodes, etc. |

The "Your task" final paragraph stays verbatim:
> "You are deeply familiar with LLM prompting techniques and LLM agent
> works from the literature. Your goal is to maximize 'fitness' by
> proposing interestingly new agents. Observe the discovered
> architectures carefully and think about what insights, lessons, or
> stepping stones can be learned from them. ... THINK OUTSIDE THE BOX."

### 3.5. Inject the llmCall config-schema cheat sheet

Insert a fixed section immediately after the upstream "Your task"
paragraph and before "# Recent behavior summary" (step 4). Purpose:
tell the meta-LLM exactly which `llmCall.config` keys the runtime
reads, and which it ignores.

Without this, the meta-LLM falls back on names it learned from ADAS
upstream (e.g. `model: "gpt-4o-mini"`) and writes dead config fields
— this is the failure mode that produced the iter_1 focus_llm bug
(`config.model` is never read; only `config.profile` is).

The cheat sheet does NOT enumerate backend profiles or advise a model:
every `llmCall` node's `profile` / `temperature` is pinned
deterministically post-edit by `_common/lib/pin_llm_profile.py`
(implementer step 3a-pin), so the proposer needs no profile guidance.

**Generation step** — call the renderer (single function):

```python
import sys
sys.path.insert(0, ".claude/commands/architect/adas-subagent/lib")
from helpers import render_backend_llm_cheat_sheet

cheat_sheet = render_backend_llm_cheat_sheet()
# Inject `cheat_sheet` into base_prompt at the position described below.
```

`render_backend_llm_cheat_sheet` (in `lib/helpers.py`) returns a fixed
~15-line Markdown block — the llmCall config schema. It takes no
arguments and reads nothing; the schema is fixed by the runtime.

**Cheat sheet** (the renderer returns exactly this block):

```markdown
## llmCall config schema — ONLY these keys are read by the runtime

| Key | Type | Meaning |
|---|---|---|
| `profile` | str | Backend LLM profile to route to |
| `temperature` | float | Sampling temperature |
| `max_tokens` | int | Output token cap |
| `system_prompt` | str | System message |
| `template` | str | Prompt body with `{port}` placeholders |
| `mode` | "single_turn" \| "conversation" | Single-shot or multi-turn |
| `n` | int | Number of samples (use n>1 for self-consistency patterns) |
| `stop` | str \| list[str] | Stop sequences |

**The `model` field is NOT read by the runtime.** Writing
`"model": "gpt-4o-mini"` does nothing — the call still routes to the
active profile. To route to a specific profile use
`"profile": "gpt-4o-mini"` instead.
```

It costs ~15 lines of prompt context but prevents an entire class of
patch-time bugs (dead config fields).

### 4. Insert the analysis summary

After the `[ARCHIVE]` injection and before "# Output Instruction and
Example", add a new section:

```
# Recent behavior summary

{analysis_summary from step 2}
```

This gives the meta-LLM concrete failure signals beyond the abstract
fitness number.

### 5. Patch schema (the `patch` field)

Replaces upstream's `code` field — but as a **change spec**, not a
typed op list. The LLM must return a JSON object with:

```json
{
  "thought": "**Insights:** ... **Overall Idea:** ... **Implementation:** ...",
  "name": "Self-Audit Chain-of-Thought via History Tracker",
  "patch": {
    "intent": "Prose: exactly what to change and why — specific enough that the implementer can realize it by editing files. e.g. 'Add a HistoryTracker llmCall node that summarizes the last 5 steps; wire its summary port into navgpt__think.context; raise terminationCondition max_steps from 5 to 15.' Describe the change — do NOT write op lists, jsonpaths, or anchors.",
    "targets": ["workspace/graphs/{graph}.json", "workspace/nodesets/<prefix>.py"]
  }
}
```

`patch.intent` is prose; `patch.targets` lists the workspace-prefixed
files the change is expected to touch. `patch` MAY be `null` for a
no-change probe. The implementer spawns a sub-agent that reads
`intent` and edits the seeded `targets` with native Edit/Write (see
`_common/implementer.md`) — there is no typed `graph_edits` DSL since
2026-05-20: an agentic implementer edits files directly, so an op enum
for a deterministic replayer was redundant.

Edit whitelist (files-contract § 7) is enforced by the **implementer**
via `_common/lib/overlay.py`, not here. The proposer's `intent` may
propose edits to any file under `workspace/{graphs,nodesets}/`.

### 6. Three sub-agent invocations (one msg_list)

This is the **load-bearing** part. Each "LLM call" in upstream
`search.py:188 / 194 / 198` becomes one `Agent({subagent_type:
"general-purpose", ...})` invocation — a **fully tool-augmented
Claude sub-agent**, not a stateless API call. Three properties that
together secure the structural anchors of upstream's 3-call Reflexion
within the modernized reasoning module paradigm:

1. **Independent sampling diversity** — each `Agent(...)` spawn is
   an independent Claude conversation with its own sampling. R0, R1,
   R2 are three independent samples, not one autoregressive trace.
   This is the property a single Claude thinking-pass cannot
   provide; spawning is the only mechanism that gives it.
2. **Tool-augmented meta-LLM** — each sub-agent has full tool
   access (Read, Grep, Bash, WebSearch, even nested Agent). It can
   ground its proposal in the archive entries, the current graph,
   `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/log.jsonl`, source code under
   `workspace/nodesets/`, etc. This is what the "coding-agent
   modernization" framing actually delivers — without tool access,
   sub-agents collapse back to stateless LLM calls.
3. **Verbatim R1 / R2 prompt text** — the Reflexion prompts below
   are copied character-for-character from upstream
   `mmlu_prompt.py:497–528`. The only adaptation is `"code"` →
   `"patch"` in the response-schema description.

**Sub-agent contract** (applies to all 3 calls):

- `subagent_type: "general-purpose"` — has all tools.
- The sub-agent is FREE to use any tool to ground its proposal.
- It MUST emit a **single fenced ```json block** as the final
  visible chunk of its return message. Proposer parses
  `parse_final_json(resp)` by extracting the LAST fenced
  ```json``` block.
- It MUST NOT edit `workspace/*` directly — patch application is
  implementer's job. The sub-agent's deliverable is patch JSON only.
- It MAY use `Agent(...)` recursively (nested sub-agents) — useful
  when grounding requires expensive grep/read passes.

**msg_list serialization**: each sub-agent receives the full
accumulating `msg_list` rendered as text inside its prompt. The
Agent tool only accepts a single user-string prompt, so we can't
natively inject prior `role: assistant` turns; rendering as text is
the closest semantic equivalent. Format:

```
=== msg_list so far (treat each [role] block as that conversation turn) ===

[system]
{system_prompt verbatim}

[user]
{base_prompt_filled verbatim — base prompt + [ARCHIVE] + Recent behavior summary}

[assistant]   ← R0's response, present from R1 onward
{json.dumps(next_solution_r0)}

[user]   ← Reflexion_prompt_1, present from R1 onward
{reflexion_1 verbatim}

[assistant]   ← R1's response, present from R2 onward
{json.dumps(next_solution_r1)}

[user]   ← Reflexion_prompt_2, present at R2 only
{reflexion_2 verbatim}

=== end msg_list ===
```

#### Call #1 — propose

```python
msg_list = [
    {"role": "system", "content": system_prompt},          # "You are a helpful assistant. ... WELL-FORMED JSON object."
    {"role": "user",   "content": base_prompt_filled},     # from step 3+4
]

resp_0 = Agent({
    "subagent_type": "general-purpose",
    "description": f"adas-subagent propose iter_{n}",
    "prompt": (
        "You are the meta-LLM in an ADAS-style architecture search "
        "loop. Your task: propose a new AgentCanvas graph + nodeset "
        "patch as the next agent design. You have full tool access "
        "(Read / Grep / Bash / WebSearch / Agent). Use whatever you "
        "need to ground your proposal in the archive, the current "
        "graph state, eval logs under outputs/eval_runs/, or related "
        "literature.\n\n"
        + render_msg_list(msg_list) +
        "\n\nReturn a SINGLE fenced ```json block as your final "
        'message containing an object with keys {"thought", "name", '
        '"patch"}. Do NOT use Edit/Write on workspace/*; the patch '
        "JSON is the only deliverable."
    ),
})
next_solution = parse_final_json(resp_0)
msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
```

#### Call #2 — Reflexion_prompt_1

`get_reflexion_prompt` verbatim from upstream
`mmlu_prompt.py:544–547`:

```python
prev_example = archive[-1] if generation_index > 0 else None
prev_example_str = (
    f"Here is the previous agent you tried:\n{json.dumps(prev_example)}\n\n"
    if prev_example else ""
)
reflexion_1 = REFLEXION_PROMPT_1.replace("[EXAMPLE]", prev_example_str)

msg_list.append({"role": "user", "content": reflexion_1})

resp_1 = Agent({
    "subagent_type": "general-purpose",
    "description": f"adas-subagent reflexion-1 iter_{n}",
    "prompt": (
        "You are the meta-LLM continuing an ADAS-style architecture "
        "search loop. The conversation so far is rendered below; the "
        "final [user] block is your current task.\n\n"
        + render_msg_list(msg_list) +
        '\n\nReturn a SINGLE fenced ```json block with keys '
        '{"reflection", "thought", "name", "patch"} per the '
        "Reflexion_prompt_1 instructions above. Do NOT edit "
        "workspace/* directly."
    ),
})
next_solution = parse_final_json(resp_1)
msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
```

`REFLEXION_PROMPT_1` — **VERBATIM from upstream `mmlu_prompt.py:497–523`**
(only `"code"` → `"patch"` in the last field description):

```
"[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings.
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"patch": Provide the corrected patch or an improved implementation. Make sure you actually implement your fix and improvement in this patch.
```

(Upstream's `"code"` field is renamed to `"patch"` — the only
character-level adaptation.)

#### Call #3 — Reflexion_prompt_2

```python
msg_list.append({"role": "user", "content": REFLEXION_PROMPT_2})

resp_2 = Agent({
    "subagent_type": "general-purpose",
    "description": f"adas-subagent reflexion-2 iter_{n}",
    "prompt": (
        "You are the meta-LLM continuing an ADAS-style architecture "
        "search loop. The conversation so far is rendered below; the "
        "final [user] block is your current task.\n\n"
        + render_msg_list(msg_list) +
        '\n\nReturn a SINGLE fenced ```json block with keys '
        '{"reflection", "thought", "name", "patch"} per the '
        "Reflexion_prompt_2 instructions above. Do NOT edit "
        "workspace/* directly."
    ),
})
next_solution = parse_final_json(resp_2)
msg_list.append({"role": "assistant", "content": json.dumps(next_solution)})
```

`REFLEXION_PROMPT_2` — **VERBATIM from upstream `mmlu_prompt.py:525–528`**
(only `"code"` → `"patch"`):

```
Using the tips in "## WRONG Implementation examples" section, revise the patch further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the patch in "patch".
```

**The R2 sub-agent's `patch` is what implementer applies.** R0/R1
outputs are preserved in `proposal.md` for forensics but not
applied.

### 7. Exception handling

Any sub-agent invocation fails — Agent tool error, sub-agent
refuses, sub-agent's return message has no parseable fenced
```json block, or `parse_final_json` raises on malformed JSON →

```
print("[adas-subagent:proposer] sub-agent exception:", e)
return status=SKIP_LLM_EXCEPTION
```

This bubbles up to loop, which: `rm -rf .staging/iter_{n}/`,
`consecutive_skips += 1`, continue to n+1. Verbatim upstream
semantics (`search.py:199–203`: `n -= 1; continue`).

### 8. Write proposal.md to staging

On success (3 calls completed, final `next_solution` parses):

```bash
mkdir -p outputs/design_runs/adas-subagent/{graph}/v{N}/.staging/iter_{n}/
```

Write `.staging/iter_n/proposal.md`:

```markdown
---
generation: {n}
iter_id: iter_{n}
name: <from next_solution.name>
parent: iter_{prev}
---

# Thought

<next_solution.thought verbatim>

# What changed (diff narrative)

<proposer writes a human-readable summary of the intended change>

# Change

```json
{patch from next_solution}
```

# Reflexion chain

<full msg_list serialized for debugging>
```

The `# Change` section holds `next_solution.patch` — the
`{intent, targets}` change spec the implementer consumes (Step 1 of
`_common/implementer.md`).

(The `reflection` field from R1/R2 outputs is preserved in the
`Reflexion chain` section for human review but **not** included in
`name` / `thought` / `patch` going forward. Upstream strips
`reflection` before archive append; we keep it in `proposal.md` for
forensics, but the Atomic Writer's archive-append step re-strips it
before writing the archive entry. Implementer retry sub-agents return
`{edit_summary, extra_targets}` — they edit natively, not a patch.)

### 9. Return to loop

Print:

```
[adas-subagent:proposer] OK — proposed "<name>"
                 reflexion calls = 3
                 targets         = {count}
                 staging         = .staging/iter_{n}/proposal.md
```

Return `status=OK` to loop.

## Outputs

| Path | What |
|------|------|
| `v{N}/.staging/iter_{n}/proposal.md` | Frontmatter + thought + diff narrative + `# Change` spec (`{intent, targets}`) + Reflexion chain |

Nothing else written until Atomic Writer commits.

## Notes

- **Sub-agent invocation replaces "pinned LLM params"**. Upstream's
  `temperature=0.8 / response_format=json_object / max_tokens=4096 /
  model=gpt-4o-2024-05-13` no longer applies — those are
  stateless-API knobs, and we no longer hit a stateless API. The
  structural properties they secured (independent sampling + JSON
  output) are now secured by:
  (a) each `Agent(...)` spawn = an independent Claude sample, so 3
  spawns = 3 independent samples just like 3 upstream API calls;
  (b) the sub-agent prompt explicitly demands a single fenced
  ```json block, parsed by `parse_final_json`.
- **Adaptation: `role: assistant` becomes quoted text in sub-agent
  prompts.** Agent tool accepts only one user-string prompt; we
  can't natively inject prior `role: assistant` turns. Rendering the
  full msg_list as text inside the user prompt is the closest
  semantic equivalent. This loses the exact role-tagging upstream
  has but preserves all textual content; sub-agents still reason
  over prior outputs.
- **archive injection is full** — every entry's `thought`, `name`,
  `graph_summary`, `diff_narrative`, `fitness` enters the prompt
  verbatim. We exclude `iter_id` from injection (debug-only). After
  ~30 generations, prompt size may need a sliding-window prune — but
  defer until measured.
- **R1's `archive[-1]` echo is separate** from `[ARCHIVE]` injection
  (verbatim upstream behavior). The most-recent entry is shown twice
  — once in the archive list, once as "Here is the previous agent you
  tried". Redundant but preserves upstream's emphasis.
- **`prev_example = None` when generation_index == 0** — verbatim
  upstream behavior; on the first evolved iter (iter_1) we don't echo
  anything as "previous", just rely on archive injection. (iter_0 is
  baseline, not evolved.)
- **No revert / rollback bookkeeping** — if the proposed patch is bad,
  implementer's debug retry handles it; if all retries fail, SKIP and
  the workspace stays at archive head.
