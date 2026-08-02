# beta-codex-agent — OpenAI Codex CLI VLN harness

The third harness cell. `beta-coding-agent` measures Anthropic's closed
scaffolding (Claude Agent SDK), `beta-react-harness` the open ReAct loop
(mini-swe-agent); this directory measures **OpenAI's closed scaffolding** by
driving the same habitat env surface through `codex exec --json`.

The bridge is `beta-coding-agent/mcp_bridge.py`, reused **verbatim** (not a
copy): tool names, schemas, descriptions, clearance math, turn-budget
broadcast, and STOP gate are byte-identical to the other two cells. Codex
mounts it as a stdio MCP server via `-c mcp_servers.env.*` config overrides;
episode placement and `env_habitat__evaluate` stay driver-side — the same
ruler as the verified baselines.

## Run recipe

```bash
# 1) env server(s), one per worker (ports 9200+; ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# 2) episodes (agentcanvas env; auth = the logged-in ChatGPT subscription):
python beta-codex-agent/run_episodes.py --episodes 0-9 --split rand100
```

Defaults: `--model gpt-5.5`, `--effort medium` (codex factory default —
"vanilla out of the box"; the driver pins both explicitly so runs don't
inherit `~/.codex/config.toml` tuning). `--bare` and `--skill` mirror the
Claude driver's conditions.

Artifacts land in `outputs/beta-codex-agent/{run}/` in the shared Monitor
layout: `episode_{i}.jsonl` (curated events, same vocabulary as the SDK
logs), `raw/episode_{i}.jsonl` (full codex event dump, image blobs elided),
`raw/episode_{i}.stderr.log`, `live_{i}/` frames, `summary.json`. The
Coding-Agent Monitor browses them via the **Codex CLI** source toggle.

## Codex-specific wiring (hard-won, keep in mind)

- **MCP tool approval**: codex approval-gates MCP tool calls and exec mode
  auto-cancels them ("user cancelled MCP tool call" → the model sees a failed
  call). The driver sets `mcp_servers.env.default_tools_approval_mode =
  "approve"`. v0.142 accepts only `prompt|approve`; the newer docs' `"auto"`
  is silently invalid.
- **Images work natively**: MCP image content reaches the model (verified
  empirically 2026-07-13 with a random-image probe; openai/codex#4819 is
  fixed in 0.142). No view_image workaround needed.
- **Prompt delivery**: codex keeps its built-in system prompt (that closed
  scaffolding is the thing under test); the task briefing rides as the one
  user prompt — the analog of opus-lab's `--builtin-system-prompt` mode.
- `-c project_doc_max_bytes=0` keeps AGENTS.md out of the session (the
  Claude cell's `setting_sources=[]` analog).

## Recorded differences vs the Claude SDK cell

- codex's built-in tools (shell, …) cannot be unmounted; the sandbox stays
  read-only and every `command_execution` is logged as a `shell` tool event
  and counted in `agent.tool_calls.shell`.
- no SDK-level `max_turns`: `--max-turns` only feeds the bridge's budget
  broadcast / STOP gate; hard caps are the env step budget (500) and
  `--episode-timeout` (2400 s).
- auth is the ChatGPT subscription (`codex login`), not per-token API
  billing: `total_cost_usd` stays null; usage tokens are summed from
  `turn.completed` events.
- reasoning: no readable think log. The driver requests
  `model_reasoning_summary="detailed"` and maps any reasoning items to
  `thinking` events, but probes (2026-07-13, codex 0.142 + gpt-5.5) show the
  API returns `summary: []` with encrypted content — reasoning is only
  visible as `reasoning_output_tokens` in usage. (Claude cell: summarized
  thinking; mini cell: whatever litellm exposes.)
