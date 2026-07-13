# beta-react-harness — M1 report

*Branch `dev/coding-agent` · generated 2026-07-13 · experiment data in
`outputs/archive/react-harness-m1-2026-07-13/` and
`outputs/archive/coding-agent-sdk-baselines-2026-07-13/`*

This document covers (1) what this push adds, (2) how the code works, and
(3) the experiment record so far — the experiment section is data only, no
analysis, so you can read the numbers fresh and build on them.

## 1. What this push adds

| commit | what |
|---|---|
| `106b792` | **`beta-react-harness/`** — the mini-swe-agent harness itself (this directory) |
| `83b9b65` | backend: Coding-Agent Monitor read endpoints take `source=claude-sdk\|mini-swe` |
| `4dcb056` | frontend: Monitor → Logs gains a **[Claude SDK \| mini-swe-agent]** source toggle |
| `5e4e226` | `beta-coding-agent/opus-lab/`: API capture proxy + session-verify utilities |
| `3f6e813` `09804a4` `3f2af02` | archived mini-harness trajectories (sonnet-5 & opus-4-8, eps 0–49 each) |
| `1c90ad8` | archived the 19 claude-SDK baseline trajectories used for comparison |

Motivation in one line: `beta-coding-agent` showed a stock coding agent can
drive VLN through the Claude Agent SDK; this push re-runs the same experiment
with the **harness as the only moved variable** — a ~500-line open ReAct loop
(mini-swe-agent) replaces the SDK — so agent capability can be measured
independently of Anthropic's closed scaffolding, on any litellm-served model,
with trajectories in a training-friendly format.

## 2. How the code works

### 2.1 Shape

mini-swe-agent's trichotomy is kept intact — the loop is upstream's, unmodified:

```
Agent (mini DefaultAgent)          while True: query → execute; exceptions carry exit messages
  │  messages (linear, append-only; trajectory == LLM input)
  ├─ Model   NavToolsModel         litellm + our tool schemas; parses tool_calls;
  │                                renders multimodal tool results (image + status JSON)
  └─ Env     HabitatEnvironment    thin session owner; routes tool calls; raises
                                   Submitted when the episode ends
                     │ execute(name, args)
                 NodesetToolSet    generic wrapper: POST /call/{fn} on an auto_host
                     │
                 HabitatToolSet    observe / step / look_around — mcp_bridge.py port
                     │ HTTP
                 env_habitat auto_host  (same server the SDK path uses)
```

### 2.2 Files

- **`toolset.py`** — `NodesetToolSet` turns a running auto_host nodeset into
  (a) tool schemas and (b) an `execute(name, args)` entry; adding another
  nodeset later = another subclass + a whitelist entry. `HabitatToolSet` is
  the SDK bridge (`beta-coding-agent/mcp_bridge.py`) ported verbatim: same
  tool descriptions/schemas, clearance readout from depth, turn-budget
  broadcast, STOP confirmation gate, `live_{i}/` frame dumps. Per-episode
  state lives on the instance (one toolset per episode).
- **`model.py`** — `NavToolsModel(LitellmModel)`: declares the toolset's
  schemas instead of mini's BASH_TOOL; `_parse_actions` maps `tool_calls` →
  `{tool, args, tool_call_id}`; `format_observation_messages` emits
  `role:"tool"` messages whose content is OpenAI parts (`image_url` +
  status-JSON text) — litellm converts per provider. Two deliberate knobs:
  `image_window` (keep only newest K frames in the API payload; **0 = off**,
  full history — used for all runs here) and a **multipart-safe
  `cache_control`** reimplementation (upstream mini 2.4.5's
  `set_cache_control` asserts single-part content and crashes on image
  parts; ours keeps the same default_end semantics).
- **`env.py`** — `HabitatEnvironment`: holds the toolset, routes parsed tool
  calls, and raises `Submitted` (mini's native exit) when a call ends the
  episode (STOP executed / step budget exhausted). Episode placement, reset,
  and metric collection stay driver-side; the agent never sees SR/SPL/pose.
- **`nav_agent.py`** — `DefaultAgent` plus a curated event stream
  (`episode_{i}.jsonl`, same event vocabulary as the SDK logs, so the
  Monitor renders both) and base64-elided trajectory dumps (`raw/`).
- **`run_episodes.py`** — the SDK driver's skeleton with the session block
  swapped for `agent.run()`: env-panel episode placement, one fresh
  agent+model+env per episode, driver-side `env_habitat__evaluate`, one
  asyncio worker per `--server-urls` entry, `summary.json` flushed per
  episode. Auth is an API key via litellm (`ANTHROPIC_API_KEY`), unlike the
  SDK path's subscription auth.
- **`check_equivalence.py`** — the gate that makes "harness is the only
  variable" checkable: tool schemas byte-identical to the MCP bridge
  (introspected in-process, bare + full variants), prompts byte-identical
  modulo jinja's stripped trailing newline, clearance math identical.
  Run it after touching toolset/prompts.

### 2.3 Monitor integration

Both drivers write the same artifact layout, so the existing Coding-Agent
Monitor serves both: backend read endpoints (`/api/coding-agent/runs*`)
take `source=claude-sdk|mini-swe` mapped to the two output roots; the page's
Logs mode has a source toggle. mini-only event kinds (`exit`, `user_text`,
`driver_error`) render natively.

### 2.4 Recorded differences vs the SDK path

- billing: litellm API key vs SDK subscription auth;
- context: no SDK-side context management — full linear history each call
  (prefix-cached; `image_window` off in all runs here);
- thinking: not configured on the mini path (plain completion);
- **turn budget: SDK runs used `max_turns=80`; mini runs used
  `step_limit=100`** (see §3.5 for how often the extra budget was used).

## 3. Experiment record (data only)

### 3.1 Setup

R2R-CE `rand100` (100-episode paper-canonical subset of val_unseen);
habitat-sim via `env_habitat` auto_host, one server per parallel worker.
Metrics are habitat's own (`env_habitat__evaluate`): success = STOP within
3 m; SPL; nDTW; oracle_success; steps. Agent-facing surface identical in
both harnesses (observe / step / look_around + clearance + budget broadcast
+ STOP gate; instruction embedded in the system prompt; no pose/map/metrics
exposed). All mini runs: full mechanisms, **no skill text**; env step budget
500; $5/episode cost cap; 2400 s wall clock. near-miss below = failures with
3.0 m ≤ distance_to_goal ≤ 5.0 m at stop.

### 3.2 mini-harness runs

| run | model | eps | n | SR | SPL | nDTW | oracle | near-miss | stop | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke2` | sonnet-5 | 0 | 1 | 0.00 | 0.00 | 0.74 | 0.00 | 1 | 1.00 | $0.18 |
| `calib10` | sonnet-5 | 0–9 | 10 | 0.20 | 0.14 | 0.50 | 0.30 | 3 | 0.90 | $2.66 |
| `sonnet10_49` | sonnet-5 | 10–49 | 40 | 0.50 | 0.35 | 0.45 | 0.55 | 3 | 0.95 | $14.41 |
| `opus10` | opus-4-8 | 0–9 | 10 | 0.20 | 0.17 | 0.61 | 0.40 | 5 | 1.00 | $6.17 |
| `opus10_19` | opus-4-8 | 10–19 | 10 | 0.80 | 0.59 | 0.59 | 0.90 | 0 | 1.00 | $7.84 |
| `opus20_49` | opus-4-8 | 20–49 | 30 | 0.37 | 0.33 | 0.51 | 0.47 | 5 | 1.00 | $13.26 |
| **opus 0–49 combined** |  | 0–49 | 50 | 0.42 | 0.35 | 0.55 | 0.54 | 10 | 1.00 | $27.27 |
| **sonnet 0–49 combined** |  | 0–49 | 50 | 0.44 | 0.31 | 0.46 | 0.50 | 6 | 0.94 | $17.08 |

### 3.3 claude-SDK baselines

| run | model | condition | eps | n | SR | SPL | nDTW | near-miss | stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sonnet100` | sonnet-5 | mechanisms, no skill | 0–99 | 100 | 0.19 | 0.16 | 0.49 | 17 | 0.35 |
| `opus50_bare` | opus-4-8 | bare | 0–49 | 50 | 0.48 | 0.46 | 0.59 | 5 | 0.72 |
| `opus50_ledger` | opus-4-8 | +ledger-nav | 0–49 | 50 | 0.50 | 0.43 | 0.59 | 8 | 1.00 |
| `gen50_ledger` | sonnet-5 | +ledger-nav | 0–49 | 50 | 0.44 | 0.30 | 0.46 | 5 | 1.00 |
| `fable50_bare` | fable-5 | bare | 0–49 | 50 | 0.64 | 0.57 | 0.58 | 4 | 0.78 |
| `fable25_ledger` | fable-5 | +ledger-nav | 0–24 | 25 | 0.68 | 0.50 | 0.57 | 4 | 1.00 |
| `fable25_49_ledger` | fable-5 | +ledger-nav | 25–49 | 25 | 0.56 | 0.42 | 0.57 | 3 | 1.00 |
| `haiku50_ledger` | haiku-4-5 | +ledger-nav | 0–49 | 50 | 0.20 | 0.12 | 0.19 | 3 | 0.78 |
| `calib10c` | (default) | mechanisms, no skill | 0–9 | 10 | 0.30 | 0.30 | 0.53 | 4 | 0.50 |
| `tune1_budget` | sonnet-5 | +ledger-nav iter1 | 0–9 | 10 | 0.10 | 0.10 | 0.45 | 3 | 1.00 |
| `tune2_pano` | sonnet-5 | +ledger-nav iter2 | 0–9 | 10 | 0.20 | 0.17 | 0.46 | 3 | 1.00 |
| `tune3_clearance` | sonnet-5 | +ledger-nav iter3 | 0–9 | 10 | 0.30 | 0.21 | 0.40 | 2 | 1.00 |
| `tune4_placement` | sonnet-5 | +ledger-nav iter4 | 0–9 | 10 | 0.30 | 0.22 | 0.45 | 3 | 1.00 |
| `tune5_stopgate` | sonnet-5 | +ledger-nav iter5 | 0–9 | 10 | 0.40 | 0.23 | 0.38 | 2 | 1.00 |
| `opus_skill_v1` | opus-4-8 | +opus-nav | 0–9 | 10 | 0.30 | 0.22 | 0.48 | 3 | 1.00 |
| `opus_skill_v2` | opus-4-8 | +opus-nav | 0–9 | 10 | 0.30 | 0.30 | 0.53 | 1 | 1.00 |
| `opus_skill_v3` | opus-4-8 | +opus-nav | 0–9 | 10 | 0.10 | 0.10 | 0.55 | 4 | 0.20 |
| `opus_skill_v4` | opus-4-8 | +opus-nav | 0–9 | 10 | 0.30 | 0.28 | 0.52 | 3 | 1.00 |

### 3.4 Same-episode-subset comparison

| condition | n | SR | SPL | nDTW | oracle | near-miss |
| --- | --- | --- | --- | --- | --- | --- |
| **eps 0–9** |  |  |  |  |  |  |
| mini opus | 10 | 0.20 | 0.17 | 0.61 | 0.40 | 5 |
| SDK opus bare | 10 | 0.30 | 0.28 | 0.44 | 0.40 | 0 |
| SDK opus +ledger | 10 | 0.20 | 0.20 | 0.52 | 0.40 | 5 |
| mini sonnet | 10 | 0.20 | 0.14 | 0.50 | 0.30 | 3 |
| SDK sonnet no-skill | 10 | 0.00 | 0.00 | 0.43 | 0.20 | 3 |
| **eps 10–19** |  |  |  |  |  |  |
| mini opus | 10 | 0.80 | 0.59 | 0.59 | 0.90 | 0 |
| SDK opus bare | 10 | 0.60 | 0.57 | 0.69 | 0.80 | 0 |
| SDK opus +ledger | 10 | 0.50 | 0.43 | 0.61 | 0.70 | 1 |
| mini sonnet | 10 | 0.30 | 0.24 | 0.37 | 0.50 | 1 |
| SDK sonnet no-skill | 10 | 0.20 | 0.11 | 0.45 | 0.40 | 0 |
| **eps 20–49** |  |  |  |  |  |  |
| mini opus | 30 | 0.37 | 0.33 | 0.51 | 0.47 | 5 |
| SDK opus bare | 30 | 0.50 | 0.48 | 0.60 | 0.57 | 5 |
| SDK opus +ledger | 30 | 0.60 | 0.51 | 0.60 | 0.60 | 2 |
| mini sonnet | 30 | 0.57 | 0.39 | 0.47 | 0.57 | 2 |
| SDK sonnet no-skill | 30 | 0.23 | 0.21 | 0.51 | 0.37 | 5 |
| **eps 0–49** |  |  |  |  |  |  |
| mini opus | 50 | 0.42 | 0.35 | 0.55 | 0.54 | 10 |
| SDK opus bare | 50 | 0.48 | 0.46 | 0.59 | 0.58 | 5 |
| SDK opus +ledger | 50 | 0.50 | 0.43 | 0.59 | 0.58 | 8 |
| mini sonnet | 50 | 0.44 | 0.31 | 0.46 | 0.50 | 6 |
| SDK sonnet no-skill | 50 | 0.18 | 0.15 | 0.49 | 0.34 | 8 |

### 3.5 Paired per-episode success counts (identical episode indices)

| pair (first vs second) | n | both | first-only | second-only | neither | exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| mini opus vs SDK opus bare | 50 | 14 | 7 | 10 | 19 | 0.6291 |
| mini opus vs SDK opus +ledger | 50 | 14 | 7 | 11 | 18 | 0.4807 |
| mini sonnet vs SDK sonnet no-skill | 50 | 7 | 15 | 2 | 26 | 0.0023 |

Turn-budget usage under the config difference (§2.4): mini episodes using
more than 80 LLM calls — sonnet 20/50 (7 of them successful), opus 11/50
(4 successful).

### 3.6 Reproducing a mini run

```bash
# 1) env server(s), one per worker (ports 9200+; ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# 2) equivalence gate (offline; run after touching toolset/prompts):
python beta-react-harness/check_equivalence.py

# 3) episodes (ANTHROPIC_API_KEY required; litellm billing):
python beta-react-harness/run_episodes.py --episodes 0-9 --split rand100 \
  --model claude-sonnet-5 --server-urls http://127.0.0.1:9200
```

Per-episode trajectories: `outputs/beta-react-harness/{run}/episode_{i}.jsonl`
(curated; rendered by the Monitor) and `raw/episode_{i}.traj.json` (full
message history, blobs elided). Archived copies of everything referenced
here are under `outputs/archive/`.
