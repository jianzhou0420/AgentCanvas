# beta-react-harness — mini-swe-agent VLN (claude-SDK path, re-harnessed)

Re-runs the `beta-coding-agent` experiment with the harness as the ONLY moved
variable: same `env_habitat` auto_host, same toolset semantics, same prompts,
same driver-side episode control and metrics — but the agent loop is
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s ~100-line
`DefaultAgent` (pip-installed, imported as a library) instead of the Claude
Agent SDK. Nodesets reach the agent through the `NodesetToolSet` concept:
tool schemas + an `execute(name, args)` entry over the auto_host HTTP surface
(`POST /call/{fn}`); `HabitatToolSet` is the first instance, FM nodesets are
later instances of the same wrapper.

## Pieces

| File | Role |
|---|---|
| `toolset.py` | `NodesetToolSet` (generic auto_host→tools wrapper) + `HabitatToolSet` — `mcp_bridge.py` ported verbatim: same descriptions/schemas, clearance readout, turn-budget broadcast, STOP confirmation gate, live artifacts. No MCP subprocess. |
| `model.py` | `NavToolsModel(LitellmModel)` — declares the toolset's schemas instead of BASH_TOOL, parses tool calls into env actions, renders observations as multimodal tool messages. `image_window` = the one declared context knob (see Honest differences). |
| `env.py` | `HabitatEnvironment` — thin session owner: routes tool calls to the toolset; raises `Submitted` when the episode ends (STOP / budget), mini's native exit path. |
| `nav_agent.py` | `DefaultAgent` + curated event stream (same `episode_{i}.jsonl` vocabulary the backend monitor reads) + blob-elided trajectory dumps. |
| `run_episodes.py` | The SDK driver with the session block swapped: env-panel placement, driver-side `env_habitat__evaluate`, per-URL workers, `summary.json` — self-hosted batch eval, mini's swebench runner unused. |
| `check_equivalence.py` | Offline equivalence gate vs the SDK path's own source: tool schemas (byte-for-byte, bare+full), prompts (byte-for-byte modulo jinja's stripped trailing newline), clearance math. |

## Run recipe

1. Habitat auto_host, exactly as the SDK path (see `beta-coding-agent/README.md`):

   ```bash
   REPO=$(git rev-parse --show-toplevel)
   cd "$REPO/agentcanvas/backend"
   PYTHONPATH="$REPO/agentcanvas/backend:$REPO" \
     ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
     --file "$REPO/workspace/nodesets/env/env_habitat.py" \
     --class EnvHabitatNodeSet --port 9200
   ```

2. Equivalence gate (offline, run after any edit to toolset/prompts):

   ```bash
   ~/miniforge3/envs/agentcanvas/bin/python beta-react-harness/check_equivalence.py
   ```

3. Episodes (agentcanvas env; **ANTHROPIC_API_KEY must be set** — see below):

   ```bash
   ~/miniforge3/envs/agentcanvas/bin/python beta-react-harness/run_episodes.py \
     --episodes 0 --split rand100 --model anthropic/claude-sonnet-5
   # bare ① condition: add --bare ; skill condition: --skill opus-nav
   ```

Artifacts: `outputs/beta-react-harness/{run}/` — `episode_{i}.jsonl` (curated
events, same vocabulary as the SDK path), `raw/episode_{i}.traj.json` (mini's
full per-step trajectory, blobs elided), `live_{i}/` (frames + actions.log),
`summary.json` (same shape; `harness: mini-swe-agent` marks the path).

## Honest differences vs the SDK path (the harness variable itself)

- **Billing/auth**: litellm uses the provider API key (the SDK path rode the
  Claude subscription and stripped the key). Costs are per-episode metered
  (`cost_usd` in summary; `--cost-limit` caps).
- **Context management**: the SDK CLI auto-compacts long sessions; mini has
  NONE — a full-length episode with every frame kept can exceed the context
  window. `--image-window K` keeps only the newest K camera frames in the API
  payload (stored trajectory keeps everything). `0` (default) = keep all,
  faithful to mini's linear-history purity.
- **Thinking**: SDK sessions ran adaptive thinking; litellm reasoning params
  are model-family-specific and NOT configured in v1 (plain completion).
- **Prompt trailing newline**: jinja strips the template's final `\n`
  (checked and accepted; semantically nil).

## Env note

`mini-swe-agent==2.4.5` is installed in the `agentcanvas` env. litellm hard-pins
`tokenizers==0.22.2`, which conflicts with `transformers 4.45.2` (needs `<0.21`);
the env keeps `tokenizers 0.20.3` — both import and work for our anthropic-only
usage. If litellm's HF token counting is ever exercised, revisit.
