# agent20 — vanilla coding-agent VLN experiment (AgentCanvas 2.0 seed)

Measures what a stock coding agent (Claude Agent SDK, zero embodied
specialization) scores on R2R-CE by driving the `env_habitat` nodeset through
an MCP bridge. Metrics come from habitat's own measures via
`env_habitat__evaluate` — the same ruler as the graphs in
`workspace/graphs/vln/verified/`.

## Pieces

| File | Role |
|---|---|
| `mcp_bridge.py` | Stdio MCP server; agent-facing toolset (`observe`, `step`) forwarding to a habitat auto_host over `POST /call/{fn}`. One bridge process per agent session per episode. |
| `run_episodes.py` | Driver: places episodes via `/env-panel/*`, runs one clean SDK session per episode, evaluates, writes `outputs/agent20/{run}/summary.json` + per-episode trajectory JSONL. |

The agent sees ONLY `observe()` (egocentric RGB) and `step(actions)`
(0=STOP, 1=fwd 0.25m, 2=left 15°, 3=right 15°). No pose, no depth, no map,
no metrics — episode control and evaluation are driver-side.

## Run recipe

1. Launch the habitat env as an auto_host subprocess (ac-vlnce interpreter,
   Py3.8; leave `AGENTCANVAS_EXECUTOR_URL` unset so event push is a no-op):

   ```bash
   REPO=$(git rev-parse --show-toplevel)
   cd "$REPO/agentcanvas/backend"
   PYTHONPATH="$REPO/agentcanvas/backend:$REPO" \
     ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
     --file "$REPO/workspace/nodesets/env/env_habitat.py" \
     --class EnvHabitatNodeSet --port 9200
   ```

   Port 9200 is outside the JobScheduler pool (8765–8769) and the user
   backend (`:8000`).

2. Drive episodes (agentcanvas env; subscription auth — the driver strips any
   inherited `ANTHROPIC_API_KEY`):

   ```bash
   ~/miniforge3/envs/agentcanvas/bin/python \
     scripts/agent20/run_episodes.py --episodes 0 --split rand100
   ```

`rand100` is the paper-canonical 100-episode subset of R2R-CE val_unseen
(identical to OpenNav/SmartWay's set). Scale plan: 1-episode smoke →
10-episode calibration (token/latency/subscription burn) → rand100.

## Watching live

Three real-time channels per run (paths under `outputs/agent20/{run}/`):

```bash
# 1. What the agent is thinking/doing — full trajectory, flushed per event
tail -f outputs/agent20/{run}/episode_{i}.jsonl | jq -r '"\(.t)s \(.kind) \(.input // .text // .texts // "" | tostring | .[0:120])"'

# 2. What the agent sees — every observe frame, plus overwritten latest.png
watch -n1 'tail -3 outputs/agent20/{run}/live_{i}/actions.log; ls outputs/agent20/{run}/live_{i} | tail -3'
# latest.png can be opened in any auto-reloading image viewer / VS Code tab

# 3. Heartbeat — the habitat server's access log (observe/step rhythm)
tail -f <auto_host stdout>
```

## Deliberate v1 choices

- `tools=[]` in the SDK options — no built-in tools (Bash/filesystem). Pure
  ReAct over the env. The full coding-agent toolset (filesystem-as-memory,
  notes) is a later ablation, not v1.
- Observation is egocentric RGB only, YAML-default resolution. Panorama /
  candidate annotation would inject embodied priors — out of scope for the
  "vanilla" claim.
- The agent must issue STOP itself (success within 3 m requires it); the
  driver never force-stops, so a non-stopping agent honestly scores its
  distance-to-goal at budget exhaustion.
