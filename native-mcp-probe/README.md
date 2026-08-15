# native-mcp-probe — real agent on a real env nodeset over native /mcp

The combination the std board never exercises: a real agent episode driven
**entirely through auto_host's native `/mcp` projection** against the raw
`env_habitat` node toolface — no bridge process, no coding-agent driver/cells,
no curated semantic tools. Standalone by design; nothing here imports from
`coding-agent/`.

```bash
# agentcanvas env; spawns its own env_habitat auto_host (ac-vlnce) and tears it down
python native-mcp-probe/probe.py                 # episode 1, opus-5, R2R-CE rand100
python native-mcp-probe/probe.py --episode 4 --model claude-haiku-4-5
python native-mcp-probe/probe.py --server http://127.0.0.1:9200   # reuse a live server
```

## What talks to what

- **Agent plane (the probe's point)** — the Claude Agent SDK connects with
  `{"type": "http", "url": ".../mcp"}`. The endpoint is narrowed server-side to
  exactly two tools via `--mcp-tools observe_egocentric,step_discrete`, and the
  SDK's `allowed_tools` mirrors that — the model's whole world is
  `observe_egocentric` + `step_discrete`, straight off the node registry.
- **Control plane (experimenter infra, invisible to the agent)** — episode
  placement via `/env-panel/field/{dataset,split,episode_index}` +
  `/env-panel/action/play`, arming via `/call env_habitat__reset`, scoring via
  `/call env_habitat__evaluate`. These are the projection's sibling endpoints
  on the same auto_host — not the bridge.

## Deliberate differences vs the std bare cell (not comparable)

- Raw `step_discrete` takes ONE action per call (the bridge batches a list) —
  one env step per agent turn, so episodes cost more turns.
- Raw `observe_egocentric` returns the full node face: RGB (real image) plus
  pose/intrinsics/instruction in the trailing JSON (ndarrays compacted to
  stubs). The std bare surface exposes RGB only — the raw face leaks more.
- No driver-side step budget: the env's own MAX_EPISODE_STEPS (500) is the
  only cutoff.

Results land in `runs/ep{N}_{stamp}/` (`result.json`, `transcript.jsonl`,
`server.log`). The run dir also records whether the SDK released its exclusive
`/mcp` session on disconnect (env_habitat is `stateful` → single-session
gate) — see `result.json:session_released`.
