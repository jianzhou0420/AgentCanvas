# opus-lab — clean-room for the opus navigation-skill hill-climb

A self-contained sibling of `beta-coding-agent/`, stripped to the minimum so the
skill is the **only** moving part. Nothing here toggles env mechanisms.

## Why a separate lab

The shared `beta-coding-agent/mcp_bridge.py` grew four tuned mechanisms
(`look_around`, depth `clearance_m`, turn-budget broadcast, STOP-confirmation
gate) behind a single `HABITAT_BARE` flag, and its driver's `--bare` conflated
two independent things: *which env mechanisms exist* and *whether the agent gets
a skill* (bare ⇒ no skill; skill ⇒ all mechanisms). You could never run
bare-tools **+** skill — which is exactly the condition an honest hill-climb
needs: hold the env fixed at the bare pair, vary only the skill.

opus-lab fixes that by construction:

- **`bridge.py`** exposes exactly `observe()` (RGB only) and `step()` — no
  clearance, no panorama, no budget broadcast, no STOP gate. There is no flag to
  turn any of that on, because none of it exists here.
- **`driver.py`** always runs the bare tools; `--skill NAME` optionally layers a
  skill onto the system prompt. No `--skill` = the pure bare baseline.

The metric ruler is unchanged: episodes are placed and scored through the same
`env_habitat` auto_host surface as the verified baselines (`evaluate` →
habitat's own SR/SPL). The agent never sees pose, depth, reward, or metrics.

## Layout

```
opus-lab/
  bridge.py                 minimal MCP: observe() + step()
  driver.py                 Agent SDK driver; bare tools + optional skill
  skills/opus-nav/SKILL.md  the skill under search (clean seed; iterate here)
```

Runs land in the shared `outputs/beta-coding-agent/{run_name}/` (run names are
prefixed `opuslab_`), so the Coding-Agent Monitor tab browses them alongside the
old driver's runs. Artifacts per run: `episode_{i}.jsonl` (curated trajectory +
final metrics), `raw/episode_{i}.jsonl` (full SDK dump, with `--raw-log`),
`summary.json`, `live_{i}/` (spectator frames).

## Running

Prereq: an `env_habitat` auto_host must already be up on the URL you pass
(default `http://127.0.0.1:9200`). Run in the `agentcanvas` conda env. Strip
`ANTHROPIC_API_KEY` so sessions bill against the Claude subscription (the driver
also pops it defensively):

```bash
conda activate agentcanvas
cd "$(git rev-parse --show-toplevel)"   # repo root

# pure bare baseline, one episode
env -u ANTHROPIC_API_KEY python beta-coding-agent/opus-lab/driver.py \
    --episodes 0 --model claude-opus-4-8

# bare tools + the opus-nav skill, eps 0-9
env -u ANTHROPIC_API_KEY python beta-coding-agent/opus-lab/driver.py \
    --episodes 0-9 --skill opus-nav --model claude-opus-4-8

# parallel across several auto_hosts (one worker per URL)
env -u ANTHROPIC_API_KEY python beta-coding-agent/opus-lab/driver.py \
    --episodes 0-9 --skill opus-nav --model claude-opus-4-8 \
    --server-urls http://127.0.0.1:9200,http://127.0.0.1:9201
```

Useful flags: `--raw-log` (dump every SDK message), `--run-name NAME` (fixed run
dir instead of a timestamp), `--max-turns N`, `--step-budget N`,
`--rgb-resolution PX`.

## Hill-climb discipline

- **Vary only the skill.** The env is frozen; a change in SR is a change in the
  skill, not the harness.
- **No test answers in the skill.** The skill must teach method, never bake in
  facts from any specific episode's ground truth (layout, path length, goal
  appearance). Invent examples; never lift them from a run.
- **Keep-if-better / rollback-if-worse**, and remember n=10 variance is ±0.1 —
  don't over-read a single-episode delta.
