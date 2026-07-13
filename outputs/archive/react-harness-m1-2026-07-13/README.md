# react-harness M1 calibration runs — 2026-07-13

First runs of `beta-react-harness/` (mini-swe-agent VLN harness; same env,
tool schemas, prompts, and metrics as the claude-SDK path — harness is the
only moved variable). All runs: R2R-CE `rand100`, full toolset
(observe / look_around / step + clearance readout + turn-budget broadcast +
STOP confirmation gate), **no skill text**, driver-side `env_habitat__evaluate`.

| run | model | episodes | SR | SPL | nDTW | oracle-SR | stop-rate | cost |
|---|---|---|---|---|---|---|---|---|
| smoke2 | claude-sonnet-5 | 0 | 0.0 (dtg 3.24 m) | 0 | 0.74 | 0 | 1.0 | $0.18 |
| calib10 | claude-sonnet-5 | 0–9 | 0.20 | 0.15 | 0.50 | 0.30 | 0.9 | $2.66 |
| opus10 | claude-opus-4-8 | 0–9 | 0.20 | 0.17 | 0.61 | 0.40 | 1.0 | $6.17 |
| opus10_19 | claude-opus-4-8 | 10–19 | 0.80 | 0.59 | 0.59 | 0.90 | 1.0 | $7.84 |
| opus20_49 | claude-opus-4-8 | 20–49 | 0.37 | 0.33 | 0.51 | 0.47 | 1.0 | $13.26 |

**mini-opus combined 0–49 (n=50): SR 0.42 · SPL 0.35 · nDTW 0.55 ·
oracle-SR 0.54 · near-miss 10** vs SDK opus50_bare 0.48 / opus50_ledger
0.50 on the identical episode set.

Headline findings (same-subset comparison against the SDK baselines in
`../coding-agent-sdk-baselines-2026-07-13/`):

- **Statistically equivalent at n=50.** Paired per-episode McNemar:
  mini vs bare discordant 7:10 (p=0.63), mini vs ledger 7:11 (p=0.48) —
  the −6…−8 pt point-estimate gap is well within noise; 33/50 episodes
  concordant. Residual soft spots for mini: SPL (0.35 vs 0.46 — less
  efficient paths) and near-miss count (10 vs 5–8).
- **Failure modes reproduce across harnesses.** On eps 0–9, mini-opus and
  SDK opus+ledger both fail with exactly 5 near-miss stops at 3.4–4.5 m
  (paths correct, stop one step early). Episode-batch difficulty structure
  (0–9 hard 0.20–0.30, 10–19 easy 0.50–0.80) reproduces in every condition.
- **Cost:** ~$0.27/ep (sonnet) / ~$0.62/ep (opus) — ≈5× cheaper than the
  SDK path's self-reported cost, thanks to prefix caching over mini's
  linear history.

Per run: `summary.json` (config + aggregate + per-episode records),
`episode_{i}.jsonl` (curated event log — same vocabulary as the SDK path;
rendered by the Coding-Agent Monitor with `source=mini-swe`), `raw/`
(mini's full per-step message trajectories, base64 blobs elided).
`live_{i}/` frame dumps are NOT archived (30–40 MB/run); rerun to regenerate.
