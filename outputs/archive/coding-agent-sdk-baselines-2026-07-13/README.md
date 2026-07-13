# coding-agent SDK baseline trajectories — archived 2026-07-13

Curated text trajectories (`summary.json` + `episode_{i}.jsonl`) of the
claude-SDK path (`beta-coding-agent/`, Claude Agent SDK over the habitat MCP
bridge), archived as the comparison baselines for the react-harness M1 runs
in `../react-harness-m1-2026-07-13/`. Frame dumps / raw SDK message dumps /
workdirs are not included (the full corpus is ~3.6 GB, gitignored).

All runs: R2R-CE `rand100`. "bare" = observe/step only, no tuned mechanisms;
otherwise full mechanisms (clearance readout, turn-budget broadcast, STOP
gate, look_around); "+skill" appends the named SKILL.md to the system prompt.

| run | model | condition | episodes | SR |
|---|---|---|---|---|
| sonnet100 | claude-sonnet-5 | mechanisms, no skill | 0–99 | 0.19 |
| opus50_bare | claude-opus-4-8 | bare | 0–49 | 0.48 |
| opus50_ledger | claude-opus-4-8 | +ledger-nav | 0–49 | 0.50 |
| gen50_ledger | claude-sonnet-5 | +ledger-nav | 0–49 | 0.44 |
| fable50_bare | claude-fable-5 | bare | 0–49 | 0.64 |
| fable25_ledger / fable25_49_ledger | claude-fable-5 | +ledger-nav | 0–24 / 25–49 | 0.68 / 0.56 |
| haiku50_ledger / haiku10_ledger | claude-haiku-4-5 | +ledger-nav | 0–49 / 0–9 | 0.20 / 0.10 |
| calib10c | (account default) | mechanisms, no skill | 0–9 | 0.30 |
| tune1–tune5 | claude-sonnet-5 | +ledger-nav, mechanism iterations | 0–9 | 0.10→0.40 |
| opus_skill_v1–v4 | claude-opus-4-8 | +opus-nav | 0–9 | 0.30/0.30/0.10/0.30 |
