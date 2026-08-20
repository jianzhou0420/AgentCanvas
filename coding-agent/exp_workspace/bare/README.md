# bare — the minimal two-tool surface (MIP main arm)

Folder = method arm: observe + step over the stdio MCP bridge, nothing else —
the MIP paper's main board (§4.2 tab:main-results / tab:bare-board), the
effort ablation (§4.3, the `*_max` tier of the same cells), and the
long-horizon RxR-bare column (§4.6 tab:long-horizon). Migrated from the
shared `bridges/mcp_bridge.py` + `prompts.py` bare branch on 2026-08-18.

## Profiles

- **r2r** — cells keep their historical `std_{sdk,mini,codex}_{model}_bare_{tier}`
  names (29 cells; qwen columns untier-ed). Frozen = `cells.STD_FROZEN`
  (R2R-CE / rand100 / 0-99 / 200 turns / rgb 512 / 500 steps / 2400 s) — NOT
  re-registered under a benchmark key, so the frozen std board (wp / hybrid
  r2r cells included) resolves exactly as before. Batches:
  O5 Q8 Ad Bd Gd Xd A B G X Q.
- **rxr** — `rxr_sdk_{model}_bare_default` (sonnet-5 / opus-4.8 / fable-5 /
  opus-5), batch `RX`. Frozen from the paper's archive runs
  (`rxr_bare_fable_rand100` et al., tab:long-horizon): RxR-CE / rand100_en /
  0-99 / **max_turns 70** / 500 steps / 2400 s, classic observe/step
  alternation (`auto_observe` baked OFF per cell — the driver's post-paper
  RxR default would flip it ON). First registered board seats for RxR; the
  pre-profile named runs stay off-board archives.

## Parity (2026-08-18 migration)

- bridge.py = byte copy of `bridges/mcp_bridge.py` (cmp-identical; the shared
  file stays until the vlnverse line — same bridge, verb-prefix env —
  migrates). mini cells keep their in-process port
  (`harnesses/mini/toolset.py`), still gated by `check_equivalence.py`
  against the shared bridge — folder copy and shared bridge must not drift.
- prompts.py = frozen bare briefing; builder output byte-identical to the
  shared `build_briefing(bare=True)`.
- All 29 pre-existing cell specs unchanged except `exp_dir`; batches
  byte-identical; tool schemas via the folder bridge byte-identical.
- nodeset/ = copy of `workspace/nodesets/env/env_habitat.py` with the ONE
  required edit: repo_root derives from FOUR parents here (folder depth),
  three in the original. py3.8-compiled; imports clean under ac-vlnce.

## Serve + run

```bash
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../../coding-agent \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --module exp_workspace.bare.nodeset --class EnvHabitatNodeSet --port 9200

python coding-agent/stdrun.py run std_sdk_fable-5_bare_default   # r2r profile
python coding-agent/stdrun.py run rxr_sdk_fable-5_bare_default   # rxr profile
```

## Boards

- r2r: the full MIP std board ran pre-exp_workspace from the same code this
  folder froze (summaries in the legacy run dirs, names unchanged).
- rxr: smoke ep0 sonnet-5 2026-08-18 through the folder chain; full boards
  pending.

Rule: NEVER edit — fork instead.
