# coding-agent — unified std experiment runner

The high-level interface over the harness cells, and (since 2026-08-03) the
ONE home of the whole coding-agent experiment: the former repo-root
`beta-coding-agent/`, `beta-react-harness/`, and `beta-codex-agent/` dirs are
unified here — live shared assets in subpackages, the frozen legacy drivers
under `legacy/`. Each legacy driver used to carry a full copy of the driver
skeleton and prompt drafts; this package collects that shared 90% once and
reduces each harness to a ~100-line adapter, so the std board
(docs → developer-guide/tmp/coding-agent/standard-experiments.html) runs from
ONE core.

```
prompts.py    briefing surfaces (bare/nav/wp/hybrid/hmeqa) + skill loader + md5 gate
cells.py      the std board as code: cells across 8 prefixes, frozen knobs, batches
driver.py     shared episode loop + EventSink (single writer of the jsonl vocabulary)
harnesses/    claude_sdk.py · mini_swe.py · codex_cli.py — one adapter each
stdrun.py     CLI: run / batch / board / compare
bridges/      the agent-facing tool surfaces (stdio MCP): mcp · wp · hybrid ·
              hmeqa · go2(+go2_host)
mini/         mini-swe-agent harness modules (toolset/model/env/nav_agent) + check_equivalence.py
analysis/     analyze_hybrid.py (interface choices, feeds the paper's hybrid section)
wp_predictor_shim/  habitat-free SmartWay predictor tree for the wp auto_host
legacy/       the three frozen pre-unification drivers + opus-lab (provenance; never edited)
```

Trimmed 2026-08-03 to the TEAPS-paper surface: the ObjectNav-family pieces
(objnav bridges, splits/, sample_episodes.py), skills/, and nodeset_mcp.py
were deleted — recover any of them from git history. The Coding-Agent
Monitor's helpers live in `scripts/eval/` (`uirun.py` Run-button entry,
`run_stats.py` stats backfill); the cells/driver code for the ObjectNav and
skill cells remains, so those cells enumerate but cannot run until their
files are restored.

## Usage

```bash
# env server(s), one per worker (ports 9200+; ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# wp cells additionally need the waypoint-predictor server. It runs in the
# ac-wp env (py3.10 + torch cu128 — GPU inference on sm_120 cards) against the
# habitat-free shim tree; see coding-agent/wp_predictor_shim/README.md.
# Checkpoints: data/smartway/waypoint_ckpt/best.pth + data/smartway/ddppo/
# gibson-2plus-resnet50.pth (symlink into VLN-MME's data).
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  SMARTWAY_REPO_PATH=$PWD/../../coding-agent/wp_predictor_shim \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  ~/miniconda3/envs/ac-wp/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/method/smartway_waypoint/__init__.py \
  --class SmartWayWaypointNodeSet --port 9210

# then (agentcanvas env):
python coding-agent/stdrun.py run std_sdk_opus-4.8_bare
python coding-agent/stdrun.py run std_sdk_fable-5_wp   # reads --wp-server (default :9210)
python coding-agent/stdrun.py batch A          # sdk × {sonnet,opus,fable} × {bare,nav}
python coding-agent/stdrun.py board            # grid status from summaries on disk
python coding-agent/stdrun.py compare std_sdk_opus-4.8_bare std_mini_opus-4.8_bare
```

Cells, not flags: 80 turns / 224 px / rand100 0-49 / 500 actions / 2400 s are
pinned in `cells.py`. `run --episodes 3,7` reruns/resumes specific indices
into the same run dir (records merge). Anything else needs `--nonstd`, which
renames the run `nonstd_*` so it can never sit on the board.

Auth per harness: sdk = Claude subscription (adapter strips a stray
`ANTHROPIC_API_KEY`); mini = requires `ANTHROPIC_API_KEY` (litellm billing);
codex = ChatGPT subscription (`codex login`).

## Design decisions

- **Outputs land in the legacy per-harness roots** (`outputs/beta-coding-agent`
  etc.), same artifact layout — the Coding-Agent Monitor and its source
  toggle work unchanged.
- **Legacy drivers are frozen, not edited** — they document how the pre-std
  archived runs were produced. New runs go through this package only.
- **The bridge stays the single tool surface**: `coding-agent/bridges/mcp_bridge.py`
  (sdk + codex spawn it; mini's byte-equivalent port is still gated by
  `coding-agent/mini/check_equivalence.py`).
- **Event vocabulary enforced by construction**: adapters can only emit
  through `driver.EventSink`, which also derives tool-call counts and
  env-step totals uniformly for all harnesses.
- Freeze discipline: `prompts.py` refuses a nav run if the ledger-nav body
  md5 drifts from `f7c74272`; changing any frozen knob is std-v2.
