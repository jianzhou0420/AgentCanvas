# coding-agent — unified std experiment runner

The high-level interface over the harness cells, and (since 2026-08-03) the
ONE home of the whole coding-agent experiment: the former repo-root
`beta-coding-agent/`, `beta-react-harness/`, and `beta-codex-agent/` dirs are
unified here — live shared assets in subpackages, the frozen legacy drivers
in git history (`d10591e`). Each legacy driver used to carry a full copy of the driver
skeleton and prompt drafts; this package collects that shared 90% once and
reduces each harness to a ~100-line adapter, so the std board
(docs → developer-guide/tmp/coding-agent/standard-experiments.html) runs from
ONE core.

```
prompts.py    briefing surfaces (bare/wp/hybrid/hmeqa/go2)
cells.py      the std board as code: 53 cells (std · go2 · hmeqa · vlnverse), frozen knobs, batches
driver.py     shared episode loop + EventSink (single writer of the jsonl vocabulary)
harnesses/    claude_sdk.py · mini_swe.py · codex_cli.py — one adapter each;
              mini/ = the mini harness's in-repo body (toolset/model/env/
              nav_agent) + check_equivalence.py, the bridges<->mini byte gate
stdrun.py     CLI: run / batch / board / compare
monitor_api.py  the run-artifact + scoring contract — the ONE surface any
              monitor consumes runs through (layout, honest-SR rule, roots,
              uirun spawn contract); backend loads it by path
bridges/      the agent-facing tool surfaces (stdio MCP): mcp · wp · hybrid ·
              hmeqa · go2(+go2_host); splits/ = tracked sampling-provenance
              manifests for the derived mip env splits (data/ is gitignored,
              so these are the one versioned record of what each split holds)
scripts/      standalone analysis scripts: analyze_hybrid.py (feeds the paper's hybrid section)
ac_support/   AgentCanvas Monitor support: uirun.py (Run-button entry) ·
              run_stats.py (stats backfill) — spawned by the backend
ac_wp_predictor_shim/  habitat-free SmartWay predictor tree for the wp auto_host
```

Trimmed 2026-08-03 to the MIP-paper surface: the ObjectNav-family pieces
(objnav bridges, their split manifests, sample_episodes.py, the objnav cells
and driver plumbing), skills/ and the nav / wp-nav / persona conditions, and
nodeset_mcp.py were all deleted, and so was `legacy/` (the equivalence
fixtures now come straight from git at `d10591e`) — recover any of it from
git history (cells/plumbing last at `a942483`, files at `cecd19c`).

## Usage

```bash
# env server(s), one per worker (ports 9200+; ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# wp cells additionally need the waypoint-predictor server. It runs in the
# ac-wp env (py3.10 + torch cu128 — GPU inference on sm_120 cards) against the
# habitat-free shim tree; see coding-agent/ac_wp_predictor_shim/README.md.
# Checkpoints: data/smartway/waypoint_ckpt/best.pth + data/smartway/ddppo/
# gibson-2plus-resnet50.pth (symlink into VLN-MME's data).
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  SMARTWAY_REPO_PATH=$PWD/../../coding-agent/ac_wp_predictor_shim \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  ~/miniconda3/envs/ac-wp/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/method/smartway_waypoint/__init__.py \
  --class SmartWayWaypointNodeSet --port 9210

# VLNVerse cells (vlnverse_*) talk to an env_vlnverse auto_host instead. It
# runs in the lean ac-vlnverse env (no simulator: Isaac Sim 5.1 renders in its
# OWN bundled python behind an msgpack RPC seam, spawned by the nodeset). Data:
# data/vlnverse/{raw_data,scene} — laid down by scripts/install/
# install_ac_vlnverse.sh, or fetched with scripts/data/fetch_{episodes,scenes}_
# vlnverse.py. Cold boot is ~25 s (Isaac + first scene).
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniconda3/envs/ac-vlnverse/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_vlnverse/__init__.py \
  --class EnvVLNVerseNodeSet --port 9260

# then (agentcanvas env):
python coding-agent/stdrun.py run std_sdk_opus-4.8_bare
python coding-agent/stdrun.py run std_sdk_fable-5_wp   # reads --wp-server (default :9210)
python coding-agent/stdrun.py batch A          # sdk × {sonnet,opus,fable} × bare_max
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
- **Legacy drivers are frozen in git history** (fixtures pinned at
  `d10591e:beta-*/run_episodes.py` by `harnesses/mini/check_equivalence.py`) — they
  document how the pre-std archived runs were produced. New runs go through
  this package only.
- **The bridge stays the single tool surface**: `coding-agent/bridges/mcp_bridge.py`
  (sdk + codex spawn it; mini's byte-equivalent port is still gated by
  `coding-agent/harnesses/mini/check_equivalence.py`).
- **Event vocabulary enforced by construction**: adapters can only emit
  through `driver.EventSink`, which also derives tool-call counts and
  env-step totals uniformly for all harnesses.
- Freeze discipline: changing any frozen knob in `cells.py` is std-v2
  territory — new cell names, never edits in place.
