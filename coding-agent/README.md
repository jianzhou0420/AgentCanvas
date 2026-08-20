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
── entries ──────────────────────────────────────────────────────────────────
stdrun.py     CLI: run / batch / board / compare — the std entry
uirun.py      the Monitor Run-button entry (ui_* runs, off-board by
              construction) — spawned by the backend
── core/ — the shared engine (import as core.*; 2026-08-20) ─────────────────
core/driver.py   shared episode loop + EventSink (single writer of the jsonl
              vocabulary)
core/cells.py    orchestration registry: model zoo, conditions, frozen std
              config, batches/E-registry. Every experiment line's cells
              register from exp_workspace/<arm>/exp.py via the loader at the
              bottom; only the imagine research line registers inline
core/prompts.py  briefing surfaces for OFF-ARM runs (ui_* free runs, the
              inline imagine line). Migrated arms carry their own frozen
              prompts.py copies — the arm copy is the truth for arm cells
core/monitor_api.py  the run-artifact + scoring contract — the ONE surface
              any monitor consumes runs through (layout, honest-SR rule,
              roots, uirun spawn contract); backend loads it by path
core/harnesses/  claude_sdk.py · mini_swe.py · codex_cli.py — one adapter
              each; mini/ = the mini harness's in-repo body (toolset/model/
              env/nav_agent) + check_equivalence.py, the byte gate between
              the arm bridge copies and mini's in-process ports
core/api/     the API layer (2026-08-19): providers.py (vendor + key
              registry, DASHSCOPE_API_KEY et al), proxy.py (litellm gateway —
              codex cross-API seats), anthropic_shim.py (anthropic-wire shim
              — sdk cross-API seats; exists because litellm's /v1/messages
              mangles openai/responses/* routes)
exp_workspace/  one folder = ONE experiment arm (2026-08-18 rule:
              orchestration is shared, execution code is DUPLICATED per arm).
              Each folder owns its exp.py (frozen knobs + cell registration,
              loaded by cells.py), bridge.py, prompts.py and nodeset/ copy.
              The shared bridges/ dir was dissolved 2026-08-20 — every bridge
              lives in its arm (cmp-gated migration; env-knob ancestors
              parked in <arm>/_upstream/ for provenance). Fork a folder to
              start a new experiment, never edit one whose boards have run.
              Serve an env with PYTHONPATH = coding-agent + agentcanvas/
              backend and `python -m app.server.auto_host --module
              exp_workspace.<exp>.nodeset --class <NodeSet> --port 92xx`
              (the folder's env python; see each folder's exp.py header).
              Folders: slam_baseline/01/02 (ac-habitat033) · bare · wp ·
              hybrid (ac-vlnce; wp/hybrid also need the :9210 predictor) ·
              vlnverse (ac-vlnverse) · hmeqa (3 profiles: hmeqa / mthm3d /
              hmeqa500; ac-hmeqa) · express (ac-hmeqa) · objnav · ovon
              (ac-objnav) · libero_bare/_full/_tb/_tbv (ac-libero, arm
              flags BAKED per folder) · go2 (bridge + robot-side host, no
              nodeset). Historical cell names unchanged everywhere; new
              seats: rxr_sdk_*_bare_default (RX) · rxr_sdk_*_wp (RXW) ·
              hmeqa500_sdk_fable-5 · xapi_* (cross-API probes)
── support ──────────────────────────────────────────────────────────────────
reporting/    run_stats.py — post-run charts + tables (driver writes
              stats.html at run end; backend backfills older runs)
scripts/      standalone analysis/data scripts (analyze_hybrid.py feeds the
              paper's hybrid section)
splits/       the single physical home of ALL split data (2026-08-18):
              *_seed42.json sampling-provenance manifests (data/ is
              gitignored, so these are the one versioned record) + r2r/ rxr/
              habitat-format split dirs; the dataset tree keeps symlinks so
              loader path templates resolve unchanged
exp_workspace/wp/ac_wp_predictor_shim/  habitat-free SmartWay predictor tree
              for the wp auto_host (wp arm owns it; hybrid/imagine point at it)
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
# ac-wp env (py3.10 + torch cu128 — GPU inference on sm_120 cards; install:
# bash scripts/install/install_ac_wp.sh) against the habitat-free shim tree;
# see coding-agent/exp_workspace/wp/ac_wp_predictor_shim/README.md. On cu121
# cards (3090) the same shim also runs under ac-smartway's python.
# Checkpoints: data/smartway/waypoint_ckpt/best.pth + data/smartway/ddppo/
# gibson-2plus-resnet50.pth (symlink into VLN-MME's data).
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  SMARTWAY_REPO_PATH=$PWD/../../coding-agent/exp_workspace/wp/ac_wp_predictor_shim \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "$(conda run -n ac-wp which python)" -m app.server.auto_host \
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

# LIBERO manipulation cells (libero_sdk_*) need the env_libero auto_host
# instead (ac-libero env; MUJOCO_GL=egl for offscreen MuJoCo rendering):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. MUJOCO_GL=egl \
  ~/miniforge3/envs/ac-libero/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_libero/__init__.py \
  --class EnvLiberoNodeSet --port 9270

# then (agentcanvas env):
python coding-agent/stdrun.py run std_sdk_opus-4.8_bare
python coding-agent/stdrun.py run std_sdk_fable-5_wp   # reads --wp-server (default :9210)
python coding-agent/stdrun.py run libero_sdk_sonnet-5 --servers http://127.0.0.1:9270
python coding-agent/stdrun.py run libero_sdk_sonnet-5_full --servers http://127.0.0.1:9270  # sensor rung: +wrist/proprio/measured-movement/auto-observe
python coding-agent/stdrun.py run libero_sdk_sonnet-5_tb --servers http://127.0.0.1:9270    # toolbox rung: atomic views + GT scene readout + move_to/gripper macros
python coding-agent/stdrun.py run libero_sdk_sonnet-5_tbv --servers http://127.0.0.1:9270   # vision toolbox: pixel_to_3d depth backprojection instead of GT get_objects
python coding-agent/stdrun.py batch A          # sdk × {sonnet,opus,fable} × bare_max
python coding-agent/stdrun.py board            # grid status from summaries on disk
python coding-agent/stdrun.py compare std_sdk_opus-4.8_bare std_mini_opus-4.8_bare
```

Cells, not flags: 200 turns / 512 px / rand100 0-99 / 500 actions / 2400 s
(std-v2) are pinned in `cells.py`. `run --episodes 3,7` reruns/resumes specific indices
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
- **The bridge stays the single tool surface**: each arm's
  `exp_workspace/<arm>/bridge.py` (sdk + codex spawn it; mini's
  byte-equivalent port is still gated by
  `coding-agent/harnesses/mini/check_equivalence.py`, whose references point
  at the bare/wp arm copies since bridges/ dissolved 2026-08-20).
- **Event vocabulary enforced by construction**: adapters can only emit
  through `driver.EventSink`, which also derives tool-call counts and
  env-step totals uniformly for all harnesses.
- Freeze discipline: changing any frozen knob in `cells.py` is std-v2
  territory — new cell names, never edits in place.
