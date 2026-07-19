# coding-agent — unified std-v1 experiment runner

The high-level interface over the three harness cells. The legacy drivers
(`beta-coding-agent/`, `beta-react-harness/`, `beta-codex-agent/`) each carry
a full copy of the driver skeleton and prompt drafts; this package collects
that shared 90% once and reduces each harness to a ~100-line adapter, so the
std-v1 board (docs → developer-guide/tmp/coding-agent/standard-experiments.html)
runs from ONE core.

```
prompts.py    BARE/FULL drafts (07-09 freeze) + skill loader + md5 gate
cells.py      the std board as code: 12 cells + codex appendix, frozen knobs, batches
driver.py     shared episode loop + EventSink (single writer of the jsonl vocabulary)
harnesses/    claude_sdk.py · mini_swe.py · codex_cli.py — one adapter each
stdrun.py     CLI: run / batch / board / compare
```

## Usage

```bash
# env server(s), one per worker (ports 9200+; ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# wp cells additionally need the waypoint-predictor server. It runs in the
# ac-wp env (py3.10 + torch cu128 — GPU inference on sm_120 cards) against the
# habitat-free shim tree; see beta-coding-agent/wp_predictor_shim/README.md.
# Checkpoints: data/smartway/waypoint_ckpt/best.pth + data/smartway/ddppo/
# gibson-2plus-resnet50.pth (symlink into VLN-MME's data).
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  SMARTWAY_REPO_PATH=$PWD/../../beta-coding-agent/wp_predictor_shim \
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
- **The bridge stays the single tool surface**: `beta-coding-agent/mcp_bridge.py`
  (sdk + codex spawn it; mini's byte-equivalent port is still gated by
  `beta-react-harness/check_equivalence.py`).
- **Event vocabulary enforced by construction**: adapters can only emit
  through `driver.EventSink`, which also derives tool-call counts and
  env-step totals uniformly for all harnesses.
- Freeze discipline: `prompts.py` refuses a nav run if the ledger-nav body
  md5 drifts from `f7c74272`; changing any frozen knob is std-v2.
