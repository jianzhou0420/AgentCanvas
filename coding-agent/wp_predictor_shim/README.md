# wp_predictor_shim — habitat-free repo tree for the waypoint predictor server

The `smartway_waypoint` nodeset loads its model code from `SMARTWAY_REPO_PATH`
(default: its `_vendored/` tree). That tree's depth-encoder file
(`vlnce_baselines/models/encoders/resnet_encoders.py`) imports
`habitat`/`habitat_baselines`, which is why the original `ac-smartway` env
carried a habitat-lab 0.1.7 install — and why it was pinned to Python 3.8 /
torch 2.1.1 (cu121), which cannot drive an RTX 5090 (sm_120 needs cu128 →
torch ≥ 2.7 → Python ≥ 3.9).

This shim is an alternative `SMARTWAY_REPO_PATH` with zero habitat
dependencies, so the predictor can run in a modern GPU env (`ac-wp`,
py3.10 + torch cu128). No vendored/nodeset file is modified:

| Entry | What it is |
|---|---|
| `waypoint_predictor/` | symlink to the vendored tree (TRM + cross-attn + bert — already habitat-free) |
| `wp_ddppo_resnet/resnet.py`, `running_mean_and_var.py` | verbatim copies of the vendored ones (torch-only) |
| `wp_ddppo_resnet/resnet_policy.py` | `ResNetEncoder` extracted verbatim from `third_party/habitat-lab/.../resnet_policy.py` (what the gibson-2plus DDPPO ckpt was trained with), imports rewritten to the sibling copies |
| `vlnce_baselines/models/encoders/resnet_encoders.py` | verbatim vendored copy; only the three habitat imports rewritten (plain `logging` logger + the `wp_ddppo_resnet` package above) |

Launch (see also `coding-agent/README.md`):

```bash
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  SMARTWAY_REPO_PATH=$PWD/../../coding-agent/wp_predictor_shim \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  ~/miniconda3/envs/ac-wp/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/method/smartway_waypoint/__init__.py \
  --class SmartWayWaypointNodeSet --port 9210
```

(`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because torch ≥ 2.6 defaults
`torch.load(weights_only=True)` and the engine loads trusted local
checkpoints written by older torch.)
