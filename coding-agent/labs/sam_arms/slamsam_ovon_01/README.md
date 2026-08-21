# slamsam_ovon_01 — HM3D-OVON ObjectNav on the SLAM+SAM map, with the target push

The slamsam lineage on ObjectNav (2026-08-18). Surface = jian's SLAM-instrument
shape, auto-observe: `step` (0/1/2/3, 30° turns; the result carries the photo),
`get_map` (SLAM map v1 — jian's integrate + numbered stable-id frontiers — with
the SAM patch of the goal word), `get_pose`, `get_trajectory`. No observe tool,
no recall on this arm.

Two additions of ours over the plain SLAM arm:
1. **Async SAM 3 on the ONE goal word**, every sensed frame (keep-latest,
   stamped at its capture pose; a patch the body walks up to and does not see
   decays).
2. **The target push** — the moment a merged detection SEES the goal, the
   running `step` leg is cut short and the result carries `TARGET DETECTED`
   (dir_deg relative to heading, positive = right; dist_m from where the body
   now stands) **plus the map image with the patch marked**; from then on every
   `step` / `get_map` result carries a `target` block re-computed from the
   current pose (last confirmed sighting, else the map patch) so the model can
   close in until dist_m < 1 m and STOP (the historic failure = stopping too
   far). Re-sightings while closing in only refresh the block; a new alert
   fires again only after the target was out of view ≥ 12 steps
   (`REALERT_GAP`). Detection supplies the where; the model still decides.

- Frozen: jian's `ovon-unseen` board line verbatim (`cells.OBJNAV_FROZEN`:
  split mip100_unseen — the scene-stratified seed-42 hundred of val_unseen —
  150 turns + $18 fuse, 500 steps); cells `slamsam_ovon_01_sdk_<m>`, batch SSO.
- Env: `nodeset/` = this folder's copy of `env_ovon` (habitat 0.2.4,
  ac-objnav; motion scalars `actual_translation_m` / `actual_dy_m` /
  `collided` + `depth_units`; `OVON_SPLIT` boot knob). Serve (cwd
  agentcanvas/backend): `PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/
  backend OVON_SPLIT=mip100_unseen <ac-objnav python> -m app.server.auto_host
  --module exp_workspace.slamsam_ovon_01.nodeset --class EnvOvonNodeSet
  --port 9241`. SAM 3: `model_sam` auto_host on 9220 (`LEAN_SAM_URL`).
- Bridge env: the driver's ObjectNav family (`OBJNAV_SERVER_URL` /
  `OBJNAV_VERB_PREFIX` / `OBJNAV_STEP_BUDGET` / `OBJNAV_LIVE_DIR`); the goal
  word is read from the seated episode via `env_ovon__reset`.
- Run: `stdrun.py run slamsam_ovon_01_sdk_opus-5 --servers http://127.0.0.1:9241
  --episodes 0` (smokes: `--nonstd --run-name …`).
- Depth: OVON normalises depth over [0.5, 5] m; a pixel nearer than 0.5 m reads
  as exactly 0.5 and is treated as INVALID (not a 0.5 m wall).
- Code lineage: forked from `slamsam_rxr_01` (2026-08-18); `toolset.py` /
  `slam_sidecar.py` from `harnesses/mini/{lean_toolset,slam_sidecar}.py`;
  `slam/` = jian efe0397 slam_r2r_02 mapping modules + our `_semantic`.
- Rule: NEVER edit after boards run — fork `slamsam_ovon_02/` instead.
