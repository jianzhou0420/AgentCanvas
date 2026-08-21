# slamsam_objnav_02 — ObjectNav (HM3D / MP3D / OVON) on the SLAM map, SAM 3 as an EXTERNAL TOOL + the final push

The slamsam lineage on ObjectNav, second arm (2026-08-18). Successor of
`slamsam_ovon_01` (the *streaming push* arm: a background detector on every
frame that cut the running step short and handed the model a `target` block).
Here the detector is an **external tool the model calls** — pure ReAct while
walking: nothing watches the frames, nothing interrupts; the model asks when
it wants to know. Two harness-side pieces remain, both user-ruled the same
night: only **high-score** detections are stamped on the map, and STOP runs a
**final push** toward the detected target.

Surface = jian's SLAM-instrument shape, auto-observe: `step` (0/1/2/3, 30°
turns; the result carries the photo — and nothing else; STOP = final push +
stop), **`detect_target()`**, `get_map` (SLAM map v1 — jian's integrate +
numbered stable-id frontiers — with the high-score detection patches under
`landmarks`), `get_pose`, `get_trajectory`. No observe tool, no recall.

`detect_target()` (free): SAM 3 for the **exact goal word** on the view the
model is looking at (the last sensed RGB-D, kept with its pose).
- **Strict** (user: "pillow is not cushion"; "置信度只有在 0.8 或者 0.85 以上才
  可以"): `SAM_SYNONYMS=0` — the goal word itself, never a synonym or fallback
  — and `SAM_SCORE_THRESH=0.85` (the stricter end of the user's band; the
  shared organ default is 0.5): weaker matches are not returned at all.
- Hit → the overlay image (each match painted, labelled with distance) +
  JSON `instances` (dir_deg relative to heading, positive = right; dist_m;
  score) + `stamped_on_map: true` — **those high-score matches, and only
  those, are stamped on the map** (`votes=2`: one look already renders; no
  decay on this arm — a strict gate misses real objects too often to let a
  miss erase a stamp; the model judges the patches). Miss → JSON only,
  nothing stamped (user: "只有在它的 score 很高 (比如 0.85 以上) 的情况下再标").
- The model reads the overlay and decides (a match is a candidate, not a
  verdict); approach = ask again: turn toward it, walk, `detect_target()` to
  re-read dist_m, until ~1 m and STOP.

**Final push on STOP** (user: "最后没有贴得足够近 … 人为地把这个往前推, 离目标足
够近"): when the model issues 0 and the gated detector sees the target in the
current view, the harness first faces the instance nearest to straight ahead
(≤ ±2 turns), then walks forward until dist ≤ `PUSH_STOP_M` 0.5 m, or blocked,
or the target leaves the view, or `PUSH_MAX_STEPS` 8 — every primitive a real,
counted env step — and only then sends STOP; the step JSON reports the ledger
under `final_push` (from_m / to_m / turned / steps / moved_m / stopped_because).
No detection in view at STOP → plain STOP.

- Frozen: the three board lines VERBATIM (`cells.OBJNAV_FROZEN`: mip100 /
  mip100_unseen — the scene-stratified seed-42 hundred of val — 150 turns +
  $18 fuse, 500 steps). One folder, three profiles, the same code:
  | profile | cells | batch | env |
  |---|---|---|---|
  | hm3d | `slamsam_hm3d_02_sdk_<m>` | SSH2 | `nodeset_objnav/` (env_objnav, hm3d_v1) |
  | mp3d | `slamsam_mp3d_02_sdk_<m>` | SSM2 | `nodeset_objnav/` (env_objnav, mp3d_v1) |
  | ovon-unseen | `slamsam_ovon_02_sdk_<m>` | SSO2 | `nodeset_ovon/` (env_ovon) |
- Envs (habitat 0.2.4, ac-objnav): `nodeset_ovon/` = this folder's copy of
  `env_ovon`, `nodeset_objnav/` = this folder's copy of `env_objnav`, both with
  the motion scalars `actual_translation_m` / `actual_dy_m` / `collided` +
  `depth_units`, and boot knobs (`OVON_SPLIT`; `OBJNAV_DATASET` /
  `OBJNAV_SPLIT`). Serve (cwd agentcanvas/backend, ac-objnav python):
  - hm3d: `PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/backend OBJNAV_DATASET=hm3d_v1 OBJNAV_SPLIT=mip100 <py> -m app.server.auto_host --module exp_workspace.slamsam_objnav_02.nodeset_objnav --class EnvObjnavNodeSet --port 9242`
  - mp3d: same with `OBJNAV_DATASET=mp3d_v1` (another port)
  - ovon: `PYTHONPATH=… OVON_SPLIT=mip100_unseen <py> -m app.server.auto_host --module exp_workspace.slamsam_objnav_02.nodeset_ovon --class EnvOvonNodeSet --port 9241`
  - SAM 3: `model_sam` auto_host on 9220 (`LEAN_SAM_URL`).
- Bridge env: the driver's ObjectNav family (`OBJNAV_SERVER_URL` /
  `OBJNAV_VERB_PREFIX` = env_objnav | env_ovon / `OBJNAV_STEP_BUDGET` /
  `OBJNAV_LIVE_DIR`); the goal word is read from the seated episode via
  `<verb>__reset` (`tv_monitor` → `tv monitor`).
- Run: `stdrun.py run slamsam_hm3d_02_sdk_opus-5 --servers http://127.0.0.1:9242
  --episodes 0` (smokes: `--nonstd --run-name …`); likewise `slamsam_mp3d_02_…`,
  `slamsam_ovon_02_…`.
- Depth: both envs normalise depth over [0.5, 5] m; a pixel nearer than 0.5 m
  reads as exactly 0.5 and is treated as INVALID (not a 0.5 m wall).
- Disk (live_dir): `obs_NNNN_stepSSS.png` (views), `obs_NNNN_map.png` (pulled
  maps), `det_NNNN_stepSSS.png` (detector overlays the model was shown),
  `bootstrap_current.png` + `bootstrap.json`, ONE `map_latest.png`.
- Baked knobs (bridge.py): MAP_MODE v1 · TURN_DEG 30 · SAM_SYNONYMS 0 ·
  SAM_SCORE_THRESH 0.85 · CAM_HEIGHT 0.88 · FINAL_PUSH on · PUSH_STOP_M 0.5 ·
  PUSH_MAX_STEPS 8 (toolset: STAMP_VOTES 2, PUSH_MAX_TURNS 2).
- Smoke notes (2026-08-18, earlier rungs of the same tool shape, dirs kept
  with `_pre_nomap` / `_nomap` suffixes): OVON EP0 "pillow" — SAM 3 painted
  sofa / armchair *cushions* as pillow at 0.73–0.88 even at a 0.7 gate (the
  0.85 gate dropped all three in the same frame; the model then explored on);
  the briefing makes the model the judge of each candidate. HM3D EP0 "chair"
  — found and closed in via repeated detect_target (1.0 m → one more step →
  STOP), dtg 1.86 twice: the episode's goal set holds ONE annotated chair
  ("folding chair_103", 2 viewpoints) and the dining chairs the model reached
  are not goals — an annotation gap the push cannot rescue; the push is for
  the general "stopped 0.3 m too far" case.
- Code lineage: forked from `slamsam_ovon_01` (2026-08-18); `toolset.py` drops
  the async per-frame SAM + streaming push, adds `detect_target` (gated
  stamping) and the final push on STOP; `slam_sidecar.py` = the 01 copy +
  `votes` on `stamp_points` (decay unused); `slam/` = jian efe0397 slam_r2r_02
  mapping modules + our `_semantic`; `nodeset_objnav/` = workspace
  `env_objnav.py` + the OVON copy's contract patches.
- Tests: `python eharness/tests/test_lean.py` (section "slamsam_objnav_02").
- Rule: NEVER edit after boards run — fork `slamsam_objnav_03/` instead.
