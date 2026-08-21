# slamsam_rxr_01 — SLAM map v1 + SAM landmarks + recall, on RxR-CE rand100_en

The slamsam lineage: jian's SLAM-map shape (`step` primitives 0/1/2/3 whose
result carries the photo, tagged frame#N; `get_map` — the SLAM map, pulled,
free) plus TWO additions of ours: the **async SAM 3 landmark layer** painted
on the map (one keep-latest job per sensed frame, stamped at its capture pose;
a label the body walks up to and does not see decays), and **`recall`** over
the frames already seen (one / recent / range — the episode's own video).
Nothing else on the model's surface: no state block, subgoals, judge, guards,
heartbeat, events, reminders, loop verdict or candidate menu. Disk per
episode: the seen frames (`obs_*`, `bootstrap_current.png`) + ONE current map
(`map_latest.png`, async keep-latest).

Map = jian's **map v1** (`slam_r2r_01`: single-floor `OccupancyMap`, numbered
frontiers with stable ids), integrated by the side-car
from the env's measured motion — NOT ORB-SLAM3. This line runs on the std
VLN-CE env, so its numbers pool with the std R2R/RxR boards and NEVER with
jian's `slam_r2r_*` (habitat-sim 0.3.3) boards.

- Frozen: `exp.py FROZEN` — benchmark r2r verbs, dataset RxR-CE, split
  rand100_en (`coding-agent/splits/rxr/rand100`), 200 turns, 500 steps.
- Env: `nodeset/` = this folder's copy of the std `env_habitat` nodeset
  (single-file; repo-root depth patched). Serve (ac-vlnce python, cwd
  agentcanvas/backend): `PYTHONPATH=<repo>/coding-agent:<repo>/agentcanvas/
  backend python -m app.server.auto_host --module
  exp_workspace.slamsam_rxr_01.nodeset --class EnvHabitatNodeSet --port 92xx`.
  SAM 3: `model_sam` auto_host on 9220 (`LEAN_SAM_URL`).
- Run: `stdrun.py run slamsam_rxr_01_sdk_opus-5 --servers http://127.0.0.1:92xx
  --episodes 0` (run dir `outputs/beta-eharness/slamsam_rxr_01_sdk_opus-5`;
  `--nonstd --run-name` for smokes).
- Rung: 01 = map v1 (jian's frontier map, stable ids); rung 02 = the same
  shell on map v2.
- Code lineage: `toolset.py` / `slam_sidecar.py` forked 2026-08-18 from
  `harnesses/mini/{lean_toolset,slam_sidecar}.py` (reference copies + their
  tests stay there); `slam/` = the pure SLAM modules (`_mapping/_frontier/
  _map_render` from jian efe0397 slam_r2r_02 with our SAM overlay, `_semantic`
  ours). Shared library imports only: `harnesses/mini/toolset.py` (tool base),
  `eharness.depthmap`, `eharness.landmarks`,
  `bridges/keywords/rand100_keywords.json` (data).
- SAM phrases: the episode's fixed keywords (rxr_rand100_en table), synonyms
  0 — never a model call.
- Rule: NEVER edit after boards run — fork `slamsam_rxr_02/` instead (02 = the map v2 rung).
