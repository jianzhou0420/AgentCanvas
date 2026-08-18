# slam_01 — ORB-SLAM3 + frontier annotation (SLAM lineage)

Folder = method arm; cells keep their lineage names per profile:
slam_r2r_01_* (R2R-CE rand100) and slam_rxr_01_* (RxR-CE rand100, canon
rand100_en — corpus switched via the profile's frozen dataset="rxr").

baseline + three read-only instruments (get_pose / get_map / get_trajectory)
+ step slam_note. Map v1: crop-to-explored render, numbered frontier circles
with STABLE ids (registry ≤1.5 m), SLAM pose source + 360° bootstrap.

- Frozen: same protocol as baseline (exp.py `_FROZEN`; extra instruments=1).
- Serve: see exp.py header. Run: `stdrun.py run slam_r2r_01_sdk_<model>`.
- Boards: slam_r2r_01_sdk_opus-5 (2026-08-18, SR 0.74/SPL 0.558, ATE 1.30 m)
  — ran pre-exp_workspace from the same code this folder froze.
- Profiles (2026-08-18, split-agnostic contract): folder = METHOD ARM;
  exp.py PROFILES declares each benchmark's frozen + cell prefix. The
  nodeset carries a corpus table (_env.CORPORA: r2r / rxr roots + file
  naming); the env panel gained a dataset field the driver pushes from
  the profile's frozen.
- Rule: NEVER edit — fork instead.
