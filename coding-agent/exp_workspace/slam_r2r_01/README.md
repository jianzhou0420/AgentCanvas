# slam_r2r_01 — ORB-SLAM3 + frontier annotation (SLAM lineage)

baseline + three read-only instruments (get_pose / get_map / get_trajectory)
+ step slam_note. Map v1: crop-to-explored render, numbered frontier circles
with STABLE ids (registry ≤1.5 m), SLAM pose source + 360° bootstrap.

- Frozen: same protocol as baseline (exp.py `_FROZEN`; extra instruments=1).
- Serve: see exp.py header. Run: `stdrun.py run slam_r2r_01_sdk_<model>`.
- Boards: slam_r2r_01_sdk_opus-5 (2026-08-18, SR 0.74/SPL 0.558, ATE 1.30 m)
  — ran pre-exp_workspace from the same code this folder froze.
- Rule: NEVER edit — fork instead.
