# slam_02 — map v2: no frontier · fixed axes · floor layers (SLAM lineage)

Folder = method arm; cells keep their lineage names (slam_r2r_02_*).

Same three instruments as 01 over map v2: frontier layer removed; crop
window snapped to 2 m world multiples and grow-only per episode; per-floor
occupancy layers (dwell-gated registry, height ref = current camera),
"floor k/n" tag, trajectory filtered per floor.

- Frozen: same protocol as baseline (exp.py `_FROZEN`; extra instruments=2).
- Serve: see exp.py header. Run: `stdrun.py run slam_r2r_02_sdk_<model>`.
- Runs: slam_r2r_02_sdk_opus-5_updown (stairs subset 8 eps, 7/8) — board
  not yet run. Design notes tracked on the internal board.
- Profiles (2026-08-18, split-agnostic contract): folder = METHOD ARM;
  exp.py PROFILES declares each benchmark's frozen + cell prefix.
- Rule: NEVER edit after boards run — fork slam_r2r_03/ instead.
