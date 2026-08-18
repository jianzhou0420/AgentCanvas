# slam_r2r_baseline — MIP bare arm (SLAM lineage)

Minimal interface: observe / step / STOP only — no instrument tools, no
briefing addendum. Env runs GT pose, no SLAM container.

- Frozen: official-rotation rand100 · eps 0-99 · 200 turns / 500 steps /
  2400 s / 512² (exp.py `_FROZEN`).
- Serve: see exp.py header. Run: `stdrun.py run slam_r2r_baseline_sdk_<model>`.
- Boards: slam_r2r_baseline_sdk_opus-5 (2026-08-18, SR 0.71/SPL 0.569) —
  ran pre-exp_workspace from the same code this folder froze.
- Rule: NEVER edit after a board runs — fork a new folder.
