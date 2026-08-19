# hmeqa — HM-EQA bare surface, three data profiles

Folder = method arm (hmeqa_bridge copy: observe / step_discrete /
answer(A-D) terminal; tilt actions frozen ON). Three profiles share this
code (split-agnostic PROFILES shape):
- **hmeqa** — explore-eqa mip100, cells `hmeqa_sdk_{trio}` (batch EQ);
  board fable = 0.76.
- **mthm3d** — MT-HM3D corpus mip100 (label-stratified), cells
  `mthm3d_{sdk×4}`. 口径 caveats: answers skew "A" (66/100) — report the
  constant-A control; published MemoryEQA numbers are full-1587.
- **hmeqa500** — the FULL 500-question val set (paper appendix-B 76.2 ran
  as named run hmeqa_full500_fable-5); NEW seat `hmeqa500_sdk_fable-5`.
  driver.HMEQA_BENCHMARKS carries the family membership.
Migrated 2026-08-18, byte-parity. Rule: NEVER edit — fork instead.
