#!/usr/bin/env python3
"""test_allocate_workers.py — sanity table for the makespan-min worker
allocator in multi_spec_eval.py.

Runs the 6 agreed mapgpt scenarios plus edge cases. No pytest dep —
plain assert + a pass/fail print so it can be invoked directly:

    python .claude/commands/architect/myloop/lib/test_allocate_workers.py

Each scenario specifies (ep_count, profile_wc) per submission and the
expected wc list. We also derive actual wave count = ceil(ep / wc) and
the wave wall (max of waves) for visibility.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Import the function under test from the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_spec_eval import allocate_workers  # noqa: E402


def _subs(pairs: list[tuple[int, int]]) -> list[dict]:
    return [{"ep_count": e, "profile_wc": w} for e, w in pairs]


def _waves(submissions: list[dict], wcs: list[int]) -> list[int]:
    return [math.ceil(s["ep_count"] / max(1, wc)) for s, wc in zip(submissions, wcs)]


SCENARIOS = [
    # name, perf_cap, [(ep, profile_wc)], expected_wc, expected_waves, expected_wall_waves
    ("K=1 perf solo",                    40, [(216, 40)],                         [40],            [6],            6),
    ("K=1 custom 3-pass",                40, [(30, 30)] * 3,                      [10] * 3,        [3] * 3,        3),
    ("K=2 all-perf",                     40, [(216, 40), (216, 40)],              [20, 20],        [11, 11],       11),
    ("K=2 perf+custom [1,3]",            40, [(216, 40)] + [(30, 30)] * 3,        [27, 4, 4, 4],   [8, 8, 8, 8],   8),
    ("K=3 all-custom 3-pass",            40, [(30, 30)] * 9,                      [4] * 9,         [8] * 9,        8),
    ("K=2 all-custom 3-pass",            40, [(30, 30)] * 6,                      [6] * 6,         [5] * 6,        5),
    ("method_max=None fallback",       None, [(30, 30), (216, 40)],               [30, 40],        [1, 6],         6),
    ("N > cap (degenerate)",              3, [(30, 30)] * 5,                      [1] * 5,         [30] * 5,       30),
    ("empty submissions",                40, [],                                  [],              [],             0),
]


def main() -> int:
    fails = 0
    print(f"{'scenario':<32} {'cap':>4} {'N':>3} {'wc':<24} {'waves':<24} {'wall':>5}")
    print("-" * 100)
    for name, cap, pairs, exp_wc, exp_waves, exp_wall in SCENARIOS:
        subs = _subs(pairs)
        got = allocate_workers(method_max=cap, submissions=subs)
        got_waves = _waves(subs, got) if got else []
        got_wall = max(got_waves) if got_waves else 0
        ok = (got == exp_wc and got_waves == exp_waves and got_wall == exp_wall)
        mark = "OK" if ok else "FAIL"
        print(f"{name:<32} {str(cap):>4} {len(subs):>3} "
              f"{str(got):<24} {str(got_waves):<24} {got_wall:>5}  [{mark}]")
        if not ok:
            print(f"  expected: wc={exp_wc}  waves={exp_waves}  wall={exp_wall}")
            fails += 1
    print("-" * 100)
    if fails:
        print(f"FAILED: {fails}/{len(SCENARIOS)} scenarios")
        return 1
    print(f"PASSED: {len(SCENARIOS)}/{len(SCENARIOS)} scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
