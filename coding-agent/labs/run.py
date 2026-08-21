#!/usr/bin/env python
"""Run a parked labs entry in place.

    python coding-agent/labs/run.py <script.py> [args...]
    python coding-agent/labs/run.py -m <module> [args...]

Establishes the labs path + old-name aliases (see _bootstrap), then executes
the target as __main__ with the remaining argv. This is the sanctioned way to
run eharness / vlaharness / ImagineVLN without moving them back onto the board.
"""
import runpy
import sys
from pathlib import Path

import _bootstrap  # labs/ is this script's dir, so this resolves


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python coding-agent/labs/run.py <script.py|-m module> [args...]",
              file=sys.stderr)
        raise SystemExit(2)
    _bootstrap.setup()
    if sys.argv[1] == "-m":
        mod = sys.argv[2]
        sys.argv = [mod, *sys.argv[3:]]
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    else:
        target = str(Path(sys.argv[1]).resolve())
        sys.argv = sys.argv[1:]
        d = str(Path(target).parent)   # target's own dir (bare intra-line imports)
        if d not in sys.path:
            sys.path.insert(0, d)
        runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
