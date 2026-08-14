"""uirun — the Coding-Agent Monitor's driver entry (UI runs, all harnesses).

The Run button's free knobs (harness / episodes / split / max turns / model /
condition / tier / extra) make every UI run off-board by construction; run
names are ui_* so they can never be mistaken for std cells. Everything else —
episode loop, EventSink vocabulary, artifact layout, evaluation — is the
shared core (driver.py) with the selected harness adapter, so UI runs and std
cells stay one implementation.

Spawned by app/services/coding_agent_runner.py; flags mirror the legacy
beta-coding-agent driver's so the runner surface stays small.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells import CONDITIONS, MODEL_EXTRA, MODELS, CellSpec, _model_id, _tier_extra
from driver import run_cell
from harnesses import get_adapter


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="sdk", choices=("sdk", "mini", "codex"))
    parser.add_argument("--episodes", required=True, help='e.g. "0", "0-9", "0,3,7"')
    parser.add_argument("--split", default="rand100")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--server-url", default="http://127.0.0.1:9200")
    parser.add_argument("--wp-server", default=None,
                        help="waypoint-predictor auto_host (wp / hybrid conditions)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=None,
                        help="board model key (resolved via cells.MODELS + MODEL_EXTRA) "
                             "or a literal harness-facing model id; blank = the CLI's default")
    parser.add_argument("--condition", default="bare", choices=tuple(CONDITIONS),
                        help="the MIP paper conditions")
    parser.add_argument("--tier", default="default", choices=("default", "max"),
                        help="effort tier — expands to the per-(harness, model) knobs the std board uses")
    parser.add_argument("--set", dest="extra", action="append", default=[],
                        metavar="KEY=VAL", help="harness extra knob (overrides tier knobs)")
    args = parser.parse_args()

    flags = CONDITIONS[args.condition]
    model_key = args.model or ""
    model_id = _model_id(args.harness, model_key) if model_key in MODELS else model_key
    extra = dict(MODEL_EXTRA.get(model_key, {}))
    extra.update(_tier_extra(args.harness, model_key, args.tier))
    for pair in args.extra:
        key, sep, val = pair.partition("=")
        if not sep:
            parser.error(f"--set expects KEY=VAL, got {pair!r}")
        extra[key] = val

    spec = CellSpec(
        name=args.run_name,
        harness=args.harness,
        model_key=model_key or "ui",
        model_id=model_id,
        condition=args.condition,
        bare=flags.get("bare", False),
        wp=flags.get("wp", False),
        hybrid=flags.get("hybrid", False),
        effort_tier=args.tier,
        extra=tuple(sorted(extra.items())),
        max_turns=args.max_turns,
    )
    asyncio.run(run_cell(
        get_adapter(args.harness), spec, [args.server_url],
        episodes_spec=args.episodes, run_name=args.run_name,
        extra={"split": args.split},
        wp_server=args.wp_server,
    ))


if __name__ == "__main__":
    main()
