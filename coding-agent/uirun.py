"""uirun — the Coding-Agent Monitor's driver entry (UI runs, claude-sdk only).

The Run button's free knobs (episodes / split / max turns / model) make every
UI run off-board by construction; run names are ui_* so they can never be
mistaken for std cells. Everything else — episode loop, EventSink vocabulary,
artifact layout, evaluation — is the shared core (driver.py) with the
claude_sdk adapter, so UI runs and std cells stay one implementation.

Spawned by app/services/coding_agent_runner.py; flags mirror the legacy
beta-coding-agent driver's so the runner surface stays small.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cells import CellSpec
from driver import run_cell
from harnesses import get_adapter


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help='e.g. "0", "0-9", "0,3,7"')
    parser.add_argument("--split", default="rand100")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--server-url", default="http://127.0.0.1:9200")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=None,
                        help="SDK model id; blank = the CLI's default model")
    args = parser.parse_args()

    spec = CellSpec(
        name=args.run_name,
        harness="sdk",
        model_key="ui",
        model_id=args.model or "",
        condition="ui",  # full toolset, no skill — the legacy UI condition
        bare=False,
        skill=None,
        max_turns=args.max_turns,
    )
    asyncio.run(run_cell(
        get_adapter("sdk"), spec, [args.server_url],
        episodes_spec=args.episodes, run_name=args.run_name,
        cfg_overrides={"split": args.split},
    ))


if __name__ == "__main__":
    main()
