"""stdrun — the high-level interface to the std-v1 standard experiments.

Cells, not flags: every frozen knob (80 turns / 224 px / rand100 0-49 / 500
actions / 2400 s) comes from the registry; the only free choices are WHICH
cell and WHICH env servers. Deviations require --nonstd, which renames the
run so it can never sit on the standard board.

Usage (agentcanvas env; habitat auto_host(s) must already be up):
    python coding-agent/stdrun.py run std_sdk_opus-4.8_bare
    python coding-agent/stdrun.py run std_mini_gpt-5.6_bare --servers http://127.0.0.1:9200,http://127.0.0.1:9201
    python coding-agent/stdrun.py run std_codex_gpt-5.5_bare --episodes 3,7   # rerun/resume two indices
    python coding-agent/stdrun.py batch A
    python coding-agent/stdrun.py board
    python coding-agent/stdrun.py compare std_sdk_opus-4.8_bare std_mini_opus-4.8_bare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cells import BATCHES, CELLS, STD_FROZEN, get_cell
from driver import run_cell
from harnesses import get_adapter


def _servers(arg: str) -> list[str]:
    return [u.strip() for u in arg.split(",") if u.strip()]


async def _run(args: argparse.Namespace) -> None:
    spec = get_cell(args.cell)
    cli_extra: dict = {}
    for kv in args.set or []:
        key, _, value = kv.partition("=")
        try:
            cli_extra[key] = json.loads(value)
        except ValueError:
            cli_extra[key] = value
    run_name = spec.name
    if cli_extra or args.run_name:
        if not args.nonstd:
            sys.exit("[std] overrides given without --nonstd — refusing "
                     "(frozen cells take no knobs; add --nonstd to run off-board)")
        run_name = args.run_name or f"nonstd_{spec.name}"
    # registry-declared model defaults (e.g. a local model's api_base +
    # image_window) are part of the cell, not a deviation
    extra = {**spec.extra_dict, **cli_extra}
    adapter = get_adapter(spec.harness)
    await run_cell(adapter, spec, _servers(args.servers),
                   episodes_spec=args.episodes, run_name=run_name, extra=extra,
                   wp_server=args.wp_server)


async def _batch(args: argparse.Namespace) -> None:
    names = BATCHES.get(args.batch)
    if not names:
        sys.exit(f"[std] unknown batch {args.batch!r}; known: {', '.join(BATCHES)}")
    print(f"[std] batch {args.batch}: {len(names)} cells, sequential")
    for name in names:
        spec = get_cell(name)
        adapter = get_adapter(spec.harness)
        await run_cell(adapter, spec, _servers(args.servers),
                       wp_server=args.wp_server)


def _load_summary(cell_name: str) -> dict | None:
    path = get_cell(cell_name).run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _board(_args: argparse.Namespace) -> None:
    # turns and rgb are printed because the cell NAME carries neither, and both
    # have changed under a stable name (turns 80→200→150→100, rgb 224→512; the
    # archived Claude baselines ran 80). Two runs with the same cell name can be
    # different protocols. Read the board, never the name, before comparing.
    print(f"{'cell':<34} {'eps':>5} {'turns':>6} {'rgb':>5} {'SR':>6} {'SPL':>6} "
          f"{'nDTW':>6} {'stop':>5}")
    for name, spec in CELLS.items():
        # for an unrun cell show what it WILL run at, not the frozen default, or
        # the board lies about a protocol it has not executed yet
        planned_turns = spec.max_turns or STD_FROZEN["max_turns"]
        planned_rgb = STD_FROZEN["rgb_resolution"]
        summary = _load_summary(name)
        if summary is None:
            print(f"{name:<34} {'—':>5} {planned_turns:>6} {planned_rgb:>5}")
            continue
        agg = summary.get("aggregate") or {}
        cfg = summary.get("config") or {}
        n = agg.get("episode_count", 0)
        errored = sum(1 for e in summary.get("episodes", []) if "error" in e)
        flag = f" ({errored} err)" if errored else ""
        print(f"{name:<34} {n:>5} {cfg.get('max_turns', planned_turns):>6} "
              f"{cfg.get('rgb_resolution', planned_rgb):>5} "
              f"{agg.get('success', float('nan')):>6.2f} "
              f"{agg.get('spl', float('nan')):>6.2f} {agg.get('ndtw', float('nan')):>6.2f} "
              f"{agg.get('stop_rate', float('nan')):>5.2f}{flag}")


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _compare(args: argparse.Namespace) -> None:
    a, b = _load_summary(args.cell_a), _load_summary(args.cell_b)
    if a is None or b is None:
        sys.exit("[std] both cells need a summary.json — run them first")

    def by_index(summary: dict) -> dict[int, dict]:
        return {int(e["index"]): e for e in summary.get("episodes", []) if "error" not in e}

    ea, eb = by_index(a), by_index(b)
    common = sorted(set(ea) & set(eb))
    mismatched = [i for i in common if ea[i].get("episode_id") != eb[i].get("episode_id")]
    if mismatched:
        sys.exit(f"[std] episode_id mismatch at indices {mismatched} — not the same episodes")

    def success(rec: dict) -> bool:
        return bool((rec.get("metrics") or {}).get("success"))

    both = sum(1 for i in common if success(ea[i]) and success(eb[i]))
    only_a = sum(1 for i in common if success(ea[i]) and not success(eb[i]))
    only_b = sum(1 for i in common if not success(ea[i]) and success(eb[i]))
    neither = len(common) - both - only_a - only_b
    p = _mcnemar_exact(only_a, only_b)
    sr_a = (both + only_a) / max(1, len(common))
    sr_b = (both + only_b) / max(1, len(common))
    print(f"n={len(common)} paired episodes")
    print(f"  {args.cell_a}: SR {sr_a:.2f}")
    print(f"  {args.cell_b}: SR {sr_b:.2f}")
    print(f"  both {both} | first-only {only_a} | second-only {only_b} | neither {neither}")
    print(f"  exact McNemar p = {p:.4f}"
          + ("  (n=50 detects big effects only)" if len(common) <= 50 else ""))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run (or resume) one cell")
    p_run.add_argument("cell")
    p_run.add_argument("--servers", default="http://127.0.0.1:9200")
    p_run.add_argument("--episodes", default=None,
                       help="override indices for reruns/resume (default: frozen 0-49)")
    p_run.add_argument("--nonstd", action="store_true",
                       help="allow deviations; renames the run off the board")
    p_run.add_argument("--run-name", default=None, help="requires --nonstd")
    p_run.add_argument("--set", action="append", metavar="KEY=VAL",
                       help="harness extra knob (requires --nonstd), e.g. effort=xhigh")
    p_run.add_argument("--wp-server", default="http://127.0.0.1:9210",
                       help="waypoint-predictor auto_host (wp cells only)")

    p_batch = sub.add_parser("batch", help="run a batch of cells sequentially")
    p_batch.add_argument("batch", choices=sorted(BATCHES))
    p_batch.add_argument("--servers", default="http://127.0.0.1:9200")
    p_batch.add_argument("--wp-server", default="http://127.0.0.1:9210",
                         help="waypoint-predictor auto_host (wp cells only)")

    sub.add_parser("board", help="show the standard board")

    p_cmp = sub.add_parser("compare", help="same-episode paired McNemar")
    p_cmp.add_argument("cell_a")
    p_cmp.add_argument("cell_b")

    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    elif args.cmd == "batch":
        asyncio.run(_batch(args))
    elif args.cmd == "board":
        _board(args)
    elif args.cmd == "compare":
        _compare(args)


if __name__ == "__main__":
    main()
