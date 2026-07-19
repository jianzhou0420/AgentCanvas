"""stdrun — the high-level interface to the std-v2 standard experiments.

Cells, not flags: every frozen knob (200 turns / 512 px / rand100 0-49 / 500
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
    python coding-agent/stdrun.py drain std_sdk_opus-4.8_bare   # finish in-flight eps, stop

Graceful drain (batch management): while a cell is running, `drain <cell>`
(or `touch <run_dir>/DRAIN`, or `kill -USR1 <pid>`) makes every worker finish
its current episode, flush, and exit WITHOUT starting a new one — in-flight
work is never cut, and the runner prints the un-run indices as a ready-to-paste
`--episodes` spec for the next resume. Lets you stop a 10-wide batch at an
episode boundary (e.g. before the 5h subscription window runs dry) instead of
hard-killing it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cells import BATCHES, CELLS, EXPERIMENTS, STD_FROZEN, get_cell, resolve_cell
from driver import run_cell
from harnesses import get_adapter


def _servers(arg: str) -> list[str]:
    return [u.strip() for u in arg.split(",") if u.strip()]


async def _run(args: argparse.Namespace) -> None:
    spec = resolve_cell(args.cell)  # cell name OR paper E-number (E1..E20)
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
        excluded = agg.get("excluded", 0)
        flag = f" ({excluded} excluded)" if excluded else ""
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

    def _engaged(rec: dict) -> bool:
        ag = rec.get("agent") or {}
        return bool(ag.get("env_steps") or ag.get("tool_calls")
                    or ag.get("called_stop"))

    def scored_success(rec: dict) -> bool | None:
        """Board口径 (mirrors driver.aggregate): an evaluated episode is
        scored as-is — cap-hit truncation and request_too_large (which crash
        with real spawn-to-death metrics + an error tag but did engage) count
        as failures; an unevaluated timeout scores 0; only a non-engaged infra
        error (account block, server crash) is excluded (returns None → not a
        paired data point)."""
        m = rec.get("metrics") or {}
        if m and (not rec.get("error") or _engaged(rec)):
            return bool(m.get("success"))
        if rec.get("error") == "timeout":
            return False
        return None

    def by_index(summary: dict) -> dict[int, dict]:
        return {int(e["index"]): e for e in summary.get("episodes", [])}

    ea, eb = by_index(a), by_index(b)
    # pair only indices where BOTH episodes are scored under the board口径;
    # an episode excluded on either side can't be a paired McNemar cell
    common = sorted(i for i in set(ea) & set(eb)
                    if scored_success(ea[i]) is not None
                    and scored_success(eb[i]) is not None)
    mismatched = [i for i in common
                  if ea[i].get("episode_id") and eb[i].get("episode_id")
                  and ea[i]["episode_id"] != eb[i]["episode_id"]]
    if mismatched:
        sys.exit(f"[std] episode_id mismatch at indices {mismatched} — not the same episodes")

    sa = {i: scored_success(ea[i]) for i in common}
    sb = {i: scored_success(eb[i]) for i in common}
    both = sum(1 for i in common if sa[i] and sb[i])
    only_a = sum(1 for i in common if sa[i] and not sb[i])
    only_b = sum(1 for i in common if not sa[i] and sb[i])
    neither = len(common) - both - only_a - only_b
    p = _mcnemar_exact(only_a, only_b)
    sr_a = (both + only_a) / max(1, len(common))
    sr_b = (both + only_b) / max(1, len(common))
    print(f"n={len(common)} paired episodes "
          f"(board口径: truncation/too_large=fail, infra excluded)")
    print(f"  {args.cell_a}: SR {sr_a:.2f}")
    print(f"  {args.cell_b}: SR {sr_b:.2f}")
    print(f"  both {both} | first-only {only_a} | second-only {only_b} | neither {neither}")
    print(f"  exact McNemar p = {p:.4f}"
          + ("  (n=50 detects big effects only)" if len(common) <= 50 else ""))


def _experiments(_args: argparse.Namespace) -> None:
    """Print the paper E-numbered experiment table with resolved knobs + status."""
    print(f"{'E#':<4} {'section':<11} {'label':<26} {'cell':<36} {'knobs':<20} {'status'}")
    for num, entry in EXPERIMENTS.items():
        spec = get_cell(entry["cell"])
        knobs = ", ".join(f"{k}={v}" for k, v in spec.extra_dict.items()) or "(vendor default)"
        summary = _load_summary(entry["cell"])
        if summary is None:
            status = "— not run"
        else:
            agg = summary.get("aggregate") or {}
            excl = agg.get("excluded", 0)
            status = (f"SR {agg.get('success', float('nan')):.2f} "
                      f"n={agg.get('episode_count', 0)}" + (f" ({excl} excl)" if excl else ""))
        print(f"{num:<4} {entry['section']:<11} {entry['label']:<26} "
              f"{entry['cell']:<36} {knobs:<20} {status}")


def _drain(args: argparse.Namespace) -> None:
    """Ask a running cell to finish in-flight episodes then stop (no new pulls).
    Touches <run_dir>/DRAIN; the runner picks it up at the next episode
    boundary, flushes, and prints the un-run indices for the next resume."""
    run_dir = get_cell(args.cell).run_dir
    if args.run_name:
        run_dir = run_dir.parent / args.run_name
    if not run_dir.exists():
        sys.exit(f"[std] no run dir at {run_dir} — nothing to drain")
    (run_dir / "DRAIN").touch()
    print(f"[std] drain requested -> {run_dir / 'DRAIN'}")
    print("[std] workers finish their current episode, flush, then exit; "
          "un-pulled indices stay pending (runner prints the resume spec).")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run (or resume) one cell (name or E-number)")
    p_run.add_argument("cell", help="cell name (std_sdk_opus-4.8_bare_max) or paper E-number (E17)")
    p_run.add_argument("--servers", default="http://127.0.0.1:9200")
    p_run.add_argument("--episodes", default=None,
                       help="override indices for reruns/resume (default: frozen 0-99)")
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

    sub.add_parser("experiments", help="show the paper E-numbered experiment table")

    p_cmp = sub.add_parser("compare", help="same-episode paired McNemar")
    p_cmp.add_argument("cell_a")
    p_cmp.add_argument("cell_b")

    p_drain = sub.add_parser(
        "drain", help="signal a running cell to finish in-flight episodes then stop")
    p_drain.add_argument("cell")
    p_drain.add_argument("--run-name", default=None,
                         help="drain a nonstd run_name instead of the cell's standard run dir")

    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    elif args.cmd == "batch":
        asyncio.run(_batch(args))
    elif args.cmd == "board":
        _board(args)
    elif args.cmd == "experiments":
        _experiments(args)
    elif args.cmd == "compare":
        _compare(args)
    elif args.cmd == "drain":
        _drain(args)


if __name__ == "__main__":
    main()
