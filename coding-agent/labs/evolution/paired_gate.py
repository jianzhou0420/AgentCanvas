"""paired_gate — same-episode paired judge for two run dirs (eharness-evo M-line).

The measurement backbone of the evolution loop: candidate arm vs parent arm,
paired by episode_id on the same frozen split, judged ONLY on discordant pairs
with a one-sided exact McNemar test (12-line kernel ported from Zetta
gating.py:12-23). Absolute "rescued >= K" criteria are banned here by design:
same-arm reruns of Claude executors flip 18-23/100 episodes (measured 08-21),
so a flip is evidence only in aggregate, never alone.

Usage:
  python paired_gate.py CANDIDATE_RUN_DIR PARENT_RUN_DIR [--alpha 0.025]
                        [--metric success] [--json OUT.json] [--flips]

Reads <run_dir>/summary.json (the unified driver schema: episodes[] with
episode_id + metrics.success). Works across beta-coding-agent and
beta-eharness boards alike. Exit code: 0 = report produced (pass or fail is in
the report, not the exit code); 2 = inputs unusable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def one_sided_exact_mcnemar(candidate_wins: int, parent_wins: int) -> float:
    """P[X >= candidate_wins], X ~ Binomial(d, 0.5), d = discordant count.

    Exact (math.comb), no approximation. d == 0 -> p = 1.0.
    """
    d = candidate_wins + parent_wins
    if d == 0:
        return 1.0
    return sum(math.comb(d, k) for k in range(candidate_wins, d + 1)) / (2 ** d)


def load_board(run_dir: Path) -> dict:
    summary = run_dir / "summary.json"
    if not summary.exists():
        sys.exit(f"[paired_gate] no summary.json under {run_dir}")
    d = json.loads(summary.read_text())
    eps = {}
    for e in d.get("episodes", []):
        m = e.get("metrics") or {}
        if "success" not in m:
            continue  # unscored episode (infra failure etc.) — never a 0
        key = str(e.get("episode_id", e.get("index")))
        eps[key] = {
            "success": float(m["success"]) >= 0.5,
            "spl": m.get("spl"),
            "index": e.get("index"),
        }
    return {
        "run": d.get("run_name", run_dir.name),
        "split": (d.get("config") or {}).get("split"),
        "episodes": eps,
        "n_total": len(d.get("episodes", [])),
    }


def compare(cand: dict, parent: dict, alpha: float) -> dict:
    shared = sorted(set(cand["episodes"]) & set(parent["episodes"]), key=str)
    if not shared:
        sys.exit("[paired_gate] zero shared episode_ids — wrong boards?")
    cw, pw, flips = 0, 0, []
    c_succ = p_succ = 0
    for k in shared:
        c = cand["episodes"][k]["success"]
        p = parent["episodes"][k]["success"]
        c_succ += c
        p_succ += p
        if c and not p:
            cw += 1
            flips.append({"episode_id": k, "flip": "candidate_rescued"})
        elif p and not c:
            pw += 1
            flips.append({"episode_id": k, "flip": "candidate_lost"})
    n = len(shared)
    pval = one_sided_exact_mcnemar(cw, pw)
    return {
        "candidate": cand["run"],
        "parent": parent["run"],
        "split": {"candidate": cand["split"], "parent": parent["split"],
                  "match": cand["split"] == parent["split"]},
        "n_paired": n,
        "n_unpaired": {"candidate_only": len(cand["episodes"]) - n,
                       "parent_only": len(parent["episodes"]) - n},
        "sr": {"candidate": round(c_succ / n, 4), "parent": round(p_succ / n, 4),
               "delta": round((c_succ - p_succ) / n, 4)},
        "discordant": {"total": cw + pw, "candidate_wins": cw, "parent_wins": pw},
        "p_one_sided_exact_mcnemar": round(pval, 6),
        "alpha": alpha,
        "significant": pval < alpha,
        "noise_note": ("same-arm rerun flips measured at 18-23/100 for Claude "
                       "executors (08-21); a flip is evidence only in aggregate. "
                       "Mechanism attribution (did the delta actually fire in "
                       "each rescued episode?) is a separate check, not implied "
                       "by this p-value."),
        "flips": flips,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate", type=Path)
    ap.add_argument("parent", type=Path)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--json", type=Path, default=None, help="write full report")
    ap.add_argument("--flips", action="store_true", help="print flip list")
    args = ap.parse_args()

    report = compare(load_board(args.candidate), load_board(args.parent), args.alpha)

    r = report
    print(f"paired_gate  {r['candidate']}  vs  {r['parent']}")
    if not r["split"]["match"]:
        print(f"  !! split mismatch: {r['split']['candidate']} vs {r['split']['parent']}")
    print(f"  paired {r['n_paired']} eps  (unpaired: {r['n_unpaired']})")
    print(f"  SR {r['sr']['candidate']:.3f} vs {r['sr']['parent']:.3f}  ΔSR {r['sr']['delta']:+.3f}")
    d = r["discordant"]
    print(f"  discordant {d['total']}  ({d['candidate_wins']} rescued / {d['parent_wins']} lost)")
    print(f"  one-sided exact McNemar p = {r['p_one_sided_exact_mcnemar']}"
          f"  ->  {'SIGNIFICANT' if r['significant'] else 'not significant'} at α={r['alpha']}")
    print(f"  note: {r['noise_note']}")
    if args.flips:
        for f in r["flips"]:
            print(f"    {f['flip']:18s} ep {f['episode_id']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  report -> {args.json}")


if __name__ == "__main__":
    main()
