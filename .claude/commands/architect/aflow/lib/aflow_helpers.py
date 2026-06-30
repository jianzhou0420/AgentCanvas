"""aflow loop helpers — parent selection (softmax-mix), anti-replay, convergence.

Ports the four AFlow building blocks that the loop / proposer need:
- get_top_rounds        : top-K archive entries by score, with iter_0 force-included
- select_round          : softmax-mix sampling (p = lam·uniform + (1-lam)·softmax(alpha · score·100))
- check_modification    : anti-replay duplicate check (verbatim | lower_ws | embed)
- build_experience_map  : group archive entries by parent_iter_id into {success, failure}
- check_convergence     : top-k mean stability over `consecutive_rounds`

Verbatim references:
- third_party/AFlow/scripts/optimizer_utils/data_utils.py:40-109
- third_party/AFlow/scripts/optimizer_utils/experience_utils.py:12-80
- third_party/AFlow/scripts/optimizer_utils/convergence_utils.py:68-113
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Archive I/O
# ══════════════════════════════════════════════════════════════════════


def read_archive(archive_path: str | Path) -> list[dict]:
    p = Path(archive_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ══════════════════════════════════════════════════════════════════════
# Parent selection (data_utils.py:40-109)
# ══════════════════════════════════════════════════════════════════════


def get_top_rounds(archive: list[dict], K: int) -> list[dict]:
    """Top-K archive entries by score (desc), with iter_0 always included.

    Verbatim from upstream data_utils.py:40-59 (round_1 unconditional
    inclusion). Drops entries with score=None (failed iters never enter
    archive in aflow, but be defensive).
    """
    scored = [e for e in archive if e.get("score") is not None]
    if not scored:
        return []
    sorted_arc = sorted(scored, key=lambda e: e["score"], reverse=True)
    top = sorted_arc[:K]
    # Force-include iter_0 (the baseline) if not already in top
    baseline = next((e for e in scored if e["iter_id"] == "iter_0"), None)
    if baseline is not None and not any(e["iter_id"] == "iter_0" for e in top):
        # Replace the lowest-scored entry with iter_0
        top = sorted(top + [baseline], key=lambda e: e["score"], reverse=True)[:K]
        # Ensure iter_0 stays in
        if not any(e["iter_id"] == "iter_0" for e in top):
            top[-1] = baseline
    return top


def select_round(top_K: list[dict], alpha: float, lam: float, rng: np.random.Generator | None = None) -> dict:
    """Softmax-mix sample one parent from top-K.

    p = lam·uniform(1/K) + (1-lam)·softmax(alpha · score·100)

    Verbatim from upstream data_utils.py:61-109. The `·100` rescale is
    REQUIRED and load-bearing — without it `score` is a fraction in [0,1]
    and exp(alpha·delta)≈1 for any practical alpha (parent selection
    silently collapses to ≈uniform).
    """
    if rng is None:
        rng = np.random.default_rng()
    if not top_K:
        raise ValueError("select_round: empty top_K")
    if len(top_K) == 1:
        return top_K[0]
    scores = np.array([e["score"] * 100 for e in top_K])  # rescale to [0,100]
    # softmax(alpha · score)
    z = alpha * scores
    z = z - z.max()  # numerical stability
    softmax = np.exp(z) / np.exp(z).sum()
    uniform = np.ones(len(top_K)) / len(top_K)
    mix = lam * uniform + (1 - lam) * softmax
    mix = mix / mix.sum()  # renormalize (paranoid)
    idx = rng.choice(len(top_K), p=mix)
    return top_K[idx]


# ══════════════════════════════════════════════════════════════════════
# Anti-replay (experience_utils.py:12-80)
# ══════════════════════════════════════════════════════════════════════


def build_experience_map(archive: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Group archive entries by parent_iter_id into {success, failure}.

    For each entry e with parent_iter_id != null:
      - if e.score > parent.score: append modification to success
      - else: append to failure
    Skips entries with modification == "(baseline)" (sentinel).
    """
    by_id = {e["iter_id"]: e for e in archive}
    exp_map: dict[str, dict[str, list[str]]] = {}
    for e in archive:
        pid = e.get("parent_iter_id")
        if pid is None:
            continue  # baseline / no parent
        if e.get("modification") == "(baseline)":
            continue
        parent = by_id.get(pid)
        if parent is None or parent.get("score") is None:
            continue
        if pid not in exp_map:
            exp_map[pid] = {"success": [], "failure": []}
        mod = e.get("modification", "")
        if e.get("score", 0) > parent.get("score", 0):
            exp_map[pid]["success"].append(mod)
        else:
            exp_map[pid]["failure"].append(mod)
    return exp_map


def _normalize_mod(s: str) -> str:
    return " ".join(s.lower().split())


def check_modification(new_mod: str, parent_exp: dict[str, list[str]], norm: str = "lower_ws") -> bool:
    """Return True if new_mod is a duplicate (reject + resample).

    Verbatim from upstream experience_utils.py:69-80 with one
    non-default option (Q3 in algorithm.html): default norm="lower_ws"
    instead of upstream's "verbatim" string equality.
    """
    candidates = list(parent_exp.get("success", [])) + list(parent_exp.get("failure", []))
    if not candidates:
        return False
    if norm == "verbatim":
        return new_mod in candidates
    elif norm == "lower_ws":
        n = _normalize_mod(new_mod)
        return n in [_normalize_mod(c) for c in candidates]
    elif norm == "embed":
        raise NotImplementedError("replay_norm=embed requires an embedding model; not enabled by default")
    else:
        raise ValueError(f"unknown replay_norm: {norm!r}")


# ══════════════════════════════════════════════════════════════════════
# Convergence (convergence_utils.py:68-113)
# ══════════════════════════════════════════════════════════════════════


def check_convergence(archive: list[dict], top_k: int, z: float, consecutive_rounds: int) -> tuple[bool, int | None, int | None]:
    """Top-k mean stability over `consecutive_rounds`.

    Verbatim from upstream convergence_utils.py:68-113. Under F3
    (validation_rounds=1) each per-round std is 0, so `sigma_delta_y`
    is always 0 and the predicate `|delta_y| <= z·sigma_delta_y`
    reduces to `delta_y == 0` regardless of z. Convergence here fires
    only on exact equality of the top-k mean across consecutive rounds
    — advisory only. See loop.md § Termination + README § F3 deviation.
    """
    scored = [e for e in archive if e.get("score") is not None]
    avg_scores = [e["score"] for e in scored]
    stds = [0.0] * len(avg_scores)  # validation_rounds=1 → per-round std is 0
    if len(avg_scores) < top_k + 1:
        return False, None, None
    convergence_count = 0
    previous_y = sigma_y_previous = None
    for i in range(len(avg_scores)):
        # top-k indices over avg_scores[:i+1] (desc)
        idx = np.argsort(avg_scores[: i + 1])[::-1][:top_k]
        top_k_scores = [avg_scores[j] for j in idx]
        top_k_stds = [stds[j] for j in idx]
        y_current = float(np.mean(top_k_scores))
        sigma_y_current = float(np.sqrt(sum(s ** 2 for s in top_k_stds) / (top_k ** 2)))
        if previous_y is not None:
            delta_y = y_current - previous_y
            sigma_delta_y = float(np.sqrt(sigma_y_current ** 2 + sigma_y_previous ** 2))
            if abs(delta_y) <= z * sigma_delta_y:
                convergence_count += 1
                if convergence_count >= consecutive_rounds:
                    return True, i - consecutive_rounds + 1, i
            else:
                convergence_count = 0
        previous_y, sigma_y_previous = y_current, sigma_y_current
    return False, None, None


# ══════════════════════════════════════════════════════════════════════
# CLI for one-shot use from bash
# ══════════════════════════════════════════════════════════════════════


def _cli():
    import argparse
    import sys

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("select_parent")
    sp.add_argument("--archive", required=True)
    sp.add_argument("--K", type=int, default=4)
    sp.add_argument("--alpha", type=float, default=0.2)
    sp.add_argument("--lam", type=float, default=0.3)
    sp.add_argument("--seed", type=int, default=None)

    se = sub.add_parser("experience_for")
    se.add_argument("--archive", required=True)
    se.add_argument("--parent", required=True)

    sc = sub.add_parser("check_dup")
    sc.add_argument("--archive", required=True)
    sc.add_argument("--parent", required=True)
    sc.add_argument("--mod", required=True)
    sc.add_argument("--norm", default="lower_ws")

    cv = sub.add_parser("convergence")
    cv.add_argument("--archive", required=True)
    cv.add_argument("--top-k", type=int, default=3)
    cv.add_argument("--z", type=float, default=0.0)
    cv.add_argument("--consecutive", type=int, default=5)

    args = p.parse_args()
    arc = read_archive(args.archive)

    if args.cmd == "select_parent":
        top = get_top_rounds(arc, K=args.K)
        rng = np.random.default_rng(args.seed)
        parent = select_round(top, alpha=args.alpha, lam=args.lam, rng=rng)
        json.dump({
            "parent_iter_id": parent["iter_id"],
            "parent_score": parent["score"],
            "top_k": [{"iter_id": e["iter_id"], "score": e["score"]} for e in top],
        }, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.cmd == "experience_for":
        exp_map = build_experience_map(arc)
        json.dump(exp_map.get(args.parent, {"success": [], "failure": []}), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.cmd == "check_dup":
        exp_map = build_experience_map(arc)
        pe = exp_map.get(args.parent, {"success": [], "failure": []})
        is_dup = check_modification(args.mod, pe, norm=args.norm)
        json.dump({"is_duplicate": is_dup, "parent": args.parent, "candidates_n": len(pe["success"]) + len(pe["failure"])}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.cmd == "convergence":
        converged, cstart, cend = check_convergence(arc, top_k=args.top_k, z=args.z, consecutive_rounds=args.consecutive)
        json.dump({"converged": converged, "conv_start": cstart, "conv_end": cend, "archive_n": len(arc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    _cli()
