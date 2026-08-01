#!/usr/bin/env python3
"""Build an RxR-CE ``rand100`` subset that mirrors R2R-CE's ``rand100``.

Two goals, one hard + one soft:

1. HARD — per-scene episode counts EQUAL R2R-CE's ``rand100`` (computed live
   from the R2R file, not hard-coded). R2R and RxR share the same 11 val_unseen
   MP3D scenes, so the quota maps across directly. This makes the two rand100s
   scene-matched: any R2R-vs-RxR gap isn't confounded by a different scene mix.

2. SOFT — within each scene's quota, choose episodes so the sample's
   INSTRUCTION-LENGTH (words) and TRAJECTORY-LENGTH (reference_path metres)
   distributions stay close to the full English val_unseen population. This
   matters because scenes differ in length character (2azQ1b91cZZ ~118 words /
   19 m vs Z6MFQCViBuw ~66 words / 12.5 m), so R2R's scene weighting alone would
   skew the length marginals; we compensate by which episodes we pick per scene.

Method: best-of-N stratified random draw (respecting the quota) to seed, then a
greedy within-scene swap polish, both minimising ks_instr + ks_traj (two-sample
Kolmogorov–Smirnov vs the en population). Swaps stay inside a scene, so the hard
quota is preserved exactly. Deterministic (fixed SEED).

Only English (``instruction.language`` in en-US / en-IN — what rxr_cma_en.yaml's
LANGUAGES filter keeps) is sampled. Outputs (role=guide, per the RxR task yaml's
{split}_{role} template):

    RxR_VLNCE_v0/rand100/rand100_guide.json.gz       (100 episodes, sorted)
    RxR_VLNCE_v0/rand100/rand100_guide_gt.json.gz     (matching GT subset)
    RxR_VLNCE_v0/rand100/rand100_follower_gt.json.gz  (empty; NDTW loads both
        roles' GT and we only run guide — see measures.py)
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# ── knobs ──
R2R = Path("data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed")
RXR = Path("data/habitat/datasets/RxR_VLNCE_v0")
SPLIT_SRC = "val_unseen"
ROLE = "guide"
EN_PREFIX = "en"
SEED = 0
N_INIT = 20000     # best-of-N stratified draws to seed
N_POLISH = 60000   # greedy within-scene swap attempts


def _load(fp: Path) -> dict:
    with gzip.open(fp, "rt") as f:
        return json.load(f)


def _dump(obj: dict, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt") as f:
        json.dump(obj, f)


def _scene(ep: dict) -> str:
    return ep["scene_id"].split("/")[-1].replace(".glb", "")


def _instr_words(ep: dict) -> int:
    return len((ep["instruction"]["instruction_text"] or "").split())


def _traj_len(ep: dict) -> float:
    p = ep.get("reference_path") or []
    return float(sum(math.dist(a, b) for a, b in zip(p, p[1:])))


def _ks(sample: np.ndarray, pop_sorted: np.ndarray) -> float:
    """Two-sample KS: max |F_sample - F_pop|. ``sample`` need not be sorted."""
    s = np.sort(sample)
    grid = np.concatenate([s, pop_sorted])
    cs = np.searchsorted(s, grid, side="right") / len(s)
    cp = np.searchsorted(pop_sorted, grid, side="right") / len(pop_sorted)
    return float(np.abs(cs - cp).max())


def main() -> None:
    # ── R2R per-scene quota (the hard constraint) ──
    r2r_eps = _load(R2R / "rand100" / "rand100.json.gz")["episodes"]
    quota = Counter(_scene(e) for e in r2r_eps)           # {scene: count}
    quota = {s: c for s, c in quota.items() if c > 0}

    # ── RxR English val_unseen population + features ──
    src = _load(RXR / SPLIT_SRC / f"{SPLIT_SRC}_{ROLE}.json.gz")
    gt = _load(RXR / SPLIT_SRC / f"{SPLIT_SRC}_{ROLE}_gt.json.gz")
    en = [
        e for e in src["episodes"]
        if (e.get("instruction", {}) or {}).get("language", "").startswith(EN_PREFIX)
    ]
    pop_instr = np.sort(np.array([_instr_words(e) for e in en], dtype=float))
    pop_traj = np.sort(np.array([_traj_len(e) for e in en], dtype=float))

    # per-scene pools (feature arrays + back-references into `en`)
    scenes = list(quota.keys())
    pool: dict[str, dict] = {}
    for s in scenes:
        idx = [i for i, e in enumerate(en) if _scene(e) == s]
        if len(idx) < quota[s]:
            raise SystemExit(f"scene {s}: en pool {len(idx)} < R2R quota {quota[s]}")
        pool[s] = {
            "gidx": np.array(idx),
            "instr": np.array([_instr_words(en[i]) for i in idx], dtype=float),
            "traj": np.array([_traj_len(en[i]) for i in idx], dtype=float),
        }

    rng = np.random.default_rng(SEED)

    def score(sel: dict) -> float:
        si = np.concatenate([pool[s]["instr"][sel[s]] for s in scenes])
        st = np.concatenate([pool[s]["traj"][sel[s]] for s in scenes])
        return _ks(si, pop_instr) + _ks(st, pop_traj)

    # ── best-of-N stratified init ──
    best_sel, best_score = None, math.inf
    for _ in range(N_INIT):
        sel = {s: rng.choice(len(pool[s]["gidx"]), quota[s], replace=False) for s in scenes}
        sc = score(sel)
        if sc < best_score:
            best_score, best_sel = sc, {s: sel[s].copy() for s in scenes}
    print(f"best-of-{N_INIT} init: combined KS = {best_score:.4f}")

    # ── greedy within-scene swap polish ──
    sel = {s: set(best_sel[s].tolist()) for s in scenes}
    swap_scenes = [s for s in scenes if len(pool[s]["gidx"]) > quota[s]]

    def score_sets(sel_sets: dict) -> float:
        si = np.concatenate([pool[s]["instr"][list(sel_sets[s])] for s in scenes])
        st = np.concatenate([pool[s]["traj"][list(sel_sets[s])] for s in scenes])
        return _ks(si, pop_instr) + _ks(st, pop_traj)

    cur = score_sets(sel)
    improved = 0
    for _ in range(N_POLISH):
        s = swap_scenes[rng.integers(len(swap_scenes))]
        inside = list(sel[s])
        outside = [i for i in range(len(pool[s]["gidx"])) if i not in sel[s]]
        a = inside[rng.integers(len(inside))]
        b = outside[rng.integers(len(outside))]
        sel[s].discard(a); sel[s].add(b)
        new = score_sets(sel)
        if new < cur - 1e-12:
            cur = new; improved += 1
        else:
            sel[s].discard(b); sel[s].add(a)
    print(f"after {N_POLISH} swap attempts ({improved} accepted): combined KS = {cur:.4f}")

    # ── materialize selection ──
    chosen = [en[int(pool[s]["gidx"][i])] for s in scenes for i in sel[s]]
    chosen.sort(key=lambda e: int(e["episode_id"]))
    out = {k: v for k, v in src.items() if k != "episodes"}
    out["episodes"] = chosen
    out_gt = {str(e["episode_id"]): gt[str(e["episode_id"])] for e in chosen}

    _dump(out, RXR / "rand100" / f"rand100_{ROLE}.json.gz")
    _dump(out_gt, RXR / "rand100" / f"rand100_{ROLE}_gt.json.gz")
    _dump({}, RXR / "rand100" / "rand100_follower_gt.json.gz")

    # ── report ──
    smp_instr = np.array([_instr_words(e) for e in chosen], dtype=float)
    smp_traj = np.array([_traj_len(e) for e in chosen], dtype=float)
    smp_scene = Counter(_scene(e) for e in chosen)

    def qs(a):
        a = np.sort(a)
        p = lambda q: a[min(len(a) - 1, int(q * len(a)))]
        return f"med={np.median(a):.1f} p25={p(.25):.1f} p75={p(.75):.1f} p90={p(.9):.1f} mean={a.mean():.1f}"

    print("\nper-scene quota (RxR sampled / R2R rand100) — must match:")
    for s in sorted(quota, key=lambda s: -quota[s]):
        flag = "" if smp_scene[s] == quota[s] else "  <-- MISMATCH"
        print(f"  {s:16s} {smp_scene[s]:3d} / {quota[s]:3d}{flag}")
    print(f"  TOTAL {sum(smp_scene.values())} / {sum(quota.values())}")
    print("\ninstruction length (words):")
    print(f"  en val_unseen : {qs(pop_instr)}")
    print(f"  rand100       : {qs(smp_instr)}   KS={_ks(smp_instr, pop_instr):.4f}")
    print("trajectory length (reference_path m):")
    print(f"  en val_unseen : {qs(pop_traj)}")
    print(f"  rand100       : {qs(smp_traj)}   KS={_ks(smp_traj, pop_traj):.4f}")


if __name__ == "__main__":
    main()
