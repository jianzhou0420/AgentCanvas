"""Rebuild the RxR-CE ``rand100`` split from recorded run summaries.

This is the GENERATOR OF RECORD for coding-agent/splits/rxr/rand100/
(first applied 2026-08-18). The canonical episode set is the int-sorted
id list every recorded RxR board shares (verified identical across all 8
July-2026 runs, bare/wp x 4 models); this script subsets the official
val_unseen guide files with that list — episodes verbatim (official
rotations/GT), so the split never carries any protocol modification.

Why not make_rxr_rand100.py: that sampler DESIGNED the set (R2R-matched
scene quota + KS length matching) but its greedy polish is
RNG/code-version sensitive and does not reproduce the historical draw
(~81/100 overlap). Run records are the only faithful source.

Safe to rerun: if the output exists it is verified against the run ids
and left untouched (exits nonzero on mismatch instead of overwriting).
"""
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RXR = REPO_ROOT / "data/habitat/datasets/RxR_VLNCE_v0"
OUT = REPO_ROOT / "coding-agent/splits/rxr/rand100"
# any full-100 recorded run works; all 8 share the same set (verified 2026-08-18)
RUN_SUMMARY = REPO_ROOT / "outputs/beta-coding-agent/rxr_bare_opus5/summary.json"


def load_gz(fp):
    with gzip.open(fp, "rt") as f:
        return json.load(f)


def dump_gz(obj, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt") as f:
        json.dump(obj, f)


def canonical_ids():
    d = json.loads(RUN_SUMMARY.read_text())
    eps = sorted(d["episodes"], key=lambda e: e["index"])
    ids = [str(e["episode_id"]) for e in eps]
    assert len(ids) == 100 and ids == sorted(ids, key=int), "run record malformed"
    return ids


def main():
    ids = canonical_ids()
    out_guide = OUT / "rand100_guide.json.gz"
    if out_guide.exists():
        have = [str(e["episode_id"]) for e in load_gz(out_guide)["episodes"]]
        if have == ids:
            print(f"{out_guide} already canonical (100 ids match run records) — nothing to do")
            return
        raise SystemExit(f"{out_guide} exists but does NOT match the run-record "
                         "ids — refusing to overwrite; investigate first.")

    src = load_gz(RXR / "val_unseen/val_unseen_guide.json.gz")
    gt = load_gz(RXR / "val_unseen/val_unseen_guide_gt.json.gz")
    by_id = {str(e["episode_id"]): e for e in src["episodes"]}
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"ids not in official val_unseen: {missing}"

    out = {k: v for k, v in src.items() if k != "episodes"}
    out["episodes"] = [by_id[i] for i in ids]
    dump_gz(out, out_guide)
    dump_gz({i: gt[i] for i in ids}, OUT / "rand100_guide_gt.json.gz")
    # NDTW loads both roles' GT; we only run guide, follower stays empty
    dump_gz({}, OUT / "rand100_follower_gt.json.gz")

    chk = load_gz(out_guide)["episodes"]
    assert [str(e["episode_id"]) for e in chk] == ids
    assert all(chk[i] == by_id[ids[i]] for i in range(100)), "episode not verbatim official"
    print(f"rebuilt {out_guide}: 100 episodes, verbatim official, ids == run records")


if __name__ == "__main__":
    main()
