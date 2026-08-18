"""Rebuild R2R-CE rand100 as a faithful subset of official val_unseen.

HISTORICAL RECORD — applied 2026-08-17, kept for provenance. The rand100
SELECTION (which 100 episodes) is inherited from SmartWay's released
protocol subset; this script did not change it. What it changed: every
episode dict + GT entry was replaced VERBATIM from the official val_unseen
files (official start_rotation, official tokens, official GT), keeping the
episode-id ORDER (board indices 0-99 stable). The SmartWay-protocol
originals (randomized spawn headings + regenerated GT) were first preserved
as the loadable sibling split ``rand100_smartway/``.

The script refuses to run again: a rerun would overwrite the
rand100_smartway backup with the ALREADY-OFFICIAL content, destroying the
protocol record.
"""
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(REPO_ROOT / "data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed")
RAND = f"{ROOT}/rand100"
BAK = f"{ROOT}/rand100_smartway"

if os.path.exists(BAK):
    raise SystemExit(
        f"{BAK} exists — the fix was already applied (2026-08-17); rerunning "
        "would overwrite the SmartWay protocol backup with official content.")


def load(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ── 1. backup as a loadable split ──
os.makedirs(BAK, exist_ok=True)
for src, dst in ((f"{RAND}/rand100.json.gz", f"{BAK}/rand100_smartway.json.gz"),
                 (f"{RAND}/rand100_gt.json.gz", f"{BAK}/rand100_smartway_gt.json.gz")):
    shutil.copy2(src, dst)
    assert md5(src) == md5(dst), f"backup mismatch: {dst}"
print("backup ok -> rand100_smartway/ (md5-verified)")

# ── 2. rebuild from official ──
old = load(f"{RAND}/rand100.json.gz")
official = load(f"{ROOT}/val_unseen/val_unseen.json.gz")
official_gt = load(f"{ROOT}/val_unseen/val_unseen_gt.json.gz")

by_id = {ep["episode_id"]: ep for ep in official["episodes"]}
order = [ep["episode_id"] for ep in old["episodes"]]
missing = [i for i in order if i not in by_id]
assert not missing, f"ids not in official val_unseen: {missing}"

new = {k: v for k, v in official.items() if k != "episodes"}
new["episodes"] = [by_id[i] for i in order]
new_gt = {str(i): official_gt[str(i)] for i in order}

# ── 3. verify against old before writing ──
rot_changed = pos_same = txt_same = 0
for o, n in zip(old["episodes"], new["episodes"]):
    assert o["episode_id"] == n["episode_id"]
    if o["start_rotation"] != n["start_rotation"]:
        rot_changed += 1
    if o["start_position"] == n["start_position"]:
        pos_same += 1
    if o["instruction"]["instruction_text"] == n["instruction"]["instruction_text"]:
        txt_same += 1
print(f"episodes: {len(new['episodes'])} (order preserved by construction)")
print(f"positions identical: {pos_same}/100, instruction text identical: {txt_same}/100")
print(f"rotations changed vs SmartWay version: {rot_changed}/100")

with gzip.open(f"{RAND}/rand100.json.gz", "wt") as f:
    json.dump(new, f)
with gzip.open(f"{RAND}/rand100_gt.json.gz", "wt") as f:
    json.dump(new_gt, f)

# ── 4. re-read and confirm rotations now match official 100/100 ──
chk = load(f"{RAND}/rand100.json.gz")
ok = sum(1 for ep in chk["episodes"]
         if ep["start_rotation"] == by_id[ep["episode_id"]]["start_rotation"])
gt_chk = load(f"{RAND}/rand100_gt.json.gz")
gt_ok = sum(1 for i in order if gt_chk[str(i)] == official_gt[str(i)])
print(f"REWRITTEN: rotations match official {ok}/100, GT entries match official {gt_ok}/100")
