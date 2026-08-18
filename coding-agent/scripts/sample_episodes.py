"""Reproducible episode sampler for the ObjectNav-family and EQA boards.

Draws the frozen MIP-N evaluation subsets (named TEAPS-N before the
2026-08-03 rename, a942483 — same files, same method) for the benchmark
lines — hm3d (objectnav_hm3d_v1), mp3d (objectnav_mp3d_v1), ovon
(HM3D-OVON, each of its three val splits), hmeqa (HM-EQA), mt_hm3d
(MT-HM3D) and express (EXPRESS-Bench) — as scene-stratified proportional
random samples. N is the only knob: ``--n 100`` gives MIP-100 (the full
board), ``--n 60`` gives MIP-60 (the V1.0 long-horizon indicator subset).
Same code, same seed, every line — only N differs. Default seed 42.

Method (the paper-appendix wording):

1. POPULATION — the benchmark's standard val split, in habitat's own
   deterministic load order. This is the same ``env.episodes`` order the
   eval runner's ``episode_index`` selects into, so a published index is
   unambiguous for anyone holding the same episode files.
2. STRATA = SCENES. Each scene's quota is proportional to its share of the
   split (largest-remainder rounding so quotas sum to exactly N).
   Proportional allocation keeps the subset an unbiased estimator of
   full-split success rate — numbers stay directly comparable to published
   full-val results. (A naive prefix would cover 1-2 scenes: the episode
   files are scene-sharded.)
3. WITHIN A SCENE — simple random sampling without replacement from
   ``numpy.random.default_rng(seed)``, scenes visited in sorted(scene)
   order so RNG consumption order is fixed.
4. NO CATEGORY CONSTRAINT — within-scene random sampling preserves the
   split's category mix in expectation; the output records per-category
   counts for the population and the sample so the (non-)drift is
   auditable rather than enforced.

Reproduction:
    - habitat-lab 0.2.4 (the benchmark-standard generation; any env that
      runs it works — no simulator or GPU is touched, only episode files)
      plus numpy. OVON additionally needs this repo checked out: its
      dataset class is vendored at workspace/nodesets/env/env_ovon/.
    - episode files under data/datasets/ (paths in _BENCHMARKS below),
      resolved from the repo root.
    - python coding-agent/scripts/sample_episodes.py  # the full board plan
      python coding-agent/scripts/sample_episodes.py --benchmark hm3d
    - Output is byte-stable: same episode files + same seed -> identical
      JSON (no timestamps, fixed key order, sorted scene visit order).

Output JSON per (benchmark, split): metadata + audit tables + ``indices``
+ per-episode records keyed by (scene_id, episode_id) — episode_id alone
is unique per scene only.

``--materialize`` additionally writes the subsets as DATASET-LAYER splits
named MIP{N} (same form as R2R-CE's rand100): the physical files live in
coding-agent/splits/ (the single split home, 2026-08-18) with a symlink
dropped next to the official splits, so the env panel keeps discovering
them (split="mip100...", episodes indexed 0..N-1). Byte-stable too (gzip mtime pinned to 0; the
CSV corpora slice raw source lines; express re-serializes the parsed
records deterministically). The env-panel registrations (env_objnav
``_DATASET_SPECS``, env_ovon ``_SPLITS``, env_hmeqa ``_MIP_N``) enumerate
the same N values — keep them in sync when adding a new N. env_express
has no mip split registration yet — its derived file is written for the
coming coding-agent wiring.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirrors the env nodesets' dataset wiring (workspace/nodesets/env/
# env_objnav.py `_DATASET_SPECS`, env_ovon `_data_path`) — the sampler must
# load episodes through the exact same habitat config the eval env uses,
# or the published indices would point at different episodes.
_OVON_SPLIT_FILES = {
    "val_seen": "val_seen",
    "val_seen_synonyms": "val_unseen_easy",
    "val_unseen": "val_unseen_hard",
}
_BENCHMARKS: dict[str, dict] = {
    "hm3d": {
        "dataset_label": "objectnav_hm3d_v1",
        "bench_config": "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        "data_path": "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
        "splits": ["val", "val_mini"],
        "type": None,  # habitat stock ObjectNav-v1
    },
    "mp3d": {
        "dataset_label": "objectnav_mp3d_v1",
        "bench_config": "benchmark/nav/objectnav/objectnav_mp3d.yaml",
        "data_path": "data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
        "splits": ["val", "val_mini"],
        "type": None,
    },
    "ovon": {
        "dataset_label": "hm3d-ovon",
        "bench_config": "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        "data_path": "data/datasets/ovon/hm3d/{split}/{file}.json.gz",
        "splits": list(_OVON_SPLIT_FILES),
        "type": "OVON-v1",  # vendored class, registered on import below
    },
    # CSV question corpora (coding-agent EQA line): the population is the
    # released question CSV in FILE ORDER — exactly the order env_hmeqa's
    # manager loads, so episode_index i = CSV row i. No habitat involved
    # (pure csv + numpy); strata/categories come from the named columns.
    # --materialize writes the derived question CSV (questions_mip{n}.csv
    # next to the official one) that env_hmeqa's panel selects as
    # split="mip{n}" — same semantics as the objnav dataset-layer mip
    # files (aligned 2026-07-29 on user request; keep env_hmeqa._MIP_N in
    # sync). mt_hm3d rides the same nodeset via dataset=mt_hm3d, so its
    # derived CSV lands in the MT corpus dir under the same filename.
    "hmeqa": {
        "dataset_label": "hm-eqa",
        "bench_config": None,
        "data_path": "data/hm3d/hmeqa/questions.csv",
        "splits": ["val"],
        "type": "csv",
        "scene_col": "scene",
        "category_col": "label",
    },
    "mt_hm3d": {
        "dataset_label": "mt-hm3d",
        "bench_config": None,
        "data_path": "data/hm3d/mt_hm3d/MT-HM3D-contextual.csv",
        "splits": ["val"],  # = the full 1,587-question contextual release
                            # (same naming as env_hmeqa: "val" is the
                            # official CSV; every MT question is eval)
        "type": "csv",
        "scene_col": "scene",
        "category_col": "label",
        # LABEL-stratified, not scene-stratified (user decision
        # 2026-08-16): scene strata degenerate on this corpus — 828
        # scenes / 100 slots hands every slot to the question-richest
        # scenes, which skew Counting to 63% of the sample vs 19% of the
        # population. Stratifying on the 4-way label column pins the
        # question-type mix to the population's instead.
        "stratify_col": "label",
    },
    # EXPRESS-Bench (env_express): the population is express-bench.json in
    # ARRAY ORDER (env_express loads the flat list; episode ids count from
    # array order). Sampled from split "all" (2,044 records over 174
    # scenes) — the benchmark's train scenes are HM3D-split train, not
    # training data; published numbers use the full set. Strata = the
    # scene hash inside scene_id ("hm3d/train/00006-HkseAnWCgqk" ->
    # "00006-HkseAnWCgqk"); categories = the record's ``type`` field.
    "express": {
        "dataset_label": "express-bench",
        "bench_config": None,
        "data_path": "data/hm3d/express_bench/express-bench.json",
        "splits": ["all"],
        "type": "json",
    },
}

# every (benchmark, split) the boards freeze — the no-argument invocation
DEFAULT_PLAN = [("hm3d", "val"), ("mp3d", "val"),
                ("ovon", "val_seen"), ("ovon", "val_seen_synonyms"),
                ("ovon", "val_unseen"), ("hmeqa", "val"),
                ("mt_hm3d", "val"), ("express", "all")]


def _register_ovon_dataset() -> None:
    """Import the vendored OVON-v1 dataset class WITHOUT importing the
    env_ovon package __init__ (which needs the backend app on sys.path)."""
    path = (REPO_ROOT / "workspace" / "nodesets" / "env" / "env_ovon"
            / "_ovon_dataset.py")
    spec = importlib.util.spec_from_file_location("_ovon_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # registers "OVON-v1" in habitat's registry


def load_dataset(benchmark: str, split: str):
    """The (benchmark, split) dataset with episodes in habitat's load order
    — the order ``episode_index`` selects into at eval time."""
    import habitat
    from habitat.config import read_write
    from habitat.config.default import get_config

    spec = _BENCHMARKS[benchmark]
    if split not in spec["splits"]:
        raise SystemExit(f"unknown split {split!r} for {benchmark} "
                         f"(expected {spec['splits']})")
    if benchmark == "ovon":
        _register_ovon_dataset()
        data_path = spec["data_path"].format(
            split=split, file=_OVON_SPLIT_FILES[split])
    else:
        data_path = spec["data_path"]

    config = get_config(spec["bench_config"])
    with read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = data_path
        if spec["type"]:
            config.habitat.dataset.type = spec["type"]
    return habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)


# ── MIP{N}: the sampled subsets materialized as DATASET-LAYER splits ──
# Same form as R2R-CE's rand100: a derived split file the env panel can
# select (split="mip100..."), so eval indexes 0..N-1 into it. Files land
# next to the official splits; the habitat dataset classes serialize
# natively (to_json rebuilds goals_by_category from the kept episodes). gzip
# with mtime=0 keeps the artifact byte-stable across runs. Path is
# parameterized by N so MIP-100 and MIP-60 sit in sibling directories
# (mip100/ next to mip60/) and never collide.
_OVON_MIP_SUFFIX = {
    "val_seen": "seen",
    "val_seen_synonyms": "seen_synonyms",
    "val_unseen": "unseen",
}


def mip_split_path(benchmark: str, split: str, n: int) -> str:
    """Dataset-layer path for the materialized MIP-N split of
    (benchmark, split). Reproduces the original mip100 paths at n=100.
    The CSV corpora (hmeqa, mt_hm3d) derive a sibling CSV, express a
    sibling JSON, rather than a json.gz.

    Since 2026-08-18 this is the DISCOVERY path only — the env panels
    keep finding the split next to the official ones, but the entry is a
    symlink into coding-agent/splits/ (the single physical home for all
    split data; see split_home_path / _publish)."""
    tag = f"mip{n}"
    if benchmark in ("hmeqa", "mt_hm3d"):
        corpus_dir = Path(_BENCHMARKS[benchmark]["data_path"]).parent
        return str(corpus_dir / f"questions_{tag}.csv")
    if benchmark == "express":
        return f"data/hm3d/express_bench/express-bench_{tag}.json"
    if benchmark == "ovon":
        name = f"{tag}_{_OVON_MIP_SUFFIX[split]}"
        return f"data/datasets/ovon/hm3d/{name}/{name}.json.gz"
    return f"data/datasets/objectnav/{benchmark}/v1/{tag}/{tag}.json.gz"


def split_home_path(benchmark: str, split: str, n: int) -> Path:
    """Physical home under coding-agent/splits/ for the materialized MIP-N
    split — same file names as mip_split_path, grouped by benchmark."""
    home = REPO_ROOT / "coding-agent" / "splits"
    tag = f"mip{n}"
    if benchmark in ("hmeqa", "mt_hm3d"):
        return home / benchmark / f"questions_{tag}.csv"
    if benchmark == "express":
        return home / "express" / f"express-bench_{tag}.json"
    if benchmark == "ovon":
        name = f"{tag}_{_OVON_MIP_SUFFIX[split]}"
        return home / "ovon" / name / f"{name}.json.gz"
    return home / "objectnav" / benchmark / tag / f"{tag}.json.gz"


def _publish(benchmark: str, split: str, n: int) -> Path:
    """Return the physical output path (coding-agent/splits/) and ensure the
    dataset-tree discovery path is a symlink to it. objnav/ovon link the
    split DIRECTORY (env discovery lists dirs); the CSV/JSON corpora link
    the file. Refuses to clobber a real (non-symlink) entry — relocate it
    into coding-agent/splits/ by hand first."""
    home = split_home_path(benchmark, split, n)
    home.parent.mkdir(parents=True, exist_ok=True)
    tree = REPO_ROOT / mip_split_path(benchmark, split, n)
    link, target = ((tree.parent, home.parent) if tree.suffix == ".gz"
                    else (tree, home))
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise SystemExit(f"{link} is a symlink to {os.readlink(link)}, "
                             f"expected {target}")
    elif link.exists():
        raise SystemExit(f"{link} is a real file/dir — move it into "
                         f"{target} and symlink it before materializing")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    return home


def materialize_csv(benchmark: str, indices: list[int], n: int) -> Path:
    """Write a CSV corpus's MIP-N derived question CSV: the header plus
    the sampled rows in ascending original-index order, original bytes
    preserved — row k of the derived file = original row ``indices[k]``,
    so ``split="mip{n}"`` + episode_index 0..N-1 selects the manifest
    order exactly like the objnav mip files."""
    import csv

    src = REPO_ROOT / _BENCHMARKS[benchmark]["data_path"]
    with open(src, newline="") as f:
        rows = sum(1 for _ in csv.DictReader(f, skipinitialspace=True))
    lines = src.read_text().splitlines(keepends=True)
    # one physical line per CSV row (header + rows; quoted fields carry no
    # embedded newlines) — a mismatch means the assumption broke, refuse.
    if len(lines) != rows + 1:
        raise SystemExit(f"{benchmark}: {src} has {len(lines)} physical "
                         f"lines, expected {rows + 1} (header + {rows} rows)")
    out_lines = [lines[0]] + [lines[i + 1] for i in sorted(indices)]
    if not out_lines[-1].endswith("\n"):
        out_lines[-1] += "\n"
    out = _publish(benchmark, "val", n)
    out.write_text("".join(out_lines))
    return out


def materialize_express(indices: list[int], n: int) -> Path:
    """Write express's MIP-N derived record list: the sampled records in
    ascending original-index order, re-serialized deterministically
    (indent=2, key order as loaded) — entry k = original array position
    ``indices[k]``."""
    src = REPO_ROOT / _BENCHMARKS["express"]["data_path"]
    records = json.loads(src.read_text())
    subset = [records[i] for i in sorted(indices)]
    out = _publish("express", "all", n)
    out.write_text(json.dumps(subset, indent=2) + "\n")
    return out


def materialize(benchmark: str, split: str, indices: list[int], n: int) -> Path:
    """Write the MIP-N dataset file for (benchmark, split): the sampled
    episodes, in ascending official-val index order."""
    import gzip

    dataset = load_dataset(benchmark, split)
    dataset.episodes = [dataset.episodes[i] for i in indices]
    if benchmark == "ovon":
        # OVON's to_json serializes the WHOLE goals_by_category dict (every
        # category of the full split), so (a) prune it to the kept episodes'
        # goals_keys, and (b) repair upstream's non-idempotent round-trip:
        # __deserialize_goal does int(object_id.split("_")[-1]), so a file
        # re-serialized with int object_ids crashes the next load — write
        # them back as strings ("123" round-trips cleanly). The vendored
        # dataset class itself stays verbatim per its do-not-edit note.
        keep = {ep.goals_key for ep in dataset.episodes}
        dataset.goals_by_category = {
            k: v for k, v in dataset.goals_by_category.items() if k in keep}
        for goals in dataset.goals_by_category.values():
            for goal in goals:
                if isinstance(goal.object_id, int):
                    goal.object_id = str(goal.object_id)
    payload = dataset.to_json().encode()
    out = _publish(benchmark, split, n)
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        fh.write(payload)
    return out


def largest_remainder_quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional quotas summing to exactly n. Ties on the fractional
    part break by scene key (sorted), keeping the result deterministic."""
    total = sum(counts.values())
    raw = {s: n * c / total for s, c in counts.items()}
    quotas = {s: int(raw[s]) for s in counts}
    short = n - sum(quotas.values())
    by_frac = sorted(counts, key=lambda s: (-(raw[s] - quotas[s]), s))
    for s in by_frac[:short]:
        quotas[s] += 1
    return quotas


def sample_split(benchmark: str, split: str, n: int, seed: int) -> dict:
    episodes = load_dataset(benchmark, split).episodes
    if n > len(episodes):
        raise SystemExit(f"{benchmark}/{split}: asked n={n} of {len(episodes)}")

    by_scene: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_scene.setdefault(ep.scene_id, []).append(i)

    quotas = largest_remainder_quotas(
        {s: len(v) for s, v in by_scene.items()}, n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for scene in sorted(by_scene):  # fixed visit order -> fixed RNG stream
        q = quotas[scene]
        if q:
            picked.extend(sorted(
                rng.choice(by_scene[scene], size=q, replace=False).tolist()))
    picked.sort()

    def cat_counts(idx_iter) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in idx_iter:
            c = getattr(episodes[i], "object_category", "") or ""
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    def short_scene(scene_id: str) -> str:
        return Path(scene_id).stem.split(".")[0]

    # CAVEAT episode_id: habitat's ObjectNavDatasetV1.from_json REASSIGNS
    # episode_id positionally on every load (episode_id = str(i)), so for
    # hm3d/mp3d it just mirrors the load index and is NOT a stable identity
    # across derived files. True identity = (scene_id, start_position);
    # geodesic_distance doubles as a content fingerprint.
    records = [{
        "index": i,
        "scene_id": episodes[i].scene_id,
        "episode_id": episodes[i].episode_id,
        "object_category": getattr(episodes[i], "object_category", ""),
        "start_position": [float(x) for x in episodes[i].start_position],
        "geodesic_distance": (episodes[i].info or {}).get("geodesic_distance"),
    } for i in picked]

    return {
        "generator": "coding-agent/sample_episodes.py",
        "method": ("scene-stratified proportional random sample "
                   "(largest-remainder quotas), without replacement, "
                   "numpy default_rng, scenes visited in sorted order"),
        "benchmark": benchmark,
        "dataset_label": _BENCHMARKS[benchmark]["dataset_label"],
        "split": split,
        "seed": seed,
        "n": n,
        "population": {
            "episodes": len(episodes),
            "scenes": len(by_scene),
            "categories": cat_counts(range(len(episodes))),
        },
        "sample": {
            "scenes": {short_scene(s): quotas[s]
                       for s in sorted(by_scene) if quotas[s]},
            "categories": cat_counts(picked),
        },
        "indices": picked,
        "episodes": records,
    }


def sample_csv(benchmark: str, split: str, n: int, seed: int) -> dict:
    """CSV-corpus counterpart of sample_split — same method (scene-
    stratified proportional sample, largest-remainder quotas, sorted-scene
    RNG order), same output shape, but the population is the question CSV
    in file order (env_hmeqa's episode_index i = CSV row i) and needs no
    habitat. Serves hmeqa and mt_hm3d (same columns, different corpus)."""
    import csv

    spec = _BENCHMARKS[benchmark]
    if split not in spec["splits"]:
        raise SystemExit(f"unknown split {split!r} for {benchmark} "
                         f"(expected {spec['splits']})")
    q_path = REPO_ROOT / spec["data_path"]
    with open(q_path, newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if n > len(rows):
        raise SystemExit(f"{benchmark}/{split}: asked n={n} of {len(rows)}")

    scene_col, category_col = spec["scene_col"], spec["category_col"]
    strat_col = spec.get("stratify_col", scene_col)
    strat_word = "scene" if strat_col == scene_col else strat_col
    by_stratum: dict[str, list[int]] = {}
    by_scene: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_stratum.setdefault(row[strat_col], []).append(i)
        by_scene.setdefault(row[scene_col], []).append(i)

    quotas = largest_remainder_quotas(
        {s: len(v) for s, v in by_stratum.items()}, n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum in sorted(by_stratum):  # fixed visit order -> fixed RNG stream
        q = quotas[stratum]
        if q:
            picked.extend(sorted(
                rng.choice(by_stratum[stratum], size=q, replace=False).tolist()))
    picked.sort()

    def label_counts(idx_iter) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in idx_iter:
            c = rows[i].get(category_col, "") or ""
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    def scene_counts(idx_iter) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in idx_iter:
            out[rows[i][scene_col]] = out.get(rows[i][scene_col], 0) + 1
        return dict(sorted(out.items()))

    records = [{
        "index": i,
        "scene": rows[i][scene_col],
        "floor": rows[i]["floor"],
        "label": rows[i].get(category_col, ""),
        "question": rows[i]["question"],
        "answer": rows[i]["answer"],
    } for i in picked]

    # scene strata: the frozen manifest shape (sample.scenes = the quota
    # table). Other strata keep the shape's keys but sample.scenes becomes
    # the OBSERVED per-scene counts and sample.strata carries the quotas.
    if strat_col == scene_col:
        sample_audit = {
            "scenes": {s: quotas[s] for s in sorted(by_scene) if quotas[s]},
            "categories": label_counts(picked),
        }
    else:
        sample_audit = {
            "strata": {s: quotas[s] for s in sorted(by_stratum) if quotas[s]},
            "scenes": scene_counts(picked),
            "categories": label_counts(picked),
        }

    return {
        "generator": "coding-agent/sample_episodes.py",
        "method": (f"{strat_word}-stratified proportional random sample "
                   "(largest-remainder quotas), without replacement, "
                   f"numpy default_rng, {strat_word}s visited in sorted order"),
        "benchmark": benchmark,
        "dataset_label": spec["dataset_label"],
        "split": split,
        "seed": seed,
        "n": n,
        "population": {
            "episodes": len(rows),
            "scenes": len(by_scene),
            "categories": label_counts(range(len(rows))),
        },
        "sample": sample_audit,
        "indices": picked,
        "episodes": records,
    }


def sample_express(split: str, n: int, seed: int) -> dict:
    """EXPRESS-Bench counterpart of sample_split — same method, same
    output shape; the population is express-bench.json in ARRAY ORDER
    (env_express's episode ids count from array order) and needs no
    habitat. Strata = the scene hash inside scene_id; categories = the
    record's ``type`` field."""
    spec = _BENCHMARKS["express"]
    if split not in spec["splits"]:
        raise SystemExit(f"unknown split {split!r} for express "
                         f"(expected {spec['splits']})")
    records_all = json.loads((REPO_ROOT / spec["data_path"]).read_text())
    if n > len(records_all):
        raise SystemExit(f"express/{split}: asked n={n} of {len(records_all)}")

    def scene_of(rec: dict) -> str:
        # "hm3d/train/00006-HkseAnWCgqk" -> "00006-HkseAnWCgqk"
        return str(rec.get("scene_id", "")).split("/")[-1]

    by_scene: dict[str, list[int]] = {}
    for i, rec in enumerate(records_all):
        by_scene.setdefault(scene_of(rec), []).append(i)

    quotas = largest_remainder_quotas(
        {s: len(v) for s, v in by_scene.items()}, n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for scene in sorted(by_scene):  # fixed visit order -> fixed RNG stream
        q = quotas[scene]
        if q:
            picked.extend(sorted(
                rng.choice(by_scene[scene], size=q, replace=False).tolist()))
    picked.sort()

    def type_counts(idx_iter) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in idx_iter:
            c = str(records_all[i].get("type", "") or "")
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    ep_records = [{
        "index": i,
        "scene_id": records_all[i]["scene_id"],
        "episode_id": records_all[i]["episode_id"],
        "trajectory_id": records_all[i]["trajectory_id"],
        "type": records_all[i]["type"],
        "question": records_all[i]["question"],
        "answer": records_all[i]["answer"],
        "geodesic_distance": records_all[i]["geodesic_distance"],
    } for i in picked]

    return {
        "generator": "coding-agent/sample_episodes.py",
        "method": ("scene-stratified proportional random sample "
                   "(largest-remainder quotas), without replacement, "
                   "numpy default_rng, scenes visited in sorted order"),
        "benchmark": "express",
        "dataset_label": spec["dataset_label"],
        "split": split,
        "seed": seed,
        "n": n,
        "population": {
            "episodes": len(records_all),
            "scenes": len(by_scene),
            "categories": type_counts(range(len(records_all))),
        },
        "sample": {
            "scenes": {s: quotas[s] for s in sorted(by_scene) if quotas[s]},
            "categories": type_counts(picked),
        },
        "indices": picked,
        "episodes": ep_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=list(_BENCHMARKS),
                    help="one benchmark only (default: the full board plan)")
    ap.add_argument("--split", help="one split (requires --benchmark)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(
        REPO_ROOT / "coding-agent" / "splits"))
    ap.add_argument("--materialize", action="store_true",
                    help="also write the MIP-N dataset-layer split files "
                         "(derived files next to the official splits; "
                         "mip{n} named after --n)")
    args = ap.parse_args()

    # habitat resolves relative data paths from CWD, same as the nodesets
    os.chdir(REPO_ROOT)

    if args.split and not args.benchmark:
        raise SystemExit("--split requires --benchmark")
    plan = ([(args.benchmark, args.split)] if args.split
            else [(b, s) for b, s in DEFAULT_PLAN if
                  args.benchmark in (None, b)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for benchmark, split in plan:
        kind = _BENCHMARKS[benchmark]["type"]
        result = (sample_csv(benchmark, split, args.n, args.seed)
                  if kind == "csv"
                  else sample_express(split, args.n, args.seed)
                  if kind == "json"
                  else sample_split(benchmark, split, args.n, args.seed))
        name = f"{benchmark}_{split}_n{args.n}_seed{args.seed}.json"
        path = out_dir / name
        path.write_text(json.dumps(result, indent=2) + "\n")
        pop, samp = result["population"], result["sample"]
        shown = (path.relative_to(REPO_ROOT)
                 if path.is_relative_to(REPO_ROOT) else path)
        print(f"[sample] {benchmark}/{split}: {args.n}/{pop['episodes']} eps "
              f"over {len(samp['scenes'])}/{pop['scenes']} scenes, "
              f"{len(samp['categories'])}/{len(pop['categories'])} categories "
              f"-> {shown}")
        if args.materialize:
            out = (materialize_csv(benchmark, result["indices"], args.n)
                   if kind == "csv"
                   else materialize_express(result["indices"], args.n)
                   if kind == "json"
                   else materialize(benchmark, split, result["indices"], args.n))
            print(f"[mip{args.n}] {benchmark}/{split} -> "
                  f"{out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
