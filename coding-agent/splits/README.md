# coding-agent/splits — evaluation split home

Single physical home for every evaluation split the coding-agent boards
run (2026-08-18). The dataset trees under `data/` keep **relative
symlinks** at the original locations, so env loaders and split discovery
resolve unchanged; the bytes live here. Generators live in
`../scripts/`; run them from the repo root.

Layout: the flat `*_seed42.json` files are sampling manifests (audit
records: indices + per-episode keys + population/sample category tables);
the subdirectories hold the materialized split files the envs load.

## Provenance

| Split | Selection | How it was drawn |
|---|---|---|
| `r2r/rand100/` | **inherited** (SmartWay) | The 100-episode selection is SmartWay's released R2R-CE protocol subset (shared by OpenNav/AgenticNav); their sampling method is not published. Contents rebuilt 2026-08-17 from official val_unseen (`../scripts/fix_rand100_official.py`): same ids, same order, episodes + GT verbatim official (annotation-aligned spawn rotations, official tokens). |
| `r2r/rand100_smartway/` | **inherited** (SmartWay, verbatim) | Byte-preserved copy of SmartWay's original files before the 2026-08-17 rebuild: uniformly randomized spawn headings + regenerated GT. Kept loadable for protocol-comparable reruns against SmartWay/OpenNav/AgenticNav published rows. |
| `r2r/heldout100/` | **ours** | Faithful val_unseen slice (episodes verbatim official), near-proportional scene quota, **disjoint from rand100** (0 overlap) — built as the held-out generalization set. Generator not archived; the files are canonical. |
| `rxr/rand100/` | **ours** | 100 English val_unseen episodes, scene quota hard-matched to R2R rand100 + instruction/trajectory-length KS matching (`../scripts/make_rxr_rand100.py`, method doc only — its greedy polish does not reproduce the historical draw). Canonical set = the id list shared by all recorded July-2026 boards; restore only via `../scripts/rebuild_rxr_rand100_from_runs.py`. Episodes verbatim official (rotations/GT untouched). |
| `objectnav/{hm3d,mp3d}/mip100/` | **ours** | `../scripts/sample_episodes.py --materialize`: scene-stratified proportional quotas (largest-remainder), within-scene simple random, seed 42, from the official val splits. Byte-stable (gzip mtime=0). |
| `ovon/mip100_{seen,seen_synonyms,unseen}/` | **ours** | Same sampler over HM3D-OVON's three val splits. |
| `hmeqa/questions_mip100.csv` | **ours** | Same sampler over the HM-EQA question CSV (scene-stratified); derived CSV slices raw source lines, byte-stable. |
| `mt_hm3d/questions_mip100.csv` | **ours** | Same sampler over MT-HM3D contextual questions, stratified on the question-type label (scene strata degenerate on this corpus). |
| `express/express-bench_mip100.json` | **ours** | Same sampler over EXPRESS-Bench (full 2,044-record set), scene-stratified. |
| `*_seed42.json` (13 files) | **ours** | Audit manifests for the MIP-N samples above (`n60` variants have manifests only; not materialized). Manifest + generator regenerate the materialized files byte-identically. |

## Regeneration policy

- **MIP-N family** (objectnav / ovon / hmeqa / mt_hm3d / express):
  reproducible — rerun `sample_episodes.py` with the same seed. Manifests
  and the CSV/JSON corpora regenerate byte-identically; the habitat
  `.json.gz` files regenerate byte-identically for the CURRENT generator
  (verified across reruns) but differ byte-wise from the pre-2026-08-18
  artifacts, which were written by an older generator generation — the
  2026-08-18 full-plan rerun replaced them with manifest-identical episode
  sequences (env-load verified; any md5 recorded before 2026-08-18 will not
  match). Episode identity, the thing board comparability rests on, is
  pinned by the manifests. `--materialize` writes here and drops the
  data-tree symlink itself.
- **`rxr/rand100`**: NOT reproducible from the sampler (RNG/code-version
  sensitive greedy search, ~81/100 overlap on rerun — the sampler now
  refuses while the canonical output exists). Restore from run records via
  `rebuild_rxr_rand100_from_runs.py`; never resample.
- **`r2r/rand100` / `rand100_smartway`**: selection is upstream's — there
  is nothing to regenerate. `fix_rand100_official.py` is a one-shot
  historical record and refuses to rerun (it would clobber the protocol
  backup).
- **`r2r/heldout100`**: files are canonical; no generator. Do not resample
  — it would break comparability with recorded held-out boards.

## Caution: comparing a split file against run records

Run records live in PRESENTATION order, not file order. For the
objnav/ovon lines, habitat 0.2.4 shuffles `dataset.episodes` IN PLACE at
env load (default `shuffle=True`, `seed=habitat.seed`=100 — a fixed
permutation for any 100-episode split), and `ObjectNavDatasetV1.from_json`
reassigns `episode_id` positionally, so a run's episode_id is a file
POSITION, not an identity. Comparing file order or raw episode_id
directly against run records is a category error — it once produced a
false "boards ran a different selection (5% overlap)" conclusion during
the 2026-08-18 audit. Correct procedure: push the file through the load
transform first (seed-100 permutation + id semantics of the dataset
class), then compare — fingerprint by (scene, object_category) for
objnav, by real-id multiset for ovon. R2R/RxR (env_habitat) are exempt:
it sets `ITERATOR_OPTIONS.SHUFFLE = False` and VLN-CE preserves real ids,
so index = file position there.

## Known composition notes

- `r2r/rand100` under-samples multi-floor episodes ~2x vs full val_unseen
  (8% vs 16% of episodes with GT vertical span >= 1 m; driven by the
  stairs-heavy scene oLBMNvg9in8 getting 3/100 vs 9.6 expected). Inherited
  with the selection — documented, not to be "fixed" by resampling.
- `rxr/rand100` matches its English val_unseen population on all measured
  margins (lengths by design; start-goal distance and vertical span as
  free dimensions); its scene mix deliberately copies R2R rand100 for
  cross-benchmark comparability instead of RxR-proportional shares.
- `r2r/heldout100` matches the population margins including vertical span
  — it carries ~2x the stairs exposure of `r2r/rand100`.
