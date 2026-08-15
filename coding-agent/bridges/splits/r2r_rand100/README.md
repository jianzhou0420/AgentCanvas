# r2r_rand100 — the paper's R2R-CE evaluation split

`rand100.json.gz` + `rand100_gt.json.gz` are the 100-episode R2R-CE
val-unseen subset every board cell evaluates on (`split="rand100"`,
episodes 0–99). It is the evaluation protocol published with
[SmartWay](https://github.com/sxyxs/SmartWay-Code) (their `eval.sh` runs
`EVAL.SPLIT rand100`) and shared by OpenNav / AgenticNav — we adopt it
unchanged so our rows are protocol-comparable with theirs.

These are not a plain filter of `val_unseen.json.gz`: the episodes carry
real `start_rotation` spawn headings (the official file has identity
rotations), instructions re-tokenized with the BERT vocabulary
(30 522 entries), and ground-truth paths regenerated for those spawns.
That is why the files are versioned here whole (~710 KB) rather than as
an episode-id manifest — the id list alone cannot regenerate them.

Install into the data tree with:

```bash
bash scripts/data/materialize_r2r_rand100.sh
```

which copies them to
`data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/rand100/` where the
`env_habitat` nodeset expects the split.
