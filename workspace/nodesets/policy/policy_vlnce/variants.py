"""R2R-CE baseline variant registry for policy_vlnce.

Source of truth: third_party/VLN-CE/vlnce_baselines/config/r2r_baselines/README.md
(VLN-CE paper Table 1, val_unseen SPL).

Each entry binds a checkpoint to its model adapter, exp_config YAML, and
policy module. The chain-entry node (adapt_env_to_canonical) exposes a
'variant' ConfigField populated from this registry — selecting a variant
is what flips the 4 strings the canvas nodes used to carry as per-graph
config.

Naming: <model>_<modifier_chain> with modifier order PM_DA_Aug. Modifiers:
  PM  = Progress Monitor auxiliary head
  DA  = DAgger training (vs teacher forcing)
  Aug = EnvDrop synthetic-episode augmentation

Inference YAML choice:
  - For variants with a separate fine-tune YAML in the README's "Config"
    column (e.g. CMA_Aug → cma_aug.yaml ⟶ cma_aug_tune.yaml), we use the
    tune YAML — it matches the released checkpoint's training schedule.
  - CMA_PM_DA_Aug is the one exception: the existing graph used cma_pm_da.yaml
    and that path is regression-validated (SR≈0.34 / SPL≈0.328 on val_unseen
    100-ep). Keep it to preserve equivalence.

Checkpoint availability:
  - Upstream VLN-CE only published 2 R2R-CE checkpoints — CMA_PM_DA_Aug.pth
    and Seq2Seq_DA.pth — i.e. one "best" CMA + one "best" Seq2Seq. The
    other 10 ablation variants from the paper's Table 1 were never
    released (issue #75 on jacobkrantz/VLN-CE confirms — open + unanswered).
  - Those 10 entries stay in the registry so the topology is complete and
    re-listed for transparency, but their dropdown label carries the
    "(not released by paper authors)" suffix so users see at selection
    time why the run will fail with a missing-file error.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    model_adaptor: str       # "cma" | "seq2seq" — adapters/models/<name>.py
    exp_config: str          # YAML path under third_party/VLN-CE/
    policy: str              # "cma_policy" | "seq2seq_policy" — policies/<name>.py
    checkpoint_path: str     # .pth path under data/habitat/checkpoints/
    paper_spl: float         # val_unseen SPL from VLN-CE paper Table 1


_R2R_YAML_PREFIX = "vlnce_baselines/config/r2r_baselines/"
_CKPT_PREFIX = "data/habitat/checkpoints/"

# Suffix appended to the dropdown label of variants whose checkpoint
# was never released by the upstream paper authors.
_UNRELEASED_SUFFIX = " (not released by paper authors)"


REGISTRY: list[VariantSpec] = [
    # CMA family — 7 variants, paper_spl desc
    VariantSpec(
        key="CMA_PM_DA_Aug",
        label="CMA +PM+DA+Aug (SPL=0.30)",
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_pm_da.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_PM_DA_Aug.pth",
        paper_spl=0.30,
    ),
    VariantSpec(
        key="CMA_DA_Aug",
        label="CMA +DA+Aug (SPL=0.26)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_da_aug_tune.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_DA_Aug.pth",
        paper_spl=0.26,
    ),
    VariantSpec(
        key="CMA_DA",
        label="CMA +DA (SPL=0.25)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_da.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_DA.pth",
        paper_spl=0.25,
    ),
    VariantSpec(
        key="Seq2Seq_DA",
        label="Seq2Seq +DA (SPL=0.23)",
        model_adaptor="seq2seq",
        exp_config=_R2R_YAML_PREFIX + "seq2seq_da.yaml",
        policy="seq2seq_policy",
        checkpoint_path=_CKPT_PREFIX + "Seq2Seq_DA.pth",
        paper_spl=0.23,
    ),
    VariantSpec(
        key="CMA",
        label="CMA clean (SPL=0.22)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA.pth",
        paper_spl=0.22,
    ),
    VariantSpec(
        key="CMA_PM_Aug",
        label="CMA +PM+Aug (SPL=0.22)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_pm_aug_tune.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_PM_Aug.pth",
        paper_spl=0.22,
    ),
    VariantSpec(
        key="Seq2Seq_PM_DA_Aug",
        label="Seq2Seq +PM+DA+Aug (SPL=0.22)" + _UNRELEASED_SUFFIX,
        model_adaptor="seq2seq",
        exp_config=_R2R_YAML_PREFIX + "seq2seq_pm_da_aug_tune.yaml",
        policy="seq2seq_policy",
        checkpoint_path=_CKPT_PREFIX + "Seq2Seq_PM_DA_Aug.pth",
        paper_spl=0.22,
    ),
    VariantSpec(
        key="CMA_PM",
        label="CMA +PM (SPL=0.19)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_pm.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_PM.pth",
        paper_spl=0.19,
    ),
    VariantSpec(
        key="CMA_Aug",
        label="CMA +Aug (SPL=0.19)" + _UNRELEASED_SUFFIX,
        model_adaptor="cma",
        exp_config=_R2R_YAML_PREFIX + "cma_aug_tune.yaml",
        policy="cma_policy",
        checkpoint_path=_CKPT_PREFIX + "CMA_Aug.pth",
        paper_spl=0.19,
    ),
    VariantSpec(
        key="Seq2Seq",
        label="Seq2Seq clean (SPL=0.18)" + _UNRELEASED_SUFFIX,
        model_adaptor="seq2seq",
        exp_config=_R2R_YAML_PREFIX + "seq2seq.yaml",
        policy="seq2seq_policy",
        checkpoint_path=_CKPT_PREFIX + "Seq2Seq.pth",
        paper_spl=0.18,
    ),
    VariantSpec(
        key="Seq2Seq_Aug",
        label="Seq2Seq +Aug (SPL=0.17)" + _UNRELEASED_SUFFIX,
        model_adaptor="seq2seq",
        exp_config=_R2R_YAML_PREFIX + "seq2seq_aug_tune.yaml",
        policy="seq2seq_policy",
        checkpoint_path=_CKPT_PREFIX + "Seq2Seq_Aug.pth",
        paper_spl=0.17,
    ),
    VariantSpec(
        key="Seq2Seq_PM",
        label="Seq2Seq +PM (SPL=0.15)" + _UNRELEASED_SUFFIX,
        model_adaptor="seq2seq",
        exp_config=_R2R_YAML_PREFIX + "seq2seq_pm.yaml",
        policy="seq2seq_policy",
        checkpoint_path=_CKPT_PREFIX + "Seq2Seq_PM.pth",
        paper_spl=0.15,
    ),
]

REGISTRY_BY_KEY: dict[str, VariantSpec] = {v.key: v for v in REGISTRY}

DEFAULT_KEY: str = "CMA_PM_DA_Aug"

assert DEFAULT_KEY in REGISTRY_BY_KEY, "DEFAULT_KEY must be a registered variant"
assert len(REGISTRY) == 12, f"expected 12 R2R-CE variants, got {len(REGISTRY)}"
