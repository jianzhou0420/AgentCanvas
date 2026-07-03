"""Seq2Seq model adapter — stage 2 + stage 4 for sequence-to-sequence VLN-CE.

The preprocessing pipeline (extract_instruction_tokens + apply_obs_transforms_batch
+ batch_obs) is identical to CMA's; obs_transforms list contents are picked up
from the seq2seq exp_config (e.g. r2r_baselines/seq2seq_da.yaml). This adapter
is therefore a thin subclass that just changes the dropdown label.
"""

from __future__ import annotations

from typing import ClassVar

from workspace.nodesets.policy.policy_adapter_vlnce.adapters.models.cma import CmaModel


class Seq2SeqModel(CmaModel):
    arch: ClassVar[str] = "seq2seq"
