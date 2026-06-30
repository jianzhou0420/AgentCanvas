"""CMA model adapter — stage 2 + stage 4 for cross-modal-attention VLN-CE.

Wraps the vlnce_baselines + habitat_baselines preprocessing pipeline:
  - extract_instruction_tokens  (instruction sensor uuid → token ids)
  - batch_obs                    (stack into torch tensors on device)
  - apply_obs_transforms_batch   (resize / center-crop / depth normalize)

For the RxR-CE family the instruction is a precomputed BERT 768-d feat;
the same vlnce-baselines pipeline picks the right sensor uuid from the
exp_config (rxr_instruction vs instruction).

Lifts the working code path from the legacy ``policy_cma.py:469-560``
verbatim, modulo the per-sample batching.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import numpy as np

from workspace.nodesets.policy.policy_vlnce.adapters.canonical import (
    CanonicalDict,
    CanonicalNavInfo,
    make_canonical_action,
)
from workspace.nodesets.policy.policy_vlnce.adapters.models.base_model import (
    VlnModelAdaptor,
)

DEFAULT_KWARGS: dict[str, Any] = {}


def _resolve_vlnce_root() -> str:
    """Resolve the VLN-CE checkout root for chdir-based exp_config loading."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "..", ".."))
    candidates = [
        os.environ.get("VLNCE_ROOT", ""),
        os.path.join(repo_root, "..", "VLN-CE"),
        os.path.join(repo_root, "third_party", "VLN-CE"),
    ]
    for c in candidates:
        c = os.path.normpath(c) if c else ""
        if c and os.path.isdir(os.path.join(c, "data")):
            return c
    # Fallback even if data/ missing — get_config still resolves YAML.
    return os.path.normpath(os.path.join(repo_root, "third_party", "VLN-CE"))


class CmaModel(VlnModelAdaptor):
    """CMA preprocessing pipeline (vlnce_baselines + habitat_baselines)."""

    arch: ClassVar[str] = "cma"

    def __init__(self, *, exp_config_path: str) -> None:
        if not exp_config_path:
            raise ValueError(
                "CmaModel requires exp_config_path (e.g. 'vlnce_baselines/config/r2r_baselines/cma_pm_da.yaml')"
            )
        self._exp_config_path = exp_config_path
        self._vlnce_root = _resolve_vlnce_root()

        # Lazy heavyweights — resolved on first canonical_to_model call so
        # module import doesn't require habitat_baselines.
        self._config: Any = None
        self._obs_transforms: list[Any] | None = None
        self._instruction_sensor_uuid: str | None = None

        # Instruction tokenization is policy-specific and lives here (model side),
        # not in the env adapter. The raw instruction text is a per-episode
        # constant, so memoize the token ids by text — canonical_to_model runs
        # every step but tokenization then happens once per episode.
        self._instr_cache_text: str | None = None
        self._instr_cache_tokens: dict[str, Any] | None = None

    # ---- public ----

    @property
    def exp_config_path(self) -> str:
        return self._exp_config_path

    @property
    def policy_config(self) -> Any:
        self._ensure_loaded()
        return self._config

    @property
    def obs_transforms(self) -> list:
        self._ensure_loaded()
        return list(self._obs_transforms or [])

    def derive_obs_space(self, sample_canonical: CanonicalDict) -> Any:
        """Apply obs_transforms_obs_space to a raw obs_space derived from canonical."""
        self._ensure_loaded()
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_obs_space,
        )

        raw_obs = self._canonical_to_raw_obs(sample_canonical)
        obs_space = self._build_raw_obs_space(raw_obs)
        return apply_obs_transforms_obs_space(obs_space, self._obs_transforms)

    def derive_action_space(self, info: CanonicalNavInfo) -> Any:
        import gym.spaces as spaces

        return spaces.Discrete(int(info.action_dim))

    def canonical_to_model(
        self,
        canonical: CanonicalDict,
        *,
        hidden_in: dict[str, Any] | None,
        device: Any,
    ) -> dict[str, Any]:
        """Build a wire-friendly model_batch (numpy only).

        We DO NOT call batch_obs / apply_obs_transforms_batch here — those
        return torch tensors, and round-tripping them through numpy across
        the auto_host wire silently produces a different downstream
        trajectory than the legacy single-node path (verified 2026-05-07:
        identical inputs but no STOP picks). Instead we emit the
        post-extract_instruction_tokens raw observations dict on the wire,
        and stage 3 (CmaPolicy.forward) does batch_obs + transforms +
        policy.act all in torch in one place.
        """
        self._ensure_loaded()
        from vlnce_baselines.common.utils import extract_instruction_tokens

        raw_obs = self._canonical_to_raw_obs(canonical)

        # extract_instruction_tokens mutates the obs in place — operate on a
        # fresh list so we don't aliase upstream callers.
        observations = [dict(raw_obs)]
        observations = extract_instruction_tokens(
            observations,
            self._instruction_sensor_uuid,
        )

        return {
            "raw_observations": observations,  # list-of-1 numpy/python obs dicts
            "instruction_sensor_uuid": self._instruction_sensor_uuid,
            "hidden_in": hidden_in,
        }

    def model_to_canonical(
        self,
        model_output: dict[str, Any],
        info: CanonicalNavInfo,
    ) -> CanonicalDict:
        action = int(model_output["action_index"])
        return make_canonical_action(action_index=action, info=info)

    # ---- helpers ----

    def _ensure_loaded(self) -> None:
        if self._config is not None:
            return
        # ORDER MATTERS: habitat_extensions + vlnce_baselines must be
        # imported BEFORE habitat_baselines.* — they register custom
        # sensors / measurements / config nodes that habitat_baselines
        # tries to instantiate during its own import chain.
        import habitat_extensions  # noqa: F401
        import vlnce_baselines  # noqa: F401
        from habitat_baselines.common.obs_transformers import (
            get_active_obs_transforms,
        )
        from vlnce_baselines.config.default import get_config

        # chdir to vlnce_root and DO NOT restore — the InstructionEncoder
        # and other CMANet sub-modules read data/datasets/...embeddings.json.gz
        # at construction time (stage 3 policy.build), which happens
        # AFTER this method returns. Mirror legacy policy_cma._load_cma_policy.
        if os.path.isdir(self._vlnce_root):
            os.chdir(self._vlnce_root)
        cfg = get_config(self._exp_config_path)

        cfg.defrost()
        cfg.TASK_CONFIG.DATASET.SPLIT = "val_unseen"
        cfg.TASK_CONFIG.DATASET.ROLES = ["guide"]
        cfg.TASK_CONFIG.DATASET.LANGUAGES = cfg.EVAL.LANGUAGES
        cfg.TASK_CONFIG.TASK.NDTW.SPLIT = "val_unseen"
        cfg.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        cfg.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        cfg.SIMULATOR_GPU_IDS = [0]
        cfg.TORCH_GPU_ID = 0
        cfg.NUM_ENVIRONMENTS = 1
        cfg.freeze()
        self._config = cfg
        self._obs_transforms = get_active_obs_transforms(cfg)
        self._instruction_sensor_uuid = cfg.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID

    def _tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize a raw R2R-CE instruction into the habitat InstructionSensor
        shape ({"tokens", "text"}), memoized by text (per-episode constant).

        This is the policy-specific step the env adapter deliberately does NOT
        do — the canonical carries raw text, and the CMA-vocab tokenization
        lives here on the model side. Replica verified 200/200 vs the dataset's
        precomputed tokens (see adapters/r2r_tokenizer.py).
        """
        if text == self._instr_cache_text and self._instr_cache_tokens is not None:
            return self._instr_cache_tokens
        from workspace.nodesets.policy.policy_vlnce.adapters.r2r_tokenizer import (
            tokenize_instruction,
        )

        tok = tokenize_instruction(text)
        self._instr_cache_text = text
        self._instr_cache_tokens = tok
        return tok

    def _canonical_to_raw_obs(self, canonical: CanonicalDict) -> dict[str, Any]:
        """Reconstruct the dict shape vlnce_baselines / habitat_baselines expect,
        from the STANDARDIZED canonical (rgb / depth / raw instruction).

        Policy-specific processing happens here, not in the env adapter:
          - raw_text (R2R-CE): tokenize → raw_obs["instruction"] = {tokens, text}
          - feat     (RxR-CE): pass embedding → raw_obs["rxr_instruction"] (BERT (T, 768))
        """
        data = canonical["data"]
        instr = data["instruction"]
        kind = instr["kind"]
        out: dict[str, Any] = {}
        out["rgb"] = np.asarray(data["rgb"])
        out["depth"] = np.asarray(data["depth"], dtype=np.float32)
        if kind == "raw_text":
            out["instruction"] = self._tokenize(instr["text"])
        elif kind == "feat":
            out["rxr_instruction"] = instr["embedding"]
        else:
            raise ValueError(f"unknown instruction kind {kind!r}")
        return out

    @staticmethod
    def _build_raw_obs_space(raw_obs: dict[str, Any]) -> Any:
        """Build a gym Dict obs_space from raw_obs sample shapes.

        Same logic as legacy policy_cma._build_spaces_from_obs.
        """
        import gym.spaces as spaces

        obs_dict: dict[str, Any] = {}
        for key, val in raw_obs.items():
            if not isinstance(val, (np.ndarray, list)):
                continue
            arr = np.asarray(val)
            if arr.size == 0:
                continue
            if np.issubdtype(arr.dtype, np.integer):
                obs_dict[key] = spaces.Box(
                    low=np.iinfo(arr.dtype).min,
                    high=np.iinfo(arr.dtype).max,
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
            else:
                obs_dict[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
        return spaces.Dict(obs_dict)
