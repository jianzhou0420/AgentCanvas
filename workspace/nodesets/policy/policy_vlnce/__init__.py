from __future__ import annotations

"""PolicyVlnceNodeSet — generic VLN-CE inference framework as a NodeSet.

5-stage adapter pattern (mirrors policy_vla 1:1, VLN-flavored):

  stage 1: env_to_canonical          — env → CanonicalDict[obs]
  stage 2: canonical_to_model        — CanonicalDict[obs] → model_batch
  stage 3: predict                   — model_batch → model_output
  stage 4: model_to_canonical        — model_output → CanonicalDict[action]
  stage 5: canonical_to_env          — CanonicalDict[action] → env action int

Filename-based discovery: every .py file under ``adapters/envs/``,
``adapters/models/``, ``policies/`` is auto-discovered as a canvas
dropdown option. Drop a file in / out → POST /api/components/reload.

Config distributed across nodes:
  - Stage 1 owns: ``env`` dropdown
  - Stage 2 owns: ``model`` dropdown + ``exp_config`` (yaml path)
  - Stage 3 owns: ``policy`` dropdown + ``checkpoint_path``
  - Stage 4 + 5: configless (reuse manager state from 2 + 1)

User-visible invariants (graph-level):
  - ``model`` and ``policy`` must be a matching pair (cma + cma_policy).
  - ``exp_config`` and ``checkpoint_path`` must be from the same VLN-CE
    baseline family (e.g. r2r_baselines/cma_pm_da.yaml + CMA_PM_DA_Aug.pth).

Recurrent state on the wire: predict has ``hidden_in`` / ``hidden_out``
ports. Wire ``iter_in.iterout_hidden → predict.hidden_in`` and
``predict.hidden_out → iter_out.hidden`` exactly like the legacy
policy_cma graph.

last updated: 2026-05-07
"""


import asyncio
import concurrent.futures
import importlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, ClassVar

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef
from app.server.batched_inference import OUTPUTS_KEY, SAMPLES_KEY
from workspace.nodesets.policy.policy_vlnce.variants import (
    DEFAULT_KEY,
    REGISTRY,
    REGISTRY_BY_KEY,
    VariantSpec,
)

log = logging.getLogger("agentcanvas.policy_vlnce")


# ══════════════════════════════════════════════════════════════════════
# Discovery — folder-scan + lazy import (mirrors policy_vla)
# ══════════════════════════════════════════════════════════════════════

_PKG_ROOT = Path(__file__).parent
_DISCOVERY_EXCLUDES: tuple[str, ...] = ("__init__",)


def _discover_modules(subpkg_relpath: str) -> list[str]:
    pkg_dir = _PKG_ROOT / subpkg_relpath
    return sorted(
        f.stem
        for f in pkg_dir.glob("*.py")
        if f.stem not in _DISCOVERY_EXCLUDES
        and not f.stem.startswith("_")
        and not f.stem.startswith("base_")
    )


def _load_module(subpkg_dotted: str, module_name: str) -> Any:
    return importlib.import_module(
        f"workspace.nodesets.policy.policy_vlnce.{subpkg_dotted}.{module_name}"
    )


def _find_subclass(module: Any, base: type) -> type:
    found = []
    for v in vars(module).values():
        if not isinstance(v, type) or v is base:
            continue
        try:
            if issubclass(v, base):
                found.append(v)
        except TypeError:
            continue
    if not found:
        raise ValueError(f"no {base.__name__} subclass found in {module.__name__}")
    if len(found) > 1:
        local = [c for c in found if c.__module__ == module.__name__]
        if local:
            return local[0]
    return found[0]


# Canvas dropdown options — frozen at module-import time.
ENV_OPTIONS: list[str] = _discover_modules("adapters/envs")
MODEL_OPTIONS: list[str] = _discover_modules("adapters/models")
POLICY_OPTIONS: list[str] = _discover_modules("policies")


# ══════════════════════════════════════════════════════════════════════
# VlnceManager — singleton policy runtime
# ══════════════════════════════════════════════════════════════════════


class VlnceManager:
    """Singleton manager hosting one VLN-CE policy on GPU.

    All torch/CUDA work runs on a pinned single-thread executor. Each of
    the three config-bearing canvas nodes (env→canonical, canonical→model,
    predict) owns one slice of state via an idempotent ``ensure_*`` call
    that short-circuits when its node's config is unchanged.
    """

    _instance: VlnceManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="policy_vlnce",
        )
        self._initialized: bool = False

        # Slice state
        self._env_adapter: Any = None  # VlnEnvAdaptor
        self._env_module: str = ""
        self._model_adaptor: Any = None  # VlnModelAdaptor
        self._model_module: str = ""
        self._policy: Any = None  # VlnPolicy on GPU
        self._policy_module: str = ""

        # ensure_* idempotency caches
        self._env_cfg_cache: tuple | None = None
        self._model_cfg_cache: tuple | None = None
        self._policy_cfg_cache: tuple | None = None

        # Cached first-seen canonical → used by ensure_policy to derive
        # obs_space (CMANet ctor needs it). Set by canonical_to_model on
        # first call. Independent of the per-step model_batch — only the
        # shape signature matters.
        self._first_canonical: Any = None

        self._loaded_meta: dict[str, Any] = {}

    # ── singleton + lifecycle ──────────────────────────────────────────

    @classmethod
    def get(cls) -> VlnceManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, **_kwargs: Any) -> None:
        with self._lock:
            self._initialized = True
            log.info("VlnceManager: initialized (lazy-load slices on first call)")

    def shutdown(self) -> None:
        with self._lock:
            self._unload_policy_unlocked()
            self._env_adapter = None
            self._model_adaptor = None
            self._first_canonical = None
            self._initialized = False

    def _unload_policy_unlocked(self) -> None:
        if self._policy is not None:
            try:
                self._policy.to("cpu")
            except Exception:
                log.debug("policy.to('cpu') raised", exc_info=True)
            del self._policy
            self._policy = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._policy_module = ""
        self._loaded_meta = {}

    # ── slice ensure_* (idempotent) ────────────────────────────────────

    def ensure_env_adapter(self, env_module: str) -> dict[str, Any]:
        with self._lock:
            if not self._initialized:
                return {"error": "VlnceManager not initialized"}
            try:
                module = _load_module("adapters.envs", env_module)
                from workspace.nodesets.policy.policy_vlnce.adapters.envs.base_env import (
                    VlnEnvAdaptor,
                )

                cls = _find_subclass(module, VlnEnvAdaptor)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
            except Exception as e:
                return {"error": f"load env module {env_module!r}: {e!r}"}

            cache_key = (env_module, json.dumps(kwargs, sort_keys=True, default=str))
            if cache_key == self._env_cfg_cache and self._env_adapter is not None:
                return {"unchanged": True}

            try:
                self._env_adapter = cls(**kwargs)
                self._env_module = env_module
                self._env_cfg_cache = cache_key
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}
            log.info("ensure_env_adapter: rebuilt module=%s", env_module)
            return {"rebuilt": True, "module": env_module}

    def ensure_model_adaptor(self, model_module: str, exp_config: str) -> dict[str, Any]:
        with self._lock:
            if not self._initialized:
                return {"error": "VlnceManager not initialized"}
            try:
                module = _load_module("adapters.models", model_module)
                from workspace.nodesets.policy.policy_vlnce.adapters.models.base_model import (
                    VlnModelAdaptor,
                )

                cls = _find_subclass(module, VlnModelAdaptor)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
                kwargs["exp_config_path"] = exp_config
            except Exception as e:
                return {"error": f"load model module {model_module!r}: {e!r}"}

            cache_key = (
                model_module,
                exp_config,
                json.dumps(kwargs, sort_keys=True, default=str),
            )
            if cache_key == self._model_cfg_cache and self._model_adaptor is not None:
                return {"unchanged": True}

            # Module change → drop policy (Net class differs).
            prev = self._model_module
            if model_module != prev and self._policy is not None:
                log.info("model module %r → %r: dropping policy", prev, model_module)
                self._unload_policy_unlocked()
                self._policy_cfg_cache = None

            try:
                self._model_adaptor = cls(**kwargs)
                self._model_module = model_module
                self._model_cfg_cache = cache_key
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}
            # exp_config change in same model_module → drop policy too
            if prev == model_module:
                self._unload_policy_unlocked()
                self._policy_cfg_cache = None
            log.info(
                "ensure_model_adaptor: rebuilt module=%s exp_config=%s",
                model_module,
                exp_config,
            )
            return {"rebuilt": True, "module": model_module, "exp_config": exp_config}

    def ensure_policy(
        self,
        policy_module: str,
        *,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> dict[str, Any]:
        with self._lock:
            if not self._initialized:
                return {"error": "VlnceManager not initialized"}
            if self._env_adapter is None:
                return {"error": "env_adapter not built — run adapt_env_to_canonical first"}
            if self._model_adaptor is None:
                return {"error": "model_adaptor not built — run adapt_canonical_to_model first"}
            if self._first_canonical is None:
                return {
                    "error": "no canonical observation seen yet — run adapt_env_to_canonical at least once"
                }

            try:
                module = _load_module("policies", policy_module)
                from workspace.nodesets.policy.policy_vlnce.policies.base_policy import (
                    VlnPolicy,
                )

                cls = _find_subclass(module, VlnPolicy)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
            except Exception as e:
                return {"error": f"load policy module {policy_module!r}: {e!r}"}

            cache_key = (
                policy_module,
                checkpoint_path,
                json.dumps(kwargs, sort_keys=True, default=str),
            )
            if cache_key == self._policy_cfg_cache and self._policy is not None:
                return {"unchanged": True}

            self._unload_policy_unlocked()

            try:
                policy = cls(**kwargs)
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}

            try:
                obs_space = self._model_adaptor.derive_obs_space(self._first_canonical)
                action_space = self._model_adaptor.derive_action_space(
                    self._env_adapter.get_canonical_info()
                )
                policy.build(
                    observation_space=obs_space,
                    action_space=action_space,
                    model_config=self._model_adaptor.policy_config.MODEL,
                    obs_transforms=self._model_adaptor.obs_transforms,
                )
            except Exception as e:
                log.exception("policy.build failed")
                return {"error": f"policy.build: {e!r}"}

            if checkpoint_path:
                try:
                    policy.load_checkpoint(checkpoint_path)
                except Exception as e:
                    return {"error": f"load_checkpoint({checkpoint_path}): {e!r}"}

            try:
                import torch

                target = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
                policy = policy.to(target).eval()
            except Exception as e:
                return {"error": f"policy.to({device}): {e!r}"}

            self._policy = policy
            self._policy_module = policy_module
            self._policy_cfg_cache = cache_key
            self._loaded_meta = {
                "policy_module": policy_module,
                "checkpoint_path": checkpoint_path,
                "total_params": int(policy.num_parameters),
                "device": str(policy.device),
            }
            log.info("ensure_policy loaded: %s", self._loaded_meta)
            return dict(self._loaded_meta)

    # ── inference path ────────────────────────────────────────────────

    def env_to_canonical(
        self, raw_obs: dict[str, Any], instruction: Any = None
    ) -> dict[str, Any]:
        with self._lock:
            if self._env_adapter is None:
                return {"error": "env_adapter not initialized"}
            try:
                canonical = self._env_adapter.env_to_canonical(raw_obs, instruction)
                # Cache the first canonical so ensure_policy can derive
                # obs_space without waiting for stage 2 to run. Lets stage 1
                # (chain entry) eagerly load the policy after env_to_canonical.
                if self._first_canonical is None:
                    self._first_canonical = canonical
                return {"canonical": canonical}
            except Exception as e:
                log.exception("env_to_canonical failed")
                return {"error": f"env_to_canonical: {e!r}"}

    def canonical_to_model(
        self, canonical: dict[str, Any], hidden_in: dict[str, Any] | None
    ) -> dict[str, Any]:
        with self._lock:
            if self._model_adaptor is None:
                return {"error": "model_adaptor not initialized"}
            # Cache first canonical so ensure_policy can derive obs_space
            # without us having to re-thread shape info through the wire.
            if self._first_canonical is None:
                self._first_canonical = canonical
            try:
                # Derive device from policy if loaded; otherwise CPU placeholder.
                # Policy is loaded by stage 3 *after* stage 2's first run, so on
                # the very first iter we may build obs_batch on CPU and migrate
                # in stage 3. In practice ensure_policy is called before any
                # canonical_to_model that actually feeds predict (graph order is
                # 1 → 2 → 3 → 4 → 5; ensure_policy is called inside 3.execute).
                # So on iter 0 we build CPU tensors; predict.forward moves them.
                import torch

                if self._policy is not None:
                    device = self._policy.device
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_batch = self._model_adaptor.canonical_to_model(
                    canonical, hidden_in=hidden_in, device=device
                )
                return {"model_batch": model_batch}
            except Exception as e:
                log.exception("canonical_to_model failed")
                return {"error": f"canonical_to_model: {e!r}"}

    def predict(self, model_batch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._policy is None:
                return {"error": "policy not initialized"}
            try:
                # Migrate tensors to the policy's device if they were built CPU-side.
                import torch

                target_device = self._policy.device

                def _to_device(d: dict[str, Any]) -> dict[str, Any]:
                    out: dict[str, Any] = {}
                    for k, v in d.items():
                        if torch.is_tensor(v):
                            out[k] = v.to(target_device)
                        elif isinstance(v, dict):
                            out[k] = _to_device(v)
                        else:
                            out[k] = v
                    return out

                model_batch = _to_device(model_batch)
                model_output = self._policy.forward(model_batch)
                return {"model_output": model_output}
            except Exception as e:
                log.exception("predict failed")
                return {"error": f"predict: {e!r}"}

    def model_to_canonical_action(self, model_output: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._model_adaptor is None or self._env_adapter is None:
                return {"error": "manager not loaded"}
            try:
                info = self._env_adapter.get_canonical_info()
                ca = self._model_adaptor.model_to_canonical(model_output, info)
                return {"canonical_action": ca}
            except Exception as e:
                log.exception("model_to_canonical_action failed")
                return {"error": f"model_to_canonical_action: {e!r}"}

    def canonical_action_to_env(self, canonical_action: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._env_adapter is None:
                return {"error": "env_adapter not initialized"}
            try:
                action = int(self._env_adapter.canonical_to_env(canonical_action))
                return {"action": action}
            except Exception as e:
                log.exception("canonical_action_to_env failed")
                return {"error": f"canonical_action_to_env: {e!r}"}


# ══════════════════════════════════════════════════════════════════════
# Variant resolution helpers
# ══════════════════════════════════════════════════════════════════════


def _resolve_variant_from_config(cfg: dict[str, Any]) -> VariantSpec:
    """Look up the VariantSpec from a node's config dict.

    Variant lives on the chain-entry node (stage 1) per the
    "env panels = env-side runtime knobs only" rule (memory:
    feedback_env_panel_scope). Method nodesets express model/ckpt
    choices as ConfigFields on a node, not on an env panel.
    """
    key = (cfg.get("variant") or "").strip() or DEFAULT_KEY
    return REGISTRY_BY_KEY.get(key, REGISTRY_BY_KEY[DEFAULT_KEY])


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve a relative checkpoint path against the project repo root.

    ``__file__`` lives at ``workspace/nodesets/policy/policy_vlnce/__init__.py``
    — four parents to reach the repo root.
    """
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.normpath(os.path.join(repo_root, path))


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> VlnceManager:
    return VlnceManager.get()


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    if kwargs:
        import functools

        return await loop.run_in_executor(
            _get_mgr().executor, functools.partial(fn, *args, **kwargs)
        )
    return await loop.run_in_executor(_get_mgr().executor, fn, *args)


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes
# ══════════════════════════════════════════════════════════════════════


_POLICY_COLOR = "blue"


_VARIANT_OPTIONS: list[dict[str, str]] = [{"value": v.key, "label": v.label} for v in REGISTRY]


# R2R-CE is currently the only env_family the registry covers (RxR-CE
# monolingual variants were dropped from v1 scope). When the registry
# grows to include rxrce variants, derive env_module from the spec
# (e.g. add ``env_module: str`` to VariantSpec) instead of pinning here.
_ENV_MODULE_FOR_VARIANT = "r2rce"


class AdaptEnvToCanonicalTool(BaseCanvasNode):
    node_type = "policy_vlnce__adapt_env_to_canonical"
    display_name = "VLN-CE: Env → Canonical"
    description = (
        "VlnEnvAdaptor.env_to_canonical — wraps a Habitat raw_obs dict into "
        "the canonical {data, info} intermediate format. As the chain entry "
        "node, this is also where the VLN-CE *variant* (model + exp_config "
        "+ policy + checkpoint) is selected; the dropdown lists 12 R2R-CE "
        "baselines from variants.REGISTRY. On execute it eagerly primes the "
        "VlnceManager (env adapter + model adapter + policy load) so "
        "downstream stages 2/3 just call into the cached singleton."
    )
    category = "policy"
    icon = "ArrowRight"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_POLICY_COLOR,
        config_fields=[
            ConfigField(
                "variant",
                "select",
                label="Variant (R2R-CE baseline)",
                default=DEFAULT_KEY,
                options=_VARIANT_OPTIONS,
            ),
        ],
    )
    input_ports = [
        PortDef("raw_obs", "TEXT", "Raw Habitat observation dict (from env observe/step)"),
        PortDef(
            "instruction",
            "ANY",
            "Raw natural-language instruction text (per-episode constant, from "
            "env_habitat's instruction_text port via an iter_in init slot, "
            "persist=true). REQUIRED so the executor waits for it before firing. "
            "The env adapter only standardizes (passes the text through verbatim); "
            "CMA-vocab tokenization happens later in canonical_to_model.",
        ),
    ]
    output_ports = [
        PortDef("canonical", "ANY", "CanonicalDict[obs] for the model adapter to consume"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        spec = _resolve_variant_from_config(self.config)
        mgr = _get_mgr()

        # Stage 1 = chain entry: prime all three slices upfront so stages
        # 2/3 don't need their own variant config. ensure_* are idempotent
        # caches, so cost is paid only when variant flips.
        res = await _run_sync(mgr.ensure_env_adapter, _ENV_MODULE_FOR_VARIANT)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_env_adapter: {res['error']}")
            return {"canonical": None}

        res = await _run_sync(mgr.ensure_model_adaptor, spec.model_adaptor, spec.exp_config)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_model_adaptor: {res['error']}")
            return {"canonical": None}

        raw_obs = inputs.get("raw_obs")
        if not isinstance(raw_obs, dict):
            self._self_log("error", "raw_obs not a dict")
            return {"canonical": None}

        instruction = inputs.get("instruction")
        if instruction is None:
            self._self_log("warn", "no instruction wired — policy will see None tokens")
        out = await _run_sync(mgr.env_to_canonical, raw_obs, instruction)
        if isinstance(out, dict) and "error" in out:
            self._self_log("error", out["error"])
            return {"canonical": None}
        canonical = out["canonical"]

        # ensure_policy needs _first_canonical (now cached inside
        # env_to_canonical above). Eagerly load the policy here so the
        # predict node can be config-less and just call mgr.predict.
        ckpt_resolved = _resolve_checkpoint_path(spec.checkpoint_path)
        res = await _run_sync(mgr.ensure_policy, spec.policy, checkpoint_path=ckpt_resolved)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_policy: {res['error']}")
            return {"canonical": None}
        if isinstance(res, dict) and res.get("policy_module"):
            self._self_log("loaded", res)
        return {"canonical": canonical}


class AdaptCanonicalToModelTool(BaseCanvasNode):
    node_type = "policy_vlnce__adapt_canonical_to_model"
    display_name = "VLN-CE: Canonical → Model"
    description = (
        "VlnModelAdaptor.canonical_to_model — applies the architecture-"
        "specific preprocessing (vlnce_baselines obs_transforms, instruction "
        "tokenize, RNN state stack). Model adapter + exp_config are selected "
        "at run time via the policy_vlnce env panel's 'variant' dropdown — "
        "no per-graph wiring."
    )
    category = "policy"
    icon = "Boxes"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("canonical", "ANY", "Output of policy_vlnce__adapt_env_to_canonical"),
        PortDef(
            "hidden_in",
            "ANY",
            "RNN state from prior iteration (None on iter 0). Wire from iter_in.iterout_hidden.",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("model_batch", "ANY", "Model-ready batch dict for policy_vlnce__predict"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        canonical = inputs.get("canonical")
        if canonical is None:
            self._self_log("error", "canonical input missing")
            return {"model_batch": None}

        out = await _run_sync(_get_mgr().canonical_to_model, canonical, inputs.get("hidden_in"))
        if isinstance(out, dict) and "error" in out:
            self._self_log("error", out["error"])
            return {"model_batch": None}
        return {"model_batch": out["model_batch"]}


class PredictTool(BaseCanvasNode):
    node_type = "policy_vlnce__predict"
    display_name = "VLN-CE: Predict"
    description = (
        "VlnPolicy.forward — loads the checkpoint (lazy, singleton-cached on "
        "(policy_module, checkpoint_path)) and runs one forward pass. The "
        "predict subprocess hosts a single policy on GPU shared across K "
        "eval workers via BatchedInferenceServer rendezvous; the K-flush is "
        "unrolled into a sequential loop because the canonical interface "
        "produces one model_batch per sample."
    )
    category = "policy"
    icon = "Cpu"
    # batched=True hits a framework asyncio loop-affinity bug under
    # worker_count>1 (BatchedInferenceServer._delayed_flush future attached
    # to wrong event loop, see project_batched_inference_validation memory).
    # Run sequential per-sample for now — shared GPU policy still serializes
    # forward passes, so the throughput cost is negligible while env steps
    # parallelize across workers.
    batched: ClassVar[bool] = False
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("model_batch", "ANY", "Output of policy_vlnce__adapt_canonical_to_model"),
    ]
    output_ports = [
        PortDef("model_output", "ANY", "Model's forward output dict"),
        PortDef(
            "hidden_out",
            "ANY",
            "RNN state for next iteration. Wire to iter_out.hidden with persist=true.",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        # batched=True nodes always receive {SAMPLES_KEY: list}; resolve
        # this up front so error returns can use the right shape.
        raw_samples = inputs.get(SAMPLES_KEY)
        is_batched_call = isinstance(raw_samples, list)
        samples = raw_samples if is_batched_call else [inputs]
        n_samples = len(samples)
        log.info(
            "PredictTool.execute: is_batched=%s n_samples=%d",
            is_batched_call,
            n_samples,
        )

        def _err_return(msg: str) -> dict:
            log.error("PredictTool err: %s", msg)
            self._self_log("error", msg)
            sample_err = {"model_output": None, "hidden_out": None}
            if is_batched_call:
                return {OUTPUTS_KEY: [sample_err for _ in range(n_samples)]}
            return sample_err

        # Stage 1 already loaded env+model+policy (variant on chain entry).
        # If predict runs before stage 1 has run, mgr.predict will return
        # an error which we surface — there's no fallback ensure_policy here.

        outputs: list[dict] = []
        for sample in samples:
            batch = sample.get("model_batch")
            if batch is None:
                self._self_log("error", "model_batch input missing")
                outputs.append({"model_output": None, "hidden_out": None})
                continue
            out = await _run_sync(_get_mgr().predict, batch)
            if isinstance(out, dict) and "error" in out:
                self._self_log("error", out["error"])
                outputs.append({"model_output": None, "hidden_out": None})
                continue
            mo = out["model_output"]
            hidden_out = {
                "rnn_states": mo["rnn_states_out"],
                "prev_actions": mo["prev_actions_out"],
                "not_done_masks": mo["not_done_masks_out"],
            }
            outputs.append({"model_output": mo, "hidden_out": hidden_out})

        if is_batched_call:
            return {OUTPUTS_KEY: outputs}
        return outputs[0]


class AdaptModelToCanonicalTool(BaseCanvasNode):
    node_type = "policy_vlnce__adapt_model_to_canonical"
    display_name = "VLN-CE: Model → Canonical"
    description = (
        "VlnModelAdaptor.model_to_canonical — extracts the action index from "
        "the policy's model_output and wraps in CanonicalDict[action]. Pure-CPU "
        "shape transform; no GPU, no env state."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("model_output", "ANY", "Output of policy_vlnce__predict"),
    ]
    output_ports = [
        PortDef("canonical_action", "ANY", "CanonicalDict[action] — feeds canonical_to_env"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        model_output = inputs.get("model_output")
        if model_output is None:
            self._self_log("error", "model_output input missing")
            return {"canonical_action": None}
        out = await _run_sync(_get_mgr().model_to_canonical_action, model_output)
        if isinstance(out, dict) and "error" in out:
            self._self_log("error", out["error"])
            return {"canonical_action": None}
        return {"canonical_action": out["canonical_action"]}


class AdaptCanonicalToEnvTool(BaseCanvasNode):
    node_type = "policy_vlnce__adapt_canonical_to_env"
    display_name = "VLN-CE: Canonical → Env Action"
    description = (
        "VlnEnvAdaptor.canonical_to_env — picks an env action index from the "
        "canonical_action. R2R-CE: int ∈ [0,3]; RxR-CE: int ∈ [0,5]."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("canonical_action", "ANY", "Output of policy_vlnce__adapt_model_to_canonical"),
    ]
    output_ports = [
        PortDef("action", "ACTION", "Discrete action index for env_*__step"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        canonical_action = inputs.get("canonical_action")
        if canonical_action is None:
            self._self_log("error", "canonical_action input missing")
            return {"action": 0}
        out = await _run_sync(_get_mgr().canonical_action_to_env, canonical_action)
        if isinstance(out, dict) and "error" in out:
            self._self_log("error", out["error"])
            return {"action": 0}
        action = int(out["action"])
        try:
            from app.standard.actions import ACTION_NAMES

            self._self_log("action_name", ACTION_NAMES.get(action, "UNKNOWN"))
        except Exception:
            pass
        self._self_log("predicted_action", action)
        return {"action": action}


# NOTE: the standalone ``policy_vlnce__tokenize_instruction`` node was retired
# 2026-06-29. Per the standardize/process split, instruction tokenization is a
# CMA-vocab-specific step and now lives in the model-side adapter
# (cma.canonical_to_model); the env adapter carries the raw instruction text.
# The tokenizer itself stays in adapters/r2r_tokenizer.py, called from cma.py.


# ══════════════════════════════════════════════════════════════════════
# PolicyVlnceNodeSet
# ══════════════════════════════════════════════════════════════════════


class PolicyVlnceNodeSet(BaseNodeSet):
    """VLN-CE inference policies (CMA, Seq2Seq, future HAMT/DUET/NaVid)."""

    name = "policy_vlnce"
    description = (
        "VLN-CE inference framework — 5 explicit canvas nodes (one per atomic "
        "adapter stage) for env→canonical, canonical→model, predict, "
        "model→canonical, canonical→env. Filename-based discovery under "
        "adapters/envs, adapters/models, policies enables drop-in extension "
        "for new VLN methods."
    )
    server_python = os.environ.get("VLNCE_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-vlnce/bin/python"))
    # The vlnce env's libicui18n (pulled in via sqlite3 by webdataset →
    # habitat_baselines.il.trainers chain) requires CXXABI_1.3.15 which
    # Ubuntu 20.04's stock libstdc++ lacks. Prepending the conda env's
    # lib/ to LD_LIBRARY_PATH selects the conda libstdc++ first.
    _vlnce_env_lib = str(Path(server_python).parent.parent / "lib")
    server_env = {
        "LD_LIBRARY_PATH": f"{_vlnce_env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    parallelism = "shared"

    def __init__(self) -> None:
        super().__init__()
        self._mgr = VlnceManager.get()

    def get_tools(self) -> list:
        return [
            AdaptEnvToCanonicalTool(),
            AdaptCanonicalToModelTool(),
            PredictTool(),
            AdaptModelToCanonicalTool(),
            AdaptCanonicalToEnvTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        if self._mgr.initialized:
            log.info("policy_vlnce already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._mgr.executor, lambda: self._mgr.initialize(**kwargs))
        log.info("PolicyVlnceNodeSet initialized")

    async def shutdown(self) -> None:
        self._mgr.shutdown()
