from __future__ import annotations

"""PolicyAdapterVlnceNodeSet — VLN-CE model-side of the General Policy Adapter pipeline.

Owns stages 2/3/4 of the 5-stage adapter pattern (mirrors
policy_adapter_vla 1:1, VLN-flavored); stages 1 & 5 (env → canonical /
canonical → env action) live in the general ``env_adapter`` nodeset
(``workspace/nodesets/env/env_adapter/envs/``), in-process in the hub. A
VLN-CE graph composes three nodesets: env_habitat + env_adapter +
policy_adapter_vlnce.

  stage 2: canonical_to_model        — CanonicalDict[obs] → model_batch
  stage 3: predict                   — model_batch → model_output
  stage 4: model_to_canonical        — model_output → CanonicalDict[action]

Filename-based discovery: every .py file under ``adapters/models/`` and
``policies/`` is auto-discovered as a canvas dropdown option. Drop a file
in / out → POST /api/components/reload.

Config distributed across nodes:
  - Stage 2 owns: ``variant`` dropdown (model + exp_config + policy +
    checkpoint bundle from variants.REGISTRY) — the chain entry.
  - Stage 3 + 4: configless (reuse manager state from 2).
  - The env side of the old variant pin (r2rce) is now the independent
    ``env_family`` select on env_adapter__vln_env_to_canonical.

Recurrent state on the wire: predict has ``hidden_in`` / ``hidden_out``
ports. Wire ``iter_in.iterout_hidden → predict.hidden_in`` and
``predict.hidden_out → iter_out.hidden`` exactly like the legacy
policy_cma graph.

last updated: 2026-07-04 (split out of the former policy_vlnce nodeset)
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
from workspace.nodesets.policy.policy_adapter_vlnce.adapters.canonical import CanonicalNavInfo
from workspace.nodesets.policy.policy_adapter_vlnce.variants import (
    DEFAULT_KEY,
    REGISTRY,
    REGISTRY_BY_KEY,
    VariantSpec,
)

log = logging.getLogger("agentcanvas.policy_adapter_vlnce")


# ══════════════════════════════════════════════════════════════════════
# Discovery — folder-scan + lazy import (mirrors policy_adapter_vla)
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
        f"workspace.nodesets.policy.policy_adapter_vlnce.{subpkg_dotted}.{module_name}"
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


# Canvas dropdown options — frozen at module-import time. (Env adapters
# are discovered by env_adapter from its envs/ folder.)
MODEL_OPTIONS: list[str] = _discover_modules("adapters/models")
POLICY_OPTIONS: list[str] = _discover_modules("policies")


# ══════════════════════════════════════════════════════════════════════
# VlnceManager — singleton policy runtime
# ══════════════════════════════════════════════════════════════════════


class VlnceManager:
    """Singleton manager hosting one VLN-CE policy on GPU.

    All torch/CUDA work runs on a pinned single-thread executor. The
    chain-entry node (canonical→model) owns the ``variant`` config and
    primes both slices via idempotent ``ensure_*`` calls that
    short-circuit when the config is unchanged.
    """

    _instance: VlnceManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="policy_adapter_vlnce",
        )
        self._initialized: bool = False

        # Slice state
        self._model_adaptor: Any = None  # VlnModelAdaptor
        self._model_module: str = ""
        self._policy: Any = None  # VlnPolicy on GPU
        self._policy_module: str = ""

        # ensure_* idempotency caches
        self._model_cfg_cache: tuple | None = None
        self._policy_cfg_cache: tuple | None = None

        # Cached first-seen canonical → used by ensure_policy to derive
        # obs_space (CMANet ctor needs it) and by model_to_canonical_action
        # for the CanonicalNavInfo (the env adapter that used to provide it
        # lives across the process boundary in env_adapter now). Set by
        # note_canonical / canonical_to_model on first sight. Independent
        # of the per-step model_batch — only the shape signature matters.
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

    def note_canonical(self, canonical: dict[str, Any]) -> None:
        """Cache the first-seen canonical (chain-entry calls this BEFORE
        ensure_policy so the policy is loaded before the first
        canonical_to_model — preserving the pre-split device placement,
        where stage 1 primed the policy eagerly)."""
        with self._lock:
            if self._first_canonical is None:
                self._first_canonical = canonical

    def ensure_model_adaptor(self, model_module: str, exp_config: str) -> dict[str, Any]:
        with self._lock:
            if not self._initialized:
                return {"error": "VlnceManager not initialized"}
            try:
                module = _load_module("adapters.models", model_module)
                from workspace.nodesets.policy.policy_adapter_vlnce.adapters.models.base_model import (
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
            if self._model_adaptor is None:
                return {"error": "model_adaptor not built — run adapt_canonical_to_model first"}
            if self._first_canonical is None:
                return {
                    "error": "no canonical observation seen yet — run adapt_canonical_to_model at least once"
                }

            try:
                module = _load_module("policies", policy_module)
                from workspace.nodesets.policy.policy_adapter_vlnce.policies.base_policy import (
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
                # The env adapter that used to provide the CanonicalNavInfo
                # lives in env_adapter now — rebuild it from the info dict
                # riding on the cached canonical (make_canonical_obs asdicts).
                action_space = self._model_adaptor.derive_action_space(
                    CanonicalNavInfo(**self._first_canonical["info"])
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
            if self._model_adaptor is None:
                return {"error": "manager not loaded"}
            if self._first_canonical is None:
                return {"error": "no canonical seen yet — run adapt_canonical_to_model first"}
            try:
                info = CanonicalNavInfo(**self._first_canonical["info"])
                ca = self._model_adaptor.model_to_canonical(model_output, info)
                return {"canonical_action": ca}
            except Exception as e:
                log.exception("model_to_canonical_action failed")
                return {"error": f"model_to_canonical_action: {e!r}"}


# ══════════════════════════════════════════════════════════════════════
# Variant resolution helpers
# ══════════════════════════════════════════════════════════════════════


def _resolve_variant_from_config(cfg: dict[str, Any]) -> VariantSpec:
    """Look up the VariantSpec from a node's config dict.

    Variant lives on the chain-entry node (stage 2, canonical→model —
    this nodeset's first pipeline stage) per the "env panels = env-side
    runtime knobs only" rule (memory: feedback_env_panel_scope). Method
    nodesets express model/ckpt choices as ConfigFields on a node, not on
    an env panel.
    """
    key = (cfg.get("variant") or "").strip() or DEFAULT_KEY
    return REGISTRY_BY_KEY.get(key, REGISTRY_BY_KEY[DEFAULT_KEY])


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve a relative checkpoint path against the project repo root.

    ``__file__`` lives at ``workspace/nodesets/policy/policy_adapter_vlnce/__init__.py``
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


class AdaptCanonicalToModelTool(BaseCanvasNode):
    node_type = "policy_adapter_vlnce__adapt_canonical_to_model"
    display_name = "VLN-CE: Canonical → Model"
    description = (
        "VlnModelAdaptor.canonical_to_model — applies the architecture-"
        "specific preprocessing (vlnce_baselines obs_transforms, instruction "
        "tokenize, RNN state stack). As this nodeset's chain entry, this is "
        "also where the VLN-CE *variant* (model + exp_config + policy + "
        "checkpoint) is selected; the dropdown lists 12 R2R-CE baselines "
        "from variants.REGISTRY. On execute it eagerly primes the "
        "VlnceManager (model adapter + policy load) so predict / "
        "model→canonical just call into the cached singleton. The env-side "
        "adapter is selected independently on env_adapter__vln_env_to_canonical."
    )
    category = "policy"
    icon = "Boxes"
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
        PortDef("canonical", "ANY", "Output of env_adapter__vln_env_to_canonical"),
        PortDef(
            "hidden_in",
            "ANY",
            "RNN state from prior iteration (None on iter 0). Wire from iter_in.iterout_hidden.",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("model_batch", "ANY", "Model-ready batch dict for policy_adapter_vlnce__predict"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        spec = _resolve_variant_from_config(self.config)
        mgr = _get_mgr()

        # Chain entry: prime both slices upfront so predict stays
        # config-less. ensure_* are idempotent caches, so cost is paid only
        # when the variant flips.
        res = await _run_sync(mgr.ensure_model_adaptor, spec.model_adaptor, spec.exp_config)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_model_adaptor: {res['error']}")
            return {"model_batch": None}

        canonical = inputs.get("canonical")
        if canonical is None:
            self._self_log("error", "canonical input missing")
            return {"model_batch": None}

        # Cache the canonical BEFORE ensure_policy (it derives obs_space
        # from it), and load the policy BEFORE canonical_to_model so the
        # first model_batch is built on the policy's device — same order
        # as the pre-split pipeline, where stage 1 primed everything.
        mgr.note_canonical(canonical)
        ckpt_resolved = _resolve_checkpoint_path(spec.checkpoint_path)
        res = await _run_sync(mgr.ensure_policy, spec.policy, checkpoint_path=ckpt_resolved)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_policy: {res['error']}")
            return {"model_batch": None}
        if isinstance(res, dict) and res.get("policy_module"):
            self._self_log("loaded", res)

        out = await _run_sync(mgr.canonical_to_model, canonical, inputs.get("hidden_in"))
        if isinstance(out, dict) and "error" in out:
            self._self_log("error", out["error"])
            return {"model_batch": None}
        return {"model_batch": out["model_batch"]}


class PredictTool(BaseCanvasNode):
    node_type = "policy_adapter_vlnce__predict"
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
        PortDef("model_batch", "ANY", "Output of policy_adapter_vlnce__adapt_canonical_to_model"),
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

        # Stage 2 already loaded model+policy (variant on the chain entry,
        # adapt_canonical_to_model). If predict runs before stage 2 has run,
        # mgr.predict will return an error which we surface — there's no
        # fallback ensure_policy here.

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
    node_type = "policy_adapter_vlnce__adapt_model_to_canonical"
    display_name = "VLN-CE: Model → Canonical"
    description = (
        "VlnModelAdaptor.model_to_canonical — extracts the action index from "
        "the policy's model_output and wraps in CanonicalDict[action]. Pure-CPU "
        "shape transform; no GPU, no env state. Its output flows to "
        "env_adapter__vln_canonical_to_env."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("model_output", "ANY", "Output of policy_adapter_vlnce__predict"),
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


# NOTE: the standalone ``tokenize_instruction`` node was retired
# 2026-06-29. Per the standardize/process split, instruction tokenization is a
# CMA-vocab-specific step and now lives in the model-side adapter
# (cma.canonical_to_model); the env adapter carries the raw instruction text.
# The tokenizer itself stays in adapters/r2r_tokenizer.py, called from cma.py.


# ══════════════════════════════════════════════════════════════════════
# PolicyAdapterVlnceNodeSet
# ══════════════════════════════════════════════════════════════════════


class PolicyAdapterVlnceNodeSet(BaseNodeSet):
    """VLN-CE inference policies (CMA, Seq2Seq, future HAMT/DUET/NaVid)."""

    name = "policy_adapter_vlnce"
    description = (
        "VLN-CE model-side of the General Policy Adapter pipeline — 3 canvas "
        "nodes: canonical→model (chain entry, owns the variant select), "
        "predict, model→canonical. Pair with env_adapter (env-side stages) "
        "and env_habitat. Filename-based discovery under adapters/models "
        "and policies enables drop-in extension for new VLN methods."
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
            AdaptCanonicalToModelTool(),
            PredictTool(),
            AdaptModelToCanonicalTool(),
        ]

    async def initialize(self, **kwargs: Any) -> None:
        if self._mgr.initialized:
            log.info("policy_adapter_vlnce already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._mgr.executor, lambda: self._mgr.initialize(**kwargs))
        log.info("PolicyAdapterVlnceNodeSet initialized")

    async def shutdown(self) -> None:
        self._mgr.shutdown()
