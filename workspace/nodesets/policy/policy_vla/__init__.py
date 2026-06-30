from __future__ import annotations

"""PolicyVlaNodeSet — generic VLA inference framework as a NodeSet.

Runs any vision-language-action policy on any robot env via the **adapter
system** vendored from the vlaworkspace training repo (``src/vlaworkspace/adaptors/``).
Inference is decomposed into 4 atomic translation stages mediated by a
``CanonicalDict`` lingua franca; the canvas exposes each stage as one node
(plus a 5th node for the model forward pass itself).

Architecture — three layers:

1. ``VlaPolicyManager`` (singleton)
     Owns one BasePolicy on GPU + one ModelAdaptor + one LiberoRobot.
     Single-thread executor enforces CUDA thread affinity. Three idempotent
     ``ensure_*`` methods (``ensure_robot`` / ``ensure_model_adaptor`` /
     ``ensure_policy``) — one per slice — let each adapter node own its
     own config; cache hits short-circuit, config changes rebuild only the
     affected slice.

2. Five canvas tool nodes — 1:1 with the 4 adapter stages + 1 model forward
     policy_vla__adapt_env_to_canonical    — stage 1: env→canonical
                                             owns robot_config_json → ensure_robot
     policy_vla__adapt_canonical_to_model  — stage 2: canonical→model_batch
                                             owns model + norm_stats_path +
                                             model_config_json → ensure_model_adaptor
     policy_vla__predict                   — model forward (BasePolicy.predict_action)
                                             owns checkpoint_path + num_inference_steps +
                                             policy_config_json → ensure_policy
                                             (heavy: GPU load + Adaptor compose)
     policy_vla__adapt_model_to_canonical  — stage 3: model_output→canonical_action
                                             no config
     policy_vla__adapt_canonical_to_env    — stage 4: canonical_action→env_action
                                             no config; takes current_state for
                                             delta-action robots

3. ``PolicyVlaNodeSet`` (collection + lifecycle)
     server_python defaults to ``$VLA_POLICY_PYTHON`` (env created by
     ``scripts/install/install_ac_vla_policy.sh``). ``parallelism="shared"``
     (mirrors policy_cma): one subprocess hosts one VlaPolicyManager
     singleton; K eval workers fan in through ``BatchedInferenceServer`` on
     the ``policy_vla__predict`` node. RT-1's recurrent ``policy_state``
     travels on the wire via the new ``policy_state_in/out`` ports +
     IterIn/IterOut feedback; Pi0/SmolVLA/DP/Droid-DP are stateless and
     leave those ports unwired.

Wire format between adapter nodes (opaque ``ANY`` dicts):
    canonical:        CanonicalDict[obs] ({data: {images, state, actions, prompt}, info})
    model_batch:      dict (model-specific shape — see ModelAdaptor.model_input())
    model_output:     dict ({"action": tensor [K, action_dim]}, model-specific extras)
    canonical_action: CanonicalDict[action] ({data: {actions: {pos,rot,gripper,joint_position}}, info})

Action chunk emitted by adapt_canonical_to_env (TEXT JSON):
    "[[ax,ay,az,arx,ary,arz,grip], …]" — feeds env_libero__step_continuous.action

The 4 currently-shipped models (Pi0, SmolVLA, DP, DROID-DP) all pair with
LiberoRobot — only one column of the model x robot Cartesian product is
filled in at this round. DROID DP reuses DPModel (confirmed via vlaworkspace
hydra config inspection 2026-05-01).

last updated: 2026-05-03
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

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef
from app.server.batched_inference import OUTPUTS_KEY, SAMPLES_KEY

log = logging.getLogger("agentcanvas.policy_vla")


# ══════════════════════════════════════════════════════════════════════
# Model registry — maps the entry node's "model" config selection → (ModelAdaptor, Policy) classes.
# Both DP and DROID DP use DPModel (confirmed via hydra config:
# diffusion_policy_libero.yaml + droid_dp_mlp_libero.yaml).
# ══════════════════════════════════════════════════════════════════════

_MODEL_TYPES: list[str] = ["pi0", "smolvla", "dp", "droid_dp", "rt1"]



# ══════════════════════════════════════════════════════════════════════
# Discovery — folder-scan + lazy import
# ──────────────────────────────────────────────────────────────────────
# Each .py file under adapters/robots/, adapters/models/, policies/ that
# defines a {RobotAdaptor, ModelAdaptor, BasePolicy} subclass and a module-
# level DEFAULT_KWARGS dict is auto-discovered as a canvas dropdown option.
# Drop in / drop out: add or delete a file → POST /api/components/reload.
# ══════════════════════════════════════════════════════════════════════

_PKG_ROOT = Path(__file__).parent
_DISCOVERY_EXCLUDES: tuple[str, ...] = ("__init__", "dp_defaults")


def _discover_modules(subpkg_relpath: str) -> list[str]:
    """Filename-based discovery — sorted .py stems in a sub-folder."""
    pkg_dir = _PKG_ROOT / subpkg_relpath
    return sorted(
        f.stem
        for f in pkg_dir.glob("*.py")
        if f.stem not in _DISCOVERY_EXCLUDES
        and not f.stem.startswith("_")
        and not f.stem.startswith("base_")
    )


def _load_module(subpkg_dotted: str, module_name: str) -> Any:
    """Lazy import of policy_vla.<subpkg>.<module_name>."""
    return importlib.import_module(
        f"workspace.nodesets.policy.policy_vla.{subpkg_dotted}.{module_name}"
    )


def _find_subclass(module: Any, base: type) -> type:
    """Return the (single) subclass of `base` defined or aliased into module.

    For variant files that re-export a class (e.g. libero_robot_absolute.py
    importing LiberoRobot), this still returns the class — different files
    can share a class but carry different DEFAULT_KWARGS.
    """
    found = []
    for v in vars(module).values():
        if not isinstance(v, type) or v is base:
            continue
        try:
            if issubclass(v, base):
                found.append(v)
        except TypeError:
            # Generic aliases (e.g. ``dict[str, Any]``) pass isinstance(_, type)
            # in 3.10 but are not real classes — issubclass raises here.
            continue
    if not found:
        raise ValueError(f"no {base.__name__} subclass found in {module.__name__}")
    if len(found) > 1:
        # Prefer class defined in this module over imported aliases.
        local = [c for c in found if c.__module__ == module.__name__]
        if local:
            return local[0]
    return found[0]


def _instantiate(spec: Any) -> Any:
    """Hydra-style ``_target_:`` resolution in nested DEFAULT_KWARGS values.

    DP / DROID-DP DEFAULT_KWARGS contain ``noise_scheduler: {_target_: ...}``;
    this turns them into actual scheduler instances. Plain values pass through.
    """
    if isinstance(spec, dict):
        if "_target_" in spec:
            target_path = spec["_target_"]
            kwargs = {k: _instantiate(v) for k, v in spec.items() if k != "_target_"}
            module_name, _, class_name = target_path.rpartition(".")
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)(**kwargs)
        return {k: _instantiate(v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [_instantiate(v) for v in spec]
    return spec


# Canvas dropdown options — frozen at module-import time into the
# AutoServerApp manifest's config_fields.options.
ROBOT_OPTIONS: list[str] = _discover_modules("adapters/robots")
MODEL_OPTIONS: list[str] = _discover_modules("adapters/models")
POLICY_OPTIONS: list[str] = _discover_modules("policies")


# ══════════════════════════════════════════════════════════════════════
# VlaPolicyManager — singleton policy runtime
# ══════════════════════════════════════════════════════════════════════


class VlaPolicyManager:
    """Singleton manager hosting one VLA policy on GPU.

    All torch/CUDA work runs on a pinned single-thread executor. The
    chain-entry node's lazy-load calls ``ensure_model_adaptor`` / ``ensure_policy``, which drop any
    previously loaded policy from GPU before instantiating the new one.
    """

    _instance: VlaPolicyManager | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="policy_vla",
        )

        # Static state
        self._initialized: bool = False

        # Episode-scoped (loaded after Env panel "load")
        self._robot: Any = None  # LiberoRobot
        self._model_type: str = ""
        self._model_adaptor: Any = None  # ModelAdaptor instance
        self._policy: Any = None  # BasePolicy on GPU
        self._adaptor: Any = None  # composed Adaptor
        self._loaded_meta: dict[str, Any] = {}

        # Config caches keyed per-node — drive ensure_* idempotency.
        # Each adapter node (env_to_canonical / canonical_to_model / predict)
        # owns one cache; ensure_* short-circuits when its node's config is
        # unchanged. None = never built; a value = last successfully built config.
        self._robot_cfg_cache: tuple | None = None
        self._model_adaptor_cfg_cache: tuple | None = None
        self._policy_cfg_cache: tuple | None = None

    # ── Singleton + lifecycle ──────────────────────────────────────────

    @classmethod
    def get(cls) -> VlaPolicyManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def executor(self) -> concurrent.futures.Executor:
        return self._executor

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def policy_loaded(self) -> bool:
        return self._policy is not None

    def initialize(self, **kwargs: Any) -> None:
        """Construct the LiberoRobot. Heavy model loads happen in load_model."""
        with self._lock:
            from workspace.nodesets.policy.policy_vla.adapters import LiberoRobot

            # Optional default_prompt for env-mode (Manager, not nodeset, owns this)
            default_prompt = kwargs.get("default_prompt")
            self._robot = LiberoRobot(default_prompt=default_prompt)
            self._initialized = True
            log.info("VlaPolicyManager: LiberoRobot ready (default_prompt=%r)", default_prompt)

    def shutdown(self) -> None:
        with self._lock:
            self._unload_unlocked()
            self._robot = None
            self._initialized = False

    def _unload_unlocked(self) -> None:
        """Drop the loaded policy from GPU and clear adapter state.

        Caller must hold the lock.
        """
        if self._policy is not None:
            try:
                # Move to CPU first so CUDA buffers are freed, then drop.
                self._policy.to("cpu")
            except Exception:
                log.debug("policy.to('cpu') raised (non-fatal)", exc_info=True)
            del self._policy
            self._policy = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._model_adaptor = None
        self._adaptor = None
        self._model_type = ""
        self._loaded_meta = {}

    # ── Model load / unload ────────────────────────────────────────────


    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._unload_unlocked()
            self._robot_cfg_cache = None
            self._model_adaptor_cfg_cache = None
            self._policy_cfg_cache = None
            return {"unloaded": True}

    # ── Distributed ensure_*: each adapter node owns one slice ─────────

    def _ensure_adaptor_unlocked(self) -> None:
        """(Re)compose Adaptor from current robot + model_adaptor. Caller holds lock."""
        if self._robot is None or self._model_adaptor is None:
            self._adaptor = None
            return
        from workspace.nodesets.policy.policy_vla.adapters import Adaptor

        self._adaptor = Adaptor(robot=self._robot, model=self._model_adaptor)
        self._adaptor.eval()

    def ensure_robot(self, robot_module: str) -> dict[str, Any]:
        """(Re)build robot adapter from ``adapters/robots/<robot_module>.py``.

        Owned by node ``policy_vla__adapt_env_to_canonical``. The selected
        module's ``DEFAULT_KWARGS`` provides ctor args; previously-set
        ``default_prompt`` is preserved across rebuilds. Module change
        invalidates the composed Adaptor so ensure_policy will recompose.
        """
        with self._lock:
            if not self._initialized:
                return {"error": "VlaPolicyManager not initialized"}
            try:
                module = _load_module("adapters.robots", robot_module)
                from workspace.nodesets.policy.policy_vla.adapters.robots.base_robot import (
                    RobotAdaptor,
                )

                cls = _find_subclass(module, RobotAdaptor)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
            except Exception as e:
                return {"error": f"load robot module {robot_module!r}: {e!r}"}

            cache_key = (robot_module, json.dumps(kwargs, sort_keys=True, default=str))
            if cache_key == self._robot_cfg_cache and self._robot is not None:
                return {"unchanged": True}

            kwargs.setdefault(
                "default_prompt",
                getattr(self._robot, "default_prompt", None) if self._robot else None,
            )
            try:
                kwargs_resolved = _instantiate(kwargs)
                self._robot = cls(**kwargs_resolved)
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}
            self._robot_cfg_cache = cache_key
            self._adaptor = None  # force recompose
            log.info("ensure_robot rebuilt: module=%s, kwargs=%s", robot_module, kwargs)
            return {"rebuilt": True, "module": robot_module, "config": kwargs}

    def ensure_model_adaptor(self, model_module: str) -> dict[str, Any]:
        """(Re)build model adapter from ``adapters/models/<model_module>.py``.

        Owned by node ``policy_vla__adapt_canonical_to_model``. The selected
        module's ``DEFAULT_KWARGS`` provides ctor args. Module change drops
        the loaded policy (Pi0Policy ≠ SmolVLAPolicy etc.).
        """
        with self._lock:
            if not self._initialized:
                return {"error": "VlaPolicyManager not initialized"}
            try:
                module = _load_module("adapters.models", model_module)
                from workspace.nodesets.policy.policy_vla.adapters.models.base_model import (
                    ModelAdaptor,
                )

                cls = _find_subclass(module, ModelAdaptor)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
            except Exception as e:
                return {"error": f"load model module {model_module!r}: {e!r}"}

            cache_key = (model_module, json.dumps(kwargs, sort_keys=True, default=str))
            if cache_key == self._model_adaptor_cfg_cache and self._model_adaptor is not None:
                return {"unchanged": True}

            # Module change → drop policy (paired with previous adapter class).
            prev = getattr(self, "_model_module", "")
            if model_module != prev and self._policy is not None:
                log.info("model module %r → %r: dropping policy", prev, model_module)
                self._unload_unlocked()
                self._policy_cfg_cache = None

            try:
                kwargs_resolved = _instantiate(kwargs)
                self._model_adaptor = cls(**kwargs_resolved)
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}
            self._model_module = model_module
            # Legacy `_model_type` field — kept so get_status()
            # still work; derived from filename.
            self._model_type = model_module.replace("_model", "")
            self._model_adaptor_cfg_cache = cache_key
            self._adaptor = None  # force recompose
            log.info("ensure_model_adaptor rebuilt: module=%s", model_module)
            return {"rebuilt": True, "module": model_module}

    def ensure_policy(
        self,
        policy_module: str,
        *,
        checkpoint_path: str = "",
        num_inference_steps: int | None = None,
        device: str = "cuda",
    ) -> dict[str, Any]:
        """(Re)build policy from ``policies/<policy_module>.py`` and load weights.

        Owned by node ``policy_vla__predict``. Heavy: builds the GPU policy,
        loads weights, recomposes Adaptor (since robot or model_adaptor may
        have been rebuilt and cleared self._adaptor). The selected module's
        ``DEFAULT_KWARGS`` provides ctor args; ``num_inference_steps`` (if
        non-None) overrides the default at the right kwarg name.
        """
        with self._lock:
            if not self._initialized:
                return {"error": "VlaPolicyManager not initialized"}
            if self._model_adaptor is None:
                return {"error": "model_adaptor not built — run adapt_canonical_to_model first"}
            try:
                module = _load_module("policies", policy_module)
                from workspace.nodesets.policy.policy_vla.policies.base_policy import BasePolicy

                cls = _find_subclass(module, BasePolicy)
                kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
            except Exception as e:
                return {"error": f"load policy module {policy_module!r}: {e!r}"}

            if num_inference_steps is not None:
                # Different policy classes name this kwarg differently; respect
                # whichever the module's DEFAULT_KWARGS already chose.
                if "num_inference_steps" in kwargs:
                    kwargs["num_inference_steps"] = num_inference_steps
                elif "num_steps" in kwargs:
                    kwargs["num_steps"] = num_inference_steps
                else:
                    kwargs["num_inference_steps"] = num_inference_steps

            cache_key = (
                policy_module,
                checkpoint_path,
                json.dumps(kwargs, sort_keys=True, default=str),
            )
            if cache_key == self._policy_cfg_cache and self._policy is not None:
                if self._adaptor is None:
                    self._ensure_adaptor_unlocked()
                return {"unchanged": True}

            # Drop any stale policy.
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

            try:
                kwargs_resolved = _instantiate(kwargs)
                policy = cls(**kwargs_resolved)
            except Exception as e:
                return {"error": f"{cls.__name__}(**{kwargs!r}): {e!r}"}

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
            self._ensure_adaptor_unlocked()

            try:
                total_params = sum(p.numel() for p in policy.parameters())
            except Exception:
                total_params = -1
            try:
                gpu_mem_mb = self._gpu_memory_mb()
            except Exception:
                gpu_mem_mb = -1

            self._loaded_meta = {
                "policy_module": policy_module,
                "checkpoint_path": checkpoint_path,
                "total_params": int(total_params),
                "gpu_mem_mb": gpu_mem_mb,
                "device": str(getattr(policy, "device", "?")),
            }
            log.info("ensure_policy loaded: %s", self._loaded_meta)
            return dict(self._loaded_meta)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "initialized": self._initialized,
                "policy_loaded": self._policy is not None,
                "model_type": self._model_type,
                "loaded_meta": dict(self._loaded_meta),
            }

    @staticmethod
    def _gpu_memory_mb() -> int:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return 0

    # ── Inference path: env → canonical → model → output → env action ──

    def env_to_canonical(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._robot is None:
                return {"error": "manager not initialized"}
            try:
                return self._robot.env_to_canonical(env_obs)
            except Exception as e:
                log.exception("env_to_canonical failed")
                return {"error": f"env_to_canonical: {e!r}"}

    def canonical_to_model(self, canonical: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._model_adaptor is None:
                return {"error": "no model loaded"}
            try:
                return self._model_adaptor.canonical_to_model(canonical)
            except Exception as e:
                log.exception("canonical_to_model failed")
                return {"error": f"canonical_to_model: {e!r}"}

    def predict(self, model_batch: dict[str, Any]) -> dict[str, Any]:
        """Numpy/dict batch → torch.tensors → policy.predict_action → numpy dict.

        Adds a batch dim (B=1) when missing; squeezes it on return.

        ``policy_state_in`` (RT-1 recurrent state — nested numpy dict whose
        leaves already carry their own batch axis) is peeled off before
        ``_batch_to_torch`` recursion and re-attached as-is, since the
        helper would otherwise unsqueeze every leaf and break the TF
        ``policy_state`` structure.
        """
        with self._lock:
            if self._policy is None:
                return {"error": "no model loaded"}
            try:
                import torch

                device = self._policy.device

                policy_state_in = None
                if isinstance(model_batch, dict) and "policy_state_in" in model_batch:
                    policy_state_in = model_batch["policy_state_in"]
                    model_batch = {k: v for k, v in model_batch.items() if k != "policy_state_in"}

                # Don't pre-cast input dtype here — each policy handles its own
                # dtype conversion in predict_action(). Pi0 casts to bf16 via
                # Pi0Observation.to(device, dtype=dtype); SmolVLA expects float32
                # inputs (its bf16 VLM internals do their own cast). Pre-casting
                # to policy.dtype breaks SmolVLA with mat1/mat2 dtype mismatch.
                torch_batch = self._batch_to_torch(model_batch, device, dtype=None)
                if policy_state_in is not None and isinstance(torch_batch, dict):
                    torch_batch["policy_state_in"] = policy_state_in

                with torch.no_grad():
                    out = self._policy.predict_action(torch_batch)

                # Normalize policy output → dict with at least "action".
                if not isinstance(out, dict):
                    out = {"action": out}

                np_out: dict[str, Any] = {}
                for k, v in out.items():
                    if hasattr(v, "detach"):
                        arr = v.detach().to("cpu").numpy()
                        # Squeeze batch dim if it's 1.
                        if arr.ndim >= 2 and arr.shape[0] == 1:
                            arr = arr[0]
                        np_out[k] = arr
                    elif k == "policy_state":
                        # RT-1 recurrent state: numpy-nested dict (already serialised
                        # by the policy). Pass through opaque — must NOT be coerced
                        # via np.asarray, which would flatten the nest.
                        np_out[k] = v
                    else:
                        np_out[k] = v
                return np_out
            except Exception as e:
                log.exception("predict failed")
                return {"error": f"predict: {e!r}"}

    def model_to_canonical_action(
        self,
        model_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage 3: model_output → canonical_action (CanonicalDict[action]).

        Calls ``ModelAdaptor.model_to_canonical``. Pure-CPU shape transform; no
        GPU, no env state. ``CanonicalInfo`` comes from the loaded RobotAdaptor.

        Returns ``{"canonical_action": CanonicalDict}`` on success.
        """
        with self._lock:
            if self._model_adaptor is None or self._robot is None:
                return {"error": "no model loaded"}
            try:
                info = self._robot.get_canonical_info()
                canonical_action = self._model_adaptor.model_to_canonical(model_output, info)
                return {"canonical_action": canonical_action}
            except Exception as e:
                log.exception("model_to_canonical_action failed")
                return {"error": f"model_to_canonical_action: {e!r}"}

    def canonical_action_to_env(
        self,
        canonical_action: dict[str, Any],
        current_state: np.ndarray | None,
    ) -> dict[str, Any]:
        """Stage 4: canonical_action → env_action.

        Calls ``RobotAdaptor.canonical_to_env``. Needs ``current_state`` to
        anchor delta actions on the current eef pose (LIBERO Pi0/DP/DROID-DP).

        Returns ``{"actions": np.ndarray (K, 7)}`` consumable by env_*__step.
        """
        with self._lock:
            if self._robot is None:
                return {"error": "robot not initialized"}
            try:
                state: dict[str, Any] | None = None
                if current_state is not None:
                    state = {"state": np.asarray(current_state, dtype=np.float32)}
                return self._robot.canonical_to_env(canonical_action, state=state)
            except Exception as e:
                log.exception("canonical_action_to_env failed")
                return {"error": f"canonical_action_to_env: {e!r}"}

    def model_to_env(
        self,
        model_output: dict[str, Any],
        current_state: np.ndarray | None,
    ) -> dict[str, Any]:
        """Backwards-compat alias — fused stage 3+4.

        Calls :meth:`model_to_canonical_action` then :meth:`canonical_action_to_env`.
        Canvas now uses the split path via two nodes; this entry point is kept
        as a back-compat alias for any direct manager callers.
        """
        res = self.model_to_canonical_action(model_output)
        if "error" in res:
            return res
        return self.canonical_action_to_env(res["canonical_action"], current_state)

    @staticmethod
    def _batch_to_torch(batch: Any, device: Any, dtype: Any) -> Any:
        """Recursively convert numpy arrays to torch tensors with a B=1 batch dim.

        Floating-point tensors are cast to ``dtype``; integer / bool kept as-is.
        Strings, ints, etc. pass through unchanged.
        """
        import torch

        if isinstance(batch, dict):
            return {k: VlaPolicyManager._batch_to_torch(v, device, dtype) for k, v in batch.items()}
        if isinstance(batch, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(batch))
            # Add batch dim if not present (heuristic: arrays that look unbatched).
            t = t.unsqueeze(0)
            t = t.to(device)
            if t.is_floating_point() and dtype is not None:
                t = t.to(dtype)
            return t
        if isinstance(batch, (np.bool_, bool)):
            return torch.tensor([bool(batch)], device=device)
        return batch  # str / int / None passthrough


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════


def _get_mgr() -> VlaPolicyManager:
    return VlaPolicyManager.get()


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    if kwargs:
        # functools.partial workaround for kwargs over run_in_executor.
        import functools

        return await loop.run_in_executor(
            _get_mgr().executor, functools.partial(fn, *args, **kwargs)
        )
    return await loop.run_in_executor(_get_mgr().executor, fn, *args)


def _maybe_json(s: Any) -> Any:
    """Decode a possibly-JSON string; pass through other types."""
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


def _to_list(arr: Any) -> Any:
    """Convert ndarray → nested list (JSON-serializable). Pass through other types."""
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32).tolist()
    return arr


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes — 3 adapters + 1 raw predict
# ══════════════════════════════════════════════════════════════════════


_POLICY_COLOR = "violet"


class AdaptEnvToCanonicalTool(BaseCanvasNode):
    node_type = "policy_vla__adapt_env_to_canonical"
    display_name = "VLA: Env → Canonical"
    description = (
        "RobotAdaptor.env_to_canonical — wraps env obs (image, wrist_image, "
        "state, prompt) into the canonical intermediate format. The dropdown "
        "lists every .py file under adapters/robots/ — drop a file in or out "
        "of that folder to add or remove a variant; each file owns its own "
        "DEFAULT_KWARGS (e.g. delta vs absolute action mode)."
    )
    category = "policy"
    icon = "ArrowRight"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_POLICY_COLOR,
        config_fields=[
            ConfigField(
                "robot",
                "select",
                label="Robot adapter",
                default=ROBOT_OPTIONS[0] if ROBOT_OPTIONS else "",
                options=[{"value": s, "label": s} for s in ROBOT_OPTIONS],
            ),
        ],
    )
    input_ports = [
        PortDef("image", "IMAGE", "Front camera (HxWx3 uint8 or CHW float)"),
        PortDef("wrist_image", "IMAGE", "Wrist camera", optional=True),
        PortDef("state", "ANY", "State vector (shape varies by robot adapter)"),
        PortDef("prompt", "TEXT", "Task language instruction", optional=True),
    ]
    output_ports = [
        PortDef("canonical", "ANY", "CanonicalDict ({data, info}) for ModelAdaptor consumption"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        robot_module = (self.config.get("robot") or "").strip()
        if not robot_module:
            self._self_log("error", "no robot adapter selected")
            return {"canonical": None}
        res = await _run_sync(_get_mgr().ensure_robot, robot_module)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_robot({robot_module!r}): {res['error']}")
            return {"canonical": None}

        env_obs = {
            "image": inputs.get("image"),
            "wrist_image": inputs.get("wrist_image"),
            "state": inputs.get("state"),
            "prompt": inputs.get("prompt") or "",
        }
        env_obs = {k: v for k, v in env_obs.items() if v is not None or k == "prompt"}
        canonical = await _run_sync(_get_mgr().env_to_canonical, env_obs)
        if isinstance(canonical, dict) and "error" in canonical:
            self._self_log("error", canonical["error"])
            return {"canonical": None}
        return {"canonical": canonical}


class AdaptCanonicalToModelTool(BaseCanvasNode):
    node_type = "policy_vla__adapt_canonical_to_model"
    display_name = "VLA: Canonical → Model"
    description = (
        "ModelAdaptor.canonical_to_model — model-specific normalize / tokenize "
        "/ pad / image format. The dropdown lists every .py file under "
        "adapters/models/; each file owns its DEFAULT_KWARGS (norm_stats path, "
        "max_token_len, ...). Changing this drops the loaded policy."
    )
    category = "policy"
    icon = "Boxes"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_POLICY_COLOR,
        config_fields=[
            ConfigField(
                "model",
                "select",
                label="Model adapter",
                default=MODEL_OPTIONS[0] if MODEL_OPTIONS else "",
                options=[{"value": s, "label": s} for s in MODEL_OPTIONS],
            ),
        ],
    )
    input_ports = [
        PortDef("canonical", "ANY", "Output of policy_vla__adapt_env_to_canonical"),
    ]
    output_ports = [
        PortDef("model_batch", "ANY", "Numpy dict in the loaded model's input shape"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        model_module = (self.config.get("model") or "").strip()
        if not model_module:
            self._self_log("error", "no model adapter selected")
            return {"model_batch": None}
        res = await _run_sync(_get_mgr().ensure_model_adaptor, model_module)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", f"ensure_model_adaptor({model_module!r}): {res['error']}")
            return {"model_batch": None}

        canonical = inputs.get("canonical")
        if canonical is None:
            self._self_log("error", "canonical input missing")
            return {"model_batch": None}
        batch = await _run_sync(_get_mgr().canonical_to_model, canonical)
        if isinstance(batch, dict) and "error" in batch:
            self._self_log("error", batch["error"])
            return {"model_batch": None}
        return {"model_batch": batch}


class PredictTool(BaseCanvasNode):
    node_type = "policy_vla__predict"
    display_name = "VLA: Predict"
    description = (
        "BasePolicy.predict_action — raw model I/O. The dropdown lists every "
        ".py file under policies/; each file owns its DEFAULT_KWARGS. "
        "checkpoint_path is per-experiment (different seeds / finetune steps "
        "for the same policy class) so it stays as a free text field. "
        "Requires policy_vla__adapt_canonical_to_model to have run first. "
        "Optional policy_state_in/out ports thread RT-1's recurrent state "
        "through an IterIn/IterOut feedback edge — leave them unwired for "
        "stateless policies (Pi0/SmolVLA/DP/Droid-DP)."
    )
    category = "policy"
    icon = "Cpu"
    # ADR-eval-002 PC-3 + this nodeset's parallelism="shared": K eval workers
    # rendezvous at this node through BatchedInferenceServer; one TF SavedModel
    # serves all of them. The TF policy is pinned to batch_size=1 so the
    # K-flush is unrolled into a sequential loop inside execute() — primary
    # win is GPU memory (one model vs N replicas), throughput is secondary.
    batched: ClassVar[bool] = True
    batch_dim: ClassVar[str] = "model_batch"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_POLICY_COLOR,
        config_fields=[
            ConfigField(
                "policy",
                "select",
                label="Policy",
                default=POLICY_OPTIONS[0] if POLICY_OPTIONS else "",
                options=[{"value": s, "label": s} for s in POLICY_OPTIONS],
            ),
            ConfigField(
                "checkpoint_path",
                "text",
                label="Checkpoint path",
                default="",
                placeholder="model.safetensors / Lightning .ckpt / dir with model.safetensors",
            ),
            ConfigField(
                "num_inference_steps",
                "number",
                label="Num inference steps (override)",
                default=10,
            ),
        ],
    )
    input_ports = [
        PortDef("model_batch", "ANY", "Output of policy_vla__adapt_canonical_to_model"),
        PortDef(
            "policy_state_in",
            "ANY",
            "Recurrent policy state from prior iteration (RT-1 only; None on step 0). "
            "Wire from iter_in.iterout_policy_state when the policy is recurrent.",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("model_output", "ANY", "Model's native output dict (e.g. {'action': ndarray})"),
        PortDef(
            "policy_state_out",
            "ANY",
            "Recurrent policy state for next iteration (RT-1 only; None for "
            "stateless policies). Wire to iter_out.policy_state with persist=true.",
            optional=True,
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        policy_module = (self.config.get("policy") or "").strip()
        if not policy_module:
            self._self_log("error", "no policy selected")
            return {"model_output": None, "policy_state_out": None}
        ckpt = (self.config.get("checkpoint_path") or "").strip()
        try:
            n_steps = int(self.config.get("num_inference_steps", 10))
        except (TypeError, ValueError):
            n_steps = 10

        res = await _run_sync(
            _get_mgr().ensure_policy,
            policy_module,
            checkpoint_path=ckpt,
            num_inference_steps=n_steps,
        )
        if isinstance(res, dict) and "error" in res:
            log.error("PredictTool: ensure_policy(%r) failed: %s", policy_module, res["error"])
            self._self_log("error", f"ensure_policy({policy_module!r}): {res['error']}")
            return {"model_output": None, "policy_state_out": None}
        if isinstance(res, dict) and res.get("policy_module"):
            self._self_log("loaded", res)

        # BatchedInferenceServer (ADR-eval-002 PC) hands us either a single
        # call (canvas Play / worker_count=1 → no rendezvous) or a list of
        # samples under SAMPLES_KEY (multi-worker fan-out). Promote the
        # single-call path to a length-1 batch so the loop body is uniform.
        raw_samples = inputs.get(SAMPLES_KEY)
        is_batched_call = isinstance(raw_samples, list)
        samples = raw_samples if is_batched_call else [inputs]

        outputs: list[dict[str, Any]] = []
        for sample in samples:
            batch = sample.get("model_batch")
            if batch is None:
                log.error("PredictTool: model_batch input missing")
                self._self_log("error", "model_batch input missing")
                outputs.append({"model_output": None, "policy_state_out": None})
                continue
            # Inject policy_state_in into the batch so VlaPolicyManager.predict
            # can pass it straight through to BasePolicy.predict_action without
            # a special-case manager method. Stateless policies will never
            # read this key.
            ps_in = sample.get("policy_state_in")
            if ps_in is not None:
                if isinstance(batch, dict):
                    batch = {**batch, "policy_state_in": ps_in}
                else:  # exotic non-dict batch — leave alone
                    log.warning("PredictTool: model_batch not a dict, dropping policy_state_in")
            out = await _run_sync(_get_mgr().predict, batch)
            if isinstance(out, dict) and "error" in out:
                log.error("PredictTool: predict failed: %s", out["error"])
                self._self_log("error", out["error"])
                outputs.append({"model_output": None, "policy_state_out": None})
                continue
            policy_state_out = (
                out.pop("policy_state") if isinstance(out, dict) and "policy_state" in out else None
            )
            if isinstance(out, dict) and "action" in out and hasattr(out["action"], "shape"):
                self._self_log("action_shape", tuple(out["action"].shape))
            outputs.append({"model_output": out, "policy_state_out": policy_state_out})

        if is_batched_call:
            return {OUTPUTS_KEY: outputs}
        return outputs[0]


class AdaptModelToCanonicalTool(BaseCanvasNode):
    node_type = "policy_vla__adapt_model_to_canonical"
    display_name = "VLA: Model → Canonical"
    description = (
        "ModelAdaptor.model_to_canonical — converts raw model output to a "
        "CanonicalDict[action] via the loaded ModelAdaptor + RobotAdaptor's "
        "CanonicalInfo. Pure-CPU shape/normalize transform; no GPU, no state. "
        "This is stage 3 of the 4-stage adapter pipeline; its output flows to "
        "policy_vla__adapt_canonical_to_env."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("model_output", "ANY", "Output of policy_vla__predict"),
    ]
    output_ports = [
        PortDef("canonical_action", "ANY", "CanonicalDict[action] — feeds canonical_to_env"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        model_output = inputs.get("model_output")
        if model_output is None:
            self._self_log("error", "model_output input missing")
            return {"canonical_action": None}
        res = await _run_sync(_get_mgr().model_to_canonical_action, model_output)
        if isinstance(res, dict) and "error" in res:
            self._self_log("error", res["error"])
            return {"canonical_action": None}
        return {"canonical_action": res.get("canonical_action")}


class AdaptCanonicalToEnvTool(BaseCanvasNode):
    node_type = "policy_vla__adapt_canonical_to_env"
    display_name = "VLA: Canonical → Env Action"
    description = (
        "RobotAdaptor.canonical_to_env — converts a CanonicalDict[action] to a "
        "JSON action chunk consumable by env_*__step.action. Needs current_state "
        "for delta-action robots (LIBERO Pi0/DP/DROID-DP) to anchor delta on "
        "the current eef pose. This is stage 4 of the 4-stage adapter pipeline."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("canonical_action", "ANY", "Output of policy_vla__adapt_model_to_canonical"),
        PortDef(
            "current_state",
            "ANY",
            "Current 8-D env state — required for delta-action robots; LIBERO defaults to absolute.",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("action_chunk", "TEXT", "JSON: [[ax,ay,az,arx,ary,arz,grip], ...]"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        canonical_action = inputs.get("canonical_action")
        if canonical_action is None:
            self._self_log("error", "canonical_action input missing")
            return {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]])}
        current_state = inputs.get("current_state")
        env_action = await _run_sync(
            _get_mgr().canonical_action_to_env, canonical_action, current_state
        )
        if isinstance(env_action, dict) and "error" in env_action:
            self._self_log("error", env_action["error"])
            return {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]])}

        actions = env_action.get("actions")
        if actions is None:
            self._self_log("error", "no 'actions' key in env_action")
            return {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]])}
        chunk = _to_list(actions)
        # Emit ONLY a list-of-lists so env_libero__step_continuous's parser sees a (K, 7) chunk
        # — even for K=1 we wrap. Single-vec passes through as well via the parser.
        if isinstance(chunk, list) and chunk and isinstance(chunk[0], (int, float)):
            chunk = [chunk]
        json_str = json.dumps(chunk)
        self._self_log("chunk_len", len(chunk) if isinstance(chunk, list) else "?")
        return {"action_chunk": json_str}


# ══════════════════════════════════════════════════════════════════════
# PolicyVlaNodeSet — the nodeset binding
# ══════════════════════════════════════════════════════════════════════


class PolicyVlaNodeSet(BaseNodeSet):
    """VLA policies (Pi0 / SmolVLA / Diffusion Policy / DROID DP) as a NodeSet.

    Loads in server mode against the ``ac-vla-policy`` conda env by
    default. ``server_python`` reads from ``$VLA_POLICY_PYTHON``.
    """

    name = "policy_vla"
    description = (
        "VLA policies — Pi0 / SmolVLA / Diffusion Policy / DROID DP, "
        "with adapter system exposed as 5 explicit canvas nodes (one per "
        "atomic adapter stage + 1 for the model forward): env→canonical, "
        "canonical→model, predict, model→canonical, canonical→env."
    )
    server_python = os.environ.get(
        "VLA_POLICY_PYTHON",
        os.path.expanduser("~/miniforge3/envs/ac-vla-policy/bin/python"),
    )
    # Subprocess env extras (TF 2.19 + tf-agents 0.19 path):
    #   - TF_USE_LEGACY_KERAS=1 — tf-agents 0.19.0 uses Keras 2 internal API
    #     (`keras._tf_keras.keras.__internal__`) which Keras 3 (TF 2.16+ default)
    #     dropped. tf-keras package + this flag re-routes tf.keras to Keras 2.
    #   - TF_FORCE_GPU_ALLOW_GROWTH=true — TF default reserves the entire GPU
    #     at process start (~22 GB on a 24 GB card just for RT-1-X). Growth mode
    #     allocates lazily, dropping real footprint to ~3-5 GB so worker_count>1
    #     and coexisting with other GPU jobs become possible.
    #   - LD_LIBRARY_PATH — point to the conda env's lib/ so the conda's
    #     libstdc++.so.6 (CXXABI_1.3.15, from libstdcxx-ng>=12) is preferred
    #     over Ubuntu 20.04's system libstdc++ 6.0.28 which lacks it. TF /
    #     tf-agents extensions fail to load without this on stock 20.04.
    _vla_env_lib = str(Path(server_python).parent.parent / "lib")
    server_env = {
        "TF_USE_LEGACY_KERAS": "1",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "LD_LIBRARY_PATH": f"{_vla_env_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    # No env panel — config is distributed across the 3 config-owning adapter
    # nodes (env_to_canonical / canonical_to_model / predict; the two action-
    # side nodes have no config). Env panels in AgentCanvas are for env-side
    # runtime knobs (suite, episode_index); pure-policy nodesets are
    # parameterized entirely through their canvas nodes.
    # Singleton subprocess shared across K eval workers (ADR-eval-002 PC-3,
    # mirrors policy_cma). RT-1 recurrent state travels on the wire via
    # policy_state_in/out + IterIn/IterOut; Pi0/SmolVLA/DP/Droid-DP are
    # stateless. PredictTool.batched=True triggers BatchedInferenceServer
    # rendezvous, but the K-flush is unrolled into a sequential loop because
    # the TF SavedModel is pinned to batch_size=1 — primary win is GPU
    # memory (one model + one USE encoder shared) not throughput.
    parallelism = "shared"
    # Pi0 first call is slow (JAX→PT conversion + torch.compile) — give 60s budget.
    default_per_step_budget_sec = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._mgr = VlaPolicyManager.get()

    def get_tools(self) -> list:
        return [
            AdaptEnvToCanonicalTool(),  # stage 1
            AdaptCanonicalToModelTool(),  # stage 2
            PredictTool(),  # model forward
            AdaptModelToCanonicalTool(),  # stage 3
            AdaptCanonicalToEnvTool(),  # stage 4
        ]

    async def initialize(self, **kwargs: Any) -> None:
        if self._mgr.initialized:
            log.info("policy_vla already initialized — skipping")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mgr.executor,
            lambda: self._mgr.initialize(**kwargs),
        )
        log.info("PolicyVlaNodeSet initialized")

    async def shutdown(self) -> None:
        self._mgr.shutdown()
