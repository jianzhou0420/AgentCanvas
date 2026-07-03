from __future__ import annotations

"""EnvAdapterNodeSet — env-side adapter stages of the General Policy Adapter pipeline.

One general nodeset owning pipeline stages 1 (env → canonical) and 5
(canonical → env action) for BOTH policy domains:

  robots/   VLA env-side adapters (LIBERO / SIMPLER RobotAdaptor subclasses,
            moved from policy_vla/adapters/robots/)
  envs/     VLN-CE env-side adapters (R2R-CE / RxR-CE VlnEnvAdaptor
            subclasses, moved from policy_vlnce/adapters/envs/)

The model-side stages (2/3/4: canonical → model, predict, model → canonical)
live in the per-domain policy nodesets ``policy_adapter_vla`` and
``policy_adapter_vlnce`` — a nodeset binds exactly one ``server_python``,
and the two model families need different conda envs. Running a graph
composes three nodesets: the env itself + this env adapter + one policy
adapter (e.g. env_habitat + env_adapter + policy_adapter_vlnce).

Everything here is pure numpy (no torch, no habitat, no model imports), so
the nodeset runs IN-PROCESS in the hub env (``server_python = None``). The
canonical dicts it emits/consumes cross the msgpack wire to the policy-side
subprocess; ``info`` is therefore always carried as a plain dict
(``dataclasses.asdict``), never a raw dataclass.

Filename-based discovery: every non-``base_``/non-underscore .py under
``robots/`` and ``envs/`` becomes a canvas dropdown option (``canonical``
schema modules are excluded). Drop a file in / out → POST
/api/components/reload.

Node inventory:
  env_adapter__vla_env_to_canonical   — VLA stage 1; owns the ``robot`` select
  env_adapter__vla_canonical_to_env   — VLA stage 5; owns the SAME ``robot``
                                        select (class identity matters:
                                        SimplerRobot vs LiberoRobot decode
                                        differently, and delta-action config
                                        is robot-side, not on the wire)
  env_adapter__vln_env_to_canonical   — VLN stage 1; owns the ``env_family``
                                        select (r2rce / rxrce)
  env_adapter__vln_canonical_to_env   — VLN stage 5; config-less (reads
                                        action_index + action_dim off the
                                        canonical_action itself)
"""

import dataclasses
import importlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.env_adapter")


# ══════════════════════════════════════════════════════════════════════
# Discovery — folder-scan + lazy import (same contract as the policy sides)
# ══════════════════════════════════════════════════════════════════════

_PKG_ROOT = Path(__file__).parent
# ``canonical`` is the schema module living beside the adapters — it defines
# the wire contract, it is not a selectable adapter.
_DISCOVERY_EXCLUDES: tuple[str, ...] = ("__init__", "canonical")


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
        f"workspace.nodesets.env.env_adapter.{subpkg_dotted}.{module_name}"
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


def _instantiate(spec: Any) -> Any:
    """Hydra-style ``_target_:`` resolution in nested DEFAULT_KWARGS values."""
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


# Canvas dropdown options — frozen at module-import time.
ROBOT_OPTIONS: list[str] = _discover_modules("robots")
ENV_OPTIONS: list[str] = _discover_modules("envs")


# ══════════════════════════════════════════════════════════════════════
# Lazy adapter caches — the in-process replacement for the old managers'
# ensure_robot / ensure_env_adapter slices. Adapters are cheap pure-numpy
# objects, immutable after construction; the lock guards cache population
# when K LoopRunners share this hub process (parallelism is a server-mode
# concept — in-process nodes all run here).
# ══════════════════════════════════════════════════════════════════════

_LOCK = threading.Lock()
_ROBOT_CACHE: dict[str, tuple[str, Any]] = {}  # robot_module → (cfg_key, RobotAdaptor)
_ENV_CACHE: dict[str, tuple[str, Any]] = {}  # env_family → (cfg_key, VlnEnvAdaptor)


def _ensure_robot(robot_module: str) -> Any:
    """(Re)build a VLA robot adapter from ``robots/<robot_module>.py``.

    Mirrors the old VlaPolicyManager.ensure_robot: DEFAULT_KWARGS from the
    module, ``default_prompt`` defaulted, rebuild only when config changes.
    Raises on failure — callers surface the message via _self_log.
    """
    module = _load_module("robots", robot_module)
    from workspace.nodesets.env.env_adapter.robots.base_robot import RobotAdaptor

    cls = _find_subclass(module, RobotAdaptor)
    kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
    kwargs.setdefault("default_prompt", None)
    cfg_key = json.dumps(kwargs, sort_keys=True, default=str)
    with _LOCK:
        hit = _ROBOT_CACHE.get(robot_module)
        if hit is not None and hit[0] == cfg_key:
            return hit[1]
        robot = cls(**_instantiate(kwargs))
        _ROBOT_CACHE[robot_module] = (cfg_key, robot)
        log.info("env_adapter: built robot adapter module=%s kwargs=%s", robot_module, kwargs)
        return robot


def _ensure_env(env_family: str) -> Any:
    """(Re)build a VLN env adapter from ``envs/<env_family>.py``.

    Mirrors the old VlnceManager.ensure_env_adapter.
    """
    module = _load_module("envs", env_family)
    from workspace.nodesets.env.env_adapter.envs.base_env import VlnEnvAdaptor

    cls = _find_subclass(module, VlnEnvAdaptor)
    kwargs = dict(getattr(module, "DEFAULT_KWARGS", {}))
    cfg_key = json.dumps(kwargs, sort_keys=True, default=str)
    with _LOCK:
        hit = _ENV_CACHE.get(env_family)
        if hit is not None and hit[0] == cfg_key:
            return hit[1]
        adapter = cls(**kwargs)
        _ENV_CACHE[env_family] = (cfg_key, adapter)
        log.info("env_adapter: built env adapter family=%s", env_family)
        return adapter


def _to_list(arr: Any) -> Any:
    """Convert ndarray → nested list (JSON-serializable). Pass through other types."""
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32).tolist()
    return arr


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes — VLA pair
# ══════════════════════════════════════════════════════════════════════


_VLA_COLOR = "violet"
_VLN_COLOR = "blue"


class VlaEnvToCanonicalTool(BaseCanvasNode):
    node_type = "env_adapter__vla_env_to_canonical"
    display_name = "VLA: Env → Canonical"
    description = (
        "RobotAdaptor.env_to_canonical — wraps env obs (image, wrist_image, "
        "state, prompt) into the canonical intermediate format. The dropdown "
        "lists every .py file under env_adapter/robots/ — drop a file in or "
        "out of that folder to add or remove a variant; each file owns its "
        "own DEFAULT_KWARGS (e.g. delta vs absolute action mode). The robot "
        "selection is env-side: it must match the graph's env nodeset "
        "(libero_robot ↔ env_libero, simpler_robot ↔ env_simpler) and the "
        "selection on env_adapter__vla_canonical_to_env."
    )
    category = "policy"
    icon = "ArrowRight"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_VLA_COLOR,
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
        try:
            robot = _ensure_robot(robot_module)
        except Exception as e:
            self._self_log("error", f"ensure_robot({robot_module!r}): {e!r}")
            return {"canonical": None}

        env_obs = {
            "image": inputs.get("image"),
            "wrist_image": inputs.get("wrist_image"),
            "state": inputs.get("state"),
            "prompt": inputs.get("prompt") or "",
        }
        env_obs = {k: v for k, v in env_obs.items() if v is not None or k == "prompt"}
        try:
            canonical = robot.env_to_canonical(env_obs)
        except Exception as e:
            log.exception("env_to_canonical failed")
            self._self_log("error", f"env_to_canonical: {e!r}")
            return {"canonical": None}
        # The canonical now crosses the msgpack wire to the policy-side
        # subprocess — a raw CanonicalInfo dataclass is not packable, so
        # flatten it here; policy_adapter_vla's stage-2 node reconstructs it.
        if dataclasses.is_dataclass(canonical.get("info")):
            canonical["info"] = dataclasses.asdict(canonical["info"])
        return {"canonical": canonical}


class VlaCanonicalToEnvTool(BaseCanvasNode):
    node_type = "env_adapter__vla_canonical_to_env"
    display_name = "VLA: Canonical → Env Action"
    description = (
        "RobotAdaptor.canonical_to_env — converts a CanonicalDict[action] to a "
        "JSON action chunk consumable by env_*__step.action. Needs current_state "
        "for delta-action robots (LIBERO Pi0/DP/DROID-DP) to anchor delta on "
        "the current eef pose. The ``robot`` select MUST match the one on "
        "env_adapter__vla_env_to_canonical — decode logic and delta-action "
        "config are robot-class-side, not carried on the wire."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_VLA_COLOR,
        config_fields=[
            ConfigField(
                "robot",
                "select",
                label="Robot adapter (same as stage 1)",
                default=ROBOT_OPTIONS[0] if ROBOT_OPTIONS else "",
                options=[{"value": s, "label": s} for s in ROBOT_OPTIONS],
            ),
        ],
    )
    input_ports = [
        PortDef("canonical_action", "ANY", "Output of policy_adapter_vla__adapt_model_to_canonical"),
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
        fallback = {"action_chunk": json.dumps([[0.0] * 6 + [-1.0]])}
        canonical_action = inputs.get("canonical_action")
        if canonical_action is None:
            self._self_log("error", "canonical_action input missing")
            return fallback
        robot_module = (self.config.get("robot") or "").strip()
        if not robot_module:
            self._self_log("error", "no robot adapter selected")
            return fallback
        try:
            robot = _ensure_robot(robot_module)
        except Exception as e:
            self._self_log("error", f"ensure_robot({robot_module!r}): {e!r}")
            return fallback

        current_state = inputs.get("current_state")
        state: dict[str, Any] | None = None
        if current_state is not None:
            state = {"state": np.asarray(current_state, dtype=np.float32)}
        try:
            env_action = robot.canonical_to_env(canonical_action, state=state)
        except Exception as e:
            log.exception("canonical_action_to_env failed")
            self._self_log("error", f"canonical_action_to_env: {e!r}")
            return fallback

        actions = env_action.get("actions")
        if actions is None:
            self._self_log("error", "no 'actions' key in env_action")
            return fallback
        chunk = _to_list(actions)
        # Emit ONLY a list-of-lists so env_libero__step_continuous's parser sees a (K, 7) chunk
        # — even for K=1 we wrap. Single-vec passes through as well via the parser.
        if isinstance(chunk, list) and chunk and isinstance(chunk[0], (int, float)):
            chunk = [chunk]
        json_str = json.dumps(chunk)
        self._self_log("chunk_len", len(chunk) if isinstance(chunk, list) else "?")
        return {"action_chunk": json_str}


# ══════════════════════════════════════════════════════════════════════
# Canvas tool nodes — VLN pair
# ══════════════════════════════════════════════════════════════════════


class VlnEnvToCanonicalTool(BaseCanvasNode):
    node_type = "env_adapter__vln_env_to_canonical"
    display_name = "VLN-CE: Env → Canonical"
    description = (
        "VlnEnvAdaptor.env_to_canonical — wraps a Habitat raw_obs dict into "
        "the canonical {data, info} intermediate format. The env_family "
        "select picks the adapter under env_adapter/envs/ (r2rce / rxrce) "
        "and is an env-side choice paired with the graph's env nodeset; the "
        "policy-side variant (model + exp_config + policy + checkpoint) is "
        "selected on policy_adapter_vlnce__adapt_canonical_to_model."
    )
    category = "policy"
    icon = "ArrowRight"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_VLN_COLOR,
        config_fields=[
            ConfigField(
                "env_family",
                "select",
                label="Env family",
                default="r2rce" if "r2rce" in ENV_OPTIONS else (ENV_OPTIONS[0] if ENV_OPTIONS else ""),
                options=[{"value": s, "label": s} for s in ENV_OPTIONS],
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
        env_family = (self.config.get("env_family") or "").strip() or "r2rce"
        try:
            adapter = _ensure_env(env_family)
        except Exception as e:
            self._self_log("error", f"ensure_env({env_family!r}): {e!r}")
            return {"canonical": None}

        raw_obs = inputs.get("raw_obs")
        if not isinstance(raw_obs, dict):
            self._self_log("error", "raw_obs not a dict")
            return {"canonical": None}

        instruction = inputs.get("instruction")
        if instruction is None:
            self._self_log("warn", "no instruction wired — policy will see None tokens")
        try:
            canonical = adapter.env_to_canonical(raw_obs, instruction)
        except Exception as e:
            log.exception("env_to_canonical failed")
            self._self_log("error", f"env_to_canonical: {e!r}")
            return {"canonical": None}
        # make_canonical_obs already stores info as a plain dict (asdict) —
        # msgpack-safe across the wire to the policy-side subprocess.
        return {"canonical": canonical}


class VlnCanonicalToEnvTool(BaseCanvasNode):
    node_type = "env_adapter__vln_canonical_to_env"
    display_name = "VLN-CE: Canonical → Env Action"
    description = (
        "Picks an env action index from the canonical_action. R2R-CE: int ∈ "
        "[0,3]; RxR-CE: int ∈ [0,5]. Config-less: action_index and action_dim "
        "both ride on the canonical_action itself, so the clip is family-"
        "agnostic (identical to every VlnEnvAdaptor.canonical_to_env)."
    )
    category = "policy"
    icon = "ArrowLeft"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_VLN_COLOR)
    input_ports = [
        PortDef("canonical_action", "ANY", "Output of policy_adapter_vlnce__adapt_model_to_canonical"),
    ]
    output_ports = [
        PortDef("action", "ACTION", "Discrete action index for env_*__step"),
    ]

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        canonical_action = inputs.get("canonical_action")
        if canonical_action is None:
            self._self_log("error", "canonical_action input missing")
            return {"action": 0}
        try:
            action_index = int(canonical_action["data"]["action_index"])
            action_dim = int(canonical_action["info"]["action_dim"])
            action = max(0, min(action_index, action_dim - 1))
        except Exception as e:
            self._self_log("error", f"canonical_action_to_env: {e!r}")
            return {"action": 0}
        try:
            from app.standard.actions import ACTION_NAMES

            self._self_log("action_name", ACTION_NAMES.get(action, "UNKNOWN"))
        except Exception:
            pass
        self._self_log("predicted_action", action)
        return {"action": action}


# ══════════════════════════════════════════════════════════════════════
# EnvAdapterNodeSet
# ══════════════════════════════════════════════════════════════════════


class EnvAdapterNodeSet(BaseNodeSet):
    """General env-side adapter — stages 1 & 5 for both policy domains."""

    name = "env_adapter"
    description = (
        "Env-side adapter stages of the General Policy Adapter pipeline — "
        "env→canonical and canonical→env for both VLA robots (LIBERO / "
        "SIMPLER) and VLN-CE env families (R2R-CE / RxR-CE). Pure numpy, "
        "runs in-process in the hub env; pairs with policy_adapter_vla / "
        "policy_adapter_vlnce for the model-side stages."
    )
    # server_python stays None: everything here is numpy-only, so the
    # nodeset loads in local mode inside the hub interpreter.

    def get_tools(self) -> list:
        return [
            VlaEnvToCanonicalTool(),  # VLA stage 1
            VlaCanonicalToEnvTool(),  # VLA stage 5
            VlnEnvToCanonicalTool(),  # VLN stage 1
            VlnCanonicalToEnvTool(),  # VLN stage 5
        ]
