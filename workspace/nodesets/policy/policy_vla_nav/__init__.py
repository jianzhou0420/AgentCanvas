from __future__ import annotations

"""PolicyVlaNavNodeSet — the trained JanusVLN Qwen3-VL-Omega VLA as two canvas nodes.

A small trained navigation policy exposed on the canvas as a plain
observation → action service, so a graph (or an outer agent driving it as a
tool) can step it exactly like any other policy:

    policy_vla_nav__reset   instruction        -> episode_ok       (clears history, loads weights)
    policy_vla_nav__act     front/left/right   -> action, action_id (one primitive per fire)

Model: ``Qwen3VLOmegaFramewiseDeepStackGatedCurrentForConditionalGeneration``
(Qwen3-VL-2B backbone + VGGT-Omega geometry + DeepStack-gated visual-token
injection), checkpoint ``39331``, run in upstream's ``--vlm_only --controller
discrete`` mode. Upstream reports R2R val_unseen (1,839 eps) SR 55.19 / SPL
50.54 / OSR 64.06 / NE 5.08.

──────────────────────────────────────────────────────────────────────
What the model actually consumes (from upstream's eval loop; do not guess)
──────────────────────────────────────────────────────────────────────
Per step it wants THREE current RGB views plus a rolling history of past
FRONT views, and it emits one of four primitives as text:

    current : left, right, front            (three separate sensors, same pose)
    history : up to num_history=20 past FRONT views, uniformly subsampled
              across the WHOLE episode (not a last-K window)
    text    : the instruction
    output  : MOVE_FORWARD | TURN_LEFT | TURN_RIGHT | STOP

History is appended AFTER each prediction, so step 0 sees an empty history.
VGGT-Omega runs on the current three views ONLY — that is what "current-3-view"
names; history frames go through the Qwen visual encoder at 0.25x resolution.

⚠ Sensor spec is load-bearing. Upstream's evaluator overrides the habitat yaml
and builds its three sensors as:

    width 720, height 640, hfov 110, position [0, 0.5, 0]
    front  orientation [0, 0,      0]
    left   orientation [0, +pi/2,  0]
    right  orientation [0, -pi/2,  0]

Feeding views rendered at a different fov / camera height is a silent
distribution shift — the policy will still answer, just worse. Configure
env_habitat's sensors to match, or accept the shift knowingly.

Action ids: this nodeset returns BOTH the upstream text action and AgentCanvas'
discrete id (0=STOP 1=FORWARD 2=LEFT 3=RIGHT), and the two action spaces already
agree — forward 0.25 m, turn 15 deg, success 3 m — so ``action_id`` feeds
``env_habitat__step_discrete`` with no conversion.

──────────────────────────────────────────────────────────────────────
Deployment
──────────────────────────────────────────────────────────────────────
Server mode on a dedicated env (the ``model_vggt_slam2`` pattern): torch
2.8.0+cu128 (Blackwell sm_120), transformers pinned to upstream's commit.
``parallelism = "replicated"`` because the front-view history is mutable
per-episode state — each eval worker gets its own process.

The env deliberately does NOT contain habitat. Upstream's ``environment.yml``
pins ``habitat-sim=0.3.3`` next to ``python=3.10``, which cannot solve
(aihabitat publishes only py3.9 builds for 0.3.3) — and it does not need to:
nothing under ``src/qwen_vl/`` or ``third_party/vggt_omega/`` imports habitat.
Only upstream's ``evaluation*.py`` glue does, which is why ``_backend.py``
re-implements that glue instead of importing it. AgentCanvas owns the
simulator via ``env_habitat``; the policy has no business shipping a second one.

Environment:
    ac-vlanav (Python 3.10)   — $VLA_NAV_PYTHON overrides the interpreter
    $VLA_NAV_REPO             — upstream checkout (default
                                ~/Desktop/Projects/DeepStack/qwen3vl-omega-current3-vlm)
    Weights: checkpoint-39331/model.safetensors (13 GB, git-lfs) carries ALL
             trained Qwen3-VL + VGGT-Omega parameters, so no separate
             VGGT-Omega checkpoint is required.
    Base arch is built from Qwen/Qwen3-VL-2B-Instruct (HF hub on first load).

License: the bundled VGGT-Omega is FAIR Noncommercial Research — the combined
release is noncommercial-research-only.

last updated: 2026-08-06 (initial)
"""

import asyncio
import functools
import logging
import os
from typing import Any, ClassVar

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.policy_vla_nav")

# The graph executor fires FRESH node_cls() instances, and AutoServerApp builds
# its manifest from a throwaway nodeset before on_startup constructs the real
# one — so nodes must reach the live session through this module global
# (_NODESET first, node backref second). Same hazard and same fix as
# model_vggt_slam2/__init__.py:80-92.
_NODESET: "PolicyVlaNavNodeSet | None" = None

_POLICY_COLOR = "blue"

_DEFAULT_REPO = os.path.expanduser(
    "~/Desktop/Projects/DeepStack/qwen3vl-omega-current3-vlm"
)


def _repo() -> str:
    return os.environ.get("VLA_NAV_REPO", _DEFAULT_REPO)


async def _run(ns: "PolicyVlaNavNodeSet", fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a session method on its pinned single-thread executor (CUDA affinity)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        ns._session.executor, functools.partial(fn, *args, **kwargs)
    )


# ══════════════════════════════════════════════════════════════════════
# Canvas nodes
# ══════════════════════════════════════════════════════════════════════


class VlaNavResetTool(BaseCanvasNode):
    """Start a new episode — clear the front-view history, load weights once."""

    node_type = "policy_vla_nav__reset"
    display_name = "VLA Nav: Reset"
    description = (
        "Begin an episode: clears the rolling front-view history and sets the "
        "default instruction. Owns the checkpoint / preprocessing config for the "
        "whole nodeset (chain entry) and triggers the one-time weight load, so "
        "policy_vla_nav__act stays config-less. Idempotent — re-firing with an "
        "unchanged config only resets episode state, it does not reload the model."
    )
    category = "policy"
    icon = "RotateCcw"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color=_POLICY_COLOR,
        config_fields=[
            ConfigField(
                "checkpoint_path",
                "text",
                label="Checkpoint dir (weights + tokenizer + processor)",
                default=os.path.join(_DEFAULT_REPO, "checkpoint-39331"),
            ),
            ConfigField(
                "base_model_path",
                "text",
                label="Base architecture (HF id or local dir)",
                default="Qwen/Qwen3-VL-2B-Instruct",
            ),
            ConfigField(
                "vggt_omega_model_path",
                "text",
                label="VGGT-Omega weights (leave EMPTY — the checkpoint carries them)",
                default="",
            ),
            ConfigField(
                "attn_implementation",
                "select",
                label="Attention kernel",
                default="flash_attention_2",
                options=[
                    {"value": "flash_attention_2", "label": "flash_attention_2 (upstream)"},
                    {"value": "sdpa", "label": "sdpa (fallback, no flash-attn build)"},
                    {"value": "eager", "label": "eager"},
                ],
            ),
            ConfigField("num_history", "text", label="Max history front views", default="20"),
            ConfigField(
                "omega_image_resolution", "text", label="Current-view resolution", default="512"
            ),
            ConfigField(
                "history_resize_ratio", "text", label="History resize ratio", default="0.25"
            ),
            ConfigField(
                "llm_max_new_tokens", "text", label="Max generated tokens", default="12"
            ),
            ConfigField("device", "text", label="Device", default="cuda"),
        ],
    )
    input_ports = [
        PortDef("instruction", "TEXT", "Episode instruction text", optional=True),
        PortDef("trigger", "ANY", "Fire to (re)start the episode", optional=True),
    ]
    output_ports = [
        PortDef("episode_ok", "BOOL", "True once the policy is loaded and history is empty"),
        PortDef("info", "TEXT", "Load summary, or the error that stopped it"),
    ]

    def __init__(self, nodeset: "PolicyVlaNavNodeSet | None" = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        if ns is None or ns._session is None:
            return {"episode_ok": False, "info": "policy_vla_nav not initialized"}

        cfg = self.config or {}

        def _num(key: str, default: float, cast=float):
            raw = str(cfg.get(key, "")).strip()
            try:
                return cast(raw) if raw else cast(default)
            except (TypeError, ValueError):
                return cast(default)

        session = ns._session
        session.num_history = _num("num_history", 20, int)
        session.omega_image_resolution = _num("omega_image_resolution", 512, int)
        session.history_resize_ratio = _num("history_resize_ratio", 0.25, float)
        session.llm_max_new_tokens = _num("llm_max_new_tokens", 12, int)

        checkpoint = str(cfg.get("checkpoint_path") or "").strip() or os.path.join(
            _repo(), "checkpoint-39331"
        )
        try:
            meta = await _run(
                ns,
                session.ensure_model,
                checkpoint_path=checkpoint,
                base_model_path=str(cfg.get("base_model_path") or "").strip()
                or "Qwen/Qwen3-VL-2B-Instruct",
                vggt_omega_model_path=str(cfg.get("vggt_omega_model_path") or "").strip()
                or None,
                device=str(cfg.get("device") or "").strip() or "cuda",
                attn_implementation=str(cfg.get("attn_implementation") or "").strip()
                or "flash_attention_2",
            )
        except Exception as exc:
            log.exception("policy_vla_nav reset: model load failed")
            self._self_log("error", f"ensure_model: {exc!r}")
            return {"episode_ok": False, "info": f"ensure_model: {exc!r}"}

        instruction = inputs.get("instruction") or ""
        await _run(ns, session.reset, str(instruction))
        if not meta.get("unchanged"):
            self._self_log("loaded", meta)
        return {
            "episode_ok": True,
            "info": (
                f"ready · ckpt={os.path.basename(checkpoint)} "
                f"history<={session.num_history} res={session.omega_image_resolution} "
                f"instr_len={len(str(instruction))}"
            ),
        }


class VlaNavActTool(BaseCanvasNode):
    """One policy step: three current views + rolling history -> one primitive."""

    node_type = "policy_vla_nav__act"
    display_name = "VLA Nav: Act"
    description = (
        "Feed the current left / right / front RGB views and get one discrete "
        "action. The policy keeps its own front-view history internally (appended "
        "AFTER each prediction, uniformly subsampled to num_history), so wire only "
        "the current frames — no history port. action_id is already in "
        "env_habitat__step_discrete's encoding (0=STOP 1=FORWARD 2=LEFT 3=RIGHT). "
        "The optional instruction port overrides the episode instruction for THIS "
        "step only and leaves the visual history untouched — that is the seam a "
        "sub-instruction outer agent drives."
    )
    category = "policy"
    icon = "Navigation"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color=_POLICY_COLOR)
    input_ports = [
        PortDef("rgb_front", "IMAGE", "Front view — orientation [0,0,0]"),
        PortDef("rgb_left", "IMAGE", "Left view — orientation [0,+pi/2,0]"),
        PortDef("rgb_right", "IMAGE", "Right view — orientation [0,-pi/2,0]"),
        PortDef(
            "instruction",
            "TEXT",
            "Overrides the episode instruction for this step only (sub-instruction seam)",
            optional=True,
        ),
    ]
    output_ports = [
        PortDef("action", "TEXT", "MOVE_FORWARD | TURN_LEFT | TURN_RIGHT | STOP"),
        PortDef("action_id", "ANY", "env_habitat discrete id — 0=STOP 1=FWD 2=LEFT 3=RIGHT"),
        PortDef("stop", "BOOL", "True when the policy asked to end the episode"),
        PortDef("raw_text", "TEXT", "Raw generation before normalization (OOD tell-tale)"),
        PortDef("recognized", "BOOL", "False when the generation matched no known action"),
        PortDef("step_id", "ANY", "Policy steps taken this episode"),
        PortDef("history_len", "ANY", "Front views held in history"),
    ]

    def __init__(self, nodeset: "PolicyVlaNavNodeSet | None" = None) -> None:
        super().__init__()
        self._nodeset = nodeset

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        ns = _NODESET or self._nodeset
        fallback = {
            "action": "STOP",
            "action_id": 0,
            "stop": True,
            "raw_text": "",
            "recognized": False,
            "step_id": 0,
            "history_len": 0,
        }
        if ns is None or ns._session is None:
            self._self_log("error", "policy_vla_nav not initialized")
            return fallback

        missing = [p for p in ("rgb_front", "rgb_left", "rgb_right") if inputs.get(p) is None]
        if missing:
            msg = f"missing required view(s): {', '.join(missing)}"
            self._self_log("error", msg)
            return fallback

        try:
            result = await _run(
                ns,
                ns._session.act,
                inputs["rgb_front"],
                inputs["rgb_left"],
                inputs["rgb_right"],
                inputs.get("instruction"),
            )
        except Exception as exc:
            log.exception("policy_vla_nav act failed")
            self._self_log("error", f"act: {exc!r}")
            return fallback

        if not result.get("recognized"):
            # Worth surfacing: an unparseable generation is the first symptom of
            # a prompt/preprocessing drift or an out-of-distribution instruction.
            self._self_log(
                "warning", f"unrecognized action text: {result.get('raw_text')!r} -> STOP"
            )
        return {
            "action": result["action"],
            "action_id": result["action_id"],
            "stop": result["stop"],
            "raw_text": result["raw_text"],
            "recognized": result["recognized"],
            "step_id": result["step_id"],
            "history_len": result["history_len"],
        }


# ══════════════════════════════════════════════════════════════════════
# PolicyVlaNavNodeSet
# ══════════════════════════════════════════════════════════════════════


class PolicyVlaNavNodeSet(BaseNodeSet):
    """Trained VLN VLA (Qwen3-VL-2B + VGGT-Omega, current-3-view) — server mode."""

    name = "policy_vla_nav"
    description = (
        "The trained JanusVLN Qwen3-VL-Omega current-3-view navigation policy as "
        "a canvas service: reset (episode start, owns the config) and act "
        "(left/right/front RGB -> one of MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / "
        "STOP). Discrete text controller, VLM-only — no ActionFormer head. Pair "
        "with env_habitat; action_id needs no conversion."
    )
    # Rolling per-episode front-view history is mutable state — one server
    # process per eval worker (model_vggt_slam2 / model_pyslam pattern).
    parallelism = "replicated"
    # Dedicated env: torch 2.8.0+cu128 for Blackwell sm_120, transformers pinned
    # to upstream's commit. Deliberately habitat-free (see module docstring).
    server_python = conda_env_python("ac-vlanav", "VLA_NAV_PYTHON")
    # Qwen3-VL-2B bf16 (~5 GB) + VGGT-Omega + up to 23 images of activations.
    expected_vram_mb = 14000
    # One act = VGGT-Omega on 3 views + a <=12-token greedy decode.
    default_per_step_budget_sec = 20.0

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None

    def get_tools(self) -> list:
        return [VlaNavResetTool(self), VlaNavActTool(self)]

    async def initialize(self, **kwargs: Any) -> None:
        global _NODESET
        # Heavy import (torch / transformers / vggt_omega) — resolvable only
        # inside the ac-vlanav server process; server_python auto-routes loads
        # to server mode so the framework env never executes this.
        from . import _backend

        self._session = _backend.VlaNavSession()
        _NODESET = self
        log.info(
            "policy_vla_nav ready (server_python=%s, repo=%s); weights load on first reset",
            self.server_python,
            _repo(),
        )

    async def shutdown(self) -> None:
        global _NODESET
        if self._session is not None:
            session = self._session
            self._session = None
            await asyncio.to_thread(session.close)
        _NODESET = None
