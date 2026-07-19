"""Prismatic VLM — generic foundation-model nodeset (score_tokens + generate).

Two capability primitives over one resident Prismatic VLM (default
``prism-dinosiglip+7b``, the Explore-EQA checkpoint):

    vlm_prismatic__score_tokens  (image, prompt, tokens) → probs
    vlm_prismatic__generate      (image, prompt) → text

``score_tokens`` is the building block for multi-choice answer scoring,
frontier visual-prompt evaluation (Explore-EQA's LSV/GSV), and calibrated
confidence estimation: Prismatic's ``get_loss(..., return_string_probabilities
=tokens)`` gives per-token likelihoods which we softmax with optional
temperature. NOTE: the checkpoint's ``string2idx`` registry covers a fixed
token set (``A``–``D``, ``Yes``/``No``); unregistered tokens raise inside
prismatic and the node reports an error.

FM-template alignment (2026-07-05): model identity is node config —
``model_id`` (default prism-dinosiglip+7b), engines
in a lazy registry keyed by the resolved id, load-failure latch, single-flight
GPU lock, config fields on the node UI. The former degraded path that
**fabricated a uniform distribution** is gone (mock output is not a
capability — house ruling): degraded/no-image/error now return empty ``probs``
with a ``degraded``/``error`` self-log, and the consumer decides.

Weights are HF-gated (TRI-ML) — set ``$HF_TOKEN`` for first download; the HF
cache works without it thereafter.

Runs **server mode** in the ``ac-hmeqa`` env (the Explore-EQA env whose
prismatic install matches the checkpoint). Override with $HMEQA_PYTHON.

Load: POST /api/components/nodesets/vlm_prismatic/load?mode=server

last updated: 2026-07-05
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.vlm_prismatic")

_DEFAULT_MODEL_ID = "prism-dinosiglip+7b"

# Curated prismatic-vlms registry ids (internal keys, not HF repos).
_MODEL_OPTIONS = [
    {"value": "prism-dinosiglip+7b", "label": "Prism DINOSigLIP 7B"},
    {"value": "prism-dinosiglip+13b", "label": "Prism DINOSigLIP 13B"},
    {"value": "prism-clip+7b", "label": "Prism CLIP 7B"},
    {"value": "prism-clip+13b", "label": "Prism CLIP 13B"},
    {"value": "prism-siglip+7b", "label": "Prism SigLIP 7B"},
    {"value": "prism-siglip+13b", "label": "Prism SigLIP 13B"},
    {"value": "reproduction-llava-v15+7b", "label": "LLaVa-v1.5 repro 7B"},
    {"value": "reproduction-llava-v15+13b", "label": "LLaVa-v1.5 repro 13B"},
]


def _to_pil_rgb(img):
    """Accept numpy/list/PIL → RGB PIL.Image."""
    from PIL import Image

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


class _PrismaticEngine:
    """Lazy registry: one resident Prismatic per ``model_id``; weights only."""

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()  # guards load AND single-flight inference

    @classmethod
    def get(cls, model_id: str = "") -> "_PrismaticEngine":
        resolved = model_id or _DEFAULT_MODEL_ID
        key = (resolved,)
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(resolved)
            return cls._instances[key]

    def ensure(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import torch
                from prismatic import load
            except Exception:
                log.exception("Prismatic import failed — is the hmeqa env active?")
                self._load_failed = True
                return False

            hf_token = os.environ.get("HF_TOKEN", "")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Prismatic VLM model_id=%s on %s", self.model_id, device)
            try:
                model = load(self.model_id, hf_token=hf_token) if hf_token else load(self.model_id)
                model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
            except Exception:
                log.exception("Prismatic load failed")
                self._load_failed = True
                return False

            self.model, self.device = model, device
            self._loaded = True
            log.info("Prismatic VLM ready (model_id=%s)", self.model_id)
            return True


def _coerce_token_list(tokens) -> list:
    if tokens is None:
        return []
    if isinstance(tokens, str):
        import json

        s = tokens.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except Exception:
            pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return [str(t) for t in tokens]


_MODEL_ID_FIELD = ConfigField(
    "model_id", "select", label="Prismatic model",
    options=list(_MODEL_OPTIONS), default=_DEFAULT_MODEL_ID,
)


# ══════════════════════════════════════════════════════════════════════
# Node 1: ScoreTokens — token-likelihood primitive
# ══════════════════════════════════════════════════════════════════════


class ScoreTokensNode(BaseCanvasNode):
    """Generic VLM token-likelihood scoring.

    Takes an image, a prompt, and a list of candidate tokens. Returns
    a softmax distribution aligned with the input token list. This is
    the building block for multi-choice answer scoring, frontier
    visual-prompt evaluation, calibrated confidence estimation, etc.

    Backend: Prismatic ``model.get_loss(image, prompt, tokens)`` —
    returns negative log-likelihoods which we softmax with optional
    temperature.

    Edge cases:
      - empty tokens list → empty probs array
      - missing image / VLM unavailable / scoring error → empty probs
        + ``degraded``/``error`` self-log (never a fabricated uniform)
    """

    node_type: ClassVar[str] = "vlm_prismatic__score_tokens"
    display_name: ClassVar[str] = "VLM Prismatic: Score Tokens"
    description: ClassVar[str] = (
        "Token-likelihood scoring — (image, prompt, tokens) → softmax probs"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "Sparkles"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            _MODEL_ID_FIELD,
            ConfigField("temperature", "slider", label="Softmax temperature", default=1.0, min=0.0, max=2.0, step=0.05),
        ],
    )

    input_ports: ClassVar[list] = [
        PortDef("image", "IMAGE", "Image to condition on"),
        PortDef("prompt", "TEXT", "Prompt"),
        PortDef("tokens", "ANY", "List of candidate tokens (e.g. ['A','B','C','D'])"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("probs", "ANY", "Softmax probabilities aligned with tokens"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        T = float(cfg.get("temperature", 1.0))
        image = inputs.get("image")
        prompt = inputs.get("prompt", "") or ""
        tokens = _coerce_token_list(inputs.get("tokens"))

        n = len(tokens)
        if n == 0:
            self._self_log("empty_tokens", True)
            return {"probs": []}

        pil = _to_pil_rgb(image)
        if pil is None:
            self._self_log("error", "no image")
            return {"probs": []}

        engine = _PrismaticEngine.get(str(cfg.get("model_id", "") or "").strip())

        def _score():
            if not engine.ensure():
                return None
            model = engine.model
            with engine._lock:
                pb = model.get_prompt_builder()
                pb.add_turn(role="human", message=prompt)
                prompt_text = pb.get_prompt()
                losses = model.get_loss(pil, prompt_text, return_string_probabilities=tokens)[0]
            losses = np.array(losses)
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))

        try:
            probs = await asyncio.to_thread(_score)
        except Exception as exc:
            log.exception("score_tokens failed")
            self._self_log("error", str(exc))
            return {"probs": []}

        if probs is None:
            self._self_log("degraded", "Prismatic engine failed to load")
            return {"probs": []}
        probs_list = [float(x) for x in np.asarray(probs).ravel().tolist()]
        self._self_log("probs", probs_list)
        return {"probs": probs_list}


# ══════════════════════════════════════════════════════════════════════
# Node 2: Generate — free-form generation
# ══════════════════════════════════════════════════════════════════════


class GenerateNode(BaseCanvasNode):
    """Free-form text generation: (image, prompt) → text."""

    node_type: ClassVar[str] = "vlm_prismatic__generate"
    display_name: ClassVar[str] = "VLM Prismatic: Generate"
    description: ClassVar[str] = "Free-form text generation given an image and prompt"
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            _MODEL_ID_FIELD,
            ConfigField(
                "max_new_tokens", "slider", label="Max new tokens",
                default=128, min=16, max=1024, step=16,
            ),
            ConfigField("temperature", "slider", label="Temperature (0 = greedy)", default=0.2, min=0.0, max=2.0, step=0.05),
        ],
    )

    input_ports: ClassVar[list] = [
        PortDef("image", "IMAGE", "Image to condition on"),
        PortDef("prompt", "TEXT", "Prompt"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Generated text"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        max_new = int(cfg.get("max_new_tokens", 128))
        T = float(cfg.get("temperature", 0.2))

        image = inputs.get("image")
        prompt = inputs.get("prompt", "") or ""

        pil = _to_pil_rgb(image)
        if pil is None:
            self._self_log("error", "no image")
            return {"text": ""}

        engine = _PrismaticEngine.get(str(cfg.get("model_id", "") or "").strip())

        def _gen():
            if not engine.ensure():
                return None
            model = engine.model
            with engine._lock:
                pb = model.get_prompt_builder()
                pb.add_turn(role="human", message=prompt)
                prompt_text = pb.get_prompt()
                return model.generate(
                    pil,
                    prompt_text,
                    do_sample=T > 0,
                    temperature=T if T > 0 else 1.0,
                    max_new_tokens=max_new,
                )

        try:
            text = await asyncio.to_thread(_gen)
        except Exception as exc:
            log.exception("generate failed")
            self._self_log("error", str(exc))
            return {"text": ""}

        if text is None:
            self._self_log("degraded", "Prismatic engine failed to load")
            return {"text": ""}
        text = str(text)
        self._self_log("text_len", len(text))
        return {"text": text}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VLMPrismaticNodeSet(BaseNodeSet):
    """Generic Prismatic VLM foundation-model nodeset.

    Loads Prismatic in a dedicated subprocess (under the ``ac-hmeqa`` env)
    and exposes ``score_tokens`` + ``generate`` as canvas-wirable
    primitives. Method nodesets consume these via canvas wires — there
    is no Python-level coupling to any specific method.
    """

    name: ClassVar[str] = "vlm_prismatic"
    description: ClassVar[str] = (
        "Prismatic VLM — generic primitives (score_tokens, generate) wired by method nodesets"
    )
    # Stateless VLM — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    server_python: ClassVar[str] = conda_env_python("ac-hmeqa", "HMEQA_PYTHON")

    def get_tools(self) -> list:
        return [ScoreTokensNode(), GenerateNode()]

    async def initialize(self, **kwargs: Any) -> None:
        # Eager warmup of the default engine — explore_eqa's first score call
        # lands within a per-step budget; loading lazily there would eat it.
        import asyncio

        await asyncio.to_thread(_PrismaticEngine.get("").ensure)

    async def shutdown(self) -> None:
        # Retain engines across reloads; freed only on subprocess teardown.
        pass
