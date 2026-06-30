"""Prismatic VLM as a generic foundation-model nodeset.

Exposes domain-agnostic primitives that any method nodeset can wire in:

  vlm_prismatic__score_tokens   — (image, prompt, tokens) → softmax probs
  vlm_prismatic__generate       — (image, prompt) → generated text

Server mode under the dedicated ``hmeqa`` env (Python 3.9 + Prismatic).
Method nodesets (e.g. ``explore_eqa``, future EQA / VLN methods) consume
these primitives via canvas wires — Prismatic stays out of method code,
the foundation-model boundary is a one-class swap.

Singleton: weights load once per subprocess on first
``initialize()`` and live until subprocess teardown.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components.bases import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
)

log = logging.getLogger("agentcanvas.vlm_prismatic")

_DEFAULT_MODEL_ID = "prism-dinosiglip+7b"

# Subprocess-local singleton. Pattern mirrors policy_cma.py.
_VLM_BUNDLE: dict | None = None
_VLM_LOAD_LOCK = threading.Lock()


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


def _ensure_loaded(model_id: str | None = None) -> dict | None:
    """Load Prismatic on first call; cache the bundle thereafter.

    Blocking — call from a thread (`asyncio.to_thread`). Lock serialises
    first-touch loaders so weights are read once even under racing
    callers.
    """
    global _VLM_BUNDLE
    if _VLM_BUNDLE is not None:
        return _VLM_BUNDLE
    with _VLM_LOAD_LOCK:
        if _VLM_BUNDLE is not None:
            return _VLM_BUNDLE
        try:
            import torch
            from prismatic import load
        except Exception:
            log.exception("Prismatic import failed — is the hmeqa env active?")
            return None

        mid = model_id or os.environ.get("VLM_PRISMATIC_MODEL_ID", _DEFAULT_MODEL_ID)
        hf_token = os.environ.get("HF_TOKEN", "")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Loading Prismatic VLM model_id=%s on %s", mid, device)
        try:
            model = load(mid, hf_token=hf_token) if hf_token else load(mid)
            model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        except Exception:
            log.exception("Prismatic load failed")
            return None

        _VLM_BUNDLE = {"model": model, "device": device, "model_id": mid}
        log.info("Prismatic VLM ready (model_id=%s)", mid)
        return _VLM_BUNDLE


def _coerce_token_list(tokens) -> list[str]:
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
      - missing image / VLM unavailable → uniform distribution
        (degraded mode, lets the graph keep running for structural
        testing without the full env)
    """

    node_type: ClassVar[str] = "vlm_prismatic__score_tokens"
    display_name: ClassVar[str] = "VLM Prismatic: Score Tokens"
    description: ClassVar[str] = (
        "Token-likelihood scoring — (image, prompt, tokens) → softmax probs"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "Sparkles"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("temperature", "number", default=1.0),
    ]

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
            self._self_log("error", "no image — uniform")
            return {"probs": [1.0 / n] * n}

        bundle = _ensure_loaded()
        if bundle is None:
            self._self_log("error", "VLM unavailable — uniform")
            return {"probs": [1.0 / n] * n}
        model = bundle["model"]

        def _score():
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
            return {"probs": [1.0 / n] * n}

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
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")

    config_schema: ClassVar[list[ConfigField]] = [
        ConfigField("max_new_tokens", "integer", default=128),
        ConfigField("temperature", "number", default=0.2),
    ]

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

        bundle = _ensure_loaded()
        if bundle is None:
            self._self_log("error", "VLM unavailable")
            return {"text": ""}
        model = bundle["model"]

        def _gen():
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

        text = str(text)
        self._self_log("text_len", len(text))
        return {"text": text}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VLMPrismaticNodeSet(BaseNodeSet):
    """Generic Prismatic VLM foundation-model nodeset.

    Loads Prismatic in a dedicated subprocess (under the ``hmeqa`` env)
    and exposes ``score_tokens`` + ``generate`` as canvas-wirable
    primitives. Method nodesets consume these via canvas wires — there
    is no Python-level coupling to any specific method.

    Override ``$VLM_PRISMATIC_MODEL_ID`` (or the per-instance config
    if/when added) to swap the underlying weights.
    """

    name: ClassVar[str] = "vlm_prismatic"
    description: ClassVar[str] = (
        "Prismatic VLM — generic primitives (score_tokens, generate) wired by method nodesets"
    )
    server_python: ClassVar[str] = os.environ.get(
        "HMEQA_PYTHON", os.path.expanduser("~/miniforge3/envs/ac-hmeqa/bin/python")
    )

    def get_tools(self) -> list:
        return [ScoreTokensNode(), GenerateNode()]

    async def initialize(self, **kwargs: Any) -> None:
        import asyncio

        await asyncio.to_thread(_ensure_loaded)

    async def shutdown(self) -> None:
        # Retain bundle across reloads; freed only on subprocess teardown.
        pass
