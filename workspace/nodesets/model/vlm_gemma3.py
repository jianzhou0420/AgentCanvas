from __future__ import annotations

"""Gemma 3 multimodal as a generic foundation-model nodeset.

Google's open multimodal model — a compact, strong image+text VLM in the same
family as Gemma 3's text models. Same generate-primitive contract as the other
VLM nodesets, one node::

    vlm_gemma3__generate  — (messages | prompt, image_paths, stop_sequences) → text

Gemma 3 is an **image** VLM (no native video path), so this nodeset exposes
``image_paths`` only. Images are attached to the last user turn as chat content
blocks and loaded through the modern
``apply_chat_template(tokenize=True, return_dict=True)`` path.

Gated model: ``google/gemma-3-*`` requires a one-time acceptance of the Gemma
license on Hugging Face (and a logged-in ``HF_TOKEN``). Without access the
engine load latches ``degraded`` (empty text) — accept the licence at the model
page once, then it downloads normally.

FM-template alignment: model identity is node config — ``model_id`` (blank =
``$GEMMA3_MODEL_ID`` or the 4B-it default), engines in a lazy registry keyed by
the resolved id, load-failure latch, generation knobs on the node UI. The
single-flight generate lock is per-engine.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11 + torch
2.8.0+cu126 + transformers 5.13.0) via ``AutoModelForImageTextToText`` +
``AutoProcessor``. On CUDA the model loads in bfloat16 with flash-attn when the
wheel is present (sdpa fallback otherwise); CPU falls back to float32. Override
the env with $GEMMA3_PYTHON. This file must stay Python-3.8-parseable.

Model default: gemma-3-4b-it (smallest multimodal Gemma 3; the 1B is text-only).
Point ``model_id`` (or $GEMMA3_MODEL_ID) at 12B / 27B-it on a bigger GPU.

Load: POST /api/components/nodesets/vlm_gemma3/load?mode=server

last updated: 2026-07-08
"""

import logging
import os
import threading
from typing import Any, ClassVar

from app.components import (
    BaseCanvasNode,
    BaseNodeSet,
    ConfigField,
    NodeUIConfig,
    PortDef,
    conda_env_python,
)

log = logging.getLogger("agentcanvas.vlm_gemma3")

_MODEL_ID_DEFAULT = os.environ.get("GEMMA3_MODEL_ID", "google/gemma-3-4b-it")


class _Gemma3Engine:
    """Lazy registry: one loaded Gemma 3 per resolved ``model_id``."""

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str = "") -> "_Gemma3Engine":
        resolved = model_id or _MODEL_ID_DEFAULT
        with cls._registry_lock:
            if resolved not in cls._instances:
                cls._instances[resolved] = cls(resolved)
            return cls._instances[resolved]

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
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except Exception:
                log.exception("Gemma3 import failed — is the ac-fm env active?")
                self._load_failed = True
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Gemma3 model_id=%s on %s", self.model_id, device)

            if device == "cuda":
                model_kwargs: dict = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    model_kwargs["attn_implementation"] = "sdpa"
            else:
                model_kwargs = {"dtype": torch.float32, "device_map": "cpu"}

            try:
                model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
                processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception:
                log.exception(
                    "Gemma3 load failed (gated? accept the Gemma licence at "
                    "https://huggingface.co/%s and ensure HF_TOKEN is set)", self.model_id)
                self._load_failed = True
                return False

            self.model, self.processor, self.device = model, processor, device
            self._loaded = True
            log.info("Gemma3 ready (model_id=%s)", self.model_id)
            return True


def _coerce_messages(messages: Any, prompt: str) -> list:
    """Normalise into a chat message list. Accepts a pre-built list, a JSON
    string, or falls back to a single user turn built from ``prompt``."""
    if isinstance(messages, list) and messages:
        return [dict(m) for m in messages]
    if isinstance(messages, str) and messages.strip():
        import json

        try:
            parsed = json.loads(messages)
            if isinstance(parsed, list) and parsed:
                return [dict(m) for m in parsed]
        except Exception:
            pass
    return [{"role": "user", "content": prompt or ""}]


def _coerce_str_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        import json

        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [s]
    return [str(x) for x in val]


def _inject_images(messages: list, image_paths: list) -> list:
    """Replace the LAST user turn's content with [image blocks…, text block]."""
    if not image_paths:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text = messages[i].get("content", "")
            if isinstance(text, list):  # already block-formatted — leave as is
                return messages
            blocks = [{"type": "image", "image": p} for p in image_paths]
            blocks.append({"type": "text", "text": text})
            messages[i] = {"role": "user", "content": blocks}
            break
    return messages


# ══════════════════════════════════════════════════════════════════════
# Node: Generate
# ══════════════════════════════════════════════════════════════════════


class GenerateNode(BaseCanvasNode):
    """Gemma 3 generation: (messages|prompt, image_paths, stops) → text.

    Either pass a full chat ``messages`` list (preferred) or a plain ``prompt``
    string. ``image_paths`` are file paths (or URLs) on shared disk attached to
    the last user turn. Stop sequences are truncated from the output.
    """

    node_type: ClassVar[str] = "vlm_gemma3__generate"
    display_name: ClassVar[str] = "Gemma 3: Generate"
    description: ClassVar[str] = (
        "Gemma 3 multimodal generation over images — (messages|prompt, image_paths) → text"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "text", "HF model id (blank = $GEMMA3_MODEL_ID or the 4B-it default)", default=""),
            ConfigField("max_new_tokens", "slider", "Max new tokens", default=2048, min=128, max=4096, step=128),
            ConfigField("temperature", "text", "Temperature (0 = greedy)", default=0.0),
            ConfigField("top_p", "text", "Top-p", default=1.0),
            ConfigField("repetition_penalty", "text", "Repetition penalty", default=1.0),
        ],
    )

    input_ports: ClassVar[list] = [
        PortDef("messages", "ANY", "Chat message list [{role, content}] (preferred)"),
        PortDef("prompt", "TEXT", "Single-turn prompt (used if messages absent)"),
        PortDef("image_paths", "ANY", "List of image file paths / URLs (shared disk)"),
        PortDef("stop_sequences", "ANY", "List of stop strings; truncated from output"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Generated text (post stop-sequence truncation)"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        import asyncio

        cfg = self.config or {}
        messages = _coerce_messages(inputs.get("messages"), inputs.get("prompt", "") or "")
        image_paths = _coerce_str_list(inputs.get("image_paths"))
        stops = _coerce_str_list(inputs.get("stop_sequences"))

        engine = _Gemma3Engine.get(str(cfg.get("model_id", "") or "").strip())

        def _gen() -> "str | None":
            if not engine.ensure():
                return None
            import torch

            model, processor = engine.model, engine.processor
            with engine._lock:
                if engine.device == "cuda":
                    torch.cuda.empty_cache()
                msgs = _inject_images([dict(m) for m in messages], image_paths)
                model_inputs = processor.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                do_sample = float(cfg.get("temperature", 0.0)) > 0
                gen_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=int(cfg.get("max_new_tokens", 2048)),
                    temperature=float(cfg.get("temperature", 0.0)),
                    top_p=float(cfg.get("top_p", 1.0)),
                    do_sample=do_sample,
                    repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
                )
                trimmed = [out[len(inp):] for inp, out in zip(model_inputs["input_ids"], gen_ids)]
                out = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            for stop in stops:
                idx = out.find(stop)
                if idx != -1:
                    out = out[:idx]
            return out

        try:
            text = await asyncio.to_thread(_gen)
        except Exception as exc:
            log.exception("Gemma3 generate failed")
            self._self_log("error", str(exc))
            return {"text": ""}

        if text is None:
            self._self_log("degraded", "Gemma3 engine failed to load (gated / no access?)")
            return {"text": ""}
        self._self_log("text_len", len(text))
        return {"text": str(text)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VLMGemma3NodeSet(BaseNodeSet):
    """Generic Gemma 3 multimodal foundation-model nodeset (images).

    Loads Gemma 3 in its own subprocess (shared ``ac-fm`` FM env) and exposes
    ``generate`` as a canvas-wirable primitive. Stateless across calls — engines
    hold loaded weights only.
    """

    name: ClassVar[str] = "vlm_gemma3"
    description: ClassVar[str] = (
        "Gemma 3 — generic generate(messages|prompt, image_paths) primitive over images (gated model)"
    )
    # K callers coalesce through one hosted copy; no per-call state.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Gemma3 is native there).
    # $GEMMA3_PYTHON overrides.
    server_python: ClassVar[str] = conda_env_python("ac-fm", "GEMMA3_PYTHON")

    def get_tools(self) -> list:
        return [GenerateNode()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info(
            "vlm_gemma3 ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        pass
