from __future__ import annotations

"""Qwen3-VL as a generic foundation-model nodeset.

The successor to ``vlm_qwen2_5_vl``: a newer, stronger vision-language model
that reasons over **images and video**. Same generate-primitive contract, one
node::

    vlm_qwen3_vl__generate  — (messages | prompt, image_paths, video_paths,
                               stop_sequences) → text

Video is the new capability over the 2.5 nodeset: pass clip paths on
``video_paths`` and Qwen3-VL reasons over the temporal sequence (VQA over a
rollout, "what changed", temporal grounding), not just a single frame. Media is
attached to the last user turn as chat content blocks and the processor loads it
via the modern ``apply_chat_template(tokenize=True, return_dict=True)`` path —
no ``qwen_vl_utils`` dependency.

FM-template alignment: model identity is node config — ``model_id`` (blank =
``$QWEN3VL_MODEL_ID`` or the 2B-Instruct default), engines in a lazy registry
keyed by the resolved id (checkpoints coexist), load-failure latch (empty text +
``degraded`` self-log), generation knobs on the node UI. The single-flight
generate lock is per-engine (one in-flight generate bounds peak VRAM under K
eval workers; KV-cache memory balloons under concurrent generate).

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11 + torch
2.8.0+cu126 + transformers 5.13.0). Weights are HF-downloaded (Qwen/Qwen3-VL-*,
ungated) and load once per subprocess on first use. On CUDA the model loads in
bfloat16 with flash-attn when the wheel is present (sdpa fallback otherwise);
CPU falls back to float32. Override the env with $QWEN3VL_PYTHON. This file must
stay Python-3.8-parseable.

Model default: Qwen3-VL-2B-Instruct (lightest; co-hosts with other FM nodesets).
Point ``model_id`` (or $QWEN3VL_MODEL_ID) at 4B / 8B / a Thinking checkpoint on
a bigger GPU.

Load: POST /api/components/nodesets/vlm_qwen3_vl/load?mode=server

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

log = logging.getLogger("agentcanvas.vlm_qwen3_vl")

_MODEL_ID_DEFAULT = os.environ.get("QWEN3VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")


class _Qwen3VLEngine:
    """Lazy registry: one loaded Qwen3-VL per resolved ``model_id``."""

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        # Single in-flight generate per engine: concurrent torch.generate
        # KV-cache/activation memory balloons and can CUDA-OOM under K workers.
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_id: str = "") -> "_Qwen3VLEngine":
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
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except Exception:
                log.exception("Qwen3-VL import failed — is the ac-fm env active?")
                self._load_failed = True
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Qwen3-VL model_id=%s on %s", self.model_id, device)

            if device == "cuda":
                model_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "cuda:0"}
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    model_kwargs["attn_implementation"] = "sdpa"
            else:
                model_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}

            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
                processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception:
                log.exception("Qwen3-VL load failed")
                self._load_failed = True
                return False

            self.model, self.processor, self.device = model, processor, device
            self._loaded = True
            log.info("Qwen3-VL ready (model_id=%s)", self.model_id)
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


def _inject_media(messages: list, image_paths: list, video_paths: list) -> list:
    """Replace the LAST user turn's content with [media blocks…, text block].

    Images and videos are attached to the most recent user message as Qwen chat
    content blocks; the processor loads them from path/URL at template time.
    A user turn already in block form is left untouched.
    """
    if not image_paths and not video_paths:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text = messages[i].get("content", "")
            if isinstance(text, list):  # already block-formatted — leave as is
                return messages
            blocks = [{"type": "image", "image": p} for p in image_paths]
            blocks += [{"type": "video", "video": p} for p in video_paths]
            blocks.append({"type": "text", "text": text})
            messages[i] = {"role": "user", "content": blocks}
            break
    return messages


# ══════════════════════════════════════════════════════════════════════
# Node: Generate
# ══════════════════════════════════════════════════════════════════════


class GenerateNode(BaseCanvasNode):
    """Qwen3-VL generation: (messages|prompt, image_paths, video_paths, stops) → text.

    Either pass a full chat ``messages`` list (preferred) or a plain ``prompt``
    string. ``image_paths`` / ``video_paths`` are file paths (or URLs) on shared
    disk attached to the last user turn. Stop sequences are truncated from the
    output.
    """

    node_type: ClassVar[str] = "vlm_qwen3_vl__generate"
    display_name: ClassVar[str] = "Qwen3-VL: Generate"
    description: ClassVar[str] = (
        "Qwen3-VL generation over images + video — (messages|prompt, image_paths, video_paths) → text"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("model_id", "text", "HF model id (blank = $QWEN3VL_MODEL_ID or the 2B default)", default=""),
            ConfigField("max_new_tokens", "slider", "Max new tokens", default=2048, min=128, max=4096, step=128),
            ConfigField("temperature", "text", "Temperature (0 = greedy)", default=0.7),
            ConfigField("top_p", "text", "Top-p", default=0.8),
            ConfigField("top_k", "text", "Top-k", default=20),
            ConfigField("repetition_penalty", "text", "Repetition penalty", default=1.0),
        ],
    )

    input_ports: ClassVar[list] = [
        PortDef("messages", "ANY", "Chat message list [{role, content}] (preferred)"),
        PortDef("prompt", "TEXT", "Single-turn prompt (used if messages absent)"),
        PortDef("image_paths", "ANY", "List of image file paths / URLs (shared disk)"),
        PortDef("video_paths", "ANY", "List of video file paths / URLs (shared disk)"),
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
        video_paths = _coerce_str_list(inputs.get("video_paths"))
        stops = _coerce_str_list(inputs.get("stop_sequences"))

        engine = _Qwen3VLEngine.get(str(cfg.get("model_id", "") or "").strip())

        def _gen() -> "str | None":
            if not engine.ensure():
                return None
            import torch

            model, processor = engine.model, engine.processor
            with engine._lock:
                if engine.device == "cuda":
                    torch.cuda.empty_cache()
                msgs = _inject_media([dict(m) for m in messages], image_paths, video_paths)
                model_inputs = processor.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                do_sample = float(cfg.get("temperature", 0.7)) > 0
                gen_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=int(cfg.get("max_new_tokens", 2048)),
                    temperature=float(cfg.get("temperature", 0.7)),
                    top_p=float(cfg.get("top_p", 0.8)),
                    top_k=int(cfg.get("top_k", 20)),
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
            log.exception("Qwen3-VL generate failed")
            self._self_log("error", str(exc))
            return {"text": ""}

        if text is None:
            self._self_log("degraded", "Qwen3-VL engine failed to load")
            return {"text": ""}
        self._self_log("text_len", len(text))
        return {"text": str(text)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VLMQwen3VLNodeSet(BaseNodeSet):
    """Generic Qwen3-VL foundation-model nodeset (images + video).

    Loads Qwen3-VL in its own subprocess (shared ``ac-fm`` FM env) and exposes
    ``generate`` as a canvas-wirable primitive. Stateless across calls — engines
    hold loaded weights only.
    """

    name: ClassVar[str] = "vlm_qwen3_vl"
    description: ClassVar[str] = (
        "Qwen3-VL — generic generate(messages|prompt, image_paths, video_paths) primitive over images and video"
    )
    # K callers coalesce through one hosted copy; no per-call state.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env; transformers Qwen3VL is native there).
    # $QWEN3VL_PYTHON overrides.
    server_python: ClassVar[str] = conda_env_python("ac-fm", "QWEN3VL_PYTHON")

    def get_tools(self) -> list:
        return [GenerateNode()]

    async def initialize(self, **kwargs: Any) -> None:
        # Lazy: weights load on the first generate (a warmup here would block
        # teardown-free reloads and eat VRAM even when the node is unused).
        log.info(
            "vlm_qwen3_vl ready (server_python=%s); engine loads lazily per model_id",
            self.server_python,
        )

    async def shutdown(self) -> None:
        # Retain engines across reloads; freed only on subprocess teardown.
        pass
