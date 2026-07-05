"""Qwen2.5-VL as a generic foundation-model nodeset.

Serves the ReAct reasoning + VQA VLM behind the ToolEQA method nodeset.
One node:

  vlm_qwen2_5_vl__generate  — (messages | prompt, image_paths, stop_sequences) → text

The contract mirrors the upstream ``QwenEngine.__call__`` used by ToolEQA's
``transformers.agents.ReactCodeAgent``: a chat message list plus a list of
image *file paths* (the agent saves frames to disk and passes paths), with
manual stop-sequence truncation. ToolEQA's method node calls this over the
standard server-mode HTTP route (NOT in-process), so heavy deps (torch,
transformers, qwen_vl_utils) live behind lazy imports and only load inside
the server subprocess.

FM-template alignment (2026-07-05, second pass): model identity is node
config — ``model_dir`` (blank = ``$QWENVL_MODEL_DIR`` or the repo-anchored
3B default), engines in a lazy registry keyed by the resolved dir (the old
module-global bundle ignored a changed dir), load-failure latch (empty text
+ ``degraded`` self-log), generation knobs on the node UI. The single-flight
generate lock is per-engine now (one in-flight generate bounds peak VRAM
under K eval workers; KV-cache memory balloons under concurrent generate).

Server mode under the shared ``ac-fm`` FM env (Python 3.11 + torch
2.8.0+cu126 + transformers 5.13.0 + qwen-vl-utils) since 2026-07-05 —
greedy generations verified byte-identical to the previous ``ac-qwenvl``
hosting under matched sdpa attention. On hosts whose glibc is too old for
the flash-attn wheel (<2.32) the loader falls back to sdpa; inside a
newer-glibc Docker base flash-attention re-activates. Weights load once
per subprocess on first ``initialize()`` and live until teardown.

Model: Qwen2.5-VL-3B-Instruct (single-3090 budget; co-hosts with DetAny3D +
Habitat — see scripts/install/install_ac_qwenvl.sh for the 3B rationale).
Point ``model_dir`` (or ``$QWENVL_MODEL_DIR``) at a 7B checkout on a bigger
GPU.

last updated: 2026-07-05
"""

from __future__ import annotations

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

log = logging.getLogger("agentcanvas.vlm_qwen2_5_vl")


def _find_repo_root() -> str:
    """Walk upward from this file until a dir containing ``data/`` is found.

    A fixed ``../../..`` breaks when this file is served from a workspace
    OVERLAY copy; fall back to the auto_host server CWD's repo root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 7):
        cand = os.path.normpath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), "..", ".."))


_DEFAULT_MODEL_DIR = os.environ.get(
    "QWENVL_MODEL_DIR",
    os.path.join(_find_repo_root(), "data", "qwen2_5_vl", "Qwen2.5-VL-3B-Instruct"),
)


class _QwenEngine:
    """Lazy registry: one loaded Qwen2.5-VL per resolved ``model_dir``."""

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        # Single in-flight generate per engine: concurrent torch.generate
        # KV-cache/activation memory balloons and can CUDA-OOM under K eval
        # workers each issuing ReAct + go_next LSV/GSV calls.
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_dir: str = "") -> "_QwenEngine":
        resolved = os.path.normpath(model_dir or _DEFAULT_MODEL_DIR)
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
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except Exception:
                log.exception("Qwen2.5-VL import failed — is the ac-fm env active?")
                self._load_failed = True
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Qwen2.5-VL model_dir=%s on %s", self.model_dir, device)

            if device == "cuda":
                # device_map MUST carry an index ("cuda:0", not "cuda") — transformers
                # 4.50's caching_allocator_warmup calls torch.cuda.mem_get_info(device)
                # which rejects the bare "cuda" string. bfloat16 explicit (flash-attn
                # warns + can misbehave under torch_dtype="auto").
                model_kwargs: dict = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "cuda:0",
                }
                # Best-effort flash-attn; fall back to sdpa if the wheel is absent.
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    model_kwargs["attn_implementation"] = "sdpa"
            else:
                model_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}

            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_dir, **model_kwargs
                )
                processor = AutoProcessor.from_pretrained(self.model_dir)
            except Exception:
                log.exception("Qwen2.5-VL load failed")
                self._load_failed = True
                return False

            self.model, self.processor, self.device = model, processor, device
            self._loaded = True
            log.info("Qwen2.5-VL ready (model_dir=%s)", self.model_dir)
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
    """Replace the LAST user turn's content with [image blocks…, text block].

    Verbatim port of upstream ``QwenEngine.call_vlm`` image handling — the
    image is attached to the most recent user message, leaving prior turns
    text-only.
    """
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
# Node: Generate — the ReAct/VQA engine
# ══════════════════════════════════════════════════════════════════════


class GenerateNode(BaseCanvasNode):
    """Qwen2.5-VL generation: (messages | prompt, image_paths, stop_sequences) → text.

    Ports mirror the upstream ``QwenEngine.__call__`` signature so ToolEQA's
    ReAct loop can use this as a drop-in ``llm_engine`` over HTTP. Either pass
    a full chat ``messages`` list (preferred — the agent builds it) or a plain
    ``prompt`` string. ``image_paths`` are file paths on shared disk attached
    to the last user turn. Stop sequences are truncated from the output.
    """

    node_type: ClassVar[str] = "vlm_qwen2_5_vl__generate"
    display_name: ClassVar[str] = "Qwen2.5-VL: Generate"
    description: ClassVar[str] = (
        "Qwen2.5-VL generation — (messages|prompt, image_paths, stop_sequences) → text"
    )
    category: ClassVar[str] = "reasoning"
    icon: ClassVar[str] = "MessageSquare"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model_dir", "text",
                "Model directory (blank = $QWENVL_MODEL_DIR or the 3B default)",
                default="",
            ),
            ConfigField(
                "max_new_tokens", "slider", "Max new tokens",
                default=2048, min=128, max=4096, step=128,
            ),
            ConfigField("temperature", "text", "Temperature (0 = greedy)", default=0.7),
            ConfigField("top_p", "text", "Top-p", default=0.8),
            ConfigField("top_k", "text", "Top-k", default=100),
            ConfigField("repetition_penalty", "text", "Repetition penalty", default=1.05),
        ],
    )

    input_ports: ClassVar[list] = [
        PortDef("messages", "ANY", "Chat message list [{role, content}] (preferred)"),
        PortDef("prompt", "TEXT", "Single-turn prompt (used if messages absent)"),
        PortDef("image_paths", "ANY", "List of image file paths (shared disk)"),
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

        engine = _QwenEngine.get(str(cfg.get("model_dir", "") or "").strip())

        def _gen() -> "str | None":
            if not engine.ensure():
                return None
            import torch
            from qwen_vl_utils import process_vision_info

            model, processor = engine.model, engine.processor
            with engine._lock:
                torch.cuda.empty_cache()
                msgs = _inject_images([dict(m) for m in messages], image_paths)
                text = processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(msgs)
                model_inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                do_sample = float(cfg.get("temperature", 0.7)) > 0
                gen_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=int(cfg.get("max_new_tokens", 2048)),
                    temperature=float(cfg.get("temperature", 0.7)),
                    top_p=float(cfg.get("top_p", 0.8)),
                    top_k=int(cfg.get("top_k", 100)),
                    do_sample=do_sample,
                    repetition_penalty=float(cfg.get("repetition_penalty", 1.05)),
                )
                trimmed = [
                    out[len(inp):] for inp, out in zip(model_inputs.input_ids, gen_ids)
                ]
                out = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            for stop in stops:
                idx = out.find(stop)
                if idx != -1:
                    out = out[:idx]
            return out

        try:
            text = await asyncio.to_thread(_gen)
        except Exception as exc:
            log.exception("Qwen2.5-VL generate failed")
            self._self_log("error", str(exc))
            return {"text": ""}

        if text is None:
            self._self_log("degraded", "Qwen2.5-VL engine failed to load")
            return {"text": ""}
        self._self_log("text_len", len(text))
        return {"text": str(text)}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class VLMQwen25VLNodeSet(BaseNodeSet):
    """Generic Qwen2.5-VL foundation-model nodeset.

    Loads Qwen2.5-VL in its own subprocess (shared ``ac-fm`` FM env)
    and exposes ``generate`` as a canvas-wirable primitive. Stateless across
    calls — engines hold loaded weights only.
    """

    name: ClassVar[str] = "vlm_qwen2_5_vl"
    description: ClassVar[str] = (
        "Qwen2.5-VL — generic generate(messages|prompt, image_paths) primitive"
    )
    # K callers coalesce through one hosted copy; no per-call state.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env) — greedy output byte-identical vs the
    # retired ac-qwenvl hosting (parity gate 2026-07-05). $QWENVL_PYTHON overrides.
    server_python: ClassVar[str] = conda_env_python("ac-fm", "QWENVL_PYTHON")

    def get_tools(self) -> list:
        return [GenerateNode()]

    async def initialize(self, **kwargs: Any) -> None:
        # Eager warmup of the default engine: ToolEQA's first ReAct call lands
        # within a per-step budget; loading lazily there would eat it.
        import asyncio

        await asyncio.to_thread(_QwenEngine.get("").ensure)

    async def shutdown(self) -> None:
        # Retain engines across reloads; freed only on subprocess teardown.
        pass
