from __future__ import annotations

"""InstructBLIP scene-caption — dedicated server-mode foundation-model nodeset.

Extracted from ``navgpt_mp3d_tools`` (where the InstructBLIP caption node
lived as a *method* node) into a clean foundation-model nodeset per the
method / foundation-model boundary principle (roadmap TODO #56): per-view
scene captioning is a generic vision primitive, consumed by NavGPT-MP3D and
DiscussNav alike — exactly like ``model_ram`` for RAM tagging.

Source: DiscussNav.py:133 (LAVIS ``blip2_t5_instruct/flant5xl``) + :169-170
(``Scene Description: {instructblip} Scene Objects: {ram};`` per direction).
HF ``transformers`` equivalent used here (LAVIS not installed).

Single tool::

    model_instructblip__caption  (views: list[{dir_id, rgb_base64}])
                                 → captions_per_dir: LIST[TEXT] (+ captions_json)

Input mirrors ``model_ram__tag_panorama`` — base64 view-tile dicts, not raw
LIST[IMAGE] — so the payload is JSON-safe across the server-mode HTTP
boundary and the caption list aligns 1:1 with the RAM tag list (both fed by
``discussnav__panorama_to_views``).

FM-template alignment (2026-07-05, second pass): fully **stateless** — the
former sha1(image)-keyed caption cache is gone. That cache was also a real
bug: keyed on image bytes only, a prompt/beam/model config change would keep
serving captions generated under the old config. Engines live in a lazy
registry keyed by ``model_name`` (the old module singleton ignored a changed
model id), with a load-failure latch (empty outputs + ``degraded`` self-log)
and a single-flight GPU inference lock. The ``device`` node config moved to
the deployment level: ``$INSTRUCTBLIP_DEVICE`` (auto → cuda when available).

Runs **server mode** (own subprocess + CUDA context) so the parent eval holds
no InstructBLIP VRAM and worker pools can coalesce onto one shared server.

Hosted in the shared ``ac-fm`` FM env (torch 2.8.0+cu126 + transformers 5.13.0
+ tokenizers 0.22) since 2026-07-05 — beam-5 captions verified byte-identical
to the previous ``agentcanvas`` hosting, and the manual image-token unpacking
below (``processor.image_token`` / ``processor.num_query_tokens``) works
unchanged under 5.x. NOT ``ac-ram``: that env has tokenizers 0.15.2 (too old
to parse the flan-t5 fast ``tokenizer.json`` → ``PyPreTokenizerTypeWrapper``
error) and no sentencepiece — InstructBlipProcessor cannot load there; that
mis-choice 500-ed every caption call in run 20260616_150115 (SR 0, 0 steps).
Override with $INSTRUCTBLIP_PYTHON to pin a different env.

last updated: 2026-07-05
"""

import asyncio
import base64
import io
import json
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

log = logging.getLogger("agentcanvas.model_instructblip")

_INSTRUCTBLIP_PROMPT = "Describe this indoor scene in details"
_MODEL_DEFAULT = "Salesforce/instructblip-flan-t5-xl"


def _resolve_device() -> str:
    dev = os.environ.get("INSTRUCTBLIP_DEVICE", "auto")
    if dev != "auto":
        return dev
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _InstructBlipEngine:
    """Lazy registry: one loaded InstructBLIP per ``model_name``; weights only."""

    _instances: ClassVar[dict] = {}
    _registry_lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()  # guards load AND single-flight inference

    @classmethod
    def get(cls, model_name: str) -> "_InstructBlipEngine":
        key = (model_name,)
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_name)
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
                from transformers import (
                    InstructBlipForConditionalGeneration,
                    InstructBlipProcessor,
                )

                device = _resolve_device()
                log.info("Loading InstructBLIP %s on %s …", self.model_name, device)
                processor = InstructBlipProcessor.from_pretrained(self.model_name)
                model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device)
                model.eval()
                self.processor, self.model, self.device = processor, model, device
                self._loaded = True
                log.info("InstructBLIP loaded (%s)", device)
                return True
            except Exception as exc:
                log.warning("InstructBLIP load failed: %s", exc)
                self._load_failed = True
                return False

    def caption(self, rgb: np.ndarray, prompt: str, max_length: int, num_beams: int) -> str:
        """Single-view beam-search description — verbatim navgpt generate path.

        Manual unpacking — InstructBlipProcessor.__call__ concatenates the
        image-token list with the text Tensor and trips a "list + Tensor"
        TypeError; bypass by calling the sub-tokenizers directly and
        prepending image tokens ourselves (verbatim navgpt workaround).
        """
        import torch
        from PIL import Image

        processor, model, dev = self.processor, self.model, self.device
        num_q = processor.num_query_tokens or 32
        img_token_str = processor.image_token.content * num_q
        cast_dtype = torch.float16 if dev == "cuda" else torch.float32
        pil = Image.fromarray(rgb).convert("RGB")
        with self._lock:
            full_text = img_token_str + prompt
            text_enc = processor.tokenizer(full_text, return_tensors="pt")
            qf_enc = processor.qformer_tokenizer(prompt, return_tensors="pt")
            img_enc = processor.image_processor(pil, return_tensors="pt")
            proc_inputs = {
                "input_ids": text_enc["input_ids"].to(dev),
                "attention_mask": text_enc["attention_mask"].to(dev),
                "qformer_input_ids": qf_enc["input_ids"].to(dev),
                "qformer_attention_mask": qf_enc["attention_mask"].to(dev),
                "pixel_values": img_enc["pixel_values"].to(dev, dtype=cast_dtype),
            }
            with torch.no_grad():
                gen = model.generate(
                    **proc_inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    do_sample=False,
                )
            decoded = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
            return (decoded[0] if decoded else "").strip()


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


class InstructBlipCaptionTool(BaseCanvasNode):
    """Per-direction scene description with InstructBLIP-FlanT5-XL, ordered output.

    Verbatim DiscussNav prompt *"Describe this indoor scene in details"*
    (DiscussNav.py:148). ``captions_per_dir`` is aligned 1:1 with the input
    ``views`` order (same base64 tiles that feed ``model_ram``).
    """

    node_type: ClassVar[str] = "model_instructblip__caption"
    display_name: ClassVar[str] = "InstructBLIP: Caption Panorama"
    description: ClassVar[str] = (
        "InstructBLIP-FlanT5-XL scene description per direction; ordered LIST[TEXT] output"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ScanEye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model_name", "text", "HuggingFace model ID",
                default=_MODEL_DEFAULT,
            ),
            ConfigField("prompt", "text", "Description prompt", default=_INSTRUCTBLIP_PROMPT),
            ConfigField(
                "max_length", "slider", "Max caption length (LAVIS default 256)",
                default=256, min=32, max=384, step=16,
            ),
            ConfigField(
                "num_beams", "slider", "Beam search width (LAVIS default 5)",
                default=5, min=1, max=8, step=1,
            ),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64} dicts (e.g. 12 directions)"),
    ]
    output_ports = [
        PortDef(
            "captions_per_dir", "LIST[TEXT]",
            "Per-direction descriptions, aligned 1:1 with `views`",
        ),
        PortDef("captions_json", "TEXT", "Same list serialised as JSON"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"captions_per_dir": [], "captions_json": "[]"}

        config = getattr(self, "config", None) or {}
        model_name = config.get("model_name", _MODEL_DEFAULT)
        prompt = config.get("prompt", _INSTRUCTBLIP_PROMPT)
        # Decoding aligned to LAVIS Blip2T5Instruct.generate defaults (beam-5,
        # max_length 256) so captions match upstream observe_view, not a greedy
        # 64-token truncation. LAVIS calls the same HF T5 .generate under the
        # hood, so matching kwargs makes the output equivalent up to float noise.
        max_length = int(config.get("max_length", 256))
        num_beams = int(config.get("num_beams", 5))

        loop = asyncio.get_running_loop()
        engine = _InstructBlipEngine.get(model_name)

        def _caption_all() -> "list[str] | None":
            import torch

            if not engine.ensure():
                return None
            out: list[str] = ["" for _ in views]
            for i, v in enumerate(views):
                if not isinstance(v, dict):
                    continue
                b64 = v.get("rgb_base64")
                if not b64:
                    continue
                try:
                    out[i] = engine.caption(_decode_rgb(b64), prompt, max_length, num_beams)
                except Exception as exc:
                    log.warning("InstructBLIP caption failed for dir %s: %s", v.get("dir_id"), exc)
            # Mitigate cumulative CUDA allocator growth across many calls.
            try:
                if engine.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return out

        captions = await loop.run_in_executor(None, _caption_all)
        if captions is None:
            self._self_log("degraded", "InstructBLIP engine failed to load")
            return {"captions_per_dir": [], "captions_json": ""}
        self._self_log("n_captions", len(captions))
        for i, c in enumerate(captions):
            self._self_log(f"caption_{i}", c[:200])
        return {"captions_per_dir": captions, "captions_json": json.dumps(captions)}


class InstructBlipNodeSet(BaseNodeSet):
    """InstructBLIP-FlanT5-XL scene captioning — dedicated server-mode FM nodeset."""

    name = "model_instructblip"
    description = "InstructBLIP-FlanT5-XL scene captioning — dedicated server-mode FM nodeset"
    # Stateless captioner — one shared server, K eval workers coalesce onto it
    # (don't replicate the ~7 GB model per worker). Bit-identical at worker_count=1.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env) — beam-5 captions byte-identical vs
    # the previous agentcanvas hosting (parity gate 2026-07-05). NOT ac-ram
    # (tokenizers 0.15.2 + no sentencepiece → processor load fails).
    # Override with $INSTRUCTBLIP_PYTHON.
    server_python = conda_env_python("ac-fm", "INSTRUCTBLIP_PYTHON")

    def get_tools(self) -> list:
        return [InstructBlipCaptionTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("InstructBlipNodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        pass
