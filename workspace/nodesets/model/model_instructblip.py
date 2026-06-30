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

Runs **server mode** (own subprocess + CUDA context) so the parent eval holds
no InstructBLIP VRAM and worker pools can coalesce onto one shared server.

Hosted in the default ``agentcanvas`` env (torch 2.4.x + transformers 4.45.2 +
tokenizers 0.20.3) — the SAME env where ``navgpt_mp3d_tools`` already ran this
exact InstructBLIP forward (verified run 20260615_173543), so the manual
image-token unpacking below (which relies on ``processor.image_token`` /
``processor.num_query_tokens``, added in transformers ~4.44) is correct here.
NOT ``ac-ram``: that env has tokenizers 0.15.2 (too old to parse the
flan-t5 fast ``tokenizer.json`` → ``PyPreTokenizerTypeWrapper`` error) and no
sentencepiece (so ``use_fast=False`` also fails) — InstructBlipProcessor cannot
load there. That mis-choice 500-ed every caption call in run 20260616_150115
(SR 0, 0 steps). A dedicated ``agentcanvas-instructblip`` env remains the ideal
per memory feedback_dedicated_env_per_model; reusing the default env avoids a
redundant multi-GB build and is version-correct for this code.

last updated: 2026-06-16
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

from app.components import BaseCanvasNode, BaseNodeSet, ConfigField, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.model_instructblip")

_INSTRUCTBLIP_PROMPT = "Describe this indoor scene in details"

# Per-server content-hash cache: sha1(rgb_base64) → caption. Reuses captions for
# byte-identical views recurring across steps/workers (upstream view_record).
_CAPTION_CACHE: dict[str, str] = {}

# Lazy singleton (per server subprocess)
_model = None
_processor = None
_device = None
_load_lock = threading.Lock()


def _get_instructblip(model_name: str = "Salesforce/instructblip-flan-t5-xl", device: str = "auto"):
    """Lazy-load InstructBLIP (FlanT5-XL). Verbatim port of the navgpt loader."""
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device
    with _load_lock:
        if _model is not None:
            return _model, _processor, _device
        import torch
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading InstructBLIP %s on %s …", model_name, device)
        processor = InstructBlipProcessor.from_pretrained(model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        _processor, _model, _device = processor, model, device
        log.info("InstructBLIP loaded (%s)", device)
        return _model, _processor, _device


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
                default="Salesforce/instructblip-flan-t5-xl",
            ),
            ConfigField("prompt", "text", "Description prompt", default=_INSTRUCTBLIP_PROMPT),
            ConfigField(
                "device", "select", "Device",
                options=[
                    {"value": "auto", "label": "Auto"},
                    {"value": "cuda", "label": "CUDA"},
                    {"value": "cpu", "label": "CPU"},
                ],
                default="auto",
            ),
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
        model_name = config.get("model_name", "Salesforce/instructblip-flan-t5-xl")
        prompt = config.get("prompt", _INSTRUCTBLIP_PROMPT)
        device = config.get("device", "auto")
        # Decoding aligned to LAVIS Blip2T5Instruct.generate defaults (beam-5,
        # max_length 256) so captions match upstream observe_view, not a greedy
        # 64-token truncation. LAVIS calls the same HF T5 .generate under the
        # hood, so matching kwargs makes the output equivalent up to float noise.
        max_length = int(config.get("max_length", 256))
        num_beams = int(config.get("num_beams", 5))

        loop = asyncio.get_running_loop()

        def _caption_all() -> list[str]:
            import torch
            from PIL import Image

            # Manual unpacking — InstructBlipProcessor.__call__ concatenates the
            # image-token list with the text Tensor and trips a "list + Tensor"
            # TypeError; bypass by calling the sub-tokenizers directly and
            # prepending image tokens ourselves (verbatim navgpt workaround).
            import hashlib

            model, processor, dev = _get_instructblip(model_name, device)
            num_q = processor.num_query_tokens or 32
            img_token_str = processor.image_token.content * num_q
            cast_dtype = torch.float16 if dev == "cuda" else torch.float32
            out: list[str] = ["" for _ in views]
            for i, v in enumerate(views):
                if not isinstance(v, dict):
                    continue
                b64 = v.get("rgb_base64")
                if not b64:
                    continue
                key = hashlib.sha1(b64.encode("ascii")).hexdigest()
                if key in _CAPTION_CACHE:
                    out[i] = _CAPTION_CACHE[key]
                    continue
                try:
                    pil = Image.fromarray(_decode_rgb(b64)).convert("RGB")
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
                    out[i] = (decoded[0] if decoded else "").strip()
                    _CAPTION_CACHE[key] = out[i]
                except Exception as exc:
                    log.warning("InstructBLIP caption failed for dir %s: %s", v.get("dir_id"), exc)
            # Mitigate cumulative CUDA allocator growth across many calls.
            try:
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return out

        captions = await loop.run_in_executor(None, _caption_all)
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
    # Default env: agentcanvas (transformers 4.45.2 + tokenizers 0.20.3) — the
    # env where this exact forward already worked via navgpt_mp3d_tools. NOT
    # ac-ram (tokenizers 0.15.2 + no sentencepiece → processor load
    # fails). Override with $INSTRUCTBLIP_PYTHON if a dedicated env is built.
    server_python = os.environ.get(
        "INSTRUCTBLIP_PYTHON",
        os.path.expanduser("~/miniforge3/envs/agentcanvas/bin/python"),
    )

    def get_tools(self) -> list:
        return [InstructBlipCaptionTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("InstructBlipNodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        pass
