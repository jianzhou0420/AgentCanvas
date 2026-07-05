from __future__ import annotations

"""BLIP-2 scene-caption — dedicated server-mode foundation-model nodeset.

Extracted from ``navgpt_mp3d_tools`` (where the BLIP-2 caption node lived as a
*method* node) into a clean foundation-model nodeset per the method /
foundation-model boundary principle (roadmap TODO #56): per-view captioning is
a generic vision primitive. NavGPT's task glue (3-elevation merging + 8-compass
direction labelling) stays behind in ``navgpt_mp3d_tools__format_captions``.

Source: NavGPT offline preprocessing (BLIP-2 ViT-G FlanT5-XL over 24 egocentric
views per viewpoint); the online loader/generate code here is moved verbatim
from ``navgpt_mp3d_tools._get_blip2`` + ``BLIP2CaptionNode.forward``
(greedy decode, ``max_new_tokens=64``, prompt ``"This is a scene of"``).

Single tool::

    model_blip2__caption  (views: list[{dir_id, rgb_base64}])
                          → captions_per_dir: LIST[TEXT] (+ captions_json)

Input mirrors ``model_ram__tag_panorama`` / ``model_instructblip__caption`` —
base64 view-tile dicts, JSON-safe across the server-mode HTTP boundary; output
aligned 1:1 with ``views``.

Runs **server mode** (own subprocess + CUDA context, ~4 GB fp16). Hosted in the
shared ``ac-fm`` FM env (torch 2.8.0+cu126 + transformers 5.13.0) since
2026-07-05 — captions verified byte-identical to the previous ``agentcanvas``
hosting (greedy decode, synthetic-image parity replay). Override with
$BLIP2_PYTHON to pin a different env.

Load: POST /api/components/nodesets/model_blip2/load?mode=server

last updated: 2026-07-05
"""

import asyncio
import base64
import io
import json
import logging
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

log = logging.getLogger("agentcanvas.model_blip2")

_BLIP2_PROMPT = "This is a scene of"

# Per-server content-hash cache: sha1(rgb_base64 + prompt + params) → caption.
# Keyed on the generation-relevant config too (unlike the older instructblip
# cache) so a prompt change in graph config can never serve stale captions.
_CAPTION_CACHE: dict[str, str] = {}

# Lazy singleton (per server subprocess)
_model = None
_processor = None
_device = None
_load_lock = threading.Lock()


def _get_blip2(model_name: str = "Salesforce/blip2-flan-t5-xl", device: str = "auto"):
    """Lazy-load BLIP-2 model + processor. Verbatim port of the navgpt loader."""
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device
    with _load_lock:
        if _model is not None:
            return _model, _processor, _device
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading BLIP-2 %s on %s …", model_name, device)
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        _processor, _model, _device = processor, model, device
        log.info("BLIP-2 loaded (%s)", device)
        return _model, _processor, _device


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


class Blip2CaptionTool(BaseCanvasNode):
    """Per-view caption with BLIP-2 FlanT5-XL, ordered output.

    Verbatim NavGPT decode: greedy ``generate(max_new_tokens=64)`` with the
    ``"This is a scene of"`` prompt prefix. ``captions_per_dir`` is aligned
    1:1 with the input ``views`` order.
    """

    node_type: ClassVar[str] = "model_blip2__caption"
    display_name: ClassVar[str] = "BLIP-2: Caption Views"
    description: ClassVar[str] = (
        "BLIP-2 FlanT5-XL caption per view; ordered LIST[TEXT] output"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "ScanEye"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "model_name", "text", "HuggingFace model ID",
                default="Salesforce/blip2-flan-t5-xl",
            ),
            ConfigField("prompt", "text", "Caption prompt prefix", default=_BLIP2_PROMPT),
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
                "max_new_tokens", "slider", "Max tokens per caption",
                default=64, min=16, max=256, step=16,
            ),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64} dicts (e.g. 24 views)"),
    ]
    output_ports = [
        PortDef(
            "captions_per_dir", "LIST[TEXT]",
            "Per-view captions, aligned 1:1 with `views`",
        ),
        PortDef("captions_json", "TEXT", "Same list serialised as JSON"),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"captions_per_dir": [], "captions_json": "[]"}

        config = getattr(self, "config", None) or {}
        model_name = config.get("model_name", "Salesforce/blip2-flan-t5-xl")
        prompt = config.get("prompt", _BLIP2_PROMPT)
        device = config.get("device", "auto")
        max_new_tokens = int(config.get("max_new_tokens", 64))

        loop = asyncio.get_running_loop()

        def _caption_all() -> list[str]:
            import hashlib

            import torch
            from PIL import Image

            model, processor, dev = _get_blip2(model_name, device)
            out: list[str] = ["" for _ in views]
            for i, v in enumerate(views):
                if not isinstance(v, dict):
                    continue
                b64 = v.get("rgb_base64")
                if not b64:
                    continue
                key = hashlib.sha1(
                    f"{model_name}|{prompt}|{max_new_tokens}|{b64}".encode("ascii", "ignore")
                ).hexdigest()
                if key in _CAPTION_CACHE:
                    out[i] = _CAPTION_CACHE[key]
                    continue
                try:
                    pil = Image.fromarray(_decode_rgb(b64)).convert("RGB")
                    enc = processor(images=pil, text=prompt, return_tensors="pt").to(
                        dev,
                        dtype=torch.float16 if dev == "cuda" else torch.float32,
                    )
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=max_new_tokens)
                    out[i] = processor.decode(gen[0], skip_special_tokens=True).strip()
                    _CAPTION_CACHE[key] = out[i]
                except Exception as exc:
                    log.warning("BLIP-2 caption failed for %s: %s", v.get("dir_id"), exc)
            return out

        captions = await loop.run_in_executor(None, _caption_all)
        self._self_log("n_captions", len(captions))
        for i, c in enumerate(captions):
            self._self_log(f"caption_{i}", c[:200])
        return {"captions_per_dir": captions, "captions_json": json.dumps(captions)}


class Blip2NodeSet(BaseNodeSet):
    """BLIP-2 FlanT5-XL captioning — dedicated server-mode FM nodeset."""

    name = "model_blip2"
    description = "BLIP-2 FlanT5-XL per-view captioning — dedicated server-mode FM nodeset"
    # Stateless captioner — one shared server, K eval workers coalesce onto it.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env) — byte-identical captions vs the
    # previous agentcanvas hosting (parity gate 2026-07-05). $BLIP2_PYTHON
    # overrides.
    server_python = conda_env_python("ac-fm", "BLIP2_PYTHON")

    def get_tools(self) -> list:
        return [Blip2CaptionTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("Blip2NodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        pass
