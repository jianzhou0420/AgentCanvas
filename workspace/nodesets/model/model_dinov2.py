from __future__ import annotations

"""DINOv2 image features — server-mode foundation-model nodeset.

Extracted from ``smartway_waypoint`` (where DINOv2 lived inside the waypoint
engine as the RGB backbone) per the method / foundation-model boundary
principle (roadmap TODO #56) and an explicit user decision (2026-07-04): the
DDPPO depth encoder stays method-internal, but the DINOv2 backbone is exposed
as a generic per-image feature primitive.

Load + forward are byte-faithful to the smartway engine (upstream
``Policy_ViewSelection_VLNBERT.py:111`` + ``base_il_trainer.py:356``):
``torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')`` frozen/eval,
preprocessed by ``AutoImageProcessor('facebook/dinov2-small')``, one batched
forward → pooled per-image embeddings (ViT-S/14-reg → 384-d). DINOv2 is a
per-image ViT (no cross-sample ops), so features computed here in input order
equal the ones the old engine computed after its clockwise reorder — the
consumer applies its own ordering.

Single tool::

    model_dinov2__extract_features  (views: ordered list of {rgb_base64})
                                    → features: TEXT JSON envelope
                                      {"shape":[N,D],"dtype":"float32","b64":…}

The envelope carries the raw C-contiguous float32 buffer base64-encoded —
byte-exact across the HTTP boundary (a JSON float list would round-trip
through decimal text) and ~4× smaller.

Runs **server mode** in the shared ``ac-fm`` FM env (Python 3.11, torch
2.8.0+cu126, transformers 5.13.0) since 2026-07-05. Numeric parity with the
old ``ac-smartway`` hosting is BYTE-EXACT under two pinned conditions, both
verified in the 2026-07-05 parity probe: the ViT forward is bit-identical
across torch 2.1.1→2.8.0, and the processor is forced to the PIL backend
(``use_fast=False`` below — transformers 5.x flipped the default to a
torchvision implementation whose resize numerics differ). Override with
$DINOV2_PYTHON. This file must stay Python-3.8-parseable (override may point
at a py3.8 env).

Load: POST /api/components/nodesets/model_dinov2/load?mode=server

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

log = logging.getLogger("agentcanvas.model_dinov2")

_HUB_REPO_DEFAULT = "facebookresearch/dinov2"
_HUB_MODEL_DEFAULT = "dinov2_vits14_reg"
_PROCESSOR_DEFAULT = "facebook/dinov2-small"


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton per (hub_model, processor_id)
# ══════════════════════════════════════════════════════════════════════


class _DinoV2Engine:
    """Lazy singleton registry: one frozen DINOv2 per (repo, model, processor)."""

    _instances: ClassVar[dict] = {}
    _lock = threading.Lock()

    def __init__(self, hub_repo: str, hub_model: str, processor_id: str) -> None:
        self.hub_repo = hub_repo
        self.hub_model = hub_model
        self.processor_id = processor_id
        self.device = None
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_failed = False

    @classmethod
    def get(cls, hub_repo: str, hub_model: str, processor_id: str) -> "_DinoV2Engine":
        key = (hub_repo, hub_model, processor_id)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(hub_repo, hub_model, processor_id)
            return cls._instances[key]

    def _ensure(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Verbatim smartway engine load (upstream
            # Policy_ViewSelection_VLNBERT.py:111): frozen, eval.
            try:
                model = torch.hub.load(self.hub_repo, self.hub_model).to(self.device)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.model = model
            except Exception as exc:
                log.warning("DINOv2 hub load failed: %s", exc)
                self._load_failed = True
                return False
            try:
                from transformers import AutoImageProcessor  # type: ignore

                # use_fast=False pins the PIL preprocessing backend: byte-equal
                # pixels/features vs the pre-2026-07-05 ac-smartway hosting.
                # transformers 5.x defaults to a torchvision backend whose
                # resize numerics differ; the kwarg is valid (no-op) on 4.x.
                self.processor = AutoImageProcessor.from_pretrained(
                    self.processor_id, use_fast=False)
            except Exception as exc:
                log.warning("DINO processor load failed: %s", exc)
                self._load_failed = True
                return False
            self._loaded = True
            log.info(
                "DINOv2 ready (%s/%s, device=%s)",
                self.hub_repo, self.hub_model, self.device,
            )
            return True

    def extract(self, images: list) -> "np.ndarray | None":
        """Batched forward over HWC uint8 arrays → (N, D) float32, input order."""
        if not self._ensure():
            return None
        import torch

        pp = self.processor(images=images, return_tensors="pt")
        pixel_values = pp["pixel_values"].to(self.device)
        with torch.no_grad():
            feats = self.model(pixel_values)
        return feats.detach().cpu().numpy().astype(np.float32)


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class ExtractFeaturesTool(BaseCanvasNode):
    """Per-image DINOv2 embeddings for an ordered view list.

    Output order matches input order; the consumer owns any reordering
    (e.g. smartway's clockwise remap inside the waypoint predictor).
    On load failure emits an empty ``features`` string — consumers keep
    their degraded-mode fallback (the old in-engine behaviour was zeros).
    """

    node_type: ClassVar[str] = "model_dinov2__extract_features"
    display_name: ClassVar[str] = "DINOv2: Extract Features"
    description: ClassVar[str] = (
        "Pooled DINOv2 embedding per view (ViT-S/14-reg → 384-d); base64-npy envelope"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Grid3x3"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField("hub_model", "text", "torch.hub model name", default=_HUB_MODEL_DEFAULT),
            ConfigField("processor_id", "text", "HF image-processor ID", default=_PROCESSOR_DEFAULT),
        ],
    )
    input_ports = [
        PortDef("views", "ANY", "Ordered list of {rgb_base64} dicts (e.g. 12 directions)"),
    ]
    output_ports = [
        PortDef(
            "features", "TEXT",
            'JSON envelope {"shape":[N,D],"dtype":"float32","b64":…} of the (N,D) buffer',
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"features": ""}

        cfg = getattr(self, "config", None) or {}
        hub_model = cfg.get("hub_model", _HUB_MODEL_DEFAULT)
        processor_id = cfg.get("processor_id", _PROCESSOR_DEFAULT)

        loop = asyncio.get_running_loop()
        engine = _DinoV2Engine.get(_HUB_REPO_DEFAULT, hub_model, processor_id)

        def _extract() -> str:
            images = []
            for v in views:
                b64 = v.get("rgb_base64") if isinstance(v, dict) else None
                if not b64:
                    return ""  # incomplete panorama → let consumer degrade
                images.append(_decode_rgb(b64))
            feats = engine.extract(images)
            if feats is None:
                return ""
            buf = np.ascontiguousarray(feats)
            return json.dumps({
                "shape": list(buf.shape),
                "dtype": "float32",
                "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            })

        envelope = await loop.run_in_executor(None, _extract)
        if envelope:
            self._self_log("shape", json.loads(envelope)["shape"])
        else:
            self._self_log("degraded", "no features (load failure or missing view)")
        return {"features": envelope}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class DinoV2NodeSet(BaseNodeSet):
    """DINOv2 per-image feature extraction — server-mode FM nodeset."""

    name = "model_dinov2"
    description = "DINOv2 (ViT-S/14-reg) pooled per-image features — server-mode FM nodeset"
    # Stateless feature extractor — one shared server across eval workers.
    parallelism = "shared"
    # Default env: ac-fm (shared FM env) — byte-equal features vs the old
    # ac-smartway hosting given the use_fast=False PIL pin above (parity gate
    # 2026-07-05). Override with $DINOV2_PYTHON.
    server_python = conda_env_python("ac-fm", "DINOV2_PYTHON")

    def get_tools(self) -> list:
        return [ExtractFeaturesTool()]

    async def initialize(self, **kwargs: Any) -> None:
        log.info("DinoV2NodeSet ready (server_python=%s)", self.server_python)

    async def shutdown(self) -> None:
        pass
