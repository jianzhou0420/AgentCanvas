from __future__ import annotations

"""RAM (Recognize Anything Model) tagging — dedicated server-mode nodeset.

Hosts RAM Swin-L 14M tag inference in its own conda env (`ac-ram`)
to avoid version coupling with `opennav` (which carries a transformers
release that broke the RAM library's `apply_chunking_to_forward` import).
Per the per-model env discipline (memory: feedback_dedicated_env_per_model),
each new model that needs server mode gets a clean conda env.

Source: DiscussNav.py:135 + 165-178 — RAM init + per-direction
``ram_img_tagging`` inside ``Vision_Perception_Experts.observe_environment``.

Single tool:

    ram_perception__tag_panorama  (views: list[{dir_id, rgb_base64}])
                                  → tags_per_dir: LIST[TEXT]

Tags every direction in order; output aligned 1:1 with input ``views``.
Each entry is the verbatim space-joined RAM tag string (after stripping
RAM's `" |"` separators).

Runs in `ac-ram` env (Python 3.10, torch 2.4.1, transformers
<4.40, recognize-anything from upstream git).

last updated: 2026-05-10
"""

import asyncio
import base64
import io
import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef

log = logging.getLogger("agentcanvas.ram_perception")

# Per-server content-hash cache: sha1(rgb_base64) → tag string. Reuses tags for
# byte-identical views recurring across steps/workers (upstream view_record).
_TAG_CACHE: dict[str, str] = {}


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)
_RAM_CKPT_DEFAULT = os.environ.get(
    "RAM_CKPT",
    os.path.join(_REPO_ROOT, "data", "opennav", "ram_swin_large_14m.pth"),
)


# ══════════════════════════════════════════════════════════════════════
# Engine — lazy singleton
# ══════════════════════════════════════════════════════════════════════


class _RAMEngine:
    """Lazy singleton: load RAM Swin-L on first tag() call, stay resident."""

    _instance: _RAMEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.ckpt = _RAM_CKPT_DEFAULT
        self.device = None
        self.model = None
        self.transform = None
        self._loaded = False

    @classmethod
    def get(cls) -> _RAMEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info("Loading RAM Swin-L from %s", self.ckpt)
            import torch
            from ram import get_transform  # type: ignore
            from ram.models import ram  # type: ignore

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ram(pretrained=self.ckpt, image_size=384, vit="swin_l")
            model.eval().to(self.device)
            self.model = model
            self.transform = get_transform(image_size=384)
            self._loaded = True
            log.info("RAM ready (device=%s)", self.device)

    def tag(self, rgb: np.ndarray) -> str:
        self._ensure()
        from PIL import Image
        from ram import inference_ram  # type: ignore

        pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        image = self.transform(pil).unsqueeze(0).to(self.device)
        result = inference_ram(image, self.model)
        if isinstance(result, tuple):
            tags_en = result[0]
        else:
            tags_en = str(result)
        return tags_en.replace(" |", "").strip()


def _decode_rgb(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════


class TagPanoramaTool(BaseCanvasNode):
    """RAM-tag every direction in a panorama, output ordered.

    Source: DiscussNav.py:165-178 — ``ram_img_tagging`` per heading inside
    ``observe_environment`` (a 12-direction sweep).
    """

    node_type: ClassVar[str] = "model_ram__tag_panorama"
    display_name: ClassVar[str] = "RAM: Tag Panorama"
    description: ClassVar[str] = (
        "Recognize Anything (Swin-L 14M tags) per direction; ordered LIST[TEXT] output"
    )
    category: ClassVar[str] = "perception"
    icon: ClassVar[str] = "Tag"
    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="violet")
    input_ports = [
        PortDef("views", "ANY", "List of {dir_id, rgb_base64} dicts (e.g. 12 directions)"),
    ]
    output_ports = [
        PortDef(
            "tags_per_dir",
            "LIST[TEXT]",
            "Ordered list of space-joined RAM tag strings, aligned 1:1 with `views`",
        ),
    ]

    async def forward(self, inputs: dict, ctx: Any = None) -> dict:
        views = inputs.get("views") or []
        if not views:
            return {"tags_per_dir": []}

        loop = asyncio.get_running_loop()
        engine = _RAMEngine.get()

        def _tag_all() -> list[str]:
            import hashlib

            out: list[str] = []
            for v in views:
                if not isinstance(v, dict):
                    out.append("")
                    continue
                b64 = v.get("rgb_base64")
                if not b64:
                    out.append("")
                    continue
                # Content-hash cache: identical view bytes (same vp+heading,
                # recurring across steps on backtrack) reuse the tag — mirrors
                # upstream DiscussNav view_record caching, cheap and transparent.
                key = hashlib.sha1(b64.encode("ascii")).hexdigest()
                if key in _TAG_CACHE:
                    out.append(_TAG_CACHE[key])
                    continue
                try:
                    tag = engine.tag(_decode_rgb(b64))
                except Exception as exc:
                    log.warning("RAM tag failed for %s: %s", v.get("dir_id"), exc)
                    tag = ""
                _TAG_CACHE[key] = tag
                out.append(tag)
            return out

        tags = await loop.run_in_executor(None, _tag_all)
        self._self_log("num_directions", len(tags))
        for i, t in enumerate(tags):
            self._self_log(f"tags_{i}", t[:200])
        return {"tags_per_dir": tags}


# ══════════════════════════════════════════════════════════════════════
# NodeSet
# ══════════════════════════════════════════════════════════════════════


class RAMPerceptionNodeSet(BaseNodeSet):
    """RAM Swin-L 14M tag inference, dedicated env."""

    name = "model_ram"
    description = "Recognize Anything Model (RAM, Swin-L 14M tags) — dedicated server-mode nodeset"
    server_python = os.environ.get(
        "RAM_PERCEPTION_PYTHON",
        os.path.expanduser("~/miniforge3/envs/ac-ram/bin/python"),
    )

    def get_tools(self) -> list:
        return [TagPanoramaTool()]

    async def initialize(self, **kwargs: Any) -> None:
        engine = _RAMEngine.get()
        if "ram_ckpt" in kwargs:
            engine.ckpt = str(kwargs["ram_ckpt"])
        log.info("RAMPerceptionNodeSet ready (ckpt=%s)", engine.ckpt)

    async def shutdown(self) -> None:
        pass
